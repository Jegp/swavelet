import threading

import jax
import jax.numpy as jnp
import numpy as np

from . import wavelet
from . import temporal_integration as ti
from .signal_utils import fft_convolve


_KERNEL_CACHE = {}
_KERNEL_CACHE_LOCK = threading.Lock()


def _get_limit_kernels(cascade_mus, dt, signal_length):
    """Module-level cache for time-causal limit kernels.

    Keyed by `(tuple of per-stage tau tuples, dt, signal_length)` so any two
    wavelets with the same scales and dt share the build cost. Kernels are
    plain numpy arrays so no JAX tracer escapes the jit boundary.
    """
    key = (
        tuple(tuple(np.asarray(stage_taus).tolist()) for stage_taus in cascade_mus),
        float(dt),
        int(signal_length),
    )
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _KERNEL_CACHE_LOCK:
        cached = _KERNEL_CACHE.get(key)
        if cached is not None:
            return cached
        kernels = np.stack([
            ti.time_causal_limit_kernel_np(stage_taus, dt, signal_length)
            for stage_taus in cascade_mus
        ])
        _KERNEL_CACHE[key] = kernels
        return kernels


class DifferenceOfTimeCausalKernelsWavelet(wavelet.Wavelet):
    """
    Difference of Time-causal kernels (DoT) wavelet - implements bandpass
    filtering using differences of time-causal limit kernels at adjacent scales.

    Based on Lindeberg (2025) "Time-causal and time-recursive wavelets"

    The DoT kernel constructs wavelets from cascaded leaky integrators (truncated
    exponentials) and computes differences between adjacent temporal scales to
    create bandpass filters with perfect reconstruction properties.

    Key properties:
    - Time-causal (only depends on past values)
    - Perfect reconstruction via channel summation
    - Scale covariant (maintains properties under temporal scaling)
    - Zero mean (satisfies wavelet admissibility criterion)
    """

    def __init__(
        self,
        n_channels: int = 5,
        dt: float = 0.01,
        mu_max: float = 0.2,
        c: float = 2.0,
        cascade_depth_max: int = 7,
        alpha_floor: float = 0.01,
        cascade_kind: str = "lindeberg",
    ):
        """
        Args:
            n_channels: Total channels. K = n_channels - 1 bandpass plus one lowpass.
            dt: Time step for integration
            mu_max: Maximum time constant (coarsest/most smoothed scale)
            c: Scale factor between adjacent scales (c > 1)
            cascade_depth_max: Target cascade depth N (paper symbol). Each channel
                uses the largest N_k <= cascade_depth_max such that the per-stage
                time constant stays above the alpha-stability floor.
            alpha_floor: Minimum acceptable alpha per cascade stage (default 0.01).
            cascade_kind: Per-channel cascade construction:
                - "lindeberg" (default): non-uniform per-stage tau following
                  Lindeberg 2016 Eq. (58); approximates the time-causal limit
                  kernel with intra-cascade tau-ratio c^2.
                - "uniform": all stages share tau = mu_k/sqrt N_k; gives a Gamma-shape
                  cascade (CLT -> Gaussian), not the limit kernel.
        """
        if cascade_kind not in ("lindeberg", "uniform"):
            raise ValueError(f"cascade_kind must be 'lindeberg' or 'uniform', got {cascade_kind!r}")
        if n_channels < 2:
            raise ValueError(
                f"n_channels must be >= 2 (one bandpass channel and one "
                f"lowpass residual at minimum), got {n_channels}"
            )
        self.n_channels = n_channels
        self.dt = dt
        self.mu_max = mu_max
        self.c = c
        self.cascade_depth_max = cascade_depth_max
        self.alpha_floor = alpha_floor
        self.cascade_kind = cascade_kind

        # K smoothing scales mu_1..mu_K ending at mu_max (K = n_channels - 1).
        # The implicit finest level mu_0 = 0 is the raw signal (used directly
        # in analyse), so we don't allocate a kernel for it.
        n_smooth = n_channels - 1
        scale_indices = jnp.arange(n_smooth)
        mus = mu_max * (c ** (-(n_smooth - 1 - scale_indices)))

        import math
        mu_stage_min = dt / (-math.log(alpha_floor))
        self.cascade_mus = []
        self.cascade_depths = []
        for mu in mus:
            mu_f = float(mu)
            N_k_stable = max(1, int((mu_f / mu_stage_min) ** 2))
            N_k = max(1, min(cascade_depth_max, N_k_stable))
            if cascade_kind == "uniform":
                stage_taus = jnp.full(N_k, mu / jnp.sqrt(float(N_k)))
            else:  # lindeberg
                stage_taus = jnp.asarray(ti.geometric_time_constants(mu_f, dt, c, N_k))
            self.cascade_mus.append(stage_taus)
            self.cascade_depths.append(N_k)

        self.params = {
            "log_mus": jnp.log(mus),  # Log of K smoothing time constants
            "log_c": jnp.log(c),  # Log of scale factor
            # Unit channel weights give the perfect-reconstruction telescoping
            # sum at default settings; lstsq adjusts them to compensate for
            # finite-mu truncation when fitted to a target signal.
            "channel_weights": jnp.ones(n_channels),
        }

    def limit_kernel_response(self, params, signal, scale_idx):
        """Convolve `signal` with the cached time-causal limit kernel for
        smoothing scale `scale_idx` (0..K-1), producing L(t; mu_{scale_idx+1}).
        """
        kernels = _get_limit_kernels(self.cascade_mus, self.dt, len(signal))
        return fft_convolve(signal, kernels[scale_idx], mode='full')[:len(signal)]

    def __call__(self, params, signal):
        """Per-channel analysis output, shape (n_channels, T).

        With K = n_channels - 1 bandpass channels at smoothing scales
        mu_1..mu_K, channels 0..K-1 carry the bandpass differences
            Delta L_1 = L(mu_1) f - f
            Delta L_k = L(mu_k) f - L(mu_{k-1}) f   for k = 2..K
        and channel K (the last one) carries the lowpass residual L(mu_K) f.
        Using the raw signal as the implicit L_0 = f makes the telescoping
        sum lowpass - sum(bandpass) recover f exactly.
        """
        scale_responses = [
            self.limit_kernel_response(params, signal, i) for i in range(self.n_channels - 1)
        ]
        encoded_differences = [scale_responses[0] - signal] + [
            scale_responses[i + 1] - scale_responses[i]
            for i in range(self.n_channels - 2)
        ]
        encoded_base = scale_responses[-1]
        return jnp.array(encoded_differences + [encoded_base])

    def reconstruct(self, params, per_channel):
        """Combine per-channel signals into a reconstruction.

        Paper Eq. (scale_bandpass_reconstruction):
            f(t) = w_K x_K(t) - sum_{k=1}^{K} w_k x_k(t),
        where the last channel (index K) is the lowpass residual and channels
        0..K-1 are the bandpass differences Delta L_k. Default unit weights
        give the telescoping sum.
        """
        weights = params["channel_weights"]
        per_channel = jnp.asarray(per_channel)
        return weights[-1] * per_channel[-1] - jnp.sum(
            weights[:-1, None] * per_channel[:-1], axis=0
        )

    def time_scales(self):
        """
        Return the temporal scales (time constants) mu_k used by each channel.
        """
        return jnp.exp(self.params["log_mus"])

    def to_nir(self):
        """Export the analysis stage as a `nir.NIRGraph`.

        Delegates to :func:`swavelet.nir_export.from_dot`.
        """
        from .nir_export import from_dot
        return from_dot(self)

    def get_readable_params(self, params=None):
        """
        Get parameters in readable format.
        """
        if params is None:
            params = self.params

        mus = jnp.exp(params["log_mus"])
        c_value = jnp.exp(params["log_c"])

        return {
            "mus": mus,
            "c": c_value,
            "scale_ratios": mus[1:] / mus[:-1],  # Ratios between scales
            "weights": params["channel_weights"],
            "cascade_depth_max": self.cascade_depth_max,
            "cascade_depths": list(self.cascade_depths),
        }

    def get_scale_space_decomposition(self, params, signal):
        """
        Get the full scale-space decomposition for visualization.

        Returns:
            scale_responses: List of L(t; mu_k, c) for k = 1..K
            bandpass_responses: List of K bandpass responses, with the first
                being Delta L_1 = L(mu_1) f - f.
        """
        scale_responses = [
            self.limit_kernel_response(params, signal, i) for i in range(self.n_channels - 1)
        ]

        bandpass_responses = [scale_responses[0] - signal] + [
            scale_responses[i + 1] - scale_responses[i]
            for i in range(self.n_channels - 2)
        ]

        return {
            "scale_responses": jnp.array(scale_responses),
            "bandpass_responses": jnp.array(bandpass_responses),
        }

    def dot_kernel_response(self, params, t, channel_k, channel_k_minus_1=None):
        """
        Compute the DoT kernel in the time domain for visualization.

        The DoT kernel is the difference of two limit kernels:
        psi_DoT(t; mu_k) = Psi(t; mu_k, c) - Psi(t; mu_{k-1}, c)

        We compute this by applying the limit kernel response to an impulse signal.

        Args:
            t: Time array
            channel_k: Index of channel k (0 to n_channels-1)
            channel_k_minus_1: Optional index of previous channel
        """
        # Create an impulse signal at t=0
        # Find the index closest to t=0
        t_array = jnp.asarray(t)
        zero_idx = jnp.argmin(jnp.abs(t_array))

        # Create impulse (delta function approximation)
        impulse = jnp.zeros_like(t_array)
        impulse = impulse.at[zero_idx].set(1.0 / self.dt)  # Scale by 1/dt for proper impulse

        # Compute the limit kernel impulse response at scale μ_k
        kernel_k = self.limit_kernel_response(params, impulse, channel_k)

        # If we have a previous scale, compute and subtract
        if channel_k_minus_1 is not None:
            kernel_k_minus_1 = self.limit_kernel_response(params, impulse, channel_k_minus_1)
            return kernel_k - kernel_k_minus_1
        else:
            return kernel_k

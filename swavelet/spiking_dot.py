import jax
import jax.numpy as jnp
import numpy as np

from . import wavelet
from . import temporal_integration as ti
from .dot import DifferenceOfTimeCausalKernelsWavelet


class SpikingDoTWavelet(wavelet.Wavelet):
    """
    Spiking implementation of Difference of Time-causal kernels (DoT) wavelets.

    Based on equations (39-41) from the paper, this implements the bandpass
    representation Delta L(t; tau_k) using dual-channel (positive/negative) spiking neurons:

    tau_m du+_k/dt = -u+_k + [L(t; tau_k) - L(t; tau_{k-1})]
    tau_m du-_k/dt = -u-_k + [L(t; tau_{k-1}) - L(t; tau_k)]

    When u+/-_k crosses theta_thr, the neuron emits a spike and resets to zero.

    Reconstruction via spike integration (equation 41):
    R_k(t) = h_exp(t; mu_r) * [s+_k(t) - s-_k(t)]
    """

    def __init__(
        self,
        n_channels: int = 5,
        dt: float = 0.01,
        mu_max: float = 0.2,
        c: float = 2.0,
        cascade_depth_max: int = 7,
        alpha_floor: float = 0.01,
        threshold: float = None,
        recon_scale_factor: float = 1.0,
        surrogate_type: str = 'sigmoid',
        enable_normalization: bool = True,
        surrogate_beta: float = 10.0,
        cascade_kind: str = "lindeberg",
    ):
        """
        Args:
            n_channels: Number of wavelet channels at different scales
            dt: Time step for integration
            mu_max: Maximum time constant (coarsest/most smoothed scale)
            c: Scale factor between adjacent scales (c > 1)
            cascade_depth_max: Target cascade depth N (paper symbol). Per-channel
                N_k is chosen as the largest N_k <= cascade_depth_max keeping each
                cascade stage's alpha = exp(-dt*sqrt N_k / mu_k) >= alpha_floor.
            alpha_floor: Minimum acceptable alpha per cascade stage (default 0.01).
            threshold: Spiking threshold for each channel. If None, uses adaptive initialization
                      based on dt and mu_scales
            recon_scale_factor: Scale factor for reconstruction time constants (1.0 = use encoding mu)
            surrogate_type: Type of surrogate gradient ('sigmoid', 'fast_sigmoid', 'arctan')

        Note:
            For DoT wavelets, each channel uses a cascade of exponentials with time constant mu.
            Membrane time constants are set equal to filter time constants by default.
        """
        if cascade_kind not in ("lindeberg", "uniform"):
            raise ValueError(f"cascade_kind must be 'lindeberg' or 'uniform', got {cascade_kind!r}")
        if n_channels < 2:
            raise ValueError(
                f"n_channels must be >= 2 (one bandpass channel and one "
                f"lowpass residual at minimum), got {n_channels}"
            )
        import math
        self.n_channels = n_channels
        self.dt = dt
        self.mu_max = mu_max
        self.c = c
        self.cascade_depth_max = cascade_depth_max
        self.alpha_floor = alpha_floor
        self.recon_scale_factor = recon_scale_factor
        self.surrogate_type = surrogate_type
        self.surrogate_beta = surrogate_beta
        self.enable_normalization = enable_normalization
        self.cascade_kind = cascade_kind

        # K smoothing scales mu_1..mu_K ending at mu_max (K = n_channels - 1).
        # The implicit finest level mu_0 = 0 is the raw signal; the
        # high-frequency bandpass channel uses Delta L_1 = L(mu_1) f - f.
        n_smooth = n_channels - 1
        scale_indices = jnp.arange(n_smooth)
        mus = mu_max * (c ** (-(n_smooth - 1 - scale_indices)))

        mu_stage_min = dt / (-math.log(alpha_floor))
        self.cascade_depths = [
            max(1, min(cascade_depth_max, int((float(mu) / mu_stage_min) ** 2)))
            for mu in mus
        ]

        # Per-smoothing-scale per-stage τ values. For "lindeberg" these follow
        # Lindeberg 2016 Eq. (58) with geometric τ progression (ratio c²);
        # for "uniform" all N_k stages share τ = μ_k/√N_k (Gamma shape).
        self.cascade_mus = []
        for k, mu in enumerate(mus):
            mu_f = float(mu)
            N_k = self.cascade_depths[k]
            if cascade_kind == "uniform":
                stage_taus = jnp.full(N_k, mu / jnp.sqrt(float(N_k)))
            else:  # lindeberg
                stage_taus = jnp.asarray(ti.geometric_time_constants(mu_f, dt, c, N_k))
            self.cascade_mus.append(stage_taus)

        # Per-output-channel (length n_channels = K+1) views of the
        # smoothing-scale arrays. Bandpass channel k (k=0..K-1) uses smoothing
        # scale mus[k]; the lowpass channel (index K) reuses the coarsest
        # smoothing scale mus[-1].
        mus_per_channel = jnp.append(mus, mus[-1])
        depths_per_channel = list(self.cascade_depths) + [self.cascade_depths[-1]]

        mu_mem_per_channel = jnp.ones(n_channels) * dt
        mu_recon_per_channel = mus_per_channel * recon_scale_factor

        if threshold is None:
            sqrtN = jnp.array([float(N_k) ** 0.5 for N_k in depths_per_channel])
            mu_eff = mus_per_channel * sqrtN
            threshold = 10 * mu_eff

        self.params = {
            "log_mus": jnp.log(mus),  # K smoothing scales
            "log_c": jnp.log(c),
            "log_threshold": jnp.log(jnp.ones(n_channels) * threshold),
            "log_mu_mem": jnp.log(mu_mem_per_channel),
            "log_mu_recon": jnp.log(mu_recon_per_channel),
            "log_weights": jnp.log(1.0 / mus_per_channel),
            "surrogate_beta": self.surrogate_beta
        }

        # Precompute per-channel filter norms for input normalization.
        if enable_normalization:
            import numpy as np
            mus_np = np.array(mus)
            norms = np.zeros(n_channels)
            for k in range(n_channels - 1):  # bandpass channels
                norms[k] = ti.bandpass_filter_norms(mus_np[k:k+1], dt, self.cascade_depths[k])[0]
            # Lowpass: reuse coarsest smoothing-scale norm.
            norms[-1] = ti.bandpass_filter_norms(mus_np[-1:], dt, self.cascade_depths[-1])[0]
            self._filter_norms = norms
        else:
            self._filter_norms = None

        # Non-spiking sibling that owns the analysis stage. Reused by
        # `bandpass_inputs` so the per-channel kernel logic isn't duplicated.
        self._analysis = DifferenceOfTimeCausalKernelsWavelet(
            n_channels=n_channels, dt=dt, mu_max=mu_max, c=c,
            cascade_depth_max=cascade_depth_max, alpha_floor=alpha_floor,
            cascade_kind=cascade_kind,
        )

    def leaky_integrate(self, signal, mu):
        """
        Leaky integrator: du/dt = -(u/mu) + input
        Discrete-time: u[n] = alpha*u[n-1] + (1-alpha)*input[n] where alpha = exp(-dt/mu)
        """
        alpha = jnp.exp(-self.dt / mu)

        def scan_fn(u_prev, x_current):
            u_new = alpha * u_prev + (1 - alpha) * x_current
            return u_new, u_new

        _, u_trace = jax.lax.scan(scan_fn, 0.0, signal)
        return u_trace

    def limit_kernel_response(self, signal, scale_idx):
        """
        Cascade the per-stage tau values for smoothing scale `scale_idx`
        (0..K-1), producing the scale-space representation L(t; mu_{scale_idx+1}).
        """
        output = signal
        for tau in self.cascade_mus[scale_idx]:
            output = self.leaky_integrate(output, tau)
        return output

    def __call__(self, params, signal, return_trace: bool = False):
        """Spike-domain analysis: returns (2*n_channels, T) spike trains.

        Each channel's bandpass input is computed by the non-spiking sibling
        `self._analysis`, prescaled, and encoded with a pair of LIF
        neurons (positive- and negative-going). The output stacks the pairs
        as `[pos_0, neg_0, pos_1, neg_1, ..., pos_K, neg_K]` (K bandpass plus
        one lowpass row) so each physical neuron has its own row.

        If `return_trace=True`, also returns the matching `(2*n_channels, T)`
        membrane potential traces.
        """
        thresholds = jnp.exp(params["log_threshold"])
        mu_mem = jnp.exp(params["log_mu_mem"])
        beta = params.get("surrogate_beta", 10.0)

        if self.enable_normalization:
            prescale_factors = 1.0 / (jnp.array(self._filter_norms) + 1e-10)
        else:
            prescale_factors = jnp.ones(self.n_channels)

        bandpass_inputs = self._analysis(self._analysis.params, signal)
        all_spikes = []
        traces = [] if return_trace else None
        for i in range(self.n_channels):
            scaled = bandpass_inputs[i] * prescale_factors[i]
            if return_trace:
                spikes_pos, trace_pos = self.integrate_and_fire_with_reset(
                    scaled, mu_mem[i], thresholds[i], beta, return_trace=True
                )
                spikes_neg, trace_neg = self.integrate_and_fire_with_reset(
                    -scaled, mu_mem[i], thresholds[i], beta, return_trace=True
                )
                traces.extend([trace_pos, trace_neg])
            else:
                spikes_pos = self.integrate_and_fire_with_reset(
                    scaled, mu_mem[i], thresholds[i], beta, return_trace=False
                )
                spikes_neg = self.integrate_and_fire_with_reset(
                    -scaled, mu_mem[i], thresholds[i], beta, return_trace=False
                )
            all_spikes.extend([spikes_pos, spikes_neg])

        spikes = jnp.array(all_spikes)
        if return_trace:
            return spikes, jnp.array(traces)
        return spikes

    def reconstruct(self, params, spikes, return_trace: bool = False):
        """Spike-domain synthesis: collapse pos/neg spike pairs into a signed
        spike train per channel, decode each channel, then reuse the non-
        spiking sibling's `reconstruct` to combine the channels.

        Args:
            spikes: `(2*n_channels, T)` -- pos/neg LIF spike trains, ordered
                as [pos_0, neg_0, pos_1, neg_1, ...].

        If `return_trace=True`, also returns the per-channel reconstructions
        before they are combined (`(n_channels, T)`).
        """
        spikes = jnp.asarray(spikes)
        signed_spikes = spikes[0::2] - spikes[1::2]   # (n_channels, T)
        weights = jax.nn.softmax(params["log_weights"])
        channel_recons = self.channel_reconstruction_from_spikes(params, signed_spikes)
        reconstruction = self._analysis.reconstruct(
            {"channel_weights": weights}, channel_recons,
        )
        if return_trace:
            return reconstruction, channel_recons
        return reconstruction

    def channel_reconstruction_from_spikes(self, params, signed_spikes):
        """Per-channel dual-frame reconstruction from signed spike trains.

        Synthesis kernels per channel:
            i = 0       : Psi(mu_1) - delta   (high-frequency band, Delta L_1)
            i = 1..K-1  : Psi(mu_{i+1}) - Psi(mu_i)   (bandpass differences)
            i = K       : Psi(mu_K)           (lowpass residual)
        Each kernel is composed with the per-channel membrane h_mem.

        Args:
            params: wavelet parameters dict (uses log_mu_mem, log_threshold).
            signed_spikes: (n_channels, T) -- signed spike trains, where
                signed_spikes[i] = spikes_pos[i] - spikes_neg[i].

        Returns:
            (n_channels, T) per-channel reconstructions scaled by per-channel
            thresholds. A weighted sum (with `log_weights`) yields the full
            reconstruction produced by `__call__`.
        """
        thresholds = jnp.exp(params["log_threshold"])
        mu_mem = jnp.exp(params["log_mu_mem"])

        diff_recons = []
        # i = 0: Delta L_1 = L(mu_1) - f. Dual kernel = Psi(mu_1) - delta;
        # the delta term passes through h_mem with normalization 1/(0 - mu_mem),
        # i.e. -h_mem * spikes / mu_mem.
        recon_0 = ti.composite_cascade_kernel(
            signed_spikes[0], self.cascade_mus[0], mu_mem[0], self.dt,
        ) - (-ti.exponential_filter(signed_spikes[0], mu_mem[0], self.dt) / mu_mem[0])
        diff_recons.append(recon_0)
        for i in range(1, self.n_channels - 1):
            # Synthesis kernel: Delta Psi_{i+1} = Psi(mu_{i+1}) - Psi(mu_i).
            recon_i = ti.composite_cascade_difference_kernel(
                signed_spikes[i],
                self.cascade_mus[i], self.cascade_mus[i - 1],
                mu_mem[i], self.dt,
            )
            diff_recons.append(recon_i)
        diff_recons = jnp.array(diff_recons) * thresholds[:-1, None]

        base_recon = ti.composite_cascade_kernel(
            signed_spikes[-1], self.cascade_mus[-1], mu_mem[-1], self.dt,
        ) * thresholds[-1]

        return jnp.concatenate([diff_recons, base_recon[None, :]], axis=0)

    def integrate_and_fire_with_reset(self, signal, mu_mem, threshold, beta, return_trace=False):
        """
        Integrate-and-fire neuron with membrane reset after spikes.
        Uses surrogate gradients for learning.
        """
        return ti.integrate_and_fire_with_reset(
            signal, mu_mem, threshold, self.dt, beta,
            surrogate_type=self.surrogate_type,
            return_trace=return_trace
        )

    def get_spike_trains(self, params, signal):
        """
        Get the raw spike trains (positive and negative) for each channel.
        Useful for analysis and visualization.
        """
        thresholds = jnp.exp(params["log_threshold"])
        mu_mem = jnp.exp(params["log_mu_mem"])
        beta = params.get("surrogate_beta", 10.0)

        # n_channels = K+1 inputs from _analysis: K bandpass differences (with
        # Delta L_1 = L(mu_1)*f - f as channel 0) plus the lowpass residual.
        bandpass_inputs = self._analysis(self._analysis.params, signal)
        scale_responses = jnp.array([
            self.limit_kernel_response(signal, k) for k in range(self.n_channels - 1)
        ])

        spike_trains_pos = []
        spike_trains_neg = []

        for i in range(self.n_channels):
            spikes_pos = self.integrate_and_fire_with_reset(
                bandpass_inputs[i], mu_mem[i], thresholds[i], beta
            )
            spikes_neg = self.integrate_and_fire_with_reset(
                -bandpass_inputs[i], mu_mem[i], thresholds[i], beta
            )

            spike_trains_pos.append(spikes_pos)
            spike_trains_neg.append(spikes_neg)

        return {
            "spike_trains_pos": jnp.array(spike_trains_pos),
            "spike_trains_neg": jnp.array(spike_trains_neg),
            "bandpass_inputs": bandpass_inputs,
            "scale_responses": scale_responses,
        }

    def to_nir(self):
        """Export the spiking analysis stage as a `nir.NIRGraph`.

        Delegates to :func:`swavelet.nir_export.from_spiking_dot`.
        """
        from .nir_export import from_spiking_dot
        return from_spiking_dot(self)

    def time_scales(self):
        """Temporal scales mu_k used by each channel."""
        return jnp.exp(self.params["log_mus"])

    def recon_time_scales(self):
        """Per-channel reconstruction time constants (may differ from `time_scales()`
        when `recon_scale_factor != 1.0`).
        """
        return jnp.exp(self.params["log_mu_recon"])

    def get_readable_params(self, params=None):
        """Get parameters in readable format."""
        if params is None:
            params = self.params

        mus = jnp.exp(params["log_mus"])
        c_value = jnp.exp(params["log_c"])

        return {
            "mus": mus,
            "c": c_value,
            "thresholds": jnp.exp(params["log_threshold"]),
            "mu_mem": jnp.exp(params["log_mu_mem"]),
            "mu_recon": jnp.exp(params["log_mu_recon"]),
            "weights": jnp.exp(params["log_weights"]),
            "cascade_depth_max": self.cascade_depth_max,
            "cascade_depths": list(self.cascade_depths),
            "surrogate_beta": params.get("surrogate_beta", 10.0),
        }

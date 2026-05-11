from math import ceil, sqrt

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erfcinv, ive

from . import wavelet
from . import temporal_integration as ti
from .signal_utils import fft_convolve


def discrete_gaussian_kernel(sigma, dt, epsilon=1e-6):
    """Lindeberg's discrete Gaussian kernel T(n; s) = e^{-s} * I_n(s), s = (sigma/dt)^2.

    Built via `scipy.special.ive` (exponentially-scaled modified Bessel I_n),
    symmetric about n=0, truncated where the cumulative mass exceeds 1 - epsilon.
    Returns a normalized NumPy array of odd length -- suitable for conversion
    to a static JAX array at wavelet-construction time.

    Unlike a sampled continuous Gaussian truncated at k*sigma, this kernel has no
    edge bias: it is the scale-space-correct discrete Gaussian (Lindeberg 1990)
    and preserves non-creation of local extrema on the integer lattice.
    """
    sigma_samples = float(sigma) / float(dt)
    s = sigma_samples * sigma_samples
    # Upper-bound kernel size from the Gaussian tail: ~ 1 + 1.5*sigma*sqrt(2)*erfcinv(epsilon)
    n_upper = int(ceil(1 + 1.5 * sqrt(2.0 * max(s, 1e-12)) * erfcinv(epsilon)))
    ns = np.arange(0, n_upper + 1)
    half = ive(ns, s)  # e^{-s} · I_n(s) — numerically safe for large s

    cum = half[0]
    length = len(half)
    for i in range(1, len(half)):
        cum += 2 * half[i]
        if cum >= 1.0 - epsilon:
            length = i + 1
            break
    half = half[:length]
    full = np.concatenate([half[:0:-1], half])
    return full / full.sum()


def _build_kernel_bank(sigmas, dt, epsilon=1e-6):
    """Build per-channel discrete Gaussian kernels, padded to common length.

    Returns `(kernels, max_len)` where `kernels` has shape (n_channels, max_len)
    as a JAX array, zero-padded symmetrically so each kernel stays centered.
    """
    kers = [discrete_gaussian_kernel(float(s), dt, epsilon) for s in sigmas]
    max_len = max(k.size for k in kers)
    if max_len % 2 == 0:
        max_len += 1  # keep odd so the center sample is unambiguous
    padded = np.zeros((len(kers), max_len))
    for i, k in enumerate(kers):
        offset = (max_len - k.size) // 2
        padded[i, offset:offset + k.size] = k
    return jnp.asarray(padded), max_len


def gaussian_half_width(sigma, dt, n_sigmas=4.0):
    """Compute kernel half-width in samples from sigma, dt, and truncation width."""
    return max(int(float(n_sigmas) * float(sigma) / float(dt) + 0.5), 1)


def gaussian_kernel(sigma, dt, n_sigmas=4.0, half_width=None):
    """
    Create a discrete 1D Gaussian kernel, normalized to unit sum.

    Args:
        sigma: Standard deviation of the Gaussian
        dt: Time step
        n_sigmas: Number of standard deviations for kernel truncation
        half_width: Pre-computed half-width in samples. Required when sigma
                    is a JAX tracer (e.g. inside vmap). If None, computed
                    from n_sigmas, sigma, and dt.

    Returns:
        Normalized Gaussian kernel array
    """
    if half_width is None:
        half_width = gaussian_half_width(sigma, dt, n_sigmas)
    t = jnp.arange(-half_width, half_width + 1) * dt
    kernel = jnp.exp(-0.5 * (t / sigma) ** 2)
    return kernel / jnp.sum(kernel)


def gaussian_filter(signal, sigma, dt, n_sigmas=4.0, half_width=None):
    """
    Apply a Gaussian filter to the signal via convolution.

    Uses 'same' mode to preserve signal length.

    Args:
        signal: Input signal (1D array)
        sigma: Standard deviation of the Gaussian
        dt: Time step
        n_sigmas: Number of standard deviations for kernel truncation
        half_width: Pre-computed half-width in samples. Required when sigma
                    is a JAX tracer (e.g. inside vmap).

    Returns:
        Filtered signal (same length as input)
    """
    kernel = gaussian_kernel(sigma, dt, n_sigmas, half_width)
    return fft_convolve(signal, kernel, mode='same')


def gaussian_difference_kernel_filter(signal, sigma_fine, sigma_coarse, dt, n_sigmas=4.0, half_width=None):
    """
    Apply difference-of-Gaussians kernel for reconstruction.

    Dual frame reconstruction: h_DoG(t) = G(sigma_coarse) - G(sigma_fine)

    Args:
        signal: Input signal
        sigma_fine: Standard deviation of finer/faster Gaussian
        sigma_coarse: Standard deviation of coarser/slower Gaussian
        dt: Time step
        n_sigmas: Number of standard deviations for kernel truncation
        half_width: Pre-computed half-width in samples. Required when sigma
                    is a JAX tracer (e.g. inside vmap).

    Returns:
        Filtered signal using difference kernel
    """
    filtered_coarse = gaussian_filter(signal, sigma_coarse, dt, n_sigmas, half_width)
    filtered_fine = gaussian_filter(signal, sigma_fine, dt, n_sigmas, half_width)
    return filtered_coarse - filtered_fine


class DifferenceOfGaussiansWavelet(wavelet.Wavelet):
    """
    Difference of Gaussians (DoG) wavelet.

    Computes N Gaussian filters at different temporal scales,
    then provides N-1 "difference channels" by taking the difference
    between consecutive filtered outputs.

    For N filters with standard deviations [sigma_1, sigma_2, ..., sigma_N], this produces
    N-1 difference channels:
        DoG_i = G(sigma_i) - G(sigma_{i+1})  for i = 1, ..., N-1

    Unlike DoE (causal exponential filters) or DoT (cascaded truncated exponentials),
    this uses full (symmetric) Gaussian kernels applied via convolution.
    """

    def __init__(
        self,
        n_channels: int = 3,
        dt: float = 0.01,
        sigma_max: float = 0.2,
        c: float = 2.0**0.5,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            n_channels: Total channels (>= 2). K = n_channels - 1 bandpass plus one lowpass.
            dt: Time step for integration
            sigma_max: Maximum standard deviation (coarsest scale)
            c: Distribution parameter for logarithmic spacing (c > 1)
            epsilon: Relative truncation error for the discrete Gaussian kernel
                (smaller epsilon -> longer kernel but better accuracy).
        """
        if n_channels < 2:
            raise ValueError(
                f"n_channels must be >= 2 (one bandpass channel and one "
                f"lowpass residual at minimum), got {n_channels}"
            )
        self.n_channels = n_channels
        self.dt = dt
        self.c = c
        self.epsilon = epsilon

        # K smoothing scales sigma_1..sigma_K ending at sigma_max
        # (K = n_channels - 1). The implicit zero-scale level L_0 = f is the
        # raw signal, used directly in analyse, so we don't allocate a kernel
        # for it.
        n_smooth = n_channels - 1
        scale_indices = np.arange(n_smooth)
        sigmas = sigma_max * (c ** (-(n_smooth - 1 - scale_indices)))

        # Precompute per-channel discrete Gaussian kernels (static arrays).
        self._kernels, self._kernel_len = _build_kernel_bank(sigmas, dt, epsilon)

        self.params = {
            "log_sigmas": jnp.log(jnp.asarray(sigmas)),
            # Unit channel weights give the perfect-reconstruction telescoping
            # sum at default settings; lstsq adjusts them to compensate for
            # finite-sigma truncation when fitted to a target signal.
            "channel_weights": jnp.ones(n_channels),
            "kernels": self._kernels,
        }

    def __call__(self, params, signal):
        """Per-channel analysis output, shape (n_channels, T).

        With K = n_channels - 1 bandpass channels at smoothing scales
        sigma_1..sigma_K, channels 0..K-1 carry the bandpass differences
            Delta L_1 = G(sigma_1) f - f
            Delta L_k = G(sigma_k) f - G(sigma_{k-1}) f   for k = 2..K
        and channel K (the last one) carries the lowpass residual G(sigma_K) f.
        Using the raw signal as the implicit L_0 = f makes the telescoping
        sum lowpass - sum(bandpass) recover f exactly.
        """
        def conv(kernel):
            return fft_convolve(signal, kernel, mode='same')
        filter_responses = jax.vmap(conv)(params["kernels"])  # (K, T)
        first_diff = filter_responses[0] - signal
        rest_diff = filter_responses[1:] - filter_responses[:-1]
        encoded_base = filter_responses[-1]
        return jnp.concatenate(
            [first_diff[None, :], rest_diff, encoded_base[None, :]], axis=0
        )

    def reconstruct(self, params, per_channel):
        """Combine per-channel signals into a reconstruction.

        Paper Eq. (scale_bandpass_reconstruction):
            f(t) = w_K x_K(t) - sum_{k=1}^{K} w_k x_k(t),
        where the last channel (index K) is the lowpass residual and channels
        0..K-1 are the bandpass differences Delta L_k (with
        Delta L_1 = G(sigma_1) f - f). With the default unit weights this
        telescopes to f exactly; learned channel_weights compensate for
        finite-sigma kernel truncation.
        """
        weights = params["channel_weights"]
        per_channel = jnp.asarray(per_channel)
        return weights[-1] * per_channel[-1] - jnp.sum(
            weights[:-1, None] * per_channel[:-1], axis=0
        )

    def time_scales(self):
        return jnp.exp(self.params["log_sigmas"])

    def get_readable_params(self, params=None):
        if params is None:
            params = self.params

        sigmas = jnp.exp(params["log_sigmas"])
        return {
            "sigmas": sigmas,
            "weights": params["channel_weights"],
        }

    def temporal_derivatives(self, params, signal, order=1):
        encoding = self(params, signal)
        reconstruction = self.reconstruct(params, encoding)

        if order == 1:
            derivative = jnp.diff(reconstruction, prepend=reconstruction[0])
        elif order == 2:
            derivative = jnp.diff(reconstruction, n=2, prepend=reconstruction[:2])
        else:
            raise ValueError(f"Derivative order {order} not implemented")

        return derivative

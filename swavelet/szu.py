import jax
import jax.numpy as jnp

from . import wavelet
from .signal_utils import fft_convolve

class SzuWavelet(wavelet.Wavelet):
    """
    Szu Causal Analytical Wavelet based on Szu et al. (1992).

    Wavelet form: h(t) = exp(-t/tau) * exp(j*2 pi ft) for t >= 0, zero otherwise

    Properties:
    - Causal (zero for t < 0)
    - Analytical (only positive frequencies)
    - Exponentially decaying envelope
    - Superior SNR for causal transient signals

    Reference:
    Szu, H. H., Telfer, B. A., & Lohmann, A. (1992).
    "Causal analytical wavelet transform."
    Optical Engineering, 31(9), 1825-1829.
    """

    def __init__(self, n_channels: int = 3, dt: float = 0.01,
                 f_min: float = 3.0, f_max: float = 20.0):
        self.n_channels = n_channels
        self.dt = dt
        self.f_min = f_min
        self.f_max = f_max

        # Saturation caps must exceed the target band so _params_to_values
        # can reproduce frequencies up to f_max.
        self.freq_cap = max(100.0, 2.0 * f_max)
        self.decay_cap = max(80.0, 2.0 * f_max)

        # Log-spaced target frequencies and decay rates across [f_min, f_max].
        # Decay-rate range mirrors the frequency range so every channel has
        # ~comparable Q.
        target_freqs = jnp.geomspace(f_min, f_max, n_channels)
        decay_min = max(1.0, f_min / 3.0)
        decay_max = max(10.0, f_max / 2.0)
        target_decays = jnp.geomspace(decay_min, decay_max, n_channels)

        # Invert the sigmoid saturation in _params_to_values:
        # x = cap / (1 + exp(-log_x/1.5))  =>  log_x = 1.5 * log(x / (cap - x))
        log_frequencies = 1.5 * jnp.log(target_freqs / (self.freq_cap - target_freqs))
        log_decay_rates = 1.5 * jnp.log(target_decays / (self.decay_cap - target_decays))

        self.params = {
            "log_frequencies": log_frequencies,
            "log_decay_rates": log_decay_rates,
            "channel_weights": jnp.ones(n_channels) / n_channels,
        }

    def _params_to_values(self, params):
        """
        Convert log-space parameters to actual values with soft bounds.

        Uses smooth transformations to prevent extreme values that cause numerical
        instability, while maintaining continuous gradients for optimization.
        """
        # Soft saturation using sigmoid-like transformation
        # x_out = cap / (1 + exp(-x_in/1.5))  →  (-inf, +inf) → (0, cap)
        frequencies = self.freq_cap / (1.0 + jnp.exp(-params["log_frequencies"] / 1.5))
        decay_rates = self.decay_cap / (1.0 + jnp.exp(-params["log_decay_rates"] / 1.5))
        return frequencies, decay_rates

    def szu_kernel(self, t, frequency, decay_rate):
        """
        Szu causal analytical wavelet kernel from Szu et al. (1992).

        h(t) = exp(-t/tau) * exp(j*2 pi ft) for t >= 0
             = 0                       for t < 0

        Args:
            t: Time array
            frequency: Center frequency in Hz
            decay_rate: Exponential decay rate = 1/tau in Hz

        Returns:
            Complex-valued normalized kernel (causal and analytical)
        """
        tau = 1.0 / decay_rate
        omega = 2 * jnp.pi * frequency

        # Normalization factor with correction for causal convolution
        # Use conservative 5x factor to compensate for causal kernels while limiting growth
        # This lower factor significantly reduces gradient incentive for unbounded decay_rate growth
        # Combined with learning rate 0.003, parameters naturally stabilize at reasonable values
        norm = 5.0 * jnp.sqrt(2 * decay_rate)

        # Causal exponential envelope (zero for t < 0)
        envelope = jnp.where(t >= 0, jnp.exp(-t / tau), 0.0)

        # Complex oscillation (analytical signal)
        oscillation = jnp.exp(1j * omega * t)

        return norm * envelope * oscillation

    def __call__(self, params, signal):
        """Per-channel CWT coefficients W_k via causal analytical Szu kernels.

        Returns a complex array of shape (n_channels, n_samples).
        """
        frequencies, decay_rates = self._params_to_values(params)
        n_samples = len(signal)
        encodings = []
        for freq, decay_rate in zip(frequencies, decay_rates):
            t_kernel = (jnp.arange(n_samples) - n_samples // 2) * self.dt
            kernel = self.szu_kernel(t_kernel, freq, decay_rate)
            W_k = fft_convolve(signal, jnp.flip(jnp.conj(kernel)), mode="same") * self.dt
            encodings.append(W_k)
        return jnp.array(encodings)

    def reconstruct(self, params, coefficients):
        """Convolve each W_k with its kernel psi_k and sum the real parts.

        Same code path whether per_channel comes from `analyse` or from a
        spike decoder.
        """
        frequencies, decay_rates = self._params_to_values(params)
        weights = params["channel_weights"]
        per_channel = jnp.asarray(coefficients)
        n_samples = per_channel.shape[-1]
        synths = []
        for i, (freq, decay_rate) in enumerate(zip(frequencies, decay_rates)):
            t_kernel = (jnp.arange(n_samples) - n_samples // 2) * self.dt
            kernel = self.szu_kernel(t_kernel, freq, decay_rate)
            synth_k = fft_convolve(per_channel[i], kernel, mode="same") * self.dt
            synths.append(weights[i] * jnp.real(synth_k))
        return jnp.sum(jnp.array(synths), axis=0)

    def get_readable_params(self, params=None):
        """
        Return human-readable (non-log) parameters.

        Args:
            params: Parameter dictionary (uses self.params if None)

        Returns:
            Dictionary with frequencies, decay_rates, time_constants, and weights
        """
        if params is None:
            params = self.params

        frequencies, decay_rates = self._params_to_values(params)
        time_constants = 1.0 / decay_rates

        return {
            "frequencies": frequencies,
            "decay_rates": decay_rates,
            "time_constants": time_constants,
            "weights": params["channel_weights"],
        }

    def time_scales(self):
        """
        Return characteristic time scales (time constants tau) for each channel.

        Returns:
            Array of time constants (tau = 1/decay_rate) in seconds
        """
        _, decay_rates = self._params_to_values(self.params)
        return 1.0 / decay_rates

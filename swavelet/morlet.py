import numpy as np
import jax.numpy as jnp
import pywt

from . import wavelet


class MorletWavelet(wavelet.Wavelet):
    """
    Morlet wavelet using PyWavelets' CWT implementation.

    Uses pywt.cwt with the complex Morlet wavelet ('cmor') for analysis.
    Reconstruction via lstsq on per-channel CWT coefficient time series.

    Parameters are log-spaced scales covering the frequency range [f_min, f_max],
    where f_max defaults to Nyquist/2 and f_min to a few Hz.
    """

    # Forward wraps pywt.cwt (pure numpy) — cannot be traced by jax.jit.
    is_jittable = False

    def __init__(self, n_channels: int = 3, dt: float = 0.01,
                 f_min: float = 3.0, f_max: float = None,
                 bandwidth: float = 1.5, center_frequency: float = 1.0):
        self.n_channels = n_channels
        self.dt = dt
        self.bandwidth = bandwidth
        self.center_frequency = center_frequency
        self.wavelet_name = f"cmor{bandwidth}-{center_frequency}"

        # Frequency range: default f_max = Nyquist / 2
        nyquist = 1.0 / (2.0 * dt)
        if f_max is None:
            f_max = nyquist / 2.0
        f_max = min(f_max, nyquist * 0.9)
        self.f_min = f_min
        self.f_max = f_max

        # Log-spaced scales (high scale = low freq)
        # scale = center_frequency / (freq * dt)
        scale_min = center_frequency / (f_max * dt)
        scale_max = center_frequency / (f_min * dt)
        self.scales = np.geomspace(scale_min, scale_max, n_channels)

        # Corresponding frequencies for reference
        self.frequencies = pywt.scale2frequency(
            self.wavelet_name, self.scales, precision=12
        ) / dt

        self.params = {
            "channel_weights": jnp.ones(n_channels) / n_channels,
        }

    def __call__(self, params, signal):
        """
        Apply Morlet CWT analysis.

        Returns:
            Complex CWT coefficients of shape (n_channels, n_samples).
        """
        sig_np = np.array(signal)

        # CWT analysis via pywt
        coeffs, _ = pywt.cwt(
            sig_np, self.scales, self.wavelet_name,
            sampling_period=self.dt, method="fft",
        )
        return jnp.array(coeffs)
    
    def reconstruct(self, params, coefficients):
        # Reconstruction: weighted sum of real parts of CWT coefficients
        reconstruction = None
        weights = params["channel_weights"]
        for i in range(self.n_channels):
            if reconstruction is None:
                reconstruction = weights[i] * jnp.real(coefficients[i])
            else:
                reconstruction = reconstruction + weights[i] * jnp.real(coefficients[i])

        return reconstruction

    def get_readable_params(self, params=None):
        if params is None:
            params = self.params
        return {
            "frequencies": self.frequencies,
            "scales": self.scales,
            "weights": params["channel_weights"],
        }

    def time_scales(self):
        return jnp.array(1.0 / self.frequencies)

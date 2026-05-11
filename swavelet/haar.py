import numpy as np
import jax.numpy as jnp
import pywt

from . import wavelet


class HaarWavelet(wavelet.Wavelet):
    """Critically-sampled Haar DWT via PyWavelets.

    Uses pywt.wavedec / pywt.waverec for analysis and synthesis.
    A full decomposition (level=None) produces exactly len(signal) coefficients
    (Nyquist cost) and reconstructs with RMSE~=0.

    n_channels maps to the DWT decomposition level; None uses the maximum
    possible depth for the given signal length.  params is an empty dict --
    there are no learnable parameters.
    """

    is_dwt = True

    def __init__(self, n_channels: int = None, dt: float = 1.0):
        self.n_channels = n_channels   # DWT level; None = full decomposition
        self.dt = dt
        self.params = {}

    def __call__(self, params, signal):
        """Haar DWT analysis.

        Returns the level-wise coefficient list `[cA_n, cD_n, ..., cD_1]`
        produced by `pywt.wavedec` (variable-length per level). Pass it back
        through `reconstruct` for the exact inverse (RMSE ~ 0).
        """
        sig_np = np.array(signal)
        return pywt.wavedec(sig_np, "haar", level=self.n_channels, mode="periodization")

    def reconstruct(self, params, coefficients):
        n = sum(len(c) for c in coefficients)
        recon_np = pywt.waverec(coefficients, "haar", mode="periodization")[:n]
        return jnp.array(recon_np)

    def time_scales(self):
        return jnp.array([])

    def get_readable_params(self, params=None):
        return {"level": self.n_channels}

"""
Cross-validation of non-spiking wavelet implementations against PyWavelets.

For wavelets with a PyWavelets equivalent we verify that our lstsq-based
reconstruction RMSE is in a similar ballpark. This catches implementation
errors in filter shapes without requiring a bit-exact match (parameterisations
and normalisation conventions will differ).

Wavelets covered
----------------
- Haar   : our filter bank  vs  pywt DWT ('haar') -- reference is exact reconstruction
- Morlet : our filter bank  vs  pywt CWT ('morl') -- both use lstsq, assert comparable
- DoG    : our filter bank  vs  pywt CWT ('gaus1') -- 1st Gaussian derivative ~= DoG

Szu is omitted (no PyWavelets equivalent).
Shannon is omitted ('shan' in pywt uses a different bandwidth/centre-frequency
parameterisation that makes scale matching ambiguous).
"""
import numpy as np
import pytest
import jax.numpy as jnp

pywt = pytest.importorskip("pywt")

from swavelet.haar import HaarWavelet
from swavelet.morlet import MorletWavelet
from swavelet.dog import DifferenceOfGaussiansWavelet


# ---------------------------------------------------------------------------
# Shared signal and helpers
# ---------------------------------------------------------------------------

# 1 s at 360 Hz — matches the MITBIH chunk size used in experiments
N = 360
DT = 1.0 / 360


def make_signal(seed=42):
    """Z-scored synthetic signal: two sinusoids + light noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) * DT
    s = (np.sin(2 * np.pi * 5 * t)
         + 0.5 * np.sin(2 * np.pi * 15 * t)
         + 0.2 * rng.standard_normal(N))
    s = (s - s.mean()) / s.std()
    return s


def lstsq_rmse(columns, signal):
    """
    Reconstruct *signal* from a list of filter-bank output vectors via lstsq.

    Each element of *columns* is a 1-D array of length N.  Column-normalises
    the matrix for numerical stability (mirrors evaluate_nonspiking).
    """
    A = np.column_stack([np.asarray(c).ravel() for c in columns])
    norms = np.linalg.norm(A, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    A_n = A / norms
    w, _, _, _ = np.linalg.lstsq(A_n, signal, rcond=None)
    recon = A_n @ w
    return float(np.sqrt(np.mean((signal - recon) ** 2)))


def our_lstsq_rmse(wavelet_obj, signal):
    """
    Run our wavelet forward pass one channel at a time (one-hot weights) and
    reconstruct via lstsq -- replicates the logic in evaluate_nonspiking.
    """
    sig_jax = jnp.array(signal)
    n_ch = wavelet_obj.n_channels
    columns = []
    for i in range(n_ch):
        p = dict(wavelet_obj.params)
        p["channel_weights"] = jnp.zeros(n_ch).at[i].set(1.0)
        recon_i = wavelet_obj.reconstruct(p, wavelet_obj(p, sig_jax))
        columns.append(np.array(recon_i))
    return lstsq_rmse(columns, signal)


# ---------------------------------------------------------------------------
# Haar
# ---------------------------------------------------------------------------

class TestHaarVsPyWavelets:
    """
    PyWavelets DWT gives exact reconstruction (RMSE ~= 0).
    Our filter bank is overcomplete but should still reconstruct a smooth
    signal well when we use enough scales.
    """

    def test_pywt_haar_exact_reconstruction(self):
        """Sanity check: PyWavelets Haar DWT reconstructs exactly."""
        signal = make_signal()
        coeffs = pywt.wavedec(signal, "haar")
        recon = pywt.waverec(coeffs, "haar")[:N]
        rmse = float(np.sqrt(np.mean((signal - recon) ** 2)))
        assert rmse < 1e-10, f"PyWavelets Haar RMSE {rmse:.2e} should be ~0"

    def test_our_haar_lstsq_reconstruction(self):
        """
        Our Haar filter bank + lstsq should reconstruct a smooth signal with
        RMSE well below 0.3 (for a z-scored signal, 0.3 means capturing > 90 %
        of variance).
        """
        signal = make_signal()
        w = HaarWavelet(n_channels=8, dt=DT)
        rmse = our_lstsq_rmse(w, signal)
        assert rmse < 0.3, (
            f"Haar lstsq RMSE {rmse:.4f} too high — filter bank may be broken"
        )

    def test_more_channels_improves_haar_reconstruction(self):
        """More Haar scales should give lower or equal lstsq RMSE."""
        signal = make_signal()
        rmse_small = our_lstsq_rmse(HaarWavelet(n_channels=4, dt=DT), signal)
        rmse_large = our_lstsq_rmse(HaarWavelet(n_channels=8, dt=DT), signal)
        assert rmse_large <= rmse_small + 1e-6, (
            f"More channels should not hurt: n=4 RMSE={rmse_small:.4f}, "
            f"n=8 RMSE={rmse_large:.4f}"
        )


# ---------------------------------------------------------------------------
# Morlet
# ---------------------------------------------------------------------------

class TestMorletVsPyWavelets:
    """
    Both our Morlet and PyWavelets 'morl' CWT are applied as filter banks and
    reconstructed via lstsq at matching centre frequencies.  We expect similar
    RMSE; a large gap would indicate a bug in our kernel shape.
    """

    # morl central frequency (cycles per unit scale) as reported by pywt
    MORL_CF = pywt.central_frequency("morl")

    def _scales_for_freqs(self, freqs_hz):
        """Convert Hz to PyWavelets CWT scale for the 'morl' wavelet."""
        return self.MORL_CF / (np.asarray(freqs_hz) * DT)

    def _pywt_morlet_rmse(self, signal, freqs_hz):
        scales = self._scales_for_freqs(freqs_hz)
        coeffs, _ = pywt.cwt(signal, scales, "morl")  # (n_scales, N), real
        return lstsq_rmse(coeffs, signal)

    def test_our_morlet_comparable_to_pywt(self):
        """
        Our Morlet lstsq RMSE should be no more than 5x worse than PyWavelets
        CWT lstsq at the same centre frequencies.
        """
        signal = make_signal()
        freqs = np.linspace(3.0, 20.0, 8)

        pywt_rmse = self._pywt_morlet_rmse(signal, freqs)
        w = MorletWavelet(n_channels=8, dt=DT)
        our_rmse = our_lstsq_rmse(w, signal)

        assert our_rmse <= 5 * pywt_rmse + 0.05, (
            f"Our Morlet RMSE {our_rmse:.4f} >> PyWavelets RMSE {pywt_rmse:.4f}"
        )

    def test_pywt_morlet_reasonable_reconstruction(self):
        """Sanity check: PyWavelets Morlet CWT lstsq should achieve low RMSE."""
        signal = make_signal()
        freqs = np.linspace(3.0, 20.0, 8)
        rmse = self._pywt_morlet_rmse(signal, freqs)
        assert rmse < 0.3, (
            f"PyWavelets Morlet RMSE {rmse:.4f} unexpectedly high"
        )


# ---------------------------------------------------------------------------
# DoG
# ---------------------------------------------------------------------------

class TestDoGVsPyWavelets:
    """
    DoG (Difference of Gaussians) is approximated by PyWavelets 'gaus1'
    (first derivative of a Gaussian), the closest available equivalent.
    """

    GAUS1_CF = pywt.central_frequency("gaus1")

    def _scales_for_sigmas(self, sigmas_s):
        """
        Convert Gaussian sigma (seconds) to PyWavelets CWT scale for 'gaus1'.
        pywt scale ~= sigma / DT  (gaus1 is unit-sigma at scale=1).
        """
        return np.asarray(sigmas_s) / DT

    def _pywt_dog_rmse(self, signal, sigmas_s):
        scales = self._scales_for_sigmas(sigmas_s)
        scales = np.clip(scales, 1.0, None)
        coeffs, _ = pywt.cwt(signal, scales, "gaus1")  # real
        return lstsq_rmse(coeffs, signal)

    def test_our_dog_comparable_to_pywt(self):
        """
        Our DoG lstsq RMSE should be within 5x of PyWavelets gaus1 CWT lstsq
        at matching scales.
        """
        signal = make_signal()
        n_ch = 8
        sigma_max = 0.05  # seconds — matches experiment default for MITBIH
        c = 2.0 ** 0.5
        sigmas = sigma_max / (c ** np.arange(n_ch - 1, -1, -1))

        pywt_rmse = self._pywt_dog_rmse(signal, sigmas)
        w = DifferenceOfGaussiansWavelet(n_channels=n_ch, dt=DT, sigma_max=sigma_max)
        our_rmse = our_lstsq_rmse(w, signal)

        assert our_rmse <= 5 * pywt_rmse + 0.05, (
            f"Our DoG RMSE {our_rmse:.4f} >> PyWavelets gaus1 RMSE {pywt_rmse:.4f}"
        )

    def test_pywt_gaus1_reasonable_reconstruction(self):
        """Sanity check: PyWavelets gaus1 CWT lstsq should achieve low RMSE."""
        signal = make_signal()
        n_ch = 8
        sigma_max = 0.05
        c = 2.0 ** 0.5
        sigmas = sigma_max / (c ** np.arange(n_ch - 1, -1, -1))
        rmse = self._pywt_dog_rmse(signal, sigmas)
        assert rmse < 0.5, (
            f"PyWavelets gaus1 RMSE {rmse:.4f} unexpectedly high"
        )

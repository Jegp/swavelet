"""
Tests for HaarWavelet -- PyWavelets critically-sampled DWT implementation.

Key properties under test:
  - Perfect reconstruction: RMSE ~= 0 (up to floating-point rounding)
  - Nyquist coefficient count: total coefficients ~= len(signal)
  - Frequency selectivity: coarser levels respond to lower frequencies
  - Real-valued, finite outputs
"""
import numpy as np
import jax.numpy as jnp
import pytest

from swavelet.haar import HaarWavelet


DT = 0.01
N_SAMPLES = 2048   # power-of-2 so DWT is exactly length-preserving


def make_signal(freq_hz, n=N_SAMPLES, dt=DT):
    t = np.arange(n) * dt
    return jnp.array(np.sin(2 * np.pi * freq_hz * t))


# ---------------------------------------------------------------------------
# Forward pass basics
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape(self):
        w = HaarWavelet()
        signal = make_signal(10.0)
        coeffs = w(w.params, signal)
        recon = w.reconstruct(w.params, coeffs)
        assert recon.shape == signal.shape

    def test_reconstruction_real_valued(self):
        w = HaarWavelet()
        signal = make_signal(10.0)
        recon = w.reconstruct(w.params, w(w.params, signal))
        assert not jnp.iscomplexobj(recon)

    def test_reconstruction_finite(self):
        w = HaarWavelet()
        signal = make_signal(10.0)
        recon = w.reconstruct(w.params, w(w.params, signal))
        assert jnp.all(jnp.isfinite(recon))

    def test_coefficients_finite(self):
        w = HaarWavelet()
        signal = make_signal(10.0)
        coeffs = w(w.params, signal)
        assert all(jnp.all(jnp.isfinite(jnp.asarray(c))) for c in coeffs)

    def test_coefficients_real_valued(self):
        w = HaarWavelet()
        signal = make_signal(10.0)
        coeffs = w(w.params, signal)
        assert not any(jnp.iscomplexobj(jnp.asarray(c)) for c in coeffs)


# ---------------------------------------------------------------------------
# Perfect reconstruction
# ---------------------------------------------------------------------------

class TestPerfectReconstruction:
    @pytest.mark.parametrize("freq", [1.0, 5.0, 20.0, 45.0])
    def test_rmse_near_zero(self, freq):
        # float32 cast in jnp.array introduces ~1e-7 rounding; threshold is 1e-6
        w = HaarWavelet()
        signal = make_signal(freq)
        recon = w.reconstruct(w.params, w(w.params, signal))
        rmse = float(jnp.sqrt(jnp.mean((signal - recon) ** 2)))
        assert rmse < 1e-6, f"RMSE={rmse:.2e} for freq={freq} Hz; expected <1e-6"

    def test_reconstruction_loss_near_zero(self):
        w = HaarWavelet()
        signal = make_signal(10.0)
        recon = w.reconstruct(w.params, w(w.params, signal))
        loss = float(jnp.mean((signal - recon) ** 2))
        assert loss < 1e-12, f"Loss={loss:.2e}; expected <1e-12"

    def test_impulse_perfect_reconstruction(self):
        w = HaarWavelet()
        signal = jnp.zeros(N_SAMPLES).at[N_SAMPLES // 2].set(1.0)
        recon = w.reconstruct(w.params, w(w.params, signal))
        rmse = float(jnp.sqrt(jnp.mean((signal - recon) ** 2)))
        assert rmse < 1e-6

    def test_white_noise_perfect_reconstruction(self):
        rng = np.random.default_rng(0)
        signal = jnp.array(rng.standard_normal(N_SAMPLES))
        w = HaarWavelet()
        recon = w.reconstruct(w.params, w(w.params, signal))
        rmse = float(jnp.sqrt(jnp.mean((signal - recon) ** 2)))
        assert rmse < 1e-6


# ---------------------------------------------------------------------------
# Nyquist coefficient count
# ---------------------------------------------------------------------------

class TestNyquistCost:
    def test_full_decomp_coefficient_count(self):
        """Full decomposition must produce exactly N coefficients."""
        w = HaarWavelet()   # level=None → full decomposition
        signal = make_signal(10.0)
        coeffs = w(w.params, signal)
        total = sum(len(c) for c in coeffs)
        assert total == N_SAMPLES, (
            f"Expected {N_SAMPLES} coefficients, got {total}"
        )

    @pytest.mark.parametrize("level", [1, 2, 4, 8])
    def test_partial_decomp_coefficient_count(self, level):
        """Partial decomposition also produces exactly N coefficients."""
        w = HaarWavelet(n_channels=level)
        signal = make_signal(10.0)
        coeffs = w(w.params, signal)
        total = sum(len(c) for c in coeffs)
        assert total == N_SAMPLES, (
            f"level={level}: expected {N_SAMPLES} coefficients, got {total}"
        )


# ---------------------------------------------------------------------------
# Frequency selectivity
# ---------------------------------------------------------------------------

class TestFrequencySelectivity:
    def _level_energy(self, signal, level):
        """Energy in the detail coefficients at a given DWT level."""
        import pywt
        sig_np = np.array(signal)
        coeffs = pywt.wavedec(sig_np, "haar")
        # coeffs[0] = approximation, coeffs[1..] = details finest→coarsest
        # level 1 = finest details (coeffs[-1]), level n = coarsest (coeffs[1])
        detail_levels = list(reversed(coeffs[1:]))   # finest first (pywt orders coarsest first)
        # Map 1-indexed level to list index (level 1 = finest = index 0)
        idx = level - 1
        if idx >= len(detail_levels):
            return 0.0
        return float(np.sum(detail_levels[idx] ** 2))

    def test_high_freq_dominates_fine_levels(self):
        """A high-frequency signal should carry more energy in fine DWT levels."""
        sig_high = make_signal(40.0)   # near Nyquist for dt=0.01
        energy_fine = self._level_energy(sig_high, level=1)
        energy_coarse = self._level_energy(sig_high, level=5)
        assert energy_fine > energy_coarse, (
            f"High-freq: fine energy {energy_fine:.4f} should exceed "
            f"coarse energy {energy_coarse:.4f}"
        )

    def test_low_freq_dominates_coarse_levels(self):
        """A low-frequency signal should carry more energy in coarse DWT levels."""
        sig_low = make_signal(0.5)
        energy_fine = self._level_energy(sig_low, level=1)
        energy_coarse = self._level_energy(sig_low, level=5)
        assert energy_coarse > energy_fine, (
            f"Low-freq: coarse energy {energy_coarse:.4f} should exceed "
            f"fine energy {energy_fine:.4f}"
        )

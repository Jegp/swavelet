import unittest

import jax
import jax.numpy as jnp

from swavelet.dog import (
    gaussian_kernel,
    gaussian_filter,
    DifferenceOfGaussiansWavelet,
)


# --- Gaussian kernel tests ---

class TestGaussianKernel:
    def test_unit_sum(self):
        kernel = gaussian_kernel(sigma=0.05, dt=0.001)
        assert jnp.allclose(jnp.sum(kernel), 1.0, atol=1e-6)

    def test_symmetry(self):
        kernel = gaussian_kernel(sigma=0.05, dt=0.001)
        assert jnp.allclose(kernel, kernel[::-1], atol=1e-8)

    def test_peak_at_center(self):
        kernel = gaussian_kernel(sigma=0.05, dt=0.001)
        center = len(kernel) // 2
        assert kernel[center] == jnp.max(kernel)

    def test_wider_sigma_wider_kernel(self):
        k_narrow = gaussian_kernel(sigma=0.02, dt=0.001)
        k_wide = gaussian_kernel(sigma=0.08, dt=0.001)
        assert len(k_wide) > len(k_narrow)

    def test_non_negative(self):
        kernel = gaussian_kernel(sigma=0.05, dt=0.001)
        assert jnp.all(kernel >= 0)


# --- Gaussian filter tests ---

class TestGaussianFilter:
    def test_preserves_dc(self):
        """A constant signal should be unchanged by Gaussian filtering."""
        signal = jnp.ones(2000) * 3.0
        filtered = gaussian_filter(signal, sigma=0.05, dt=0.001)
        margin = 250
        assert jnp.allclose(filtered[margin:-margin], 3.0, atol=1e-4)

    def test_output_length(self):
        signal = jnp.ones(1000)
        filtered = gaussian_filter(signal, sigma=0.05, dt=0.001)
        assert filtered.shape == signal.shape

    def test_smoothing_reduces_variance(self):
        """Filtering white noise should reduce variance."""
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (2000,))
        filtered = gaussian_filter(noise, sigma=0.05, dt=0.001)
        assert jnp.var(filtered) < jnp.var(noise)

    def test_larger_sigma_more_smoothing(self):
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (2000,))
        f1 = gaussian_filter(noise, sigma=0.02, dt=0.001)
        f2 = gaussian_filter(noise, sigma=0.08, dt=0.001)
        assert jnp.var(f2) < jnp.var(f1)


# --- Impulse response tests ---

class TestImpulseResponse:
    def test_impulse_produces_nonzero_differences(self):
        """An impulse should produce nonzero DoG difference channels."""
        n = 2000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfGaussiansWavelet(n_channels=4, dt=dt, sigma_max=0.1)
        encoding = w(w.params, impulse)

        # Each difference channel should have meaningful amplitude
        for i in range(w.n_channels - 1):
            peak = jnp.max(jnp.abs(encoding[i]))
            assert peak > 0.1, f"Difference channel {i} peak {peak} too small"

    def test_impulse_response_amplitude_ordering(self):
        """Finer scales should have higher peak amplitude for an impulse."""
        n = 4000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfGaussiansWavelet(n_channels=4, dt=dt, sigma_max=0.1)
        encoding = w(w.params, impulse)

        # Finer scales (lower index) = narrower Gaussian = higher peak for impulse
        peaks = [float(jnp.max(jnp.abs(encoding[i]))) for i in range(w.n_channels - 1)]
        for i in range(len(peaks) - 1):
            assert peaks[i] > peaks[i + 1], (
                f"Expected finer scale {i} peak ({peaks[i]:.4f}) > "
                f"coarser scale {i+1} peak ({peaks[i+1]:.4f})"
            )

    def test_impulse_base_channel_is_lowpass(self):
        """Base channel (coarsest Gaussian) should be a smooth lowpass response."""
        n = 2000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfGaussiansWavelet(n_channels=4, dt=dt, sigma_max=0.1)
        encoding = w(w.params, impulse)

        base = encoding[-1]
        # Should have a single smooth peak
        assert jnp.max(base) > 0
        # Peak should be near the impulse location
        peak_idx = jnp.argmax(base)
        assert abs(int(peak_idx) - n // 2) < 200

    def test_difference_channels_are_zero_mean(self):
        """DoG difference channels should have approximately zero integral (bandpass)."""
        n = 4000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfGaussiansWavelet(n_channels=4, dt=dt, sigma_max=0.1)
        encoding = w(w.params, impulse)

        for i in range(w.n_channels - 1):
            channel_sum = jnp.sum(encoding[i]) * dt
            peak = jnp.max(jnp.abs(encoding[i]))
            # Sum should be small relative to peak
            assert abs(float(channel_sum)) < 0.1 * float(peak), (
                f"Channel {i} sum {channel_sum:.4f} not near zero (peak={peak:.4f})"
            )


# --- Wavelet reconstruction tests ---

class TestReconstruction:
    def test_reconstruction_preserves_signal_shape(self):
        signal = jnp.sin(jnp.linspace(0, 10, 1000))
        w = DifferenceOfGaussiansWavelet(n_channels=3, dt=0.01, sigma_max=0.2)
        recon = w.reconstruct(w.params, w(w.params, signal))
        assert recon.shape == signal.shape

    def test_reconstruction_loss_finite(self):
        signal = jnp.sin(jnp.linspace(0, 10, 1000))
        w = DifferenceOfGaussiansWavelet(n_channels=3, dt=0.01, sigma_max=0.2)
        recon = w.reconstruct(w.params, w(w.params, signal))
        loss = jnp.mean((signal - recon) ** 2)
        assert jnp.isfinite(loss)

    def test_unit_weights_reconstruction(self):
        """With unit weights, reconstruction should capture most of a smooth signal."""
        dt = 0.001
        t = jnp.arange(0, 2.0, dt)
        signal = jnp.sin(2 * jnp.pi * 5 * t)

        w = DifferenceOfGaussiansWavelet(n_channels=4, dt=dt, sigma_max=0.05)
        params = w.params.copy()
        params["channel_weights"] = jnp.ones(4)

        recon = w.reconstruct(params, w(params, signal))
        # Ignore edges
        s = signal[200:-200]
        r = recon[200:-200]
        correlation = jnp.abs(jnp.corrcoef(s, r)[0, 1])
        assert correlation > 0.5, f"Reconstruction correlation {correlation:.3f} too low"


# --- Parameter tests ---

class TestParameters:
    def test_time_scales_ordered(self):
        w = DifferenceOfGaussiansWavelet(n_channels=5, dt=0.01, sigma_max=0.2)
        scales = w.time_scales()
        # Should be monotonically increasing (finest to coarsest)
        for i in range(len(scales) - 1):
            assert scales[i] < scales[i + 1]

    def test_readable_params(self):
        w = DifferenceOfGaussiansWavelet(n_channels=3, dt=0.01, sigma_max=0.2)
        rp = w.get_readable_params()
        assert "sigmas" in rp
        assert "weights" in rp
        # K-1 smoothing scales: implicit sigma_0 = 0 contributes the raw signal.
        assert len(rp["sigmas"]) == 2

    def test_different_c_changes_scale_spacing(self):
        w1 = DifferenceOfGaussiansWavelet(n_channels=3, sigma_max=0.2, c=2.0**0.5)
        w2 = DifferenceOfGaussiansWavelet(n_channels=3, sigma_max=0.2, c=2.0)
        s1 = w1.time_scales()
        s2 = w2.time_scales()
        # Coarsest should be the same
        assert jnp.allclose(s1[-1], s2[-1])
        # Finer scales should differ
        assert not jnp.allclose(s1[0], s2[0])


# --- Gradient tests ---

class TestGradients:
    def test_loss_has_finite_gradients(self):
        """Reconstruction loss should produce finite gradients for all params."""
        dt = 0.001
        t = jnp.arange(0, 1.0, dt)
        signal = jnp.sin(2 * jnp.pi * 10 * t)

        w = DifferenceOfGaussiansWavelet(n_channels=3, dt=dt, sigma_max=0.05)
        grad_fn = jax.grad(lambda p, s: jnp.mean((s - w.reconstruct(p, w(p, s))) ** 2))
        grads = grad_fn(w.params, signal)

        for key, g in grads.items():
            assert jnp.all(jnp.isfinite(g)), f"Non-finite gradient for {key}"


# --- Telescoping reconstruction contract ---

class TestTelescopeReconstruction(unittest.TestCase):
    """Contract for the bandpass + lowpass channel layout.

    n_channels = K total output channels: K-1 bandpass differences and one
    lowpass residual. The first bandpass channel is Delta L_1 = G(sigma_1)*f - f,
    using the raw signal as the implicit zero-scale level L_0 = f. With unit
    weights the telescoping sum then exactly recovers the input signal.
    """

    def test_n_channels_must_be_at_least_two(self):
        """n_channels < 2 leaves no room for both a bandpass and a lowpass."""
        with self.assertRaises(ValueError):
            DifferenceOfGaussiansWavelet(n_channels=1, dt=0.001, sigma_max=0.05)

    def test_analyse_output_shape(self):
        """analyse returns (n_channels, T): K-1 bandpass + 1 lowpass."""
        dt = 0.001
        n = 1000
        signal = jnp.sin(jnp.linspace(0, 10, n))
        for K in (2, 3, 5):
            w = DifferenceOfGaussiansWavelet(n_channels=K, dt=dt, sigma_max=0.05)
            encoding = w(w.params, signal)
            self.assertEqual(
                encoding.shape, (K, n),
                f"K={K}: expected ({K},{n}), got {encoding.shape}",
            )

    def test_unit_weights_reconstruction_is_exact(self):
        """With default unit weights the telescope must recover f exactly,
        not G(sigma_finest)*f. This is a regression test for the missing
        Delta L_1 = G(sigma_1)*f - f channel."""
        dt = 0.001
        t = jnp.arange(0, 2.0, dt)
        signal = jnp.sin(2 * jnp.pi * 5 * t) + 0.3 * jnp.cos(2 * jnp.pi * 13 * t)

        w = DifferenceOfGaussiansWavelet(n_channels=4, dt=dt, sigma_max=0.05)
        recon = w.reconstruct(w.params, w(w.params, signal))
        # DoG (Gaussian smoothing, mode='same') -- edge effects on both sides.
        margin = 250
        s = signal[margin:-margin]
        r = recon[margin:-margin]
        max_err = float(jnp.max(jnp.abs(s - r)))
        self.assertTrue(
            jnp.allclose(s, r, atol=1e-4),
            f"max abs error: {max_err:.3e}",
        )

    def test_telescope_invariant_holds(self):
        """analyse output must satisfy lowpass - sum(bandpass) == signal,
        directly mirroring Tony's pytempscsp reconstruction identity."""
        dt = 0.001
        t = jnp.arange(0, 2.0, dt)
        signal = jnp.sin(2 * jnp.pi * 5 * t)

        w = DifferenceOfGaussiansWavelet(n_channels=4, dt=dt, sigma_max=0.05)
        encoding = w(w.params, signal)
        bandpass = encoding[:-1]
        lowpass = encoding[-1]
        recon = lowpass - jnp.sum(bandpass, axis=0)
        margin = 250
        self.assertTrue(
            jnp.allclose(signal[margin:-margin], recon[margin:-margin], atol=1e-4)
        )

    def test_first_bandpass_uses_raw_signal_as_L0(self):
        """The first bandpass channel must reference f directly, not a smoothed
        version. Verified by the DC test: for a constant input, the first
        bandpass channel must be (approximately) zero, which only holds when
        L_0 = f and G(sigma_1)*const = const."""
        dt = 0.001
        n = 2000
        signal = jnp.ones(n) * 2.5

        w = DifferenceOfGaussiansWavelet(n_channels=4, dt=dt, sigma_max=0.05)
        encoding = w(w.params, signal)
        # Symmetric kernel boundary effects on both sides.
        margin = 300
        first_bandpass = encoding[0, margin:-margin]
        max_dev = float(jnp.max(jnp.abs(first_bandpass)))
        self.assertTrue(
            jnp.allclose(first_bandpass, 0.0, atol=1e-4),
            f"First bandpass on a DC signal should be ~0 in the interior, "
            f"got max |x|={max_dev:.3e}",
        )


if __name__ == "__main__":
    unittest.main()

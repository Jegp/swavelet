import unittest

import jax
import jax.numpy as jnp
import pytest

from swavelet.dot import DifferenceOfTimeCausalKernelsWavelet


# --- Impulse response tests ---

class TestImpulseResponse:
    def test_impulse_produces_nonzero_differences(self):
        """An impulse should produce nonzero DoT difference channels."""
        n = 2000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=4, dt=dt, mu_max=0.1)
        encoding = w(w.params, impulse)

        for i in range(w.n_channels - 1):
            peak = jnp.max(jnp.abs(encoding[i]))
            assert peak > 0.1, f"Difference channel {i} peak {peak} too small"

    def test_impulse_response_amplitude_ordering(self):
        """Finer scales should have higher peak amplitude for an impulse."""
        n = 4000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=4, dt=dt, mu_max=0.1)
        encoding = w(w.params, impulse)

        peaks = [float(jnp.max(jnp.abs(encoding[i]))) for i in range(w.n_channels - 1)]
        for i in range(len(peaks) - 1):
            assert peaks[i] > peaks[i + 1], (
                f"Expected finer scale {i} peak ({peaks[i]:.4f}) > "
                f"coarser scale {i+1} peak ({peaks[i+1]:.4f})"
            )

    def test_impulse_base_channel_is_lowpass(self):
        """Base channel (coarsest cascade) should be a smooth lowpass response."""
        n = 2000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=4, dt=dt, mu_max=0.1)
        encoding = w(w.params, impulse)

        base = encoding[-1]
        assert jnp.max(base) > 0
        # Causal: peak should be at or after the impulse
        peak_idx = jnp.argmax(base)
        assert int(peak_idx) >= n // 2

    def test_difference_channels_are_bandpass(self):
        """DoT difference channels should have approximately zero integral (bandpass)."""
        n = 4000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[100].set(1.0 / dt)  # Early impulse so response can decay

        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=4, dt=dt, mu_max=0.1)
        encoding = w(w.params, impulse)

        for i in range(w.n_channels - 1):
            channel_sum = jnp.sum(encoding[i]) * dt
            peak = jnp.max(jnp.abs(encoding[i]))
            assert abs(float(channel_sum)) < 0.15 * float(peak), (
                f"Channel {i} sum {channel_sum:.4f} not near zero (peak={peak:.4f})"
            )

    def test_causal_response(self):
        """DoT channels should be causal -- zero before the impulse."""
        n = 2000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=4, dt=dt, mu_max=0.1)
        encoding = w(w.params, impulse)

        for i in range(w.n_channels):
            # FFT convolution leaks ~1e-6 numerical noise before the impulse.
            # Causality to 1e-5 is enough to exclude a real violation.
            assert jnp.allclose(encoding[i, :n // 2], 0.0, atol=1e-5), (
                f"Channel {i} has non-zero response before impulse"
            )

    def test_cascade_depth_affects_response(self):
        """Different target cascade depths should produce different responses."""
        n = 2000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        # mu_max=0.1 is large relative to dt=0.001, so both N_max=3 and N_max=7
        # fit comfortably under the stability floor across all channels. The
        # per-channel N_k will saturate at cascade_depth_max for both wavelets,
        # so the impulse responses differ as intended.
        w1 = DifferenceOfTimeCausalKernelsWavelet(n_channels=3, dt=dt, mu_max=0.1, cascade_depth_max=3)
        w2 = DifferenceOfTimeCausalKernelsWavelet(n_channels=3, dt=dt, mu_max=0.1, cascade_depth_max=7)

        enc1 = w1(w1.params, impulse)
        enc2 = w2(w2.params, impulse)

        assert not jnp.allclose(enc1[0], enc2[0], atol=1e-3)


# --- Wavelet reconstruction tests ---

class TestReconstruction:
    def test_reconstruction_preserves_signal_shape(self):
        signal = jnp.sin(jnp.linspace(0, 10, 1000))
        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=3, dt=0.01, mu_max=0.2)
        recon = w.reconstruct(w.params, w(w.params, signal))
        assert recon.shape == signal.shape

    def test_reconstruction_loss_finite(self):
        signal = jnp.sin(jnp.linspace(0, 10, 1000))
        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=3, dt=0.01, mu_max=0.2)
        recon = w.reconstruct(w.params, w(w.params, signal))
        loss = jnp.mean((signal - recon) ** 2)
        assert jnp.isfinite(loss)

    def test_unit_weights_reconstruction(self):
        """With unit weights, reconstruction should capture the signal shape."""
        dt = 0.001
        t = jnp.arange(0, 2.0, dt)
        signal = jnp.sin(2 * jnp.pi * 5 * t)

        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=4, dt=dt, mu_max=0.05)
        params = w.params.copy()
        params["channel_weights"] = jnp.ones(4)

        recon = w.reconstruct(params, w(params, signal))
        s = signal[200:-200]
        r = recon[200:-200]
        correlation = jnp.abs(jnp.corrcoef(s, r)[0, 1])
        assert correlation > 0.5, f"Reconstruction |correlation| {correlation:.3f} too low"


# --- Parameter tests ---

class TestParameters:
    def test_time_scales_ordered(self):
        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=5, dt=0.01, mu_max=0.2)
        scales = w.time_scales()
        for i in range(len(scales) - 1):
            assert scales[i] < scales[i + 1]

    def test_readable_params(self):
        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=3, dt=0.01, mu_max=0.2)
        rp = w.get_readable_params()
        assert "mus" in rp
        assert "weights" in rp
        assert "cascade_depth_max" in rp
        assert "cascade_depths" in rp
        # K-1 smoothing scales: implicit mu_0 = 0 contributes the raw signal.
        assert len(rp["cascade_depths"]) == 2

    def test_different_c_changes_scale_spacing(self):
        w1 = DifferenceOfTimeCausalKernelsWavelet(n_channels=3, mu_max=0.2, c=2.0)
        w2 = DifferenceOfTimeCausalKernelsWavelet(n_channels=3, mu_max=0.2, c=3.0)
        s1 = w1.time_scales()
        s2 = w2.time_scales()
        assert jnp.allclose(s1[-1], s2[-1])
        assert not jnp.allclose(s1[0], s2[0])


# --- Per-channel cascade depth (N_k) tests ---

class TestAdaptiveCascadeDepth:
    def test_length_matches_n_smooth(self):
        """cascade_depths covers the K-1 smoothing scales mu_1..mu_{K-1}."""
        w = DifferenceOfTimeCausalKernelsWavelet(
            n_channels=4, dt=0.001, mu_max=0.1, cascade_depth_max=7)
        assert len(w.cascade_depths) == 3

    def test_all_channels_saturate_when_mu_is_large(self):
        """When every mu_k is comfortably above the per-stage stability floor,
        every channel should hit the target cascade_depth_max."""
        # mu_max=0.5 and c=sqrt(2) keep mu_min well above dt / ln(100).
        w = DifferenceOfTimeCausalKernelsWavelet(
            n_channels=4, dt=1e-3, mu_max=0.5, c=2.0 ** 0.5,
            cascade_depth_max=7, alpha_floor=0.01)
        assert all(d == 7 for d in w.cascade_depths)

    def test_finest_channel_shrinks_when_mu_is_small(self):
        """When mu_finest is near dt, N_k for that channel should drop below
        N_max. Coarser channels should still get larger N_k."""
        # dt=6.25e-5 (16 kHz), mu_max=2e-3, c=3.7, n_channels=5 → 4 smoothing
        # scales, mu_finest ≈ 40 μs. Per-stage floor = dt / ln(10) ≈ 27 μs at
        # alpha_floor=0.1, so N_k,finest = (40/27)^2 ≈ 2 < N_max.
        w = DifferenceOfTimeCausalKernelsWavelet(
            n_channels=5, dt=6.25e-5, mu_max=2e-3, c=3.684,
            cascade_depth_max=7, alpha_floor=0.1)
        depths = w.cascade_depths
        # Ordering: finest channel (index 0) uses smallest N_k
        assert depths[0] <= depths[-1]
        # Finest should be strictly below N_max at this alpha_floor
        assert depths[0] < 7
        # Coarsest should still reach N_max
        assert depths[-1] == 7

    def test_minimum_N_is_one(self):
        """Even with extremely small mu, N_k floors at 1 (pure DoE)."""
        w = DifferenceOfTimeCausalKernelsWavelet(
            n_channels=3, dt=1.0, mu_max=0.5, c=2.0,  # μ < dt everywhere
            cascade_depth_max=7, alpha_floor=0.01)
        assert all(d >= 1 for d in w.cascade_depths)

    def test_doe_equivalence_when_max_is_one(self):
        """cascade_depth_max=1 forces every channel to N_k=1 (DoE)."""
        w = DifferenceOfTimeCausalKernelsWavelet(
            n_channels=5, dt=0.001, mu_max=0.1, cascade_depth_max=1)
        assert all(d == 1 for d in w.cascade_depths)

    def test_per_channel_cascade_mus_have_correct_length(self):
        """self.cascade_mus[k] must contain exactly N_k entries."""
        w = DifferenceOfTimeCausalKernelsWavelet(
            n_channels=5, dt=6.25e-5, mu_max=2e-3, c=3.684,
            cascade_depth_max=7, alpha_floor=0.1)
        for k, N_k in enumerate(w.cascade_depths):
            assert w.cascade_mus[k].shape == (N_k,)

    def test_reconstruction_still_runs_with_mixed_N(self):
        """Perfect reconstruction plumbing must work when channels have
        different cascade depths (the telescoping sum doesn't care)."""
        w = DifferenceOfTimeCausalKernelsWavelet(
            n_channels=5, dt=6.25e-5, mu_max=2e-3, c=3.684,
            cascade_depth_max=7, alpha_floor=0.1)
        # Sanity: we actually have mixed depths, otherwise the test is trivial.
        assert len(set(w.cascade_depths)) > 1
        signal = jnp.sin(jnp.linspace(0, 10, 4096))
        recon = w.reconstruct(w.params, w(w.params, signal))
        assert recon.shape == signal.shape
        assert jnp.all(jnp.isfinite(recon))


# --- Telescoping reconstruction contract ---

class TestTelescopeReconstruction(unittest.TestCase):
    """Contract for the bandpass + lowpass channel layout.

    n_channels = K total output channels: K-1 bandpass differences and one
    lowpass residual. The first bandpass channel is Delta L_1 = L(mu_1)*f - f,
    using the raw signal as the implicit zero-scale level L_0 = f. With unit
    weights the telescoping sum then exactly recovers the input signal.
    """

    def test_n_channels_must_be_at_least_two(self):
        """n_channels < 2 leaves no room for both a bandpass and a lowpass."""
        with self.assertRaises(ValueError):
            DifferenceOfTimeCausalKernelsWavelet(n_channels=1, dt=0.001, mu_max=0.05)

    def test_analyse_output_shape(self):
        """analyse returns (n_channels, T): K-1 bandpass + 1 lowpass."""
        dt = 0.001
        n = 1000
        signal = jnp.sin(jnp.linspace(0, 10, n))
        for K in (2, 3, 5):
            w = DifferenceOfTimeCausalKernelsWavelet(n_channels=K, dt=dt, mu_max=0.05)
            encoding = w(w.params, signal)
            self.assertEqual(
                encoding.shape, (K, n),
                f"K={K}: expected ({K},{n}), got {encoding.shape}",
            )

    def test_unit_weights_reconstruction_is_exact(self):
        """With default unit weights the telescope must recover f exactly,
        not L(mu_finest)*f. This is a regression test for the missing
        Delta L_1 = L(mu_1)*f - f channel."""
        dt = 0.001
        t = jnp.arange(0, 2.0, dt)
        signal = jnp.sin(2 * jnp.pi * 5 * t) + 0.3 * jnp.cos(2 * jnp.pi * 13 * t)

        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=4, dt=dt, mu_max=0.05)
        recon = w.reconstruct(w.params, w(w.params, signal))
        # DoT is causal -- edge effects only at the start.
        margin = 200
        max_err = float(jnp.max(jnp.abs(signal[margin:] - recon[margin:])))
        self.assertTrue(
            jnp.allclose(signal[margin:], recon[margin:], atol=1e-4),
            f"max abs error: {max_err:.3e}",
        )

    def test_telescope_invariant_holds(self):
        """analyse output must satisfy lowpass - sum(bandpass) == signal,
        directly mirroring Tony's pytempscsp reconstruction identity."""
        dt = 0.001
        t = jnp.arange(0, 2.0, dt)
        signal = jnp.sin(2 * jnp.pi * 5 * t)

        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=4, dt=dt, mu_max=0.05)
        encoding = w(w.params, signal)
        bandpass = encoding[:-1]
        lowpass = encoding[-1]
        recon = lowpass - jnp.sum(bandpass, axis=0)
        margin = 200
        self.assertTrue(jnp.allclose(signal[margin:], recon[margin:], atol=1e-4))

    def test_first_bandpass_uses_raw_signal_as_L0(self):
        """The first bandpass channel must reference f directly, not a smoothed
        version. Verified by the DC test: for a constant input, the first
        bandpass channel must be (approximately) zero, which only holds when
        L_0 = f and L(mu_1)*const = const."""
        dt = 0.001
        n = 2000
        signal = jnp.ones(n) * 2.5

        w = DifferenceOfTimeCausalKernelsWavelet(n_channels=4, dt=dt, mu_max=0.05)
        encoding = w(w.params, signal)
        margin = 300  # let causal smoothing settle
        first_bandpass = encoding[0, margin:]
        max_dev = float(jnp.max(jnp.abs(first_bandpass)))
        self.assertTrue(
            jnp.allclose(first_bandpass, 0.0, atol=1e-4),
            f"First bandpass on a DC signal should be ~0 once smoothing has "
            f"settled, got max |x|={max_dev:.3e}",
        )


if __name__ == "__main__":
    unittest.main()

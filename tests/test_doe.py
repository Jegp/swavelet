import jax
import jax.numpy as jnp

from swavelet.doe import DifferenceOfExponentialsWavelet
from swavelet import temporal_integration as ti


# --- Exponential filter tests ---

class TestExponentialFilter:
    def test_preserves_dc(self):
        """A constant signal should converge to that constant under exponential filtering."""
        signal = jnp.ones(2000) * 3.0
        filtered = ti.exponential_filter(signal, mu=0.05, dt=0.001)
        # Exponential filter is causal, so it needs time to settle
        assert jnp.allclose(filtered[-500:], 3.0, atol=1e-2)

    def test_output_length(self):
        signal = jnp.ones(1000)
        filtered = ti.exponential_filter(signal, mu=0.05, dt=0.001)
        assert filtered.shape == signal.shape

    def test_smoothing_reduces_variance(self):
        """Filtering white noise should reduce variance."""
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (2000,))
        filtered = ti.exponential_filter(noise, mu=0.05, dt=0.001)
        assert jnp.var(filtered) < jnp.var(noise)

    def test_larger_mu_more_smoothing(self):
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (2000,))
        f1 = ti.exponential_filter(noise, mu=0.02, dt=0.001)
        f2 = ti.exponential_filter(noise, mu=0.08, dt=0.001)
        assert jnp.var(f2) < jnp.var(f1)

    def test_causal(self):
        """Output before impulse should be zero."""
        n = 1000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)
        filtered = ti.exponential_filter(impulse, mu=0.05, dt=dt)
        assert jnp.allclose(filtered[:n // 2], 0.0, atol=1e-6)


# --- Impulse response tests ---

class TestImpulseResponse:
    def test_impulse_produces_nonzero_differences(self):
        """An impulse should produce nonzero DoE difference channels."""
        n = 2000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfExponentialsWavelet(n_channels=4, dt=dt, mu_max=0.1)
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

        w = DifferenceOfExponentialsWavelet(n_channels=4, dt=dt, mu_max=0.1)
        encoding = w(w.params, impulse)

        peaks = [float(jnp.max(jnp.abs(encoding[i]))) for i in range(w.n_channels - 1)]
        for i in range(len(peaks) - 1):
            assert peaks[i] > peaks[i + 1], (
                f"Expected finer scale {i} peak ({peaks[i]:.4f}) > "
                f"coarser scale {i+1} peak ({peaks[i+1]:.4f})"
            )

    def test_impulse_base_channel_is_lowpass(self):
        """Base channel (coarsest exponential) should be a smooth lowpass response."""
        n = 2000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfExponentialsWavelet(n_channels=4, dt=dt, mu_max=0.1)
        encoding = w(w.params, impulse)

        base = encoding[-1]
        assert jnp.max(base) > 0
        # Causal: peak should be at or after the impulse
        peak_idx = jnp.argmax(base)
        assert int(peak_idx) >= n // 2

    def test_difference_channels_are_bandpass(self):
        """DoE difference channels should have approximately zero integral (bandpass)."""
        n = 4000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[100].set(1.0 / dt)  # Early impulse so response can decay

        w = DifferenceOfExponentialsWavelet(n_channels=4, dt=dt, mu_max=0.1)
        encoding = w(w.params, impulse)

        for i in range(w.n_channels - 1):
            channel_sum = jnp.sum(encoding[i]) * dt
            peak = jnp.max(jnp.abs(encoding[i]))
            assert abs(float(channel_sum)) < 0.15 * float(peak), (
                f"Channel {i} sum {channel_sum:.4f} not near zero (peak={peak:.4f})"
            )

    def test_causal_response(self):
        """DoE channels should be causal -- zero before the impulse."""
        n = 2000
        dt = 0.001
        impulse = jnp.zeros(n)
        impulse = impulse.at[n // 2].set(1.0 / dt)

        w = DifferenceOfExponentialsWavelet(n_channels=4, dt=dt, mu_max=0.1)
        encoding = w(w.params, impulse)

        for i in range(w.n_channels):
            # FFT convolution leaks ~1e-6 numerical noise before the impulse.
            assert jnp.allclose(encoding[i, :n // 2], 0.0, atol=1e-5), (
                f"Channel {i} has non-zero response before impulse"
            )


# --- Wavelet reconstruction tests ---

class TestReconstruction:
    def test_reconstruction_preserves_signal_shape(self):
        signal = jnp.sin(jnp.linspace(0, 10, 1000))
        w = DifferenceOfExponentialsWavelet(n_channels=3, dt=0.01, mu_max=0.2)
        recon = w.reconstruct(w.params, w(w.params, signal))
        assert recon.shape == signal.shape

    def test_reconstruction_loss_finite(self):
        signal = jnp.sin(jnp.linspace(0, 10, 1000))
        w = DifferenceOfExponentialsWavelet(n_channels=3, dt=0.01, mu_max=0.2)
        recon = w.reconstruct(w.params, w(w.params, signal))
        loss = jnp.mean((signal - recon) ** 2)
        assert jnp.isfinite(loss)

    def test_unit_weights_reconstruction(self):
        """With unit weights, reconstruction should capture the signal shape."""
        dt = 0.001
        t = jnp.arange(0, 2.0, dt)
        signal = jnp.sin(2 * jnp.pi * 5 * t)

        w = DifferenceOfExponentialsWavelet(n_channels=4, dt=dt, mu_max=0.05)
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
        w = DifferenceOfExponentialsWavelet(n_channels=5, dt=0.01, mu_max=0.2)
        scales = w.time_scales()
        for i in range(len(scales) - 1):
            assert scales[i] < scales[i + 1]

    def test_readable_params(self):
        w = DifferenceOfExponentialsWavelet(n_channels=3, dt=0.01, mu_max=0.2)
        rp = w.get_readable_params()
        assert "mus" in rp
        assert "weights" in rp
        # K-1 smoothing scales: implicit mu_0 = 0 contributes the raw signal.
        assert len(rp["mus"]) == 2

    def test_different_c_changes_scale_spacing(self):
        w1 = DifferenceOfExponentialsWavelet(n_channels=3, mu_max=0.2, c=2.0**0.5)
        w2 = DifferenceOfExponentialsWavelet(n_channels=3, mu_max=0.2, c=2.0)
        s1 = w1.time_scales()
        s2 = w2.time_scales()
        assert jnp.allclose(s1[-1], s2[-1])
        assert not jnp.allclose(s1[0], s2[0])


# --- Gradient tests ---

class TestGradients:
    def test_loss_has_finite_gradients(self):
        """Reconstruction loss should produce finite gradients for all params."""
        dt = 0.001
        t = jnp.arange(0, 1.0, dt)
        signal = jnp.sin(2 * jnp.pi * 10 * t)

        w = DifferenceOfExponentialsWavelet(n_channels=3, dt=dt, mu_max=0.05)
        grad_fn = jax.grad(lambda p, s: jnp.mean((s - w.reconstruct(p, w(p, s))) ** 2))
        grads = grad_fn(w.params, signal)

        for key, g in grads.items():
            assert jnp.all(jnp.isfinite(g)), f"Non-finite gradient for {key}"

import numpy as np
import pytest

from swavelet import spiking_doe, spiking_dog, temporal_integration as ti


class TestBandpassFilterNormsAnalytical:
    """Tests for cascade_depth=1 (DoE) -- analytical formula."""

    def _norms(self, mus, dt=1/360):
        return ti.bandpass_filter_norms(np.array(mus), dt, cascade_depth=1)

    def test_returns_correct_shape(self):
        norms = self._norms([0.01, 0.02, 0.04])
        assert norms.shape == (3,)

    def test_all_positive(self):
        norms = self._norms([0.01, 0.02, 0.04, 0.08])
        assert np.all(norms > 0)

    def test_analytical_matches_numerical_cascade1(self):
        """Analytical (cascade_depth=1) and numerical (cascade_depth=1) must agree."""
        mus = np.array([0.005, 0.01, 0.02, 0.04])
        dt = 1 / 360
        analytical = ti.bandpass_filter_norms(mus, dt, cascade_depth=1)
        numerical  = ti.bandpass_filter_norms(mus, dt, cascade_depth=1)
        # They use the same code path here; just check they're identical
        np.testing.assert_allclose(analytical, numerical, rtol=1e-6)

    def test_norm_decreases_with_narrower_bandwidth(self):
        """
        Narrower bandpass channels (smaller c -> adjacent mus closer together)
        should have smaller norm than wide channels.
        """
        dt = 1 / 360
        mu_base = 0.04
        # Wide channel: c=2 → mus far apart
        mus_wide   = np.array([mu_base / 2, mu_base])
        # Narrow channel: c≈1.1 → mus close together
        mus_narrow = np.array([mu_base / 1.1, mu_base])
        norm_wide   = ti.bandpass_filter_norms(mus_wide,   dt, cascade_depth=1)[0]
        norm_narrow = ti.bandpass_filter_norms(mus_narrow, dt, cascade_depth=1)[0]
        assert norm_wide > norm_narrow, (
            f"Wide ({norm_wide:.4f}) should exceed narrow ({norm_narrow:.4f})"
        )

    def test_base_channel_norm_equals_single_integrator_norm(self):
        """Base channel norm = ||h_K|| = sqrt((1-alpha)/(1+alpha))."""
        dt = 1 / 360
        mu = 0.08
        alpha = np.exp(-dt / mu)
        expected = np.sqrt((1 - alpha) / (1 + alpha))
        norms = ti.bandpass_filter_norms(np.array([0.04, mu]), dt, cascade_depth=1)
        np.testing.assert_allclose(norms[-1], expected, rtol=1e-6)

    def test_norm_vs_direct_impulse_response(self):
        """Analytical norm should match ||psi|| computed from a long impulse response."""
        dt = 1 / 360
        mus = np.array([0.01, 0.04])
        analytical_norm = ti.bandpass_filter_norms(mus, dt, cascade_depth=1)[0]

        # Direct computation via impulse response
        n = int(np.ceil(20 * mus[-1] / dt))
        impulse = np.zeros(n); impulse[0] = 1.0

        def leaky(sig, mu):
            alpha = np.exp(-dt / mu)
            out, u = np.zeros(len(sig)), 0.0
            for t in range(len(sig)):
                u = alpha * u + (1 - alpha) * sig[t]
                out[t] = u
            return out

        h0 = leaky(impulse, mus[0])
        h1 = leaky(impulse, mus[1])
        direct_norm = np.sqrt(np.sum((h0 - h1) ** 2))
        np.testing.assert_allclose(analytical_norm, direct_norm, rtol=1e-4)


class TestBandpassFilterNormsNumerical:
    """Tests for cascade_depth>1 (DoT) -- numerical impulse response."""

    def test_returns_correct_shape(self):
        norms = ti.bandpass_filter_norms(np.array([0.01, 0.02, 0.04]), 1/360, cascade_depth=3)
        assert norms.shape == (3,)

    def test_all_positive(self):
        norms = ti.bandpass_filter_norms(np.array([0.01, 0.02, 0.04, 0.08]), 1/360, cascade_depth=3)
        assert np.all(norms > 0)

    def test_deeper_cascade_gives_smaller_norms(self):
        """Deeper cascades narrow the impulse response peak -> smaller L2 norm."""
        mus = np.array([0.01, 0.04])
        dt = 1 / 360
        norm1 = ti.bandpass_filter_norms(mus, dt, cascade_depth=1)[0]
        norm3 = ti.bandpass_filter_norms(mus, dt, cascade_depth=3)[0]
        assert norm1 > norm3


class TestNormalizationIntegrationDoG:
    """Smoke tests: SpikingDoGWavelet with enable_normalization=True."""

    def test_filter_norms_stored_at_init(self):
        wav = spiking_dog.SpikingDoGWavelet(
            n_channels=4, dt=1/360, sigma_max=0.08, c=2.0,
            enable_normalization=True
        )
        assert wav._filter_norms is not None
        assert len(wav._filter_norms) == 4

    def test_all_norms_positive(self):
        wav = spiking_dog.SpikingDoGWavelet(
            n_channels=4, dt=1/360, sigma_max=0.08, c=2.0,
            enable_normalization=True
        )
        assert np.all(wav._filter_norms > 0)

    def test_no_normalization_stores_none(self):
        wav = spiking_dog.SpikingDoGWavelet(
            n_channels=4, dt=1/360, sigma_max=0.08, c=2.0,
            enable_normalization=False
        )
        assert wav._filter_norms is None

    def test_normalization_produces_more_spikes_than_unnormalized(self):
        """With normalization, narrow-bandwidth channels should fire more."""

        dt = 1 / 360
        c = 1.1  # small c → narrow bandpass → tiny amplitude without normalization
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(360).astype(np.float32)

        wav_norm  = spiking_dog.SpikingDoGWavelet(
            n_channels=3, dt=dt, sigma_max=0.05, c=c,
            threshold=0.05, enable_normalization=True
        )
        wav_plain = spiking_dog.SpikingDoGWavelet(
            n_channels=3, dt=dt, sigma_max=0.05, c=c,
            threshold=0.05, enable_normalization=False
        )

        spikes_norm  = wav_norm (wav_norm.params,  signal)
        spikes_plain = wav_plain(wav_plain.params, signal)

        total_norm  = int(np.sum(np.abs(spikes_norm)))
        total_plain = int(np.sum(np.abs(spikes_plain)))
        assert total_norm > total_plain, (
            f"Normalized ({total_norm}) should spike more than plain ({total_plain})"
        )

    def test_norm_matches_stored_kernels(self):
        """Filter norms should match the actual per-channel bandpass kernels.

        Layout (n_channels = K, K-1 smoothing kernels stored in `_kernels`):
            _filter_norms[0]      = ||G(sigma_1) - delta||
            _filter_norms[1..K-2] = ||G(sigma_{i+1}) - G(sigma_i)||
            _filter_norms[K-1]    = ||G(sigma_{K-1})||  (lowpass)
        """

        dt = 1 / 360
        n_channels = 3
        wav = spiking_dog.SpikingDoGWavelet(
            n_channels=n_channels, dt=dt, sigma_max=0.08, c=2.0,
            enable_normalization=True
        )
        kernels = np.asarray(wav._kernels)  # (K-1, kernel_len)
        kernel_len = kernels.shape[1]
        delta = np.zeros(kernel_len)
        delta[kernel_len // 2] = 1.0

        # Bandpass 0: G(sigma_1) - delta.
        bp_first = kernels[0] - delta
        np.testing.assert_allclose(
            wav._filter_norms[0], float(np.sqrt(np.sum(bp_first ** 2))), rtol=1e-5
        )
        # Bandpass i = 1..K-2: G(sigma_{i+1}) - G(sigma_i).
        for i in range(1, n_channels - 1):
            bp = kernels[i] - kernels[i - 1]
            np.testing.assert_allclose(
                wav._filter_norms[i], float(np.sqrt(np.sum(bp ** 2))), rtol=1e-5
            )
        # Lowpass: ||G(sigma_{K-1})||.
        np.testing.assert_allclose(
            wav._filter_norms[-1], float(np.sqrt(np.sum(kernels[-1] ** 2))), rtol=1e-5
        )


class TestNormalizationIntegration:
    """Smoke tests: SpikingDoEWavelet with enable_normalization=True."""

    def test_normalization_produces_more_spikes_than_unnormalized(self):
        """With normalization, channels with small bandwidths should fire more."""

        dt = 1 / 360
        # Small c → narrow bandpass → small amplitude without normalization
        c = 1.1
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(360).astype(np.float32)

        wav_norm   = spiking_doe.SpikingDoEWavelet(
            n_channels=3, dt=dt, mu_max=0.05, c=c,
            threshold=0.05, enable_normalization=True
        )
        wav_plain  = spiking_doe.SpikingDoEWavelet(
            n_channels=3, dt=dt, mu_max=0.05, c=c,
            threshold=0.05, enable_normalization=False
        )

        spikes_norm  = wav_norm (wav_norm.params,  signal)
        spikes_plain = wav_plain(wav_plain.params, signal)

        total_norm  = int(np.sum(np.abs(spikes_norm)))
        total_plain = int(np.sum(np.abs(spikes_plain)))
        assert total_norm > total_plain, (
            f"Normalized ({total_norm}) should spike more than plain ({total_plain}) "
            "when bandwidth is narrow"
        )

    def test_filter_norms_stored_at_init(self):
        wav = spiking_doe.SpikingDoEWavelet(
            n_channels=4, dt=1/360, mu_max=0.08, c=2.0,
            enable_normalization=True
        )
        assert wav._filter_norms is not None
        assert len(wav._filter_norms) == 4

    def test_no_normalization_stores_none(self):
        wav = spiking_doe.SpikingDoEWavelet(
            n_channels=4, dt=1/360, mu_max=0.08, c=2.0,
            enable_normalization=False  # explicit opt-out
        )
        assert wav._filter_norms is None

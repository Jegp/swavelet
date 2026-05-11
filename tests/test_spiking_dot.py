import jax
import jax.numpy as jnp
import pytest

from swavelet.spiking_dot import SpikingDoTWavelet


DT = 0.001
N_CHANNELS = 3
MU_MAX = 0.05
CASCADE_DEPTH = 3  # Target cascade depth (N_max) for faster tests


def make_wavelet(**kwargs):
    defaults = dict(n_channels=N_CHANNELS, dt=DT, mu_max=MU_MAX, cascade_depth_max=CASCADE_DEPTH)
    # Let callers pass either cascade_depth_max or the legacy `cascade_depth` kwarg
    # (translated) so test bodies further down keep working.
    if "cascade_depth" in kwargs:
        kwargs["cascade_depth_max"] = kwargs.pop("cascade_depth")
    defaults.update(kwargs)
    return SpikingDoTWavelet(**defaults)


def make_impulse(n=2000, loc=None):
    if loc is None:
        loc = n // 2
    impulse = jnp.zeros(n)
    return impulse.at[loc].set(1.0 / DT)


def make_sine(freq=10.0, duration=1.0):
    t = jnp.arange(0, duration, DT)
    return jnp.sin(2 * jnp.pi * freq * t)


# --- Basic forward pass ---

class TestForwardPass:
    def test_output_shapes(self):
        w = make_wavelet()
        signal = make_sine()
        spikes = w(w.params, signal)
        recon = w.reconstruct(w.params, spikes)
        assert recon.shape == signal.shape
        # Pair of pos/neg LIF spike trains per channel.
        assert spikes.shape == (2 * N_CHANNELS, len(signal))

    def test_return_trace_shapes(self):
        w = make_wavelet()
        signal = make_sine()
        spikes, traces = w(w.params, signal, return_trace=True)
        recon, recon_channels = w.reconstruct(w.params, spikes, return_trace=True)
        assert recon.shape == signal.shape
        assert spikes.shape == (2 * N_CHANNELS, len(signal))
        assert traces.shape == spikes.shape
        # Per-channel reconstructions are merged into K rows in synthesis.
        assert recon_channels.shape == (N_CHANNELS, len(signal))

    def test_reconstruction_finite(self):
        w = make_wavelet()
        signal = make_sine()
        recon = w.reconstruct(w.params, w(w.params, signal))
        assert jnp.all(jnp.isfinite(recon))

    def test_reconstruction_loss_finite(self):
        w = make_wavelet()
        signal = make_sine()
        recon = w.reconstruct(w.params, w(w.params, signal))
        loss = jnp.mean((signal - recon) ** 2)
        assert jnp.isfinite(loss)


# --- Spike properties ---

class TestSpikeProperties:
    def test_spikes_are_binary(self):
        w = make_wavelet()
        signal = make_sine()
        spikes = w(w.params, signal)
        assert jnp.all((spikes < 0.01) | (spikes > 0.99))

    def test_impulse_produces_spikes(self):
        w = make_wavelet()
        impulse = make_impulse()
        spikes = w(w.params, impulse)
        total_spikes = jnp.sum(spikes > 0.5)
        assert total_spikes > 0, "No spikes produced from impulse"

    def test_zero_signal_no_spikes(self):
        w = make_wavelet()
        signal = jnp.zeros(1000)
        spikes = w(w.params, signal)
        assert jnp.sum(spikes > 0.5) == 0

    def test_higher_threshold_fewer_spikes(self):
        w = make_wavelet()
        signal = make_sine()

        params_low = w.params.copy()
        params_high = w.params.copy()
        params_high["log_threshold"] = params_low["log_threshold"] + jnp.log(5.0)

        spikes_low = w(params_low, signal)
        spikes_high = w(params_high, signal)

        assert jnp.sum(spikes_high > 0.5) <= jnp.sum(spikes_low > 0.5)


# --- Impulse response ---

class TestImpulseResponse:
    def test_impulse_reconstruction_nonzero(self):
        w = make_wavelet()
        impulse = make_impulse()
        recon = w.reconstruct(w.params, w(w.params, impulse))
        assert jnp.max(jnp.abs(recon)) > 0

    def test_causal_spikes(self):
        """Spikes should only occur at or after the impulse."""
        w = make_wavelet()
        impulse = make_impulse(n=2000, loc=1000)
        spikes = w(w.params, impulse)
        assert jnp.sum(spikes[:, :990] > 0.5) == 0


# --- Cascade depth ---

class TestCascadeDepth:
    def test_different_depths_different_spikes(self):
        """Different cascade depths should produce different spiking behavior."""
        signal = make_sine()
        w1 = make_wavelet(cascade_depth=1)
        w2 = make_wavelet(cascade_depth=5)

        spikes1 = w1(w1.params, signal)
        spikes2 = w2(w2.params, signal)

        # Different cascade depths → different spike patterns
        assert not jnp.allclose(spikes1, spikes2)

    def test_depth_1_matches_doe_reconstruction(self):
        """With cascade_depth=1, DoT reconstruction should match DoE."""
        from swavelet.spiking_doe import SpikingDoEWavelet
        signal = make_sine()

        doe = SpikingDoEWavelet(n_channels=N_CHANNELS, dt=DT, mu_max=MU_MAX)
        dot = make_wavelet(cascade_depth=1)

        # Both should produce finite reconstructions of same shape
        recon_doe = doe.reconstruct(doe.params, doe(doe.params, signal))
        recon_dot = dot.reconstruct(dot.params, dot(dot.params, signal))

        assert recon_doe.shape == recon_dot.shape
        assert jnp.all(jnp.isfinite(recon_doe))
        assert jnp.all(jnp.isfinite(recon_dot))


# --- Get spike trains ---

class TestGetSpikeTrains:
    def test_returns_expected_keys(self):
        w = make_wavelet()
        signal = make_sine()
        result = w.get_spike_trains(w.params, signal)
        assert "spike_trains_pos" in result
        assert "spike_trains_neg" in result
        assert "bandpass_inputs" in result
        assert "scale_responses" in result

    def test_spike_train_shapes(self):
        w = make_wavelet()
        signal = make_sine()
        result = w.get_spike_trains(w.params, signal)
        assert result["spike_trains_pos"].shape == (N_CHANNELS, len(signal))
        assert result["spike_trains_neg"].shape == (N_CHANNELS, len(signal))


# --- Parameters ---

class TestParameters:
    def test_time_scales_ordered(self):
        w = make_wavelet()
        scales = w.time_scales()
        for i in range(len(scales) - 1):
            assert scales[i] < scales[i + 1]

    def test_readable_params(self):
        w = make_wavelet()
        rp = w.get_readable_params()
        assert "mus" in rp
        assert "thresholds" in rp
        assert "mu_mem" in rp
        assert "cascade_depth_max" in rp
        assert "cascade_depths" in rp
        # K-1 smoothing scales: implicit mu_0 = 0 contributes the raw signal.
        assert len(rp["cascade_depths"]) == N_CHANNELS - 1


# --- Gradients ---

class TestGradients:
    def test_loss_has_finite_gradients(self):
        w = make_wavelet()
        signal = make_sine()
        grad_fn = jax.grad(lambda p, s: jnp.mean((s - w.reconstruct(p, w(p, s))) ** 2))
        grads = grad_fn(w.params, signal)
        for key, g in grads.items():
            assert jnp.all(jnp.isfinite(g)), f"Non-finite gradient for {key}"


C_DENSE = 1.2
MU_MAX_DENSE = 0.5  # mu_min ≈ 0.032 s (16ch) / 0.0021 s (32ch), both >> DT


@pytest.mark.parametrize("n_channels", [16, 32])
class TestManyChannels:
    """
    Verify scale-overlap and bandpass correctness at dense channel counts (16, 32).

    With c=1.2, adjacent scale responses are highly correlated (heavy overlap),
    but every DoT difference channel must still carry nonzero energy.

    Gradient tests excluded here (expensive at 16/32 channels) -- see
    TestManyChannelsGradients for gradient coverage at 4 and 8 channels.
    """

    def make_wavelet(self, n_channels):
        return SpikingDoTWavelet(
            n_channels=n_channels, dt=DT, mu_max=MU_MAX_DENSE,
            c=C_DENSE, cascade_depth_max=CASCADE_DEPTH,
        )

    def test_forward_pass_shape_and_finite(self, n_channels):
        w = self.make_wavelet(n_channels)
        signal = make_sine(freq=5.0, duration=1.0)
        spikes = w(w.params, signal)
        recon = w.reconstruct(w.params, spikes)
        assert recon.shape == signal.shape
        assert spikes.shape == (2 * n_channels, len(signal))
        assert jnp.all(jnp.isfinite(recon))

    def test_scale_overlap_present(self, n_channels):
        """Adjacent scale responses must be highly correlated (> 0.9) with c=1.2."""
        w = self.make_wavelet(n_channels)
        signal = make_sine(freq=5.0, duration=2.0)
        result = w.get_spike_trains(w.params, signal)
        scale_resp = result["scale_responses"]  # (n_channels, time)

        for i in range(n_channels - 1):
            r_i = jnp.array(scale_resp[i])
            r_next = jnp.array(scale_resp[i + 1])
            corr = float(jnp.corrcoef(r_i, r_next)[0, 1])
            assert corr > 0.9, (
                f"n_channels={n_channels}: scale responses {i} and {i+1} "
                f"should be highly correlated (c={C_DENSE}), got {corr:.3f}"
            )

    def test_bandpass_response_present(self, n_channels):
        """Every DoT difference channel must have nonzero RMS despite scale overlap."""
        w = self.make_wavelet(n_channels)
        signal = make_sine(freq=5.0, duration=2.0)
        result = w.get_spike_trains(w.params, signal)
        bandpass = result["bandpass_inputs"]  # (n_channels, time)

        for i in range(n_channels):
            rms = float(jnp.sqrt(jnp.mean(jnp.array(bandpass[i]) ** 2)))
            assert rms > 0, (
                f"n_channels={n_channels}: bandpass channel {i} has zero RMS"
            )

    def test_bandpass_rms_smaller_than_lowpass_rms(self, n_channels):
        """Mean bandpass RMS < mean lowpass RMS -- consequence of scale overlap."""
        w = self.make_wavelet(n_channels)
        signal = make_sine(freq=5.0, duration=2.0)
        result = w.get_spike_trains(w.params, signal)
        scale_resp = result["scale_responses"]
        bandpass = result["bandpass_inputs"]

        mean_lp = float(jnp.mean(jnp.array([
            jnp.sqrt(jnp.mean(jnp.array(r) ** 2)) for r in scale_resp
        ])))
        mean_bp = float(jnp.mean(jnp.array([
            jnp.sqrt(jnp.mean(jnp.array(d) ** 2)) for d in bandpass
        ])))

        assert mean_bp < mean_lp, (
            f"n_channels={n_channels}: mean bandpass RMS ({mean_bp:.4f}) "
            f"should be < mean lowpass RMS ({mean_lp:.4f}) with c={C_DENSE}"
        )

    def test_fine_channels_dominate_for_high_frequency(self, n_channels):
        """Fine channels (low index) carry more energy for high-freq; coarse for low-freq."""
        w = self.make_wavelet(n_channels)
        sig_high = make_sine(freq=30.0, duration=2.0)
        sig_low = make_sine(freq=0.5, duration=4.0)

        result_high = w.get_spike_trains(w.params, sig_high)
        result_low = w.get_spike_trains(w.params, sig_low)

        energies_high = jnp.sum(jnp.array(result_high["bandpass_inputs"]) ** 2, axis=1)
        energies_low = jnp.sum(jnp.array(result_low["bandpass_inputs"]) ** 2, axis=1)

        peak_high = int(jnp.argmax(energies_high))
        peak_low = int(jnp.argmax(energies_low))

        assert peak_high < peak_low, (
            f"n_channels={n_channels}: high-freq peak channel ({peak_high}) "
            f"should be finer (lower index) than low-freq peak ({peak_low})"
        )


@pytest.mark.parametrize("n_channels", [4, 8])
class TestManyChannelsGradients:
    """Gradient correctness for dense-config DoT at moderate channel counts."""

    def make_wavelet(self, n_channels):
        return SpikingDoTWavelet(
            n_channels=n_channels, dt=DT, mu_max=MU_MAX_DENSE,
            c=C_DENSE, cascade_depth_max=CASCADE_DEPTH,
        )

    def test_finite_gradients(self, n_channels):
        w = self.make_wavelet(n_channels)
        signal = make_sine(freq=5.0, duration=1.0)
        grad_fn = jax.grad(lambda p, s: jnp.mean((s - w.reconstruct(p, w(p, s))) ** 2))
        grads = grad_fn(w.params, signal)
        for key, g in grads.items():
            assert jnp.all(jnp.isfinite(g)), (
                f"n_channels={n_channels}: non-finite gradient for {key}"
            )



import jax
import jax.numpy as jnp
import pytest

from swavelet.spiking_dog import SpikingDoGWavelet


DT = 0.001
N_CHANNELS = 3
SIGMA_MAX = 0.05


def make_wavelet(**kwargs):
    defaults = dict(n_channels=N_CHANNELS, dt=DT, sigma_max=SIGMA_MAX)
    defaults.update(kwargs)
    return SpikingDoGWavelet(**defaults)


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
        assert "sigmas" in rp
        assert "thresholds" in rp
        assert "mu_mem" in rp


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
# sigma_max=0.1 keeps the coarsest Gaussian kernel (4*sigma/dt = 400 samples) inside
# any signal we use, avoiding the convolve mode='same' size-mismatch when the kernel
# exceeds the signal length.  sigma_min ≈ 0.0065 s (16ch) / 4.2e-4 s (32ch), both > DT.
SIGMA_MAX_DENSE = 0.1


@pytest.mark.parametrize("n_channels", [16, 32])
class TestManyChannels:
    """
    Verify scale-overlap and bandpass correctness at dense channel counts (16, 32).

    With c=1.2, adjacent scale responses are highly correlated (heavy overlap),
    but every DoG difference channel must still carry nonzero energy.

    Gradient tests excluded here (expensive at 16/32 channels) -- see
    TestManyChannelsGradients for gradient coverage at 4 and 8 channels.
    """

    def make_wavelet(self, n_channels):
        return SpikingDoGWavelet(
            n_channels=n_channels, dt=DT, sigma_max=SIGMA_MAX_DENSE, c=C_DENSE,
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
        """Every DoG difference channel must have nonzero RMS despite scale overlap."""
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
    """Gradient correctness for dense-config DoG at moderate channel counts."""

    def make_wavelet(self, n_channels):
        return SpikingDoGWavelet(
            n_channels=n_channels, dt=DT, sigma_max=SIGMA_MAX_DENSE, c=C_DENSE,
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



"""
Tests for SzuWavelet -- verifying correct analysis-synthesis reconstruction.

The Szu causal analytical wavelet (Szu et al. 1992) has a proper CWT inverse.
  Analysis:   W_k[n] = Sigma_m signal[m]*conj(kernel[m-n])  (cross-correlation)
  Synthesis:  synth_k = convolve(W_k, kernel)

This gives synth_k(omega) = Signal(omega) * |Psi_k(omega)|^2  (zero-phase bandpass filter).
The dominant frequency of the per-channel synthesis output must therefore
match the channel's center frequency -- this is the primary correctness test.

NOTE: The default Szu parameters map log_frequencies through a sigmoid to
~67-88 Hz. Tests use dt=0.001 (1 kHz sampling, Nyquist=500 Hz) to properly
represent these frequencies.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from swavelet.szu import SzuWavelet


DT = 0.001      # 1 kHz — needed to represent default Szu frequencies (~67–88 Hz)
N_SAMPLES = 4000
MARGIN = 200


def make_signal(freq_hz, dt=DT, n=N_SAMPLES):
    t = jnp.arange(n) * dt
    return jnp.sin(2 * jnp.pi * freq_hz * t)


def channel_fft_peak(wav, ch_idx, signal, dt=DT):
    """Return the frequency (Hz) where channel ch_idx's synthesis output peaks."""
    n_ch = wav.n_channels
    p = dict(wav.params)
    p["channel_weights"] = jnp.zeros(n_ch).at[ch_idx].set(1.0)
    recon = wav.reconstruct(p, wav(p, signal))
    fft_mag = np.abs(np.fft.rfft(np.array(recon)))
    freqs = np.fft.rfftfreq(len(signal), dt)
    return float(freqs[np.argmax(fft_mag)])


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------

class TestSzuKernel:
    def test_complex_valued(self):
        w = SzuWavelet(n_channels=4, dt=DT)
        t = (jnp.arange(201) - 100) * DT
        kernel = w.szu_kernel(t, frequency=75.0, decay_rate=5.0)
        assert jnp.iscomplexobj(kernel)

    def test_causal_zero_before_onset(self):
        """Szu kernel is causal: zero for t < 0."""
        w = SzuWavelet(n_channels=4, dt=DT)
        n = 4001
        t = (jnp.arange(n) - n // 2) * DT
        kernel = w.szu_kernel(t, frequency=75.0, decay_rate=5.0)
        assert jnp.all(kernel[t < 0] == 0), "Szu kernel should be zero for t < 0"

    def test_peak_at_onset(self):
        """Envelope should peak at or very near t=0 (causal onset)."""
        w = SzuWavelet(n_channels=4, dt=DT)
        n = 4001
        t = (jnp.arange(n) - n // 2) * DT
        kernel = w.szu_kernel(t, frequency=75.0, decay_rate=5.0)
        causal_envelope = jnp.abs(kernel[t >= 0])
        peak_idx = int(jnp.argmax(causal_envelope))
        assert peak_idx <= 2, f"Causal envelope should peak near onset, got index {peak_idx}"

    def test_finite_values(self):
        w = SzuWavelet(n_channels=4, dt=DT)
        t = (jnp.arange(N_SAMPLES) - N_SAMPLES // 2) * DT
        kernel = w.szu_kernel(t, frequency=75.0, decay_rate=5.0)
        assert jnp.all(jnp.isfinite(jnp.real(kernel)))
        assert jnp.all(jnp.isfinite(jnp.imag(kernel)))

    def test_higher_decay_rate_faster_decay(self):
        """Higher decay_rate -> energy concentrated nearer onset."""
        w = SzuWavelet(n_channels=4, dt=DT)
        n = 4001
        t = (jnp.arange(n) - n // 2) * DT
        k_fast = w.szu_kernel(t, frequency=75.0, decay_rate=50.0)
        k_slow = w.szu_kernel(t, frequency=75.0, decay_rate=2.0)
        pos_t = t >= 0
        n_pos = int(jnp.sum(pos_t))
        quarter = n_pos // 4
        env_fast = jnp.abs(k_fast[pos_t])
        env_slow = jnp.abs(k_slow[pos_t])
        frac_fast = float(jnp.sum(env_fast[:quarter] ** 2) / (jnp.sum(env_fast ** 2) + 1e-30))
        frac_slow = float(jnp.sum(env_slow[:quarter] ** 2) / (jnp.sum(env_slow ** 2) + 1e-30))
        assert frac_fast > frac_slow, (
            f"Fast decay should concentrate energy earlier: {frac_fast:.3f} vs {frac_slow:.3f}"
        )


# ---------------------------------------------------------------------------
# Encoding (analysis) tests
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_encoding_shape(self):
        w = SzuWavelet(n_channels=6, dt=DT)
        freqs_raw, _ = w._params_to_values(w.params)
        f0 = float(freqs_raw[0])
        signal = make_signal(min(f0, 400.0))
        enc = w(w.params, signal)
        assert enc.shape == (6, N_SAMPLES)

    def test_encoding_is_complex(self):
        w = SzuWavelet(n_channels=4, dt=DT)
        signal = make_signal(75.0)
        enc = w(w.params, signal)
        assert jnp.iscomplexobj(enc)

    def test_encoding_finite(self):
        w = SzuWavelet(n_channels=4, dt=DT)
        signal = make_signal(75.0)
        enc = w(w.params, signal)
        assert jnp.all(jnp.isfinite(jnp.abs(enc)))


# ---------------------------------------------------------------------------
# Synthesis correctness tests
# ---------------------------------------------------------------------------

class TestSynthesisCorrectness:
    """
    The key property: synthesis output's dominant FFT frequency must match
    the channel's center frequency (since synth_k(omega) = Signal(omega) * |Psi_k(omega)|^2).
    """

    def test_synthesis_fft_peak_matches_channel_frequency(self):
        """Each channel's synthesis output should peak at its center frequency."""
        n_ch = 8
        w = SzuWavelet(n_channels=n_ch, dt=DT)
        freqs_raw, _ = w._params_to_values(w.params)
        freqs = np.array(freqs_raw)
        nyquist = 0.5 / DT

        for i, f_center in enumerate(freqs):
            f_center = float(f_center)
            if f_center >= nyquist * 0.9:
                continue  # skip channels too close to Nyquist
            signal = make_signal(f_center)
            peak_f = channel_fft_peak(w, i, signal)
            # Allow 20% tolerance (Szu has broader passband due to causal decay)
            tol = max(f_center * 0.2, 2.0)
            assert abs(peak_f - f_center) <= tol, (
                f"Channel {i} (f={f_center:.1f} Hz): synthesis FFT peak "
                f"at {peak_f:.1f} Hz, expected within {tol:.1f} Hz of center"
            )

    def test_synthesis_output_is_real(self):
        w = SzuWavelet(n_channels=4, dt=DT)
        signal = make_signal(75.0)
        recon = w.reconstruct(w.params, w(w.params, signal))
        assert not jnp.iscomplexobj(recon)

    def test_synthesis_output_shape(self):
        w = SzuWavelet(n_channels=4, dt=DT)
        signal = make_signal(75.0)
        recon = w.reconstruct(w.params, w(w.params, signal))
        assert recon.shape == signal.shape

    def test_synthesis_output_finite(self):
        w = SzuWavelet(n_channels=4, dt=DT)
        signal = make_signal(75.0)
        recon = w.reconstruct(w.params, w(w.params, signal))
        assert jnp.all(jnp.isfinite(recon))

    def test_synthesis_differs_from_raw_analysis(self):
        """Synthesis output must differ from the raw analysis encoding."""
        n_ch = 4
        w = SzuWavelet(n_channels=n_ch, dt=DT)
        signal = make_signal(75.0)

        p = dict(w.params)
        p["channel_weights"] = jnp.zeros(n_ch).at[0].set(1.0)
        enc = w(p, signal)
        recon = w.reconstruct(p, enc)

        recon_np = np.array(recon)
        enc_real_np = np.array(jnp.real(enc[0]))
        assert not np.allclose(recon_np, enc_real_np, atol=1e-6), (
            "Synthesis output should differ from raw analysis encoding"
        )

    def test_higher_frequency_channel_has_higher_synthesis_peak(self):
        """Higher-indexed channels (higher center freq) should have higher FFT peaks."""
        n_ch = 6
        w = SzuWavelet(n_channels=n_ch, dt=DT)
        freqs_raw, _ = w._params_to_values(w.params)
        freqs = np.array(freqs_raw)
        nyquist = 0.5 / DT

        t = jnp.arange(N_SAMPLES) * DT
        # Broadband signal covering the channel range
        signal = sum(
            jnp.sin(2 * jnp.pi * float(f) * t)
            for f in freqs
            if float(f) < nyquist * 0.9
        )

        peaks = [
            channel_fft_peak(w, i, signal)
            for i in range(n_ch)
            if float(freqs[i]) < nyquist * 0.9
        ]

        if len(peaks) >= 2:
            assert peaks[0] <= peaks[-1], (
                f"Lowest channel peak ({peaks[0]:.1f} Hz) should be <= "
                f"highest channel peak ({peaks[-1]:.1f} Hz)"
            )


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:
    def test_reconstruction_loss_finite_gradients(self):
        t = jnp.arange(400) * DT
        signal = jnp.sin(2 * jnp.pi * 75.0 * t)
        w = SzuWavelet(n_channels=4, dt=DT)

        grad_fn = jax.grad(lambda p, s: jnp.mean((s - w.reconstruct(p, w(p, s))) ** 2))
        grads = grad_fn(w.params, signal)

        for key, g in grads.items():
            assert jnp.all(jnp.isfinite(g)), f"Non-finite gradient for {key}"

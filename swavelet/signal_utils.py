import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
import numpy as np


class SignalGenerator:
    """
    Utility for generating and manipulating test signals for wavelet experiments.
    Focuses on signals that can be temporally dilated/compressed for scale covariance testing.
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt

    def generate_multi_sine(
        self,
        duration: float,
        frequencies: List[float],
        amplitudes: List[float] = None,
        phases: List[float] = None,
        noise_level: float = 0.0
    ) -> jnp.ndarray:
        """
        Generate multi-component sine wave signal.

        Args:
            duration: Signal duration in seconds
            frequencies: List of frequencies (Hz)
            amplitudes: List of amplitudes (default: all 1.0)
            phases: List of phase shifts (default: all 0.0)
            noise_level: Gaussian noise standard deviation

        Returns:
            Signal array
        """
        n_samples = int(duration / self.dt)
        t = jnp.arange(n_samples) * self.dt

        if amplitudes is None:
            amplitudes = [1.0] * len(frequencies)
        if phases is None:
            phases = [0.0] * len(frequencies)

        signal = jnp.zeros(n_samples)
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            signal += amp * jnp.sin(2 * jnp.pi * freq * t + phase)

        if noise_level > 0:
            key = jax.random.PRNGKey(42)
            noise = jax.random.normal(key, signal.shape) * noise_level
            signal += noise

        return signal

    def generate_chirp(
        self,
        duration: float,
        f0: float,
        f1: float,
        noise_level: float = 0.0
    ) -> jnp.ndarray:
        """
        Generate linear chirp signal (frequency sweep).

        Args:
            duration: Signal duration in seconds
            f0: Starting frequency (Hz)
            f1: Ending frequency (Hz)
            noise_level: Gaussian noise standard deviation

        Returns:
            Signal array
        """
        n_samples = int(duration / self.dt)
        t = jnp.arange(n_samples) * self.dt

        # Linear frequency sweep
        phase = 2 * jnp.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
        signal = jnp.sin(phase)

        if noise_level > 0:
            key = jax.random.PRNGKey(42)
            noise = jax.random.normal(key, signal.shape) * noise_level
            signal += noise

        return signal

    def generate_pulse_train(
        self,
        duration: float,
        pulse_rate: float,
        pulse_width: float,
        pulse_amplitude: float = 1.0,
        noise_level: float = 0.0
    ) -> jnp.ndarray:
        """
        Generate periodic pulse train.

        Args:
            duration: Signal duration in seconds
            pulse_rate: Pulse rate (Hz)
            pulse_width: Pulse width in seconds
            pulse_amplitude: Pulse amplitude
            noise_level: Gaussian noise standard deviation

        Returns:
            Signal array
        """
        n_samples = int(duration / self.dt)
        t = jnp.arange(n_samples) * self.dt

        period = 1.0 / pulse_rate
        pulse_samples = int(pulse_width / self.dt)

        signal = jnp.zeros(n_samples)

        # Add pulses at regular intervals
        for pulse_start in jnp.arange(0, duration, period):
            start_idx = int(pulse_start / self.dt)
            end_idx = min(start_idx + pulse_samples, n_samples)
            if start_idx < n_samples:
                signal = signal.at[start_idx:end_idx].set(pulse_amplitude)

        if noise_level > 0:
            key = jax.random.PRNGKey(42)
            noise = jax.random.normal(key, signal.shape) * noise_level
            signal += noise

        return signal

    def generate_exponential_decay(
        self,
        duration: float,
        tau: float,
        amplitude: float = 1.0,
        noise_level: float = 0.0
    ) -> jnp.ndarray:
        """
        Generate exponential decay signal.

        Args:
            duration: Signal duration in seconds
            tau: Time constant
            amplitude: Initial amplitude
            noise_level: Gaussian noise standard deviation

        Returns:
            Signal array
        """
        n_samples = int(duration / self.dt)
        t = jnp.arange(n_samples) * self.dt

        signal = amplitude * jnp.exp(-t / tau)

        if noise_level > 0:
            key = jax.random.PRNGKey(42)
            noise = jax.random.normal(key, signal.shape) * noise_level
            signal += noise

        return signal

    def dilate_signal(self, signal: jnp.ndarray, dilation_factor: float) -> jnp.ndarray:
        """
        Temporally dilate/compress a signal by resampling.

        Args:
            signal: Input signal
            dilation_factor: >1 for dilation (slower), <1 for compression (faster)

        Returns:
            Dilated signal (same length as original)
        """
        n_original = len(signal)

        # Create new time indices scaled by dilation factor
        original_indices = jnp.arange(n_original)
        dilated_indices = jnp.arange(n_original) / dilation_factor

        # Interpolate to get dilated signal
        # For indices outside the original range, use boundary values
        dilated_signal = jnp.interp(
            dilated_indices,
            original_indices,
            signal,
            left=signal[0],
            right=signal[-1]
        )

        return dilated_signal

    def create_scale_test_signals(
        self,
        base_signal: jnp.ndarray,
        scale_factors: List[float]
    ) -> Dict[float, jnp.ndarray]:
        """
        Create a set of temporally scaled versions of a signal.

        Args:
            base_signal: Original signal
            scale_factors: List of temporal scale factors

        Returns:
            Dictionary mapping scale factor to scaled signal
        """
        scaled_signals = {}
        for factor in scale_factors:
            scaled_signals[factor] = self.dilate_signal(base_signal, factor)

        return scaled_signals


class SignalMetrics:
    """
    Utilities for computing signal analysis metrics.
    """

    @staticmethod
    def compute_snr(signal: jnp.ndarray, noise: jnp.ndarray) -> float:
        """Compute signal-to-noise ratio in dB."""
        signal_power = jnp.mean(signal**2)
        noise_power = jnp.mean(noise**2)
        return 10 * jnp.log10(signal_power / noise_power)

    @staticmethod
    def compute_mse(signal1: jnp.ndarray, signal2: jnp.ndarray) -> float:
        """Compute mean squared error between two signals."""
        return jnp.mean((signal1 - signal2)**2)

    @staticmethod
    def compute_normalized_mse(signal1: jnp.ndarray, signal2: jnp.ndarray) -> float:
        """Compute normalized mean squared error (NMSE)."""
        mse = SignalMetrics.compute_mse(signal1, signal2)
        signal_power = jnp.mean(signal1**2)
        return mse / signal_power

    @staticmethod
    def compute_correlation(signal1: jnp.ndarray, signal2: jnp.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        return jnp.corrcoef(signal1, signal2)[0, 1]

    @staticmethod
    def compute_spectral_distortion(
        signal1: jnp.ndarray,
        signal2: jnp.ndarray,
        dt: float
    ) -> float:
        """
        Compute spectral distortion between two signals.
        Uses the log-spectral distance metric.
        """
        # Compute power spectral densities
        fft1 = jnp.fft.fft(signal1)
        fft2 = jnp.fft.fft(signal2)

        psd1 = jnp.abs(fft1)**2
        psd2 = jnp.abs(fft2)**2

        # Add small constant to avoid log(0)
        epsilon = 1e-12
        psd1 = psd1 + epsilon
        psd2 = psd2 + epsilon

        # Log-spectral distance
        log_ratio = jnp.log(psd1 / psd2)
        spectral_dist = jnp.sqrt(jnp.mean(log_ratio**2))

        return spectral_dist

    @staticmethod
    def compute_spike_rate(spikes: jnp.ndarray, dt: float, window_size: float = 0.1) -> jnp.ndarray:
        """
        Compute instantaneous spike rate using a sliding window.

        Args:
            spikes: Binary spike train
            dt: Time step
            window_size: Window size in seconds for rate estimation

        Returns:
            Instantaneous spike rate array
        """
        window_samples = int(window_size / dt)
        kernel = jnp.ones(window_samples) / window_size

        # Convolve with causal padding
        padded_spikes = jnp.pad(spikes, (window_samples-1, 0), mode='constant')
        rate = jnp.convolve(padded_spikes, kernel, mode='valid')

        return rate

    @staticmethod
    def compute_event_count(spikes: jnp.ndarray) -> int:
        """Count total number of spike events."""
        return int(jnp.sum(jnp.abs(spikes)))


class RealtimeProcessor:
    """
    Framework for real-time signal processing and reconstruction.
    Processes signals step-by-step to simulate online processing.
    """

    def __init__(self, wavelet_obj, dt: float = 0.01, buffer_size: int = 100):
        self.wavelet = wavelet_obj
        self.dt = dt
        self.buffer_size = buffer_size

        # Internal state
        self.input_buffer = jnp.zeros(buffer_size)
        self.reconstruction_buffer = jnp.zeros(buffer_size)
        self.step_count = 0

    def reset(self):
        """Reset processor state."""
        self.input_buffer = jnp.zeros(self.buffer_size)
        self.reconstruction_buffer = jnp.zeros(self.buffer_size)
        self.step_count = 0

    def process_step(self, new_sample: float) -> Tuple[float, Dict]:
        """
        Process a single new sample and return reconstruction + diagnostics.

        Args:
            new_sample: New input sample

        Returns:
            (reconstructed_sample, diagnostics_dict)
        """
        # Shift buffer and add new sample
        self.input_buffer = jnp.roll(self.input_buffer, -1)
        self.input_buffer = self.input_buffer.at[-1].set(new_sample)

        # Process current buffer window
        if hasattr(self.wavelet, 'spiking_response'):
            # For spiking wavelets, get spike response
            response = self.wavelet.spiking_response(self.wavelet.params, self.input_buffer)
            spikes = response

            # Get reconstruction (for spiking wavelets that support it)
            if hasattr(self.wavelet, 'spike_rate_response'):
                reconstruction = self.wavelet.spike_rate_response(self.wavelet.params, self.input_buffer)
            else:
                reconstruction = response

            # Extract current sample reconstruction (middle of buffer for causality)
            current_reconstruction = reconstruction[self.buffer_size // 2]

            # Compute diagnostics
            diagnostics = {
                'spike_count': SignalMetrics.compute_event_count(spikes),
                'current_spike': spikes[-1],  # Most recent spike
                'buffer_energy': float(jnp.mean(self.input_buffer**2))
            }

        elif hasattr(self.wavelet, 'get_spike_trains'):
            # Spiking DoT/DoE/DoG wavelet
            response, _ = self.wavelet(self.wavelet.params, self.input_buffer)

            # Get spike trains for diagnostics
            spike_data = self.wavelet.get_spike_trains(self.wavelet.params, self.input_buffer)
            total_spikes = jnp.sum(spike_data['spike_trains_pos']) + jnp.sum(spike_data['spike_trains_neg'])

            current_reconstruction = response[self.buffer_size // 2]

            diagnostics = {
                'spike_count': float(total_spikes),
                'current_spike': 0.0,  # Would need to extract from specific channel
                'buffer_energy': float(jnp.mean(self.input_buffer**2))
            }

        else:
            # For non-spiking wavelets
            if hasattr(self.wavelet, 'morlet_response'):
                response = self.wavelet.morlet_response(self.wavelet.params, self.input_buffer)
            elif hasattr(self.wavelet, 'haar_response'):
                response = self.wavelet.haar_response(self.wavelet.params, self.input_buffer)
            elif hasattr(self.wavelet, 'difference_of_exponentials_response'):
                response = self.wavelet.difference_of_exponentials_response(self.wavelet.params, self.input_buffer)
            else:
                response, _ = self.wavelet(self.wavelet.params, self.input_buffer)

            current_reconstruction = response[self.buffer_size // 2]

            diagnostics = {
                'spike_count': 0.0,
                'current_spike': 0.0,
                'buffer_energy': float(jnp.mean(self.input_buffer**2))
            }

        # Update reconstruction buffer
        self.reconstruction_buffer = jnp.roll(self.reconstruction_buffer, -1)
        self.reconstruction_buffer = self.reconstruction_buffer.at[-1].set(current_reconstruction)

        self.step_count += 1
        diagnostics['step_count'] = self.step_count

        return float(current_reconstruction), diagnostics

    def process_signal(self, signal: jnp.ndarray) -> Tuple[jnp.ndarray, List[Dict]]:
        """
        Process entire signal step by step.

        Args:
            signal: Input signal

        Returns:
            (reconstruction_array, diagnostics_list)
        """
        self.reset()

        reconstructions = []
        all_diagnostics = []

        for sample in signal:
            recon, diag = self.process_step(float(sample))
            reconstructions.append(recon)
            all_diagnostics.append(diag)

        return jnp.array(reconstructions), all_diagnostics


# Predefined signal configurations for experiments
EXPERIMENT_SIGNALS = {
    'multi_sine_5_15': {
        'type': 'multi_sine',
        'duration': 2.0,
        'frequencies': [5.0, 15.0],
        'amplitudes': [1.0, 0.7],
        'noise_level': 0.1
    },

    'chirp_2_20': {
        'type': 'chirp',
        'duration': 2.0,
        'f0': 2.0,
        'f1': 20.0,
        'noise_level': 0.05
    },

    'pulse_train_10hz': {
        'type': 'pulse_train',
        'duration': 2.0,
        'pulse_rate': 10.0,
        'pulse_width': 0.02,
        'pulse_amplitude': 1.0,
        'noise_level': 0.05
    },

    'exp_decay_fast': {
        'type': 'exponential_decay',
        'duration': 1.0,
        'tau': 0.1,
        'amplitude': 1.0,
        'noise_level': 0.02
    }
}


def generate_experiment_signal(signal_name: str, dt: float = 0.01) -> jnp.ndarray:
    """
    Generate a predefined experimental signal.

    Args:
        signal_name: Name from EXPERIMENT_SIGNALS
        dt: Time step

    Returns:
        Generated signal
    """
    if signal_name not in EXPERIMENT_SIGNALS:
        raise ValueError(f"Unknown signal: {signal_name}. Available: {list(EXPERIMENT_SIGNALS.keys())}")

    config = EXPERIMENT_SIGNALS[signal_name]
    generator = SignalGenerator(dt=dt)

    if config['type'] == 'multi_sine':
        return generator.generate_multi_sine(
            config['duration'],
            config['frequencies'],
            config.get('amplitudes'),
            config.get('phases'),
            config.get('noise_level', 0.0)
        )
    elif config['type'] == 'chirp':
        return generator.generate_chirp(
            config['duration'],
            config['f0'],
            config['f1'],
            config.get('noise_level', 0.0)
        )
    elif config['type'] == 'pulse_train':
        return generator.generate_pulse_train(
            config['duration'],
            config['pulse_rate'],
            config['pulse_width'],
            config.get('pulse_amplitude', 1.0),
            config.get('noise_level', 0.0)
        )
    elif config['type'] == 'exponential_decay':
        return generator.generate_exponential_decay(
            config['duration'],
            config['tau'],
            config.get('amplitude', 1.0),
            config.get('noise_level', 0.0)
        )
    else:
        raise ValueError(f"Unknown signal type: {config['type']}")


def fft_convolve(signal, kernel, mode='same'):
    """FFT-based convolution -- O(N log N) replacement for jnp.convolve.

    Supports mode='same' (centred, len = len(signal)) and
    mode='full' (len = len(signal) + len(kernel) - 1).
    Uses rfft for real inputs, full fft for complex inputs.
    """
    n_sig = len(signal)
    n_ker = len(kernel)
    n_out = n_sig + n_ker - 1
    n_fft = 1 << (n_out - 1).bit_length()  # next power of 2
    is_complex = jnp.iscomplexobj(signal) or jnp.iscomplexobj(kernel)
    if is_complex:
        out = jnp.fft.ifft(
            jnp.fft.fft(signal, n=n_fft) * jnp.fft.fft(kernel, n=n_fft),
        )[:n_out]
    else:
        out = jnp.fft.irfft(
            jnp.fft.rfft(signal, n=n_fft) * jnp.fft.rfft(kernel, n=n_fft),
            n=n_fft,
        )[:n_out]
    if mode == 'full':
        return out
    # mode == 'same': centred slice of length n_sig
    start = (n_ker - 1) // 2
    return out[start:start + n_sig]

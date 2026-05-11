"""
Spiking Difference of Gaussians (DoG) Wavelet

Implements DoG filters with spiking neurons. Uses parallel Gaussian filters
at different scales (standard deviations), takes differences to create bandpass
channels, then applies dual-channel integrate-and-fire neurons for spike encoding.

Unlike SpikingDoTWavelet (cascaded exponentials) or SpikingDoEWavelet (single
exponentials), this uses full symmetric Gaussian kernels applied via convolution.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import wavelet
from . import temporal_integration as ti
from .dog import _build_kernel_bank, DifferenceOfGaussiansWavelet
from .signal_utils import fft_convolve


def composite_gaussian_membrane(signal, gauss_kernel, mu_mem, sigma, dt):
    """Apply discrete Gaussian kernel followed by membrane exponential filter.

    Composite kernel: G(sigma) (x) h_mem(mu_mem), normalized by 1/(sigma - mu_mem)
    (same scaling as the continuous derivation -- small discretization
    correction is absorbed by the lstsq weights at evaluation time).
    """
    gauss_filtered = fft_convolve(signal, gauss_kernel, mode='same')
    result = ti.exponential_filter(gauss_filtered, mu_mem, dt)
    return result / (sigma - mu_mem + 1e-10)


def composite_gaussian_difference(signal, kernel_fine, kernel_coarse, mu_mem, sigma_fine, sigma_coarse, dt):
    """Composite reconstruction kernel for a DoG bandpass channel + membrane.

    Computes: [G(sigma_fine) (x) h_mem] - [G(sigma_coarse) (x) h_mem].
    """
    term_fine = composite_gaussian_membrane(signal, kernel_fine, mu_mem, sigma_fine, dt)
    term_coarse = composite_gaussian_membrane(signal, kernel_coarse, mu_mem, sigma_coarse, dt)
    return term_fine - term_coarse


class SpikingDoGWavelet(wavelet.Wavelet):
    """
    Spiking implementation of Difference of Gaussians (DoG) wavelets.

    Architecture:
        Signal -> N Gaussian Filters (parallel, via convolution)
               -> N-1 DoG Differences (filter[i] - filter[i+1])
               -> Dual-Channel I&F Neurons (pos/neg spike encoding)
               -> Composite Gaussian-Membrane Kernels (reconstruction)
               -> Weighted Sum -> Reconstruction

    Key differences from SpikingDoTWavelet:
    - Uses full Gaussian kernels (symmetric) instead of causal cascaded exponentials
    - Gaussian filters applied via convolution, not recurrent integration
    - Better frequency selectivity due to Gaussian's optimal time-frequency localization
    - Non-causal (signal is filtered symmetrically)
    """

    def __init__(
        self,
        n_channels: int = 3,
        dt: float = 0.01,
        sigma_max: float = 0.2,
        c: float = 2.0**0.5,
        epsilon: float = 1e-6,
        threshold: float = None,
        surrogate_type: str = 'sigmoid',
        surrogate_beta: float = 10.0,
        enable_normalization: bool = True,
    ):
        """
        Args:
            n_channels: Number of filter scales (produces n_channels-1 difference channels)
            dt: Time step for integration
            sigma_max: Maximum standard deviation (coarsest scale)
            c: Distribution parameter for logarithmic spacing (c > 1)
            epsilon: Relative truncation error for the discrete Gaussian kernel
                (Lindeberg 1990; smaller -> longer kernel, better accuracy).
            threshold: Spiking threshold. If None, uses adaptive initialization.
            surrogate_type: Type of surrogate gradient ('sigmoid', 'fast_sigmoid', 'arctan')
            surrogate_beta: Steepness of surrogate gradient
            enable_normalization: If True, normalise each channel's LIF input by the
                L2 norm of its bandpass filter so all channels get unit-RMS drive.
        """
        if n_channels < 2:
            raise ValueError(
                f"n_channels must be >= 2 (one bandpass channel and one "
                f"lowpass residual at minimum), got {n_channels}"
            )
        self.n_channels = n_channels
        self.dt = dt
        self.c = float(c)
        self.epsilon = epsilon
        self.surrogate_type = surrogate_type
        self.surrogate_beta = surrogate_beta
        self.enable_normalization = enable_normalization

        # K smoothing scales sigma_1..sigma_K ending at sigma_max
        # (K = n_channels - 1). The implicit zero-scale level L_0 = signal is
        # the raw input; the high-frequency bandpass channel is
        # Delta L_1 = G(sigma_1) f - f.
        n_smooth = n_channels - 1
        scale_indices = np.arange(n_smooth)
        sigmas_np = sigma_max * (self.c ** (-(n_smooth - 1 - scale_indices)))
        sigmas = jnp.asarray(sigmas_np)

        # Precompute K discrete Gaussian kernels (static arrays).
        self._kernels, self._kernel_len = _build_kernel_bank(sigmas_np, dt, epsilon)

        # Per-output-channel (length n_channels = K+1) views of the
        # smoothing-scale arrays. Bandpass channel k (0..K-1) uses
        # sigma_{k+1} = sigmas[k]; the lowpass channel (index K) reuses the
        # coarsest smoothing scale sigmas[-1].
        sigmas_per_channel = jnp.append(sigmas, sigmas[-1])

        mu_mem_per_channel = jnp.ones(n_channels) * dt

        if threshold is None:
            threshold = 10 * sigmas_per_channel

        # Precompute bandpass L2 norms for LIF input prescaling.
        if enable_normalization:
            kernels_np = np.asarray(self._kernels)
            kernel_len = kernels_np.shape[1]
            center = kernel_len // 2
            delta = np.zeros(kernel_len)
            delta[center] = 1.0
            norms = np.zeros(n_channels)
            # Bandpass 0: G(sigma_1) - delta. Use centered delta so subtraction matches
            # the same-mode convolution alignment.
            bp_first = kernels_np[0] - delta
            norms[0] = float(np.sqrt(np.sum(bp_first ** 2)))
            for i in range(1, n_channels - 1):
                bp = kernels_np[i] - kernels_np[i - 1]
                norms[i] = float(np.sqrt(np.sum(bp ** 2)))
            norms[-1] = float(np.sqrt(np.sum(kernels_np[-1] ** 2)))
            self._filter_norms = norms
        else:
            self._filter_norms = None

        self.params = {
            "log_sigmas": jnp.log(sigmas),  # K smoothing scales
            "log_threshold": jnp.log(jnp.ones(n_channels) * threshold),
            "log_mu_mem": jnp.log(mu_mem_per_channel),
            "log_mu_recon": jnp.log(sigmas_per_channel),  # per-channel decay (K+1)
            "log_weights": jnp.log(1.0 / sigmas_per_channel),
            "surrogate_beta": self.surrogate_beta,
        }

        # Non-spiking sibling that owns the analysis stage. Reused for the
        # decoupled per-spike scheme so the per-channel kernel logic isn't
        # duplicated.
        self._analysis = DifferenceOfGaussiansWavelet(
            n_channels=n_channels, dt=dt, sigma_max=sigma_max, c=c, epsilon=epsilon,
        )

    def integrate_and_fire_with_reset(self, signal, mu_mem, threshold, beta, return_trace=False):
        return ti.integrate_and_fire_with_reset(
            signal, mu_mem, threshold, self.dt, beta,
            surrogate_type=self.surrogate_type,
            return_trace=return_trace
        )

    def __call__(self, params, signal, return_trace: bool = False):
        """Spike-domain analysis: returns (2*n_channels, T) spike trains.

        Each channel's bandpass input is computed by the non-spiking sibling
        `self._analysis`, prescaled, and encoded with a pair of LIF
        neurons (positive- and negative-going). The output stacks the pairs
        as `[pos_0, neg_0, pos_1, neg_1, ..., pos_K, neg_K]` (K bandpass plus
        one lowpass row) so each physical neuron has its own row.

        If `return_trace=True`, also returns the matching `(2*n_channels, T)`
        membrane potential traces.
        """
        thresholds = jnp.exp(params["log_threshold"])
        mu_mem = jnp.exp(params["log_mu_mem"])
        beta = params.get("surrogate_beta", 10.0)
        if self.enable_normalization:
            prescale_factors = 1.0 / (jnp.array(self._filter_norms) + 1e-10)
        else:
            prescale_factors = jnp.ones(self.n_channels)

        bandpass_inputs = self._analysis(self._analysis.params, signal)

        all_spikes = []
        traces = [] if return_trace else None
        for i in range(self.n_channels):
            scaled = bandpass_inputs[i] * prescale_factors[i]
            if return_trace:
                spikes_pos, trace_pos = self.integrate_and_fire_with_reset(
                    scaled, mu_mem[i], thresholds[i], beta, return_trace=True
                )
                spikes_neg, trace_neg = self.integrate_and_fire_with_reset(
                    -scaled, mu_mem[i], thresholds[i], beta, return_trace=True
                )
                traces.extend([trace_pos, trace_neg])
            else:
                spikes_pos = self.integrate_and_fire_with_reset(
                    scaled, mu_mem[i], thresholds[i], beta, return_trace=False
                )
                spikes_neg = self.integrate_and_fire_with_reset(
                    -scaled, mu_mem[i], thresholds[i], beta, return_trace=False
                )
            all_spikes.extend([spikes_pos, spikes_neg])

        spikes = jnp.array(all_spikes)
        if return_trace:
            return spikes, jnp.array(traces)
        return spikes

    def reconstruct(self, params, spikes, return_trace: bool = False):
        """Spike-domain synthesis: collapse pos/neg spike pairs into a signed
        spike train per channel, decode each channel, then reuse the non-
        spiking sibling's `reconstruct` to combine the channels.

        Args:
            spikes: `(2*n_channels, T)` -- pos/neg LIF spike trains, ordered
                as [pos_0, neg_0, pos_1, neg_1, ...].

        If `return_trace=True`, also returns the per-channel reconstructions
        before they are combined (`(n_channels, T)`).
        """
        spikes = jnp.asarray(spikes)
        signed_spikes = spikes[0::2] - spikes[1::2]   # (n_channels, T)
        weights = jax.nn.softmax(params["log_weights"])
        channel_recons = self.channel_reconstruction_from_spikes(params, signed_spikes)
        reconstruction = self._analysis.reconstruct(
            {"channel_weights": weights}, channel_recons,
        )
        if return_trace:
            return reconstruction, channel_recons
        return reconstruction

    def channel_reconstruction_from_spikes(self, params, signed_spikes):
        """Per-channel dual-frame reconstruction from signed spike trains.

        Synthesis kernels per channel:
            i = 0       : G(sigma_1) - delta   (high-frequency band, Delta L_1)
            i = 1..K-1  : G(sigma_{i+1}) - G(sigma_i)   (bandpass differences)
            i = K       : G(sigma_K)           (lowpass residual)
        Each is composed with the per-channel membrane h_mem.

        Args:
            params: wavelet parameters dict (uses log_sigmas, log_mu_mem, log_threshold).
            signed_spikes: (n_channels, T) signed spike trains, where
                signed_spikes[i] = spikes_pos[i] - spikes_neg[i].

        Returns:
            (n_channels, T) per-channel reconstructions scaled by per-channel
            thresholds.
        """
        sigmas = jnp.exp(params["log_sigmas"])  # K smoothing scales
        thresholds = jnp.exp(params["log_threshold"])
        mu_mem = jnp.exp(params["log_mu_mem"])

        diff_recons = []
        # Channel 0: dual = G(sigma_1) - delta. The delta term passes through
        # h_mem with normalization 1/(0 - mu_mem) = -1/mu_mem, contributing
        # -h_mem * spikes / mu_mem; subtracting that gives + h_mem * spikes / mu_mem.
        recon_0 = (
            composite_gaussian_membrane(
                signed_spikes[0], self._kernels[0], mu_mem[0], sigmas[0], self.dt,
            )
            + ti.exponential_filter(signed_spikes[0], mu_mem[0], self.dt) / mu_mem[0]
        )
        diff_recons.append(recon_0)
        for i in range(1, self.n_channels - 1):
            # Synthesis kernel: Delta Psi_{i+1} = G(sigma_{i+1}) - G(sigma_i).
            recon_i = composite_gaussian_difference(
                signed_spikes[i],
                self._kernels[i], self._kernels[i - 1],
                mu_mem[i], sigmas[i], sigmas[i - 1],
                self.dt,
            )
            diff_recons.append(recon_i)
        diff_recons = jnp.array(diff_recons) * thresholds[:-1, None]

        base_recon = composite_gaussian_membrane(
            signed_spikes[-1], self._kernels[-1], mu_mem[-1], sigmas[-1], self.dt,
        ) * thresholds[-1]

        return jnp.concatenate([diff_recons, base_recon[None, :]], axis=0)

    def get_spike_trains(self, params, signal):
        """Get the raw spike trains for each channel."""
        thresholds = jnp.exp(params["log_threshold"])
        mu_mem = jnp.exp(params["log_mu_mem"])
        beta = params.get("surrogate_beta", 10.0)

        bandpass_inputs = self._analysis(self._analysis.params, signal)

        def conv(kernel):
            return fft_convolve(signal, kernel, mode='same')
        filter_responses = jax.vmap(conv)(self._kernels)  # (K, T)

        spike_trains_pos = []
        spike_trains_neg = []
        for i in range(self.n_channels):
            spikes_pos = self.integrate_and_fire_with_reset(
                bandpass_inputs[i], mu_mem[i], thresholds[i], beta
            )
            spikes_neg = self.integrate_and_fire_with_reset(
                -bandpass_inputs[i], mu_mem[i], thresholds[i], beta
            )
            spike_trains_pos.append(spikes_pos)
            spike_trains_neg.append(spikes_neg)

        return {
            "spike_trains_pos": jnp.array(spike_trains_pos),
            "spike_trains_neg": jnp.array(spike_trains_neg),
            "bandpass_inputs": bandpass_inputs,
            "scale_responses": filter_responses,
        }

    def time_scales(self):
        return jnp.exp(self.params["log_sigmas"])

    def get_readable_params(self, params=None):
        if params is None:
            params = self.params

        sigmas = jnp.exp(params["log_sigmas"])
        return {
            "sigmas": sigmas,
            "thresholds": jnp.exp(params["log_threshold"]),
            "mu_mem": jnp.exp(params["log_mu_mem"]),
            "weights": jax.nn.softmax(params["log_weights"]),
            "surrogate_beta": params.get("surrogate_beta", 10.0),
            "epsilon": self.epsilon,
        }

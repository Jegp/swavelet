"""
Spiking Difference of Exponentials (DoE) Wavelet

DoE is a special case of the Spiking DoT wavelet with cascade_depth_max=1
(a single exponential filter per scale instead of a cascade).
"""

import jax.numpy as jnp

from .spiking_dot import SpikingDoTWavelet


class SpikingDoEWavelet(SpikingDoTWavelet):
    """
    Spiking implementation of Difference of Exponentials (DoE) wavelets.

    Combines DoE filters (parallel exponential filters with differences) with
    dual-channel integrate-and-fire neurons for spike encoding.

    This is equivalent to SpikingDoTWavelet with cascade_depth_max=1.
    """

    def __init__(
        self,
        n_channels: int = 3,
        dt: float = 0.01,
        mu_max: float = 0.2,
        c: float = 2.0**0.5,
        threshold: float = None,
        surrogate_type: str = 'sigmoid',
        surrogate_beta: float = 10.0,
        enable_normalization: bool = True,
    ):
        super().__init__(
            n_channels=n_channels,
            dt=dt,
            mu_max=mu_max,
            c=c,
            cascade_depth_max=1,  # DoE is DoT with a single-exponential cascade
            threshold=threshold,
            surrogate_type=surrogate_type,
            enable_normalization=enable_normalization,
            surrogate_beta=surrogate_beta,
        )

    def get_spike_trains(self, params, signal):
        """Get spike trains with DoE-compatible keys."""
        result = super().get_spike_trains(params, signal)
        # Add DoE-specific alias
        result["doe_outputs"] = result["bandpass_inputs"]
        result["filter_responses"] = result["scale_responses"]
        return result

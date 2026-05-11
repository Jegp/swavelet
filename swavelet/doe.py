"""
Difference of Exponentials (DoE) wavelet.

DoE is a special case of the Difference of Time-causal kernels (DoT) wavelet
with cascade_depth_max=1 (a single exponential filter per scale instead of a
cascade).
"""

import jax.numpy as jnp

from .dot import DifferenceOfTimeCausalKernelsWavelet


class DifferenceOfExponentialsWavelet(DifferenceOfTimeCausalKernelsWavelet):
    """
    Difference of Exponentials (DoE) wavelet.

    Computes N single exponential filters at different temporal scales,
    then provides N-1 "difference channels" by taking the literal difference
    between consecutive filtered outputs.

    This is equivalent to DoT with cascade_depth_max=1.
    """

    def __init__(
        self,
        n_channels: int = 3,
        dt: float = 0.01,
        mu_max: float = 0.2,
        c: float = 2.0**0.5,
    ):
        super().__init__(
            n_channels=n_channels,
            dt=dt,
            mu_max=mu_max,
            c=c,
            cascade_depth_max=1,
        )
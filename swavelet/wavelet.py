import jax.numpy as jnp

class Wavelet:
    """
    Base class for all wavelets with unified interface.

    All wavelet subclasses should implement:
    - __call__(params, signal): Apply the wavelet transform and return per-channel coefficients
    - reconstruct(params, coefficients): Combine coefficients back into a reconstruction
    - time_scales(): Return characteristic time scales
    - get_readable_params(params): Return human-readable parameters
    """

    def __call__(self, params, signal):
        """Apply the wavelet transform to the input signal.

        For wavelets with bandpass + lowpass-residual structure (DoE/DoT/DoG),
        this returns the bandpass differences plus the coarsest-scale lowpass.
        For CWT-like wavelets it returns the per-channel transform coefficients.

        Returns:
            encoding: the per-channel analysis output.
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def reconstruct(self, params, coefficients) -> jnp.ndarray:
        """Combine per-channel signals into a reconstruction.

        This method implements the wavelet's synthesis stage
        (e.g. telescoping sum, CWT synthesis convolution, IDWT).
        For spike-based pipelines this is the single seam: pass
        spike-decoded output to get a reconstruction through the
        same code path.
        """
        raise NotImplementedError("Subclasses must implement reconstruct")

    def time_scales(self) -> jnp.ndarray:
        """Return characteristic time scales for each channel."""
        raise NotImplementedError("Subclasses must implement time_scales")

    def recon_time_scales(self) -> jnp.ndarray:
        """Per-channel reconstruction time constants (seconds).

        For spike-based reconstruction, spikes from channel k decay with this
        time constant. Defaults to `time_scales()` -- override when the
        reconstruction kernel has an independent scaling (e.g. DoT's
        `recon_scale_factor`).
        """
        return self.time_scales()

    def get_readable_params(self, params=None):
        """Return human-readable parameter dictionary."""
        raise NotImplementedError("Subclasses must implement get_readable_params")
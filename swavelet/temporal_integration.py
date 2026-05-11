"""
Shared temporal integration functions for cascade exponential filters.

Implements the Lindeberg time-causal temporal scale-space model.
This module also contains functions for computing stable wavelet parameters
based on dataset characteristics and extracting time constants for visualization.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np


def geometric_time_constants(mu_scale_seconds, dt, c, N):
    """Per-stage tau values (seconds) approximating Lindeberg's time-causal
    limit kernel at scale `mu_scale_seconds` with `N` cascade stages.

    Geometric tau progression in sample^2 units, per-stage mu derived from
    Delta tau, then converted back to our tau-in-seconds convention so the
    existing `exponential_filter`/`apply_cascade` recursions
    (alpha = exp(-dt/tau)) produce the equivalent impulse response.

    For comparison, the "uniform" cascade has all stages at tau = mu/sqrt N -- a
    Gamma-shape approximation that converges to a Gaussian (CLT).

    Args:
        mu_scale_seconds: target scale sigma (= our mu, in seconds).
        dt: sample period in seconds.
        c: intra-cascade distribution parameter (>1); geometric tau ratio c^2.
        N: number of cascade stages.

    Returns:
        NumPy array of length N -- per-stage tau in seconds, ordered finest-to-coarsest
        (the first stage has the smallest Delta tau).
    """
    sigma_samples = float(mu_scale_seconds) / float(dt)
    tau_samples_prev = 0.0
    out = np.empty(N)
    for i in range(N):
        tau_samples_cur = sigma_samples * sigma_samples / (c ** (2 * (N - i - 1)))
        dtau = tau_samples_cur - tau_samples_prev
        # Mu_stage (in sample units) from Delta tau.
        mu_stage = (-1.0 + math.sqrt(1.0 + 4.0 * dtau)) / 2.0
        # Convert sample-unit mus to our tau-in-seconds:
        if mu_stage <= 0.0:
            raise ValueError(
                f"Non-positive mu_stage={mu_stage} at stage {i} "
                f"(sigma_samples={sigma_samples}, N={N}, c={c}). "
                "Increase mu_scale_seconds or reduce N."
            )
        out[i] = dt / math.log1p(1.0 / mu_stage)
        tau_samples_prev = tau_samples_cur
    return out


def logarithmic_time_constants(tau, K, c):
    """
    Compute time constants for logarithmic distribution of scale levels.

    From equations (18), (19), (20) in Lindeberg 2016:
        tau_k = c^(2(k-K)) * tau_max
        mu_1 = c^(1-K) * sqrt(tau_max)
        mu_k = sqrt(tau_k - tau_{k-1}) for k >= 2

    Args:
        tau: Target variance of composed kernel
        K: Number of cascade stages
        c: Distribution parameter (c > 1)

    Returns:
        Array of K time constants
    """
    sqrt_tau = jnp.sqrt(tau)

    # First time constant (eq 19)
    mu_1 = c ** (1 - K) * sqrt_tau

    # Remaining time constants (eq 20)
    mus = [mu_1]
    for k in range(2, K + 1):
        mu_k = c ** (k - K - 1) * jnp.sqrt(c**2 - 1) * sqrt_tau
        mus.append(mu_k)

    return jnp.array(mus)


def uniform_time_constants(tau, K):
    """
    Compute time constants for uniform distribution of scale levels.

    From equation (17) in Lindeberg 2016:
        mu_k = sqrt(tau_max / K)

    Args:
        tau: Target variance of composed kernel
        K: Number of cascade stages

    Returns:
        Array of K identical time constants
    """
    mu = jnp.sqrt(tau / K)
    return jnp.ones(K) * mu


def time_causal_limit_kernel_np(mus_cascade, dt, length):
    """Numpy impulse response of the cascade for length `length`.

    For DoT, this approximates the time-causal limit kernel
    Psi(t; mu, c) as a finite cascade of truncated exponentials.
    Returned as a numpy array so the caller can store it as a
    constant (no JAX tracing) and FFT-convolve it against arbitrary
    inputs inside jit-compiled code.
    """
    import scipy.signal
    impulse = np.zeros(length)
    impulse[0] = 1.0
    u = impulse
    for tau in np.asarray(mus_cascade, dtype=float):
        alpha = float(np.exp(-dt / tau))
        # h_exp[n] = (1 - alpha) * alpha^n  -> y[n] = alpha y[n-1] + (1-alpha) x[n]
        u = scipy.signal.lfilter([1.0 - alpha], [1.0, -alpha], u)
    return u


def exponential_filter(signal, mu, dt):
    """
    Single exponential filter (first-order integrator).

    Implements equation (15) from Lindeberg 2016:
        d_t L = (1/mu) * (input - L)

    Which corresponds to equation (10):
        h_exp(t; mu) = (1/mu) * exp(-t/mu)

    Discrete form:
        L[n+1] = exp(-dt/mu) * L[n] + (1 - exp(-dt/mu)) * input[n]

    Args:
        signal: Input signal
        mu: Time constant
        dt: Time step

    Returns:
        Filtered signal
    """
    alpha = jnp.exp(-dt / mu)

    def scan_fn(u_prev, x_current):
        # Exponential smoothing (leaky integration)
        u_new = alpha * u_prev + (1 - alpha) * x_current
        return u_new, u_new

    _, u_trace = jax.lax.scan(scan_fn, 0.0, signal)
    return u_trace


def apply_cascade(signal, mus, dt, return_intermediates=False):
    """
    Apply cascade of K exponential filters with time constants mus.

    Each stage applies: u_{k+1} = h_exp(mu_k) * u_k
    where h_exp is the exponential kernel from equation (10).

    This is implemented via the differential equation (15):
        d_t L(t; tau_k) = (1/mu_k) * (L(t; tau_{k-1}) - L(t; tau_k))

    Args:
        signal: Input signal
        mus: Array of K time constants [mu_1, mu_2, ..., mu_K]
        dt: Time step
        return_intermediates: If True, return all intermediate states

    Returns:
        If return_intermediates=False:
            Output after cascading all K filters
        If return_intermediates=True:
            (final_output, intermediates) where intermediates is (K, time)
    """
    K = len(mus)
    u = signal

    if return_intermediates:
        intermediates = []
        for k in range(K):
            u = exponential_filter(u, mus[k], dt)
            intermediates.append(u)
        return u, jnp.array(intermediates)
    else:
        # Just cascade through without storing
        for k in range(K):
            u = exponential_filter(u, mus[k], dt)
        return u


def cascade_kernel(t, mus, dt):
    """
    Compute the cascade kernel in time domain by applying to impulse.

    This is the convolution of K exponential kernels:
        h_composed(t) = (h_exp(mu_1) * h_exp(mu_2) * ... * h_exp(mu_K))(t)

    Args:
        t: Time points
        mus: Array of K time constants
        dt: Time step

    Returns:
        Cascade kernel evaluated at time points t
    """
    # Create impulse signal
    impulse = jnp.zeros_like(t)
    impulse = impulse.at[0].set(1.0 / dt)

    # Apply cascade to impulse
    return apply_cascade(impulse, mus, dt)


def composite_exponential_kernel(signal, mu1, mu2, dt, apply_normalization=True):
    """
    Compute convolution of two exponential filters: exp(mu1) (x) exp(mu2).

    From the analytical formula for convolution of two exponentials:
        h_1 (x) h_2 = 1/(mu_1 - mu_2) * [e^(-t/mu_1) - e^(-t/mu_2)]

    The normalization factor 1/(mu_1 - mu_2) is critical for correct amplitude.

    Args:
        signal: Input signal
        mu1: First exponential time constant
        mu2: Second exponential time constant
        dt: Time step
        apply_normalization: If True, apply 1/(mu_1 - mu_2) normalization from
                            the analytical convolution formula

    Returns:
        Signal filtered by composite kernel
    """
    # Cascade two exponential filters
    alpha1 = jnp.exp(-dt / mu1)
    alpha2 = jnp.exp(-dt / mu2)

    def scan_fn(state, x):
        u1, u2 = state
        u1_new = alpha1 * u1 + (1 - alpha1) * x
        u2_new = alpha2 * u2 + (1 - alpha2) * u1_new
        return (u1_new, u2_new), u2_new

    _, result = jax.lax.scan(scan_fn, (0.0, 0.0), signal)

    if apply_normalization:
        # Apply the analytical normalization factor: 1/(μ₁ - μ₂)
        # This comes from the convolution formula:
        #   exp(mu_1) \circ exp(mu_2) = 1/(mu_1 - mu_2) * [e^(-t/mu_1) - e^(-t/mu_2)]
        normalization = 1.0 / (mu1 - mu2 + 1e-10)
        result = result * normalization

    return result


def composite_difference_kernel(signal, mu_fine, mu_coarse, mu_mem, dt):
    """
    Composite reconstruction kernel for DoE + membrane integration.

    Uses the analytical convolution formula for exponentials:
        conv(exp(mu_1), exp(mu_2)) = 1/(mu_1 - mu_2) * [e^(-t/mu_1) - e^(-t/mu_2)]

    The normalization factors 1/(mu - tau_mem) are critical for correct amplitude
    when reconstructing from spikes or membrane outputs.

    Args:
        signal: Input signal (typically signed spikes)
        mu_fine: Time constant of finer/faster filter
        mu_coarse: Time constant of coarser/slower filter
        mu_mem: Membrane time constant used during spike encoding
        dt: Time step

    Returns:
        Reconstructed signal using composite kernel
    """
    # First term: conv(exp(mu_mem), exp(mu_coarse)) with normalization
    # Normalization factor: 1/(mu_coarse - mu_mem)
    term1 = composite_exponential_kernel(signal, mu_coarse, mu_mem, dt)

    # Second term: conv(exp(mu_mem), exp(mu_fine)) with normalization
    # Normalization factor: 1/(mu_fine - mu_mem)
    term2 = composite_exponential_kernel(signal, mu_fine, mu_mem, dt)

    # Difference gives the full composite DoE + membrane kernel
    return term2 - term1


def difference_kernel_filter(signal, mu_fine, mu_coarse, dt):
    """
    Apply difference-of-exponentials kernel for reconstruction.

    This implements the dual frame reconstruction for DoE/DoT wavelets:
        h_DoE(t) = h_exp(t; mu_coarse) - h_exp(t; mu_fine)

    where mu_fine < mu_coarse (finer/faster scale first).

    NOTE: For spiking wavelets with membrane integration, use
    composite_difference_kernel() instead to account for tau_mem.

    Args:
        signal: Input signal (typically signed spikes)
        mu_fine: Time constant of finer/faster filter
        mu_coarse: Time constant of coarser/slower filter
        dt: Time step

    Returns:
        Filtered signal using difference kernel
    """
    # Apply both exponential filters
    filtered_coarse = exponential_filter(signal, mu_coarse, dt)
    filtered_fine = exponential_filter(signal, mu_fine, dt)

    # Return difference (coarse - fine matches DoE convention)
    # DoE_k = L_{k+1} - L_k where k+1 is coarser (larger mu)
    return filtered_coarse - filtered_fine


def difference_cascade_kernel_filter(signal, mus_fine, mus_coarse, dt):
    """
    Apply difference-of-cascaded-exponentials kernel for DoT reconstruction.

    This implements the dual frame reconstruction for DoT wavelets:
        h_DoT(t) = Psi(t; tau_coarse, c) - Psi(t; tau_fine, c)

    where Psi is the time-causal limit kernel (cascade of exponentials).

    Following the frame theory:
        f(t) = L_base(t) + Sigma_k [spikes_k (x) (Psi(tau_k) - Psi(tau_{k-1}))]

    Args:
        signal: Input signal (typically signed spikes)
        mus_fine: Array of time constants for finer/faster cascade
        mus_coarse: Array of time constants for coarser/slower cascade
        dt: Time step

    Returns:
        Filtered signal using DoT difference kernel
    """
    # Apply both cascaded filters
    filtered_coarse = apply_cascade(signal, mus_coarse, dt)
    filtered_fine = apply_cascade(signal, mus_fine, dt)

    # Return difference
    return filtered_coarse - filtered_fine


def composite_cascade_kernel(signal, mus_cascade, mu_mem, dt):
    """Apply cascade Psi(mu) followed by h_mem with 1/(mu_eff - mu_mem) normalization."""
    cascaded = apply_cascade(signal, mus_cascade, dt)
    result = exponential_filter(cascaded, mu_mem, dt)
    # Effective time constant: sqrt of sum of squared time constants
    mu_eff = jnp.sqrt(jnp.sum(mus_cascade ** 2))
    normalization = 1.0 / (mu_eff - mu_mem + 1e-10)
    return result * normalization


def composite_cascade_difference_kernel(signal, mus_fine, mus_coarse, mu_mem, dt):
    """Apply [Psi(mu_fine) - Psi(mu_coarse)] (x) h_mem with per-term 1/(mu_eff - mu_mem) normalization."""
    term_fine = composite_cascade_kernel(signal, mus_fine, mu_mem, dt)
    term_coarse = composite_cascade_kernel(signal, mus_coarse, mu_mem, dt)
    return term_fine - term_coarse


def difference_wavelet_reconstruction(
    encoded_differences,
    encoded_base,
    difference_kernels,
    base_kernel,
    weights=None
):
    """
    Generic reconstruction for difference-based wavelets using frame theory.

    Implements: f(t) = weight_base * [base (x) base_kernel] + Sigma_k weight_k * [diff_k (x) diff_kernel_k]

    Args:
        encoded_differences: List/array of n_diff_channels encoded difference signals
                            (e.g., spike trains or continuous signals)
        encoded_base: Encoded base scale signal
        difference_kernels: List of n_diff_channels functions, each with signature
                           kernel_fn(signal) -> reconstructed
        base_kernel: Function with signature kernel_fn(signal) -> reconstructed
        weights: Optional array (n_diff_channels + 1,) of channel weights, where:
                - weights[0:n_diff_channels] are for difference channels
                - weights[-1] is for the base channel
                If None, uses uniform weights for all channels.

    Returns:
        Reconstructed signal
    """
    n_diff_channels = len(encoded_differences)

    if weights is None:
        # Uniform weights for all channels (differences + base)
        weights = jnp.ones(n_diff_channels + 1) / (n_diff_channels + 1)

    # Reconstruct base scale with its weight (last weight)
    base_recon = weights[-1] * base_kernel(encoded_base)

    # Reconstruct each difference channel
    diff_recons = []
    for i in range(n_diff_channels):
        recon = difference_kernels[i](encoded_differences[i])
        diff_recons.append(weights[i] * recon)

    # Sum: f = weighted_base + sum of weighted_differences
    reconstruction = base_recon + jnp.sum(jnp.array(diff_recons), axis=0)

    return reconstruction


def get_time_constants(wavelet_name, params):
    """
    Extract time constants from wavelet parameters for visualization.

    Returns a dict with time constant arrays for doe, dot, spiking_doe, spiking_dot.
    Returns None for other wavelet types.

    Args:
        wavelet_name: Name of the wavelet (doe, dot, spiking_doe, spiking_dot, etc.)
        params: Dictionary of wavelet parameters

    Returns:
        Dictionary with time constant information, or None for non-temporal wavelets
    """
    if wavelet_name == "doe":
        # DoE: time constants are sqrt of variances (taus)
        taus = jnp.exp(params["log_taus"])
        time_constants = jnp.sqrt(taus)
        return {"filter_time_constants": time_constants}

    elif wavelet_name == "dot":
        # DoT: tau_scales are the temporal scales
        tau_scales = jnp.exp(params["log_tau_scales"])
        return {"tau_scales": tau_scales}

    elif wavelet_name == "spiking_doe":
        # Spiking DoE: filter time constants + membrane time constants
        taus = jnp.exp(params["log_taus"])
        filter_time_constants = jnp.sqrt(taus)
        tau_mem = jnp.exp(params["log_tau_mem"])
        return {
            "filter_time_constants": filter_time_constants,
            "membrane_time_constants": tau_mem
        }

    elif wavelet_name == "spiking_dot":
        # Spiking DoT: tau_scales + per-channel membrane time constants
        tau_scales = jnp.exp(params["log_tau_scales"])
        tau_mem = jnp.exp(params["log_tau_mem"])
        return {
            "tau_scales": tau_scales,
            "membrane_time_constants": tau_mem  # Now per-channel like spiking_doe
        }

    return None


def bandpass_filter_norms(mus, dt, cascade_depth=1):
    """
    Compute L2 norms of the bandpass filters for each spiking channel.

    For cascade_depth=1 (DoE): analytical formula via geometric series.
    For cascade_depth>1 (DoT): numerical impulse response.

    Each bandpass filter for channel i is:
        psi_i = cascade(mu_i) - cascade(mu_{i+1})   (fine minus coarse)
    The final entry is the base channel (coarsest scale, no differencing).

    Args:
        mus: Array of K scale time constants, fine->coarse order
        dt: Time step
        cascade_depth: Number of leaky integrators cascaded per scale

    Returns:
        norms: numpy array of shape (K,) -- L2 norms of (K-1) bandpass
               filters followed by the base-channel filter norm
    """
    import numpy as np
    mus = np.asarray(mus, dtype=float)
    K = len(mus)
    norms = np.zeros(K)

    if cascade_depth == 1:
        alphas = np.exp(-dt / mus)
        h_norms_sq = (1 - alphas) / (1 + alphas)
        for i in range(K - 1):
            ai, aj = alphas[i], alphas[i + 1]
            cross = (1 - ai) * (1 - aj) / (1 - ai * aj)
            norm_sq = h_norms_sq[i] + h_norms_sq[i + 1] - 2 * cross
            norms[i] = np.sqrt(max(norm_sq, 1e-30))
        norms[-1] = np.sqrt(h_norms_sq[-1])
    else:
        # Numerical: apply cascade to a unit impulse, take L2 norm
        n_steps = int(np.ceil(1.0 * float(mus[-1]) / dt)) + 1
        scale_resps = []
        for mu in mus:
            alpha = np.exp(-dt / float(mu))
            resp = np.zeros(n_steps)
            resp[0] = 1.0
            for _ in range(cascade_depth):
                out = np.zeros(n_steps)
                u = 0.0
                for t in range(n_steps):
                    u = alpha * u + (1 - alpha) * resp[t]
                    out[t] = u
                resp = out
            scale_resps.append(resp)
        for i in range(K - 1):
            bp = scale_resps[i] - scale_resps[i + 1]
            norms[i] = np.sqrt(np.sum(bp ** 2))
        norms[-1] = np.sqrt(np.sum(scale_resps[-1] ** 2))

    return norms


def compute_stable_wavelet_params(n_channels, dt, cascade_depth=7, target_alpha_min=0.01,
                                   f_min=None, f_max=None, wavelet_type='dot',
                                   enable_normalization=False):
    """
    Compute tau_max and c for DoT/DoE wavelets based on desired frequency coverage.

    The wavelet channels span from tau_min (highest frequency) to tau_max (lowest frequency).
    For both DoE and DoT, center frequency relates to tau as f ~ 1/(2*pi*sqrt(tau)).

    This function adaptively finds stable parameters by:
    - For DoT: adjusting cascade_depth and c to ensure alpha >= target_alpha_min
    - For DoE: adjusting c to ensure single exponential stability

    Args:
        n_channels: Number of wavelet channels
        dt: Time step (1/sample_rate)
        cascade_depth: Initial cascade depth (default: 7). May be reduced for stability.
        target_alpha_min: Minimum acceptable alpha value (default: 0.01)
        f_min: Minimum frequency to cover (Hz). If None, defaults to 0.5 Hz.
        f_max: Maximum frequency to cover (Hz). If None, defaults to Nyquist/4.
        wavelet_type: 'dot' or 'doe' - determines stability checking approach
        enable_normalization: If True, relax the minimum-c floor for DoT from 1.3
            to 1.05. Amplitude collapse from narrow channels is then handled by
            bandpass_filter_norms; c must still be > 1 for channels to be distinct.

    Returns: (tau_max, c)
    """
    import numpy as np

    sample_rate = 1.0 / dt
    nyquist = sample_rate / 2

    # Default frequency range based on signal characteristics
    if f_min is None:
        f_min = 0.5  # 0.5 Hz default minimum
    if f_max is None:
        f_max = nyquist / 4  # Conservative: Nyquist/4 to avoid aliasing issues

    # Clamp f_max to reasonable limit
    f_max = min(f_max, nyquist / 2)

    # Convert frequencies to variance parameters
    # For both DoE and DoT, the center frequency relates to tau as f ~ 1/(2*pi*sqrt(tau))
    tau_max = (1.0 / (2 * np.pi * f_min)) ** 2  # Coarsest scale (lowest frequency)
    tau_min_target = (1.0 / (2 * np.pi * f_max)) ** 2  # Finest scale (highest frequency)

    # Compute ideal c from desired tau range: tau_min = tau_max * c^(-2*(n-1))
    if n_channels > 1:
        tau_ratio = tau_max / tau_min_target
        c_ideal = tau_ratio ** (1.0 / (2 * (n_channels - 1)))
    else:
        c_ideal = 2 ** 0.5  # Default for single channel

    # Adaptive stability adjustment
    if wavelet_type == 'dot':
        # Without normalization, small c causes near-zero bandpass amplitudes, so
        # enforce a generous floor. With normalization the amplitude is corrected,
        # but c must still be meaningfully > 1 so adjacent channels are distinct.
        MIN_C_DOT = 1.05 if enable_normalization else 1.3

        # Use either MIN_C_DOT or the ideal c, whichever is larger
        c = max(MIN_C_DOT, c_ideal)

        # Compute minimum stable tau at finest cascade subdivision
        # We need: alpha = exp(-dt/mu_0) >= target_alpha_min
        # => mu_0 >= dt / (-ln(target_alpha_min))
        mu_0_min = dt / (-np.log(target_alpha_min))

        # From mu_0 = (-1 + sqrt(1 + 4*tau_level_0)) / 2, solve for tau_level_0:
        # tau_level_0 = mu_0 * (mu_0 + 1)
        tau_level_0_min = mu_0_min * (mu_0_min + 1)

        # For DoT with cascade, the finest subdivision is:
        # tau_level_0 = tau_min / c^(2*(cascade_depth-1))
        # where tau_min = tau_max * c^(-2*(n_channels-1))

        # Combined: tau_level_0 = tau_max * c^(-2*(n_channels + cascade_depth - 2))
        # For stability: tau_level_0 >= tau_level_0_min
        # => tau_max * c^(-2*(n_channels + cascade_depth - 2)) >= tau_level_0_min

        # Instead of changing tau_max, adjust cascade_depth to meet stability
        # Solve for maximum cascade_depth:
        # tau_max * c^(-2*(n_channels + d - 2)) >= tau_level_0_min
        # c^(-2*(n_channels + d - 2)) >= tau_level_0_min / tau_max
        # -2*(n_channels + d - 2) >= ln(tau_level_0_min / tau_max) / ln(c)
        # d <= 2 - n_channels - ln(tau_level_0_min / tau_max) / (2 * ln(c))

        max_cascade_depth = 2 - n_channels - np.log(tau_level_0_min / tau_max) / (2 * np.log(c))

        if cascade_depth > max_cascade_depth:
            # Reduce cascade_depth for stability (preserves frequency range)
            cascade_depth = max(3, int(np.floor(max_cascade_depth)))
            # Note: Reduces internal smoothness but preserves frequency coverage

        tau_min_actual = tau_max * (c ** (-2 * (n_channels - 1)))

    else:  # wavelet_type == 'doe'
        # DoE: just ensure mu >= dt / (-ln(target_alpha_min))
        mu_min_required = dt / (-np.log(target_alpha_min))
        tau_min_required = mu_min_required ** 2
        min_c_doe = 1.05  # always: c must be meaningfully > 1

        if tau_min_target < tau_min_required:
            # Can't achieve target f_max, need larger tau_min
            tau_min_actual = tau_min_required
            # Recompute c for this tau_min
            if n_channels > 1:
                tau_ratio = tau_max / tau_min_actual
                c = tau_ratio ** (1.0 / (2 * (n_channels - 1)))
                c = max(min_c_doe, c)
            else:
                c = 2 ** 0.5
        else:
            # Target is achievable: use c_ideal so the full [f_min, f_max] band is covered.
            # Stability is already enforced by tau_min_required above; no upper cap needed.
            c = max(min_c_doe, c_ideal)
            tau_min_actual = tau_max * (c ** (-2 * (n_channels - 1)))

    if wavelet_type == 'dot':
        return tau_max, c, cascade_depth
    else:
        return tau_max, c


# ==============================================================================
# Reconstruction Weight Functions
# ==============================================================================

def compute_reconstruction_weight(theta_thr, mu1, mu2, isi, dt=None):
    """
    Per-spike reconstruction weight from the composed exponential derivation.

    Given a spike at time t1 from a bandpass kernel kappa(t; mu_1, mu_2) integrated
    by a LIF neuron, estimates the constant input amplitude I_1 that caused
    the spike over interval [t_0, t_1] where isi = t_1 - t_0.

    The bandpass kernel is normalized as:
        kappa(t; mu_1, mu_2) = (mu_1^-^1 e^{-t/mu_1} - mu_2^-^1 e^{-t/mu_2}) / C_kappa
    with C_kappa = (mu_2 - mu_1) / (mu_1 mu_2).

    Assuming constant input I_1 over [t_0, t_1]:
        I_1 = theta_thr * C_kappa / (e^{-Delta t/mu_2} - e^{-Delta t/mu_1})

    where Delta t = t_1 - t_0 and C_kappa = (mu_2 - mu_1)/(mu_1 mu_2).

    In discrete time the leaky integrator accumulates (1 - alpha) ~= dt/mu per
    step instead of the continuous 1/mu, so C_kappa must be divided by dt to
    match the discrete gain.  When dt is provided this correction is applied
    automatically; all arguments (mu1, mu2, isi) should then be in seconds.

    If mu1 and mu2 are close, the kernel approaches a single exponential with
    time constant mu1 ~= mu2, and the reconstruction weight approaches theta_thr.

    Args:
        theta_thr: Spiking threshold (scalar or array)
        mu1: Finer/faster time constant in seconds (mu_1 < mu_2)
        mu2: Coarser/slower time constant in seconds
        isi: Inter-spike interval(s) Delta t = t_1 - t_0 in seconds
        dt: Time step in seconds.  When given, applies the discrete-time
            correction (divides C_kappa by dt).

    Returns:
        Estimated constant input amplitude(s) I_1
    """
    if mu2 - mu1 < 1e-5:
        return theta_thr / (dt if dt is not None else 1.0)  # limit as mu2 -> mu1 is theta
    c_kappa = (mu2 - mu1) / (mu1 * mu2 + 1e-10)
    if dt is not None:
        c_kappa = c_kappa / dt
    exp_diff = 1 / (jnp.exp(-isi / mu2) - jnp.exp(-isi / mu1))
    return theta_thr * c_kappa * exp_diff


def compute_spike_isis(spike_indices, dt):
    """
    Compute inter-spike intervals from spike index arrays.

    For the first spike, the ISI is the time from t=0 to the spike.

    Args:
        spike_indices: Array of spike time indices (integer)
        dt: Time step

    Returns:
        Array of inter-spike intervals (same length as spike_indices)
    """
    if len(spike_indices) == 0:
        return jnp.array([])
    indices = jnp.array(spike_indices, dtype=jnp.float32)
    isis = jnp.concatenate([indices[:1], jnp.diff(indices)]) * dt
    return isis


# ==============================================================================
# Spiking Neuron Functions
# ==============================================================================

def surrogate_spike_function(x, beta=10.0, surrogate_type='sigmoid'):
    """
    Simple surrogate spike function for gradient-based learning.
    Forward: Heaviside step
    Backward: Smooth approximation

    Args:
        x: Membrane potential relative to threshold (u - threshold)
        beta: Steepness parameter for surrogate gradient
        surrogate_type: Type of surrogate ('sigmoid', 'fast_sigmoid', 'arctan')

    Returns:
        Binary spikes {0, 1} with smooth gradients for backpropagation
    """
    # Forward pass: Heaviside
    spikes = jnp.where(x >= 0, 1.0, 0.0)

    # Backward: smooth approximation for gradients
    if surrogate_type == 'sigmoid':
        smooth_spikes = jax.nn.sigmoid(beta * x)
    elif surrogate_type == 'fast_sigmoid':
        smooth_spikes = 0.5 * (1 + x / (1 + jnp.abs(x)))
    elif surrogate_type == 'arctan':
        smooth_spikes = 0.5 * (1 + (2/jnp.pi) * jnp.arctan(beta * x))
    else:
        smooth_spikes = jax.nn.sigmoid(beta * x)

    # Straight-through estimator
    return spikes + smooth_spikes - jax.lax.stop_gradient(smooth_spikes)


def integrate_and_fire_with_reset(signal, mu_mem, threshold, dt, beta=10.0,
                                   surrogate_type='sigmoid', return_trace=False):
    """
    Integrate-and-fire neuron with membrane reset after spikes.
    Uses surrogate gradients for learning.

    Uses detached reset to prevent gradient explosion: the reset operation
    doesn't backpropagate gradients through the recurrent connection.
    This prevents gradient explosion while still allowing learning of
    mu_mem and threshold through the spike generation.

    Args:
        signal: Input signal to integrate
        mu_mem: Membrane time constant
        threshold: Spike threshold
        dt: Time step
        beta: Steepness parameter for surrogate gradient (default: 10.0)
        surrogate_type: Type of surrogate ('sigmoid', 'fast_sigmoid', 'arctan')
        return_trace: If True, return (spikes, trace), else just spikes

    Returns:
        If return_trace=False: spikes only (shape: time,)
        If return_trace=True: (spikes, trace) where trace is membrane potential after reset
    """
    alpha = jnp.exp(-dt / mu_mem)

    if return_trace:
        def scan_fn(u_prev, x_current):
            # Leaky integration
            u_new = alpha * u_prev + (1 - alpha) * x_current

            # Check for spike with surrogate gradient
            # Gradients flow through here for learning threshold and mu_mem
            membrane_relative = u_new - threshold
            spike = surrogate_spike_function(
                membrane_relative, beta, surrogate_type
            )

            # Detached reset: stop gradient flow through the reset term
            # This prevents gradient explosion in long sequences while
            # preserving gradients for spike generation (threshold, mu_mem)
            reset_membrane = u_new * (1 - jax.lax.stop_gradient(spike))

            # Output spike and membrane potential (after reset) for visualization
            return reset_membrane, (spike, reset_membrane)

        _, (spikes, trace) = jax.lax.scan(scan_fn, 0.0, signal)
        return spikes, trace
    else:
        def scan_fn(u_prev, x_current):
            # Leaky integration
            u_new = alpha * u_prev + (1 - alpha) * x_current

            # Check for spike with surrogate gradient
            membrane_relative = u_new - threshold
            spike = surrogate_spike_function(
                membrane_relative, beta, surrogate_type
            )

            # Detached reset
            reset_membrane = u_new * (1 - jax.lax.stop_gradient(spike))

            return reset_membrane, spike

        _, spikes = jax.lax.scan(scan_fn, 0.0, signal)
        return spikes

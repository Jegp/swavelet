"""
Encoding efficiency experiment: nRMSE and coherence ratio across wavelet families.

For each wavelet family this script evaluates `n_eval_signals` validation
signals per dataset, sweeping the scale ratio `c` (with K derived from c).
The spiking variants use the canonical per-spike decoupled lstsq scheme
(paper Eq. per_channel_decoupled_lstsq); only that scheme is run, since
the table only reports its results.

Usage:
    python experiment_encoding_efficiency.py --n-eval-signals 100 --datasets speech mitbih
    python experiment_encoding_efficiency.py --n-eval-signals 2 --datasets mitbih  # smoke test
"""

import json
import argparse
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import jax
import jax.numpy as jnp
import tqdm

from swavelet import morlet
from swavelet import haar, szu
from swavelet import doe, dot, dog
from swavelet import spiking_doe, spiking_dot, spiking_dog

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
# Scale ratio between adjacent channels. With [f_min, f_max] fixed by the
# dataset, the bandpass count K is determined by c via K = ceil(log_c(f_max/f_min)).
C_VALUES = [2 ** 0.5, 2.0]
SPIKING_THRESHOLDS = [0.1]
N_EVAL_SIGNALS = 100
# DoT target cascade depth (paper symbol: N)
DOT_CASCADE_DEPTH_MAX = 7
SEGMENT_SIZE = 1000  # samples per segment for the segmented per-spike solve


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def get_dataset_config(dataset_name):
    configs = {
        "speech": {
            "sampling_rate": 16000,
            "dt": 1.0 / 16000,
            "chunk_size": 16000,
            "f_min": 80,
            "f_max": 8000,
        },
        "mitbih": {
            "sampling_rate": 360,
            "dt": 1.0 / 360,
            "chunk_size": 360,
            "f_min": 0.5,
            "f_max": 180,
        },
    }
    return configs[dataset_name]


def load_eval_signals(dataset_name, config, n_signals):
    """Load and z-score normalize evaluation signals from the validation split."""
    root_path = Path(__file__).parent.parent

    if dataset_name == "speech":
        import librispeech_dataset
        ds = librispeech_dataset.LibriSpeechDataset(
            root_path / "LibriSpeech",
            ["train-clean-100"],
            chunk_size=config["chunk_size"],
            split="val",
            val_fraction=0.1,
            seed=42,
        )
    else:
        import mitbih_dataset
        ds = mitbih_dataset.MITBIHDataset(
            root_path / "mitbih",
            chunk_size=config["chunk_size"],
            split="val",
            val_fraction=0.2,
            seed=42,
        )

    indices = np.random.default_rng(42).choice(len(ds), size=min(n_signals, len(ds)), replace=False)
    signals = []
    for idx in indices:
        sig = np.array(ds[int(idx)], dtype=np.float64)
        std = sig.std()
        if std > 1e-8:
            sig = (sig - sig.mean()) / std
        signals.append(sig)
    return signals


# ---------------------------------------------------------------------------
# Wavelet factory
# ---------------------------------------------------------------------------
def effective_f_max(dt, f_max):
    """Clamp f_max so the finest channel stays a comfortable margin below Nyquist."""
    nyquist = 1.0 / (2.0 * dt)
    return min(f_max, nyquist / 2.0)


def n_channels_for_c(c, dt, f_min, f_max):
    """Total channels n = K + 1, where K = ceil(log_c(f_max_eff / f_min))
    is the bandpass count and the +1 is the lowpass residual.
    f_max_eff = min(f_max, Nyquist/2).
    """
    f_max_eff = effective_f_max(dt, f_max)
    return int(1 + math.ceil(math.log(f_max_eff / f_min) / math.log(c)))


def create_all_wavelets(c, dt, f_min, f_max, chunk_size):
    """Create the wavelets reported in the table at a fixed scale ratio c.

    Total channel count `n_channels` (= K + 1, where K bandpass channels
    span [f_min, min(f_max, Nyquist/2)] plus one lowpass residual) is
    derived via `n_channels_for_c` so the finest channel stays clear of
    the leaky-integrator stability floor. All wavelet families share the
    same (c, n_channels, mu_max) so per-bit comparisons are apples-to-apples.

    Returns dict of {name: (wavelet, is_spiking)}.
    """
    f_max_eff = effective_f_max(dt, f_max)
    n_channels = n_channels_for_c(c, dt, f_min, f_max)
    mu_max = 1.0 / (2 * np.pi * f_min)

    wavelets = {}
    wavelets["morlet"] = (morlet.MorletWavelet(n_channels=n_channels, dt=dt, f_min=f_min, f_max=f_max_eff), False)
    wavelets["szu"] = (szu.SzuWavelet(n_channels=n_channels, dt=dt, f_min=f_min, f_max=f_max_eff), False)
    wavelets["dog"] = (dog.DifferenceOfGaussiansWavelet(n_channels=n_channels, dt=dt, sigma_max=mu_max, c=c), False)
    wavelets["doe"] = (doe.DifferenceOfExponentialsWavelet(n_channels=n_channels, dt=dt, mu_max=mu_max, c=c), False)
    wavelets["dot"] = (dot.DifferenceOfTimeCausalKernelsWavelet(n_channels=n_channels, dt=dt, mu_max=mu_max, c=c, cascade_depth_max=DOT_CASCADE_DEPTH_MAX), False)
    wavelets["spiking_dog"] = (spiking_dog.SpikingDoGWavelet(n_channels=n_channels, dt=dt, sigma_max=mu_max, c=c), True)
    wavelets["spiking_doe"] = (spiking_doe.SpikingDoEWavelet(n_channels=n_channels, dt=dt, mu_max=mu_max, c=c), True)
    wavelets["spiking_dot"] = (spiking_dot.SpikingDoTWavelet(n_channels=n_channels, dt=dt, mu_max=mu_max, c=c, cascade_depth_max=DOT_CASCADE_DEPTH_MAX), True)
    return wavelets, n_channels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _normalized_lstsq(A, y):
    """Solve lstsq with column-normalized A for numerical stability."""
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms > 0, col_norms, 1.0)
    A_norm = A / col_norms[None, :]
    w_norm, _, _, _ = np.linalg.lstsq(A_norm, y, rcond=None)
    return A_norm @ w_norm, w_norm / col_norms


def _coherence_ratio(signal, reconstruction, x, x_tilde):
    """Coherence ratio rho_K = ||f - tilde f|| / sqrt(sum_k ||x_k - tilde x_k||^2).

    Quantifies how the synthesis combines per-channel errors: rho_K < 1 means
    the synthesis cancels per-channel error (coherent), rho_K > 1 means it
    amplifies. Inputs are aligned on the same time grid (no lag adjustment).
    """
    num = float(np.linalg.norm(np.asarray(signal) - np.asarray(reconstruction)))
    den_sq = float(np.sum((np.asarray(x) - np.asarray(x_tilde)) ** 2))
    if den_sq <= 0:
        return float("inf") if num > 0 else 0.0
    return num / float(np.sqrt(den_sq))


def _reconstruction_metrics(signal, reconstruction):
    """Compute RMSE, nRMSE (= RMSE / signal_rms), and alignment lag."""
    corr = np.correlate(signal, reconstruction, mode='full')
    lag = int(np.argmax(corr)) - (len(reconstruction) - 1)

    if lag >= 0:
        sig_aligned = signal[lag:]
        rec_aligned = reconstruction[:len(sig_aligned)]
    else:
        rec_aligned = reconstruction[-lag:]
        sig_aligned = signal[:len(rec_aligned)]

    rmse = float(np.sqrt(np.mean((sig_aligned - rec_aligned) ** 2)))
    signal_rms = float(np.sqrt(np.mean(sig_aligned ** 2)))
    nrmse = rmse / signal_rms if signal_rms > 0 else float("inf")
    return rmse, signal_rms, nrmse, lag


# ---------------------------------------------------------------------------
# Evaluation: DWT (haar) and non-spiking CWT
# ---------------------------------------------------------------------------
def evaluate_dwt(wavelet, signal):
    """DWT evaluation (perfect reconstruction)."""
    sig_np = np.array(signal)
    coeffs = wavelet(wavelet.params, sig_np)
    recon = wavelet.reconstruct(wavelet.params, coeffs)
    recon_np = np.array(recon)
    rmse, signal_rms, nrmse, lag = _reconstruction_metrics(sig_np, recon_np[:len(sig_np)])
    reals = len(sig_np)
    return {"rmse": rmse, "signal_rms": signal_rms, "nrmse": nrmse, "lag": lag, "reals": reals}


def _wavelet_meta(wav):
    """Per-wavelet metadata (cascade depths for DoT/DoE)."""
    meta = {}
    if hasattr(wav, "cascade_depths"):
        depths = [int(d) for d in wav.cascade_depths]
        meta["cascade_depths"] = depths
        meta["cascade_depth_min"] = min(depths)
        meta["cascade_depth_max"] = max(depths)
    return meta


def _get_jitted_per_channel_basis(wavelet):
    """Cache a jit+vmap forward that builds all one-hot channel reconstructions at once."""
    if not hasattr(wavelet, '_jitted_per_ch_basis'):
        def basis(params, sig):
            encoding = wavelet(params, sig)
            def one(cw):
                p = dict(params)
                p["channel_weights"] = cw
                return wavelet.reconstruct(p, encoding)
            cw_id = jnp.eye(wavelet.n_channels)
            recons = jax.vmap(one)(cw_id)
            return recons, encoding
        wavelet._jitted_per_ch_basis = jax.jit(basis)
    return wavelet._jitted_per_ch_basis


def evaluate_nonspiking(wavelet, params, signal, dt):
    """Non-spiking wavelet with lstsq per-channel weights.

    Reals: one real per encoding component (complex counts as 2).
    """
    sig_jax = jnp.array(signal)
    n_ch = wavelet.n_channels

    if getattr(wavelet, "is_jittable", True):
        basis_fn = _get_jitted_per_channel_basis(wavelet)
        recons, encoding = basis_fn(params, sig_jax)
        A = np.asarray(recons).T
    else:
        encoding = wavelet(params, sig_jax)
        channel_recons = []
        for i in range(n_ch):
            p = dict(params)
            p["channel_weights"] = jnp.zeros(n_ch).at[i].set(1.0)
            recon_i = wavelet.reconstruct(p, encoding)
            channel_recons.append(np.asarray(recon_i))
        A = np.column_stack(channel_recons)

    recon, _ = _normalized_lstsq(A, signal)
    rmse, signal_rms, nrmse, lag = _reconstruction_metrics(signal, recon)

    enc = np.asarray(encoding)
    reals = enc.size * (2 if np.iscomplexobj(enc) else 1)

    return {"rmse": rmse, "signal_rms": signal_rms, "nrmse": nrmse, "lag": lag, "reals": reals}


# ---------------------------------------------------------------------------
# Spiking helpers: spike extraction and per-channel reconstruction kernels
# ---------------------------------------------------------------------------
def _get_jitted_forward(wavelet):
    if not hasattr(wavelet, '_jitted_call'):
        wavelet._jitted_call = jax.jit(lambda p, s: wavelet(p, s))
    return wavelet._jitted_call


def _broadcast_mu(mu_recon, n_channels):
    arr = np.atleast_1d(np.asarray(mu_recon, dtype=float))
    if arr.size == 1:
        return np.full(n_channels, float(arr[0]))
    if arr.size != n_channels:
        raise ValueError(f"mu_recon length {arr.size} != n_channels {n_channels}")
    return arr.astype(float)


def _get_per_channel_recon_tau(wavelet):
    """Per-channel reconstruction time constants (seconds) as a numpy array."""
    params = getattr(wavelet, "params", {})
    if "log_mu_recon" in params:
        return np.asarray(jnp.exp(params["log_mu_recon"]))
    if "log_sigmas" in params:
        return np.asarray(jnp.exp(params["log_sigmas"]))
    if "log_mus" in params:
        return np.asarray(jnp.exp(params["log_mus"]))
    return np.full(wavelet.n_channels, float(getattr(wavelet, "mu_max", 0.1)))


def _get_spike_indices(wavelet, params, signal, threshold):
    """Run forward pass; return (idx_pos, idx_neg, total_spikes) per channel."""
    sig_jax = jnp.array(signal)
    n_channels = wavelet.n_channels
    params_override = dict(params)
    params_override["log_threshold"] = jnp.log(jnp.ones(n_channels) * threshold)

    forward_fn = _get_jitted_forward(wavelet)
    all_spikes = forward_fn(params_override, sig_jax)
    all_spikes = np.array(all_spikes)

    all_idx_pos = [np.where(all_spikes[2 * k] > 0.5)[0] for k in range(n_channels)]
    all_idx_neg = [np.where(all_spikes[2 * k + 1] > 0.5)[0] for k in range(n_channels)]
    total_spikes = sum(len(p) + len(n) for p, n in zip(all_idx_pos, all_idx_neg))
    return all_idx_pos, all_idx_neg, total_spikes


# ---------------------------------------------------------------------------
# Decoupled per-spike reconstruction
# ---------------------------------------------------------------------------
def _per_channel_events(idx_pos, idx_neg):
    evs = [(int(t), 1.0) for t in np.asarray(idx_pos, dtype=int)]
    evs.extend((int(t), -1.0) for t in np.asarray(idx_neg, dtype=int))
    evs.sort()
    return evs


def _segment_design(seg_evs_per_ch, B, decays):
    """(K, B, n_max) padded per-spike design tensor for one segment."""
    K = len(seg_evs_per_ch)
    n_evs = np.array([len(e) for e in seg_evs_per_ch])
    n_max = int(n_evs.max()) if K > 0 else 0
    if n_max == 0:
        return np.zeros((K, B, 0)), n_evs, 0
    A_batch = np.zeros((K, B, n_max))
    j_col = np.arange(B)[:, None]
    for k in range(K):
        n_k = n_evs[k]
        if n_k == 0:
            continue
        t_k = np.array([e[0] for e in seg_evs_per_ch[k]])
        s_k = np.array([e[1] for e in seg_evs_per_ch[k]])
        diff = np.maximum(j_col - t_k, 0)
        A_batch[k, :, :n_k] = np.where(j_col >= t_k, s_k * decays[k] ** diff, 0.0)
    return A_batch, n_evs, n_max


def _segment_gram_with_ridge(A_batch, n_evs):
    """Per-channel A^T A with a tiny diagonal ridge plus identity on padded positions."""
    K, _, n_max = A_batch.shape
    G_batch = np.einsum('kti,ktj->kij', A_batch, A_batch)
    for k in range(K):
        n_k = n_evs[k]
        if n_k == 0:
            G_batch[k] = np.eye(n_max)
            continue
        diag_mean = float(np.mean(np.diag(G_batch[k, :n_k, :n_k])))
        if diag_mean > 0:
            G_batch[k, np.arange(n_k), np.arange(n_k)] += 1e-10 * diag_mean
        if n_k < n_max:
            pad = np.arange(n_k, n_max)
            G_batch[k, pad, pad] = 1.0
    return G_batch


def _batched_cho_solve(G_batch, rhs_batch, A_batch, residuals, n_evs):
    """Batched Cholesky + triangular solves; per-channel lstsq fallback on failure."""
    try:
        L = np.linalg.cholesky(G_batch)
        y = np.linalg.solve(L, rhs_batch[..., None])
        return np.linalg.solve(np.swapaxes(L, -1, -2), y)[..., 0]
    except np.linalg.LinAlgError:
        K, _, n_max = A_batch.shape
        w = np.zeros((K, n_max))
        for k in range(K):
            n_k = n_evs[k]
            if n_k == 0:
                continue
            w_k, *_ = np.linalg.lstsq(A_batch[k, :, :n_k], residuals[k], rcond=None)
            w[k, :n_k] = w_k
        return w


def _solve_segment_decoupled(seg_evs_per_ch, residuals, decays, B):
    A_batch, n_evs, n_max = _segment_design(seg_evs_per_ch, B, decays)
    if n_max == 0:
        K = len(seg_evs_per_ch)
        return np.zeros((K, B)), np.zeros(K)
    G_batch = _segment_gram_with_ridge(A_batch, n_evs)
    rhs_batch = np.einsum('kti,kt->ki', A_batch, residuals)
    w_batch = _batched_cho_solve(G_batch, rhs_batch, A_batch, residuals, n_evs)
    recon_seg = np.einsum('kti,ki->kt', A_batch, w_batch)
    last_row = np.einsum('ki,ki->k', A_batch[:, B - 1, :], w_batch)
    return recon_seg, last_row


def _reconstruct_per_spike_decoupled(signal, all_idx_pos, all_idx_neg, total_spikes, dt, mu_recon, bandpass_targets, wavelet):
    """Decoupled per-channel per-spike reconstruction (paper canonical scheme).

    For each channel `k`, solves a per-channel least squares problem fitting
    per-spike low-pass columns to that channel's analysis output `x_k`. The
    full reconstruction goes through the wavelet's `reconstruct`.

    Reals: 2 * total_spikes (timestamp + weight per spike).
    """
    N = len(signal)
    K = len(all_idx_pos)

    if total_spikes == 0:
        signal_rms = float(np.sqrt(np.mean(signal ** 2)))
        return {
            "rmse": signal_rms, "signal_rms": signal_rms, "nrmse": 1.0,
            "lag": 0, "coherence_ratio": None, "reals": 0, "total_spikes": 0,
        }

    mu_arr = _broadcast_mu(mu_recon, K)
    decays = np.exp(-dt / mu_arr)
    targets = np.asarray(bandpass_targets)

    events_per_ch = [_per_channel_events(all_idx_pos[k], all_idx_neg[k]) for k in range(K)]
    ev_ptrs = [0] * K
    reconstructions = np.zeros((K, N))
    carries = np.zeros(K)

    for seg_start in range(0, N, SEGMENT_SIZE):
        seg_end = min(seg_start + SEGMENT_SIZE, N)
        B = seg_end - seg_start
        j_arr = np.arange(B)

        carry_vecs = carries[:, None] * (decays[:, None] ** j_arr[None, :])
        residuals = targets[:, seg_start:seg_end] - carry_vecs

        seg_evs_per_ch = []
        for k in range(K):
            here = []
            evs_k = events_per_ch[k]
            while ev_ptrs[k] < len(evs_k) and evs_k[ev_ptrs[k]][0] < seg_end:
                t, sign = evs_k[ev_ptrs[k]]
                here.append((t - seg_start, sign))
                ev_ptrs[k] += 1
            seg_evs_per_ch.append(here)

        recon_seg, last_row = _solve_segment_decoupled(seg_evs_per_ch, residuals, decays, B)
        reconstructions[:, seg_start:seg_end] = carry_vecs + recon_seg
        carries = carries * (decays ** B) + decays * last_row

    analysis = wavelet._analysis
    reconstruction = np.asarray(analysis.reconstruct(analysis.params, reconstructions))
    rmse, signal_rms, nrmse, lag = _reconstruction_metrics(signal, reconstruction)
    coherence_ratio = _coherence_ratio(signal, reconstruction, targets, reconstructions)
    reals = int(total_spikes * 2)
    return {
        "rmse": rmse, "signal_rms": signal_rms, "nrmse": nrmse, "lag": lag,
        "coherence_ratio": coherence_ratio, "reals": reals, "total_spikes": total_spikes,
    }


def evaluate_spiking(wavelet, signal, threshold, dt):
    """Per-signal evaluation for a spiking wavelet using the decoupled scheme."""
    mu_recon = _get_per_channel_recon_tau(wavelet)
    all_idx_pos, all_idx_neg, total_spikes = _get_spike_indices(
        wavelet, wavelet.params, signal, threshold,
    )
    analysis = wavelet._analysis
    bandpass_targets = np.asarray(analysis(analysis.params, jnp.asarray(signal)))
    return _reconstruct_per_spike_decoupled(
        signal, all_idx_pos, all_idx_neg, total_spikes, dt, mu_recon,
        bandpass_targets, wavelet,
    )


# ---------------------------------------------------------------------------
# Parallel runners
# ---------------------------------------------------------------------------
def _run_parallel(fn, signals, n_workers, warn_label):
    """Evaluate fn(sig) over signals in parallel; returns dict of result lists."""
    keys = ("rmse", "reals", "nrmse", "lag", "spikes", "coherence")
    out = {k: [] for k in keys}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(fn, sig) for sig in signals]
        for fut in as_completed(futures):
            try:
                r = fut.result()
                out["rmse"].append(r["rmse"])
                out["reals"].append(r["reals"])
                out["nrmse"].append(r["nrmse"])
                out["lag"].append(r["lag"])
                if "total_spikes" in r:
                    out["spikes"].append(r["total_spikes"])
                if r.get("coherence_ratio") is not None:
                    out["coherence"].append(r["coherence_ratio"])
            except Exception as e:
                print(f"      WARN: {warn_label}: {e}")
    return out


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(datasets, c_values, spiking_thresholds, n_eval_signals, output_dir, mode="all", n_workers=8):
    """Run the encoding-efficiency sweep. `mode` selects which families run:
    "all" (default), "nonspiking" (haar + CWT only), or "spiking" (spiking only).
    """
    output_dir = Path(output_dir)
    all_results = []
    run_nonspiking = mode in ("all", "nonspiking")
    run_spiking = mode in ("all", "spiking")

    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        config = get_dataset_config(dataset_name)
        dt = config["dt"]
        signals = load_eval_signals(dataset_name, config, n_eval_signals)
        print(f"  Loaded {len(signals)} eval signals (chunk_size={config['chunk_size']})")

        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Haar DWT: critically sampled, runs once outside the c-sweep.
        if run_nonspiking:
            print("\n  haar_dwt (full decomposition)")
            haar_wav = haar.HaarWavelet(dt=dt)
            data = _run_parallel(
                lambda sig: evaluate_dwt(haar_wav, sig),
                signals, n_workers, "haar_dwt",
            )
        else:
            data = {"rmse": []}
        if data["rmse"]:
            result = {
                "dataset": dataset_name,
                "wavelet": "haar",
                "is_spiking": False,
                "is_dwt": True,
                "n_channels": None,
                "threshold": None,
                "rmse_mean": float(np.mean(data["rmse"])),
                "rmse_std": float(np.std(data["rmse"])),
                "nrmse_mean": float(np.mean(data["nrmse"])),
                "nrmse_std": float(np.std(data["nrmse"])),
                "lag_mean": float(np.mean(data["lag"])),
                "reals_mean": float(np.mean(data["reals"])),
                "reals_std": float(np.std(data["reals"])),
                "spikes_mean": None,
                "n_eval_signals": len(data["rmse"]),
            }
            with open(dataset_dir / "haar.json", "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)

        for c in c_values:
            wavelets, n_ch = create_all_wavelets(c, dt, config["f_min"], config["f_max"], config["chunk_size"])
            print(f"\n  c={c:.4g}  (n_channels={n_ch})")

            for wname, (wav, is_spiking) in tqdm.tqdm(
                list(wavelets.items()), desc=f"    wavelets (c={c:.3g}, n_ch={n_ch})", leave=False,
            ):
                if is_spiking and not run_spiking:
                    continue
                if not is_spiking and not run_nonspiking:
                    continue
                if is_spiking:
                    for thresh in spiking_thresholds:
                        data = _run_parallel(
                            lambda sig, t=thresh: evaluate_spiking(wav, sig, t, dt),
                            signals, n_workers, f"{wname} n_ch={n_ch} thresh={thresh}",
                        )
                        if not data["rmse"]:
                            continue
                        entry_name = f"{wname}_per_spike_decoupled"
                        model_result = {
                            "dataset": dataset_name,
                            "wavelet": entry_name,
                            "is_spiking": True,
                            "weight_scheme": "per_spike_decoupled",
                            "n_channels": n_ch,
                            "c": float(c),
                            "threshold": thresh,
                            "rmse_mean": float(np.mean(data["rmse"])),
                            "rmse_std": float(np.std(data["rmse"])),
                            "nrmse_mean": float(np.mean(data["nrmse"])),
                            "nrmse_std": float(np.std(data["nrmse"])),
                            "lag_mean": float(np.mean(data["lag"])),
                            "reals_mean": float(np.mean(data["reals"])),
                            "reals_std": float(np.std(data["reals"])),
                            "spikes_mean": float(np.mean(data["spikes"])) if data["spikes"] else None,
                            "coherence_mean": float(np.mean(data["coherence"])) if data["coherence"] else None,
                            "coherence_std": float(np.std(data["coherence"])) if data["coherence"] else None,
                            "n_eval_signals": len(data["rmse"]),
                            **_wavelet_meta(wav),
                        }
                        model_path = dataset_dir / f"{entry_name}_n{n_ch}_c{c:.3g}_thresh{thresh:.4g}.json"
                        with open(model_path, "w") as f:
                            json.dump(model_result, f, indent=2)
                        all_results.append(model_result)
                else:
                    data = _run_parallel(
                        lambda sig: evaluate_nonspiking(wav, wav.params, sig, dt),
                        signals, n_workers, f"{wname} n_ch={n_ch}",
                    )
                    if not data["rmse"]:
                        continue
                    model_result = {
                        "dataset": dataset_name,
                        "wavelet": wname,
                        "is_spiking": False,
                        "n_channels": n_ch,
                        "c": float(c),
                        "threshold": None,
                        "rmse_mean": float(np.mean(data["rmse"])),
                        "rmse_std": float(np.std(data["rmse"])),
                        "nrmse_mean": float(np.mean(data["nrmse"])),
                        "nrmse_std": float(np.std(data["nrmse"])),
                        "lag_mean": float(np.mean(data["lag"])),
                        "reals_mean": float(np.mean(data["reals"])),
                        "reals_std": float(np.std(data["reals"])),
                        "spikes_mean": None,
                        "n_eval_signals": len(data["rmse"]),
                        **_wavelet_meta(wav),
                    }
                    model_path = dataset_dir / f"{wname}_n{n_ch}_c{c:.3g}.json"
                    with open(model_path, "w") as f:
                        json.dump(model_result, f, indent=2)
                    all_results.append(model_result)

                jax.clear_caches()

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encoding efficiency experiment across wavelet families")
    parser.add_argument("--output-dir", type=str, default="results/encoding_efficiency",
                        help="Output directory for per-model JSON files")
    parser.add_argument("--n-eval-signals", type=int, default=N_EVAL_SIGNALS,
                        help=f"Number of eval signals per dataset (default: {N_EVAL_SIGNALS})")
    parser.add_argument("--datasets", nargs="+", default=["mitbih", "speech"],
                        choices=["mitbih", "speech"],
                        help="Datasets to evaluate (default: mitbih speech)")
    parser.add_argument("--c-values", nargs="+", type=float, default=C_VALUES,
                        help=f"Scale ratios c to sweep; K = 1 + ceil(log_c(f_max/f_min)). "
                             f"Default: {C_VALUES}")
    parser.add_argument("--thresholds", nargs="+", type=float, default=SPIKING_THRESHOLDS,
                        help=f"Spiking thresholds to sweep (default: {SPIKING_THRESHOLDS})")
    parser.add_argument("--mode", choices=["all", "spiking", "nonspiking"], default="all",
                        help="Which families to evaluate (default: all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel threads for signal evaluation (default: 8)")
    args = parser.parse_args()

    results = run_experiment(
        datasets=args.datasets,
        c_values=args.c_values,
        spiking_thresholds=args.thresholds,
        n_eval_signals=args.n_eval_signals,
        output_dir=args.output_dir,
        mode=args.mode,
        n_workers=args.workers,
    )

    print(f"\nSaved {len(results)} results to {args.output_dir}")

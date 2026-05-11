"""
Diagnostic comparison: Morlet vs Spiking DoE (per-channel weights) at 16 channels on MIT-BIH.

Run from project root:
    python examples/spiking_doe_diagnostic.py
"""
# %%

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "neurips"))

import temporal_integration as ti
import morlet
import spiking_doe
import mitbih_dataset

# %%
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_CHANNELS = 16
DATASET_ROOT = ROOT / "mitbih"
CHUNK_SIZE = 360
DT = 1.0 / 360
F_MIN = 0.5
F_MAX = 45.0
N_SIGNALS = 10
SEED = 42
THRESHOLDS = [0.001, 0.01, 0.1]

# %%
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sep(title="", width=70):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'=' * pad} {title} {'=' * pad}")
    else:
        print("=" * width)


def load_signals(n):
    ds = mitbih_dataset.MITBIHDataset(
        DATASET_ROOT, chunk_size=CHUNK_SIZE, split="val", val_fraction=0.2, seed=SEED
    )
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    signals = []
    for idx in indices:
        sig = np.array(ds[int(idx)], dtype=np.float64)
        std = sig.std()
        if std > 1e-8:
            sig = (sig - sig.mean()) / std
        signals.append(sig)
    return signals


def build_channel_matrix(all_idx_pos, all_idx_neg, mu_recon, N):
    n_ch = len(all_idx_pos)
    A = np.zeros((N, n_ch))
    n = np.arange(N)
    for k, (idx_p, idx_n) in enumerate(zip(all_idx_pos, all_idx_neg)):
        for tau_arr, sign in [(idx_p, 1.0), (idx_n, -1.0)]:
            if len(tau_arr) == 0:
                continue
            elapsed = (n[:, None] - np.array(tau_arr, dtype=int)[None, :]) * DT
            safe_decay = np.exp(-np.clip(elapsed, 0, None) / mu_recon)
            A[:, k] += sign * np.where(elapsed >= 0, safe_decay, 0.0).sum(axis=1)
    return A


def normalized_lstsq(A, y):
    """Solve lstsq with column-normalized A for numerical stability.

    Returns (reconstruction, weights_in_original_scale).
    """
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms > 0, col_norms, 1.0)
    A_norm = A / col_norms[None, :]
    w_norm, _, _, _ = np.linalg.lstsq(A_norm, y, rcond=None)
    return A_norm @ w_norm, w_norm / col_norms

# %%
# ---------------------------------------------------------------------------
# Morlet diagnostics
# ---------------------------------------------------------------------------
def diagnose_morlet(wav, signals):
    sep("MORLET — 16 channels")
    jit_fwd = jax.jit(lambda p, s: wav(p, s))

    all_rmse, all_reals, all_C = [], [], []

    for sig_idx, sig in enumerate(signals):
        sig_jax = jnp.array(sig)
        recon_native, encoding = jit_fwd(wav.params, sig_jax)
        recon_native = np.array(recon_native)
        enc_np = np.array(encoding)

        # Fit scalar frame normalization constant C
        C = np.dot(recon_native, sig) / np.dot(recon_native, recon_native)
        recon = C * recon_native
        rmse = float(np.sqrt(np.mean((sig - recon) ** 2)))

        n_enc_reals = enc_np.size * (2 if np.iscomplexobj(enc_np) else 1)
        reals = n_enc_reals + 1  # +1 for C
        all_rmse.append(rmse)
        all_reals.append(reals)
        all_C.append(C)

        if sig_idx == 0:
            print(f"\n  [Signal 0 detail]")
            print(f"  Encoding shape : {enc_np.shape}  dtype={enc_np.dtype}")
            print(f"  Reals          : {reals:,}  ({n_enc_reals} encoding + 1 for C)")
            print(f"  C              : {C:.6f}")
            print(f"  RMSE (native)  : {float(np.sqrt(np.mean((sig - recon_native) ** 2))):.6f}")
            print(f"  RMSE (scaled)  : {rmse:.6f}")

    print(f"\n  Summary over {len(signals)} signals:")
    print(f"    C      mean={np.mean(all_C):.6f}  std={np.std(all_C):.6f}")
    print(f"    RMSE   mean={np.mean(all_rmse):.6f}  std={np.std(all_rmse):.6f}"
          f"  min={np.min(all_rmse):.6f}  max={np.max(all_rmse):.6f}")
    print(f"    Reals  mean={np.mean(all_reals):,.0f}")
    return float(np.mean(all_rmse)), float(np.mean(all_reals))

# %%
# ---------------------------------------------------------------------------
# Spiking DoE per-channel diagnostics
# ---------------------------------------------------------------------------
def diagnose_spiking_doe(wav, signals, threshold):
    sep(f"SPIKING DoE — 16 channels  threshold={threshold}")
    mu_recon = float(wav.mu_max)
    N = CHUNK_SIZE
    jit_fwd = jax.jit(lambda p, s: wav(p, s))

    print(f"  mu_recon = {mu_recon:.6f} s")

    all_rmse_before, all_rmse_after, all_reals, all_total_spikes = [], [], [], []

    for sig_idx, sig in enumerate(signals):
        sig_jax = jnp.array(sig)
        params_thr = dict(wav.params)
        params_thr["log_threshold"] = jnp.log(jnp.ones(wav.n_channels) * threshold)

        _, spikes_arr = jit_fwd(params_thr, sig_jax)
        spikes_arr = np.array(spikes_arr)  # (2*n_ch, time)

        n_pairs = spikes_arr.shape[0] // 2
        all_idx_pos = [np.where(spikes_arr[2 * k] > 0.5)[0] for k in range(n_pairs)]
        all_idx_neg = [np.where(spikes_arr[2 * k + 1] > 0.5)[0] for k in range(n_pairs)]
        spikes_pos = [len(p) for p in all_idx_pos]
        spikes_neg = [len(n) for n in all_idx_neg]
        total_spikes = sum(spikes_pos) + sum(spikes_neg)
        all_total_spikes.append(total_spikes)

        A_ch = build_channel_matrix(all_idx_pos, all_idx_neg, mu_recon, N)

        # Before optimization: unit weights on each channel
        recon_before = A_ch.sum(axis=1)
        rmse_before = np.sqrt(np.mean((sig - recon_before) ** 2))
        all_rmse_before.append(rmse_before)

        # After optimization: lstsq per-channel weights
        if total_spikes > 0:
            recon_after, w_opt = normalized_lstsq(A_ch, sig)
        else:
            w_opt = np.zeros(wav.n_channels)
            recon_after = np.zeros(N)
        rmse_after = np.sqrt(np.mean((sig - recon_after) ** 2))
        all_rmse_after.append(rmse_after)

        reals = total_spikes * (2 if np.iscomplexobj(sig) else 1)
        all_reals.append(reals)

        if sig_idx == 0:
            print(f"\n  [Signal 0 detail]")
            print(f"  Total spikes        : {total_spikes}")
            print(f"  Reals               : {reals:,}")
            print(f"  RMSE before opt     : {rmse_before:.6f}  (unit weights)")
            print(f"  RMSE after opt      : {rmse_after:.6f}  (lstsq weights)")

            sv = np.linalg.svd(A_ch, compute_uv=False)
            cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
            print(f"  Channel matrix      : shape={A_ch.shape}  "
                  f"rank={np.linalg.matrix_rank(A_ch)}  cond={cond:.2e}")

            print(f"\n  {'ch':>3}  {'pos':>5}  {'neg':>5}  {'total':>6}  "
                  f"{'col_norm':>9}  {'weight':>9}  {'rmse_without_ch':>15}")
            for ch in range(wav.n_channels):
                col_norm = np.linalg.norm(A_ch[:, ch])
                # RMSE when this channel is removed from the reconstruction
                recon_without = recon_after - A_ch[:, ch] * w_opt[ch]
                rmse_without = np.sqrt(np.mean((sig - recon_without) ** 2))
                print(f"  {ch:3d}  {spikes_pos[ch]:5d}  {spikes_neg[ch]:5d}  "
                      f"{spikes_pos[ch]+spikes_neg[ch]:6d}  "
                      f"{col_norm:9.4f}  {w_opt[ch]:9.4f}  {rmse_without:15.6f}")

    print(f"\n  Summary over {len(signals)} signals:")
    print(f"    Total spikes   mean={np.mean(all_total_spikes):.1f}  "
          f"std={np.std(all_total_spikes):.1f}")
    print(f"    Reals          mean={np.mean(all_reals):,.0f}")
    print(f"    RMSE (before)  mean={np.mean(all_rmse_before):.6f}  "
          f"std={np.std(all_rmse_before):.6f}")
    print(f"    RMSE (after)   mean={np.mean(all_rmse_after):.6f}  "
          f"std={np.std(all_rmse_after):.6f}")
    return float(np.mean(all_rmse_after)), float(np.mean(all_reals))

# %%
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sep("SETUP")
    tau_max_doe, c_doe = ti.compute_stable_wavelet_params(
        N_CHANNELS, DT, f_min=F_MIN, f_max=F_MAX, wavelet_type='doe'
    )
    mu_max_doe = float(np.sqrt(tau_max_doe))
    print(f"  DoE params: mu_max={mu_max_doe:.6f} s  c={c_doe:.4f}")

    wav_morlet = morlet.MorletWavelet(n_channels=N_CHANNELS, dt=DT)
    wav_sdoe = spiking_doe.SpikingDoEWavelet(
        n_channels=N_CHANNELS, dt=DT, mu_max=mu_max_doe, c=c_doe
    )

    print(f"  Loading {N_SIGNALS} MIT-BIH signals...")
    signals = load_signals(N_SIGNALS)
    print(f"  Loaded {len(signals)} signals of length {len(signals[0])}")
    sig0 = signals[0]
    print(f"  Signal 0: mean={sig0.mean():.4f}  std={sig0.std():.4f}"
          f"  min={sig0.min():.4f}  max={sig0.max():.4f}")

    morlet_rmse, morlet_reals = diagnose_morlet(wav_morlet, signals)

    sdoe_results = []
    for thr in THRESHOLDS:
        rmse, reals = diagnose_spiking_doe(wav_sdoe, signals, thr)
        sdoe_results.append((thr, rmse, reals))

    sep("COMPARISON SUMMARY")
    print(f"\n  {'Method':<38}  {'RMSE mean':>10}  {'Reals mean':>12}")
    print(f"  {'-'*38}  {'-'*10}  {'-'*12}")
    print(f"  {'Morlet (16ch)':<38}  {morlet_rmse:10.6f}  {morlet_reals:12,.0f}")
    for thr, rmse, reals in sdoe_results:
        label = f"Spiking DoE per-ch (thr={thr})"
        print(f"  {label:<38}  {rmse:10.6f}  {reals:12,.0f}")

# %%
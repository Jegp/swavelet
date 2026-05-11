# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: spiking-wavelets (3.13.6)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Threshold sweep: nRMSE vs $\theta_{\rm thr}$
#
# Tests the prediction that reconstruction error scales linearly with the spiking
# threshold (equation 15). Sweeps $\theta_{\rm thr}$ for DoE, DoT, DoG at fixed $K$.

# %%
import os
from pathlib import Path

# Pin this sweep to a specific CUDA device. Must happen BEFORE any JAX import
# so JAX only sees the selected GPU. Set to None (or "") to run on CPU.
CUDA_DEVICE = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "" if CUDA_DEVICE is None else str(CUDA_DEVICE)

import json

import numpy as np
import jax
import tqdm

from .experiment_encoding_efficiency import (
    get_dataset_config,
    load_eval_signals,
    create_all_wavelets,
    evaluate_spiking_per_spike_decoupled,
    _run_parallel,
)

OUTPUT_DIR = Path("results/threshold_sweep")
DATASETS = ["speech", "mitbih"]
# Scale ratio between adjacent channels (matches encoding-efficiency sweep).
# K is derived from c and [f_min, f_max] inside create_all_wavelets.
C = 2.0
N_THRESHOLDS = 8
THRESHOLD_MIN = 0.001
THRESHOLD_MAX = 20.0
N_EVAL_SIGNALS = 10
N_WORKERS = 8
# Per-dataset chunk size override for the threshold sweep. The curve shape is
# chunk-length-independent, so we use shorter windows to keep the low-threshold
# end tractable (per-spike lstsq cost scales ~ chunk × n_spikes²).
CHUNK_SIZE_OVERRIDE = {"speech": 2000, "mitbih": 180}

THRESHOLDS = list(np.logspace(np.log10(THRESHOLD_MIN), np.log10(THRESHOLD_MAX), N_THRESHOLDS))

# %% [markdown]
# ## Sweep

# %%
all_results = []


def _save_progress():
    """Write per-dataset and combined JSON with whatever is collected so far.
    Called after every (wavelet, weight_scheme, threshold) so an interrupted
    sweep still leaves usable partial results on disk.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in sorted(set(r["dataset"] for r in all_results)):
        ds_dir = OUTPUT_DIR / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        ds_results = [r for r in all_results if r["dataset"] == ds]
        with open(ds_dir / "threshold_sweep.json", "w") as f:
            json.dump(ds_results, f, indent=2)
    with open(OUTPUT_DIR / "threshold_sweep_all.json", "w") as f:
        json.dump(all_results, f, indent=2)


for dataset_name in DATASETS:
    print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")

    config = dict(get_dataset_config(dataset_name))
    if dataset_name in CHUNK_SIZE_OVERRIDE:
        config["chunk_size"] = CHUNK_SIZE_OVERRIDE[dataset_name]
    dt = config["dt"]
    signals = load_eval_signals(dataset_name, config, N_EVAL_SIGNALS)
    print(f"  Loaded {len(signals)} eval signals (chunk_size={config['chunk_size']})")

    dataset_dir = OUTPUT_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    wavelets, n_channels = create_all_wavelets(
        C, dt, config["f_min"], config["f_max"], config["chunk_size"]
    )

    for wname in ["spiking_doe", "spiking_dot", "spiking_dog"]:
        if wname not in wavelets:
            print(f"  Skipping {wname} (not available for c={C})")
            continue

        wav, _ = wavelets[wname]
        mu_recon = float(wav.mu_max) if hasattr(wav, "mu_max") else float(dt * 10)

        # Paper K = number of bandpass channels = n_channels - 1
        # (the remaining channel is the lowpass residual).
        print(f"\n  {wname} (c={C}, K={n_channels - 1})")

        weight_scheme = "per_spike_decoupled"
        eval_fn = evaluate_spiking_per_spike_decoupled
        print(f"    {weight_scheme}:")
        for thresh in tqdm.tqdm(THRESHOLDS, desc="      θ_thr", leave=False):
            rmse_list, reals_list, spikes_list, nrmse_list, lag_list = _run_parallel(
                lambda sig, t=thresh: eval_fn(wav, wav.params, sig, t, dt, mu_recon),
                signals, N_WORKERS,
                f"{wname} {weight_scheme} thresh={thresh}",
            )
            if rmse_list:
                all_results.append({
                    "dataset": dataset_name,
                    "wavelet": wname,
                    "weight_scheme": weight_scheme,
                    "c": float(C),
                    "n_channels": int(n_channels),
                    "threshold": float(thresh),
                    "rmse_mean": float(np.mean(rmse_list)),
                    "rmse_std": float(np.std(rmse_list)),
                    "nrmse_mean": float(np.mean(nrmse_list)),
                    "nrmse_std": float(np.std(nrmse_list)),
                    "lag_mean": float(np.mean(lag_list)),
                    "reals_mean": float(np.mean(reals_list)),
                    "reals_std": float(np.std(reals_list)),
                    "spikes_mean": float(np.mean(spikes_list)) if spikes_list else None,
                    "n_eval_signals": len(rmse_list),
                })
                _save_progress()

        jax.clear_caches()

# %% [markdown]
# ## Final save

# %%
_save_progress()
print(f"Saved {len(all_results)} results to {OUTPUT_DIR / 'threshold_sweep_all.json'}")

# %%

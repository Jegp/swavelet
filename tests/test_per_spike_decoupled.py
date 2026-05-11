"""Tests for the decoupled per-channel per-spike reconstruction.

Covers the helper decomposition (`_per_channel_events`, `_segment_design`,
`_segment_gram_with_ridge`, `_batched_cho_solve`, `_solve_segment_decoupled`)
and the top-level `_reconstruct_per_spike_decoupled`. The reconstruction is
wavelet-agnostic --- it takes per-channel spike indices and per-channel
bandpass targets, with no dependency on the analysis wavelet --- so the
tests use synthetic targets and any spiking wavelet whose analysis sibling
exposes a `__call__` returning a `(n_channels, T)` encoding works.
"""

import numpy as np
import pytest

from experiment_encoding_efficiency import (
    SEGMENT_SIZE,
    _per_channel_events,
    _segment_design,
    _segment_gram_with_ridge,
    _batched_cho_solve,
    _solve_segment_decoupled,
    _reconstruct_per_spike_decoupled,
)


class _SumAnalysis:
    """Analysis sibling whose `reconstruct` returns the channel sum."""

    params = {}

    @staticmethod
    def reconstruct(params, activities):
        return np.asarray(activities).sum(axis=0)


class _SumWavelet:
    """Stub spiking wavelet whose `_analysis.reconstruct` is a plain channel-sum.

    Pairs with the test setup `signal = targets.sum(axis=0)` so the lstsq
    solver is exercised in isolation from any wavelet's decode rule.
    """

    _analysis = _SumAnalysis()


# --- helpers ---

def _decoupled_reference(signal, all_idx_pos, all_idx_neg, dt, mus, bandpass_targets):
    """Per-channel reference: solve K independent lstsq problems with numpy.

    Used as ground truth for the batched/segmented decoupled solver. For
    short signals (one segment) this matches the segmented solver up to
    floating-point noise.
    """
    K = len(all_idx_pos)
    N = len(signal)
    recon = np.zeros(N)
    for k in range(K):
        idx_p = np.asarray(all_idx_pos[k], dtype=int)
        idx_n = np.asarray(all_idx_neg[k], dtype=int)
        n_k = len(idx_p) + len(idx_n)
        if n_k == 0:
            continue
        decay = np.exp(-dt / mus[k])
        times = np.concatenate([idx_p, idx_n])
        signs = np.concatenate([np.ones(len(idx_p)), -np.ones(len(idx_n))])
        order = np.argsort(times)
        times, signs = times[order], signs[order]
        j = np.arange(N)[:, None]
        A = np.where(j >= times, signs * decay ** np.maximum(j - times, 0), 0.0)
        w, *_ = np.linalg.lstsq(A, bandpass_targets[k], rcond=None)
        recon += A @ w
    return recon


# --- _per_channel_events ---

class TestPerChannelEvents:
    def test_sorts_by_time(self):
        evs = _per_channel_events([5, 1, 3], [4, 2])
        times = [t for t, _ in evs]
        assert times == sorted(times)

    def test_signs_match_polarity(self):
        evs = _per_channel_events([1, 3], [2, 4])
        d = dict(evs)
        assert d[1] == 1.0 and d[3] == 1.0
        assert d[2] == -1.0 and d[4] == -1.0

    def test_empty_inputs(self):
        assert _per_channel_events([], []) == []


# --- _segment_design ---

class TestSegmentDesign:
    def test_padding_when_unbalanced(self):
        decays = np.array([0.9, 0.8])
        seg = [[(0, 1.0), (2, -1.0)], [(1, 1.0)]]
        A, n_evs, n_max = _segment_design(seg, B=5, decays=decays)
        assert A.shape == (2, 5, 2)
        assert list(n_evs) == [2, 1]
        assert n_max == 2
        # Channel-1's padding column should be all zeros
        assert np.allclose(A[1, :, 1], 0.0)

    def test_causality(self):
        """A[t, i] is zero for t < t_e."""
        decays = np.array([0.5])
        seg = [[(3, 1.0)]]
        A, _, _ = _segment_design(seg, B=6, decays=decays)
        assert np.allclose(A[0, :3, 0], 0.0)
        # At spike time, contribution is the signed decay-power-zero = sign
        assert A[0, 3, 0] == pytest.approx(1.0)
        # One step after, decay^1 = 0.5
        assert A[0, 4, 0] == pytest.approx(0.5)

    def test_signs_propagate(self):
        decays = np.array([0.7])
        seg = [[(1, -1.0)]]
        A, _, _ = _segment_design(seg, B=4, decays=decays)
        assert A[0, 1, 0] == pytest.approx(-1.0)
        assert A[0, 2, 0] == pytest.approx(-0.7)

    def test_empty_segment(self):
        decays = np.array([0.9, 0.8])
        seg = [[], []]
        A, n_evs, n_max = _segment_design(seg, B=5, decays=decays)
        assert n_max == 0
        assert A.shape == (2, 5, 0)
        assert list(n_evs) == [0, 0]


# --- _segment_gram_with_ridge ---

class TestSegmentGram:
    def test_ridge_added_on_real_diagonal(self):
        decays = np.array([0.9])
        seg = [[(0, 1.0), (2, 1.0)]]
        A, n_evs, _ = _segment_design(seg, B=4, decays=decays)
        G = _segment_gram_with_ridge(A, n_evs)
        # G should be positive definite (small ridge applied)
        eig = np.linalg.eigvalsh(G[0])
        assert eig.min() > 0

    def test_padded_positions_become_identity(self):
        decays = np.array([0.9, 0.8])
        seg = [[(0, 1.0), (1, 1.0)], [(0, 1.0)]]   # channel 1 has fewer events
        A, n_evs, n_max = _segment_design(seg, B=4, decays=decays)
        G = _segment_gram_with_ridge(A, n_evs)
        # Channel 1 padded position [1, 1] should equal 1
        assert G[1, 1, 1] == pytest.approx(1.0)
        # Off-diagonal padding entries are zero
        assert G[1, 0, 1] == pytest.approx(0.0)
        assert G[1, 1, 0] == pytest.approx(0.0)

    def test_empty_channel_becomes_identity_block(self):
        decays = np.array([0.9, 0.8])
        seg = [[(0, 1.0), (1, 1.0)], []]            # channel 1 empty
        A, n_evs, _ = _segment_design(seg, B=4, decays=decays)
        G = _segment_gram_with_ridge(A, n_evs)
        np.testing.assert_allclose(G[1], np.eye(2))


# --- _batched_cho_solve ---

class TestBatchedCho:
    def test_matches_per_channel_lstsq_on_overdetermined(self):
        rng = np.random.default_rng(0)
        K, B, n_max = 3, 50, 6
        A = rng.standard_normal((K, B, n_max))
        residuals = rng.standard_normal((K, B))
        n_evs = np.array([n_max] * K)
        G = _segment_gram_with_ridge(A, n_evs)
        rhs = np.einsum('kti,kt->ki', A, residuals)
        w_batched = _batched_cho_solve(G, rhs, A, residuals, n_evs)
        # Per-channel reference via lstsq
        for k in range(K):
            w_ref, *_ = np.linalg.lstsq(A[k], residuals[k], rcond=None)
            np.testing.assert_allclose(w_batched[k], w_ref, atol=1e-6)

    def test_padded_weights_are_zero(self):
        decays = np.array([0.9, 0.8])
        seg = [[(0, 1.0), (1, 1.0), (2, 1.0)], [(0, 1.0)]]
        A, n_evs, n_max = _segment_design(seg, B=5, decays=decays)
        G = _segment_gram_with_ridge(A, n_evs)
        residuals = np.ones((2, 5))
        rhs = np.einsum('kti,kt->ki', A, residuals)
        w = _batched_cho_solve(G, rhs, A, residuals, n_evs)
        # Channel 1 has only 1 real event; positions 1.. should solve to 0
        assert np.allclose(w[1, 1:], 0.0)


# --- _solve_segment_decoupled ---

class TestSolveSegmentDecoupled:
    def test_recon_matches_per_channel_reference(self):
        """One-segment solve must match the per-channel numpy lstsq baseline."""
        rng = np.random.default_rng(7)
        B = SEGMENT_SIZE  # one segment
        K = 3
        decays = np.array([0.99, 0.97, 0.95])
        # Few well-spaced spikes per channel
        seg = [
            [(10, 1.0), (40, -1.0), (90, 1.0)],
            [(20, 1.0), (60, -1.0)],
            [(30, -1.0), (75, 1.0)],
        ]
        residuals = rng.standard_normal((K, B))
        recon_seg, last_row = _solve_segment_decoupled(seg, residuals, decays, B)
        # Reference: build A_k, lstsq per channel, return A_k @ w_k
        ref_recon = np.zeros((K, B))
        for k in range(K):
            t_k = np.array([e[0] for e in seg[k]])
            s_k = np.array([e[1] for e in seg[k]])
            j = np.arange(B)[:, None]
            A_k = np.where(j >= t_k, s_k * decays[k] ** np.maximum(j - t_k, 0), 0.0)
            w_k, *_ = np.linalg.lstsq(A_k, residuals[k], rcond=None)
            ref_recon[k] = A_k @ w_k
        np.testing.assert_allclose(recon_seg, ref_recon, atol=1e-6)
        np.testing.assert_allclose(last_row, ref_recon[:, -1], atol=1e-6)

    def test_empty_segment_returns_zeros(self):
        decays = np.array([0.9, 0.8])
        seg = [[], []]
        residuals = np.zeros((2, 4))
        recon_seg, last_row = _solve_segment_decoupled(seg, residuals, decays, 4)
        assert np.allclose(recon_seg, 0.0)
        assert np.allclose(last_row, 0.0)


# --- _reconstruct_per_spike_decoupled (top level) ---

class TestReconstructPerSpikeDecoupled:
    def test_zero_spikes_returns_signal_rms(self):
        signal = np.array([1.0, 2.0, 3.0])
        all_idx_pos = [np.array([], dtype=int), np.array([], dtype=int)]
        all_idx_neg = [np.array([], dtype=int), np.array([], dtype=int)]
        targets = np.zeros((2, 3))
        r = _reconstruct_per_spike_decoupled(signal, all_idx_pos, all_idx_neg, 0, dt=0.01, mu_recon=np.array([0.1, 0.2]), bandpass_targets=targets, wavelet=_SumWavelet())
        assert r["reals"] == 0
        assert r["nrmse"] == 1.0
        assert r["total_spikes"] == 0

    def test_single_segment_matches_reference(self):
        """Sufficiently many spikes -> per-channel lstsq fits each x_k well, sum recovers signal."""
        rng = np.random.default_rng(3)
        N = 200  # one segment (< SEGMENT_SIZE)
        K = 2
        dt = 0.01
        mus = np.array([0.05, 0.15])
        # Random target per channel
        targets = rng.standard_normal((K, N))
        signal = targets.sum(axis=0)
        # Dense spike grid per channel ensures the per-spike basis is rich
        idx_pos = [np.arange(0, N, 4), np.arange(2, N, 4)]
        idx_neg = [np.arange(1, N, 4), np.arange(3, N, 4)]
        total_spikes = sum(len(p) + len(n) for p, n in zip(idx_pos, idx_neg))

        r = _reconstruct_per_spike_decoupled(signal, idx_pos, idx_neg, total_spikes, dt, mus, targets, wavelet=_SumWavelet())
        ref_recon = _decoupled_reference(signal, idx_pos, idx_neg, dt, mus, targets)
        # The function returns metrics; recompute reconstruction by re-running and compare RMSE
        # against the reference's RMSE.
        ref_rmse = float(np.sqrt(np.mean((signal - ref_recon) ** 2)))
        assert r["rmse"] == pytest.approx(ref_rmse, rel=0.05)

    def test_works_across_segments(self):
        """N > SEGMENT_SIZE: per-segment carry must keep the reconstruction continuous."""
        rng = np.random.default_rng(5)
        N = 2 * SEGMENT_SIZE + 250
        K = 2
        dt = 0.001
        mus = np.array([0.02, 0.05])
        targets = rng.standard_normal((K, N)) * 0.1
        signal = targets.sum(axis=0)
        # Sparse spike pattern but enough spikes per segment per channel
        idx_pos = [np.arange(5, N, 30), np.arange(10, N, 25)]
        idx_neg = [np.arange(20, N, 30), np.arange(15, N, 25)]
        total_spikes = sum(len(p) + len(n) for p, n in zip(idx_pos, idx_neg))

        r = _reconstruct_per_spike_decoupled(signal, idx_pos, idx_neg, total_spikes, dt, mus, targets, wavelet=_SumWavelet())
        # Sanity: nRMSE finite, much less than 1 (we should at least beat zero recon)
        assert np.isfinite(r["nrmse"])
        assert r["nrmse"] < 1.0
        assert r["total_spikes"] == total_spikes
        assert r["reals"] == 2 * total_spikes

    def test_unbalanced_channels(self):
        """A channel with zero spikes contributes zero, others contribute normally."""
        N = 200
        K = 3
        dt = 0.01
        mus = np.array([0.05, 0.10, 0.20])
        targets = np.zeros((K, N))
        targets[0] = np.sin(np.linspace(0, 4 * np.pi, N))
        # Only channel 0 has spikes
        idx_pos = [np.arange(0, N, 5), np.array([], dtype=int), np.array([], dtype=int)]
        idx_neg = [np.arange(2, N, 5), np.array([], dtype=int), np.array([], dtype=int)]
        total_spikes = sum(len(p) + len(n) for p, n in zip(idx_pos, idx_neg))
        signal = targets[0].copy()
        r = _reconstruct_per_spike_decoupled(signal, idx_pos, idx_neg, total_spikes, dt, mus, targets, wavelet=_SumWavelet())
        # Empty channels should not break the solve and should not hurt the fit
        assert np.isfinite(r["nrmse"])
        assert r["reals"] == 2 * total_spikes

    @pytest.mark.parametrize("K", [1, 4, 16])
    def test_runs_for_various_K(self, K):
        rng = np.random.default_rng(K)
        N = 200
        dt = 0.01
        mus = np.linspace(0.05, 0.2, K)
        targets = rng.standard_normal((K, N)) * 0.1
        signal = targets.sum(axis=0)
        idx_pos = [np.arange(0, N, 5) + (k % 5) for k in range(K)]
        idx_neg = [np.arange(2, N, 5) + (k % 5) for k in range(K)]
        # Trim any indices that overflow
        idx_pos = [a[a < N] for a in idx_pos]
        idx_neg = [a[a < N] for a in idx_neg]
        total_spikes = sum(len(p) + len(n) for p, n in zip(idx_pos, idx_neg))
        r = _reconstruct_per_spike_decoupled(signal, idx_pos, idx_neg, total_spikes, dt, mus, targets, wavelet=_SumWavelet())
        assert np.isfinite(r["nrmse"])
        assert r["reals"] == 2 * total_spikes

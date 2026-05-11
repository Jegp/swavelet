"""Tests for `to_nir` on the DoT/DoE family (spiking and non-spiking)."""

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from swavelet import DoE, DoT, SpikingDoE, SpikingDoT


# --- Shared validation ---

class TestValidation:
    def test_rejects_cascade_depth_at_or_above_10(self):
        """to_nir refuses cascades whose stages exceed the alpha-stability floor."""
        w = DoT(n_channels=3, dt=0.01, mu_max=1.0, cascade_depth_max=9)
        # Bypass __init__ caps post-hoc to trip the export check.
        w.cascade_depths = [10, 10]
        w.cascade_mus = [
            list(w.cascade_mus[0]) + [w.cascade_mus[0][-1]],
            list(w.cascade_mus[1]) + [w.cascade_mus[1][-1]],
        ]
        with pytest.raises(ValueError, match="depth >= 10"):
            w.to_nir()

    def test_rejects_unequal_cascade_depths(self):
        """A single multi-channel LI bank can't pack unequal depths."""
        w = DoT(n_channels=3, dt=0.01, mu_max=1.0, cascade_depth_max=3)
        w.cascade_depths = [3, 2]
        w.cascade_mus = [w.cascade_mus[0], w.cascade_mus[1][:2]]
        with pytest.raises(ValueError, match="equal cascade depths"):
            w.to_nir()


# --- Non-spiking ---

class TestDoTNirExport:
    def _wav(self, **kwargs):
        return DoT(n_channels=4, dt=0.01, mu_max=1.0, cascade_depth_max=3, **kwargs)

    def test_returns_nirgraph(self):
        assert isinstance(self._wav().to_nir(), nir.NIRGraph)

    def test_single_input_and_output(self):
        g = self._wav().to_nir()
        inputs = [n for n, node in g.nodes.items() if isinstance(node, nir.Input)]
        outputs = [n for n, node in g.nodes.items() if isinstance(node, nir.Output)]
        assert len(inputs) == 1
        assert len(outputs) == 1

    def test_output_is_n_channels_wide(self):
        w = self._wav()
        g = w.to_nir()
        assert g.nodes["output"].output_type["output"].tolist() == [w.n_channels]

    def test_linear_chain(self):
        """Input -> fanout -> LI stages -> connectivity -> Output."""
        w = self._wav()
        depth = w.cascade_depths[0]
        g = w.to_nir()
        for name in ("input", "fanout", "connectivity", "output"):
            assert name in g.nodes
        for stage in range(depth):
            assert f"li_stage_{stage}" in g.nodes

    def test_one_li_bank_per_stage(self):
        w = self._wav()
        depth = w.cascade_depths[0]
        g = w.to_nir()
        li_nodes = [n for n, node in g.nodes.items() if isinstance(node, nir.LI)]
        assert len(li_nodes) == depth

    def test_li_bank_taus_match_per_stage(self):
        """Stage s of the LI bank carries tau = [cascade_mus[k][s] for k]."""
        w = self._wav()
        K = w.n_channels - 1
        g = w.to_nir()
        for stage in range(w.cascade_depths[0]):
            node = g.nodes[f"li_stage_{stage}"]
            expected = np.array([float(w.cascade_mus[k][stage]) for k in range(K)])
            np.testing.assert_allclose(node.tau, expected, rtol=1e-5)

    def test_connectivity_matrix_structure(self):
        """Leading row = e_1; middle rows = forward differences; last row = e_K."""
        w = self._wav()
        K = w.n_channels - 1
        g = w.to_nir()
        W = g.nodes["connectivity"].weight
        assert W.shape == (w.n_channels, K)
        # row 0: just L_1
        np.testing.assert_array_equal(W[0], np.eye(1, K, k=0).reshape(K))
        # rows 1..K-1: forward differences
        for k in range(1, K):
            expected = np.zeros(K); expected[k - 1] = -1.0; expected[k] = 1.0
            np.testing.assert_array_equal(W[k], expected)
        # row K: lowpass = L_K
        expected_lp = np.zeros(K); expected_lp[K - 1] = 1.0
        np.testing.assert_array_equal(W[K], expected_lp)


class TestDoENirExport:
    def test_returns_nirgraph(self):
        w = DoE(n_channels=3, dt=0.01, mu_max=0.1)
        assert isinstance(w.to_nir(), nir.NIRGraph)

    def test_one_li_bank_only(self):
        """DoE has cascade depth 1 -> a single multi-channel LI stage."""
        w = DoE(n_channels=4, dt=0.01, mu_max=0.1)
        g = w.to_nir()
        li_nodes = [n for n, node in g.nodes.items() if isinstance(node, nir.LI)]
        assert len(li_nodes) == 1

    def test_li_bank_tau_matches_smoothing_scales(self):
        w = DoE(n_channels=3, dt=0.01, mu_max=0.1)
        K = w.n_channels - 1
        g = w.to_nir()
        node = g.nodes["li_stage_0"]
        expected = np.array([float(w.cascade_mus[k][0]) for k in range(K)])
        np.testing.assert_allclose(node.tau, expected, rtol=1e-6)


# --- Spiking ---

class TestSpikingDoTNirExport:
    def _wav(self, **kwargs):
        return SpikingDoT(
            n_channels=4, dt=0.001, mu_max=1.0, cascade_depth_max=3, **kwargs
        )

    def test_returns_nirgraph(self):
        assert isinstance(self._wav().to_nir(), nir.NIRGraph)

    def test_linear_chain(self):
        """Input -> fanout -> LI stages -> connectivity -> LIF -> Output."""
        w = self._wav()
        g = w.to_nir()
        for name in ("input", "fanout", "connectivity", "lif", "output"):
            assert name in g.nodes
        for stage in range(w.cascade_depths[0]):
            assert f"li_stage_{stage}" in g.nodes

    def test_single_output_n_channels_wide(self):
        w = self._wav()
        g = w.to_nir()
        outputs = [n for n, node in g.nodes.items() if isinstance(node, nir.Output)]
        assert len(outputs) == 1
        assert g.nodes["output"].output_type["output"].tolist() == [w.n_channels]

    def test_one_multi_channel_lif(self):
        w = self._wav()
        g = w.to_nir()
        lifs = [(n, node) for n, node in g.nodes.items() if isinstance(node, nir.LIF)]
        assert len(lifs) == 1
        _, lif = lifs[0]
        assert lif.tau.shape == (w.n_channels,)

    def test_lif_tau_matches_log_mu_mem(self):
        import jax.numpy as jnp
        w = self._wav()
        mu_mem = np.asarray(jnp.exp(w.params["log_mu_mem"]))
        g = w.to_nir()
        np.testing.assert_allclose(g.nodes["lif"].tau, mu_mem, rtol=1e-6)

    def test_lif_thresholds_match_log_threshold(self):
        import jax.numpy as jnp
        w = self._wav()
        thr = np.asarray(jnp.exp(w.params["log_threshold"]))
        g = w.to_nir()
        np.testing.assert_allclose(g.nodes["lif"].v_threshold, thr, rtol=1e-6)

    def test_connectivity_folds_prescale(self):
        """Each connectivity row's magnitude equals the channel's prescale."""
        w = self._wav()
        K = w.n_channels - 1
        prescale = 1.0 / (np.asarray(w._filter_norms) + 1e-10)
        g = w.to_nir()
        W = g.nodes["connectivity"].weight
        # row 0: leading bandpass = prescale[0] on L_1
        np.testing.assert_allclose(W[0, 0], prescale[0], rtol=1e-6)
        # rows 1..K-1: forward differences scaled by prescale[k]
        for k in range(1, K):
            np.testing.assert_allclose(W[k, k - 1], -prescale[k], rtol=1e-6)
            np.testing.assert_allclose(W[k, k], prescale[k], rtol=1e-6)
        # row K: lowpass = prescale[K] on L_K
        np.testing.assert_allclose(W[K, K - 1], prescale[K], rtol=1e-6)

    def test_no_cubalif(self):
        """The minimal encoding uses plain LIF, not CubaLIF."""
        w = self._wav()
        g = w.to_nir()
        cubas = [n for n, node in g.nodes.items() if isinstance(node, nir.CubaLIF)]
        assert cubas == []


class TestSpikingDoENirExport:
    def test_returns_nirgraph(self):
        w = SpikingDoE(n_channels=4, dt=0.001, mu_max=0.05)
        assert isinstance(w.to_nir(), nir.NIRGraph)

    def test_one_li_bank_only(self):
        w = SpikingDoE(n_channels=4, dt=0.001, mu_max=0.05)
        g = w.to_nir()
        li_nodes = [n for n, node in g.nodes.items() if isinstance(node, nir.LI)]
        assert len(li_nodes) == 1

    def test_six_node_skeleton(self):
        """Input + fanout + 1 LI + connectivity + LIF + Output = 6 nodes."""
        w = SpikingDoE(n_channels=4, dt=0.001, mu_max=0.05)
        g = w.to_nir()
        assert len(g.nodes) == 6

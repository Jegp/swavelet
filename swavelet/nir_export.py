"""NIR export for the DoT family wavelets (and their spiking variants).

Both exports produce a single linear chain ending at one `nir.Output`:

- Non-spiking: ``Input -> fanout Affine -> LI bank -> connectivity Affine
  -> Output``. The connectivity Affine maps the LI bank's K-vec into the
  n_channels output: bandpass rows have ±1 pairs (`L_k − L_{k-1}`); the
  lowpass row is a single +1 on the coarsest scale.

- Spiking: same skeleton with a multi-channel `LIF` between the
  connectivity Affine and the Output. Each channel's per-channel
  prescale factor is folded into the connectivity row's magnitude.

Both require equal cascade depths across channels.
"""
from __future__ import annotations

import numpy as np

_MAX_CASCADE_DEPTH = 10


def _validate(wavelet):
    max_depth = max(wavelet.cascade_depths)
    if max_depth >= _MAX_CASCADE_DEPTH:
        raise ValueError(
            f"NIR export rejects cascades of depth >= {_MAX_CASCADE_DEPTH} "
            f"(deeper cascades hit the alpha-stability floor and produce LI "
            f"taus that aren't numerically representable on most neuromorphic "
            f"targets); got cascade_depths={list(wavelet.cascade_depths)}"
        )
    if len(set(wavelet.cascade_depths)) > 1:
        raise ValueError(
            f"NIR export requires equal cascade depths across channels (one "
            f"multi-channel LI bank per stage); got "
            f"cascade_depths={list(wavelet.cascade_depths)}"
        )


def _connectivity_matrix(n_channels, prescale=None):
    """Build the (n_channels, K) row-stochastic connectivity matrix.

    Row 0 is the leading bandpass (just `L_1`); rows 1..K-1 are forward
    differences; row K is the lowpass tap. When `prescale` is given (one
    factor per output channel) it's folded into the row magnitudes.
    """
    K = n_channels - 1
    if prescale is None:
        prescale = np.ones(n_channels)
    conn = np.zeros((n_channels, K))
    conn[0, 0] = float(prescale[0])
    for k in range(1, K):
        conn[k, k - 1] = -float(prescale[k])
        conn[k, k] = float(prescale[k])
    conn[K, K - 1] = float(prescale[K])
    return conn


def _li_bank_nodes(wavelet, nir, source_name, cascade_mus):
    """Append fanout + multi-channel LI bank stages onto `nodes`/`edges`,
    returning the terminal node name and the populated dicts."""
    K = wavelet.n_channels - 1
    depth = wavelet.cascade_depths[0]
    nodes: dict = {}
    edges: list = []

    nodes["fanout"] = nir.Affine(weight=np.ones((K, 1)), bias=np.zeros(K))
    edges.append((source_name, "fanout"))

    prev = "fanout"
    for stage in range(depth):
        name = f"li_stage_{stage}"
        stage_taus = np.array([float(cascade_mus[k][stage]) for k in range(K)])
        nodes[name] = nir.LI(
            tau=stage_taus, r=np.ones(K), v_leak=np.zeros(K),
        )
        edges.append((prev, name))
        prev = name
    return prev, nodes, edges


# ---------------------------------------------------------------------------
# Non-spiking DoT (and DoE via inheritance)
# ---------------------------------------------------------------------------

def from_dot(wavelet):
    """Build a `nir.NIRGraph` for a non-spiking DoT/DoE wavelet."""
    import nir
    _validate(wavelet)

    nodes: dict = {}
    edges: list = []
    nodes["input"] = nir.Input(input_type={"input": np.array([1])})

    li_terminal, li_nodes, li_edges = _li_bank_nodes(
        wavelet, nir, "input", wavelet.cascade_mus,
    )
    nodes.update(li_nodes)
    edges.extend(li_edges)

    conn = _connectivity_matrix(wavelet.n_channels)
    nodes["connectivity"] = nir.Affine(
        weight=conn, bias=np.zeros(wavelet.n_channels),
    )
    edges.append((li_terminal, "connectivity"))

    nodes["output"] = nir.Output(
        output_type={"output": np.array([wavelet.n_channels])},
    )
    edges.append(("connectivity", "output"))

    return nir.NIRGraph(nodes=nodes, edges=edges, type_check=False)


# ---------------------------------------------------------------------------
# Spiking DoT (and SpikingDoE via inheritance)
# ---------------------------------------------------------------------------

def from_spiking_dot(wavelet):
    """Build a `nir.NIRGraph` for a spiking DoT/DoE wavelet."""
    import nir
    import jax.numpy as jnp
    _validate(wavelet)

    thresholds = np.asarray(jnp.exp(wavelet.params["log_threshold"]))
    mu_mem = np.asarray(jnp.exp(wavelet.params["log_mu_mem"]))
    if wavelet.enable_normalization and wavelet._filter_norms is not None:
        prescale = 1.0 / (np.asarray(wavelet._filter_norms) + 1e-10)
    else:
        prescale = np.ones(wavelet.n_channels)

    nodes: dict = {}
    edges: list = []
    nodes["input"] = nir.Input(input_type={"input": np.array([1])})

    li_terminal, li_nodes, li_edges = _li_bank_nodes(
        wavelet, nir, "input", wavelet._analysis.cascade_mus,
    )
    nodes.update(li_nodes)
    edges.extend(li_edges)

    conn = _connectivity_matrix(wavelet.n_channels, prescale=prescale)
    nodes["connectivity"] = nir.Affine(
        weight=conn, bias=np.zeros(wavelet.n_channels),
    )
    edges.append((li_terminal, "connectivity"))

    nodes["lif"] = nir.LIF(
        tau=mu_mem.astype(float),
        r=np.ones(wavelet.n_channels),
        v_leak=np.zeros(wavelet.n_channels),
        v_threshold=thresholds.astype(float),
        v_reset=np.zeros(wavelet.n_channels),
    )
    edges.append(("connectivity", "lif"))

    nodes["output"] = nir.Output(
        output_type={"output": np.array([wavelet.n_channels])},
    )
    edges.append(("lif", "output"))

    return nir.NIRGraph(nodes=nodes, edges=edges, type_check=False)

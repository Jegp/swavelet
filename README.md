# Scale covariant and spiking wavelets

[Wavelets](https://en.wikipedia.org/wiki/Wavelet) that are guaranteed to be scale covariant, implemented as standard transforms and with [spiking neurons](https://snnbook.net).


## Why

Standard wavelets are excellent at robust signal processing but work offline: they need the full signal.
Real-time streams (audio, biosignals, sensors) need *causal* wavelets, and signals at unpredictable rates benefit from *scale covariance*: the response transforms predictably under time rescaling.

`swavelet` contains time-causal wavelets that can be used directly as neuromorphic signal encoders, deployable directly to neuromorphic hardware via [NIR][nir], both as spiking and non-spiking variants.

The spiking variants double as event-driven ADCs, encoding continuous inputs straight into sparse spike trains, skipping the uniform sampler entirely.

![Wavelet spike coding diagram](https://github.com/Jegp/swavelet/blob/main/diagram.png?raw=true)


## Usage

**Install** via [uv](https://docs.astral.sh/uv/): `uv add swavelet` or via [pip](https://pypi.org/project/pip/): `pip install swavelet`

### Encoding and decoding (non-spiking)
```python
import jax.numpy as jnp
from swavelet import DoT

dt = 1e-3                        # Simulation time delta
t = jnp.arange(0, 1.0, dt)       # Duration: 1 second
signal = jnp.sin(2 * jnp.pi * t) # A simple sinusoidal

w = DoT(n_channels=4, dt=dt, mu_max=0.05) # 4-channel wavelet
coeffs = w(w.params, signal)              # analysis: (n_channels, T)
recon = w.reconstruct(w.params, coeffs)   # synthesis: back to signal
```

### Encoding and decoding (spiking)
The same shape works for every wavelet — call returns coefficients, `reconstruct` inverts them. Spiking variants substitute spike trains for coefficients:

```python
from swavelet import SpikingDoT

w = SpikingDoT(n_channels=4, dt=dt, mu_max=0.05)
spikes = w(w.params, signal) # (2 * n_channels, T)
# Use the spikes in your ML pipeline or reconstruct:
recon  = w.reconstruct(w.params, spikes)
```

### Export to NIR
```python
import nir
from swavelet import SpikingDoE

w = SpikingDoE(n_channels=4, dt=dt, mu_max=0.05)
nir_graph = w.to_nir()
nir.write("spiking_doe.nir", nir_graph)
```

The graph is a single chain: a fanout `Affine` broadcasts the scalar input across the K smoothing channels, the multi-channel `LI` bank produces `L_1..L_K`, the connectivity `Affine` wires each bandpass row with ±1 weights between adjacent scales (and the lowpass row taps `L_K` directly), and a multi-channel `LIF` emits the spike output. Diagram via [NirViz](https://github.com/open-neuromorphic/nirviz):

![Spiking DoE NIR graph](https://github.com/Jegp/swavelet/blob/main/spiking_doe_nir.png?raw=true)

### Wavelet implementations (spiking and non-spiking)

- DoG: Difference of Gaussian
- DoT: Difference of time-causal limit kernel
- DoE: Difference of truncated exponential (DoT with cascade depth=1)

| Wavelet     | Causal | Recovery            | [NIR][nir] | Use case                                |
| ----------- | :----: | ------------------- | :--------: | --------------------------------------- |
| DoG         |   ✗    | Perfect             |     ✗      | Offline, symmetric scale-space          |
| DoT         |   ✓    | Perfect             |     ✓      | Streaming, full causal scale-space      |
| DoE         |   ✓    | Perfect             |     ✓      | Streaming, cheapest (cascade depth 1)   |
| Spiking DoG |   ✗    | Quantized           |     ✗      | Offline neuromorphic encoding           |
| Spiking DoT |   ✓    | Quantized + delayed |     ✓      | SNN streaming, biologically plausible   |
| Spiking DoE |   ✓    | Quantized + delayed |     ✓      | Lightest SNN variant for streaming      |

[nir]: https://neuroir.org/

## Acknowledgements

Please cite [our paper](https://ieeexplore.ieee.org/document/11463688):
```bibtex
@inproceedings{pedersen2026scalecovariant,
  title = {Scale-{{Covariant Spiking Wavelets}}},
  booktitle = {2026 {{IEEE International Conference}} on {{Acoustics}}, {{Speech}} and {{Signal Processing}} ({{ICASSP}})},
  author = {Pedersen, Jens Egholm and Lindeberg, Tony and Gerstoft, Peter},
  year = 2026,
  month = may,
  pages = {20347--20351},
  issn = {2379-190X},
  doi = {10.1109/ICASSP55912.2026.11463688},
  urldate = {2026-05-10},
}
```

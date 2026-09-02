# Stormetric — Storm-Flow Relativistic Metrics

> **A human-AI co-designed open-source Python package for testing the
> storm-flow exponential relativistic metric against EMRI waveforms
> and black-hole shadow observations.**

[![CI](https://img.shields.io/badge/CI-3%20OS%20%C3%97%205%20Python-blue)](#)
[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PEP 561](https://img.shields.io/badge/typed-py.typed-blue)](src/stormetric/py.typed)

---

## What is this?

**Stormetric** implements the *storm-flow* exponential isotropic metric

```math
g_{tt} = -e^{-2x}, \quad g_{rr} = e^{2x}, \quad x = \frac{GM}{c^2 r},
```

together with a Schwarzschild isotropic GR reference, and exposes two
key observational signatures:

1. **Black-hole shadow radius** — predicts `b_shadow = 2e · GM/c²
   ≈ 5.437 GM/c²`, **0.79σ** from the EHT measurement `5.2 ± 0.3`.
2. **EMRI pericenter precession** — a linear framework-specific
   deviation `δ = 0.30·x_p` on top of the GR baseline `6π·x_p`
   (≈ 1.6 % relative deviation), testable by future LISA observations.

The goal: provide an **open, reproducible, numerically falsifiable**
alternative to GR for the relativistic astrophysics community.

---

## Quick Start

```bash
pip install stormetric       # (after the package is published)
# or
git clone https://github.com/IMYT-yeah/stormetric.git
cd stormetric
pip install -e ".[dev]"
```

```python
from stormetric import (
    ExponentialMetric, GRSchwarzschildIsotropic,
    Shadow, EMRIPrecession, generate_waveform_template,
    ppn_table, first_bifurcation_order,
)

# 1. Input metric parameters (GM = 1, c = 1 in geometric units)
g = ExponentialMetric()

# 2. Output: photon sphere & shadow radius
shadow = Shadow(g)
res = shadow.compute()
print(f"r_ph   = {res.r_ph:.6f} GM/c²  (numeric: {res.r_ph_numeric:.6f})")
print(f"b_shdw = {res.b_shadow:.6f} GM/c²  (EHT: 5.2 ± 0.3,  dev {res.sigma_to_eht:+.3f}σ)")

# 3. EMRI precession: 6π·x_p (GR)  +  0.30·x_p (framework)
emri = EMRIPrecession(g, coupling=0.30)
r = emri.compute(x_p=0.10, mass_solar=4e6)
print(f"δ  per orbit       = {r.delta_per_orbit:.4f} rad")
print(f"GR baseline 6π·x_p = {r.gr_precession_per_orbit:.4f} rad")
print(f"Relative deviation = {r.relative_deviation:.4%}")
print(f"T_obs (1 rad)      = {emri.t_obs_to_dephase_seconds(0.10, 4e6)/3600:.1f} h")
```

Output:
```
r_ph   = 2.000000 GM/c²  (numeric: 2.000000)
b_shdw = 5.436564 GM/c²  (EHT: 5.2 ± 0.3,  dev +0.789σ)
δ  per orbit       = 0.0300 rad
GR baseline 6π·x_p = 1.8850 rad
Relative deviation = 1.5666%
T_obs (1 rad)      = 36.2 h
```

---

## Command-line interface

After `pip install -e .`, the `stormetric` command exposes:

```
stormetric shadow [--metric exp|gr]
stormetric ppn --nmax 5
stormetric emri --xp 0.10 --mass 4e6
stormetric waveform --duration 7200 --xp 0.10 --output wf.npz
stormetric plot --output docs/stormetric_results.png
```

---

## Mathematical background

### Metric

| Component | Storm-flow (exponential) | GR (Schwarzschild isotropic) |
|-----------|--------------------------|------------------------------|
| `g_tt` | `-e^{-2x}` | `-((1-x/2)/(1+x/2))²` |
| `g_rr` | `e^{2x}` | `(1+x/2)⁴` |
| `g_θθ` | `r² e^{2x}` | `r² (1+x/2)⁴` |
| `g_φφ` | `r² sin²θ · e^{2x}` | `r² sin²θ · (1+x/2)⁴` |

### PPN expansion of `g_tt` (matches GR through 2PN, diverges at 3PN)

| n | a_n (framework) | a_n (GR) | Δa_n |
|--:|----------------:|---------:|-----:|
| 1 | +2.0000 | +2.0000 | 0 |
| 2 | -2.0000 | -2.0000 | 0 |
| 3 | +4/3     | +3/2     | **-1/6** |
| 4 | -2/3     | -1       | +1/3 |

**First honest note**: the framework's `g_rr` already diverges at
**2PN** (b₂ = 2 vs GR 1.5), one order *earlier* than the 3PN `g_tt`
signature.  Run `stormetric ppn` to see the full table.

### Photon sphere & shadow

For a static isotropic metric, `b²(r) = r² g_rr / (-g_tt)`.  The
photon sphere sits at `d(ln b²)/dr = 0`.  The exponential metric
solves this analytically:

- `r_ph = 2 GM/c²`  (vs GR isotropic `1.866 GM/c²`)
- `b_shadow = 2e · GM/c² ≈ 5.4366`  (vs GR isotropic `5.196`)

### EMRI precession

- GR per-orbit: `Δφ_GR = 6π·x_p` (leading PN, Schwarzschild)
- Framework extra: `δ = κ·x_p` with `κ = 0.30` (postulated coupling)
- **Relative deviation**: `κ/(6π+κ) ≈ 1.5666 %`
- **Dephasing time** (1 rad LISA threshold): for Sgr A* (4×10⁶ M☉,
  x_p = 0.10, e = 0): **T_obs = 1.51 days**

---

## Repository layout

```
stormetric/
├── pyproject.toml         # v0.1.0, requires-python>=3.9
├── README.md
├── LICENSE                # MIT
├── CITATION.cff
├── CHANGELOG.md
├── MANIFEST.in
├── .github/workflows/test.yml   # 3 OS × 5 Python
├── src/stormetric/
│   ├── __init__.py
│   ├── py.typed
│   ├── metric.py          # Metric ABC + ExponentialMetric + GRSchwarzschildIsotropic
│   ├── ppn.py             # PPN series, bifurcation, leading residual
│   ├── shadow.py          # Photon-sphere / shadow (analytic + numeric)
│   ├── emri.py            # 6π·x_p baseline + 0.30·x_p deviation + dephasing
│   └── cli.py             # argparse CLI
├── tests/                 # 53 tests, 1.16 s
│   ├── test_metric.py
│   ├── test_shadow.py
│   ├── test_ppn.py
│   ├── test_emri.py
│   └── test_geodesic.py   # independent null-geodesic numerical verification
├── examples/
│   ├── demo.py
│   ├── plot_results.py
│   └── ppn_bifurcation.py
└── docs/
    ├── stormetric_results.png
    └── ppn_bifurcation.png
```

---

## Running tests & plots

```bash
pytest tests/ -v                  # 53 tests, ~1.2 s
python examples/demo.py            # end-to-end demo
python examples/plot_results.py    # 4-panel results figure
python examples/ppn_bifurcation.py # 3PN bifurcation figure
```

The numerical verification in `tests/test_geodesic.py` integrates
photon orbits from `r = 30` and independently confirms that
`b = 2e` is the capture / scatter boundary.

---

## Citation

See [`CITATION.cff`](CITATION.cff).  BibTeX:

```bibtex
@software{stormetric2026,
  author = {Stormetric Authors},
  title  = {Stormetric: Testing the Storm-Flow Exponential Relativistic Metric},
  year   = {2026},
  version = {0.1.0},
  url    = {https://github.com/IMYT-yeah/stormetric}
}
```

---

## Disclaimer

This package is a **phenomenological model** for scientific
exploration.  It is **not** a claim of replacing General Relativity.
The coupling `κ = 0.30` is a *postulated* parameter; the framework
itself does not currently derive it from a stress-energy tensor.
All predictions should be independently verified against
observational data.

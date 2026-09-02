# Changelog

All notable changes to Stormetric are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and the project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-09-02

### Added
- `GRSchwarzschildIsotropic` metric class for direct GR reference.
- `ppn` module: PPN coefficient table, `truncated_g00`, `truncated_grr`,
  `first_bifurcation_order`, `leading_residual_coefficient`.
- Generic `Shadow.photon_sphere_radius_numeric()` (brentq root) with
  analytic override for the exponential metric.
- `EMRIPrecession` GR-baseline (6π·x_p), relative deviation
  κ/(6π+κ) ≈ 1.6%, `t_obs_to_dephase_seconds`, full `EMRIResult`.
- `generate_waveform_template` now uses a linear-chirp carrier
  and records `phase_total`, `phase_gr`, `delta_phi`.
- CLI entry point: `stormetric shadow | ppn | emri | waveform | plot`.
- `py.typed` marker (PEP 561).
- `CITATION.cff` and this `CHANGELOG.md`.
- New tests: `tests/test_metric.py`, `tests/test_shadow.py`,
  `tests/test_ppn.py`, `tests/test_emri.py`, `tests/test_geodesic.py`.
- New example: `examples/ppn_bifurcation.py` (3PN bifurcation plot).
- Scientific honesty note: framework *g_rr* diverges from GR at 2PN
  (framework 2 vs GR 1.5) — *earlier* than the g_tt 3PN signature.

### Fixed
- `metric: Metric | None` annotation was Python 3.10+ only; now uses
  `Optional[Metric]` and `from __future__ import annotations` so the
  package works on the declared `python_requires = ">=3.9"`.
- `g_phi_phi` default `theta=0.0` (≡ sin θ = 0) replaced with
  `theta=π/2` (equatorial plane), matching the shadow code path.
- `Shadow._b_squared` no longer hard-codes the exponential form;
  it now reads `g_tt`/`g_rr` from the supplied `Metric` instance.

## [0.1.0] — initial release

- Exponential-metric shadow radius (analytic 2e).
- PPN 1PN/2PN match, 3PN 4/3 signature.
- Linear precession law δ = 0.30·x_p.

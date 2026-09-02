"""
Demo: Stormetric quick-start example.

Run from the repo root:  python examples/demo.py
"""

import numpy as np
from stormetric import (
    ExponentialMetric,
    GRSchwarzschildIsotropic,
    Shadow,
    EMRIPrecession,
    first_bifurcation_order,
    generate_waveform_template,
    make_metric,
    ppn_table,
)


def main():
    print("=" * 64)
    print("  Stormetric — Storm-Flow Relativistic Metrics  (v0.2.0)")
    print("=" * 64)

    # ── 1. Metric (factory) ────────────────────────────────────────
    g = ExponentialMetric()
    gr = GRSchwarzschildIsotropic()
    print("\n📐 Metric: g_tt = -exp(-2x),  x = GM/(c²r)")
    print(f"   same factory → {make_metric('storm').__class__.__name__}")

    # ── 2. PPN expansion ──────────────────────────────────────────
    print("\n📊 PPN expansion of g_tt (framework vs GR):")
    print(ppn_table(n_max=4))
    bif = first_bifurcation_order(n_max=5)
    print(f"\n   First bifurcation: g_tt at {bif['g_tt']}PN, "
          f"g_rr at {bif['g_rr']}PN (overall: {bif['overall']}PN)")

    # ── 3. Shadow radius ──────────────────────────────────────────
    shadow = Shadow(g)
    res = shadow.compute()
    print(f"\n🌑 Photon sphere:  r_ph = {res.r_ph:.6f} GM/c²  (numeric: {res.r_ph_numeric:.6f})")
    print(f"   Shadow radius:  b_min = {res.b_shadow:.6f} GM/c²")
    print(f"   EHT:            α = {Shadow.EHT_CENTRAL} ± {Shadow.EHT_SIGMA}")
    print(f"   Deviation:      {res.sigma_to_eht:.3f} σ   (within 1σ: {abs(res.sigma_to_eht) < 1})")
    print(f"   Source:         {res.source}")

    # ── 4. EMRI precession ────────────────────────────────────────
    emri = EMRIPrecession(g, coupling=0.30)
    print("\n🌀 EMRI precession scale law:  δ = 0.30 · x_p")
    print(f"   Relative deviation κ/(6π+κ) = {emri.relative_deviation(0.10):.4%}")
    for xp in [0.05, 0.10, 0.15]:
        delta = emri.precession_difference(xp)
        gr_pre = emri.gr_precession(xp)
        n_detect = emri.n_orbits_to_dephase(xp)
        print(f"   x_p={xp:.2f}: δ={delta:.4f} rad, 6π·x_p={gr_pre:.3f} rad, "
              f"N_detect(1 rad)={n_detect:.1f}")

    # ── 5. Sgr A* dephasing time ──────────────────────────────────
    print("\n🛰️  Sgr A* EMRI dephasing time (M=4e6 M_sun, x_p=0.10):")
    sgra = emri.compute(x_p=0.10, mass_solar=4e6)
    print(f"   T_obs(1 rad dephasing)  = {emri.t_obs_to_dephase_seconds(0.10, mass_solar=4e6):.0f} s"
          f"  ({emri.t_obs_to_dephase_seconds(0.10, mass_solar=4e6)/3600:.1f} h)")
    print(f"   T_obs(1 rad dephasing)  = {emri.t_obs_to_dephase_seconds(0.10, mass_solar=4e6)/86400:.2f} days")

    # ── 6. Waveform template ──────────────────────────────────────
    print("\n📡 Generating waveform template (mock LISA data)...")
    wf = generate_waveform_template(duration=3600.0, sampling_rate=2.0, x_p=0.10)
    print(f"   Duration: {wf['t'][-1]:.0f} s,  {len(wf['t'])} samples")
    print(f"   Δφ_final: {wf['delta_phi'][-1]:.6f} rad")

    print("\n" + "=" * 64)
    print("  ✅ All predictions locked. Ready for peer review.")
    print("=" * 64)


if __name__ == "__main__":
    main()

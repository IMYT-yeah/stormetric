"""
Generate publication-quality plots for Stormetric.

Run from the repo root:  python examples/plot_results.py [--output FILE]

Output: a 2×2 figure with
  (a) Photon-sphere & shadow radius
  (b) PPN expansion: framework vs GR (full functions)
  (c) EMRI precession: δ = 0.30 · x_p
  (d) Mock EMRI waveform (LISA band)
"""

import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stormetric import (
    ExponentialMetric,
    GRSchwarzschildIsotropic,
    Shadow,
    EMRIPrecession,
    generate_waveform_template,
)


def main(output: str = "docs/stormetric_results.png") -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Stormetric — Storm-Flow Metric: Observational Signatures",
                 fontsize=14, fontweight="bold")

    g = ExponentialMetric()
    gr = GRSchwarzschildIsotropic()
    shadow = Shadow(g)
    emri = EMRIPrecession(g)

    # ── (a) Photon sphere & shadow ────────────────────────────────
    ax = axes[0, 0]
    r = np.linspace(1.5, 20, 500)
    b = np.sqrt(shadow.b_squared(r))
    ax.plot(r, b, color="#2c3e50", lw=2, label="b(r)")
    r_ph = shadow.photon_sphere_radius()
    b_min = shadow.shadow_radius()
    ax.axvline(r_ph, color="red", ls="--", alpha=0.7, label=f"r_ph = {r_ph:.1f}")
    ax.scatter([r_ph], [b_min], color="red", s=80, zorder=5)
    ax.annotate(f"b_min = {b_min:.3f}", xy=(r_ph, b_min),
                xytext=(r_ph + 2, b_min + 1), fontsize=10, color="red")
    ax.axhline(Shadow.EHT_CENTRAL, color="green", ls=":", alpha=0.8,
               label=f"EHT: {Shadow.EHT_CENTRAL} ± {Shadow.EHT_SIGMA}")
    ax.fill_between(r, Shadow.EHT_CENTRAL - Shadow.EHT_SIGMA,
                    Shadow.EHT_CENTRAL + Shadow.EHT_SIGMA,
                    color="green", alpha=0.1)
    ax.set_xlabel("r (GM/c²)")
    ax.set_ylabel("Impact parameter b (GM/c²)")
    ax.set_title("(a) Photon Sphere & Shadow Radius")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── (b) PPN expansion: framework vs GR ────────────────────────
    ax = axes[0, 1]
    x = np.linspace(0.01, 0.5, 200)
    g00_framework = g.g_tt(g.r_of_x(x))
    g00_gr = gr.g_tt(gr.r_of_x(x))
    ax.plot(x, g00_framework, color="#e74c3c", lw=2.5, label="Exponential metric")
    ax.plot(x, g00_gr, color="#2980b9", lw=2, ls="--", label="GR (Schwarzschild)")
    # 3PN divergence band
    ax.axvspan(0.2, 0.5, color="orange", alpha=0.1)
    ax.annotate("3PN divergence\n(framework 4/3 vs GR 3/2)", xy=(0.35, -0.55),
                fontsize=9, ha="center", color="darkorange")
    ax.set_xlabel("x = GM/(c²r)")
    ax.set_ylabel("g₀₀")
    ax.set_title("(b) PPN Expansion: Framework vs GR")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── (c) EMRI precession: δ vs x_p ─────────────────────────────
    ax = axes[1, 0]
    xp_vals = np.linspace(0.01, 0.20, 100)
    delta_vals = emri.precession_difference(xp_vals)
    gr_vals = emri.gr_precession(xp_vals)
    ax.plot(xp_vals, delta_vals, color="#8e44ad", lw=2.5, label="δ (framework)")
    ax.plot(xp_vals, gr_vals, color="#2980b9", lw=2, ls="--", label="6π·x_p (GR)")
    ax.fill_between(xp_vals, 0, delta_vals, alpha=0.15, color="#8e44ad",
                    label=f"Δφ_deviation ≈ {emri.relative_deviation(0.10):.2%} of GR")
    for xp in [0.05, 0.10, 0.15]:
        d = emri.precession_difference(xp)
        ax.scatter([xp], [d], color="black", s=40)
        ax.annotate(f"({xp:.2f}, {d:.3f})", xy=(xp, d),
                    xytext=(xp + 0.015, d + 0.4), fontsize=8)
    ax.set_xlabel("x_p (dimensionless pericenter parameter)")
    ax.set_ylabel("Precession per orbit (rad)")
    ax.set_title("(c) EMRI Precession: δ = 0.30 · x_p vs GR 6π·x_p")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # ── (d) Waveform template (mock LISA) ─────────────────────────
    ax = axes[1, 1]
    wf = generate_waveform_template(duration=3600.0, sampling_rate=4.0, x_p=0.10)
    t = wf["t"]
    ax.plot(t, wf["h_plus"], color="#16a085", lw=1, label="h₊ (GR + correction)")
    ax.plot(t, wf["h_cross"], color="#e67e22", lw=1, ls="--", label="h× (orthogonal)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain amplitude")
    ax.set_title("(d) Mock EMRI Waveform (LISA band)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/stormetric_results.png")
    args = parser.parse_args()
    main(args.output)

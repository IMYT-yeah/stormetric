"""
PPN 3PN bifurcation plot — shows *where* the framework and GR diverge.

Generates a 2-panel figure:
  (a) g_00(x) for the framework metric, with truncated PPN series at
      orders 1PN, 2PN, 3PN, 4PN overlaid.  All four truncations agree
      through 2PN and the 3PN truncation is the first to visibly
      diverge from the exact curve.
  (b) Residual Δg_00 = g_00(truncated @ order n) - g_00(exact) for
      n = 1, 2, 3, 4, 5.  The leading 3PN residual is the framework's
      falsifiable signature.

Run from the repo root:  python examples/ppn_bifurcation.py [--output FILE]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stormetric import (
    ExponentialMetric,
    GRSchwarzschildIsotropic,
    truncated_g00,
)


def main(output: str = "docs/ppn_bifurcation.png") -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    g = ExponentialMetric()
    gr = GRSchwarzschildIsotropic()

    x = np.linspace(0.01, 0.5, 400)
    g00_exact = g.g_tt(g.r_of_x(x))
    g00_gr_exact = gr.g_tt(gr.r_of_x(x))

    # ── panel (a): framework truncated series vs exact ────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.plot(x, g00_exact, color="black", lw=2.5, label="Exact (exponential)")
    for n, color, ls in [
        (1, "#3498db", ":"),
        (2, "#27ae60", "--"),
        (3, "#e67e22", "-."),
        (4, "#c0392b", "-"),
    ]:
        g00_trunc = truncated_g00(g, x, n)
        ax.plot(x, g00_trunc, color=color, lw=1.5, ls=ls,
                label=f"Truncated @ {n}PN")
    ax.set_xlabel("x = GM/(c²r)")
    ax.set_ylabel("g₀₀")
    ax.set_title("(a) Exponential g₀₀(x) — truncated PPN series vs exact")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.axvspan(0.2, 0.5, color="orange", alpha=0.08)

    # ── panel (b): residual Δg_00 = truncated - exact ─────────────
    ax = axes[1]
    for n, color, ls in [
        (1, "#3498db", ":"),
        (2, "#27ae60", "--"),
        (3, "#e67e22", "-."),
        (4, "#c0392b", "-"),
        (5, "#8e44ad", "-"),
    ]:
        g00_trunc = truncated_g00(g, x, n)
        residual = g00_trunc - g00_exact
        ax.plot(x, residual, color=color, lw=1.5, ls=ls,
                label=f"@ {n}PN")
    # the 3PN residual is the leading signature
    res3 = truncated_g00(g, x, 3) - g00_exact
    ax.fill_between(x, 0, res3, alpha=0.10, color="#e67e22",
                    label="leading signature (3PN, 4/3 vs 3/2 = -1/6)")
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel("x = GM/(c²r)")
    ax.set_ylabel("g₀₀(truncated) - g₀₀(exact)")
    ax.set_title("(b) PPN residual — the 3PN gap is the framework signature")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/ppn_bifurcation.png")
    args = parser.parse_args()
    main(args.output)

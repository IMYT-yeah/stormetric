"""
Command-line interface for Stormetric.

Examples
--------

    stormetric shadow --metric exp
    stormetric ppn --metric exp --nmax 5
    stormetric emri --xp 0.10 --mass 4e6
    stormetric waveform --duration 7200 --xp 0.10 --output wf.npz
    stormetric plot --output docs/stormetric_results.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .metric import make_metric
from .ppn import (
    first_bifurcation_order,
    ppn_table,
    truncated_g00,
    truncated_grr,
)
from .shadow import Shadow
from .emri import EMRIPrecession, generate_waveform_template


def _metric(args) -> object:
    return make_metric(args.metric)


def cmd_shadow(args) -> int:
    g = _metric(args)
    s = Shadow(g)
    res = s.compute()
    print(f"Metric            : {g.__class__.__name__}")
    print(f"Photon sphere r_ph: {res.r_ph:.6f} GM/c²  (numeric: {res.r_ph_numeric:.6f})")
    print(f"Shadow radius b   : {res.b_shadow:.6f} GM/c²  (numeric: {res.b_shadow_numeric:.6f})")
    print(f"EHT (5.2 ± 0.3)   : deviation = {res.sigma_to_eht:+.3f} σ")
    print(f"Source            : {res.source}")
    return 0


def cmd_ppn(args) -> int:
    g = _metric(args)
    print(ppn_table(n_max=args.nmax))
    print()
    bif = first_bifurcation_order(n_max=args.nmax)
    print(f"First bifurcation order:")
    print(f"  g_tt : {bif['g_tt']}PN")
    print(f"  g_rr : {bif['g_rr']}PN")
    print(f"  overall: {bif['overall']}PN")

    if args.bifurcation_plot is not None:
        # Save a small CSV with truncated vs exact g_00 at a grid of x
        x = np.linspace(0.01, 0.5, 50)
        with open(args.bifurcation_plot, "w", encoding="utf-8") as fh:
            fh.write("x,g00_exact,g00_truncated_1pn,g00_truncated_2pn,g00_truncated_3pn\n")
            for xi in x:
                exact = g.g_tt(g.r_of_x(np.array([xi])))[0]
                g1 = truncated_g00(g, np.array([xi]), 1)[0]
                g2 = truncated_g00(g, np.array([xi]), 2)[0]
                g3 = truncated_g00(g, np.array([xi]), 3)[0]
                fh.write(f"{xi:.6f},{exact:.10f},{g1:.10f},{g2:.10f},{g3:.10f}\n")
        print(f"\nWrote: {args.bifurcation_plot}")
    return 0


def cmd_emri(args) -> int:
    g = _metric(args)
    emri = EMRIPrecession(g, coupling=args.kappa)
    res = emri.compute(x_p=args.xp, mass_solar=args.mass,
                       eccentricity=args.eccentricity)
    print(f"Pericenter parameter x_p  : {res.x_p:.3f}")
    print(f"Framework coupling κ       : {res.kappa:.3f}")
    print(f"δ per orbit                : {res.delta_per_orbit:.6f} rad")
    print(f"GR baseline (6π·x_p)       : {res.gr_precession_per_orbit:.6f} rad")
    print(f"Total framework            : {res.delta_per_orbit + res.gr_precession_per_orbit:.6f} rad")
    print(f"Relative deviation         : {res.relative_deviation:.4%}")
    print(f"Orbits to 1 rad dephasing  : {res.n_orbits_to_dephase:.2f}")
    print(f"Orbital period (geo. units): {res.period_geometric:.4f}")
    t_obs_s = emri.t_obs_to_dephase_seconds(
        args.xp, mass_solar=args.mass, eccentricity=args.eccentricity
    )
    print(f"T_obs to 1 rad (seconds)  : {t_obs_s:.2f}")
    print(f"T_obs to 1 rad (days)     : {t_obs_s/86400.0:.3f}")
    print(f"T_obs to 1 rad (years)    : {t_obs_s/(365.25*86400.0):.4f}")
    print(f"Dephased within 1 year?    : {'YES' if res.dephased else 'no'}")
    return 0


def cmd_waveform(args) -> int:
    wf = generate_waveform_template(
        duration=args.duration, sampling_rate=args.sampling_rate,
        x_p=args.xp, coupling=args.kappa, mass_solar=args.mass,
    )
    if args.output:
        np.savez(args.output, **wf)
        print(f"Wrote: {args.output}  (n={len(wf['t'])} samples)")
    else:
        print(f"h_plus range    : [{wf['h_plus'].min():.3f}, {wf['h_plus'].max():.3f}]")
        print(f"delta_phi final : {wf['delta_phi'][-1]:.6f} rad")
        print(f"phase_total_end : {wf['phase_total'][-1]:.3f} rad")
    return 0


def cmd_plot(args) -> int:
    from examples.plot_results import main as plot_main  # type: ignore
    plot_main(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stormetric",
        description="Stormetric — Storm-Flow Relativistic Metrics",
    )
    p.add_argument("--version", action="version", version=f"stormetric {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--metric", default="exp",
                        choices=["exp", "gr", "storm", "schw"],
                        help="Metric family (default: exp = exponential)")

    sp = sub.add_parser("shadow", parents=[common],
                        help="Compute photon-sphere and shadow radius")
    sp.set_defaults(func=cmd_shadow)

    pp = sub.add_parser("ppn", parents=[common],
                        help="Print PPN coefficient table and bifurcation order")
    pp.add_argument("--nmax", type=int, default=5)
    pp.add_argument("--bifurcation-plot", default=None,
                    help="Optional CSV file with truncated-vs-exact g_00")
    pp.set_defaults(func=cmd_ppn)

    em = sub.add_parser("emri", parents=[common],
                        help="EMRI pericenter precession & dephasing time")
    em.add_argument("--xp", type=float, default=0.10,
                    help="Pericenter parameter (default 0.10)")
    em.add_argument("--mass", type=float, default=1e6,
                    help="Central mass in solar masses (default 1e6)")
    em.add_argument("--eccentricity", type=float, default=0.0)
    em.add_argument("--kappa", type=float, default=0.30)
    em.set_defaults(func=cmd_emri)

    wf = sub.add_parser("waveform", parents=[common],
                        help="Generate mock EMRI waveform template")
    wf.add_argument("--duration", type=float, default=86400.0)
    wf.add_argument("--sampling-rate", type=float, default=1.0)
    wf.add_argument("--xp", type=float, default=0.10)
    wf.add_argument("--mass", type=float, default=1e6)
    wf.add_argument("--kappa", type=float, default=0.30)
    wf.add_argument("--output", default=None)
    wf.set_defaults(func=cmd_waveform)

    pl = sub.add_parser("plot", parents=[common],
                        help="Generate publication-quality results figure")
    pl.add_argument("--output", default="docs/stormetric_results.png")
    pl.set_defaults(func=cmd_plot)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

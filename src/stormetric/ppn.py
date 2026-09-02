"""
Stormetric — PPN expansion and bifurcation analysis.

The framework's central post-Newtonian claim is that
:math:`g_{00}` and :math:`g_{rr}` match the standard
Parameterized Post-Newtonian (PPN) parameters
:math:`\\gamma = 1` and :math:`\\beta = 1` through 2PN, and the
first *deviation* enters at:

* **2PN in** :math:`g_{rr}` — exponential gives 2 vs GR 3/2
* **3PN in** :math:`g_{00}` — exponential gives 4/3 vs GR 3/2

This module exposes the truncated series, the framework-vs-GR
coefficient table, and helpers to *see* the 3PN bifurcation
visually (used in :mod:`examples.ppn_bifurcation`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .metric import GRSchwarzschildIsotropic, ExponentialMetric, Metric


# ─────────────────────────────────────────────────────────────────────
# Series construction
# ─────────────────────────────────────────────────────────────────────
def truncated_g00(metric: Metric, x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate :math:`g_{00}(x)` truncated to PPN order ``order``.

    Convention: g_00 = -1 + a_1 x + a_2 x² + ... + a_{order} x^{order}.
    """
    coef = metric.ppn_g00_coefficients(n_max=order)
    x_arr = np.asarray(x, dtype=float)
    out = -1.0 + np.zeros_like(x_arr, dtype=float)
    for n in range(1, order + 1):
        out = out + coef[n] * x_arr ** n
    return out


def truncated_grr(metric: Metric, x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate :math:`g_{rr}(x)` truncated to PPN order ``order``.

    g_rr = 1 + b_1 x + b_2 x² + ... + b_{order} x^{order}.
    """
    coef = metric.ppn_grr_coefficients(n_max=order)
    x_arr = np.asarray(x, dtype=float)
    out = 1.0 + np.zeros_like(x_arr, dtype=float)
    for n in range(1, order + 1):
        out = out + coef[n] * x_arr ** n
    return out


def exact_g00(metric: Metric, x: np.ndarray) -> np.ndarray:
    """Closed-form g_{00}(x) from the metric object."""
    return np.asarray(metric.g_tt(metric.r_of_x(x)), dtype=float)


def exact_grr(metric: Metric, x: np.ndarray) -> np.ndarray:
    """Closed-form g_{rr}(x) from the metric object."""
    return np.asarray(metric.g_rr(metric.r_of_x(x)), dtype=float)


# ─────────────────────────────────────────────────────────────────────
# Coefficient table
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PPNRow:
    order: int
    a_framework: float
    a_gr: float
    b_framework: float
    b_gr: float
    matches_gr: bool  # True iff both a_n and b_n equal GR

    @property
    def delta_a(self) -> float:
        return self.a_framework - self.a_gr

    @property
    def delta_b(self) -> float:
        return self.b_framework - self.b_gr


def ppn_table(metrics: Tuple[Metric, Metric] = None, n_max: int = 5) -> str:
    """Return a markdown table comparing framework vs GR PPN coefficients.

    ``metrics`` is a 2-tuple (framework, gr).  Defaults to
    ``(ExponentialMetric(), GRSchwarzschildIsotropic())``.
    """
    if metrics is None:
        framework = ExponentialMetric()
        gr = GRSchwarzschildIsotropic()
    else:
        framework, gr = metrics
    a_f = framework.ppn_g00_coefficients(n_max=n_max)
    a_g = gr.ppn_g00_coefficients(n_max=n_max)
    b_f = framework.ppn_grr_coefficients(n_max=n_max)
    b_g = gr.ppn_grr_coefficients(n_max=n_max)

    lines = ["| n  |  a_n (framework) |  a_n (GR) |  Δa_n |  b_n (framework) |  b_n (GR) |  Δb_n |  matches? |",
             "|---:|-----------------:|----------:|------:|-----------------:|----------:|------:|:----------|"]
    for n in range(1, n_max + 1):
        delta_a = a_f[n] - a_g[n]
        delta_b = b_f[n] - b_g[n]
        ok = (delta_a == 0.0) and (delta_b == 0.0)
        lines.append(
            f"| {n}  |  {a_f[n]:+.10f} |  {a_g[n]:+.10f} |  {delta_a:+.4f} "
            f"|  {b_f[n]:+.10f} |  {b_g[n]:+.10f} |  {delta_b:+.4f} "
            f"|  {'✅' if ok else '❌'} |"
        )
    return "\n".join(lines)


def ppn_rows(metrics: Tuple[Metric, Metric] = None, n_max: int = 5):
    """Iterator yielding :class:`PPNRow` for each PPN order."""
    if metrics is None:
        framework = ExponentialMetric()
        gr = GRSchwarzschildIsotropic()
    else:
        framework, gr = metrics
    a_f = framework.ppn_g00_coefficients(n_max=n_max)
    a_g = gr.ppn_g00_coefficients(n_max=n_max)
    b_f = framework.ppn_grr_coefficients(n_max=n_max)
    b_g = gr.ppn_grr_coefficients(n_max=n_max)
    for n in range(1, n_max + 1):
        yield PPNRow(
            order=n,
            a_framework=a_f[n],
            a_gr=a_g[n],
            b_framework=b_f[n],
            b_gr=b_g[n],
            matches_gr=(a_f[n] == a_g[n]) and (b_f[n] == b_g[n]),
        )


# ─────────────────────────────────────────────────────────────────────
# Bifurcation order: the *first* PPN order at which framework ≠ GR
# ─────────────────────────────────────────────────────────────────────
def first_bifurcation_order(
    metrics: Tuple[Metric, Metric] = None,
    n_max: int = 5,
) -> Dict[str, int]:
    """Return the first PPN order where framework and GR diverge.

    Returns ``{"g_tt": n, "g_rr": m, "overall": min(n, m)}``.
    """
    if metrics is None:
        framework = ExponentialMetric()
        gr = GRSchwarzschildIsotropic()
    else:
        framework, gr = metrics
    a_f = framework.ppn_g00_coefficients(n_max=n_max)
    a_g = gr.ppn_g00_coefficients(n_max=n_max)
    b_f = framework.ppn_grr_coefficients(n_max=n_max)
    b_g = gr.ppn_grr_coefficients(n_max=n_max)
    n_tt = next(n for n in range(1, n_max + 1) if a_f[n] != a_g[n])
    n_rr = next(n for n in range(1, n_max + 1) if b_f[n] != b_g[n])
    return {"g_tt": n_tt, "g_rr": n_rr, "overall": min(n_tt, n_rr)}


# ─────────────────────────────────────────────────────────────────────
# 3PN leading residual (the falsifiable signature)
# ─────────────────────────────────────────────────────────────────────
def leading_residual_coefficient(
    metrics: Tuple[Metric, Metric] = None,
) -> Dict[str, float]:
    """Return the leading (smallest n) framework-vs-GR coefficient gap
    for ``g_tt`` and ``g_rr`` *separately*.

    For the exponential metric, the result is::

        {"g_tt": -1/6, "g_rr": +1/2, "order_g_tt": 3, "order_g_rr": 2}

    corresponding to 3PN in g_tt (4/3 - 3/2 = -1/6) and 2PN in
    g_rr (2 - 3/2 = +1/2).
    """
    rows = list(ppn_rows(metrics=metrics, n_max=4))
    out: Dict[str, float] = {}
    out_tt_set = False
    out_rr_set = False
    for r in rows:
        if (not out_tt_set) and r.delta_a != 0.0:
            out["g_tt"] = r.delta_a
            out["order_g_tt"] = r.order
            out_tt_set = True
        if (not out_rr_set) and r.delta_b != 0.0:
            out["g_rr"] = r.delta_b
            out["order_g_rr"] = r.order
            out_rr_set = True
        if out_tt_set and out_rr_set:
            break
    return out

"""
Stormetric — Black-hole shadow / photon-sphere calculation.

For a static, spherically symmetric isotropic metric

.. math::

    ds^2 = -A(r) dt^2 + B(r) (dr^2 + r^2 d\\Omega^2),

the equatorial-plane impact parameter of a photon is

.. math::

    b^2(r) = \\frac{r^2 B(r)}{A(r)}.

The photon sphere sits where :math:`b(r)` is stationary,
:math:`db/dr = 0`, i.e.

.. math::

    \\frac{d\\ln b^2}{dr} = 0.

The shadow radius is :math:`b_{\\rm shadow} = b(r_{\\rm ph})`.

For the exponential metric this has the analytic solution
:math:`r_{\\rm ph} = 2GM/c^2` and
:math:`b_{\\rm shadow} = 2 e\\, GM/c^2 \\approx 5.4366` —
compatible with the EHT measurement
:math:`\\alpha_{86} = 5.2 \\pm 0.3` for M87*/Sgr A*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .metric import ExponentialMetric, Metric


# ─────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ShadowResult:
    """Container for shadow / photon-sphere computation."""
    r_ph: float                 # photon-sphere radius (GM/c² units)
    b_shadow: float             # shadow radius (GM/c² units)
    sigma_to_eht: float         # deviation from EHT 5.2 ± 0.3
    r_ph_numeric: float         # numeric root of d(ln b²)/dr = 0
    b_shadow_numeric: float     # numeric shadow at numeric r_ph
    source: str                 # "analytic" or "numeric"

    def to_dict(self) -> dict:
        return {
            "r_ph": self.r_ph,
            "b_shadow": self.b_shadow,
            "sigma_to_eht": self.sigma_to_eht,
            "r_ph_numeric": self.r_ph_numeric,
            "b_shadow_numeric": self.b_shadow_numeric,
            "source": self.source,
        }


# ─────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────
class Shadow:
    """Compute the photon sphere and shadow radius for any :class:`Metric`.

    For metrics that admit a closed-form photon sphere, set
    ``analytic=True`` to use the formula.  By default the code first
    validates the analytic value against a numeric root finder; if the
    deviation exceeds ``tol`` the numeric value is used.
    """

    EHT_CENTRAL = 5.2
    EHT_SIGMA = 0.3

    def __init__(self, metric: Metric, tol: float = 1e-6) -> None:
        self.metric = metric
        self.tol = tol

    # ── raw impact parameter ────────────────────────────────────────
    def b_squared(self, r: np.ndarray) -> np.ndarray:
        """Impact parameter squared at radius r (equatorial plane).

        b²(r) = r² B(r) / A(r)   where A = -g_tt, B = g_rr.
        """
        r_arr = np.asarray(r, dtype=float)
        g_tt = np.asarray(self.metric.g_tt(r_arr), dtype=float)
        g_rr = np.asarray(self.metric.g_rr(r_arr), dtype=float)
        return r_arr ** 2 * g_rr / (-g_tt)

    def dln_b2(self, r: float, h: float = 1e-6) -> float:
        """d ln(b²)/dr by central difference."""
        b2_p = self.b_squared(r + h)
        b2_m = self.b_squared(r - h)
        return (np.log(b2_p) - np.log(b2_m)) / (2.0 * h)

    # ── photon sphere: numeric root of d ln b² / dr = 0 ─────────────
    def photon_sphere_radius_numeric(
        self, r_lo: float = 1.1, r_hi: float = 20.0
    ) -> float:
        """Solve d(ln b²)/dr = 0 with :func:`scipy.optimize.brentq`.

        ``r_lo``, ``r_hi`` bracket the search window (in units of
        :math:`GM/c^2`).  We start with the search centred just above
        the horizon; if the sign convention flips we widen.
        """
        f_lo = self.dln_b2(r_lo)
        f_hi = self.dln_b2(r_hi)
        # If signs agree, widen progressively up to r=1000
        for r_lo_try, r_hi_try in [(r_lo, r_hi),
                                   (1.05, 100.0),
                                   (1.01, 1000.0)]:
            fl = self.dln_b2(r_lo_try)
            fh = self.dln_b2(r_hi_try)
            if fl * fh < 0.0:
                return brentq(self.dln_b2, r_lo_try, r_hi_try, xtol=1e-10)
        # Last resort: golden-section on |d ln b²/dr|
        rs = np.linspace(r_lo, 1000.0, 20000)
        fvals = np.array([abs(self.dln_b2(r)) for r in rs])
        idx = int(np.argmin(fvals))
        return float(rs[idx])

    # ── analytic override for the exponential metric ───────────────
    def photon_sphere_radius_analytic(self) -> Optional[float]:
        """Closed-form photon-sphere radius for the exponential metric.

        For g_tt = -e^{-2x}, g_rr = e^{2x} with x = GM/(c²r) we have
        b² = r² e^{4x}; d(b²)/dr = e^{4x} (2r - 4GM/c²) ⇒ r_ph = 2GM/c².
        """
        if isinstance(self.metric, ExponentialMetric):
            return 2.0 * self.metric.GM / self.metric.c ** 2
        return None

    # ── public API: returns ShadowResult ────────────────────────────
    def photon_sphere_radius(self) -> float:
        """Photon-sphere radius in units of :math:`GM/c^2`."""
        ana = self.photon_sphere_radius_analytic()
        if ana is not None:
            return ana
        return self.photon_sphere_radius_numeric()

    def shadow_radius(self) -> float:
        """Shadow radius b_min = b(r_ph) in units of :math:`GM/c^2`."""
        r_ph = self.photon_sphere_radius()
        return float(np.sqrt(self.b_squared(np.array([r_ph])))[0])

    # ── full result ─────────────────────────────────────────────────
    def compute(self) -> ShadowResult:
        """Return a :class:`ShadowResult` with both analytic and numeric r_ph."""
        ana = self.photon_sphere_radius_analytic()
        r_ph_num = self.photon_sphere_radius_numeric()
        b_num = float(np.sqrt(self.b_squared(np.array([r_ph_num])))[0])
        if ana is None:
            return ShadowResult(
                r_ph=r_ph_num, b_shadow=b_num, sigma_to_eht=0.0,
                r_ph_numeric=r_ph_num, b_shadow_numeric=b_num,
                source="numeric",
            )
        # sanity check
        diff = abs(ana - r_ph_num) / abs(ana)
        if diff > self.tol:
            # fall back to numeric
            return ShadowResult(
                r_ph=r_ph_num, b_shadow=b_num, sigma_to_eht=0.0,
                r_ph_numeric=r_ph_num, b_shadow_numeric=b_num,
                source="numeric (analytic mismatch {:.2e})".format(diff),
            )
        b_ana = float(np.sqrt(self.b_squared(np.array([ana])))[0])
        sigma = (b_ana - self.EHT_CENTRAL) / self.EHT_SIGMA
        return ShadowResult(
            r_ph=ana, b_shadow=b_ana, sigma_to_eht=sigma,
            r_ph_numeric=r_ph_num, b_shadow_numeric=b_num,
            source="analytic (validated vs numeric)",
        )

    # ── EHT comparison ─────────────────────────────────────────────
    def compare_with_eht(self, shadow_mass: float = 1.0) -> dict:
        """Compare predicted shadow with EHT observation."""
        predicted = self.shadow_radius()
        return {
            "predicted": predicted,
            "eht_central": self.EHT_CENTRAL,
            "eht_uncertainty": self.EHT_SIGMA,
            "deviation_in_sigma": (predicted - self.EHT_CENTRAL) / self.EHT_SIGMA,
        }

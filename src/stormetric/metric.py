"""
Stormetric — Metric definitions.

This module provides:

* :class:`Metric` — abstract base class for any static, spherically symmetric
  metric written in isotropic-like coordinates
  :math:`ds^2 = -A(r) dt^2 + B(r) (dr^2 + r^2 d\\Omega^2)`.

* :class:`ExponentialMetric` — the *storm-flow* ansatz

  .. math::

      g_{tt} = -e^{-2x}, \\quad
      g_{rr} = e^{2x}, \\quad
      x = \\frac{GM}{c^2 r}.

* :class:`GRSchwarzschildIsotropic` — Schwarzschild in isotropic coordinates,
  the standard GR reference for PPN / shadow comparisons

  .. math::

      g_{tt} = -\\left(\\frac{1-x/2}{1+x/2}\\right)^2, \\quad
      g_{rr} = \\left(1+\\frac{x}{2}\\right)^4.

Both metrics share the same dimensionless radial coordinate :math:`x` and
agree with the classical PPN parameters
:math:`\\gamma=1` and :math:`\\beta=1` through 2PN.  The first
post-GR signature appears at:

* **2PN in** :math:`g_{rr}` (framework 2 vs GR 1.5)
* **3PN in** :math:`g_{tt}` (framework 4/3 vs GR 1.5)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np

Number = Union[float, int, np.ndarray]


class Metric(ABC):
    """Abstract base class for a static, spherically symmetric isotropic metric.

    All subclasses must implement :meth:`g_tt` and :meth:`g_rr`.  The angular
    components follow from spherical symmetry:

    .. math::

        g_{\\theta\\theta} = r^2 B(r), \\quad
        g_{\\phi\\phi}     = r^2 \\sin^2\\theta\\, B(r).
    """

    def __init__(self, GM: float = 1.0, c: float = 1.0) -> None:
        self.GM = float(GM)
        self.c = float(c)
        # Schwarzschild radius in these units:
        self.r_s = 2.0 * self.GM / self.c ** 2

    # ── abstract metric components ──────────────────────────────────────
    @abstractmethod
    def g_tt(self, r: Number) -> Number:
        """Time-time component g_{00} (negative)."""

    @abstractmethod
    def g_rr(self, r: Number) -> Number:
        """Radial component g_{11} (positive)."""

    # ── angular components (spherical symmetry) ───────────────────────
    def g_theta_theta(self, r: Number) -> Number:
        r_arr = np.asarray(r, dtype=float)
        return r_arr ** 2 * self.g_rr(r_arr)

    def g_phi_phi(self, r: Number, theta: float = math.pi / 2) -> Number:
        """Azimuthal component g_{33}.

        Default ``theta = pi/2`` selects the equatorial plane (the natural
        choice for photon-sphere / shadow / orbital calculations).
        """
        r_arr = np.asarray(r, dtype=float)
        return r_arr ** 2 * np.sin(theta) ** 2 * self.g_rr(r_arr)

    # ── helpers ───────────────────────────────────────────────────────
    def x_of_r(self, r: Number) -> Number:
        """Dimensionless coordinate :math:`x = GM / (c^2 r)`."""
        return self.GM / (self.c ** 2 * np.asarray(r, dtype=float))

    def r_of_x(self, x: Number) -> Number:
        """Inverse: :math:`r = GM / (c^2 x)`."""
        return self.GM / (self.c ** 2 * np.asarray(x, dtype=float))

    # ── PPN: subclasses must declare their series ─────────────────────
    @abstractmethod
    def ppn_g00_coefficients(self, n_max: int = 6) -> Dict[int, float]:
        """Coefficients :math:`a_n` in :math:`g_{00} = -1 + \\sum a_n x^n`."""

    @abstractmethod
    def ppn_grr_coefficients(self, n_max: int = 6) -> Dict[int, float]:
        """Coefficients :math:`b_n` in :math:`g_{rr} = 1 + \\sum b_n x^n`."""


# ─────────────────────────────────────────────────────────────────────
# Exponential (storm-flow) metric
# ─────────────────────────────────────────────────────────────────────
class ExponentialMetric(Metric):
    """The exponential isotropic metric (storm-flow framework).

    .. math::

        g_{tt} = -e^{-2x}, \\quad g_{rr} = e^{2x}.
    """

    name = "exponential"

    def g_tt(self, r: Number) -> Number:
        return -np.exp(-2.0 * self.x_of_r(r))

    def g_rr(self, r: Number) -> Number:
        return np.exp(2.0 * self.x_of_r(r))

    def ppn_g00_coefficients(self, n_max: int = 6) -> Dict[int, float]:
        # -e^{-2x} = -(1 - 2x + 2x² - 4/3 x³ + 2/3 x⁴ - 4/15 x⁵ + 4/45 x⁶ - ...)
        #          = -1 + 2x - 2x² + 4/3 x³ - 2/3 x⁴ + 4/15 x⁵ - 4/45 x⁶ + ...
        return {
            n: ((-1) ** (n + 1)) * (2 ** n) / math.factorial(n)
            for n in range(1, n_max + 1)
        }

    def ppn_grr_coefficients(self, n_max: int = 6) -> Dict[int, float]:
        # e^{2x} = 1 + 2x + 2x² + 4/3 x³ + 2/3 x⁴ + 4/15 x⁵ + 4/45 x⁶ + ...
        return {n: (2 ** n) / math.factorial(n) for n in range(1, n_max + 1)}

    def __repr__(self) -> str:  # pragma: no cover
        return f"ExponentialMetric(GM={self.GM}, c={self.c})"


# ─────────────────────────────────────────────────────────────────────
# Schwarzschild (GR reference) in isotropic coordinates
# ─────────────────────────────────────────────────────────────────────
class GRSchwarzschildIsotropic(Metric):
    """Schwarzschild metric in isotropic coordinates — the GR baseline.

    .. math::

        g_{tt} = -((1 - x/2) / (1 + x/2))^2, \\quad
        g_{rr} = (1 + x/2)^4.
    """

    name = "gr_schwarzschild_isotropic"

    def g_tt(self, r: Number) -> Number:
        x = self.x_of_r(r)
        return -((1.0 - x / 2.0) / (1.0 + x / 2.0)) ** 2

    def g_rr(self, r: Number) -> Number:
        x = self.x_of_r(r)
        return (1.0 + x / 2.0) ** 4

    def ppn_g00_coefficients(self, n_max: int = 6) -> Dict[int, float]:
        # (1 - x/2)² = 1 - x + x²/4
        # (1 + x/2)^{-2} = 1 - x + 3x²/4 - x³/2 + 5x⁴/16 - 7x⁵/32 + 21 x⁶/128 - ...
        # Product gives g_tt = -(1 - 2x + 2x² - 1.5x³ + x⁴ - 1.25x⁵ + 7/8 x⁶ - ...)
        #                  = -1 + 2x - 2x² + 1.5x³ - x⁴ + 1.25x⁵ - 7/8 x⁶ + ...
        coef_tt = {1: 2.0, 2: -2.0, 3: 3.0 / 2.0, 4: -1.0,
                   5: 5.0 / 4.0, 6: -7.0 / 8.0}
        return {n: coef_tt[n] for n in range(1, n_max + 1)}

    def ppn_grr_coefficients(self, n_max: int = 6) -> Dict[int, float]:
        # (1 + x/2)⁴ = 1 + 2x + 1.5x² + 0.5x³ + (1/16) x⁴ + ...
        # i.e. coefficients are C(4, n) / 2^n
        return {n: math.comb(4, n) / (2 ** n) for n in range(1, n_max + 1)}

    def __repr__(self) -> str:  # pragma: no cover
        return f"GRSchwarzschildIsotropic(GM={self.GM}, c={self.c})"


# ─────────────────────────────────────────────────────────────────────
# factory
# ─────────────────────────────────────────────────────────────────────
_REGISTRY: Dict[str, type] = {
    cls.name: cls for cls in (ExponentialMetric, GRSchwarzschildIsotropic)
}


def make_metric(name: str, GM: float = 1.0, c: float = 1.0) -> Metric:
    """Construct a metric by short name.

    >>> make_metric("exp")
    ExponentialMetric(GM=1.0, c=1.0)
    >>> make_metric("gr")
    GRSchwarzschildIsotropic(GM=1.0, c=1.0)
    """
    key = name.lower().strip()
    aliases = {"exp": "exponential", "storm": "exponential",
               "gr": "gr_schwarzschild_isotropic", "schw": "gr_schwarzschild_isotropic"}
    key = aliases.get(key, key)
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown metric {name!r}. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](GM=GM, c=c)

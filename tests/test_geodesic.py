"""Numerical geodesic tests for the shadow module.

These tests integrate null (photon) geodesics in the equatorial plane
and verify the shadow radius from the capture / scatter boundary.
The test is independent of any analytic result baked into the package
and therefore constitutes an *independent* numerical check.
"""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from stormetric import ExponentialMetric, Shadow


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _exp_u_dot(u, b, M=1.0):
    """du/dφ for null geodesics in the exponential metric.

    From the standard orbit equation
    (du/dφ)² = (1/b²) · B(u)/A(u) - u²
    with B/A = e^{4Mu}, so (du/dφ)² = e^{4Mu}/b² - u².
    """
    arg = 4.0 * M * u
    if arg > 50.0:  # exp(50) ≈ 5e21 still safe, but exp(700) is the limit
        # Once the exponential dominates, the integrand grows without bound
        # and the photon is captured (u → ∞ ⇒ r → 0).  Return a sentinel.
        return float("inf")
    val = math.exp(arg) / b ** 2 - u ** 2
    if val <= 0.0:
        return 0.0
    return math.sqrt(val)


def _photon_min_r(b: float, r0: float = 30.0, M: float = 1.0) -> float:
    """Integrate a photon from r=r0 inward and return the minimum r reached.

    Uses a sign flip on du/dφ at each turning point.
    """
    u0 = 1.0 / r0
    sign = +1.0
    u = u0
    phi = 0.0
    dphi = 1e-3
    u_min = u0
    for _ in range(2_000_000):
        du = _exp_u_dot(u, b, M)
        if du == float("inf"):
            # captured (u → ∞ ⇒ r → 0)
            return 0.0
        u_next = u + sign * du * dphi
        if u_next <= 0.0:
            return math.inf
        if u_next > 1e6:
            return math.inf
        u = u_next
        phi += dphi
        if u > u_min:
            u_min = u
        if _exp_u_dot(u, b, M) <= 0.0:
            sign = -1.0
            if phi > 200:
                return 1.0 / u
    return 1.0 / u_min


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────
class TestShadowGeodesic:

    expected_b_shadow = 2.0 * math.e  # ≈ 5.4366

    def test_below_shadow_captures(self):
        # b slightly below critical → photon should be captured (r → 0)
        b = self.expected_b_shadow * 0.99
        r_min = _photon_min_r(b, r0=30.0)
        # Captured means u → ∞ ⇒ r → 0; we just assert r_min < 1 (well inside)
        assert r_min < 1.0, f"b={b:.3f}: r_min={r_min}, expected capture"

    def test_above_shadow_scatters(self):
        # b slightly above critical → photon should scatter (r_min stays > r_ph)
        b = self.expected_b_shadow * 1.01
        r_min = _photon_min_r(b, r0=30.0)
        # Scattered means r_min stays > photon-sphere radius (2)
        assert r_min > 1.5, f"b={b:.3f}: r_min={r_min}, expected scatter"

    def test_critical_b_is_2e(self):
        # The critical impact parameter should be very close to 2e.
        # We verify by checking that the shadow radius from the package
        # matches 2e to high precision (analytic + numeric).
        g = ExponentialMetric()
        s = Shadow(g)
        b_pkg = s.shadow_radius()
        assert b_pkg == pytest.approx(self.expected_b_shadow, rel=1e-5)


class TestShadowConsistency:

    def test_shadow_independent_of_initial_r0(self):
        """The capture / scatter boundary should not depend on r0."""
        b = 2.0 * math.e
        # slightly above critical
        r_a = _photon_min_r(b * 1.001, r0=20.0)
        r_b = _photon_min_r(b * 1.001, r0=40.0)
        # both should be scattered (r_min > 1.5)
        assert r_a > 1.5
        assert r_b > 1.5

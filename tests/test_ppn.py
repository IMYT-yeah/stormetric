"""Tests for the PPN module."""

import math

import numpy as np
import pytest

from stormetric import (
    ExponentialMetric,
    GRSchwarzschildIsotropic,
    first_bifurcation_order,
    leading_residual_coefficient,
    ppn_rows,
    ppn_table,
    truncated_g00,
    truncated_grr,
)


class TestTruncatedSeries:

    def test_truncated_g00_at_order_1(self):
        g = ExponentialMetric()
        x = np.array([0.1, 0.2, 0.3])
        # 1PN: -1 + 2x
        expected = -1.0 + 2.0 * x
        assert np.allclose(truncated_g00(g, x, 1), expected)

    def test_truncated_g00_at_order_3(self):
        g = ExponentialMetric()
        x = np.array([0.1])
        # 3PN: -1 + 2x - 2x² + 4/3 x³
        expected = np.array([-1.0 + 2.0 * 0.1 - 2.0 * 0.01 + (4.0/3.0) * 0.001])
        assert np.allclose(truncated_g00(g, x, 3), expected, atol=1e-12)

    def test_truncated_g00_converges_to_exact(self):
        g = ExponentialMetric()
        x = np.array([0.1, 0.2])
        for order in [1, 2, 3, 4, 5, 6]:
            trunc = truncated_g00(g, x, order)
            exact = g.g_tt(g.r_of_x(x))
            err = np.max(np.abs(trunc - exact))
            # leading residual term is |a_{n+1}| x^{n+1} with a_{n+1} = 2^{n+1}/(n+1)!
            # observed residual is bounded by (full tail sum) — we use 3x safety
            expected = 3.0 * (2.0 ** (order + 1) / math.factorial(order + 1)) * np.max(np.abs(x)) ** (order + 1)
            assert err < expected, f"order={order}: err={err:.3e} > {expected:.3e}"


class TestPPNTable:

    def test_ppn_table_md_format(self):
        md = ppn_table(n_max=4)
        # Headers + 4 rows + separator
        lines = md.split("\n")
        assert len(lines) == 6
        assert "a_n (framework)" in lines[0]
        assert "b_n (GR)" in lines[0]

    def test_ppn_rows_returns_4(self):
        rows = list(ppn_rows(n_max=4))
        assert len(rows) == 4

    def test_ppn_row_1pn_matches(self):
        rows = list(ppn_rows(n_max=4))
        # 1PN: both g_tt and g_rr should match
        assert rows[0].matches_gr is True
        assert rows[0].a_framework == pytest.approx(2.0)
        assert rows[0].b_framework == pytest.approx(2.0)


class TestBifurcation:

    def test_first_bifurcation_g_tt_is_3pn(self):
        bif = first_bifurcation_order(n_max=5)
        assert bif["g_tt"] == 3

    def test_first_bifurcation_g_rr_is_2pn(self):
        bif = first_bifurcation_order(n_max=5)
        # Honest note: framework g_rr diverges from GR at 2PN
        # (2 vs 1.5) — earlier than g_tt (3PN).
        assert bif["g_rr"] == 2

    def test_overall_bifurcation(self):
        bif = first_bifurcation_order(n_max=5)
        assert bif["overall"] == 2

    def test_leading_residual(self):
        coef = leading_residual_coefficient()
        # g_tt 3PN gap: 4/3 - 3/2 = -1/6 ≈ -0.1667
        assert coef["g_tt"] == pytest.approx(-1.0 / 6.0)
        # g_rr 2PN gap: 2 - 3/2 = +1/2
        assert coef["g_rr"] == pytest.approx(0.5)
        assert coef["order_g_tt"] == 3
        assert coef["order_g_rr"] == 2

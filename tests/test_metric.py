"""Tests for the metric module."""

import numpy as np
import pytest

from stormetric import (
    ExponentialMetric,
    GRSchwarzschildIsotropic,
    Metric,
    make_metric,
)


class TestExponentialMetric:

    def test_g_tt_at_infinity(self):
        g = ExponentialMetric()
        assert g.g_tt(np.array([1e10]))[0] == pytest.approx(-1.0, rel=1e-6)

    def test_g_rr_at_infinity(self):
        g = ExponentialMetric()
        assert g.g_rr(np.array([1e10]))[0] == pytest.approx(1.0, rel=1e-6)

    def test_g_tt_at_horizon(self):
        g = ExponentialMetric(GM=1.0, c=1.0)
        # at r = 2 GM/c² (the "horizon-like" radius for the exp metric),
        # x = 1/2, g_tt = -e^{-1} ≈ -0.368
        r = 2.0
        assert g.g_tt(np.array([r]))[0] == pytest.approx(-np.exp(-1.0))

    def test_ppn_g00_coefficients(self):
        g = ExponentialMetric()
        c = g.ppn_g00_coefficients(n_max=4)
        assert c[1] == pytest.approx(2.0)
        assert c[2] == pytest.approx(-2.0)
        assert c[3] == pytest.approx(4.0 / 3.0)
        assert c[4] == pytest.approx(-2.0 / 3.0)

    def test_ppn_grr_coefficients(self):
        g = ExponentialMetric()
        c = g.ppn_grr_coefficients(n_max=4)
        assert c[1] == pytest.approx(2.0)
        assert c[2] == pytest.approx(2.0)   # 2 vs GR's 1.5 → diverges at 2PN

    def test_g_phi_phi_default_equatorial(self):
        g = ExponentialMetric()
        r = np.array([1.0, 2.0, 5.0])
        # default theta=π/2 → sin² = 1
        assert g.g_phi_phi(r)[0] == pytest.approx(g.g_rr(r)[0])


class TestGRSchwarzschildIsotropic:

    def test_g_tt_at_infinity(self):
        g = GRSchwarzschildIsotropic()
        assert g.g_tt(np.array([1e10]))[0] == pytest.approx(-1.0, rel=1e-6)

    def test_g_rr_at_infinity(self):
        g = GRSchwarzschildIsotropic()
        assert g.g_rr(np.array([1e10]))[0] == pytest.approx(1.0, rel=1e-6)

    def test_ppn_g00_coefficients(self):
        g = GRSchwarzschildIsotropic()
        c = g.ppn_g00_coefficients(n_max=4)
        assert c[1] == pytest.approx(2.0)
        assert c[2] == pytest.approx(-2.0)
        assert c[3] == pytest.approx(1.5)
        assert c[4] == pytest.approx(-1.0)

    def test_ppn_grr_coefficients(self):
        g = GRSchwarzschildIsotropic()
        c = g.ppn_grr_coefficients(n_max=4)
        assert c[1] == pytest.approx(2.0)
        assert c[2] == pytest.approx(1.5)
        assert c[3] == pytest.approx(0.5)


class TestFactory:

    def test_make_exp(self):
        g = make_metric("exp")
        assert isinstance(g, ExponentialMetric)

    def test_make_storm_alias(self):
        assert isinstance(make_metric("storm"), ExponentialMetric)

    def test_make_gr(self):
        g = make_metric("gr")
        assert isinstance(g, GRSchwarzschildIsotropic)

    def test_make_schw_alias(self):
        assert isinstance(make_metric("schw"), GRSchwarzschildIsotropic)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            make_metric("nonsense_metric")

    def test_factory_round_trip(self):
        for name, expected_cls in [("exp", ExponentialMetric),
                                    ("gr", GRSchwarzschildIsotropic)]:
            g = make_metric(name)
            assert isinstance(g, expected_cls)

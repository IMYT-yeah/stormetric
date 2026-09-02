"""Tests for the EMRI module."""

import math

import numpy as np
import pytest

from stormetric import EMRIPrecession, ExponentialMetric, generate_waveform_template


class TestPrecession:

    def test_linear_scale_law(self):
        g = ExponentialMetric()
        emri = EMRIPrecession(g, coupling=0.30)
        assert emri.precession_difference(0.10) == pytest.approx(0.030, rel=1e-6)
        assert emri.precession_difference(0.05) == pytest.approx(0.015, rel=1e-6)

    def test_gr_baseline_6pi(self):
        g = ExponentialMetric()
        emri = EMRIPrecession(g, coupling=0.30)
        # Δφ_GR = 6π · x_p
        assert emri.gr_precession(0.10) == pytest.approx(6.0 * math.pi * 0.10, rel=1e-9)

    def test_total_precession(self):
        g = ExponentialMetric()
        emri = EMRIPrecession(g, coupling=0.30)
        # Δφ_total = 6π x_p + κ x_p = (6π + κ) x_p
        assert emri.total_precession(0.10) == pytest.approx(
            (6.0 * math.pi + 0.30) * 0.10, rel=1e-9
        )

    def test_relative_deviation_kappa_over_6pi(self):
        g = ExponentialMetric()
        emri = EMRIPrecession(g, coupling=0.30)
        # κ / (6π + κ) ≈ 1.59%  (≈ κ / 6π for small κ)
        expected = 0.30 / (6.0 * math.pi + 0.30)
        assert emri.relative_deviation(0.10) == pytest.approx(expected, rel=1e-9)
        assert emri.relative_deviation(0.10) == pytest.approx(0.0159, abs=1e-3)

    def test_cumulative(self):
        g = ExponentialMetric()
        emri = EMRIPrecession(g)
        assert emri.cumulative_precession(0.10, 100) == pytest.approx(3.0, rel=1e-9)


class TestDetectability:

    def test_n_orbits_to_dephase(self):
        g = ExponentialMetric()
        emri = EMRIPrecession(g, coupling=0.30)
        # N = 1 / (κ x_p) at threshold 1 rad
        assert emri.n_orbits_to_dephase(0.10) == pytest.approx(
            1.0 / (0.30 * 0.10), rel=1e-9
        )

    def test_orbital_period_geometric(self):
        g = ExponentialMetric()
        emri = EMRIPrecession(g)
        # T = 2π a^{3/2} with a = 1/x_p (for e=0)
        assert emri.orbital_period_geometric(0.10) == pytest.approx(
            2.0 * math.pi * (1.0 / 0.10) ** 1.5, rel=1e-9
        )

    def test_t_obs_dephase_seconds_positive(self):
        g = ExponentialMetric()
        emri = EMRIPrecession(g)
        t = emri.t_obs_to_dephase_seconds(0.10, mass_solar=4e6)
        assert t > 0
        # Sgr A* ~ 4e6 M_sun, x_p=0.10: should be hours-to-days
        assert 100 < t < 1e8  # 100 s < t < ~3 years

    def test_compute_returns_dataclass(self):
        g = ExponentialMetric()
        emri = EMRIPrecession(g)
        res = emri.compute(x_p=0.10, mass_solar=4e6)
        assert res.x_p == 0.10
        assert res.kappa == 0.30
        assert res.relative_deviation == pytest.approx(0.30 / (6.0 * math.pi + 0.30))


class TestWaveform:

    def test_generate_template_shape(self):
        wf = generate_waveform_template(duration=100.0, x_p=0.10)
        assert len(wf["t"]) == len(wf["h_plus"]) == len(wf["h_cross"])
        assert wf["coupling"] == 0.30
        assert wf["x_p"] == 0.10

    def test_delta_phi_grows_linearly(self):
        wf = generate_waveform_template(duration=1000.0, x_p=0.10)
        assert wf["delta_phi"][-1] > wf["delta_phi"][0]

    def test_phase_total_equals_gr_plus_delta(self):
        wf = generate_waveform_template(duration=100.0, x_p=0.10)
        assert np.allclose(wf["phase_total"], wf["phase_gr"] + wf["delta_phi"])


class TestInvalidInputs:

    def test_coupling_out_of_range_raises(self):
        g = ExponentialMetric()
        with pytest.raises(ValueError):
            EMRIPrecession(g, coupling=1.5)
        with pytest.raises(ValueError):
            EMRIPrecession(g, coupling=-0.1)

    def test_xp_out_of_range_raises(self):
        g = ExponentialMetric()
        emri = EMRIPrecession(g)
        with pytest.raises(ValueError):
            emri.precession_difference(-0.1)
        with pytest.raises(ValueError):
            emri.precession_difference(1.5)

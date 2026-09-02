"""
Stormetric — EMRI pericenter precession and waveform templates.

In General Relativity, a bound orbit around a Schwarzschild mass
exhibits a perihelion advance per orbit

.. math::

    \\Delta\\varphi_{\\rm GR} = 6\\pi x_p, \\quad x_p = \\frac{GM}{c^2 r_p},

to leading post-Newtonian order (here ``r_p`` is the pericenter
distance and :math:`x_p` the dimensionless pericenter parameter
typical of EMRIs: 0.05 – 0.15).

The storm-flow framework postulates an *additional* linear
contribution

.. math::

    \\delta = \\kappa\\, x_p, \\quad \\kappa = 0.30,

so the total per-orbit precession is
:math:`\\Delta\\varphi_{\\rm total} = (6\\pi + \\kappa)\\,x_p`
and the *relative* deviation from GR is

.. math::

    \\frac{\\delta}{\\Delta\\varphi_{\\rm GR}} = \\frac{\\kappa}{6\\pi} \\approx 1.6\\%.

This module exposes:

* :class:`EMRIPrecession` — turn :math:`x_p` and mission parameters
  into (i) per-orbit phase, (ii) cumulative precession, and
  (iii) **dephasing time** (the observation time required for the
  framework deviation to accumulate a 1 rad phase shift, the
  LISA matched-filter detection threshold).
* :func:`generate_waveform_template` — a mock chirp waveform that
  encodes both the GR baseline and the framework phase correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metric import Metric


# ─────────────────────────────────────────────────────────────────────
# Detection thresholds & GR baseline
# ─────────────────────────────────────────────────────────────────────
GR_PRECESSION_COEFF = 6.0 * math.pi  # 6π — leading-PN Schwarzschild perihelion
LISA_DEPHASING_THRESHOLD_RAD = 1.0    # standard detection criterion (rad)


# ─────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────
@dataclass
class EMRIResult:
    """Container for an EMRI precession / detectability calculation."""
    x_p: float
    kappa: float
    delta_per_orbit: float        # κ·x_p
    gr_precession_per_orbit: float  # 6π·x_p
    relative_deviation: float     # κ / (6π + κ)
    n_orbits_to_dephase: float    # 1/(κ·x_p) orbits
    period_geometric: float       # orbital period in geometric units
    t_obs_to_dephase_geometric: float  # geometric time to accumulate 1 rad
    dephased: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class EMRIPrecession:
    """EMRI pericenter precession in the exponential (storm-flow) metric.

    Parameters
    ----------
    metric : Metric
        A :class:`Metric` instance (kept for interface symmetry, the
        precession law itself is parametrised by ``kappa``).
    coupling : float
        Phenomenological coupling :math:`\\kappa` (default 0.30).
    """

    def __init__(self, metric: Metric, coupling: float = 0.30) -> None:
        if not 0.0 < coupling < 1.0:
            raise ValueError(
                f"coupling κ must be in (0, 1), got {coupling}"
            )
        self.metric = metric
        self.coupling = float(coupling)

    # ── per-orbit quantities ──────────────────────────────────────
    def precession_difference(self, x_p):
        """Framework-specific per-orbit extra precession δ = κ·x_p.

        Parameters
        ----------
        x_p : float or array-like
            Dimensionless pericenter parameter :math:`x_p = GM/(c² r_p)`.
            Must lie in (0, 1).
        """
        x_p = np.asarray(x_p, dtype=float)
        if np.any(x_p <= 0) or np.any(x_p >= 1):
            raise ValueError(f"x_p must be in (0, 1), got {x_p}")
        return self.coupling * x_p

    def gr_precession(self, x_p):
        """Leading-PN GR baseline per orbit: :math:`6\\pi x_p`."""
        x_p = np.asarray(x_p, dtype=float)
        return GR_PRECESSION_COEFF * x_p

    def total_precession(self, x_p):
        """Framework total: GR baseline + framework deviation."""
        return self.gr_precession(x_p) + self.precession_difference(x_p)

    def cumulative_precession(self, x_p: float, n_passages: int) -> float:
        """Cumulative extra precession after ``n_passages`` orbits.

        Useful for matched-filter dephasing budgets.
        """
        return float(n_passages * self.precession_difference(x_p))

    def relative_deviation(self, x_p):
        """Fractional deviation from GR: :math:`\\kappa / (6\\pi + \\kappa)`.

        This is the falsifiable *relative* number.  For κ=0.30 it
        equals 1.59%.
        """
        return self.coupling / (GR_PRECESSION_COEFF + self.coupling)

    # ── detectability (LISA) ──────────────────────────────────────
    def n_orbits_to_dephase(self, x_p: float, threshold: float = LISA_DEPHASING_THRESHOLD_RAD) -> float:
        """Number of orbits required to accumulate ``threshold`` radians
        of extra precession (default 1 rad, the LISA detection threshold).
        """
        delta = self.precession_difference(x_p)
        return threshold / delta

    def orbital_period_geometric(self, x_p: float, eccentricity: float = 0.0) -> float:
        """Orbital period in geometric units (GM = c = 1).

        For a Kepler orbit,
        :math:`T = 2\\pi a^{3/2}` with :math:`a = r_p/(1-e)`.
        """
        a = (1.0 / x_p) / max(1.0 - eccentricity, 1e-12)
        return 2.0 * math.pi * a ** 1.5

    def t_obs_to_dephase_geometric(
        self, x_p: float, eccentricity: float = 0.0,
        threshold: float = LISA_DEPHASING_THRESHOLD_RAD,
    ) -> float:
        """Observation time in geometric units to accumulate
        ``threshold`` rad of extra precession."""
        n = self.n_orbits_to_dephase(x_p, threshold=threshold)
        T = self.orbital_period_geometric(x_p, eccentricity=eccentricity)
        return n * T

    def t_obs_to_dephase_seconds(
        self, x_p: float, mass_solar: float = 1e6,
        eccentricity: float = 0.0,
        threshold: float = LISA_DEPHASING_THRESHOLD_RAD,
    ) -> float:
        """Same as :meth:`t_obs_to_dephase_geometric` but in seconds,
        given the central mass in solar masses.

        Useful for Sgr A* (``mass_solar ≈ 4e6``) or M87* (``mass_solar ≈ 6.5e9``).
        """
        GM_c3_sun = 4.925490947e-6  # GM_sun / c^3 in seconds
        gm_c3 = mass_solar * GM_c3_sun
        return self.t_obs_to_dephase_geometric(
            x_p, eccentricity=eccentricity, threshold=threshold
        ) * gm_c3

    # ── full result ───────────────────────────────────────────────
    def compute(self, x_p: float, mass_solar: float = 1e6,
                eccentricity: float = 0.0) -> EMRIResult:
        """Return a :class:`EMRIResult` with all the observables."""
        delta = float(self.precession_difference(x_p))
        gr = float(self.gr_precession(x_p))
        return EMRIResult(
            x_p=float(x_p),
            kappa=self.coupling,
            delta_per_orbit=delta,
            gr_precession_per_orbit=gr,
            relative_deviation=self.relative_deviation(x_p),
            n_orbits_to_dephase=self.n_orbits_to_dephase(x_p),
            period_geometric=self.orbital_period_geometric(
                x_p, eccentricity=eccentricity
            ),
            t_obs_to_dephase_geometric=self.t_obs_to_dephase_geometric(
                x_p, eccentricity=eccentricity
            ),
            dephased=(
                self.t_obs_to_dephase_seconds(
                    x_p, mass_solar=mass_solar, eccentricity=eccentricity
                )
                < 1.0 * 365.25 * 86400.0  # within 1 year
            ),
        )


# ─────────────────────────────────────────────────────────────────────
# Waveform template
# ─────────────────────────────────────────────────────────────────────
def generate_waveform_template(
    duration: float = 86400.0,
    sampling_rate: float = 1.0,
    x_p: float = 0.10,
    coupling: float = 0.30,
    metric: Optional[Metric] = None,
    mass_solar: float = 1e6,
    f0_mHz: float = 1.0,
    fdot_mHz_per_s: float = 1e-9,
) -> dict:
    """Generate a mock EMRI waveform with the framework phase correction.

    The carrier is a linear-chirp sinusoid (approximate inspiral):

    .. math::

        f(t) = f_0 + \\dot f\\, t,
        \\quad \\Phi(t) = 2\\pi \\int_0^t f(t')\\,dt'
             = 2\\pi f_0 t + \\pi \\dot f\\, t^2.

    The framework deviation :math:`\\delta = \\kappa\\, x_p` is added as
    a per-orbit phase shift, which in time reads

    .. math::

        \\Delta\\Phi(t) = \\delta \\cdot \\frac{t}{T_{\\rm orb}(x_p)}.

    Returns
    -------
    dict with keys:
      - t            : time array (s)
      - h_plus       : + polarization
      - h_cross      : × polarization
      - phase_total  : total phase (rad)
      - phase_gr     : GR-only phase
      - delta_phi    : framework extra phase (rad)
      - coupling     : κ
      - x_p          : pericenter parameter
    """
    if metric is None:
        from .metric import ExponentialMetric
        metric = ExponentialMetric()

    emri = EMRIPrecession(metric, coupling=coupling)
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    # GR baseline chirp
    f0 = f0_mHz * 1e-3           # Hz
    fdot = fdot_mHz_per_s * 1e-3  # Hz/s
    phase_gr = 2.0 * np.pi * f0 * t + np.pi * fdot * t ** 2

    # Framework extra phase: δ · (t / T_orbit)
    delta = emri.precession_difference(x_p)
    T = emri.orbital_period_geometric(x_p)
    # Convert T from geometric to seconds
    GM_c3_sun = 4.925490947e-6
    T_sec = T * mass_solar * GM_c3_sun
    delta_phi = delta * (t / T_sec)

    phase_total = phase_gr + delta_phi
    h_plus = np.sin(phase_total)
    h_cross = np.cos(phase_total)
    return {
        "t": t,
        "h_plus": h_plus,
        "h_cross": h_cross,
        "phase_total": phase_total,
        "phase_gr": phase_gr,
        "delta_phi": delta_phi,
        "coupling": coupling,
        "x_p": x_p,
        "f0_mHz": f0_mHz,
        "fdot_mHz_per_s": fdot_mHz_per_s,
    }

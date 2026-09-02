"""
Stormetric — Storm-Flow Relativistic Metrics
=============================================

A human-AI co-designed Python package for testing alternative
relativistic metrics against EMRI waveforms and black-hole shadow
observations.

Modules
-------
* :mod:`metric` — :class:`Metric`, :class:`ExponentialMetric`,
  :class:`GRSchwarzschildIsotropic`
* :mod:`ppn` — PPN expansion and 3PN bifurcation analysis
* :mod:`shadow` — photon sphere and shadow radius (analytic + numeric)
* :mod:`emri` — pericenter precession and waveform templates

Command-line entry point
------------------------
After ``pip install stormetric`` you can run ``stormetric`` to
inspect shadow, PPN coefficients, EMRI precession, etc::

    stormetric shadow
    stormetric ppn --metric exp
    stormetric emri --xp 0.10 --mass 4e6
    stormetric plot --output docs/stormetric_results.png
"""

from __future__ import annotations

__version__ = "0.2.0"

from .metric import (
    ExponentialMetric,
    GRSchwarzschildIsotropic,
    Metric,
    make_metric,
)
from .ppn import (
    PPNRow,
    first_bifurcation_order,
    leading_residual_coefficient,
    ppn_rows,
    ppn_table,
    truncated_g00,
    truncated_grr,
)
from .shadow import Shadow, ShadowResult
from .emri import (
    EMRIResult,
    EMRIPrecession,
    generate_waveform_template,
)

__all__ = [
    "ExponentialMetric",
    "GRSchwarzschildIsotropic",
    "Metric",
    "make_metric",
    "PPNRow",
    "first_bifurcation_order",
    "leading_residual_coefficient",
    "ppn_rows",
    "ppn_table",
    "truncated_g00",
    "truncated_grr",
    "Shadow",
    "ShadowResult",
    "EMRIResult",
    "EMRIPrecession",
    "generate_waveform_template",
]

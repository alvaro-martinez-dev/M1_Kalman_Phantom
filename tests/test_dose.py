"""Validation of src/phantom/dose.py."""
import numpy as np
import pytest

from src.constants import WATER
from src.phantom.materials import bethe_bloch, range_csda
from src.phantom.dose import (
    dose_per_length, dose_in_step, bragg_peak_depth, MEV_PER_G_TO_GY,
)


def test_dose_per_length_units():
    dEdx = bethe_bloch(200.0, WATER)
    dose = dose_per_length(200.0, WATER)
    assert np.isclose(dose, dEdx / WATER.density)


def test_dose_in_step_equals_rate():
    step = 1.0
    D1 = dose_in_step(200.0, step, WATER)
    D2 = dose_per_length(200.0, WATER) * step * MEV_PER_G_TO_GY
    assert np.isclose(D1, D2, rtol=1e-12)


def test_dose_order_of_magnitude():
    """200 MeV proton, 1 mm water -> ~1e-9 Gy."""
    D = dose_in_step(200.0, 1.0, WATER)
    assert 1e-10 < D < 1e-8


def test_bragg_peak_at_range():
    assert np.isclose(bragg_peak_depth(200.0, WATER),
                      range_csda(200.0, WATER))
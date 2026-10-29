
"""Validation of materials.py against NIST PSTAR and analytic checks."""
import numpy as np
import pytest
from src.materials import WATER, BONE, bethe_bloch, range_csda, highland_width


def test_bethe_bloch_water_200_MeV():
    """NIST PSTAR: dE/dx ≈ 4.465 MeV/cm at 200 MeV in water."""
    dEdx = bethe_bloch(200.0, WATER)
    assert 4.0 < dEdx < 5.0, f"dE/dx = {dEdx:.3f} MeV/cm, expected ~4.47"


def test_range_water_200_MeV():
    """NIST PSTAR: CSDA range ≈ 25.96 cm at 200 MeV in water."""
    r = range_csda(200.0, WATER)
    assert 24.0 < r < 28.0, f"range = {r:.2f} cm, expected ~25.96"


def test_bethe_bloch_grows_at_low_energy():
    """dE/dx must increase as the proton slows down."""
    assert bethe_bloch(50.0, WATER) > bethe_bloch(200.0, WATER)


def test_bone_stops_faster_than_water():
    """Bone is denser, so the range in bone is shorter."""
    assert range_csda(200.0, BONE) < range_csda(200.0, WATER)


def test_highland_width_200_MeV_water():
    """At 200 MeV, 10 cm of water: theta_0 ≈ 18-20 mrad."""
    theta = highland_width(200.0, 10.0, WATER)
    assert 0.010 < theta < 0.030, f"theta_0 = {theta*1000:.1f} mrad"

"""Validation of materials.py against NIST PSTAR and analytic checks.

Units: energy MeV, length mm.
"""
import pytest
from src.materials import bethe_bloch, range_csda, highland_width
from src.constants import WATER, BONE


def test_bethe_bloch_water_200_MeV():
    """NIST PSTAR: dE/dx ≈ 0.4465 MeV/mm at 200 MeV in water (4.465 MeV/cm)."""
    dEdx = bethe_bloch(200.0, WATER)
    assert 0.4 < dEdx < 0.5, f"dE/dx = {dEdx:.4f} MeV/mm, expected ~0.4465"


def test_range_water_200_MeV():
    """NIST PSTAR: CSDA range ≈ 259.6 mm at 200 MeV in water."""
    r = range_csda(200.0, WATER)
    assert 240.0 < r < 280.0, f"range = {r:.2f} mm, expected ~259.6"


def test_bethe_bloch_grows_at_low_energy():
    """dE/dx must increase as the proton slows down."""
    assert bethe_bloch(50.0, WATER) > bethe_bloch(200.0, WATER)


def test_bone_stops_faster_than_water():
    """Bone is denser, so the range in bone is shorter."""
    assert range_csda(200.0, BONE) < range_csda(200.0, WATER)


def test_highland_width_200_MeV_water():
    """At 200 MeV, 100 mm of water: theta_0 ≈ 18-20 mrad."""
    theta = highland_width(200.0, 100.0, WATER)
    assert 0.010 < theta < 0.030, f"theta_0 = {theta*1000:.1f} mrad"
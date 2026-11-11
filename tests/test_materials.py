"""Validation of src/materials.py against NIST PSTAR and analytic checks.

Units: energy MeV, length mm.
"""
import numpy as np
import pytest
from src.phantom.materials import density_effect, beta_gamma
from src.constants import (
    WATER, BONE, PROTON, MU_MINUS, ELECTRON, ALPHA,
)
from src.phantom.materials import (
    bethe_bloch, range_csda, range_csda_fast, highland_width, E_MIN_MEV,
)


@pytest.mark.parametrize("material, particle", [(WATER, PROTON), (BONE, PROTON), (WATER, MU_MINUS)])
@pytest.mark.parametrize("T", [0.5, 3.0, 50.0, 200.0, 1234.5, 5.0e4])
def test_range_table_matches_the_integral(material, particle, T):
    assert range_csda_fast(T, material, particle) == pytest.approx(range_csda(T, material, particle), rel=2e-6)


def test_range_table_outside_its_limits():
    assert range_csda_fast(0.05, WATER) == 0.0
    assert range_csda_fast(2.0e5, WATER) == range_csda(2.0e5, WATER)     # falls back to the integral


# ---------------------------------------------------------------
# Reference values: NIST PSTAR, protons in water
# ---------------------------------------------------------------

def test_bethe_bloch_water_200_MeV():
    """PSTAR: dE/dx = 4.465 MeV/cm = 0.4465 MeV/mm at 200 MeV."""
    dEdx = bethe_bloch(200.0, WATER)
    assert 0.442 < dEdx < 0.451, f"dE/dx = {dEdx:.4f} MeV/mm"


def test_bethe_bloch_water_100_MeV():
    """PSTAR: dE/dx = 7.299 MeV/cm = 0.7299 MeV/mm at 100 MeV."""
    dEdx = bethe_bloch(100.0, WATER)
    assert 0.720 < dEdx < 0.740, f"dE/dx = {dEdx:.4f} MeV/mm"


def test_range_water_200_MeV():
    """PSTAR: CSDA range = 259.6 mm at 200 MeV."""
    r = range_csda(200.0, WATER)
    assert 254.0 < r < 265.0, f"range = {r:.2f} mm"


def test_range_water_100_MeV():
    """PSTAR: CSDA range = 77.1 mm at 100 MeV."""
    r = range_csda(100.0, WATER)
    assert 75.5 < r < 78.5, f"range = {r:.2f} mm"


# ---------------------------------------------------------------
# Physical trends
# ---------------------------------------------------------------

def test_bethe_bloch_grows_at_low_energy():
    """dE/dx increases as the particle slows down (Bragg peak)."""
    assert bethe_bloch(50.0, WATER) > bethe_bloch(200.0, WATER)


def test_bone_stops_faster_than_water():
    """Bone is denser: shorter range."""
    assert range_csda(200.0, BONE) < range_csda(200.0, WATER)


# ---------------------------------------------------------------
# Generalization to other particles
# ---------------------------------------------------------------

def test_bethe_bloch_alpha_is_4x_proton():
    """Alpha has z=+2; at the same beta, dE/dx is 4x the proton's."""
    T_alpha = 200.0 * (ALPHA.mass / PROTON.mass)  # same kinetic energy per nucleon
    dEdx_p = bethe_bloch(200.0, WATER, particle=PROTON)
    dEdx_a = bethe_bloch(T_alpha, WATER, particle=ALPHA)
    ratio = dEdx_a / dEdx_p
    assert 3.8 < ratio < 4.2, f"alpha/proton ratio = {ratio:.3f}"


def test_highland_negative_charge_positive_width():
    """Highland uses |z|: negative particle gives positive width."""
    theta = highland_width(200.0, 100.0, WATER, particle=MU_MINUS)
    assert theta > 0


def test_electron_raises():
    """Bethe-Bloch is not valid for electrons."""
    with pytest.raises(NotImplementedError):
        bethe_bloch(100.0, WATER, particle=ELECTRON)


# ---------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------

def test_below_E_min_raises():
    """Bethe-Bloch not valid below E_MIN."""
    with pytest.raises(ValueError):
        bethe_bloch(0.05, WATER)


def test_highland_zero_thickness_raises():
    """Zero thickness: log(0) is undefined."""
    with pytest.raises(ValueError):
        highland_width(200.0, 0.0, WATER)


def test_highland_negative_thickness_raises():
    with pytest.raises(ValueError):
        highland_width(200.0, -1.0, WATER)


# ---------------------------------------------------------------
# Highland additivity
# ---------------------------------------------------------------

def test_highland_width_200_MeV_water():
    """At 200 MeV, 100 mm water: theta_0 ~ 20 mrad."""
    theta = highland_width(200.0, 100.0, WATER)
    assert 0.010 < theta < 0.030, f"theta_0 = {theta*1000:.1f} mrad"


def test_highland_additivity():
    """n_slices kicks add in quadrature to the single-layer width."""
    x = 100.0
    n = 10
    theta_total = highland_width(200.0, x, WATER, n_slices=1)
    theta_step = highland_width(200.0, x, WATER, n_slices=n)
    combined = np.sqrt(n) * theta_step
    assert np.isclose(combined, theta_total, rtol=1e-12)


def test_density_effect_zero_at_low_energy():
    """At 200 MeV in water, beta*gamma < x0 -> delta = 0."""
    beta, gamma = beta_gamma(200.0)
    delta = density_effect(beta * gamma, WATER)
    assert delta == 0.0


def test_density_effect_positive_at_high_energy():
    """For beta*gamma > x1, delta must be > 0 (high-energy limit)."""
    delta = density_effect(1000.0, WATER)
    assert delta > 0.0


def test_bethe_bloch_unchanged_at_200_MeV_with_delta():
    """At 200 MeV, delta = 0: dE/dx unchanged."""
    dEdx = bethe_bloch(200.0, WATER)
    assert 0.442 < dEdx < 0.451
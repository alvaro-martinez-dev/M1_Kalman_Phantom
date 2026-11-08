"""Validation of src/phantom/dose.py and Validation of WEPL and dose profile against the Geometry class."""
import numpy as np
import pytest


from src.constants import WATER, BONE
from src.geometry.geometry import Geometry
from src.phantom.materials import material_object
from src.phantom.wepl import wepl_density, wepl_stopping_power
from src.phantom.dose import dose_profile
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
    
    


@pytest.fixture
def layered_phantom():
    """WATER 200 mm + BONE 100 mm, magnet 500 mm after phantom start."""
    return Geometry(
        z_start_phantom=100.0,
        phantom_layers=[
            {"material": "WATER", "thickness": 200.0},
            {"material": "BONE",  "thickness": 100.0},
        ],
        z_start_magnet=500.0,
        z_length_magnet=300.0,
        spacing_after_magnet=200.0,
    )


def test_material_object_water():
    assert material_object("WATER") is WATER


def test_material_object_bone():
    assert material_object("BONE") is BONE


def test_material_object_unknown_raises():
    with pytest.raises(ValueError, match="Unknown material"):
        material_object("UNOBTAINIUM")


def test_wepl_density_layered(layered_phantom):
    # 200*1.0 + 100*1.85 = 200 + 185 = 385 mm water-equivalent
    assert abs(wepl_density(layered_phantom) - 385.0) < 1e-9


def test_wepl_stopping_power_close_to_density(layered_phantom):
    wepl_d = wepl_density(layered_phantom)
    wepl_s = wepl_stopping_power(layered_phantom, T_MeV=200.0)
    # At 200 MeV, the two estimates should be close (not identical)
    assert abs(wepl_s - wepl_d) / wepl_d < 0.15


def test_dose_profile_monotonic_energy(layered_phantom):
    z_arr, T_arr, D_arr = dose_profile(layered_phantom, T0_MeV=200.0, n_steps=200)
    assert np.all(np.diff(T_arr) <= 1e-9)


def test_dose_profile_cumulative_positive(layered_phantom):
    z_arr, T_arr, D_arr = dose_profile(layered_phantom, T0_MeV=200.0, n_steps=200)
    assert np.all(D_arr >= 0)
    assert np.all(np.diff(D_arr) >= -1e-15)
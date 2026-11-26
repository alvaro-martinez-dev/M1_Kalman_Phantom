"""Validation of src/phantom/dose.py and Validation of WEPL and dose profile against the Geometry class."""
import numpy as np
import pytest


from src.constants import WATER, BONE, PROTON
from src.geometry.geometry import Geometry
from src.phantom.materials import material_object
from src.phantom.wepl import wepl_density, wepl_stopping_power
from src.phantom.dose import dose_profile
from src.phantom.materials import bethe_bloch, range_csda
from src.phantom.dose import (
    dose_per_length, dose_in_step, bragg_peak_depth, MEV_PER_G_TO_GY,
)


def test_dose_per_length_units():
    dEdx = bethe_bloch(200.0, WATER, PROTON)
    dose = dose_per_length(200.0, WATER, PROTON)
    assert np.isclose(dose, dEdx / WATER.density)


def test_dose_in_step_equals_rate():
    step = 1.0
    D1 = dose_in_step(200.0, step, WATER, PROTON)
    D2 = dose_per_length(200.0, WATER, PROTON) * step * MEV_PER_G_TO_GY
    assert np.isclose(D1, D2, rtol=1e-12)


def test_dose_order_of_magnitude():
    """200 MeV proton, 1 mm water -> ~1e-9 Gy."""
    D = dose_in_step(200.0, 1.0, WATER, PROTON)
    assert 1e-10 < D < 1e-8


def test_bragg_peak_at_range():
    assert np.isclose(bragg_peak_depth(200.0, WATER, PROTON),
                      range_csda(200.0, WATER, PROTON))
    
    


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
        spacing_after_last=200.0,
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
    wepl_s = wepl_stopping_power(layered_phantom, T_in_MeV=200.0, particle=PROTON)
    # At 200 MeV, the two estimates should be close (not identical)
    assert abs(wepl_s - wepl_d) / wepl_d < 0.15


def test_dose_profile_monotonic_energy(layered_phantom):
    z_arr, T_arr, D_arr = dose_profile(layered_phantom, T0_MeV=200.0, particle=PROTON, n_steps=200)
    assert np.all(np.diff(T_arr) <= 1e-9)


def test_dose_profile_is_a_bragg_curve(layered_phantom):
    """The 200 MeV proton stops in the phantom (range ~260 mm of water < WEPL ~385 mm): the local
    dose is >= 0, grows inside each material up to the Bragg peak and is 0 after the stop.
    It is not monotonic overall: it drops after the peak, and also at the water/bone interface
    (dose is per unit mass, and bone has lower Z/A than water)."""
    z_arr, T_arr, D_arr = dose_profile(layered_phantom, T0_MeV=200.0, particle=PROTON, n_steps=200)
    assert np.all(D_arr >= 0)

    stopped = T_arr == 0.0
    assert stopped.any() and np.all(D_arr[stopped] == 0.0)
    i_peak = int(np.argmax(D_arr))
    # the peak is in the step where it stops or the one before: the step of the stop is only partly
    # crossed, so its MEAN dose can be below that of the previous (full) step
    i_stop = np.flatnonzero(~stopped)[-1]
    assert i_stop - 1 <= i_peak <= i_stop

    # before the peak, the dose grows inside each material. Steps that cross an interface are left
    # out: their dose is the mean over both materials (per unit mass, and bone has lower Z/A)
    start = np.array([layered_phantom.material_at(z) for z in z_arr[:i_peak + 1]])
    end = np.array([layered_phantom.material_at(z - 1e-6) for z in z_arr[1:i_peak + 2]])
    pure = start == end
    same_material = pure[1:] & pure[:-1] & (start[1:] == start[:-1])
    assert same_material.sum() > 100
    assert np.all(np.diff(D_arr[:i_peak + 1])[same_material] > 0)


def mass_per_area(geometry, z_arr):
    # [g/cm^2] of each step of a dose_profile grid (no step crosses an interface in these tests)
    return np.array([material_object(geometry.material_at(0.5 * (a + b))).density * (b - a) * 0.1
                     for a, b in zip(z_arr[:-1], z_arr[1:])])


def test_dose_profile_conserves_energy(layered_phantom):
    """A proton that stops deposits all its energy: sum of dose x mass of the steps = T0."""
    z_arr, T_arr, D_arr = dose_profile(layered_phantom, T0_MeV=200.0, particle=PROTON, n_steps=300)
    deposited = np.sum(D_arr[:-1] / MEV_PER_G_TO_GY * mass_per_area(layered_phantom, z_arr))
    assert deposited == pytest.approx(200.0, rel=1e-9)


def test_dose_profile_energy_does_not_depend_on_the_step(layered_phantom):
    """The energy at a given depth is exact (range inversion), not an Euler integration: the same
    for any number of steps, and equal to the energy of the CSDA residual range."""
    z_a, T_a, _ = dose_profile(layered_phantom, T0_MeV=200.0, particle=PROTON, n_steps=30)
    z_b, T_b, _ = dose_profile(layered_phantom, T0_MeV=200.0, particle=PROTON, n_steps=300)
    assert T_b[::10] == pytest.approx(T_a, rel=1e-9, abs=1e-9)
    # in the first (water) layer: range(T(z)) = range(T0) - depth
    i = 5
    depth = z_a[i] - layered_phantom.z_start_phantom
    assert range_csda(T_a[i], WATER, PROTON) == pytest.approx(range_csda(200.0, WATER, PROTON) - depth, rel=1e-6)


def test_dose_profile_stops_at_the_csda_range():
    """In pure water the proton stops at its CSDA range (here 77.1 mm at 100 MeV, PSTAR)."""
    water = Geometry(z_start_phantom=0.0, phantom_layers=[{"material": "WATER", "thickness": 150.0}])
    z_arr, T_arr, _ = dose_profile(water, T0_MeV=100.0, particle=PROTON, n_steps=1500)
    z_stop = z_arr[np.flatnonzero(T_arr > 0)[-1]]
    assert z_stop == pytest.approx(range_csda(100.0, WATER, PROTON), abs=0.1)


def test_energy_from_range_inverts_the_range():
    from src.phantom.dose import energy_from_range
    for T in (5.0, 70.0, 230.0):
        R = range_csda(T, WATER, PROTON)
        assert energy_from_range(R, WATER, PROTON, T_hi=300.0) == pytest.approx(T, rel=1e-6)
    assert energy_from_range(0.0, WATER, PROTON, T_hi=300.0) == 0.0


def test_radiation_weighting_factors():
    """ICRP 103: fixed values, and the neutron curve (2.5 when thermal, peak ~20.7 near 1 MeV,
    2.84 at 1 GeV, tending to 2.5 at high energy)."""
    from src.phantom.dose import equivalent_dose, radiation_weighting_factor
    assert radiation_weighting_factor("photon") == 1.0
    assert radiation_weighting_factor("proton") == 2.0
    assert radiation_weighting_factor("alpha") == 20.0
    assert radiation_weighting_factor("neutron", 1.0) == pytest.approx(5.0 + 17.0 * np.exp(-np.log(2.0)**2 / 6))
    assert radiation_weighting_factor("neutron", 1e-8) == pytest.approx(2.5, abs=0.1)     # thermal
    assert radiation_weighting_factor("neutron", 1000.0) == pytest.approx(2.836, abs=0.005)
    assert max(radiation_weighting_factor("neutron", E) for E in np.geomspace(0.1, 10, 200)) == pytest.approx(20.7, abs=0.1)
    assert equivalent_dose(1e-3, "neutron", 1.0) == pytest.approx(1e-3 * radiation_weighting_factor("neutron", 1.0))
    with pytest.raises(ValueError):
        radiation_weighting_factor("neutron")


def test_check_dose_limits_categories():
    from src.phantom.dose import check_dose_limits
    assert check_dose_limits(0.5e-3) == "public"
    assert check_dose_limits(5e-3) == "occupational"
    assert check_dose_limits(50e-3) == "above"
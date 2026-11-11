# tests/test_scattering.py
"""Multiple scattering per slice: Highland with local momentum, (dx, dtx) covariance and sampling."""
import numpy as np
import pytest

from src.constants import BONE, PROTON, WATER
from src.geometry.geometry import Geometry
from src.phantom.materials import highland_width
from src.propagator.propagator import momentum
from src.propagator.scattering import (phantom_slices, sample_kick, slice_covariance,
                                       theta0_slice)


def q_p(T_MeV):
    return PROTON.charge / momentum(T_MeV, PROTON)


# ----- theta0 ----------------------------------------------------------------------------------

def test_single_slice_is_the_highland_width_of_the_layer():
    assert theta0_slice(q_p(200.0), 0.0, 100.0, 100.0, WATER) == pytest.approx(
        highland_width(200.0, 100.0, WATER), rel=1e-12)


@pytest.mark.parametrize("n", [2, 10, 100])
def test_slices_add_up_to_the_layer_at_constant_momentum(n):
    """The log term is evaluated for the whole layer: the result does not depend on n."""
    L = 100.0
    theta_i = theta0_slice(q_p(200.0), 0.0, L / n, L, WATER)
    assert np.sqrt(n) * theta_i == pytest.approx(highland_width(200.0, L, WATER), rel=1e-12)
    assert theta_i == pytest.approx(highland_width(200.0, L, WATER, n_slices=n), rel=1e-12)


def test_slower_particle_scatters_more():
    assert theta0_slice(q_p(50.0), 0.0, 1.0, 100.0, WATER) > theta0_slice(q_p(200.0), 0.0, 1.0, 100.0, WATER)


def test_inclined_track_crosses_more_material():
    tx = 0.3
    straight = theta0_slice(q_p(200.0), 0.0, 10.0, 10.0, WATER)
    inclined = theta0_slice(q_p(200.0), tx, 10.0, 10.0, WATER)
    assert inclined == pytest.approx(highland_width(200.0, 10.0 * np.sqrt(1 + tx**2), WATER), rel=1e-12)
    assert inclined > straight


def test_denser_material_scatters_more():
    assert theta0_slice(q_p(200.0), 0.0, 1.0, 10.0, BONE) > theta0_slice(q_p(200.0), 0.0, 1.0, 10.0, WATER)


def test_bad_slice_length():
    with pytest.raises(ValueError, match="dz"):
        theta0_slice(q_p(200.0), 0.0, 20.0, 10.0, WATER)
    with pytest.raises(ValueError, match="dz"):
        theta0_slice(q_p(200.0), 0.0, 0.0, 10.0, WATER)


# ----- covariance and sampling -----------------------------------------------------------------

def test_covariance_of_a_slice():
    theta0, dz = 1e-3, 6.0
    cov = slice_covariance(theta0, 0.0, dz)
    assert cov == pytest.approx(theta0**2 * np.array([[12.0, 3.0], [3.0, 1.0]]))
    # correlation sqrt(3)/2, positive definite
    assert cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) == pytest.approx(np.sqrt(3) / 2)
    assert cov[0, 0] > 0 and cov[0, 0] * cov[1, 1] - cov[0, 1]**2 > 0   # 2x2 positive definite


def test_covariance_in_tx_grows_with_the_angle():
    tx = 0.5
    ratio = slice_covariance(1e-3, tx, 5.0) / slice_covariance(1e-3, 0.0, 5.0)
    assert ratio == pytest.approx(np.full((2, 2), (1 + tx**2)**2))


def test_sampled_kicks_have_the_slice_covariance():
    rng = np.random.default_rng(1)
    theta0, tx, dz = 2e-3, 0.2, 5.0
    kicks = np.array([sample_kick(theta0, tx, dz, rng) for _ in range(200_000)])
    assert kicks.mean(axis=0) == pytest.approx([0.0, 0.0], abs=3e-4 * np.array([dz, 1.0]).max())
    # element by element (np.cov goes through BLAS)
    dx, dtx = (kicks - kicks.mean(axis=0)).T
    sample_cov = np.array([[np.mean(dx * dx), np.mean(dx * dtx)],
                           [np.mean(dx * dtx), np.mean(dtx * dtx)]])
    assert sample_cov == pytest.approx(slice_covariance(theta0, tx, dz), rel=2e-2)


# ----- slices of the phantom -------------------------------------------------------------------

def layered_geometry():
    return Geometry(z_start_phantom=50.0,
                    phantom_layers=[{"material": "WATER", "thickness": 200.0},
                                    {"material": "BONE", "thickness": 15.0}],
                    z_start_magnet=400.0, z_length_magnet=300.0, spacing_after_last=100.0)


def test_slices_never_cross_an_interface():
    slices = phantom_slices(layered_geometry(), 10.0)
    assert len(slices) == 20 + 2                       # 200/10 and ceil(15/10) = 2 slices of 7.5
    assert slices[0] == (50.0, 60.0, "WATER", 200.0)
    assert slices[19][1] == pytest.approx(250.0) and slices[20][0] == pytest.approx(250.0)
    z_a, z_b, material, dz_layer = slices[-1]
    assert (z_a, z_b, dz_layer) == pytest.approx((257.5, 265.0, 15.0)) and material == "BONE"
    assert all(b - a <= 10.0 + 1e-9 for a, b, _, _ in slices)
    # contiguous, covering the whole phantom
    assert all(s[1] == pytest.approx(t[0]) for s, t in zip(slices[:-1], slices[1:]))


def test_slice_longer_than_the_layer_gives_one_slice_per_layer():
    assert len(phantom_slices(layered_geometry(), 1000.0)) == 2


def test_no_phantom_no_slices():
    g = Geometry(z_start_magnet=400.0, z_length_magnet=300.0, spacing_after_last=100.0)
    assert phantom_slices(g, 5.0) == []


@pytest.mark.parametrize("value", [0.0, -1.0, "5"])
def test_bad_max_slice(value):
    with pytest.raises(ValueError, match="max_slice"):
        phantom_slices(layered_geometry(), value)

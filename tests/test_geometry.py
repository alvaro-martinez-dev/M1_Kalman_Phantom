"""Input checks of src/geometry: a wrong geometry or detector placement must fail early.

Units: length mm.
"""
import pytest

from src.geometry.detectors import DetectorPlanes
from src.geometry.geometry import Geometry
from src.utils.config_loader import make_detector_planes, make_geometry


WATER_200 = [{"material": "WATER", "thickness": 200.0}]

GOOD_GEOMETRY = dict(z_start_phantom=50.0, phantom_layers=WATER_200, z_start_magnet=400.0,
                     z_length_magnet=300.0, spacing_after_magnet=100.0)

GOOD_PLACEMENT = dict(n_before_phantom=1, n_after_phantom_before_magnet=1, n_after_magnet=1,
                      inside_phantom=True, n_inside_magnet=1)


def geometry(**changes):
    return Geometry(**{**GOOD_GEOMETRY, **changes})


def planes(**changes):
    return DetectorPlanes(geometry(), **{**GOOD_PLACEMENT, **changes})


# ---------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------

def test_good_geometry():
    g = geometry()
    assert g.z_end_phantom == 250.0
    assert g.z_end_system == 800.0


def test_integers_from_the_yaml_are_accepted():
    """yaml reads 100 as int: it must work like 100.0."""
    g = geometry(spacing_after_magnet=100, z_start_phantom=50)
    assert g.z_end_system == 800.0


def test_edge_cases_are_accepted():
    """Phantom at z = 0, phantom touching the magnet, no space after the magnet, no free planes."""
    g = geometry(z_start_phantom=0.0, z_start_magnet=200.0, spacing_after_magnet=0.0)
    DetectorPlanes(g, 0, 0, 0, False, 0)


# ---------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------

@pytest.mark.parametrize("key, value", [
    ("z_start_phantom", -1.0),
    ("z_start_magnet", -1.0),
    ("z_length_magnet", 0.0),
    ("z_length_magnet", -300.0),
    ("spacing_after_magnet", -1.0),
])
def test_negative_or_zero_lengths(key, value):
    with pytest.raises(ValueError, match=key):
        geometry(**{key: value})


@pytest.mark.parametrize("value", ["50mm", None, True, float("nan"), float("inf")])
def test_not_a_number(value):
    with pytest.raises(ValueError, match="z_start_phantom"):
        geometry(z_start_phantom=value)


def test_phantom_overlapping_the_magnet():
    with pytest.raises(ValueError, match="overlap"):
        geometry(z_start_magnet=200.0)   # phantom ends at 250


@pytest.mark.parametrize("thickness", [0.0, -10.0, "10"])
def test_bad_thickness(thickness):
    with pytest.raises(ValueError, match="phantom layer 2"):
        geometry(phantom_layers=WATER_200 + [{"material": "BONE", "thickness": thickness}])


@pytest.mark.parametrize("layers", [
    [],                                                 # no layers
    None,
    [{"material": "WATER"}],                            # no thickness
    [{"material": "WATER", "thickness": 10, "x": 1}],   # unknown key (typo)
    [{"material": "WATR", "thickness": 10}],            # unknown material
    [{"material": "PROTON", "thickness": 10}],          # constant that is not a Material
])
def test_bad_layers(layers):
    with pytest.raises(ValueError, match="phantom"):
        geometry(phantom_layers=layers)


# ---------------------------------------------------------------
# Detector planes
# ---------------------------------------------------------------

@pytest.mark.parametrize("key, value", [
    ("n_before_phantom", -1),
    ("n_after_magnet", 1.5),
    ("n_inside_magnet", 2.0),
    ("n_after_phantom_before_magnet", "2"),
    ("n_inside_magnet", True),
    ("inside_phantom", 1),
    ("inside_phantom", "yes"),
])
def test_bad_placement(key, value):
    with pytest.raises(ValueError, match=key):
        planes(**{key: value})


def test_planes_need_a_geometry():
    with pytest.raises(ValueError, match="Geometry"):
        DetectorPlanes(GOOD_GEOMETRY, **GOOD_PLACEMENT)


# ---------------------------------------------------------------
# yaml blocks
# ---------------------------------------------------------------

def test_missing_key_in_the_yaml_is_named():
    cfg = {"spacing_after_magnet": 100, "phantom": {"layers": WATER_200},
           "magnet": {"z_start": 400.0, "z_length": 300.0}}
    with pytest.raises(KeyError, match="geometry.phantom.z_start"):
        make_geometry(cfg)


def test_missing_placement_key_is_named():
    placement = {k: v for k, v in GOOD_PLACEMENT.items() if k != "n_after_magnet"}
    with pytest.raises(KeyError, match="placement.n_after_magnet"):
        make_detector_planes(geometry(), {"placement": placement})

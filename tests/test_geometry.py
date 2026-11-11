"""Input checks of src/geometry: a wrong geometry or detector placement must fail early.

Units: length mm.
"""
import numpy as np
import pytest

from src.geometry.detectors import DetectorPlanes
from src.geometry.geometry import Geometry
from src.utils.config_loader import make_detector_planes, make_geometry


WATER_200 = [{"material": "WATER", "thickness": 200.0}]

GOOD_GEOMETRY = dict(z_start_phantom=50.0, phantom_layers=WATER_200, z_start_magnet=400.0,
                     z_length_magnet=300.0, spacing_after_last=100.0)

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
    g = geometry(spacing_after_last=100, z_start_phantom=50)
    assert g.z_end_system == 800.0


def test_edge_cases_are_accepted():
    """Phantom at z = 0, phantom touching the magnet, no space after the magnet, no free planes."""
    g = geometry(z_start_phantom=0.0, z_start_magnet=200.0, spacing_after_last=0.0)
    DetectorPlanes(g, 0, 0, 0, False, 0)


# ---------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------

@pytest.mark.parametrize("key, value", [
    ("z_start_phantom", -1.0),
    ("z_start_magnet", -1.0),
    ("z_length_magnet", 0.0),
    ("z_length_magnet", -300.0),
    ("spacing_after_last", -1.0),
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

def yaml_geometry(phantom=True, magnet=True):
    return {"spacing_after_last": 100,
            "phantom": {"enabled": phantom, "z_start": 50.0, "layers": WATER_200},
            "magnet": {"enabled": magnet, "z_start": 400.0, "z_length": 300.0}}


def test_missing_key_in_the_yaml_is_named():
    cfg = yaml_geometry()
    del cfg["phantom"]["z_start"]
    with pytest.raises(KeyError, match="geometry.phantom.z_start"):
        make_geometry(cfg)


def test_missing_placement_key_is_named():
    placement = {k: v for k, v in GOOD_PLACEMENT.items() if k != "n_after_magnet"}
    with pytest.raises(KeyError, match="placement.n_after_magnet"):
        make_detector_planes(geometry(), {"enabled": True, "placement": placement})


@pytest.mark.parametrize("block", ["phantom", "magnet"])
def test_missing_enabled_is_named(block):
    cfg = yaml_geometry()
    del cfg[block]["enabled"]
    with pytest.raises(KeyError, match=f"geometry.{block}.enabled"):
        make_geometry(cfg)


@pytest.mark.parametrize("value", ["false", 0, None])
def test_enabled_must_be_a_bool(value):
    """A typo (or 'no' read as a string) must not silently remove an element."""
    with pytest.raises(ValueError, match="enabled"):
        make_geometry(yaml_geometry(phantom=value))


def test_disabled_elements_from_the_yaml():
    g = make_geometry(yaml_geometry(phantom=False))
    assert not g.has_phantom and g.has_magnet
    g = make_geometry(yaml_geometry(magnet=False))
    assert g.has_phantom and not g.has_magnet


def test_disabled_detectors_from_the_yaml():
    assert make_detector_planes(geometry(), {"enabled": False}) is None


# ---------------------------------------------------------------
# Phantom and/or magnet absent
# ---------------------------------------------------------------

def test_half_given_element_is_rejected():
    with pytest.raises(ValueError, match="magnet"):
        Geometry(z_start_magnet=400.0, spacing_after_last=100.0)


def test_no_phantom():
    g = Geometry(z_start_magnet=400.0, z_length_magnet=300.0, spacing_after_last=100.0)
    assert g.z_start_phantom is None and g.z_end_phantom is None and g.phantom_layers == []
    assert g.z_end_system == 800.0
    assert g.material_at(100.0) is None
    assert [label for _, label in g.boundaries()] == ["magnet_start", "magnet_end", "system_end"]


def test_no_magnet():
    g = Geometry(z_start_phantom=50.0, phantom_layers=WATER_200, spacing_after_last=100.0)
    assert g.z_start_magnet is None and g.z_end_magnet is None
    assert g.z_end_system == 350.0               # after the phantom
    assert g.region_at(500.0).value == "outside"
    assert [label for _, label in g.boundaries()] == ["phantom_start", "phantom_end", "system_end"]


def test_neither_phantom_nor_magnet():
    g = Geometry(spacing_after_last=500.0)
    assert g.z_end_system == 500.0
    assert g.boundaries() == [(500.0, "system_end")]


def test_merged_regions_keep_the_number_of_free_planes():
    """Without phantom/magnet the regions around it merge and get the planes of both."""
    placement = dict(n_before_phantom=1, n_after_phantom_before_magnet=2, n_after_magnet=3,
                     inside_phantom=True, n_inside_magnet=1)
    full = DetectorPlanes(geometry(), **placement)
    no_phantom = DetectorPlanes(Geometry(z_start_magnet=400.0, z_length_magnet=300.0,
                                         spacing_after_last=100.0), **placement)
    no_magnet = DetectorPlanes(Geometry(z_start_phantom=50.0, phantom_layers=WATER_200,
                                        spacing_after_last=100.0), **placement)
    nothing = DetectorPlanes(Geometry(spacing_after_last=500.0), **placement)

    assert len(full.z_free_planes) == 1 + 2 + 1 + 3
    assert len(no_phantom.z_free_planes) == (1 + 2) + 1 + 3
    assert len(no_magnet.z_free_planes) == 1 + (2 + 3)      # n_inside_magnet ignored
    assert len(nothing.z_free_planes) == 1 + 2 + 3

    names = no_phantom.named_planes()
    assert list(names)[:4] == ["D_before_magnet_1", "D_before_magnet_2", "D_before_magnet_3", "D_magnet_start"]
    assert names["D_before_magnet_1"] == pytest.approx(100.0)
    assert "D_phantom_start" not in names
    assert "D_magnet_start" not in no_magnet.named_planes()
    assert list(nothing.named_planes().values()) == pytest.approx(np.linspace(0.0, 500.0, 8)[1:-1])

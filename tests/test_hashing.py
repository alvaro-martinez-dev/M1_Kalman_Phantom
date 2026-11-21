# tests/test_hashing.py
"""Hashes of the folder names of data/: deterministic, sensitive to what changes the result and
blind to what does not."""
import copy

from src.geometry.detectors import DetectorPlanes
from src.geometry.geometry import Geometry
from src.geometry.layout import layout_name
from src.utils.hashing import (HASH_LENGTH, beam_hash, canonical, kalman_hash, magnet_hash, propagator_hash,
                               short_hash)

BEAM = {"particle": "PROTON", "x0": 0.0, "sigma_x0": 1.0, "theta0": 0.0, "sigma_theta0": 1e-3,
        "p0": 700.0, "sigma_p0": 3.0}
MAGNET = {"enabled": True, "z_start": 400.0, "z_length": 300.0, "x_half_gap": 50.0, "y_height": 400, "B0": 1.0,
          "model": "biot_savart", "biot_savart": {"n_filaments": 81},
          "interpolation": {"enabled": True, "dx": 1.0, "dz": 0.5}}
PROPAGATOR = {"input": "init_beam/a", "seed": 45, "n_max": 1000, "n_workers": None, "step": 1.0,
              "adaptive": False, "tolerances": {"x": 1e-6, "tx": 1e-8, "qp_rel": 1e-8},
              "scattering": {"enabled": True, "max_slice": 5.0}}


def test_hash_is_short_hex_and_deterministic():
    h = short_hash({"b": 1, "a": [1.0, None]})
    assert h == short_hash({"a": [1.0, None], "b": 1})          # key order does not matter
    assert len(h) == HASH_LENGTH and int(h, 16) >= 0


def test_integers_and_floats_from_the_yaml_give_the_same_hash():
    assert canonical({"a": 100, "b": [1, True]}) == {"a": 100.0, "b": [1.0, True]}
    assert beam_hash({**BEAM, "p0": 700}, 1000, 42) == beam_hash(BEAM, 1000, 42)


def test_beam_hash_covers_the_block_the_size_and_the_seed():
    h = beam_hash(BEAM, 1000, 42)
    assert beam_hash(BEAM, 1000, 43) != h
    assert beam_hash(BEAM, 2000, 42) != h
    assert beam_hash({**BEAM, "p0": 701.0}, 1000, 42) != h


def test_magnet_hash_ignores_what_does_not_change_the_field():
    h = magnet_hash(MAGNET, 900.0)
    assert magnet_hash({**MAGNET, "B0": 0.5}, 900.0) != h
    assert magnet_hash(MAGNET, 1000.0) != h                    # interpolation grid covers the system

    hard = {**copy.deepcopy(MAGNET), "model": "hard_edge", "interpolation": {"enabled": False, "dx": 1.0, "dz": 0.5}}
    other = copy.deepcopy(hard)
    other["biot_savart"]["n_filaments"] = 20                   # keys of the unused model
    other["interpolation"]["dz"] = 2.0                         # grid of a disabled interpolation
    assert magnet_hash(other, 900.0) == magnet_hash(hard, 900.0)
    assert magnet_hash(hard, 1000.0) == magnet_hash(hard, 900.0)   # no grid: the extent does not matter

    assert magnet_hash({**MAGNET, "enabled": False}, 900.0) == "off"


def test_propagator_hash_only_physics_options():
    h = propagator_hash(PROPAGATOR)
    assert propagator_hash({**PROPAGATOR, "seed": 7, "n_max": 10, "n_workers": 4, "input": "x"}) == h
    assert propagator_hash({**PROPAGATOR, "step": 0.5}) != h
    assert propagator_hash({**PROPAGATOR, "scattering": {"enabled": False, "max_slice": 5.0}}) != h
    # tolerances only matter with adaptive steps
    assert propagator_hash({**PROPAGATOR, "tolerances": {"x": 1.0, "tx": 1.0, "qp_rel": 1.0}}) == h


def test_kalman_hash():
    h = kalman_hash(True, False, (1e-4, 1e-7, 1e-6))
    assert kalman_hash(True, False, [1e-4, 1e-7, 1e-6]) == h
    assert kalman_hash(False, False, (1e-4, 1e-7, 1e-6)) != h
    assert kalman_hash(True, True, (1e-4, 1e-7, 1e-6)) != h
    # range constraints: off is the hash of the fits made before the option existed
    assert kalman_hash(True, False, (1e-4, 1e-7, 1e-6), range_constraints=False) == h
    assert kalman_hash(True, False, (1e-4, 1e-7, 1e-6), range_constraints=True) != h


def test_layout_name_starts_with_its_hash():
    g = Geometry(z_start_phantom=50.0, phantom_layers=[{"material": "WATER", "thickness": 50.0},
                                                       {"material": "BONE", "thickness": 20.0}],
                 z_start_magnet=400.0, z_length_magnet=300.0, spacing_after_last=100.0)
    detectors = DetectorPlanes(g, n_before_phantom=1, n_after_phantom_before_magnet=1, n_after_magnet=1,
                               inside_phantom=True, n_inside_magnet=1)
    name, _, hash_ = layout_name(g, detectors, 50.0)
    assert name.startswith(f"layout_Lay{hash_}_") and "_2Layers_1PhDet_1MagDet" in name

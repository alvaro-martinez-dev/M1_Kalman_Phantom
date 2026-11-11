# tests/test_mc_propagation.py
"""Monte Carlo of the full propagation: true state at every plane, stopped tracks, reproducibility."""
import numpy as np
import pandas as pd
import pytest

from src.constants import JPSI, PROTON
from src.field import HardEdgeField
from src.geometry.detectors import DetectorPlanes
from src.geometry.geometry import Geometry
from src.montecarlo.propagation import plane_list, propagate_tracks
from src.montecarlo.tracks import track_table
from src.propagator.propagator import Propagator, kinetic_energy, momentum
from src.utils.dataset_io import load_dataset, load_table, save_dataset

Z_PH, Z_MAG, L_MAG, HALF_GAP = 50.0, 400.0, 300.0, 50.0


def system(thickness=200.0):
    g = Geometry(z_start_phantom=Z_PH, phantom_layers=[{"material": "WATER", "thickness": thickness}],
                 z_start_magnet=Z_MAG, z_length_magnet=L_MAG, spacing_after_last=100.0)
    detectors = DetectorPlanes(g, n_before_phantom=1, n_after_phantom_before_magnet=1, n_after_magnet=1,
                               inside_phantom=False, n_inside_magnet=1)
    field = HardEdgeField(B0=1.0, z_start=Z_MAG, length=L_MAG, half_gap=HALF_GAP)
    prop = Propagator(g, field, HALF_GAP, step=5.0, max_slice=20.0)
    return prop, plane_list(detectors)


def beam(T_list, particle=PROTON):
    p = np.array([momentum(T, particle) for T in T_list])
    n = len(T_list)
    return track_table(np.arange(n), particle, x0=np.zeros(n), z0=np.zeros(n), px0=np.zeros(n), pz0=p)


def test_plane_list_merges_planes_at_the_same_z():
    g = Geometry(z_start_phantom=Z_PH, phantom_layers=[{"material": "WATER", "thickness": 350.0}],
                 z_start_magnet=Z_MAG, z_length_magnet=L_MAG, spacing_after_last=100.0)   # touching
    planes = plane_list(DetectorPlanes(g, 0, 0, 0, False, 0))
    assert [name for name, _ in planes] == ["D_phantom_start", "D_phantom_end+D_magnet_start", "D_magnet_end"]


def test_every_plane_is_reached_by_a_fast_proton():
    prop, planes = system()
    states, tracks = propagate_tracks(beam([400.0, 400.0]), prop, planes, np.random.default_rng(0))
    assert len(states) == 2 * len(planes) and not tracks["stopped"].any()
    one = states[states.track_id == 0]
    assert list(one["plane"]) == [name for name, _ in planes]
    assert one["z"].to_numpy() == pytest.approx([z for _, z in planes])
    assert (np.diff(one["T"].to_numpy()) <= 1e-9).all()          # energy only decreases
    assert tracks["n_planes"].tolist() == [len(planes)] * 2
    # the two tracks start equal: only the scattering makes them differ
    assert states[states.track_id == 0]["x"].iloc[-1] != states[states.track_id == 1]["x"].iloc[-1]


def test_stopped_track_has_no_states_after_it_stops():
    prop, planes = system()
    states, tracks = propagate_tracks(beam([100.0, 400.0]), prop, planes, np.random.default_rng(0))
    slow = tracks.iloc[0]
    assert slow["stopped"] and Z_PH < slow["z_final"] < Z_PH + 200.0
    assert states[states.track_id == 0]["z"].max() <= slow["z_final"]
    assert slow["n_planes"] == (states.track_id == 0).sum() < len(planes)
    assert not tracks.iloc[1]["stopped"]


def test_without_rng_it_is_the_deterministic_mean_track():
    prop, planes = system()
    a, _ = propagate_tracks(beam([300.0]), prop, planes, rng=None)
    b, _ = propagate_tracks(beam([300.0]), prop, planes, rng=None)
    pd.testing.assert_frame_equal(a, b)
    assert a[a.z < Z_MAG]["x"].abs().max() == 0.0                 # on axis until the magnet (no kicks)


def test_same_seed_same_dataset():
    prop, planes = system()
    run = lambda: propagate_tracks(beam([300.0, 350.0]), prop, planes, np.random.default_rng(5))
    pd.testing.assert_frame_equal(run()[0], run()[0])


def test_kinetic_energy_column_matches_q_over_p():
    prop, planes = system()
    states, _ = propagate_tracks(beam([300.0]), prop, planes, np.random.default_rng(1))
    T = [kinetic_energy(q_p, PROTON) for q_p in states["q_over_p"]]
    assert states["T"].to_numpy() == pytest.approx(T)


def test_neutral_or_mixed_input_is_rejected():
    prop, planes = system()
    with pytest.raises(ValueError, match="PROTON|proton|only"):
        propagate_tracks(beam([300.0], particle=JPSI), prop, planes, None)
    neutral = Propagator(prop.geometry, prop.field, HALF_GAP, particle=JPSI)
    with pytest.raises(ValueError, match="neutral"):
        propagate_tracks(beam([300.0], particle=JPSI), neutral, planes, None)


def test_backward_tracks_are_rejected():
    prop, planes = system()
    df = beam([300.0])
    df.loc[0, "pz0"] = -1.0
    with pytest.raises(ValueError, match="backwards"):
        propagate_tracks(df, prop, planes, None)


def test_dataset_with_an_extra_table(tmp_path):
    prop, planes = system()
    states, tracks = propagate_tracks(beam([300.0]), prop, planes, np.random.default_rng(2))
    out = save_dataset(tracks, tmp_path / "prop", {"seed": 2}, tables={"states": states})
    tracks_back, metadata = load_dataset(out)
    pd.testing.assert_frame_equal(tracks_back, tracks)
    pd.testing.assert_frame_equal(load_table(out, "states"), states)
    assert metadata["seed"] == 2
    with pytest.raises(ValueError, match="tracks"):
        save_dataset(tracks, tmp_path / "bad", {}, tables={"tracks": states})

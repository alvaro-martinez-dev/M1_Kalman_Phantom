# tests/test_mc_propagation.py
"""Monte Carlo of the full propagation: true state at every plane, stopped tracks, reproducibility."""
import numpy as np
import pandas as pd
import pytest

from src.constants import JPSI, PROTON
from src.field import HardEdgeField
from src.geometry.detectors import DetectorPlanes
from src.geometry.geometry import Geometry
from src.montecarlo.propagation import (plane_list, propagate_tracks, select_tracks, selection_rng,
                                        start_state, track_rng)
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
    states, tracks = propagate_tracks(beam([400.0, 400.0]), prop, planes, 0)
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
    states, tracks = propagate_tracks(beam([100.0, 400.0]), prop, planes, 0)
    slow = tracks.iloc[0]
    assert slow["stopped"] and Z_PH < slow["z_final"] < Z_PH + 200.0
    assert states[states.track_id == 0]["z"].max() <= slow["z_final"]
    assert slow["n_planes"] == (states.track_id == 0).sum() < len(planes)
    assert not tracks.iloc[1]["stopped"]


def test_without_seed_it_is_the_deterministic_mean_track():
    prop, planes = system()
    a, _ = propagate_tracks(beam([300.0]), prop, planes, seed=None)
    b, _ = propagate_tracks(beam([300.0]), prop, planes, seed=None)
    pd.testing.assert_frame_equal(a, b)
    assert a[a.z < Z_MAG]["x"].abs().max() == 0.0                 # on axis until the magnet (no kicks)


def test_same_seed_same_dataset():
    prop, planes = system()
    run = lambda: propagate_tracks(beam([300.0, 350.0]), prop, planes, 5)
    pd.testing.assert_frame_equal(run()[0], run()[0])


def test_a_track_alone_is_the_same_as_in_the_dataset():
    """Each track has its own random stream (seed, input_row): propagating it again on its own,
    with other output points, gives exactly the same track as in the full dataset."""
    prop, planes = system()
    seed = 9
    chosen = select_tracks(beam([250.0, 300.0, 350.0, 400.0]), 3, selection_rng(seed))
    states, _ = propagate_tracks(chosen, prop, planes, seed)

    row = chosen.iloc[1]
    plane_z = [z for _, z in planes]
    dense_z = sorted(set(np.arange(10.0, 790.0, 7.0)) | set(plane_z))      # many more output points
    alone = prop.propagate(start_state(row), prop.geometry.z_end_system, z_out=dense_z,
                           rng=track_rng(seed, row["input_row"]))
    at_planes = {round(s.z, 9): s for s in alone.at_z_out if s is not None}
    in_dataset = states[states.input_row == row["input_row"]]
    for z, x, tx in zip(in_dataset["z"], in_dataset["x"], in_dataset["tx"]):
        s = at_planes[round(z, 9)]
        assert (s.x, s.tx) == pytest.approx((x, tx), abs=1e-9)


def test_kicks_of_a_track_do_not_depend_on_the_other_tracks():
    prop, planes = system()
    df = beam([300.0] * 6)
    all_six, _ = propagate_tracks(select_tracks(df, None, None), prop, planes, 4)
    only_some, _ = propagate_tracks(select_tracks(df, None, None).iloc[[1, 4]], prop, planes, 4)
    for input_row in (1, 4):
        a = all_six[all_six.input_row == input_row][["z", "x", "tx"]].reset_index(drop=True)
        b = only_some[only_some.input_row == input_row][["z", "x", "tx"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
    # and different tracks get different kicks
    assert all_six[all_six.input_row == 1]["x"].iloc[-1] != all_six[all_six.input_row == 4]["x"].iloc[-1]


def test_kinetic_energy_column_matches_q_over_p():
    prop, planes = system()
    states, _ = propagate_tracks(beam([300.0]), prop, planes, 1)
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


def test_select_tracks_at_random_keeps_order_and_event_id():
    df = beam([300.0] * 50)
    chosen = select_tracks(df, 10, np.random.default_rng(0))
    assert len(chosen) == 10 and chosen["event_id"].is_unique
    assert chosen["event_id"].is_monotonic_increasing
    assert set(chosen["event_id"]) <= set(df["event_id"])
    assert chosen["event_id"].tolist() != list(range(10))          # not just the first ones
    assert (chosen["input_row"] == chosen["event_id"]).all()       # row in the input (= event_id for a beam)
    again = select_tracks(df, 10, np.random.default_rng(0))
    pd.testing.assert_frame_equal(chosen, again)                   # same seed, same tracks


def test_select_tracks_all_of_them():
    df = beam([300.0] * 5)
    every = select_tracks(df, None, None)
    assert every["input_row"].tolist() == list(range(5))
    pd.testing.assert_frame_equal(every.drop(columns="input_row"), df)
    assert len(select_tracks(df, 5, np.random.default_rng(0))) == 5   # n_max = n: all, allowed


@pytest.mark.parametrize("n_max, match", [(6, "larger than the number of tracks"), (0, ">= 1"),
                                          (-1, ">= 0"), (2.5, "integer"), ("1e3", "integer")])
def test_select_tracks_bad_n_max(n_max, match):
    with pytest.raises(ValueError, match=match):
        select_tracks(beam([300.0] * 5), n_max, np.random.default_rng(0))


def test_dataset_with_an_extra_table(tmp_path):
    prop, planes = system()
    states, tracks = propagate_tracks(beam([300.0]), prop, planes, 2)
    out = save_dataset(tracks, tmp_path / "prop", {"seed": 2}, tables={"states": states})
    tracks_back, metadata = load_dataset(out)
    pd.testing.assert_frame_equal(tracks_back, tracks)
    pd.testing.assert_frame_equal(load_table(out, "states"), states)
    assert metadata["seed"] == 2
    with pytest.raises(ValueError, match="tracks"):
        save_dataset(tracks, tmp_path / "bad", {}, tables={"tracks": states})

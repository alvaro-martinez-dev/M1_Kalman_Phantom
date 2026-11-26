# tests/test_depth_dose.py
"""Depth-dose curve from the Monte Carlo (src/phantom/depth_dose.py): energy conservation, agreement
with the analytic CSDA curve (src/phantom/dose.py) and reproducibility."""
import numpy as np
import pytest

from src.constants import PROTON
from src.field import NoField
from src.geometry.geometry import Geometry
from src.montecarlo.propagation import start_state, track_rng
from src.montecarlo.tracks import track_table
from src.phantom.depth_dose import csda_stop_depth, depth_dose, depth_grid, track_deposits
from src.phantom.dose import dose_profile
from src.propagator.propagator import Propagator, kinetic_energy, momentum

Z_PH = 50.0
LAYERS = [{"material": "WATER", "thickness": 50.0}, {"material": "BONE", "thickness": 20.0},
          {"material": "WATER", "thickness": 50.0}]


def phantom(layers=LAYERS):
    return Geometry(z_start_phantom=Z_PH, phantom_layers=layers, spacing_after_last=50.0)


def beam(T_list, x0=None, tx0=None):
    n = len(T_list)
    p = np.array([momentum(T, PROTON) for T in T_list])
    x0 = np.zeros(n) if x0 is None else np.asarray(x0, float)
    tx0 = np.zeros(n) if tx0 is None else np.asarray(tx0, float)
    pz = p / np.sqrt(1 + tx0**2)
    return track_table(np.arange(n), PROTON, x0=x0, z0=np.zeros(n), px0=pz * tx0, pz0=pz)


def test_depth_grid_has_a_node_at_every_interface():
    g = phantom()
    z = depth_grid(g, 0.3)
    for z_i in g.z_phantom_interfaces:
        assert np.min(np.abs(z - z_i)) < 1e-9
    assert np.all(np.diff(z) <= 0.3 + 1e-9) and z[0] == Z_PH and z[-1] == pytest.approx(Z_PH + 120.0)


@pytest.mark.parametrize("T0", [60.0, 80.0, 200.0])
def test_a_track_deposits_its_energy(T0):
    """A track that stops deposits exactly its energy at the phantom entrance; one that crosses it,
    its entrance minus its exit energy. Scattering on."""
    g = phantom()
    prop = Propagator(g, NoField(), np.inf, PROTON, step=1.0, max_slice=5.0)
    z_edges = depth_grid(g, 0.5)
    T_in, z_stop, index, x_mid, dE = track_deposits(prop, beam([T0]).iloc[0], 0, 7, z_edges)
    assert T_in == pytest.approx(T0)                      # no energy loss in vacuum
    if np.isnan(z_stop):
        # crosses: it deposits its entrance energy minus its exit energy (same track again, same kicks)
        result = prop.propagate(start_state(beam([T0]).iloc[0]), z_edges[-1], rng=track_rng(7, 0))
        assert not result.stopped
        assert dE.sum() == pytest.approx(T0 - kinetic_energy(result.final.q_p, PROTON), rel=1e-9)
    else:
        assert dE.sum() == pytest.approx(T0, rel=1e-12)
        assert z_edges[index[-1]] <= z_stop <= z_edges[index[-1] + 1] + 0.05
    assert np.all(dE >= 0) and np.all(np.diff(index) >= 0)


def test_monoenergetic_pencil_beam_matches_the_analytic_curve():
    """No scattering, no spread, along z: the Monte Carlo (RK4 of Bethe-Bloch) gives the CSDA curve
    of dose_profile (range-table inversion) bin by bin, and stops at the CSDA range."""
    g = phantom()
    T0, dz = 80.0, 0.25
    prop = Propagator(g, NoField(), np.inf, PROTON, step=0.25, max_slice=None)
    dd = depth_dose(prop, beam([T0]), [0], kick_seed=None, dz=dz, n_workers=1)
    z_a, T_a, D_a = dose_profile(g, T0, PROTON, n_steps=int(round(120.0 / dz)))
    assert dd.z_edges == pytest.approx(z_a)
    stop_bin = int(np.flatnonzero(dd.energy > 0)[-1])
    # bins crossed: same dose to 1e-4, and to 1e-3 in the last mm, where dE/dx grows fast and the RK4
    # step (0.25 mm) has a larger error; the bin of the stop holds the rest of the energy in both
    assert dd.dose[:stop_bin - 4] == pytest.approx(D_a[:stop_bin - 4], rel=1e-4)
    assert dd.dose[stop_bin - 4:stop_bin - 1] == pytest.approx(D_a[stop_bin - 4:stop_bin - 1], rel=1e-3)
    assert dd.dose[stop_bin - 1:stop_bin + 1].sum() == pytest.approx(D_a[stop_bin - 1:stop_bin + 1].sum(), rel=1e-3)
    assert dd.z_stop[0] == pytest.approx(csda_stop_depth(g, T0, PROTON), abs=0.05)


def test_csda_stop_depth():
    g = phantom([{"material": "WATER", "thickness": 200.0}])
    from src.phantom.materials import range_csda
    from src.constants import WATER
    assert csda_stop_depth(g, 100.0, PROTON) == pytest.approx(Z_PH + range_csda(100.0, WATER, PROTON), rel=1e-6)
    assert csda_stop_depth(g, 300.0, PROTON) is None          # range ~ 51 cm > 20 cm: crosses


def test_parallel_equals_serial_and_is_reproducible():
    g = phantom()
    prop = Propagator(g, NoField(), np.inf, PROTON, step=1.0, max_slice=5.0)
    rng = np.random.default_rng(1)
    tracks = beam(rng.normal(80.0, 1.5, 12), x0=rng.normal(0, 1, 12), tx0=rng.normal(0, 1e-3, 12))
    rows = np.arange(12)
    serial = depth_dose(prop, tracks, rows, kick_seed=3, dz=0.5, n_workers=1)
    parallel = depth_dose(prop, tracks, rows, kick_seed=3, dz=0.5, n_workers=2)
    again = depth_dose(prop, tracks, rows, kick_seed=3, dz=0.5, n_workers=1)
    assert parallel.energy == pytest.approx(serial.energy, rel=1e-12)
    assert again.energy == pytest.approx(serial.energy, rel=1e-12)
    np.testing.assert_allclose(parallel.z_stop, serial.z_stop)
    # each stopped track deposits all its entrance energy
    stopped = ~np.isnan(serial.z_stop)
    assert stopped.all()
    assert serial.energy.sum() == pytest.approx(serial.T_entrance.sum(), rel=1e-12)


def test_scattering_changes_x_not_the_energy_balance():
    """The kicks spread the deposits in x; the energy deposited is still the entrance energy."""
    g = phantom()
    prop = Propagator(g, NoField(), np.inf, PROTON, step=1.0, max_slice=5.0)
    dd = depth_dose(prop, beam([80.0] * 20), np.arange(20), kick_seed=11, dz=0.5, n_workers=1)
    assert dd.x_mid.std() > 0.0
    x_edges = np.linspace(-20, 20, 81)
    emap = dd.energy_map(x_edges)
    assert emap.sum() == pytest.approx(80.0, rel=1e-6)        # per proton

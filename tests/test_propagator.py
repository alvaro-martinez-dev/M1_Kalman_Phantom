# tests/test_propagator.py
"""Propagator: free space, energy loss in the phantom and bending in the field, region by region."""
import numpy as np
import pytest
from scipy.optimize import brentq

from src.constants import KAPPA, PROTON, WATER
from src.field import HardEdgeField, InterpolatedField, MagneticField, NoField
from src.geometry.geometry import Geometry
from src.phantom.materials import range_csda
from src.propagator.propagator import Propagator, State, kinetic_energy, momentum
from src.propagator.scattering import slice_covariance, theta0_slice

Z_PH, L_WATER = 50.0, 200.0                 # water phantom from 50 to 250 mm
Z_MAG, L_MAG = 400.0, 300.0                 # magnet from 400 to 700 mm
HALF_GAP = 50.0


def geometry(thickness=L_WATER):
    return Geometry(z_start_phantom=Z_PH, phantom_layers=[{"material": "WATER", "thickness": thickness}],
                    z_start_magnet=Z_MAG, z_length_magnet=L_MAG, spacing_after_last=100.0)


class UniformField(MagneticField):
    """By = B0 everywhere: checks that the propagator itself cuts the field at |x| > half gap."""
    def __init__(self, B0):
        self.B0 = B0

    def B(self, x, y, z):
        x = np.asarray(x, dtype=float)
        out = np.zeros(x.shape + (3,))
        out[..., 1] = self.B0
        return out


def proton_state(z, T_MeV, x=0.0, tx=0.0):
    return State(z=z, x=x, tx=tx, q_p=PROTON.charge / momentum(T_MeV, PROTON))


def no_field_propagator(thickness=L_WATER, **kwargs):
    return Propagator(geometry(thickness), UniformField(0.0), HALF_GAP, **kwargs)


# ----- free space ------------------------------------------------------------------------------

def test_free_space_before_phantom_is_a_straight_line():
    s = no_field_propagator().propagate(proton_state(0.0, 200.0, x=1.0, tx=0.01), Z_PH).final
    assert s.z == Z_PH
    assert s.x == pytest.approx(1.0 + 0.01 * Z_PH)
    assert s.tx == 0.01


# ----- phantom ---------------------------------------------------------------------------------

def csda_exit_energy(T_in, path):
    """T_out such that R(T_in) - R(T_out) = path."""
    R_in = range_csda(T_in, WATER)
    return brentq(lambda T: R_in - range_csda(T, WATER) - path, 1.0, T_in)


@pytest.mark.parametrize("adaptive", [False, True])
def test_phantom_energy_loss_matches_csda(adaptive):
    res = no_field_propagator(adaptive=adaptive).propagate(proton_state(0.0, 200.0), Z_PH + L_WATER)
    assert not res.stopped
    assert kinetic_energy(res.final.q_p, PROTON) == pytest.approx(csda_exit_energy(200.0, L_WATER), rel=1e-4)


def test_phantom_inclined_track_uses_the_path_length():
    tx = 0.2
    res = no_field_propagator().propagate(proton_state(0.0, 200.0, tx=tx), Z_PH + L_WATER)
    path = L_WATER * np.sqrt(1 + tx**2)
    assert kinetic_energy(res.final.q_p, PROTON) == pytest.approx(csda_exit_energy(200.0, path), rel=1e-4)
    assert res.final.tx == tx
    assert res.final.x == pytest.approx(tx * (Z_PH + L_WATER))


def test_proton_stops_at_the_csda_range():
    res = no_field_propagator(thickness=300.0).propagate(proton_state(0.0, 200.0), 1000.0, z_out=[100.0, 500.0])
    assert res.stopped
    assert res.final.z == pytest.approx(Z_PH + range_csda(200.0, WATER))
    assert res.at_z_out[0] is not None and res.at_z_out[1] is None


# ----- field -----------------------------------------------------------------------------------

def test_hard_edge_magnet_matches_the_circle():
    B0, T = 1.0, 500.0
    field = HardEdgeField(B0=B0, z_start=Z_MAG, length=L_MAG, half_gap=HALF_GAP)
    prop = Propagator(geometry(), field, HALF_GAP, step=0.5)
    res = prop.propagate(proton_state(300.0, T), Z_MAG + L_MAG)

    r = momentum(T, PROTON) / (KAPPA * B0)      # [mm], positive charge bends to -x for By > 0
    root = np.sqrt(r**2 - L_MAG**2)
    assert res.final.x == pytest.approx(-(r - root), abs=1e-6)
    assert res.final.tx == pytest.approx(-L_MAG / root, rel=1e-8)
    assert res.final.q_p == res.final.q_p       # |p| constant in the field


def test_no_field_beyond_the_half_gap():
    prop = Propagator(geometry(), UniformField(1.0), HALF_GAP)
    inside = prop.propagate(proton_state(300.0, 500.0, x=0.0), 700.0).final
    outside = prop.propagate(proton_state(300.0, 500.0, x=HALF_GAP + 10.0), 700.0).final
    assert inside.tx < -0.01                    # bends
    assert outside.tx == 0.0 and outside.x == HALF_GAP + 10.0   # straight


def test_field_is_used_between_phantom_and_magnet():
    """Fringe field: the region between phantom end (250) and magnet start (400) is not free."""
    prop = Propagator(geometry(), UniformField(1.0), HALF_GAP)
    s = prop.propagate(proton_state(Z_PH + L_WATER, 500.0), Z_MAG).final
    assert s.tx < 0.0


def test_field_is_used_before_the_phantom():
    """The vacuum before the phantom is not assumed field-free: a field reaching it bends the track."""
    prop = Propagator(geometry(), UniformField(1.0), HALF_GAP)
    s = prop.propagate(proton_state(0.0, 500.0), Z_PH).final
    tx_expected = -KAPPA * 1.0 * proton_state(0.0, 500.0).q_p * Z_PH     # small angle
    assert s.tx == pytest.approx(tx_expected, rel=1e-3)


def test_straight_line_outside_the_field_support():
    field = HardEdgeField(B0=1.0, z_start=Z_MAG, length=L_MAG, half_gap=HALF_GAP)
    prop = Propagator(geometry(), field, HALF_GAP)
    assert prop.field_is_zero(Z_PH + L_WATER, Z_MAG)
    assert prop.field_is_zero(Z_MAG + L_MAG, 800.0)
    assert not prop.field_is_zero(Z_MAG, Z_MAG + L_MAG)
    s = prop.propagate(proton_state(Z_PH + L_WATER, 500.0, x=1.0, tx=0.01), Z_MAG).final
    assert s.tx == 0.01 and s.x == pytest.approx(1.0 + 0.01 * (Z_MAG - Z_PH - L_WATER))


def test_field_support():
    hard = HardEdgeField(B0=1.0, z_start=Z_MAG, length=L_MAG, half_gap=HALF_GAP)
    assert hard.z_support() == (Z_MAG, Z_MAG + L_MAG)
    interp = InterpolatedField.from_field(hard, x_max=HALF_GAP, z_min=Z_MAG - 50.0, z_max=Z_MAG + L_MAG + 50.0,
                                          dx=1.0, dz=2.0)
    assert interp.z_support() == pytest.approx((Z_MAG - 50.0, Z_MAG + L_MAG + 50.0))
    assert UniformField(1.0).z_support() == (-np.inf, np.inf)


# ----- phantom + field (coupled) ---------------------------------------------------------------

@pytest.mark.parametrize("adaptive", [False, True])
def test_field_inside_phantom_bends_more_as_the_proton_slows_down(adaptive):
    B0, T0 = 1.0, 200.0
    z_in, z_out = Z_PH, Z_PH + L_WATER
    z_grid = np.linspace(z_in, z_out, 401)
    with_field = Propagator(geometry(), UniformField(B0), HALF_GAP, adaptive=adaptive)
    res = with_field.propagate(proton_state(z_in, T0), z_out, z_out=z_grid[1:-1])

    # The bent track is longer than the layer, so it loses more energy: CSDA along the real path
    tx = np.array([0.0] + [s.tx for s in res.at_z_out] + [res.final.tx])
    path = np.trapezoid(np.sqrt(1 + tx**2), z_grid)
    assert path > L_WATER
    T_out = kinetic_energy(res.final.q_p, PROTON)
    assert T_out == pytest.approx(csda_exit_energy(T0, path), rel=1e-4)
    assert T_out < csda_exit_energy(T0, L_WATER)

    # Small angles: tx_out = -kappa B int q/p(z) dz, with q/p(z) from the propagation without field
    no_field = no_field_propagator().propagate(proton_state(z_in, T0), z_out, z_out=z_grid[1:-1])
    q_p = [proton_state(z_in, T0).q_p] + [s.q_p for s in no_field.at_z_out] + [no_field.final.q_p]
    tx_expected = -KAPPA * B0 * np.trapezoid(q_p, z_grid)
    assert res.final.tx == pytest.approx(tx_expected, rel=1e-2)

    # ... which is more than in vacuum with the initial momentum (the curvature grows)
    tx_vacuum = -KAPPA * B0 * proton_state(z_in, T0).q_p * L_WATER
    assert res.final.tx < tx_vacuum < 0


def test_no_field_inside_phantom_beyond_the_half_gap():
    prop = Propagator(geometry(), UniformField(1.0), HALF_GAP)
    s = prop.propagate(proton_state(Z_PH, 200.0, x=HALF_GAP + 10.0), Z_PH + L_WATER).final
    assert s.tx == 0.0


def test_field_map_starting_inside_phantom_warns_and_is_a_cut():
    source = HardEdgeField(B0=1.0, z_start=Z_MAG, length=L_MAG, half_gap=HALF_GAP)
    field = InterpolatedField.from_field(source, x_max=HALF_GAP, z_min=Z_PH + 100.0,   # inside the water
                                         z_max=Z_MAG + L_MAG, dx=1.0, dz=2.0)
    with pytest.warns(UserWarning, match="inside the phantom"):
        prop = Propagator(geometry(), field, HALF_GAP)
    assert np.any(np.abs(prop.cuts(0.0, 900.0) - (Z_PH + 100.0)) < 1e-9)


def test_smooth_field_does_not_warn(recwarn):
    Propagator(geometry(), UniformField(1.0), HALF_GAP)
    assert len(recwarn) == 0


# ----- cuts and z_out --------------------------------------------------------------------------

def test_states_at_z_out():
    prop = no_field_propagator()
    z_out = [20.0, Z_PH, 150.0, 800.0]          # one of them on a boundary
    res = prop.propagate(proton_state(0.0, 200.0, tx=0.01), 800.0, z_out=z_out)
    assert [s.z for s in res.at_z_out] == pytest.approx(z_out)
    assert res.final == res.at_z_out[-1]


def test_no_phantom_no_energy_loss_and_field_before_the_magnet():
    """Without phantom the whole vacuum before the magnet is integrated with the field."""
    g = Geometry(z_start_magnet=Z_MAG, z_length_magnet=L_MAG, spacing_after_last=100.0)
    res = Propagator(g, UniformField(1.0), HALF_GAP).propagate(proton_state(0.0, 200.0), Z_MAG)
    assert res.final.q_p == proton_state(0.0, 200.0).q_p
    assert res.final.tx < 0.0


def test_no_magnet_with_no_field_is_a_straight_line_after_the_phantom():
    g = Geometry(z_start_phantom=Z_PH, phantom_layers=[{"material": "WATER", "thickness": L_WATER}],
                 spacing_after_last=100.0)
    prop = Propagator(g, NoField(), HALF_GAP)
    assert prop.field_is_zero(0.0, 1e9)
    s = prop.propagate(proton_state(0.0, 200.0, x=1.0, tx=0.01), g.z_end_system).final
    assert s.tx == 0.01 and s.x == pytest.approx(1.0 + 0.01 * g.z_end_system)


# ----- multiple scattering ---------------------------------------------------------------------

def test_slice_edges_are_cuts_and_do_not_change_the_mean_track():
    with_slices = no_field_propagator(max_slice=20.0)
    assert all(np.any(np.abs(with_slices.cuts(0.0, 900.0) - z) < 1e-9) for z in np.arange(70.0, 250.0, 20.0))
    s0 = proton_state(0.0, 200.0, x=1.0, tx=0.02)
    a = no_field_propagator().propagate(s0, 400.0).final
    b = with_slices.propagate(s0, 400.0).final            # rng=None: no kicks
    assert (b.x, b.tx, b.q_p) == pytest.approx((a.x, a.tx, a.q_p), rel=1e-9, abs=1e-12)


def test_kicks_need_slices():
    with pytest.raises(ValueError, match="max_slice"):
        no_field_propagator().propagate(proton_state(0.0, 200.0), 400.0, rng=np.random.default_rng(0))


def test_same_seed_same_track():
    prop = no_field_propagator(max_slice=20.0)
    run = lambda: prop.propagate(proton_state(0.0, 200.0), 400.0, rng=np.random.default_rng(7)).final
    assert run() == run()
    assert run().tx != 0.0


def test_kicks_only_inside_the_phantom():
    """Kicks are applied in the phantom only: before it, the track is untouched."""
    prop = no_field_propagator(max_slice=20.0)
    s = prop.propagate(proton_state(0.0, 200.0, x=1.0, tx=0.01), Z_PH, rng=np.random.default_rng(3)).final
    assert s.x == pytest.approx(1.0 + 0.01 * Z_PH) and s.tx == 0.01


def test_spread_at_the_phantom_exit_matches_the_slice_covariances():
    """Var(x), Var(tx), Cov at the exit = sum of the slice covariances transported to the exit."""
    T0, max_slice, n_tracks = 200.0, 20.0, 1500
    z_exit = Z_PH + L_WATER
    prop = no_field_propagator(max_slice=max_slice, step=5.0)
    s0 = proton_state(Z_PH, T0)

    # Expected: mean track -> theta0 of each slice -> covariance transported as a straight line
    edges = [sl[0] for sl in prop.slices] + [prop.slices[-1][1]]
    mean = prop.propagate(s0, z_exit, z_out=edges[1:-1])
    states = [s0] + mean.at_z_out + [mean.final]
    expected = np.zeros((2, 2))
    for entry, exit_ in zip(states[:-1], states[1:]):
        dz = exit_.z - entry.z
        p_mid = 0.5 * (PROTON.charge / entry.q_p + PROTON.charge / exit_.q_p)
        theta0 = theta0_slice(PROTON.charge / p_mid, 0.0, dz, L_WATER, WATER)
        d = z_exit - exit_.z                                 # drift from the kick to the exit
        F = np.array([[1.0, d], [0.0, 1.0]])
        expected += F @ slice_covariance(theta0, 0.0, dz) @ F.T

    rng = np.random.default_rng(11)
    out = np.array([[r.x, r.tx] for r in
                    (prop.propagate(s0, z_exit, rng=rng).final for _ in range(n_tracks))])
    assert out.mean(axis=0) == pytest.approx([0.0, 0.0], abs=4 * np.sqrt(np.diag(expected) / n_tracks).max())
    assert np.cov(out.T) == pytest.approx(expected, rel=0.12)


# ----- from the yaml ---------------------------------------------------------------------------

def test_make_propagator_from_the_config():
    import copy
    from src.field import NoField, build_field
    from src.utils.config_loader import load_config, make_geometry, make_propagator

    cfg = load_config("config")
    cfg["geometry"]["magnet"]["model"] = "hard_edge"            # fast to build
    geometry = make_geometry(cfg["geometry"])
    prop = make_propagator(cfg, geometry, build_field(cfg, geometry), PROTON)
    assert prop.max_slice == cfg["propagator"]["scattering"]["max_slice"]
    assert prop.step == cfg["propagator"]["step"] and len(prop.slices) > 0

    off = copy.deepcopy(cfg)
    off["propagator"]["scattering"]["enabled"] = False
    off["geometry"]["magnet"]["enabled"] = False
    geometry_off = make_geometry(off["geometry"])
    prop = make_propagator(off, geometry_off, build_field(off, geometry_off), PROTON)
    assert prop.max_slice is None and prop.slices == []
    assert isinstance(prop.field, NoField) and prop.x_half_gap == np.inf


def test_make_propagator_rejects_a_bad_step():
    from src.field import build_field
    from src.utils.config_loader import load_config, make_geometry, make_propagator

    cfg = load_config("config")
    cfg["geometry"]["magnet"]["model"] = "hard_edge"
    cfg["propagator"]["step"] = 0.0
    geometry = make_geometry(cfg["geometry"])
    with pytest.raises(ValueError, match="propagator.step"):
        make_propagator(cfg, geometry, build_field(cfg, geometry), PROTON)


def test_field_map_covers_the_whole_system():
    """build_field: the map spans [0, z_end_system] in z and the gap in x, so it never ends inside
    the system (no warning, no artificial cut of the fringe field), whatever the geometry."""
    import warnings
    from src.field import build_field
    from src.utils.config_loader import load_config, make_geometry, make_propagator

    cfg = load_config("config")
    cfg["geometry"]["magnet"]["model"] = "hard_edge"
    geometry = make_geometry(cfg["geometry"])
    field = build_field(cfg, geometry)
    z_lo, z_hi = field.z_support()
    assert z_lo <= 0.0 and z_hi >= geometry.z_end_system
    x_nodes = field._xs
    assert x_nodes[-1] >= cfg["geometry"]["magnet"]["x_half_gap"] and x_nodes[-1] < cfg["geometry"]["magnet"]["x_half_gap"] + 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_propagator(cfg, geometry, field, PROTON)


def test_backward_not_implemented():
    with pytest.raises(NotImplementedError):
        no_field_propagator().propagate(proton_state(100.0, 200.0), 50.0)

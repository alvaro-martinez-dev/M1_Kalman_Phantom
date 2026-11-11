# tests/test_propagator.py
"""Propagator: free space, energy loss in the phantom and bending in the field, region by region."""
import numpy as np
import pytest
from scipy.optimize import brentq

from src.constants import KAPPA, PROTON, WATER
from src.field import HardEdgeField, InterpolatedField, MagneticField
from src.geometry.geometry import Geometry
from src.phantom.materials import range_csda
from src.propagator.propagator import Propagator, State, kinetic_energy, momentum

Z_PH, L_WATER = 50.0, 200.0                 # water phantom from 50 to 250 mm
Z_MAG, L_MAG = 400.0, 300.0                 # magnet from 400 to 700 mm
HALF_GAP = 50.0


def geometry(thickness=L_WATER):
    return Geometry(z_start_phantom=Z_PH, phantom_layers=[{"material": "WATER", "thickness": thickness}],
                    z_start_magnet=Z_MAG, z_length_magnet=L_MAG, spacing_after_magnet=100.0)


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
    margin = Z_MAG - (Z_PH + 100.0)                # map starts at z = 150, inside the water
    field = InterpolatedField.from_field(source, Z_MAG, L_MAG, HALF_GAP, margin, 1.0, 2.0)
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


def test_backward_not_implemented():
    with pytest.raises(NotImplementedError):
        no_field_propagator().propagate(proton_state(100.0, 200.0), 50.0)

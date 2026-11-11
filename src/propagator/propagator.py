# src/propagator/propagator.py
#
# Propagates the state of a track (x, tx, q/p) along z through the geometry, region by region.
# Units: mm, MeV, T; q in e, q_p = q/p in e/MeV, tx = dx/dz.
#
# The z axis is cut at every physical boundary (Geometry.boundaries()), at every jump of the field
# (field.z_discontinuities(), e.g. the ends of an interpolated map) and at every z where the state
# is requested (e.g. detector planes). Each piece is propagated according to what there is in it
# (material and field), not according to where it is with respect to the phantom or the magnet:
#
#   material                      energy loss + field, coupled: rk4 on y = (x, tx, q/p) with
#                                 eom_phantom_field. The fringe field can reach the phantom
#   vacuum, field may be != 0     rkn4 on x with eom_magnet, |p| constant. The field is NOT cut at
#                                 the magnet box: the fringe field before the phantom, between
#                                 phantom and magnet and after the magnet is not negligible
#   vacuum, field = 0             exact straight line (propagate_free_space). Only a shortcut: RKN
#                                 with zero field gives the same line. "Field = 0" means outside
#                                 field.z_support() (whose ends are cuts, as jumps of the field)
#
# In the field, for |x| > x_half_gap (outside the plates) the field is taken as zero.
#
# Multiple scattering (src/propagator/scattering.py): with max_slice, each phantom layer is cut into
# slices (phantom_slices), whose edges are also cuts. propagate(..., rng=None) is deterministic (the
# mean track, for the Kalman filter). With an rng (Monte Carlo) a random (dx, dtx) kick is applied
# at the end of each slice, with theta0 from the mean momentum and tx between its entrance and exit.
#
# Not included yet:
#   - backward propagation (z_end < z)
#   - energy-loss straggling (q/p is not kicked)

import warnings
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from src.constants import PROTON, Particle
from src.geometry.geometry import Geometry, TOL
from src.phantom.materials import E_MIN_MEV, material_object, range_csda, range_csda_fast
from src.propagator.equations import eom_magnet, eom_phantom_field, propagate_free_space
from src.propagator.rk4 import rk4, rkn4
from src.propagator.scattering import phantom_slices, sample_kick, theta0_slice

# A particle is considered stopped in a layer if its residual range after crossing it would be
# shorter than the range at this kinetic energy (~20 um in water at 1 MeV): near the end of the
# range dE/ds diverges and the integration of q/p is not reliable
T_STOP_MEV = 1.0


@dataclass(frozen=True)
class State:
    z: float      # [mm]
    x: float      # [mm]
    tx: float     # dx/dz
    q_p: float    # [e/MeV]


@dataclass
class PropagationResult:
    final: State                  # state at z_end, or where the particle stopped
    at_z_out: list                # State at each requested z_out (None if the particle stopped before it)
    stopped: bool = False         # True if the particle stopped inside the phantom


# T_MeV -> |p| and q_p -> T_MeV for a given particle
def momentum(T_MeV, particle):
    return np.sqrt(T_MeV * (T_MeV + 2.0 * particle.mass))


def kinetic_energy(q_p, particle):
    p = abs(particle.charge / q_p)
    return np.sqrt(p**2 + particle.mass**2) - particle.mass


@lru_cache(maxsize=None)
def _stop_margin(material, particle):
    # Residual range [mm] below which the particle is considered stopped (see T_STOP_MEV)
    return range_csda(T_STOP_MEV, material, particle)


def _clamped(fun, z_a, z_b):
    # The RK also evaluates fun at the ends of the segment. With a discontinuous field (hard edge,
    # end of a field map) the end on a boundary would see the field of the next region, so z is
    # kept TOL inside the segment. For a smooth field this changes nothing
    z_lo, z_hi = min(z_a, z_b) + TOL, max(z_a, z_b) - TOL
    def fun_clamped(z, *args):
        return fun(min(max(z, z_lo), z_hi), *args)
    return fun_clamped


class Propagator:

    # geometry: Geometry (phantom, magnet)
    # field: MagneticField (see src.field.build_field)
    # x_half_gap: [mm] half gap between the plates; for |x| > x_half_gap there is no field
    # particle: Particle being propagated
    # step: [mm] RK step (fixed step), or initial and maximum step (adaptive)
    # adaptive: use adaptive steps (rkn4 / rk45) instead of fixed ones
    # tol_x: [mm] local error allowed in x (only if adaptive)
    # tol_tx: local error allowed in tx in the phantom (only if adaptive)
    # tol_qp_rel: relative local error allowed in q/p in the phantom (only if adaptive)
    # max_slice: [mm] maximum slice length for multiple scattering in the phantom (see
    #            scattering.phantom_slices), or None for no slices (no scattering possible)
    def __init__(self, geometry: Geometry, field, x_half_gap, particle: Particle = PROTON,
                 step=1.0, adaptive=False, tol_x=1e-6, tol_tx=1e-8, tol_qp_rel=1e-8, max_slice=None):
        self.geometry = geometry
        self.field = field
        self.x_half_gap = x_half_gap
        self.particle = particle
        self.step = step
        self.adaptive = adaptive
        self.tol_x = tol_x
        self.tol_tx = tol_tx
        self.tol_qp_rel = tol_qp_rel

        # Scattering slices: (z_a, z_b, material_name, dz_layer) in beam order, and their edges
        self.max_slice = max_slice
        self.slices = phantom_slices(geometry, max_slice) if max_slice is not None else []
        self._slice_starts = np.array([s[0] for s in self.slices])
        self._slice_ends = np.array([s[1] for s in self.slices])

        # A jump of the field inside the phantom (e.g. the field map starts there) means that a
        # non-zero field is being truncated: the propagation is still cut there, but warn
        g = geometry
        jumps = [z for z in field.z_discontinuities()
                 if g.has_phantom and g.z_start_phantom < z < g.z_end_phantom]
        if jumps:
            warnings.warn(f"The field jumps inside the phantom at z = {jumps} mm (e.g. the field map "
                          "starts there): the field is truncated. Make the field map cover the "
                          "whole phantom (build_field does it)", stacklevel=2)


    def cuts(self, z0, z_end, z_out=()):
        # Sorted z values where the propagation is cut: z0, z_end, the physical boundaries,
        # the jumps of the field, the scattering slice edges and z_out
        inside = [z for z, _ in self.geometry.boundaries() if z0 < z < z_end]
        inside += [z for z in self.field.z_discontinuities() if z0 < z < z_end]
        inside += [z for z in self._slice_ends if z0 < z < z_end]
        inside += [z for z in z_out if z0 < z < z_end]
        z_all = np.unique(np.concatenate(([z0, z_end], inside)))
        # merge cuts closer than TOL (e.g. a detector plane on a boundary)
        keep = np.concatenate(([True], np.diff(z_all) > TOL))
        z_all = z_all[keep]
        z_all[-1] = z_end
        return z_all


    def propagate(self, state: State, z_end, z_out=(), rng=None):
        # state: State at state.z
        # z_end: [mm] final z, must be >= state.z
        # z_out: [mm] z values where the state is also returned (e.g. detector planes)
        # rng: numpy Generator for the scattering kicks (Monte Carlo), or None for the deterministic
        #      mean track. Needs max_slice. The state at a slice end (and at a z_out there) includes
        #      the kick of that slice
        # Returns: PropagationResult
        if z_end < state.z:
            raise NotImplementedError("Backward propagation (z_end < state.z) is not implemented yet")
        if rng is not None and self.max_slice is None:
            raise ValueError("Scattering kicks (rng) need the phantom slices: build the Propagator with max_slice")

        z_cuts = self.cuts(state.z, z_end, z_out)
        visited = {}    # z_cut -> State
        current = state
        slice_entry = state    # state at the entrance of the current slice (or the initial state)

        for z_a, z_b in zip(z_cuts[:-1].tolist(), z_cuts[1:].tolist()):
            if self._is_edge(z_a, self._slice_starts):
                slice_entry = current
            current, stopped = self.propagate_segment(current, z_a, z_b)
            if stopped:
                return self.result(current, z_out, visited, stopped=True)
            if rng is not None and self._is_edge(z_b, self._slice_ends):
                current = self.scatter(slice_entry, current, rng)
            visited[z_b] = current

        return self.result(current, z_out, visited, stopped=False)


    @staticmethod
    def _is_edge(z, edges):
        # True if z is one of the edges (within TOL)
        return len(edges) > 0 and np.min(np.abs(edges - z)) <= TOL


    def scatter(self, entry, exit, rng):
        # Random kick at the end of a slice, from entry (State at its entrance, or where the
        # propagation started) to exit (State at its end, before the kick). Returns the kicked State
        _, _, material_name, dz_layer = self.slices[int(np.argmin(np.abs(self._slice_ends - exit.z)))]
        dz = exit.z - entry.z
        if dz <= TOL or self.particle.charge == 0:
            return exit

        # mean momentum and slope in the slice
        p_mid = 0.5 * (abs(self.particle.charge / entry.q_p) + abs(self.particle.charge / exit.q_p))
        q_p_mid = np.sign(exit.q_p) * abs(self.particle.charge) / p_mid
        tx_mid = 0.5 * (entry.tx + exit.tx)

        theta0 = theta0_slice(q_p_mid, tx_mid, dz, dz_layer, material_object(material_name), self.particle)
        dx, dtx = sample_kick(theta0, tx_mid, dz, rng)
        return replace(exit, x=exit.x + dx, tx=exit.tx + dtx)


    def result(self, final, z_out, visited, stopped):
        # State at each z_out: the visited cut closest to it (cuts were merged within TOL)
        z_visited = np.array(list(visited))
        at_z_out = []
        for z in z_out:
            if len(z_visited) and np.min(np.abs(z_visited - z)) <= TOL:
                at_z_out.append(visited[z_visited[np.argmin(np.abs(z_visited - z))]])
            else:
                at_z_out.append(None)
        return PropagationResult(final=final, at_z_out=at_z_out, stopped=stopped)


    def propagate_segment(self, state, z_a, z_b):
        # Propagates from z_a to z_b, with z_a and z_b inside one region. Returns (State, stopped)
        z_mid = 0.5 * (z_a + z_b)
        material_name = self.geometry.material_at(z_mid)

        if self.particle.charge == 0:   # neutral: straight line everywhere
            return self.free(state, z_b), False
        if material_name is not None:
            return self.phantom(state, z_b, material_object(material_name))
        if self.field_is_zero(z_a, z_b):
            return self.free(state, z_b), False
        return self.magnetic(state, z_b), False


    def field_is_zero(self, z_a, z_b):
        # True if [z_a, z_b] is outside the support of the field. The ends of the support are cuts,
        # so a segment is either fully inside or fully outside it
        z_lo, z_hi = self.field.z_support()
        return max(z_a, z_b) <= z_lo + TOL or min(z_a, z_b) >= z_hi - TOL


    def free(self, state, z_b):
        x, tx = propagate_free_space(state.x, state.tx, state.z, z_b)
        return replace(state, z=z_b, x=x, tx=tx)


    def phantom(self, state, z_b, material):
        # One layer: energy loss and field coupled, rk4 on y = (x, tx, q/p)
        particle = self.particle
        path_factor = np.sqrt(1.0 + state.tx**2)           # ds/dz at the entrance
        T = kinetic_energy(state.q_p, particle)

        # Does it stop in this layer? (CSDA range vs path length in the layer). The path is taken
        # as straight with the entrance tx: the field bends the track very little inside the phantom
        residual = range_csda_fast(T, material, particle) if T > E_MIN_MEV else 0.0
        if residual <= (z_b - state.z) * path_factor + _stop_margin(material, particle):
            return self.stopped_at(state, residual / path_factor), True

        fun = _clamped(eom_phantom_field(self.field, material, particle, x_half_gap=self.x_half_gap),
                       state.z, z_b)
        y0 = np.array([state.x, state.tx, state.q_p])
        try:
            if self.adaptive:
                tol = np.array([self.tol_x, self.tol_tx, self.tol_qp_rel * abs(state.q_p)])
                _, y = rk4(fun, state.z, z_b, y0, self.step, adaptative=True, tol=tol)
            else:
                _, y = rk4(fun, state.z, z_b, y0, self.step)
        except ValueError:
            # bethe_bloch refused an energy below E_MIN inside an RK stage: the particle is at the
            # end of its range, so it is stopped where the CSDA range says
            return self.stopped_at(state, residual / path_factor), True

        x, tx, q_p = (float(v) for v in y[-1])
        return State(z=z_b, x=x, tx=tx, q_p=q_p), False


    def stopped_at(self, state, dz):
        # State where the particle stops, dz after state.z (straight line from state, the bending
        # by the field in this last piece is neglected). q/p is the one at E_MIN_MEV
        x, tx = propagate_free_space(state.x, state.tx, state.z, state.z + dz)
        q_p = self.particle.charge / momentum(E_MIN_MEV, self.particle)
        return State(z=state.z + dz, x=x, tx=tx, q_p=q_p)


    def magnetic(self, state, z_b):
        # Field region: |p| constant, x'' = f(z, x, x') with rkn4
        fun = _clamped(eom_magnet(self.field, state.q_p, self.particle, x_half_gap=self.x_half_gap),
                       state.z, z_b)
        if self.adaptive:
            _, x, tx = rkn4(fun, state.z, z_b, state.x, state.tx, self.step, adaptative=True,
                            tol=self.tol_x, h_max=self.step)
        else:
            _, x, tx = rkn4(fun, state.z, z_b, state.x, state.tx, self.step)
        return replace(state, z=z_b, x=float(x[-1]), tx=float(tx[-1]))

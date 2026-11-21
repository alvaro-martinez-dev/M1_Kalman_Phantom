# src/kalman/transport.py
#
# Prediction step of the Kalman filter: transports the state (x, tx, q/p) from one z to the next
# with the same deterministic propagator as the Monte Carlo, and returns
#
#   x_pred: predicted state at z_b (the mean track, no scattering kicks)
#   F:      3x3 Jacobian d x_pred / d x, by finite differences on Propagator.propagate_segment
#   Q:      3x3 process noise (multiple scattering) accumulated between z_a and z_b
#
# The propagation is cut exactly as in the Monte Carlo (Propagator.cuts: boundaries, field jumps,
# scattering slices), and each piece is propagated with propagate_segment, which chooses by itself
# between straight line, RKN (vacuum with field) and RK4 (material + field). So F follows the same
# model as the Monte Carlo everywhere, including the fringe field.
#
# Accumulation over the pieces i = 1..n between z_a and z_b:
#   F = F_n ... F_1
#   Q <- F_i Q F_i^T          after each piece
#   Q[:2, :2] += Q_slice      at the end of each scattering slice, where the Monte Carlo kicks
# Q_slice is slice_covariance(theta0_slice(...)) evaluated on the predicted (mean) track, with the
# mean q/p and tx of the slice, as Propagator.scatter does. No energy-loss straggling (q/p is not
# kicked in the Monte Carlo either), so Q has no q/p terms.
#
# Fixed RK steps, always. F comes from finite differences: (propagate(x + dx) - propagate(x)) / dx.
# This is only smooth if the nominal and the perturbed propagations take EXACTLY the same RK steps.
# With adaptive steps the controller may accept/reject steps differently for the perturbed state, and
# each propagation is then off by up to the tolerance; divided by the tiny dx this is a large error
# in F (e.g. tol_x / dtx = 1e-6 mm / 1e-7 = 10 mm in dx/dtx, which is a few hundred mm). So the
# Transport uses its own copy of the propagator with adaptive=False and the same step, whatever the
# Monte Carlo used. This is not a model mismatch: the RK step is a numerical choice, not physics.
# Fixed and adaptive integrations of the same equations agree within the tolerance (~1e-6 mm), far
# below the resolution and the scattering. The physics (geometry, field, energy loss, scattering
# slices) is shared with the Monte Carlo. Real fitters (GENFIT, ACTS) keep adaptive steps and transport
# F analytically through the same RK stages instead; for this toy model fixed steps are enough.
# With adaptive=True in the Monte Carlo, `step` is only the MAXIMUM step: check that the fixed step is
# converged (results with step and step/2 must agree well below sigma_x). A warning reminds it.
#
# Units: mm, MeV; q in e, q/p in e/MeV.

import copy
import warnings

import numpy as np

from src.geometry.geometry import TOL
from src.phantom.materials import E_MIN_MEV, material_object
from src.propagator.propagator import State, kinetic_energy, momentum
from src.propagator.scattering import slice_covariance, theta0_slice

# Default finite-difference steps for (x [mm], tx, q/p [relative])
FD_STEPS = (1.0e-4, 1.0e-7, 1.0e-6)

# Safety limit of the bisection of reach_limit (it normally stops at s_tol after ~20-40 steps)
MAX_BISECTIONS = 200


class TrackStopped(Exception):
    # The predicted (mean) track stops in the phantom before z_b
    pass


class Transport:

    # propagator: Propagator of the Monte Carlo (same geometry, field, step and max_slice). It is not
    #             modified: the Transport keeps a copy with fixed steps (see the header).
    #             Without max_slice there are no slices: Q = 0
    # fd_steps: finite-difference steps for (x [mm], tx, q/p [relative to |q/p|])
    # central: central differences (2 extra propagations per variable) instead of forward ones (1)
    def __init__(self, propagator, fd_steps=FD_STEPS, central=False):
        if propagator.particle.charge == 0:
            raise ValueError(f"{propagator.particle.name} is neutral: it leaves no hits to fit")
        if propagator.adaptive:
            warnings.warn(f"The Monte Carlo used adaptive steps; the Kalman transport uses fixed steps of "
                          f"{propagator.step} mm (its maximum step) for smooth Jacobians. Check that this "
                          "step is converged (see src/kalman/transport.py)", stacklevel=2)
        self.propagator = copy.copy(propagator)    # shallow: field, geometry and slices are shared
        self.propagator.adaptive = False
        self.fd_steps = np.asarray(fd_steps, dtype=float)
        self.central = central


    # x: state (x, tx, q/p) at z_a
    # z_a, z_b: [mm] start and end, z_b >= z_a
    # Returns: (x_pred at z_b, F, Q)
    # Raises TrackStopped if the mean track stops before z_b
    def predict(self, x, z_a, z_b):
        prop = self.propagator
        x = np.asarray(x, dtype=float)
        F = np.eye(3)
        Q = np.zeros((3, 3))
        if z_b - z_a <= TOL:
            return x.copy(), F, Q

        current = x.copy()
        slice_entry = current      # state at the entrance of the current slice
        z_cuts = prop.cuts(z_a, z_b)
        for za, zb in zip(z_cuts[:-1].tolist(), z_cuts[1:].tolist()):
            if prop._is_edge(za, prop._slice_starts):
                slice_entry = current
            new, F_seg = self.segment(current, za, zb)
            F = F_seg @ F
            Q = F_seg @ Q @ F_seg.T
            if prop._is_edge(zb, prop._slice_ends):
                Q[:2, :2] += self.slice_noise(slice_entry, new, zb)
            current = new

        return current, F, Q


    # Propagates the state x from za to zb (one piece, inside one region) and its Jacobian.
    # Near the end of the range (e.g. after a range constraint, src/kalman/constraints.py) the state
    # may reach zb while a perturbed one stops: then the difference is taken on the other side
    def segment(self, x, za, zb):
        new = self._propagate(x, za, zb)
        steps = self.fd_steps.copy()
        steps[2] *= abs(x[2])
        F = np.empty((3, 3))
        for j in range(3):
            dx = np.zeros(3)
            dx[j] = steps[j]
            plus = self._try_propagate(x + dx, za, zb)
            minus = self._try_propagate(x - dx, za, zb) if self.central else None
            if self.central and plus is not None and minus is not None:
                F[:, j] = (plus - minus) / (2 * steps[j])
            elif plus is not None:
                F[:, j] = (plus - new) / steps[j]
            else:
                minus = minus if minus is not None else self._propagate(x - dx, za, zb)
                F[:, j] = (new - minus) / steps[j]
        return new, F


    def _try_propagate(self, x, za, zb):
        # As _propagate, but None if the track stops
        try:
            return self._propagate(x, za, zb)
        except TrackStopped:
            return None


    # True if the mean track that starts at z_a with the state x (its q/p replaced by sign(q) * s,
    # s = |q|/p)
    # reaches z_b, i.e. does not stop in the phantom on the way
    def reaches(self, x, s, z_a, z_b):
        prop = self.propagator
        q_p = np.sign(prop.particle.charge) * s
        result = prop.propagate(State(z=z_a, x=float(x[0]), tx=float(x[1]), q_p=float(q_p)), z_b)
        return not result.stopped


    # True if there is phantom material between z_a and z_b (elsewhere a track cannot stop)
    def crosses_material(self, z_a, z_b):
        g = self.propagator.geometry
        return g.has_phantom and z_a < g.z_end_phantom - TOL and z_b > g.z_start_phantom + TOL


    # Boundary s* = |q|/p at z_a (see src/kalman/constraints.py): the mean track that starts at z_a with
    # the position and slope of x reaches z_b if and only if s < s*.
    # Found by bisection in s with the propagator itself, so the criterion of stopping is exactly the
    # one of the Monte Carlo (CSDA range, stop margin, layers, path length sqrt(1 + tx^2), field).
    # s_tol: [e/MeV] precision of s*
    # Returns: s*, or np.inf if the track cannot stop between z_a and z_b (no material, or it reaches
    #          z_b even at the lowest energy of the model)
    def reach_limit(self, x, z_a, z_b, s_tol):
        prop = self.propagator
        if z_b - z_a <= TOL or not self.crosses_material(z_a, z_b):
            return np.inf
        m = prop.particle.mass
        s_of_T = lambda T: abs(prop.particle.charge) / momentum(T, prop.particle)   # s = |q|/p, as reaches

        # Bracket: s_stop (stops) > s* >= s_reach (reaches). The slowest particle of the model
        T_lo = E_MIN_MEV * (1.0 + 1e-6)
        if self.reaches(x, s_of_T(T_lo), z_a, z_b):
            return np.inf
        s_stop = s_of_T(T_lo)
        T_hi = max(1.0, kinetic_energy(x[2], prop.particle))
        while not self.reaches(x, s_of_T(T_hi), z_a, z_b):
            s_stop = s_of_T(T_hi)
            T_hi *= 2.0
            if T_hi > 1.0e6 * max(1.0, m):
                raise RuntimeError(f"The track does not reach z = {z_b} mm from z = {z_a} mm at any energy")
        s_reach = s_of_T(T_hi)

        # Bisection (geometric mean: s spans decades)
        for _ in range(MAX_BISECTIONS):
            if s_stop - s_reach <= s_tol:
                break
            s_mid = np.sqrt(s_stop * s_reach)
            if not s_reach < s_mid < s_stop:     # floating-point resolution reached
                break
            if self.reaches(x, s_mid, z_a, z_b):
                s_reach = s_mid
            else:
                s_stop = s_mid
        return 0.5 * (s_stop + s_reach)


    def _propagate(self, x, za, zb):
        state, stopped = self.propagator.propagate_segment(State(z=za, x=x[0], tx=x[1], q_p=x[2]), za, zb)
        if stopped:
            raise TrackStopped(f"the predicted track stops at z = {state.z:.2f} mm (before {zb:.2f} mm)")
        return np.array([state.x, state.tx, state.q_p])


    # 2x2 covariance of the (dx, dtx) kick at the end of the slice ending at z_end, evaluated on the
    # mean track as Propagator.scatter does: mean momentum and slope between entry and exit.
    # dz is the full slice length (the Monte Carlo kicks once per slice, even if the prediction
    # started inside it, e.g. from a detector plane placed inside a slice)
    def slice_noise(self, entry, exit, z_end):
        prop = self.propagator
        z_s, z_e, material_name, dz_layer = prop.slices[int(np.argmin(np.abs(prop._slice_ends - z_end)))]
        dz = z_e - z_s
        q = prop.particle.charge
        p_mid = 0.5 * (abs(q / entry[2]) + abs(q / exit[2]))
        q_p_mid = np.sign(exit[2]) * abs(q) / p_mid
        tx_mid = 0.5 * (entry[1] + exit[1])
        theta0 = theta0_slice(q_p_mid, tx_mid, dz, dz_layer, material_object(material_name), prop.particle)
        return slice_covariance(theta0, tx_mid, dz)

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
# Units: mm, MeV; q in e, q/p in e/MeV.

import numpy as np

from src.geometry.geometry import TOL
from src.phantom.materials import material_object
from src.propagator.propagator import State
from src.propagator.scattering import slice_covariance, theta0_slice

# Default finite-difference steps for (x [mm], tx, q/p [relative])
FD_STEPS = (1.0e-4, 1.0e-7, 1.0e-6)


class TrackStopped(Exception):
    # The predicted (mean) track stops in the phantom before z_b
    pass


class Transport:

    # propagator: Propagator of the Monte Carlo (same geometry, field, steps and max_slice).
    #             Without max_slice there are no slices: Q = 0
    # fd_steps: finite-difference steps for (x [mm], tx, q/p [relative to |q/p|])
    # central: central differences (2 extra propagations per variable) instead of forward ones (1)
    def __init__(self, propagator, fd_steps=FD_STEPS, central=False):
        if propagator.particle.charge == 0:
            raise ValueError(f"{propagator.particle.name} is neutral: it leaves no hits to fit")
        self.propagator = propagator
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


    # Propagates the state x from za to zb (one piece, inside one region) and its Jacobian
    def segment(self, x, za, zb):
        new = self._propagate(x, za, zb)
        steps = self.fd_steps.copy()
        steps[2] *= abs(x[2])
        F = np.empty((3, 3))
        for j in range(3):
            dx = np.zeros(3)
            dx[j] = steps[j]
            if self.central:
                F[:, j] = (self._propagate(x + dx, za, zb) - self._propagate(x - dx, za, zb)) / (2 * steps[j])
            else:
                F[:, j] = (self._propagate(x + dx, za, zb) - new) / steps[j]
        return new, F


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

# src/propagator/scattering.py
#
# Multiple Coulomb scattering in the phantom, slice by slice. Units: mm, MeV; q in e, q_p in e/MeV.
#
# The same model is used by:
#   - the Monte Carlo: sample_kick draws (dx, dtx) from the covariance of each slice
#   - the Kalman filter: slice_covariance is the process noise Q of that slice
# so the scattering that is simulated is exactly the one the filter expects.
#
# 1. RMS projected angle of a slice (Highland), with the LOCAL momentum of the slice:
#
#       theta0_i^2 = (13.6 z / (beta_i p_i))^2 * (s_i / X0) * [1 + 0.038 ln(s_layer / X0)]^2
#
#    s_i = dz_i sqrt(1 + tx^2) is the path length in the slice, s_layer the one in the whole layer.
#    The log term is not additive, so it is evaluated for the whole layer: with a constant momentum
#    the slices add up in quadrature to the Highland width of the layer (as highland_width with
#    n_slices), whatever the number of slices. With the local momentum the angle grows as the
#    particle slows down. Same log convention as src.phantom.materials.highland_width.
#
# 2. Covariance of (dx, dtx) at the end of a slice of length dz along z. For a thick scatterer the
#    track is displaced as well as deflected; in terms of the projected angle theta and the
#    displacement perpendicular to the track, with path length s:
#
#       Var(theta) = theta0^2,  Var(d_perp) = theta0^2 s^2 / 3,  Cov = theta0^2 s / 2
#
#    With tx = tan(theta): dtx = (1 + tx^2) dtheta, and at fixed z dx = d_perp sqrt(1 + tx^2).
#    Using s = dz sqrt(1 + tx^2):
#
#       Cov(dx, dtx) = theta0^2 (1 + tx^2)^2 [[dz^2 / 3, dz / 2],
#                                             [dz / 2,   1     ]]
#
# Energy-loss straggling is not included (q/p is not kicked).

import numpy as np

from src.constants import HIGHLAND_CONST, PROTON, Material, Particle
from src.geometry.geometry import Geometry, TOL
from src.utils.validation import check_real

LOG_COEFF = 0.038     # coefficient of the log term of the Highland formula


def theta0_slice(q_p, tx, dz, dz_layer, material: Material, particle: Particle = PROTON):
    # RMS projected scattering angle [rad] of one slice (see 1. above)
    # q_p: [e/MeV] q/p at the slice (e.g. at its entrance); tx: dx/dz at the slice
    # dz: [mm] length of the slice along z; dz_layer: [mm] length along z of the whole layer
    # Returns 0 for a neutral particle
    if particle.charge == 0:
        return 0.0
    if not 0.0 < dz <= dz_layer + TOL:
        raise ValueError(f"Need 0 < dz <= dz_layer, got dz = {dz} mm, dz_layer = {dz_layer} mm")

    p = abs(particle.charge / q_p)
    beta = p / np.sqrt(p**2 + particle.mass**2)
    path_factor = np.sqrt(1.0 + tx**2)                    # ds/dz
    s, s_layer = dz * path_factor, dz_layer * path_factor

    log_term = 1.0 + LOG_COEFF * np.log(s_layer / material.X0)
    return HIGHLAND_CONST * abs(particle.charge) / (beta * p) * np.sqrt(s / material.X0) * log_term


def slice_covariance(theta0, tx, dz):
    # 2x2 covariance of (dx, dtx) at the end of a slice of length dz [mm] along z (see 2. above)
    # theta0: [rad] RMS projected angle of the slice (theta0_slice); tx: dx/dz at the slice
    a2 = theta0**2 * (1.0 + tx**2)**2
    return a2 * np.array([[dz**2 / 3.0, dz / 2.0],
                          [dz / 2.0,    1.0     ]])


def sample_kick(theta0, tx, dz, rng):
    # One random (dx, dtx) with covariance slice_covariance(theta0, tx, dz)
    # rng: numpy Generator
    # Two independent N(0, 1): u gives the angle, v the extra displacement
    #   dtx = a u,   dx = a dz (u / 2 + v / (2 sqrt(3))),   a = theta0 (1 + tx^2)
    u, v = rng.standard_normal(2)
    a = theta0 * (1.0 + tx**2)
    return a * dz * (0.5 * u + v / (2.0 * np.sqrt(3.0))), a * u


def phantom_slices(geometry: Geometry, max_slice):
    # Slices of the phantom where a kick is applied at the end
    # max_slice: [mm] maximum length of a slice along z. Each layer is cut into the fewest equal
    #            slices not longer than max_slice, so a slice never crosses an interface
    # Returns: list of (z_a, z_b, material_name, dz_layer), in beam order. Empty without phantom
    max_slice = check_real("max_slice", max_slice, minimum=TOL, strict=True)
    if not geometry.has_phantom:
        return []

    slices = []
    z_interfaces = geometry.z_phantom_interfaces
    for layer, z_a, z_b in zip(geometry.phantom_layers, z_interfaces[:-1], z_interfaces[1:]):
        dz_layer = float(z_b - z_a)
        n = max(1, int(np.ceil(dz_layer / max_slice - TOL)))
        edges = np.linspace(z_a, z_b, n + 1)
        slices += [(float(a), float(b), layer["material"], dz_layer) for a, b in zip(edges[:-1], edges[1:])]
    return slices

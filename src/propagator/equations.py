# Equations of motion for each region. Units: mm, MeV, T; q in e, q_p = q/p in e/MeV.
import numpy as np

from src.constants import KAPPA, PROTON, Particle, Material
from src.phantom.materials import bethe_bloch


def propagate_free_space(x0, tx, z0, z1):
    """Straight line from z0 to z1: no field, no material, so tx and q/p are unchanged.

    Exact solution of x'' = 0, so no Runge-Kutta is needed here.
    Returns (x1, tx).
    """
    return x0 + tx * (z1 - z0), tx


def eom_magnet(field, q_p, particle: Particle = PROTON, kappa=KAPPA, x_half_gap=np.inf):
    """x'' = f(z, x, x'); |p| is constant inside the field.

    For |x| > x_half_gap (outside the gap between the plates) the field is taken as zero,
    so the track there is a straight line.
    """
    def fun_magnet(z, x, xprime):
        By = np.where(np.abs(x) <= x_half_gap, field.By(x, z), 0.0)
        return -kappa * q_p * (1 + xprime**2)**1.5 * By
    return fun_magnet


# This is the eom for the phantom but assuming that there can be a remaining field from the magnet
# still present !!!
def eom_phantom_field(field, material: Material, particle: Particle = PROTON, kappa=KAPPA,
                      x_half_gap=np.inf):
    """y' = f(z, y) with y = (x, tx, q/p): energy loss and magnetic field at the same time.

        x'   = tx
        tx'  = -kappa (q/p) (1 + tx^2)^(3/2) By(x, z)
        q/p' = (1/q) (q/p)^2 (S / beta) sqrt(1 + tx^2),    S = -dE/ds > 0 (Bethe-Bloch)

    Nothing is constant: the curvature grows as the particle slows down, and the path length
    per unit z changes as tx changes. For |x| > x_half_gap the field is taken as zero.
    """
    q, m = particle.charge, particle.mass

    def fun_phantom_field(z, y):
        x, tx, q_p = y
        By = field.By(x, z) if abs(x) <= x_half_gap else 0.0
        p = q / q_p
        E = np.sqrt(p**2 + m**2)
        beta = abs(p) / E
        S = bethe_bloch(E - m, material, particle)  # -dE/ds > 0 [MeV/mm]
        path_factor = np.sqrt(1 + tx**2)            # ds/dz
        return np.array([
            tx,
            -kappa * q_p * path_factor**3 * By,
            (q_p**2 / q) * S / beta * path_factor,
        ])
    return fun_phantom_field

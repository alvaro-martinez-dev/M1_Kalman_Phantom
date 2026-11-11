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


def eom_magnet(field, q_p, particle: Particle = PROTON, kappa=KAPPA):
    """x'' = f(z, x, x'); |p| is constant inside the field."""
    def fun_magnet(z, x, xprime):
        return -kappa * q_p * (1 + xprime**2)**1.5 * field.By(x, z)
    return fun_magnet


def eom_phantom(tx, material: Material, particle: Particle = PROTON):
    """(q/p)' = f(z, q/p); tx is constant (multiple scattering is added apart)."""
    q, m = particle.charge, particle.mass
    path_factor = np.sqrt(1 + tx**2)              # ds/dz

    def fun_phantom(z, q_p):
        p = q / q_p
        E = np.sqrt(p**2 + m**2)
        beta = abs(p) / E
        S = bethe_bloch(E - m, material, particle)  # -dE/ds > 0 [MeV/mm]
        return (q_p**2 / q) * S / beta * path_factor
    return fun_phantom

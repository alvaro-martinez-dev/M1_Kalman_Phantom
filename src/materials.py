"""
Materials for proton tracking: energy loss (Bethe-Bloch) and
multiple scattering (Highland).

Units: energy MeV, length mm.
Material definitions come from src.constants (single source of truth).
"""
import numpy as np
from src.constants import (
    K_BETHE, M_ELECTRON, PROTON, HIGHLAND_CONST,
    Material, WATER,
)

M_P = PROTON.mass  # MeV/c^2
M_E = M_ELECTRON   # MeV/c^2


def beta_gamma(E_kin_MeV):
    """Return (beta, gamma) for a proton of kinetic energy E_kin."""
    gamma = (E_kin_MeV + M_P) / M_P
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    return beta, gamma


def bethe_bloch(E_kin_MeV, material):
    """
    Mean energy loss dE/ds for a proton in `material`, in MeV/mm.

    Per unit *path length* s. To get dE/dz, multiply by
    ds/dz = sqrt(1 + tx^2).
    """
    beta, gamma = beta_gamma(E_kin_MeV)
    T_max = (2.0 * M_E * beta**2 * gamma**2
             / (1.0 + 2.0 * gamma * M_E / M_P + (M_E / M_P)**2))
    term = (0.5 * np.log(2.0 * M_E * beta**2 * gamma**2 * T_max
                         / material.I**2)
            - beta**2)
    # K_BETHE is in MeV cm^2 / mol; density in g/cm^3.
    # dE/ds in MeV/cm = K * (Z/A) * rho / beta^2 * term.
    # To get MeV/mm, divide by 10.
    dEdx_cm = K_BETHE * material.Z_A * material.density / beta**2 * term
    return dEdx_cm / 10.0


def range_csda(E_kin_MeV, material, n_steps=20000):
    """
    CSDA range of a proton in `material`, in mm.

    Integrates dE / (dE/ds) from E_kin down to 0.
    """
    E = float(E_kin_MeV)
    dE = E / n_steps
    rng = 0.0
    for _ in range(n_steps):
        rng += dE / bethe_bloch(E, material)
        E -= dE
        if E <= 0.0:
            break
    return rng


def highland_width(E_kin_MeV, thickness_mm, material, charge=1):
    """
    RMS scattering angle (rad) of a proton after crossing `thickness_mm`
    of `material`, from the Highland formula.
    """
    beta, _ = beta_gamma(E_kin_MeV)
    p_MeV = np.sqrt((E_kin_MeV + M_P)**2 - M_P**2)  # MeV/c
    x_over_X0 = thickness_mm / material.X0  # both in mm
    return (HIGHLAND_CONST / (beta * p_MeV) * charge
            * np.sqrt(x_over_X0)
            * (1.0 + 0.038 * np.log(x_over_X0)))
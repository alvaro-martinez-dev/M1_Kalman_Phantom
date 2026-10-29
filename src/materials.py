
"""
Materials for proton tracking: energy loss (Bethe-Bloch) and
multiple scattering (Highland).

Units:
  - energy: MeV
  - length: cm
  - density: g/cm^3

Reference: PDG, Passage of particles through matter.
Validation: NIST PSTAR database.
"""
from dataclasses import dataclass
import numpy as np

# Physical constants
K_BB = 0.307075       # MeV mol^-1 cm^2
M_E = 0.51099895      # MeV/c^2
M_P = 938.27208816    # MeV/c^2


@dataclass(frozen=True)
class Material:
    """A material with its Bethe-Bloch and Highland parameters."""
    name: str
    rho: float       # g/cm^3
    Z_over_A: float  # dimensionless
    I: float         # mean excitation energy, MeV
    X0: float        # radiation length, cm


# --- Material table (add more as needed) ---
WATER = Material("water", rho=1.0,  Z_over_A=0.555, I=75e-6,  X0=36.08)
BONE  = Material("bone",  rho=1.85, Z_over_A=0.514, I=91.9e-6, X0=10.6)


def beta_gamma(E_kin_MeV):
    """Return (beta, gamma) for a proton of kinetic energy E_kin."""
    gamma = (E_kin_MeV + M_P) / M_P
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    return beta, gamma


def bethe_bloch(E_kin_MeV, material):
    """
    Mean energy loss dE/ds for a proton in `material`, in MeV/cm.

    This is per unit *path length* s. To get dE/dz, multiply by
    ds/dz = sqrt(1 + tx^2).
    """
    beta, gamma = beta_gamma(E_kin_MeV)
    T_max = (2.0 * M_E * beta**2 * gamma**2
             / (1.0 + 2.0 * gamma * M_E / M_P + (M_E / M_P)**2))
    term = (0.5 * np.log(2.0 * M_E * beta**2 * gamma**2 * T_max
                         / material.I**2)
            - beta**2)
    return K_BB * material.Z_over_A * material.rho / beta**2 * term


def range_csda(E_kin_MeV, material, n_steps=20000):
    """
    CSDA range of a proton in `material`, in cm.

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


def highland_width(E_kin_MeV, thickness_cm, material, charge=1):
    """
    RMS scattering angle (rad) of a proton after crossing `thickness_cm`
    of `material`, from the Highland formula.
    """
    beta, _ = beta_gamma(E_kin_MeV)
    p_MeV = np.sqrt((E_kin_MeV + M_P)**2 - M_P**2)  # MeV/c
    x_over_X0 = thickness_cm / material.X0
    return (13.6 / (beta * p_MeV) * charge
            * np.sqrt(x_over_X0)
            * (1.0 + 0.038 * np.log(x_over_X0)))

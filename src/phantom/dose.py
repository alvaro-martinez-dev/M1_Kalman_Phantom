"""
Deposited dose from energy loss.

Dose is energy deposited per unit mass.
Unit conversions:
    1 MeV/g = 1.602176634e-10 Gy
    1 Gy    = 1 J/kg
"""
import numpy as np

from src.constants import Material
from src.phantom.materials import bethe_bloch, range_csda

# 1 MeV/g = 1.602176634e-10 Gy
MEV_PER_G_TO_GY = 1.602176634e-10


def dose_per_length(T_MeV, material: Material, particle=None):
    """
    Dose per unit path length, D' = (1/rho) * dE/ds, in MeV/g/mm.
    """
    kwargs = {} if particle is None else {"particle": particle}
    dEdx = bethe_bloch(T_MeV, material, **kwargs)  # MeV/mm
    return dEdx / material.density                  # MeV/g/mm


def dose_in_step(T_MeV, step_mm, material: Material, particle=None):
    """
    Dose deposited by a particle crossing `step_mm` of `material`
    at kinetic energy T_MeV, in Gy.

    Per unit area (transverse), the mass of the step is rho * step_mm
    (in g/cm^2 after the mm->cm conversion), and the energy lost is
    dE = (dE/dx) * step_mm. So D = dE / mass_per_area.
    """
    kwargs = {} if particle is None else {"particle": particle}
    dEdx = bethe_bloch(T_MeV, material, **kwargs)       # MeV/mm
    dE = dEdx * step_mm                                  # MeV
    mass_per_area = material.density * step_mm * 0.1     # g/cm^2
    D_MeV_per_g = dE / mass_per_area                     # MeV/g
    return D_MeV_per_g * MEV_PER_G_TO_GY                 # Gy


def bragg_peak_depth(T_MeV, material: Material, particle=None):
    """
    Approximate depth (mm) of the Bragg peak: where the particle stops.
    Uses the CSDA range.
    """
    kwargs = {} if particle is None else {"particle": particle}
    return range_csda(T_MeV, material, **kwargs)
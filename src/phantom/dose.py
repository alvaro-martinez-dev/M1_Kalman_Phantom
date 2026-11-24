"""
Deposited dose from energy loss.

Dose is energy deposited per unit mass.
Unit conversions:
    1 MeV/g = 1.602176634e-10 Gy
    1 Gy    = 1 J/kg
    "Dose in Gy for a fluence of 1 particle/cm². Multiply by your actual fluence to get the physical dose."

The particle is always required (no default): forgetting it must raise an error instead of
silently computing the dose of a proton (see src/phantom/materials.py).
"""
import numpy as np

from src.constants import Material, Particle
from src.phantom.materials import E_MIN_MEV, material_object
from src.phantom.materials import bethe_bloch, range_csda

# 1 MeV/g = 1.602176634e-10 Gy
MEV_PER_G_TO_GY = 1.602176634e-10


def dose_per_length(T_MeV, material: Material, particle: Particle):
    """
    Dose per unit path length, D' = (1/rho) * dE/ds, in MeV/g/mm.
    """
    dEdx = bethe_bloch(T_MeV, material, particle)  # MeV/mm
    return dEdx / material.density                  # MeV/g/mm


def dose_in_step(T_MeV, step_mm, material: Material, particle: Particle):
    """
    Dose deposited by a particle crossing `step_mm` of `material`
    at kinetic energy T_MeV, in Gy.

    Per unit area (transverse), the mass of the step is rho * step_mm
    (in g/cm^2 after the mm->cm conversion), and the energy lost is
    dE = (dE/dx) * step_mm. So D = dE / mass_per_area.
    """
    dEdx = bethe_bloch(T_MeV, material, particle)       # MeV/mm
    dE = dEdx * step_mm                                  # MeV
    mass_per_area = material.density * step_mm * 0.1     # g/cm^2
    D_MeV_per_g = dE / mass_per_area                     # MeV/g
    return D_MeV_per_g * MEV_PER_G_TO_GY                 # Gy


def bragg_peak_depth(T_MeV, material: Material, particle: Particle):
    """
    Approximate depth (mm) of the Bragg peak: where the particle stops.
    Uses the CSDA range.
    """
    return range_csda(T_MeV, material, particle)

def dose_profile(geometry, T0_MeV, particle: Particle, n_steps=200):
    """
    Local dose profile along z (Bragg curve).
    Returns the dose deposited in each step (Gy), NOT the cumulative dose.
    """
    if not geometry.has_phantom:
        raise ValueError("dose_profile needs a phantom: the geometry has none (phantom disabled)")

    # --- ESTAS LÍNEAS FALTABAN ---
    z_arr = np.linspace(geometry.z_start_phantom, geometry.z_end_phantom, n_steps + 1)
    dz = z_arr[1] - z_arr[0]
    # -----------------------------

    T = float(T0_MeV)
    T_arr = np.zeros_like(z_arr)
    D_local = np.zeros_like(z_arr)  # <--- Ahora es local, no acumulada

    for i, z in enumerate(z_arr):
        if T <= E_MIN_MEV:
            T_arr[i:] = 0.0
            D_local[i:] = 0.0
            break
        T_arr[i] = T
        mat_name = geometry.material_at(z)
        if mat_name is None:
            D_local[i] = 0.0
            continue
        mat = material_object(mat_name)

        D_local[i] = dose_in_step(T, dz, mat, particle)

        dEdx = bethe_bloch(T, mat, particle)
        T = max(T - dEdx * dz, 0.0)

    return z_arr, T_arr, D_local

def check_dose_limits(dose_Gy: float):
    """
    Compares the calculated dose with international standards.
    Reference: ICRP Publication 103 (2007) for radiation protection limits.
    Reference: ICRU Report 83 for radiotherapy (typical tumor dose is ~2 Gy per fraction).
    """
    # Public exposure limit: 1 mSv/year = 0.001 Gy
    # Occupational limit: 20 mSv/year = 0.020 Gy
    # Therapy limit (single fraction): 2 Gy

    if dose_Gy < 0.001:
        print(f"Dose {dose_Gy:.2e} Gy is BELOW the public safety limit. Safe.")
    elif dose_Gy < 0.02:
        print(f"Dose {dose_Gy:.2e} Gy is within OCCUPATIONAL limits.")
    elif dose_Gy < 2.0:
        print(f"Dose {dose_Gy:.2e} Gy is above protection limits but below a typical RADIOTHERAPY fraction (2 Gy).")
    else:
        print(f"Dose {dose_Gy:.2e} Gy EXCEEDS typical radiotherapy fraction. Ensure this is intended for target volume only.")

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

def dose_profile(geometry, T0_MeV, n_steps=200, particle=None):
    """
    Dose profile along z for a proton of initial kinetic energy T0_MeV
    crossing the phantom described by `geometry`.

    Simplified: at each step, look up the material at z, compute dE/dx,
    deposit that dose, and update the kinetic energy. Stops when T
    reaches E_MIN_MEV (the proton is stopped).

    Returns
    -------
    z_arr : ndarray
        z positions (mm) inside the phantom.
    T_arr : ndarray
        Kinetic energy (MeV) at each z.
    D_cumulative : ndarray
        Cumulative dose (Gy) up to each z.
    """
    import numpy as np
    from src.phantom.materials import E_MIN_MEV, material_object

    if not geometry.has_phantom:
        raise ValueError("dose_profile needs a phantom: the geometry has none (phantom disabled)")

    z_arr = np.linspace(geometry.z_start_phantom,
                        geometry.z_end_phantom, n_steps + 1)
    dz = z_arr[1] - z_arr[0]

    T = float(T0_MeV)
    T_arr = np.zeros_like(z_arr)
    D_cumulative = np.zeros_like(z_arr)
    for i, z in enumerate(z_arr):
        if T <= E_MIN_MEV:
            # Proton stopped: remaining kinetic energy is zero, and the
            # dose already deposited is constant from here on.
            T_arr[i:] = 0.0
            D_cumulative[i:] = D_cumulative[i - 1] if i > 0 else 0.0
            break
        T_arr[i] = T
        mat_name = geometry.material_at(z)
        if mat_name is None:
            # Outside the phantom: no dose, keep the cumulative value
            D_cumulative[i] = D_cumulative[i - 1] if i > 0 else 0.0
            continue
        mat = material_object(mat_name)
        D_cumulative[i] = D_cumulative[i - 1] if i > 0 else 0.0
        D_cumulative[i] += dose_in_step(T, dz, mat, particle=particle)
        dEdx = bethe_bloch(T, mat, **({} if particle is None else {"particle": particle}))
        T = max(T - dEdx * dz, 0.0)

    return z_arr, T_arr, D_cumulative
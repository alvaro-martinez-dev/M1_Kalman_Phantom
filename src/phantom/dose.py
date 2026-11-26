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
from scipy.optimize import brentq

from src.constants import Material, Particle
from src.phantom.materials import E_MIN_MEV, material_object
from src.phantom.materials import bethe_bloch, range_csda, range_csda_fast

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

# Kinetic energy [MeV] whose CSDA range in `material` is R_mm (inverse of range_csda_fast), for
# 0 < R_mm < range_csda_fast(T_hi). Returns 0.0 for R_mm <= 0 (the particle has stopped)
def energy_from_range(R_mm, material: Material, particle: Particle, T_hi):
    if R_mm <= 0.0:
        return 0.0
    return brentq(lambda T: range_csda_fast(T, material, particle) - R_mm, E_MIN_MEV, T_hi,
                  xtol=1e-12, rtol=1e-12)


def dose_profile(geometry, T0_MeV, particle: Particle, n_steps=200):
    """
    Local dose profile along z through the phantom (Bragg curve), CSDA, along a straight line.

    The kinetic energy at each z is exact within the CSDA (no step error): in each layer the
    residual range is R_layer(T_in) - (z - z_layer_start), inverted with energy_from_range.
    The dose of each step is the energy lost in it divided by its mass per unit area (a step that
    crosses an interface takes the mass of each material), so the steps add up to the energy of the
    particle: sum(D_local * mass_per_area) = T0 if it stops in the phantom (energy conservation).

    Returns (z_arr, T_arr, D_local), each of n_steps + 1 values:
        z_arr: [mm] edges of the steps, from the entrance to the exit of the phantom
        T_arr: [MeV] kinetic energy at z_arr (0 once the particle has stopped)
        D_local: [Gy per particle/cm^2] mean dose in the step [z_arr[i], z_arr[i+1]], NOT cumulative
                 (the last value, after the exit, is 0). The step where the particle stops gets all
                 its remaining energy (also the part below E_MIN_MEV, where Bethe-Bloch is not used)
    Not included: range straggling, nuclear interactions, the spread of the beam.
    """
    if not geometry.has_phantom:
        raise ValueError("dose_profile needs a phantom: the geometry has none (phantom disabled)")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    z_arr = np.linspace(geometry.z_start_phantom, geometry.z_end_phantom, n_steps + 1)
    interfaces = geometry.z_phantom_interfaces
    materials = [material_object(layer["material"]) for layer in geometry.phantom_layers]

    # Energy at the entrance of each layer, and the residual range it gives in that layer
    T_in = [float(T0_MeV)]
    for k, mat in enumerate(materials):
        R = range_csda_fast(T_in[k], mat, particle) if T_in[k] > E_MIN_MEV else 0.0
        T_in.append(energy_from_range(R - (interfaces[k + 1] - interfaces[k]), mat, particle, T_in[k]))

    def energy_at(z):
        k = min(int(np.searchsorted(interfaces, z, side="right")) - 1, len(materials) - 1)
        if T_in[k] <= 0.0:
            return 0.0
        R = range_csda_fast(T_in[k], materials[k], particle) if T_in[k] > E_MIN_MEV else 0.0
        return energy_from_range(R - (z - interfaces[k]), materials[k], particle, T_in[k])

    T_arr = np.array([energy_at(z) for z in z_arr])
    T_arr[-1] = T_in[-1]      # exactly the energy at the exit (0 if it stopped)

    # Mass per unit area of each step [g/cm^2]: each material in it with its own density
    edges = np.union1d(z_arr, interfaces)
    rho = np.array([materials[min(int(np.searchsorted(interfaces, 0.5 * (a + b), side="right")) - 1,
                                  len(materials) - 1)].density for a, b in zip(edges[:-1], edges[1:])])
    mass_fine = rho * np.diff(edges) * 0.1                            # mm -> cm
    step_of_fine = np.searchsorted(z_arr, edges[:-1], side="right") - 1
    mass_per_area = np.bincount(step_of_fine, weights=mass_fine, minlength=n_steps)[:n_steps]

    D_local = np.zeros_like(z_arr)
    D_local[:-1] = (T_arr[:-1] - T_arr[1:]) / mass_per_area * MEV_PER_G_TO_GY
    return z_arr, T_arr, D_local


# ICRP 103 radiation weighting factor w_R: equivalent dose H [Sv] = w_R * absorbed dose D [Gy]
# kind: "photon", "electron", "muon", "proton", "pion", "alpha", "ion" or "neutron"
# E_neutron_MeV: neutron energy [MeV], needed only for neutrons (continuous function of ICRP 103)
def radiation_weighting_factor(kind, E_neutron_MeV=None):
    fixed = {"photon": 1.0, "electron": 1.0, "muon": 1.0, "proton": 2.0, "pion": 2.0,
             "alpha": 20.0, "ion": 20.0}
    if kind in fixed:
        return fixed[kind]
    if kind != "neutron":
        raise ValueError(f"Unknown radiation kind {kind!r}: one of {sorted(fixed) + ['neutron']}")
    if E_neutron_MeV is None or E_neutron_MeV <= 0:
        raise ValueError("The w_R of neutrons depends on their energy: give E_neutron_MeV > 0")
    E = E_neutron_MeV
    if E < 1.0:
        return 2.5 + 18.2 * np.exp(-np.log(E)**2 / 6.0)
    if E <= 50.0:
        return 5.0 + 17.0 * np.exp(-np.log(2.0 * E)**2 / 6.0)
    return 2.5 + 3.25 * np.exp(-np.log(0.04 * E)**2 / 6.0)


def equivalent_dose(dose_Gy, kind, E_neutron_MeV=None):
    """Equivalent dose [Sv] of an absorbed dose [Gy] of one kind of radiation (ICRP 103)."""
    return radiation_weighting_factor(kind, E_neutron_MeV) * dose_Gy


# ICRP 103 annual limits of EFFECTIVE dose, for radiation protection (planned exposure)
LIMIT_PUBLIC_SV_PER_YEAR = 1.0e-3
LIMIT_OCCUPATIONAL_SV_PER_YEAR = 20.0e-3


def check_dose_limits(dose_Sv_per_year: float):
    """
    Compares an annual EFFECTIVE dose [Sv/year] (e.g. behind a shielding wall) with the ICRP 103
    limits: 1 mSv/year for the public, 20 mSv/year for workers (averaged over 5 years).

    The limits are in sievert, not gray: convert an absorbed dose with equivalent_dose first (for
    neutrons w_R is up to ~20). They apply to radiation protection, never to the dose given to the
    patient, which is prescribed (e.g. ~2 Gy per fraction to the tumour, ICRU 78) and not limited.
    Returns: "public", "occupational" or "above" (the category the dose falls into), and prints it
    """
    if dose_Sv_per_year < LIMIT_PUBLIC_SV_PER_YEAR:
        category, text = "public", "below the public limit (1 mSv/year)"
    elif dose_Sv_per_year < LIMIT_OCCUPATIONAL_SV_PER_YEAR:
        category, text = "occupational", "above the public limit, below the occupational one (20 mSv/year)"
    else:
        category, text = "above", "ABOVE the occupational limit (20 mSv/year)"
    print(f"Effective dose {dose_Sv_per_year * 1e3:.3g} mSv/year: {text}")
    return category

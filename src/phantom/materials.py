"""
Materials for proton tracking: energy loss (Bethe-Bloch) and
multiple scattering (Highland).

Units: energy MeV, length mm.

Material and particle definitions come from src.constants
(single source of truth).
"""
import numpy as np
from scipy.integrate import quad
import src.constants as constants
from src.constants import (
    K_BETHE, M_ELECTRON, PROTON, HIGHLAND_CONST,
    Particle, Material,
)

# K_BETHE is in MeV cm^2 / mol and densities in g/cm^3, so the raw
# Bethe-Bloch formula gives MeV/cm. The project works in mm.
MEV_PER_CM_TO_MM = 0.1  # 1 MeV/cm = 0.1 MeV/mm

# Bethe-Bloch is valid for kinetic energies down to ~0.1 MeV.
# Below that the approximations break and the formula can give
# unphysical (even negative) dE/dx.
E_MIN_MEV = 0.1


def beta_gamma(T_MeV, particle: Particle = PROTON):
    """
    Return (beta, gamma) for a particle of *kinetic* energy T_MeV (MeV).

    Parameters
    ----------
    T_MeV : float or array_like
        Kinetic energy in MeV.
    particle : Particle
        Particle from src.constants (default: PROTON).
    """
    gamma = (T_MeV + particle.mass) / particle.mass
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    return beta, gamma

def density_effect(beta_gamma, material):
    """
    Sternheimer parameterization of the density-effect correction delta.

    Returns delta (dimensionless). If the material has no Sternheimer
    parameters (all None), returns 0 -- which is the correct limit at
    low beta*gamma and the default for materials where it is negligible.

    Reference: Sternheimer, Berger & Seltzer, Atomic Data and Nuclear
    Data Tables 30 (1984) 261; PDG, Passage of particles through matter.
    """
    if material.x0 is None:
        return 0.0
    x = np.log10(beta_gamma)
    if x < material.x0:
        return 0.0
    elif x < material.x1:
        return (2.0 * np.log(10.0) * x - material.Cbar
                + material.a * (material.x1 - x) ** material.k)
    else:
        return 2.0 * np.log(10.0) * x - material.Cbar

def bethe_bloch(T_MeV, material: Material, particle: Particle = PROTON):
    """
    Mean energy loss -dE/ds for a charged particle in `material`,
    in MeV/mm (per unit path length s).

    Parameters
    ----------
    T_MeV : float or array_like
        Kinetic energy in MeV. Must be >= E_MIN_MEV.
    material : Material
        From src.constants.
    particle : Particle
        From src.constants (default: PROTON).

    Returns
    -------
    dEdx : float or array_like
        Energy loss in MeV/mm (positive).

    Notes
    -----
    PDG formula (eq. 34.5, with delta=0):
        -dE/ds = K z^2 (Z/A) (1/beta^2)
                 [ 0.5 ln(2 m_e c^2 beta^2 gamma^2 T_max / I^2) - beta^2 ]
    where s is the path length and T is the *kinetic* energy.

    Valid for moderately relativistic charged particles heavier than
    electrons (muons, pions, protons, alphas, ...). NOT valid for
    electrons/positrons (bremsstrahlung dominates).
    """
    z = abs(particle.charge)
    if z == 0:
        raise ValueError(f"Particle {particle.name!r} is neutral: no Bethe-Bloch")
    if particle.mass < 10.0:
        raise NotImplementedError(
            f"Bethe-Bloch not implemented for {particle.name!r}: "
            "electrons lose energy mainly by bremsstrahlung"
        )
    if np.any(np.asarray(T_MeV) < E_MIN_MEV):
        raise ValueError(
            f"Kinetic energy below E_MIN = {E_MIN_MEV} MeV: "
            "Bethe-Bloch not valid"
        )

    beta, gamma = beta_gamma(T_MeV, particle)
    M = particle.mass
    T_max = (2.0 * M_ELECTRON * beta**2 * gamma**2
             / (1.0 + 2.0 * gamma * M_ELECTRON / M
                + (M_ELECTRON / M)**2))
    log_term = 0.5 * np.log(2.0 * M_ELECTRON * beta**2 * gamma**2
                            * T_max / material.I**2)
    bg = beta * gamma
    delta = density_effect(bg, material)
    dEdx_cm = (K_BETHE * z**2 * material.Z_A * material.density / beta**2 * (log_term - beta**2 - 0.5 * delta))
    
    return dEdx_cm * MEV_PER_CM_TO_MM


def range_csda(T_MeV, material: Material, particle: Particle = PROTON):
    """
    CSDA range of a particle in `material`, in mm.

    Integrates 1 / (dE/ds) from E_MIN_MEV to T_MeV using
    scipy.integrate.quad. The integral stops at E_MIN because
    Bethe-Bloch is not valid below it; the missing piece of range
    is < 1 um and negligible.

    Parameters
    ----------
    T_MeV : float
        Kinetic energy in MeV.
    material : Material
    particle : Particle

    Returns
    -------
    rng : float
        Range in mm.
    """
    def integrand(T):
        return 1.0 / bethe_bloch(T, material, particle=particle)

    E_lo = E_MIN_MEV * 1.0001  # small margin vs boundary checks
    rng, _ = quad(integrand, E_lo, T_MeV,
                  limit=200, epsabs=1e-10, epsrel=1e-10)
    return rng


def highland_width(T_MeV, thickness_mm, material: Material,
                   particle: Particle = PROTON, n_slices: int = 1):
    """
    RMS multiple-scattering angle (rad) after crossing `thickness_mm`
    of `material`, from the Highland formula:
        theta_0 = (13.6 MeV / (beta c p)) z sqrt(x/X0)
                  * [1 + 0.038 ln(x/X0)]

    Parameters
    ----------
    T_MeV : float
        Kinetic energy in MeV.
    thickness_mm : float
        Thickness of the material. Must be > 0.
    material : Material
    particle : Particle
    n_slices : int
        If 1 (default), return the Highland width for a *single layer*
        of the given thickness -- the physically correct value.
        If > 1, return the per-slice width such that `n_slices`
        independent kicks added in quadrature reproduce the single-layer
        width. Use this when the MC or the Kalman filter propagates in
        steps through a thick layer: the log term in Highland is NOT
        additive, so applying the formula slice-by-slice without this
        correction underestimates the total scattering.

    Returns
    -------
    theta_0 : float
        RMS angle in radians (always positive).
    """
    if thickness_mm <= 0:
        raise ValueError(
            f"thickness_mm must be > 0 (got {thickness_mm}): "
            "no scattering in zero thickness, and log(0) is undefined"
        )
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1 (got {n_slices})")

    z = abs(particle.charge)
    if z == 0:
        raise ValueError(f"Particle {particle.name!r} is neutral: no scattering")

    beta, _ = beta_gamma(T_MeV, particle)
    p_MeV = np.sqrt((T_MeV + particle.mass)**2 - particle.mass**2)
    x_over_X0 = thickness_mm / material.X0
    theta_total = (HIGHLAND_CONST / (beta * p_MeV) * z
                   * np.sqrt(x_over_X0)
                   * (1.0 + 0.038 * np.log(x_over_X0)))
    return theta_total / np.sqrt(n_slices)


def material_object(material_name: str) -> Material:
    """
    Map a material name (as stored in Geometry.phantom_layers, e.g.
    "WATER" or "BONE") to the corresponding Material object in
    src.constants.

    Geometry.check_layers already validates that the name refers to a
    Material in src.constants, so this is a safe typed accessor.
    """
    material = getattr(constants, material_name, None)
    if not isinstance(material, Material):
        raise ValueError(
            f"Unknown material: {material_name!r}. "
            f"Expected the name of a Material in src.constants."
        )
    return material
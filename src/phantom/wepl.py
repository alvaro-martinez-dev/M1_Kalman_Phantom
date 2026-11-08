"""
WEPL (water-equivalent path length) for a phantom described by Geometry.

WEPL is the thickness of water that would cause the same energy loss.
Two approximations are provided: one based on density, one based on
stopping-power ratios.
"""
from src.constants import WATER
from src.geometry.geometry import Geometry
from src.phantom.materials import material_object, bethe_bloch


def wepl_density(geometry: Geometry) -> float:
    """
    First-order WEPL (mm): sum_i thickness_i * (rho_i / rho_water).

    Good approximation for thin phantoms and high-energy protons, where
    the stopping-power ratio is close to the density ratio.
    """
    rho_water = WATER.density
    total = 0.0
    for layer in geometry.phantom_layers:
        mat = material_object(layer["material"])
        total += layer["thickness"] * (mat.density / rho_water)
    return total


def wepl_stopping_power(geometry: Geometry, T_MeV: float) -> float:
    """
    WEPL (mm) from stopping-power ratios at kinetic energy T_MeV:

        WEPL = sum_i thickness_i * (dE/dx)_i(T) / (dE/dx)_water(T)

    More accurate than the density-based WEPL when the phantom materials
    have very different Z or I. Still approximate: it does not integrate
    the energy change along the path.
    """
    dEdx_water = bethe_bloch(T_MeV, WATER)
    total = 0.0
    for layer in geometry.phantom_layers:
        mat = material_object(layer["material"])
        dEdx_mat = bethe_bloch(T_MeV, mat)
        total += layer["thickness"] * (dEdx_mat / dEdx_water)
    return total
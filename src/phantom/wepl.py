"""
WEPL (water-equivalent path length) for a phantom described by Geometry.

WEPL is the thickness of water that would cause the same energy loss.
Two approximations are provided: one based on density, one based on
stopping-power ratios.
"""
from src.constants import WATER
from src.geometry.geometry import Geometry
from src.phantom.materials import material_object, bethe_bloch, range_csda_fast
from src.constants import WATER

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


def wepl_stopping_power(geometry: Geometry, T_in_MeV: float, particle=None) -> float:
    """
    Exact WEPL definition: WEPL = R_water(T_in) - R_water(T_out).
    This is the quantity measured in proton CT. It requires finding T_out
    by subtracting the material range from T_in step by step.
    """
    T_current = T_in_MeV
    
    for layer in geometry.phantom_layers:
        mat = material_object(layer["material"])
        thickness = layer["thickness"]
        
        # Range of the particle in the current material
        range_in_mat = range_csda_fast(T_current, mat, particle)
        
        if range_in_mat <= thickness:
            # The particle stops inside this layer. The remaining WEPL is just the water range of the current energy.
            return range_csda_fast(T_current, WATER, particle)
        
        # To find T_out, we need to invert the range. Since we have a spline table, we can just find the energy that corresponds to (range_in_mat - thickness).
        # Note: For a fast approximation, you can use a root-finding algorithm, but using the table interpolation is faster.
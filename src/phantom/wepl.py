"""
WEPL (water-equivalent path length) for a phantom described by Geometry.

WEPL is the thickness of water that would cause the same energy loss.
Two approximations are provided: one based on density, one based on
stopping-power ratios.
"""
from src.constants import WATER, Particle
from src.geometry.geometry import Geometry
from src.phantom.materials import material_object, bethe_bloch, range_csda_fast

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


from scipy.optimize import brentq
from src.phantom.materials import E_MIN_MEV

def wepl_stopping_power(geometry: Geometry, T_MeV: float, particle=None) -> float:
    """
    Exact WEPL definition: WEPL = R_water(T_in) - R_water(T_out).
    This is the quantity measured in proton CT. It requires finding T_out
    by subtracting the material range from T_in step by step.
    """
    T_current = T_MeV
    
    for layer in geometry.phantom_layers:
        mat = material_object(layer["material"])
        thickness = layer["thickness"]
        
        # Range of the particle in the current material
        range_in_mat = range_csda_fast(T_current, mat, particle)
        
        if range_in_mat <= thickness:
            # The particle stops inside this layer. 
            # The final energy is 0, so the WEPL is the full water range of the initial energy.
            return range_csda_fast(T_MeV, WATER, particle)
        
        # We need to find T_out such that the range in the material equals the remaining range.
        remaining_range = range_in_mat - thickness
        
        # Define the function to find the root (where range(T) - remaining_range == 0)
        def root_func(T):
            return range_csda_fast(T, mat, particle) - remaining_range
        
        # The root is bracketed between the minimum valid energy (0.75 MeV) and the current energy.
        T_out = brentq(root_func, E_MIN_MEV, T_current)
        T_current = T_out
        
    # The particle has exited the phantom.
    # WEPL is the difference between the water range of the initial and final energy.
    R_water_in = range_csda_fast(T_MeV, WATER, particle)
    R_water_out = range_csda_fast(T_current, WATER, particle)
    
    return R_water_in - R_water_out
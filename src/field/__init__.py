"""Magnetic field models.

Public API:
    build_field(cfg, geometry) -> MagneticField
"""
from .base import MagneticField
from .hard_edge import HardEdgeField
from .biot_savart import BiotSavartField
from .interpolated import InterpolatedField
from .no_field import NoField


def build_field(cfg: dict, geometry) -> MagneticField:
    """Build the field model selected in the config.

    Args:
        cfg: full config dict, as returned by src.utils.config_loader.load_config.
            The magnet block is read from cfg["geometry"]["magnet"] (B0 in tesla).
        geometry: Geometry built from cfg (make_geometry), for the extent of the field map.

    If the magnet is disabled (magnet.enabled: false) there is no field: NoField.
    If interpolation is enabled, the source field is tabulated on a grid
    and the returned object is an InterpolatedField. The grid covers the whole system in z,
    [0, z_end_system], so the map never ends inside it (no artificial cut of the fringe field),
    and the magnet gap in x, |x| <= x_half_gap + 2 dx: the propagator takes the field as zero
    outside the gap, and the 2 extra cells keep the spline accurate at the plates.
    """
    from src.utils.config_loader import is_enabled   # local: config_loader imports src.geometry

    m = cfg["geometry"]["magnet"]
    if not is_enabled(m, "geometry.magnet"):
        return NoField()

    if m["model"] == "hard_edge":
        src = HardEdgeField(m["B0"], m["z_start"], m["z_length"], m["x_half_gap"])
    elif m["model"] == "biot_savart":
        src = BiotSavartField(m["B0"], m["z_start"], m["z_length"],
                              m["x_half_gap"], m["y_height"],
                              m["biot_savart"]["n_filaments"])
    else:
        raise ValueError(f"Unknown field model: {m['model']}")

    ip = m["interpolation"]
    if ip["enabled"]:
        return InterpolatedField.from_field(
            src, x_max=m["x_half_gap"] + 2 * ip["dx"], z_min=0.0, z_max=geometry.z_end_system,
            dx=ip["dx"], dz=ip["dz"])
    return src


__all__ = ["MagneticField", "HardEdgeField", "BiotSavartField",
           "InterpolatedField", "NoField", "build_field"]
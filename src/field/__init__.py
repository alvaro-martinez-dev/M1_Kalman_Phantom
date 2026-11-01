"""Magnetic field models.

Public API:
    build_field(cfg) -> MagneticField
"""
from .base import MagneticField
from .hard_edge import HardEdgeField
from .biot_savart import BiotSavartField
from .interpolated import InterpolatedField


def build_field(cfg: dict) -> MagneticField:
    """Build the field model selected in the config.

    If interpolation is enabled, the source field is tabulated on a grid
    and the returned object is an InterpolatedField.
    """
    m = cfg["magnet"]
    if m["model"] == "hard_edge":
        src = HardEdgeField(m["B0_nat"], m["z_start"], m["length"], m["half_gap"])
    elif m["model"] == "biot_savart":
        src = BiotSavartField(m["B0_nat"], m["z_start"], m["length"],
                              m["half_gap"], m["height"],
                              m["biot_savart"]["n_filaments"])
    else:
        raise ValueError(f"Unknown field model: {m['model']}")

    ip = m["interpolation"]
    if ip["enabled"]:
        return InterpolatedField.from_field(
            src, m["z_start"], m["length"], m["half_gap"],
            ip["margin"], ip["dx"], ip["dz"])
    return src


__all__ = ["MagneticField", "HardEdgeField", "BiotSavartField",
           "InterpolatedField", "build_field"]
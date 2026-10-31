"""Biot-Savart field from two finite current sheets (skeleton).

Not implemented yet. The plan:
    - each plate is discretised into n_filaments wires at different y;
    - each wire is a closed rectangular loop in the xz plane;
    - the field of each straight segment is computed analytically;
    - the result is normalised so that By at the centre equals B0_nat.
"""
import numpy as np
from .base import MagneticField


class BiotSavartField(MagneticField):
    """Field from two parallel current sheets with return paths.

    Args:
        B0_nat: target field magnitude at the centre, in GeV^2.
        z_start: entrance plane of the magnet [mm].
        length: magnet length along z [mm].
        half_gap: plates located at x = +/- half_gap [mm].
        height: extent of the plates in y [mm].
        n_filaments: number of wires used to discretise each plate.
    """

    def __init__(self, B0_nat, z_start, length, half_gap, height, n_filaments):
        self.B0_nat = B0_nat
        self.z_start = z_start
        self.z_end = z_start + length
        self.half_gap = half_gap
        self.height = height
        self.n_filaments = n_filaments
        # TODO: build the wire segments and the normalisation to B0_nat

    def B(self, x, y, z):
        """Not implemented yet."""
        raise NotImplementedError("Biot-Savart: next step")
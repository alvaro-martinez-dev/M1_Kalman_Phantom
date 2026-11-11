"""Hard-edge field model: uniform inside the magnet, zero outside."""
import numpy as np
from .base import MagneticField


class HardEdgeField(MagneticField):
    """Uniform By = B0_nat inside the magnet box, zero outside.

    The box is defined by:
        z_start <= z <= z_start + length
        |x| <= half_gap
    No y dependence (infinite height in y).

    Args:
        B0_nat: field magnitude in natural units, MeV^2 (B0[T] * constants.tesla).
        z_start: entrance plane of the magnet [mm].
        length: magnet length along z [mm].
        half_gap: half-width of the gap in x [mm].
    """

    def __init__(self, B0_nat, z_start, length, half_gap):
        self.B0_nat = B0_nat
        self.z_start = z_start
        self.z_end = z_start + length
        self.half_gap = half_gap

    def B(self, x, y, z):
        """Return the field vector in MeV^2, shape (..., 3)."""
        x, y, z = np.broadcast_arrays(*map(np.asarray, (x, y, z)))
        inside = (z >= self.z_start) & (z <= self.z_end) & (np.abs(x) <= self.half_gap)
        out = np.zeros(x.shape + (3,))
        out[..., 1] = np.where(inside, self.B0_nat, 0.0)
        return out
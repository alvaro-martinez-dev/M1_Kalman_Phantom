"""Hard-edge field model: uniform inside the magnet, zero outside."""
import numpy as np
from .base import MagneticField


class HardEdgeField(MagneticField):
    """Uniform By = B0 inside the magnet box, zero outside.

    The box is defined by:
        z_start <= z <= z_start + length
        |x| <= half_gap
    No y dependence (infinite height in y).

    Args:
        B0: field magnitude [T].
        z_start: entrance plane of the magnet [mm].
        length: magnet length along z [mm].
        half_gap: half-width of the gap in x [mm].
    """

    def __init__(self, B0, z_start, length, half_gap):
        self.B0 = B0
        self.z_start = z_start
        self.z_end = z_start + length
        self.half_gap = half_gap

    def z_discontinuities(self):
        return (self.z_start, self.z_end)

    def B(self, x, y, z):
        """Return the field vector in T, shape (..., 3)."""
        x, y, z = np.broadcast_arrays(*map(np.asarray, (x, y, z)))
        inside = (z >= self.z_start) & (z <= self.z_end) & (np.abs(x) <= self.half_gap)
        out = np.zeros(x.shape + (3,))
        out[..., 1] = np.where(inside, self.B0, 0.0)
        return out
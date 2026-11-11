"""No magnetic field: used when the magnet is disabled."""
import numpy as np
from .base import MagneticField


class NoField(MagneticField):
    """B = 0 everywhere. Its support is empty, so the propagator uses straight lines in vacuum."""

    def z_support(self):
        return (np.inf, -np.inf)    # empty interval: every vacuum segment is field-free

    def B(self, x, y, z):
        """Return the field vector in T, shape (..., 3): all zeros."""
        x, y, z = np.broadcast_arrays(*map(np.asarray, (x, y, z)))
        return np.zeros(x.shape + (3,))

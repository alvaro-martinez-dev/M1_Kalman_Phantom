"""Abstract interface for all magnetic field models.

Units: positions in mm, field in tesla.
"""
from abc import ABC, abstractmethod
import numpy as np


class MagneticField(ABC):
    """Abstract base class for magnetic fields."""

    @abstractmethod
    def B(self, x, y, z) -> np.ndarray:
        """Return the field vector [Bx, By, Bz] at the given positions.

        Args:
            x, y, z: positions in mm (scalars or broadcastable arrays).

        Returns:
            Array of shape (..., 3) with the field in T.
        """

    def z_discontinuities(self):
        """z values [mm] where the field jumps (e.g. hard edges, end of a field map).

        The propagator cuts the integration there, so that no RK step crosses a jump.
        Default: none (smooth field).
        """
        return ()

    def z_support(self):
        """(z_min, z_max) [mm] outside which the field is identically zero.

        The propagator uses the exact straight line in vacuum segments outside this interval
        and RKN inside it. Default: (-inf, inf), i.e. the field may be non-zero anywhere
        (e.g. Biot-Savart, which decays but never vanishes). The ends of a finite support are
        jumps of the field, so they must also be in z_discontinuities().
        """
        return (-np.inf, np.inf)

    def By(self, x, z):
        """Return the By component at y = 0.

        This is the only component used by the propagator, since the
        magnetic field points along y and the track is in the xz plane.
        """
        return self.B(x, 0.0, z)[..., 1]
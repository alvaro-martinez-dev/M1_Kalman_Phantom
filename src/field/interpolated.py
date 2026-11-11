"""Tabulated By(x, z) at y = 0, interpolated with a cubic spline.

The source field is evaluated once on a regular (x, z) grid (build_field: the
whole system in z, the magnet gap in x). During tracking, By is read by interpolation,
which is much faster than re-evaluating the source (e.g. Biot-Savart) at
every RK4 substep.

Outside the grid, the field is set to zero.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from .base import MagneticField


class InterpolatedField(MagneticField):
    """Cubic-spline interpolation of By(x, z) on a rectangular grid.

    Args:
        xs: 1D array of x nodes [mm], strictly increasing.
        zs: 1D array of z nodes [mm], strictly increasing.
        By_grid: array of shape (len(xs), len(zs)) with By in T.
    """

    def __init__(self, xs, zs, By_grid):
        self._xs, self._zs = xs, zs
        self._spline = RectBivariateSpline(xs, zs, By_grid, kx=3, ky=3)
        # limits of the grid as Python floats, for the scalar path of By
        self._x_max = float(max(abs(xs[0]), abs(xs[-1])))
        self._z_lo, self._z_hi = float(zs[0]), float(zs[-1])

    def By(self, x, z):
        """By [T] at y = 0. Scalars (the propagator, at every RK stage) skip the array
        handling of B, which costs more than the spline itself; arrays go through B."""
        if np.ndim(x) == 0 and np.ndim(z) == 0:
            if abs(x) > self._x_max or not self._z_lo <= z <= self._z_hi:
                return 0.0
            return float(self._spline.ev(x, z))
        return super().By(x, z)

    @classmethod
    def from_field(cls, source, x_max, z_min, z_max, dx, dz):
        """Tabulate a source field on the grid x in [-x_max, x_max], z in [z_min, z_max].

        The nodes are spaced dx, dz; the last node is at or just beyond x_max / z_max,
        so the grid always covers the requested range.

        Args:
            source: any MagneticField instance (hard-edge, Biot-Savart, ...).
            x_max: half width of the grid in x [mm].
            z_min, z_max: extent of the grid in z [mm].
            dx, dz: grid spacing [mm].
        """
        nx = int(np.ceil(x_max / dx - 1e-9))
        nz = int(np.ceil((z_max - z_min) / dz - 1e-9))
        xs = dx * np.arange(-nx, nx + 1)
        zs = z_min + dz * np.arange(nz + 1)
        grid = source.By(xs[:, None], zs[None, :])   # shape (nx, nz)
        return cls(xs, zs, grid)

    def z_discontinuities(self):
        """The field is set to zero outside the grid, so it can jump at both ends of it."""
        return (float(self._zs[0]), float(self._zs[-1]))

    def z_support(self):
        """Zero outside the grid."""
        return (float(self._zs[0]), float(self._zs[-1]))

    def B(self, x, y, z):
        """Return the field vector in T, shape (..., 3).

        Only By is non-zero. Points outside the grid return zero.
        The y argument is ignored (the field is evaluated at y = 0).
        """
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        # Points inside the tabulated domain
        inside = ((np.abs(x) <= self._xs[-1]) &
                  (z >= self._zs[0]) & (z <= self._zs[-1]))
        # Clip to the domain so the spline is never evaluated outside it
        xc = np.clip(x, self._xs[0], self._xs[-1])
        zc = np.clip(z, self._zs[0], self._zs[-1])
        by = np.where(inside, self._spline.ev(xc, zc), 0.0)
        out = np.zeros(x.shape + (3,))
        out[..., 1] = by
        return out
"""Tabulated By(x, z) at y = 0, interpolated with a cubic spline.

The source field is evaluated once on a regular (x, z) grid that extends
beyond the magnet by a margin. During tracking, By is read by interpolation,
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

    @classmethod
    def from_field(cls, source, z_start, length, half_gap, margin, dx, dz):
        """Tabulate a source field on a grid covering the magnet plus margin.

        Grid extent:
            x in [-(half_gap + margin), +(half_gap + margin)]
            z in [z_start - margin, z_start + length + margin]

        Args:
            source: any MagneticField instance (hard-edge, Biot-Savart, ...).
            z_start, length, half_gap: magnet geometry [mm].
            margin: extra extent beyond the magnet in x and z [mm].
            dx, dz: grid spacing [mm].
        """
        x_max = half_gap + margin
        z0, z1 = z_start - margin, z_start + length + margin
        xs = np.arange(-x_max, x_max + dx / 2, dx)
        zs = np.arange(z0, z1 + dz / 2, dz)
        grid = source.By(xs[:, None], zs[None, :])   # shape (nx, nz)
        return cls(xs, zs, grid)

    def z_discontinuities(self):
        """The field is set to zero outside the grid, so it can jump at both ends of it."""
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
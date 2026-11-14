"""Biot-Savart field from two finite current sheets.

Each plate (at x = +x_half_gap and x = -x_half_gap) is discretised into
n_filaments wires, evenly spaced in y over [-y_height/2, y_height/2]. For
each y, the two wires (one per plate) are connected by return paths
at z = z_start and z = z_end, forming a closed rectangular loop in
the xz plane. All loops carry the same unit current (mu0 = I = 1).

Normalisation: the field is rescaled so that the line integral of By along
the axis (x = 0, y = 0) equals B0 * z_length. B0 is therefore the mean field
along the central trajectory, and the bending power is fixed independently of
the geometry of the return paths. The integral runs over
[z_start - z_int_halfrange, z_end + z_int_halfrange]; the tails of the field
decay slowly, so this range is part of the definition of B0.
"""
import numpy as np
from .base import MagneticField


def _segment_field(r, p1, p2):
    """Biot-Savart field (mu0 = I = 1) of a straight segment p1 -> p2.

    Args:
        r: query points, array of shape (..., 3).
        p1, p2: segment endpoints, shape (3,).

    Returns:
        Field array of shape (..., 3), in units of 1/length (mu0 = I = 1).
    """
    L = p2 - p1
    a = r - p1
    b = r - p2

    cross_La = np.cross(L, a)
    cross_norm2 = np.sum(cross_La ** 2, axis=-1, keepdims=True)
    cross_norm2 = np.maximum(cross_norm2, 1e-30)  # avoid blow-up on the wire

    a_norm = np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-30)
    b_norm = np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-30)

    cos_a = np.sum(L * a, axis=-1, keepdims=True) / a_norm
    cos_b = np.sum(L * b, axis=-1, keepdims=True) / b_norm

    prefactor = (cos_a - cos_b) / (4.0 * np.pi * cross_norm2)
    return prefactor * cross_La


class BiotSavartField(MagneticField):
    """Field from two parallel current sheets with return paths.

    Args:
        B0: mean field along the axis, i.e. (integral of By dz at x = 0) / z_length [T].
        z_start: entrance plane of the magnet [mm].
        z_length: magnet length along z [mm].
        x_half_gap: plates located at x = +/- x_half_gap [mm].
        y_height: extent of the plates in y [mm].
        n_filaments: number of wires used to discretise each plate.
        z_int_halfrange: the normalisation integral covers the magnet extended by
            this amount on each side [mm].
        z_int_step: step of the trapezoidal rule of the normalisation integral [mm].
    """

    def __init__(self, B0, z_start, z_length, x_half_gap, y_height, n_filaments,
                 z_int_halfrange=2400.0, z_int_step=1.0):
        self.B0 = B0
        self.z_start = z_start
        self.z_length = z_length
        self.z_end = z_start + z_length
        self.x_half_gap = x_half_gap
        self.y_height = y_height
        self.n_filaments = n_filaments
        self.z_int_halfrange = z_int_halfrange
        self.z_int_step = z_int_step

        self._loops = self._build_loops()   # closed loops: div B = 0 by construction

        # Linearity in the current: compute the shape with unit current, then rescale
        # so that the axis integral of By equals B0 * z_length.
        z = np.arange(self.z_start - z_int_halfrange,
                      self.z_end + z_int_halfrange + z_int_step / 2.0, z_int_step)
        raw_integral = np.trapezoid(self._raw_By(np.zeros_like(z), z), z)
        if abs(raw_integral) < 1e-300:
            raise ValueError("Axis integral of the unit field is zero; check geometry.")
        self._scale = self.B0 * self.z_length / raw_integral

    def _build_loops(self):
        """Return the (p1, p2) segment pairs of every filament loop."""
        ys = np.linspace(-self.y_height / 2.0, self.y_height / 2.0, self.n_filaments)
        d = self.x_half_gap
        z0, z1 = self.z_start, self.z_end

        segments = []
        for y in ys:
            corners = [
                np.array([+d, y, z0]),
                np.array([+d, y, z1]),
                np.array([-d, y, z1]),
                np.array([-d, y, z0]),
            ]
            for i in range(4):
                segments.append((corners[i], corners[(i + 1) % 4]))
        return segments

    def _raw_By(self, x, z):
        """By summed over all loops with unit current (mu0 = I = 1), at y = 0."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        x, z = np.broadcast_arrays(x, z)
        r = np.stack([x, np.zeros_like(x), z], axis=-1)

        total = np.zeros(r.shape)
        for p1, p2 in self._loops:
            total += _segment_field(r, p1, p2)
        return total[..., 1]               # only the y component

    def B(self, x, y, z):
        """Return the field vector in T, shape (..., 3).

        Only By is modelled (Bx, Bz are left at zero, as in HardEdgeField).
        The y argument is ignored: the field is evaluated at y = 0, the
        same convention used by InterpolatedField.
        """
        by = self._scale * self._raw_By(x, z)
        out = np.zeros(by.shape + (3,))
        out[..., 1] = by
        return out
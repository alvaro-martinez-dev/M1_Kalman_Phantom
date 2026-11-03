"""Biot-Savart field from two finite current sheets.

Each plate (at x = +half_gap and x = -half_gap) is discretised into
n_filaments wires, evenly spaced in y over [-height/2, height/2]. For
each y, the two wires (one per plate) are connected by return paths
at z = z_start and z = z_end, forming a closed rectangular loop in
the xz plane. All loops carry the same unit current (mu0 = I = 1);
the total field is then rescaled so that By at the magnet centre
equals B0_nat.
"""
import numpy as np
from .base import MagneticField


def _segment_field(r, p1, p2):  #implement Biot Savart law for a single segment
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
        B0_nat: target field magnitude at the centre, in natural units
            (same convention as HardEdgeField.B0_nat, see constants.py).
        z_start: entrance plane of the magnet [mm].
        length: magnet length along z [mm].
        half_gap: plates located at x = +/- half_gap [mm].
        height: extent of the plates in y [mm].
        n_filaments: number of wires used to discretise each plate.
    """

    def __init__(self, B0_nat, z_start, length, half_gap, height, n_filaments):
        self.B0_nat = B0_nat                         #here we save the geometry
        self.z_start = z_start
        self.z_end = z_start + length
        self.half_gap = half_gap
        self.height = height
        self.n_filaments = n_filaments

        self._loops = self._build_loops()            #here we build the filaments loops, closed so div(B) = 0 by costruction
        z_centre = 0.5 * (self.z_start + self.z_end)
        by_unit = self._raw_By(np.array([0.0]), np.array([z_centre]))[0]
        if abs(by_unit) < 1e-300:
            raise ValueError("Unit field at centre is zero; check geometry.")
        self._scale = self.B0_nat / by_unit          #scalar factor, that bc of the linear dependence on I of the magnetic field avoid us not to know I, we can just set a target B0

    def _build_loops(self):                          #here we actually build the loop
        """Return the (p1, p2) segment pairs of every filament loop."""
        ys = np.linspace(-self.height / 2.0, self.height / 2.0, self.n_filaments)
        d = self.half_gap
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
        """we calculate total By with the principle of superposition from all loops with unit current (mu0 = I = 1), at y = 0."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        x, z = np.broadcast_arrays(x, z)
        r = np.stack([x, np.zeros_like(x), z], axis=-1)

        total = np.zeros(r.shape)
        for p1, p2 in self._loops:
            total += _segment_field(r, p1, p2)
        return total[..., 1]            #output is ONLY y component

    def B(self, x, y, z):
        """Return the field vector in natural units, shape (..., 3).

        Only By is modelled (Bx, Bz are left at zero, as in HardEdgeField).
        The y argument is ignored: the field is evaluated at y = 0, the
        same convention used by InterpolatedField.
        """
        by = self._scale * self._raw_By(x, z)
        out = np.zeros(by.shape + (3,))
        out[..., 1] = by
        return out
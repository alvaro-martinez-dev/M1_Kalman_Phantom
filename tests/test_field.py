# tests/test_field.py
# tests/test_field.py
"""Tests for the magnetic field models (src/field).

Units: positions in mm, field in tesla.
"""
import numpy as np
import pytest
from scipy.integrate import quad

from field import (HardEdgeField, BiotSavartField, InterpolatedField,
                   NoField, MagneticField)
from field.biot_savart import _segment_field

# Reference geometry
Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT = 400.0, 300.0, 50.0, 100.0
Z_END = Z_START + Z_LENGTH
Z_CENTRE = Z_START + Z_LENGTH / 2.0


# ---------------------------------------------------------------- HardEdge
class TestHardEdgeField:

    def make(self, B0=1.0):
        return HardEdgeField(B0=B0, z_start=Z_START, length=Z_LENGTH,
                             half_gap=X_HALF_GAP)

    def test_symmetry_x(self):
        f = self.make()
        assert f.By(10.0, 450.0) == f.By(-10.0, 450.0)

    def test_inside_value(self):
        assert self.make(B0=1.5).By(0.0, Z_CENTRE) == 1.5

    def test_outside_is_zero(self):
        f = self.make()
        assert f.By(0.0, 100.0) == 0.0         # before the magnet
        assert f.By(0.0, 800.0) == 0.0         # after the magnet
        assert f.By(100.0, Z_CENTRE) == 0.0    # outside the gap in x

    def test_only_By_nonzero(self):
        b = self.make().B(0.0, 0.0, Z_CENTRE)
        assert b.shape == (3,)
        assert b[0] == 0.0 and b[2] == 0.0

    def test_vectorised_shape(self):
        x = np.linspace(-60, 60, 7)[:, None]
        z = np.linspace(300, 800, 11)[None, :]
        assert self.make().By(x, z).shape == (7, 11)

    def test_discontinuities_and_support(self):
        f = self.make()
        assert tuple(f.z_discontinuities()) == (Z_START, Z_END)
        assert tuple(f.z_support()) == (Z_START, Z_END)


# ----------------------------------------------------------------- NoField
class TestNoField:

    def test_zero_everywhere(self):
        f = NoField()
        assert np.all(f.B(1.0, 2.0, 3.0) == 0.0)
        assert f.By(np.array([0.0, 10.0]), 500.0).shape == (2,)

    def test_empty_support(self):
        lo, hi = NoField().z_support()
        assert lo > hi


# ------------------------------------------------------------- Biot-Savart
class TestSegmentField:
    """Analytic finite-segment formula against independent references."""

    def test_vs_numerical_integration(self):
        """Compare with quad on dB = (1/4pi) dl x r / r^3 (mu0 = I = 1)."""
        p1 = np.array([0.0, 0.0, -30.0])
        p2 = np.array([0.0, 0.0, 70.0])
        r = np.array([[20.0, 5.0, 10.0]])

        def integrand(t, comp):
            pos = p1 + t * (p2 - p1)
            dl = p2 - p1
            d = r[0] - pos
            return (np.cross(dl, d)[comp] / (4 * np.pi * np.linalg.norm(d) ** 3))

        expected = np.array([quad(integrand, 0, 1, args=(c,))[0] for c in range(3)])
        got = _segment_field(r, p1, p2)[0]
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-14)

    def test_infinite_wire_limit(self):
        """Very long wire along z: |B| -> 1/(2 pi d)."""
        half = 1e6
        p1 = np.array([0.0, 0.0, -half])
        p2 = np.array([0.0, 0.0, half])
        d = 25.0
        b = _segment_field(np.array([[d, 0.0, 0.0]]), p1, p2)[0]
        assert abs(np.linalg.norm(b) - 1.0 / (2 * np.pi * d)) < 1e-9
        # direction: z x x = +y
        assert b[1] > 0 and abs(b[0]) < 1e-12 and abs(b[2]) < 1e-12


class TestBiotSavartField:

    def make(self, B0=1.0, n=20, y_height=Y_HEIGHT):
        return BiotSavartField(B0=B0, z_start=Z_START, z_length=Z_LENGTH,
                               x_half_gap=X_HALF_GAP, y_height=y_height,
                               n_filaments=n)

    def test_centre_normalisation(self):
        assert self.make(B0=1.0).By(0.0, Z_CENTRE) == pytest.approx(1.0, abs=1e-12)
        assert self.make(B0=0.7).By(0.0, Z_CENTRE) == pytest.approx(0.7, abs=1e-12)

    def test_sign_follows_B0(self):
        assert self.make(B0=-1.0).By(0.0, Z_CENTRE) == pytest.approx(-1.0, abs=1e-12)

    def test_linear_in_B0(self):
        z = np.linspace(300, 800, 21)
        np.testing.assert_allclose(self.make(B0=2.0).By(10.0, z),
                                   2.0 * self.make(B0=1.0).By(10.0, z),
                                   rtol=1e-12, atol=1e-14)

    def test_symmetry_x(self):
        f = self.make()
        x = np.array([5.0, 20.0, 40.0])
        np.testing.assert_allclose(f.By(x, 500.0), f.By(-x, 500.0),
                                   rtol=1e-10, atol=1e-12)

    def test_symmetry_z_about_centre(self):
        f = self.make()
        dz = np.array([10.0, 80.0, 200.0, 350.0])
        np.testing.assert_allclose(f.By(0.0, Z_CENTRE + dz),
                                   f.By(0.0, Z_CENTRE - dz),
                                   rtol=1e-10, atol=1e-12)

    def test_only_By_nonzero_and_shape(self):
        b = self.make().B(0.0, 0.0, Z_CENTRE)
        assert b.shape == (3,)
        assert b[0] == 0.0 and b[2] == 0.0

    def test_vectorised_shape(self):
        x = np.linspace(-20, 20, 5)[:, None]
        z = np.linspace(300, 800, 9)[None, :]
        assert self.make().By(x, z).shape == (5, 9)

    def test_default_support_is_unbounded(self):
        f = self.make()
        lo, hi = f.z_support()
        assert lo == -np.inf and hi == np.inf
        assert tuple(f.z_discontinuities()) == ()

    def test_is_a_magnetic_field(self):
        assert isinstance(self.make(), MagneticField)


# ------------------------------------------------------------ Interpolated
class TestInterpolatedField:

    def test_grid_covers_requested_range(self):
        src = HardEdgeField(1.0, Z_START, Z_LENGTH, X_HALF_GAP)
        f = InterpolatedField.from_field(src, x_max=52.0, z_min=0.0,
                                         z_max=1000.0, dx=1.0, dz=0.5)
        assert f._xs[0] <= -52.0 and f._xs[-1] >= 52.0
        assert f._zs[0] == 0.0 and f._zs[-1] >= 1000.0
        assert 0.0 in f._xs                      # grid symmetric, contains x = 0

    def test_support_and_discontinuities_are_grid_ends(self):
        src = HardEdgeField(1.0, Z_START, Z_LENGTH, X_HALF_GAP)
        f = InterpolatedField.from_field(src, 52.0, 0.0, 1000.0, 1.0, 0.5)
        ends = (float(f._zs[0]), float(f._zs[-1]))
        assert tuple(f.z_support()) == ends
        assert tuple(f.z_discontinuities()) == ends

    def test_zero_outside_grid(self):
        src = HardEdgeField(1.0, Z_START, Z_LENGTH, X_HALF_GAP)
        f = InterpolatedField.from_field(src, 52.0, 0.0, 1000.0, 1.0, 0.5)
        assert f.By(0.0, -10.0) == 0.0
        assert f.By(0.0, 1500.0) == 0.0
        assert f.By(200.0, 500.0) == 0.0

    def test_matches_biot_savart_in_smooth_region(self):
        """Away from the steep fringe, the spline reproduces the source."""
        src = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT, 20)
        f = InterpolatedField.from_field(src, x_max=52.0, z_min=200.0,
                                         z_max=900.0, dx=1.0, dz=0.5)
        x = np.linspace(-25.0, 25.0, 11)[:, None]
        z = np.linspace(450.0, 650.0, 21)[None, :]
        exact = src.By(x, z)
        interp = f.By(x, z)
        assert np.max(np.abs(interp - exact)) < 1e-4

    def test_error_near_edge_is_reported(self):
        """Not an assertion on accuracy: prints the spline error at the steep
        edge z = z_start, to decide the grid spacing (run with pytest -s)."""
        src = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT, 20)
        f = InterpolatedField.from_field(src, 52.0, 200.0, 900.0, 1.0, 0.5)
        z = np.linspace(Z_START - 10, Z_START + 10, 81)
        err = np.max(np.abs(f.By(0.0, z) - src.By(0.0, z)))
        print(f"\nmax interpolation error within 10 mm of z_start: {err:.3e} T")
        assert np.isfinite(err)
class TestInterpolatedScalarPath:
    """By has a fast path for scalar arguments (used by the propagator at every
    RK stage). It must agree exactly with the array path through B."""

    @staticmethod
    def make():
        src = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT, 20)
        return InterpolatedField.from_field(src, x_max=52.0, z_min=0.0, z_max=1000.0, dx=1.0, dz=0.5)

    def test_scalar_returns_python_float(self):
        f = self.make()
        assert isinstance(f.By(0.0, 550.0), float)
        assert isinstance(f.By(0.0, -10.0), float)      # outside the grid

    def test_scalar_matches_array_path(self):
        f = self.make()
        rng = np.random.default_rng(0)
        xs = rng.uniform(-60.0, 60.0, 200)     # includes points beyond |x| = 52
        zs = rng.uniform(-50.0, 1050.0, 200)   # includes points beyond the z range
        array_path = f.B(xs, 0.0, zs)[..., 1]
        scalar_path = np.array([f.By(float(x), float(z)) for x, z in zip(xs, zs)])
        np.testing.assert_allclose(scalar_path, array_path, rtol=1e-12, atol=1e-14)

    def test_grid_edges_are_inclusive_in_both_paths(self):
        """x = +/-x_max and z = z_lo, z_hi belong to the grid in both paths;
        a hair beyond them the field is zero in both paths."""
        f = self.make()
        eps = 1e-6
        for x, z in [(f._x_max, 500.0), (-f._x_max, 500.0),
                     (0.0, f._z_lo), (0.0, f._z_hi)]:
            assert f.By(x, z) == pytest.approx(f.B(x, 0.0, z)[1], abs=1e-14)
        for x, z in [(f._x_max + eps, 500.0), (-f._x_max - eps, 500.0),
                     (0.0, f._z_lo - eps), (0.0, f._z_hi + eps)]:
            assert f.By(x, z) == 0.0
            assert f.B(x, 0.0, z)[1] == 0.0

    def test_zero_dim_numpy_inputs(self):
        """0-d numpy scalars go through the fast path and must still work."""
        f = self.make()
        assert f.By(np.float64(10.0), np.float64(500.0)) == pytest.approx(
            f.By(np.array([10.0]), np.array([500.0]))[0], abs=1e-14)

def test_edge_error_at_dz_0p5():
    """Interpolation error near the steep edges at the chosen dz = 0.5 mm."""
    src = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT, 20)
    f = InterpolatedField.from_field(src, x_max=52.0, z_min=0.0,
                                     z_max=1000.0, dx=1.0, dz=0.5)
    z = np.linspace(Z_START - 10, Z_START + 10, 801)
    assert np.max(np.abs(f.By(0.0, z) - src.By(0.0, z))) < 2e-4
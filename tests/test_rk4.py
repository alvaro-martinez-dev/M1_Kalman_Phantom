# tests/test_rk4.py
"""Validation of the integrators in src/propagator/rk4.py against exact solutions."""
import numpy as np
import pytest

from src.field import HardEdgeField
from src.propagator.rk4 import (
    rkn4, rkn4_terms, rkn4_error, rk4, rk4_terms, rk45_terms,
)

KAPPA = 1e-3  # curvature of the test track [1/mm], radius = 1000 mm


# ----- Problems with exact solution ----------------------------------------------------------

def circle_rhs(z, x, xprime):
    """x'' = f(z, x, x') for a circular track, starting flat at z = 0."""
    return -KAPPA * (1 + xprime**2)**1.5


def circle_x(z):
    return np.sqrt(1/KAPPA**2 - z**2) - 1/KAPPA


def circle_xprime(z):
    return -z / np.sqrt(1/KAPPA**2 - z**2)


def gauss_rhs(z, x):
    """x' = -2 z x, with solution x = exp(-z^2)."""
    return -2 * z * x


def gauss_x(z):
    return np.exp(-z**2)


def oscillator_rhs(z, x):
    """x' = (x1, -x0), with solution (cos z, -sin z)."""
    return np.array([x[1], -x[0]])


def oscillator_x(z):
    return np.array([np.cos(z), -np.sin(z)])


# ----- Helpers -------------------------------------------------------------------------------

def slopes(errors):
    """Convergence orders, for errors obtained by halving h each time."""
    errors = np.array(errors)
    return np.log2(errors[:-1] / errors[1:])


def fixed_step_rk45(fun, z0, z_end, x0, n):
    """Integrate with n equal Dormand-Prince steps, to measure the order of rk45_terms."""
    h = (z_end - z0) / n
    x = np.asarray(x0, dtype=float)
    for i in range(n):
        z = z0 + i*h
        x, _, _ = rk45_terms(fun, z, x, h, fun(z, x))
    return x


# ----- rkn4: single step ---------------------------------------------------------------------

def test_rkn4_error_formula():
    assert rkn4_error(2.0, 1.0, 2.0, 3.0, 5.0) == pytest.approx(4.0 * abs(1 - 2 - 3 + 5))


def test_rkn4_terms_exact_for_constant_acceleration():
    """With x'' = a the step is exact and the error estimate is zero."""
    a, x0, xp0, h = 3.0, 1.0, 2.0, 0.5
    x, xprime, err = rkn4_terms(lambda z, x, xp: a, 0.0, x0, xp0, h)
    assert x == pytest.approx(x0 + xp0*h + 0.5*a*h**2)
    assert xprime == pytest.approx(xp0 + a*h)
    assert err == pytest.approx(0.0)


def test_rkn4_terms_exact_for_acceleration_cubic_in_z():
    """x' is integrated with Simpson's rule, which is exact for a cubic x''(z)."""
    _, xprime, _ = rkn4_terms(lambda z, x, xp: 4*z**3, 1.0, 0.0, 0.0, 0.5)
    assert xprime == pytest.approx(1.5**4 - 1.0)


def test_rkn4_terms_local_error_is_order_5():
    z0, hs = 100.0, (40.0, 20.0, 10.0, 5.0)
    x0, xp0 = np.array([circle_x(z0)]), np.array([circle_xprime(z0)])
    errors = [abs(rkn4_terms(circle_rhs, z0, x0, xp0, h)[0][0] - circle_x(z0 + h)) for h in hs]
    assert np.all(np.abs(slopes(errors) - 5) < 0.3)


def test_rkn4_error_estimate_scales_as_h4():
    z0, hs = 100.0, (40.0, 20.0, 10.0, 5.0, 2.5)
    x0, xp0 = np.array([circle_x(z0)]), np.array([circle_xprime(z0)])
    estimates = [rkn4_terms(circle_rhs, z0, x0, xp0, h)[2][0] for h in hs]
    assert np.all(np.abs(slopes(estimates) - 4) < 0.1)


# ----- rkn4: fixed step ----------------------------------------------------------------------

def test_rkn4_fixed_matches_exact_solution():
    z, x, xprime = rkn4(circle_rhs, 0.0, 800.0, [0.0], [0.0], 2.0)
    assert np.max(np.abs(x[:, 0] - circle_x(z))) < 1e-7
    assert np.max(np.abs(xprime[:, 0] - circle_xprime(z))) < 1e-7


def test_rkn4_fixed_grid():
    z, x, xprime = rkn4(circle_rhs, 0.0, 10.0, [0.0], [0.0], 3.0)
    assert z[0] == 0.0 and z[-1] == 10.0
    assert len(z) == 5  # ceil(10/3) = 4 steps
    assert np.allclose(np.diff(z), 2.5)
    assert x.shape == (5, 1) and xprime.shape == (5, 1)


def test_rkn4_fixed_is_order_4_in_x():
    errors = [abs(rkn4(circle_rhs, 0.0, 800.0, [0.0], [0.0], h)[1][-1, 0] - circle_x(800.0))
              for h in (40.0, 20.0, 10.0, 5.0)]
    assert np.all(np.abs(slopes(errors) - 4) < 0.2)


def test_rkn4_fixed_is_order_4_in_xprime():
    errors = [abs(rkn4(circle_rhs, 0.0, 800.0, [0.0], [0.0], h)[2][-1, 0] - circle_xprime(800.0))
              for h in (20.0, 10.0, 5.0, 2.5)]
    assert np.all(np.abs(slopes(errors) - 4) < 0.35)


def test_rkn4_fixed_backwards():
    z, x, _ = rkn4(circle_rhs, 0.0, -800.0, [0.0], [0.0], 2.0)
    assert z[-1] == -800.0
    assert x[-1, 0] == pytest.approx(circle_x(-800.0), abs=1e-7)


# ----- rkn4: adaptive step -------------------------------------------------------------------

def test_rkn4_adaptive_reaches_z_end_exactly():
    z, _, _ = rkn4(circle_rhs, 0.0, 800.0, [0.0], [0.0], 1.0, adaptative=True, tol=1e-5)
    assert z[0] == 0.0 and z[-1] == 800.0
    assert np.all(np.diff(z) > 0)


def test_rkn4_adaptive_error_decreases_with_tol():
    errors = [abs(rkn4(circle_rhs, 0.0, 800.0, [0.0], [0.0], 1.0, adaptative=True, tol=tol)[1][-1, 0]
                  - circle_x(800.0))
              for tol in (1e-3, 1e-5, 1e-7)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] < 1e-3 and errors[2] < 1e-7


@pytest.mark.parametrize("h0", [1e-3, 1.0, 500.0])
def test_rkn4_adaptive_every_stored_step_satisfies_tol(h0):
    """Rejected steps must not be stored, whatever the initial guess of h."""
    tol = 1e-5
    z, x, xprime = rkn4(circle_rhs, 0.0, 800.0, [0.0], [0.0], h0, adaptative=True, tol=tol)
    for k in range(len(z) - 1):
        x_new, xprime_new, err = rkn4_terms(circle_rhs, z[k], x[k], xprime[k], z[k + 1] - z[k])
        assert np.max(err) <= tol * (1 + 1e-6)
        assert x_new == pytest.approx(x[k + 1], abs=1e-9)
        assert xprime_new == pytest.approx(xprime[k + 1], abs=1e-9)


def test_rkn4_adaptive_grows_h_when_error_is_zero():
    z, _, _ = rkn4(lambda z, x, xp: 0.0*x, 0.0, 1000.0, [0.0], [1.0], 1.0, adaptative=True, tol=1e-5)
    assert np.diff(z)[:3] == pytest.approx([1.0, 5.0, 25.0])


def test_rkn4_adaptive_backwards():
    z, x, _ = rkn4(circle_rhs, 0.0, -800.0, [0.0], [0.0], 1.0, adaptative=True, tol=1e-5)
    assert z[-1] == -800.0
    assert np.all(np.diff(z) < 0)
    assert x[-1, 0] == pytest.approx(circle_x(-800.0), abs=1e-5)


def test_rkn4_adaptive_scalar_state():
    z, x, xprime = rkn4(circle_rhs, 0.0, 800.0, 0.0, 0.0, 1.0, adaptative=True, tol=1e-5)
    assert x.shape == z.shape and xprime.shape == z.shape
    assert x[-1] == pytest.approx(circle_x(800.0), abs=1e-5)


def test_rkn4_adaptive_vector_state():
    rhs = lambda z, x, xp: np.array([circle_rhs(z, x[0], xp[0]), 0.0])
    z, x, _ = rkn4(rhs, 0.0, 800.0, [0.0, 3.0], [0.0, 0.5], 1.0, adaptative=True, tol=1e-5)
    assert x.shape == (len(z), 2)
    assert x[-1, 0] == pytest.approx(circle_x(800.0), abs=1e-5)
    assert x[-1, 1] == pytest.approx(403.0)


def test_rkn4_adaptive_respects_h_max():
    z, _, _ = rkn4(circle_rhs, 0.0, 800.0, [0.0], [0.0], 1.0, adaptative=True, tol=1e-5, h_max=5.0)
    assert np.max(np.diff(z)) <= 5.0 + 1e-12


def test_rkn4_adaptive_zero_length_interval():
    z, x, xprime = rkn4(circle_rhs, 5.0, 5.0, [1.0], [2.0], 1.0, adaptative=True, tol=1e-5)
    assert z.tolist() == [5.0] and x.tolist() == [[1.0]] and xprime.tolist() == [[2.0]]


def test_rkn4_adaptive_unreachable_tol_raises():
    with pytest.raises(RuntimeError):
        rkn4(circle_rhs, 0.0, 800.0, [0.0], [0.0], 1.0, adaptative=True, tol=1e-30, h_min=0.5)


# ----- rkn4 with the real hard-edge field ------------------------------------------------------

Z_IN, Z_OUT = 400.0, 500.0
EPS = 1e-9  # keeps the end of a section strictly on one side of the discontinuity


def magnet_rhs():
    field = HardEdgeField(B0=1.0, z_start=Z_IN, length=Z_OUT - Z_IN, half_gap=1e9)
    return lambda z, x, xp: -KAPPA * field.By(x, z) * (1 + xp**2)**1.5


def magnet_exact(z_end):
    """Straight, then circle inside the magnet, then straight again."""
    d = Z_OUT - Z_IN
    root = np.sqrt(1/KAPPA**2 - d**2)
    return (root - 1/KAPPA) + (-d/root) * (z_end - Z_OUT)


def integrate_sections(fun, sections, **kwargs):
    x, xprime = [0.0], [0.0]
    for a, b in sections:
        _, vec_x, vec_xprime = rkn4(fun, a, b, x, xprime, 1.0, adaptative=True, tol=1e-5, **kwargs)
        x, xprime = vec_x[-1], vec_xprime[-1]
    return x[0]


def test_rkn4_adaptive_hard_edge_by_sections():
    sections = [(0.0, Z_IN - EPS), (Z_IN, Z_OUT), (Z_OUT + EPS, 1000.0)]
    assert integrate_sections(magnet_rhs(), sections) == pytest.approx(magnet_exact(1000.0), abs=1e-3)


def test_rkn4_adaptive_hard_edge_with_h_max_sees_the_magnet():
    assert integrate_sections(magnet_rhs(), [(0.0, 1000.0)], h_max=50.0) == pytest.approx(
        magnet_exact(1000.0), abs=0.1)


@pytest.mark.xfail(strict=True, reason="Known limitation: without h_max a step can jump over the whole "
                                       "magnet, because the error is zero outside it")
def test_rkn4_adaptive_hard_edge_without_h_max_sees_the_magnet():
    assert integrate_sections(magnet_rhs(), [(0.0, 1000.0)]) == pytest.approx(magnet_exact(1000.0), abs=0.1)


# ----- rk4 and rk45: single step ---------------------------------------------------------------

def test_rk4_terms_exact_for_cubic():
    """RK4 reduces to Simpson's rule when f depends only on z."""
    x = rk4_terms(lambda z, x: 4*z**3, 1.0, 2.0, 0.5)
    assert x == pytest.approx(2.0 + 1.5**4 - 1.0)


def test_rk45_terms_exact_for_quartic():
    """The 5th order solution is exact when x' is a polynomial of degree 4 in z."""
    f = lambda z, x: 5*z**4
    x, _, _ = rk45_terms(f, 1.0, 2.0, 0.5, f(1.0, 2.0))
    assert x == pytest.approx(2.0 + 1.5**5 - 1.0)


def test_rk45_error_estimate_zero_for_cubic_nonzero_for_quartic():
    """The embedded solution is order 4, so both solutions agree for a cubic x'."""
    cubic = lambda z, x: 4*z**3
    quartic = lambda z, x: 5*z**4
    _, err_cubic, _ = rk45_terms(cubic, 1.0, 2.0, 0.5, cubic(1.0, 2.0))
    _, err_quartic, _ = rk45_terms(quartic, 1.0, 2.0, 0.5, quartic(1.0, 2.0))
    assert err_cubic == pytest.approx(0.0, abs=1e-14)
    assert err_quartic > 1e-6


def test_rk45_terms_k7_is_f_at_the_new_point():
    x0 = gauss_x(0.3)
    x_new, _, k7 = rk45_terms(gauss_rhs, 0.3, x0, 0.2, gauss_rhs(0.3, x0))
    assert k7 == pytest.approx(gauss_rhs(0.5, x_new))


def test_rk45_terms_local_error_is_order_6():
    z0 = 0.3
    x0 = oscillator_x(z0)
    k1 = oscillator_rhs(z0, x0)
    errors = [np.max(np.abs(rk45_terms(oscillator_rhs, z0, x0, h, k1)[0] - oscillator_x(z0 + h)))
              for h in (0.4, 0.2, 0.1, 0.05)]
    assert np.all(np.abs(slopes(errors) - 6) < 0.2)


def test_rk45_error_estimate_scales_as_h5():
    z0 = 0.3
    x0 = oscillator_x(z0)
    k1 = oscillator_rhs(z0, x0)
    estimates = [np.max(rk45_terms(oscillator_rhs, z0, x0, h, k1)[1]) for h in (0.4, 0.2, 0.1, 0.05)]
    assert np.all(np.abs(slopes(estimates) - 5) < 0.2)


def test_rk45_is_order_5():
    errors = [np.max(np.abs(fixed_step_rk45(oscillator_rhs, 0.0, 10.0, [1.0, 0.0], n) - oscillator_x(10.0)))
              for n in (20, 40, 80)]
    assert np.all(np.abs(slopes(errors) - 5) < 0.3)


# ----- rk4: fixed step -------------------------------------------------------------------------

def test_rk4_fixed_matches_exact_solution():
    z, x = rk4(gauss_rhs, 0.0, 2.0, 1.0, 0.01)
    assert np.max(np.abs(x - gauss_x(z))) < 1e-8


def test_rk4_fixed_grid():
    z, x = rk4(gauss_rhs, 0.0, 10.0, 1.0, 3.0)
    assert z[0] == 0.0 and z[-1] == 10.0
    assert len(z) == 5
    assert np.allclose(np.diff(z), 2.5)
    assert x.shape == (5,)


def test_rk4_fixed_is_order_4():
    errors = [abs(rk4(gauss_rhs, 0.0, 2.0, 1.0, h)[1][-1] - gauss_x(2.0)) for h in (0.1, 0.05, 0.025)]
    assert np.all(np.abs(slopes(errors) - 4) < 0.3)


def test_rk4_fixed_backwards():
    z, x = rk4(gauss_rhs, 0.0, -2.0, 1.0, 0.01)
    assert z[-1] == -2.0
    assert x[-1] == pytest.approx(gauss_x(-2.0), abs=1e-8)


def test_rk4_fixed_vector_state():
    z, x = rk4(oscillator_rhs, 0.0, 10.0, [1.0, 0.0], 0.01)
    assert x.shape == (len(z), 2)
    assert x[-1] == pytest.approx(oscillator_x(10.0), abs=1e-8)


# ----- rk4: adaptive step ----------------------------------------------------------------------

def test_rk4_adaptive_reaches_z_end_exactly():
    z, _ = rk4(gauss_rhs, 0.0, 2.0, 1.0, 0.1, adaptative=True, tol=1e-8)
    assert z[0] == 0.0 and z[-1] == 2.0
    assert np.all(np.diff(z) > 0)


def test_rk4_adaptive_error_decreases_with_tol():
    errors = [abs(rk4(gauss_rhs, 0.0, 2.0, 1.0, 0.1, adaptative=True, tol=tol)[1][-1] - gauss_x(2.0))
              for tol in (1e-4, 1e-6, 1e-8, 1e-10)]
    assert errors[0] > errors[1] > errors[2] > errors[3]
    assert errors[0] < 1e-4 and errors[3] < 1e-10


@pytest.mark.parametrize("h0", [1e-3, 0.1, 5.0])
def test_rk4_adaptive_every_stored_step_satisfies_tol(h0):
    tol = 1e-8
    z, x = rk4(gauss_rhs, 0.0, 2.0, 1.0, h0, adaptative=True, tol=tol)
    for k in range(len(z) - 1):
        x_new, err, _ = rk45_terms(gauss_rhs, z[k], x[k], z[k + 1] - z[k], gauss_rhs(z[k], x[k]))
        assert np.max(err) <= tol * (1 + 1e-6)
        assert x_new == pytest.approx(x[k + 1], abs=1e-12)


def test_rk4_adaptive_tol_per_component():
    """With one tol per component, each component of every step satisfies its own tol."""
    rhs = lambda z, x: np.array([oscillator_rhs(z, x[:2])[0], oscillator_rhs(z, x[:2])[1] * 1e-3])
    tol = np.array([1e-8, 1e-11])
    z, x = rk4(lambda z, x: rhs(z, x), 0.0, 5.0, [1.0, 0.0], 0.1, adaptative=True, tol=tol)
    for k in range(len(z) - 1):
        _, err, _ = rk45_terms(rhs, z[k], x[k], z[k + 1] - z[k], rhs(z[k], x[k]))
        assert np.all(err <= tol * (1 + 1e-6))


def test_rk4_adaptive_rejects_non_positive_tol_component():
    with pytest.raises(ValueError):
        rk4(oscillator_rhs, 0.0, 1.0, [1.0, 0.0], 0.1, adaptative=True, tol=np.array([1e-6, 0.0]))


def test_rk4_adaptive_reuses_k1():
    """Each attempted step costs 6 new evaluations, plus 1 for the very first k1."""
    calls = [0]

    def counted(z, x):
        calls[0] += 1
        return gauss_rhs(z, x)

    z, _ = rk4(counted, 0.0, 2.0, 1.0, 0.1, adaptative=True, tol=1e-8)
    assert (calls[0] - 1) % 6 == 0
    assert calls[0] - 1 >= 6 * (len(z) - 1)


def test_rk4_adaptive_backwards():
    z, x = rk4(gauss_rhs, 0.0, -2.0, 1.0, 0.1, adaptative=True, tol=1e-8)
    assert z[-1] == -2.0
    assert np.all(np.diff(z) < 0)
    assert x[-1] == pytest.approx(gauss_x(-2.0), abs=1e-8)


def test_rk4_adaptive_vector_state():
    z, x = rk4(oscillator_rhs, 0.0, 10.0, [1.0, 0.0], 0.1, adaptative=True, tol=1e-9)
    assert x.shape == (len(z), 2)
    assert x[-1] == pytest.approx(oscillator_x(10.0), abs=1e-7)


def test_rk4_adaptive_respects_h_max():
    z, _ = rk4(gauss_rhs, 0.0, 2.0, 1.0, 0.1, adaptative=True, tol=1e-6, h_max=0.05)
    assert np.max(np.diff(z)) <= 0.05 + 1e-12


def test_rk4_adaptive_zero_length_interval():
    z, x = rk4(gauss_rhs, 1.0, 1.0, 3.0, 0.1, adaptative=True, tol=1e-6)
    assert z.tolist() == [1.0] and x.tolist() == [3.0]


def test_rk4_adaptive_unreachable_tol_raises():
    with pytest.raises(RuntimeError):
        rk4(gauss_rhs, 0.0, 2.0, 1.0, 0.1, adaptative=True, tol=1e-30, h_min=0.05)


# ----- Input validation and behaviour with NaN -------------------------------------------------

def run_rkn4(h=1.0, **kwargs):
    return rkn4(circle_rhs, 0.0, 10.0, [0.0], [0.0], h, **kwargs)


def run_rk4(h=1.0, **kwargs):
    return rk4(gauss_rhs, 0.0, 1.0, 1.0, h, **kwargs)


@pytest.mark.parametrize("solver", [run_rkn4, run_rk4])
@pytest.mark.parametrize("h", [0.0, -1.0])
def test_non_positive_h_raises(solver, h):
    with pytest.raises(ValueError):
        solver(h=h)


@pytest.mark.parametrize("solver", [run_rkn4, run_rk4])
@pytest.mark.parametrize("tol", [None, 0.0, -1e-5])
def test_adaptive_needs_positive_tol(solver, tol):
    with pytest.raises(ValueError):
        solver(adaptative=True, tol=tol)


class TooManyCalls(Exception):
    """Raised by the test functions below to stop a loop that would never end."""


def nan_after_guard(limit=20000):
    calls = [0]

    def guard():
        calls[0] += 1
        if calls[0] > limit:
            raise TooManyCalls(f"more than {limit} evaluations: the loop does not stop")

    return guard


def test_rkn4_adaptive_stops_on_nan():
    """A NaN in fun must give an exception, not an endless loop."""
    guard = nan_after_guard()

    def rhs(z, x, xp):
        guard()
        return np.nan * np.ones_like(x)

    with pytest.raises((RuntimeError, ValueError)):
        rkn4(rhs, 0.0, 10.0, [0.0], [0.0], 1.0, adaptative=True, tol=1e-5)


def test_rk4_adaptive_stops_on_nan():
    """A NaN in fun must give an exception, not an endless loop."""
    guard = nan_after_guard()

    def rhs(z, x):
        guard()
        return np.nan * np.ones_like(x)

    with pytest.raises((RuntimeError, ValueError)):
        rk4(rhs, 0.0, 10.0, [1.0], 1.0, adaptative=True, tol=1e-5)

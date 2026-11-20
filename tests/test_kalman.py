# tests/test_kalman.py
"""Kalman filter and smoother: Jacobians, process noise against the Monte Carlo kicks, exactness of
the linear case (no field, no material) and the fit of many tracks."""
import numpy as np
import pandas as pd
import pytest

from src.constants import PROTON
from src.field import HardEdgeField, NoField
from src.geometry.geometry import Geometry
from src.kalman.filter import run_filter
from src.kalman.fit import fit_tracks, fit_tracks_parallel
from src.kalman.smoother import run_smoother
from src.kalman.transport import Transport
from src.propagator.propagator import Propagator, State, momentum

Z_PH, Z_MAG, L_MAG, HALF_GAP = 50.0, 400.0, 300.0, 50.0
Q_P = PROTON.charge / momentum(230.0, PROTON)     # 230 MeV proton


def vacuum_transport():
    # magnet present but no field: straight lines everywhere
    g = Geometry(z_start_magnet=Z_MAG, z_length_magnet=L_MAG, spacing_after_last=100.0)
    return Transport(Propagator(g, NoField(), np.inf, step=5.0))


def magnet_transport(central=False):
    g = Geometry(z_start_magnet=Z_MAG, z_length_magnet=L_MAG, spacing_after_last=100.0)
    field = HardEdgeField(B0=1.0, z_start=Z_MAG, z_length=L_MAG, x_half_gap=HALF_GAP)
    return Transport(Propagator(g, field, HALF_GAP, step=5.0), central=central)


def phantom_propagator(thickness=100.0, max_slice=20.0):
    g = Geometry(z_start_phantom=Z_PH, phantom_layers=[{"material": "WATER", "thickness": thickness}],
                 spacing_after_last=100.0)
    return Propagator(g, NoField(), np.inf, step=5.0, max_slice=max_slice)


def test_free_space_jacobian_is_the_straight_line_and_no_noise():
    x_pred, F, Q = vacuum_transport().predict(np.array([1.0, 0.01, Q_P]), 0.0, 100.0)
    assert x_pred == pytest.approx([2.0, 0.01, Q_P])
    assert F == pytest.approx(np.array([[1.0, 100.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), abs=1e-6)
    assert np.all(Q == 0.0)


def test_magnet_jacobian_forward_and_central_differences_agree():
    x = np.array([0.5, 0.002, Q_P])
    _, F_fwd, _ = magnet_transport(central=False).predict(x, 380.0, 720.0)
    _, F_cen, _ = magnet_transport(central=True).predict(x, 380.0, 720.0)
    assert F_fwd == pytest.approx(F_cen, rel=1e-4, abs=1e-8)
    # the bending depends on q/p: this is what measures the momentum
    assert abs(F_cen[0, 2]) > 0 and abs(F_cen[1, 2]) > 0
    assert F_cen[2] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)    # |p| constant in vacuum


def test_phantom_energy_loss_enters_the_jacobian():
    transport = Transport(phantom_propagator())
    x_pred, F, _ = transport.predict(np.array([0.0, 0.0, Q_P]), 0.0, 200.0)
    assert x_pred[2] > Q_P                     # slows down: |q/p| grows
    assert F[2, 2] > 1.0                       # and a slower particle loses more energy


def test_process_noise_matches_the_spread_of_the_monte_carlo_kicks():
    prop = phantom_propagator()
    x0 = np.array([0.0, 0.0, Q_P])
    _, _, Q = Transport(prop).predict(x0, 0.0, 200.0)

    rng = np.random.default_rng(1)
    finals = np.array([[s.x, s.tx] for s in
                       (prop.propagate(State(z=0.0, x=0.0, tx=0.0, q_p=Q_P), 200.0, rng=rng).final
                        for _ in range(1000))])
    cov = np.cov(finals.T)
    # 1000 samples: the variances are known to ~4.5 %
    assert cov[0, 0] == pytest.approx(Q[0, 0], rel=0.15)
    assert cov[1, 1] == pytest.approx(Q[1, 1], rel=0.15)
    assert cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) == pytest.approx(
        Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1]), abs=0.05)
    # No energy-loss straggling: q/p only gets the (tiny) variance of the kicks transported by F
    # (a larger tx means a longer path and more energy loss)
    assert Q[2, 2] < 1e-12 * Q_P**2


def linear_problem():
    z = np.array([25.0, 100.0, 200.0, 350.0, 500.0])
    x0 = np.array([0.0, 0.0, Q_P])
    C0 = np.diag([1.0, 1e-6, 1e-10])
    rng = np.random.default_rng(3)
    truth = rng.multivariate_normal(x0, C0)
    sigma = np.full(len(z), 0.1)
    m = truth[0] + truth[1] * z + rng.normal(0.0, sigma)
    return z, m, sigma, x0, C0


def test_linear_case_filter_and_smoother_are_the_exact_posterior():
    z, m, sigma, x0, C0 = linear_problem()
    res = run_filter(vacuum_transport(), x0, C0, 0.0, z, m, sigma)
    smooth = run_smoother(res)

    # Batch posterior of the state at z = 0: m = A theta + noise, A = [1, z, 0]
    A = np.stack([np.ones_like(z), z, np.zeros_like(z)], axis=1)
    W = np.diag(1.0 / sigma**2)
    C_post = np.linalg.inv(np.linalg.inv(C0) + A.T @ W @ A)
    theta = C_post @ (np.linalg.solve(C0, x0) + A.T @ W @ m)
    for k, zk in enumerate(z):
        F = np.array([[1.0, zk, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert smooth.x_smooth[k] == pytest.approx(F @ theta, rel=1e-6, abs=1e-9)
        assert smooth.C_smooth[k][:2, :2] == pytest.approx((F @ C_post @ F.T)[:2, :2], rel=1e-5, abs=1e-14)

    # the last smoothed state is the last filtered one, and q/p is not measured without field
    assert smooth.x_smooth[-1] == pytest.approx(res.x_filt[-1])
    assert res.x_filt[-1][2] == pytest.approx(Q_P)
    assert res.ndf == len(z) and np.all(res.chi2_k >= 0)


def test_parallel_fit_is_the_serial_one():
    rng = np.random.default_rng(5)
    planes = np.array([25.0, 100.0, 200.0, 350.0, 500.0])
    rows = []
    for track_id in range(12):
        x, tx = rng.normal(0.0, 1.0), rng.normal(0.0, 1e-3)
        for k, z in enumerate(planes):
            rows.append((track_id, track_id, track_id, k, f"D{k}", z, x + tx * z + rng.normal(0, 0.1), 0.1))
    hits = pd.DataFrame(rows, columns=["track_id", "input_row", "event_id", "plane_index", "plane", "z",
                                       "x_meas", "sigma_x"])
    prior = (0.0, np.array([0.0, 0.0, Q_P]), np.diag([1.0, 1e-6, 1e-10]))

    fit_s, tracks_s = fit_tracks(hits, vacuum_transport(), prior)
    fit_p, tracks_p = fit_tracks_parallel(hits, vacuum_transport(), prior, n_workers=1, progress=False)
    pd.testing.assert_frame_equal(fit_s, fit_p)
    pd.testing.assert_frame_equal(tracks_s, tracks_p)
    assert (tracks_s["status"] == "ok").all() and (tracks_s["ndf"] == len(planes)).all()

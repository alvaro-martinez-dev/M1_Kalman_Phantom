# tests/test_range_constraints.py
"""Range constraints of the Kalman filter (truncated Gaussian, src/kalman/constraints.py): the
truncation itself, the boundary s* found with the propagator, that tracks with energy to spare are
not affected and that tracks that stop in the phantom are fitted with correct errors."""
import numpy as np
import pandas as pd
import pytest

from src.constants import PROTON, WATER, Particle
from src.field import NoField
from src.geometry.geometry import Geometry
from src.kalman.constraints import truncate_qp
from src.kalman.filter import apply_reach_if_needed, run_filter
from src.kalman.fit import fit_tracks
from src.kalman.transport import Transport, TrackStopped
from src.phantom.materials import range_csda
from src.propagator.propagator import Propagator, State, _stop_margin, kinetic_energy, momentum


def water_transport(layers=(100.0,), z_start=50.0, step=1.0):
    g = Geometry(z_start_phantom=z_start,
                 phantom_layers=[{"material": "WATER", "thickness": t} for t in layers],
                 spacing_after_last=100.0)
    return Transport(Propagator(g, NoField(), np.inf, PROTON, step=step))


def qp(T):
    return PROTON.charge / momentum(T, PROTON)


# ----- truncation ---------------------------------------------------------------------------------

@pytest.mark.parametrize("bound", ["hi", "lo", "both"])
def test_truncation_is_the_moments_of_the_truncated_gaussian(bound):
    mu = np.array([0.5, 1e-3, 3e-3])
    sd = np.array([1.0, 2e-3, 1e-4])
    corr = np.array([[1.0, 0.3, -0.6], [0.3, 1.0, 0.4], [-0.6, 0.4, 1.0]])
    C = corr * np.outer(sd, sd)
    s_lo = mu[2] - 0.3 * sd[2] if bound in ("lo", "both") else None
    s_hi = mu[2] + 0.8 * sd[2] if bound in ("hi", "both") else None

    x_t, C_t = truncate_qp(mu, C, +1, s_lo=s_lo, s_hi=s_hi)

    samples = np.random.default_rng(0).multivariate_normal(mu, C, size=2_000_000)
    keep = np.ones(len(samples), bool)
    if s_lo is not None:
        keep &= samples[:, 2] > s_lo
    if s_hi is not None:
        keep &= samples[:, 2] < s_hi
    kept = samples[keep]
    # ~1e6 samples kept: statistical precision ~1e-3 of the sigmas
    assert np.all(np.abs(x_t - kept.mean(axis=0)) < 5e-3 * sd)
    assert np.all(np.abs(C_t - np.cov(kept.T)) < 1e-2 * np.outer(sd, sd))


def test_truncation_negative_charge_uses_s_equal_to_one_over_p():
    # q/p < 0: s = -q/p. Truncating s < s_hi must give q/p > -s_hi
    x, C = np.array([0.0, 0.0, -3e-3]), np.diag([1.0, 1e-6, 1e-8])
    x_t, _ = truncate_qp(x, C, -1, s_hi=3e-3)
    assert -3e-3 < x_t[2] < 0.0


def test_truncation_far_in_the_tail_is_finite_and_inside():
    # mean 60 sigma beyond the bound: the mass is on the bound
    x, C = np.array([0.0, 0.0, 4e-3]), np.diag([1.0, 1e-6, 1e-10])
    x_t, C_t = truncate_qp(x, C, +1, s_hi=4e-3 - 60 * 1e-5)
    assert np.all(np.isfinite(x_t)) and np.all(np.isfinite(C_t))
    assert x_t[2] < 4e-3 - 60 * 1e-5 and C_t[2, 2] < 1e-10


# ----- boundary s* ----------------------------------------------------------------------------------

@pytest.mark.parametrize("tx", [0.0, 0.1])
def test_reach_limit_is_the_csda_range(tx):
    # One water layer, no slices: the track gets from 50 to 90 mm iff its residual range is larger than
    # the path, 40 sqrt(1 + tx^2) mm, plus the stop margin of the propagator
    transport = water_transport()
    s_star = transport.reach_limit(np.array([0.0, tx, qp(100.0)]), 50.0, 90.0, s_tol=1e-13)
    T_star = kinetic_energy(s_star, PROTON)      # s = q/p for a proton
    expected = 40.0 * np.sqrt(1.0 + tx**2) + _stop_margin(WATER, PROTON)
    assert range_csda(T_star, WATER, PROTON) == pytest.approx(expected, rel=1e-4)


def test_reach_limit_with_charge_two_is_the_csda_range():
    # s = |q|/p (= sign(q) q/p, as the filter uses it), not 1/p: for an alpha the boundary must still be
    # the energy whose CSDA range is the path, which needs the bracket of the bisection in |q|/p
    alpha = Particle("alpha", 3727.37941, +2)
    g = Geometry(z_start_phantom=50.0, phantom_layers=[{"material": "WATER", "thickness": 100.0}],
                 spacing_after_last=100.0)
    transport = Transport(Propagator(g, NoField(), np.inf, particle=alpha, step=1.0))
    x = np.array([0.0, 0.0, alpha.charge / momentum(300.0, alpha)])
    s_star = transport.reach_limit(x, 50.0, 60.0, s_tol=1e-13)
    assert np.isfinite(s_star)
    T_star = kinetic_energy(s_star, alpha)        # q/p = s for a positive charge
    expected = 10.0 + _stop_margin(WATER, alpha)
    assert range_csda(T_star, WATER, alpha) == pytest.approx(expected, rel=1e-4)


def test_reach_limit_without_material_is_infinite():
    transport = water_transport()
    assert transport.reach_limit(np.array([0.0, 0.0, qp(100.0)]), 0.0, 40.0, s_tol=1e-12) == np.inf
    assert transport.reach_limit(np.array([0.0, 0.0, qp(100.0)]), 160.0, 250.0, s_tol=1e-12) == np.inf


# ----- filter -------------------------------------------------------------------------------------

def test_tracks_with_energy_to_spare_are_not_affected():
    # 230 MeV protons (range ~33 cm) through 10 cm of water: the constraints are irrelevant, so the
    # fit is exactly the unconstrained one
    transport = water_transport()
    z = np.array([25.0, 50.0, 100.0, 150.0, 250.0])
    m = np.array([0.1, -0.05, 0.02, 0.0, 0.08])
    x0, C0 = np.array([0.0, 0.0, qp(230.0)]), np.diag([1.0, 1e-6, (3.0 / momentum(230.0, PROTON)**2)**2])
    free = run_filter(transport, x0, C0, 0.0, z, m, np.full(5, 0.1))
    constrained = run_filter(transport, x0, C0, 0.0, z, m, np.full(5, 0.1), z_next=None,
                             range_constraints=True)
    assert np.array_equal(free.x_filt, constrained.x_filt) and np.array_equal(free.C_filt, constrained.C_filt)
    assert np.all(np.isnan(constrained.s_reach)) and np.isnan(constrained.s_hi) and np.isnan(constrained.s_lo)


def test_a_track_beyond_the_predicted_range_is_fitted_with_the_constraint():
    # The prior says ~100 MeV (range ~77 mm), but the track left a hit at the end of 100 mm of water:
    # without constraints the predicted track stops before it; with them, the energy is pushed up
    transport = water_transport()
    z = np.array([25.0, 50.0, 100.0, 150.0])
    m = np.zeros(4)
    p0 = momentum(100.0, PROTON)
    x0, C0 = np.array([0.0, 0.0, 1.0 / p0]), np.diag([1.0, 1e-6, (20.0 / p0**2)**2])
    with pytest.raises(TrackStopped):
        run_filter(transport, x0, C0, 0.0, z, m, np.full(4, 0.1))

    res = run_filter(transport, x0, C0, 0.0, z, m, np.full(4, 0.1), range_constraints=True)
    assert np.isfinite(res.s_hi) and np.isnan(res.s_lo)
    assert res.x_filt[0][2] < res.s_hi                       # the bound holds (before the water)
    assert res.x_filt[0][2] < x0[2]                          # and the energy grew with respect to the prior


def test_safety_net_makes_the_prediction_possible():
    # A state whose mean stops in the water before z = 140 mm is truncated so that its mean reaches it
    transport = water_transport()
    x, C = np.array([0.0, 0.0, qp(100.0)]), np.diag([1.0, 1e-6, (20.0 / momentum(100.0, PROTON)**2)**2])
    assert not transport.reaches(x, x[2], 40.0, 140.0)
    x_t, _, s_reach = apply_reach_if_needed(transport, x, C, 40.0, 140.0)
    assert np.isfinite(s_reach) and x_t[2] < s_reach and transport.reaches(x_t, x_t[2], 40.0, 140.0)
    # and does nothing if the mean gets there
    assert np.isnan(apply_reach_if_needed(transport, x, C, 40.0, 60.0)[2])


def stopping_sample(n_tracks, seed=11):
    # Eight water layers of 5 mm from z = 50 mm, a plane at each interface (and at 10 and 190 mm).
    # No field and no scattering, so the energy is only known from the prior and from where the track
    # stopped. 60 +- ~9 MeV protons (range ~31 mm): almost all stop in the phantom
    transport = water_transport(layers=(5.0,) * 8, step=0.5)
    prop = transport.propagator
    planes = np.concatenate(([10.0], prop.geometry.z_phantom_interfaces, [prop.geometry.z_end_system]))
    p0, sigma_p = momentum(60.0, PROTON), 20.0
    prior = (0.0, np.array([0.0, 0.0, 1.0 / p0]), np.diag([1.0, 1e-6, (sigma_p / p0**2)**2]))

    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for track_id in range(n_tracks):
        start = State(z=0.0, x=rng.normal(0.0, 1.0), tx=rng.normal(0.0, 1e-3),
                      q_p=1.0 / rng.normal(p0, sigma_p))
        res = prop.propagate(start, planes[-1], z_out=planes.tolist())
        for k, s in enumerate(res.at_z_out):
            if s is None:
                break
            rows.append((track_id, track_id, track_id, k, f"D{k}", s.z, s.x + rng.normal(0.0, 0.1), 0.1))
            truth.append((track_id, k, s.q_p, start.q_p))
    hits = pd.DataFrame(rows, columns=["track_id", "input_row", "event_id", "plane_index", "plane", "z",
                                       "x_meas", "sigma_x"])
    truth = pd.DataFrame(truth, columns=["track_id", "plane_index", "q_over_p_true", "q_over_p0_true"])
    return transport, prior, planes, hits, truth


def test_range_constraints_need_the_planes():
    # without the planes the stop of a track cannot be seen: an error, not a silent half constraint
    transport, prior, planes, hits, _ = stopping_sample(3)
    with pytest.raises(ValueError, match="planes"):
        fit_tracks(hits, transport, prior, range_constraints=True)
    fit_tracks(hits, transport, prior)    # without range constraints the planes are not needed


def test_tracks_that_stop_are_fitted_with_correct_errors():
    transport, prior, planes, hits, truth = stopping_sample(300)
    fit, tracks = fit_tracks(hits, transport, prior, smoother=True, planes_z=planes, range_constraints=True)

    # without constraints many of them could not be fitted at all
    _, free = fit_tracks(hits, transport, prior, smoother=True)
    assert (free["status"] == "stopped").sum() > 20

    assert (tracks["status"] == "ok").all()
    stopped = tracks[tracks["range_end"]]
    assert len(stopped) > 200
    assert np.isfinite(stopped["s_lo"]).all()

    df = fit.merge(truth, on=["track_id", "plane_index"])
    df = df[df["track_id"].isin(stopped["track_id"])]
    # the true initial s is inside the bounds (exact model: no scattering, no straggling)
    bounds = stopped.set_index("track_id")
    first = df[df["plane_index"] == 0]
    assert (first["q_over_p0_true"] > first["track_id"].map(bounds["s_lo"])).all()
    s_hi = first["track_id"].map(bounds["s_hi"])
    assert (np.isnan(s_hi) | (first["q_over_p0_true"] < s_hi)).all()

    # q/p pulls are N(0, 1) at every hit except the last one (see src/kalman/filter.py)
    df = df.assign(back=df.groupby("track_id")["plane_index"].transform("max") - df["plane_index"])
    for name, d in (("phantom entrance", df[df["plane_index"] == 1]), ("before the last hit", df[df["back"] >= 1])):
        pull = (d["q_over_p_smooth"] - d["q_over_p_true"]) / d["q_over_p_smooth_err"]
        assert abs(pull.mean()) < 0.15, name
        assert 0.9 < pull.std() < 1.1, name

    # and the range measures the energy: at the phantom entrance the error is several times smaller
    # than that of the beam alone
    entrance = df[df["plane_index"] == 1]
    rel_err = (entrance["q_over_p_smooth_err"] / entrance["q_over_p_smooth"]).median()
    assert rel_err < 0.3 * np.sqrt(prior[2][2, 2]) / prior[1][2]

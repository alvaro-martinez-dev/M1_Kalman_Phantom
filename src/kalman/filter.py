# src/kalman/filter.py
#
# Forward Kalman filter of one track. State x = (x, tx, q/p) [mm, -, e/MeV], covariance C (3x3).
#
# Measurement: each detector plane measures only x, so every hit is a scalar
#   m_k = H x_k + eps_k,   H = [1, 0, 0],   eps_k ~ N(0, V_k),   V_k = sigma_x**2
#
# For each hit k (in beam order), starting from the prior (x_0, C_0) at z_0:
#   predict:  x_pred = f(x_{k-1}),  C_pred = F C_{k-1} F^T + Q            (src/kalman/transport.py)
#   update:   r = m - H x_pred,  S = H C_pred H^T + V,  K = C_pred H^T / S
#             x_filt = x_pred + K r
#             C_filt = (I - K H) C_pred (I - K H)^T + K V K^T             (Joseph form)
#             chi2_k = r^2 / S
#
# With a prior that is the true distribution of the tracks (the beam, see src/kalman/fit.py), the
# predicted residuals r / sqrt(S) are independent N(0, 1), so chi2 = sum chi2_k follows a chi2
# with ndf = n_hits (not n_hits - 3: the prior is information, not a free fit).
#
# The filter keeps everything the smoother needs (x_pred, C_pred, x_filt, C_filt, F at each hit).
#
# Range constraints (range_constraints=True, see src/kalman/constraints.py). The presence or absence
# of hits bounds the energy, which the positions alone barely measure inside the phantom. Without
# energy-loss straggling the energy loss is deterministic, so all this information is about the
# INITIAL energy, and it is applied to the prior at z0, where its Gaussian is accurate:
#   reached the last hit z_last         ->  s0 < s_hi = s*(z0, z_last)
#   no hit on the next plane z_next     ->  s0 > s_lo = s*(z0, z_next)   (the track stopped)
# both in one truncation. The filter then propagates a state that already carries the range
# information, and the smoother brings it to every hit.
# Why not hit by hit, at the end of the range: there q/p is very non-linear in the energy, a Gaussian
# in q/p is a poor description, and truncating and re-propagating it hit by hit biases the estimate
# (tested: q/p pulls with sigma 3-6). On the prior the q/p pulls are N(0, 1) at every hit except the
# LAST one of a track that stops: there the true energy is anywhere between ~1 MeV and the energy to
# cross the layer, which no Gaussian in q/p describes (pull sigma ~2; the same without constraints).
# The scattering errors on x, tx are barely affected: q/p only enters them through theta0.
# Safety net: if, even so, the mean track would stop before the next hit (the positions moved q/p),
# the state is truncated there so that the prediction exists (n_reach counts it; rare).
# A bound further than N_SIGMA_IRRELEVANT sigma from the mean is skipped (one cheap propagation to
# check it), so for tracks with energy to spare the fit is exactly the unconstrained one.
# Assumptions: 100 % efficient planes (a missing hit means that the particle stopped), no straggling
# (the bounds are sharp), the path length of the mean track (the scattering lengthens it only to
# second order in the angles).

from dataclasses import dataclass, field

import numpy as np

from src.kalman.constraints import N_SIGMA_IRRELEVANT, QP, truncate_qp

H = np.array([[1.0, 0.0, 0.0]])

# Precision of the bound s* of a range constraint, in units of the sigma of s of the state
S_TOL_SIGMA = 1.0e-3


@dataclass
class FilterResult:
    z: np.ndarray        # (n,) z of the hits [mm]
    m: np.ndarray        # (n,) measured x [mm]
    V: np.ndarray        # (n,) measurement variances [mm^2]
    x_pred: np.ndarray   # (n, 3) predicted state at each hit
    C_pred: np.ndarray   # (n, 3, 3)
    x_filt: np.ndarray   # (n, 3) filtered state at each hit (after its update)
    C_filt: np.ndarray   # (n, 3, 3)
    F: np.ndarray        # (n, 3, 3) F[k]: transport from the previous hit (or the prior) to hit k
    r: np.ndarray        # (n,) predicted residuals m - H x_pred
    S: np.ndarray        # (n,) their variances
    chi2_k: np.ndarray   # (n,) chi2 increment of each hit
    # Range constraints (NaN: not applied, either disabled or irrelevant, i.e. > N_SIGMA_IRRELEVANT away)
    s_reach: np.ndarray = field(default=None)   # (n,) safety-net bound s < s_reach[k] before hit k [e/MeV]
    s_hi: float = np.nan                        # bound s0 < s_hi on the prior: reached the last hit [e/MeV]
    s_lo: float = np.nan                        # bound s0 > s_lo on the prior: did not reach z_next [e/MeV]

    @property
    def chi2(self):
        return float(self.chi2_k.sum())

    @property
    def ndf(self):
        return len(self.z)


# x_pred, C_pred: predicted state and covariance; m: measured x; V: its variance
# Returns: (x_filt, C_filt, r, S)
def update(x_pred, C_pred, m, V):
    r = m - (H @ x_pred)[0]
    S = (H @ C_pred @ H.T)[0, 0] + V
    K = (C_pred @ H.T)[:, 0] / S
    I_KH = np.eye(3) - np.outer(K, H[0])
    x_filt = x_pred + K * r
    C_filt = I_KH @ C_pred @ I_KH.T + V * np.outer(K, K)
    return x_filt, C_filt, r, S


# Range constraints on the prior at z0 (see the header):
#   reached the last hit z_last          ->  s0 < s_hi = s*(z0, z_last)
#   no hit on the next plane z_next      ->  s0 > s_lo = s*(z0, z_next)    (only if z_next is given)
# Both bounds are applied together, in one truncation, where the Gaussian of the prior is accurate.
# A bound further than N_SIGMA_IRRELEVANT sigma is skipped (one propagation to check it).
# Returns: (x, C, s_hi, s_lo), the bounds NaN if not applied
def apply_range_to_prior(transport, x, C, z0, z_last, z_next):
    sign_q = np.sign(transport.propagator.particle.charge)
    mu, sd = sign_q * x[QP], np.sqrt(max(C[QP, QP], 0.0))
    if sd == 0.0:
        return x, C, np.nan, np.nan
    tol = _s_tol(mu, sd)

    s_hi = np.nan
    if (transport.crosses_material(z0, z_last)
            and not transport.reaches(x, mu + N_SIGMA_IRRELEVANT * sd, z0, z_last)):
        s_hi = transport.reach_limit(x, z0, z_last, s_tol=tol)       # the slow tail did not get there
    s_lo = np.nan
    if (z_next is not None and transport.crosses_material(z0, z_next)
            and transport.reaches(x, max(mu - N_SIGMA_IRRELEVANT * sd, 0.01 * mu), z0, z_next)):
        s_lo = transport.reach_limit(x, z0, z_next, s_tol=tol)       # the fast tail would get there

    s_hi = s_hi if np.isfinite(s_hi) else np.nan
    s_lo = s_lo if np.isfinite(s_lo) else np.nan
    if np.isnan(s_hi) and np.isnan(s_lo):
        return x, C, s_hi, s_lo
    x, C = truncate_qp(x, C, sign_q, s_lo=None if np.isnan(s_lo) else s_lo,
                       s_hi=None if np.isnan(s_hi) else s_hi)
    return x, C, s_hi, s_lo


# Safety net, before predicting from z_a to the next hit at z_b: if the MEAN track would stop before
# z_b (the positions moved q/p after the prior constraint, or there was none), truncate to s < s*
# so that the prediction exists. With the prior constraint this is rare.
# Returns: (x, C, s_reach), with s_reach = NaN if not applied
def apply_reach_if_needed(transport, x, C, z_a, z_b):
    if not transport.crosses_material(z_a, z_b):
        return x, C, np.nan
    sign_q = np.sign(transport.propagator.particle.charge)
    mu, sd = sign_q * x[QP], np.sqrt(max(C[QP, QP], 0.0))
    if sd == 0.0 or transport.reaches(x, mu, z_a, z_b):
        return x, C, np.nan
    s_reach = transport.reach_limit(x, z_a, z_b, s_tol=_s_tol(mu, sd))
    if not np.isfinite(s_reach):
        return x, C, np.nan
    x, C = truncate_qp(x, C, sign_q, s_hi=s_reach)
    return x, C, s_reach


def _s_tol(mu, sd):
    # precision of the bound: a small fraction of the sigma of s (never below double precision)
    return max(S_TOL_SIGMA * sd, 1e-12 * mu)


# transport: Transport (src/kalman/transport.py)
# x0, C0, z0: prior state, covariance and its z [mm]
# z, m, sigma: (n,) z [mm], measured x [mm] and resolution [mm] of the hits, in beam order (z > z0)
# z_next: [mm] z of the plane after the last hit, if the track left NO hit there (so it stopped
#         before it); None if the last hit is on the last plane. Only used with range_constraints
# range_constraints: apply the range constraints to the prior (see the header)
# Returns: FilterResult
# Raises transport.TrackStopped if the predicted track stops before a hit (without range constraints
# this happens near the end of the range; with them, only in pathological cases)
def run_filter(transport, x0, C0, z0, z, m, sigma, z_next=None, range_constraints=False):
    z, m = np.asarray(z, dtype=float), np.asarray(m, dtype=float)
    V = np.asarray(sigma, dtype=float)**2
    n = len(z)
    if n == 0:
        raise ValueError("No hits to fit")
    if np.any(np.diff(np.concatenate(([z0], z))) < 0):
        raise ValueError("The hits must be in beam order, after z0")
    if z_next is not None and z_next <= z[-1]:
        raise ValueError(f"z_next = {z_next} mm must be after the last hit (z = {z[-1]} mm)")

    out = {name: np.empty((n, 3)) for name in ("x_pred", "x_filt")}
    out.update({name: np.empty((n, 3, 3)) for name in ("C_pred", "C_filt", "F")})
    r, S = np.empty(n), np.empty(n)
    s_reach = np.full(n, np.nan)

    x, C, z_prev = np.asarray(x0, dtype=float), np.asarray(C0, dtype=float), z0
    s_hi = s_lo = np.nan
    if range_constraints:
        x, C, s_hi, s_lo = apply_range_to_prior(transport, x, C, z0, z[-1], z_next)

    for k in range(n):
        if range_constraints:
            x, C, s_reach[k] = apply_reach_if_needed(transport, x, C, z_prev, z[k])
            if k > 0 and np.isfinite(s_reach[k]):   # the constrained state is the filtered one
                out["x_filt"][k - 1], out["C_filt"][k - 1] = x, C
        x_pred, F, Q = transport.predict(x, z_prev, z[k])
        C_pred = F @ C @ F.T + Q
        x, C, r[k], S[k] = update(x_pred, C_pred, m[k], V[k])
        out["x_pred"][k], out["C_pred"][k], out["F"][k] = x_pred, C_pred, F
        out["x_filt"][k], out["C_filt"][k] = x, C
        z_prev = z[k]

    return FilterResult(z=z, m=m, V=V, r=r, S=S, chi2_k=r**2 / S, s_reach=s_reach, s_hi=s_hi, s_lo=s_lo,
                        **out)

# src/kalman/constraints.py
#
# Range constraints of the Kalman filter: inequality constraints on the energy, applied by truncating
# the Gaussian of the state (constrained Kalman filtering by density truncation, D. Simon and
# D. L. Simon, IET Control Theory Appl. 4 (2010) 1303; D. Simon, "Kalman filtering with state
# constraints: a survey", same journal, 1303-1318).
#
# Why: inside the phantom the positions of the hits say almost nothing about the energy (only through
# the multiple scattering), and a track that stops there never reaches the magnet. Its energy
# information is WHERE it stopped, which the detector gives through the presence or absence of hits:
#   it left a hit on the plane at z_k          ->  it had enough energy to get there
#   it left no hit on the plane after its last ->  it had not enough energy to get there
# (100 % efficient planes: a missing hit means that the particle stopped). Each statement bounds
# s = sign(q) q/p = 1/p (> 0 for both signs of the charge; a larger s is a slower particle):
#   reaches z_b from z_a          <=>  s < s*(z_a, z_b)
#   does not reach z_b from z_a   <=>  s > s*(z_a, z_b)
# The filter (src/kalman/filter.py) applies them to the prior, z_a = z0, where the Gaussian is accurate.
# with s* the boundary value, found with the propagator itself (Transport.reach_limit), so the
# criterion of "reaches / stops" is exactly the one that decided in the Monte Carlo whether there was
# a hit.
#
# Truncation: the state (x, tx, q/p) ~ N(mu, C) is replaced by the mean and covariance of the same
# Gaussian restricted to the allowed interval of s. For s, the moments of the 1D truncated normal
# (mean m', variance v'). The other components are linear in s plus an independent Gaussian part
# (exact for a Gaussian), so with g = C[:, 2] / C[2, 2]:
#   mu' = mu + g (q/p' - q/p),     C' = C + g g^T (v' - C[2, 2])
# The constraint is not a measurement with a residual: it does not add to chi2 nor to ndf.
#
# Units: q/p and s in e/MeV and 1/MeV.

import numpy as np
from scipy.stats import truncnorm

QP = 2    # index of q/p in the state (x, tx, q/p)

# A bound further than this many sigma from the mean changes nothing that matters (relative change
# of the mean ~1e-6 sigma): the truncation is skipped, and so is the search of s* (Transport.reach_limit)
N_SIGMA_IRRELEVANT = 5.0


# x, C: state (3,) and covariance (3, 3)
# sign_q: sign of the charge (+1 / -1)
# s_lo, s_hi: allowed interval of s = sign_q * q/p (None: unbounded on that side)
# Returns: (x', C') of the truncated Gaussian. Unchanged if q/p has no variance or the interval is
#          unbounded
def truncate_qp(x, C, sign_q, s_lo=None, s_hi=None):
    x, C = np.asarray(x, dtype=float), np.asarray(C, dtype=float)
    var = C[QP, QP]
    if var <= 0.0 or (s_lo is None and s_hi is None):
        return x.copy(), C.copy()
    if s_lo is not None and s_hi is not None and not s_lo < s_hi:
        raise ValueError(f"Empty interval of s: [{s_lo}, {s_hi}]")

    mu, sd = sign_q * x[QP], np.sqrt(var)
    a = -np.inf if s_lo is None else (s_lo - mu) / sd
    b = np.inf if s_hi is None else (s_hi - mu) / sd
    mean, new_var = truncnorm.stats(a, b, loc=mu, scale=sd, moments="mv")
    mean, new_var = float(mean), float(new_var)
    if not (np.isfinite(mean) and np.isfinite(new_var)) or new_var <= 0.0:
        # Bound absurdly far from the mean (tens of sigma): numerically the mass is on the bound.
        # Exponential tail: mean ~ bound + sd^2 / distance inside, variance ~ (sd^2 / distance)^2
        bound, dist = (s_lo, max(a, 1.0)) if s_lo is not None and a > 0 else (s_hi, max(-b, 1.0))
        step = sd / dist
        mean = bound + step if bound == s_lo else bound - step
        new_var = step**2

    g = C[:, QP] / var
    x_new = x + g * (sign_q * mean - x[QP])
    C_new = C + np.outer(g, g) * (new_var - var)
    return x_new, 0.5 * (C_new + C_new.T)

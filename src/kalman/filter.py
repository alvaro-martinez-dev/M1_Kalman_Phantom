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

from dataclasses import dataclass

import numpy as np

H = np.array([[1.0, 0.0, 0.0]])


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


# transport: Transport (src/kalman/transport.py)
# x0, C0, z0: prior state, covariance and its z [mm]
# z, m, sigma: (n,) z [mm], measured x [mm] and resolution [mm] of the hits, in beam order (z > z0)
# Returns: FilterResult
# Raises transport.TrackStopped if the predicted track stops before a hit
def run_filter(transport, x0, C0, z0, z, m, sigma):
    z, m = np.asarray(z, dtype=float), np.asarray(m, dtype=float)
    V = np.asarray(sigma, dtype=float)**2
    n = len(z)
    if n == 0:
        raise ValueError("No hits to fit")
    if np.any(np.diff(np.concatenate(([z0], z))) < 0):
        raise ValueError("The hits must be in beam order, after z0")

    out = {name: np.empty((n, 3)) for name in ("x_pred", "x_filt")}
    out.update({name: np.empty((n, 3, 3)) for name in ("C_pred", "C_filt", "F")})
    r, S = np.empty(n), np.empty(n)

    x, C, z_prev = np.asarray(x0, dtype=float), np.asarray(C0, dtype=float), z0
    for k in range(n):
        x_pred, F, Q = transport.predict(x, z_prev, z[k])
        C_pred = F @ C @ F.T + Q
        x, C, r[k], S[k] = update(x_pred, C_pred, m[k], V[k])
        out["x_pred"][k], out["C_pred"][k], out["F"][k] = x_pred, C_pred, F
        out["x_filt"][k], out["C_filt"][k] = x, C
        z_prev = z[k]

    return FilterResult(z=z, m=m, V=V, r=r, S=S, chi2_k=r**2 / S, **out)

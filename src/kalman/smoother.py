# src/kalman/smoother.py
#
# Rauch-Tung-Striebel smoother: runs backwards over the output of the forward filter so that the
# estimate at each hit uses ALL the hits of the track, not only the previous ones.
#
#   x_smooth[n-1] = x_filt[n-1],  C_smooth[n-1] = C_filt[n-1]
#   for k = n-2 .. 0:
#       A_k        = C_filt[k] F[k+1]^T C_pred[k+1]^-1
#       x_smooth[k] = x_filt[k] + A_k (x_smooth[k+1] - x_pred[k+1])
#       C_smooth[k] = C_filt[k] + A_k (C_smooth[k+1] - C_pred[k+1]) A_k^T
#
# Smoothed residuals (the hit is included in the estimate, so its variance is V - C_xx, not V + C_xx):
#   r_smooth = m - x_smooth[x],  R_smooth = V - C_smooth[0, 0]
#
# This matters most for q/p: it is measured by the bending in the magnet, after the phantom, so the
# forward filter reaches the phantom planes knowing little more than the prior; the smoother brings
# the information of the magnet back to them.

from dataclasses import dataclass

import numpy as np


@dataclass
class SmootherResult:
    x_smooth: np.ndarray   # (n, 3)
    C_smooth: np.ndarray   # (n, 3, 3)
    r: np.ndarray          # (n,) smoothed residuals m - x_smooth[:, 0]
    R: np.ndarray          # (n,) their variances V - C_smooth[:, 0, 0]


# result: FilterResult (src/kalman/filter.py)
# Returns: SmootherResult
def run_smoother(result):
    n = len(result.z)
    x_s = result.x_filt.copy()
    C_s = result.C_filt.copy()

    for k in range(n - 2, -1, -1):
        # A = C_filt F^T C_pred^-1  <=>  A^T = C_pred^-1 F C_filt (C symmetric)
        A = np.linalg.solve(result.C_pred[k + 1], result.F[k + 1] @ result.C_filt[k]).T
        x_s[k] = result.x_filt[k] + A @ (x_s[k + 1] - result.x_pred[k + 1])
        C_s[k] = result.C_filt[k] + A @ (C_s[k + 1] - result.C_pred[k + 1]) @ A.T
        C_s[k] = 0.5 * (C_s[k] + C_s[k].T)

    r = result.m - x_s[:, 0]
    R = result.V - C_s[:, 0, 0]
    return SmootherResult(x_smooth=x_s, C_smooth=C_s, r=r, R=R)

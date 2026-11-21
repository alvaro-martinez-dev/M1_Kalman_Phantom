"""Interpolation error of the Biot-Savart map as a function of |x|, from the axis to the plates.

The error is the maximum over z (inside the magnet, away from its edges) of |map - source|.
It is computed for the config dx and for finer dx, and the largest |x| below which the error
stays under a few tolerances is reported. Maps are local in z (the magnet +/- 50 mm): the
x behaviour does not depend on the rest of the z range.

    .venv\Scripts\python.exe scans/scan__interp_x_error.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.utils.config_loader import load_config
from src.field import BiotSavartField, InterpolatedField

m = load_config("config")["geometry"]["magnet"]
Z_START, Z_LENGTH = m["z_start"], m["z_length"]
X_HALF_GAP, Y_HEIGHT = m["x_half_gap"], m["y_height"]
N_FIL = m["biot_savart"]["n_filaments"]
DX_CONFIG, DZ = m["interpolation"]["dx"], m["interpolation"]["dz"]
Z_END = Z_START + Z_LENGTH
PITCH = Y_HEIGHT / (N_FIL - 1)

Z_EDGE_MARGIN = 20.0      # stay away from the magnet edges (their z behaviour is a separate issue)
Z_MAP_MARGIN = 50.0       # the local map extends this much beyond the magnet in z
DX_LIST = (DX_CONFIG, DX_CONFIG / 2.0, DX_CONFIG / 4.0)
TOLS = (1e-3, 1e-4, 1e-5)
ZONES = ((0, 10), (10, 25), (25, 40), (40, 45), (45, 48), (48, 50), (50, 52))

src = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT, N_FIL)
z_centre = Z_START + Z_LENGTH / 2.0
print(f"x_half_gap = {X_HALF_GAP} mm, filament pitch = {PITCH:.3f} mm, n_fil = {N_FIL}, "
      f"a filament lies on y = 0: {N_FIL % 2 == 1}")
print(f"By(0, z_centre)/B0 = {src.By(0.0, z_centre):.4f}  (scale of the field)")

x_eval = np.arange(0.0, X_HALF_GAP + 2.0 * DX_CONFIG + 1e-9, 0.25)
z_eval = np.linspace(Z_START + Z_EDGE_MARGIN, Z_END - Z_EDGE_MARGIN, 25)

errors = {}                                   # dx -> (x values, max error over z)
for dx in DX_LIST:
    x_max = X_HALF_GAP + 2.0 * dx
    fmap = InterpolatedField.from_field(
        src, x_max=x_max, z_min=Z_START - Z_MAP_MARGIN, z_max=Z_END + Z_MAP_MARGIN,
        dx=dx, dz=DZ)
    xs = x_eval[x_eval <= x_max]
    X, Z = np.meshgrid(xs, z_eval, indexing="ij")
    errors[dx] = (xs, np.max(np.abs(fmap.By(X, Z) - src.By(X, Z)), axis=1))

print("\nmax |map - source| over z [units of B0], per |x| zone")
print(f"{'|x| zone [mm]':>14} " + " ".join(f"{'dx=' + format(dx, 'g'):>12}" for dx in DX_LIST))
for lo, hi in ZONES:
    cells = []
    for dx in DX_LIST:
        xs, err = errors[dx]
        sel = (xs >= lo) & (xs <= hi)
        cells.append(f"{err[sel].max():12.3e}" if sel.any() else f"{'-':>12}")
    print(f"{f'{lo}-{hi}':>14} " + " ".join(cells))

print("\nlargest |x| such that the error is below tol for every smaller |x|")
print(f"{'tol':>8} " + " ".join(f"{'dx=' + format(dx, 'g'):>12}" for dx in DX_LIST))
for tol in TOLS:
    cells = []
    for dx in DX_LIST:
        xs, err = errors[dx]
        bad = np.nonzero(err >= tol)[0]
        x_safe = xs[-1] if bad.size == 0 else (xs[bad[0] - 1] if bad[0] > 0 else float("nan"))
        cells.append(f"{x_safe:12.2f}")
    print(f"{tol:8.0e} " + " ".join(cells))
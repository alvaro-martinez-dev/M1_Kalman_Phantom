"""Jump of By across a current plate: (By(d+delta) - By(d-delta)) * pitch -> 1 (mu0 K = 1/pitch).

Ampere's law for a current sheet K along z: the component of B parallel to the sheet and
perpendicular to K (By here) jumps by mu0 K across it. The jump is local, so it holds for the
finite magnet of the config, not only in the infinite-plate limit.

Prints the jump for several delta at the config geometry and at a finer filament pitch,
the linear extrapolation to delta -> 0, and the equivalent surface current.

    .venv\Scripts\python.exe scans\scan_plate_jump.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.utils.config_loader import load_config
from src.field import BiotSavartField

MU0_SI = 4e-7 * np.pi            # T m / A

_m = load_config("config")["geometry"]["magnet"]
Z_START, Z_LENGTH = _m["z_start"], _m["z_length"]
X_HALF_GAP, Y_HEIGHT = _m["x_half_gap"], _m["y_height"]
N_CONFIG = _m["biot_savart"]["n_filaments"]
Z_CENTRE = Z_START + Z_LENGTH / 2.0

REFINE = (1, 4)                  # pitch refinement factors (1 = as in the config)
DELTAS = (1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0)   # [mm]


def jump_times_pitch(f, pitch, delta, y=0.0, side=+1):
    """(By_outer - By_inner) * pitch across the plate at x = side * x_half_gap."""
    d = side * X_HALF_GAP
    out = f._raw_B(d + side * delta, y, Z_CENTRE)[1]       #without _scale mu0I is normalized to 1, so By is in units of mu0 K = 1/pitch
    inn = f._raw_B(d - side * delta, y, Z_CENTRE)[1]
    return (out - inn) * pitch


for k in REFINE:
    n = (N_CONFIG - 1) * k + 1
    pitch = Y_HEIGHT / (n - 1)
    f = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT, n)

    print(f"\n=== n_filaments = {n}, pitch = {pitch:.3f} mm "
          f"(y_height = {Y_HEIGHT}, z_length = {Z_LENGTH}) ===")
    print(f"{'delta [mm]':>10} {'delta/pitch':>11} {'jump x=+d':>10} {'jump x=-d':>10} "
          f"{'jump, y=p/2':>12}")
    rows = []
    for delta in DELTAS:
        jp = jump_times_pitch(f, pitch, delta, side=+1)
        jm = jump_times_pitch(f, pitch, delta, side=-1)
        jy = jump_times_pitch(f, pitch, delta, y=pitch / 2.0, side=+1)
        rows.append((delta, jp))
        print(f"{delta:10.1f} {delta / pitch:11.2f} {jp:10.5f} {jm:10.5f} {jy:12.5f}")

    # linear extrapolation to delta -> 0, using only delta >= 2 pitch (lattice effects gone)
    sel = [(d, j) for d, j in rows if d >= 2.0 * pitch and d <= 20.0]
    if len(sel) >= 2:
        d_arr, j_arr = np.array(sel).T
        slope, j0 = np.polyfit(d_arr, j_arr, 1)
        print(f"extrapolated jump * pitch at delta -> 0: {j0:.5f}   (ideal: 1)")

    # equivalent current: mu0 K [T] = |scale| / pitch, for B0 = 1 T
    mu0K_T = abs(f._scale) / pitch
    print(f"B0 = 1 T  ->  mu0 K = {mu0K_T:.4f} T,  K = {mu0K_T / MU0_SI:.4g} A/m")
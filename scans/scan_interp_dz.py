"""Interpolation error of the Biot-Savart map vs grid spacing dz (run from repo root).
.venv\Scripts\python.exe  scan\scan_interp_dz.py  to call from windows
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.utils.config_loader import load_config
from src.field import BiotSavartField, InterpolatedField

_m = load_config("config")["geometry"]["magnet"]
Z_START = _m["z_start"]
Z_LENGTH = _m["z_length"]
X_HALF_GAP = _m["x_half_gap"]
Y_HEIGHT = _m["y_height"]
DX = _m["interpolation"]["dx"]
N_FIL = _m["n_filaments"]

Z_END = Z_START + Z_LENGTH

src = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT, N_FIL)

# fine evaluation grid for the comparison
z_cmp = np.linspace(0.0, 1000.0, 20001)           # 0.05 mm spacing
edge = (np.abs(z_cmp - Z_START) < 10) | (np.abs(z_cmp - Z_END) < 10)

ref = {x: src.By(x, z_cmp) for x in (0.0, 25.0)}
ref_int = {x: np.trapezoid(ref[x], z_cmp) for x in ref}

print(f"{'dz [mm]':>8} {'n_z':>6} {'max err edge x=0':>17} {'max err all x=0':>16} "
      f"{'dInt x=0':>10} {'dInt x=25':>10}   (errors in T, T*mm)")
for dz in (2.0, 1.0, 0.5, 0.25, 0.1):
    f = InterpolatedField.from_field(src, x_max=X_HALF_GAP + 2 * DX,
                                     z_min=0.0, z_max=1000.0, dx=DX, dz=dz)
    e0 = f.By(0.0, z_cmp) - ref[0.0]
    d0 = np.trapezoid(f.By(0.0, z_cmp), z_cmp) - ref_int[0.0]
    d25 = np.trapezoid(f.By(25.0, z_cmp), z_cmp) - ref_int[25.0]
    print(f"{dz:8.2f} {len(f._zs):6d} {np.max(np.abs(e0[edge])):17.3e} "
          f"{np.max(np.abs(e0)):16.3e} {d0:10.3e} {d25:10.3e}")
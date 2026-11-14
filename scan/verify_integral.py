"""Checks on the field integral: convergence in the integration range and
in the z step, for several plate heights. Run from repo root:
    .venv\\Scripts\\python.exe scripts\\verify_integral.py   that's for windows
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from field import BiotSavartField

Z_START, Z_LENGTH, X_HALF_GAP = 400.0, 300.0, 50.0
PITCH = 5.0


def integral(f, x, z_lo, z_hi, dz):
    z = np.arange(z_lo, z_hi + dz / 2, dz)
    return np.trapezoid(f.By(x, z), z)


print("--- 1. convergence in the integration range (y_height = 400, dz = 0.5) ---")
f = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, 400.0, 81)
print(f"{'half-range [mm]':>16} {'I(0)/B0L':>10} {'I(25)/B0L':>10}")
for half in (300.0, 600.0, 1200.0, 2400.0, 4800.0):
    lo, hi = Z_START - half, Z_START + Z_LENGTH + half
    i0 = integral(f, 0.0, lo, hi, 0.5) / Z_LENGTH
    i25 = integral(f, 25.0, lo, hi, 0.5) / Z_LENGTH
    print(f"{half:16.0f} {i0:10.4f} {i25:10.4f}")

print("\n--- 2. convergence in dz (y_height = 400, range +/-2400 mm) ---")
print(f"{'dz [mm]':>8} {'I(0)/B0L':>10}")
for dz in (2.0, 1.0, 0.5, 0.25):
    print(f"{dz:8.2f} {integral(f, 0.0, Z_START - 2400, Z_START + Z_LENGTH + 2400, dz) / Z_LENGTH:10.4f}")

print("\n--- 3. dependence on y_height (range +/-2400 mm, dz = 0.5) ---")
print(f"{'y_height':>9} {'n_fil':>6} {'I(0)/B0L':>10} {'I(25)/B0L':>10} {'I25/I0':>8}")
for y_height in (100.0, 400.0, 800.0):
    n = int(round(y_height / PITCH)) + 1
    g = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, y_height, n)
    lo, hi = Z_START - 2400, Z_START + Z_LENGTH + 2400
    i0 = integral(g, 0.0, lo, hi, 0.5) / Z_LENGTH
    i25 = integral(g, 25.0, lo, hi, 0.5) / Z_LENGTH
    print(f"{y_height:9.0f} {n:6d} {i0:10.4f} {i25:10.4f} {i25 / i0:8.4f}")
"""Scan of Biot-Savart uniformity and field integral vs plate height y_height.

The filament pitch in y is kept at about 5 mm, so n_filaments grows with y_height.
Run from repo root:
    .venv\\Scripts\\python.exe scripts\\scan_biot_savart_height.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from field import BiotSavartField

Z_START, Z_LENGTH, X_HALF_GAP = 400.0, 300.0, 50.0
Z_CENTRE = Z_START + Z_LENGTH / 2.0
FILAMENT_PITCH = 5.0                                   # mm, spacing between filaments in y

z_int = np.linspace(-600.0, 1600.0, 2201)              # wide range: the tails decay slowly
z_win = np.linspace(Z_CENTRE - 100, Z_CENTRE + 100, 201)
x_win = np.linspace(-25.0, 25.0, 101)

print(f"{'y_height':>9} {'n_fil':>6} {'dev_z':>8} {'dev_x':>8} "
      f"{'I(0)/B0L':>9} {'I(25)/B0L':>10} {'I25/I0':>8}")
for y_height in (100.0, 200.0, 400.0, 800.0):
    n = int(round(y_height / FILAMENT_PITCH)) + 1
    f = BiotSavartField(B0=1.0, z_start=Z_START, z_length=Z_LENGTH,
                        x_half_gap=X_HALF_GAP, y_height=y_height, n_filaments=n)
    dev_z = np.max(np.abs(f.By(0.0, z_win) - 1.0))
    dev_x = np.max(np.abs(f.By(x_win, Z_CENTRE) - 1.0))
    i0 = np.trapezoid(f.By(0.0, z_int), z_int) / Z_LENGTH
    i25 = np.trapezoid(f.By(25.0, z_int), z_int) / Z_LENGTH
    print(f"{y_height:9.0f} {n:6d} {dev_z:8.4f} {dev_x:8.4f} "
          f"{i0:9.4f} {i25:10.4f} {i25 / i0:8.4f}")
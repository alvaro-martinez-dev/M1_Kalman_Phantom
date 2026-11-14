"""Scan of n_filaments: uniformity, interpolation error and field integral.

Run from repo root:
    .venv\\Scripts\\python.exe scripts\\scan_filaments.py   that's for windows
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from field import BiotSavartField, InterpolatedField

Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT = 400.0, 300.0, 50.0, 100.0
Z_END = Z_START + Z_LENGTH
Z_CENTRE = Z_START + Z_LENGTH / 2.0
DX, DZ = 1.0, 0.1                      # interpolation grid, as in config

z_win = np.linspace(Z_CENTRE - 100, Z_CENTRE + 100, 201)
z_int = np.linspace(0.0, 1000.0, 20001)
z_edge = np.linspace(Z_START - 10, Z_START + 10, 801)
x_win = np.linspace(-25.0, 25.0, 101)

print(f"{'n_fil':>5} {'pitch[mm]':>9} {'dev_z':>8} {'dev_x':>8} "
      f"{'err_edge[T]':>12} {'I(x=0)/B0L':>11} {'I(x=25)/B0L':>12} {'time[s]':>8}")

for n in (20, 40, 80, 160):
    src = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT, n)
    pitch = Y_HEIGHT / (n - 1)

    dev_z = np.max(np.abs(src.By(0.0, z_win) - 1.0))
    dev_x = np.max(np.abs(src.By(x_win, Z_CENTRE) - 1.0))
    i0 = np.trapezoid(src.By(0.0, z_int), z_int) / Z_LENGTH
    i25 = np.trapezoid(src.By(25.0, z_int), z_int) / Z_LENGTH

    # time to build the interpolated map (what the propagator would use)
    import time
    t0 = time.perf_counter()
    f = InterpolatedField.from_field(src, x_max=X_HALF_GAP + 2 * DX,
                                     z_min=0.0, z_max=1000.0, dx=DX, dz=DZ)
    dt = time.perf_counter() - t0
    err_edge = np.max(np.abs(f.By(0.0, z_edge) - src.By(0.0, z_edge)))

    print(f"{n:5d} {pitch:9.2f} {dev_z:8.4f} {dev_x:8.4f} "
          f"{err_edge:12.3e} {i0:11.4f} {i25:12.4f} {dt:8.2f}")
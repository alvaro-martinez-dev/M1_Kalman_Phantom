"""Truncation of the interpolated Biot-Savart map, in units of B0 (B0 set to 1 T).

1. z: field at the ends of the grid and bending integral lost outside it.
2. x: field just outside the plates, which the map (and the propagator) sets to zero.
3. interpolation error inside the grid, away from the steep edges.
4. bending integral at x = 0, 10, 25 mm:
     full source (normalisation range)  vs  source on the grid  vs  map.

    .venv\Scripts\python.exe scans\scan_map_truncation.py
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.utils.config_loader import load_config, make_geometry
from src.field import BiotSavartField, InterpolatedField, build_field

cfg = copy.deepcopy(load_config("config"))
m = cfg["geometry"]["magnet"]
m["B0"] = 1.0                                   # results in units of B0
geometry = make_geometry(cfg["geometry"])

if m["model"] != "biot_savart" or not m["interpolation"]["enabled"]:
    raise SystemExit("config: model must be biot_savart with interpolation enabled")

Z_START, Z_LENGTH = m["z_start"], m["z_length"]
X_HALF_GAP, Y_HEIGHT = m["x_half_gap"], m["y_height"]
N_FIL = m["biot_savart"]["n_filaments"]
DX, DZ = m["interpolation"]["dx"], m["interpolation"]["dz"]
Z_END = Z_START + Z_LENGTH
Z_CENTRE = Z_START + Z_LENGTH / 2.0
Z_GRID_MIN, Z_GRID_MAX = 0.0, float(geometry.z_end_system)
X_GRID = X_HALF_GAP + 2 * DX
NORM_HALF_RANGE = 2400.0        # default z_int_halfrange of BiotSavartField

src = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT, N_FIL)
fmap = build_field(cfg, geometry)
assert isinstance(fmap, InterpolatedField)

print(f"grid: z in [{Z_GRID_MIN:.0f}, {Z_GRID_MAX:.0f}] mm (dz={DZ}), |x| <= {X_GRID:.1f} mm (dx={DX})")
print(f"magnet: z in [{Z_START:.0f}, {Z_END:.0f}] mm, x_half_gap={X_HALF_GAP}, "
      f"y_height={Y_HEIGHT}, n_fil={N_FIL}")

# 1. z truncation
print("\n--- 1. z: field at the grid ends (the map jumps to 0 beyond them) ---")
print(f"By(0, z={Z_GRID_MIN:.0f})/B0 = {src.By(0.0, Z_GRID_MIN):.3e}   "
      f"By(0, z={Z_GRID_MAX:.0f})/B0 = {src.By(0.0, Z_GRID_MAX):.3e}")

# 2. x truncation
print("\n--- 2. x: field just outside the plates, set to zero by the map ---")
print(f"By(0, z_centre)/B0 = {src.By(0.0, Z_CENTRE):.4f}   (reference, inside the gap)")
print(f"{'x - x_half_gap [mm]':>20} {'By/B0':>12}")
for delta in (1.0, 2 * DX, 10.0, 25.0, 50.0):
    print(f"{delta:20.1f} {src.By(X_HALF_GAP + delta, Z_CENTRE):12.4e}")

# 3. interpolation error inside the grid
print("\n--- 3. map - source inside the grid (random points) ---")
rng = np.random.default_rng(0)
xs = rng.uniform(-25.0, 25.0, 4000)
zs = rng.uniform(Z_GRID_MIN, Z_GRID_MAX, 4000)
err = np.abs(fmap.By(xs, zs) - src.By(xs, zs))
far = (np.abs(zs - Z_START) > 10.0) & (np.abs(zs - Z_END) > 10.0)
print(f"max |error|, |x| <= 25 mm, 10 mm away from the magnet edges: {err[far].max():.3e}")
print(f"max |error|, |x| <= 25 mm, anywhere in z:                    {err.max():.3e}")

# 4. bending integrals, relative to B0 * z_length
print("\n--- 4. bending integral / (B0 z_length) ---")
z_full = np.arange(Z_START - NORM_HALF_RANGE, Z_END + NORM_HALF_RANGE + DZ / 2, DZ)
z_grid = np.arange(Z_GRID_MIN, Z_GRID_MAX + DZ / 2, DZ)
print(f"{'x [mm]':>7} {'source, full':>13} {'source, grid':>13} {'map':>13} "
      f"{'full-grid':>11} {'map-grid':>11}")
for x in (0.0, 10.0, 25.0):
    i_full = np.trapezoid(src.By(x, z_full), z_full) / Z_LENGTH
    i_grid = np.trapezoid(src.By(x, z_grid), z_grid) / Z_LENGTH
    i_map = np.trapezoid(fmap.By(x, z_grid), z_grid) / Z_LENGTH
    print(f"{x:7.1f} {i_full:13.6f} {i_grid:13.6f} {i_map:13.6f} "
          f"{i_full - i_grid:11.2e} {i_map - i_grid:11.2e}")
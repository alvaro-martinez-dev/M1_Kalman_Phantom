"""Diagnostic of the large map-source error found by scan_map_truncation.py (exploration).

Runs the same random-point comparison for the n_filaments of the config and for n_filaments - 1
(parity flips: with an odd number a filament lies exactly in the y = 0 plane, where the map is
tabulated). It also integrates the map off the grid nodes, which block 4 of the previous scan
did not (there the map equals the source by construction, since it interpolates at the nodes).

    .venv\Scripts\python.exe scans\diagnose_map_error.py
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
Z_END = Z_START + Z_LENGTH
X_HALF_GAP, Y_HEIGHT = m["x_half_gap"], m["y_height"]
N_CONFIG = m["biot_savart"]["n_filaments"]
DX, DZ = m["interpolation"]["dx"], m["interpolation"]["dz"]
Z_MAP_MIN, Z_MAP_MAX = 300.0, 800.0              # local map, enough for the magnet and its edges

rng = np.random.default_rng(0)
xs = rng.uniform(-25.0, 25.0, 4000)
zs = rng.uniform(Z_MAP_MIN, Z_MAP_MAX, 4000)
dist_edge = np.minimum(np.abs(zs - Z_START), np.abs(zs - Z_END))
z_off = np.arange(Z_MAP_MIN, Z_MAP_MAX - DZ, DZ) + 0.37 * DZ   # not on the grid nodes

for n in (N_CONFIG, N_CONFIG - 1):
    ys_fil = np.linspace(-Y_HEIGHT / 2.0, Y_HEIGHT / 2.0, n)
    src = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, Y_HEIGHT, n)
    fmap = InterpolatedField.from_field(src, x_max=X_HALF_GAP + 2.0 * DX,
                                        z_min=Z_MAP_MIN, z_max=Z_MAP_MAX, dx=DX, dz=DZ)
    s, f = src.By(xs, zs), fmap.By(xs, zs)
    err = np.abs(f - s)
    i = int(np.argmax(err))
    print(f"\n=== n_filaments = {n}, filament on y = 0: {bool(np.any(np.isclose(ys_fil, 0.0)))} ===")
    print(f"max error {err[i]:.3e} at z = {zs[i]:.3f}, x = {xs[i]:.3f} "
          f"({dist_edge[i]:.3f} mm from the nearest magnet edge); "
          f"source = {s[i]:.3e}, map = {f[i]:.3e}")
    for band in (0.5, 1.0, 5.0, 10.0):
        print(f"max error more than {band:4.1f} mm from the edges: {err[dist_edge > band].max():.3e}")
    for x in (0.37, 10.37, 25.37):
        d = (np.trapezoid(fmap.By(x, z_off), z_off)
             - np.trapezoid(src.By(x, z_off), z_off)) / Z_LENGTH
        print(f"off-node integral, map - source, x = {x:5.2f} mm: {d:.2e}  (units of B0 z_length)")
"""Tail of By(0, z)/B0 outside the magnet, read from configs/config.yaml (exploration).

Run from repo root:
    .venv\Scripts\python.exe  scan\tail_outside_magnet.py   that's for windows
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config, make_geometry
from src.field import BiotSavartField

cfg = load_config("config")
m = cfg["geometry"]["magnet"]
geometry = make_geometry(cfg["geometry"])

# Raw Biot-Savart source (not the interpolated map), so the tail is not truncated
src = BiotSavartField(B0=1.0, z_start=m["z_start"], z_length=m["z_length"],
                      x_half_gap=m["x_half_gap"], y_height=m["y_height"],
                      n_filaments=m["biot_savart"]["n_filaments"])

z_start, z_end = m["z_start"], m["z_start"] + m["z_length"]
print(f"model={m['model']}  y_height={m['y_height']}  n_fil={m['biot_savart']['n_filaments']}")
print(f"{'dist. from edge [mm]':>21} {'z_before':>9} {'By/B0 before':>13} "
      f"{'z_after':>9} {'By/B0 after':>12}")
for d in (0.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0):
    zb, za = z_start - d, z_end + d
    print(f"{d:21.0f} {zb:9.0f} {src.By(0.0, zb):13.5f} "
          f"{za:9.0f} {src.By(0.0, za):12.5f}")
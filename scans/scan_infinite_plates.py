"""Infinite-plate limit: ratio By_between_plates / (mu0 K) for growing plate size.

In the raw units (mu0 = 1, unit current per filament) the surface current is
K = 1 / pitch, so ratio = By_raw * pitch, and it must tend to 1.

    .venv\Scripts\python.exe scans\scan_infinite_plates.py  just for windows
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.field import BiotSavartField

_m = load_config("config")["geometry"]["magnet"]
X_HALF_GAP = _m["x_half_gap"]        # plate position, from the config
PITCH = _m["y_height"] / (_m["biot_savart"]["n_filaments"] - 1)   # pitch of the real magnet
SCAN_PITCH = 5.0


def ratio(z_start, z_length, x_half_gap, y_height, n):
    """By(0, 0, z_centre) / (mu0 K), raw units. z_int_* only affect the normalisation,
    which is irrelevant here, so a cheap one is used."""
    f = BiotSavartField(1.0, z_start, z_length, x_half_gap, y_height, n,
                        z_int_halfrange=10.0, z_int_step=5.0)
    pitch = y_height / (n - 1)
    return f._raw_By(0.0, z_start + z_length / 2.0) * pitch


print(f"{'size [mm]':>10} {'n_fil':>6} {'By/(mu0 K)':>11} {'deficit':>9}")
for size in (500.0, 1000.0, 2000.0, 4000.0, 8000.0):
    n = int(round(size / SCAN_PITCH)) + 1
    r = ratio(-size / 2.0, size, X_HALF_GAP, size, n)
    print(f"{size:10.0f} {n:6d} {abs(r):11.4f} {1.0 - abs(r):9.4f}")

m = load_config("config")["geometry"]["magnet"]
n = m["biot_savart"]["n_filaments"]
r = ratio(m["z_start"], m["z_length"], m["x_half_gap"], m["y_height"], n)
print(f"\nconfig geometry (z_length={m['z_length']}, x_half_gap={m['x_half_gap']}, "
      f"y_height={m['y_height']}, n_fil={n}): By(centre)/(mu0 K) = {r:.4f}")
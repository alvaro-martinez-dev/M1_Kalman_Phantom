"""Visual check of the Biot-Savart field (run from repo root).

Produces data/field_plots/biot_savart_profile.png with:
    - By(z) at x = 0 and x = 40 mm, from z = 200 to 800 mm
    - By(x) at z = 550 mm (magnet centre), from x = -50 to 50 mm
and data/field_plots/biot_savart_profile.json with the metadata of the run
(config, parameters, numerical results, date, git commit).

.venv\Scripts\python.exe  scan\check_biot_savart.py  just for windows
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from src.utils.config_loader import load_config
from src.field import BiotSavartField

# Output
OUT_DIR = ROOT / "data" / "field_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEM = "biot_savart_profile_" + datetime.now().strftime("%Y%m%d_%H%M%S")

# Geometry, read from configs/config.yaml
_m = load_config("config")["geometry"]["magnet"]
Z_START = _m["z_start"]                          # entrance plane of the magnet [mm, z]
Z_LENGTH = _m["z_length"]                        # magnet length along z        [mm, z]
X_HALF_GAP = _m["x_half_gap"]                    # plates at x = +/- X_HALF_GAP [mm, x]
Y_HEIGHT = _m["y_height"]                        # extent of the plates in y    [mm, y]
N_FILAMENTS = _m["biot_savart"]["n_filaments"]   # wires per plate

Z_END = Z_START + Z_LENGTH           # exit plane   [mm, z]
Z_CENTRE = Z_START + Z_LENGTH / 2.0  # magnet centre [mm, z]

B0_USED = 1.0   # unit mean field: the plot shows the shape of the field, in units of B0
field = BiotSavartField(
    B0=B0_USED, z_start=Z_START, z_length=Z_LENGTH,
    x_half_gap=X_HALF_GAP, y_height=Y_HEIGHT, n_filaments=N_FILAMENTS,
)

# Profile along z
z = np.linspace(200.0, 800.0, 600)
by_z_x0 = field.By(0.0, z)
by_z_x40 = field.By(40.0, z)

# Profile along x at the magnet centre
x = np.linspace(-50.0, 50.0, 201)
by_x = field.By(x, Z_CENTRE)

# Uniformity metric: max |By - By_centre| for |z - Z_CENTRE| < 100 mm, at x = 0
by_centre = field.By(0.0, Z_CENTRE)
mask = np.abs(z - Z_CENTRE) < 100.0
max_dev = np.max(np.abs(by_z_x0[mask] - by_centre))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(z, by_z_x0, label="x = 0")
ax1.plot(z, by_z_x40, label="x = 40 mm")
ax1.axvline(Z_START, ls="--", c="gray", label="magnet edges")
ax1.axvline(Z_END, ls="--", c="gray")
ax1.axvspan(Z_CENTRE - 100, Z_CENTRE + 100, color="green", alpha=0.08,
            label="uniformity window")
ax1.set_xlabel("z [mm]")
ax1.set_ylabel("By [T]")
ax1.set_title("By along z")
ax1.legend()

ax2.plot(x, by_x, c="C2")
ax2.axvspan(-25, 25, color="orange", alpha=0.1, label="proton region |x| < 25 mm")
ax2.axhline(by_centre, ls=":", c="gray")
ax2.set_xlabel("x [mm]")
ax2.set_ylabel("By [T]")
ax2.set_title(f"By along x at z = {Z_CENTRE:.0f} mm")
ax2.legend()

fig.tight_layout()
png_path = OUT_DIR / f"{STEM}.png"
fig.savefig(png_path, dpi=120)
plt.close(fig)

# Metadata
try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
        stderr=subprocess.DEVNULL, text=True).strip()
except Exception:
    commit = None

metadata = {
    "script": Path(__file__).name,
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_commit": commit,
    "config_file": "configs/config.yaml",
    "magnet_config": _m,                       # full magnet block, as read from the yaml
    "field": {
        "model": "BiotSavartField (raw source, no interpolation)",
        "B0_used_T": B0_USED,
        "z_start_mm": Z_START,
        "z_length_mm": Z_LENGTH,
        "x_half_gap_mm": X_HALF_GAP,
        "y_height_mm": Y_HEIGHT,
        "n_filaments": N_FILAMENTS,
    },
    "plot_ranges": {
        "z_mm": [float(z[0]), float(z[-1]), len(z)],
        "x_mm": [float(x[0]), float(x[-1]), len(x)],
        "uniformity_window_mm": 100.0,
    },
    "results": {
        "By_centre_T": float(by_centre),
        "max_abs_deviation_T": float(max_dev),
        "relative_deviation": float(max_dev / by_centre),
    },
    "note": "Deterministic script (no random numbers, no seed).",
}
json_path = OUT_DIR / f"{STEM}.json"
with open(json_path, "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print("By centre            =", by_centre)
print("max |By - centre|    =", max_dev, "(|z - centre| < 100 mm, x = 0)")
print("relative deviation   =", max_dev / by_centre)
print("saved:", png_path)
print("saved:", json_path)
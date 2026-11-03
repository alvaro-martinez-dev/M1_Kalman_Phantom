"""Visual check of the Biot-Savart field (run from repo root).

Produces biot_savart_check_v2.png with:
    - By(z) at x = 0 and x = 40 mm, from z = 200 to 800 mm
    - By(x) at z = 550 mm (magnet centre), from x = -50 to 50 mm
and prints the centre value and the max deviation from it in |z - 550| < 100 mm.
"""
import sys
from pathlib import Path

# Make src/ importable (field, constants, ...) when running as a script
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import matplotlib.pyplot as plt
from field import BiotSavartField

# Geometry (coordinate in the suffix)
Z_START = 400.0      # entrance plane of the magnet          [mm, z]
Z_LENGTH = 300.0     # magnet length along z                 [mm, z]
X_HALF_GAP = 50.0    # plates at x = +/- X_HALF_GAP          [mm, x]
Y_HEIGHT = 100.0     # extent of the plates in y             [mm, y]

Z_END = Z_START + Z_LENGTH           # exit plane            [mm, z]
Z_CENTRE = Z_START + Z_LENGTH / 2.0  # magnet centre         [mm, z]

field = BiotSavartField(
    B0_nat=1.0, z_start=Z_START, z_length=Z_LENGTH,
    x_half_gap=X_HALF_GAP, y_height=Y_HEIGHT, n_filaments=20,
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
ax1.set_ylabel("By [MeV^2]")
ax1.set_title("By lungo z")
ax1.legend()

ax2.plot(x, by_x, c="C2")
ax2.axvspan(-25, 25, color="orange", alpha=0.1, label="proton region |x| < 25 mm")
ax2.axhline(by_centre, ls=":", c="gray")
ax2.set_xlabel("x [mm]")
ax2.set_ylabel("By [MeV^2]")
ax2.set_title("By lungo x a z = 550 mm")
ax2.legend()

fig.tight_layout()
fig.savefig("biot_savart_check_v2.png", dpi=120)

print("By(0, 550)           =", by_centre)
print("max |By - centre|    =", max_dev, "(|z - 550| < 100 mm, x = 0)")
print("relative deviation   =", max_dev / by_centre)
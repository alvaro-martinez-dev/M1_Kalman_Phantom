"""Visual check of the Biot-Savart field (run from repo root)."""
import sys
from pathlib import Path

# Make src/ importable (field, constants, ...) when running as a script
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import matplotlib.pyplot as plt
from field import BiotSavartField

field = BiotSavartField(
    B0_nat=1.0, z_start=400.0, length=300.0,
    half_gap=50.0, height=100.0, n_filaments=20,
)

z = np.linspace(200, 600, 200)
plt.figure()
plt.plot(z, field.By(0.0, z), label="x=0")
plt.plot(z, field.By(40.0, z), label="x=40 mm")
plt.axvline(400, ls="--", c="gray")
plt.axvline(700, ls="--", c="gray")
plt.xlabel("z [mm]")
plt.ylabel("By [MeV^2]")
plt.legend()
plt.title("Biot-Savart: profilo lungo z")
plt.savefig("biot_savart_check.png", dpi=120)
print("By al centro:", field.By(0.0, 500.0))
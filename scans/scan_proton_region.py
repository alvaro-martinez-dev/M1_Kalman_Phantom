"""Where do the protons go? x of simulated tracks in the magnet and at the planes (exploration).

Tracks are sampled from a beam block of the config and propagated with the real Propagator
(scattering kicks included). Part 1: distribution of max |x| inside the magnet, x at the planes.
Part 2: variation of the bending integral I(x) = int By dz with x, for several plate heights,
at the x reached by the protons.

ASSUMPTIONS (not seen in the code): beams are in cfg["beams"][name] with the keys read by
make_beam; p0 and sigma_p0 are in MeV; theta0 is an angle (tx = tan theta); tracks start at z = 0.

    .venv\Scripts\python.exe scans\scan_proton_region.py --n 300
    .venv\Scripts\python.exe scans\scan_proton_region.py --n 5000 --beam <name>
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.field import BiotSavartField, build_field
from src.montecarlo.propagation import plane_list
from src.propagator.propagator import State, kinetic_energy
from src.utils.config_loader import (get_constant, load_config, make_detector_planes,
                                     make_geometry, make_propagator)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=300, help="number of tracks")
parser.add_argument("--beam", default=None, help="name of the beam block (default: the first)")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

Z0 = 0.0                       # initial z of the tracks (assumption)
N_MAGNET_POINTS = 31           # states requested inside the magnet
SCAN_PITCH = 5.0               # filament pitch of the height scan [mm]
HEIGHTS = (200.0, 400.0, 800.0)

cfg = load_config("config")
m = cfg["geometry"]["magnet"]
Z_START, Z_LENGTH, X_HALF_GAP = m["z_start"], m["z_length"], m["x_half_gap"]
Z_END = Z_START + Z_LENGTH

geometry = make_geometry(cfg["geometry"])
detectors = make_detector_planes(geometry, cfg["geometry"]["detector_planes"])
planes = plane_list(detectors) if detectors is not None else []
z_end_system = float(geometry.z_end_system)
field = build_field(cfg, geometry)

beams = cfg["beams"]
name = args.beam or next(iter(beams))
b = beams[name]
print(f"beams in the config: {list(beams)}; using '{name}': {b}")
particle = get_constant(b["particle"])
propagator = make_propagator(cfg, geometry, field, particle)

# ---- sample the beam --------------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
n = args.n
x0 = rng.normal(b["x0"], b["sigma_x0"], n)
tx0 = np.tan(rng.normal(b["theta0"], b["sigma_theta0"], n))
p0 = rng.normal(b["p0"], b["sigma_p0"], n)
q_p0 = particle.charge / p0
print(f"sampled: <x0> = {x0.mean():.3f} mm, <tx0> = {tx0.mean():.2e}, <p0> = {p0.mean():.1f} MeV "
      f"-> T = {kinetic_energy(q_p0.mean(), particle):.1f} MeV (check the units of p0)")

# ---- propagate --------------------------------------------------------------------------------
z_magnet = np.linspace(Z_START, Z_END, N_MAGNET_POINTS)
z_out = np.unique(np.concatenate(([z for _, z in planes], z_magnet, [z_end_system])))
x_out = np.full((n, len(z_out)), np.nan)
kick_rng = np.random.default_rng(args.seed + 1) if propagator.max_slice is not None else None
n_stopped = 0
for i in range(n):
    res = propagator.propagate(State(Z0, x0[i], tx0[i], q_p0[i]), z_end_system,
                               z_out=list(z_out), rng=kick_rng)
    if res.stopped:
        n_stopped += 1
        continue
    for j, s in enumerate(res.at_z_out):
        if s is not None:
            x_out[i, j] = s.x
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{n} tracks", flush=True)
print(f"stopped in the phantom: {n_stopped}/{n}")

# ---- part 1: x of the protons -----------------------------------------------------------------
in_magnet = (z_out >= Z_START - 1e-6) & (z_out <= Z_END + 1e-6)
xm = x_out[:, in_magnet]
ok = ~np.isnan(xm).any(axis=1)
max_abs = np.abs(xm[ok]).max(axis=1)
print(f"\n=== max |x| inside the magnet (z in [{Z_START:.0f}, {Z_END:.0f}] mm), "
      f"{ok.sum()} tracks ===")
for q in (50, 90, 99, 99.9, 100):
    print(f"  percentile {q:>5}: {np.percentile(max_abs, q):8.2f} mm")
print(f"  fraction with max|x| > 25 mm:  {np.mean(max_abs > 25.0):.4f}")
print(f"  fraction with max|x| > x_half_gap = {X_HALF_GAP}: {np.mean(max_abs > X_HALF_GAP):.4f}"
      "   (there the propagator sets the field to zero)")

j_exit = int(np.argmin(np.abs(z_out - Z_END)))
x_exit = x_out[ok, j_exit]
print(f"  x at the magnet exit: mean = {x_exit.mean():+.2f} mm, std = {x_exit.std():.2f} mm "
      "(sign: compare with the PDF, +7 cm at the last plane)")

after = (z_out > Z_END + 1e-6) & ~np.isnan(x_out).all(axis=0)
if after.any():
    far = np.abs(x_out[:, after]).max(axis=1)
    print(f"  after the magnet: max |x| = {np.nanmax(far):.1f} mm, "
          f"fraction beyond x_half_gap = {np.nanmean(far > X_HALF_GAP):.4f}")

if planes:
    print(f"\n{'plane':>10} {'z [mm]':>8} {'<x> [mm]':>9} {'std x':>8} {'max|x|':>8}")
    for pname, z in planes:
        col = x_out[:, int(np.argmin(np.abs(z_out - z)))]
        col = col[~np.isnan(col)]
        if len(col):
            print(f"{pname:>10} {z:8.1f} {col.mean():9.2f} {col.std():8.2f} {np.abs(col).max():8.2f}")

# ---- part 2: bending integral vs x, for several plate heights ---------------------------------
x_eval = sorted({10.0, 25.0, round(float(np.percentile(max_abs, 99)), 1),
                 round(float(np.percentile(max_abs, 99.9)), 1)})
z_int = np.arange(Z_START - 2400.0, Z_END + 2400.0 + 0.5, 1.0)
configs = [("config", m["y_height"], m["biot_savart"]["n_filaments"])]
configs += [(f"y_height = {y:g}", y, 2 * int(round(y / (2 * SCAN_PITCH)))) for y in HEIGHTS]

print("\n=== I(x)/I(0) - 1  [%], I = int By dz at constant x (an estimate: the real path bends) ===")
print(f"{'geometry':>18} {'n_fil':>6} " + " ".join(f"{'x=' + format(x, 'g'):>9}" for x in x_eval))
for label, y_height, n_fil in configs:
    f = BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, y_height, n_fil)
    i0 = np.trapezoid(f.By(0.0, z_int), z_int)
    row = [100.0 * (np.trapezoid(f.By(x, z_int), z_int) / i0 - 1.0) for x in x_eval]
    print(f"{label:>18} {n_fil:6d} " + " ".join(f"{v:9.3f}" for v in row))
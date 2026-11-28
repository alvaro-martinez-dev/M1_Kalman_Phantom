"""Proton region for every beam of the config, and the bending integral along the real paths.

For each beam: tracks sampled as Initial_Beam.generate does (same order of the draws, same seed),
propagated with the real Propagator through the magnet, then
  1. distribution of max |x| in the magnet (the 99th percentile with a bootstrap interval)
  2. per track, the bending integral along the path in the magnet, I_path = int By(x(z), z) dz,
     relative to the axis I_axis = int By(0, z) dz, for several plate heights. The field is
     zero for |x| > x_half_gap, as in the propagator. Both integrals use the same z nodes.

    .venv\Scripts\python.exe scans\scan_proton_region.py --n 300
    .venv\Scripts\python.exe scans\scan_proton_region.py --n 2000 --beams proton proton_low
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.field import BiotSavartField, build_field
from src.propagator.propagator import State, kinetic_energy
from src.utils.config_loader import get_constant, load_config, make_geometry, make_propagator, make_beam

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=300, help="tracks per beam")
parser.add_argument("--beams", nargs="*", default=None, help="beam names (default: all of the config)")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

N_MAGNET_POINTS = 31
SCAN_PITCH = 5.0                       # filament pitch of the height scan [mm]
HEIGHTS = (200.0, 400.0, 800.0)
N_BOOT = 200

cfg = load_config("config")
m = cfg["geometry"]["magnet"]
Z_START, Z_LENGTH, X_HALF_GAP = m["z_start"], m["z_length"], m["x_half_gap"]
Z_END = Z_START + Z_LENGTH
geometry = make_geometry(cfg["geometry"])
field = build_field(cfg, geometry)
z_magnet = np.linspace(Z_START, Z_END, N_MAGNET_POINTS)

geoms = [("config", m["y_height"], m["biot_savart"]["n_filaments"])]
geoms += [(f"y_height = {y:g}", y, 2 * int(round(y / (2 * SCAN_PITCH)))) for y in HEIGHTS]
fields = {label: BiotSavartField(1.0, Z_START, Z_LENGTH, X_HALF_GAP, y, n) for label, y, n in geoms}
i_axis = {label: np.trapezoid(f.By(0.0, z_magnet), z_magnet) for label, f in fields.items()}

print(f"magnet z in [{Z_START:.0f}, {Z_END:.0f}] mm, x_half_gap = {X_HALF_GAP} mm, "
      f"{args.n} tracks per beam, seed = {args.seed}")

for name in (args.beams or list(cfg["beams"])):
    b = cfg["beams"][name]
    particle = get_constant(b["particle"])
    propagator = make_propagator(cfg, geometry, field, particle)

    df_beam = make_beam(b).generate(args.n, seed=args.seed)
    x0 = df_beam["x0"].to_numpy()
    tx0 = (df_beam["px0"] / df_beam["pz0"]).to_numpy()
    q_p0 = particle.charge / np.hypot(df_beam["px0"], df_beam["pz0"]).to_numpy()
    kick_rng = np.random.default_rng(args.seed + 1) if propagator.max_slice is not None else None

    x_path, n_stopped = [], 0
    for i in range(args.n):
        res = propagator.propagate(State(0.0, x0[i], tx0[i], q_p0[i]), Z_END,
                                   z_out=list(z_magnet), rng=kick_rng)
        row = [s.x if s is not None else np.nan for s in res.at_z_out]
        if res.stopped or np.isnan(row).any():
            n_stopped += 1
            continue
        x_path.append(row)
    x_path = np.array(x_path)

    T = kinetic_energy(particle.charge / b["p0"], particle)
    print(f"\n=========== beam '{name}': {particle.name}, p0 = {b['p0']} MeV (T = {T:.1f} MeV), "
          f"stopped before the magnet end: {n_stopped}/{args.n} ===========")
    if len(x_path) == 0:
        continue

    max_abs = np.abs(x_path).max(axis=1)
    p99 = np.percentile(max_abs, 99)
    boot = [np.percentile(np.random.default_rng(k).choice(max_abs, len(max_abs)), 99)
            for k in range(N_BOOT)]
    print(f"max|x| in the magnet [mm]: median {np.median(max_abs):.1f}, p90 {np.percentile(max_abs, 90):.1f}, "
          f"p99 {p99:.1f} (90% interval {np.percentile(boot, 5):.1f}-{np.percentile(boot, 95):.1f}), "
          f"max {max_abs.max():.1f}")
    print(f"fraction > 25 mm: {np.mean(max_abs > 25.0):.3f}   "
          f"fraction > x_half_gap: {np.mean(max_abs > X_HALF_GAP):.4f}   "
          f"<x> at the exit: {x_path[:, -1].mean():+.1f} mm (std {x_path[:, -1].std():.1f})")

    print(f"{'bending integral along the path / axis - 1 [%]':>48}  mean     std   |99th pct|")
    for label, f in fields.items():
        by = np.where(np.abs(x_path) <= X_HALF_GAP, f.By(x_path, z_magnet[None, :]), 0.0)
        delta = 100.0 * (np.trapezoid(by, z_magnet, axis=1) / i_axis[label] - 1.0)
        print(f"{label:>48} {delta.mean():7.3f} {delta.std():7.3f} {np.percentile(np.abs(delta), 99):9.3f}")
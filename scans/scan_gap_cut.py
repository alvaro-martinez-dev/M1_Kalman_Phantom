"""Effect of the field cut |x| > x_half_gap on the propagation (exploration).

The same tracks are propagated twice, with the same scattering kicks (a random stream per track,
the same in the two runs):
  ref  : the production propagator (the field is zero for |x| > x_half_gap)
  free : no cut (x_half_gap = inf), with a field map that extends to |x| <= --x-wide
The difference at the detector planes is the effect of the cut. Tracks are split in two groups:
those that never leave the gap (there the two runs must agree to numerical precision: a built-in
check) and those that do.

    .venv\Scripts\python.exe scans\scan_gap_cut.py --n 300
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.field import BiotSavartField, InterpolatedField, build_field
from src.montecarlo.propagation import plane_list
from src.propagator.propagator import State
from src.utils.config_loader import (get_constant, load_config, make_beam, make_detector_planes,
                                     make_geometry, make_propagator)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=300, help="number of tracks")
parser.add_argument("--beam", default="proton", help="beam block of the config")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--x-wide", type=float, default=200.0, help="half width in x of the wide map [mm]")
args = parser.parse_args()

N_MAGNET_POINTS = 31

cfg = load_config("config")
m = cfg["geometry"]["magnet"]
if m["model"] != "biot_savart" or not m["interpolation"]["enabled"]:
    raise SystemExit("config: this scan needs model = biot_savart with interpolation enabled")
Z_START, Z_LENGTH, X_HALF_GAP = m["z_start"], m["z_length"], m["x_half_gap"]
Z_END = Z_START + Z_LENGTH
ip = m["interpolation"]

geometry = make_geometry(cfg["geometry"])
detectors = make_detector_planes(geometry, cfg["geometry"]["detector_planes"])
planes = plane_list(detectors)
z_end_system = float(geometry.z_end_system)
particle = get_constant(cfg["beams"][args.beam]["particle"])

# ---- the two propagators ----------------------------------------------------------------------
field_ref = build_field(cfg, geometry)
source = BiotSavartField(m["B0"], Z_START, Z_LENGTH, X_HALF_GAP, m["y_height"],
                         m["biot_savart"]["n_filaments"])
print(f"building the wide map, |x| <= {args.x_wide:g} mm (dx = {ip['dx']}, dz = {ip['dz']}) ...", flush=True)
field_wide = InterpolatedField.from_field(source, x_max=args.x_wide, z_min=0.0, z_max=z_end_system,
                                          dx=ip["dx"], dz=ip["dz"])

prop_ref = make_propagator(cfg, geometry, field_ref, particle)
prop_free = make_propagator(cfg, geometry, field_wide, particle)
prop_free.x_half_gap = np.inf                    # no cut

# ---- the beam: the project code ---------------------------------------------------------------
df_beam = make_beam(cfg["beams"][args.beam]).generate(args.n, seed=args.seed)
n = len(df_beam)
x0 = df_beam["x0"].to_numpy()
tx0 = (df_beam["px0"] / df_beam["pz0"]).to_numpy()
q_p0 = particle.charge / np.hypot(df_beam["px0"], df_beam["pz0"]).to_numpy()

z_magnet = np.linspace(Z_START, Z_END, N_MAGNET_POINTS)
z_out = np.unique(np.concatenate(([z for _, z in planes], z_magnet, [z_end_system])))


def run(propagator):
    # (n, n_z_out, 2) array with x and tx at each z_out; NaN if the track stopped
    out = np.full((n, len(z_out), 2), np.nan)
    for i in range(n):
        rng = np.random.default_rng([args.seed, i]) if propagator.max_slice is not None else None
        res = propagator.propagate(State(0.0, x0[i], tx0[i], q_p0[i]), z_end_system,
                                   z_out=list(z_out), rng=rng)
        if res.stopped:
            continue
        for j, s in enumerate(res.at_z_out):
            if s is not None:
                out[i, j] = (s.x, s.tx)
    return out


print("propagating with the cut (production) ...", flush=True)
ref = run(prop_ref)
print("propagating without the cut ...", flush=True)
free = run(prop_free)

# ---- compare ----------------------------------------------------------------------------------
valid = ~np.isnan(ref).any(axis=(1, 2)) & ~np.isnan(free).any(axis=(1, 2))
ref, free = ref[valid], free[valid]
after_start = z_out >= Z_START - 1e-6
leaves = (np.abs(ref[:, after_start, 0]) > X_HALF_GAP).any(axis=1)

print(f"\n{valid.sum()}/{n} tracks reach the end; x_half_gap = {X_HALF_GAP} mm; "
      f"{leaves.sum()} leave the gap at some z >= z_start ({100.0 * leaves.mean():.2f}%)")
if leaves.any():
    big = np.abs(ref[leaves][:, after_start, 0]).max(axis=1)
    print(f"max |x| of the tracks that leave: median {np.median(big):.1f} mm, max {big.max():.1f} mm "
          f"(the wide map ends at {args.x_wide:g} mm)")

cols = [(pname, int(np.argmin(np.abs(z_out - z))), z) for pname, z in planes if z >= Z_START - 1e-6]
for label, mask in (("never leave the gap (expected: no difference)", ~leaves),
                    ("leave the gap", leaves)):
    print(f"\n--- tracks that {label}: {mask.sum()} ---")
    if not mask.any():
        continue
    print(f"{'plane':>20} {'z [mm]':>8} {'mean dx [um]':>13} {'max|dx| [um]':>13} {'max|dtx| [urad]':>16}")
    for pname, j, z in cols:
        dx = 1e3 * (free[mask, j, 0] - ref[mask, j, 0])
        dtx = 1e6 * (free[mask, j, 1] - ref[mask, j, 1])
        print(f"{pname:>20} {z:8.1f} {dx.mean():13.3f} {np.abs(dx).max():13.3f} {np.abs(dtx).max():16.3f}")
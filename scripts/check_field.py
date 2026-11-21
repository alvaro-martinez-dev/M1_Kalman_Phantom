# scripts/check_field.py
#
# Visual check of the magnetic field built from a yaml config (src.field.build_field, i.e. exactly
# the field the propagator uses: model, B0 and interpolation of geometry.magnet). Saves in
# data/field/field_Magn<hash>_<model>[_interp]/:
#   field.pdf:    1. By(z) along the whole system, at x = 0 and x = x_offset, with the magnet edges
#                    and the uniformity window around the magnet centre
#                 2. By(x) across the gap at the magnet centre, with the plates at +-x_half_gap
#                 If the interpolation is enabled, the source field (no interpolation) is drawn
#                 dashed on top, to see the interpolation error by eye
#   metadata.yml: date, git commit, the magnet block used, the plot ranges and the results
#                 (By at the centre, uniformity, field integral)
# with hash = magnet_hash of src/utils/hashing.py: deterministic hash of what changes the field in
#             the magnet block (and the extent of the system, z_end_system, which sets the
#             interpolation grid, if interpolated): two configs with a different field never share
#             a folder. The propagations made with this field carry the same Magn<hash> in their name
#
# Usage (from the repo root):
#   python -m scripts.check_field                          # uses configs/config.yaml
#   python -m scripts.check_field --overwrite              # replace an existing field folder
#   python -m scripts.check_field --x-offset 30 --window 50
#   python -m scripts.check_field --config my_config --overwrite

import argparse
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from matplotlib.figure import Figure

from plotting.plot_layout import COLOR_MAGNET, COLOR_MUTED, COLOR_TEXT, PAGE_SIZE
from src.field import build_field
from src.utils.config_loader import CONFIG_DIR, get_key, is_enabled, load_config, make_geometry
from src.utils.dataset_io import DATA_DIR, REPO_ROOT, git_state
from src.utils.hashing import magnet_hash, magnet_parameters


REPORT_NAME = "field.pdf"
HEIGHT_RATIOS = [1, 2, 1]       # margin / plots / margin: the plots take half the page height
WIDTH_RATIOS = [1, 8, 8, 1]     # margin / By(z) / By(x) / margin

N_Z = 2000                # points of the By(z) profiles (whole system)
N_X = 401                 # points of the By(x) profile (across the gap)
X_OFFSET_FRACTION = 0.8   # default x of the second By(z) profile, in units of x_half_gap
WINDOW = 100.0            # [mm] default half width of the uniformity window around the centre

COLOR_X0 = "#2a78d6"
COLOR_XOFF = "#1baf7a"
COLOR_SOURCE = "#0b0b0b"
COLOR_WINDOW = "#1baf7a"
WINDOW_ALPHA = 0.08


# Draws the two profiles on an A4 landscape page and saves it as a pdf
# profiles: dict with the arrays of main(); source: same keys for the source field, or None
def plot_field(path, profiles, source, magnet, x_offset, window, title):

    z, x = profiles["z"], profiles["x"]
    z_start, z_end = magnet["z_start"], magnet["z_start"] + magnet["z_length"]
    z_centre = 0.5 * (z_start + z_end)
    x_half_gap = magnet["x_half_gap"]

    fig = Figure(figsize=PAGE_SIZE, layout="constrained")
    fig.suptitle(title, fontsize=11, color=COLOR_TEXT)
    # empty rows/columns around the two plots, so they keep a natural aspect and stay centred
    grid = fig.add_gridspec(3, 4, width_ratios=WIDTH_RATIOS, height_ratios=HEIGHT_RATIOS)
    ax_z, ax_x = fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])

    # 1. By along z
    ax_z.plot(z, profiles["By_z_x0"], c=COLOR_X0, label="x = 0")
    ax_z.plot(z, profiles["By_z_xoff"], c=COLOR_XOFF, label=f"x = {x_offset:g} mm")
    if source is not None:
        ax_z.plot(z, source["By_z_x0"], c=COLOR_SOURCE, ls="--", lw=0.8, label="source (no interp.)")
        ax_z.plot(z, source["By_z_xoff"], c=COLOR_SOURCE, ls="--", lw=0.8)
    ax_z.axvline(z_start, ls="--", c=COLOR_MAGNET, label="magnet edges")
    ax_z.axvline(z_end, ls="--", c=COLOR_MAGNET)
    ax_z.axvspan(z_centre - window, z_centre + window, color=COLOR_WINDOW, alpha=WINDOW_ALPHA,
                 label=f"uniformity window (±{window:g} mm)")
    ax_z.set_xlabel("z [mm]")
    ax_z.set_ylabel("By [T]")
    ax_z.set_title("By along z", color=COLOR_TEXT)
    ax_z.legend(fontsize=8)

    # 2. By along x at the magnet centre
    ax_x.plot(x, profiles["By_x"], c=COLOR_X0, label=f"z = {z_centre:g} mm")
    if source is not None:
        ax_x.plot(x, source["By_x"], c=COLOR_SOURCE, ls="--", lw=0.8, label="source (no interp.)")
    ax_x.axvline(-x_half_gap, ls="--", c=COLOR_MAGNET, label="plates")
    ax_x.axvline(x_half_gap, ls="--", c=COLOR_MAGNET)
    ax_x.axhline(profiles["By_centre"], ls=":", c=COLOR_MUTED, label="By at the centre")
    ax_x.set_xlabel("x [mm]")
    ax_x.set_ylabel("By [T]")
    ax_x.set_title("By along x at the magnet centre", color=COLOR_TEXT)
    ax_x.legend(fontsize=8)

    fig.savefig(path)
    return path


# field: MagneticField; z, x: sampling arrays
# Returns: dict with the three profiles and By at the centre
def sample_profiles(field, z, x, x_offset, z_centre):

    return {
        "z": z,
        "x": x,
        "By_z_x0": field.By(0.0, z),
        "By_z_xoff": field.By(x_offset, z),
        "By_x": field.By(x, z_centre),
        "By_centre": float(field.By(0.0, z_centre)),
    }


def main():

    parser = argparse.ArgumentParser(description="Plot the magnetic field built from a yaml config.")
    parser.add_argument("--config", default="config",
                        help="name of a yaml in configs/ (without extension) or path to a yaml")
    parser.add_argument("--x-offset", type=float, default=None,
                        help=f"x [mm] of the second By(z) profile (default: {X_OFFSET_FRACTION} x_half_gap)")
    parser.add_argument("--window", type=float, default=WINDOW,
                        help="half width [mm] of the uniformity window around the magnet centre "
                             f"(default: {WINDOW:g})")
    parser.add_argument("--name", default=None,
                        help="field folder name (default: field_Magn<hash>_<model>[_interp])")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the field folder if it already exists")
    args = parser.parse_args()

    # 1. Config
    config_path = Path(args.config)
    if config_path.suffix not in (".yaml", ".yml"):
        config_path = CONFIG_DIR / f"{args.config}.yaml"
    cfg = load_config(config_path)
    magnet = get_key(cfg["geometry"], "magnet", "geometry")
    if not is_enabled(magnet, "geometry.magnet"):
        raise ValueError("The magnet is disabled (geometry.magnet.enabled): there is no field to plot")

    # 2. Field (as the propagator sees it) and, if interpolated, its source
    geometry = make_geometry(cfg["geometry"])
    field = build_field(cfg, geometry)
    interpolated = get_key(magnet, "interpolation.enabled", "geometry.magnet")
    source = None
    if interpolated:
        cfg_source = copy.deepcopy(cfg)
        cfg_source["geometry"]["magnet"]["interpolation"]["enabled"] = False
        source = build_field(cfg_source, geometry)

    # 3. Profiles and results
    z_start, z_length = magnet["z_start"], magnet["z_length"]
    x_half_gap = magnet["x_half_gap"]
    z_centre = z_start + 0.5 * z_length
    x_offset = args.x_offset if args.x_offset is not None else X_OFFSET_FRACTION * x_half_gap
    if not 0.0 <= args.window <= 0.5 * z_length:
        raise ValueError(f"--window must be in [0, z_length / 2 = {0.5 * z_length:g}] mm, got {args.window:g}")

    z = np.linspace(0.0, geometry.z_end_system, N_Z)
    x = np.linspace(-x_half_gap, x_half_gap, N_X)
    profiles = sample_profiles(field, z, x, x_offset, z_centre)
    source_profiles = sample_profiles(source, z, x, x_offset, z_centre) if source else None

    by_centre = profiles["By_centre"]
    in_window = np.abs(z - z_centre) <= args.window
    max_dev = float(np.max(np.abs(profiles["By_z_x0"][in_window] - by_centre)))
    integral = float(np.trapezoid(profiles["By_z_x0"], z))   # [T mm] along the whole system, x = 0
    results = {
        "By_centre_T": by_centre,
        "max_abs_deviation_in_window_T": max_dev,
        "relative_deviation_in_window": max_dev / by_centre if by_centre else None,
        "By_integral_x0_T_mm": integral,
        "effective_length_mm": integral / by_centre if by_centre else None,
    }
    if source_profiles:
        results["max_abs_interpolation_error_z_T"] = float(np.max(np.abs(
            np.concatenate([profiles["By_z_x0"] - source_profiles["By_z_x0"],
                            profiles["By_z_xoff"] - source_profiles["By_z_xoff"]]))))
        results["max_abs_interpolation_error_x_T"] = float(np.max(np.abs(
            profiles["By_x"] - source_profiles["By_x"])))

    # 4. Output folder
    hash_ = magnet_hash(magnet, geometry.z_end_system)
    name = args.name or f"field_Magn{hash_}_{magnet['model']}{'_interp' if interpolated else ''}"
    out_dir = DATA_DIR / "field" / name
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists (use --overwrite to replace it)")
        # the folder itself is not removed (OneDrive may lock it), only its files
        for file in out_dir.iterdir():
            if file.is_file():
                file.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5. Metadata
    config_path = config_path.resolve()
    if config_path.is_relative_to(REPO_ROOT):
        config_path = config_path.relative_to(REPO_ROOT)
    commit, dirty = git_state()
    created = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "created": created,
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "config_source": str(config_path),
        "field_class": type(field).__name__,
        "magnet_hash": hash_,
        "magnet_parameters": magnet_parameters(magnet, geometry.z_end_system),   # what the hash is computed from
        "z_end_system": geometry.z_end_system,
        "plot_ranges": {
            "z_mm": [float(z[0]), float(z[-1]), N_Z],
            "x_mm": [float(x[0]), float(x[-1]), N_X],
            "x_offset_mm": x_offset,
            "window_half_width_mm": args.window,
        },
        "results": results,
        "magnet": magnet,
    }
    with open(out_dir / "metadata.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    # 6. Plot
    commit_text = str(commit)[:7] + (" + uncommitted changes" if dirty else "")
    title = (f"{type(field).__name__} ({magnet['model']}, B0 = {magnet['B0']:g} T)\n"
             f"created {created}  |  commit {commit_text}")
    plot_field(out_dir / REPORT_NAME, profiles, source_profiles, magnet, x_offset, args.window, title)

    print(f"Field: {type(field).__name__} ({magnet['model']}, B0 = {magnet['B0']:g} T)")
    print(f"  By at the centre (z = {z_centre:g} mm)       = {by_centre:.6g} T")
    print(f"  max |By - centre| (|z - centre| <= {args.window:g}) = {max_dev:.3g} T"
          + (f"  ({max_dev / by_centre:.3g} relative)" if by_centre else ""))
    print(f"  integral of By dz at x = 0         = {integral:.6g} T mm")
    if source_profiles:
        print(f"  max |interpolated - source|        = "
              f"{max(results['max_abs_interpolation_error_z_T'], results['max_abs_interpolation_error_x_T']):.3g} T")
    print(f"\nSaved field check to {out_dir}")


if __name__ == "__main__":
    main()

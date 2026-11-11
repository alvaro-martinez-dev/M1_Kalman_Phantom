# scripts/generate_layout.py
#
# Builds the geometry (and the detector planes) from a yaml config and saves the layout table
# in data/geometry/layout_<n>Det_<m>Lay<p>PhDet_<k>MagDet_<hash>/ to check the generated positions:
#   layout.csv:   one row per z (boundaries + detector planes, region/material on each side)
#   metadata.yml: date, git commit, the layout parameters with their hash and the geometry block used
#   report.pdf:   schematic of the system and the table, if run.plot is true (see plotting/plot_layout.py)
# with n = total number of distinct detector planes, m = number of phantom layers,
#      p = detector planes at the internal interfaces of the phantom (m - 1 if inside_phantom, else 0),
#      k = detector planes strictly inside the magnet (n_inside_magnet),
#      hash = deterministic hash of the layout parameters (see layout_hash): two configs that
#             differ in any position, thickness or number of planes never share a folder
#
# Usage (from the repo root):
#   python -m scripts.generate_layout                         # uses configs/config.yaml
#   python -m scripts.generate_layout --no-detectors          # only the physical boundaries (0Det),
#                                                             # whatever detector_planes.enabled says
#   python -m scripts.generate_layout --overwrite             # replace an existing layout folder
#   python -m scripts.generate_layout --no-plot               # skips the report of run.plot
#   python -m scripts.generate_layout --config my_config --overwrite

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml

from plotting.plot_layout import REPORT_NAME, load_layout, plot_layout
from src.geometry.layout import layout_table
from src.utils.config_loader import CONFIG_DIR, get_key, load_config, make_detector_planes, make_geometry
from src.utils.dataset_io import DATA_DIR, REPO_ROOT, git_state
from src.utils.validation import check_real


HASH_LENGTH = 6   # hex characters of the layout hash in the folder name


# Parameters that define the layout, already validated and normalised by Geometry and
# DetectorPlanes (eg 100 and 100.0 give the same value). Field parameters (B0, model...)
# and sigma_x are left out: they do not change the layout.
# A disabled phantom/magnet has its parameters at None.
# x_half_gap: transverse size of the magnet, drawn in the report (None without magnet)
# Returns: dict
def layout_parameters(geometry, detectors, x_half_gap):

    return {
        "z_start_phantom": geometry.z_start_phantom,
        "phantom_layers": geometry.phantom_layers,
        "z_start_magnet": geometry.z_start_magnet,
        "z_length_magnet": geometry.z_length_magnet,
        "x_half_gap": x_half_gap if geometry.has_magnet else None,
        "spacing_after_last": geometry.spacing_after_last,
        "placement": None if detectors is None else {
            "n_before_phantom": detectors.n_before_phantom,
            "n_after_phantom_before_magnet": detectors.n_after_phantom_before_magnet,
            "n_after_magnet": detectors.n_after_magnet,
            "inside_phantom": detectors.inside_phantom,
            "n_inside_magnet": detectors.n_inside_magnet,
        },
    }


# Deterministic hash of the layout parameters: same parameters -> same hash, any change -> another
# Returns: the first HASH_LENGTH hex characters of the sha256 of the parameters as sorted json
def layout_hash(parameters):

    text = json.dumps(parameters, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:HASH_LENGTH]


def main():

    parser = argparse.ArgumentParser(description="Save the z layout of the geometry and detector planes.")
    parser.add_argument("--config", default="config",
                        help="name of a yaml in configs/ (without extension) or path to a yaml")
    parser.add_argument("--detectors", action=argparse.BooleanOptionalAction, default=None,
                        help="include the detector planes (default: detector_planes.enabled of the yaml)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the layout folder if it already exists")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None,
                        help=f"write {REPORT_NAME} in the layout folder (default: run.plot of the yaml)")
    args = parser.parse_args()

    # 1. Config
    config_path = Path(args.config)
    if config_path.suffix not in (".yaml", ".yml"):
        config_path = CONFIG_DIR / f"{args.config}.yaml"
    cfg = load_config(config_path)
    geometry_cfg = cfg["geometry"]

    # 2. Geometry, detectors and table
    geometry = make_geometry(geometry_cfg)
    detectors_cfg = geometry_cfg["detector_planes"]
    if args.detectors is not None:      # the command line overrides the yaml
        detectors_cfg = {**detectors_cfg, "enabled": args.detectors}
    detectors = make_detector_planes(geometry, detectors_cfg)
    table = layout_table(geometry, detectors)

    # 3. Save
    n_detectors = int(table["is_detector"].sum())
    n_layers = len(geometry.phantom_layers)
    n_phantom_detectors = len(detectors.z_phantom_layers) if detectors else 0
    n_magnet_detectors = detectors.n_inside_magnet if detectors and geometry.has_magnet else 0
    x_half_gap = check_real("x_half_gap", get_key(geometry_cfg, "magnet.x_half_gap", "geometry"),
                            minimum=0.0, strict=True)
    parameters = layout_parameters(geometry, detectors, x_half_gap)
    hash_ = layout_hash(parameters)

    name = (f"layout_{n_detectors}Det_{n_layers}Lay{n_phantom_detectors}PhDet_"
            f"{n_magnet_detectors}MagDet_{hash_}")
    out_dir = DATA_DIR / "geometry" / name
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists (use --overwrite to replace it)")
        # the folder itself is not removed (OneDrive may lock it), only its files
        for file in out_dir.iterdir():
            if file.is_file():
                file.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(out_dir / "layout.csv", index=False)

    config_path = config_path.resolve()
    if config_path.is_relative_to(REPO_ROOT):
        config_path = config_path.relative_to(REPO_ROOT)
    commit, dirty = git_state()
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "config_source": str(config_path),
        "with_detectors": detectors is not None,
        "n_detector_planes": n_detectors,
        "n_phantom_layers": n_layers,
        "n_phantom_detector_planes": n_phantom_detectors,
        "n_magnet_detector_planes": n_magnet_detectors,
        "layout_hash": hash_,
        "layout_parameters": parameters,   # what the hash is computed from
        "geometry": geometry_cfg,
    }
    with open(out_dir / "metadata.yml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    print(table.to_string(index=False))
    print(f"\nSaved layout to {out_dir}")

    # 4. Report (the layout is already saved: a failure here does not lose it)
    if cfg["run"].get("plot", False) if args.plot is None else args.plot:
        try:
            report = plot_layout(*load_layout(out_dir), out_dir / REPORT_NAME)
            print(f"Saved report to {report}")
        except Exception as error:
            print(f"WARNING: report not generated ({error!r}). To retry:\n"
                  f"  python -m plotting.plot_layout {out_dir}")


if __name__ == "__main__":
    main()

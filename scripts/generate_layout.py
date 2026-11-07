# scripts/generate_layout.py
#
# Builds the geometry (and the detector planes) from a yaml config and saves the layout table
# in data/geometry/layout_<n>D_<m>INT/ to check the generated positions:
#   layout.csv:   one row per z (boundaries + detector planes, region/material on each side)
#   metadata.yml: date, git commit and the geometry block actually used
# with n = number of distinct detector planes, m = number of phantom interfaces
# (entrance and exit included, ie #layers + 1)
#
# Usage (from the repo root):
#   python -m scripts.generate_layout                         # uses configs/config.yaml
#   python -m scripts.generate_layout --no-detectors          # only the physical boundaries (0D)
#   python -m scripts.generate_layout --config my_config --overwrite

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from src.geometry.layout import layout_table
from src.utils.config_loader import CONFIG_DIR, load_config, make_detector_planes, make_geometry
from src.utils.dataset_io import DATA_DIR, REPO_ROOT, git_state


def main():

    parser = argparse.ArgumentParser(description="Save the z layout of the geometry and detector planes.")
    parser.add_argument("--config", default="config",
                        help="name of a yaml in configs/ (without extension) or path to a yaml")
    parser.add_argument("--detectors", action=argparse.BooleanOptionalAction, default=True,
                        help="include the detector planes (default: yes)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the layout folder if it already exists")
    args = parser.parse_args()

    # 1. Config
    config_path = Path(args.config)
    if config_path.suffix not in (".yaml", ".yml"):
        config_path = CONFIG_DIR / f"{args.config}.yaml"
    geometry_cfg = load_config(config_path)["geometry"]

    # 2. Geometry, detectors and table
    geometry = make_geometry(geometry_cfg)
    detectors = make_detector_planes(geometry, geometry_cfg["detector_planes"]) if args.detectors else None
    table = layout_table(geometry, detectors)

    # 3. Save
    n_detectors = int(table["is_detector"].sum())
    n_interfaces = len(geometry.z_phantom_interfaces)

    out_dir = DATA_DIR / "geometry" / f"layout_{n_detectors}D_{n_interfaces}INT"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists (use --overwrite to replace it)")
    # the folder is not removed (OneDrive may lock it): its two files are just rewritten
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
        "with_detectors": args.detectors,
        "n_detector_planes": n_detectors,
        "n_phantom_interfaces": n_interfaces,
        "geometry": geometry_cfg,
    }
    with open(out_dir / "metadata.yml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    print(table.to_string(index=False))
    print(f"\nSaved layout to {out_dir}")


if __name__ == "__main__":
    main()

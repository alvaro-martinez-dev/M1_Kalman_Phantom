# scripts/generate_hits.py
#
# Hits on the detector planes: takes a propagation dataset (data/propagation/...), smears the true
# x at each plane with the detector resolution and saves the measurements in
# data/hits/<propagation name>_hits_sigma<sigma_x>mm_seed<seed>/ (so the name carries the hashes of
# the propagation: Beam, Lay, Magn, Prop):
#   hits.parquet:   one row per (track, plane reached), only what the detector measures
#                   (see src/montecarlo/hits.py)
#   tracks.parquet: one row per track (particle, number of hits)
#   metadata.yml:   date, git commit, the propagation dataset used (and its own input, seed and
#                   layout), the seed, sigma_x and the planes
#   report.pdf:     residuals and pulls against the MC truth, if run.plot is true (plotting/plot_hits.py)
#
# It does not need the geometry or the field: the z of the planes are in the propagation dataset.
#
# Usage (from the repo root):
#   python -m scripts.generate_hits                              # uses configs/config.yaml
#   python -m scripts.generate_hits --sigma-x 0.05               # overrides hits.sigma_x
#   python -m scripts.generate_hits --no-plot                    # skips the report of run.plot
#   python -m scripts.generate_hits --input propagation/<name> --seed 7 --name test --overwrite

import argparse
from pathlib import Path

from plotting.plot_hits import REPORT_NAME, plot_hits
from src.montecarlo.hits import make_hits
from src.utils.config_loader import CONFIG_DIR, get_key, load_config
from src.utils.dataset_io import DATA_DIR, REPO_ROOT, load_dataset, load_table, save_dataset


def main():

    parser = argparse.ArgumentParser(description="Generate the detector hits of a propagation dataset.")
    parser.add_argument("--config", default="config",
                        help="name of a yaml in configs/ (without extension) or path to a yaml")
    parser.add_argument("--input", default=None,
                        help="propagation dataset, relative to data/ (default: hits.input)")
    parser.add_argument("--sigma-x", type=float, default=None,
                        help="[mm] resolution of every plane (default: hits.sigma_x)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the measurement errors (default: hits.seed)")
    parser.add_argument("--name", default=None,
                        help="dataset folder name (default: <input name>_hits_sigma<sigma_x>mm_seed<seed>)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the dataset folder if it already exists")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None,
                        help=f"write {REPORT_NAME} in the dataset folder (default: run.plot of the yaml)")
    args = parser.parse_args()

    # 1. Config and propagation dataset
    config_path = Path(args.config)
    if config_path.suffix not in (".yaml", ".yml"):
        config_path = CONFIG_DIR / f"{args.config}.yaml"
    cfg = load_config(config_path)
    if "hits" not in cfg:
        raise KeyError("missing block 'hits' in the yaml")

    input_path = args.input or get_key(cfg["hits"], "input", "hits")
    seed = args.seed if args.seed is not None else get_key(cfg["hits"], "seed", "hits")
    sigma_x = args.sigma_x if args.sigma_x is not None else get_key(cfg["hits"], "sigma_x", "hits")

    tracks, prop_metadata = load_dataset(input_path)
    states = load_table(input_path, "states")
    if seed == prop_metadata.get("seed"):
        raise ValueError(f"hits.seed = {seed} is the seed of the propagation {input_path}: use another one")

    # 2. Hits
    hits, hit_tracks = make_hits(states, tracks, sigma_x, seed)

    # 3. Save
    input_name = Path(input_path).name
    name = args.name or f"{input_name}_hits_sigma{float(sigma_x):g}mm_seed{seed}"
    config_path = config_path.resolve()
    if config_path.is_relative_to(REPO_ROOT):
        config_path = config_path.relative_to(REPO_ROOT)
    params = {
        "config_source": str(config_path),
        # propagation dataset used, and what it was made from (to trace the hits to the beam)
        "input": str(input_path),
        "input_created": prop_metadata.get("created"),
        "input_git_commit": prop_metadata.get("git_commit"),
        "propagation": {
            "input": prop_metadata.get("input"),          # initial beam dataset
            "seed": prop_metadata.get("seed"),
            "n_tracks": prop_metadata.get("n_tracks"),
            "n_stopped": prop_metadata.get("n_stopped"),
            "layout_name": prop_metadata.get("layout_name"),
            # hashes of what the propagation was made from (also in its folder name)
            "beam_hash": prop_metadata.get("beam_hash"),
            "layout_hash": prop_metadata.get("layout_hash"),
            "magnet_hash": prop_metadata.get("magnet_hash"),
            "propagator_hash": prop_metadata.get("propagator_hash"),
        },
        "seed": seed,
        "sigma_x": float(sigma_x),
        "particle": prop_metadata.get("particle"),
        "n_tracks": len(hit_tracks),
        "n_hits": len(hits),
        "planes": prop_metadata.get("planes"),
    }
    out_dir = save_dataset(hit_tracks, DATA_DIR / "hits" / name, params,
                           overwrite=args.overwrite, tables={"hits": hits})
    print(f"Saved {len(hits)} hits of {len(hit_tracks)} tracks (sigma_x = {float(sigma_x):g} mm) to {out_dir}")

    # 4. Report (the dataset is already saved: a failure here does not lose it)
    if cfg["run"].get("plot", False) if args.plot is None else args.plot:
        try:
            print(f"Saved report to {plot_hits(out_dir)}")
        except Exception as error:
            print(f"WARNING: report not generated ({error!r}). To retry:\n"
                  f"  python -m plotting.plot_hits {out_dir}")


if __name__ == "__main__":
    main()

# scripts/generate_init_beam.py
#
# Generates an initial beam dataset from a yaml config and saves it in data/init_beam/<name>/
#
# Usage (from the repo root):
#   python -m scripts.generate_init_beam                      # uses configs/config.yaml and run.beam
#   python -m scripts.generate_init_beam --beam proton
#   python -m scripts.generate_init_beam --config my_config --beam pion --name pion_test --overwrite

import argparse
from pathlib import Path

from src.utils.config_loader import CONFIG_DIR, load_config, make_beam
from src.utils.dataset_io import DATA_DIR, REPO_ROOT, save_dataset


def main():

    parser = argparse.ArgumentParser(description="Generate an initial beam dataset.")
    parser.add_argument("--config", default="config",
                        help="name of a yaml in configs/ (without extension) or path to a yaml")
    parser.add_argument("--beam", default=None,
                        help="block of beams: to simulate (default: run.beam of the yaml)")
    parser.add_argument("--name", default=None,
                        help="dataset folder name (default: <beam>_n<n_events>_seed<seed>)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the dataset folder if it already exists")
    args = parser.parse_args()

    # 1. Config (same path resolution as load_config)
    config_path = Path(args.config)
    if config_path.suffix not in (".yaml", ".yml"):
        config_path = CONFIG_DIR / f"{args.config}.yaml"
    cfg = load_config(config_path)

    run = cfg["run"]
    beam_name = args.beam or run["beam"]
    if beam_name not in cfg["beams"]:
        raise ValueError(f"beam '{beam_name}' not in beams: {list(cfg['beams'])}")
    n_events, seed = run["n_events"], run["seed"]

    # 2. Generation
    df = make_beam(cfg["beams"][beam_name]).generate(n_events, seed=seed)

    # 3. Save
    name = args.name or f"{beam_name}_n{n_events}_seed{seed}"
    config_path = config_path.resolve()
    if config_path.is_relative_to(REPO_ROOT):
        config_path = config_path.relative_to(REPO_ROOT)  # e.g. configs/config.yaml
    params = {
        "config_source": str(config_path),
        "run": {"beam": beam_name, "n_events": n_events, "seed": seed},
        "beam": cfg["beams"][beam_name],
    }
    out_dir = save_dataset(df, DATA_DIR / "init_beam" / name, params,
                           overwrite=args.overwrite)
    print(f"Saved {len(df)} tracks to {out_dir}")


if __name__ == "__main__":
    main()

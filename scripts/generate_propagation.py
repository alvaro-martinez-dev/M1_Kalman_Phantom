# scripts/generate_propagation.py
#
# Monte Carlo of the full propagation: takes a dataset of initial tracks (e.g. data/init_beam/...),
# propagates every track through the system with the scattering kicks and saves the TRUE state at
# each detector plane in data/propagation/<name>/:
#   tracks.parquet: one row per track (stopped, final state, planes reached)
#   states.parquet: one row per (track, plane reached), see src/montecarlo/propagation.py
#   metadata.yml:   date, git commit, input, seed and the geometry/field/propagator blocks used
#
# Usage (from the repo root):
#   python -m scripts.generate_propagation                       # uses configs/config.yaml
#   python -m scripts.generate_propagation --n-max 200           # only the first 200 tracks (quick test)
#   python -m scripts.generate_propagation --input init_beam/proton_n1000_seed1 --name test --overwrite

import argparse
from pathlib import Path

import numpy as np

from src.field import build_field
from src.montecarlo.propagation import plane_list, propagate_tracks
from src.utils.config_loader import (CONFIG_DIR, get_constant, get_key, load_config, make_detector_planes,
                                     make_geometry, make_propagator)
from src.utils.dataset_io import DATA_DIR, REPO_ROOT, load_dataset, save_dataset


def main():

    parser = argparse.ArgumentParser(description="Propagate a dataset of initial tracks through the system.")
    parser.add_argument("--config", default="config",
                        help="name of a yaml in configs/ (without extension) or path to a yaml")
    parser.add_argument("--input", default=None,
                        help="dataset of initial tracks, relative to data/ (default: propagator.input)")
    parser.add_argument("--n-max", type=int, default=None,
                        help="propagate only the first n tracks (default: all)")
    parser.add_argument("--name", default=None,
                        help="dataset folder name (default: <input name>_prop_seed<seed>[_n<n-max>])")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the dataset folder if it already exists")
    args = parser.parse_args()

    # 1. Config and input tracks
    config_path = Path(args.config)
    if config_path.suffix not in (".yaml", ".yml"):
        config_path = CONFIG_DIR / f"{args.config}.yaml"
    cfg = load_config(config_path)

    input_path = args.input or get_key(cfg["propagator"], "input", "propagator")
    seed = get_key(cfg["propagator"], "seed", "propagator")
    df_tracks, input_metadata = load_dataset(input_path)
    if args.n_max is not None:
        df_tracks = df_tracks.iloc[:args.n_max]

    particles = df_tracks["particle"].unique()
    if len(particles) != 1:
        raise ValueError(f"The input must contain a single particle species, got {list(particles)}")
    particle = get_constant(_constant_name(particles[0]))

    # 2. System: geometry, detector planes, field, propagator
    geometry = make_geometry(cfg["geometry"])
    detectors = make_detector_planes(geometry, cfg["geometry"]["detector_planes"])
    if detectors is None:
        raise ValueError("The detector planes are disabled (geometry.detector_planes.enabled): "
                         "there is nowhere to store the states")
    planes = plane_list(detectors)
    field = build_field(cfg, geometry)
    propagator = make_propagator(cfg, geometry, field, particle)

    # 3. Propagation (no kicks if the scattering is disabled)
    rng = np.random.default_rng(seed) if propagator.max_slice is not None else None
    print(f"Propagating {len(df_tracks)} {particle.name} tracks through {len(planes)} planes "
          f"(scattering {'on' if rng is not None else 'off'})")
    states, tracks = propagate_tracks(df_tracks, propagator, planes, rng,
                                      progress_every=max(1, len(df_tracks) // 20))

    # 4. Save
    input_name = Path(input_path).name
    name = args.name or (f"{input_name}_prop_seed{seed}" + (f"_n{args.n_max}" if args.n_max is not None else ""))
    config_path = config_path.resolve()
    if config_path.is_relative_to(REPO_ROOT):
        config_path = config_path.relative_to(REPO_ROOT)
    params = {
        "config_source": str(config_path),
        "input": str(input_path),
        "input_created": input_metadata.get("created"),
        "seed": seed,
        "n_tracks": len(tracks),
        "n_stopped": int(tracks["stopped"].sum()),
        "particle": particle.name,
        "planes": [{"name": n, "z": float(z)} for n, z in planes],
        "geometry": cfg["geometry"],
        "propagator": cfg["propagator"],
    }
    out_dir = save_dataset(tracks, DATA_DIR / "propagation" / name, params,
                           overwrite=args.overwrite, tables={"states": states})
    print(f"Saved {len(tracks)} tracks ({params['n_stopped']} stopped) and {len(states)} plane states to {out_dir}")


def _constant_name(particle_name):
    # Name of the constant in src/constants.py of a Particle, from its name in the track table
    import src.constants as constants
    for key, value in vars(constants).items():
        if isinstance(value, constants.Particle) and value.name == particle_name:
            return key
    raise ValueError(f"No Particle named {particle_name!r} in src/constants.py")


if __name__ == "__main__":
    main()

# scripts/generate_propagation.py
#
# Monte Carlo of the full propagation: takes a dataset of initial tracks (e.g. data/init_beam/...),
# propagates every track through the system with the scattering kicks and saves the TRUE state at
# each detector plane in data/propagation/<name>/:
#   tracks.parquet: one row per track (stopped, final state, planes reached)
#   states.parquet: one row per (track, plane reached), see src/montecarlo/propagation.py
#   metadata.yml:   date, git commit, input, seed, the layout it corresponds to (layout_name, the
#                   folder generate_layout would write for this geometry) and the geometry/field/
#                   propagator blocks used
#   tracks_on_layout.pdf: a few tracks drawn on the system, if run.plot is true (plotting/plot_tracks.py)
#
# Usage (from the repo root):
#   python -m scripts.generate_propagation                       # uses configs/config.yaml
#   python -m scripts.generate_propagation --n-max 200           # 200 random tracks (overrides propagator.n_max)
#   python -m scripts.generate_propagation --no-plot             # skips the plot of run.plot
#   python -m scripts.generate_propagation --input init_beam/proton_n1000_seed1 --name test --overwrite

import argparse
from pathlib import Path

from plotting.plot_tracks import REPORT_NAME, plot_tracks
from src.field import build_field
from src.geometry.layout import layout_name
from src.montecarlo.propagation import plane_list, propagate_tracks, select_tracks, selection_rng
from src.utils.config_loader import (CONFIG_DIR, get_key, load_config, make_detector_planes, make_geometry,
                                     make_propagator, particle_by_name)
from src.utils.dataset_io import DATA_DIR, REPO_ROOT, load_dataset, save_dataset


def main():

    parser = argparse.ArgumentParser(description="Propagate a dataset of initial tracks through the system.")
    parser.add_argument("--config", default="config",
                        help="name of a yaml in configs/ (without extension) or path to a yaml")
    parser.add_argument("--input", default=None,
                        help="dataset of initial tracks, relative to data/ (default: propagator.input)")
    parser.add_argument("--n-max", type=int, default=None,
                        help="propagate only n tracks, chosen at random (default: propagator.n_max)")
    parser.add_argument("--name", default=None,
                        help="dataset folder name (default: <input name>_prop_seed<seed>[_n<n_max>])")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the dataset folder if it already exists")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None,
                        help=f"write {REPORT_NAME} in the dataset folder (default: run.plot of the yaml)")
    args = parser.parse_args()

    # 1. Config and input tracks
    config_path = Path(args.config)
    if config_path.suffix not in (".yaml", ".yml"):
        config_path = CONFIG_DIR / f"{args.config}.yaml"
    cfg = load_config(config_path)

    input_path = args.input or get_key(cfg["propagator"], "input", "propagator")
    seed = get_key(cfg["propagator"], "seed", "propagator")
    n_max = args.n_max if args.n_max is not None else get_key(cfg["propagator"], "n_max", "propagator")
    df_input, input_metadata = load_dataset(input_path)

    # Independent random streams from the seed: the choice of the tracks, and the kicks of each track
    df_tracks = select_tracks(df_input, n_max, selection_rng(seed))

    particles = df_tracks["particle"].unique()
    if len(particles) != 1:
        raise ValueError(f"The input must contain a single particle species, got {list(particles)}")
    particle = particle_by_name(particles[0])

    # 2. System: geometry, detector planes, field, propagator
    geometry = make_geometry(cfg["geometry"])
    detectors = make_detector_planes(geometry, cfg["geometry"]["detector_planes"])
    if detectors is None:
        raise ValueError("The detector planes are disabled (geometry.detector_planes.enabled): "
                         "there is nowhere to store the states")
    planes = plane_list(detectors)
    x_half_gap = get_key(cfg["geometry"], "magnet.x_half_gap", "geometry") if geometry.has_magnet else None
    layout, _, layout_hash = layout_name(geometry, detectors, x_half_gap)
    field = build_field(cfg, geometry)
    propagator = make_propagator(cfg, geometry, field, particle)

    # 3. Propagation (no kicks if the scattering is disabled)
    kick_seed = seed if propagator.max_slice is not None else None
    print(f"Propagating {len(df_tracks)} of the {len(df_input)} {particle.name} tracks of {input_path} "
          f"through {len(planes)} planes "
          f"(scattering {'on' if kick_seed is not None else 'off'})")
    states, tracks = propagate_tracks(df_tracks, propagator, planes, kick_seed,
                                      progress_every=max(1, len(df_tracks) // 20))

    # 4. Save
    input_name = Path(input_path).name
    name = args.name or (f"{input_name}_prop_seed{seed}" + (f"_n{n_max}" if n_max is not None else ""))
    config_path = config_path.resolve()
    if config_path.is_relative_to(REPO_ROOT):
        config_path = config_path.relative_to(REPO_ROOT)
    params = {
        "config_source": str(config_path),
        "input": str(input_path),
        "input_created": input_metadata.get("created"),
        "seed": seed,
        "n_input_tracks": len(df_input),
        "n_max": n_max,                     # null: all the tracks; else chosen at random (seed)
        "n_tracks": len(tracks),
        "n_stopped": int(tracks["stopped"].sum()),
        "particle": particle.name,
        "layout_name": layout,              # data/geometry/<layout_name> if generate_layout was run on it
        "layout_hash": layout_hash,
        "planes": [{"name": n, "z": float(z)} for n, z in planes],
        "geometry": cfg["geometry"],
        "propagator": cfg["propagator"],
    }
    out_dir = save_dataset(tracks, DATA_DIR / "propagation" / name, params,
                           overwrite=args.overwrite, tables={"states": states})
    print(f"Saved {len(tracks)} tracks ({params['n_stopped']} stopped) and {len(states)} plane states to {out_dir}")

    # 5. Plot (the dataset is already saved: a failure here does not lose it)
    if cfg["run"].get("plot", False) if args.plot is None else args.plot:
        try:
            print(f"Saved plot to {plot_tracks(out_dir)}")
        except Exception as error:
            print(f"WARNING: plot not generated ({error!r}). To retry:\n"
                  f"  python -m plotting.plot_tracks {out_dir}")


if __name__ == "__main__":
    main()

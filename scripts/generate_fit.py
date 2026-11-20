# scripts/generate_fit.py
#
# Kalman fit of the tracks of a hits dataset (data/hits/...): forward Kalman filter and, if
# kalman.smoother is true, the RTS smoother (src/kalman/). Saves in
# data/fit/<hits name>_Kal<hash>[_n<n_max>]/, with hash = kalman_hash of src/utils/hashing.py
# (smoother, central_differences, finite-difference steps); the hits name already carries the
# hashes of the propagation (Beam, Lay, Magn, Prop):
#   tracks.parquet: one row per track (status, chi2, ndf), see src/kalman/fit.py
#   fit.parquet:    one row per (track, hit): predicted, filtered and smoothed states with errors
#                   and correlations, and the predicted and smoothed residuals
#   metadata.yml:   date, git commit, the hits dataset used and what it was made from (propagation,
#                   initial beam), the prior, the options of the fit and a summary (chi2/ndf)
#   report.pdf:     the fit against the MC truth, if run.plot is true (plotting/plot_fit.py)
#
# The model of the fit (geometry, field, propagator, prior) is read from the metadata of the
# datasets, not from the yaml: the yaml only chooses the input and the options of the fit.
#
# Usage (from the repo root):
#   python -m scripts.generate_fit                               # uses configs/config.yaml
#   python -m scripts.generate_fit --n-max 200                   # only the first 200 tracks
#   python -m scripts.generate_fit --no-smoother --workers 4
#   python -m scripts.generate_fit --input hits/<name> --name test --overwrite --no-plot

import argparse
import time
from pathlib import Path

import numpy as np

from plotting.plot_fit import REPORT_NAME, plot_fit
from src.kalman.fit import fit_tracks_parallel, load_model
from src.kalman.transport import FD_STEPS
from src.propagator.multiprocess_propagation import resolve_workers
from src.utils.config_loader import CONFIG_DIR, get_key, load_config
from src.utils.dataset_io import DATA_DIR, REPO_ROOT, save_dataset
from src.utils.hashing import kalman_hash
from src.utils.validation import check_bool, check_count


def main():

    parser = argparse.ArgumentParser(description="Kalman fit of the tracks of a hits dataset.")
    parser.add_argument("--config", default="config",
                        help="name of a yaml in configs/ (without extension) or path to a yaml")
    parser.add_argument("--input", default=None,
                        help="hits dataset, relative to data/ (default: kalman.input)")
    parser.add_argument("--n-max", type=int, default=None,
                        help="fit only the first n tracks (default: kalman.n_max)")
    parser.add_argument("--smoother", action=argparse.BooleanOptionalAction, default=None,
                        help="also run the RTS smoother (default: kalman.smoother)")
    parser.add_argument("--workers", type=int, default=None,
                        help="number of processes (default: kalman.n_workers; 1 = serial)")
    parser.add_argument("--name", default=None,
                        help="dataset folder name (default: <input name>_Kal<hash>[_n<n_max>])")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the dataset folder if it already exists")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None,
                        help=f"write {REPORT_NAME} in the dataset folder (default: run.plot of the yaml)")
    args = parser.parse_args()

    # 1. Config
    config_path = Path(args.config)
    if config_path.suffix not in (".yaml", ".yml"):
        config_path = CONFIG_DIR / f"{args.config}.yaml"
    cfg = load_config(config_path)
    if not cfg.get("kalman"):
        raise KeyError("missing block 'kalman' in the yaml")

    def get(path):
        return get_key(cfg["kalman"], path, "kalman")

    input_path = args.input or get("input")
    n_max = args.n_max if args.n_max is not None else get("n_max")
    n_max = None if n_max is None else check_count("kalman.n_max", n_max)
    smoother = check_bool("kalman.smoother", args.smoother if args.smoother is not None else get("smoother"))
    central = check_bool("kalman.central_differences", get("central_differences"))
    n_workers = resolve_workers(args.workers if args.workers is not None else get("n_workers"))

    # 2. Model of the fit, from the metadata of the hits, propagation and initial beam datasets
    model = load_model(input_path, central=central)
    hits = model["hits"]
    track_ids = np.unique(hits["track_id"])
    if n_max is not None:
        if not 1 <= n_max <= len(track_ids):
            raise ValueError(f"n_max = {n_max} must be between 1 and the {len(track_ids)} tracks of {input_path} "
                             "(or null for all of them)")
        track_ids = track_ids[:n_max]
        hits = hits[hits["track_id"].isin(track_ids)]

    # 3. Fit
    print(f"Fitting {len(track_ids)} {model['particle'].name} tracks ({len(hits)} hits) of {input_path} "
          f"(smoother {'on' if smoother else 'off'}, {n_workers} processes)")
    t_start = time.perf_counter()
    fit, fit_tracks = fit_tracks_parallel(hits, model["transport"], model["prior"], smoother=smoother,
                                          n_workers=n_workers)
    elapsed = time.perf_counter() - t_start

    # 4. Save
    ok = fit_tracks[fit_tracks["status"] == "ok"]
    hits_meta, prop_meta = model["metadata"]["hits"], model["metadata"]["propagation"]
    z0, x0, C0 = model["prior"]
    fd_steps = [float(s) for s in model["transport"].fd_steps]
    kal_hash = kalman_hash(smoother, central, fd_steps)
    name = args.name or (f"{Path(input_path).name}_Kal{kal_hash}" + (f"_n{n_max}" if n_max is not None else ""))
    config_path = config_path.resolve()
    if config_path.is_relative_to(REPO_ROOT):
        config_path = config_path.relative_to(REPO_ROOT)
    params = {
        "config_source": str(config_path),
        # hits dataset used, and what it was made from (to trace the fit to the beam)
        "input": str(input_path),
        "input_created": hits_meta.get("created"),
        "input_git_commit": hits_meta.get("git_commit"),
        "propagation": {
            "input": model["propagation_path"],     # where the MC truth is (states.parquet)
            "created": prop_meta.get("created"),
            "git_commit": prop_meta.get("git_commit"),
            "seed": prop_meta.get("seed"),
            "layout_name": prop_meta.get("layout_name"),
            # hashes of what the propagation was made from (also in the folder name)
            "beam_hash": prop_meta.get("beam_hash"),
            "layout_hash": prop_meta.get("layout_hash"),
            "magnet_hash": prop_meta.get("magnet_hash"),
            "propagator_hash": prop_meta.get("propagator_hash"),
        },
        "beam": {"input": model["beam_path"], **model["metadata"]["beam"]["beam"]},
        "particle": model["particle"].name,
        "sigma_x": hits_meta.get("sigma_x"),
        "planes": hits_meta.get("planes"),
        # options of the fit (kalman_hash: smoother, central_differences, fd_steps)
        "kalman_hash": kal_hash,
        "n_max": n_max,
        "smoother": smoother,
        "central_differences": central,
        "fd_steps": fd_steps,
        "prior": {"z0": float(z0), "x0": [float(v) for v in x0], "C0_diag": [float(v) for v in np.diag(C0)]},
        # summary
        "n_tracks": len(fit_tracks),
        "n_ok": len(ok),
        "n_stopped": int((fit_tracks["status"] == "stopped").sum()),
        "n_hits": len(fit),
        "chi2_ndf_mean": float((ok["chi2"] / ok["ndf"]).mean()) if len(ok) else None,
        "time_s": round(elapsed, 1),
        "n_workers": n_workers,
    }
    out_dir = save_dataset(fit_tracks, DATA_DIR / "fit" / name, params,
                           overwrite=args.overwrite, tables={"fit": fit})
    print(f"Saved the fit of {len(fit_tracks)} tracks ({params['n_ok']} ok, {params['n_stopped']} stopped, "
          f"mean chi2/ndf = {params['chi2_ndf_mean']:.3f}) in {elapsed:.0f} s to {out_dir}")

    # 5. Report (the dataset is already saved: a failure here does not lose it)
    if cfg["run"].get("plot", False) if args.plot is None else args.plot:
        try:
            print(f"Saved report to {plot_fit(out_dir)}")
        except Exception as error:
            print(f"WARNING: report not generated ({error!r}). To retry:\n"
                  f"  python -m plotting.plot_fit {out_dir}")


if __name__ == "__main__":
    main()

# src/kalman/fit.py
#
# Fits the tracks of a hits dataset (data/hits/...) with the Kalman filter and, optionally, the
# RTS smoother on top of it (the smoother needs the filter, so both come from the same run and can
# be compared directly).
#
# The model of the fit is the one of the Monte Carlo, read from the metadata of the datasets:
#   hits metadata -> propagation dataset: geometry, field and propagator blocks (Propagator)
#                 -> initial beam dataset: beam block (prior)
# so the filter transports with exactly the geometry, field, steps and scattering slices that were
# simulated, whatever configs/config.yaml says now. The code must be the one of the propagation:
# a change of the field or propagator code afterwards biases the fit (regenerate the datasets).
#
# Prior: the beam at z0 = 0 (src/montecarlo/init_beam.py), a Gaussian in (x, theta, p):
#   x0 = beam.x0,  tx0 = tan(beam.theta0),  q/p0 = q / beam.p0
#   C0 = diag(sigma_x0^2, ((1 + tx0^2) sigma_theta0)^2, (|q| sigma_p0 / p0^2)^2)   (linearised)
# It uses only the parameters of the beam, never the true initial state of each track.
#
# The fit does not read the MC truth (states.parquet of the propagation): the comparison with the
# truth (residuals, pulls) is done afterwards (plotting/plot_fit.py), joining by (track_id, plane_index).
#
# Output, two tables:
#   fit: one row per (track, hit), the identifiers of the hit and, for stage in pred, filt, smooth:
#       x_<stage>, tx_<stage>, q_over_p_<stage> and their errors <var>_<stage>_err (sqrt of the
#       diagonal of the covariance) and its correlations corr_x_tx_<stage>, corr_x_q_over_p_<stage>,
#       corr_tx_q_over_p_<stage>; res_pred, res_pred_err, chi2_k (predicted residuals);
#       res_smooth, res_smooth_err (smoothed residuals). The smooth columns only if smoother=True;
#       s_reach: safety-net bound s < s_reach [1/MeV] on s = 1/p at the previous hit, applied only if
#       the mean track would not reach this hit (NaN if not applied)
#   fit_tracks: one row per track: track_id, input_row, event_id, n_hits, status, chi2, ndf,
#       range_end, s_hi, s_lo, n_reach
#       status: "ok", or "stopped" if the predicted track stops before one of its hits (no fit rows)
#       range_end: the track has no hit on the plane after its last hit, i.e. it stopped (planes are
#                  100 % efficient). It is detector information, not MC truth
#       s_hi, s_lo: bounds s_lo < s0 < s_hi [1/MeV] on s = 1/p of the prior: it reached its last hit,
#                   it did not reach the next plane (NaN if not applied, e.g. irrelevant)
#       n_reach: number of safety-net bounds applied
#
# Range constraints (range_constraints=True): the presence/absence of hits bounds the energy, see
# src/kalman/constraints.py and src/kalman/filter.py. The plane after the last hit of each track is
# found in the list of planes of the hits dataset (plane_index is the index in that list).
#
# fit_tracks_parallel runs the same fit in several processes (as multiprocess_propagation): the
# tracks are independent, so the result does not depend on the number of processes.

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.field import build_field
from src.kalman.filter import run_filter
from src.kalman.smoother import run_smoother
from src.kalman.transport import FD_STEPS, Transport, TrackStopped
from src.propagator.multiprocess_propagation import resolve_workers
from src.utils.config_loader import make_geometry, make_propagator, particle_by_name
from src.utils.dataset_io import load_dataset, load_table

STATE_VARS = ("x", "tx", "q_over_p")
# Off-diagonal pairs ((index, name), (index, name)) of the covariance stored as correlations
CORR_PAIRS = [((a, STATE_VARS[a]), (b, STATE_VARS[b])) for a in range(3) for b in range(a + 1, 3)]
HIT_ID_COLUMNS = ["track_id", "input_row", "event_id", "plane_index", "plane", "z", "x_meas", "sigma_x"]
TRACK_COLUMNS = ["track_id", "input_row", "event_id", "n_hits", "status", "chi2", "ndf",
                 "range_end", "s_hi", "s_lo", "n_reach"]

# Chunks per process (as multiprocess_propagation) and of a serial run (only for the progress bar)
CHUNKS_PER_WORKER = 8
SERIAL_CHUNKS = 200


# beam: beam block of the yaml (x0, sigma_x0, theta0, sigma_theta0, p0, sigma_p0)
# particle: Particle of the beam
# Returns: (z0, x0, C0), see the header
def beam_prior(beam, particle):
    tx0 = np.tan(beam["theta0"])
    p0 = beam["p0"]
    x0 = np.array([beam["x0"], tx0, particle.charge / p0])
    C0 = np.diag([beam["sigma_x0"]**2,
                  ((1.0 + tx0**2) * beam["sigma_theta0"])**2,
                  (abs(particle.charge) * beam["sigma_p0"] / p0**2)**2])
    return 0.0, x0, C0


# hits_path: hits dataset, relative to data/ (e.g. "hits/<name>")
# fd_steps: finite-difference steps of the Jacobians (default: transport.FD_STEPS)
# central: central differences instead of forward ones
# Returns: dict with hits, tracks (DataFrames), transport (Transport), prior (z0, x0, C0), particle,
#          planes_z (z of the planes of the hits dataset, indexed by plane_index),
#          the paths of the propagation and initial beam datasets and the metadata of the three
def load_model(hits_path, fd_steps=FD_STEPS, central=False):
    tracks, hits_meta = load_dataset(hits_path)
    hits = load_table(hits_path, "hits")

    prop_path = hits_meta["input"]
    _, prop_meta = load_dataset(prop_path)
    if prop_meta.get("created") != hits_meta.get("input_created"):
        warnings.warn(f"{prop_path} was regenerated after the hits {hits_path} were made from it: the "
                      "model of the fit is the new propagation, but the hits come from the old one "
                      "(regenerate the hits)", stacklevel=2)
    beam_path = prop_meta["input"]
    _, beam_meta = load_dataset(beam_path)
    if "beam" not in beam_meta:
        raise ValueError(f"{beam_path} is not an initial beam dataset (no 'beam' block in its metadata): "
                         "the prior of the fit is the beam")

    cfg = {"geometry": prop_meta["geometry"], "propagator": prop_meta["propagator"]}
    particle = particle_by_name(prop_meta["particle"])
    geometry = make_geometry(cfg["geometry"])
    field = build_field(cfg, geometry)
    propagator = make_propagator(cfg, geometry, field, particle)

    if "planes" not in hits_meta:
        raise ValueError(f"{hits_path} has no list of planes in its metadata (needed for the range constraints)")
    planes_z = np.array([float(p["z"]) for p in hits_meta["planes"]])

    return {
        "hits": hits, "tracks": tracks, "particle": particle, "planes_z": planes_z,
        "transport": Transport(propagator, fd_steps=fd_steps, central=central),
        "prior": beam_prior(beam_meta["beam"], particle),
        "propagation_path": str(prop_path), "beam_path": str(beam_path),
        "metadata": {"hits": hits_meta, "propagation": prop_meta, "beam": beam_meta},
    }


# Columns of the fit table for one stage, from (n, 3) states and (n, 3, 3) covariances: state,
# errors and the correlation coefficients corr_<a>_<b>_<stage> (with the errors they give back the
# full covariance, which the NEES of plotting/plot_fit.py needs)
def _stage_columns(stage, x, C):
    err = np.sqrt(np.diagonal(C, axis1=1, axis2=2))
    cols = {}
    for j, var in enumerate(STATE_VARS):
        cols[f"{var}_{stage}"] = x[:, j]
        cols[f"{var}_{stage}_err"] = err[:, j]
    for (a, va), (b, vb) in CORR_PAIRS:
        cols[f"corr_{va}_{vb}_{stage}"] = C[:, a, b] / (err[:, a] * err[:, b])
    return cols


# planes_z: (n_planes,) z [mm] of the planes of the hits dataset, indexed by plane_index
# hits_track: hits of ONE track (rows of hits.parquet)
# Returns: z of the plane after the last hit (where the track left no hit: it stopped before it), or
#          None if the last hit is on the last plane (or planes_z is None)
def next_plane_z(planes_z, hits_track):
    if planes_z is None:
        return None
    k_next = int(hits_track["plane_index"].max()) + 1
    return float(planes_z[k_next]) if k_next < len(planes_z) else None


# transport: Transport; prior: (z0, x0, C0)
# hits_track: hits of ONE track (rows of hits.parquet), any order
# smoother: also run the RTS smoother
# planes_z: z of the planes of the dataset (next_plane_z), for the stop constraint
# range_constraints: apply the range constraints (src/kalman/filter.py)
# Returns: (DataFrame of its fit rows, FilterResult, SmootherResult or None)
# Raises TrackStopped if the predicted track stops before one of its hits
def fit_track(transport, prior, hits_track, smoother=True, planes_z=None, range_constraints=False):
    hits_track = hits_track.sort_values("z")
    z0, x0, C0 = prior
    res = run_filter(transport, x0, C0, z0, hits_track["z"].to_numpy(),
                     hits_track["x_meas"].to_numpy(), hits_track["sigma_x"].to_numpy(),
                     z_next=next_plane_z(planes_z, hits_track), range_constraints=range_constraints)

    cols = {c: hits_track[c].to_numpy() for c in HIT_ID_COLUMNS}
    cols.update(_stage_columns("pred", res.x_pred, res.C_pred))
    cols.update(_stage_columns("filt", res.x_filt, res.C_filt))
    cols.update(res_pred=res.r, res_pred_err=np.sqrt(res.S), chi2_k=res.chi2_k)

    smooth = None
    if smoother:
        smooth = run_smoother(res)
        cols.update(_stage_columns("smooth", smooth.x_smooth, smooth.C_smooth))
        cols.update(res_smooth=smooth.r, res_smooth_err=np.sqrt(np.clip(smooth.R, 0.0, None)))
    cols.update(s_reach=res.s_reach)

    return pd.DataFrame(cols), res, smooth


# hits: hits table (hits.parquet), or the hits of some of its tracks; transport, prior: as in fit_track
# planes_z, range_constraints: as in fit_track
# Returns: (fit DataFrame, fit_tracks DataFrame), see the header, with the tracks sorted by track_id
def fit_tracks(hits, transport, prior, smoother=True, planes_z=None, range_constraints=False):
    fit_rows, track_rows = [], []
    for track_id, h in hits.groupby("track_id", sort=True):
        first = h.iloc[0]
        range_end = next_plane_z(planes_z, h) is not None
        try:
            df, res, _ = fit_track(transport, prior, h, smoother=smoother, planes_z=planes_z,
                                   range_constraints=range_constraints)
            fit_rows.append(df)
            status, chi2, ndf = "ok", res.chi2, res.ndf
            s_hi, s_lo, n_reach = res.s_hi, res.s_lo, int(np.isfinite(res.s_reach).sum())
        except TrackStopped:
            status, chi2, ndf, s_hi, s_lo, n_reach = "stopped", np.nan, 0, np.nan, np.nan, 0
        track_rows.append((track_id, first["input_row"], first["event_id"], len(h), status, chi2, ndf,
                           range_end, s_hi, s_lo, n_reach))

    fit = pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame()
    return fit, pd.DataFrame(track_rows, columns=TRACK_COLUMNS)


# Each worker receives the transport (propagator with its field, ~MB) once, not once per chunk
_WORKER_ARGS = {}


def _init_worker(transport, prior, smoother, planes_z, range_constraints):
    _WORKER_ARGS.update(transport=transport, prior=prior, smoother=smoother, planes_z=planes_z,
                        range_constraints=range_constraints)


def _run_chunk(hits_chunk):
    return fit_tracks(hits_chunk, **_WORKER_ARGS)


# Same arguments and output as fit_tracks, plus
# n_workers: number of processes (None: half the CPUs; 1: serial, in this process)
# progress: show a tqdm progress bar (in tracks)
# On Windows the calling script must be protected by  if __name__ == "__main__":
def fit_tracks_parallel(hits, transport, prior, smoother=True, n_workers=None, progress=True,
                        planes_z=None, range_constraints=False):
    track_ids = np.unique(hits["track_id"])
    n_tracks = len(track_ids)
    n_workers = min(resolve_workers(n_workers), max(1, n_tracks))

    # Contiguous chunks of tracks, in track_id order
    n_chunks = SERIAL_CHUNKS if n_workers == 1 else n_workers * CHUNKS_PER_WORKER
    bounds = np.linspace(0, n_tracks, min(n_tracks, n_chunks) + 1).astype(int)
    by_track = hits.sort_values(["track_id", "z"], kind="stable")
    first_row = np.searchsorted(by_track["track_id"].to_numpy(), track_ids)
    row_bounds = np.append(first_row, len(by_track))[bounds]
    chunks = [by_track.iloc[a:b] for a, b in zip(row_bounds[:-1], row_bounds[1:])]
    sizes = np.diff(bounds)

    results = [None] * len(chunks)
    bar = tqdm(total=n_tracks, unit="track", disable=not progress, dynamic_ncols=True,
               desc="Fitting" + (f" ({n_workers} processes)" if n_workers > 1 else ""))
    with bar:
        if n_workers == 1:
            for k, chunk in enumerate(chunks):
                results[k] = fit_tracks(chunk, transport, prior, smoother=smoother, planes_z=planes_z,
                                        range_constraints=range_constraints)
                bar.update(sizes[k])
        else:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                     initargs=(transport, prior, smoother, planes_z, range_constraints)) as pool:
                futures = {pool.submit(_run_chunk, chunk): k for k, chunk in enumerate(chunks)}
                for future in as_completed(futures):
                    k = futures[future]
                    results[k] = future.result()
                    bar.update(sizes[k])

    fits = [f for f, _ in results if not f.empty]
    fit = pd.concat(fits, ignore_index=True) if fits else pd.DataFrame()
    return fit, pd.concat([t for _, t in results], ignore_index=True)

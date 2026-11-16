# src/montecarlo/hits.py
#
# Hits on the detector planes: the measurement of each plane, from the TRUE state of the
# propagation (states table of src/montecarlo/propagation.py) plus the detector resolution.
# The system is 2D (x, z): each plane measures only the position x of the track,
#   x_meas = x_true + N(0, sigma_x)
# so for the Kalman filter each hit is a scalar measurement with H = [1, 0, 0] on (x, tx, q/p)
# and variance V = sigma_x**2.
#
# Only what the detector measures is kept: the truth stays in the propagation dataset and is
# joined by (track_id, plane_index) to validate (residuals, pulls). The z of the hit is the z of
# the plane, which is known (it is not measured).
#
# Random numbers: each track has its own stream, from the seed of the hits and the row of the track
# in the initial beam dataset (input_row), as the scattering kicks of the propagation. The hits of a
# track do not depend on the other tracks, on their order or on how many there are.
#
# Output, two tables:
#   hits:   one row per (track, plane reached), in beam order
#       track_id, input_row, event_id, plane_index, plane, z [mm], x_meas [mm], sigma_x [mm]
#   tracks: one row per track
#       track_id, input_row, event_id, particle, n_hits
#
# Units: mm.

import numpy as np
import pandas as pd

from src.utils.validation import check_real

# Label of the random stream of the hits (the propagation uses 0 and 1, see propagation.py)
STREAM_HITS = 2

HIT_COLUMNS = ["track_id", "input_row", "event_id", "plane_index", "plane", "z", "x_meas", "sigma_x"]
TRACK_COLUMNS = ["track_id", "input_row", "event_id", "particle", "n_hits"]


def hit_rng(seed, input_row):
    # numpy Generator for the measurement errors of the track in row input_row of the initial beam
    return np.random.default_rng([seed, STREAM_HITS, int(input_row)])


# states: states table of a propagation (one row per track and plane reached, grouped by track and
#         in beam order inside each track, as propagate_tracks writes it)
# tracks: tracks table of the same propagation (for the particle and the tracks with no plane)
# sigma_x: [mm] resolution of every plane (> 0)
# seed: integer seed of the measurement errors (hit_rng)
# Returns: (hits DataFrame, tracks DataFrame), see the header
def make_hits(states, tracks, sigma_x, seed):

    sigma_x = check_real("sigma_x", sigma_x, minimum=0.0, strict=True)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer, got {seed!r}")

    track_id = states["track_id"].to_numpy()
    if np.any(np.diff(track_id) < 0):
        raise ValueError("states must be grouped by track_id in increasing order (as propagate_tracks writes it)")

    # Rows of each track: one block of consecutive rows per track_id
    starts = np.flatnonzero(np.concatenate(([True], np.diff(track_id) != 0))) if len(states) else np.array([], int)
    ends = np.append(starts[1:], len(states))
    input_rows = states["input_row"].to_numpy()

    noise = np.empty(len(states))
    for a, b in zip(starts, ends):
        noise[a:b] = hit_rng(seed, input_rows[a]).normal(0.0, sigma_x, size=b - a)

    hits = pd.DataFrame({
        "track_id": states["track_id"].to_numpy(),
        "input_row": input_rows,
        "event_id": states["event_id"].to_numpy(),
        "plane_index": states["plane_index"].to_numpy(),
        "plane": states["plane"].to_numpy(),
        "z": states["z"].to_numpy(),
        "x_meas": states["x"].to_numpy() + noise,
        "sigma_x": np.full(len(states), sigma_x),
    }, columns=HIT_COLUMNS)

    # One row per track of the propagation, also those with no hit (n_hits = 0)
    n_hits = hits.groupby("track_id").size()
    tracks_out = tracks[["track_id", "input_row", "event_id", "particle"]].copy()
    tracks_out["n_hits"] = tracks_out["track_id"].map(n_hits).fillna(0).astype(int)

    return hits, tracks_out[TRACK_COLUMNS]

# src/propagator/multiprocess_propagation.py
#
# Parallel version of src/montecarlo/propagation.py:propagate_tracks: the tracks are split into
# contiguous chunks and each one is propagated with propagate_tracks, in several processes
# (src/utils/parallel.py).
#
# The result is EXACTLY the same as the serial one: each track has its own random stream
# (track_rng(seed, input_row)), so its kicks do not depend on which process propagates it, and the
# chunks are joined back in their original order (track_id renumbered as in the serial run).
#
# Windows starts the workers by importing the main script again: the script that calls this must
# be protected by  if __name__ == "__main__":  (scripts/generate_propagation.py is).

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.montecarlo.propagation import propagate_tracks
from src.utils.parallel import chunk_bounds, resolve_workers, run_chunks


# Same arguments and output as propagate_tracks, plus
# n_workers: number of processes (None: half the CPUs; 1: serial, in this process)
# progress: show a tqdm progress bar (in tracks). In parallel it advances one chunk at a time
def propagate_tracks_parallel(df_tracks, propagator, planes, seed, z_end=None, n_workers=None,
                              progress=True):

    n_tracks = len(df_tracks)
    n_workers = min(resolve_workers(n_workers), max(1, n_tracks))
    if z_end is None:
        z_end = propagator.geometry.z_end_system

    # input_row identifies the random stream of each track: without it, propagate_tracks would use
    # the row number inside the chunk, so it is fixed here to the row number in df_tracks
    if "input_row" not in df_tracks:
        df_tracks = df_tracks.copy()
        df_tracks.insert(0, "input_row", np.arange(n_tracks))

    bounds = chunk_bounds(n_tracks, n_workers)
    chunks = [df_tracks.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    with tqdm(total=n_tracks, unit="track", disable=not progress, dynamic_ncols=True,
              desc="Propagating" + (f" ({n_workers} processes)" if n_workers > 1 else "")) as bar:
        results = run_chunks(propagate_tracks, chunks, n_workers,
                             dict(propagator=propagator, planes=planes, seed=seed, z_end=z_end),
                             progress=bar.update)

    # Join in chunk order; track_id inside each chunk starts at 0, so shift it to its global value
    states, tracks = [], []
    for start, (chunk_states, chunk_tracks) in zip(bounds[:-1].tolist(), results):
        chunk_states["track_id"] += start
        chunk_tracks["track_id"] += start
        states.append(chunk_states)
        tracks.append(chunk_tracks)
    return pd.concat(states, ignore_index=True), pd.concat(tracks, ignore_index=True)

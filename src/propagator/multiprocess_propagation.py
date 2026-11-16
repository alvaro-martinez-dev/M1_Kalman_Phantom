# src/propagator/multiprocess_propagation.py
#
# Parallel version of src/montecarlo/propagation.py:propagate_tracks, with multiprocessing
# (concurrent.futures.ProcessPoolExecutor). The tracks are split into contiguous chunks and each
# worker process runs propagate_tracks on its chunks.
#
# The result is EXACTLY the same as the serial one: each track has its own random stream
# (track_rng(seed, input_row)), so its kicks do not depend on which process propagates it, and the
# chunks are joined back in their original order (track_id renumbered as in the serial run).
#
# Windows starts the workers by importing the main script again: the script that calls this must
# be protected by  if __name__ == "__main__":  (scripts/generate_propagation.py is).

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.montecarlo.propagation import propagate_tracks
from src.utils.validation import check_count

# Chunks per worker: more chunks balance the load better (tracks that stop in the phantom are
# faster) and give a finer progress, at a small cost per chunk
CHUNKS_PER_WORKER = 8
# Chunks of the serial run (only for the progress bar; the cost per chunk is negligible)
SERIAL_CHUNKS = 200


def resolve_workers(n_workers):
    # n_workers: integer >= 1, or None for half the logical CPUs (~ the physical cores: with
    # hyperthreading, more processes than cores is slower). Returns the number of processes.
    # Each process costs a few seconds to start (imports scipy/pandas): with few tracks, use 1
    if n_workers is None:
        return max(1, (os.cpu_count() or 2) // 2)
    n_workers = check_count("n_workers", n_workers)
    if n_workers == 0:
        raise ValueError("n_workers must be >= 1 (or null for half the CPUs)")
    return n_workers


# Each worker receives the propagator (~MB with an interpolated field) once, not once per chunk
_WORKER_ARGS = {}


def _init_worker(propagator, planes, seed, z_end):
    _WORKER_ARGS.update(propagator=propagator, planes=planes, seed=seed, z_end=z_end)


def _run_chunk(df_chunk):
    return propagate_tracks(df_chunk, **_WORKER_ARGS)


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

    # Serial: small chunks, only so that the bar advances; parallel: CHUNKS_PER_WORKER per process
    n_chunks = SERIAL_CHUNKS if n_workers == 1 else n_workers * CHUNKS_PER_WORKER
    bounds = np.linspace(0, n_tracks, min(n_tracks, n_chunks) + 1).astype(int)
    starts = bounds[:-1].tolist()
    chunks = [df_tracks.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    results = [None] * len(chunks)
    bar = tqdm(total=n_tracks, unit="track", disable=not progress, dynamic_ncols=True,
               desc="Propagating" + (f" ({n_workers} processes)" if n_workers > 1 else ""))
    with bar:
        if n_workers == 1:
            for k, chunk in enumerate(chunks):
                results[k] = propagate_tracks(chunk, propagator, planes, seed, z_end=z_end)
                bar.update(len(chunk))
        else:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                     initargs=(propagator, planes, seed, z_end)) as pool:
                futures = {pool.submit(_run_chunk, chunk): k for k, chunk in enumerate(chunks)}
                for future in as_completed(futures):
                    k = futures[future]
                    results[k] = future.result()
                    bar.update(len(chunks[k]))

    # Join in chunk order; track_id inside each chunk starts at 0, so shift it to its global value
    states, tracks = [], []
    for start, (chunk_states, chunk_tracks) in zip(starts, results):
        chunk_states["track_id"] += start
        chunk_tracks["track_id"] += start
        states.append(chunk_states)
        tracks.append(chunk_tracks)
    return pd.concat(states, ignore_index=True), pd.concat(tracks, ignore_index=True)

# src/utils/parallel.py
#
# Runs a function over chunks of independent items (tracks), serially or in parallel processes
# (concurrent.futures.ProcessPoolExecutor). Used by the propagation (src/propagator/
# multiprocess_propagation.py), the Kalman fit (src/kalman/fit.py) and the depth-dose
# (src/phantom/depth_dose.py).
#
# The result does not depend on the number of processes: each item has its own random stream (e.g.
# track_rng(seed, input_row)), and the results are returned in the order of the chunks.
#
# Each worker receives the shared arguments (e.g. a propagator with its field map, ~MB) once, when it
# starts, not once per chunk. func must be a module-level function (it is pickled to the workers).
#
# Windows starts the workers by importing the main script again: the script that calls this must be
# protected by  if __name__ == "__main__":  (the scripts of scripts/ are).

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.utils.validation import check_count

# Chunks per worker: more chunks balance the load better (tracks that stop in the phantom are faster)
# and give a finer progress, at a small cost per chunk
CHUNKS_PER_WORKER = 8
# Chunks of a serial run (only for the progress bar; the cost per chunk is negligible)
SERIAL_CHUNKS = 200


def resolve_workers(n_workers):
    # n_workers: integer >= 1, or None for half the logical CPUs (~ the physical cores: with
    # hyperthreading, more processes than cores is slower). Returns the number of processes,
    # never more than the logical CPUs of this computer (the yaml may come from another one).
    # Each process costs a few seconds to start (imports scipy/pandas): with few tracks, use 1
    n_cpus = os.cpu_count() or 1    # cpu_count() can return None if it cannot be determined
    if n_workers is None:
        return max(1, n_cpus // 2)
    n_workers = check_count("n_workers", n_workers)
    if n_workers == 0:
        raise ValueError("n_workers must be >= 1 (or null for half the CPUs)")
    if n_workers > n_cpus:
        warnings.warn(f"n_workers = {n_workers} but this computer has {n_cpus} logical CPUs: "
                      f"using {n_cpus} processes (null uses half of them, usually the fastest)",
                      stacklevel=2)
        return n_cpus
    return n_workers


# n_items: number of items to split; n_workers: processes (resolve_workers)
# Returns: (n_chunks + 1,) bounds of contiguous chunks of items, chunk k = items[bounds[k]:bounds[k+1]]
def chunk_bounds(n_items, n_workers):
    n_chunks = SERIAL_CHUNKS if n_workers == 1 else n_workers * CHUNKS_PER_WORKER
    return np.linspace(0, n_items, max(1, min(n_items, n_chunks)) + 1).astype(int)


_WORKER = {}


def _init_worker(func, shared):
    _WORKER.update(func=func, shared=shared)


def _run_chunk(chunk):
    return _WORKER["func"](chunk, **_WORKER["shared"])


# func: module-level function, called as func(chunk, **shared)
# chunks: list of chunks (e.g. DataFrames or arrays of rows)
# n_workers: processes (1: serial, in this process)
# shared: dict of keyword arguments of func, the same for every chunk
# progress: callable(n) called after each chunk with its size (e.g. a tqdm update), or None
# sizes: size of each chunk for progress (default: len of each chunk)
# Returns: list of the results of func, in the order of the chunks
def run_chunks(func, chunks, n_workers, shared, progress=None, sizes=None):
    sizes = [len(chunk) for chunk in chunks] if sizes is None else list(sizes)
    results = [None] * len(chunks)
    if n_workers == 1:
        for k, chunk in enumerate(chunks):
            results[k] = func(chunk, **shared)
            if progress:
                progress(sizes[k])
        return results

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(func, shared)) as pool:
        futures = {pool.submit(_run_chunk, chunk): k for k, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            k = futures[future]
            results[k] = future.result()
            if progress:
                progress(sizes[k])
    return results

# src/montecarlo/propagation.py
#
# Monte Carlo of the full propagation: each track of a track table (see tracks.py) is propagated
# through the system with the scattering kicks, and its TRUE state is stored at every detector
# plane. The detector resolution is not applied here (see the hits stage): the propagation is
# slow and the smearing is not, so several resolutions can be tried on the same propagation.
#
# Output, two tables:
#   states: one row per (track, plane reached), in beam order
#       track_id, event_id, plane_index, plane, z [mm], x [mm], tx, q_over_p [e/MeV], T [MeV]
#   tracks: one row per track
#       track_id, event_id, particle, stopped, z_final, x_final, tx_final, q_over_p_final, T_final,
#       n_planes (planes reached)
# A track that stops in the phantom has no rows in the planes after its stopping point (so the
# absorption shows up as fewer tracks per plane), and tracks.stopped is True.
#
# Units: mm, MeV; q in e.

import time

import numpy as np
import pandas as pd

from src.geometry.geometry import TOL
from src.propagator.propagator import State, kinetic_energy


# detectors: DetectorPlanes
# Returns: list of (name, z) of the distinct planes in beam order. Planes closer than TOL are one
#          plane, whose name joins all of them with "+" (e.g. D_phantom_end+D_magnet_start)
def plane_list(detectors):

    merged = []
    for name, z in sorted(detectors.named_planes().items(), key=lambda item: item[1]):
        if merged and z - merged[-1][1] <= TOL:
            merged[-1] = (f"{merged[-1][0]}+{name}", merged[-1][1])
        else:
            merged.append((name, z))
    return merged


# df_tracks: track table (tracks.py), all of the particle of the propagator
# propagator: Propagator (built with max_slice for scattering)
# planes: list of (name, z) of the detector planes (plane_list)
# rng: numpy Generator for the scattering kicks, or None for the deterministic mean tracks
# z_end: [mm] where the propagation ends (default: the end of the system)
# progress_every: print the progress every this many tracks (None: silent)
# Returns: (states DataFrame, tracks DataFrame), see the header
def propagate_tracks(df_tracks, propagator, planes, rng, z_end=None, progress_every=None):

    particle = propagator.particle
    if particle.charge == 0:
        raise ValueError(f"{particle.name} is neutral: it leaves no hits in the detector planes "
                         "(e.g. decay it first)")
    if not (df_tracks["particle"] == particle.name).all():
        raise ValueError(f"df_tracks must contain only {particle.name} (the particle of the propagator)")
    if (df_tracks["pz0"] <= 0).any():
        raise ValueError("Tracks going backwards (pz0 <= 0) cannot be propagated")
    if z_end is None:
        z_end = propagator.geometry.z_end_system

    plane_names = [name for name, _ in planes]
    plane_z = np.array([z for _, z in planes])

    state_rows, track_rows = [], []
    t_start = time.perf_counter()
    for track_id, row in enumerate(df_tracks.itertuples(index=False)):
        start = State(z=float(row.z0), x=float(row.x0), tx=float(row.tx0), q_p=float(row.q_over_p0))
        ahead = np.flatnonzero((plane_z > start.z + TOL) & (plane_z <= z_end + TOL))   # planes after the start
        result = propagator.propagate(start, z_end, z_out=plane_z[ahead].tolist(), rng=rng)

        n_planes = 0
        for k, s in zip(ahead, result.at_z_out):
            if s is None:             # stopped before this plane
                break
            state_rows.append((track_id, row.event_id, int(k), plane_names[k], s.z, s.x, s.tx, s.q_p,
                               kinetic_energy(s.q_p, particle)))
            n_planes += 1

        f = result.final
        track_rows.append((track_id, row.event_id, particle.name, result.stopped, f.z, f.x, f.tx, f.q_p,
                           kinetic_energy(f.q_p, particle), n_planes))

        if progress_every and (track_id + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t_start
            left = elapsed / (track_id + 1) * (len(df_tracks) - track_id - 1)
            print(f"  {track_id + 1}/{len(df_tracks)} tracks, {elapsed:.0f} s, ~{left:.0f} s left", flush=True)

    states = pd.DataFrame(state_rows, columns=["track_id", "event_id", "plane_index", "plane", "z", "x", "tx",
                                               "q_over_p", "T"])
    tracks = pd.DataFrame(track_rows, columns=["track_id", "event_id", "particle", "stopped", "z_final",
                                               "x_final", "tx_final", "q_over_p_final", "T_final", "n_planes"])
    return states, tracks

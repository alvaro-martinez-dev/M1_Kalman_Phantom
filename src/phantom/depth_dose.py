# src/phantom/depth_dose.py
#
# Depth-dose curve (Bragg curve) and lateral dose from the Monte Carlo: the tracks of a propagation
# dataset are propagated AGAIN through the phantom with the state on a fine grid of z, and the energy
# each one loses between two nodes of the grid is deposited in that bin.
#
# Why again and not stored: the propagation keeps the state only at the detector planes (four of them
# in the phantom), far too coarse for a peak a few mm wide, and the state on a fine grid would be tens
# of MB per dataset. Each track has its own random stream (track_rng(seed, input_row)), so it is
# propagated again into EXACTLY the same kicks. The extra output points are extra cuts of the
# fixed-step RK, which change the state by its truncation error (~1e-3 mm, see plotting/plot_tracks.py):
# negligible for the dose. The tracks are propagated only up to the end of the phantom (the kicks are
# given only there, so this does not change them), which skips the magnet.
#
# Deposit of one track, with T_i the kinetic energy at the node z_i:
#   - bin [z_i, z_{i+1}] crossed: T_i - T_{i+1}, at the mean x of the track in the bin
#   - bin where it stops: ALL its remaining energy T_i (also the < E_MIN_MEV the propagator does not
#     follow), so a track that stops deposits exactly its energy at the entrance of the phantom
# CSDA, as the propagator: no straggling, no nuclear interactions, no secondary particles.
#
# Dose: per bin of depth, the energy over the mass per unit area of the bin, per proton:
#   D_i = sum(E_dep,i) / N / (rho_i dz_i)      [Gy per proton/cm^2]
# the dose of a broad beam of unit fluence, the same quantity as src.phantom.dose.dose_profile, so
# the two can be compared directly. The grid always has a node at each interface, so no bin mixes
# two materials.
#
# Units: mm, MeV; dose in Gy per (proton/cm^2).

from dataclasses import dataclass

import numpy as np

from src.geometry.geometry import TOL
from src.montecarlo.propagation import start_state, track_rng
from src.phantom.dose import MEV_PER_G_TO_GY, energy_from_range
from src.phantom.materials import E_MIN_MEV, material_object, range_csda_fast
from src.propagator.propagator import kinetic_energy
from src.utils.parallel import chunk_bounds, resolve_workers, run_chunks


@dataclass
class DepthDose:
    z_edges: np.ndarray        # (n_bins + 1,) [mm] edges of the depth bins (nodes at every interface)
    density: np.ndarray        # (n_bins,) [g/cm^3] material of each bin
    energy: np.ndarray         # (n_bins,) [MeV] energy deposited in each bin, summed over the tracks
    n_tracks: int              # tracks propagated
    T_entrance: np.ndarray     # (n_tracks,) [MeV] kinetic energy at the entrance of the phantom
    z_stop: np.ndarray         # (n_tracks,) [mm] where each track stops, NaN if it crosses the phantom
    bin_index: np.ndarray      # (n_deposits,) depth bin of each deposit
    x_mid: np.ndarray          # (n_deposits,) [mm] mean x of the track in that bin
    dE: np.ndarray             # (n_deposits,) [MeV] energy of that deposit

    @property
    def dose(self):
        # [Gy per proton/cm^2] mean dose of each bin (see the header)
        mass_per_area = self.density * np.diff(self.z_edges) * 0.1          # g/cm^2 (mm -> cm)
        return self.energy / self.n_tracks / mass_per_area * MEV_PER_G_TO_GY

    @property
    def z_mid(self):
        return 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    # (x_edges, energy[n_bins, n_x]) [MeV per proton] of the (z, x) map, x binned with x_edges
    def energy_map(self, x_edges):
        energy, _, _ = np.histogram2d(self.bin_index, self.x_mid, weights=self.dE,
                                      bins=[np.arange(len(self.z_edges)), x_edges])
        return energy / self.n_tracks


# geometry: Geometry with a phantom; dz: [mm] target width of the depth bins
# Returns: edges from the entrance to the exit of the phantom, every dz inside each layer, with a node
#          at every interface (each layer is cut into the fewest equal bins not wider than dz)
def depth_grid(geometry, dz):
    if not geometry.has_phantom:
        raise ValueError("The depth-dose needs a phantom: the geometry has none (phantom disabled)")
    if not dz > 0:
        raise ValueError(f"dz must be > 0, got {dz}")
    edges = [geometry.z_phantom_interfaces[0]]
    for z_a, z_b in zip(geometry.z_phantom_interfaces[:-1], geometry.z_phantom_interfaces[1:]):
        n = max(1, int(np.ceil((z_b - z_a) / dz - 1e-9)))
        edges += np.linspace(z_a, z_b, n + 1)[1:].tolist()
    return np.array(edges)


# Density of the material of each bin of a depth grid
def bin_densities(geometry, z_edges):
    return np.array([material_object(geometry.material_at(0.5 * (a + b))).density
                     for a, b in zip(z_edges[:-1], z_edges[1:])])


# One track propagated again through the phantom on the grid z_edges
# input_track: row of the initial beam dataset; input_row: its row there (its random stream)
# kick_seed: seed of the kicks of the propagation (None: no scattering, the mean track)
# Returns: (T at the entrance [MeV], z where it stops or NaN, bin indices, mean x in each bin, dE [MeV])
def track_deposits(propagator, input_track, input_row, kick_seed, z_edges):
    start = start_state(input_track)
    rng = track_rng(kick_seed, input_row) if (kick_seed is not None and propagator.max_slice is not None) else None
    if start.z > z_edges[0] + TOL:
        raise ValueError(f"The track starts at z = {start.z} mm, inside the phantom (it starts at {z_edges[0]} mm)")
    result = propagator.propagate(start, float(z_edges[-1]), z_out=z_edges.tolist(), rng=rng)

    # nodes reached, in order (a prefix: None after the stop). A phantom that starts where the track
    # starts has no output point there (z_out must be after the start): the start state is that node
    at_nodes = list(result.at_z_out)
    if at_nodes[0] is None and abs(start.z - z_edges[0]) <= TOL:
        at_nodes[0] = start
    reached = []
    for s in at_nodes:
        if s is None:
            break
        reached.append(s)
    T = np.array([kinetic_energy(s.q_p, propagator.particle) for s in reached])
    x = np.array([s.x for s in reached])
    n = len(reached)
    if n == 0:                                                   # stopped before the phantom: impossible
        return np.nan, float(result.final.z), np.array([], int), np.array([]), np.array([])

    index = np.arange(n - 1)
    x_mid = 0.5 * (x[:-1] + x[1:])
    dE = T[:-1] - T[1:]
    z_stop = np.nan
    if result.stopped:
        # it stops in the bin after its last node reached: all its remaining energy goes there. The
        # stop can be a few um past the next node (stop margin of the propagator): still this bin
        k = min(n - 1, len(z_edges) - 2)
        z_stop = float(result.final.z)
        index = np.append(index, k)
        x_mid = np.append(x_mid, 0.5 * (x[-1] + result.final.x))
        dE = np.append(dE, T[-1])
    return float(T[0]), z_stop, index, x_mid, dE


# Deposits of a list of tracks (one chunk, src/utils/parallel.py); args as track_deposits
def _deposits_of(input_rows, propagator, input_tracks, kick_seed, z_edges):
    out = [track_deposits(propagator, input_tracks.iloc[int(r)], int(r), kick_seed, z_edges) for r in input_rows]
    T0 = np.array([o[0] for o in out])
    z_stop = np.array([o[1] for o in out])
    index = np.concatenate([o[2] for o in out]).astype(np.int32)
    x_mid = np.concatenate([o[3] for o in out]).astype(np.float32)
    dE = np.concatenate([o[4] for o in out])
    return T0, z_stop, index, x_mid, dE


# propagator: Propagator of the propagation (plotting.plot_tracks.rebuild_system)
# input_tracks: initial beam dataset of the propagation; input_rows: rows of the tracks to use
# kick_seed: seed of the propagation (its metadata "seed"), None for no scattering
# dz: [mm] width of the depth bins; n_workers: processes (None: half the CPUs, 1: serial)
# progress: callable(n) called with the tracks done after each chunk (e.g. a tqdm update), or None
# Returns: DepthDose. The result does not depend on n_workers (each track has its own stream)
def depth_dose(propagator, input_tracks, input_rows, kick_seed, dz, n_workers=None, progress=None):
    geometry = propagator.geometry
    z_edges = depth_grid(geometry, dz)
    input_rows = np.asarray(input_rows, dtype=int)
    n_tracks = len(input_rows)
    if n_tracks == 0:
        raise ValueError("No tracks to propagate")
    n_workers = min(resolve_workers(n_workers), n_tracks)

    bounds = chunk_bounds(n_tracks, n_workers)
    chunks = [input_rows[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    results = run_chunks(_deposits_of, chunks, n_workers,
                         dict(propagator=propagator, input_tracks=input_tracks, kick_seed=kick_seed,
                              z_edges=z_edges), progress=progress)

    T0, z_stop, index, x_mid, dE = (np.concatenate([r[j] for r in results]) for j in range(5))
    energy = np.bincount(index, weights=dE, minlength=len(z_edges) - 1)
    return DepthDose(z_edges=z_edges, density=bin_densities(geometry, z_edges), energy=energy,
                     n_tracks=n_tracks, T_entrance=T0, z_stop=z_stop, bin_index=index, x_mid=x_mid, dE=dE)


# z [mm] where a particle of kinetic energy T0 entering the phantom stops in the CSDA, along z (the
# range of each layer in turn), or None if it crosses the whole phantom
def csda_stop_depth(geometry, T0, particle):
    T = float(T0)
    interfaces = geometry.z_phantom_interfaces
    for layer, z_a, z_b in zip(geometry.phantom_layers, interfaces[:-1], interfaces[1:]):
        material = material_object(layer["material"])
        R = range_csda_fast(T, material, particle) if T > E_MIN_MEV else 0.0
        if R <= z_b - z_a:
            return float(z_a + R)
        T = energy_from_range(R - (z_b - z_a), material, particle, T)
    return None

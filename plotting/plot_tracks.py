# plotting/plot_tracks.py
#
# A few tracks of a propagation dataset (data/propagation/, see scripts/generate_propagation.py)
# drawn on the schematic of its system, in the (z, x) plane: the full trajectory as a line and its
# true state at each detector plane as a dot. Written by generate_propagation if run.plot is true.
#
# Along each track, symbols show the field it crosses, B_y(x, z) of the model (zero outside the
# gap, as in the propagator): ⊙ for B_y > 0, out of the page (z to the right and x up give
# z x x = +y towards the reader), ⊗ for B_y < 0, into the page, with size growing with |B_y|.
# Points with |B_y| below FIELD_MIN_FRACTION of B0 are not marked.
#
# Everything is rebuilt from the metadata of the dataset (geometry, detector planes, field,
# propagator), so the tracks are always drawn on the system they were propagated in: no layout
# folder is needed (the layout it corresponds to is recorded in the metadata, layout_name).
#
# The dataset only stores the states at the planes, so the chosen tracks are propagated again
# with many output points. Each track has its own random stream (src/montecarlo/propagation.py),
# so this is exactly the same track as in the dataset (checked at the planes), not a new one.
#
# Usage (from the repo root), to redo the plot of an existing dataset:
#   python -m plotting.plot_tracks propagation/proton_n100000_seed42_prop_seed45_n1000
#   python -m plotting.plot_tracks <propagation> --n-tracks 7 --seed 3 --dz 1.0 --out tracks.pdf

import argparse
import warnings
from pathlib import Path

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from plotting.plot_layout import COLORS_MATERIAL, COLOR_TEXT, LEGEND_COLUMNS, MARGIN_RATIOS, PAGE_SIZE, draw_layout
from src.field import build_field
from src.geometry.geometry import TOL
from src.geometry.layout import layout_table
from src.montecarlo.propagation import start_state, track_rng
from src.propagator.propagator import kinetic_energy
from src.utils.config_loader import make_detector_planes, make_geometry, make_propagator, particle_by_name
from src.utils.dataset_io import dataset_path, load_dataset, load_table


REPORT_NAME = "tracks_on_layout.pdf"
# Categorical slots in fixed order (same palette as the layout): one per track, never cycled.
# The orange slot is left out: it is the magnet, and a track in it reads as the plates
COLORS_TRACKS = list(COLORS_MATERIAL)
MAX_TRACKS = len(COLORS_TRACKS)
COLOR_SURFACE = "#fcfcfb"
LINE_WIDTH = 2.0
HALO_WIDTH = 4.0            # surface ring around each line, separates it from the fills and other lines
MARKER_SIZE = 5.0
TOL_PLANES = 1e-6           # [mm] allowed difference between the redrawn track and the dataset at the planes

# Field symbols along the tracks
FIELD_DZ = 30.0             # [mm] spacing of the symbols along z
FIELD_CLEAR_OF_PLANES = 10.0  # [mm] no symbol this close to a plane (its dot is there)
FIELD_MIN_FRACTION = 0.05   # |B_y| / |B0| below which no symbol is drawn
FIELD_SIZE_MIN, FIELD_SIZE_MAX = 4.0, 9.0   # marker size at FIELD_MIN_FRACTION and at |B_y| >= |B0|
FIELD_INNER_RATIO = 0.55    # size of the inner dot / cross relative to the circle
FIELD_EDGE_WIDTH = 0.9


# prop_metadata: metadata.yml of the propagation dataset
# Returns: (Propagator, layout table) of the system it was generated with
def rebuild_system(prop_metadata):

    cfg = {"geometry": prop_metadata["geometry"], "propagator": prop_metadata["propagator"]}
    geometry = make_geometry(cfg["geometry"])
    detectors = make_detector_planes(geometry, cfg["geometry"]["detector_planes"])
    propagator = make_propagator(cfg, geometry, build_field(cfg, geometry), particle_by_name(prop_metadata["particle"]))
    return propagator, layout_table(geometry, detectors)


# Propagates again one track of the dataset with output every dz [mm] (and at the planes)
# Returns: (z, x) arrays of the trajectory, from its start to the end of the system or where it stops
def dense_track(propagator, input_track, input_row, kick_seed, plane_z, dz):

    start = start_state(input_track)
    z_end = propagator.geometry.z_end_system
    z_out = np.union1d(np.arange(start.z + dz, z_end, dz), plane_z[plane_z > start.z + TOL])
    rng = track_rng(kick_seed, input_row) if propagator.max_slice is not None else None
    result = propagator.propagate(start, z_end, z_out=z_out.tolist(), rng=rng)

    states = [start] + [s for s in result.at_z_out if s is not None] + [result.final]
    return np.array([s.z for s in states]), np.array([s.x for s in states])


# Draws one ⊙ (sign > 0) or ⊗ (sign < 0) of marker size `size` at (z, x)
def draw_field_symbol(ax, z, x, sign, size, zorder=6):

    ax.plot(z, x, linestyle="none", marker="o", markersize=size, markerfacecolor=COLOR_SURFACE,
            markeredgecolor=COLOR_TEXT, markeredgewidth=FIELD_EDGE_WIDTH, zorder=zorder)
    inner = dict(marker=".", markersize=size * FIELD_INNER_RATIO * 0.9) if sign > 0 else \
        dict(marker="x", markersize=size * FIELD_INNER_RATIO, markeredgewidth=FIELD_EDGE_WIDTH)
    ax.plot(z, x, linestyle="none", color=COLOR_TEXT, zorder=zorder + 0.1, **inner)


# Field symbols along one trajectory (z, x): B_y of the model where the track is, every FIELD_DZ
def draw_field_along(ax, propagator, z, x, plane_z, B0):

    z_marks = np.arange(z[0] + FIELD_DZ / 2, z[-1], FIELD_DZ)
    z_marks = z_marks[np.min(np.abs(z_marks[:, None] - plane_z[None, :]), axis=1) > FIELD_CLEAR_OF_PLANES]
    for zm in z_marks:
        xm = float(np.interp(zm, z, x))
        By = propagator.field.By(xm, zm) if abs(xm) <= propagator.x_half_gap else 0.0
        fraction = abs(By) / abs(B0)
        if fraction < FIELD_MIN_FRACTION:
            continue
        size = FIELD_SIZE_MIN + (FIELD_SIZE_MAX - FIELD_SIZE_MIN) * min(1.0, (fraction - FIELD_MIN_FRACTION)
                                                                        / (1.0 - FIELD_MIN_FRACTION))
        draw_field_symbol(ax, zm, xm, np.sign(By), size)


# Legend handles of the field symbols
def field_legend_handles():

    out = []
    for sign, label in ((+1, "B_y > 0, out of the page"), (-1, "B_y < 0, into the page")):
        circle = Line2D([], [], linestyle="none", marker="o", markersize=FIELD_SIZE_MAX,
                        markerfacecolor=COLOR_SURFACE, markeredgecolor=COLOR_TEXT, markeredgewidth=FIELD_EDGE_WIDTH)
        inner = Line2D([], [], linestyle="none", color=COLOR_TEXT,
                       **(dict(marker=".", markersize=FIELD_SIZE_MAX * FIELD_INNER_RATIO * 0.9) if sign > 0 else
                          dict(marker="x", markersize=FIELD_SIZE_MAX * FIELD_INNER_RATIO,
                               markeredgewidth=FIELD_EDGE_WIDTH)))
        out.append(((circle, inner), f"{label} (size ~ |B_y|)"))
    return out


# prop_path: propagation dataset folder (absolute, or relative to data/)
# n_tracks: tracks to draw, chosen at random with seed; dz: [mm] spacing of the drawn points
# out_path: output pdf (default: REPORT_NAME in the dataset folder)
# Returns: out_path as a Path
def plot_tracks(prop_path, n_tracks=5, seed=0, dz=2.0, out_path=None):

    if not 1 <= n_tracks <= MAX_TRACKS:
        raise ValueError(f"n_tracks must be between 1 and {MAX_TRACKS} (one distinct color per track)")

    # 1. Dataset, its input tracks and its system
    prop_dir = dataset_path(prop_path)
    tracks, prop_metadata = load_dataset(prop_dir)
    states = load_table(prop_dir, "states")
    input_tracks, input_metadata = load_dataset(prop_metadata["input"])
    if input_metadata.get("created") != prop_metadata.get("input_created"):
        warnings.warn(f"{prop_metadata['input']} was regenerated after this propagation: the tracks may differ")
    propagator, table = rebuild_system(prop_metadata)
    particle = propagator.particle
    plane_z = np.array([plane["z"] for plane in prop_metadata["planes"]])
    n_tracks = min(n_tracks, len(tracks))
    magnet = prop_metadata["geometry"]["magnet"]
    B0 = float(magnet["B0"]) if propagator.geometry.has_magnet and magnet["B0"] else None   # None: no symbols

    # 2. Schematic of the system
    fig = Figure(figsize=PAGE_SIZE, layout="constrained")
    fig.suptitle(f"{n_tracks} random tracks of {prop_dir.name}\n"
                 f"on {prop_metadata.get('layout_name', 'its system')}  |  {particle.name}, "
                 f"scattering {'on' if propagator.max_slice is not None else 'off'}",
                 fontsize=11, color=COLOR_TEXT)
    grid = fig.add_gridspec(3, 3, width_ratios=MARGIN_RATIOS, height_ratios=MARGIN_RATIOS)
    ax = fig.add_subplot(grid[1, 1])
    layout_handles = draw_layout(ax, table, prop_metadata)

    # 3. Tracks chosen at random, propagated again with many points
    chosen = tracks.iloc[np.sort(np.random.default_rng(seed).choice(len(tracks), size=n_tracks, replace=False))]
    halo = [path_effects.Stroke(linewidth=HALO_WIDTH, foreground=COLOR_SURFACE), path_effects.Normal()]
    track_handles, x_extent = [], 0.0
    for color, track in zip(COLORS_TRACKS, chosen.itertuples(index=False)):
        input_row = int(track.input_row)
        z, x = dense_track(propagator, input_tracks.iloc[input_row], input_row, prop_metadata["seed"], plane_z, dz)

        # the redrawn track must be the one of the dataset
        at_planes = states[states.input_row == input_row]
        if len(at_planes) and np.max(np.abs(np.interp(at_planes["z"], z, x) - at_planes["x"].to_numpy())) > TOL_PLANES:
            warnings.warn(f"Track {input_row} redrawn differs from the dataset at the planes "
                          "(was the code or the input changed after the propagation?)")

        ax.plot(z, x, color=color, linewidth=LINE_WIDTH, path_effects=halo, zorder=4)
        ax.plot(at_planes["z"], at_planes["x"], linestyle="none", marker="o", markersize=MARKER_SIZE,
                markerfacecolor=color, markeredgecolor=COLOR_SURFACE, markeredgewidth=1.0, zorder=5)
        if B0:
            draw_field_along(ax, propagator, z, x, plane_z, B0)
        T0 =kinetic_energy(input_tracks.iloc[input_row]["q_over_p0"], particle)
        end = "stops" if track.stopped else f"{track.T_final:.0f} MeV"
        track_handles.append(Line2D([], [], color=color, linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE,
                                    label=f"track {input_row}: {T0:.0f} MeV -> {end}"))
        x_extent = max(x_extent, np.max(np.abs(x)))

    # the schematic shows +-2.5 half gaps: widen it if a track goes further
    if x_extent * 1.05 > ax.get_ylim()[1]:
        ax.set_ylim(-x_extent * 1.05, x_extent * 1.05)

    entries = [(h, h.get_label()) for h in track_handles]
    entries.append((Line2D([], [], linestyle="none", marker="o", markersize=MARKER_SIZE, markerfacecolor=COLOR_TEXT,
                           markeredgecolor=COLOR_SURFACE), "true state at a detector plane"))
    if B0:
        entries += field_legend_handles()
    entries += [(h, h.get_label()) for h in layout_handles]
    ax.legend([h for h, _ in entries], [label for _, label in entries],
              handler_map={tuple: HandlerTuple(ndivide=1)},       # ⊙ / ⊗: circle and inner mark overlaid
              loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=min(len(entries), LEGEND_COLUMNS), fontsize=8, frameon=False)

    out_path = Path(out_path) if out_path is not None else prop_dir / REPORT_NAME
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig)
    return out_path


def main():

    parser = argparse.ArgumentParser(description="Draw a few tracks of a propagation dataset on its system.")
    parser.add_argument("propagation", help="propagation dataset folder (absolute, or relative to data/)")
    parser.add_argument("--n-tracks", type=int, default=5, help=f"tracks to draw, chosen at random (1-{MAX_TRACKS})")
    parser.add_argument("--seed", type=int, default=0, help="seed of the choice of the tracks")
    parser.add_argument("--dz", type=float, default=2.0, help="[mm] spacing of the points of each trajectory")
    parser.add_argument("--out", default=None, help=f"output pdf (default: {REPORT_NAME} in the propagation folder)")
    args = parser.parse_args()

    print(f"Saved {plot_tracks(args.propagation, args.n_tracks, args.seed, args.dz, args.out)}")


if __name__ == "__main__":
    main()

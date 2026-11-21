# plotting/plot_hits.py
#
# Validation report of a hits dataset (data/hits/, see scripts/generate_hits.py): a single
# multi-page pdf that compares the hits with the MC truth of their propagation dataset (joined by
# track_id, plane_index; the truth is read from the propagation, it is not in the hits):
#   1. Residuals x_meas - x_true and pulls (x_meas - x_true) / sigma_x of all the hits, against
#      N(0, sigma_x) and N(0, 1), and the residual against x_true (the error must not depend on x)
#   2. Per detector plane: number of hits, mean and std of the residuals, against 0 and sigma_x
#   3. A few tracks: their hits (with ±sigma_x) and their true states at the planes, and their residuals
#
# Usage (from the repo root), to redo the report of an existing dataset:
#   python -m plotting.plot_hits hits/proton_Beamca5fcd_Lay255033_Magn5a08f2_Propb14662_seed45_n1000_hits_sigma0.1mm_seed46

import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.stats import norm

from plotting.plot_init_beam import (CMAP_COUNTS, COLORBAR_ASPECT, COLOR_AXIS, COLOR_DATA, COLOR_EXPECTED,
                                     COLOR_MUTED, COLOR_TEXT, PAGE_SIZE, PANEL_ROWS, expected_panel,
                                     n_bins, stats_title, style_axes)
from plotting.plot_layout import COLORS_MATERIAL
from src.utils.dataset_io import dataset_path, load_dataset, load_table

REPORT_NAME = "report.pdf"

UM = 1e3                    # mm -> um: the residuals are drawn in um
COLORS_TRACKS = list(COLORS_MATERIAL)
N_EXAMPLE_TRACKS = 5        # tracks of page 3 (at most len(COLORS_TRACKS))
MARKER_SIZE = 5.0


# Hits joined with the true x of their propagation dataset
# Returns: DataFrame of the hits with the columns x_true, residual [mm] and pull
def hits_with_truth(hits, metadata):

    states = load_table(metadata["input"], "states")
    merged = hits.merge(states[["track_id", "plane_index", "x"]], on=["track_id", "plane_index"],
                        how="left", validate="one_to_one")
    if merged["x"].isna().any():
        raise ValueError(f"{merged['x'].isna().sum()} hits have no true state in {metadata['input']}: "
                         "the hits do not come from this propagation")
    merged = merged.rename(columns={"x": "x_true"})
    merged["residual"] = merged["x_meas"] - merged["x_true"]
    merged["pull"] = merged["residual"] / merged["sigma_x"]
    return merged


# Page with an empty figure and the header that identifies the dataset
def new_page(title, metadata):

    commit = str(metadata["git_commit"])[:7]
    if metadata["git_uncommitted_changes"]:
        commit += " + uncommitted changes"
    subtitle = (f"{metadata['particle']}  |  {metadata['n_tracks']} tracks, {metadata['n_hits']} hits  |  "
                f"$\\sigma_x$ = {metadata['sigma_x']:g} mm  |  seed {metadata['seed']}  |  "
                f"from {Path(metadata['input']).name}\ncreated {metadata['created']}  |  commit {commit}")

    fig = Figure(figsize=PAGE_SIZE, layout="constrained")
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11, color=COLOR_TEXT)
    return fig


# Histogram of x against N(mean, sigma), with its pulls below (as plot_init_beam)
def gaussian_panel(ax, ax_pull, x, mean, sigma, xlabel):

    edges = np.linspace(min(mean - 4 * sigma, x.min()), max(mean + 4 * sigma, x.max()), n_bins(len(x)) + 1)
    expected = len(x) * np.diff(norm.cdf(edges, mean, sigma))
    chi2, ndof = expected_panel(ax, ax_pull, None, x, edges, expected, xlabel=xlabel)
    ax.set_title(stats_title(x, mean, sigma, chi2, ndof), fontsize=8, color=COLOR_TEXT, loc="left")
    ax.set_ylabel("Hits / bin", fontsize=9, color=COLOR_MUTED)


# Page 1: residuals and pulls of all the hits, and the residual against x_true
def page_residuals(df, metadata):

    fig = new_page("Residuals and pulls of all the hits", metadata)
    grid = fig.add_gridspec(2, 4, height_ratios=PANEL_ROWS, width_ratios=[1, 1, 1, 1 / COLORBAR_ASPECT])
    sigma = metadata["sigma_x"]

    ax = fig.add_subplot(grid[0, 0])
    ax_pull = fig.add_subplot(grid[1, 0], sharex=ax)
    gaussian_panel(ax, ax_pull, df["residual"].to_numpy() * UM, 0.0, sigma * UM,
                   r"$x_\mathrm{meas} - x_\mathrm{true}$ [µm]")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax = fig.add_subplot(grid[0, 1])
    ax_pull = fig.add_subplot(grid[1, 1], sharex=ax)
    gaussian_panel(ax, ax_pull, df["pull"].to_numpy(), 0.0, 1.0,
                   r"$(x_\mathrm{meas} - x_\mathrm{true}) / \sigma_x$")

    # The measurement error must not depend on where the track crosses the plane
    x_true, residual = df["x_true"].to_numpy(), df["residual"].to_numpy() * UM
    ax = fig.add_subplot(grid[:, 2])
    ax.set_box_aspect(1)
    *_, mesh = ax.hist2d(x_true, residual, bins=n_bins(len(df)), cmap=CMAP_COUNTS, cmin=1)
    ax.axhline(0, color=COLOR_EXPECTED, linewidth=1)
    ax.set_title(f"r = {np.corrcoef(x_true, residual)[0, 1]:+.4f} ± {1 / np.sqrt(len(df)):.4f}  (expected 0)",
                 fontsize=8, color=COLOR_TEXT, loc="left")
    ax.set_xlabel(r"$x_\mathrm{true}$ [mm]", fontsize=10, color=COLOR_TEXT)
    ax.set_ylabel(r"$x_\mathrm{meas} - x_\mathrm{true}$ [µm]", fontsize=10, color=COLOR_TEXT)
    style_axes(ax)
    cax = fig.add_subplot(grid[:, 3])
    cax.set_box_aspect(COLORBAR_ASPECT)   # same height as its square panel
    colorbar = fig.colorbar(mesh, cax=cax, label="Hits / bin")
    colorbar.ax.tick_params(labelsize=8, colors=COLOR_MUTED)
    colorbar.ax.yaxis.label.set_size(9)

    return fig


# Statistics of the residuals of each plane
# Returns: dict of arrays, one value per plane of the metadata (n = 0 and nan for a plane with no hit)
def per_plane(df, planes):

    groups = df.groupby("plane_index")["residual"]
    index = np.arange(len(planes))
    n = groups.size().reindex(index, fill_value=0).to_numpy()
    mean = groups.mean().reindex(index).to_numpy()
    std = groups.std().reindex(index).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"n": n, "mean": mean, "std": std,
                "mean_err": std / np.sqrt(n),                  # error of the mean
                "std_err": std / np.sqrt(2 * (n - 1))}         # error of the std (Gaussian)


# chi2/ndof text of values against an expected constant, over the planes with hits
def chi2_text(values, errors, expected):

    used = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    chi2 = np.sum(((values[used] - expected) / errors[used])**2)
    return f"$\\chi^2$/ndof = {chi2:.1f}/{used.sum()} = {chi2 / max(used.sum(), 1):.2f}"


# Page 2: per detector plane, number of hits and mean and std of the residuals
def page_planes(df, metadata):

    fig = new_page("Residuals per detector plane", metadata)
    planes = metadata["planes"]
    sigma = metadata["sigma_x"] * UM
    stats = per_plane(df, planes)
    index = np.arange(len(planes))
    axes = fig.subplots(3, 1, sharex=True)

    # 1. Hits per plane: fewer than tracks after the planes where some tracks stop
    ax = axes[0]
    ax.bar(index, stats["n"], width=0.6, color=COLOR_DATA, label="Hits")
    ax.axhline(metadata["n_tracks"], color=COLOR_EXPECTED, linewidth=1.5, label="Tracks")
    ax.set_ylabel("Hits", fontsize=9, color=COLOR_MUTED)
    ax.set_ylim(0, 1.05 * max(metadata["n_tracks"], stats["n"].max(initial=0)))
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), fontsize=8, frameon=False, ncols=2)

    # 2. Mean residual (bias): 0 within its error
    ax = axes[1]
    ax.errorbar(index, stats["mean"] * UM, yerr=stats["mean_err"] * UM, fmt="o", markersize=MARKER_SIZE,
                color=COLOR_DATA, label="Mean ± error")
    ax.axhline(0, color=COLOR_EXPECTED, linewidth=1.5, label="Expected: 0")
    ax.set_ylabel("Mean residual [µm]", fontsize=9, color=COLOR_MUTED)
    ax.set_title(chi2_text(stats["mean"], stats["mean_err"], 0.0), fontsize=8, color=COLOR_TEXT, loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncols=2)

    # 3. Std of the residuals (resolution): sigma_x within its error
    ax = axes[2]
    ax.errorbar(index, stats["std"] * UM, yerr=stats["std_err"] * UM, fmt="o", markersize=MARKER_SIZE,
                color=COLOR_DATA, label="Std ± error")
    ax.axhline(sigma, color=COLOR_EXPECTED, linewidth=1.5, label=f"Expected: $\\sigma_x$ = {sigma:g} µm")
    ax.set_ylabel("Std of the residuals [µm]", fontsize=9, color=COLOR_MUTED)
    ax.set_title(chi2_text(stats["std"], stats["std_err"], metadata["sigma_x"]),   # all in mm
                 fontsize=8, color=COLOR_TEXT, loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncols=2)

    for ax in axes:
        style_axes(ax)      # before the plane names: it formats the x axis as numbers
        ax.tick_params(axis="x", labelsize=7)
    ax.set_xticks(index, [f"{p['name']}\nz = {p['z']:g}" for p in planes], rotation=30, ha="right")
    return fig


# Page 3: hits and true states of a few tracks chosen at random (seed), and their residuals
def page_examples(df, metadata, n_tracks, seed):

    fig = new_page("Hits of a few tracks", metadata)
    sigma = metadata["sigma_x"]
    ax_x, ax_res = fig.subplots(1, 2, width_ratios=[1.6, 1])

    track_ids = np.unique(df["track_id"])
    n_tracks = min(n_tracks, len(track_ids), len(COLORS_TRACKS))
    chosen = np.sort(np.random.default_rng(seed).choice(track_ids, size=n_tracks, replace=False))
    offsets = np.linspace(-0.2, 0.2, n_tracks) if n_tracks > 1 else [0.0]   # side by side at each plane
    for color, track_id, offset in zip(COLORS_TRACKS, chosen, offsets):
        t = df[df["track_id"] == track_id].sort_values("plane_index")
        # true state at the planes, joined by straight segments (not the trajectory between planes)
        ax_x.plot(t["z"], t["x_true"], color=color, linewidth=1, marker="o", markersize=MARKER_SIZE,
                  markerfacecolor="white", label=f"Track {track_id}")
        ax_x.errorbar(t["z"], t["x_meas"], yerr=t["sigma_x"], fmt="none", color=color, linewidth=1.5)
        ax_x.plot(t["z"], t["x_meas"], linestyle="none", marker="o", markersize=MARKER_SIZE * 0.6, color=color)
        # points only: the errors of different planes are independent, a line would suggest otherwise
        ax_res.plot(t["plane_index"] + offset, t["residual"] * UM, color=color, linestyle="none",
                    marker="o", markersize=MARKER_SIZE, label=f"Track {track_id}")

    ax_x.set_title("○ true state at the plane   ● hit (± $\\sigma_x$)", fontsize=8, color=COLOR_TEXT, loc="left")
    ax_x.set_xlabel("z [mm]", fontsize=10, color=COLOR_TEXT)
    ax_x.set_ylabel("x [mm]", fontsize=10, color=COLOR_TEXT)
    ax_x.legend(loc="best", fontsize=8, frameon=False)

    ax_res.axhspan(-sigma * UM, sigma * UM, color=COLOR_AXIS, alpha=0.4, linewidth=0, label="± $\\sigma_x$")
    ax_res.axhline(0, color=COLOR_EXPECTED, linewidth=1)
    ax_res.set_title("Residuals of these tracks (|r| < $\\sigma_x$ in 68 % of the hits)",
                     fontsize=8, color=COLOR_TEXT, loc="left")
    ax_res.set_ylabel(r"$x_\mathrm{meas} - x_\mathrm{true}$ [µm]", fontsize=10, color=COLOR_TEXT)

    for ax in (ax_x, ax_res):
        style_axes(ax)      # before the plane names: it formats the x axis as numbers
    ax_res.set_xticks(np.arange(len(metadata["planes"])), [p["name"] for p in metadata["planes"]],
                      rotation=30, ha="right")
    ax_res.tick_params(axis="x", labelsize=7)
    return fig


# hits_dir: folder of a hits dataset (absolute, or relative to data/)
# out_path: output pdf (default: REPORT_NAME in the dataset folder)
# n_tracks, seed: number of tracks of page 3 and the seed of their choice
# Returns: out_path as a Path
def plot_hits(hits_dir, out_path=None, n_tracks=N_EXAMPLE_TRACKS, seed=0):

    hits_dir = dataset_path(hits_dir)
    _, metadata = load_dataset(hits_dir)
    df = hits_with_truth(load_table(hits_dir, "hits"), metadata)
    if df.empty:
        raise ValueError(f"{hits_dir} has no hits")

    # All the pages are built first: a failure does not leave a partial pdf behind
    pages = [page_residuals(df, metadata), page_planes(df, metadata),
             page_examples(df, metadata, n_tracks, seed)]

    out_path = Path(out_path) if out_path is not None else hits_dir / REPORT_NAME
    with PdfPages(out_path) as pdf:
        for fig in pages:
            pdf.savefig(fig)
    return out_path


def main():

    parser = argparse.ArgumentParser(description="Redo the report of a hits dataset.")
    parser.add_argument("dataset", help="folder of the dataset (absolute, or relative to data/)")
    parser.add_argument("--n-tracks", type=int, default=N_EXAMPLE_TRACKS,
                        help=f"tracks drawn in page 3 (default {N_EXAMPLE_TRACKS}, at most {len(COLORS_TRACKS)})")
    parser.add_argument("--seed", type=int, default=0, help="seed of the choice of those tracks (default 0)")
    parser.add_argument("--out", default=None, help=f"output pdf (default: {REPORT_NAME} in the dataset folder)")
    args = parser.parse_args()

    print(f"Saved report to {plot_hits(args.dataset, args.out, args.n_tracks, args.seed)}")


if __name__ == "__main__":
    main()

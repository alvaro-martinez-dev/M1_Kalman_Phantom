# plotting/plot_init_beam.py
#
# Validation report of an init_beam dataset: a single multi-page pdf with
#   1. Generated parameters (x0, theta0, p0) against the Gaussians of metadata.yml, with pulls
#   2. Derived quantities of the track table
#   3. Correlations between the generated parameters
#
# Usage (from the repo root), to redo the report of an existing dataset:
#   python -m plotting.plot_init_beam init_beam/proton_n100000_seed42

import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy.stats import norm

from src.utils.dataset_io import DATA_DIR, load_dataset


REPORT_NAME = "report.pdf"

PAGE_SIZE = (11.69, 8.27)   # A4 landscape, in inches
PANEL_ASPECT = 0.75         # height / width of a histogram
PULL_ASPECT = 0.25          # height / width of the pulls below it
COLORBAR_ASPECT = 20        # height / width of a colorbar

COLOR_DATA = "#2a78d6"
COLOR_EXPECTED = "#eb6834"
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_AXIS = "#c3c2b7"
CMAP_COUNTS = LinearSegmentedColormap.from_list(
    "counts", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# Column: (axis label, factor from the units of the track table to the units of the plot)
VARIABLES = {
    "x0":        (r"$x_0$ [mm]", 1),
    "z0":        (r"$z_0$ [mm]", 1),
    "theta0":    (r"$\theta_0$ [mrad]", 1e3),
    "tx0":       (r"$t_{x0}$ [$10^{-3}$]", 1e3),
    "p0":        (r"$p_0$ [MeV]", 1),
    "px0":       (r"$p_{x0}$ [MeV]", 1),
    "pz0":       (r"$p_{z0}$ [MeV]", 1),
    "E0":        (r"$E_0$ [MeV]", 1),
    "T0":        (r"$T_0$ [MeV]", 1),
    "beta0":     (r"$\beta_0$", 1),
    "q_over_p0": (r"$q/p_0$ [e/GeV]", 1e3),
}

GENERATED = ["x0", "theta0", "p0"]
DERIVED = ["z0", "tx0", "px0", "pz0", "E0", "T0", "beta0", "q_over_p0"]
CORRELATIONS = [("x0", "theta0"), ("x0", "p0"), ("theta0", "p0")]

MIN_EXPECTED_FOR_PULL = 5   # bins with fewer expected tracks are left out of the pulls


# Number of histogram bins for n tracks
def n_bins(n):

    return int(np.clip(np.sqrt(n), 10, 60))


# Column of df in the units of the plot
def values(df, column):

    return df[column].to_numpy() * VARIABLES[column][1]


def is_constant(x):

    return np.ptp(x) == 0


def style_axes(ax):

    ax.tick_params(labelsize=8, colors=COLOR_MUTED, length=3)
    ax.ticklabel_format(useOffset=False, axis="x")
    ax.spines[:].set_color(COLOR_AXIS)


# Panel of a column with a single value, where a histogram would say nothing
def constant_panel(ax, column, value):

    ax.set_axis_off()
    ax.text(0.5, 0.5, f"{VARIABLES[column][0]}\nconstant = {value:g}", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color=COLOR_MUTED)


# Page with an empty figure and the header that identifies the dataset
def new_page(title, particle, metadata):

    run = metadata["run"]
    commit = str(metadata["git_commit"])[:7]
    if metadata["git_uncommitted_changes"]:
        commit += " + uncommitted changes"
    subtitle = (f"{particle}  |  {metadata['n_rows']} tracks  |  seed {run['seed']}  |  "
                f"created {metadata['created']}  |  commit {commit}")

    fig = Figure(figsize=PAGE_SIZE, layout="constrained")
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11, color=COLOR_TEXT)

    return fig


# Histogram of a generated parameter and its Gaussian (ax), with the pulls below (ax_pull)
def gaussian_panel(ax, ax_pull, df, column, beam):

    factor = VARIABLES[column][1]
    x = values(df, column)
    n = len(x)
    mean, sigma = beam[column] * factor, beam[f"sigma_{column}"] * factor

    # 1. Observed and expected counts (the range never hides tracks out of the Gaussian)
    edges = np.linspace(min(mean - 4 * sigma, x.min()), max(mean + 4 * sigma, x.max()), n_bins(n) + 1)
    counts, _ = np.histogram(x, edges)
    expected = n * np.diff(norm.cdf(edges, mean, sigma))

    # 2. Pulls
    used = expected >= MIN_EXPECTED_FOR_PULL
    pulls = (counts[used] - expected[used]) / np.sqrt(expected[used])
    chi2, ndof = np.sum(pulls**2), len(pulls)
    centers = (edges[:-1] + edges[1:]) / 2

    # 3. Histogram
    ax.stairs(counts, edges, fill=True, color=COLOR_DATA, label="Generated")
    ax.stairs(expected, edges, color=COLOR_EXPECTED, linewidth=1.5, label="Expected")
    ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
    ax.tick_params(labelbottom=False)
    ax.set_title(f"mean = {x.mean():.3f} ± {x.std() / np.sqrt(n):.3f}  (expected {mean:g})\n"
                 f"std = {x.std():.3f} ± {x.std() / np.sqrt(2 * n):.3f}  (expected {sigma:g})\n"
                 f"$\\chi^2$/ndof = {chi2:.1f}/{ndof} = {chi2 / ndof:.2f}",
                 fontsize=8, color=COLOR_TEXT, loc="left")

    # 4. Pulls
    limit = max(5, np.ceil(np.abs(pulls).max()))
    ax_pull.scatter(centers[used], pulls, s=7, color=COLOR_DATA)
    ax_pull.axhline(0, color=COLOR_EXPECTED, linewidth=1)
    ax_pull.set_ylim(-limit, limit)
    ax_pull.set_yticks([-3, 0, 3])
    ax_pull.set_xlabel(VARIABLES[column][0], fontsize=10, color=COLOR_TEXT)
    ax_pull.set_ylabel("Pull", fontsize=9, color=COLOR_MUTED)

    # 5. Fixed proportions, with the pulls right below the histogram
    ax.set_box_aspect(PANEL_ASPECT)
    ax.set_anchor("S")
    ax_pull.set_box_aspect(PULL_ASPECT)
    ax_pull.set_anchor("N")

    style_axes(ax)
    style_axes(ax_pull)


# Page 1: generated parameters against the Gaussians of the metadata
def page_generated(df, metadata, particle):

    fig = new_page("Generated parameters", particle, metadata)
    grid = fig.add_gridspec(2, len(GENERATED), height_ratios=[1.6, 1])  # centers the panels in the page

    for i, column in enumerate(GENERATED):
        x = values(df, column)
        if is_constant(x):
            constant_panel(fig.add_subplot(grid[:, i]), column, x[0])
            continue
        ax = fig.add_subplot(grid[0, i])
        ax_pull = fig.add_subplot(grid[1, i], sharex=ax)
        gaussian_panel(ax, ax_pull, df, column, metadata["beam"])

    # Legend in the first panel that is not constant
    for ax in fig.axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            break

    return fig


# Page 2: derived quantities of the track table
def page_derived(df, metadata, particle):

    fig = new_page("Derived quantities", particle, metadata)
    axes = fig.subplots(2, 4)

    # The rows stay together: the free space of the page goes above and below them
    for ax in axes[0]:
        ax.set_anchor("S")
    for ax in axes[1]:
        ax.set_anchor("N")

    for ax, column in zip(axes.flat, DERIVED):
        x = values(df, column)
        ax.set_box_aspect(PANEL_ASPECT)
        if is_constant(x):
            constant_panel(ax, column, x[0])
            continue
        ax.stairs(*np.histogram(x, n_bins(len(x))), fill=True, color=COLOR_DATA)
        ax.set_title(f"mean = {x.mean():.4g}, std = {x.std():.3g}", fontsize=8,
                     color=COLOR_TEXT, loc="left")
        ax.set_xlabel(VARIABLES[column][0], fontsize=10, color=COLOR_TEXT)
        ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
        ax.locator_params(axis="x", nbins=5)
        style_axes(ax)

    return fig


# Page 3: pairs of generated parameters, independent by construction
def page_correlations(df, metadata, particle):

    fig = new_page("Correlations between generated parameters", particle, metadata)
    grid = fig.add_gridspec(1, 2 * len(CORRELATIONS), width_ratios=[COLORBAR_ASPECT, 1] * len(CORRELATIONS))

    for i, (col_x, col_y) in enumerate(CORRELATIONS):
        x, y = values(df, col_x), values(df, col_y)
        ax = fig.add_subplot(grid[0, 2 * i])
        ax.set_box_aspect(1)
        if is_constant(x) or is_constant(y):
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{VARIABLES[col_x][0]} vs {VARIABLES[col_y][0]}\n"
                    "not available (constant column)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color=COLOR_MUTED)
            continue
        *_, mesh = ax.hist2d(x, y, bins=n_bins(len(x)), cmap=CMAP_COUNTS, cmin=1)
        ax.set_title(f"r = {np.corrcoef(x, y)[0, 1]:+.4f} ± {1 / np.sqrt(len(x)):.4f}  (expected 0)",
                     fontsize=8, color=COLOR_TEXT, loc="left")
        ax.set_xlabel(VARIABLES[col_x][0], fontsize=10, color=COLOR_TEXT)
        ax.set_ylabel(VARIABLES[col_y][0], fontsize=10, color=COLOR_TEXT)
        style_axes(ax)
        ax.ticklabel_format(useOffset=False, axis="y")
        cax = fig.add_subplot(grid[0, 2 * i + 1])
        cax.set_box_aspect(COLORBAR_ASPECT)   # same height as its square panel
        colorbar = fig.colorbar(mesh, cax=cax, label="Tracks / bin")
        colorbar.ax.tick_params(labelsize=8, colors=COLOR_MUTED)
        colorbar.ax.yaxis.label.set_size(9)

    return fig


# df: track DataFrame of the beam
# metadata: dict of metadata.yml of the dataset (see load_dataset)
# out_path: path of the output pdf
# Returns: out_path as a Path
def plot_init_beam(df, metadata, out_path):

    out_path = Path(out_path)
    particle = df["particle"].iloc[0]

    # All the pages are built first: a failure does not leave a partial pdf behind
    pages = [page(df, metadata, particle)
             for page in (page_generated, page_derived, page_correlations)]

    with PdfPages(out_path) as pdf:
        for fig in pages:
            pdf.savefig(fig)

    return out_path


def main():

    parser = argparse.ArgumentParser(description="Redo the report of an initial beam dataset.")
    parser.add_argument("dataset", help="folder of the dataset (absolute, or relative to data/)")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        path = DATA_DIR / path

    df, metadata = load_dataset(path)
    print(f"Saved report to {plot_init_beam(df, metadata, path / REPORT_NAME)}")


if __name__ == "__main__":
    main()

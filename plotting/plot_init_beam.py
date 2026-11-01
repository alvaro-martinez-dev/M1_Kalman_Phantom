# plotting/plot_init_beam.py
#
# Validation report of an init_beam dataset: a single multi-page pdf with
#   1. Generated parameters (x0, theta0, p0) against the Gaussians of metadata.yml, with pulls
#   2. Derived quantities of the track table against their expected distributions, with pulls
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

from src.utils.config_loader import get_constant
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
N_INTEGRATION = 2001        # points of the numerical integrals of the expected distributions
INTEGRATION_RANGE = 8       # half width of those integrals, in sigmas


# Number of histogram bins for n tracks
def n_bins(n):

    return int(np.clip(np.sqrt(n), 10, 60))


# Column of df in the units of the plot
def values(df, column):

    return df[column].to_numpy() * VARIABLES[column][1]


# True also if the column only varies by rounding errors (e.g. E0 of a beam with sigma_p0 = 0)
def is_constant(x):

    return np.ptp(x) <= 1e-12 * np.abs(x).max()


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


# Gaussian CDF, also valid for sigma = 0 (a step at the mean)
def gaussian_cdf(x, mean, sigma):

    return norm.cdf(x, mean, sigma) if sigma > 0 else (x >= mean).astype(float)


# Points and weights to average a function over a Gaussian (a single value if sigma = 0)
def gaussian_points(mean, sigma):

    u = np.linspace(-INTEGRATION_RANGE, INTEGRATION_RANGE, N_INTEGRATION)
    weights = norm.pdf(u)

    return mean + sigma * u, weights / weights.sum()


# Expected CDF of a derived column, from the Gaussians of p0 and theta0 of the beam block.
# The formulas are independent of track_table, so that the comparison also validates it.
# Assumes p > 0 and a forward beam (|theta| < pi/2).
# column: one of the non constant columns of DERIVED
# beam: beam block of the metadata
# Returns: function of the values of the column in the units of the track table. It decreases
#          with q_over_p0 if q > 0: only the differences between bin edges are meaningful.
def expected_cdf(column, beam):

    particle = get_constant(beam["particle"])
    m, q = particle.mass, particle.charge
    theta0, sigma_theta0 = beam["theta0"], beam["sigma_theta0"]
    p0, sigma_p0 = beam["p0"], beam["sigma_p0"]

    def cdf_p(p):
        return gaussian_cdf(p, p0, sigma_p0)

    # 1. Functions of a single generated parameter, inverted
    if column == "tx0":
        return lambda t: gaussian_cdf(np.arctan(t), theta0, sigma_theta0)
    if column == "E0":
        return lambda E: cdf_p(np.sqrt(np.maximum(E**2 - m**2, 0)))
    if column == "T0":
        return lambda T: cdf_p(np.sqrt(np.maximum((T + m)**2 - m**2, 0)))
    if column == "beta0":
        return lambda beta: cdf_p(m * beta / np.sqrt(1 - beta**2))
    if column == "q_over_p0":
        return lambda q_over_p: cdf_p(q / q_over_p)

    # 2. Functions of both, averaged over the parameter that keeps the integrand smooth:
    #    P(px < a) = <P(theta < arcsin(a / p))>_p,  P(pz < a) = <P(p < a / cos(theta))>_theta
    #    If the other parameter is fixed the integrand is a step, so the result is written directly.
    if column == "px0":
        if sigma_theta0 == 0:
            return lambda a: cdf_p(a / np.sin(theta0))
        points, weights = gaussian_points(p0, sigma_p0)
        def cdf_px(a):
            angle = np.arcsin(np.clip(a[:, None] / points, -1, 1))
            return gaussian_cdf(angle, theta0, sigma_theta0) @ weights
        return cdf_px
    if column == "pz0":
        if sigma_p0 == 0:
            def cdf_pz_fixed_p(a):
                angle = np.arccos(np.clip(a / p0, -1, 1))
                return 1 - gaussian_cdf(angle, theta0, sigma_theta0) + gaussian_cdf(-angle, theta0, sigma_theta0)
            return cdf_pz_fixed_p
        points, weights = gaussian_points(theta0, sigma_theta0)
        return lambda a: cdf_p(a[:, None] / np.cos(points)) @ weights

    raise ValueError(f"No expected distribution for {column}")


# Expected mean and std of a derived column, in the units of the track table.
# Every column is a product f(p) * g(theta) of independent parameters, so that
# <f g> = <f> <g> and <(f g)^2> = <f^2> <g^2>, averaged over the Gaussians of the beam block.
# Same assumptions as expected_cdf.
def expected_mean_std(column, beam):

    particle = get_constant(beam["particle"])
    m, q = particle.mass, particle.charge
    p, weights_p = gaussian_points(beam["p0"], beam["sigma_p0"])
    theta, weights_theta = gaussian_points(beam["theta0"], beam["sigma_theta0"])
    E = np.sqrt(p**2 + m**2)

    f, g = {
        "tx0":       (1, np.tan(theta)),
        "px0":       (p, np.sin(theta)),
        "pz0":       (p, np.cos(theta)),
        "E0":        (E, 1),
        "T0":        (E - m, 1),
        "beta0":     (p / E, 1),
        "q_over_p0": (q / p, 1),
    }[column]

    f, g = f * np.ones_like(p), g * np.ones_like(theta)
    mean = (f @ weights_p) * (g @ weights_theta)
    second_moment = (f**2 @ weights_p) * (g**2 @ weights_theta)

    return mean, np.sqrt(max(second_moment - mean**2, 0))


# Title of a panel: expected and generated mean ± std, with the decimals given by the
# generated std (> 0, the column is not constant), and the chi2 of the pulls
def stats_title(x, mean, sigma, chi2, ndof):

    decimals = max(0, 3 - int(np.floor(np.log10(x.std()))))

    # Adding 0.0 turns a rounded -0.0 into 0.0
    def pair(mean, sigma):
        return f"{round(mean, decimals) + 0.0:.{decimals}f} ± {sigma:.{decimals}f}"

    return (f"Exp. mean ± std: {pair(mean, sigma)}\n"
            f"Gen. mean ± std: {pair(x.mean(), x.std())}\n"
            f"$\\chi^2$/ndof = {chi2:.1f}/{ndof} = {chi2 / ndof:.2f}")


# Histogram of a column and its expected counts (ax), with the pulls below (ax_pull)
# Returns: chi2, ndof of the pulls
def expected_panel(ax, ax_pull, column, x, edges, expected):

    # 1. Observed counts and pulls
    counts, _ = np.histogram(x, edges)
    used = expected >= MIN_EXPECTED_FOR_PULL
    pulls = (counts[used] - expected[used]) / np.sqrt(expected[used])
    chi2, ndof = np.sum(pulls**2), len(pulls)
    centers = (edges[:-1] + edges[1:]) / 2

    # 2. Histogram
    ax.stairs(counts, edges, fill=True, color=COLOR_DATA, label="Generated")
    ax.stairs(expected, edges, color=COLOR_EXPECTED, linewidth=1.5, label="Expected")
    ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
    ax.tick_params(labelbottom=False)

    # 3. Pulls
    limit = max(5, np.ceil(np.abs(pulls).max()))
    ax_pull.scatter(centers[used], pulls, s=7, color=COLOR_DATA)
    ax_pull.axhline(0, color=COLOR_EXPECTED, linewidth=1)
    ax_pull.set_ylim(-limit, limit)
    ax_pull.set_yticks([-3, 0, 3])
    ax_pull.set_xlabel(VARIABLES[column][0], fontsize=10, color=COLOR_TEXT)
    ax_pull.set_ylabel("Pull", fontsize=9, color=COLOR_MUTED)

    # 4. Fixed proportions, with the pulls right below the histogram
    ax.set_box_aspect(PANEL_ASPECT)
    ax.set_anchor("S")
    ax_pull.set_box_aspect(PULL_ASPECT)
    ax_pull.set_anchor("N")

    style_axes(ax)
    style_axes(ax_pull)

    return chi2, ndof


# Panel of a generated parameter against its Gaussian
def gaussian_panel(ax, ax_pull, df, column, beam):

    factor = VARIABLES[column][1]
    x = values(df, column)
    n = len(x)
    mean, sigma = beam[column] * factor, beam[f"sigma_{column}"] * factor

    # The range never hides tracks out of the Gaussian
    edges = np.linspace(min(mean - 4 * sigma, x.min()), max(mean + 4 * sigma, x.max()), n_bins(n) + 1)
    expected = n * np.diff(norm.cdf(edges, mean, sigma))

    chi2, ndof = expected_panel(ax, ax_pull, column, x, edges, expected)
    ax.set_title(stats_title(x, mean, sigma, chi2, ndof), fontsize=8, color=COLOR_TEXT, loc="left")


# Panel of a derived quantity against its expected distribution
def derived_panel(ax, ax_pull, df, column, beam):

    factor = VARIABLES[column][1]
    x = values(df, column)
    n = len(x)
    edges = np.linspace(x.min(), x.max(), n_bins(n) + 1)
    cdf = expected_cdf(column, beam)
    expected = n * np.abs(np.diff(cdf(edges / factor)))
    mean, sigma = expected_mean_std(column, beam)

    chi2, ndof = expected_panel(ax, ax_pull, column, x, edges, expected)
    ax.set_title(stats_title(x, mean * factor, sigma * factor, chi2, ndof),
                 fontsize=7, color=COLOR_TEXT, loc="left")   # 4 panels per row
    ax_pull.locator_params(axis="x", nbins=5)


# Legend in the first panel of the figure that is not constant
def add_legend(fig):

    for ax in fig.axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            break


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

    add_legend(fig)

    return fig


# Page 2: derived quantities of the track table against their expected distributions
def page_derived(df, metadata, particle):

    fig = new_page("Derived quantities", particle, metadata)
    # Two rows of panels, each with its histogram and its pulls below
    grid = fig.add_gridspec(4, 4, height_ratios=[PANEL_ASPECT, PULL_ASPECT] * 2)

    for i, column in enumerate(DERIVED):
        row, col = 2 * (i // 4), i % 4
        x = values(df, column)
        if is_constant(x):
            constant_panel(fig.add_subplot(grid[row:row + 2, col]), column, x[0])
            continue
        ax = fig.add_subplot(grid[row, col])
        ax_pull = fig.add_subplot(grid[row + 1, col], sharex=ax)
        derived_panel(ax, ax_pull, df, column, metadata["beam"])

    add_legend(fig)

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

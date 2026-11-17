# plotting/plot_fit.py
#
# Validation report of a Kalman fit (data/fit/, see scripts/generate_fit.py): a single multi-page
# pdf that compares the fitted states with the MC truth of the propagation the hits come from
# (joined by track_id, plane_index; the truth is read from the propagation, the fit never sees it).
# Filter (filt) and smoother (smooth, if the fit has it) are drawn side by side:
#   1. Pulls (fit - true) / sigma_fit of x, tx and q/p at all the planes, against N(0, 1), one page
#      per stage: the errors of the fit are right if the pulls are N(0, 1)
#   2. chi2 of the tracks against chi2(ndf), and its probability against a flat distribution
#   3. Per detector plane: resolution std(fit - true) against the error the fit gives, sqrt(<sigma^2>)
#   4. Per detector plane: mean and std of the pulls, against 0 and 1
#   5. A few tracks: fit - true of x and q/p along z with their ±1 sigma bands, and the hits
#
# Usage (from the repo root), to redo the report of an existing fit:
#   python -m plotting.plot_fit fit/<name>

import argparse
import warnings
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.stats import chi2 as chi2_dist

from plotting.plot_hits import chi2_text, gaussian_panel
from plotting.plot_init_beam import (COLOR_AXIS, COLOR_EXPECTED, COLOR_MUTED, COLOR_TEXT, PAGE_SIZE,
                                     expected_panel, n_bins, style_axes)
from plotting.plot_layout import COLORS_MATERIAL
from src.utils.dataset_io import dataset_path, load_dataset, load_table

REPORT_NAME = "report.pdf"

STAGES = {"filt": ("Filter", COLORS_MATERIAL[0]), "smooth": ("Smoother", COLORS_MATERIAL[1])}
# Variable: (label, label of fit - true, factor of fit - true and of the errors to the plotted units).
# q/p is drawn relative to the true value, in %
VARIABLES = {
    "x": ("x", r"$x_\mathrm{fit} - x_\mathrm{true}$ [µm]", 1e3),
    "tx": (r"$t_x$", r"$t_{x,\mathrm{fit}} - t_{x,\mathrm{true}}$ [mrad]", 1e3),
    "q_over_p": ("q/p", r"$(q/p)_\mathrm{fit} / (q/p)_\mathrm{true} - 1$ [%]", 1e2),
}
RESOLUTION_UNITS = {"x": "µm", "tx": "mrad", "q_over_p": "% of q/p"}
N_EXAMPLE_TRACKS = 3       # tracks of the last page
MARKER_SIZE = 5.0
BAND_ALPHA = 0.25


# Fit rows joined with the true state of the propagation dataset
# Returns: DataFrame of the fit with, for each variable and stage, d_<var>_<stage> (fit - true, in the
#          plotted units), e_<var>_<stage> (error, same units) and pull_<var>_<stage>
def fit_with_truth(fit, metadata, stages):

    prop = metadata["propagation"]
    _, prop_meta = load_dataset(prop["input"])
    if prop_meta.get("created") != prop.get("created"):
        warnings.warn(f"{prop['input']} was regenerated after this fit: the truth may not be the one of the hits")
    states = load_table(prop["input"], "states")
    df = fit.merge(states[["track_id", "plane_index", "x", "tx", "q_over_p"]].rename(
        columns={v: f"{v}_true" for v in VARIABLES}), on=["track_id", "plane_index"], how="left",
        validate="one_to_one")
    if df["x_true"].isna().any():
        raise ValueError(f"{df['x_true'].isna().sum()} fitted hits have no true state in {prop['input']}")

    for var, (_, _, factor) in VARIABLES.items():
        scale = factor / np.abs(df[f"{var}_true"]) if var == "q_over_p" else factor
        for st in stages:
            d = df[f"{var}_{st}"] - df[f"{var}_true"]
            df[f"pull_{var}_{st}"] = d / df[f"{var}_{st}_err"]
            df[f"d_{var}_{st}"] = d * scale
            df[f"e_{var}_{st}"] = df[f"{var}_{st}_err"] * scale
    return df


# Page with an empty figure and the header that identifies the fit
def new_page(title, metadata):

    commit = str(metadata["git_commit"])[:7]
    if metadata["git_uncommitted_changes"]:
        commit += " + uncommitted changes"
    subtitle = (f"{metadata['particle']}  |  {metadata['n_tracks']} tracks ({metadata['n_ok']} fitted, "
                f"{metadata['n_stopped']} stopped), {metadata['n_hits']} hits  |  "
                f"$\\sigma_x$ = {metadata['sigma_x']:g} mm  |  smoother {'on' if metadata['smoother'] else 'off'}\n"
                f"from {Path(metadata['input']).name}\ncreated {metadata['created']}  |  commit {commit}")
    fig = Figure(figsize=PAGE_SIZE, layout="constrained")
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11, color=COLOR_TEXT)
    return fig


# Page 1 (one per stage): pulls of x, tx and q/p at all the planes against N(0, 1)
def page_pulls(df, metadata, stage):

    fig = new_page(f"Pulls of the {STAGES[stage][0].lower()} at all the planes", metadata)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.6, 1])
    for j, (var, (label, _, _)) in enumerate(VARIABLES.items()):
        ax = fig.add_subplot(grid[0, j])
        ax_pull = fig.add_subplot(grid[1, j], sharex=ax)
        gaussian_panel(ax, ax_pull, df[f"pull_{var}_{stage}"].to_numpy(), 0.0, 1.0,
                       f"({label}$_\\mathrm{{fit}}$ − {label}$_\\mathrm{{true}}$) / $\\sigma_\\mathrm{{fit}}$")
        if j == 0:
            ax.legend(["Fit", "N(0, 1)"], loc="upper right", fontsize=8, frameon=False)
    fig.text(0.5, 0.02, "One entry per hit. The entries of a track are correlated (q/p barely changes along it), "
             "so the $\\chi^2$ of the histogram, which assumes independent entries, can be > 1 with std ≈ 1",
             ha="center", fontsize=8, color=COLOR_MUTED)
    return fig


# Page 2: chi2 of the tracks against chi2(ndf) and its probability against a flat distribution
def page_chi2(tracks, metadata):

    fig = new_page("$\\chi^2$ of the tracks (predicted residuals of the filter)", metadata)
    ok = tracks[tracks["status"] == "ok"]
    chi2, ndf = ok["chi2"].to_numpy(), ok["ndf"].to_numpy()
    grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])

    # chi2: the tracks may have different ndf (fewer hits if they stop), so the expected counts
    # are the sum of the chi2(ndf) of each group of tracks
    ax = fig.add_subplot(grid[0, 0])
    ax_pull = fig.add_subplot(grid[1, 0], sharex=ax)
    edges = np.linspace(0, max(np.quantile(chi2, 0.999), chi2_dist.ppf(0.999, ndf.max())), n_bins(len(chi2)) + 1)
    expected = sum(np.sum(ndf == n) * np.diff(chi2_dist.cdf(edges, n)) for n in np.unique(ndf))
    c2, nd = expected_panel(ax, ax_pull, None, chi2, edges, expected, xlabel="$\\chi^2$ of the track")
    ndf_text = f"ndf = {ndf[0]}" if len(np.unique(ndf)) == 1 else f"ndf = {ndf.min()}-{ndf.max()}"
    ax.set_title(f"{ndf_text}  |  mean $\\chi^2$/ndf = {np.mean(chi2 / ndf):.3f} (expected 1)\n"
                 f"histogram vs $\\chi^2$(ndf): $\\chi^2$/ndof = {c2:.1f}/{nd} = {c2 / max(nd, 1):.2f}",
                 fontsize=8, color=COLOR_TEXT, loc="left")
    ax.legend(["Fit", "$\\chi^2$(ndf)"], loc="upper right", fontsize=8, frameon=False)

    # probability: flat if the chi2 follow chi2(ndf)
    prob = chi2_dist.sf(chi2, ndf)
    ax = fig.add_subplot(grid[0, 1])
    ax_pull = fig.add_subplot(grid[1, 1], sharex=ax)
    edges = np.linspace(0, 1, 21)
    c2, nd = expected_panel(ax, ax_pull, None, prob, edges, np.full(20, len(prob) / 20),
                            xlabel="P($\\chi^2$, ndf)")
    ax.set_title(f"mean = {prob.mean():.3f} (expected 0.5)\n"
                 f"histogram vs flat: $\\chi^2$/ndof = {c2:.1f}/{nd} = {c2 / max(nd, 1):.2f}",
                 fontsize=8, color=COLOR_TEXT, loc="left")
    ax.set_ylim(0, None)
    return fig


# Per plane statistics of a column: dict of arrays, one value per plane (nan for a plane with no rows)
def per_plane(df, column, n_planes):

    groups = df.groupby("plane_index")[column]
    index = np.arange(n_planes)
    n = groups.size().reindex(index, fill_value=0).to_numpy()
    mean = groups.mean().reindex(index).to_numpy()
    std = groups.std().reindex(index).to_numpy()
    rms = np.sqrt((df[column]**2).groupby(df["plane_index"]).mean().reindex(index).to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"n": n, "mean": mean, "std": std, "rms": rms,
                "mean_err": std / np.sqrt(n), "std_err": std / np.sqrt(2 * (n - 1))}


# Plane names on the x axis. style_axes must be called on ALL the axes before (it formats the x
# axis as numbers, which fails once a shared axis has the names)
# narrow: only the names, more rotated (for a grid of several panels side by side)
def plane_axis(ax, planes, narrow=False):
    labels = [p["name"] if narrow else f"{p['name']}\nz = {p['z']:g}" for p in planes]
    ax.set_xticks(np.arange(len(planes)), labels, rotation=60 if narrow else 30, ha="right",
                  rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=6 if narrow else 7)


# Page 3: per plane, resolution std(fit - true) (points) and error given by the fit (open markers)
def page_resolution(df, metadata, stages):

    fig = new_page("Resolution per detector plane: std(fit − true) ● against the error of the fit ○", metadata)
    planes = metadata["planes"]
    index = np.arange(len(planes))
    offsets = np.linspace(-0.12, 0.12, len(stages)) if len(stages) > 1 else [0.0]
    axes = fig.subplots(3, 1, sharex=True)
    for ax in axes:
        style_axes(ax)
    for ax, (var, (label, _, _)) in zip(axes, VARIABLES.items()):
        for st, offset in zip(stages, offsets):
            name, color = STAGES[st]
            res = per_plane(df, f"d_{var}_{st}", len(planes))
            err = per_plane(df, f"e_{var}_{st}", len(planes))
            ax.errorbar(index + offset, res["std"], yerr=res["std_err"], fmt="o", markersize=MARKER_SIZE,
                        color=color, label=f"{name}: std(fit − true)")
            ax.plot(index + offset, err["rms"], linestyle="none", marker="o", markersize=MARKER_SIZE + 2,
                    markerfacecolor="none", markeredgecolor=color, label=f"{name}: error of the fit")
        if var == "x":
            ax.axhline(metadata["sigma_x"] * 1e3, color=COLOR_EXPECTED, linewidth=1,
                       label=f"$\\sigma_x$ of the hits = {metadata['sigma_x'] * 1e3:g} µm")
        ax.set_ylabel(f"Resolution [{RESOLUTION_UNITS[var]}]", fontsize=9, color=COLOR_MUTED)
        ax.set_title(f"{label}", fontsize=9, color=COLOR_TEXT, loc="left")
        ax.set_ylim(0, None)
        ax.legend(loc="upper right", fontsize=7, frameon=False, ncols=len(stages) + 1)
        plane_axis(ax, planes)
    return fig


# Page 4: per plane, mean and std of the pulls of x, tx and q/p, against 0 and 1
def page_pull_planes(df, metadata, stages):

    fig = new_page("Pulls per detector plane: mean (expected 0) and std (expected 1)", metadata)
    planes = metadata["planes"]
    index = np.arange(len(planes))
    offsets = np.linspace(-0.12, 0.12, len(stages)) if len(stages) > 1 else [0.0]
    axes = fig.subplots(2, 3, sharex=True)
    for ax in axes.flat:
        style_axes(ax)
    for j, (var, (label, _, _)) in enumerate(VARIABLES.items()):
        titles = {0: [], 1: []}
        for st, offset in zip(stages, offsets):
            name, color = STAGES[st]
            s = per_plane(df, f"pull_{var}_{st}", len(planes))
            axes[0, j].errorbar(index + offset, s["mean"], yerr=s["mean_err"], fmt="o", markersize=MARKER_SIZE,
                                color=color, label=name)
            axes[1, j].errorbar(index + offset, s["std"], yerr=s["std_err"], fmt="o", markersize=MARKER_SIZE,
                                color=color, label=name)
            titles[0].append(f"{name}: " + chi2_text(s["mean"], s["mean_err"], 0.0))
            titles[1].append(f"{name}: " + chi2_text(s["std"], s["std_err"], 1.0))
        for row, expected in ((0, 0.0), (1, 1.0)):
            ax = axes[row, j]
            ax.axhline(expected, color=COLOR_EXPECTED, linewidth=1.5)
            ax.set_title(f"{label}\n" + "\n".join(titles[row]), fontsize=7, color=COLOR_TEXT, loc="left")
            ax.set_ylabel("Mean of the pulls" if row == 0 else "Std of the pulls", fontsize=9, color=COLOR_MUTED)
            plane_axis(ax, planes, narrow=True)
        axes[0, j].legend(loc="upper right", fontsize=7, frameon=False, ncols=len(stages))
    return fig


# Page 5: a few tracks chosen at random (seed): fit - true of x and q/p along z, with ±1 sigma bands
def page_examples(df, metadata, stages, n_tracks, seed):

    fig = new_page("A few tracks: fit − true along z, with the ±1σ band of the fit", metadata)
    track_ids = np.unique(df["track_id"])
    n_tracks = min(n_tracks, len(track_ids))
    chosen = np.sort(np.random.default_rng(seed).choice(track_ids, size=n_tracks, replace=False))
    axes = fig.subplots(2, n_tracks, sharex=True, squeeze=False)
    for j, track_id in enumerate(chosen):
        t = df[df["track_id"] == track_id].sort_values("z")
        for row, var in enumerate(("x", "q_over_p")):
            ax = axes[row, j]
            for st in stages:
                name, color = STAGES[st]
                d, e = t[f"d_{var}_{st}"].to_numpy(), t[f"e_{var}_{st}"].to_numpy()
                ax.fill_between(t["z"], d - e, d + e, color=color, alpha=BAND_ALPHA, linewidth=0)
                ax.plot(t["z"], d, color=color, linewidth=1.2, marker="o", markersize=MARKER_SIZE * 0.7, label=name)
            if var == "x":       # the hits, relative to the truth
                ax.errorbar(t["z"], (t["x_meas"] - t["x_true"]) * 1e3, yerr=t["sigma_x"] * 1e3, fmt="o",
                            markersize=MARKER_SIZE * 0.6, color=COLOR_MUTED, linewidth=1, label="Hit")
            ax.axhline(0, color=COLOR_EXPECTED, linewidth=1)
            ax.set_ylabel(VARIABLES[var][1], fontsize=9, color=COLOR_MUTED)
            if row == 0:
                ax.set_title(f"Track {track_id}", fontsize=9, color=COLOR_TEXT, loc="left")
            else:
                ax.set_xlabel("z [mm]", fontsize=10, color=COLOR_TEXT)
            style_axes(ax)
    axes[0, 0].legend(loc="best", fontsize=7, frameon=False)
    return fig


# fit_dir: folder of a fit dataset (absolute, or relative to data/)
# out_path: output pdf (default: REPORT_NAME in the dataset folder)
# n_tracks, seed: number of tracks of the last page and the seed of their choice
# Returns: out_path as a Path
def plot_fit(fit_dir, out_path=None, n_tracks=N_EXAMPLE_TRACKS, seed=0):

    fit_dir = dataset_path(fit_dir)
    tracks, metadata = load_dataset(fit_dir)
    fit = load_table(fit_dir, "fit")
    if fit.empty:
        raise ValueError(f"{fit_dir} has no fitted tracks")
    stages = ["filt", "smooth"] if metadata["smoother"] else ["filt"]
    df = fit_with_truth(fit, metadata, stages)

    # All the pages are built first: a failure does not leave a partial pdf behind
    pages = [page_pulls(df, metadata, st) for st in stages]
    pages += [page_chi2(tracks, metadata), page_resolution(df, metadata, stages),
              page_pull_planes(df, metadata, stages), page_examples(df, metadata, stages, n_tracks, seed)]

    out_path = Path(out_path) if out_path is not None else fit_dir / REPORT_NAME
    with PdfPages(out_path) as pdf:
        for fig in pages:
            pdf.savefig(fig)
    return out_path


def main():

    parser = argparse.ArgumentParser(description="Redo the report of a Kalman fit.")
    parser.add_argument("dataset", help="folder of the fit (absolute, or relative to data/)")
    parser.add_argument("--n-tracks", type=int, default=N_EXAMPLE_TRACKS,
                        help=f"tracks drawn in the last page (default {N_EXAMPLE_TRACKS})")
    parser.add_argument("--seed", type=int, default=0, help="seed of the choice of those tracks (default 0)")
    parser.add_argument("--out", default=None, help=f"output pdf (default: {REPORT_NAME} in the dataset folder)")
    args = parser.parse_args()

    print(f"Saved report to {plot_fit(args.dataset, args.out, args.n_tracks, args.seed)}")


if __name__ == "__main__":
    main()

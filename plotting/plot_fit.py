# plotting/plot_fit.py
#
# Validation report of a Kalman fit (data/fit/, see scripts/generate_fit.py): a single multi-page
# pdf that compares the fitted states with the MC truth of the propagation the hits come from
# (joined by track_id, plane_index; the truth is read from the propagation, the fit never sees it).
# Filter (filt) and smoother (smooth, if the fit has it) are drawn together:
#   1. Pulls (fit - true) / sigma_fit of x, tx and q/p against N(0, 1), one row per stage: the
#      errors of the fit are right if the pulls are N(0, 1). ONE plane per track, chosen at random:
#      the pulls of the planes of a track are correlated (q/p is a single number per track, the
#      smoother uses the same hits at every plane), so a histogram of all the planes would not
#      have independent entries and its chi2 against N(0, 1) would mean nothing
#   2-3. Fit and true values of x, tx and q/p overlaid, at the entrance and exit of the phantom (2:
#      D_phantom_start, D_phantom_end) and of the magnet (3: D_magnet_start, D_magnet_end): what the
#      pulls measure, seen directly. A missing element leaves its rows empty, saying so. The fit is a
#      posterior mean, pulled towards the prior, so its distribution is NARROWER than the true one
#      where the measurements say little: Var(fit) ~ Var(true) - Var(fit - true). E.g. q/p of the
#      filter before the magnet is almost the prior for every track (a spike: its peak is cut and
#      the panel says so)
#   4. chi2 of the tracks against chi2(ndf), and its probability against a flat distribution
#   4b. Innovations of the filter, r / sqrt(S) (predicted residuals, the terms of the chi2): the only
#      test that does not need the MC truth, so the one that could be made on real data.
#      Per plane: mean and std against 0 and 1 (the chi2 of the track sums them over the planes, where
#      a plane too wide and another too narrow compensate each other).
#      Whiteness: with a right model (F, Q, V) the innovations of a track are independent, so the
#      correlation r between those of two planes, over the tracks, is 0 within 1/sqrt(n): r sqrt(n) ~
#      N(0, 1) and the sum of its squares over the pairs of planes ~ chi2(pairs). It sees a wrong F or Q
#      (e.g. scattering in the wrong place) even when the variances look right (Bar-Shalom, Li and
#      Kirubarajan, Estimation with Applications to Tracking and Navigation, Wiley 2001, sec. 5.4)
#   5. NEES = (fit - true)^T C^-1 (fit - true) against chi2(3): tests the whole covariance, also the
#      correlations that the pulls do not see. Histogram at one plane per track, and mean per plane
#      (only for fits that store the correlations, corr_* columns of src/kalman/fit.py)
#   6. Per detector plane: resolution std(fit - true) against the error the fit gives, sqrt(<sigma^2>)
#   7. Per detector plane: mean and std of the pulls, against 0 and 1
#   8. Tracks that stop in the phantom (fits with range_end, see src/kalman/fit.py): how many, the
#      coverage of the range bounds (true initial 1/p inside [s_lo, s_hi]), the energy resolution at
#      the phantom entrance against that of the beam, and the pulls against the number of hits
#      before the last one
#   8b. Proton radiography (tracks that cross the phantom, fits with smoother): WEPL of each track,
#      R_water(T_in) - R_water(T_out) (src/phantom/wepl.py), against the true one from the MC energies
#      at D_phantom_start and D_phantom_end. T_in is not measured (the planes only measure x): the
#      nominal energy of the beam, or the true one (ideal). T_out (it is the same anywhere after the
#      phantom: vacuum, and B does no work) from:
#        - the full fit, the smoother at D_phantom_end. But the fit KNOWS the phantom: its prior (the
#          beam) is slowed down through the MODEL phantom, with weight sigma_smooth^2 / sigma_filt^2 in
#          the smoother (~0.5 here), so the errors of T_in and T_out cancel and the WEPL leans to the
#          model. Not a measurement of an unknown phantom (proton CT)
#        - the magnet only: the prior removed from the smoother (remove_prior, the filter at
#          D_phantom_end is the prior plus the phantom hits, which say almost nothing of q/p). This is
#          the measurement: the pull of its q/p against the truth checks it
#      Error budget: sigma_WEPL = sigma_T / S_water(T) for the beam spread and for T_out, in quadrature
#   9. A few tracks: fit - true of x and q/p along z with their ±1 sigma bands, and the hits
#
# The LAST hit of a track that stops is left out of pages 1, 2, 3, 5, 6 and 7: there the true energy
# is anywhere between ~1 MeV and the energy to cross the layer, which no Gaussian in q/p describes
# (see src/kalman/filter.py), and it would hide how the rest of the fit behaves. Page 8 shows it apart.
#
# Usage (from the repo root), to redo the report of an existing fit:
#   python -m plotting.plot_fit fit/<name>

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

from plotting.plot_hits import chi2_text, gaussian_panel
from plotting.plot_init_beam import (COLOR_AXIS, COLOR_EXPECTED, COLOR_MUTED, COLOR_TEXT, PAGE_SIZE,
                                     PANEL_ROWS, expected_panel, n_bins, style_axes)
from plotting.plot_layout import COLORS_MATERIAL
from plotting.plot_tracks import plane_index_named
from src.kalman.fit import CORR_PAIRS
from src.phantom.depth_dose import csda_stop_depth
from src.phantom.wepl import wepl_error, wepl_from_energies, wepl_range_definition
from src.propagator.propagator import kinetic_energy
from src.utils.config_loader import make_geometry, particle_by_name
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
# Page of fit and true values overlaid: (axis label, factor to the plotted units)
VALUES = {"x": (r"$x$ [mm]", 1), "tx": (r"$t_x$ [mrad]", 1e3), "q_over_p": (r"$q/p$ [e/GeV]", 1e3)}
COLOR_TRUE = "#c3c2b7"      # filled histogram of the true values, under the lines of the fit
SPIKE_FACTOR = 3            # a fit histogram this many times taller than the true one is cut (pages 2-3)
# Pages of fit and true values overlaid: element -> planes (one row each), entrance and exit
FIT_VS_TRUE_PAGES = {"phantom": ["D_phantom_start", "D_phantom_end"],
                     "magnet": ["D_magnet_start", "D_magnet_end"]}
SIDE_MARGIN = [1, 7, 1]     # width ratios margin / panels / margin of the per-plane resolution page
N_EXAMPLE_TRACKS = 3       # tracks of the last page
COLOR_MAGNET = COLORS_MATERIAL[3]  # WEPL page: magnet only, entrance energy of the beam
COLOR_IDEAL = COLORS_MATERIAL[2]   # WEPL page: magnet only, true entrance energy
N_X_BINS = 8               # WEPL page: bins of x at the phantom exit
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
    if has_correlations(df, stages):
        for st in stages:
            df[f"nees_{st}"] = nees(df, st)
    return df


# Marks the rows of the tracks that stop in the phantom (range_end of the fit tracks table; False for
# fits made before it existed): stops (bool, per row), hits_to_last (hits after this one in its
# track, 0 = last hit) and last_of_stop (the last hit of a track that stops, see the header)
def mark_stopping(df, tracks):
    stopping = set(tracks.loc[tracks["range_end"], "track_id"]) if "range_end" in tracks else set()
    df["stops"] = df["track_id"].isin(stopping)
    df["hits_to_last"] = df.groupby("track_id")["plane_index"].transform("max") - df["plane_index"]
    df["last_of_stop"] = df["stops"] & (df["hits_to_last"] == 0)
    return df


EXCLUDED_NOTE = "Without the last hit of the tracks that stop in the phantom (page 'Tracks that stop')."


def has_correlations(df, stages):
    return all(f"corr_{va}_{vb}_{st}" in df for (_, va), (_, vb) in CORR_PAIRS for st in stages)


# NEES of each row: with z = pulls and R = correlation matrix, (fit - true)^T C^-1 (fit - true) = z^T R^-1 z
# (same number, but R is well conditioned while C mixes mm, rad and e/MeV)
def nees(df, stage):
    z = np.stack([df[f"pull_{v}_{stage}"].to_numpy() for v in VARIABLES], axis=1)
    R = np.broadcast_to(np.eye(3), (len(df), 3, 3)).copy()
    for (a, va), (b, vb) in CORR_PAIRS:
        R[:, a, b] = R[:, b, a] = df[f"corr_{va}_{vb}_{stage}"].to_numpy()
    return np.einsum("ni,ni->n", z, np.linalg.solve(R, z[:, :, None])[:, :, 0])


# One row per track, at a plane chosen at random (seed): independent entries for a histogram
def one_per_track(df, seed):
    return df.groupby("track_id").sample(n=1, random_state=seed)


# Page with an empty figure and the header that identifies the fit
# excludes_last_of_stop: the page leaves out the last hit of the tracks that stop (see the header):
#                        say it, if the fit has such tracks
def new_page(title, metadata, excludes_last_of_stop=False):

    commit = str(metadata["git_commit"])[:7]
    if metadata["git_uncommitted_changes"]:
        commit += " + uncommitted changes"
    range_text = (f", {metadata['n_range_end']} stop in the phantom" if "n_range_end" in metadata else "")
    constraints = ("  |  range constraints " + ("on" if metadata["range_constraints"] else "off")
                   if "range_constraints" in metadata else "")
    subtitle = (f"{metadata['particle']}  |  {metadata['n_tracks']} tracks ({metadata['n_ok']} fitted, "
                f"{metadata['n_stopped']} failed{range_text}), {metadata['n_hits']} hits  |  "
                f"$\\sigma_x$ = {metadata['sigma_x']:g} mm  |  smoother {'on' if metadata['smoother'] else 'off'}"
                f"{constraints}\n"
                f"from {Path(metadata['input']).name}\ncreated {metadata['created']}  |  commit {commit}")
    if excludes_last_of_stop and metadata.get("n_range_end", 0) > 0:
        subtitle += f"\n{EXCLUDED_NOTE}"
    fig = Figure(figsize=PAGE_SIZE, layout="constrained")
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11, color=COLOR_TEXT)
    return fig


# Page 1: pulls of x, tx and q/p against N(0, 1), at one random plane per track, one row per stage
# sample: one_per_track(df, seed), the same rows for every stage
def page_pulls(sample, metadata, stages):

    fig = new_page("Pulls (one random plane per track)", metadata, True)
    grid = fig.add_gridspec(2 * len(stages), 3, height_ratios=PANEL_ROWS * len(stages))
    for i, st in enumerate(stages):
        name, _ = STAGES[st]
        for j, (var, (label, _, _)) in enumerate(VARIABLES.items()):
            ax = fig.add_subplot(grid[2 * i, j])
            ax_pull = fig.add_subplot(grid[2 * i + 1, j], sharex=ax)
            gaussian_panel(ax, ax_pull, sample[f"pull_{var}_{st}"].to_numpy(), 0.0, 1.0,
                           f"({label}$_\\mathrm{{fit}}$ − {label}$_\\mathrm{{true}}$) / $\\sigma_\\mathrm{{fit}}$")
            ax.set_title(f"{name}\n" + ax.get_title(loc="left"), fontsize=8, color=COLOR_TEXT, loc="left")
            ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
            if j == 0:
                ax.legend(["Fit", "N(0, 1)"], loc="upper right", fontsize=8, frameon=False)
    fig.supxlabel("One entry per track, at a plane chosen at random: the planes of a track are correlated "
                  "(q/p is one number per track), so only one of them gives independent entries. "
                  "Per plane: page 'Pulls per detector plane'", fontsize=8, color=COLOR_MUTED)
    return fig


# Pages 2-3: fit (each stage) and true values of x, tx and q/p overlaid, one row per plane of `names`
# (entrance and exit of an element: FIT_VS_TRUE_PAGES). A plane the fit does not have (element
# disabled) or that no track reached gives a row of empty panels that say so. The fit histogram is
# narrower than the true one where the fit leans on the prior (see the header); a fit histogram much
# taller than the true one is cut
# element: "phantom" or "magnet", for the title; names: plane names, one row each
def page_fit_vs_true(df, metadata, stages, element, names):

    fig = new_page(f"Fit and true values overlaid: entrance and exit of the {element}", metadata, True)
    planes = metadata["planes"]
    grid = fig.add_gridspec(len(names), len(VALUES))
    handles = None
    for i, name in enumerate(names):
        k = plane_index_named(planes, name)      # merged planes (A+B) are found by any of their names
        at = df[df["plane_index"] == k] if k is not None else df.iloc[:0]
        for j, (var, (xlabel, factor)) in enumerate(VALUES.items()):
            ax = fig.add_subplot(grid[i, j])
            style_axes(ax)
            if at.empty:
                ax.set_axis_off()
                reason = (f"no {name} plane in this fit (no {element}?)" if k is None
                          else f"no fitted track at {planes[k]['name']}")
                ax.text(0.5, 0.5, f"{xlabel}\n{reason}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color=COLOR_MUTED)
                continue
            series = [("True", at[f"{var}_true"].to_numpy() * factor, COLOR_TRUE)]
            series += [(STAGES[st][0], at[f"{var}_{st}"].to_numpy() * factor, STAGES[st][1]) for st in stages]
            # range: ALL the true values (so that its histogram has every track) and at least 99.8 % of each
            # fit series (a quantile, so that one outlier of the fit does not squeeze the rest)
            true = series[0][1]
            low = min([true.min()] + [np.quantile(v, 0.001) for _, v, _ in series[1:]])
            high = max([true.max()] + [np.quantile(v, 0.999) for _, v, _ in series[1:]])
            edges = np.linspace(low, high, n_bins(len(at)) + 1)
            peaks = []
            for n, (name, values, color) in enumerate(series):
                if n == 0:
                    counts, _, _ = ax.hist(values, edges, color=color, label=name)
                else:
                    counts, _, _ = ax.hist(values, edges, histtype="step", color=color, linewidth=1.5, label=name)
                peaks.append(counts.max())
            # a fit histogram SPIKE_FACTOR times taller than the true one (a fit that is almost the prior)
            # is cut, so that the others stay readable
            cut = [name for (name, _, _), peak in zip(series[1:], peaks[1:]) if peak > SPIKE_FACTOR * peaks[0]]
            if cut:
                kept = [peak for (name, _, _), peak in zip(series, peaks) if name not in cut]
                ax.set_ylim(0, 1.15 * max(kept))
            ax.set_title(("" if j else f"{planes[k]['name']} (z = {planes[k]['z']:g} mm), {len(at)} tracks\n")
                         + f"std: {series[0][0]} {series[0][1].std():.3g}\n"
                         + "  |  ".join(f"{name} {values.std():.3g}" for name, values, _ in series[1:])
                         + (f"\n{', '.join(cut)}: peak off scale" if cut else ""),
                         fontsize=8, color=COLOR_TEXT, loc="left")
            ax.set_xlabel(xlabel, fontsize=9, color=COLOR_TEXT)
            ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
            handles = ax.get_legend_handles_labels()
    if handles:
        fig.legend(*handles, loc="outside lower center", ncols=len(handles[0]), fontsize=9, frameon=False)
    return fig


# Page 2: chi2 of the tracks against chi2(ndf) and its probability against a flat distribution
def page_chi2(tracks, metadata):

    fig = new_page("$\\chi^2$ of the tracks (predicted residuals of the filter)", metadata)
    ok = tracks[tracks["status"] == "ok"]
    chi2, ndf = ok["chi2"].to_numpy(), ok["ndf"].to_numpy()
    grid = fig.add_gridspec(2, 2, height_ratios=PANEL_ROWS)

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


# Correlation between the innovations of every pair of planes, over the tracks that have both
# df: rows with an "innovation" column (r / sqrt(S)), one per (track, plane)
# Returns: (r, n), (n_planes, n_planes) arrays: Pearson correlation (NaN on the diagonal and for pairs
#          with fewer than 3 tracks) and number of tracks with both planes
def innovation_correlations(df, n_planes):

    wide = df.pivot(index="track_id", columns="plane_index", values="innovation")
    v = wide.reindex(columns=range(n_planes)).to_numpy()
    r = np.full((n_planes, n_planes), np.nan)
    n = np.zeros((n_planes, n_planes), dtype=int)
    for i in range(n_planes):
        for j in range(i + 1, n_planes):
            both = np.isfinite(v[:, i]) & np.isfinite(v[:, j])
            n[i, j] = n[j, i] = both.sum()
            if both.sum() >= 3:
                r[i, j] = r[j, i] = np.corrcoef(v[both, i], v[both, j])[0, 1]
    return r, n


# Page of the innovations of the filter (see 4b in the header). Top: mean and std of r / sqrt(S) per
# plane. Bottom: correlation matrix between planes, and r sqrt(n) of each pair against the distance
# between the two planes
def page_innovations(df, metadata):

    fig = new_page("Innovations of the filter r / √S: per plane and whiteness (no MC truth needed)", metadata,
                   True)
    planes = metadata["planes"]
    n_planes = len(planes)
    index = np.arange(n_planes)
    color = STAGES["filt"][1]
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])

    # Per plane: mean (expected 0) and std (expected 1)
    s = per_plane(df, "innovation", n_planes)
    for col, (stat, expected, ylabel) in enumerate((("mean", 0.0, "Mean of r / √S"),
                                                    ("std", 1.0, "Std of r / √S"))):
        ax = fig.add_subplot(grid[0, col])
        style_axes(ax)
        ax.errorbar(index, s[stat], yerr=s[f"{stat}_err"], fmt="o", markersize=MARKER_SIZE, color=color)
        ax.axhline(expected, color=COLOR_EXPECTED, linewidth=1.5)
        ax.set_title(f"{ylabel} per plane (expected {expected:g})\n"
                     + chi2_text(s[stat], s[f"{stat}_err"], expected), fontsize=8, color=COLOR_TEXT, loc="left")
        ax.set_ylabel(ylabel, fontsize=9, color=COLOR_MUTED)
        plane_axis(ax, planes, narrow=True)

    # Whiteness: correlation between planes. Under independence r sqrt(n) ~ N(0, 1) for each pair
    r, n = innovation_correlations(df, n_planes)
    upper = np.triu(np.isfinite(r), k=1)
    z = np.where(upper, r * np.sqrt(n), np.nan)
    n_pairs = int(upper.sum())
    chi2 = float(np.nansum(z**2))
    names = [p["name"] for p in planes]

    ax = fig.add_subplot(grid[1, 0])
    if n_pairs:
        typical = 1.0 / np.sqrt(np.median(n[upper]))
        limit = max(np.nanmax(np.abs(r)), 3.0 * typical)
        image = ax.imshow(r, cmap="RdBu_r", vmin=-limit, vmax=limit)
        for i, j in zip(*np.nonzero(np.isfinite(r))):
            ax.text(j, i, f"{int(round(100 * r[i, j]))}", ha="center", va="center", fontsize=5, color=COLOR_TEXT)
        fig.colorbar(image, ax=ax, shrink=0.85).ax.tick_params(labelsize=7, colors=COLOR_MUTED)
        ax.set_title(f"Correlation between planes, ×100 in the cells (expected 0)\n"
                     f"statistical error 1/√n ≈ {typical:.3f}", fontsize=8, color=COLOR_TEXT, loc="left")
    else:
        ax.text(0.5, 0.5, "No pair of planes with tracks in both", ha="center", va="center", color=COLOR_MUTED)
    ax.set_xticks(index, names, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(index, names)
    ax.tick_params(labelsize=6, colors=COLOR_MUTED, length=2)

    ax = fig.add_subplot(grid[1, 1])
    style_axes(ax)
    if n_pairs:
        i, j = np.nonzero(upper)
        ax.axhspan(-2.0, 2.0, color=COLOR_AXIS, alpha=0.4, linewidth=0, label="±2 (95 % of the pairs)")
        ax.axhline(0.0, color=COLOR_EXPECTED, linewidth=1.5)
        ax.plot(j - i, z[i, j], "o", markersize=MARKER_SIZE, color=color, label="Pair of planes")
        outside = np.mean(np.abs(z[i, j]) > 2.0)
        ax.set_title(f"r √n of each pair of planes (expected N(0, 1))\n"
                     f"$\\sum (r\\sqrt{{n}})^2$ = {chi2:.1f} for {n_pairs} pairs: "
                     f"$\\chi^2$/ndof = {chi2 / n_pairs:.2f}, P = {chi2_dist.sf(chi2, n_pairs):.3f}\n"
                     f"{100 * outside:.0f} % of the pairs outside ±2 (expected 5 %)",
                     fontsize=8, color=COLOR_TEXT, loc="left")
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_xlabel("Distance between the two planes (in planes)", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("r √n", fontsize=9, color=COLOR_MUTED)
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

    fig = new_page("Resolution per detector plane: std(fit − true) ● against the error of the fit ○", metadata,
                   True)
    planes = metadata["planes"]
    index = np.arange(len(planes))
    offsets = np.linspace(-0.12, 0.12, len(stages)) if len(stages) > 1 else [0.0]
    # panels in the central column only (margins left and right), one legend for all, below them
    grid = fig.add_gridspec(len(VARIABLES), 3, width_ratios=SIDE_MARGIN)
    axes = [fig.add_subplot(grid[i, 1]) for i in range(len(VARIABLES))]
    for ax in axes:
        style_axes(ax)
        if ax is not axes[0]:
            ax.sharex(axes[0])
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
        top = ax.get_ylim()[1] * 1.1
        ax.set_ylim(-0.06 * top, top)     # a little below 0: points near 0 (e.g. tx after the phantom) not cut
        plane_axis(ax, planes)
        if ax is not axes[-1]:
            ax.tick_params(axis="x", labelbottom=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=len(stages) + 1,
               fontsize=8, frameon=False)
    return fig


# Page 4: per plane, mean and std of the pulls of x, tx and q/p, against 0 and 1
def page_pull_planes(df, metadata, stages):

    fig = new_page("Pulls per detector plane: mean (expected 0) and std (expected 1)", metadata, True)
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


# Page 5: NEES against chi2(3). Top: histogram at one random plane per track (sample), one column
# per stage. Bottom: mean NEES per plane (expected 3; its error is std / sqrt(n), the planes are
# correlated with each other but each point is right on its own)
def page_nees(df, sample, metadata, stages):

    fig = new_page("NEES = (fit − true)$^T$ C$^{-1}$ (fit − true): the full covariance against $\\chi^2$(3)",
                   metadata)
    grid = fig.add_gridspec(3, len(stages), height_ratios=PANEL_ROWS + [1.4])
    for j, st in enumerate(stages):
        name, _ = STAGES[st]
        x = sample[f"nees_{st}"].to_numpy()
        ax = fig.add_subplot(grid[0, j])
        ax_pull = fig.add_subplot(grid[1, j], sharex=ax)
        edges = np.linspace(0, max(np.quantile(x, 0.999), chi2_dist.ppf(0.999, 3)), n_bins(len(x)) + 1)
        expected = len(x) * np.diff(chi2_dist.cdf(edges, 3))
        c2, nd = expected_panel(ax, ax_pull, None, x, edges, expected, xlabel=f"NEES of the {name.lower()}")
        prob = chi2_dist.sf(x, 3)
        ax.set_title(f"{name}, one random plane per track\n"
                     f"mean = {x.mean():.3f} ± {x.std() / np.sqrt(len(x)):.3f} (expected 3)\n"
                     f"mean P($\\chi^2$, 3) = {prob.mean():.3f} (expected 0.5)\n"
                     f"histogram vs $\\chi^2$(3): $\\chi^2$/ndof = {c2:.1f}/{nd} = {c2 / max(nd, 1):.2f}",
                     fontsize=8, color=COLOR_TEXT, loc="left")
        ax.legend(["Fit", "$\\chi^2$(3)"], loc="upper right", fontsize=8, frameon=False)

    planes = metadata["planes"]
    index = np.arange(len(planes))
    offsets = np.linspace(-0.12, 0.12, len(stages)) if len(stages) > 1 else [0.0]
    ax = fig.add_subplot(grid[2, :])
    style_axes(ax)
    titles = []
    for st, offset in zip(stages, offsets):
        name, color = STAGES[st]
        s = per_plane(df, f"nees_{st}", len(planes))
        ax.errorbar(index + offset, s["mean"], yerr=s["mean_err"], fmt="o", markersize=MARKER_SIZE,
                    color=color, label=name)
        titles.append(f"{name}: " + chi2_text(s["mean"], s["mean_err"], 3.0))
    ax.axhline(3.0, color=COLOR_EXPECTED, linewidth=1.5)
    ax.set_title("Mean NEES per plane (expected 3)\n" + "\n".join(titles), fontsize=8, color=COLOR_TEXT, loc="left")
    ax.set_ylabel("Mean NEES", fontsize=9, color=COLOR_MUTED)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncols=len(stages))
    plane_axis(ax, planes, narrow=True)
    fig.text(0.5, 0.005, EXCLUDED_NOTE, ha="center", fontsize=8, color=COLOR_MUTED)
    return fig


# True initial q/p of each fitted track, from the initial beam dataset (input_row is its row there).
# Returns: Series indexed by track_id, or None if the beam dataset cannot be read
def true_initial_qp(tracks, metadata):
    try:
        beam_tracks, _ = load_dataset(metadata["beam"]["input"])
    except (KeyError, FileNotFoundError):
        return None
    return tracks.set_index("track_id")["input_row"].map(beam_tracks["q_over_p0"])


# [MeV] error of the kinetic energy given by an error err of q/p: sigma_T = beta sigma_p (arrays)
def kinetic_energy_error(q_p, err, particle):
    p = np.abs(particle.charge / q_p)
    return (p / np.sqrt(p**2 + particle.mass**2)) * p * np.abs(err / q_p)


# Nominal kinetic energy of the beam and its spread [MeV], from p0 and sigma_p0 of the beam block
def beam_energy(metadata, particle):
    p0, sigma_p0 = metadata["beam"]["p0"], metadata["beam"]["sigma_p0"]
    E0 = np.sqrt(p0**2 + particle.mass**2)
    return E0 - particle.mass, p0 / E0 * sigma_p0


# Range verification of the tracks that stop in the phantom (and stop in the MC): where they stop
# along z, from the fitted kinetic energy at the phantom entrance (CSDA through the layers,
# src.phantom.depth_dose.csda_stop_depth), against the true stop of the propagation (z_final)
# Returns: DataFrame indexed by track_id, [mm]: true; <stage> and <stage>_err (sigma_T dz/dT, numeric)
#          for each stage (NaN if the fitted energy crosses the phantom); prior (CSDA of the nominal beam
#          energy, the same for all); planes (middle of the gap between the last plane with a hit and
#          the next one, what the planes alone say). None without an entrance plane or a true stop
def stop_depths(df, metadata, stages):

    k_in = plane_index_named(metadata["planes"], "D_phantom_start")
    if k_in is None:
        return None
    particle = particle_by_name(metadata["particle"])
    prop_tracks, prop_meta = load_dataset(metadata["propagation"]["input"])
    geometry = make_geometry(prop_meta["geometry"])
    true = prop_tracks.loc[prop_tracks["stopped"]].set_index("track_id")["z_final"]

    def depth(T):
        z = csda_stop_depth(geometry, T, particle)
        return np.nan if z is None else z

    entrance = df[df["stops"] & (df["plane_index"] == k_in)].set_index("track_id")
    entrance = entrance[entrance.index.isin(true.index)]
    if entrance.empty:
        return None
    out = pd.DataFrame({"true": true.loc[entrance.index]})
    h = 0.01      # [MeV] step of the numeric derivative dz/dT
    for st in stages:
        q_p, err = entrance[f"q_over_p_{st}"].to_numpy(), entrance[f"q_over_p_{st}_err"].to_numpy()
        T, sigma_T = kinetic_energy(q_p, particle), kinetic_energy_error(q_p, err, particle)
        out[st] = [depth(t) for t in T]
        out[f"{st}_err"] = sigma_T * np.array([(depth(t + h) - depth(t - h)) / (2 * h) for t in T])
    out["prior"] = depth(beam_energy(metadata, particle)[0])

    planes = metadata["planes"]
    last = df[df["last_of_stop"]].set_index("track_id")["plane_index"].reindex(out.index)
    z_last = np.array([planes[int(k)]["z"] for k in last])
    z_next = np.array([planes[int(k) + 1]["z"] if int(k) + 1 < len(planes) else np.nan for k in last])
    out["planes"] = 0.5 * (z_last + z_next)
    return out


# Page 6: tracks that stop in the phantom. Top left: summary and coverage of the range bounds. Top
# middle: kinetic energy at the phantom entrance, fit - true, against the spread of the beam. Top right:
# stop depth from the fitted energy against the true one (range verification). Bottom:
# mean and std of the pulls of q/p (each stage) and x (smoother) against the hits left to the last one
def page_stopping(df, tracks, metadata, stages):

    fig = new_page("Tracks that stop in the phantom: energy from the range (range constraints)", metadata)
    particle = particle_by_name(metadata["particle"])
    stopping = tracks[tracks["range_end"]]
    rows = df[df["stops"]]
    # top: summary, energy at the entrance, stop depth; bottom: pulls (mean, std)
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1])

    # Summary and coverage: the true initial s = |q|/p must be inside [s_lo, s_hi] (NaN: unbounded)
    ax = fig.add_subplot(grid[0, 0:2])
    ax.axis("off")
    lines = [f"Tracks fitted: {int((tracks['status'] == 'ok').sum())}",
             f"Stop in the phantom (no hit on the plane\nafter the last): {len(stopping)}",
             f"With a range bound on the prior: {int((np.isfinite(tracks['s_hi']) | np.isfinite(tracks['s_lo'])).sum())}",
             f"Safety-net bounds (mean track not\nreaching a hit): {int(tracks['n_reach'].sum())}"]
    qp0 = true_initial_qp(tracks, metadata)
    if qp0 is not None:
        s0 = np.sign(particle.charge) * qp0
        lo = tracks.set_index("track_id")["s_lo"].fillna(-np.inf)
        hi = tracks.set_index("track_id")["s_hi"].fillna(np.inf)
        bounded = tracks.set_index("track_id")["s_lo"].notna() | tracks.set_index("track_id")["s_hi"].notna()
        inside = ((s0 > lo) & (s0 < hi))[bounded]
        lines.append(f"Coverage: true initial 1/p inside its bounds\nin {inside.sum()} / {len(inside)} tracks"
                     + (f" ({100 * inside.mean():.2f} %)" if len(inside) else ""))
        lines.append("(below 100 % only through what the bounds\nneglect: scattering lengthens the path)")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8, color=COLOR_TEXT, transform=ax.transAxes)
    # (here and not as a footnote, which the constrained layout puts over the x labels of the bottom row)
    ax.text(0.0, 0.0, "Pulls below: at the last hit the true energy is\nanywhere between ~1 MeV and the energy to cross\n"
            "the layer. No Gaussian in q/p describes it, so its\nq/p pull is not N(0, 1) (src/kalman/filter.py)",
            va="bottom", ha="left", fontsize=7, color=COLOR_MUTED, transform=ax.transAxes)

    # Energy at the phantom entrance: fit - true, against the spread of the energy of the beam
    entrance = [k for k, p in enumerate(metadata["planes"]) if "phantom_start" in p["name"]]
    ax = fig.add_subplot(grid[0, 2:4])
    style_axes(ax)
    if entrance:
        e = rows[rows["plane_index"] == entrance[0]]
        T_true = kinetic_energy(e["q_over_p_true"].to_numpy(), particle)
        titles = []
        for st in stages:
            name, color = STAGES[st]
            q_p, err = e[f"q_over_p_{st}"].to_numpy(), e[f"q_over_p_{st}_err"].to_numpy()
            T_fit = kinetic_energy(q_p, particle)
            sigma_T = kinetic_energy_error(q_p, err, particle)
            dT = T_fit - T_true
            ax.hist(dT, bins=n_bins(len(dT)), histtype="step", color=color, linewidth=1.5, label=name)
            titles.append(f"{name}: std(fit − true) {dT.std():.2f} MeV,\n   error of the fit "
                          f"{np.sqrt(np.mean(sigma_T**2)):.2f} MeV")
        _, sigma_T_beam = beam_energy(metadata, particle)
        titles.append(f"Beam alone: {sigma_T_beam:.2f} MeV")
        ax.set_title("Kinetic energy at the phantom entrance\n" + "\n".join(titles), fontsize=7,
                     color=COLOR_TEXT, loc="left")
        ax.set_xlabel("$T_\\mathrm{fit} - T_\\mathrm{true}$ [MeV]", fontsize=9, color=COLOR_TEXT)
        ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "No plane at the phantom entrance", ha="center", va="center", color=COLOR_MUTED)

    # Range verification: stop depth from the fitted entrance energy (CSDA) against the true one
    ax = fig.add_subplot(grid[0, 4:6])
    style_axes(ax)
    depths = stop_depths(df, metadata, stages)
    if depths is None:
        ax.text(0.5, 0.5, "No plane at the phantom entrance,\nor no true stop", ha="center", va="center",
                color=COLOR_MUTED)
    else:
        prior = (depths["prior"] - depths["true"]).to_numpy()
        residuals = {st: (depths[st] - depths["true"]).to_numpy() for st in stages}
        values = np.concatenate([prior] + [r[np.isfinite(r)] for r in residuals.values()])
        edges = np.linspace(np.quantile(values, 0.001), np.quantile(values, 0.999), n_bins(len(depths)) + 1)
        ax.hist(prior, edges, color=COLOR_TRUE, label="Beam prior alone")
        titles = [f"Beam prior alone: mean {prior.mean():+.2f}, std {prior.std():.2f}"]
        for st, r in residuals.items():
            name, color = STAGES[st]
            ok = np.isfinite(r)
            ax.hist(r[ok], edges, histtype="step", color=color, linewidth=1.5, label=name)
            titles.append(f"{name}: mean {r[ok].mean():+.2f}, std {r[ok].std():.2f}, "
                          f"error {np.sqrt(np.nanmean(depths[f'{st}_err']**2)):.2f}")
        planes = (depths["planes"] - depths["true"]).to_numpy()
        titles.append(f"Planes alone (middle of the gap where it stops):\n   std {planes.std():.1f}")
        ax.set_title("Stop depth [mm]: CSDA of the fitted entrance energy\n− true (MC)\n" + "\n".join(titles),
                     fontsize=7, color=COLOR_TEXT, loc="left")
        ax.set_xlabel("$z_\\mathrm{stop,\\,fit} - z_\\mathrm{stop,\\,true}$ [mm]", fontsize=9, color=COLOR_TEXT)
        ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
        ax.legend(loc="upper right", fontsize=7, frameon=False)

    # Pulls against the hits left to the last one: N(0, 1) except at the last hit (0)
    n_back = int(rows["hits_to_last"].max()) + 1
    index = np.arange(n_back)
    series = [(f"q/p, {STAGES[st][0].lower()}", f"pull_q_over_p_{st}", STAGES[st][1], "o") for st in stages]
    series.append(("x, " + STAGES[stages[-1]][0].lower(), f"pull_x_{stages[-1]}", COLOR_MUTED, "s"))
    offsets = np.linspace(-0.15, 0.15, len(series))
    for col, (stat, expected, ylabel) in enumerate((("mean", 0.0, "Mean of the pulls"),
                                                    ("std", 1.0, "Std of the pulls"))):
        ax = fig.add_subplot(grid[1, 3 * col:3 * col + 3])
        style_axes(ax)
        for (label, column, color, marker), offset in zip(series, offsets):
            groups = rows.groupby("hits_to_last")[column]
            n = groups.size().reindex(index, fill_value=0).to_numpy()
            mean, std = groups.mean().reindex(index).to_numpy(), groups.std().reindex(index).to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                err = std / np.sqrt(n) if stat == "mean" else std / np.sqrt(2 * (n - 1))
            ax.errorbar(index + offset, mean if stat == "mean" else std, yerr=err, fmt=marker,
                        markersize=MARKER_SIZE, color=color, label=label)
        ax.axhline(expected, color=COLOR_EXPECTED, linewidth=1.5)
        ax.set_xticks(index, ["last hit" if k == 0 else f"{k} before" for k in index], fontsize=7)
        ax.set_xlabel("Hit of the track, counted back from its last one", fontsize=9, color=COLOR_TEXT)
        ax.set_ylabel(ylabel, fontsize=9, color=COLOR_MUTED)
        if col == 0:
            ax.legend(loc="best", fontsize=7, frameon=False)
    return fig


# Removes a Gaussian prior from a posterior, one variable (1D): with 1/post^2 = 1/prior^2 + 1/meas^2 and
# post_mean = post^2 (prior_mean/prior^2 + meas_mean/meas^2), what the measurements alone say is
#   1/meas^2 = 1/post^2 - 1/prior^2,   meas_mean = meas^2 (post_mean/post^2 - prior_mean/prior^2)
# Used for the q/p at the phantom exit: posterior = smoother, prior = filter there (beam prior slowed down
# through the model phantom + the phantom hits, which say almost nothing of q/p), so what is left is the
# measurement of the magnet and the planes after the phantom. 1D: it neglects the correlations of q/p
# with x and tx in the two states (small, checked by the pull of the result against the truth)
# Returns: (meas_mean, meas_sigma), NaN where the prior is not wider than the posterior
def remove_prior(post_mean, post_sigma, prior_mean, prior_sigma):
    post_mean, post_sigma = np.asarray(post_mean, float), np.asarray(post_sigma, float)
    prior_mean, prior_sigma = np.asarray(prior_mean, float), np.asarray(prior_sigma, float)
    info = 1.0 / post_sigma**2 - 1.0 / prior_sigma**2
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = info > 0
        meas_var = np.where(ok, 1.0 / np.where(ok, info, 1.0), np.nan)
        meas_mean = meas_var * (post_mean / post_sigma**2 - prior_mean / prior_sigma**2)
    return meas_mean, np.sqrt(meas_var)


# WEPL of each track that crosses the phantom (page 8b, see the header), or None if the fit has no
# smoother or no plane at the entrance or exit of the phantom, or no track crosses it
# Returns: DataFrame, one row per track: x_out (true x at the exit) [mm], w_prior (weight of the prior in
#          the smoother), pull_magnet (of the magnet-only q/p against the truth), and WEPL [mm]:
#          true; full (smoother, T_in of the beam); magnet (magnet only, T_in of the beam); magnet_ideal
#          (magnet only, true T_in); sigma_full and sigma_magnet (error of T_out of each, in WEPL)
def wepl_per_track(df, metadata):

    if "q_over_p_smooth" not in df:      # the filter at the phantom exit has not seen the magnet yet
        return None
    k_in = plane_index_named(metadata["planes"], "D_phantom_start")
    k_out = plane_index_named(metadata["planes"], "D_phantom_end")
    if k_in is None or k_out is None:
        return None
    particle = particle_by_name(metadata["particle"])
    columns = ["track_id", "x_true", "q_over_p_true", "q_over_p_filt", "q_over_p_filt_err", "q_over_p_smooth",
               "q_over_p_smooth_err"]
    out = df.loc[(df["plane_index"] == k_out) & ~df["stops"], columns]
    both = out.merge(df.loc[df["plane_index"] == k_in, ["track_id", "q_over_p_true"]], on="track_id",
                     suffixes=("", "_in"), validate="one_to_one")
    if both.empty:
        return None

    T_true_in = kinetic_energy(both["q_over_p_true_in"].to_numpy(), particle)
    T_true_out = kinetic_energy(both["q_over_p_true"].to_numpy(), particle)
    T_beam = np.full(len(both), beam_energy(metadata, particle)[0])
    qp_s, err_s = both["q_over_p_smooth"].to_numpy(), both["q_over_p_smooth_err"].to_numpy()
    qp_f, err_f = both["q_over_p_filt"].to_numpy(), both["q_over_p_filt_err"].to_numpy()
    qp_m, err_m = remove_prior(qp_s, err_s, qp_f, err_f)
    T_full, T_magnet = kinetic_energy(qp_s, particle), kinetic_energy(qp_m, particle)
    return pd.DataFrame({
        "track_id": both["track_id"].to_numpy(),
        "x_out": both["x_true"].to_numpy(),
        "true": wepl_from_energies(T_true_in, T_true_out, particle),
        "full": wepl_from_energies(T_beam, T_full, particle),
        "magnet": wepl_from_energies(T_beam, T_magnet, particle),
        "magnet_ideal": wepl_from_energies(T_true_in, T_magnet, particle),
        "sigma_full": wepl_error(T_full, kinetic_energy_error(qp_s, err_s, particle), particle),
        "sigma_magnet": wepl_error(T_magnet, kinetic_energy_error(qp_m, err_m, particle), particle),
        "w_prior": (err_s / err_f)**2,
        "pull_magnet": (qp_m - both["q_over_p_true"].to_numpy()) / err_m,
    })


# Page 8b: proton radiography, WEPL of each track from the fit (see the header). Top left: WEPL
# reconstructed and true. Top right: residual against the expected Gaussian. Bottom left: error budget.
# Bottom right: residual against the true x at the exit of the phantom
def page_wepl(w, metadata):

    fig = new_page("Proton radiography: WEPL = $R_\\mathrm{water}(T_\\mathrm{in}) - R_\\mathrm{water}(T_\\mathrm{out})$ "
                   "of each track that crosses the phantom", metadata)
    particle = particle_by_name(metadata["particle"])
    T_beam, sigma_T_beam = beam_energy(metadata, particle)
    _, prop_meta = load_dataset(metadata["propagation"]["input"])
    wepl_model = wepl_range_definition(make_geometry(prop_meta["geometry"]), T_beam, particle)
    w = w[np.isfinite(w["magnet"])]

    def rms(column):
        return float(np.sqrt(np.mean(w[column]**2)))

    sigma_beam = float(wepl_error(T_beam, sigma_T_beam, particle))
    sigma_full, sigma_magnet = rms("sigma_full"), rms("sigma_magnet")
    # (label, column, expected std of fit - true, color)
    recos = [("Full fit (uses the MODEL phantom)", "full", sigma_full, STAGES["smooth"][1]),
             ("Magnet only, $T_\\mathrm{in}$ of the beam", "magnet", np.hypot(sigma_beam, sigma_magnet), COLOR_MAGNET),
             ("Magnet only, $T_\\mathrm{in}$ true", "magnet_ideal", sigma_magnet, COLOR_IDEAL)]
    residual = {col: (w[col] - w["true"]).to_numpy() for _, col, _, _ in recos}
    grid = fig.add_gridspec(2, 2)

    # (a) WEPL reconstructed and true. The true one is almost a single value (layers of uniform
    # thickness): its peak is cut, so that the reconstructed ones stay readable
    ax = fig.add_subplot(grid[0, 0])
    style_axes(ax)
    values = [w[col] for _, col, _, _ in recos]
    edges = np.linspace(min(np.quantile(v, 0.001) for v in values), max(np.quantile(v, 0.999) for v in values),
                        n_bins(len(w)) + 1)
    ax.hist(w["true"], edges, color=COLOR_TRUE,
            label=f"True (MC): mean {w['true'].mean():.2f} mm, std {w['true'].std():.2f} mm (peak off scale)")
    peak = 0.0
    for label, col, _, color in recos:
        counts, _, _ = ax.hist(w[col], edges, histtype="step", color=color, linewidth=1.5, label=label)
        peak = max(peak, counts.max())
    ax.set_ylim(0, 1.35 * peak)
    ax.axvline(wepl_model, color=COLOR_EXPECTED, linewidth=1.2, linestyle="--",
               label=f"CSDA along z at the beam energy, {T_beam:.1f} MeV: {wepl_model:.2f} mm")
    ax.set_title(f"WEPL of the {len(w)} tracks that cross the phantom", fontsize=8, color=COLOR_TEXT, loc="left")
    ax.set_xlabel("WEPL [mm]", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
    ax.legend(loc="upper left", fontsize=6, frameon=False)

    # (b) residual against the Gaussian of the expected std
    ax = fig.add_subplot(grid[0, 1])
    style_axes(ax)
    limit = max(np.quantile(np.abs(r), 0.999) for r in residual.values())
    edges = np.linspace(-limit, limit, n_bins(len(w)) + 1)
    z = np.linspace(-limit, limit, 400)
    titles = []
    for label, col, expected, color in recos:
        r = residual[col]
        ax.hist(r, edges, histtype="step", color=color, linewidth=1.5, label=label)
        ax.plot(z, len(r) * np.diff(edges)[0] * norm.pdf(z, 0.0, expected), color=color, linewidth=1, linestyle=":")
        titles.append(f"{label.replace(' (uses the MODEL phantom)', '')}: mean {r.mean():+.2f} ± "
                      f"{r.std() / np.sqrt(len(r)):.2f}, std {r.std():.2f} ({100 * r.std() / w['true'].mean():.1f} %), "
                      f"expected {expected:.2f}")
    pull = w["pull_magnet"].to_numpy()
    titles.append(f"Pull of the magnet-only q/p: mean {pull.mean():+.3f}, std {pull.std():.3f} (expected 0, 1)")
    ax.set_title("WEPL fit − true [mm] (dotted: Gaussian of the expected std)\n" + "\n".join(titles),
                 fontsize=7, color=COLOR_TEXT, loc="left")
    ax.set_xlabel("$\\mathrm{WEPL}_\\mathrm{fit} - \\mathrm{WEPL}_\\mathrm{true}$ [mm]", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
    ax.legend(loc="upper right", fontsize=6, frameon=False)

    # (c) resolution of each reconstruction: observed std (bar, colour of the reconstruction) against
    # the expected one (black line), with how the expected one is made under its name
    ax = fig.add_subplot(grid[1, 0])
    style_axes(ax)
    how = {"full": ("Full fit (model phantom)",
                    f"error of $T_\\mathrm{{out}}$ only, {sigma_full:.2f}: the $T_\\mathrm{{in}}$ error cancels"),
           "magnet": ("Magnet only, $T_\\mathrm{in}$ of the beam",
                      f"√(beam {sigma_beam:.2f}² + magnet {sigma_magnet:.2f}²)"),
           "magnet_ideal": ("Magnet only, $T_\\mathrm{in}$ true", f"magnet {sigma_magnet:.2f} only")}
    y = np.arange(len(recos))[::-1]
    for yi, (_, col, expected, color) in zip(y, recos):
        observed = residual[col].std()
        ax.barh(yi, observed, height=0.5, color=color)
        ax.plot([expected, expected], [yi - 0.36, yi + 0.36], color=COLOR_TEXT, linewidth=2)
        ax.text(max(observed, expected), yi, f"  {observed:.2f} (expected {expected:.2f})", va="center",
                fontsize=7, color=COLOR_TEXT)
    ax.set_yticks(y, [f"{how[col][0]}\n{how[col][1]}" for _, col, _, _ in recos], fontsize=7)
    ax.set_xlim(0, 1.45 * max(max(residual[col].std(), e) for _, col, e, _ in recos))
    ax.set_ylim(-0.6, len(recos) - 0.4)
    ax.set_xlabel("Std of WEPL fit − true [mm]", fontsize=9, color=COLOR_TEXT)
    ax.legend(handles=[Patch(facecolor=COLOR_MUTED, label="Observed: std(fit − true)"),
                       Line2D([], [], color=COLOR_TEXT, linewidth=2, label="Expected from the errors")],
              loc="lower right", fontsize=7, frameon=False)
    ax.set_title(f"Resolution of each reconstruction: observed (bar) vs expected (line)\n"
                 f"Each error source in WEPL: $\\sigma_T$ / $S_\\mathrm{{water}}(T)$ (beam: $\\sigma_T$ = "
                 f"{sigma_T_beam:.2f} MeV, magnet: from the fit)\n"
                 f"Full fit: {100 * w['w_prior'].mean():.0f} % of its $T_\\mathrm{{out}}$ comes from the beam prior,\n"
                 f"slowed down by the MODEL phantom: not a measurement", fontsize=7, color=COLOR_TEXT, loc="left")

    # (d) residual against the true x at the exit: mean ± std in bins of equal population
    ax = fig.add_subplot(grid[1, 1])
    style_axes(ax)
    x = w["x_out"].to_numpy()
    bins = np.unique(np.quantile(x, np.linspace(0, 1, N_X_BINS + 1)))
    which = np.clip(np.searchsorted(bins, x, side="right") - 1, 0, len(bins) - 2)
    centers = np.array([x[which == b].mean() for b in range(len(bins) - 1)])
    step = 0.15 * np.min(np.diff(centers)) if len(centers) > 1 else 0.1
    for (label, col, _, color), offset in zip(recos, (-step, 0.0, step)):
        r = residual[col]
        mean = np.array([r[which == b].mean() for b in range(len(bins) - 1)])
        std = np.array([r[which == b].std() for b in range(len(bins) - 1)])
        ax.errorbar(centers + offset, mean, yerr=std, fmt="o", markersize=MARKER_SIZE, color=color, capsize=2,
                    label=label)
    ax.axhline(0.0, color=COLOR_EXPECTED, linewidth=1.2)
    ax.set_title(f"WEPL fit − true against the true x at the exit ({len(bins) - 1} bins of equal population)\n"
                 "points: mean, bars: std. Flat for a phantom of uniform layers", fontsize=7, color=COLOR_TEXT,
                 loc="left")
    ax.set_xlabel("x at the exit of the phantom [mm]", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("WEPL fit − true [mm]", fontsize=9, color=COLOR_MUTED)
    # below the axis, out of the points (short names: the colours are those of the other panels)
    short = {"full": "Full fit", "magnet": "Magnet, $T_\\mathrm{in}$ of the beam",
             "magnet_ideal": "Magnet, $T_\\mathrm{in}$ true"}
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [short[col] for _, col, _, _ in recos], loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncols=len(recos), fontsize=7, frameon=False)
    return fig


# Page 7: a few tracks chosen at random (seed): fit - true of x and q/p along z, with ±1 sigma bands
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
    df = mark_stopping(fit_with_truth(fit, metadata, stages), tracks)
    df["innovation"] = df["res_pred"] / df["res_pred_err"]     # r / sqrt(S) of the filter (page 4b)
    main = df[~df["last_of_stop"]]        # see the header

    sample = one_per_track(main, seed)

    # All the pages are built first: a failure does not leave a partial pdf behind
    pages = [page_pulls(sample, metadata, stages)]
    pages += [page_fit_vs_true(main, metadata, stages, element, names)
              for element, names in FIT_VS_TRUE_PAGES.items()]
    pages.append(page_chi2(tracks, metadata))
    pages.append(page_innovations(main, metadata))
    if has_correlations(df, stages):
        pages.append(page_nees(main, sample, metadata, stages))
    else:
        warnings.warn(f"{fit_dir} does not store the correlations of the covariance (fit made before they "
                      "were added): no NEES page. Redo the fit to get it")
    pages += [page_resolution(main, metadata, stages), page_pull_planes(main, metadata, stages)]
    if df["stops"].any():
        pages.append(page_stopping(df, tracks, metadata, stages))
    wepl = wepl_per_track(df, metadata)
    if wepl is not None:
        pages.append(page_wepl(wepl, metadata))
    elif not metadata["smoother"]:
        warnings.warn(f"{fit_dir} has no smoother: no WEPL page (the filter at the phantom exit has not seen "
                      "the magnet). Redo the fit with kalman.smoother: true to get it")
    pages.append(page_examples(df, metadata, stages, n_tracks, seed))

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

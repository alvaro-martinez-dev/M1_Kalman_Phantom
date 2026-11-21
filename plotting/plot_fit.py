# plotting/plot_fit.py
#
# Validation report of a Kalman fit (data/fit/, see scripts/generate_fit.py): a single multi-page
# pdf that compares the fitted states with the MC truth of the propagation the hits come from
# (joined by track_id, plane_index; the truth is read from the propagation, the fit never sees it).
# Filter (filt) and smoother (smooth, if the fit has it) are drawn side by side:
#   1. Pulls (fit - true) / sigma_fit of x, tx and q/p against N(0, 1), one page per stage: the
#      errors of the fit are right if the pulls are N(0, 1). ONE plane per track, chosen at random:
#      the pulls of the planes of a track are correlated (q/p is a single number per track, the
#      smoother uses the same hits at every plane), so a histogram of all the planes would not
#      have independent entries and its chi2 against N(0, 1) would mean nothing
#   2. chi2 of the tracks against chi2(ndf), and its probability against a flat distribution
#   3. Per detector plane: resolution std(fit - true) against the error the fit gives, sqrt(<sigma^2>)
#   4. Per detector plane: mean and std of the pulls, against 0 and 1
#   5. NEES = (fit - true)^T C^-1 (fit - true) against chi2(3): tests the whole covariance, also the
#      correlations that the pulls do not see. Histogram at one plane per track, and mean per plane
#      (only for fits that store the correlations, corr_* columns of src/kalman/fit.py)
#   6. Tracks that stop in the phantom (fits with range_end, see src/kalman/fit.py): how many, the
#      coverage of the range bounds (true initial 1/p inside [s_lo, s_hi]), the energy resolution at
#      the phantom entrance against that of the beam, and the pulls against the number of hits
#      before the last one
#   7. A few tracks: fit - true of x and q/p along z with their ±1 sigma bands, and the hits
#
# The LAST hit of a track that stops is left out of pages 1, 3, 4 and 5: there the true energy is
# anywhere between ~1 MeV and the energy to cross the layer, which no Gaussian in q/p describes (see
# src/kalman/filter.py), and it would hide how the rest of the fit behaves. Page 6 shows it apart.
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
                                     PANEL_ROWS, expected_panel, n_bins, style_axes)
from plotting.plot_layout import COLORS_MATERIAL
from src.kalman.fit import CORR_PAIRS
from src.propagator.propagator import kinetic_energy
from src.utils.config_loader import particle_by_name
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


# Page 1 (one per stage): pulls of x, tx and q/p against N(0, 1), at one random plane per track
# sample: one_per_track(df, seed), the same rows for every stage
def page_pulls(sample, metadata, stage):

    fig = new_page(f"Pulls of the {STAGES[stage][0].lower()} (one random plane per track)", metadata, True)
    grid = fig.add_gridspec(2, 3, height_ratios=PANEL_ROWS)
    for j, (var, (label, _, _)) in enumerate(VARIABLES.items()):
        ax = fig.add_subplot(grid[0, j])
        ax_pull = fig.add_subplot(grid[1, j], sharex=ax)
        gaussian_panel(ax, ax_pull, sample[f"pull_{var}_{stage}"].to_numpy(), 0.0, 1.0,
                       f"({label}$_\\mathrm{{fit}}$ − {label}$_\\mathrm{{true}}$) / $\\sigma_\\mathrm{{fit}}$")
        ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
        if j == 0:
            ax.legend(["Fit", "N(0, 1)"], loc="upper right", fontsize=8, frameon=False)
    fig.text(0.5, 0.02, "One entry per track, at a plane chosen at random: the planes of a track are correlated "
             "(q/p is one number per track), so only one of them gives independent entries. "
             "Per plane: page 4", ha="center", fontsize=8, color=COLOR_MUTED)
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


# Page 6: tracks that stop in the phantom. Top left: summary and coverage of the range bounds. Top
# right: kinetic energy at the phantom entrance, fit - true, against the spread of the beam. Bottom:
# mean and std of the pulls of q/p (each stage) and x (smoother) against the hits left to the last one
def page_stopping(df, tracks, metadata, stages):

    fig = new_page("Tracks that stop in the phantom: energy from the range (range constraints)", metadata)
    particle = particle_by_name(metadata["particle"])
    stopping = tracks[tracks["range_end"]]
    rows = df[df["stops"]]
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Summary and coverage: the true initial s = |q|/p must be inside [s_lo, s_hi] (NaN: unbounded)
    ax = fig.add_subplot(grid[0, 0])
    ax.axis("off")
    lines = [f"Tracks fitted: {int((tracks['status'] == 'ok').sum())}",
             f"Stop in the phantom (no hit on the plane after the last): {len(stopping)}",
             f"With a range bound on the prior: {int((np.isfinite(tracks['s_hi']) | np.isfinite(tracks['s_lo'])).sum())}",
             f"Safety-net bounds (mean track not reaching a hit): {int(tracks['n_reach'].sum())}"]
    qp0 = true_initial_qp(tracks, metadata)
    if qp0 is not None:
        s0 = np.sign(particle.charge) * qp0
        lo = tracks.set_index("track_id")["s_lo"].fillna(-np.inf)
        hi = tracks.set_index("track_id")["s_hi"].fillna(np.inf)
        bounded = tracks.set_index("track_id")["s_lo"].notna() | tracks.set_index("track_id")["s_hi"].notna()
        inside = ((s0 > lo) & (s0 < hi))[bounded]
        lines.append(f"Coverage: true initial 1/p inside its bounds in {inside.sum()} / {len(inside)} tracks"
                     + (f" ({100 * inside.mean():.2f} %)" if len(inside) else ""))
        lines.append("(below 100 % only through what the bounds neglect: scattering lengthens the path)")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, color=COLOR_TEXT, transform=ax.transAxes)
    # (here and not as a footnote, which the constrained layout puts over the x labels of the bottom row)
    ax.text(0.0, 0.0, "Pulls below: at the last hit the true energy is anywhere between ~1 MeV and the\n"
            "energy to cross the layer. No Gaussian in q/p describes it, so its q/p pull is not\n"
            "N(0, 1) (src/kalman/filter.py)", va="bottom", ha="left", fontsize=8, color=COLOR_MUTED,
            transform=ax.transAxes)

    # Energy at the phantom entrance: fit - true, against the spread of the energy of the beam
    entrance = [k for k, p in enumerate(metadata["planes"]) if "phantom_start" in p["name"]]
    ax = fig.add_subplot(grid[0, 1])
    style_axes(ax)
    if entrance:
        e = rows[rows["plane_index"] == entrance[0]]
        T_true = kinetic_energy(e["q_over_p_true"].to_numpy(), particle)
        titles = []
        for st in stages:
            name, color = STAGES[st]
            q_p, err = e[f"q_over_p_{st}"].to_numpy(), e[f"q_over_p_{st}_err"].to_numpy()
            T_fit = kinetic_energy(q_p, particle)
            p = np.abs(particle.charge / q_p)
            sigma_T = (p / np.sqrt(p**2 + particle.mass**2)) * p * np.abs(err / q_p)    # beta * sigma_p
            dT = T_fit - T_true
            ax.hist(dT, bins=n_bins(len(dT)), histtype="step", color=color, linewidth=1.5, label=name)
            titles.append(f"{name}: std(fit − true) = {dT.std():.2f} MeV, error of the fit = "
                          f"{np.sqrt(np.mean(sigma_T**2)):.2f} MeV")
        beam = metadata["beam"]
        p0 = beam["p0"]
        sigma_T_beam = p0 / np.sqrt(p0**2 + particle.mass**2) * beam["sigma_p0"]
        titles.append(f"Beam alone: {sigma_T_beam:.2f} MeV")
        ax.set_title("Kinetic energy at the phantom entrance\n" + "\n".join(titles), fontsize=8,
                     color=COLOR_TEXT, loc="left")
        ax.set_xlabel("$T_\\mathrm{fit} - T_\\mathrm{true}$ [MeV]", fontsize=10, color=COLOR_TEXT)
        ax.set_ylabel("Tracks / bin", fontsize=9, color=COLOR_MUTED)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "No plane at the phantom entrance", ha="center", va="center", color=COLOR_MUTED)

    # Pulls against the hits left to the last one: N(0, 1) except at the last hit (0)
    n_back = int(rows["hits_to_last"].max()) + 1
    index = np.arange(n_back)
    series = [(f"q/p, {STAGES[st][0].lower()}", f"pull_q_over_p_{st}", STAGES[st][1], "o") for st in stages]
    series.append(("x, " + STAGES[stages[-1]][0].lower(), f"pull_x_{stages[-1]}", COLOR_MUTED, "s"))
    offsets = np.linspace(-0.15, 0.15, len(series))
    for col, (stat, expected, ylabel) in enumerate((("mean", 0.0, "Mean of the pulls"),
                                                    ("std", 1.0, "Std of the pulls"))):
        ax = fig.add_subplot(grid[1, col])
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
    main = df[~df["last_of_stop"]]        # see the header

    sample = one_per_track(main, seed)

    # All the pages are built first: a failure does not leave a partial pdf behind
    pages = [page_pulls(sample, metadata, st) for st in stages]
    pages += [page_chi2(tracks, metadata), page_resolution(main, metadata, stages),
              page_pull_planes(main, metadata, stages)]
    if has_correlations(df, stages):
        pages.append(page_nees(main, sample, metadata, stages))
    else:
        warnings.warn(f"{fit_dir} does not store the correlations of the covariance (fit made before they "
                      "were added): no NEES page. Redo the fit to get it")
    if df["stops"].any():
        pages.append(page_stopping(df, tracks, metadata, stages))
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

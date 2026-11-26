# plotting/plot_bragg.py
#
# Bragg curve and dose of a propagation dataset (data/propagation/...): its tracks are propagated again
# through the phantom on a fine grid of z (src/phantom/depth_dose.py: same random streams, so the same
# tracks) and the energy they lose is deposited along z and x. One pdf, bragg.pdf, in the dataset folder:
#   1. Depth-dose curve of the Monte Carlo against the analytic CSDA curve of a single proton of the mean
#      beam energy along z (src/phantom/dose.py:dose_profile), with the layers of the phantom and the
#      kinetic energy the proton has left, T(z) (analytic): it falls with depth, while the dose (the
#      energy it loses per unit mass) grows up to the peak. The Monte Carlo adds the energy spread of the beam and the multiple
#      scattering (longer paths); like the analytic curve it has no straggling and no nuclear
#      interactions, so its peak is sharper than a measured one. Protons that cross the phantom only
#      give the entrance region (no peak)
#   2. Energy deposited in the (z, x) plane, and lateral profiles at a few depths: the beam widens with
#      the multiple scattering
#   3. Depth where the protons stop against the CSDA range, with the spread expected from the energy
#      spread of the beam alone
#
# Written by scripts/generate_propagation.py after each propagation (if run.plot is true), with the
# propagator.bragg block of the yaml (n_tracks, dz, dx). That block is kept in the metadata of the
# propagation, so redoing the report uses the same values; the command line overrides them.
#
# Usage (from the repo root), to redo the report of an existing propagation:
#   python -m plotting.plot_bragg propagation/<name>
#   python -m plotting.plot_bragg propagation/<name> --n-tracks 500 --dz 0.5

import argparse
import warnings
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

from plotting.plot_init_beam import COLOR_AXIS, COLOR_DATA, COLOR_EXPECTED, COLOR_MUTED, COLOR_TEXT, PAGE_SIZE, \
    n_bins, style_axes
from plotting.plot_layout import COLORS_MATERIAL, FILL_ALPHA
from plotting.plot_tracks import rebuild_system
from src.phantom.depth_dose import csda_stop_depth, depth_dose
from src.phantom.dose import dose_profile
from src.propagator.propagator import kinetic_energy, momentum
from src.utils.dataset_io import dataset_path, load_dataset
from src.utils.validation import check_count

REPORT_NAME = "bragg.pdf"
DOSE_SCALE = 1e9            # Gy -> nGy per proton/cm^2 on the plots
COLOR_ENERGY = COLORS_MATERIAL[5]    # the kinetic energy and its axis (page 1)
# Defaults of the propagator.bragg block of the yaml (used if the propagation metadata has none)
DEFAULTS = {"n_tracks": 2000, "dz": 0.25, "dx": 0.5}
N_PROFILES = 4              # lateral profiles of page 2, from the entrance to the peak
X_QUANTILE = 0.001          # x range of the map: the deposits between this quantile and 1 - it
Z_MARGIN = 0.15             # page 2: if the protons stop, z shown up to the last deposit + this
                            # fraction of the depth with deposits (page 1 always shows the whole phantom)


# [z_lo, z_hi] shown on page 2: the whole phantom, or up to a little after the last deposit if the
# protons stop in it (the rest of the phantom would be an empty half of the map)
def z_view(dd):
    last = dd.z_edges[np.flatnonzero(dd.energy > 0)[-1] + 1]
    if np.isnan(dd.z_stop).all():
        return dd.z_edges[0], dd.z_edges[-1]
    return dd.z_edges[0], min(dd.z_edges[-1], last + Z_MARGIN * (last - dd.z_edges[0]))


# Header of every page: what dataset, how many tracks, the beam and the grid
def new_page(title, info):
    fig = Figure(figsize=PAGE_SIZE, layout="constrained")
    fig.suptitle(f"{title}\n{info}", fontsize=11, color=COLOR_TEXT)
    return fig


# Layers of the phantom shaded, a color per material (order of first appearance, as plot_layout)
def shade_layers(ax, geometry):
    colors, handles = {}, []
    interfaces = geometry.z_phantom_interfaces
    for layer, z_a, z_b in zip(geometry.phantom_layers, interfaces[:-1], interfaces[1:]):
        name = layer["material"]
        if name not in colors:
            colors[name] = COLORS_MATERIAL[len(colors) % len(COLORS_MATERIAL)]
            handles.append(ax.axvspan(z_a, z_b, color=colors[name], alpha=FILL_ALPHA * 0.5, linewidth=0,
                                      label=name.lower()))
        else:
            ax.axvspan(z_a, z_b, color=colors[name], alpha=FILL_ALPHA * 0.5, linewidth=0)
    return handles


# Page 1: depth-dose of the Monte Carlo against the analytic curve of one proton of the mean energy
def page_depth_dose(dd, geometry, particle, T_beam, info):
    fig = new_page("Depth-dose (Bragg) curve: Monte Carlo against the analytic CSDA curve", info)
    ax = fig.add_subplot()
    style_axes(ax)
    layers = shade_layers(ax, geometry)

    z_a, T_a, D_a = dose_profile(geometry, T_beam, particle, n_steps=len(dd.z_edges) - 1)
    mc = ax.stairs(dd.dose * DOSE_SCALE, dd.z_edges, color=COLOR_DATA, linewidth=1.8,
                   label=f"Monte Carlo ({dd.n_tracks} tracks: beam spread + scattering)")
    an = ax.stairs(D_a[:-1] * DOSE_SCALE, z_a, color=COLOR_EXPECTED, linewidth=1.2, linestyle="--",
                   label=f"Analytic CSDA, one proton of {T_beam:.1f} MeV along z")
    ax.set_xlabel("Depth z [mm]", fontsize=10, color=COLOR_TEXT)
    ax.set_ylabel("Dose = energy lost per unit mass [nGy per proton/cm²]", fontsize=10, color=COLOR_TEXT)
    ax.set_xlim(dd.z_edges[0], dd.z_edges[-1])      # the whole phantom: every layer shown, a check of the layout
    ax.set_ylim(0, None)

    # kinetic energy LEFT to the proton (it falls with depth), on its own axis, in its own color
    ax_T = ax.twinx()
    ax_T.plot(z_a[T_a > 0], T_a[T_a > 0], color=COLOR_ENERGY, linewidth=1.2,
              label="Kinetic energy remaining, T(z) [MeV]: decreases with depth (analytic)")
    ax_T.set_ylabel("Kinetic energy remaining, T(z) [MeV]", fontsize=10, color=COLOR_ENERGY)
    ax_T.tick_params(labelsize=8, colors=COLOR_ENERGY)
    ax_T.spines["right"].set_color(COLOR_ENERGY)
    ax_T.set_ylim(0, None)

    lines = ["The proton loses energy as it goes (T falls): the slower it is, the more it loses per mm, so the "
             "dose grows up to the peak, where it stops"]
    z_csda = csda_stop_depth(geometry, T_beam, particle)
    stopped = ~np.isnan(dd.z_stop)
    if stopped.any():
        i_peak = int(np.argmax(dd.dose))
        ax.axvline(dd.z_mid[i_peak], color=COLOR_DATA, linewidth=0.8, linestyle=":")
        lines.append(f"Monte Carlo: peak at z = {dd.z_mid[i_peak]:.2f} mm, "
                     f"{stopped.sum()} / {dd.n_tracks} protons stop, mean stop z = {np.nanmean(dd.z_stop):.2f} mm")
        lines.append(f"peak / entrance dose = {dd.dose[i_peak] / dd.dose[0]:.2f} (Monte Carlo), "
                     f"{np.max(D_a) / D_a[0]:.2f} (analytic)")
    else:
        lines.append(f"No proton stops in the phantom: only the entrance region of the curve (no peak). "
                     f"Exit energy, analytic: {T_a[-1]:.1f} MeV")
    if z_csda is not None:
        ax.axvline(z_csda, color=COLOR_EXPECTED, linewidth=0.8, linestyle=":")
        lines.append(f"Analytic: stops at z = {z_csda:.2f} mm (CSDA range)")
    ax.set_title("\n".join(lines), fontsize=9, color=COLOR_TEXT, loc="left")
    handles = [mc, an] + ax_T.get_legend_handles_labels()[0] + layers
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=8, frameon=False)
    fig.supxlabel("Dose of a broad beam of unit fluence (energy per mass of each bin). CSDA in both: no "
                  "straggling and no nuclear interactions, so the peak is sharper than a measured one",
                  fontsize=8, color=COLOR_MUTED)
    return fig


# Page 2: energy deposited in (z, x), and lateral profiles at N_PROFILES depths
def page_lateral(dd, geometry, dx, info):
    fig = new_page("Energy deposited in the (z, x) plane and lateral profiles", info)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])

    order = np.argsort(dd.x_mid)
    cum = np.cumsum(dd.dE[order]) / dd.dE.sum()
    x_lo = dd.x_mid[order][np.searchsorted(cum, X_QUANTILE)]
    x_hi = dd.x_mid[order][min(np.searchsorted(cum, 1 - X_QUANTILE), len(cum) - 1)]
    x_edges = np.arange(np.floor(x_lo / dx) * dx, np.ceil(x_hi / dx) * dx + dx / 2, dx)
    if len(x_edges) < 3:
        x_edges = np.linspace(x_lo - dx, x_hi + dx, 4)
    emap = dd.energy_map(x_edges)                                   # MeV per proton
    density = emap / (np.diff(dd.z_edges)[:, None] * np.diff(x_edges)[None, :])   # MeV / mm^2

    ax = fig.add_subplot(grid[0])
    image = ax.pcolormesh(dd.z_edges, x_edges, density.T, cmap="magma_r", shading="flat")
    for z in geometry.z_phantom_interfaces[1:-1]:
        ax.axvline(z, color=COLOR_AXIS, linewidth=0.8, linestyle="--")
    fig.colorbar(image, ax=ax, label="Energy deposited per proton [MeV/mm²]").ax.tick_params(labelsize=8)
    ax.set_xlim(*z_view(dd))
    ax.set_xlabel("Depth z [mm]", fontsize=10, color=COLOR_TEXT)
    ax.set_ylabel("x [mm]", fontsize=10, color=COLOR_TEXT)
    ax.tick_params(labelsize=8, colors=COLOR_MUTED)
    ax.set_title(f"x range: {100 * (1 - 2 * X_QUANTILE):.1f} % of the energy; dashed: interfaces",
                 fontsize=9, color=COLOR_TEXT, loc="left")

    # profiles from the entrance to the peak of the dose (beyond it few protons are left: noisy profiles)
    ax = fig.add_subplot(grid[1])
    style_axes(ax)
    picks = np.unique(np.linspace(0, int(np.argmax(dd.dose)), N_PROFILES).astype(int))
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    titles = []
    for color, k in zip(COLORS_MATERIAL, picks):
        profile = emap[k]
        if profile.sum() <= 0:
            continue
        mean = np.average(x_mid, weights=profile)
        sigma = np.sqrt(np.average((x_mid - mean)**2, weights=profile))
        ax.stairs(profile / profile.max(), x_edges, color=color, linewidth=1.5, label=f"z = {dd.z_mid[k]:.1f} mm")
        titles.append(f"z = {dd.z_mid[k]:.1f} mm: rms width {sigma:.2f} mm")
    ax.set_xlabel("x [mm]", fontsize=10, color=COLOR_TEXT)
    ax.set_ylabel("Energy deposited (normalised to its maximum)", fontsize=9, color=COLOR_MUTED)
    ax.set_title("Lateral profiles: " + "  |  ".join(titles), fontsize=9, color=COLOR_TEXT, loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    return fig


# Page 3: where the protons stop, against the CSDA range of the mean beam energy and the spread that
# the energy spread of the beam alone would give
def page_stopping(dd, geometry, particle, beam, T_beam, info):
    fig = new_page("Depth where the protons stop", info)
    ax = fig.add_subplot()
    style_axes(ax)
    z_stop = dd.z_stop[~np.isnan(dd.z_stop)]
    if len(z_stop) == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"No proton stops in the phantom (all {dd.n_tracks} cross it)", ha="center",
                va="center", fontsize=12, color=COLOR_MUTED, transform=ax.transAxes)
        return fig

    layers = shade_layers(ax, geometry)
    ax.hist(z_stop, bins=n_bins(len(z_stop)), histtype="step", color=COLOR_DATA, linewidth=1.8,
            label=f"Monte Carlo: {len(z_stop)} protons")
    lines = [f"Monte Carlo: mean {z_stop.mean():.2f} mm, std {z_stop.std(ddof=1):.2f} mm"]
    z_csda = csda_stop_depth(geometry, T_beam, particle)
    if z_csda is not None:
        ax.axvline(z_csda, color=COLOR_EXPECTED, linewidth=1.5, label=f"CSDA range of {T_beam:.1f} MeV")
        lines.append(f"Analytic (CSDA, along z): {z_csda:.2f} mm. The scattering lengthens the path, so the "
                     "protons stop slightly before it")
    if beam is not None:
        # spread from the energy spread of the beam alone: stop depth at p0 +- sigma_p0
        T_lo, T_hi = (kinetic_energy(particle.charge / (beam["p0"] + s * beam["sigma_p0"]), particle) for s in (-1, 1))
        z_lo, z_hi = csda_stop_depth(geometry, T_lo, particle), csda_stop_depth(geometry, T_hi, particle)
        if z_lo is not None and z_hi is not None:
            lines.append(f"Spread expected from the energy spread of the beam alone (σ_p = {beam['sigma_p0']:g} "
                         f"MeV/c): {0.5 * (z_hi - z_lo):.2f} mm")
    # zoom on the stops (and the CSDA line), the layers stay shaded around them
    lo, hi = z_stop.min(), z_stop.max()
    if z_csda is not None:
        lo, hi = min(lo, z_csda), max(hi, z_csda)
    pad = max(0.5, 0.3 * (hi - lo))
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("z where the proton stops [mm]", fontsize=10, color=COLOR_TEXT)
    ax.set_ylabel("Protons / bin", fontsize=10, color=COLOR_MUTED)
    ax.set_title("\n".join(lines), fontsize=9, color=COLOR_TEXT, loc="left")
    handles = ax.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=8, frameon=False)
    return fig


# Parameters of the report of a propagation: the propagator.bragg block it was generated with (in its
# metadata), else DEFAULTS. Explicit values (not None) override them. n_tracks None in the block: all
def bragg_parameters(meta, n_tracks=None, dz=None, dx=None):
    block = (meta.get("propagator") or {}).get("bragg") or {}
    params = {key: block.get(key, default) for key, default in DEFAULTS.items()}
    params.update({k: v for k, v in (("n_tracks", n_tracks), ("dz", dz), ("dx", dx)) if v is not None})
    return params


# prop_path: propagation dataset (relative to data/ or absolute)
# n_tracks: tracks propagated again (the first n of the dataset, a random sample of the beam)
# dz, dx: [mm] bins in depth and in x
#   (None: the propagator.bragg block of the propagation, see bragg_parameters)
# n_workers: processes (None: half the CPUs)
# out_path: output pdf (default: REPORT_NAME in the dataset folder)
# Returns: out_path as a Path
def plot_bragg(prop_path, n_tracks=None, dz=None, dx=None, n_workers=None, out_path=None, progress=True):

    prop_dir = dataset_path(prop_path)
    tracks, meta = load_dataset(prop_dir)
    params = bragg_parameters(meta, n_tracks, dz, dx)
    n_tracks, dz, dx = params["n_tracks"], params["dz"], params["dx"]
    input_tracks, input_meta = load_dataset(meta["input"])
    if input_meta.get("created") != meta.get("input_created"):
        warnings.warn(f"{meta['input']} was regenerated after this propagation: the tracks may differ")
    propagator, _ = rebuild_system(meta)
    if not propagator.geometry.has_phantom:
        raise ValueError(f"{prop_dir.name} has no phantom: there is no dose to compute")
    particle = propagator.particle
    kick_seed = meta["seed"] if propagator.max_slice is not None else None

    n_tracks = len(tracks) if n_tracks is None else min(check_count("n_tracks", n_tracks), len(tracks))
    rows = tracks.sort_values("track_id")["input_row"].to_numpy()[:n_tracks]
    with tqdm(total=n_tracks, unit="track", desc="Dose", disable=not progress, dynamic_ncols=True) as bar:
        dd = depth_dose(propagator, input_tracks, rows, kick_seed, dz, n_workers=n_workers, progress=bar.update)

    beam = input_meta.get("beam")
    T_beam = (kinetic_energy(particle.charge / beam["p0"], particle) if beam is not None
              else float(np.mean(dd.T_entrance)))
    commit = str(meta.get("git_commit", "?"))[:7] + (" + uncommitted changes" if meta.get("git_uncommitted_changes") else "")
    # energy spread of the beam: sigma_T = beta sigma_p
    beta = momentum(T_beam, particle) / (T_beam + particle.mass)
    spread = f" ± {beta * beam['sigma_p0']:.2f}" if beam is not None else ""
    info = (f"{particle.name}, T = {T_beam:.1f}{spread} MeV  |  {dd.n_tracks} of the {len(tracks)} tracks  |  "
            f"scattering {'on' if kick_seed is not None else 'off'}  |  dz = {dz:g} mm\n"
            f"from {prop_dir.name}\ncreated {meta.get('created')}  |  commit {commit}")

    pages = [page_depth_dose(dd, propagator.geometry, particle, T_beam, info),
             page_lateral(dd, propagator.geometry, dx, info),
             page_stopping(dd, propagator.geometry, particle, beam, T_beam, info)]
    out_path = Path(out_path) if out_path is not None else prop_dir / REPORT_NAME
    with PdfPages(out_path) as pdf:
        for fig in pages:
            pdf.savefig(fig)
    return out_path


def main():

    parser = argparse.ArgumentParser(description="Redo the Bragg curve and dose report of a propagation dataset.")
    parser.add_argument("dataset", help="folder of the propagation (absolute, or relative to data/)")
    parser.add_argument("--n-tracks", type=int, default=None,
                        help="tracks propagated again (default: propagator.bragg.n_tracks of the propagation)")
    parser.add_argument("--dz", type=float, default=None, help="[mm] depth bins (default: propagator.bragg.dz)")
    parser.add_argument("--dx", type=float, default=None, help="[mm] x bins of the map (default: propagator.bragg.dx)")
    parser.add_argument("--workers", type=int, default=None, help="processes (default: half the CPUs)")
    parser.add_argument("--out", default=None, help=f"output pdf (default: {REPORT_NAME} in the dataset folder)")
    args = parser.parse_args()

    out = plot_bragg(args.dataset, n_tracks=args.n_tracks, dz=args.dz, dx=args.dx, n_workers=args.workers,
                     out_path=args.out)
    print(f"Saved report to {out}")


if __name__ == "__main__":
    main()

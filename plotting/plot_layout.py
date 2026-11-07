# plotting/plot_layout.py
#
# Report of a layout folder of data/geometry/ (see scripts/generate_layout.py): a pdf with
#   1. Schematic of the system along z: phantom layers (coloured by material), magnet,
#      physical boundaries and detector planes
#   2. The layout table itself, to read the exact positions
#
# Usage (from the repo root), to redo the report of an existing layout:
#   python -m plotting.plot_layout geometry/layout_11Det_2Lay0PhDet_2MagDet_4b77bc

import argparse
from pathlib import Path

import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.utils.dataset_io import DATA_DIR


REPORT_NAME = "report.pdf"

PAGE_SIZE = (11.69, 8.27)   # A4 landscape, in inches
MARGIN_RATIOS = [1, 8, 1]   # margin / schematic / margin of the page 1, in both directions
FONTSIZE_PLANES = 9         # names of the detector planes and boundaries
X_RANGE = 2.5               # half height of the schematic, in units of the magnet x_half_gap
PLATE_WIDTH = 4             # linewidth of the magnet plates
LEGEND_COLUMNS = 4          # a wider legend would squeeze the schematic

COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_AXIS = "#c3c2b7"
COLOR_MAGNET = "#eb6834"
# Materials in order of first appearance along the beam (fixed order, never cycled)
COLORS_MATERIAL = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FILL_ALPHA = 0.3


# path: layout folder (absolute, or relative to data/)
# Returns: (layout DataFrame, metadata dict)
def load_layout(path):

    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = DATA_DIR / path

    # keep_default_na: the empty names/materials stay "" instead of NaN
    table = pd.read_csv(path / "layout.csv", keep_default_na=False)
    with open(path / "metadata.yml") as f:
        metadata = yaml.safe_load(f)

    return table, metadata


# Page with an empty figure and the header that identifies the layout
def new_page(title, metadata):

    commit = str(metadata["git_commit"])[:7]
    if metadata["git_uncommitted_changes"]:
        commit += " + uncommitted changes"
    subtitle = (f"{metadata['n_detector_planes']} detector planes  |  "
                f"{metadata['n_phantom_layers']} phantom layer(s)  |  "
                f"created {metadata['created']}  |  commit {commit}")

    fig = Figure(figsize=PAGE_SIZE, layout="constrained")
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11, color=COLOR_TEXT)

    return fig


# Segments between consecutive rows of the table: list of (z_start, z_end, region, material).
# Consecutive segments of the same region and material (eg the magnet split by a detector
# plane inside it) are merged into one
def segments(table):

    rows = table.to_dict("records")
    merged = []
    for a, b in zip(rows[:-1], rows[1:]):
        key = (a["region_after"], a["material_after"])
        if merged and merged[-1][2:] == key:
            merged[-1] = (merged[-1][0], b["z"], *key)
        else:
            merged.append((a["z"], b["z"], *key))
    return merged


# Page 1: schematic of the system along z
def page_schematic(table, metadata):

    fig = new_page("System layout along z", metadata)
    # Empty rows/columns around the schematic leave margins in the page
    grid = fig.add_gridspec(3, 3, width_ratios=MARGIN_RATIOS, height_ratios=MARGIN_RATIOS)
    ax = fig.add_subplot(grid[1, 1])

    x_half_gap = metadata["geometry"]["magnet"]["x_half_gap"]

    # 1. Regions: phantom layers coloured by material (no finite x size: they fill the plot),
    #    magnet as its two plates at x = +-x_half_gap with the gap between them, nothing outside
    material_colors = {}
    for z_a, z_b, region, material in segments(table):
        if region == "phantom":
            color = material_colors.setdefault(material, COLORS_MATERIAL[len(material_colors) % len(COLORS_MATERIAL)])
            ax.axvspan(z_a, z_b, color=color, alpha=FILL_ALPHA, linewidth=0)
            label = material
        elif region == "magnet":
            ax.fill_between([z_a, z_b], -x_half_gap, x_half_gap, color=COLOR_MAGNET, alpha=FILL_ALPHA, linewidth=0)
            ax.hlines([-x_half_gap, x_half_gap], z_a, z_b, color=COLOR_MAGNET, linewidth=PLATE_WIDTH)
            label = "magnet"
        else:
            continue
        ax.text((z_a + z_b) / 2, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=COLOR_TEXT)

    # Beam axis
    ax.axhline(0, color=COLOR_MUTED, linewidth=0.8, linestyle=":")

    # 2. Detector planes (solid) and physical boundaries without detector (dashed)
    for row in table.itertuples():
        if row.is_detector:
            ax.axvline(row.z, color=COLOR_TEXT, linewidth=1.2)
            name = row.detector
        else:
            ax.axvline(row.z, color=COLOR_MUTED, linewidth=1, linestyle="--")
            name = row.boundary
        ax.text(row.z, 0.03, f" {name}  (z = {row.z:g})", transform=ax.get_xaxis_transform(),
                rotation=90, ha="right", va="bottom", fontsize=FONTSIZE_PLANES, color=COLOR_MUTED,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))  # readable over the plates

    # 3. Axes: z along the beam, x transverse and centred on the beam axis (x = 0)
    z_min, z_max = 0.0, table["z"].max()
    margin = 0.03 * (z_max - z_min)
    ax.set_xlim(z_min - margin, z_max + margin)
    ax.set_ylim(-X_RANGE * x_half_gap, X_RANGE * x_half_gap)
    ax.set_xlabel("z [mm]", fontsize=10, color=COLOR_TEXT)
    ax.set_ylabel("x [mm]", fontsize=10, color=COLOR_TEXT)
    ax.tick_params(labelsize=8, colors=COLOR_MUTED, length=3)
    ax.spines[:].set_color(COLOR_AXIS)

    # 4. Legend
    handles = [Patch(color=c, alpha=FILL_ALPHA, label=f"phantom: {m}") for m, c in material_colors.items()]
    if (table["region_after"] == "magnet").any():
        handles += [Line2D([], [], color=COLOR_MAGNET, linewidth=PLATE_WIDTH, label=f"magnet plates (x = ±{x_half_gap:g})"),
                    Patch(color=COLOR_MAGNET, alpha=FILL_ALPHA, label="magnet gap")]
    handles += [Line2D([], [], color=COLOR_MUTED, linewidth=0.8, linestyle=":", label="beam axis (x = 0)"),
                Line2D([], [], color=COLOR_TEXT, linewidth=1.2, label="detector plane"),
                Line2D([], [], color=COLOR_MUTED, linewidth=1, linestyle="--", label="boundary, no detector")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(len(handles), LEGEND_COLUMNS),
              fontsize=8, frameon=False)

    return fig


# Page 2: the layout table
def page_table(table, metadata):

    fig = new_page("Layout table", metadata)
    ax = fig.add_subplot()
    ax.set_axis_off()

    cells = table.copy()
    cells["z"] = cells["z"].map(lambda z: f"{z:g}")
    cells["is_detector"] = cells["is_detector"].map(lambda d: "yes" if d else "")
    columns = ["z [mm]" if c == "z" else c for c in cells.columns]

    tab = ax.table(cellText=cells.values, colLabels=columns, loc="upper center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.auto_set_column_width(range(len(columns)))
    tab.scale(1, 1.8)
    for (row, _), cell in tab.get_celld().items():
        cell.set_edgecolor(COLOR_AXIS)
        cell.get_text().set_color(COLOR_TEXT)
        if row == 0:
            cell.get_text().set_fontweight("bold")

    return fig


# table: layout DataFrame (see src/geometry/layout.py)
# metadata: dict of metadata.yml of the layout folder
# out_path: path of the output pdf
# Returns: out_path as a Path
def plot_layout(table, metadata, out_path):

    out_path = Path(out_path)

    # All the pages are built first: a failure does not leave a partial pdf behind
    pages = [page(table, metadata) for page in (page_schematic, page_table)]

    with PdfPages(out_path) as pdf:
        for fig in pages:
            pdf.savefig(fig)

    return out_path


def main():

    parser = argparse.ArgumentParser(description="Redo the report of a layout folder.")
    parser.add_argument("layout", help="layout folder (absolute, or relative to data/)")
    args = parser.parse_args()

    path = Path(args.layout)
    if not path.exists():
        path = DATA_DIR / path

    table, metadata = load_layout(path)
    print(f"Saved report to {plot_layout(table, metadata, path / REPORT_NAME)}")


if __name__ == "__main__":
    main()

# This .py builds a table of everything placed along z (physical boundaries and, optionally,
# detector planes), with the region and material on each side. It is meant to be saved and
# inspected to check that the positions have been generated as expected

import pandas as pd

from src.utils.hashing import short_hash
from .geometry import Geometry, TOL


# Parameters that define the layout, already validated and normalised by Geometry and
# DetectorPlanes (eg 100 and 100.0 give the same value). Field parameters (B0, model...)
# are left out: they do not change the layout (the resolution sigma_x is not in geometry).
# A disabled phantom/magnet has its parameters at None.
# x_half_gap: transverse size of the magnet, drawn in the report (None without magnet)
# Returns: dict
def layout_parameters(geometry, detectors, x_half_gap):

    return {
        "z_start_phantom": geometry.z_start_phantom,
        "phantom_layers": geometry.phantom_layers,
        "z_start_magnet": geometry.z_start_magnet,
        "z_length_magnet": geometry.z_length_magnet,
        "x_half_gap": x_half_gap if geometry.has_magnet else None,
        "spacing_after_last": geometry.spacing_after_last,
        "placement": None if detectors is None else {
            "n_before_phantom": detectors.n_before_phantom,
            "n_after_phantom_before_magnet": detectors.n_after_phantom_before_magnet,
            "n_after_magnet": detectors.n_after_magnet,
            "inside_phantom": detectors.inside_phantom,
            "n_inside_magnet": detectors.n_inside_magnet,
        },
    }


# Deterministic hash of the layout parameters: same parameters -> same hash, any change -> another
# (the Lay<hash> of the folder names, see src/utils/hashing.py). The parameters are already
# normalised by Geometry and DetectorPlanes
def layout_hash(parameters):

    return short_hash(parameters)


# Name of the folder of data/geometry/ of a layout (scripts/generate_layout.py):
#   layout_Lay<hash>_<n>Det_<m>Layers_<p>PhDet_<k>MagDet
# Returns: (name, parameters, hash), so a propagation can record which layout it was made on
def layout_name(geometry, detectors, x_half_gap):

    table = layout_table(geometry, detectors)
    n_detectors = int(table["is_detector"].sum())
    n_layers = len(geometry.phantom_layers)
    n_phantom_detectors = len(detectors.z_phantom_layers) if detectors else 0
    n_magnet_detectors = detectors.n_inside_magnet if detectors and geometry.has_magnet else 0
    parameters = layout_parameters(geometry, detectors, x_half_gap)
    hash_ = layout_hash(parameters)
    name = (f"layout_Lay{hash_}_{n_detectors}Det_{n_layers}Layers_{n_phantom_detectors}PhDet_"
            f"{n_magnet_detectors}MagDet")
    return name, parameters, hash_


def _merge_close(points):
    # points: list of (z, name, kind). Groups the ones closer than TOL into a single z.
    # Returns: list of (z, {kind: [names]})
    merged = []
    for z, name, kind in sorted(points, key=lambda p: p[0]):
        if merged and z - merged[-1][0] <= TOL:
            merged[-1][1].setdefault(kind, []).append(name)
        else:
            merged.append((z, {kind: [name]}))
    return merged


def layout_table(geometry: Geometry, detectors=None):
    # geometry: Geometry
    # detectors: DetectorPlanes built on that geometry, or None for the bare geometry
    # Returns: DataFrame, one row per distinct z in beam order, with columns
    #   z [mm], boundary (physical boundaries at z, joined by "+"), detector (detector plane
    #   names at z, joined by "+"), is_detector, region/material just before and just after z
    if detectors is not None and detectors.geometry is not geometry:
        raise ValueError("detectors must be built on the same Geometry object")

    points = [(z, name, "boundary") for z, name in geometry.boundaries()]
    if detectors is not None:
        points += [(z, name, "detector") for name, z in detectors.named_planes().items()]

    rows = []
    for z, names in _merge_close(points):
        material_before = geometry.material_at(z - TOL)
        material_after = geometry.material_at(z)
        rows.append({
            "z": z,
            "boundary": "+".join(names.get("boundary", [])),
            "detector": "+".join(names.get("detector", [])),
            "is_detector": "detector" in names,
            "region_before": geometry.region_at(z - TOL).value,
            "region_after": geometry.region_at(z).value,
            "material_before": material_before or "",
            "material_after": material_after or "",
        })

    return pd.DataFrame(rows)

# This .py builds a table of everything placed along z (physical boundaries and, optionally,
# detector planes), with the region and material on each side. It is meant to be saved and
# inspected to check that the positions have been generated as expected

import pandas as pd

from .geometry import Geometry, TOL


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

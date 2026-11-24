# tests/test_plot_tracks.py
"""Pages 2-4 of plotting/plot_tracks.py (states at D_phantom_end, D_magnet_end and the last plane):
the planes are found whatever elements the system has, and the pages are drawn even without data."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_pdf import PdfPages

from plotting.plot_tracks import page_states, plane_index_named, state_locations
from src.constants import MU_PLUS, PROTON


def metadata(phantom=True, magnet=True, planes=None):
    if planes is None:
        planes = ([{"name": "D_phantom_start", "z": 50.0}, {"name": "D_phantom_end", "z": 170.0}] if phantom else []) \
               + ([{"name": "D_magnet_start", "z": 400.0}, {"name": "D_magnet_end", "z": 700.0}] if magnet else []) \
               + [{"name": "D_after_magnet_1", "z": 750.0}]
    return {"geometry": {"phantom": {"enabled": phantom}, "magnet": {"enabled": magnet}}, "planes": planes}


def states_at(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(0, 1, n), "tx": rng.normal(0, 1e-3, n),
                         "q_over_p": 1 / rng.normal(700, 3, n)})


def test_merged_planes_are_found_by_any_name():
    planes = [{"name": "D_phantom_start", "z": 0.0}, {"name": "D_phantom_end+D_magnet_start", "z": 1.0}]
    assert plane_index_named(planes, "D_phantom_end") == 1
    assert plane_index_named(planes, "D_magnet_start") == 1
    assert plane_index_named(planes, "D_magnet_end") is None


@pytest.mark.parametrize("phantom, magnet", [(True, True), (True, False), (False, True), (False, False)])
def test_locations_with_and_without_elements(phantom, magnet):
    meta = metadata(phantom, magnet)
    (l1, k1, r1), (l2, k2, r2), (l3, k3, r3) = state_locations(meta)
    assert (l1, l2) == ("D_phantom_end", "D_magnet_end")
    assert (k1 is not None) == phantom and (r1 is None) == phantom
    assert (k2 is not None) == magnet and (r2 is None) == magnet
    if phantom:
        assert meta["planes"][k1]["name"] == "D_phantom_end"
    if magnet:
        assert meta["planes"][k2]["name"] == "D_magnet_end"
    assert k3 == len(meta["planes"]) - 1 and r3 is None       # the last plane always exists


def test_no_planes_gives_three_empty_locations():
    assert all(k is None and reason for _, k, reason in state_locations(metadata(planes=[])))


@pytest.mark.parametrize("phantom, magnet", [(True, True), (False, False)])
def test_pages_are_drawn_with_and_without_data(tmp_path, phantom, magnet):
    meta = metadata(phantom, magnet)
    with PdfPages(tmp_path / "pages.pdf") as pdf:
        for label, k, reason in state_locations(meta):
            at_plane = states_at(200) if k is not None else states_at(0)
            plane = meta["planes"][k] if k is not None else None
            pdf.savefig(page_states(at_plane, plane, label, reason, 200, PROTON, "dataset"))
        # a plane that exists but no track reached (e.g. all stopped), and constant columns
        pdf.savefig(page_states(states_at(0), meta["planes"][-1], "last", None, 200, MU_PLUS, "dataset"))
        constant = pd.DataFrame({"x": np.zeros(50), "tx": np.zeros(50), "q_over_p": np.full(50, 1 / 700)})
        pdf.savefig(page_states(constant, meta["planes"][-1], "last", None, 50, PROTON, "dataset"))
    assert (tmp_path / "pages.pdf").stat().st_size > 0

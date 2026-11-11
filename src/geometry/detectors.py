# This .py has the aim of determining the position of the detector planes along the machine.
# The physical layout (phantom, interfaces, magnet) lives in Geometry: the detector planes
# only read it, so the geometry exists and is the same whether there are planes or not.
#
# There is always a plane at each boundary of the elements present (phantom start/end, magnet
# start/end), plus the free planes, equispaced in the free regions between them. The placement is
# given for the full system (phantom + magnet). If an element is absent, its keys are ignored
# (inside_phantom, n_inside_magnet) and the two free regions around it merge into one that gets the
# planes of both, so the number of free planes does not change when an element is disabled:
#
#   phantom + magnet   before_phantom | gap            | magnet | after_magnet
#   no phantom         before_magnet (before + gap)    | magnet | after_magnet
#   no magnet          before_phantom | after_phantom (gap + after)
#   neither            free (before + gap + after)

import warnings

import numpy as np

from src.utils.validation import check_bool, check_count

from .geometry import Geometry, TOL


class DetectorPlanes():

    def __init__(self, geometry: Geometry, n_before_phantom, n_after_phantom_before_magnet,
                 n_after_magnet, inside_phantom, n_inside_magnet):

        if not isinstance(geometry, Geometry):
            raise ValueError(f"geometry must be a Geometry, got {type(geometry).__name__}")
        self.geometry = geometry

        self.n_before_phantom = check_count("n_before_phantom", n_before_phantom)
        self.n_after_phantom_before_magnet = check_count("n_after_phantom_before_magnet", n_after_phantom_before_magnet)
        self.n_after_magnet = check_count("n_after_magnet", n_after_magnet)
        self.inside_phantom = check_bool("inside_phantom", inside_phantom)
        self.n_inside_magnet = check_count("n_inside_magnet", n_inside_magnet)

        self.z_phantom_layers = self.compute_z_phantom_layers()
        self.z_free_planes = self.compute_z_free_planes()


    def compute_z_phantom_layers(self):
        # One plane at each internal interface: (#interfaces - 2) planes. None without phantom
        if not (self.inside_phantom and self.geometry.has_phantom):
            return np.array([])

        return self.geometry.z_phantom_interfaces[1:-1]


    def free_regions(self):
        # [(name, z_a, z_b, n)] of the free regions in beam order: n planes strictly between z_a and z_b
        g = self.geometry
        n_before, n_gap, n_after = self.n_before_phantom, self.n_after_phantom_before_magnet, self.n_after_magnet

        if g.has_phantom and g.has_magnet:
            return [("before_phantom", 0.0, g.z_start_phantom, n_before),
                    ("gap", g.z_end_phantom, g.z_start_magnet, n_gap),
                    ("magnet", g.z_start_magnet, g.z_end_magnet, self.n_inside_magnet),
                    ("after_magnet", g.z_end_magnet, g.z_end_system, n_after)]
        if g.has_magnet:
            return [("before_magnet", 0.0, g.z_start_magnet, n_before + n_gap),
                    ("magnet", g.z_start_magnet, g.z_end_magnet, self.n_inside_magnet),
                    ("after_magnet", g.z_end_magnet, g.z_end_system, n_after)]
        if g.has_phantom:
            return [("before_phantom", 0.0, g.z_start_phantom, n_before),
                    ("after_phantom", g.z_end_phantom, g.z_end_system, n_gap + n_after)]
        return [("free", 0.0, g.z_end_system, n_before + n_gap + n_after)]


    def fixed_planes(self):
        # [(name, z)] of the planes at the boundaries of the elements present, in beam order
        g = self.geometry
        out = []
        if g.has_phantom:
            out += [("D_phantom_start", g.z_start_phantom)]
            out += [(f"D_layer_{k}_to_{k + 1}", z) for k, z in enumerate(self.z_phantom_layers.tolist(), start=1)]
            out += [("D_phantom_end", g.z_end_phantom)]
        if g.has_magnet:
            out += [("D_magnet_start", g.z_start_magnet), ("D_magnet_end", g.z_end_magnet)]
        return out


    @staticmethod
    def _planes_between(z_a, z_b, n):
        # n equispaced planes strictly between z_a and z_b (empty if n = 0)
        return np.linspace(z_a, z_b, n + 2)[1:-1]


    def compute_z_free_planes(self):
        # Computes the detector planes that are not fixed by construction (see free_regions)
        return np.concatenate([np.array([])] + [self._planes_between(z_a, z_b, n)
                                                for _, z_a, z_b, n in self.free_regions()])


    def compute_z_detector_planes(self):
        z_all_planes = np.sort(np.concatenate((
            [z for _, z in self.fixed_planes()],
            self.z_free_planes,
        )))

        # A zero-length region collapses its planes onto the same z: keep only one and warn
        is_new_plane = np.concatenate(([True], np.diff(z_all_planes) > TOL))
        n_removed = np.count_nonzero(~is_new_plane)
        if n_removed > 0:
            warnings.warn(
                f"Removed {n_removed} overlapping detector plane(s) at z = "
                f"{z_all_planes[~is_new_plane].tolist()} mm: "
                f"{z_all_planes.size - n_removed} planes kept out of {z_all_planes.size}.",
                stacklevel=2,
            )

        return z_all_planes[is_new_plane]


    def _named_free_planes(self, name, z_a, z_b, n):
        # [(D_<name>_k, z)] of the free planes of a region. A zero-length region has none
        if z_b - z_a <= TOL:
            return []

        planes = self._planes_between(z_a, z_b, n).tolist()
        return [(f"D_{name}_{k}", z) for k, z in enumerate(planes, start=1)]


    def named_planes(self):
        # {name: z} of every detector plane, in beam order. A plane shared by two boundaries
        # (eg phantom touching the magnet) has both names, so use compute_z_detector_planes to count planes
        planes = self.fixed_planes()
        for region in self.free_regions():
            planes += self._named_free_planes(*region)
        # stable sort: planes at the same z keep the beam order of fixed_planes
        return dict(sorted(planes, key=lambda plane: plane[1]))

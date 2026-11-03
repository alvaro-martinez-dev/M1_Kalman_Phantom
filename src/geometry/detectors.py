# This .py has the aim of determining the position of the detector planes along the machine

import warnings

import numpy as np

from .geometry import Geometry, TOL


class DetectorPlanes(Geometry):

    def __init__(self, z_start_phantom, phantom_layers, z_start_magnet, z_length_magnet,
                 spacing_after_magnet, n_before_phantom, n_after_phantom_before_magnet,
                 n_after_magnet, inside_phantom, n_inside_magnet):

        super().__init__(z_start_phantom, phantom_layers, z_start_magnet, z_length_magnet,
                         spacing_after_magnet)

        self.n_before_phantom = n_before_phantom
        self.n_after_phantom_before_magnet = n_after_phantom_before_magnet
        self.n_after_magnet = n_after_magnet
        self.inside_phantom = inside_phantom # boolean value
        self.n_inside_magnet = n_inside_magnet

        self.z_phantom_layers = self.compute_z_phantom_layers()
        self.z_free_planes = self.compute_z_free_planes()


    def compute_z_phantom_layers(self):
        # One plane at each internal interface: (#interfaces - 2) planes
        if not self.inside_phantom:
            return np.array([])

        return self.z_phantom_interfaces[1:-1]


    @staticmethod
    def _planes_between(z_a, z_b, n):
        # n equispaced planes strictly between z_a and z_b (empty if n = 0)
        return np.linspace(z_a, z_b, n + 2)[1:-1]


    def compute_z_free_planes(self):
        # Computes the detector planes that are not fixed by construction,
        # ie the ones before the phantom, after phantom and before magnet, inside magnet, after magnet
        return np.concatenate([
            self._planes_between(0.0, self.z_start_phantom, self.n_before_phantom),
            self._planes_between(self.z_end_phantom, self.z_start_magnet, self.n_after_phantom_before_magnet),
            self._planes_between(self.z_start_magnet, self.z_end_magnet, self.n_inside_magnet),
            self._planes_between(self.z_end_magnet, self.z_end_system, self.n_after_magnet),
        ])


    def compute_z_detector_planes(self):

        z_all_planes = np.sort(np.concatenate((
            [self.z_start_phantom, self.z_end_phantom, self.z_start_magnet, self.z_end_magnet],
            self.z_phantom_layers,
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
        # {D_<name>_k: z} of the free planes of a region. A zero-length region has none
        if z_b - z_a <= TOL:
            return {}

        planes = self._planes_between(z_a, z_b, n).tolist()
        return {f"D_{name}_{k}": z for k, z in enumerate(planes, start=1)}


    def named_planes(self):
        # {name: z} of every detector plane, in beam order. A plane shared by two boundaries
        # (eg phantom touching the magnet) has both names, so use compute_z_detector_planes to count planes
        layer_planes = self.z_phantom_layers.tolist()

        return {
            **self._named_free_planes("before_phantom", 0.0, self.z_start_phantom, self.n_before_phantom),
            "D_phantom_start": self.z_start_phantom,
            **{f"D_layer_{k}_to_{k + 1}": z for k, z in enumerate(layer_planes, start=1)},
            "D_phantom_end": self.z_end_phantom,
            **self._named_free_planes("gap", self.z_end_phantom, self.z_start_magnet, self.n_after_phantom_before_magnet),
            "D_magnet_start": self.z_start_magnet,
            **self._named_free_planes("magnet", self.z_start_magnet, self.z_end_magnet, self.n_inside_magnet),
            "D_magnet_end": self.z_end_magnet,
            **self._named_free_planes("after_magnet", self.z_end_magnet, self.z_end_system, self.n_after_magnet),
        }

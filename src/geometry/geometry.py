# This .py has the aim of determining the physical layout of the machine along z:
# where the phantom, its layers and the magnet are, and what there is at each z

from enum import Enum

import numpy as np

TOL = 1e-6 # [mm] two z positions closer than this are considered the same


class Region(Enum):
    OUTSIDE = "outside"
    PHANTOM = "phantom"
    MAGNET = "magnet"


class Geometry():

    def __init__(self, z_start_phantom, phantom_layers, z_start_magnet, z_length_magnet,
                 spacing_after_magnet):

        self.z_start_phantom = z_start_phantom
        self.phantom_layers = phantom_layers
        self.z_start_magnet = z_start_magnet
        self.z_length_magnet = z_length_magnet
        self.spacing_after_magnet = spacing_after_magnet

        self.z_end_phantom, self.z_end_magnet = self.compute_z_end()
        self.z_end_system = self.z_end_magnet + self.spacing_after_magnet
        self.z_phantom_interfaces = self.compute_z_phantom_interfaces()


    def compute_z_end(self):
        # Finds the end of the phantom and the magnet
        z_end_phantom = self.z_start_phantom + sum(layer["thickness"] for layer in self.phantom_layers)
        z_end_magnet = self.z_start_magnet + self.z_length_magnet

        # guarantee that the phantom does not overlap the magnet (touching is allowed, magnet behind)
        if z_end_phantom > self.z_start_magnet + TOL:
            raise ValueError(
                f"The phantom must not overlap the magnet: "
                f"z_end_phantom = {z_end_phantom} mm, z_start_magnet = {self.z_start_magnet} mm"
            )

        return z_end_phantom, z_end_magnet


    def compute_z_phantom_interfaces(self):
        # All the layer boundaries, including the entrance and the exit of the phantom: (#layers + 1) values
        thicknesses = [layer["thickness"] for layer in self.phantom_layers]
        return self.z_start_phantom + np.concatenate(([0.0], np.cumsum(thicknesses)))


    def _layer_index(self, z):
        # Index of the phantom layer that contains z, or None if z is out of the phantom.
        # Each layer covers [start, end), so a z on an interface belongs to the next layer
        index = int(np.searchsorted(self.z_phantom_interfaces, z, side="right")) - 1
        if 0 <= index < len(self.phantom_layers):
            return index
        return None


    def region_at(self, z):
        # Region (Region.PHANTOM, Region.MAGNET or Region.OUTSIDE) at a single position z.
        # Regions cover [start, end), so a z on a boundary belongs to what comes next
        if self._layer_index(z) is not None:
            return Region.PHANTOM
        if self.z_start_magnet <= z < self.z_end_magnet:
            return Region.MAGNET
        return Region.OUTSIDE


    def material_at(self, z):
        # Material of the phantom layer at a single position z, as written in phantom_layers.
        # None means that there is no material there (out of the phantom)
        index = self._layer_index(z)
        if index is None:
            return None
        return self.phantom_layers[index]["material"]


    def boundaries(self):
        # (z, label) of every physical boundary, in beam order
        n_layers = len(self.phantom_layers)
        labels = (["phantom_start"]
                  + [f"layer_{k}_to_{k + 1}" for k in range(1, n_layers)]
                  + ["phantom_end"])
        return list(zip(self.z_phantom_interfaces.tolist(), labels)) + [
            (self.z_start_magnet, "magnet_start"),
            (self.z_end_magnet, "magnet_end"),
            (self.z_end_system, "system_end"),
        ]

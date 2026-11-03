# This .py has the aim of building the complete machine, with phantom, magnet and detector planes

import numpy as np

class Geometry():

    def __init__(self, z_start_phantom, phantom_layers, z_start_magnet, z_length_magnet, 
                 spacing_after_magnet, n_before_phantom, n_after_phantom_before_magnet,
                 n_after_magnet, inside_phantom, n_inside_magnet):
        self.z_start_ph = z_start_phantom
        self.ph_layers = phantom_layers
        self.z_st_magn = z_start_magnet
        self.z_length_magn = z_length_magnet
        self.space_aft_magn = spacing_after_magnet
        self.n_bf_ph = n_before_phantom
        self.n_aft_ph_bf_magn = n_after_phantom_before_magnet
        self.n_aft_magn = n_after_magnet
        self.inside_ph = inside_phantom # boolean value
        self.n_inside_magn = n_inside_magnet


    def compute_distances(self):
        # Finds the end of the phantom and the magnet
        z_end_phantom = self.z_start_ph + sum(layer["thickness"] for layer in self.ph_layers)
        z_end_magnet = self.z_st_magn + self.z_length_magn

        # guarantee that the phantom ends before the magnet starts (no overlap, magnet behind)
        if z_end_phantom >= self.z_st_magn:
            raise ValueError(
                f"The phantom must end before the magnet starts: "
                f"z_end_phantom = {z_end_phantom} mm, z_start_magnet = {self.z_st_magn} mm"
            )

        return z_end_phantom, z_end_magnet



    def compute_inside_phantom_planes(self):
        # One plane at each internal interface: (#interfaces - 2) planes
        if not self.inside_ph:
            return np.array([])

        thicknesses = [layer["thickness"] for layer in self.ph_layers]
        interfaces = self.z_start_ph + np.concatenate(([0.0], np.cumsum(thicknesses)))
        return interfaces[1:-1]



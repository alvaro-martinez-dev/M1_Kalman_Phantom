# This .py has the aim of building the complete machine, with phantom, magnet and detector planes

import numpy as np

class Geometry():

    def __init__(self, z_start_phantom, phantom_layers, z_start_magnet, z_length_magnet, 
                 spacing_after_magnet, n_before_phantom, n_after_phantom_before_magnet,
                 n_after_magnet, n_inside_phantom, n_inside_magnet):
        self.z_start_ph = z_start_phantom
        self.ph_layers = phantom_layers
        self.z_st_magn = z_start_magnet
        self.z_length_magn = z_length_magnet
        self.space_aft_magn = spacing_after_magnet
        self.n_bf_ph = n_before_phantom
        self.n_aft_ph_bf_magn = n_after_phantom_before_magnet
        self.n_aft_magn = n_after_magnet
        self.n_inside_ph = n_inside_phantom
        self.n_inside_magn = n_inside_magnet


    def compute_distances(self):
        # Finds the end of the phantom, the magnet and the system
        z_end_phantom = self.z_start_ph + sum(layers["thcikness"] for layers in self.ph_layers)
        z_end_magnet = self.z_st_magn + 



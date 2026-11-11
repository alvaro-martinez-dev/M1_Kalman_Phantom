# This .py has the aim of determining the physical layout of the machine along z:
# where the phantom, its layers and the magnet are, and what there is at each z.
#
# The phantom and the magnet are optional (enabled: true/false in the yaml):
#   - no phantom: z_start_phantom = z_end_phantom = None, phantom_layers = [], no interfaces
#   - no magnet:  z_start_magnet = z_length_magnet = z_end_magnet = None
# The system ends spacing_after_last after the last element present (magnet, else phantom, else z = 0)

from enum import Enum

import numpy as np

import src.constants as constants
from src.constants import Material
from src.utils.validation import check_real

TOL = 1e-6 # [mm] two z positions closer than this are considered the same


class Region(Enum):
    OUTSIDE = "outside"
    PHANTOM = "phantom"
    MAGNET = "magnet"


class Geometry():

    # z_start_phantom, phantom_layers: both None (no phantom) or both given
    # z_start_magnet, z_length_magnet: both None (no magnet) or both given
    # spacing_after_last: [mm] from the end of the last element present to the end of the system
    def __init__(self, z_start_phantom=None, phantom_layers=None, z_start_magnet=None,
                 z_length_magnet=None, spacing_after_last=0.0):

        self.has_phantom = self.check_present("phantom", z_start_phantom=z_start_phantom,
                                              phantom_layers=phantom_layers)
        self.has_magnet = self.check_present("magnet", z_start_magnet=z_start_magnet,
                                             z_length_magnet=z_length_magnet)

        if self.has_phantom:
            # The beam starts at z = 0: nothing can be placed before it
            self.z_start_phantom = check_real("z_start_phantom", z_start_phantom, minimum=0.0)
            self.phantom_layers = self.check_layers(phantom_layers)
        else:
            self.z_start_phantom, self.phantom_layers = None, []

        if self.has_magnet:
            self.z_start_magnet = check_real("z_start_magnet", z_start_magnet, minimum=0.0)
            self.z_length_magnet = check_real("z_length_magnet", z_length_magnet, minimum=TOL, strict=True)
        else:
            self.z_start_magnet = self.z_length_magnet = None

        self.spacing_after_last = check_real("spacing_after_last", spacing_after_last, minimum=0.0)

        self.z_end_phantom, self.z_end_magnet = self.compute_z_end()
        self.z_end_system = self.compute_z_end_system()
        self.z_phantom_interfaces = self.compute_z_phantom_interfaces()


    @staticmethod
    def check_present(element, **values):
        # An element is present if all its parameters are given, absent if none is
        given = [value is not None for value in values.values()]
        if any(given) and not all(given):
            raise ValueError(f"The {element} needs all of {list(values)} or none of them (no {element}), "
                             f"got {values!r}")
        return all(given)


    @staticmethod
    def check_layers(phantom_layers):
        # A phantom that is present has at least one layer, each one
        # {material: <Material in src/constants.py>, thickness: > 0}.
        # A zero thickness is rejected: it would put two interfaces at the same z
        if not isinstance(phantom_layers, (list, tuple)) or len(phantom_layers) == 0:
            raise ValueError(f"phantom_layers must be a non-empty list of layers, got {phantom_layers!r} "
                             "(to remove the phantom, disable it)")

        checked = []
        for k, layer in enumerate(phantom_layers, start=1):
            name = f"phantom layer {k}"
            if not isinstance(layer, dict) or set(layer) != {"material", "thickness"}:
                raise ValueError(f"{name} must be {{material: ..., thickness: ...}}, got {layer!r}")

            material = layer["material"]
            if not isinstance(material, str) or not isinstance(getattr(constants, material, None), Material):
                raise ValueError(f"{name}: material {material!r} is not a Material of src/constants.py")

            thickness = check_real(f"{name}: thickness", layer["thickness"], minimum=TOL, strict=True)
            checked.append({"material": material, "thickness": thickness})

        return checked


    def compute_z_end(self):
        # Finds the end of the phantom and the magnet (None if absent)
        z_end_phantom = z_end_magnet = None
        if self.has_phantom:
            z_end_phantom = self.z_start_phantom + sum(layer["thickness"] for layer in self.phantom_layers)
        if self.has_magnet:
            z_end_magnet = self.z_start_magnet + self.z_length_magnet

        # guarantee that the phantom does not overlap the magnet (touching is allowed, magnet behind)
        if self.has_phantom and self.has_magnet and z_end_phantom > self.z_start_magnet + TOL:
            raise ValueError(
                f"The phantom must not overlap the magnet: "
                f"z_end_phantom = {z_end_phantom} mm, z_start_magnet = {self.z_start_magnet} mm"
            )

        return z_end_phantom, z_end_magnet


    def compute_z_end_system(self):
        # spacing_after_last after the end of the last element present (z = 0 if there is none)
        ends = [z for z in (self.z_end_phantom, self.z_end_magnet) if z is not None]
        z_last = max(ends) if ends else 0.0
        return z_last + self.spacing_after_last


    def compute_z_phantom_interfaces(self):
        # All the layer boundaries, including the entrance and the exit of the phantom: (#layers + 1)
        # values. Empty if there is no phantom
        if not self.has_phantom:
            return np.array([])
        thicknesses = [layer["thickness"] for layer in self.phantom_layers]
        return self.z_start_phantom + np.concatenate(([0.0], np.cumsum(thicknesses)))


    def _layer_index(self, z):
        # Index of the phantom layer that contains z, or None if z is out of the phantom.
        # Each layer covers [start, end), so a z on an interface belongs to the next layer
        if not self.has_phantom:
            return None
        index = int(np.searchsorted(self.z_phantom_interfaces, z, side="right")) - 1
        if 0 <= index < len(self.phantom_layers):
            return index
        return None


    def region_at(self, z):
        # Region (Region.PHANTOM, Region.MAGNET or Region.OUTSIDE) at a single position z.
        # Regions cover [start, end), so a z on a boundary belongs to what comes next
        if self._layer_index(z) is not None:
            return Region.PHANTOM
        if self.has_magnet and self.z_start_magnet <= z < self.z_end_magnet:
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
        # (z, label) of every physical boundary of the elements present, in beam order
        out = []
        if self.has_phantom:
            n_layers = len(self.phantom_layers)
            labels = (["phantom_start"]
                      + [f"layer_{k}_to_{k + 1}" for k in range(1, n_layers)]
                      + ["phantom_end"])
            out += list(zip(self.z_phantom_interfaces.tolist(), labels))
        if self.has_magnet:
            out += [(self.z_start_magnet, "magnet_start"), (self.z_end_magnet, "magnet_end")]
        out.append((self.z_end_system, "system_end"))
        return out

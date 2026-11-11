# src/utils/config_loader.py
#
# Reads the yaml config files of configs/ and builds the objects they describe.
# Particles and decay channels are referenced in the yaml by the name of their
# constant in src/constants.py (e.g. PROTON, JPSI_MUMU).

from pathlib import Path

import yaml

import src.constants as constants
from src.geometry.detectors import DetectorPlanes
from src.geometry.geometry import Geometry
from src.montecarlo.init_beam import Initial_Beam
from src.utils.validation import check_bool, check_real


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


# name: name of a yaml in configs/ without extension (e.g. "config"),
#       or a path to any yaml file
# Returns: dict with the content of the yaml
def load_config(name):

    path = Path(name)
    if path.suffix not in (".yaml", ".yml"):
        path = CONFIG_DIR / f"{name}.yaml"

    with open(path) as f:
        return yaml.safe_load(f)


# name: name of a constant in src/constants.py (e.g. "PROTON", "JPSI_MUMU")
# Returns: the Particle, DecayChannel... with that name
def get_constant(name):

    if not hasattr(constants, name):
        raise ValueError(f"{name} is not defined in src/constants.py")

    return getattr(constants, name)


# beam_cfg: one block of beams.<name> in the yaml
# Returns: Initial_Beam built from it
def make_beam(beam_cfg):

    return Initial_Beam(
        x0=beam_cfg["x0"],
        sigma_x0=beam_cfg["sigma_x0"],
        theta0=beam_cfg["theta0"],
        sigma_theta0=beam_cfg["sigma_theta0"],
        p0=beam_cfg["p0"],
        sigma_p0=beam_cfg["sigma_p0"],
        particle=get_constant(beam_cfg["particle"]),
    )


# cfg: a block of the yaml
# path: dotted keys inside it (e.g. "phantom.z_start")
# block: name of cfg in the yaml, for the error message (e.g. "geometry")
# Returns: the value, or raises a KeyError that says which key of the yaml is missing
def get_key(cfg, path, block):

    value = cfg
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"missing key '{block}.{path}' in the yaml")
        value = value[key]

    return value


# cfg: a block of the yaml with an "enabled" key
# path: name of the block, for the error message (e.g. "geometry.phantom")
# Returns: the value of enabled, which must be true or false (a typo must not disable an element)
def is_enabled(cfg, path):

    if not isinstance(cfg, dict) or "enabled" not in cfg:
        raise KeyError(f"missing key '{path}.enabled' in the yaml")
    enabled = cfg["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError(f"'{path}.enabled' must be true or false, got {enabled!r}")

    return enabled


# geometry_cfg: the geometry block of the yaml
# Returns: Geometry built from it (phantom, interfaces, magnet; no detectors).
#          A disabled phantom/magnet is absent: its other keys are not read
def make_geometry(geometry_cfg):

    def get(path):
        return get_key(geometry_cfg, path, "geometry")

    phantom = {}
    if is_enabled(get("phantom"), "geometry.phantom"):
        phantom = dict(z_start_phantom=get("phantom.z_start"), phantom_layers=get("phantom.layers"))

    magnet = {}
    if is_enabled(get("magnet"), "geometry.magnet"):
        magnet = dict(z_start_magnet=get("magnet.z_start"), z_length_magnet=get("magnet.z_length"))

    return Geometry(**phantom, **magnet, spacing_after_last=get("spacing_after_last"))


# cfg: full config dict (load_config)
# geometry: Geometry built from cfg (make_geometry)
# field: MagneticField built from cfg (src.field.build_field)
# particle: Particle that is propagated (e.g. the one of the beam, or a decay product)
# Returns: Propagator built from the propagator block of the yaml. Without magnet x_half_gap is
#          not read (there is no field: NoField). The scattering kicks also need an rng in propagate
def make_propagator(cfg, geometry, field, particle):

    from src.propagator.propagator import Propagator   # local: the propagator imports src.field

    def get(path):
        return get_key(cfg["propagator"], path, "propagator")

    x_half_gap = get_key(cfg["geometry"], "magnet.x_half_gap", "geometry") if geometry.has_magnet else float("inf")
    max_slice = get("scattering.max_slice") if is_enabled(get("scattering"), "propagator.scattering") else None

    return Propagator(
        geometry, field, x_half_gap, particle=particle,
        step=check_real("propagator.step", get("step"), minimum=0.0, strict=True),
        adaptive=check_bool("propagator.adaptive", get("adaptive")),
        tol_x=check_real("propagator.tolerances.x", get("tolerances.x"), minimum=0.0, strict=True),
        tol_tx=check_real("propagator.tolerances.tx", get("tolerances.tx"), minimum=0.0, strict=True),
        tol_qp_rel=check_real("propagator.tolerances.qp_rel", get("tolerances.qp_rel"), minimum=0.0, strict=True),
        max_slice=max_slice,
    )


# geometry: Geometry the planes are placed on
# detectors_cfg: the geometry.detector_planes block of the yaml
# Returns: DetectorPlanes built from it, or None if the detector planes are disabled
def make_detector_planes(geometry, detectors_cfg):

    if not is_enabled(detectors_cfg, "geometry.detector_planes"):
        return None

    def get(key):
        return get_key(detectors_cfg, f"placement.{key}", "geometry.detector_planes")

    return DetectorPlanes(
        geometry,
        n_before_phantom=get("n_before_phantom"),
        n_after_phantom_before_magnet=get("n_after_phantom_before_magnet"),
        n_after_magnet=get("n_after_magnet"),
        inside_phantom=get("inside_phantom"),
        n_inside_magnet=get("n_inside_magnet"),
    )

# src/utils/hashing.py
#
# Deterministic hashes of the input parameters of each kind of folder of data/, so that the name
# of a folder says what it was generated from:
#   same parameters -> same hash (on any computer),  any change that matters -> another hash
#
#   Beam<h>   beam block (except the particle)    data/init_beam/<beam>_n<n>_seed<s>_Beam<h>
#             + n_events + seed
#   Lay<h>    geometry + detector planes          data/geometry/layout_Lay<h>_...  (src/geometry/layout.py)
#   Magn<h>   magnet block (+ extent if interp.)  data/field/field_Magn<h>_...
#   Prop<h>   physics options of the propagator   data/propagation/<beam>_Beam<h>_Lay<h>_Magn<h>_Prop<h>_seed<s>[_n<n>]
#   Kal<h>    options of the Kalman fit           data/fit/<hits>_Kal<h>[_n<n>]
#
# A child dataset cites its parents by their hashes, so each hash covers ALL the parameters of its
# folder that change the result (Beam includes the seed and n_events), and only those: n_workers,
# plot options or keys of a model that is not used do not change any hash.
#
# Hash: first HASH_LENGTH hex characters of the sha256 of the parameters as json with sorted keys.
# The parameters taken from the yaml are normalised first (canonical: an integer and the same float
# give the same hash, e.g. 100 and 100.0).

import hashlib
import json
from numbers import Integral, Real

HASH_LENGTH = 6


# Hash of an already normalised json-serialisable object (dicts, lists, numbers, strings, None)
def short_hash(obj):
    text = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:HASH_LENGTH]


# Normalises values read from a yaml: numbers (not bools) to float, recursively
def canonical(obj):
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (Integral, Real)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    raise TypeError(f"Cannot hash a {type(obj).__name__}: {obj!r}")


# beam_cfg: block of beams: of the yaml (particle, x0, sigma_x0, theta0, sigma_theta0, p0, sigma_p0)
# The particle is left out on purpose: two beams of different species (e.g. protons and muons) with
# the same kinematics, n_events and seed get the same Beam<h>, so it is seen at a glance that they
# were generated alike. The species is still in the folder name (<beam>_...) and in the metadata.
def beam_parameters(beam_cfg, n_events, seed):
    kinematics = {key: value for key, value in beam_cfg.items() if key != "particle"}
    return {"beam": canonical(kinematics), "n_events": int(n_events), "seed": int(seed)}


def beam_hash(beam_cfg, n_events, seed):
    return short_hash(beam_parameters(beam_cfg, n_events, seed))


# magnet_cfg: geometry.magnet block of the yaml; z_end_system: [mm] end of the system (Geometry)
# Only what changes the field the propagator sees (src.field.build_field): the keys of the other
# model and the interpolation grid when it is not used (disabled, or hard_edge) are left out. The
# extent of the system sets the interpolation grid, so it is included only if the field is interpolated.
# Returns: dict, or None if the magnet is disabled (no field)
def magnet_parameters(magnet_cfg, z_end_system):
    if not magnet_cfg.get("enabled", True):
        return None
    params = {key: canonical(magnet_cfg[key]) for key in ("z_start", "z_length", "x_half_gap", "B0", "model")}
    if magnet_cfg["model"] == "biot_savart":
        params["y_height"] = canonical(magnet_cfg["y_height"])
        params["biot_savart"] = canonical(magnet_cfg["biot_savart"])
    # Only the Biot-Savart field is tabulated: hard_edge is always evaluated exactly (build_field) and
    # ignores the interpolation block. Before that change hard_edge was interpolated too, so those
    # datasets (with interpolation) keep a hash that the exact field never gets
    interpolation = magnet_cfg.get("interpolation", {})
    if magnet_cfg["model"] == "biot_savart" and interpolation.get("enabled", False):
        params["interpolation"] = {"dx": canonical(interpolation["dx"]), "dz": canonical(interpolation["dz"])}
        params["z_end_system"] = canonical(z_end_system)
    return params


# "off" without magnet, so that the name says it: Magnoff
def magnet_hash(magnet_cfg, z_end_system):
    params = magnet_parameters(magnet_cfg, z_end_system)
    return "off" if params is None else short_hash(params)


# prop_cfg: propagator block of the yaml. Only the physics options (step, adaptive, tolerances if
# adaptive, scattering slices): input, seed, n_max and n_workers are not part of the model
def propagator_parameters(prop_cfg):
    adaptive = bool(prop_cfg["adaptive"])
    scattering = prop_cfg.get("scattering", {})
    return {
        "step": canonical(prop_cfg["step"]),
        "adaptive": adaptive,
        "tolerances": canonical(prop_cfg["tolerances"]) if adaptive else None,
        "max_slice": canonical(scattering["max_slice"]) if scattering.get("enabled", False) else None,
    }


def propagator_hash(prop_cfg):
    return short_hash(propagator_parameters(prop_cfg))


# Options of the Kalman fit that change its output. range_constraints only enters when true, so the
# fits made before the option existed (unconstrained) keep their hash
def kalman_parameters(smoother, central_differences, fd_steps, range_constraints=False):
    params = {"smoother": bool(smoother), "central_differences": bool(central_differences),
              "fd_steps": canonical(list(fd_steps))}
    if range_constraints:
        params["range_constraints"] = True
    return params


def kalman_hash(smoother, central_differences, fd_steps, range_constraints=False):
    return short_hash(kalman_parameters(smoother, central_differences, fd_steps, range_constraints))

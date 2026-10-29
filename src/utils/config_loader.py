# src/utils/config_loader.py
#
# Reads the yaml config files of configs/ and builds the objects they describe.
# Particles and decay channels are referenced in the yaml by the name of their
# constant in src/constants.py (e.g. PROTON, JPSI_MUMU).

from pathlib import Path

import yaml

import src.constants as constants
from src.montecarlo.init_beam import Initial_Beam


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
# Returns: Initial_Beam built from it (the "decay" key, if any, is ignored)
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

#here we upload the yalm file and we convert the B0 in tesla to B0_nat in T

from pathlib import Path
import yaml
from .constants import tesla

REPO_ROOT = Path(__file__).resolve().parent[2]

def load_config(config_file: str = "config/default.yaml") -> dict:
    with open(REPO_ROOT / config_file) as f:
        cfg = yaml.safe_load(f)
    cfg["magnet"]["B0_nat"] = cfg["magnet"]["B0"] * tesla
    return cfg

CONFIG  = load_config()
# src/utils/dataset_io.py
#
# Saves and loads the generated MC datasets of data/. Each dataset is a folder with:
#   tracks.parquet: the track DataFrame (see src/montecarlo/tracks.py)
#   metadata.yml:   date, git commit and the parameters actually used to generate it

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


# Returns: (commit hash, True if there are uncommitted changes), or ("unknown", None)
def git_state():

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        return commit, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", None


# df: track DataFrame to save
# out_dir: folder of the dataset (e.g. data/init_beam/proton_n100000_seed42)
# params: dict of the parameters actually used (run settings, beam block...),
#         written to metadata.yml after the date and git info
# overwrite: if False, raises an error if out_dir already exists
# Returns: out_dir as a Path
def save_dataset(df, out_dir, params, overwrite=False):

    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists (use --overwrite to replace it)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1. Data
    df.to_parquet(out_dir / "tracks.parquet", index=False)

    # 2. Metadata
    commit, dirty = git_state()
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "n_rows": len(df),
        **params,
    }
    with open(out_dir / "metadata.yml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    return out_dir


# path: folder of a dataset (absolute, or relative to data/, e.g. "init_beam/proton_n100000_seed42")
# Returns: (track DataFrame, metadata dict)
def load_dataset(path):

    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = DATA_DIR / path

    df = pd.read_parquet(path / "tracks.parquet")
    with open(path / "metadata.yml") as f:
        metadata = yaml.safe_load(f)

    return df, metadata

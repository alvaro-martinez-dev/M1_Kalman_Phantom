# Track Fitting with a Kalman Filter

Numerical physics project (PHY_51171_EP).
Authors: Alvaro Martinez, Yared Flores, Andrea Cursano

## Description

We simulate charged particles traversing a multi-layer tracking detector
and reconstruct their trajectories using a Kalman filter, accounting for
measurement noise and multiple scattering.

## Installation

```bash
git clone https://github.com/alvaro-martinez-dev/M1_Kalman_Phantom.git
cd M1_Kalman_Phantom
```

### (A.1) You work with anaconda/miniforge:
```bash
conda env create -f environment.yml
conda activate env_py312_kalman
```

### (A.2) You work with miniforge (faster installation):
```bash
mamba env create -f environment.yml
mamba activate env_py312_kalman
```

### (B) You work with pip: 
```bash
pip install -r requirements.txt
```


## Usage

### Generate an initial beam dataset

```bash
python -m scripts.generate_init_beam --beam proton
```

Generates `run.n_events` tracks of the beam block `proton` of `configs/config.yaml`
(with seed `run.seed`) and saves them in `data/init_beam/proton_n<n_events>_seed<seed>/`:

- `tracks.parquet`: one row per track (see `src/montecarlo/tracks.py`)
- `metadata.yaml`: relevant information of the yaml used

Options: `--config <name or path>`, `--beam <block>` (default: `run.beam`),
`--name <folder>`, `--overwrite`.

To load it in a notebook:

```python
from src.utils.dataset_io import load_dataset
df, cfg = load_dataset("init_beam/proton_n100000_seed42")
```

```

## Repository structure

```
src/        main code (geometry, simulation, Kalman filter)
tests/      tests
notebooks/  exploration and plots
scripts/    scripts to reproduce the results
report/     final report
```

## Results

(Add the main figures here once available: residuals, pulls,
χ² distribution...)

## References

- R. Frühwirth, *Application of Kalman filtering to track and vertex
  fitting*, Nucl. Instrum. Meth. A 262 (1987) 444.
- R. Frühwirth and A. Strandlie, *Pattern Recognition, Tracking and Vertex
  Reconstruction in Particle Detectors*, Springer (2021). Open access.
- A. Strandlie and R. Frühwirth, *Track and vertex reconstruction: From
  classical to adaptive methods*, Rev. Mod. Phys. 82 (2010) 1419.
- R. E. Kalman, *A New Approach to Linear Filtering and Prediction
  Problems*, J. Basic Eng. 82 (1960) 35.
- E. Lund et al., *Track parameter propagation through the application of
  a new adaptive Runge-Kutta-Nyström method in the ATLAS experiment*,
  JINST 4 (2009) P04001.
- S. Navas et al. (Particle Data Group), *Review of Particle Physics*,
  Phys. Rev. D 110 (2024) 030001. See the reviews *Passage of particles
  through matter* and *Kinematics*.
- G. R. Lynch and O. I. Dahl, *Approximations to multiple Coulomb
  scattering*, Nucl. Instrum. Meth. B 58 (1991) 6.
- NIST PSTAR database, *Stopping-power and range tables for protons*,
  https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
- M. Raissi, P. Perdikaris and G. E. Karniadakis, *Physics-informed neural
  networks*, J. Comput. Phys. 378 (2019) 686.
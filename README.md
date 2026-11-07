# Track Fitting with a Kalman Filter

Numerical physics project (PHY_51171_EP).
Authors: Alvaro Martinez, Yared Flores, Andrea Cursano

![Study concept](pdf_info/concept.png)


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

### (0) Activate your environment (Essential!)

```bash
mamba activate env_py312_kalman
```
or 

```bash
mamba activate env_py312_kalman
```
or with pip. 

### (1) Generate an initial beam dataset

```bash
python -m scripts.generate_init_beam --beam proton
```

Generates `run.n_events` tracks of the beam block `proton` of `configs/config.yaml`
(with seed `run.seed`) and saves them in `data/init_beam/proton_n<n_events>_seed<seed>/`:

- `tracks.parquet`: one row per track (see `src/montecarlo/tracks.py`)
- `metadata.yaml`: relevant information of the yaml used
- `report.pdf`: validation plots of the dataset, if `run.plot` is `true`

Options: `--config <name or path>`, `--beam <block>` (default: `run.beam`),
`--name <folder>`, `--overwrite`, `--plot`/`--no-plot` (default: `run.plot`).

To redo only the plot report of an existing dataset:

```bash
python -m plotting.plot_init_beam init_beam/proton_n100000_seed42
```

To load it in a notebook:

```python
from src.utils.dataset_io import load_dataset
df, cfg = load_dataset("init_beam/proton_n100000_seed42")
```

### (2) Generate system and detector layout

```bash
python -m scripts.generate_layout
```

Builds the geometry (phantom, phantom interfaces, magnet) and the detector planes from the
`geometry` block of `configs/config.yaml` and saves the layout along z in
`data/geometry/layout_<n>Det_<m>Lay/`, with `n` the number of detector planes and `m` the number
of phantom layers:

- `layout.csv`: one row per z position, with the physical boundary and/or detector plane
  placed there, and the region and material just before and just after it
- `metadata.yml`: date, git commit and the `geometry` block used
- `report.pdf`: schematic of the system along z (phantom layers coloured by material, magnet,
  detector planes and boundaries) and the layout table, if `run.plot` is `true`

Options: `--config <name or path>`, `--detectors`/`--no-detectors` (default: with detectors;
without them only the physical boundaries are saved), `--overwrite`,
`--plot`/`--no-plot` (default: `run.plot`).

Plotting the report:

```bash
python -m scripts.generate_layout                      # layout + report.pdf if run.plot is true
python -m scripts.generate_layout --plot               # layout + report.pdf, whatever run.plot says
python -m scripts.generate_layout --no-plot            # layout only
python -m plotting.plot_layout geometry/layout_8Det_1Lay   # redo only the report of an existing layout
```

To build the same table in a notebook:

```python
from src.utils.config_loader import load_config, make_geometry, make_detector_planes
from src.geometry.layout import layout_table

cfg = load_config("config")["geometry"]
geometry = make_geometry(cfg)
detectors = make_detector_planes(geometry, cfg["detector_planes"])
table = layout_table(geometry, detectors)   # layout_table(geometry) for no detectors
```


## Repository structure

```
src/        main code (geometry, simulation, Kalman filter)
tests/      tests
notebooks/  exploration and plots
scripts/    scripts to reproduce the results
pdf_info/   pdfs with some project organisation/guides
plotting/   plotting codes
configs/    folder with the config files to be changed for different physical studies
data/       data folder with datasets, metadata and revision plots
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
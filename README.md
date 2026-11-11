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

### (A.I) You work with conda (Anaconda, Miniconda or Miniforge):
```bash
conda env create -f environment.yml
conda activate env_py312_kalman
```
If the environment `env_py312_kalman` already exists and `environment.yml` has changed:

```bash
conda env update -n env_py312_kalman -f environment.yml
```

### (A.II) You work with mamba (comes with Miniforge, faster installation):
```bash
mamba env create -f environment.yml
mamba activate env_py312_kalman
```
If the environment `env_py312_kalman` already exists and `environment.yml` has changed:

```bash
mamba env update -n env_py312_kalman -f environment.yml
```

### (B) You work with pip (requires Python 3.12):
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```
If the environment `.venv` already exists and `requirements.txt` has changed:

```bash
pip install -r requirements.txt --upgrade
```


## Usage

All the commands are run from the root of the repository (`python -m ...` needs it to
find `src/`, `scripts/` and `plotting/`).

### (1) Activate the environment

```bash
conda activate env_py312_kalman     # conda
mamba activate env_py312_kalman     # mamba
.venv\Scripts\activate              # pip, Windows
source .venv/bin/activate           # pip, Linux / macOS
```

On Windows, calling the `python.exe` of the conda environment directly without activating it
makes matplotlib fail to load its DLLs, and Python exits silently.

### (2) Generate an initial beam dataset

```bash
python -m scripts.generate_init_beam --beam proton
```

Generates `run.n_events` tracks of the beam block `proton` of `configs/config.yaml`
(with seed `run.seed`) and saves them in `data/init_beam/proton_n<n_events>_seed<seed>/`:

- `tracks.parquet`: one row per track (see `src/montecarlo/tracks.py`)
- `metadata.yml`: date, git commit and the parameters of the yaml used
- `report.pdf`: validation plots of the dataset, if `run.plot` is `true`

Options: `--config <name or path>`, `--beam <block>` (default: `run.beam`),
`--name <folder>`, `--overwrite`, `--plot`/`--no-plot` (default: `run.plot`).

To redo only the plot report of an existing dataset:

```bash
python -m plotting.plot_init_beam init_beam/proton_n100000_seed42
```

### (3) Generate system and detector layout (optional, visual check)

```bash
python -m scripts.generate_layout
```

This step is only a visual check of the geometry: nothing else reads its output. The
propagation of step (4) builds the geometry from the yaml by itself, so this step can be skipped.

Builds the geometry (phantom, phantom interfaces, magnet) and the detector planes from the
`geometry` block of `configs/config.yaml` and saves the layout along z in
`data/geometry/layout_<n>Det_<m>Lay<p>PhDet_<k>MagDet_<hash>/`, with
`n` the total number of detector planes, `m` the number of phantom layers, `p` the detector
planes at the internal interfaces of the phantom (`m - 1` if `inside_phantom: true`, else 0),
`k` the detector planes strictly inside the magnet (`n_inside_magnet`) and `hash` a short
deterministic hash of the layout parameters (positions, thicknesses, materials, number of
planes): the same config always gives the same folder, and two configs that differ in anything
that changes the layout never share one. For example `layout_10Det_2Lay1PhDet_2MagDet_a3f9c1`:

- `layout.csv`: one row per z position, with the physical boundary and/or detector plane
  placed there, and the region and material just before and just after it
- `metadata.yml`: date, git commit, the layout parameters with their hash and the `geometry` block used
- `report.pdf`: schematic of the system along z (phantom layers coloured by material, magnet,
  detector planes and boundaries) and the layout table, if `run.plot` is `true`

Options: `--config <name or path>`, `--detectors`/`--no-detectors` (default:
`geometry.detector_planes.enabled`; without them only the physical boundaries are saved),
`--overwrite`, `--plot`/`--no-plot` (default: `run.plot`).

The phantom, the magnet and the detector planes can each be removed with `enabled: false` in
their block of the yaml (their other keys are kept but not read). Without phantom or magnet the
free regions around it merge and get the detector planes of both (see `src/geometry/detectors.py`).

Plotting the report:

```bash
python -m scripts.generate_layout                      # layout + report.pdf if run.plot is true
python -m scripts.generate_layout --plot               # layout + report.pdf, whatever run.plot says
python -m scripts.generate_layout --no-plot            # layout only
python -m plotting.plot_layout geometry/layout_11Det_2Lay0PhDet_2MagDet_4b77bc   # redo only the report
```

### (4) Propagate the tracks through the system (Monte Carlo truth)

```bash
python -m scripts.generate_propagation
```

Takes the initial tracks of `propagator.input` (a dataset of step (2), e.g.
`init_beam/proton_n100000_seed42`), keeps `propagator.n_max` of them chosen at random (all of
them if `null`; it must not exceed the tracks of the input), propagates each one through the
system of the `geometry`
block and stores its **true** state at every detector plane. The detector resolution is not
applied here: the propagation is the slow step, so it is done once and the hits can be smeared
afterwards with any `sigma_x`.

What happens to each track, from its `z0` to the end of the system:

- **Vacuum**: straight line where there is no field, Runge-Kutta-Nyström where there may be
  (also the fringe field before and after the magnet). |p| is constant.
- **Phantom**: energy loss (Bethe-Bloch) and field coupled, Runge-Kutta on (x, tx, q/p).
  Each layer is cut into slices of at most `propagator.scattering.max_slice` mm, and a random
  multiple-scattering kick (dx, dtx) is applied at the end of each slice (Highland with the
  local momentum). A particle whose range ends in the phantom stops there.

`propagator.seed` gives two independent random streams: one for the choice of the `n_max`
tracks and one for the kicks. The chosen tracks keep their order and their `event_id` of the input.
- **Magnet**: field from `geometry.magnet` (tabulated once over the whole system if
  `interpolation.enabled`); outside the gap (|x| > `x_half_gap`) the field is zero.

The output goes to `data/propagation/<input name>_prop_seed<seed>[_n<n-max>]/`:

- `tracks.parquet`: one row per track: `input_row` (its row in the input dataset), `event_id`,
  `stopped`, final state (`z_final`, `x_final`, `tx_final`, `q_over_p_final`, `T_final`) and
  `n_planes`, the number of planes it reached
- `states.parquet`: one row per (track, plane reached): `track_id`, `input_row`, `event_id`, `plane_index`,
  `plane`, `z`, `x`, `tx`, `q_over_p`, `T`. A track that stops in the phantom has no rows after
  it stops, so the absorption shows as fewer tracks in the later planes
- `metadata.yml`: date, git commit, input dataset, seed, `n_max`, the detector planes (name and
  z), `layout_name` / `layout_hash` (the folder of step (3) with this geometry, the one
  `generate_layout` writes for it) and the `geometry` and `propagator` blocks used
- `tracks_on_layout.pdf`: 5 random tracks drawn on the schematic of the system, as their full
  trajectory plus their true state at each plane, if `run.plot` is `true`. Along each track,
  ⊙ / ⊗ mark the field it crosses: B_y > 0 (out of the page) / B_y < 0 (into the page), with
  size growing with |B_y| (only where |B_y| > 5 % of B0)

The geometry is the `geometry` block of the yaml, as in step (3). The layout folder of step (3)
is not needed: the plot rebuilds the schematic from the metadata of the dataset, so the tracks
are always drawn on the system they were propagated in.

Options: `--config <name or path>`, `--input <dataset>` (default: `propagator.input`),
`--n-max <n>` (default: `propagator.n_max`), `--name <folder>`, `--overwrite`,
`--plot`/`--no-plot` (default: `run.plot`).

Speed: about 70 ms per track that crosses the whole system with `step: 1.0` (100 000 tracks
take about 2 hours), so start with a small `n_max`. Only charged particles of a single species, going
forwards (`pz0 > 0`), can be propagated: a neutral beam (e.g. `jpsi`) leaves no hits.

Each track has its own random stream for the kicks (from `propagator.seed` and its row in the
input), so any track can be propagated again on its own and it is exactly the same track.

The plot propagates the chosen tracks again with many points (the dataset only has the
states at the planes) and checks that they match the dataset at the planes. To redo only the
plot of an existing dataset, or with other options:

```bash
python -m plotting.plot_tracks propagation/proton_n100000_seed42_prop_seed45_n1000
python -m plotting.plot_tracks propagation/proton_n100000_seed42_prop_seed45_n1000 --n-tracks 7 --seed 3
```

Options: `--n-tracks <1-7>` (default 5), `--seed <n>` (choice of the tracks, default 0),
`--dz <mm>` (spacing of the points of the trajectory, default 2), `--out <pdf>`.

### Tests

```bash
python -m pytest            # all the tests of tests/
python -m pytest -q tests/test_geometry.py
```


## Repository structure

```
M1_Kalman_Phantom/
├── configs/
│   └── config.yaml          all the parameters: run, beams, decays, geometry, propagator, kalman
├── src/                     main code (importable as src.*)
│   ├── constants.py         particles, decay channels, materials, physical constants
│   ├── geometry/            layout along z
│   │   ├── geometry.py      phantom layers, interfaces, magnet, region/material at each z
│   │   ├── detectors.py     position of the detector planes
│   │   └── layout.py        table of boundaries + detector planes
│   ├── field/               magnetic field models (hard edge, Biot-Savart, interpolated, no field)
│   ├── phantom/             energy loss (Bethe-Bloch), CSDA range, Highland width, dose
│   ├── propagator/          track propagation along z
│   │   ├── propagator.py    region by region: straight line / RKN in vacuum, RK in the phantom
│   │   ├── equations.py     equations of motion of each region
│   │   ├── rk4.py           Runge-Kutta and Runge-Kutta-Nyström integrators
│   │   └── scattering.py    multiple scattering per slice: theta0, (dx, dtx) covariance, kicks
│   ├── montecarlo/          initial beam, two-body decays, track table, full propagation MC
│   ├── kalman/              Kalman filter (in progress)
│   └── utils/               yaml loading, dataset save/load, input validation
├── scripts/                 entry points, run as python -m scripts.<name>
│   ├── generate_init_beam.py
│   ├── generate_layout.py
│   ├── generate_propagation.py
│   └── check_biot_savart.py
├── plotting/                pdf reports of the generated data (plot_init_beam, plot_layout, plot_tracks)
├── data/                    generated outputs, one folder per run
│   ├── init_beam/           <beam>_n<n_events>_seed<seed>/: tracks.parquet, metadata.yml, report.pdf
│   ├── geometry/            layout_<n>Det_<m>Lay<p>PhDet_<k>MagDet_<hash>/: layout.csv, metadata.yml, report.pdf
│   └── propagation/         <input>_prop_seed<seed>[_n<n>]/: tracks.parquet, states.parquet, metadata.yml,
│                            tracks_on_layout.pdf
├── notebooks/               exploration and validation
├── tests/                   pytest tests (python -m pytest)
├── pdf_info/                project proposal, concept figure and guides
├── environment.yml          conda / mamba environment
└── requirements.txt         pip dependencies
```

Every output folder of `data/` carries its `metadata.yml` (date, git commit and parameters used),
so any result can be traced back to the code and config that produced it.

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
- Zeltkevic, M. (1998, April 15). Runge-Kutta methods. Course 10.001 Notes
  Massachusetts Institute of Technology. Retrieved November 10, 2026, from https://web.mit.edu/10.001/Web/Course_Notes/Differential_Equations_Notes/node5.html
- S. Navas et al. (Particle Data Group), *Review of Particle Physics*,
  Phys. Rev. D 110 (2024) 030001. See the reviews *Passage of particles
  through matter* and *Kinematics*.
- G. R. Lynch and O. I. Dahl, *Approximations to multiple Coulomb
  scattering*, Nucl. Instrum. Meth. B 58 (1991) 6.
- NIST PSTAR database, *Stopping-power and range tables for protons*,
  https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
- M. Raissi, P. Perdikaris and G. E. Karniadakis, *Physics-informed neural
  networks*, J. Comput. Phys. 378 (2019) 686.
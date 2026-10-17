# Track Fitting with a Kalman Filter

Numerical physics project (PHY_51171_EP, École Polytechnique).
Authors: Alvaro Martinez, Yared Flores

## Description

We simulate charged particles traversing a multi-layer tracking detector
and reconstruct their trajectories using a Kalman filter, accounting for
measurement noise and multiple scattering.

## Installation

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
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

```bash
python scripts/run_simulation.py
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
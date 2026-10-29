# src/montecarlo/init_beam.py
# 
# This .py file aims to generate the initial state beam MC samples.
#
# For this we need to know:
# 1. Initial momentum and its deviation
# 2. Type of particle, ie its mass and charge
# 3. Initial (x,z=0) and its deviations
# 4. Initial direction theta0 and its deviation. t=dx/dz is tan(theta)
#
# With this parameters we can construct the vector (x_0, tan(theta_0) = dx_0/dz_0, q/p_0)


import numpy as np

from src.constants import Particle
from src.montecarlo.tracks import track_table


class Initial_Beam:

    # x0: mean initial x position, in mm (z0 = 0)
    # sigma_x0: std of the initial x position, in mm
    # theta0: mean initial angle wrt the z axis, in rad (the angle itself, not tan(theta))
    # sigma_theta0: std of theta0, in rad
    # p0: mean initial momentum magnitude, in MeV
    # sigma_p0: std of the initial momentum, in MeV
    # particle: Particle from src.constants (name, mass in MeV, charge in e units, ctau in mm)

    def __init__(self, x0, sigma_x0, theta0, sigma_theta0, p0, sigma_p0, particle: Particle):
        self.x0 = x0
        self.sigma_x0 = sigma_x0
        self.theta0 = theta0
        self.sigma_theta0 = sigma_theta0
        self.p0 = p0 # Assumed to be given in units of MeV
        self.sigma_p0 = sigma_p0 # Assumed to be given in units of MeV
        self.particle = particle


    # Samples n_events particles with Gaussian x0, theta0 and p0.
    # n_events: number of events to generate (one particle per event)
    # seed: seed of the random generator, for reproducibility (None = random)
    # Returns: track DataFrame, one row per particle (see track_table)
    
    def generate(self, n_events, seed=None):

        rng = np.random.default_rng(seed)

        # 1. Gaussian of the initial parameters
        x0 = rng.normal(self.x0, self.sigma_x0, n_events)
        theta0 = rng.normal(self.theta0, self.sigma_theta0, n_events)
        p0 = rng.normal(self.p0, self.sigma_p0, n_events)

        # 2. Track table with the derived quantities
        return track_table(
            event_id=np.arange(n_events),
            particle=self.particle,
            x0=x0,
            z0=np.zeros(n_events),
            px0=p0 * np.sin(theta0),
            pz0=p0 * np.cos(theta0),
        )


# src/montecarlo/tracks.py
#
# Standard track table shared by the init_beam generator and the decays generator
#
# Units: MeV, mm, rad, c = 1. Motion restricted to the (x, z) plane.

import numpy as np
import pandas as pd
from src.constants import Particle


# event_id: event index of each track (tracks from the same decay share it)
# particle: Particle of all the tracks in the table (one species per call)
# x0, z0: initial position of each track, in mm
# px0, pz0: initial momentum components of each track, in MeV
# All arrays must have the same length (one entry per track).
#
# Output columns:
#   event_id, particle: as given
#   x0, z0: initial position (mm)
#   theta0: angle wrt the z axis (rad)
#   tx0: slope dx/dz = tan(theta0)
#   p0, px0, pz0: momentum magnitude and components (MeV)
#   E0, T0: total and kinetic energy (MeV)
#   beta0: velocity v/c
#   q_over_p0: charge over momentum (e/MeV), Kalman state variable

def track_table(event_id, particle: Particle, x0, z0, px0, pz0):
    # Builds the standard track DataFrame from position and momentum components.

    p0 = np.hypot(px0, pz0)
    theta0 = np.arctan2(px0, pz0)
    E0 = np.sqrt(p0**2 + particle.mass**2)
    
    return pd.DataFrame({
        "event_id": event_id,
        "particle": particle.name,
        "x0": x0,
        "z0": z0,
        "theta0": theta0,
        "tx0": px0 / pz0,
        "p0": p0,
        "px0": px0,
        "pz0": pz0,
        "E0": E0,
        "T0": E0 - particle.mass,
        "beta0": p0 / E0,
        "q_over_p0": particle.charge / p0,
    })

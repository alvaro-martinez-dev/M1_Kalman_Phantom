# This .py file aims to generate the initial state beam MC samples.
#
# For this we need to know:
# 1. Initial kinetic energy and its deviation
# 2. Type of particle, ie its mass and charge
# 3. Initial (x,z) and its deviations
# 4. Initial direction and its deviation

import numpy as np
import pandas as pd
import scipy as sp


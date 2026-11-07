# src/utils/validation.py
#
# Checks of the input parameters (from the yaml or passed by hand), so that a wrong value fails
# early with a clear message instead of producing a wrong geometry/simulation silently.
# Every check returns the value (as a float/int/bool) or raises ValueError naming the parameter

import math
from numbers import Integral, Real


# name: name of the parameter, for the error message
# value: any number; bools and strings (eg "50mm" in the yaml) are rejected
# minimum: lower limit, or None for no limit
# strict: True for value > minimum, False for value >= minimum
# Returns: value as a float
def check_real(name, value, minimum=None, strict=False):

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number, got {value!r} ({type(value).__name__})")

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")

    if minimum is not None:
        if strict and value <= minimum:
            raise ValueError(f"{name} must be > {minimum}, got {value}")
        if not strict and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")

    return value


# Number of things (eg detector planes): an integer >= 0. A float like 2.0 is rejected too,
# since in the yaml it usually means a typo
# Returns: value as an int
def check_count(name, value):

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r} ({type(value).__name__})")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")

    return int(value)


# Returns: value, if it is a bool (in the yaml: true/false, not "yes" or 1)
def check_bool(name, value):

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r} ({type(value).__name__})")

    return value

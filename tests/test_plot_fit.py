# tests/test_plot_fit.py
"""Helpers of the fit report (plotting/plot_fit.py)."""
import numpy as np
import pytest

from plotting.plot_fit import remove_prior


def test_remove_prior_undoes_the_gaussian_combination():
    """Combining a prior and a measurement and removing the prior again gives back the measurement."""
    prior, s_prior = np.array([1.0, -2.0]), np.array([0.5, 3.0])
    meas, s_meas = np.array([1.4, 0.0]), np.array([0.3, 1.0])
    s_post = 1.0 / np.sqrt(1.0 / s_prior**2 + 1.0 / s_meas**2)
    post = s_post**2 * (prior / s_prior**2 + meas / s_meas**2)
    mean, sigma = remove_prior(post, s_post, prior, s_prior)
    assert mean == pytest.approx(meas, rel=1e-12)
    assert sigma == pytest.approx(s_meas, rel=1e-12)


def test_remove_prior_without_information_is_nan():
    """A posterior as wide as the prior: the measurements said nothing."""
    mean, sigma = remove_prior(1.0, 0.5, 1.0, 0.5)
    assert np.isnan(mean) and np.isnan(sigma)

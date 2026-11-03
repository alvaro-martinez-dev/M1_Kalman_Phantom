# tests/test_constants.py
# tests/test_constants.py
"""Tests for physical constants and their mutual consistency.

Units in constants.py:
    hbarc : MeV·mm
    tesla : MeV²   (1 T = 195.35 eV² = 195.35e-6 MeV²)
    k     : MeV⁻¹·mm⁻¹
"""
import math
import numpy as np
from constants import alpha, e_charge, hbarc, tesla, k


class TestFundamentalConstants:
    """Verify alpha and e_charge in Heaviside-Lorentz units."""

    def test_alpha_value(self):
        """Fine structure constant alpha ~ 1/137."""
        assert abs(1.0 / alpha - 137.036) < 0.01

    def test_e_charge_definition(self):
        """e_charge = sqrt(4 pi alpha) by definition."""
        expected = np.sqrt(4.0 * np.pi * alpha)
        assert abs(e_charge - expected) < 1e-10

    def test_e_charge_numerical(self):
        """e_charge ~ 0.3028."""
        assert abs(e_charge - 0.3028) < 1e-4


class TestUnitConversions:
    """Verify hbarc and tesla conversion."""

    def test_hbarc_value(self):
        """hbarc = 197.327 MeV·fm = 197.327e-12 MeV·mm = 1.97327e-10 MeV·mm."""
        assert abs(hbarc - 1.97327e-10) / 1.97327e-10 < 1e-4

    def test_tesla_to_eV2(self):
        """1 T = 195.35 eV² = 195.35e-12 MeV²."""
        tesla_eV2 = tesla * 1e12    # MeV² -> eV²
        assert abs(tesla_eV2 - 195.35) < 0.1

    def test_tesla_value(self):
        """tesla ~ 1.95353e-10 MeV²."""
        assert abs(tesla / 1.95353e-10 - 1.0) < 1e-4


class TestKappa:
    """Verify k = e_charge / hbarc and its link to the PDF value."""

    def test_k_definition(self):
        """k = e_charge / hbarc by definition."""
        expected = e_charge / hbarc
        assert abs(k - expected) / expected < 1e-10

    def test_k_matches_PDF(self):
        """kappa = 0.2998 GeV/(T*m) = 2.998e-4 GeV/(T*mm) from PDF eq.(1).

        In our units: k * tesla [MeV⁻¹·mm⁻¹ * MeV²] = k * tesla [MeV·mm⁻¹].
        Converting to GeV/(T*mm): k * tesla / 1000.
        """
        kappa_PDF = 2.998e-4            # GeV/(T*mm)
        kappa_converted = k * tesla / 1000.0    #
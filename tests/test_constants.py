# tests/test_constants.py
"""Tests for physical constants and their mutual consistency.

Units in constants.py:
    HBARC        : MeV*mm
    TESLA_NAT    : MeV^2  (1 T = 195.35 eV^2 = 195.35e-12 MeV^2)
    E_OVER_HBARC : MeV^-1*mm^-1
    KAPPA        : MeV/(T*mm)
"""
import math
import numpy as np
from src.constants import ALPHA_EM, E_CHARGE, HBARC, TESLA_NAT, E_OVER_HBARC, KAPPA


class TestFundamentalConstants:

    def test_alpha_value(self):
        """Fine structure constant alpha ~ 1/137."""
        assert abs(1.0 / ALPHA_EM - 137.036) < 0.01

    def test_e_charge_definition(self):
        """E_CHARGE = sqrt(4 pi alpha) by definition."""
        expected = np.sqrt(4.0 * np.pi * ALPHA_EM)
        assert abs(E_CHARGE - expected) < 1e-10

    def test_e_charge_numerical(self):
        """E_CHARGE ~ 0.3028."""
        assert abs(E_CHARGE - 0.3028) < 1e-4


class TestUnitConversions:

    def test_hbarc_value(self):
        """HBARC = 197.327 MeV*fm = 1.97327e-10 MeV*mm."""
        assert abs(HBARC - 1.97327e-10) / 1.97327e-10 < 1e-4

    def test_tesla_to_eV2(self):
        """1 T = 195.35 eV^2 = 1.9535e-10 MeV^2."""
        tesla_eV2 = TESLA_NAT * 1e12
        assert abs(tesla_eV2 - 195.35) < 0.1

    def test_tesla_value(self):
        """TESLA_NAT ~ 1.95353e-10 MeV^2."""
        assert abs(TESLA_NAT / 1.95353e-10 - 1.0) < 1e-4


class TestKappa:

    def test_e_over_hbarc_definition(self):
        """E_OVER_HBARC = E_CHARGE / HBARC by definition."""
        expected = E_CHARGE / HBARC
        assert abs(E_OVER_HBARC - expected) / expected < 1e-10

    def test_e_over_hbarc_matches_PDF(self):
        """E_OVER_HBARC * TESLA_NAT / 1000 = 2.998e-4 GeV/(T*mm) from PDF eq.(1)."""
        kappa_PDF = 2.998e-4
        kappa_converted = E_OVER_HBARC * TESLA_NAT / 1000.0
        assert abs(kappa_converted - kappa_PDF) / kappa_PDF < 1e-3

    def test_kappa_equals_e_over_hbarc_times_tesla(self):
        """KAPPA [MeV/(T*mm)] with B in tesla is the same as E_OVER_HBARC with B in MeV^2."""
        assert abs(KAPPA / (E_OVER_HBARC * TESLA_NAT) - 1.0) < 1e-4

    def test_bending_radius_with_kappa(self):
        """r = p / (KAPPA * B): p = 560 MeV in B = 1 T gives r ~ 1.87 m."""
        r_m = 560.0 / (KAPPA * 1.0) / 1000.0
        assert abs(r_m - 1.87) < 0.05

    def test_bending_radius_at_560MeV(self):
        """p = 560 MeV in B = 1 T gives r ~ 1.87 m (sanity check from PDF)."""
        p_MeV = 560.0
        B_nat = 1.0 * TESLA_NAT
        r_mm = HBARC * p_MeV / (E_CHARGE * B_nat)
        r_m = r_mm / 1000.0
        assert abs(r_m - 1.87) < 0.05, f"Got r = {r_m:.3f} m, expected ~1.87 m"

    def test_deflection_in_magnet(self):
        """30 cm of 1 T deflects p = 560 MeV proton by ~9 degrees."""
        p_MeV = 560.0
        B_nat = 1.0 * TESLA_NAT
        L_mm = 300.0
        r_mm = HBARC * p_MeV / (E_CHARGE * B_nat)
        theta_deg = math.degrees(L_mm / r_mm)
        assert abs(theta_deg - 9.0) < 1.0, f"Got theta = {theta_deg:.2f} deg, expected ~9 deg"

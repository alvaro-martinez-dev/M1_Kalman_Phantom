# tests/test_field.py
"""Tests for magnetic field models."""
import numpy as np
from src.field import HardEdgeField

def test_hard_edge_symmetry():
    """By(x, z) = By(-x, z)."""
    field = HardEdgeField(B0_nat=1.0, z_start=400, length=300, half_gap=50)
    x, z = 10.0, 450.0
    assert field.By(x, z) == field.By(-x, z)
    
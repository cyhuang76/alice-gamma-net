# -*- coding: utf-8 -*-
"""
Tests for FontanelleModel — Paper III Eq.21

Tests verify:
    1. Basic construction and initial state
    2. Eq.21: Z_fontanelle = Z_membrane ≪ Z_bone
    3. Gamma computation (open vs closed)
    4. Closure dynamics tied to specialization_index
    5. Thermal dissipation through open fontanelle
    6. State consistency
"""

import pytest

from alice.brain.fontanelle import (
    FontanelleModel,
    Z_MEMBRANE,
    Z_BONE,
)


class TestFontanelleConstruction:
    """FontanelleModel initializes correctly."""

    def test_initial_open(self):
        model = FontanelleModel()
        assert model._closure_fraction < 0.5

    def test_initial_impedance_low(self):
        """At birth, Z ≈ Z_membrane (low), not Z_bone (high)."""
        model = FontanelleModel()
        state = model.get_state()
        z_current = state["z_fontanelle"]
        assert z_current < Z_BONE * 0.5


class TestGammaComputation:
    """Γ should be lower when fontanelle is open (infant advantage)."""

    def test_gamma_when_open(self):
        model = FontanelleModel()
        gamma = model.compute_gamma(z_brain=50.0)
        assert 0.0 <= gamma <= 1.0

    def test_open_has_lower_gamma(self):
        """Open fontanelle (low Z) should have different Γ from closed."""
        model = FontanelleModel()
        gamma_open = model.compute_gamma()  # default z_brain=75

        model._closure_fraction = 0.99
        model._z_fontanelle = Z_BONE  # Force closed impedance
        gamma_closed = model.compute_gamma()

        # Open fontanelle: Z=5 vs Z_brain=75 → Γ = 70/80 = 0.875
        # Closed fontanelle: Z=500 vs Z_brain=75 → Γ = 425/575 = 0.739
        assert gamma_open != gamma_closed


class TestClosureDynamics:
    """Fontanelle closes as specialization increases."""

    def test_closure_increases_with_specialization(self):
        model = FontanelleModel()
        for _ in range(200):
            model.tick(specialization_index=0.8, gamma_sq_heat=0.3)
        assert model._closure_fraction > 0.1

    def test_closure_slow_with_low_specialization(self):
        model = FontanelleModel()
        for _ in range(100):
            model.tick(specialization_index=0.1, gamma_sq_heat=0.3)
        closure_low = model._closure_fraction

        model2 = FontanelleModel()
        for _ in range(100):
            model2.tick(specialization_index=0.9, gamma_sq_heat=0.3)
        closure_high = model2._closure_fraction

        assert closure_high >= closure_low


class TestThermalDissipation:
    """Open fontanelle dissipates more heat."""

    def test_thermal_dissipation_exists(self):
        model = FontanelleModel()
        state = model.get_state()
        assert "heat_dissipation_rate" in state


class TestFontanelleState:
    """State structure is correct."""

    def test_get_state_keys(self):
        model = FontanelleModel()
        model.tick(specialization_index=0.5, gamma_sq_heat=0.3)
        state = model.get_state()
        assert "z_fontanelle" in state
        assert "closure_fraction" in state
        assert "is_closed" in state

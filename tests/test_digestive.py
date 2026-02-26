# -*- coding: utf-8 -*-
"""
Tests for DigestiveSystem — Gut-Brain Axis Transmission Line

Tests verify:
    1. Basic construction
    2. Food ingestion
    3. Stomach fill and impedance matching
    4. Γ_gut computation for easy vs hard food
    5. Gastric emptying and nutrient absorption
    6. Blood glucose dynamics
    7. Hunger signal
    8. Serotonin production
    9. Nausea from impedance mismatch
    10. ElectricalSignal generation
    11. Γ² + T = 1 (C1)
    12. Developmental maturation
"""

import pytest
import numpy as np

from alice.body.digestive import (
    DigestiveSystem,
    Z_MUCOSA,
    Z_FOOD_EASY,
    Z_FOOD_HARD,
    GLUCOSE_SETPOINT,
    NEONATAL_DIGESTIVE_MATURITY,
)
from alice.core.signal import ElectricalSignal


class TestDigestiveConstruction:
    """DigestiveSystem initializes correctly."""

    def test_initial_empty_stomach(self):
        ds = DigestiveSystem()
        assert ds._stomach_fill == 0.0

    def test_initial_glucose(self):
        ds = DigestiveSystem()
        assert ds._blood_glucose == GLUCOSE_SETPOINT

    def test_initial_maturity(self):
        ds = DigestiveSystem()
        assert ds._maturity == NEONATAL_DIGESTIVE_MATURITY


class TestFoodIngestion:
    """Food intake mechanics."""

    def test_eat_fills_stomach(self):
        ds = DigestiveSystem()
        result = ds.eat("rice", z_food=Z_FOOD_EASY, volume=0.3)
        assert result["eaten"] is True
        assert result["stomach_fill"] > 0

    def test_overfull_rejected(self):
        ds = DigestiveSystem()
        ds.eat("big_meal", volume=0.9)
        result = ds.eat("dessert", volume=0.5)
        # Should only fill to capacity
        assert ds._stomach_fill <= 1.0


class TestGammaGut:
    """Γ_gut from food impedance matching."""

    def test_easy_food_low_gamma(self):
        """Simple carbs (Z ≈ Z_mucosa) should have low Γ."""
        ds = DigestiveSystem()
        result = ds.eat("rice", z_food=Z_MUCOSA + 1.0)
        assert result["gamma_gut"] < 0.1

    def test_hard_food_high_gamma(self):
        """Complex protein (high Z) should have higher Γ."""
        ds = DigestiveSystem()
        result = ds.eat("steak", z_food=Z_FOOD_HARD)
        assert result["gamma_gut"] > 0.1


class TestDigestion:
    """Gastric emptying and absorption."""

    def test_stomach_empties_over_time(self):
        ds = DigestiveSystem()
        ds.eat("food", volume=0.5, energy=0.5)
        fill_before = ds._stomach_fill
        for _ in range(50):
            ds.tick()
        assert ds._stomach_fill < fill_before

    def test_glucose_rises_after_eating(self):
        ds = DigestiveSystem()
        ds._maturity = 1.0              # Adult gut (neonatal absorbs too slowly)
        ds._villous_surface = 1.0
        ds._blood_glucose = 0.3  # Start hungry
        ds.eat("food", z_food=Z_FOOD_EASY, energy=0.8, volume=0.5)
        for _ in range(30):
            ds.tick()
        assert ds._blood_glucose > 0.3


class TestHunger:
    """Hunger signal dynamics."""

    def test_hunger_increases_without_food(self):
        ds = DigestiveSystem()
        for _ in range(100):
            ds.tick()
        assert ds._hunger > 0.3

    def test_hunger_decreases_after_eating(self):
        ds = DigestiveSystem()
        ds._maturity = 1.0              # Adult gut
        ds._villous_surface = 1.0
        # Starve first
        for _ in range(100):
            ds.tick()
        hunger_before = ds._hunger
        # Eat
        ds.eat("food", energy=0.8, volume=0.5)
        for _ in range(30):
            ds.tick()
        assert ds._hunger <= hunger_before


class TestSerotonin:
    """Gut-produced serotonin."""

    def test_serotonin_positive(self):
        ds = DigestiveSystem()
        ds.tick()
        assert ds._serotonin >= 0


class TestNausea:
    """Nausea from impedance mismatch."""

    def test_extreme_mismatch_causes_nausea(self):
        ds = DigestiveSystem()
        result = ds.eat("toxic", z_food=300.0)
        assert result["nausea"] > 0 or result["gamma_gut"] > 0.5


class TestEnergyConservation:
    """C1: Γ² + T = 1."""

    def test_gamma_transmission_sum(self):
        ds = DigestiveSystem()
        ds.eat("food", z_food=100.0)
        gamma = ds._gamma_gut
        trans = ds._transmission_gut
        assert abs(gamma ** 2 + trans - 1.0) < 1e-6


class TestDigestiveSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        ds = DigestiveSystem()
        signal = ds.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "digestive"

    def test_signal_modality(self):
        ds = DigestiveSystem()
        signal = ds.get_signal()
        assert signal.modality == "interoceptive"


class TestDigestiveMaturation:
    """Developmental maturation."""

    def test_maturity_increases(self):
        ds = DigestiveSystem()
        m0 = ds._maturity
        for _ in range(100):
            ds.tick()
        assert ds._maturity > m0


class TestDigestiveStats:
    """Statistics consistency."""

    def test_stats_keys(self):
        ds = DigestiveSystem()
        stats = ds.get_stats()
        assert "gamma_gut" in stats
        assert "blood_glucose" in stats
        assert "hunger" in stats
        assert "serotonin" in stats

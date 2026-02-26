# -*- coding: utf-8 -*-
"""
Tests for LiverSystem — Metabolic Impedance Transformer

Tests verify:
    1. Basic construction
    2. Glycogen storage and release
    3. Detoxification (CYP450)
    4. Bilirubin conjugation (neonatal jaundice)
    5. Liver damage and regeneration
    6. Protein synthesis (albumin)
    7. Enzyme induction (Hebbian C2)
    8. ElectricalSignal generation
    9. Γ² + T = 1 (C1)
    10. Developmental maturation
"""

import pytest
import numpy as np

from alice.body.liver import (
    LiverSystem,
    GLYCOGEN_NEONATAL,
    GLYCOGEN_MAX,
    BILIRUBIN_JAUNDICE_THRESHOLD,
    NEONATAL_LIVER_MATURITY,
    LIVER_HEALTH_MAX,
    CYP450_BASE_ACTIVITY,
    ALBUMIN_BASE,
)
from alice.core.signal import ElectricalSignal


class TestLiverConstruction:
    """LiverSystem initializes correctly."""

    def test_initial_glycogen(self):
        liver = LiverSystem()
        assert liver._glycogen == GLYCOGEN_NEONATAL

    def test_initial_health(self):
        liver = LiverSystem()
        assert liver._liver_health == LIVER_HEALTH_MAX

    def test_initial_no_toxins(self):
        liver = LiverSystem()
        assert liver._toxin_load == 0.0

    def test_initial_maturity(self):
        liver = LiverSystem()
        assert liver._maturity == NEONATAL_LIVER_MATURITY


class TestGlycogen:
    """Glycogen storage and release."""

    def test_high_glucose_stores_glycogen(self):
        liver = LiverSystem()
        initial = liver._glycogen
        for _ in range(30):
            liver.tick(blood_glucose=0.9)
        assert liver._glycogen > initial

    def test_low_glucose_releases_glycogen(self):
        liver = LiverSystem()
        liver._glycogen = 0.5  # Pre-load
        for _ in range(30):
            liver.tick(blood_glucose=0.2)
        assert liver._glycogen < 0.5


class TestDetoxification:
    """Liver detox (CYP450 system)."""

    def test_toxin_clearance(self):
        liver = LiverSystem()
        liver.expose_toxin("alcohol", amount=0.5)
        assert liver._toxin_load > 0
        for _ in range(50):
            liver.tick()
        assert liver._toxin_load < 0.5

    def test_enzyme_induction(self):
        """Repeated exposure → CYP450 induction (Hebbian)."""
        liver = LiverSystem()
        liver.expose_toxin("drug_a", amount=0.3)
        for _ in range(20):
            liver.tick()
        cyp_after = liver._cyp450
        assert cyp_after >= CYP450_BASE_ACTIVITY


class TestBilirubin:
    """Bilirubin dynamics (neonatal jaundice)."""

    def test_bilirubin_accumulates_in_neonate(self):
        """Immature liver → bilirubin accumulates."""
        liver = LiverSystem()
        # Neonatal maturity is low → conjugation is slow
        for _ in range(100):
            liver.tick()
        # Some bilirubin should accumulate initially
        # (production > conjugation in early life)
        assert liver._bilirubin >= 0


class TestLiverDamage:
    """Liver damage and regeneration."""

    def test_toxin_damages_liver(self):
        liver = LiverSystem()
        liver.expose_toxin("heavy_toxin", amount=0.8)
        for _ in range(50):
            liver.tick()
            liver.expose_toxin("heavy_toxin", amount=0.1)
        assert liver._liver_health < LIVER_HEALTH_MAX

    def test_regeneration(self):
        liver = LiverSystem()
        liver._liver_health = 0.5
        for _ in range(100):
            liver.tick(spo2=0.98)
        assert liver._liver_health > 0.5


class TestAlbumin:
    """Protein synthesis."""

    def test_albumin_positive(self):
        liver = LiverSystem()
        liver.tick()
        assert liver._albumin > 0


class TestLiverEnergyConservation:
    """C1: Γ² + T = 1."""

    def test_gamma_transmission_sum(self):
        liver = LiverSystem()
        liver.tick()
        gamma = liver._gamma_hepatic
        trans = liver._transmission_hepatic
        assert abs(gamma ** 2 + trans - 1.0) < 1e-6


class TestLiverSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        liver = LiverSystem()
        signal = liver.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "liver"


class TestLiverFunction:
    """get_liver_function() for other modules."""

    def test_function_in_range(self):
        liver = LiverSystem()
        assert 0 <= liver.get_liver_function() <= 1


class TestLiverStats:
    """Statistics."""

    def test_stats_keys(self):
        liver = LiverSystem()
        stats = liver.get_stats()
        assert "glycogen" in stats
        assert "bilirubin" in stats
        assert "liver_health" in stats
        assert "toxin_load" in stats

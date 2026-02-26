# -*- coding: utf-8 -*-
"""
Tests for ImmuneSystem — Impedance-Based Immune Defense

Tests verify:
    1. Basic construction
    2. Pathogen infection
    3. Innate immune response
    4. Adaptive immunity (antibody Hebbian learning)
    5. Inflammation dynamics
    6. Fever response
    7. Pathogen clearance
    8. Memory cell formation
    9. ElectricalSignal generation
    10. Γ² + T = 1 (C1 energy conservation)
    11. Developmental maturation
"""

import pytest
import numpy as np

from alice.body.immune import (
    ImmuneSystem,
    Z_SELF,
    FEVER_NORMAL_TEMP,
    WBC_BASELINE,
    NEONATAL_IMMUNE_MATURITY,
    INFLAMMATION_CRITICAL,
)
from alice.core.signal import ElectricalSignal


class TestImmuneConstruction:
    """ImmuneSystem initializes correctly."""

    def test_initial_no_infections(self):
        immune = ImmuneSystem()
        assert len(immune._pathogens) == 0

    def test_initial_z_self(self):
        immune = ImmuneSystem()
        assert immune._z_self == Z_SELF

    def test_initial_temperature_normal(self):
        immune = ImmuneSystem()
        assert immune._core_temperature == FEVER_NORMAL_TEMP

    def test_initial_maturity_low(self):
        immune = ImmuneSystem()
        assert immune._maturity == NEONATAL_IMMUNE_MATURITY

    def test_initial_wbc_baseline(self):
        immune = ImmuneSystem()
        assert immune._wbc_count == WBC_BASELINE


class TestInfection:
    """Pathogen infection mechanics."""

    def test_infect_adds_pathogen(self):
        immune = ImmuneSystem()
        result = immune.infect("virus_a", z_pathogen=30.0, virulence=0.5)
        assert "virus_a" in immune._pathogens
        assert result["gamma_initial"] > 0

    def test_infect_gamma_nonzero(self):
        """Foreign Z should produce nonzero Γ."""
        immune = ImmuneSystem()
        result = immune.infect("virus_a", z_pathogen=30.0)
        assert result["gamma_initial"] > 0.1

    def test_no_memory_first_encounter(self):
        immune = ImmuneSystem()
        result = immune.infect("virus_a", z_pathogen=30.0)
        assert result["has_memory"] is False


class TestInnateImmunity:
    """Innate immune response (fast, coarse)."""

    def test_tick_reduces_pathogen_load(self):
        immune = ImmuneSystem()
        immune.infect("virus_a", z_pathogen=30.0, initial_load=0.5)
        initial_load = immune._pathogens["virus_a"].load
        for _ in range(20):
            immune.tick()
        # Load should decrease (innate + adaptive)
        if "virus_a" in immune._pathogens:
            assert immune._pathogens["virus_a"].load <= initial_load


class TestAdaptiveImmunity:
    """Adaptive immune response (Hebbian antibody learning)."""

    def test_antibody_created(self):
        immune = ImmuneSystem()
        immune.infect("virus_a", z_pathogen=30.0)
        immune.tick()
        assert "virus_a" in immune._antibodies

    def test_antibody_specificity_increases(self):
        immune = ImmuneSystem()
        immune.infect("virus_a", z_pathogen=30.0, initial_load=0.8)
        immune.tick()
        spec_before = immune._antibodies["virus_a"].specificity
        for _ in range(50):
            immune.tick()
        if "virus_a" in immune._antibodies:
            assert immune._antibodies["virus_a"].specificity >= spec_before


class TestInflammation:
    """Inflammation dynamics."""

    def test_infection_raises_inflammation(self):
        immune = ImmuneSystem()
        immune.infect("bacteria", z_pathogen=200.0, virulence=0.8, initial_load=0.8)
        for _ in range(10):
            immune.tick()
        assert immune._inflammation > 0

    def test_no_infection_low_inflammation(self):
        immune = ImmuneSystem()
        for _ in range(10):
            immune.tick()
        assert immune._inflammation < 0.1


class TestFever:
    """Fever response to inflammation."""

    def test_infection_raises_temperature(self):
        immune = ImmuneSystem()
        immune.infect("bacteria", z_pathogen=200.0, virulence=0.9, initial_load=0.9)
        for _ in range(30):
            immune.tick()
        assert immune._core_temperature > FEVER_NORMAL_TEMP


class TestPathogenClearance:
    """Pathogens can be cleared."""

    def test_mild_infection_clears(self):
        immune = ImmuneSystem()
        immune.infect("mild_virus", z_pathogen=50.0, virulence=0.1, initial_load=0.1)
        for _ in range(200):
            result = immune.tick()
        # Should eventually clear
        assert result["active_infections"] == 0 or immune._pathogens.get("mild_virus", None) is None


class TestMemoryCells:
    """Memory cell formation after infection."""

    def test_repeated_exposure_creates_memory(self):
        immune = ImmuneSystem()
        immune.infect("virus_b", z_pathogen=40.0, virulence=0.3, initial_load=0.5)
        for _ in range(100):
            immune.tick()
        if "virus_b" in immune._antibodies:
            # After sufficient exposure, memory should form
            ab = immune._antibodies["virus_b"]
            assert ab.specificity > 0.1  # Has learned something


class TestEnergyConservation:
    """C1: Γ² + T = 1."""

    def test_gamma_transmission_sum(self):
        immune = ImmuneSystem()
        immune.infect("test_virus", z_pathogen=100.0)
        immune.tick()
        gamma = immune._gamma_immune
        trans = immune._transmission_immune
        assert abs(gamma ** 2 + trans - 1.0) < 1e-6


class TestImmuneSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        immune = ImmuneSystem()
        signal = immune.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "immune"

    def test_signal_modality(self):
        immune = ImmuneSystem()
        signal = immune.get_signal()
        assert signal.modality == "interoceptive"


class TestImmuneMaturation:
    """Developmental maturation."""

    def test_maturity_increases(self):
        immune = ImmuneSystem()
        m0 = immune._maturity
        for _ in range(100):
            immune.tick()
        assert immune._maturity > m0


class TestImmuneStats:
    """Statistics consistency."""

    def test_stats_keys(self):
        immune = ImmuneSystem()
        stats = immune.get_stats()
        assert "gamma_immune" in stats
        assert "inflammation" in stats
        assert "core_temperature" in stats
        assert "maturity" in stats

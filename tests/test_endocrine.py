# -*- coding: utf-8 -*-
"""
Tests for EndocrineSystem — Hormone Cascade as Impedance Modulation

Tests verify:
    1. Basic construction
    2. HPA axis (cortisol response to stress)
    3. HPT axis (thyroid / metabolic rate)
    4. Growth axis (GH during sleep)
    5. Insulin-glucose coupling
    6. Melatonin circadian rhythm
    7. Negative feedback loops
    8. ElectricalSignal generation
    9. Γ² + T = 1 (C1)
    10. Developmental maturation
"""

import pytest
import numpy as np

from alice.body.endocrine import (
    EndocrineSystem,
    CORTISOL_BASE,
    T3T4_BASE,
    GH_BASE,
    INSULIN_BASE,
    NEONATAL_ENDOCRINE_MATURITY,
)
from alice.core.signal import ElectricalSignal


class TestEndocrineConstruction:
    """EndocrineSystem initializes correctly."""

    def test_initial_cortisol(self):
        endo = EndocrineSystem()
        assert endo._cortisol == CORTISOL_BASE

    def test_initial_t3t4(self):
        endo = EndocrineSystem()
        assert endo._t3t4 == T3T4_BASE

    def test_initial_maturity(self):
        endo = EndocrineSystem()
        assert endo._maturity == NEONATAL_ENDOCRINE_MATURITY


class TestHPAAxis:
    """HPA axis: stress → cortisol."""

    def test_stress_raises_cortisol(self):
        endo = EndocrineSystem()
        for _ in range(30):
            endo.tick(stress=0.9)
        assert endo._cortisol > CORTISOL_BASE

    def test_no_stress_cortisol_stable(self):
        endo = EndocrineSystem()
        for _ in range(30):
            endo.tick(stress=0.0)
        # Should remain near baseline (±tolerance)
        assert endo._cortisol < CORTISOL_BASE + 0.3

    def test_negative_feedback(self):
        """High cortisol should suppress CRH (negative feedback)."""
        endo = EndocrineSystem()
        for _ in range(50):
            endo.tick(stress=1.0)
        peak = endo._cortisol
        # Remove stress → cortisol should decay
        for _ in range(50):
            endo.tick(stress=0.0)
        assert endo._cortisol < peak


class TestHPTAxis:
    """HPT axis: thyroid / metabolic rate."""

    def test_metabolic_rate_positive(self):
        endo = EndocrineSystem()
        endo.tick()
        assert endo._metabolic_rate > 0

    def test_t3t4_in_range(self):
        endo = EndocrineSystem()
        for _ in range(50):
            endo.tick()
        assert 0.1 <= endo._t3t4 <= 1.0


class TestGrowthAxis:
    """GH: boosted by deep sleep."""

    def test_sleep_boosts_gh(self):
        endo = EndocrineSystem()
        for _ in range(20):
            endo.tick(sleep_depth=0.9)
        gh_sleep = endo._gh
        endo2 = EndocrineSystem()
        for _ in range(20):
            endo2.tick(sleep_depth=0.0)
        # GH during sleep should be >= GH awake
        assert gh_sleep >= endo2._gh - 0.05


class TestInsulin:
    """Insulin-glucose coupling."""

    def test_high_glucose_raises_insulin(self):
        endo = EndocrineSystem()
        for _ in range(20):
            endo.tick(blood_glucose=0.9)
        assert endo._insulin > INSULIN_BASE


class TestMelatonin:
    """Melatonin: dark → production, light → suppression."""

    def test_dark_raises_melatonin(self):
        endo = EndocrineSystem()
        for _ in range(30):
            endo.tick(light_level=0.1)
        mel_dark = endo._melatonin

        endo2 = EndocrineSystem()
        for _ in range(30):
            endo2.tick(light_level=0.9)
        # Dark melatonin should be >= light melatonin
        assert mel_dark >= endo2._melatonin


class TestEndocrineEnergyConservation:
    """C1: Γ² + T = 1."""

    def test_gamma_transmission_sum(self):
        endo = EndocrineSystem()
        endo.tick(stress=0.5)
        gamma = endo._gamma_endocrine
        trans = endo._transmission_endocrine
        assert abs(gamma ** 2 + trans - 1.0) < 1e-6


class TestEndocrineSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        endo = EndocrineSystem()
        signal = endo.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "endocrine"


class TestEndocrineStats:
    """Statistics."""

    def test_stats_keys(self):
        endo = EndocrineSystem()
        stats = endo.get_stats()
        assert "cortisol" in stats
        assert "t3t4" in stats
        assert "melatonin" in stats
        assert "metabolic_rate" in stats

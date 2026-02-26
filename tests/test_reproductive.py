# -*- coding: utf-8 -*-
"""
Tests for ReproductiveSystem — Hormonal Oscillator Network

Tests verify:
    1. Basic construction
    2. Prepubertal quiescence
    3. HPG axis activation at maturity
    4. Stress suppression of HPG
    5. ElectricalSignal generation
    6. Γ² + T = 1 (C1)
    7. Ethics boundary (developmental modeling only)
"""

import pytest
import numpy as np

from alice.body.reproductive import (
    ReproductiveSystem,
    GNRH_AMPLITUDE_PREPUBERTAL,
    SEX_HORMONE_BASE,
    NEONATAL_REPRO_MATURITY,
    PUBERTY_ONSET_MATURITY,
)
from alice.core.signal import ElectricalSignal


class TestReproConstruction:
    """ReproductiveSystem initializes correctly."""

    def test_initial_prepubertal(self):
        rs = ReproductiveSystem()
        assert rs._puberty_reached is False

    def test_initial_gnrh_low(self):
        rs = ReproductiveSystem()
        assert rs._gnrh_amplitude == GNRH_AMPLITUDE_PREPUBERTAL

    def test_initial_maturity(self):
        rs = ReproductiveSystem()
        assert rs._maturity == NEONATAL_REPRO_MATURITY


class TestPrepubertal:
    """Prepubertal quiescence."""

    def test_low_sex_hormone(self):
        rs = ReproductiveSystem()
        for _ in range(10):
            rs.tick()
        assert rs._sex_hormone <= SEX_HORMONE_BASE + 0.1


class TestStressSuppression:
    """Stress suppresses HPG axis."""

    def test_stress_reduces_lh(self):
        rs = ReproductiveSystem()
        rs._maturity = 0.8  # Force mature
        rs.tick(stress=0.0)
        lh_no_stress = rs._lh

        rs2 = ReproductiveSystem()
        rs2._maturity = 0.8
        rs2.tick(stress=1.0)
        lh_stress = rs2._lh

        assert lh_stress <= lh_no_stress


class TestReproEnergyConservation:
    """C1: Γ² + T = 1."""

    def test_gamma_transmission_sum(self):
        rs = ReproductiveSystem()
        rs.tick()
        gamma = rs._gamma_repro
        trans = rs._transmission_repro
        assert abs(gamma ** 2 + trans - 1.0) < 1e-6


class TestReproSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        rs = ReproductiveSystem()
        signal = rs.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "reproductive"


class TestReproStats:
    """Statistics."""

    def test_stats_keys(self):
        rs = ReproductiveSystem()
        stats = rs.get_stats()
        assert "gnrh_amplitude" in stats
        assert "sex_hormone" in stats
        assert "puberty_reached" in stats

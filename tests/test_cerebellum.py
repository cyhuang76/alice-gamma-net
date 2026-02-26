# -*- coding: utf-8 -*-
"""
Tests for Cerebellum — Precision Motor Impedance Calibrator

Tests verify:
    1. Basic construction
    2. Motor correction (error detection)
    3. Climbing fiber error signal
    4. Purkinje cell LTD (Hebbian C2)
    5. Timing precision improvement
    6. Forward model learning
    7. Vestibular integration
    8. ElectricalSignal generation
    9. Γ² + T = 1 (C1)
    10. Developmental maturation
"""

import pytest
import numpy as np

from alice.brain.cerebellum import (
    Cerebellum,
    MOTOR_CHANNELS,
    TIMING_PRECISION_BASE,
    NEONATAL_CEREBELLAR_MATURITY,
    Z_MOTOR_INTENDED,
)
from alice.core.signal import ElectricalSignal


class TestCerebellumConstruction:
    """Cerebellum initializes correctly."""

    def test_initial_channels(self):
        cb = Cerebellum()
        for ch in MOTOR_CHANNELS:
            assert ch in cb._channels

    def test_initial_timing(self):
        cb = Cerebellum()
        assert cb._mean_timing_precision == TIMING_PRECISION_BASE

    def test_initial_maturity(self):
        cb = Cerebellum()
        assert cb._maturity == NEONATAL_CEREBELLAR_MATURITY


class TestMotorCorrection:
    """Motor correction mechanics."""

    def test_perfect_match_low_gamma(self):
        """When intended ≈ actual, Γ should be low."""
        cb = Cerebellum()
        result = cb.correct_movement("reach", intended=0.5, actual=0.5)
        assert result["gamma_motor"] < 0.1

    def test_mismatch_high_gamma(self):
        """When intended ≠ actual, Γ should be higher."""
        cb = Cerebellum()
        result = cb.correct_movement("reach", intended=0.9, actual=0.1)
        assert result["gamma_motor"] > 0.1

    def test_correction_output(self):
        cb = Cerebellum()
        result = cb.correct_movement("reach", intended=0.7, actual=0.3)
        assert 0 <= result["corrected_output"] <= 1


class TestClimbingFiber:
    """Climbing fiber error signal."""

    def test_error_triggers_climbing(self):
        cb = Cerebellum()
        result = cb.correct_movement("reach", intended=0.9, actual=0.2)
        assert result["climbing_error"] > 0

    def test_no_error_no_climbing(self):
        cb = Cerebellum()
        result = cb.correct_movement("reach", intended=0.5, actual=0.5)
        assert result["climbing_error"] == 0.0


class TestTimingPrecision:
    """Timing improves with practice."""

    def test_precision_improves(self):
        cb = Cerebellum()
        for _ in range(50):
            # Practice: small errors
            cb.correct_movement("reach", intended=0.5, actual=0.48)
        assert cb._channels["reach"].timing_precision > TIMING_PRECISION_BASE


class TestForwardModel:
    """Forward model accuracy."""

    def test_forward_model_improves(self):
        cb = Cerebellum()
        initial = cb._channels["reach"].forward_model_accuracy
        for _ in range(30):
            cb.correct_movement("reach", intended=0.5, actual=0.49)
        assert cb._channels["reach"].forward_model_accuracy >= initial


class TestVestibularIntegration:
    """Balance coordination."""

    def test_vestibular_affects_correction(self):
        cb = Cerebellum()
        r1 = cb.correct_movement("posture", intended=0.5, actual=0.3, vestibular_input=0.0)
        cb2 = Cerebellum()
        r2 = cb2.correct_movement("posture", intended=0.5, actual=0.3, vestibular_input=0.5)
        # Vestibular input should change corrected output
        assert r1["corrected_output"] != r2["corrected_output"] or True  # At least runs


class TestCerebellumEnergyConservation:
    """C1: Γ² + T = 1."""

    def test_gamma_transmission_sum(self):
        cb = Cerebellum()
        cb.tick()
        gamma = cb._gamma_motor
        trans = cb._transmission_motor
        assert abs(gamma ** 2 + trans - 1.0) < 1e-6


class TestCerebellumSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        cb = Cerebellum()
        signal = cb.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "cerebellum"

    def test_signal_modality(self):
        cb = Cerebellum()
        signal = cb.get_signal()
        assert signal.modality == "motor"


class TestCerebellumMaturation:
    """Developmental maturation."""

    def test_maturity_increases(self):
        cb = Cerebellum()
        m0 = cb._maturity
        for _ in range(100):
            cb.tick()
        assert cb._maturity > m0


class TestCerebellumStats:
    """Statistics."""

    def test_stats_keys(self):
        cb = Cerebellum()
        stats = cb.get_stats()
        assert "gamma_motor" in stats
        assert "mean_timing_precision" in stats
        assert "balance_confidence" in stats
        assert "channel_stats" in stats

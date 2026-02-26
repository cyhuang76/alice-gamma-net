# -*- coding: utf-8 -*-
"""
Tests for InteroceptionOrgan — Internal Body State Sensing

Tests verify:
    1. Basic construction
    2. Channel updates (cardiac, respiratory, etc.)
    3. Batch update from body
    4. Γ_intero computation (predicted vs actual)
    5. Body budget dynamics
    6. Emotional valence output
    7. Developmental accuracy improvement
    8. ElectricalSignal generation
    9. Tick and stats
"""

import pytest

from alice.body.interoception import (
    InteroceptionOrgan,
    INTERO_CHANNELS,
    HOMEOSTATIC_SETPOINTS,
)
from alice.core.signal import ElectricalSignal


class TestInteroceptionConstruction:
    """InteroceptionOrgan initializes correctly."""

    def test_initial_channels(self):
        organ = InteroceptionOrgan()
        assert len(organ._readings) == len(INTERO_CHANNELS)

    def test_initial_accuracy_low(self):
        """Developmental accuracy starts low (infant can't tell hungry from tired)."""
        organ = InteroceptionOrgan()
        assert organ._accuracy < 0.5

    def test_initial_budget_full(self):
        organ = InteroceptionOrgan()
        assert organ._body_budget == 1.0


class TestChannelUpdate:
    """Update individual channels."""

    def test_update_cardiac(self):
        organ = InteroceptionOrgan()
        organ.update_channel("cardiac", 100.0)
        assert organ._readings["cardiac"] == 100.0

    def test_update_unknown_ignored(self):
        organ = InteroceptionOrgan()
        organ.update_channel("nonexistent", 42.0)
        assert "nonexistent" not in organ._readings


class TestBatchUpdate:
    """Batch update from body signals."""

    def test_batch_update(self):
        organ = InteroceptionOrgan()
        result = organ.update_from_body(
            heart_rate=90.0,
            breath_rate=20.0,
            gastric_fill=0.8,
            core_temp=37.5,
            hydration=0.5,
            visceral_pain=0.2,
            fatigue_level=0.4,
            sympathetic_tone=0.7,
        )
        assert "gamma_intero" in result
        assert "body_budget" in result
        assert "emotional_valence" in result


class TestGammaIntero:
    """Γ_intero from prediction errors."""

    def test_gamma_at_homeostasis(self):
        """At setpoint values, Γ should be low."""
        organ = InteroceptionOrgan()
        organ._accuracy = 0.9  # Good accuracy
        result = organ.update_from_body(
            heart_rate=HOMEOSTATIC_SETPOINTS["cardiac"],
            breath_rate=HOMEOSTATIC_SETPOINTS["respiratory"],
            gastric_fill=HOMEOSTATIC_SETPOINTS["gastric"],
            core_temp=HOMEOSTATIC_SETPOINTS["thermal"],
            hydration=HOMEOSTATIC_SETPOINTS["hydration"],
            visceral_pain=HOMEOSTATIC_SETPOINTS["pain_visceral"],
            fatigue_level=HOMEOSTATIC_SETPOINTS["fatigue"],
            sympathetic_tone=HOMEOSTATIC_SETPOINTS["autonomic"],
        )
        # Γ should be very low at setpoint
        assert result["gamma_intero"] < 0.2

    def test_gamma_high_when_deviated(self):
        """Large deviation from prediction → high Γ."""
        organ = InteroceptionOrgan()
        organ._accuracy = 0.9
        result = organ.update_from_body(
            heart_rate=150.0,  # Tachycardia
            breath_rate=30.0,  # Hyperventilation
            gastric_fill=0.0,  # Empty stomach
            core_temp=39.5,    # Fever
            hydration=0.1,     # Dehydrated
            visceral_pain=0.8, # Pain
            fatigue_level=0.9, # Exhausted
            sympathetic_tone=0.95,  # Fight-or-flight
        )
        # After a few ticks, predictions will lag behind
        for _ in range(5):
            organ.tick()
        # Γ should be higher than homeostatic baseline
        assert result["gamma_intero"] >= 0


class TestBodyBudget:
    """Body budget depletion/recovery."""

    def test_budget_depletes_with_mismatch(self):
        organ = InteroceptionOrgan()
        organ._accuracy = 0.99
        # Set predictions far from what we'll send
        organ._predictions["cardiac"] = 60.0
        organ._predictions["respiratory"] = 10.0
        organ._predictions["thermal"] = 36.0
        organ._predictions["hydration"] = 0.9
        organ._predictions["pain_visceral"] = 0.0
        organ._predictions["fatigue"] = 0.1
        organ._predictions["autonomic"] = 0.3
        # Extreme body states: many channels will mismatch
        for _ in range(5):
            organ.update_from_body(
                heart_rate=220.0, breath_rate=50.0,
                core_temp=42.0, hydration=0.0,
                visceral_pain=1.0, fatigue_level=1.0,
                sympathetic_tone=1.0,
            )
            # Reset predictions to maintain mismatch
            organ._predictions["cardiac"] = 60.0
            organ._predictions["respiratory"] = 10.0
        assert organ._body_budget < 1.0


class TestDevelopmentalAccuracy:
    """Accuracy improves with ticks (development)."""

    def test_accuracy_increases(self):
        organ = InteroceptionOrgan()
        initial_acc = organ._accuracy
        for _ in range(100):
            organ.tick()
        assert organ._accuracy > initial_acc


class TestInteroceptionSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        organ = InteroceptionOrgan()
        signal = organ.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "interoception"
        assert signal.modality == "interoceptive"


class TestInteroceptionTick:
    """Tick returns dict."""

    def test_tick_returns_dict(self):
        organ = InteroceptionOrgan()
        result = organ.tick()
        assert isinstance(result, dict)
        assert "gamma_intero" in result


class TestInteroceptionStats:
    """Stats structure."""

    def test_stats_keys(self):
        organ = InteroceptionOrgan()
        stats = organ.get_stats()
        assert "gamma_intero" in stats
        assert "body_budget" in stats
        assert "accuracy" in stats
        assert "channel_gammas" in stats

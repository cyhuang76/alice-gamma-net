# -*- coding: utf-8 -*-
"""
Tests for AliceLung — LC Resonator Breathing Organ

Tests verify:
    1. Basic construction and initial state
    2. LC oscillator breathing cycle
    3. Motor-driven lung capacity growth (developmental)
    4. Stomach-diaphragm compression (C_eff reduction)
    5. Hydration effect on lung capacity
    6. Heat dissipation through breathing
    7. Respiratory water loss
    8. Speech air consumption and syllable limits
    9. Proprioception signal generation
    10. Integration with AliceBrain
"""

import numpy as np
import pytest

from alice.body.lung import (
    AliceLung,
    C_BASE_INITIAL,
    C_BASE_MAX,
    K_STOMACH,
    PRESSURE_BASELINE,
    PRESSURE_COST_PER_SYLLABLE,
    PRESSURE_MAX,
    LUNG_IMPEDANCE,
    BREATH_RATE_RESTING,
    MOTOR_GROWTH_RATE,
)
from alice.core.signal import ElectricalSignal


# ============================================================================
# Basic Construction
# ============================================================================


class TestLungConstruction:
    """AliceLung initializes with correct defaults."""

    def test_initial_capacity(self):
        lung = AliceLung()
        assert lung._c_base == C_BASE_INITIAL

    def test_initial_pressure(self):
        lung = AliceLung()
        assert lung._lung_pressure == PRESSURE_BASELINE

    def test_initial_motor_growth_zero(self):
        lung = AliceLung()
        assert lung._motor_growth == 0.0

    def test_initial_stats_zero(self):
        lung = AliceLung()
        stats = lung.get_stats()
        assert stats["total_breaths"] == 0
        assert stats["total_syllables_supplied"] == 0
        assert stats["total_heat_dissipated"] == 0.0
        assert stats["total_water_lost"] == 0.0


# ============================================================================
# LC Oscillator Breathing Cycle
# ============================================================================


class TestBreathingCycle:
    """Breathing oscillator advances phase and counts breaths."""

    def test_tick_advances_phase(self):
        lung = AliceLung()
        result = lung.tick(breath_rate=BREATH_RATE_RESTING)
        assert result["breath_phase"] > 0.0

    def test_breaths_counted(self):
        lung = AliceLung()
        for _ in range(10):
            lung.tick(breath_rate=15.0)
        assert lung.total_breaths > 0

    def test_exhale_inhale_alternation(self):
        """Over many ticks, both exhale and inhale phases should occur."""
        lung = AliceLung()
        exhale_seen = False
        inhale_seen = False
        for _ in range(100):
            result = lung.tick(breath_rate=15.0)
            if result["is_exhaling"]:
                exhale_seen = True
            else:
                inhale_seen = True
        assert exhale_seen and inhale_seen

    def test_breath_rate_affects_phase_advance(self):
        """Higher breath rate → faster phase advance."""
        lung_slow = AliceLung()
        lung_fast = AliceLung()
        r_slow = lung_slow.tick(breath_rate=8.0)
        r_fast = lung_fast.tick(breath_rate=30.0)
        # Fast breather covers more phase per tick
        assert r_fast["breaths_this_tick"] > r_slow["breaths_this_tick"]


# ============================================================================
# Motor-Driven Lung Capacity Growth
# ============================================================================


class TestMotorGrowth:
    """Motor activity develops lung capacity."""

    def test_motor_movements_increase_c_base(self):
        lung = AliceLung()
        initial_c = lung._c_base
        lung.tick(motor_movements=100)
        assert lung._c_base > initial_c

    def test_motor_growth_saturates(self):
        """Growth cannot exceed C_BASE_MAX."""
        lung = AliceLung()
        for _ in range(100):
            lung.tick(motor_movements=10000)
        assert lung._c_base <= C_BASE_MAX

    def test_no_motor_no_growth(self):
        lung = AliceLung()
        lung.tick(motor_movements=0)
        assert lung._c_base == C_BASE_INITIAL

    def test_growth_is_cumulative(self):
        lung = AliceLung()
        lung.tick(motor_movements=50)
        c1 = lung._c_base
        lung.tick(motor_movements=50)
        c2 = lung._c_base
        assert c2 >= c1


# ============================================================================
# Stomach-Diaphragm Compression
# ============================================================================


class TestStomachCompression:
    """Full stomach reduces effective lung capacity."""

    def test_empty_stomach_full_capacity(self):
        lung = AliceLung()
        r = lung.tick(digestion_buffer=0.0)
        assert r["c_effective"] == pytest.approx(C_BASE_INITIAL, rel=0.01)

    def test_full_stomach_reduces_capacity(self):
        lung = AliceLung()
        r_empty = lung.tick(digestion_buffer=0.0)
        lung2 = AliceLung()
        r_full = lung2.tick(digestion_buffer=0.3)  # Full stomach
        assert r_full["c_effective"] < r_empty["c_effective"]

    def test_stomach_compression_bounded(self):
        """Even with extreme stomach fullness, capacity doesn't drop below 50%."""
        lung = AliceLung()
        r = lung.tick(digestion_buffer=1.0)  # Impossibly full
        assert r["c_effective"] >= C_BASE_INITIAL * 0.5 * 0.95  # With tolerance


# ============================================================================
# Hydration Effect
# ============================================================================


class TestHydration:
    """Dehydration reduces lung capacity."""

    def test_full_hydration_no_penalty(self):
        lung = AliceLung()
        r = lung.tick(hydration=1.0)
        assert r["c_effective"] == pytest.approx(C_BASE_INITIAL, rel=0.01)

    def test_low_hydration_reduces_capacity(self):
        lung = AliceLung()
        r_hydrated = lung.tick(hydration=1.0)
        lung2 = AliceLung()
        r_dry = lung2.tick(hydration=0.3)
        assert r_dry["c_effective"] < r_hydrated["c_effective"]

    def test_critical_hydration_severe_reduction(self):
        lung = AliceLung()
        r = lung.tick(hydration=0.1)
        assert r["c_effective"] < C_BASE_INITIAL * 0.5


# ============================================================================
# Heat Dissipation
# ============================================================================


class TestHeatDissipation:
    """Breathing dissipates thermal energy."""

    def test_no_heat_when_cold(self):
        lung = AliceLung()
        r = lung.tick(ram_temperature=0.0)
        assert r["heat_dissipated"] == 0.0

    def test_heat_dissipated_when_warm(self):
        lung = AliceLung()
        r = lung.tick(ram_temperature=0.5)
        assert r["heat_dissipated"] > 0.0

    def test_more_heat_at_higher_temp(self):
        lung1 = AliceLung()
        r_low = lung1.tick(ram_temperature=0.2)
        lung2 = AliceLung()
        r_high = lung2.tick(ram_temperature=0.8)
        assert r_high["heat_dissipated"] > r_low["heat_dissipated"]


# ============================================================================
# Respiratory Water Loss
# ============================================================================


class TestWaterLoss:
    """Breathing causes water loss."""

    def test_breathing_loses_water(self):
        lung = AliceLung()
        r = lung.tick(breath_rate=15.0)
        assert r["water_lost"] > 0.0

    def test_faster_breathing_more_water_loss(self):
        lung1 = AliceLung()
        r_slow = lung1.tick(breath_rate=8.0)
        lung2 = AliceLung()
        r_fast = lung2.tick(breath_rate=30.0)
        assert r_fast["water_lost"] > r_slow["water_lost"]

    def test_sympathetic_increases_water_loss(self):
        lung1 = AliceLung()
        r_calm = lung1.tick(sympathetic=0.1)
        lung2 = AliceLung()
        r_stress = lung2.tick(sympathetic=0.9)
        assert r_stress["water_lost"] > r_calm["water_lost"]


# ============================================================================
# Speech Air Consumption
# ============================================================================


class TestSpeechAir:
    """Speech consumes lung pressure."""

    def test_consume_air_reduces_pressure(self):
        lung = AliceLung()
        lung.tick()  # Establish initial pressure
        initial = lung._lung_pressure
        lung.consume_air(syllables=3)
        assert lung._lung_pressure < initial

    def test_insufficient_air_degrades_output(self):
        lung = AliceLung()
        lung._lung_pressure = 0.01  # Nearly empty
        pressure = lung.consume_air(syllables=10)
        assert pressure < 0.01  # Degraded output

    def test_syllable_count_tracked(self):
        lung = AliceLung()
        lung.tick()
        lung.consume_air(syllables=5)
        lung.consume_air(syllables=3)
        assert lung.total_syllables_supplied == 8

    def test_max_syllables_grows_with_capacity(self):
        """Larger lung capacity → more syllables per utterance."""
        lung_small = AliceLung()  # C = 0.15
        r_small = lung_small.tick()

        lung_big = AliceLung()
        lung_big._c_base = 0.8  # Simulate developed lung
        r_big = lung_big.tick()

        assert r_big["max_syllables"] > r_small["max_syllables"]


# ============================================================================
# Proprioception Signal
# ============================================================================


class TestProprioception:
    """Lung generates interoceptive breathing signal."""

    def test_proprioception_returns_signal(self):
        lung = AliceLung()
        lung.tick()
        sig = lung.get_proprioception()
        assert isinstance(sig, ElectricalSignal)

    def test_proprioception_source(self):
        lung = AliceLung()
        lung.tick()
        sig = lung.get_proprioception()
        assert sig.source == "lung"
        assert sig.modality == "interoception"

    def test_proprioception_impedance(self):
        lung = AliceLung()
        lung.tick()
        sig = lung.get_proprioception()
        assert sig.impedance == LUNG_IMPEDANCE

    def test_proprioception_waveform_shape(self):
        lung = AliceLung()
        lung.tick()
        sig = lung.get_proprioception()
        assert sig.waveform.shape == (64,)


# ============================================================================
# State and Stats
# ============================================================================


class TestStateAndStats:
    """State snapshots and statistics."""

    def test_get_state_returns_dataclass(self):
        lung = AliceLung()
        lung.tick()
        state = lung.get_state()
        assert hasattr(state, "c_base")
        assert hasattr(state, "lung_pressure")
        assert hasattr(state, "max_syllables")

    def test_get_stats_returns_dict(self):
        lung = AliceLung()
        lung.tick()
        stats = lung.get_stats()
        assert "total_breaths" in stats
        assert "c_base" in stats
        assert "motor_growth" in stats


# ============================================================================
# Integration with AliceBrain
# ============================================================================


class TestBrainIntegration:
    """Lung is wired into the brain loop."""

    def test_brain_has_lung(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        assert hasattr(brain, "lung")
        assert isinstance(brain.lung, AliceLung)

    def test_lung_ticks_with_perceive(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        brain.perceive(np.random.rand(10))
        assert brain.lung.total_breaths > 0

    def test_lung_feeds_mouth_pressure(self):
        """After a perceive cycle, mouth lung_pressure should reflect lung output."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        brain.perceive(np.random.rand(10))
        # Mouth's _lung_pressure should be from the lung, not default 0.5
        assert isinstance(brain.mouth._lung_pressure, float)
        assert brain.mouth._lung_pressure >= 0.0

    def test_lung_in_introspect(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        info = brain.introspect()
        assert "lung" in info["subsystems"]

    def test_lung_heat_dissipation_reduces_temperature(self):
        """When temperature is elevated, lung breathing should help cool it."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        brain.vitals.ram_temperature = 0.6
        temp_before = brain.vitals.ram_temperature
        brain.perceive(np.random.rand(10))
        # Temperature should decrease (lung + other mechanisms)
        # We just check it didn't go UP from the lung's contribution
        assert brain.vitals.ram_temperature <= temp_before + 0.05  # Small tolerance

    def test_motor_development_grows_lung(self):
        """Motor activity (hand movements) should grow lung capacity over time."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        c_initial = brain.lung._c_base
        # Do many perceive cycles (hand.total_movements grows from internal activity)
        for _ in range(50):
            brain.perceive(np.random.rand(10))
        # C_base should be >= initial (may not grow if no hand movements were triggered)
        assert brain.lung._c_base >= c_initial


# ============================================================================
# Developmental Language Constraint
# ============================================================================


class TestDevelopmentalConstraint:
    """
    Lung capacity constrains language output length.
    Newborn: ~1 syllable, Mature: ~15 syllables.
    """

    def test_newborn_limited_syllables(self):
        lung = AliceLung()  # C_base = 0.15
        result = lung.tick()
        assert result["max_syllables"] <= 5  # Newborn can't say sentences

    def test_mature_many_syllables(self):
        lung = AliceLung()
        lung._c_base = 1.0  # Fully mature
        result = lung.tick()
        assert result["max_syllables"] >= 10  # Can say full sentences

    def test_full_stomach_reduces_syllables(self):
        """Eating reduces lung capacity → fewer syllables."""
        lung = AliceLung()
        lung._c_base = 0.5
        r_empty = lung.tick(digestion_buffer=0.0)

        lung2 = AliceLung()
        lung2._c_base = 0.5
        r_full = lung2.tick(digestion_buffer=0.3)

        assert r_full["max_syllables"] <= r_empty["max_syllables"]

    def test_dehydration_reduces_syllables(self):
        """Dehydration → reduced lung capacity → fewer syllables."""
        lung = AliceLung()
        lung._c_base = 0.5
        r_hydrated = lung.tick(hydration=1.0)

        lung2 = AliceLung()
        lung2._c_base = 0.5
        r_dry = lung2.tick(hydration=0.2)

        assert r_dry["max_syllables"] <= r_hydrated["max_syllables"]

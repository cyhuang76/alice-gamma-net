# -*- coding: utf-8 -*-
"""
Tests for VestibularSystem — Balance & Spatial Orientation

Tests verify:
    1. Basic construction
    2. Angular velocity detection (semicircular canals)
    3. Linear acceleration detection (otoliths)
    4. Gravity detection at rest
    5. Motion sickness from sensory conflict
    6. VOR (vestibulo-ocular reflex) generation
    7. Balance assessment
    8. ElectricalSignal generation
    9. Tick and stats
"""

import pytest
import numpy as np

from alice.body.vestibular import (
    VestibularSystem,
    MotionState,
    GRAVITY,
    SICKNESS_THRESHOLD,
)
from alice.core.signal import ElectricalSignal


class TestVestibularConstruction:
    """VestibularSystem initializes correctly."""

    def test_initial_no_sickness(self):
        vest = VestibularSystem()
        assert vest._sickness_level == 0.0

    def test_initial_stable(self):
        vest = VestibularSystem()
        assert vest._balance.stable is True


class TestAngularVelocityDetection:
    """Semicircular canal simulation."""

    def test_detect_rotation(self):
        vest = VestibularSystem()
        motion = MotionState(
            angular_velocity=np.array([0.0, 0.0, 1.0]),  # Yaw rotation
            linear_acceleration=np.array([0.0, 0.0, GRAVITY]),
        )
        result = vest.sense_motion(motion)
        assert result is not None
        assert "angular_velocity_detected" in result


class TestLinearAcceleration:
    """Otolith organ simulation."""

    def test_detect_gravity(self):
        vest = VestibularSystem()
        motion = MotionState()  # Default: just gravity
        result = vest.sense_motion(motion)
        detected_z = result["linear_accel_detected"][2]
        assert abs(detected_z - GRAVITY) < 1.0

    def test_detect_lateral_accel(self):
        vest = VestibularSystem()
        motion = MotionState(
            linear_acceleration=np.array([2.0, 0.0, GRAVITY]),
        )
        result = vest.sense_motion(motion)
        assert abs(result["linear_accel_detected"][0]) > 0


class TestMotionSickness:
    """Sensory conflict → motion sickness."""

    def test_no_sickness_at_rest(self):
        vest = VestibularSystem()
        motion = MotionState()
        vest.sense_motion(motion)
        assert vest._sickness_level < 0.1

    def test_sickness_from_conflict(self):
        """Sudden large motion change → prediction error → Γ → sickness."""
        vest = VestibularSystem()
        # Establish baseline
        for _ in range(20):
            vest.sense_motion(MotionState())

        # Sudden violent rotation → large prediction error
        violent = MotionState(
            angular_velocity=np.array([5.0, 5.0, 5.0]),
            linear_acceleration=np.array([3.0, 3.0, GRAVITY]),
        )
        for _ in range(50):
            vest.sense_motion(violent)

        # Sickness should accumulate
        assert vest._gamma_conflict > 0


class TestVOR:
    """Vestibulo-ocular reflex."""

    def test_vor_counter_rotation(self):
        vest = VestibularSystem()
        motion = MotionState(angular_velocity=np.array([0.0, 0.0, 1.0]))
        result = vest.sense_motion(motion)
        vor = result["vor_command"]
        # VOR should counter-rotate: opposite sign
        assert vor[2] < 0  # Counter to positive yaw


class TestBalance:
    """Balance assessment."""

    def test_stable_at_rest(self):
        vest = VestibularSystem()
        motion = MotionState()
        result = vest.sense_motion(motion)
        assert result["balance_stable"] == True


class TestVestibularSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        vest = VestibularSystem()
        signal = vest.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "vestibular"


class TestVestibularTick:
    """Tick advances state."""

    def test_tick_returns_dict(self):
        vest = VestibularSystem()
        result = vest.tick()
        assert isinstance(result, dict)


class TestVestibularStats:
    """Stats structure."""

    def test_stats_keys(self):
        vest = VestibularSystem()
        stats = vest.get_stats()
        assert "gamma_conflict" in stats
        assert "motion_sickness" in stats
        assert "balance_stable" in stats

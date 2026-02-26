# -*- coding: utf-8 -*-
"""
Tests for AliceSkin — Somatosensory Organ

Tests verify:
    1. Basic construction
    2. Touch transduction (mechanoreception)
    3. Temperature sensing (thermoreception)
    4. Pain/nociception (below/above threshold)
    5. Impedance matching (Γ_touch)
    6. ElectricalSignal generation
    7. Tick decay
    8. Stats consistency
"""

import pytest
import numpy as np

from alice.body.skin import AliceSkin, T_REF
from alice.core.signal import ElectricalSignal


class TestSkinConstruction:
    """AliceSkin initializes correctly."""

    def test_initial_temperature(self):
        skin = AliceSkin()
        stats = skin.get_stats()
        assert abs(stats["skin_temperature"] - T_REF) < 2.0

    def test_initial_no_pain(self):
        skin = AliceSkin()
        stats = skin.get_stats()
        assert stats["pain_level"] == 0.0


class TestTouchTransduction:
    """Touch stimuli produce correct impedance matching."""

    def test_touch_returns_dict(self):
        skin = AliceSkin()
        result = skin.touch(pressure=0.5, object_temperature=25.0, object_impedance=75.0)
        assert isinstance(result, dict)
        assert "gamma_touch" in result

    def test_perfect_match_low_gamma(self):
        """When Z_object ≈ Z_skin, Γ should be near 0."""
        skin = AliceSkin()
        stats = skin.get_stats()
        z_skin = stats.get("z_skin", 75.0)
        result = skin.touch(pressure=0.5, object_impedance=z_skin)
        assert result["gamma_touch"] < 0.1

    def test_mismatch_high_gamma(self):
        """When Z_object ≫ Z_skin, Γ should be high."""
        skin = AliceSkin()
        result = skin.touch(pressure=0.5, object_impedance=5000.0)
        assert result["gamma_touch"] > 0.5


class TestTemperatureSensing:
    """Temperature detection (thermoreception)."""

    def test_temperature_reading(self):
        skin = AliceSkin()
        result = skin.touch(pressure=0.0, object_temperature=40.0, object_impedance=75.0)
        assert "skin_temperature" in result

    def test_extreme_temperature_triggers_pain(self):
        """Very hot should produce pain signal after sustained contact."""
        skin = AliceSkin()
        # Apply repeated extreme temperature to push skin temp above T_PAIN_HOT
        for _ in range(30):
            result = skin.touch(pressure=0.0, object_temperature=80.0, object_impedance=75.0)
        assert result.get("pain_level", 0) > 0


class TestNociception:
    """Pain detection (nociception)."""

    def test_high_pressure_pain(self):
        """High pressure should trigger pain."""
        skin = AliceSkin()
        result = skin.touch(pressure=0.95, object_impedance=75.0)
        assert result.get("pain_level", 0.0) > 0


class TestSkinSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        skin = AliceSkin()
        skin.touch(pressure=0.5, object_temperature=25.0, object_impedance=75.0)
        signal = skin.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "skin"

    def test_signal_modality(self):
        skin = AliceSkin()
        signal = skin.get_signal()
        assert signal.modality == "somatosensory"


class TestSkinTick:
    """Tick advances state."""

    def test_tick_returns_dict(self):
        skin = AliceSkin()
        result = skin.tick()
        assert isinstance(result, dict)


class TestSkinStats:
    """Stats structure."""

    def test_stats_keys(self):
        skin = AliceSkin()
        stats = skin.get_stats()
        assert "total_touches" in stats

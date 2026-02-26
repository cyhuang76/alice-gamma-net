# -*- coding: utf-8 -*-
"""
Tests for AliceNose — Olfactory Organ

Tests verify:
    1. Basic construction
    2. Sniffing clean air
    3. Sniffing an odorant (detection)
    4. Olfactory adaptation (habituation)
    5. Thalamus bypass (always true)
    6. Hedonic response
    7. ElectricalSignal generation
    8. Tick recovery
    9. Stats consistency
"""

import pytest
import numpy as np

from alice.body.nose import (
    AliceNose,
    OdorProfile,
    N_RECEPTOR_TYPES,
    DETECTION_THRESHOLD,
    NOSE_IMPEDANCE,
)
from alice.core.signal import ElectricalSignal


class TestNoseConstruction:
    """AliceNose initializes correctly."""

    def test_receptor_count(self):
        nose = AliceNose()
        assert len(nose._receptor_impedances) == N_RECEPTOR_TYPES

    def test_initial_no_detection(self):
        nose = AliceNose()
        assert nose._detection_strength == 0.0


class TestCleanAir:
    """Sniffing clean air returns no detection."""

    def test_sniff_none(self):
        nose = AliceNose()
        result = nose.sniff(None)
        assert result["detected"] is False
        assert result["detection_strength"] == 0.0


class TestOdorantDetection:
    """Sniffing odorant returns correct detection."""

    def test_sniff_odorant(self):
        nose = AliceNose()
        rose = OdorProfile(name="rose", z_molecular=50.0, hedonic_value=0.8, volatility=0.5)
        result = nose.sniff(rose)
        assert isinstance(result, dict)
        assert "detected" in result
        assert "detection_strength" in result

    def test_matched_odorant_detected(self):
        """Odorant with impedance near a receptor should be detected."""
        nose = AliceNose()
        # Use impedance close to a receptor
        z_close = float(nose._receptor_impedances[0])
        odorant = OdorProfile(name="match", z_molecular=z_close, hedonic_value=0.5, volatility=0.3)
        result = nose.sniff(odorant)
        assert result["detected"] is True
        assert result["detection_strength"] > DETECTION_THRESHOLD


class TestAdaptation:
    """Olfactory adaptation (fastest habituation of any sense)."""

    def test_adaptation_increases(self):
        nose = AliceNose()
        z_close = float(nose._receptor_impedances[5])
        odorant = OdorProfile(name="persistent", z_molecular=z_close, hedonic_value=0.3, volatility=0.1)

        # First sniff
        result1 = nose.sniff(odorant)
        # Repeated sniffs → adaptation
        for _ in range(20):
            nose.sniff(odorant)
        result2 = nose.sniff(odorant)

        # Detection should decrease with adaptation
        assert result2["detection_strength"] <= result1["detection_strength"]


class TestThalamusBypass:
    """Olfaction bypasses thalamus — always."""

    def test_bypass_flag(self):
        nose = AliceNose()
        odorant = OdorProfile(name="test", z_molecular=50.0, hedonic_value=0.0, volatility=0.5)
        result = nose.sniff(odorant)
        assert result["bypasses_thalamus"] is True


class TestHedonicResponse:
    """Pleasant/noxious distinction."""

    def test_pleasant_positive(self):
        nose = AliceNose()
        odorant = OdorProfile(name="flower", z_molecular=50.0, hedonic_value=0.9, volatility=0.5)
        result = nose.sniff(odorant)
        assert result["hedonic_response"] >= 0

    def test_noxious_negative(self):
        nose = AliceNose()
        odorant = OdorProfile(name="rotten", z_molecular=50.0, hedonic_value=-0.9, volatility=0.5)
        result = nose.sniff(odorant)
        assert result["hedonic_response"] <= 0


class TestNoseSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        nose = AliceNose()
        signal = nose.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "nose"
        assert signal.modality == "olfactory"


class TestNoseTick:
    """Tick advances adaptation state."""

    def test_tick_returns_dict(self):
        nose = AliceNose()
        result = nose.tick()
        assert isinstance(result, dict)


class TestNoseStats:
    """Stats structure."""

    def test_stats_keys(self):
        nose = AliceNose()
        nose.sniff(OdorProfile(name="x", z_molecular=50.0, hedonic_value=0.0, volatility=0.5))
        stats = nose.get_stats()
        assert "total_sniffs" in stats
        assert stats["total_sniffs"] == 1

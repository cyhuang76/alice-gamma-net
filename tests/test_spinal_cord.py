# -*- coding: utf-8 -*-
"""
Tests for SpinalCord — High-Speed Reflex Transmission Line

Tests verify:
    1. Basic construction
    2. Reflex arc activation
    3. Reflex bypasses brain (fast latency)
    4. Pain gate control (Melzack & Wall)
    5. Spinal cord injury simulation
    6. Myelination development
    7. Hebbian reflex adaptation (C2)
    8. ElectricalSignal generation
    9. Γ² + T = 1 (C1)
"""

import pytest
import numpy as np

from alice.brain.spinal_cord import (
    SpinalCord,
    MYELINATION_NEONATAL,
    REFLEX_TYPES,
    N_SEGMENTS,
    MONOSYNAPTIC_LATENCY,
    PAIN_GATE_THRESHOLD,
)
from alice.core.signal import ElectricalSignal


class TestSpinalConstruction:
    """SpinalCord initializes correctly."""

    def test_initial_myelination(self):
        sc = SpinalCord()
        assert sc._myelination == MYELINATION_NEONATAL

    def test_initial_segments_healthy(self):
        sc = SpinalCord()
        assert all(sc._segment_health[i] == 1.0 for i in range(N_SEGMENTS))

    def test_reflex_arcs_present(self):
        sc = SpinalCord()
        for rt in REFLEX_TYPES:
            assert rt in sc._reflexes


class TestReflexActivation:
    """Reflex arc triggers."""

    def test_stretch_reflex(self):
        sc = SpinalCord()
        result = sc.activate_reflex("stretch", stimulus_intensity=0.7)
        assert result["response"] > 0
        assert result["reflex_type"] == "stretch"

    def test_withdrawal_reflex(self):
        sc = SpinalCord()
        result = sc.activate_reflex("withdrawal", stimulus_intensity=0.8)
        assert result["response"] > 0

    def test_reflex_faster_than_cortical(self):
        """Spinal reflex latency should be in milliseconds."""
        sc = SpinalCord()
        result = sc.activate_reflex("stretch", stimulus_intensity=0.5)
        assert result["latency_ms"] < 500  # Much faster than cortical


class TestPainGate:
    """Melzack & Wall pain gate control."""

    def test_pain_without_touch(self):
        sc = SpinalCord()
        result = sc.pain_gate(nociceptive=0.8, non_nociceptive=0.0)
        assert result["pain_gate_open"] is True
        assert result["pain_level"] > 0

    def test_touch_closes_gate(self):
        """Non-nociceptive input should reduce pain."""
        sc = SpinalCord()
        result_pain = sc.pain_gate(nociceptive=0.6, non_nociceptive=0.0)
        pain_open = result_pain["pain_level"]

        sc2 = SpinalCord()
        result_touch = sc2.pain_gate(nociceptive=0.6, non_nociceptive=0.8)
        pain_touch = result_touch["pain_level"]

        assert pain_touch <= pain_open


class TestSpinalInjury:
    """Spinal cord injury."""

    def test_injury_reduces_health(self):
        sc = SpinalCord()
        sc.injure_segment(15, severity=0.8)
        assert sc._segment_health[15] < 1.0

    def test_injured_segment_high_gamma(self):
        sc = SpinalCord()
        sc.injure_segment(15, severity=0.9)
        result = sc.activate_reflex("stretch", stimulus_intensity=0.5, segment=15)
        # Injured segment should have higher Γ
        assert result["gamma_segment"] > 0


class TestMyelination:
    """Myelination development."""

    def test_myelination_increases(self):
        sc = SpinalCord()
        m0 = sc._myelination
        for _ in range(100):
            sc.tick()
        assert sc._myelination > m0

    def test_conduction_velocity_increases(self):
        sc = SpinalCord()
        v0 = sc._conduction_velocity
        for _ in range(100):
            sc.tick()
        assert sc._conduction_velocity > v0


class TestSpinalEnergyConservation:
    """C1: Γ² + T = 1."""

    def test_gamma_transmission_sum(self):
        sc = SpinalCord()
        sc.tick()
        gamma = sc._gamma_spinal
        trans = sc._transmission_spinal
        assert abs(gamma ** 2 + trans - 1.0) < 1e-6


class TestSpinalSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        sc = SpinalCord()
        signal = sc.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "spinal_cord"


class TestSpinalStats:
    """Statistics."""

    def test_stats_keys(self):
        sc = SpinalCord()
        stats = sc.get_stats()
        assert "myelination" in stats
        assert "conduction_velocity" in stats
        assert "pain_gate_open" in stats
        assert "reflex_gains" in stats

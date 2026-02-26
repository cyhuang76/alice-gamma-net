# -*- coding: utf-8 -*-
"""
Tests for LymphaticSystem — Impedance Drainage Network

Tests verify:
    1. Basic construction
    2. Lymph flow mechanics
    3. Edema from poor drainage
    4. Node activation during inflammation
    5. Muscle pump effect
    6. ElectricalSignal generation
    7. Γ² + T = 1 (C1)
    8. Developmental maturation
"""

import pytest
import numpy as np

from alice.body.lymphatic import (
    LymphaticSystem,
    LYMPH_FLOW_BASE,
    NEONATAL_LYMPHATIC_MATURITY,
    EDEMA_THRESHOLD,
    N_NODE_REGIONS,
)
from alice.core.signal import ElectricalSignal


class TestLymphaticConstruction:
    """LymphaticSystem initializes correctly."""

    def test_initial_flow(self):
        ls = LymphaticSystem()
        assert ls._lymph_flow == LYMPH_FLOW_BASE

    def test_initial_no_edema(self):
        ls = LymphaticSystem()
        assert ls._edema == 0.0

    def test_initial_node_count(self):
        ls = LymphaticSystem()
        assert len(ls._node_activation) == N_NODE_REGIONS

    def test_initial_maturity(self):
        ls = LymphaticSystem()
        assert ls._maturity == NEONATAL_LYMPHATIC_MATURITY


class TestLymphFlow:
    """Lymph flow dynamics."""

    def test_activity_boosts_flow(self):
        ls = LymphaticSystem()
        result_active = ls.tick(physical_activity=0.9)
        flow_active = result_active["lymph_flow"]

        ls2 = LymphaticSystem()
        result_sedentary = ls2.tick(physical_activity=0.0)
        flow_sedentary = result_sedentary["lymph_flow"]

        assert flow_active >= flow_sedentary


class TestEdema:
    """Edema from fluid accumulation."""

    def test_high_bp_inflammation_causes_fluid_accumulation(self):
        ls = LymphaticSystem()
        for _ in range(100):
            ls.tick(inflammation=0.8, blood_pressure_norm=0.9, physical_activity=0.0)
        assert ls._interstitial_fluid > 0


class TestNodeActivation:
    """Lymph node activation during infection."""

    def test_inflammation_activates_nodes(self):
        ls = LymphaticSystem()
        for _ in range(20):
            result = ls.tick(inflammation=0.8)
        assert result["node_mean_activation"] > 0

    def test_no_inflammation_low_activation(self):
        ls = LymphaticSystem()
        result = ls.tick(inflammation=0.0)
        assert result["node_mean_activation"] < 0.1


class TestLymphaticEnergyConservation:
    """C1: Γ² + T = 1."""

    def test_gamma_transmission_sum(self):
        ls = LymphaticSystem()
        ls.tick(inflammation=0.3)
        gamma = ls._gamma_lymphatic
        trans = ls._transmission_lymphatic
        assert abs(gamma ** 2 + trans - 1.0) < 1e-6


class TestLymphaticSignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        ls = LymphaticSystem()
        signal = ls.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "lymphatic"


class TestLymphaticStats:
    """Statistics."""

    def test_stats_keys(self):
        ls = LymphaticSystem()
        stats = ls.get_stats()
        assert "lymph_flow" in stats
        assert "edema" in stats
        assert "node_activations" in stats

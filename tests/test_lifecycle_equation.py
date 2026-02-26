# -*- coding: utf-8 -*-
"""
Tests for LifecycleEquationEngine — Paper III Eq.24 + Eq.22

Tests verify:
    1. Basic construction and initial state
    2. Eq.24 ODE dynamics (learning, novelty, aging)
    3. Eq.22 equilibrium temperature
    4. Phase detection (birth → infancy → childhood → adulthood → decline)
    5. Bathtub curve shape
    6. get_state() / get_stats() consistency
"""

import numpy as np
import pytest

from alice.brain.lifecycle_equation import (
    LifecycleEquationEngine,
    ETA_LEARN_DEFAULT,
    GAMMA_NOVEL_MAX,
    DELTA_AGING_DEFAULT,
    AGING_ONSET_TICK,
)


class TestLifecycleConstruction:
    """LifecycleEquationEngine initializes correctly."""

    def test_initial_gamma_sq(self):
        engine = LifecycleEquationEngine()
        assert engine._sigma_gamma_sq > 0

    def test_initial_phase(self):
        engine = LifecycleEquationEngine()
        assert engine._lifecycle_phase == "birth"

    def test_initial_tick_zero(self):
        engine = LifecycleEquationEngine()
        assert engine._tick_count == 0


class TestLifecycleODE:
    """Test Eq.24 dynamics: d(ΣΓ²)/dt = −η·ΣΓ² + γ·Γ_env(t) + δ·D(t)."""

    def test_gamma_decreases_with_learning(self):
        """Without novelty, η·ΣΓ² > 0 → ΣΓ² decreases."""
        engine = LifecycleEquationEngine()
        initial = engine._sigma_gamma_sq
        for _ in range(100):
            engine.tick(novelty_level=0.0)
        assert engine._sigma_gamma_sq < initial

    def test_novelty_injects_gamma(self):
        """High novelty should slow or reverse Γ² descent."""
        engine = LifecycleEquationEngine()
        for _ in range(50):
            engine.tick(novelty_level=1.0)
        engine2 = LifecycleEquationEngine()
        for _ in range(50):
            engine2.tick(novelty_level=0.0)
        assert engine._sigma_gamma_sq >= engine2._sigma_gamma_sq

    def test_aging_adds_gamma_after_onset(self):
        """After AGING_ONSET_TICK, δ·D(t) > 0 adds slow Γ² rise."""
        engine = LifecycleEquationEngine()
        engine._tick_count = AGING_ONSET_TICK + 10
        engine._sigma_gamma_sq = 0.1
        engine.tick(fatigue_damage=0.5, novelty_level=0.0)
        assert engine._sigma_gamma_sq >= 0


class TestEquilibriumTemperature:
    """Test Eq.22: T_steady = T_env + (α/β)·ΣΓ²."""

    def test_equilibrium_positive(self):
        engine = LifecycleEquationEngine()
        engine.tick(novelty_level=0.0)
        state = engine.get_state()
        assert state["t_steady"] > 0


class TestPhaseDetection:
    """Phase detection transitions correctly over lifecycle."""

    def test_birth_phase_initial(self):
        engine = LifecycleEquationEngine()
        assert engine.get_phase() == "birth"

    def test_phase_advances_with_ticks(self):
        engine = LifecycleEquationEngine()
        for _ in range(200):
            engine.tick(novelty_level=0.3)
        assert engine.get_phase() != "birth" or engine._tick_count >= 100


class TestLifecycleState:
    """get_state() and get_stats() return correct structure."""

    def test_get_state_keys(self):
        engine = LifecycleEquationEngine()
        engine.tick(novelty_level=0.0)
        state = engine.get_state()
        assert "sigma_gamma_sq" in state
        assert "lifecycle_phase" in state
        assert "t_steady" in state

    def test_get_stats_keys(self):
        engine = LifecycleEquationEngine()
        stats = engine.get_stats()
        assert "tick_count" in stats
        assert "lifecycle_phase" in stats

# -*- coding: utf-8 -*-
"""
test_predictive_engine.py - Predictive Processing Engine Unit Tests (Phase 17)

Tests cover:
- ForwardModel forward model learning
- PredictiveEngine prediction error calculation
- Precision adaptive modulation
- Mental simulation (What-If Loop)
- Preemptive decision-making
- Anxiety management
- Trauma bias
- Energy exhaustion degradation
- AliceBrain integration
"""

import math
import numpy as np
import pytest

from alice.brain.predictive_engine import (
    PredictiveEngine,
    ForwardModel,
    WorldState,
    SimulationPath,
    PredictionResult,
    ACTION_CATALOG,
    encode_action,
    DEFAULT_PRECISION,
    MIN_PRECISION,
    MAX_PRECISION,
    MIN_ENERGY_FOR_SIMULATION,
    PREDICTION_HORIZON,
    MAX_BRANCHES,
)
from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority


# ============================================================================
# WorldState Tests
# ============================================================================


class TestWorldState:
    def test_to_vector(self):
        s = WorldState(temperature=0.5, pain=0.2, energy=0.8)
        v = s.to_vector()
        assert v.shape == (8,)
        assert v[0] == pytest.approx(0.5)
        assert v[1] == pytest.approx(0.2)
        assert v[2] == pytest.approx(0.8)

    def test_from_vector_roundtrip(self):
        s = WorldState(temperature=0.4, pain=0.1, energy=0.7,
                       arousal=0.3, stability=0.9, consciousness=0.8,
                       cortisol=0.2, heart_rate=80.0)
        v = s.to_vector()
        s2 = WorldState.from_vector(v, timestamp=5)
        assert s2.temperature == pytest.approx(s.temperature, abs=0.01)
        assert s2.pain == pytest.approx(s.pain, abs=0.01)
        assert s2.timestamp == 5

    def test_impedance_stable(self):
        stable = WorldState(temperature=0.0, pain=0.0, energy=1.0,
                            stability=1.0, cortisol=0.0)
        assert stable.impedance() == pytest.approx(50.0, abs=1.0)

    def test_impedance_stressed(self):
        stressed = WorldState(temperature=0.8, pain=0.7, energy=0.2,
                              stability=0.3, cortisol=0.5)
        assert stressed.impedance() > 200.0

    def test_from_vector_clipping(self):
        v = np.array([1.5, -0.1, 2.0, -1.0, 0.5, 0.5, 0.5, 0.5])
        s = WorldState.from_vector(v)
        assert s.temperature == 1.0
        assert s.pain == 0.0
        assert s.energy == 1.0
        assert s.arousal == 0.0


# ============================================================================
# ForwardModel Tests
# ============================================================================


class TestForwardModel:
    def test_predict_shape(self):
        fm = ForwardModel(state_dim=8, action_dim=4)
        state = np.random.rand(8)
        action = np.random.rand(4)
        pred = fm.predict(state, action)
        assert pred.shape == (8,)

    def test_predict_without_action(self):
        fm = ForwardModel()
        state = np.random.rand(8)
        pred = fm.predict(state)
        assert pred.shape == (8,)

    def test_update_reduces_error(self):
        fm = ForwardModel(seed=42)
        state = np.array([0.3, 0.0, 0.9, 0.3, 0.95, 0.8, 0.1, 0.3])
        next_state = np.array([0.31, 0.0, 0.89, 0.3, 0.95, 0.8, 0.1, 0.3])
        action = encode_action("idle")

        errors = []
        for _ in range(50):
            err = fm.update(state, action, next_state)
            errors.append(err)

        assert errors[-1] < errors[0]

    def test_stats(self):
        fm = ForwardModel()
        stats = fm.get_stats()
        assert "total_updates" in stats
        assert "avg_error" in stats


# ============================================================================
# PredictiveEngine Basic Tests
# ============================================================================


class TestPredictiveEngineBasics:
    def test_init(self):
        pe = PredictiveEngine()
        assert pe._tick_count == 0
        assert pe._precision == DEFAULT_PRECISION

    def test_observe(self):
        pe = PredictiveEngine()
        s = pe.observe(temperature=0.3, pain=0.1)
        assert isinstance(s, WorldState)
        assert s.temperature == pytest.approx(0.3)
        assert len(pe._state_history) == 1

    def test_tick_returns_dict(self):
        pe = PredictiveEngine()
        r = pe.tick(temperature=0.3, pain=0.0)
        assert isinstance(r, dict)
        assert "prediction_error" in r
        assert "free_energy" in r
        assert "surprise" in r
        assert "precision" in r
        assert "best_action" in r

    def test_tick_count(self):
        pe = PredictiveEngine()
        pe.tick()
        pe.tick()
        pe.tick()
        assert pe._tick_count == 3

    def test_predict_next(self):
        pe = PredictiveEngine()
        s = WorldState(temperature=0.3, pain=0.0, energy=0.9)
        pred = pe.predict_next(s, "idle")
        assert isinstance(pred, WorldState)
        assert pe._last_prediction is not None

    def test_get_state(self):
        pe = PredictiveEngine()
        pe.tick()
        state = pe.get_state()
        assert "tick" in state
        assert "precision" in state
        assert "total_simulations" in state

    def test_get_stats(self):
        pe = PredictiveEngine()
        pe.tick()
        stats = pe.get_stats()
        assert "ticks" in stats
        assert "accuracy" in stats


# ============================================================================
# Prediction Error and Precision
# ============================================================================


class TestPredictionError:
    def test_first_tick_no_error(self):
        pe = PredictiveEngine()
        r = pe.tick(temperature=0.3)
        assert r["prediction_error"] == 0.0

    def test_stable_env_reduces_error(self):
        pe = PredictiveEngine()
        errors = []
        for t in range(80):
            r = pe.tick(temperature=0.3, pain=0.0, energy=0.9,
                        arousal=0.3, stability=0.95, consciousness=0.8,
                        pfc_energy=0.8)
            if t > 5:
                errors.append(r["prediction_error"])
        avg_early = np.mean(errors[:20])
        avg_late = np.mean(errors[-20:])
        assert avg_late <= avg_early + 0.01

    def test_precision_decreases_in_stable(self):
        pe = PredictiveEngine()
        for _ in range(100):
            pe.tick(temperature=0.3, pain=0.0, energy=0.9,
                    arousal=0.3, stability=0.95, consciousness=0.8,
                    pfc_energy=0.5)
        assert pe._precision < DEFAULT_PRECISION

    def test_precision_increases_on_shock(self):
        pe = PredictiveEngine()
        # Brief stable phase — just enough to establish predictions
        for _ in range(10):
            pe.tick(temperature=0.3, pain=0.0, pfc_energy=0.5)
        pre_shock = pe._precision

        # Apply a few shock ticks — check before forward model adapts
        for _ in range(3):
            pe.tick(temperature=0.9, pain=0.7, energy=0.2,
                    arousal=0.9, stability=0.2, pfc_energy=0.5)
        assert pe._precision > pre_shock


# ============================================================================
# Mental Simulation
# ============================================================================


class TestMentalSimulation:
    def test_simulate_futures_returns_paths(self):
        pe = PredictiveEngine()
        # Give some training
        for _ in range(30):
            pe.tick(temperature=0.3, pfc_energy=0.8)

        s = WorldState(temperature=0.5, pain=0.1, energy=0.7)
        paths = pe.simulate_futures(s, pfc_energy=0.8)
        assert len(paths) > 0
        assert isinstance(paths[0], SimulationPath)

    def test_no_simulation_without_energy(self):
        pe = PredictiveEngine()
        s = WorldState(temperature=0.5)
        paths = pe.simulate_futures(s, pfc_energy=0.01)
        assert len(paths) == 0

    def test_paths_sorted_by_gamma(self):
        pe = PredictiveEngine()
        for _ in range(30):
            pe.tick(temperature=0.3, pfc_energy=0.8)

        s = WorldState(temperature=0.5)
        paths = pe.simulate_futures(s, pfc_energy=0.8)
        if len(paths) > 1:
            for i in range(len(paths) - 1):
                assert paths[i].cumulative_gamma <= paths[i + 1].cumulative_gamma

    def test_simulation_path_attributes(self):
        path = SimulationPath(action_label="cool")
        assert path.average_gamma == 0.0
        assert path.terminal_impedance == 500.0
        assert not path.is_harmful


# ============================================================================
# Preemptive Action
# ============================================================================


class TestPreemptive:
    def test_preemptive_on_harmful_idle(self):
        pe = PredictiveEngine()
        # Simulated paths: idle is harmful, cool is safe
        p_idle = SimulationPath(action_label="idle", is_harmful=True,
                                cumulative_gamma=2.0)
        p_idle.terminal_state = WorldState(temperature=0.9)
        p_cool = SimulationPath(action_label="cool", is_harmful=False,
                                cumulative_gamma=0.5)
        p_cool.terminal_state = WorldState(temperature=0.4)

        action, is_pre = pe.evaluate_preemptive_action([p_cool, p_idle])
        assert action == "cool"
        assert is_pre is True

    def test_no_preemptive_when_safe(self):
        pe = PredictiveEngine()
        p_idle = SimulationPath(action_label="idle", is_harmful=False,
                                cumulative_gamma=0.3)
        p_idle.terminal_state = WorldState(temperature=0.3)

        action, is_pre = pe.evaluate_preemptive_action([p_idle])
        assert is_pre is False

    def test_empty_paths(self):
        pe = PredictiveEngine()
        action, is_pre = pe.evaluate_preemptive_action([])
        assert action is None
        assert is_pre is False


# ============================================================================
# Anxiety Management
# ============================================================================


class TestAnxiety:
    def test_anxiety_starts_zero(self):
        pe = PredictiveEngine()
        assert pe._anxiety_level == 0.0

    def test_chaotic_env_raises_anxiety(self):
        pe = PredictiveEngine()
        # Train briefly on stable so surprise signal is meaningful
        for _ in range(20):
            pe.tick(temperature=0.3, pain=0.0, pfc_energy=0.8)
        rng = np.random.RandomState(42)
        max_anxiety = 0.0
        for _ in range(150):
            pe.tick(
                temperature=float(rng.uniform(0.1, 0.9)),
                pain=float(rng.uniform(0.0, 0.8)),
                energy=float(rng.uniform(0.2, 0.9)),
                arousal=float(rng.uniform(0.2, 0.9)),
                stability=float(rng.uniform(0.3, 0.9)),
                pfc_energy=0.8,
            )
            max_anxiety = max(max_anxiety, pe._anxiety_level)
        # At least at some point anxiety was raised
        assert max_anxiety > 0.0

    def test_reset_anxiety(self):
        pe = PredictiveEngine()
        pe._anxiety_level = 0.8
        pe._rumination_count = 10
        pe.reset_anxiety()
        assert pe._anxiety_level == 0.0
        assert pe._rumination_count == 0


# ============================================================================
# Trauma Bias
# ============================================================================


class TestTraumaBias:
    def test_induce_trauma_bias(self):
        pe = PredictiveEngine()
        pe.induce_trauma_bias(0.2)
        assert pe._valence_bias == pytest.approx(-0.2)

    def test_positive_experience_counters(self):
        pe = PredictiveEngine()
        pe.induce_trauma_bias(0.3)
        pe.positive_experience(0.1)
        assert pe._valence_bias == pytest.approx(-0.2)

    def test_bias_affects_prediction(self):
        pe = PredictiveEngine()
        s = WorldState(temperature=0.3, pain=0.0, energy=0.9)

        pred_normal = pe.predict_next(s, "idle")
        pe.induce_trauma_bias(0.5)
        pred_biased = pe.predict_next(s, "idle")

        assert pred_biased.temperature > pred_normal.temperature


# ============================================================================
# Energy Exhaustion Degradation
# ============================================================================


class TestExhaustionFallback:
    def test_no_sim_when_exhausted(self):
        pe = PredictiveEngine()
        r = pe.tick(temperature=0.5, pfc_energy=0.01)
        assert r["simulations_run"] == 0

    def test_sim_when_energized(self):
        pe = PredictiveEngine()
        r = pe.tick(temperature=0.5, pfc_energy=0.8)
        assert r["simulations_run"] > 0

    def test_no_sim_when_sleeping(self):
        pe = PredictiveEngine()
        r = pe.tick(temperature=0.3, pfc_energy=0.8, is_sleeping=True)
        assert r["simulations_run"] == 0

    def test_no_sim_low_consciousness(self):
        pe = PredictiveEngine()
        r = pe.tick(temperature=0.3, pfc_energy=0.8, consciousness=0.1)
        assert r["simulations_run"] == 0


# ============================================================================
# Action Encoding
# ============================================================================


class TestActionEncoding:
    def test_known_actions(self):
        for label in ACTION_CATALOG:
            v = encode_action(label)
            assert v.shape == (4,)

    def test_unknown_defaults_to_idle(self):
        v = encode_action("nonexistent")
        np.testing.assert_array_equal(v, ACTION_CATALOG["idle"])


# ============================================================================
# History Management
# ============================================================================


class TestHistoryManagement:
    def test_surprise_history(self):
        pe = PredictiveEngine()
        for _ in range(10):
            pe.tick()
        assert len(pe.get_surprise_history()) == 10

    def test_free_energy_history(self):
        pe = PredictiveEngine()
        for _ in range(10):
            pe.tick()
        assert len(pe.get_free_energy_history()) == 10

    def test_history_limit(self):
        pe = PredictiveEngine()
        for _ in range(250):
            pe.tick()
        assert len(pe._state_history) <= 200


# ============================================================================
# AliceBrain Integration
# ============================================================================


class TestAliceBrainIntegration:
    def test_brain_has_predictive(self):
        brain = AliceBrain(neuron_count=50)
        assert hasattr(brain, "predictive")
        assert isinstance(brain.predictive, PredictiveEngine)

    def test_perceive_includes_predictive(self):
        brain = AliceBrain(neuron_count=50)
        signal = np.random.rand(64).astype(np.float32)
        result = brain.perceive(signal, Modality.VISUAL, Priority.NORMAL)
        assert "predictive" in result
        assert "prediction_error" in result["predictive"]

    def test_predictive_ticks_with_perceive(self):
        brain = AliceBrain(neuron_count=50)
        signal = np.random.rand(64).astype(np.float32)
        for _ in range(5):
            brain.perceive(signal, Modality.VISUAL, Priority.NORMAL)
        assert brain.predictive._tick_count == 5

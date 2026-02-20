# -*- coding: utf-8 -*-
"""
test_metacognition.py — Phase 18 Metacognition Engine Unit Tests
"""

import math
import numpy as np
import pytest

from alice.brain.metacognition import (
    MetacognitionEngine,
    MetacognitionResult,
    CognitiveSnapshot,
    CounterfactualResult,
    SYSTEM2_ENGAGE_THRESHOLD,
    SYSTEM2_DISENGAGE_THRESHOLD,
    CORRECTION_THRESHOLD,
    DOUBT_THRESHOLD,
    INSIGHT_THRESHOLD,
    BASE_THINKING_RATE,
    MIN_THINKING_RATE,
    MAX_THINKING_RATE,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
    RUMINATION_THRESHOLD,
    EMA_ALPHA,
)


# ============================================================================
# MetacognitionEngine Basics
# ============================================================================

class TestMetacognitionBasics:

    def test_init_defaults(self):
        me = MetacognitionEngine()
        assert me._gamma_thinking == 0.0
        assert me._system_mode == 1
        assert me._confidence == 0.5
        assert me._thinking_rate == BASE_THINKING_RATE
        assert me._total_ticks == 0

    def test_tick_returns_dict(self):
        me = MetacognitionEngine()
        result = me.tick()
        assert isinstance(result, dict)
        assert "gamma_thinking" in result
        assert "thinking_rate" in result
        assert "confidence" in result
        assert "system_mode" in result
        assert "meta_report" in result

    def test_tick_increments_counter(self):
        me = MetacognitionEngine()
        me.tick()
        me.tick()
        me.tick()
        assert me._total_ticks == 3


# ============================================================================
# Thinking Impedance
# ============================================================================

class TestThinkingImpedance:

    def test_low_load_low_impedance(self):
        me = MetacognitionEngine()
        for _ in range(10):
            r = me.tick(prediction_error=0.01, anxiety=0.01, pfc_energy=0.95)
        assert r["gamma_thinking"] < 0.15

    def test_high_load_high_impedance(self):
        me = MetacognitionEngine()
        for _ in range(15):
            r = me.tick(
                prediction_error=0.8, free_energy=5.0, anxiety=0.9,
                pfc_energy=0.15, binding_gamma=0.7,
            )
        assert r["gamma_thinking"] > 0.3

    def test_pain_increases_impedance(self):
        me1 = MetacognitionEngine()
        me2 = MetacognitionEngine()
        for _ in range(10):
            r_no_pain = me1.tick(prediction_error=0.3, pain=0.0)
            r_pain = me2.tick(prediction_error=0.3, pain=0.8)
        assert r_pain["gamma_thinking"] > r_no_pain["gamma_thinking"]

    def test_impedance_bounded_0_1(self):
        me = MetacognitionEngine()
        r = me.tick(
            prediction_error=1.0, free_energy=100.0, anxiety=1.0,
            pfc_energy=0.0, binding_gamma=1.0, pain=1.0,
        )
        assert 0.0 <= r["gamma_thinking"] <= 1.0


# ============================================================================
# System 1/2 Switching
# ============================================================================

class TestSystemSwitching:

    def test_starts_system1(self):
        me = MetacognitionEngine()
        r = me.tick()
        assert r["system_mode"] == 1

    def test_high_impedance_switches_to_system2(self):
        me = MetacognitionEngine()
        for _ in range(20):
            r = me.tick(
                prediction_error=0.9, free_energy=8.0, anxiety=0.9,
                pfc_energy=0.1, binding_gamma=0.8,
            )
        assert r["system_mode"] == 2

    def test_hysteresis_prevents_oscillation(self):
        me = MetacognitionEngine()
        # Enter System 2
        for _ in range(20):
            me.tick(
                prediction_error=0.9, anxiety=0.9, pfc_energy=0.1,
                free_energy=8.0, binding_gamma=0.8,
            )
        assert me._system_mode == 2

        # Partially lower — should stay System 2 (hysteresis)
        for _ in range(3):
            r = me.tick(prediction_error=0.3, anxiety=0.3, pfc_energy=0.6)
        assert r["system_mode"] == 2  # Still in System 2

    def test_full_recovery_returns_system1(self):
        me = MetacognitionEngine()
        for _ in range(20):
            me.tick(
                prediction_error=0.9, anxiety=0.9, pfc_energy=0.1,
                free_energy=8.0, binding_gamma=0.8,
            )
        for _ in range(40):
            r = me.tick(prediction_error=0.01, anxiety=0.01, pfc_energy=0.95)
        assert r["system_mode"] == 1


# ============================================================================
# Thinking Rate
# ============================================================================

class TestThinkingRate:

    def test_low_impedance_high_rate(self):
        me = MetacognitionEngine()
        for _ in range(15):
            r = me.tick(prediction_error=0.01, anxiety=0.01, pfc_energy=0.95)
        assert r["thinking_rate"] > 0.8

    def test_high_impedance_low_rate(self):
        me = MetacognitionEngine()
        for _ in range(20):
            r = me.tick(
                prediction_error=0.9, free_energy=8.0, anxiety=0.9,
                pfc_energy=0.1, binding_gamma=0.8,
            )
        assert r["thinking_rate"] < 0.5

    def test_rate_bounded(self):
        me = MetacognitionEngine()
        r = me.tick(
            prediction_error=1.0, anxiety=1.0, pfc_energy=0.0,
            free_energy=100.0, binding_gamma=1.0,
        )
        assert r["thinking_rate"] >= MIN_THINKING_RATE
        assert r["thinking_rate"] <= MAX_THINKING_RATE


# ============================================================================
# Confidence Estimation
# ============================================================================

class TestConfidence:

    def test_high_precision_high_confidence(self):
        me = MetacognitionEngine()
        for _ in range(20):
            r = me.tick(prediction_error=0.01, precision=0.05, pfc_energy=0.95)
        assert r["confidence"] > 0.6

    def test_low_precision_low_confidence(self):
        me = MetacognitionEngine()
        for _ in range(60):
            r = me.tick(
                prediction_error=0.9, precision=1.5, anxiety=0.9,
                pfc_energy=0.1, free_energy=10.0, binding_gamma=0.8,
            )
        assert r["confidence"] < DOUBT_THRESHOLD
        assert r["is_doubting"]

    def test_confidence_bounded(self):
        me = MetacognitionEngine()
        for _ in range(100):
            r = me.tick(precision=0.01, prediction_error=0.0)
        assert r["confidence"] <= MAX_CONFIDENCE
        assert r["confidence"] >= MIN_CONFIDENCE


# ============================================================================
# Counterfactual Reasoning
# ============================================================================

class TestCounterfactual:

    def test_regret_on_bad_choice(self):
        me = MetacognitionEngine()
        me.update_action_value("flee", 0.9)
        me.update_action_value("idle", 0.1)

        for _ in range(15):
            r = me.tick(last_action="idle")
        assert r["regret"] > 0.0

    def test_relief_on_good_choice(self):
        me = MetacognitionEngine()
        me.update_action_value("flee", 0.9)
        me.update_action_value("idle", 0.1)

        for _ in range(15):
            r = me.tick(last_action="flee")
        assert r["relief"] > 0.0

    def test_no_action_decays(self):
        me = MetacognitionEngine()
        me.update_action_value("flee", 0.9)
        me.update_action_value("idle", 0.1)

        # Build up regret
        for _ in range(10):
            me.tick(last_action="idle")
        regret_peak = me._regret

        # No action → decay
        for _ in range(20):
            me.tick(last_action=None)
        assert me._regret < regret_peak


# ============================================================================
# Self-Correction
# ============================================================================

class TestSelfCorrection:

    def test_correction_triggered(self):
        me = MetacognitionEngine()
        corrected = False
        for _ in range(30):
            r = me.tick(
                prediction_error=0.9, free_energy=8.0, anxiety=0.9,
                pfc_energy=0.1, binding_gamma=0.8,
            )
            if r["is_correcting"]:
                corrected = True
                break
        assert corrected

    def test_cooldown_limits_corrections(self):
        me = MetacognitionEngine()
        corrections = 0
        for _ in range(20):
            r = me.tick(
                prediction_error=0.9, free_energy=8.0, anxiety=0.9,
                pfc_energy=0.1, binding_gamma=0.8,
            )
            if r["is_correcting"]:
                corrections += 1
        # With 5-tick cooldown, max corrections in 20 ticks ≈ 3-4
        assert 1 <= corrections <= 6


# ============================================================================
# Insight Detection
# ============================================================================

class TestInsight:

    def test_insight_on_impedance_drop(self):
        me = MetacognitionEngine()
        # Build high impedance
        for _ in range(20):
            me.tick(
                prediction_error=0.8, free_energy=5.0, anxiety=0.7,
                pfc_energy=0.2, binding_gamma=0.6,
            )

        # Sudden clarity
        insight_seen = False
        for _ in range(10):
            r = me.tick(
                prediction_error=0.01, free_energy=0.05, anxiety=0.02,
                pfc_energy=0.95, binding_gamma=0.05,
            )
            if r["is_insight"]:
                insight_seen = True
                break
        assert insight_seen

    def test_no_insight_on_gradual_change(self):
        me = MetacognitionEngine()
        # Gradually increase and decrease — no sudden drop
        insight_count = 0
        for i in range(60):
            level = 0.3 + 0.2 * math.sin(i * 0.1)
            r = me.tick(prediction_error=level, anxiety=level * 0.5)
            if r["is_insight"]:
                insight_count += 1
        # Gradual change should rarely trigger insight
        assert insight_count <= 2


# ============================================================================
# Rumination
# ============================================================================

class TestRumination:

    def test_rumination_on_persistent_regret(self):
        me = MetacognitionEngine()
        me._action_values["flee"] = 0.95
        me._action_values["idle"] = 0.05

        ruminating = False
        for _ in range(50):
            r = me.tick(last_action="idle")
            if r["is_ruminating"]:
                ruminating = True
                break
        assert ruminating

    def test_rumination_count_increases(self):
        me = MetacognitionEngine()
        me._action_values["flee"] = 0.95
        me._action_values["idle"] = 0.05

        for _ in range(50):
            r = me.tick(last_action="idle")
        assert r["rumination_count"] > 0


# ============================================================================
# Sleep
# ============================================================================

class TestSleep:

    def test_sleep_min_rate(self):
        me = MetacognitionEngine()
        r = me.tick(is_sleeping=True)
        assert r["thinking_rate"] == MIN_THINKING_RATE

    def test_sleep_system1(self):
        me = MetacognitionEngine()
        r = me.tick(is_sleeping=True)
        assert r["system_mode"] == 1


# ============================================================================
# Action Value Updates
# ============================================================================

class TestActionValues:

    def test_update_existing(self):
        me = MetacognitionEngine()
        me.update_action_value("idle", 0.5, lr=0.5)
        assert me._action_values["idle"] == pytest.approx(0.25, abs=0.01)

    def test_update_new_action(self):
        me = MetacognitionEngine()
        me.update_action_value("dance", 0.8)
        assert "dance" in me._action_values
        assert me._action_values["dance"] == 0.8

    def test_inject_regret(self):
        me = MetacognitionEngine()
        me.inject_regret(0.5)
        assert me._regret == 0.5

    def test_inject_relief(self):
        me = MetacognitionEngine()
        me.inject_relief(0.5)
        assert me._relief == 0.5


# ============================================================================
# History and State
# ============================================================================

class TestHistory:

    def test_history_grows(self):
        me = MetacognitionEngine()
        for _ in range(10):
            me.tick()
        assert len(me._gamma_history) == 10
        assert len(me._confidence_history) == 10

    def test_get_state(self):
        me = MetacognitionEngine()
        me.tick()
        state = me.get_state()
        assert "gamma_thinking" in state
        assert "system2_ratio" in state
        assert "action_values" in state

    def test_properties(self):
        me = MetacognitionEngine()
        me.tick(prediction_error=0.5, anxiety=0.5)
        assert isinstance(me.gamma_thinking, float)
        assert isinstance(me.confidence, float)
        assert isinstance(me.thinking_rate, float)
        assert me.system_mode in (1, 2)


# ============================================================================
# AliceBrain Integration
# ============================================================================

class TestAliceBrainIntegration:

    def test_brain_has_metacognition(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        assert hasattr(brain, "metacognition")
        assert isinstance(brain.metacognition, MetacognitionEngine)

    def test_perceive_returns_metacognition(self):
        from alice.alice_brain import AliceBrain
        from alice.core.protocol import Modality, Priority
        brain = AliceBrain()
        t = np.linspace(0, 0.1, 64)
        signal = (0.5 * np.sin(2 * np.pi * 40.0 * t)).astype(np.float32)
        result = brain.perceive(signal, Modality.AUDITORY, Priority.NORMAL)
        assert "metacognition" in result
        meta = result["metacognition"]
        assert "gamma_thinking" in meta
        assert "system_mode" in meta

    def test_introspect_has_metacognition(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        intro = brain.introspect()
        assert "metacognition" in intro["subsystems"]

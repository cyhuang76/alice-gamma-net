# -*- coding: utf-8 -*-
"""
Tests for CognitiveFlexibilityEngine — Cognitive Flexibility Engine
Phase 8: Physics Model for High-Intensity Task Switching

Coverage:
  1. Initialization and defaults
  2. Task switch cost calculation
  3. Task-set inertia (charge/discharge)
  4. Mixing cost
  5. Perseveration error
  6. Training plasticity (reconfig acceleration, flexibility gain)
  7. Familiar task-pair discount
  8. Decay and use-it-or-lose-it
  9. Esports player simulation
  10. Integration test (AliceBrain see/hear rapid alternation)
"""

import pytest
import numpy as np
import time

from alice.brain.cognitive_flexibility import (
    CognitiveFlexibilityEngine,
    RECONFIG_TAU_INITIAL,
    RECONFIG_TAU_MIN,
    RECONFIG_LEARNING_RATE,
    INERTIA_CHARGE_RATE,
    INERTIA_MAX_CHARGE,
    INERTIA_WEIGHT,
    MIXING_RT_PENALTY,
    MAX_ACTIVE_TASKSETS,
    MAX_ACTIVE_TASKSETS_TRAINED,
    FLEXIBILITY_INITIAL,
    FLEXIBILITY_MAX,
    FAMILIAR_SWITCH_DISCOUNT,
    PERSEVERATION_ERROR_THRESHOLD,
    PERSEVERATION_ERROR_ENERGY_THRESHOLD,
)


# ============================================================================
# 1. Initialization
# ============================================================================

class TestInitialization:
    def test_engine_starts_empty(self):
        engine = CognitiveFlexibilityEngine()
        assert engine.get_current_task() is None
        assert engine.get_flexibility_index() == FLEXIBILITY_INITIAL
        assert engine.get_reconfig_tau() == RECONFIG_TAU_INITIAL

    def test_initial_switch_success_rate(self):
        engine = CognitiveFlexibilityEngine()
        assert engine.get_switch_success_rate() == 1.0  # No switches → 100%

    def test_training_level_starts_novice(self):
        engine = CognitiveFlexibilityEngine()
        assert engine.training_level == "novice"


# ============================================================================
# 2. Task Switch Cost
# ============================================================================

class TestSwitchCost:
    def test_same_task_zero_cost(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        result = engine.attempt_switch("visual")
        assert result.switch_cost_ms == 0.0
        assert result.energy_spent == 0.0

    def test_different_task_has_cost(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        result = engine.attempt_switch("auditory")
        assert result.switch_cost_ms > 0
        assert result.reconfig_delay_ms > 0
        assert not result.perseveration_error

    def test_reconfig_delay_matches_tau(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        result = engine.attempt_switch("auditory")
        # Reconfig delay ≈ τ_reconfig × 1000 (ms)
        assert result.reconfig_delay_ms == pytest.approx(
            RECONFIG_TAU_INITIAL * 1000, abs=1.0
        )

    def test_switch_records_from_and_to(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        result = engine.attempt_switch("auditory")
        assert result.from_task == "visual"
        assert result.to_task == "auditory"

    def test_switch_updates_current_task(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        engine.attempt_switch("auditory")
        assert engine.get_current_task() == "auditory"


# ============================================================================
# 3. Task-Set Inertia
# ============================================================================

class TestInertia:
    def test_task_charges_over_ticks(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        for _ in range(50):
            engine.tick()
        inertia = engine.get_inertia("visual")
        assert inertia > 0
        assert inertia == pytest.approx(50 * INERTIA_CHARGE_RATE, abs=0.01)

    def test_inertia_caps_at_max(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        for _ in range(500):
            engine.tick()
        assert engine.get_inertia("visual") == INERTIA_MAX_CHARGE

    def test_inactive_task_discharges(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        for _ in range(50):
            engine.tick()
        charge_before = engine.get_inertia("visual")

        engine.attempt_switch("auditory")
        for _ in range(50):
            engine.tick()
        charge_after = engine.get_inertia("visual")
        assert charge_after < charge_before

    def test_inertia_increases_switch_cost(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        # Immediate switch → low inertia
        result1 = engine.attempt_switch("auditory")

        # After long execution → high inertia
        engine2 = CognitiveFlexibilityEngine()
        engine2.notify_task("visual")
        for _ in range(80):
            engine2.tick()
        result2 = engine2.attempt_switch("auditory")

        assert result2.inertia_penalty_ms > result1.inertia_penalty_ms


# ============================================================================
# 4. Mixing Cost
# ============================================================================

class TestMixingCost:
    def test_single_task_no_mixing(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        overhead = engine.get_switch_overhead()
        assert overhead["active_tasksets"] == 1

    def test_multi_task_has_mixing_cost(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        engine.attempt_switch("auditory")
        # visual still has residual activation
        overhead = engine.get_switch_overhead()
        assert overhead["active_tasksets"] >= 2
        assert overhead["mixing_cost_ms"] > 0


# ============================================================================
# 5. Perseveration Error
# ============================================================================

class TestPerseverationError:
    def test_high_inertia_low_energy_causes_error(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        # Fill inertia
        for _ in range(200):
            engine.tick()
        assert engine.get_inertia("visual") >= PERSEVERATION_ERROR_THRESHOLD

        # Simulate low PFC energy
        engine.sync_pfc_energy(0.1)
        result = engine.attempt_switch("auditory")
        assert result.perseveration_error

    def test_perseveration_stays_on_old_task(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        for _ in range(200):
            engine.tick()
        engine.sync_pfc_energy(0.1)
        engine.attempt_switch("auditory")
        # Perseveration error → stay on old task
        assert engine.get_current_task() == "visual"

    def test_forced_switch_ignores_perseveration(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        for _ in range(200):
            engine.tick()
        engine.sync_pfc_energy(0.1)
        result = engine.attempt_switch("auditory", forced=True)
        assert not result.perseveration_error
        assert engine.get_current_task() == "auditory"

    def test_normal_conditions_no_perseveration(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        # Normal energy + low inertia
        engine.sync_pfc_energy(0.8)
        result = engine.attempt_switch("auditory")
        assert not result.perseveration_error


# ============================================================================
# 6. Training Plasticity
# ============================================================================

class TestTrainingPlasticity:
    def test_reconfig_tau_decreases_with_training(self):
        engine = CognitiveFlexibilityEngine()
        initial_tau = engine.get_reconfig_tau()
        # Do 100 visual↔auditory switches
        for i in range(100):
            task = "visual" if i % 2 == 0 else "auditory"
            engine.attempt_switch(task)
        trained_tau = engine.get_reconfig_tau()
        assert trained_tau < initial_tau

    def test_flexibility_index_increases_with_training(self):
        engine = CognitiveFlexibilityEngine()
        initial_flex = engine.get_flexibility_index()
        for i in range(200):
            task = "visual" if i % 2 == 0 else "auditory"
            engine.attempt_switch(task)
        trained_flex = engine.get_flexibility_index()
        assert trained_flex > initial_flex

    def test_reconfig_tau_has_floor(self):
        engine = CognitiveFlexibilityEngine()
        for i in range(10000):
            task = "visual" if i % 2 == 0 else "auditory"
            engine.attempt_switch(task)
        assert engine.get_reconfig_tau() >= RECONFIG_TAU_MIN

    def test_flexibility_has_ceiling(self):
        engine = CognitiveFlexibilityEngine()
        for i in range(10000):
            task = "visual" if i % 2 == 0 else "auditory"
            engine.attempt_switch(task)
        assert engine.get_flexibility_index() <= FLEXIBILITY_MAX

    def test_taskset_slots_expand_with_training(self):
        engine = CognitiveFlexibilityEngine()
        initial_slots = engine.get_max_active_tasksets()
        # Do 600 successful switches → should unlock at least 1 new slot
        for i in range(600):
            task = f"task_{i % 3}"
            engine.attempt_switch(task)
        assert engine.get_max_active_tasksets() > initial_slots

    def test_training_level_progresses(self):
        engine = CognitiveFlexibilityEngine()
        assert engine.training_level == "novice"
        for i in range(60):
            engine.attempt_switch(f"task_{i % 2}")
        assert engine.training_level == "intermediate"


# ============================================================================
# 7. Familiar Task-Pair Discount
# ============================================================================

class TestFamiliarSwitchDiscount:
    def test_familiar_pair_reduces_reconfig(self):
        engine = CognitiveFlexibilityEngine()
        # Do 15 A→B switches (exceeding 10 threshold)
        for i in range(30):
            task = "visual" if i % 2 == 0 else "auditory"
            engine.attempt_switch(task)

        # Now visual→auditory has >10 times
        engine.attempt_switch("visual")
        result = engine.attempt_switch("auditory")

        # Reconfig delay should be less than full discount of initial value
        full_reconfig_ms = engine.get_reconfig_tau() * 1000
        assert result.reconfig_delay_ms < full_reconfig_ms * 0.9

    def test_unfamiliar_pair_no_discount(self):
        engine = CognitiveFlexibilityEngine()
        engine.notify_task("visual")
        # First switch to a completely new task → no discount
        result = engine.attempt_switch("tactile")
        assert result.reconfig_delay_ms == pytest.approx(
            RECONFIG_TAU_INITIAL * 1000, abs=5.0
        )


# ============================================================================
# 8. Decay
# ============================================================================

class TestDecay:
    def test_flexibility_decays_without_practice(self):
        engine = CognitiveFlexibilityEngine()
        for i in range(200):
            engine.attempt_switch(f"task_{i % 2}")
        trained_flex = engine.get_flexibility_index()

        # 5000 ticks without practice
        for _ in range(5000):
            engine.tick()
        decayed_flex = engine.get_flexibility_index()
        assert decayed_flex < trained_flex

    def test_reconfig_tau_decays_without_practice(self):
        engine = CognitiveFlexibilityEngine()
        for i in range(200):
            engine.attempt_switch(f"task_{i % 2}")
        trained_tau = engine.get_reconfig_tau()

        for _ in range(5000):
            engine.tick()
        decayed_tau = engine.get_reconfig_tau()
        assert decayed_tau > trained_tau

    def test_decay_slower_than_learning(self):
        """Use-it-or-lose-it asymmetry: learning is fast, forgetting is slow"""
        engine = CognitiveFlexibilityEngine()
        for i in range(200):
            engine.attempt_switch(f"task_{i % 2}")
        trained_flex = engine.get_flexibility_index()
        total_gain = trained_flex - FLEXIBILITY_INITIAL

        for _ in range(5000):
            engine.tick()
        decayed_flex = engine.get_flexibility_index()
        decay_loss = trained_flex - decayed_flex

        # Decay amount < 50% of total gain
        assert decay_loss < total_gain * 0.5


# ============================================================================
# 9. Esports Player Simulation
# ============================================================================

class TestEsportsSimulation:
    def test_reaction_time_improvement(self):
        """After 5000 switch training, reconfig time drops >50%"""
        engine = CognitiveFlexibilityEngine()
        initial_tau = engine.get_reconfig_tau()

        for i in range(5000):
            task = ["visual", "auditory", "motor"][i % 3]
            engine.attempt_switch(task)

        improvement = 1.0 - (engine.get_reconfig_tau() / initial_tau)
        assert improvement > 0.50

    def test_flexibility_becomes_expert(self):
        """Extensive training → flexibility index approaches ceiling"""
        engine = CognitiveFlexibilityEngine()
        for i in range(5000):
            engine.attempt_switch(f"task_{i % 3}")
        assert engine.get_flexibility_index() > 0.7

    def test_rapid_alternation_cost_decreases(self):
        """Average cost of rapid alternation should decrease with training"""
        engine = CognitiveFlexibilityEngine()

        # Average cost of the first 10 switches
        early_costs = []
        for i in range(10):
            task = "visual" if i % 2 == 0 else "auditory"
            result = engine.attempt_switch(task)
            if result.switch_cost_ms > 0:
                early_costs.append(result.switch_cost_ms)

        # Train 2000 times
        for i in range(2000):
            engine.attempt_switch(f"task_{i % 3}")

        # Average cost of the last 10 switches
        late_costs = []
        for i in range(10):
            task = "visual" if i % 2 == 0 else "auditory"
            result = engine.attempt_switch(task)
            if result.switch_cost_ms > 0:
                late_costs.append(result.switch_cost_ms)

        avg_early = np.mean(early_costs) if early_costs else 999
        avg_late = np.mean(late_costs) if late_costs else 999
        assert avg_late < avg_early

    def test_use_it_or_lose_it_asymmetry(self):
        """Skills learned still retain >50% after 5000 ticks"""
        engine = CognitiveFlexibilityEngine()
        for i in range(1000):
            engine.attempt_switch(f"task_{i % 2}")

        trained_flex = engine.get_flexibility_index()
        gain = trained_flex - FLEXIBILITY_INITIAL

        for _ in range(5000):
            engine.tick()
        remaining = engine.get_flexibility_index() - FLEXIBILITY_INITIAL
        assert remaining > gain * 0.50


# ============================================================================
# 10. Integration Tests
# ============================================================================

class TestIntegration:
    def test_brain_has_cognitive_flexibility(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        assert hasattr(brain, "cognitive_flexibility")
        assert isinstance(brain.cognitive_flexibility, CognitiveFlexibilityEngine)

    def test_see_sets_visual_task(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        pixels = np.random.rand(64, 64).astype(np.float32)
        brain.see(pixels)
        assert brain.cognitive_flexibility.get_current_task() == "visual"

    def test_hear_sets_auditory_task(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        sound = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600))
        brain.hear(sound.astype(np.float32))
        assert brain.cognitive_flexibility.get_current_task() == "auditory"

    def test_see_hear_alternation_creates_switches(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        pixels = np.random.rand(64, 64).astype(np.float32)
        sound = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600)).astype(np.float32)

        brain.see(pixels)
        brain.hear(sound)
        brain.see(pixels)

        state = brain.cognitive_flexibility.get_state()
        assert state["total_switches"] >= 2

    def test_cognitive_flexibility_in_brain_state(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        state = brain.introspect()
        assert "cognitive_flexibility" in state["subsystems"]

    def test_rapid_alternation_has_overhead(self):
        """Rapid see/hear alternation → mixed environment should have active tasksets > 1"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        pixels = np.random.rand(64, 64).astype(np.float32)
        sound = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600)).astype(np.float32)

        brain.see(pixels)
        brain.hear(sound)

        overhead = brain.cognitive_flexibility.get_switch_overhead()
        assert overhead["active_tasksets"] >= 2

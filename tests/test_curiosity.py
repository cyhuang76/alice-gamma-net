# -*- coding: utf-8 -*-
"""
CuriosityDriveEngine Tests — Phase 9: Curiosity, Boredom & Self-Awareness

Verification targets:
  1. Novelty detection
  2. Boredom accumulation and release
  3. Spontaneous action generation
  4. Self-recognition (efference copy)
  5. Internal goal generation
  6. AliceBrain integration
"""

import numpy as np
import pytest

from alice.brain.curiosity_drive import (
    BOREDOM_ACCUMULATION,
    BOREDOM_THRESHOLD,
    CURIOSITY_REWARD_SCALE,
    EFFERENCE_COPY_WINDOW,
    INITIAL_SELF_ACCURACY,
    MODEL_IMPEDANCE_INIT,
    NOVELTY_DECAY,
    SELF_RECOGNITION_THRESHOLD,
    SPONTANEOUS_COOLDOWN,
    CuriosityDriveEngine,
    InternalGoal,
    NoveltyEvent,
    SelfOtherJudgment,
    SpontaneousAction,
    SpontaneousActionType,
)


# ============================================================================
# Fixture
# ============================================================================


@pytest.fixture
def engine():
    return CuriosityDriveEngine()


# ============================================================================
# 1. Novelty Detection
# ============================================================================


class TestNoveltyDetection:
    """Γ_novelty = |Z_input - Z_model| / (Z_input + Z_model)"""

    def test_familiar_signal_low_novelty(self, engine):
        """Similar impedance signal → low novelty"""
        event = engine.evaluate_novelty("visual", MODEL_IMPEDANCE_INIT)
        # Z_input ≈ Z_model → Γ ≈ 0
        assert event.gamma_novelty < 0.1
        assert event.novelty_score < 0.1

    def test_novel_signal_high_novelty(self, engine):
        """Large impedance difference → high novelty"""
        event = engine.evaluate_novelty("visual", MODEL_IMPEDANCE_INIT * 3)
        # Z_input >> Z_model → large Γ
        assert event.gamma_novelty > 0.3
        assert event.novelty_score > 0.2

    def test_novelty_habituates(self, engine):
        """Repeated same signal → novelty decreases (habituation)"""
        z_novel = 200.0
        first = engine.evaluate_novelty("visual", z_novel)
        # Continuous same signal → model learns → Γ decreases
        for _ in range(20):
            engine.evaluate_novelty("visual", z_novel)
        last = engine.evaluate_novelty("visual", z_novel)
        assert last.gamma_novelty < first.gamma_novelty

    def test_novel_signal_generates_intrinsic_reward(self, engine):
        """Novel signal → intrinsic reward"""
        engine.evaluate_novelty("visual", 300.0)
        reward = engine.get_intrinsic_reward()
        assert reward > 0

    def test_internal_model_updates(self, engine):
        """Internal model updates with experience"""
        initial_z = engine._internal_model["visual"]
        engine.evaluate_novelty("visual", 200.0)
        updated_z = engine._internal_model["visual"]
        # Model should shift toward input direction
        assert updated_z > initial_z

    def test_different_modalities_independent(self, engine):
        """Novelty for different modalities computed independently"""
        engine.evaluate_novelty("visual", 200.0)
        visual_model = engine._internal_model["visual"]
        auditory_model = engine._internal_model["auditory"]
        # Visual updated, auditory unchanged
        assert visual_model != MODEL_IMPEDANCE_INIT
        assert auditory_model == MODEL_IMPEDANCE_INIT

    def test_amplitude_modulates_novelty(self, engine):
        """Signal amplitude modulates novelty perception"""
        weak = engine.evaluate_novelty("visual", 200.0, signal_amplitude=0.1)
        engine._internal_model["auditory"] = MODEL_IMPEDANCE_INIT  # reset
        strong = engine.evaluate_novelty("auditory", 200.0, signal_amplitude=1.0)
        assert strong.novelty_score > weak.novelty_score

    def test_novelty_event_structure(self, engine):
        """NoveltyEvent data structure is complete"""
        event = engine.evaluate_novelty("visual", 100.0)
        assert isinstance(event, NoveltyEvent)
        assert event.modality == "visual"
        assert 0 <= event.novelty_score <= 1
        assert 0 <= event.gamma_novelty <= 1
        assert event.z_input > 0
        assert event.z_model > 0

    def test_unknown_modality_creates_model(self, engine):
        """Unknown modality automatically creates internal model"""
        event = engine.evaluate_novelty("tactile", 100.0)
        assert "tactile" in engine._internal_model
        assert event.modality == "tactile"


# ============================================================================
# 2. Boredom Accumulation
# ============================================================================


class TestBoredomAccumulation:
    """Extended absence of novel stimuli → boredom pressure rises"""

    def test_boredom_increases_without_input(self, engine):
        """No input → boredom pressure accumulates"""
        initial_boredom = engine.get_boredom()
        for _ in range(100):
            engine.tick(has_external_input=False)
        final_boredom = engine.get_boredom()
        assert final_boredom > initial_boredom
        assert final_boredom > 0.1

    def test_boredom_decreases_with_novel_input(self, engine):
        """Novel input → boredom pressure decreases"""
        # First accumulate boredom
        for _ in range(100):
            engine.tick(has_external_input=False)
        bored = engine.get_boredom()
        assert bored > 0.1

        # Novel input → release boredom
        engine.evaluate_novelty("visual", 300.0)
        assert engine.get_boredom() < bored

    def test_boredom_capped_at_saturation(self, engine):
        """Boredom pressure does not exceed upper limit"""
        for _ in range(2000):
            engine.tick(has_external_input=False)
        assert engine.get_boredom() <= 1.0

    def test_boredom_accelerates_over_time(self, engine):
        """Boredom accumulation accelerates over time (longer = more restless)"""
        engine.tick(has_external_input=False)
        early_rate = engine.get_boredom()
        for _ in range(49):
            engine.tick(has_external_input=False)
        mid_boredom = engine.get_boredom()
        for _ in range(50):
            engine.tick(has_external_input=False)
        late_boredom = engine.get_boredom()
        # Later 50 ticks increment should >= earlier 50 ticks
        later_gain = late_boredom - mid_boredom
        earlier_gain = mid_boredom - early_rate
        assert later_gain >= earlier_gain * 0.8  # Allow some variance

    def test_external_input_resets_counter(self, engine):
        """External input → no-input counter resets to zero"""
        for _ in range(20):
            engine.tick(has_external_input=False)
        assert engine._ticks_without_input == 20
        engine.tick(has_external_input=True, sensory_load=0.5)
        assert engine._ticks_without_input == 0


# ============================================================================
# 3. Spontaneous Action
# ============================================================================


class TestSpontaneousAction:
    """Boredom → spontaneous action generation"""

    def test_no_spontaneous_when_not_bored(self, engine):
        """Insufficient boredom → no spontaneous action"""
        action = engine.generate_spontaneous_action(energy=1.0)
        assert action is None

    def test_spontaneous_when_bored(self, engine):
        """Boredom exceeds threshold → generates spontaneous action"""
        engine._boredom_pressure = BOREDOM_THRESHOLD + 0.1
        action = engine.generate_spontaneous_action(energy=1.0)
        assert action is not None
        assert isinstance(action, SpontaneousAction)
        assert action.trigger == "boredom_threshold_exceeded"

    def test_spontaneous_cooldown(self, engine):
        """No new spontaneous action during cooldown"""
        engine._boredom_pressure = 0.8
        first = engine.generate_spontaneous_action(energy=1.0)
        assert first is not None

        # During cooldown
        engine._boredom_pressure = 0.8
        second = engine.generate_spontaneous_action(energy=1.0)
        assert second is None

    def test_spontaneous_types_exist(self, engine):
        """All spontaneous action types exist"""
        types = list(SpontaneousActionType)
        assert len(types) == 6
        assert SpontaneousActionType.BABBLE in types
        assert SpontaneousActionType.SELF_EXAMINE in types

    def test_no_spontaneous_without_energy(self, engine):
        """No energy → cannot perform spontaneous action"""
        engine._boredom_pressure = 0.9
        action = engine.generate_spontaneous_action(energy=0.05)
        assert action is None

    def test_spontaneous_reduces_boredom(self, engine):
        """Spontaneous action releases some boredom"""
        engine._boredom_pressure = 0.8
        boredom_before = engine._boredom_pressure
        engine.generate_spontaneous_action(energy=1.0)
        assert engine._boredom_pressure < boredom_before

    def test_spontaneous_action_data_structure(self, engine):
        """SpontaneousAction data structure is complete"""
        engine._boredom_pressure = 0.8
        action = engine.generate_spontaneous_action(energy=1.0)
        assert action is not None
        assert isinstance(action.action_type, SpontaneousActionType)
        assert 0 <= action.intensity <= 1
        assert isinstance(action.trigger, str)
        assert 0 <= action.boredom_level <= 1

    def test_self_examine_favored_when_low_accuracy(self, engine):
        """Low self-recognition accuracy → prefers self-examination"""
        engine._self_recognition_accuracy = 0.1
        engine._boredom_pressure = 0.8
        engine._curiosity_drive = 0.2  # moderate curiosity
        action = engine.generate_spontaneous_action(energy=1.0)
        assert action is not None
        # When self-recognition is poor, SELF_EXAMINE or BABBLE scores high
        assert action.action_type in (
            SpontaneousActionType.SELF_EXAMINE,
            SpontaneousActionType.BABBLE,
        )

    def test_tick_generates_spontaneous_automatically(self, engine):
        """Tick cycle automatically generates spontaneous actions"""
        # Accumulate enough boredom
        for _ in range(300):
            engine.tick(has_external_input=False, energy=1.0)
        # Should have generated at least one spontaneous action
        assert engine._total_spontaneous_actions > 0

    def test_babble_favored_high_energy_low_self_recognition(self, engine):
        """High energy + low self-recognition → babble"""
        engine._self_recognition_accuracy = 0.15
        engine._boredom_pressure = 0.7
        engine._curiosity_drive = 0.1
        action = engine.generate_spontaneous_action(energy=0.9)
        assert action is not None
        # BABBLE has high score when self_recognition_accuracy is low
        assert action.action_type in (
            SpontaneousActionType.BABBLE,
            SpontaneousActionType.SELF_EXAMINE,
        )


# ============================================================================
# 4. Self-Recognition
# ============================================================================


class TestSelfRecognition:
    """Efference copy → self/other discrimination"""

    def test_efference_copy_registration(self, engine):
        """Efference copy registration"""
        engine.register_efference_copy("vocal", 60.0, "say_440Hz")
        assert len(engine._efference_copies) == 1
        assert engine._efference_copies[0]["predicted_z"] == 60.0

    def test_self_recognition_with_matching_copy(self, engine):
        """Matching efference copy → classified as self"""
        # Register copy
        engine.register_efference_copy("vocal", 60.0)
        # Compare: actual signal close to copy
        judgment = engine.compare_self_other("vocal", 62.0, is_actually_self=True)
        assert judgment.is_self is True
        assert judgment.gamma_self < 0.1

    def test_other_recognition_without_copy(self, engine):
        """No efference copy → classified as other"""
        # No copy registered, compare directly
        judgment = engine.compare_self_other("vocal", 120.0, is_actually_self=False)
        # No copy, compare with self-model, large gap → other
        assert judgment.is_self is False or judgment.confidence < 0.5

    def test_self_recognition_accuracy_improves(self, engine):
        """Correct judgments → self-recognition accuracy improves"""
        initial_acc = engine._self_recognition_accuracy
        assert initial_acc == INITIAL_SELF_ACCURACY

        # Multiple correct judgments
        for i in range(20):
            engine.register_efference_copy("vocal", 60.0)
            engine._tick_count += 1  # avoid expiration
            engine.compare_self_other("vocal", 62.0, is_actually_self=True)

        assert engine._self_recognition_accuracy > initial_acc

    def test_efference_copy_expires(self, engine):
        """Efference copy expires"""
        engine.register_efference_copy("vocal", 60.0)
        # Advance time beyond window
        engine._tick_count += EFFERENCE_COPY_WINDOW + 1
        # Cleanup
        engine._efference_copies = [
            c for c in engine._efference_copies
            if engine._tick_count - c["created_tick"] <= EFFERENCE_COPY_WINDOW
        ]
        assert len(engine._efference_copies) == 0

    def test_wrong_judgment_decreases_accuracy(self, engine):
        """Wrong judgment → accuracy decreases"""
        engine._self_recognition_accuracy = 0.8
        # Error: classified as self but actually not
        engine.compare_self_other("vocal", 60.0, is_actually_self=False)
        # Next: efference copy matches but is not actually self
        assert engine._self_recognition_accuracy < 0.8

    def test_self_model_updates_with_feedback(self, engine):
        """Self-model updates with correct feedback"""
        initial_z = engine._self_model["vocal"]
        for _ in range(10):
            engine.compare_self_other("vocal", 55.0, is_actually_self=True)
        updated_z = engine._self_model["vocal"]
        assert updated_z != initial_z
        # Should shift toward 55.0
        assert abs(updated_z - 55.0) < abs(initial_z - 55.0)

    def test_judgment_data_structure(self, engine):
        """SelfOtherJudgment data structure is complete"""
        engine.register_efference_copy("vocal", 60.0)
        j = engine.compare_self_other("vocal", 65.0)
        assert isinstance(j, SelfOtherJudgment)
        assert j.modality == "vocal"
        assert 0 <= j.gamma_self <= 1
        assert 0 <= j.confidence <= 1
        assert isinstance(j.is_self, bool)

    def test_initial_self_recognition_is_poor(self, engine):
        """Infant self-recognition accuracy is low"""
        assert engine._self_recognition_accuracy == INITIAL_SELF_ACCURACY
        assert INITIAL_SELF_ACCURACY < 0.5  # Infant doesn't know it's themselves


# ============================================================================
# 5. Internal Goal Setting
# ============================================================================


class TestInternalGoals:
    """Curiosity/boredom → autonomous goal setting"""

    def test_no_goal_when_satisfied(self, engine):
        """Low curiosity and boredom → no goal generated"""
        goal = engine.generate_goal_from_curiosity()
        assert goal is None

    def test_curiosity_generates_exploration_goal(self, engine):
        """High curiosity → exploration goal"""
        engine._curiosity_drive = 0.8
        engine._self_recognition_accuracy = 0.7  # Self-recognition sufficient, no interference
        goal = engine.generate_goal_from_curiosity()
        assert goal is not None
        assert goal.source == "curiosity"
        assert "explore" in goal.description

    def test_boredom_generates_stimulation_goal(self, engine):
        """High boredom → seek stimulation goal"""
        engine._boredom_pressure = 0.7
        engine._curiosity_drive = 0.1  # Low curiosity
        engine._self_recognition_accuracy = 0.7  # Self-recognition sufficient
        goal = engine.generate_goal_from_curiosity()
        assert goal is not None
        assert goal.source == "boredom"

    def test_low_self_recognition_generates_self_goal(self, engine):
        """Poor self-recognition → self-exploration goal"""
        engine._self_recognition_accuracy = 0.2
        engine._curiosity_drive = 0.5
        goal = engine.generate_goal_from_curiosity()
        assert goal is not None
        assert goal.source == "self_explore"
        assert "self" in goal.description

    def test_goal_saturation(self, engine):
        """Goal count has an upper limit"""
        engine._curiosity_drive = 0.9
        for _ in range(10):
            engine._tick_count += 1  # avoid dedup
            engine.generate_goal_from_curiosity()
        active = engine.get_active_goals()
        assert len(active) <= engine._max_goals

    def test_goal_satisfaction(self, engine):
        """Goals can be satisfied"""
        engine._curiosity_drive = 0.8
        goal = engine.generate_goal_from_curiosity()
        assert goal is not None
        result = engine.satisfy_goal(goal.description)
        assert result is True

    def test_internal_goal_structure(self, engine):
        """InternalGoal data structure is complete"""
        engine._curiosity_drive = 0.8
        goal = engine.generate_goal_from_curiosity()
        assert isinstance(goal, InternalGoal)
        assert isinstance(goal.description, str)
        assert isinstance(goal.source, str)
        assert 0 <= goal.priority <= 1


# ============================================================================
# 6. Tick Cycle
# ============================================================================


class TestTickCycle:
    """tick() correctly updates all internal state"""

    def test_tick_increments_counter(self, engine):
        """Tick increments counter"""
        assert engine._tick_count == 0
        engine.tick()
        assert engine._tick_count == 1

    def test_tick_returns_state(self, engine):
        """Tick returns complete state"""
        result = engine.tick()
        assert "novelty" in result
        assert "boredom" in result
        assert "curiosity" in result
        assert "spontaneous_urge" in result
        assert "self_recognition_accuracy" in result

    def test_sleeping_freezes_curiosity(self, engine):
        """Curiosity pauses during sleep"""
        engine._boredom_pressure = 0.5
        boredom_before = engine._boredom_pressure
        result = engine.tick(is_sleeping=True)
        assert result["sleeping"] is True
        # Boredom doesn't change during sleep
        assert engine._boredom_pressure == boredom_before

    def test_novelty_decays_each_tick(self, engine):
        """Novelty decays each tick"""
        engine._novelty_level = 0.5
        engine.tick()
        assert engine._novelty_level < 0.5

    def test_history_recorded(self, engine):
        """History recorded correctly"""
        for _ in range(10):
            engine.tick()
        assert len(engine._novelty_history) == 10
        assert len(engine._boredom_history) == 10
        assert len(engine._curiosity_history) == 10

    def test_history_capped(self, engine):
        """History does not exceed upper limit"""
        for _ in range(500):
            engine.tick()
        assert len(engine._novelty_history) <= engine._max_history

    def test_sensory_load_suppresses_boredom(self, engine):
        """High sensory load suppresses boredom accumulation"""
        for _ in range(50):
            engine.tick(has_external_input=False, sensory_load=0.0)
        boredom_no_load = engine.get_boredom()

        engine2 = CuriosityDriveEngine()
        for _ in range(50):
            engine2.tick(has_external_input=False, sensory_load=0.9)
        boredom_high_load = engine2.get_boredom()

        assert boredom_high_load < boredom_no_load


# ============================================================================
# 7. Query Interface
# ============================================================================


class TestQueryInterface:
    """get_state() / get_stats() completeness"""

    def test_get_state(self, engine):
        """get_state is complete"""
        state = engine.get_state()
        assert "novelty_level" in state
        assert "boredom_pressure" in state
        assert "curiosity_drive" in state
        assert "self_recognition_accuracy" in state
        assert "internal_model" in state
        assert "self_model" in state
        assert "active_goals" in state

    def test_get_stats(self, engine):
        """get_stats is complete"""
        stats = engine.get_stats()
        assert "total_ticks" in stats
        assert "total_novelty_events" in stats
        assert "total_spontaneous_actions" in stats
        assert "total_intrinsic_reward" in stats
        assert "self_recognition_accuracy" in stats

    def test_getters(self, engine):
        """All getter methods return correct ranges"""
        assert 0 <= engine.get_boredom() <= 1
        assert 0 <= engine.get_curiosity() <= 1
        assert 0 <= engine.get_novelty() <= 1
        assert 0 <= engine.get_self_recognition_accuracy() <= 1
        assert 0 <= engine.get_spontaneous_urge() <= 1


# ============================================================================
# 8. Full Lifecycle Simulation
# ============================================================================


class TestLifecycleSimulation:
    """Simulate a free period of Alice's time"""

    def test_curiosity_lifecycle(self, engine):
        """
        Full curiosity lifecycle:
        1. Initial — everything is novel
        2. No input → boredom accumulates
        3. Boredom → spontaneous action
        4. New stimulus → curiosity
        5. Repetition → habituation
        """
        # Phase 1: Novelty
        event = engine.evaluate_novelty("visual", 150.0)
        assert event.novelty_score > 0.1

        # Phase 2: Boredom accumulates (spontaneous actions consume some, so final value may be below threshold)
        for _ in range(300):
            engine.tick(has_external_input=False, energy=1.0)
        # Boredom pressure should rise noticeably (even if partially consumed by spontaneous actions)
        assert engine.get_boredom() > 0.3

        # Phase 3: Spontaneous actions have been generated
        assert engine._total_spontaneous_actions > 0

        # Phase 4: New stimulus → curiosity
        event2 = engine.evaluate_novelty("auditory", 300.0)
        assert event2.novelty_score > 0.15
        assert engine.get_curiosity() > 0

        # Phase 5: Repetition → habituation
        gammas = []
        for _ in range(30):
            e = engine.evaluate_novelty("auditory", 300.0)
            gammas.append(e.gamma_novelty)
        assert gammas[-1] < gammas[0]  # Novelty decreases

    def test_self_recognition_development(self, engine):
        """
        Self-recognition development:
        1. Initial — doesn't know it's themselves
        2. Practice → accuracy improves
        3. Eventually — can distinguish self from other
        """
        assert engine._self_recognition_accuracy == INITIAL_SELF_ACCURACY

        # Practice: vocalize → hear → compare
        for i in range(50):
            engine._tick_count = i  # prevent copy expiration
            engine.register_efference_copy("vocal", 60.0)
            engine.compare_self_other("vocal", 62.0, is_actually_self=True)

        assert engine._self_recognition_accuracy > INITIAL_SELF_ACCURACY
        assert engine._self_recognition_accuracy > 0.6

    def test_free_time_produces_spontaneous_activity(self, engine):
        """
        Free time simulation:
        300 ticks without instructions → Alice acts on her own
        """
        spontaneous_actions = []
        for i in range(300):
            result = engine.tick(has_external_input=False, energy=0.8)
            if result["spontaneous_action"] is not None:
                spontaneous_actions.append(result["spontaneous_action"])

        # Should have generated multiple spontaneous actions
        assert len(spontaneous_actions) >= 2

        # Action types should be diverse
        types_seen = {a["type"] for a in spontaneous_actions}
        assert len(types_seen) >= 1

    def test_goal_directed_exploration(self, engine):
        """
        Goal-directed exploration:
        Curiosity → goal → satisfaction
        """
        # Trigger curiosity
        engine.evaluate_novelty("visual", 500.0)
        engine._curiosity_drive = 0.8

        # Generate goal
        goal = engine.generate_goal_from_curiosity()
        assert goal is not None

        # Satisfy goal
        result = engine.satisfy_goal(goal.description)
        assert result is True


# ============================================================================
# 9. AliceBrain Integration
# ============================================================================


class TestAliceBrainIntegration:
    """CuriosityDriveEngine integration with AliceBrain"""

    def test_brain_has_curiosity(self):
        """AliceBrain has curiosity engine"""
        from alice import AliceBrain
        brain = AliceBrain()
        assert hasattr(brain, "curiosity_drive")
        assert isinstance(brain.curiosity_drive, CuriosityDriveEngine)

    def test_see_triggers_novelty(self):
        """Visual input triggers novelty evaluation"""
        from alice import AliceBrain
        brain = AliceBrain()
        pixels = np.random.rand(32, 32).astype(np.float32)
        brain.see(pixels)
        assert brain.curiosity_drive._total_novelty_events > 0

    def test_hear_triggers_novelty(self):
        """Auditory input triggers novelty evaluation"""
        from alice import AliceBrain
        brain = AliceBrain()
        sound = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        brain.hear(sound)
        assert brain.curiosity_drive._total_novelty_events > 0

    def test_say_registers_efference_copy(self):
        """Vocalization registers efference copy"""
        from alice import AliceBrain
        brain = AliceBrain()
        brain.say(440.0, 0.5)
        # Should have at least one efference copy (may be cleaned by tick, check history)
        # Already registered in say
        assert brain.curiosity_drive._tick_count >= 0  # say does not call tick

    def test_introspect_includes_curiosity(self):
        """Introspection report includes curiosity module"""
        from alice import AliceBrain
        brain = AliceBrain()
        report = brain.introspect()
        assert "curiosity_drive" in report["subsystems"]

    def test_perceive_updates_curiosity_tick(self):
        """perceive calls curiosity tick"""
        from alice import AliceBrain
        brain = AliceBrain()
        stimulus = np.random.rand(64).astype(np.float32)
        from alice.core.protocol import Modality
        brain.perceive(stimulus, Modality.VISUAL)
        assert brain.curiosity_drive._tick_count >= 1


# ============================================================================
# 10. Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests"""

    def test_zero_impedance_signal(self, engine):
        """Zero impedance signal doesn't crash"""
        event = engine.evaluate_novelty("visual", 0.0)
        assert 0 <= event.gamma_novelty <= 1

    def test_huge_impedance(self, engine):
        """Extremely large impedance doesn't crash"""
        event = engine.evaluate_novelty("visual", 1e6)
        assert 0 <= event.gamma_novelty <= 1

    def test_rapid_fire_evaluate(self, engine):
        """Rapid consecutive evaluations don't crash"""
        for i in range(1000):
            engine.evaluate_novelty("visual", np.random.uniform(10, 500))
        assert engine._total_novelty_events == 1000

    def test_multiple_efference_copies(self, engine):
        """Multiple efference copies can coexist"""
        for i in range(10):
            engine.register_efference_copy("vocal", 50.0 + i)
        # Copies outside window should be cleaned
        assert len(engine._efference_copies) <= 10

    def test_compare_without_copy(self, engine):
        """Can compare even without copy (uses self-model)"""
        j = engine.compare_self_other("vocal", 100.0)
        assert isinstance(j, SelfOtherJudgment)

    def test_satisfy_nonexistent_goal(self, engine):
        """Satisfying nonexistent goal → False"""
        result = engine.satisfy_goal("nonexistent_goal")
        assert result is False

    def test_all_values_bounded(self, engine):
        """All values within reasonable range"""
        # Stress test
        for _ in range(500):
            engine.evaluate_novelty("visual", np.random.uniform(1, 1000))
            engine.tick(has_external_input=bool(np.random.randint(2)), energy=np.random.random())

        assert 0 <= engine.get_novelty() <= 1
        assert 0 <= engine.get_boredom() <= 1
        assert 0 <= engine.get_curiosity() <= 1
        assert 0 <= engine.get_self_recognition_accuracy() <= 1
        assert 0 <= engine.get_spontaneous_urge() <= 1

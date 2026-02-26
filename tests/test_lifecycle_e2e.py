# -*- coding: utf-8 -*-
"""
End-to-End Lifecycle Tests — Phase 22 Complete Life Validation

Validates the complete perceive→sleep→wake→learn→act cycle,
ensuring all subsystems cooperate across multiple ticks.
"""

import pytest
import numpy as np

from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority


@pytest.fixture
def brain():
    """Create AliceBrain for testing"""
    return AliceBrain(neuron_count=10)


# ============================================================================
# 1. Basic Lifecycle: perceive → multi-tick stability
# ============================================================================


class TestBasicLifecycle:
    """Basic lifecycle - multi-tick stability"""

    def test_100_tick_stability(self, brain):
        """100 continuous perception ticks without crash"""
        for i in range(100):
            stimulus = np.random.randn(64) * 0.3
            result = brain.perceive(stimulus, Modality.AUDITORY)
            assert result is not None
            assert "vitals" in result

        vitals = brain.vitals.get_vitals()
        assert vitals["total_ticks"] == 100
        assert vitals["consciousness"] > 0.1  # No collapse

    def test_perceive_returns_all_subsystems(self, brain):
        """perceive returns all subsystem results"""
        stimulus = np.random.randn(64)
        result = brain.perceive(stimulus, Modality.VISUAL)

        expected_keys = [
            "vitals", "autonomic", "sleep", "consciousness",
            "life_loop", "impedance_adaptation", "predictive",
            "narrative_memory", "recursive_grammar",
            "semantic_pressure", "homeostatic_drive",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_homeostatic_drive_in_perceive(self, brain):
        """perceive returns homeostatic drive data"""
        stimulus = np.random.randn(64)
        result = brain.perceive(stimulus, Modality.AUDITORY)
        hd = result["homeostatic_drive"]
        assert "glucose" in hd
        assert "hydration" in hd
        assert "hunger_drive" in hd
        assert "thirst_drive" in hd


# ============================================================================
# 2. Hunger/Thirst Homeostatic Drive
# ============================================================================


class TestHomeostaticDriveLifecycle:
    """Homeostatic drive - hunger/thirst integrated into main loop"""

    def test_glucose_depletes_over_time(self, brain):
        """Glucose gradually decreases after continuous perception"""
        initial_glucose = brain.homeostatic_drive.glucose
        for _ in range(50):
            brain.perceive(np.random.randn(64), Modality.AUDITORY)
        assert brain.homeostatic_drive.glucose < initial_glucose

    def test_hunger_drive_emerges(self, brain):
        """Hunger drive emerges when glucose falls below threshold"""
        brain.homeostatic_drive.glucose = 0.4  # Low glucose
        brain.perceive(np.random.randn(64), Modality.AUDITORY)
        assert brain.homeostatic_drive.hunger_drive > 0

    def test_thirst_drive_emerges(self, brain):
        """Thirst drive emerges when hydration falls below threshold"""
        brain.homeostatic_drive.hydration = 0.5  # Low hydration
        brain.perceive(np.random.randn(64), Modality.AUDITORY)
        assert brain.homeostatic_drive.thirst_drive > 0

    def test_eating_restores_glucose(self, brain):
        """Glucose gradually recovers after eating"""
        brain.homeostatic_drive.glucose = 0.3
        brain.homeostatic_drive.eat()

        # Digestion takes multiple ticks
        for _ in range(20):
            brain.homeostatic_drive.tick()
        assert brain.homeostatic_drive.glucose > 0.3

    def test_drinking_restores_hydration(self, brain):
        """Hydration gradually recovers after drinking"""
        brain.homeostatic_drive.hydration = 0.4
        brain.homeostatic_drive.drink()

        for _ in range(20):
            brain.homeostatic_drive.tick()
        assert brain.homeostatic_drive.hydration > 0.4

    def test_hunger_irritability_affects_valence(self, brain):
        """Hunger-induced irritability should affect valence (shift emotion negative)"""
        brain.homeostatic_drive.glucose = 0.2  # Very hungry
        result = brain.perceive(np.random.randn(64), Modality.AUDITORY)
        # Cognitive penalty > 0 under hunger
        assert result["homeostatic_drive"]["cognitive_penalty"] > 0

    def test_sleep_reduces_metabolism(self, brain):
        """Metabolic rate decreases during sleep — glucose consumed more slowly"""
        brain.homeostatic_drive.glucose = 0.8
        signal = brain.homeostatic_drive.tick(is_sleeping=True)
        glucose_after_sleep = brain.homeostatic_drive.glucose

        brain.homeostatic_drive.glucose = 0.8
        signal = brain.homeostatic_drive.tick(is_sleeping=False)
        glucose_after_wake = brain.homeostatic_drive.glucose

        # Less consumption during sleep
        assert glucose_after_sleep > glucose_after_wake

    def test_satiety_suppresses_hunger(self, brain):
        """Satiety suppresses hunger drive"""
        brain.homeostatic_drive.glucose = 0.4  # Low glucose
        brain.homeostatic_drive.tick()
        hunger_before_eat = brain.homeostatic_drive.hunger_drive

        brain.homeostatic_drive.eat()
        brain.homeostatic_drive.tick()  # Satiety begins
        hunger_after_eat = brain.homeostatic_drive.hunger_drive

        # Satiety should suppress hunger
        assert hunger_after_eat < hunger_before_eat

    def test_gamma_hunger_impedance(self, brain):
        """Γ_hunger correctly reflects glucose deviation"""
        brain.homeostatic_drive.glucose = 1.0
        assert brain.homeostatic_drive.get_gamma_hunger() == 0.0

        brain.homeostatic_drive.glucose = 0.5
        gamma = brain.homeostatic_drive.get_gamma_hunger()
        assert 0 < gamma < 1

    def test_starvation_tracking(self, brain):
        """Low glucose tracks starvation ticks"""
        brain.homeostatic_drive.glucose = 0.1  # Severe hypoglycemia
        for _ in range(10):
            brain.homeostatic_drive.tick()
        assert brain.homeostatic_drive._starvation_ticks > 0


# ============================================================================
# 3. Physics Reward Engine
# ============================================================================


class TestPhysicsRewardLifecycle:
    """Physics reward engine — impedance matching replaces Q-learning"""

    def test_choose_action(self, brain):
        """Action selection works correctly"""
        action, explored = brain.physics_reward.choose_action(
            "test_state", ["a", "b", "c"]
        )
        assert action in ["a", "b", "c"]

    def test_update_returns_rpe(self, brain):
        """update returns RPE (dopamine signal)"""
        rpe = brain.physics_reward.update("s1", "a1", 1.0, "s2")
        assert isinstance(rpe, float)

    def test_positive_rpe_lowers_impedance(self, brain):
        """Positive RPE → channel impedance decreases"""
        brain.physics_reward.update("s1", "a1", 0.0, "s2")  # Establish baseline
        z_before = brain.physics_reward._channels[("s1", "a1")].impedance
        brain.physics_reward.update("s1", "a1", 1.0, "s2")  # Positive reward
        z_after = brain.physics_reward._channels[("s1", "a1")].impedance
        assert z_after < z_before

    def test_negative_rpe_raises_impedance(self, brain):
        """Negative RPE → channel impedance increases"""
        # First establish high expectation
        for _ in range(5):
            brain.physics_reward.update("s2", "a2", 1.0, "s3")
        z_before = brain.physics_reward._channels[("s2", "a2")].impedance
        # Then disappointed
        brain.physics_reward.update("s2", "a2", -1.0, "s3")
        z_after = brain.physics_reward._channels[("s2", "a2")].impedance
        assert z_after > z_before

    def test_act_uses_physics_reward(self, brain):
        """act() uses physics_reward instead of old Q-table"""
        result = brain.act("test_state", ["walk", "run", "stop"])
        assert "chosen_action" in result
        assert result["chosen_action"] in ["walk", "run", "stop"]

    def test_learn_from_feedback_updates_physics_reward(self, brain):
        """learn_from_feedback updates physics_reward"""
        result = brain.learn_from_feedback("s1", "a1", 1.0, "s2", ["a1", "a2"])
        assert "dopamine_signal" in result
        assert brain.physics_reward._total_updates > 0

    def test_dopamine_pipeline_unified(self, brain):
        """Dopamine pipeline unified: physics_reward → basal_ganglia"""
        brain.learn_from_feedback("s1", "a1", 2.0, "s2")
        # Dopamine should be injected into basal ganglia
        assert brain.basal_ganglia._dopamine_level != 0.5  # Has been modulated

    def test_replay_works(self, brain):
        """Experience replay works correctly"""
        for i in range(20):
            brain.physics_reward.update(f"s{i}", "a1", float(i % 3), f"s{i+1}")
        error = brain.physics_reward.replay()
        assert error >= 0

    def test_q_value_compatibility(self, brain):
        """get_q_value maintains API compatibility"""
        brain.physics_reward.update("s1", "a1", 1.0, "s2")
        q = brain.physics_reward.get_q_value("s1", "a1")
        assert isinstance(q, float)

    def test_channel_gamma_decreases_with_learning(self, brain):
        """Channel Γ decreases after repeated positive rewards"""
        for _ in range(10):
            brain.physics_reward.update("s_learn", "a_learn", 1.0, "s_next")
        gamma = brain.physics_reward.get_channel_gamma("s_learn", "a_learn")
        initial_gamma = (100.0 - 75.0) / (100.0 + 75.0)
        assert gamma < initial_gamma

    def test_introspect_includes_physics_reward(self, brain):
        """introspect() includes physics_reward statistics"""
        info = brain.introspect()
        rl_stats = info["subsystems"]["reinforcement_learning"]
        assert "total_channels" in rl_stats  # PhysicsRewardEngine fields
        assert "avg_gamma" in rl_stats


# ============================================================================
# 4. End-to-End: perceive → learn → act → reward
# ============================================================================


class TestPerceiveLearnActCycle:
    """Complete perceive→learn→act→reward closed loop"""

    def test_full_perceive_learn_act_cycle(self, brain):
        """Full closed loop: perceive → learn → act"""
        # Perceive
        result = brain.perceive(np.random.randn(64), Modality.AUDITORY)
        assert result is not None

        # Act
        act_result = brain.act("heard_sound", ["respond", "ignore", "explore"])
        action = act_result["chosen_action"]

        # Learn
        learn_result = brain.learn_from_feedback(
            "heard_sound", action, 1.0, "after_respond"
        )
        assert learn_result["dopamine_signal"] is not None

    def test_habit_formation_via_physics_reward(self, brain):
        """Repeated successful behavior → habit formation (Γ_action→0)"""
        for i in range(30):
            brain.act("daily_routine", ["brush_teeth", "skip"])
            brain.learn_from_feedback("daily_routine", "brush_teeth", 1.0, "clean")
            brain.basal_ganglia.update_after_action("daily_routine", "brush_teeth", 1.0, True)

        ch = brain.basal_ganglia.get_channel("daily_routine", "brush_teeth")
        assert ch is not None
        assert ch.gamma_habit < 0.5  # Trending toward habit

    def test_multi_cycle_consistency(self, brain):
        """Multiple perceive-act cycles produce no inconsistencies"""
        for i in range(20):
            brain.perceive(np.random.randn(64), Modality.VISUAL)
            brain.act(f"state_{i}", ["a", "b"])
            brain.learn_from_feedback(f"state_{i}", "a", 0.5, f"state_{i+1}")

        stats = brain.physics_reward.get_stats()
        assert stats["total_updates"] == 20


# ============================================================================
# 5. End-to-End: perceive → sleep → wake → recovery
# ============================================================================


class TestSleepWakeCycle:
    """Sleep-wake cycle"""

    def test_energy_depletion_and_recovery(self, brain):
        """Energy recovers through sleep after depletion"""
        # Deplete energy
        brain.autonomic.energy = 0.3
        brain.autonomic.sympathetic = 0.8

        # Simulate sleep period
        for _ in range(30):
            brain.autonomic.tick(is_sleeping=True)

        # Energy should have recovered somewhat
        assert brain.autonomic.energy > 0.3

    def test_glucose_depletes_during_wake_cycle(self, brain):
        """Glucose gradually depletes during wakefulness"""
        initial = brain.homeostatic_drive.glucose
        for _ in range(100):
            brain.perceive(np.random.randn(64), Modality.AUDITORY)
        assert brain.homeostatic_drive.glucose < initial

    def test_sleep_consolidation_with_homeostasis(self, brain):
        """Homeostatic system also operates during sleep consolidation"""
        brain.homeostatic_drive.glucose = 0.6
        # Sleep tick
        brain.homeostatic_drive.tick(is_sleeping=True)
        glucose_after_sleep = brain.homeostatic_drive.glucose
        # Low metabolic rate during sleep → less glucose consumed (but still consuming)
        assert glucose_after_sleep < 0.6


# ============================================================================
# 6. Pain Collapse → Recovery
# ============================================================================


class TestPainCollapseRecovery:
    """Pain collapse and recovery"""

    def test_pain_escalation_and_freeze(self, brain):
        """Pain escalates until freeze"""
        brain.vitals.ram_temperature = 0.95
        for _ in range(10):
            brain.vitals.tick(
                critical_queue_len=10,
                high_queue_len=5,
                total_queue_len=30,
                sensory_activity=0.8,
                emotional_valence=-0.5,
                left_brain_activity=0.5,
                right_brain_activity=0.5,
                cycle_elapsed_ms=10.0,
                reflected_energy=0.8,
            )
        assert brain.vitals.pain_level > 0.5

    def test_natural_recovery_from_pain(self, brain):
        """System naturally recovers after stress is removed"""
        brain.vitals.ram_temperature = 0.8
        brain.vitals.pain_level = 0.6

        for _ in range(50):
            brain.vitals.tick(
                critical_queue_len=0,
                high_queue_len=0,
                total_queue_len=0,
                sensory_activity=0.0,
                emotional_valence=0.0,
                left_brain_activity=0.0,
                right_brain_activity=0.0,
                cycle_elapsed_ms=1.0,
                reflected_energy=0.0,
            )
        assert brain.vitals.pain_level < 0.6
        assert brain.vitals.ram_temperature < 0.8


# ============================================================================
# 7. Homeostatic Drive Unit Tests
# ============================================================================


class TestHomeostaticDriveUnit:
    """HomeostaticDriveEngine unit tests"""

    def test_init_values(self):
        from alice.brain.homeostatic_drive import HomeostaticDriveEngine
        hd = HomeostaticDriveEngine()
        assert hd.glucose == 1.0
        assert hd.hydration == 1.0
        assert hd.hunger_drive == 0.0
        assert hd.thirst_drive == 0.0

    def test_tick_returns_signal(self):
        from alice.brain.homeostatic_drive import HomeostaticDriveEngine
        hd = HomeostaticDriveEngine()
        signal = hd.tick()
        assert hasattr(signal, "hunger_intensity")
        assert hasattr(signal, "thirst_intensity")
        assert hasattr(signal, "needs_food")
        assert hasattr(signal, "needs_water")
        assert signal.metabolic_rate > 0

    def test_high_sympathetic_increases_metabolism(self):
        from alice.brain.homeostatic_drive import HomeostaticDriveEngine
        hd = HomeostaticDriveEngine()
        sig_low = hd.tick(sympathetic=0.1)
        hd.glucose = 1.0  # Reset
        sig_high = hd.tick(sympathetic=0.9)
        assert sig_high.metabolic_rate > sig_low.metabolic_rate

    def test_dehydration_pain(self):
        from alice.brain.homeostatic_drive import HomeostaticDriveEngine
        hd = HomeostaticDriveEngine()
        hd.hydration = 0.15  # Severe dehydration
        signal = hd.tick()
        assert signal.pain_contribution > 0

    def test_get_state(self):
        from alice.brain.homeostatic_drive import HomeostaticDriveEngine
        hd = HomeostaticDriveEngine()
        hd.tick()
        state = hd.get_state()
        assert "glucose" in state
        assert "hydration" in state
        assert "gamma_hunger" in state
        assert "gamma_thirst" in state

    def test_reset(self):
        from alice.brain.homeostatic_drive import HomeostaticDriveEngine
        hd = HomeostaticDriveEngine()
        hd.glucose = 0.1
        hd.hydration = 0.2
        hd.reset()
        assert hd.glucose == 1.0
        assert hd.hydration == 1.0

    def test_energy_recovery_factor(self):
        from alice.brain.homeostatic_drive import HomeostaticDriveEngine
        hd = HomeostaticDriveEngine()
        hd.glucose = 1.0
        sig_full = hd.tick()
        hd.glucose = 0.3
        sig_low = hd.tick()
        assert sig_full.energy_recovery_factor >= sig_low.energy_recovery_factor


# ============================================================================
# 8. Physics Reward Engine Unit Tests
# ============================================================================


class TestPhysicsRewardUnit:
    """PhysicsRewardEngine unit tests"""

    def test_init(self):
        from alice.brain.physics_reward import PhysicsRewardEngine
        pr = PhysicsRewardEngine()
        assert pr._dopamine == 0.5
        assert len(pr._channels) == 0

    def test_channel_creation(self):
        from alice.brain.physics_reward import PhysicsRewardEngine
        pr = PhysicsRewardEngine()
        ch = pr._get_or_create_channel("s1", "a1")
        assert ch.impedance == 100.0
        assert ch.visit_count == 0

    def test_gamma_calculation(self):
        from alice.brain.physics_reward import PhysicsRewardEngine, Z_SOURCE
        pr = PhysicsRewardEngine()
        ch = pr._get_or_create_channel("s1", "a1")
        expected_gamma = abs(100.0 - Z_SOURCE) / (100.0 + Z_SOURCE)
        assert abs(ch.gamma - expected_gamma) < 0.001

    def test_transmission_efficiency(self):
        from alice.brain.physics_reward import PhysicsRewardEngine
        pr = PhysicsRewardEngine()
        ch = pr._get_or_create_channel("s1", "a1")
        assert 0 < ch.transmission < 1  # Non-perfect match

    def test_choose_empty_actions(self):
        from alice.brain.physics_reward import PhysicsRewardEngine
        pr = PhysicsRewardEngine()
        action, explored = pr.choose_action("s1", [])
        assert action == ""

    def test_dopamine_injection(self):
        from alice.brain.physics_reward import PhysicsRewardEngine
        pr = PhysicsRewardEngine()
        pr.inject_dopamine(0.3)
        assert pr._dopamine > 0.5

    def test_get_stats(self):
        from alice.brain.physics_reward import PhysicsRewardEngine
        pr = PhysicsRewardEngine()
        pr.update("s1", "a1", 1.0, "s2")
        stats = pr.get_stats()
        assert stats["total_updates"] == 1
        assert stats["total_channels"] == 1

    def test_learned_channel(self):
        from alice.brain.physics_reward import PhysicsRewardEngine
        pr = PhysicsRewardEngine()
        # Large positive rewards → impedance drops below learned threshold
        for _ in range(50):
            pr.update("s_train", "a_train", 2.0, "s_next")
        ch = pr._channels[("s_train", "a_train")]
        # Impedance should have significantly decreased
        assert ch.impedance < 100.0

    def test_get_state(self):
        from alice.brain.physics_reward import PhysicsRewardEngine
        pr = PhysicsRewardEngine()
        pr.update("s1", "a1", 1.0, "s2")
        state = pr.get_state()
        assert "channels" in state
        assert "s1:a1" in state["channels"]


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

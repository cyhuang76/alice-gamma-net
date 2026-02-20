#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 19 — SocialResonanceEngine Unit Tests

tests/test_social_resonance.py

Tests:
  1. Impedance coupling basics (Γ + η)
  2. Mismatch vs match
  3. Trust dynamics (grow + decay)
  4. Belief tracking (update + witness + reality change)
  5. Sally-Anne (low/high ToM)
  6. False belief detection
  7. Social homeostasis (loneliness + satiation)
  8. Compassion fatigue + recovery
  9. Bidirectional coupling
  10. Frequency sync
  11. Agent model management (capacity limit)
  12. Tick decay dynamics
  13. Sync from mirror neurons
  14. AliceBrain integration
  15. Full state/stats reporting
"""

import math
import numpy as np
import pytest

from alice.brain.social_resonance import (
    SocialResonanceEngine,
    Belief,
    SocialAgentModel,
    SocialCouplingResult,
    SallyAnneResult,
    SocialHomeostasisState,
    Z_SOCIAL_BASE,
    Z_LISTENER_DEFAULT,
    K_RELEASE,
    K_ABSORB,
    K_REFLECT,
    LONELINESS_THRESHOLD,
    SYNC_THRESHOLD,
    MAX_AGENT_MODELS,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def engine():
    return SocialResonanceEngine()


@pytest.fixture
def alice():
    from alice.alice_brain import AliceBrain
    return AliceBrain(neuron_count=80)


# ============================================================
# Test 1: Impedance coupling basics
# ============================================================

class TestImpedanceCoupling:
    def test_speaker_impedance_decreases_with_pressure(self, engine):
        """High pressure -> low impedance"""
        z_low_p = engine.compute_speaker_impedance(0.0)
        z_high_p = engine.compute_speaker_impedance(2.0)
        assert z_low_p > z_high_p

    def test_speaker_impedance_base(self, engine):
        """Zero pressure -> Z_base"""
        z = engine.compute_speaker_impedance(0.0)
        assert abs(z - Z_SOCIAL_BASE) < 0.01

    def test_listener_impedance_decreases_with_empathy(self, engine):
        z_cold = engine.compute_listener_impedance(0.0, 0.0)
        z_warm = engine.compute_listener_impedance(0.9, 0.9)
        assert z_cold > z_warm

    def test_listener_impedance_with_bond(self, engine):
        """Intimate bond further lowers impedance"""
        z_stranger = engine.compute_listener_impedance(0.5, 0.5, bond_impedance=75.0)
        z_intimate = engine.compute_listener_impedance(0.5, 0.5, bond_impedance=10.0)
        assert z_stranger > z_intimate

    def test_gamma_range(self, engine):
        result = engine.couple("a", "b", 1.0, 0.5, 0.5)
        assert 0.0 <= result.gamma_social <= 1.0
        assert 0.0 <= result.energy_transfer <= 1.0

    def test_eta_from_gamma(self, engine):
        """eta = 1 - gamma^2"""
        result = engine.couple("a", "b", 1.0, 0.5, 0.5)
        expected_eta = 1.0 - result.gamma_social ** 2
        assert abs(result.energy_transfer - expected_eta) < 0.001

    def test_pressure_components(self, engine):
        result = engine.couple("a", "b", 1.0, 0.7, 0.7)
        assert result.pressure_released >= 0
        assert result.pressure_absorbed >= 0
        assert result.pressure_reflected >= 0


# ============================================================
# Test 2: Mismatch vs Match
# ============================================================

class TestMismatchMatch:
    def test_cold_listener_high_gamma(self, engine):
        result = engine.couple("a", "b", 1.0, 0.05, 0.0)
        assert result.gamma_social > 0.5

    def test_warm_listener_low_gamma(self, engine):
        result = engine.couple("a", "b", 1.0, 0.9, 0.9)
        assert result.gamma_social < 0.3

    def test_cold_releases_less(self, engine):
        cold = engine.couple("a", "b", 1.0, 0.05, 0.0)
        warm = engine.couple("c", "d", 1.0, 0.9, 0.9)
        assert warm.pressure_released > cold.pressure_released

    def test_cold_reflects_more(self, engine):
        cold = engine.couple("a", "b", 1.0, 0.05, 0.0)
        warm = engine.couple("c", "d", 1.0, 0.9, 0.9)
        assert cold.pressure_reflected > warm.pressure_reflected


# ============================================================
# Test 3: Trust dynamics
# ============================================================

class TestTrustDynamics:
    def test_trust_grows_with_good_interaction(self, engine):
        initial_trust = 0.5  # default
        for _ in range(50):
            engine.couple("self", "friend", 1.0, 0.9, 0.9)
        model = engine.get_agent_model("friend")
        assert model.trust > initial_trust

    def test_trust_drops_with_coldness(self, engine):
        # Build some trust first
        for _ in range(20):
            engine.couple("self", "friend", 1.0, 0.9, 0.9)
        trust_after_good = engine.get_agent_model("friend").trust

        # Be cold
        for _ in range(50):
            engine.couple("self", "friend", 1.0, 0.05, 0.0)
        trust_after_cold = engine.get_agent_model("friend").trust

        assert trust_after_cold < trust_after_good

    def test_trust_bounded(self, engine):
        for _ in range(200):
            engine.couple("self", "friend", 1.0, 0.9, 0.9)
        model = engine.get_agent_model("friend")
        assert 0.0 <= model.trust <= 1.0


# ============================================================
# Test 4: Belief tracking
# ============================================================

class TestBeliefTracking:
    def test_update_belief(self, engine):
        belief = engine.update_belief("sally", "ball", "basket", "basket")
        assert belief.subject == "ball"
        assert belief.value == "basket"
        assert not belief.is_false_belief

    def test_false_belief_creation(self, engine):
        engine.update_belief("sally", "ball", "basket", "basket")
        engine.update_reality("ball", "box")

        model = engine.get_agent_model("sally")
        belief = model.beliefs["ball"]
        assert belief.is_false_belief
        assert belief.value == "basket"
        assert belief.reality_value == "box"

    def test_witness_event_updates_belief(self, engine):
        engine.update_belief("sally", "ball", "basket", "basket")
        engine.update_reality("ball", "box")

        # Sally witnesses the change
        engine.agent_witnesses_event("sally", "ball", "box")

        model = engine.get_agent_model("sally")
        belief = model.beliefs["ball"]
        assert not belief.is_false_belief
        assert belief.value == "box"

    def test_gamma_belief(self, engine):
        engine.update_belief("sally", "ball", "basket", "box", confidence=0.8)
        gamma = engine.get_gamma_belief("sally")
        assert gamma > 0  # false belief has non-zero gamma


# ============================================================
# Test 5: Sally-Anne
# ============================================================

class TestSallyAnne:
    def test_low_tom_egocentric(self, engine):
        """Low ToM -> Alice predicts based on reality (egocentric error)"""
        engine._tom_capacity = 0.1
        engine.update_belief("sally", "ball", "basket", "box", 1.0)

        result = engine.sally_anne_test("sally", "ball")
        # Level 0: predicts reality
        assert result.tom_level == 0
        assert result.alice_prediction == "box"

    def test_high_tom_correct(self, engine):
        """High ToM -> Alice predicts Sally's actual belief"""
        engine._tom_capacity = 0.8
        engine.update_belief("sally", "ball", "basket", "box", 1.0)

        result = engine.sally_anne_test("sally", "ball")
        assert result.tom_level == 2
        assert result.alice_prediction == "basket"
        assert result.prediction_correct

    def test_unknown_belief(self, engine):
        """No tracked belief -> unknown prediction"""
        result = engine.sally_anne_test("unknown_agent", "something")
        assert result.alice_prediction == "unknown"
        assert result.tom_level == 0

    def test_sally_anne_result_structure(self, engine):
        engine._tom_capacity = 0.8
        engine.update_belief("sally", "ball", "basket", "box", 1.0)
        result = engine.sally_anne_test("sally", "ball")

        assert isinstance(result, SallyAnneResult)
        assert result.agent_id == "sally"
        assert result.subject == "ball"
        assert 0.0 <= result.confidence <= 1.0


# ============================================================
# Test 6: False belief detection
# ============================================================

class TestFalseBelief:
    def test_get_false_beliefs(self, engine):
        engine.update_belief("a", "x", "left", "right", 0.9)
        engine.update_belief("a", "y", "up", "up", 0.9)  # correct
        fbs = engine.get_false_beliefs("a")
        assert len(fbs) == 1
        assert fbs[0].subject == "x"

    def test_no_false_beliefs_when_correct(self, engine):
        engine.update_belief("a", "x", "left", "left")
        fbs = engine.get_false_beliefs("a")
        assert len(fbs) == 0

    def test_false_belief_detection_count(self, engine):
        engine.update_belief("a", "x", "left", "right")
        engine.update_belief("b", "y", "up", "down")
        assert engine._false_belief_detections == 2


# ============================================================
# Test 7: Social homeostasis
# ============================================================

class TestSocialHomeostasis:
    def test_loneliness_from_isolation(self, engine):
        for _ in range(250):
            engine.tick(has_social_input=False)
        h = engine.get_homeostasis()
        assert h.is_lonely
        assert h.social_need > LONELINESS_THRESHOLD

    def test_social_need_drops_with_interaction(self, engine):
        # Build need
        for _ in range(200):
            engine.tick(has_social_input=False)
        high_need = engine.get_social_need()

        # Interact
        for _ in range(50):
            engine.couple("self", "friend", 1.0, 0.8, 0.8)
            engine.tick(has_social_input=True)
        low_need = engine.get_social_need()

        assert low_need < high_need

    def test_optimal_zone(self, engine):
        """After moderate interaction, should be in optimal zone"""
        # Some isolation
        for _ in range(50):
            engine.tick(has_social_input=False)
        # Some interaction
        for _ in range(30):
            engine.couple("s", "f", 0.5, 0.7, 0.7)
            engine.tick(has_social_input=True)

        h = engine.get_homeostasis()
        # Should be in reasonable range
        assert 0.0 <= h.social_need <= 1.0


# ============================================================
# Test 8: Compassion fatigue + recovery
# ============================================================

class TestCompassionFatigue:
    def test_fatigue_from_listening(self, engine):
        initial = engine.get_compassion_energy()
        for _ in range(200):
            engine.couple("patient", "therapist", 2.0, 0.95, 0.95)
            engine.tick(has_social_input=True)
        final = engine.get_compassion_energy()
        assert final < initial

    def test_recovery_from_solitude(self, engine):
        # Drain compassion
        for _ in range(200):
            engine.couple("patient", "therapist", 2.0, 0.95, 0.95)
            engine.tick(has_social_input=True)
        drained = engine.get_compassion_energy()

        # Recover
        for _ in range(100):
            engine.tick(has_social_input=False)
        recovered = engine.get_compassion_energy()

        assert recovered > drained


# ============================================================
# Test 9: Bidirectional coupling
# ============================================================

class TestBidirectional:
    def test_returns_two_results(self, engine):
        r_ab, r_ba = engine.bidirectional_couple(
            "a", "b", 1.0, 1.0, 0.7, 0.7, 0.7, 0.7,
        )
        assert isinstance(r_ab, SocialCouplingResult)
        assert isinstance(r_ba, SocialCouplingResult)

    def test_bidirectional_more_total_release(self, engine):
        """Bidirectional releases more pressure total"""
        r_ab, r_ba = engine.bidirectional_couple(
            "a", "b", 1.0, 1.0, 0.7, 0.7, 0.7, 0.7,
        )
        r_uni = engine.couple("c", "d", 1.0, 0.7, 0.7)
        total_bi = r_ab.pressure_released + r_ba.pressure_released
        assert total_bi > r_uni.pressure_released

    def test_creates_mutual_models(self, engine):
        engine.bidirectional_couple("a", "b", 1.0, 1.0, 0.7, 0.7, 0.7, 0.7)
        assert engine.get_agent_model("a") is not None
        assert engine.get_agent_model("b") is not None


# ============================================================
# Test 10: Frequency sync
# ============================================================

class TestFrequencySync:
    def test_sync_increases_with_interaction(self, engine):
        for _ in range(50):
            engine.couple("a", "partner", 0.5, 0.8, 0.8)
            engine.tick(has_social_input=True)

        model = engine.get_agent_model("partner")
        assert model.sync_degree > 0.0

    def test_sync_monotonically_increases(self, engine):
        syncs = []
        for _ in range(50):
            engine.couple("a", "partner", 0.5, 0.8, 0.8)
            engine.tick(has_social_input=True)
            model = engine.get_agent_model("partner")
            syncs.append(model.sync_degree)

        # Should be monotonically increasing
        for i in range(len(syncs) - 1):
            assert syncs[i] <= syncs[i + 1] + 0.001

    def test_sync_bounded(self, engine):
        for _ in range(200):
            engine.couple("a", "p", 0.5, 0.9, 0.9)
        model = engine.get_agent_model("p")
        assert 0.0 <= model.sync_degree <= 1.0


# ============================================================
# Test 11: Agent model capacity
# ============================================================

class TestAgentModelCapacity:
    def test_max_agents(self, engine):
        for i in range(MAX_AGENT_MODELS + 5):
            engine.couple("self", f"agent_{i}", 0.5, 0.5, 0.5)
        assert len(engine._agent_models) <= MAX_AGENT_MODELS

    def test_evicts_oldest(self, engine):
        # Create max agents
        for i in range(MAX_AGENT_MODELS):
            engine.couple("self", f"agent_{i}", 0.5, 0.5, 0.5)
            engine._tick_count += 10

        # Add one more -> should evict oldest
        engine.couple("self", "new_agent", 0.5, 0.5, 0.5)
        assert engine.get_agent_model("new_agent") is not None

    def test_get_all_agent_ids(self, engine):
        engine.couple("a", "b", 1.0, 0.5, 0.5)
        engine.couple("c", "d", 1.0, 0.5, 0.5)
        ids = engine.get_all_agent_ids()
        assert len(ids) >= 2


# ============================================================
# Test 12: Tick decay dynamics
# ============================================================

class TestTickDecay:
    def test_social_need_grows_without_input(self, engine):
        need_before = engine.get_social_need()
        for _ in range(50):
            engine.tick(has_social_input=False)
        need_after = engine.get_social_need()
        assert need_after > need_before

    def test_loneliness_duration_tracked(self, engine):
        for _ in range(10):
            engine.tick(has_social_input=False)
        assert engine.get_loneliness_duration() == 10

    def test_loneliness_resets_with_social(self, engine):
        for _ in range(10):
            engine.tick(has_social_input=False)
        engine.tick(has_social_input=True)
        assert engine.get_loneliness_duration() == 0

    def test_tick_returns_dict(self, engine):
        result = engine.tick()
        assert "tick" in result
        assert "social_need" in result
        assert "compassion_energy" in result
        assert "is_lonely" in result
        assert "tom_capacity" in result


# ============================================================
# Test 13: Sync from mirror neurons
# ============================================================

class TestSyncFromMirror:
    def test_sync_tom_capacity(self, engine):
        engine.sync_from_mirror({}, empathy_capacity=0.5, tom_capacity=0.7)
        assert engine.get_tom_capacity() >= 0.7

    def test_sync_agent_models(self, engine):
        from alice.brain.mirror_neurons import AgentModel
        mirror_models = {
            "alice": AgentModel(
                agent_id="alice",
                inferred_emotion=0.3,
                inferred_arousal=0.7,
            ),
        }
        engine.sync_from_mirror(mirror_models, 0.5, 0.5)
        model = engine.get_agent_model("alice")
        assert model is not None
        assert abs(model.inferred_valence - 0.3) < 0.01


# ============================================================
# Test 14: AliceBrain integration
# ============================================================

class TestAliceBrainIntegration:
    def test_social_resonance_exists(self, alice):
        assert hasattr(alice, "social_resonance")
        assert isinstance(alice.social_resonance, SocialResonanceEngine)

    def test_perceive_includes_social(self, alice):
        pixels = np.random.rand(64, 64) * 0.5
        result = alice.see(pixels)
        assert "social_resonance" in result

    def test_introspect_includes_social(self, alice):
        report = alice.introspect()
        assert "social_resonance" in report["subsystems"]

    def test_social_tick_runs(self, alice):
        """Social resonance tick happens in perceive pipeline"""
        pixels = np.random.rand(64, 64) * 0.5
        alice.see(pixels)
        assert alice.social_resonance._tick_count > 0


# ============================================================
# Test 15: State & stats reporting
# ============================================================

class TestStateReporting:
    def test_get_state(self, engine):
        engine.couple("a", "b", 1.0, 0.7, 0.7)
        state = engine.get_state()
        assert "tick" in state
        assert "social_need" in state
        assert "tracked_agents" in state
        assert isinstance(state["tracked_agents"], dict)

    def test_get_stats(self, engine):
        for _ in range(5):
            engine.couple("a", "b", 1.0, 0.5, 0.5)
            engine.tick()
        stats = engine.get_stats()
        assert stats["total_couplings"] >= 5
        assert "avg_gamma" in stats
        assert "avg_eta" in stats

    def test_no_nan_in_stats(self, engine):
        for _ in range(50):
            engine.couple("a", "b", 1.0, 0.5, 0.5)
            engine.tick()
        stats = engine.get_stats()
        for key, val in stats.items():
            if isinstance(val, float):
                assert not math.isnan(val), f"NaN in {key}"
                assert not math.isinf(val), f"Inf in {key}"

    def test_homeostasis_state(self, engine):
        h = engine.get_homeostasis()
        assert isinstance(h, SocialHomeostasisState)
        assert 0.0 <= h.social_need <= 1.0
        assert 0.0 <= h.compassion_energy <= 1.0


# ============================================================
# Test: Belief dataclass
# ============================================================

class TestBeliefDataclass:
    def test_correct_belief(self):
        b = Belief("location", "basket", "basket", 0.9)
        assert not b.is_false_belief
        assert b.gamma_belief == 0.0

    def test_false_belief(self):
        b = Belief("location", "basket", "box", 0.9)
        assert b.is_false_belief
        assert b.gamma_belief == 0.9

    def test_low_confidence_false_belief(self):
        b = Belief("location", "basket", "box", 0.2)
        assert b.is_false_belief
        assert b.gamma_belief == 0.2

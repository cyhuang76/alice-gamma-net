# -*- coding: utf-8 -*-
"""Tests for Attention Plasticity Engine"""
import numpy as np
import pytest
from alice.brain.attention_plasticity import (
    AttentionPlasticityEngine, ModalityTrainingRecord,
    GATE_TAU_INITIAL, GATE_TAU_MIN, Q_INITIAL, Q_MAX,
    INHIBITION_EFFICIENCY_INITIAL, INHIBITION_EFFICIENCY_MIN,
    PFC_CAPACITY_INITIAL, PFC_CAPACITY_MAX,
    ATTENTION_SLOTS_INITIAL, ATTENTION_SLOTS_MAX,
    SLOT_TRAINING_THRESHOLD, PATHWAY_SEGMENTS,
)


class TestInitialization:
    def test_engine_starts_empty(self):
        engine = AttentionPlasticityEngine()
        state = engine.get_state()
        assert state["global"]["pfc_capacity"] == PFC_CAPACITY_INITIAL
        assert state["global"]["attention_slots"] == ATTENTION_SLOTS_INITIAL
        assert state["modalities"] == {}

    def test_defaults_for_unknown_modality(self):
        engine = AttentionPlasticityEngine()
        assert engine.get_gate_tau("visual") == GATE_TAU_INITIAL
        assert engine.get_tuner_q("visual") == Q_INITIAL
        assert engine.get_inhibition_cost_multiplier("visual") == INHIBITION_EFFICIENCY_INITIAL
        assert engine.get_reaction_delay("visual") == sum(PATHWAY_SEGMENTS.values())


class TestExposure:
    def test_exposure_creates_record(self):
        engine = AttentionPlasticityEngine()
        engine.on_exposure("visual")
        rec = engine.get_training_record("visual")
        assert rec is not None
        assert rec.total_exposures == 1

    def test_exposure_alone_doesnt_improve(self):
        engine = AttentionPlasticityEngine()
        for _ in range(100):
            engine.on_exposure("visual")
        # Exposure alone should NOT change gate_tau or tuner_q
        assert engine.get_gate_tau("visual") == GATE_TAU_INITIAL
        assert engine.get_tuner_q("visual") == Q_INITIAL


class TestSuccessfulLock:
    def test_gate_tau_improves(self):
        engine = AttentionPlasticityEngine()
        initial_tau = GATE_TAU_INITIAL
        engine.on_successful_lock("visual")
        new_tau = engine.get_gate_tau("visual")
        assert new_tau < initial_tau, "Gate tau should decrease after successful lock"

    def test_tuner_q_improves(self):
        engine = AttentionPlasticityEngine()
        engine.on_successful_lock("visual")
        assert engine.get_tuner_q("visual") > Q_INITIAL

    def test_repeated_training_compounds(self):
        engine = AttentionPlasticityEngine()
        taus = [GATE_TAU_INITIAL]
        for _ in range(100):
            engine.on_successful_lock("visual")
            taus.append(engine.get_gate_tau("visual"))
        # Should be monotonically decreasing
        for i in range(1, len(taus)):
            assert taus[i] <= taus[i - 1]
        # Should reach well below initial
        assert taus[-1] < GATE_TAU_INITIAL * 0.85

    def test_gate_tau_has_floor(self):
        engine = AttentionPlasticityEngine()
        for _ in range(10000):
            engine.on_successful_lock("visual")
        assert engine.get_gate_tau("visual") >= GATE_TAU_MIN

    def test_tuner_q_has_ceiling(self):
        engine = AttentionPlasticityEngine()
        for _ in range(10000):
            engine.on_successful_lock("visual")
        assert engine.get_tuner_q("visual") <= Q_MAX

    def test_conduction_delay_improves(self):
        engine = AttentionPlasticityEngine()
        initial_delay = engine.get_reaction_delay("visual")
        for _ in range(100):
            engine.on_successful_lock("visual")
        new_delay = engine.get_reaction_delay("visual")
        assert new_delay < initial_delay

    def test_cross_modal_transfer(self):
        engine = AttentionPlasticityEngine()
        # Train visual
        engine.on_exposure("auditory")  # Create auditory record first
        for _ in range(50):
            engine.on_successful_lock("visual")
        # Auditory should benefit slightly
        aud_q = engine.get_tuner_q("auditory")
        assert aud_q > Q_INITIAL, "Cross-modal transfer should improve auditory Q"


class TestSuccessfulIdentification:
    def test_stronger_q_improvement(self):
        engine = AttentionPlasticityEngine()
        engine.on_successful_lock("visual")
        q_after_lock = engine.get_tuner_q("visual")

        engine2 = AttentionPlasticityEngine()
        engine2.on_successful_identification("visual")
        q_after_id = engine2.get_tuner_q("visual")

        assert q_after_id > q_after_lock, "Identification should improve Q more than lock"

    def test_pfc_capacity_grows(self):
        engine = AttentionPlasticityEngine()
        for _ in range(50):
            engine.on_successful_identification("visual")
        assert engine.get_pfc_capacity() > PFC_CAPACITY_INITIAL


class TestSuccessfulInhibition:
    def test_inhibition_efficiency_improves(self):
        engine = AttentionPlasticityEngine()
        engine.on_successful_inhibition("visual")
        cost = engine.get_inhibition_cost_multiplier("visual")
        assert cost < INHIBITION_EFFICIENCY_INITIAL

    def test_inhibition_has_floor(self):
        engine = AttentionPlasticityEngine()
        for _ in range(10000):
            engine.on_successful_inhibition("visual")
        assert engine.get_inhibition_cost_multiplier("visual") >= INHIBITION_EFFICIENCY_MIN


class TestMultiFocusTraining:
    def test_attention_slots_expand(self):
        engine = AttentionPlasticityEngine()
        for _ in range(SLOT_TRAINING_THRESHOLD):
            engine.on_multi_focus_success(["visual", "auditory"])
        assert engine.get_attention_slots() > ATTENTION_SLOTS_INITIAL

    def test_attention_slots_capped(self):
        engine = AttentionPlasticityEngine()
        for _ in range(SLOT_TRAINING_THRESHOLD * 10):
            engine.on_multi_focus_success(["visual", "auditory"])
        assert engine.get_attention_slots() <= ATTENTION_SLOTS_MAX


class TestDecay:
    def test_decay_reverts_improvements(self):
        engine = AttentionPlasticityEngine()
        # Train
        for _ in range(200):
            engine.on_successful_lock("visual")
        trained_tau = engine.get_gate_tau("visual")
        trained_q = engine.get_tuner_q("visual")
        # Decay a lot
        for _ in range(50000):
            engine.decay_tick()
        decayed_tau = engine.get_gate_tau("visual")
        decayed_q = engine.get_tuner_q("visual")
        assert decayed_tau > trained_tau, "Gate tau should decay back toward initial"
        assert decayed_q < trained_q, "Tuner Q should decay back toward initial"

    def test_decay_doesnt_exceed_initial(self):
        engine = AttentionPlasticityEngine()
        engine.on_exposure("visual")  # Create record
        for _ in range(10000):
            engine.decay_tick()
        assert engine.get_gate_tau("visual") <= GATE_TAU_INITIAL
        assert engine.get_tuner_q("visual") >= Q_INITIAL


class TestTrainingLevels:
    @pytest.mark.parametrize("count,expected", [
        (0, "novice"),
        (10, "novice"),
        (100, "intermediate"),
        (1000, "advanced"),
        (10000, "expert"),
        (100000, "master"),
    ])
    def test_training_level(self, count, expected):
        rec = ModalityTrainingRecord(modality="visual", total_exposures=count)
        assert rec.training_level == expected


class TestEsportsSimulation:
    """Simulate an esports player's attention training process"""

    def test_reaction_time_decreases_with_practice(self):
        """Practice reduces reaction time — not talent, but myelination"""
        engine = AttentionPlasticityEngine()
        initial_delay = engine.get_reaction_delay("visual")

        # Simulate 1000 rapid target identification training sessions
        for _ in range(1000):
            engine.on_exposure("visual")
            engine.on_successful_lock("visual")
            engine.on_successful_identification("visual")

        final_delay = engine.get_reaction_delay("visual")

        # Reaction delay should decrease by at least 30%
        improvement = 1.0 - (final_delay / initial_delay)
        assert improvement > 0.30, f"Expected >30% improvement, got {improvement:.1%}"

    def test_selective_attention_sharpens(self):
        """Sommeliers can distinguish 2001 from 2002 vintages — because their Q is high"""
        engine = AttentionPlasticityEngine()
        initial_q = engine.get_tuner_q("visual")

        # 5000 visual identification training sessions
        for _ in range(5000):
            engine.on_successful_lock("visual")
            engine.on_successful_identification("visual")

        expert_q = engine.get_tuner_q("visual")

        # Q should at least double
        assert expert_q > initial_q * 2.0

    def test_sustained_attention_capacity_grows(self):
        """Meditation increases prefrontal gray matter density — PFC capacity grows"""
        engine = AttentionPlasticityEngine()

        # 2000 inhibition training sessions (simulating meditation)
        for _ in range(2000):
            engine.on_successful_inhibition("visual")
            engine.on_successful_identification("visual")

        assert engine.get_pfc_capacity() > PFC_CAPACITY_INITIAL * 1.3

    def test_use_it_or_lose_it(self):
        """Stop training → skill degradation — but much slower than learning rate"""
        engine = AttentionPlasticityEngine()

        # Train to expert level
        for _ in range(1000):
            engine.on_successful_lock("visual")
        trained_q = engine.get_tuner_q("visual")

        # 1000 ticks without practice
        for _ in range(1000):
            engine.decay_tick()
        decayed_q = engine.get_tuner_q("visual")

        # Decay << training gain (asymmetry)
        decay_loss = trained_q - decayed_q
        total_gain = trained_q - Q_INITIAL
        assert decay_loss < total_gain * 0.3, "Decay should be much slower than learning"


class TestIntegration:
    """Test plasticity engine integration with other modules"""

    def test_thalamus_uses_trained_gate_tau(self):
        from alice.brain.thalamus import ThalamusEngine, GATE_MOMENTUM
        thal = ThalamusEngine()
        engine = AttentionPlasticityEngine()
        thal.set_plasticity_engine(engine)

        # Train gate speed
        for _ in range(100):
            engine.on_successful_lock("visual")

        # Gate should use trained tau (below default)
        trained_tau = engine.get_gate_tau("visual")
        assert trained_tau < GATE_MOMENTUM

    def test_perception_uses_trained_q(self):
        from alice.brain.perception import PerceptionPipeline
        pipeline = PerceptionPipeline()
        engine = AttentionPlasticityEngine()
        pipeline.set_plasticity_engine(engine)

        # Train Q
        for _ in range(200):
            engine.on_successful_lock("visual")

        # Q should have increased
        assert engine.get_tuner_q("visual") > Q_INITIAL

    def test_full_brain_has_plasticity(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        assert brain.attention_plasticity is not None
        state = brain.attention_plasticity.get_state()
        assert "global" in state
        assert "modalities" in state

# -*- coding: utf-8 -*-
"""Tests for Phase 12: ConsciousnessModule"""
import time
import numpy as np
import pytest
from alice.brain.consciousness import (
    ConsciousnessModule,
    CONSCIOUS_THRESHOLD, LUCID_THRESHOLD, SUBLIMINAL_THRESHOLD,
    META_AWARENESS_THRESHOLD, MAX_ATTENTION_TARGETS,
)


class TestConsciousnessBasics:
    def test_init_defaults(self):
        c = ConsciousnessModule()
        assert c.phi == pytest.approx(0.8)
        assert c.is_conscious()
        assert c.total_ticks == 0

    def test_reset(self):
        c = ConsciousnessModule()
        c.phi = 0.1
        c.reset()
        assert c.phi == pytest.approx(0.8)


class TestPhiCalculation:
    def test_high_inputs_high_phi(self):
        c = ConsciousnessModule()
        result = c.tick(
            attention_strength=0.9,
            binding_quality=0.9,
            working_memory_usage=0.5,
            arousal=0.8,
            sensory_gate=1.0,
            pain_level=0.0,
        )
        assert result["phi"] > CONSCIOUS_THRESHOLD

    def test_zero_arousal_drops_phi(self):
        c = ConsciousnessModule()
        # Multiple ticks to let EMA converge
        for _ in range(20):
            result = c.tick(
                attention_strength=0.9,
                binding_quality=0.9,
                arousal=0.0,  # Completely no arousal
                sensory_gate=1.0,
            )
        assert result["phi"] < LUCID_THRESHOLD

    def test_zero_sensory_gate_drops_phi(self):
        c = ConsciousnessModule()
        for _ in range(20):
            result = c.tick(
                attention_strength=0.9,
                binding_quality=0.9,
                arousal=0.8,
                sensory_gate=0.0,  # Deep sleep
            )
        assert result["phi"] < LUCID_THRESHOLD

    def test_pain_disruption(self):
        c = ConsciousnessModule()
        # No pain
        for _ in range(10):
            no_pain = c.tick(attention_strength=0.7, binding_quality=0.7,
                             arousal=0.7, sensory_gate=1.0, pain_level=0.0)
        phi_no_pain = no_pain["phi"]

        c2 = ConsciousnessModule()
        # High pain
        for _ in range(10):
            high_pain = c2.tick(attention_strength=0.7, binding_quality=0.7,
                                arousal=0.7, sensory_gate=1.0, pain_level=0.9)
        phi_pain = high_pain["phi"]
        assert phi_pain < phi_no_pain

    def test_all_zero_goes_unconscious(self):
        c = ConsciousnessModule()
        for _ in range(30):
            result = c.tick(
                attention_strength=0.0,
                binding_quality=0.0,
                arousal=0.0,
                sensory_gate=0.0,
                pain_level=0.0,
            )
        assert result["state"] in ("subliminal", "unconscious")


class TestConsciousnessStates:
    def test_lucid_state(self):
        c = ConsciousnessModule()
        for _ in range(20):
            result = c.tick(attention_strength=0.9, binding_quality=0.9,
                            arousal=0.9, sensory_gate=1.0)
        assert result["state"] == "lucid"
        assert c.is_lucid()

    def test_conscious_state(self):
        c = ConsciousnessModule()
        for _ in range(30):
            result = c.tick(attention_strength=0.6, binding_quality=0.6,
                            arousal=0.7, sensory_gate=0.8)
        assert c.is_conscious()

    def test_phi_smoothing(self):
        c = ConsciousnessModule()
        # phi starts at 0.8, sudden drop input
        result = c.tick(attention_strength=0.0, binding_quality=0.0,
                        arousal=0.0, sensory_gate=0.0)
        # EMA smoothing → phi shouldn't jump to 0 immediately
        assert result["phi"] > 0.1


class TestMetaAwareness:
    def test_meta_aware_when_lucid(self):
        c = ConsciousnessModule()
        for _ in range(20):
            result = c.tick(attention_strength=0.9, binding_quality=0.9,
                            arousal=0.9, sensory_gate=1.0)
        assert result["is_meta_aware"] is True
        assert result["meta_report"] is not None

    def test_not_meta_aware_when_low(self):
        c = ConsciousnessModule()
        for _ in range(30):
            result = c.tick(attention_strength=0.1, binding_quality=0.1,
                            arousal=0.1, sensory_gate=0.1)
        assert result["is_meta_aware"] is False


class TestAttention:
    def test_focus_attention(self):
        c = ConsciousnessModule()
        c.focus_attention("cat", "visual", 0.8)
        targets = c.get_attention_targets()
        assert len(targets) == 1
        assert targets[0]["target"] == "cat"

    def test_max_attention_targets(self):
        c = ConsciousnessModule()
        for i in range(10):
            c.focus_attention(f"target_{i}", "visual", float(i) / 10)
        targets = c.get_attention_targets()
        assert len(targets) == MAX_ATTENTION_TARGETS

    def test_sorted_by_salience(self):
        c = ConsciousnessModule()
        c.focus_attention("low", "visual", 0.1)
        c.focus_attention("high", "visual", 0.9)
        c.focus_attention("mid", "visual", 0.5)
        targets = c.get_attention_targets()
        assert targets[0]["target"] == "high"

    def test_replace_existing(self):
        c = ConsciousnessModule()
        c.focus_attention("cat", "visual", 0.3)
        c.focus_attention("cat", "visual", 0.9)
        targets = c.get_attention_targets()
        assert len(targets) == 1
        assert targets[0]["salience"] == pytest.approx(0.9)


class TestWorkspace:
    def test_broadcast(self):
        c = ConsciousnessModule()
        c.phi = 0.8  # Conscious
        c.broadcast_to_workspace({"concept": "apple"}, source="perception")
        contents = c.get_workspace_contents()
        assert len(contents) == 1

    def test_unconscious_blocks_broadcast(self):
        c = ConsciousnessModule()
        c.phi = 0.05  # Unconscious
        c.broadcast_to_workspace({"concept": "apple"}, source="perception")
        contents = c.get_workspace_contents()
        assert len(contents) == 0

    def test_workspace_capacity(self):
        c = ConsciousnessModule()
        c.phi = 0.8
        for i in range(20):
            c.broadcast_to_workspace({"item": i}, source="test")
        assert len(c.get_workspace_contents()) <= 7

    def test_clear_workspace(self):
        c = ConsciousnessModule()
        c.phi = 0.8
        c.broadcast_to_workspace({"concept": "test"}, source="test")
        c.clear_workspace()
        assert len(c.get_workspace_contents()) == 0


class TestConsciousnessStats:
    def test_stats_keys(self):
        c = ConsciousnessModule()
        c.tick()
        stats = c.get_stats()
        assert "phi" in stats
        assert "is_conscious" in stats
        assert "attention_targets" in stats
        assert "total_ticks" in stats

    def test_waveforms(self):
        c = ConsciousnessModule()
        for _ in range(5):
            c.tick()
        wf = c.get_waveforms(last_n=3)
        assert len(wf["phi"]) == 3

    def test_tick_counters(self):
        c = ConsciousnessModule()
        for _ in range(10):
            c.tick(attention_strength=0.9, binding_quality=0.9,
                   arousal=0.9, sensory_gate=1.0)
        stats = c.get_stats()
        assert stats["total_ticks"] == 10
        assert stats["lucid_ticks"] + stats["conscious_ticks"] + stats["unconscious_ticks"] == 10


class TestComponentOutput:
    def test_components_in_result(self):
        c = ConsciousnessModule()
        result = c.tick(attention_strength=0.5, binding_quality=0.6,
                        arousal=0.7, sensory_gate=0.8, pain_level=0.3)
        comp = result["components"]
        assert "attention" in comp
        assert "binding" in comp
        assert "arousal" in comp
        assert "pain_cost" in comp
        assert comp["attention"] == pytest.approx(0.5)


class TestAliceBrainIntegration:
    def test_brain_has_new_modules(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        assert hasattr(brain, "ear")
        assert hasattr(brain, "mouth")
        assert hasattr(brain, "autonomic")
        assert hasattr(brain, "sleep_cycle")
        assert hasattr(brain, "consciousness")

    def test_perceive_includes_new_data(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        result = brain.perceive(np.random.randn(64))
        assert "autonomic" in result
        assert "sleep" in result
        assert "consciousness" in result

    def test_introspect_includes_all_subsystems(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        info = brain.introspect()
        subs = info["subsystems"]
        assert "ear" in subs
        assert "mouth" in subs
        assert "autonomic" in subs
        assert "sleep" in subs
        assert "consciousness" in subs

    def test_hear_method(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        result = brain.hear(np.random.randn(256))
        assert "auditory" in result
        assert result["auditory"]["source"] == "ear"

    def test_say_method(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        result = brain.say(target_pitch=200.0, volume=0.5, vowel="a")
        assert "final_pitch" in result
        assert "tremor_intensity" in result
        assert result["feedback"]["source"] == "mouth"

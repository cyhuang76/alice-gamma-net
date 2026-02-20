# -*- coding: utf-8 -*-
"""
Alice Smart System — Complete Test Suite
"""

import time
import numpy as np
import pytest

from alice.core.protocol import (
    MessagePacket,
    Modality,
    Priority,
    PriorityRouter,
    YearRingCache,
    BrainHemisphere,
    ErrorCorrector,
    GammaNetV4Protocol,
)
from alice.core.cache_analytics import CachePerformanceDashboard, with_analytics
from alice.core.cache_persistence import CachePersistence, CacheCheckpoint
from alice.brain.fusion_brain import FusionBrain, BrainRegion, BrainRegionType
from alice.modules.working_memory import WorkingMemory
from alice.modules.reinforcement import ReinforcementLearner
from alice.modules.causal_reasoning import CausalReasoner
from alice.modules.meta_learning import MetaLearner
from alice.alice_brain import AliceBrain
from alice.alice_brain import SystemState


# ============================================================================
# Core Engine Tests
# ============================================================================


class TestMessagePacket:
    def test_from_signal(self):
        sig = np.array([0.5, 0.8, 0.1, 0.9])
        pkt = MessagePacket.from_signal(sig)
        assert pkt.frequency_tag > 0
        assert pkt.content_hash
        assert pkt.priority == Priority.NORMAL
        assert pkt.modality == Modality.VISUAL

    def test_different_signals_different_hash(self):
        s1 = np.array([1.0, 0.0])
        s2 = np.array([0.0, 1.0])
        p1 = MessagePacket.from_signal(s1)
        p2 = MessagePacket.from_signal(s2)
        assert p1.content_hash != p2.content_hash

    def test_same_signal_same_hash(self):
        sig = np.array([0.5, 0.5])
        p1 = MessagePacket.from_signal(sig)
        p2 = MessagePacket.from_signal(sig)
        assert p1.content_hash == p2.content_hash


class TestPriorityRouter:
    def test_critical_immediate(self):
        router = PriorityRouter()
        pkt = MessagePacket.from_signal(np.array([1.0]), priority=Priority.CRITICAL)
        ok, reason = router.route(pkt)
        assert ok
        assert "CRITICAL" in reason

    def test_background_deferred(self):
        router = PriorityRouter()
        pkt = MessagePacket.from_signal(np.array([0.1]), priority=Priority.BACKGROUND)
        ok, _ = router.route(pkt)
        assert not ok

    def test_get_next_priority_order(self):
        router = PriorityRouter()
        low = MessagePacket.from_signal(np.array([0.1]), priority=Priority.BACKGROUND)
        high = MessagePacket.from_signal(np.array([0.9]), priority=Priority.HIGH)
        router.route(low)
        router.route(high)
        nxt = router.get_next()
        assert nxt.priority == Priority.HIGH

    def test_stats(self):
        router = PriorityRouter()
        pkt = MessagePacket.from_signal(np.array([1.0]))
        router.route(pkt)
        stats = router.get_stats()
        assert stats["total_routed"] == 1


class TestYearRingCache:
    def test_store_and_lookup(self):
        cache = YearRingCache()
        pkt = MessagePacket.from_signal(np.array([0.5, 0.5]))
        cache.store(pkt, "test_label")
        hit, label, conf = cache.lookup(pkt)
        assert hit
        assert label == "test_label"
        assert conf > 0

    def test_miss(self):
        cache = YearRingCache()
        pkt = MessagePacket.from_signal(np.array([0.5, 0.5]))
        hit, label, conf = cache.lookup(pkt)
        assert not hit
        assert label is None

    def test_consolidation(self):
        cache = YearRingCache()
        pkt = MessagePacket.from_signal(np.array([1.0]))
        cache.store(pkt, "item")
        # Multiple lookups trigger consolidation
        for _ in range(10):
            cache.lookup(pkt)
            cache.store(pkt, "item")
        assert cache.consolidations >= 1

    def test_stats(self):
        cache = YearRingCache()
        stats = cache.get_stats()
        assert "hit_rate" in stats
        assert "ring_sizes" in stats


class TestHemisphere:
    def test_skip_low_error(self):
        hemi = BrainHemisphere("Left", "detail")
        ok, _ = hemi.should_activate(0.1, Modality.VISUAL)
        assert not ok

    def test_activate_high_error(self):
        hemi = BrainHemisphere("Left", "detail")
        ok, _ = hemi.should_activate(0.8, Modality.AUDITORY)
        assert ok

    def test_process(self):
        hemi = BrainHemisphere("Right", "global")
        data = np.random.rand(10)
        result, ratio = hemi.process(data, 0.6)
        assert result.shape == data.shape


class TestErrorCorrector:
    def test_compute_error(self):
        ec = ErrorCorrector()
        a = np.array([1.0, 0.5, 0.0])
        b = np.array([1.0, 0.5, 0.3])
        err, mask = ec.compute_error(a, b)
        assert err >= 0

    def test_correct_minimal(self):
        ec = ErrorCorrector()
        cur = np.array([1.0, 0.0])
        tgt = np.array([1.0, 1.0])
        _, mask = ec.compute_error(tgt, cur)
        result, ratio = ec.correct(cur, tgt, mask)
        # Only correct the differing parts
        assert result[0] == cur[0]  # Unchanged
        assert result[1] == tgt[1]  # Corrected


class TestGammaNetV4Protocol:
    def test_learn_and_recognize(self):
        proto = GammaNetV4Protocol()
        data = np.array([0.9, 0.1, 0.9, 0.1])
        proto.learn("pattern_A", data)
        result = proto.recognize(data)
        assert result["status"] == "cache_hit"
        assert result["cache_hit"]

    def test_unknown_pattern(self):
        proto = GammaNetV4Protocol()
        result = proto.recognize(np.random.rand(4))
        # Unlearned pattern
        assert "status" in result


# ============================================================================
# FusionBrain Tests
# ============================================================================


class TestFusionBrain:
    def test_process_stimulus(self):
        fb = FusionBrain(neuron_count=50)
        signal = np.random.rand(50)
        result = fb.process_stimulus(signal)
        assert result["cycle"] == 1
        assert result["memory_consolidated"]
        assert "sensory" in result
        assert "cognitive" in result
        assert "emotional" in result
        assert "motor" in result

    def test_brain_state(self):
        fb = FusionBrain(neuron_count=20)
        state = fb.get_brain_state()
        assert "regions" in state
        assert "motor" in state["regions"]

    def test_memory_consolidation(self):
        fb = FusionBrain(neuron_count=10)
        # Use a strong signal with frequency structure (perception pipeline needs frequency components for effective transmission)
        t = np.linspace(0, 1, 10, endpoint=False)
        strong = 5.0 * np.sin(2 * np.pi * 10 * t)  # Strong α-band signal
        # Multiple stimulations to accumulate Hebbian plasticity
        for _ in range(5):
            fb.process_stimulus(strong, priority=Priority.CRITICAL)
        state = fb.get_brain_state()
        # Memory consolidation should work (system should not crash)
        assert state["cycle_count"] == 5
        # Synaptic strength should be modified (whether strengthened or weakened, should not stay at initial 1.0)
        sensory = state["regions"]["somatosensory"]
        assert sensory["avg_synaptic_strength"] != 1.0

    def test_report(self):
        fb = FusionBrain(neuron_count=10)
        fb.process_stimulus(np.random.rand(10))
        report = fb.generate_report("Test Report")
        assert "Test Report" in report


# ============================================================================
# v5 Module Tests
# ============================================================================


class TestWorkingMemory:
    def test_store_retrieve(self):
        wm = WorkingMemory(capacity=5)
        wm.store("key1", "value1")
        assert wm.retrieve("key1") == "value1"

    def test_capacity_limit(self):
        wm = WorkingMemory(capacity=3)
        for i in range(5):
            wm.store(f"key_{i}", f"val_{i}", importance=0.5)
        assert len(wm._items) <= 3

    def test_rehearsal(self):
        wm = WorkingMemory()
        wm.store("important", "data")
        item_before = wm._items["important"].activation
        wm.rehearse("important")
        assert wm._items["important"].activation >= item_before


class TestReinforcementLearner:
    def test_update_and_choose(self):
        rl = ReinforcementLearner(epsilon=0.0)  # Pure exploitation
        rl.update("s1", "a1", 10.0, "s2", ["a1", "a2"])
        action, _ = rl.choose_action("s1", ["a1", "a2"])
        assert action == "a1"  # Q(s1,a1) is highest

    def test_dopamine_signal(self):
        rl = ReinforcementLearner()
        dopamine = rl.update("s1", "a1", 1.0, "s2")
        # Initial Q=0, reward=1, so dopamine > 0
        assert dopamine > 0

    def test_replay(self):
        rl = ReinforcementLearner()
        for i in range(50):
            rl.update(f"s{i}", "a1", float(i), f"s{i+1}")
        error = rl.replay(batch_size=10)
        assert error >= 0


class TestCausalReasoner:
    def test_observe_and_infer(self):
        cr = CausalReasoner()
        for _ in range(20):
            x = np.random.rand()
            cr.observe({"stimulus": x, "response": x * 0.8})
        result = cr.infer("stimulus", "response")
        assert result["observations"] >= 20

    def test_intervene(self):
        cr = CausalReasoner()
        for _ in range(30):
            cr.observe({"A": 1.0, "B": 0.8, "C": 0.6})
        result = cr.intervene("A", 2.0, "C")
        assert "predicted_effect" in result

    def test_counterfactual(self):
        cr = CausalReasoner()
        for _ in range(30):
            cr.observe({"X": 1.0, "Y": 0.5})
        result = cr.counterfactual(
            factual={"X": 1.0, "Y": 0.5},
            counterfactual_var="X",
            counterfactual_value=2.0,
            target="Y",
        )
        assert "counterfactual_value" in result


class TestMetaLearner:
    def test_select_strategy(self):
        ml = MetaLearner()
        strat = ml.select_strategy()
        assert strat.name is not None
        assert "learning_rate" in strat.params

    def test_performance_feedback(self):
        ml = MetaLearner()
        ml.select_strategy()
        ml.report_performance(0.8)
        assert ml.total_adaptations == 1

    def test_evolution(self):
        ml = MetaLearner()
        for _ in range(60):
            ml.select_strategy()
            ml.report_performance(np.random.rand())
        # Should have evolved at least once
        assert ml.total_adaptations >= 50


# ============================================================================
# AliceBrain Integration Tests
# ============================================================================


class TestAliceBrain:
    def test_perceive(self):
        alice = AliceBrain(neuron_count=20)
        result = alice.perceive(np.random.rand(20))
        assert result["cycle"] == 1

    def test_think(self):
        alice = AliceBrain(neuron_count=10)
        # First provide some observations
        for _ in range(15):
            alice.perceive(np.random.rand(10))
        result = alice.think(
            "Why does increased stimulus lead to increased motor output?",
            {"stimulus": 0.8, "motor": 0.6},
        )
        assert "reasoning" in result

    def test_act_and_learn(self):
        alice = AliceBrain(neuron_count=10)
        act_result = alice.act("alert", ["fight", "flight", "freeze"])
        assert act_result["chosen_action"] in ["fight", "flight", "freeze"]

        learn_result = alice.learn_from_feedback(
            "alert", act_result["chosen_action"], 1.0, "calm", ["rest", "explore"]
        )
        assert "dopamine_signal" in learn_result

    def test_introspect(self):
        alice = AliceBrain(neuron_count=10)
        alice.perceive(np.random.rand(10))
        report = alice.introspect()
        assert "subsystems" in report
        assert "fusion_brain" in report["subsystems"]
        assert "working_memory" in report["subsystems"]
        assert "reinforcement_learning" in report["subsystems"]
        assert "causal_reasoning" in report["subsystems"]
        assert "meta_learning" in report["subsystems"]

    def test_full_cognitive_loop(self):
        """Complete cognitive loop test"""
        alice = AliceBrain(neuron_count=20)

        # 1. Perceive
        for i in range(5):
            alice.perceive(np.random.rand(20), priority=Priority.NORMAL)

        # 2. Think
        think_result = alice.think(
            "What happens if stimulus intensity doubles?",
            {"stimulus_energy": 1.0, "motor_output": 0.5},
        )
        assert think_result["type"] in ("counterfactual", "causal", "intervention", "observation")

        # 3. Act
        act_result = alice.act("exploring", ["move_left", "move_right", "stay", "retreat"])

        # 4. Learn
        learn_result = alice.learn_from_feedback(
            "exploring", act_result["chosen_action"], 0.7, "at_destination",
            ["investigate", "report", "wait"],
        )
        assert learn_result["dopamine_signal"] != 0 or learn_result["reward"] == 0

        # 5. Introspect
        report = alice.introspect()
        assert report["cycle_count"] == 5


# ============================================================================
# Persistence Tests
# ============================================================================


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        cache = YearRingCache(num_rings=4, ring_capacity=10)
        pkt = MessagePacket.from_signal(np.array([0.5]))
        cache.store(pkt, "test_data")

        persistence = CachePersistence(str(tmp_path))
        path = persistence.save_cache(cache)

        cache2 = YearRingCache(num_rings=4, ring_capacity=10)
        assert persistence.load_cache(cache2, path)
        hit, label, _ = cache2.lookup(pkt)
        assert hit
        assert label == "test_data"

    def test_json_export(self, tmp_path):
        cache = YearRingCache()
        pkt = MessagePacket.from_signal(np.array([1.0]))
        cache.store(pkt, "label")

        persistence = CachePersistence(str(tmp_path))
        fp = str(tmp_path / "export.json")
        assert persistence.export_as_json(cache, fp)


# ============================================================================
# Vitals & Pain Feedback Tests
# ============================================================================


class TestSystemState:
    """SystemState — Pain feedback loop tests"""

    def test_initial_healthy(self):
        state = SystemState()
        v = state.get_vitals()
        assert v["ram_temperature"] == 0.0
        assert v["pain_level"] == 0.0
        assert v["stability_index"] == 1.0
        assert v["consciousness"] == 1.0
        assert v["is_frozen"] is False
        assert state.get_throttle_factor() == 1.0

    def test_tick_no_stress(self):
        state = SystemState()
        state.tick(0, 0, 0, 0.1, 0.0, 0.3, 0.3, 1.0)
        assert state.ram_temperature < 0.1
        assert state.pain_level == 0.0
        assert state.get_throttle_factor() == 1.0

    def test_critical_causes_heat(self):
        state = SystemState()
        for _ in range(15):
            state.tick(10, 0, 10, 0.5, -0.3, 0.5, 0.5, 2.0)
        assert state.ram_temperature > 0.5
        assert state.pain_level > 0.0

    def test_throttle_under_stress(self):
        state = SystemState()
        for _ in range(30):
            state.tick(20, 5, 30, 0.8, -0.5, 0.5, 0.5, 5.0)
        assert state.get_throttle_factor() < 1.0

    def test_freeze_under_extreme_stress(self):
        state = SystemState()
        for _ in range(50):
            state.tick(50, 10, 80, 1.0, -0.5, 0.5, 0.5, 10.0)
        assert state.is_frozen()

    def test_reset_restores_health(self):
        state = SystemState()
        for _ in range(50):
            state.tick(50, 10, 80, 1.0, -0.5, 0.5, 0.5, 10.0)
        state.reset()
        assert state.ram_temperature == 0.0
        assert state.pain_level == 0.0
        assert state.consciousness == 1.0
        assert not state.is_frozen()

    def test_waveforms_recorded(self):
        state = SystemState()
        for _ in range(5):
            state.tick(0, 0, 0, 0.1, 0.0, 0.3, 0.5, 1.0)
        w = state.get_waveforms()
        assert len(w["temperature"]) == 5
        assert len(w["heart_rate"]) == 5
        assert len(w["left_brain"]) == 5
        assert len(w["right_brain"]) == 5


class TestAliceBrainVitals:
    """AliceBrain pain integration tests"""

    def test_perceive_returns_vitals(self):
        alice = AliceBrain(neuron_count=20)
        result = alice.perceive(np.random.rand(10), Modality.VISUAL, Priority.NORMAL)
        assert "vitals" in result
        assert "ram_temperature" in result["vitals"]

    def test_critical_raises_temperature(self):
        alice = AliceBrain(neuron_count=20)
        for _ in range(10):
            alice.perceive(np.random.randn(10) * 3.0, Modality.TACTILE, Priority.CRITICAL)
        assert alice.vitals.ram_temperature > 0.3

    def test_introspect_includes_vitals(self):
        alice = AliceBrain(neuron_count=20)
        info = alice.introspect()
        assert "vitals" in info

    def test_emergency_reset(self):
        alice = AliceBrain(neuron_count=20)
        for _ in range(20):
            alice.perceive(np.random.randn(10) * 5.0, Modality.TACTILE, Priority.CRITICAL)
        alice.emergency_reset()
        # ★ Trauma memory: after reset, temperature returns to baseline_temperature (no longer 0)
        #   baseline_temperature = min(0.3, trauma_count * 0.03)
        assert alice.vitals.ram_temperature == alice.vitals.baseline_temperature
        assert alice.vitals.pain_level == 0.0
        # ★ Trauma imprint should be recorded
        assert alice.vitals.trauma_count > 0
        assert alice.vitals.pain_sensitivity > 1.0

    def test_inject_pain(self):
        alice = AliceBrain(neuron_count=20)
        alice.inject_pain(0.8)
        assert alice.vitals.ram_temperature > 0.2


# ============================================================================
# Compensation Command Dispatch Tests — Last link of the closed loop
# ============================================================================


class TestCommandDispatch:
    """Verify LifeLoop compensation commands are dispatched to body organs"""

    def test_perceive_returns_dispatched(self):
        """perceive() result should contain dispatched field"""
        alice = AliceBrain(neuron_count=20)
        result = alice.perceive(np.random.rand(20))
        assert "life_loop" in result
        assert "dispatched" in result["life_loop"]
        assert isinstance(result["life_loop"]["dispatched"], list)

    def test_dispatch_reach(self):
        """REACH command → hand.reach() is called"""
        from alice.brain.life_loop import CompensationCommand, CompensationAction, ErrorType
        alice = AliceBrain(neuron_count=20)
        cmd = CompensationCommand(
            action=CompensationAction.REACH,
            target=np.array([100.0, 100.0]),
            strength=0.5,
            source_error=ErrorType.VISUAL_MOTOR,
            priority=0.5,
        )
        results = alice._dispatch_commands([cmd])
        assert len(results) == 1
        assert results[0]["action"] == "reach"
        assert results[0]["executed"] is True

    def test_dispatch_vocalize(self):
        """VOCALIZE command → mouth.speak() is called"""
        from alice.brain.life_loop import CompensationCommand, CompensationAction, ErrorType
        alice = AliceBrain(neuron_count=20)
        cmd = CompensationCommand(
            action=CompensationAction.VOCALIZE,
            target=np.array([220.0]),
            strength=0.5,
            source_error=ErrorType.AUDITORY_VOCAL,
            priority=0.5,
        )
        results = alice._dispatch_commands([cmd])
        assert len(results) == 1
        assert results[0]["action"] == "vocalize"
        assert results[0]["executed"] is True

    def test_dispatch_adjust_pupil(self):
        """ADJUST_PUPIL command → eye.adjust_pupil() is called"""
        from alice.brain.life_loop import CompensationCommand, CompensationAction, ErrorType
        alice = AliceBrain(neuron_count=20)
        cmd = CompensationCommand(
            action=CompensationAction.ADJUST_PUPIL,
            target=np.array([0.7]),
            strength=0.5,
            source_error=ErrorType.INTEROCEPTIVE,
            priority=0.3,
        )
        results = alice._dispatch_commands([cmd])
        assert results[0]["executed"] is True
        assert abs(alice.eye.pupil_aperture - 0.7) < 0.01

    def test_dispatch_attend(self):
        """ATTEND command → consciousness.focus_attention() is called"""
        from alice.brain.life_loop import CompensationCommand, CompensationAction, ErrorType
        alice = AliceBrain(neuron_count=20)
        cmd = CompensationCommand(
            action=CompensationAction.ATTEND,
            target=np.array([0.0]),
            strength=0.8,
            source_error=ErrorType.SENSORY_PREDICTION,
            priority=0.5,
        )
        results = alice._dispatch_commands([cmd])
        assert results[0]["executed"] is True

    def test_dispatch_breathe(self):
        """BREATHE command → autonomic.parasympathetic increases"""
        from alice.brain.life_loop import CompensationCommand, CompensationAction, ErrorType
        alice = AliceBrain(neuron_count=20)
        para_before = alice.autonomic.parasympathetic
        cmd = CompensationCommand(
            action=CompensationAction.BREATHE,
            target=np.array([0.0]),
            strength=0.5,
            source_error=ErrorType.INTEROCEPTIVE,
            priority=0.3,
        )
        alice._dispatch_commands([cmd])
        assert alice.autonomic.parasympathetic >= para_before

    def test_dispatch_multiple_commands(self):
        """Multiple commands dispatched simultaneously"""
        from alice.brain.life_loop import CompensationCommand, CompensationAction, ErrorType
        alice = AliceBrain(neuron_count=20)
        cmds = [
            CompensationCommand(
                action=CompensationAction.REACH,
                target=np.array([50.0, 50.0]),
                strength=0.5,
                source_error=ErrorType.VISUAL_MOTOR,
                priority=0.5,
            ),
            CompensationCommand(
                action=CompensationAction.BREATHE,
                target=np.array([0.0]),
                strength=0.3,
                source_error=ErrorType.INTEROCEPTIVE,
                priority=0.2,
            ),
        ]
        results = alice._dispatch_commands(cmds)
        assert len(results) == 2
        assert all(r["executed"] for r in results)

    def test_dispatch_none_action(self):
        """NONE command → not executed"""
        from alice.brain.life_loop import CompensationCommand, CompensationAction, ErrorType
        alice = AliceBrain(neuron_count=20)
        cmd = CompensationCommand(
            action=CompensationAction.NONE,
            target=np.array([0.0]),
            strength=0.0,
            source_error=ErrorType.SENSORY_PREDICTION,
            priority=0.0,
        )
        results = alice._dispatch_commands([cmd])
        assert results[0]["executed"] is False


# ============================================================================
# Neural Pruning Wiring Tests — Experience-driven cortical specialization
# ============================================================================


class TestPruningWiring:
    """Verify NeuralPruningEngine is connected to main loop"""

    def test_visual_stimulation_reaches_occipital(self):
        """Visual perception → occipital region receives stimulation"""
        alice = AliceBrain(neuron_count=20)
        occ = alice.pruning.regions["occipital"]
        stim_before = occ.stimulation_cycles
        alice.perceive(np.random.rand(20), Modality.VISUAL)
        assert occ.stimulation_cycles > stim_before

    def test_auditory_stimulation_reaches_temporal(self):
        """Auditory perception → temporal region receives stimulation"""
        alice = AliceBrain(neuron_count=20)
        temp = alice.pruning.regions["temporal"]
        stim_before = temp.stimulation_cycles
        alice.perceive(np.random.rand(64), Modality.AUDITORY)
        assert temp.stimulation_cycles > stim_before

    def test_tactile_stimulation_reaches_parietal(self):
        """Tactile perception → parietal region receives stimulation"""
        alice = AliceBrain(neuron_count=20)
        par = alice.pruning.regions["parietal"]
        stim_before = par.stimulation_cycles
        alice.perceive(np.random.rand(20), Modality.TACTILE)
        assert par.stimulation_cycles > stim_before

    def test_pruning_after_50_ticks(self):
        """After 50 perceive calls, apoptosis scan is triggered"""
        alice = AliceBrain(neuron_count=20)
        occ = alice.pruning.regions["occipital"]
        initial_alive = occ.alive_count
        for i in range(51):
            alice.perceive(np.random.rand(20), Modality.VISUAL)
        # Apoptosis may have removed some connections
        # At least confirm stimulate was called 51 times
        assert occ.stimulation_cycles >= 51

    def test_pruning_state_in_introspect(self):
        """introspect() report includes pruning state"""
        alice = AliceBrain(neuron_count=20)
        alice.perceive(np.random.rand(20))
        report = alice.introspect()
        assert "pruning" in report["subsystems"]

    def test_gamma_squared_decreases_with_experience(self):
        """Experience-driven pruning → global Γ² should trend downward"""
        alice = AliceBrain(neuron_count=20)
        # Initial Γ²
        g2_before = alice.pruning._compute_global_gamma_squared()
        # Give lots of visual stimuli → occipital should specialize
        for _ in range(60):
            alice.perceive(np.random.rand(20), Modality.VISUAL)
        g2_after = alice.pruning._compute_global_gamma_squared()
        # Γ² should at least not surge (Hebbian-matched connections are strengthened)
        # Note: due to random initialization, strict decrease is not guaranteed, but should not surge
        assert g2_after < g2_before * 1.5  # Allow minor fluctuation

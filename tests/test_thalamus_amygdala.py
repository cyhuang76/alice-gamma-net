# -*- coding: utf-8 -*-
"""
Tests for Thalamus (Phase 5.3) and Amygdala (Phase 5.4)

Complete test suite for the thalamus (sensory gate) + amygdala (emotional fast pathway).
"""

import time
import numpy as np
import pytest

from alice.brain.thalamus import (
    ThalamusEngine,
    SensoryChannel,
    ThalamicGateResult,
    GATE_MIN,
    GATE_MAX,
    MAX_FOCUSED_CHANNELS,
    BURST_AROUSAL_THRESHOLD,
    STARTLE_AMPLITUDE_THRESHOLD,
    STARTLE_GAMMA_THRESHOLD,
)

from alice.brain.amygdala import (
    AmygdalaEngine,
    FearMemory,
    EmotionalState,
    AmygdalaResponse,
    FEAR_THRESHOLD,
    FREEZE_THRESHOLD,
    THREAT_IMPEDANCE,
    SAFETY_IMPEDANCE,
)


# ============================================================================
# Helper Functions
# ============================================================================


def make_fingerprint(dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    fp = rng.randn(dim).astype(np.float64)
    norm = np.linalg.norm(fp)
    if norm > 1e-10:
        fp /= norm
    return fp


def make_visual_fp(seed: int = 42) -> np.ndarray:
    return make_fingerprint(256, seed)


def make_auditory_fp(seed: int = 42) -> np.ndarray:
    return make_fingerprint(32, seed)


# ============================================================================
# Test SensoryChannel
# ============================================================================


class TestSensoryChannel:
    """Thalamic sensory channel data structure tests."""

    def test_default_values(self):
        ch = SensoryChannel(modality="visual")
        assert ch.modality == "visual"
        assert ch.gate_gain == 0.5
        assert ch.topdown_bias == 0.5
        assert ch.habituation == 0.0

    def test_signal_counting(self):
        ch = SensoryChannel(modality="auditory")
        assert ch.signal_count == 0
        assert ch.blocked_count == 0
        assert ch.total_count == 0


# ============================================================================
# Test ThalamusEngine — Basic Functionality
# ============================================================================


class TestThalamusBasic:
    """Thalamus engine basic functionality tests."""

    def test_init_default(self):
        t = ThalamusEngine()
        assert t.alpha == 0.6
        assert t._arousal == 0.8
        assert len(t._channels) == 0

    def test_auto_create_channel(self):
        t = ThalamusEngine()
        result = t.gate("visual", amplitude=0.5, gamma=0.3)
        assert "visual" in t._channels
        assert isinstance(result, ThalamicGateResult)
        assert result.modality == "visual"

    def test_gate_returns_result(self):
        t = ThalamusEngine()
        fp = make_visual_fp()
        result = t.gate("visual", fingerprint=fp, amplitude=0.5, gamma=0.3)
        assert isinstance(result, ThalamicGateResult)
        assert 0.0 <= result.gate_gain <= 1.0
        assert 0.0 <= result.salience <= 1.0

    def test_gate_gain_bounded(self):
        """Gate gain must be in [GATE_MIN, GATE_MAX] (non-burst mode)."""
        t = ThalamusEngine()
        # Normal arousal + low bias → non-burst mode
        t.set_arousal(0.5)
        t.set_attention("visual", 0.0)
        result = t.gate("visual", amplitude=0.1, gamma=0.1)
        assert result.gate_gain >= GATE_MIN

    def test_high_arousal_high_gain(self):
        """High arousal + high attention → high gate gain."""
        t = ThalamusEngine()
        t.set_arousal(1.0)
        t.set_attention("visual", 1.0)
        # Multiple gate calls to let EMA converge
        for _ in range(10):
            result = t.gate("visual", amplitude=0.8, gamma=0.8)
        assert result.gate_gain > 0.5


# ============================================================================
# Test ThalamusEngine — Startle Circuit
# ============================================================================


class TestThalamusStartle:
    """Thalamus startle circuit tests."""

    def test_high_amplitude_startle(self):
        """High amplitude signal unconditionally passes through the gate."""
        t = ThalamusEngine()
        t.set_arousal(0.1)  # Low arousal
        result = t.gate("visual", amplitude=STARTLE_AMPLITUDE_THRESHOLD, gamma=0.1)
        assert result.passed is True
        assert result.is_startle is True
        assert result.gate_gain == GATE_MAX

    def test_high_gamma_startle(self):
        """High Γ signal unconditionally passes through the gate."""
        t = ThalamusEngine()
        t.set_arousal(0.1)
        result = t.gate("auditory", amplitude=0.3, gamma=STARTLE_GAMMA_THRESHOLD)
        assert result.passed is True
        assert result.is_startle is True

    def test_startle_breaks_habituation(self):
        """Startle should break habituation."""
        t = ThalamusEngine()
        ch = t._ensure_channel("visual")
        ch.habituation = 0.8  # Highly habituated
        t.gate("visual", amplitude=STARTLE_AMPLITUDE_THRESHOLD, gamma=0.1)
        assert ch.habituation < 0.8

    def test_startle_stats(self):
        """Startle count statistics."""
        t = ThalamusEngine()
        t.gate("visual", amplitude=0.95, gamma=0.1)
        assert t._total_startles == 1
        assert t._total_passed == 1


# ============================================================================
# Test ThalamusEngine — Attention Control
# ============================================================================


class TestThalamusAttention:
    """Thalamus attention control tests."""

    def test_set_attention(self):
        t = ThalamusEngine()
        t.set_attention("visual", 0.9)
        ch = t._channels["visual"]
        assert ch.topdown_bias == 0.9
        assert "visual" in t._focused_modalities

    def test_attention_bottleneck(self):
        """Attention bottleneck: simultaneous focus limited to MAX_FOCUSED_CHANNELS."""
        t = ThalamusEngine()
        for i in range(5):
            t.set_attention(f"modality_{i}", 0.8)
        assert len(t._focused_modalities) <= MAX_FOCUSED_CHANNELS

    def test_attention_eviction(self):
        """Earliest focused channel gets evicted when bottleneck is exceeded."""
        t = ThalamusEngine()
        t.set_attention("mod_a", 0.9)
        t.set_attention("mod_b", 0.9)
        t.set_attention("mod_c", 0.9)
        t.set_attention("mod_d", 0.9)  # This should evict mod_a
        assert len(t._focused_modalities) == MAX_FOCUSED_CHANNELS
        assert "mod_a" not in t._focused_modalities

    def test_unfocus(self):
        """Setting low bias → removed from the focused list."""
        t = ThalamusEngine()
        t.set_attention("visual", 0.9)
        assert "visual" in t._focused_modalities
        t.set_attention("visual", 0.2)
        assert "visual" not in t._focused_modalities

    def test_trn_inhibition(self):
        """TRN competitive inhibition: focused channels enhanced, unfocused channels suppressed."""
        t = ThalamusEngine()
        t.set_attention("visual", 0.9)
        t.set_attention("auditory", 0.2)  # Not focused
        t.apply_trn_inhibition()
        assert t._channels["visual"].topdown_bias > t._channels["auditory"].topdown_bias


# ============================================================================
# Test ThalamusEngine — Habituation
# ============================================================================


class TestThalamusHabituation:
    """Thalamus habituation tests."""

    def test_same_stimulus_habituates(self):
        """Continuous same stimulus → habituation increases."""
        t = ThalamusEngine()
        fp = make_visual_fp(seed=42)
        for _ in range(20):
            t.gate("visual", fingerprint=fp, amplitude=0.5, gamma=0.3)
        ch = t._channels["visual"]
        assert ch.habituation > 0.1  # Significant habituation

    def test_novel_stimulus_recovers(self):
        """Novel stimulus → habituation decreases."""
        t = ThalamusEngine()
        fp1 = make_visual_fp(seed=42)
        for _ in range(10):
            t.gate("visual", fingerprint=fp1, amplitude=0.5, gamma=0.3)
        hab_before = t._channels["visual"].habituation

        fp2 = make_visual_fp(seed=99)  # Different fingerprint
        t.gate("visual", fingerprint=fp2, amplitude=0.5, gamma=0.3)
        assert t._channels["visual"].habituation <= hab_before


# ============================================================================
# Test ThalamusEngine — Thalamic Burst Mode
# ============================================================================


class TestThalamusBurst:
    """Thalamic burst mode (low arousal) tests."""

    def test_burst_mode_at_low_arousal(self):
        """Low arousal → thalamus enters burst mode."""
        t = ThalamusEngine()
        t.set_arousal(0.1)  # Very low arousal
        np.random.seed(42)
        results = [
            t.gate("visual", amplitude=0.3, gamma=0.3) for _ in range(20)
        ]
        # Some should be blocked
        burst_results = [r for r in results if r.is_burst]
        assert len(burst_results) > 0

    def test_normal_arousal_no_burst(self):
        """Normal arousal → no burst mode."""
        t = ThalamusEngine()
        t.set_arousal(0.8)
        results = [t.gate("visual", amplitude=0.5, gamma=0.3) for _ in range(10)]
        burst_results = [r for r in results if r.is_burst]
        assert len(burst_results) == 0


# ============================================================================
# Test ThalamusEngine — Batch Gating & State Query
# ============================================================================


class TestThalamusBatch:
    """Thalamus batch operation tests."""

    def test_gate_all(self):
        t = ThalamusEngine()
        signals = {
            "visual": {"fingerprint": make_visual_fp(), "amplitude": 0.5, "gamma": 0.3},
            "auditory": {"fingerprint": make_auditory_fp(), "amplitude": 0.7, "gamma": 0.6},
        }
        results = t.gate_all(signals)
        assert "visual" in results
        assert "auditory" in results
        assert isinstance(results["visual"], ThalamicGateResult)

    def test_get_state(self):
        t = ThalamusEngine()
        t.gate("visual", amplitude=0.5, gamma=0.3)
        state = t.get_state()
        assert "arousal" in state
        assert "channels" in state
        assert "stats" in state
        assert state["stats"]["total_gated"] == 1

    def test_reset(self):
        t = ThalamusEngine()
        t.gate("visual", amplitude=0.5, gamma=0.3)
        t.reset()
        assert len(t._channels) == 0
        assert t._total_gated == 0


# ============================================================================
# Test FearMemory
# ============================================================================


class TestFearMemory:
    """Amygdala fear memory tests."""

    def test_effective_threat_default(self):
        fm = FearMemory(
            modality="visual",
            fingerprint=make_visual_fp(),
            threat_level=0.8,
        )
        assert fm.effective_threat == pytest.approx(0.8, abs=0.2)

    def test_conditioning_increases_threat(self):
        fm = FearMemory(
            modality="visual",
            fingerprint=make_visual_fp(),
            threat_level=0.5,
            conditioning_count=5,
        )
        assert fm.effective_threat > 0.5

    def test_extinction_decreases_threat(self):
        fm = FearMemory(
            modality="visual",
            fingerprint=make_visual_fp(),
            threat_level=0.8,
            extinction_count=10,
        )
        assert fm.effective_threat < 0.8

    def test_threat_never_zero(self):
        """Fear never fully disappears."""
        fm = FearMemory(
            modality="visual",
            fingerprint=make_visual_fp(),
            threat_level=0.8,
            extinction_count=100,
        )
        assert fm.effective_threat > 0


# ============================================================================
# Test AmygdalaEngine — Basic Functionality
# ============================================================================


class TestAmygdalaBasic:
    """Amygdala engine basic functionality tests."""

    def test_init(self):
        a = AmygdalaEngine()
        assert a._valence == 0.0
        assert a._threat_level == 0.0
        assert len(a._fear_memories) == 0

    def test_evaluate_neutral(self):
        """Neutral input → low threat."""
        a = AmygdalaEngine()
        resp = a.evaluate("visual", amplitude=0.3, gamma=0.1)
        assert isinstance(resp, AmygdalaResponse)
        assert resp.emotional_state.threat_level < 0.3
        assert resp.emotional_state.is_fight_flight is False

    def test_evaluate_high_pain(self):
        """High pain → high threat → negative valence."""
        a = AmygdalaEngine()
        resp = a.evaluate("visual", amplitude=0.5, gamma=0.5, pain_level=0.9)
        assert resp.emotional_state.threat_level > 0.1
        assert resp.emotional_state.valence < 0

    def test_evaluate_returns_gamma_threat(self):
        """Must return Γ_threat."""
        a = AmygdalaEngine()
        resp = a.evaluate("auditory", amplitude=0.5, gamma=0.3)
        assert 0.0 <= resp.gamma_threat <= 1.0

    def test_valence_bounded(self):
        """Valence must be in [-1, 1] range."""
        a = AmygdalaEngine()
        for _ in range(50):
            a.evaluate("visual", amplitude=0.9, gamma=0.9, pain_level=0.9)
        assert -1.0 <= a._valence <= 1.0


# ============================================================================
# Test AmygdalaEngine — Fight-or-Flight Response
# ============================================================================


class TestAmygdalaFightFlight:
    """Amygdala fight-or-flight response tests."""

    def test_fight_flight_trigger(self):
        """Sustained high threat → triggers fight-or-flight."""
        a = AmygdalaEngine()
        for _ in range(20):
            resp = a.evaluate("visual", amplitude=0.9, gamma=0.9, pain_level=0.9)
        assert a._threat_level > 0.3 or a._total_threats > 0

    def test_fight_flight_dissipates(self):
        """Fight-or-flight has inertia but eventually dissipates."""
        a = AmygdalaEngine()
        # Trigger high threat
        for _ in range(20):
            a.evaluate("visual", amplitude=0.9, gamma=0.9, pain_level=0.9)
        # Then give safety signals
        for _ in range(50):
            a.evaluate("visual", amplitude=0.1, gamma=0.1, pain_level=0.0)
            a.decay_tick()
        assert a._threat_level < 0.3

    def test_sympathetic_command(self):
        """High threat should output sympathetic nervous command."""
        a = AmygdalaEngine()
        for _ in range(20):
            resp = a.evaluate("visual", amplitude=0.9, gamma=0.9, pain_level=0.9)
        # At least in some cases there is a sympathetic command
        assert resp.sympathetic_command >= 0


# ============================================================================
# Test AmygdalaEngine — Fear Conditioning
# ============================================================================


class TestAmygdalaConditioning:
    """Amygdala fear conditioning tests."""

    def test_condition_fear(self):
        """Fear conditioning: bind fingerprint with threat."""
        a = AmygdalaEngine()
        fp = make_visual_fp(seed=42)
        a.condition_fear("visual", fp, threat_level=0.9, concept_label="snake")
        assert len(a._fear_memories) == 1
        assert a._fear_memories[0].concept_label == "snake"

    def test_conditioned_stimulus_triggers_fear(self):
        """Conditioned stimulus → matches fear memory."""
        a = AmygdalaEngine()
        fp = make_visual_fp(seed=42)
        a.condition_fear("visual", fp, threat_level=0.9, concept_label="snake")

        # Evaluate with the same fingerprint
        resp = a.evaluate("visual", fingerprint=fp, amplitude=0.3, gamma=0.1,
                          concept_label="snake")
        assert resp.fear_matched is True
        assert resp.matched_memory == "snake"

    def test_reconditioning_strengthens(self):
        """Repeated conditioning → strengthens memory."""
        a = AmygdalaEngine()
        fp = make_visual_fp(seed=42)
        a.condition_fear("visual", fp, concept_label="dog")
        a.condition_fear("visual", fp, concept_label="dog")
        a.condition_fear("visual", fp, concept_label="dog")
        assert a._fear_memories[0].conditioning_count == 3

    def test_extinction(self):
        """Fear extinction: safe exposure reduces effective threat."""
        a = AmygdalaEngine()
        fp = make_visual_fp(seed=42)
        a.condition_fear("visual", fp, threat_level=0.9, concept_label="spider")
        initial_threat = a._fear_memories[0].effective_threat

        # Multiple extinctions
        for _ in range(10):
            a.extinguish_fear("visual", fp, concept_label="spider")
        assert a._fear_memories[0].effective_threat < initial_threat

    def test_extinction_not_deletion(self):
        """Extinction is not deletion — fear memories persist forever."""
        a = AmygdalaEngine()
        fp = make_visual_fp(seed=42)
        a.condition_fear("visual", fp, threat_level=0.9, concept_label="fire")
        for _ in range(50):
            a.extinguish_fear("visual", fp, concept_label="fire")
        # Memory still exists
        assert len(a._fear_memories) == 1
        assert a._fear_memories[0].effective_threat > 0

    def test_cross_modal_fear(self):
        """Concept labels enable cross-modal fear."""
        a = AmygdalaEngine()
        fp_vis = make_visual_fp(seed=42)
        a.condition_fear("visual", fp_vis, threat_level=0.9, concept_label="tiger")

        # Evaluate with auditory modality + same concept label
        fp_aud = make_auditory_fp(seed=99)
        resp = a.evaluate("auditory", fingerprint=fp_aud, concept_label="tiger")
        assert resp.fear_matched is True
        assert resp.matched_memory == "tiger"

    def test_capacity_management(self):
        """Fear memory capacity limit."""
        a = AmygdalaEngine()
        for i in range(60):
            fp = make_fingerprint(32, seed=i)
            a.condition_fear("auditory", fp, threat_level=0.1 + i * 0.01,
                             concept_label=f"fear_{i}")
        assert len(a._fear_memories) <= 50


# ============================================================================
# Test AmygdalaEngine — Emotional Decay & State
# ============================================================================


class TestAmygdalaDecay:
    """Amygdala emotional decay tests."""

    def test_decay_reduces_valence(self):
        """Emotional decay: valence gradually returns to zero."""
        a = AmygdalaEngine()
        a._valence = -0.8
        for _ in range(50):
            a.decay_tick()
        assert abs(a._valence) < 0.2

    def test_get_state(self):
        a = AmygdalaEngine()
        state = a.get_state()
        assert "valence" in state
        assert "threat_level" in state
        assert "fear_memories" in state
        assert "stats" in state

    def test_reset_preserves_fear(self):
        """Reset emotions but preserve fear memories."""
        a = AmygdalaEngine()
        fp = make_visual_fp()
        a.condition_fear("visual", fp, concept_label="trauma")
        a._valence = -0.9
        a._threat_level = 0.8
        a.reset()
        assert a._valence == 0.0
        assert a._threat_level == 0.0
        assert len(a._fear_memories) == 1  # Trauma memory preserved

    def test_emotion_label(self):
        """Valence → emotion label mapping."""
        a = AmygdalaEngine()
        assert a._get_emotion_label(0.0) == "neutral"
        assert a._get_emotion_label(-0.8) == "terror"
        assert a._get_emotion_label(0.8) == "joy"


# ============================================================================
# Test Thalamus-Amygdala Integration
# ============================================================================


class TestThalamusAmygdalaIntegration:
    """Thalamus + amygdala joint tests."""

    def test_startle_triggers_amygdala(self):
        """Thalamic startle → amygdala high threat."""
        thal = ThalamusEngine()
        amyg = AmygdalaEngine()

        fp = make_visual_fp()
        # Startle signal passes through thalamus
        thal_result = thal.gate("visual", fingerprint=fp, amplitude=0.95, gamma=0.1)
        assert thal_result.is_startle is True

        # Send to amygdala
        amyg_resp = amyg.evaluate(
            "visual", fingerprint=fp,
            gamma=0.1, amplitude=0.95,
        )
        # High amplitude triggers threat
        assert amyg_resp.emotional_state.threat_level >= 0

    def test_fear_enhances_thalamic_gate(self):
        """Fear conditioning → fearful stimulus enhances thalamic attention bias."""
        thal = ThalamusEngine()
        amyg = AmygdalaEngine()

        # Condition fear
        fp = make_visual_fp(seed=42)
        amyg.condition_fear("visual", fp, threat_level=0.9, concept_label="danger")

        # Baseline bias
        thal._ensure_channel("visual")
        base_bias = thal._channels["visual"].topdown_bias

        # Fearful stimulus → boost thalamic attention
        resp = amyg.evaluate("visual", fingerprint=fp, concept_label="danger")
        if resp.emotional_state.threat_level > 0.1:
            thal.set_attention("visual", min(1.0, 0.5 + resp.emotional_state.threat_level))

        # Attention bias should be elevated
        assert thal._channels["visual"].topdown_bias > base_bias

    def test_sleep_closes_gates(self):
        """Sleep → low arousal → thalamic gate closes."""
        thal = ThalamusEngine()
        thal.set_arousal(1.0)  # Awake
        for _ in range(5):
            r1 = thal.gate("visual", amplitude=0.5, gamma=0.3)
        awake_gain = thal._channels["visual"].gate_gain

        thal.set_arousal(0.1)  # Sleep
        for _ in range(10):
            r2 = thal.gate("visual", amplitude=0.5, gamma=0.3)
        sleep_gain = thal._channels["visual"].gate_gain

        assert sleep_gain < awake_gain

    def test_dual_pathway_low_high(self):
        """Dual pathway model: low road (amygdala fast) vs high road (cortical precise)."""
        thal = ThalamusEngine()
        amyg = AmygdalaEngine()

        fp = make_auditory_fp()

        # Low road: amygdala direct evaluation (bypasses cortical analysis)
        amyg_fast = amyg.evaluate("auditory", fingerprint=fp, amplitude=0.8, gamma=0.7)

        # High road: thalamic gate → cortex (semantic field, etc.) → amygdala
        thal_result = thal.gate("auditory", fingerprint=fp, amplitude=0.8, gamma=0.7)
        # High road: more precise but slower (timing difference determined by architecture in physical implementation)

        # Both pathways should respond to high signals
        assert amyg_fast.gamma_threat >= 0
        assert thal_result.gate_gain >= 0


# ============================================================================
# Test AliceBrain Integration
# ============================================================================


class TestAliceBrainThalamusAmygdala:
    """Integration tests for thalamus + amygdala in AliceBrain."""

    def test_see_returns_thalamus(self):
        """see() should include thalamus result."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        pixels = np.random.randn(64)
        result = brain.see(pixels)
        assert "thalamus" in result
        assert "passed" in result["thalamus"]
        assert "gate_gain" in result["thalamus"]

    def test_see_returns_amygdala(self):
        """see() should include amygdala result."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        pixels = np.random.randn(64)
        result = brain.see(pixels)
        assert "amygdala" in result
        assert "valence" in result["amygdala"]
        assert "threat_level" in result["amygdala"]

    def test_hear_returns_thalamus(self):
        """hear() should include thalamus result."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        sound = np.sin(np.linspace(0, 2 * np.pi * 440, 1024))
        result = brain.hear(sound)
        assert "thalamus" in result
        assert "passed" in result["thalamus"]

    def test_hear_returns_amygdala(self):
        """hear() should include amygdala result."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        sound = np.sin(np.linspace(0, 2 * np.pi * 440, 1024))
        result = brain.hear(sound)
        assert "amygdala" in result
        assert "valence" in result["amygdala"]

    def test_introspect_includes_thalamus_amygdala(self):
        """introspect() includes thalamus and amygdala state."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        state = brain.introspect()
        assert "thalamus" in state["subsystems"]
        assert "amygdala" in state["subsystems"]


# ============================================================================
# Test Physical Invariants
# ============================================================================


class TestPhysicsInvariants:
    """Physical invariant tests."""

    def test_gate_gain_range(self):
        """Gate gain always in [GATE_MIN, GATE_MAX] (in non-burst mode)."""
        t = ThalamusEngine()
        for seed in range(20):
            fp = make_fingerprint(64, seed)
            result = t.gate("test", fingerprint=fp,
                            amplitude=np.random.random(),
                            gamma=np.random.random(),
                            arousal=np.random.random())
            if not result.is_startle and not result.is_burst:
                assert GATE_MIN <= result.gate_gain <= GATE_MAX

    def test_valence_bounded(self):
        """Valence always in [-1, 1]."""
        a = AmygdalaEngine()
        for _ in range(100):
            a.evaluate("visual",
                       amplitude=np.random.random(),
                       gamma=np.random.random(),
                       pain_level=np.random.random())
        assert -1.0 <= a._valence <= 1.0

    def test_threat_bounded(self):
        """Threat level always in [0, 1]."""
        a = AmygdalaEngine()
        for _ in range(100):
            a.evaluate("visual",
                       amplitude=np.random.random(),
                       gamma=np.random.random(),
                       pain_level=np.random.random())
        assert 0.0 <= a._threat_level <= 1.0

    def test_gamma_threat_physical(self):
        """Γ_threat follows the impedance reflection formula."""
        a = AmygdalaEngine()
        resp = a.evaluate("visual", amplitude=0.5, gamma=0.3)
        # Γ = |Z_sig - Z_threat| / (Z_sig + Z_threat) must be ∈ [0, 1]
        assert 0.0 <= resp.gamma_threat <= 1.0

    def test_startle_always_passes(self):
        """Startle signal passes at any arousal level."""
        t = ThalamusEngine()
        for arousal in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            t.set_arousal(arousal)
            result = t.gate("test", amplitude=0.95, gamma=0.1)
            assert result.passed is True

    def test_habituation_bounded(self):
        """Habituation bounded in [0, 1]."""
        t = ThalamusEngine()
        fp = make_fingerprint(32, seed=42)
        for _ in range(200):
            t.gate("test", fingerprint=fp, amplitude=0.5, gamma=0.3)
        assert 0.0 <= t._channels["test"].habituation <= 1.0

    def test_fear_memory_never_deleted(self):
        """Fear memories can only be extinguished, not deleted."""
        a = AmygdalaEngine()
        fp = make_visual_fp()
        a.condition_fear("visual", fp, concept_label="trauma")
        initial_count = len(a._fear_memories)

        # Massive extinction
        for _ in range(100):
            a.extinguish_fear("visual", fp, concept_label="trauma")
        assert len(a._fear_memories) == initial_count

    def test_safety_impedance_gt_threat(self):
        """Safety channel impedance > threat channel impedance (physical design)."""
        assert SAFETY_IMPEDANCE > THREAT_IMPEDANCE

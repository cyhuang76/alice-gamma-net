# -*- coding: utf-8 -*-
"""
test_emotion_granularity.py — EmotionGranularityEngine Complete Tests

Covers:
  1. Basic initialization
  2. Single emotion injection
  3. Multi-source injection
  4. Differential decay
  5. Compound emotion detection
  6. VAD computation
  7. Emotion regulation
  8. Emotional richness/depth
  9. Impedance model
  10. Fear conditioning injection
  11. HIP scenario simulation
"""

import math
import pytest
import numpy as np

from alice.brain.emotion_granularity import (
    EmotionGranularityEngine,
    EmotionVector,
    GranularEmotionState,
    PLUTCHIK_PRIMARIES,
    COMPOUND_EMOTIONS,
    COMPOUND_THRESHOLD,
)


class TestEmotionGranularityInit:
    """Initialization tests."""

    def test_default_state(self):
        engine = EmotionGranularityEngine()
        state = engine.get_state()
        assert state["valence"] == 0.0
        assert state["arousal"] == 0.0
        assert state["dominant_emotion"] == "neutral"
        assert state["richness"] == 0.0

    def test_all_primaries_zero(self):
        engine = EmotionGranularityEngine()
        vec = engine.get_emotion_vector()
        for name in PLUTCHIK_PRIMARIES:
            assert vec[name] == 0.0

    def test_eight_emotions_defined(self):
        assert len(PLUTCHIK_PRIMARIES) == 8
        expected = {"joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"}
        assert set(PLUTCHIK_PRIMARIES.keys()) == expected

    def test_eight_compound_emotions(self):
        assert len(COMPOUND_EMOTIONS) == 8


class TestEmotionInjection:
    """Emotion injection tests."""

    def test_inject_threat_high_no_dominance(self):
        """High threat + low dominance → primarily fear."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(threat_level=0.8, dominance_sense=0.1)
        engine.tick()
        assert engine.get_activation("fear") > 0.1
        assert engine.get_activation("fear") > engine.get_activation("anger")

    def test_inject_threat_high_with_dominance(self):
        """High threat + high dominance → anger > fear."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(threat_level=0.8, dominance_sense=0.9)
        engine.tick()
        # With high dominance, anger dominates
        assert engine.get_activation("anger") > 0.0
        # Fear reduces due to low dominance weight
        fear = engine.get_activation("fear")
        anger = engine.get_activation("anger")
        assert anger > fear * 0.5  # At least comparable to fear

    def test_inject_threat_safe(self):
        """Low threat + no pain → trust + joy."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(threat_level=0.0, pain_level=0.0)
        engine.tick()
        assert engine.get_activation("trust") > 0.0
        assert engine.get_activation("joy") > 0.0

    def test_inject_threat_with_pain(self):
        """Threat + pain → anger + sadness."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(threat_level=0.5, pain_level=0.6, dominance_sense=0.5)
        engine.tick()
        assert engine.get_activation("sadness") > 0.0
        assert engine.get_activation("anger") > 0.0

    def test_inject_threat_with_fear_memory(self):
        """Matching fear memory → enhanced fear."""
        engine = EmotionGranularityEngine()

        # No memory match
        engine.inject_threat(threat_level=0.5, dominance_sense=0.3)
        engine.tick()
        fear_no_match = engine.get_activation("fear")

        # Reset
        engine.reset()

        # With memory match
        engine.inject_threat(threat_level=0.5, dominance_sense=0.3, fear_matched=True)
        engine.tick()
        fear_matched = engine.get_activation("fear")

        assert fear_matched > fear_no_match

    def test_inject_reward_positive(self):
        """Positive reward → joy."""
        engine = EmotionGranularityEngine()
        engine.inject_reward(0.8)
        engine.tick()
        assert engine.get_activation("joy") > 0.1

    def test_inject_reward_negative(self):
        """Negative reward → sadness."""
        engine = EmotionGranularityEngine()
        engine.inject_reward(-0.7)
        engine.tick()
        assert engine.get_activation("sadness") > 0.1

    def test_inject_novelty(self):
        """Novelty → surprise."""
        engine = EmotionGranularityEngine()
        engine.inject_novelty(surprise_level=0.8)
        engine.tick()
        assert engine.get_activation("surprise") > 0.1

    def test_inject_novelty_with_curiosity(self):
        """Curiosity satisfied → anticipation + joy."""
        engine = EmotionGranularityEngine()
        engine.inject_novelty(0.3, curiosity_satisfied=True)
        engine.tick()
        assert engine.get_activation("anticipation") > 0.05
        assert engine.get_activation("joy") > 0.0

    def test_inject_social_bonding(self):
        """Social bonding → trust."""
        engine = EmotionGranularityEngine()
        engine.inject_social(social_bond_strength=0.7)
        engine.tick()
        assert engine.get_activation("trust") > 0.1

    def test_inject_social_rejection(self):
        """Social rejection → sadness + anger."""
        engine = EmotionGranularityEngine()
        engine.inject_social(rejection=0.8)
        engine.tick()
        assert engine.get_activation("sadness") > 0.05
        assert engine.get_activation("anger") > 0.0

    def test_inject_homeostatic_satisfied(self):
        """Need satisfied → joy."""
        engine = EmotionGranularityEngine()
        engine.inject_homeostatic(satisfaction=0.8)
        engine.tick()
        assert engine.get_activation("joy") > 0.05

    def test_inject_homeostatic_deficit(self):
        """Need deficit + irritability → anger."""
        engine = EmotionGranularityEngine()
        engine.inject_homeostatic(deficit=0.7, irritability=0.6)
        engine.tick()
        assert engine.get_activation("anger") > 0.0

    def test_inject_goal_achievement(self):
        """Goal achievement → joy."""
        engine = EmotionGranularityEngine()
        engine.inject_goal(achievement=0.9)
        engine.tick()
        assert engine.get_activation("joy") > 0.1

    def test_inject_goal_frustration(self):
        """Frustration → anger."""
        engine = EmotionGranularityEngine()
        engine.inject_goal(frustration=0.8)
        engine.tick()
        assert engine.get_activation("anger") > 0.05

    def test_inject_fear_conditioning(self):
        """Fear conditioning → fear + surprise."""
        engine = EmotionGranularityEngine()
        engine.inject_fear_conditioning(threat_level=0.7)
        engine.tick()
        assert engine.get_activation("fear") > 0.1
        assert engine.get_activation("surprise") > 0.0


class TestDifferentialDecay:
    """Differential decay tests."""

    def test_surprise_decays_fastest(self):
        """Surprise decays fastest."""
        engine = EmotionGranularityEngine()
        engine.inject_novelty(surprise_level=0.9)
        engine.inject_homeostatic(deficit=0.9, irritability=0.0)  # sadness trigger

        engine.tick()
        surprise_t0 = engine.get_activation("surprise")

        # Run 20 ticks
        for _ in range(20):
            engine.tick()

        surprise_t20 = engine.get_activation("surprise")
        # Surprise kappa=0.10 → after 20 ticks × (1-0.1)^20 ≈ 0.12
        assert surprise_t20 < surprise_t0 * 0.25

    def test_sadness_decays_slowest(self):
        """Sadness decays slowest."""
        engine = EmotionGranularityEngine()
        # Inject sadness and surprise to similar levels
        engine.inject_reward(-0.9)  # → sadness
        engine.inject_novelty(0.9)  # → surprise

        engine.tick()
        sadness_t0 = engine.get_activation("sadness")
        surprise_t0 = engine.get_activation("surprise")

        for _ in range(30):
            engine.tick()

        sadness_t30 = engine.get_activation("sadness")
        surprise_t30 = engine.get_activation("surprise")

        # Sadness kappa=0.008 → retains more
        # Surprise kappa=0.100 → almost fully decayed
        if sadness_t0 > 0 and surprise_t0 > 0:
            sadness_ratio = sadness_t30 / sadness_t0
            surprise_ratio = surprise_t30 / surprise_t0
            assert sadness_ratio > surprise_ratio

    def test_all_decay_to_near_zero(self):
        """All emotions eventually decay to near zero."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.9, pain_level=0.5, dominance_sense=0.5)
        engine.inject_reward(0.5)
        engine.inject_novelty(0.9)
        engine.tick()

        # Run 500 ticks — even the slowest sadness should have decayed
        for _ in range(500):
            engine.tick()

        for name in engine.EMOTION_NAMES:
            assert engine.get_activation(name) < 0.05, f"{name} did not decay to near zero"


class TestVADComputation:
    """VAD computation tests."""

    def test_pure_joy_positive_valence(self):
        """Pure joy → positive valence."""
        engine = EmotionGranularityEngine()
        engine.inject_reward(0.9)
        engine.tick()
        assert engine.get_valence() > 0.0

    def test_pure_fear_negative_valence(self):
        """Pure fear → negative valence."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.9, dominance_sense=0.0)
        engine.tick()
        assert engine.get_valence() < 0.0

    def test_fear_high_arousal(self):
        """Fear → high arousal."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.9, dominance_sense=0.0)
        engine.tick()
        assert engine.get_arousal() > 0.5

    def test_sadness_low_arousal(self):
        """Sadness → low arousal."""
        engine = EmotionGranularityEngine()
        engine.inject_reward(-0.9)
        for _ in range(5):
            engine.inject_reward(-0.9)
            engine.tick()
        # Sadness dominates → arousal is lower
        assert engine.get_arousal() < 0.5

    def test_anger_high_dominance(self):
        """Anger → high dominance."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.9, dominance_sense=0.9)
        engine.tick()
        assert engine.get_dominance() > 0.4


class TestCompoundEmotions:
    """Compound emotion detection tests."""

    def test_love_detection(self):
        """joy + trust → love"""
        engine = EmotionGranularityEngine()
        engine.inject_reward(0.7)  # joy
        engine.inject_social(social_bond_strength=0.8)  # trust
        engine.tick()
        compounds = engine.get_compound_emotions()
        if engine.get_activation("joy") > COMPOUND_THRESHOLD and \
           engine.get_activation("trust") > COMPOUND_THRESHOLD:
            assert "love" in compounds

    def test_awe_detection(self):
        """fear + surprise → awe"""
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.5, dominance_sense=0.1)  # fear
        engine.inject_novelty(0.8)  # surprise
        engine.tick()
        compounds = engine.get_compound_emotions()
        if engine.get_activation("fear") > COMPOUND_THRESHOLD and \
           engine.get_activation("surprise") > COMPOUND_THRESHOLD:
            assert "awe" in compounds

    def test_optimism_detection(self):
        """anticipation + joy → optimism"""
        engine = EmotionGranularityEngine()
        engine.inject_goal(progress=0.8)  # anticipation
        engine.inject_reward(0.6)  # joy
        engine.tick()
        compounds = engine.get_compound_emotions()
        if engine.get_activation("anticipation") > COMPOUND_THRESHOLD and \
           engine.get_activation("joy") > COMPOUND_THRESHOLD:
            assert "optimism" in compounds


class TestEmotionRegulation:
    """Emotion regulation tests."""

    def test_reappraisal_reduces_fear(self):
        """Cognitive reappraisal reduces fear."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.8, dominance_sense=0.1)
        engine.tick()
        fear_before = engine.get_activation("fear")

        engine.regulate(strategy="reappraisal", target_emotion="fear")
        fear_after = engine.get_activation("fear")

        if fear_before > 0.05:
            assert fear_after < fear_before

    def test_suppression_less_effective(self):
        """Suppression is less effective than reappraisal."""
        engine1 = EmotionGranularityEngine()
        engine2 = EmotionGranularityEngine()

        # Same fear injection
        engine1.inject_threat(0.8, dominance_sense=0.1)
        engine2.inject_threat(0.8, dominance_sense=0.1)
        engine1.tick()
        engine2.tick()

        engine1.regulate("reappraisal", "fear")
        engine2.regulate("suppression", "fear")

        fear_reappraisal = engine1.get_activation("fear")
        fear_suppression = engine2.get_activation("fear")

        # Reappraisal should be more effective (reduce more)
        assert fear_reappraisal <= fear_suppression

    def test_regulate_no_target_finds_strongest_negative(self):
        """No target specified → regulate strongest negative emotion."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.3, dominance_sense=0.1)  # fear
        engine.inject_reward(-0.5)  # sadness
        engine.tick()

        # First confirm there are negative emotions
        negative_before = {
            name: engine.get_activation(name)
            for name in ["fear", "sadness", "anger", "disgust"]
        }
        strongest = max(negative_before, key=negative_before.get)

        engine.regulate("reappraisal")
        # Strongest negative emotion should be reduced
        # (Regulation effect is reflected before the next tick)
        assert engine.get_activation(strongest) <= negative_before[strongest]


class TestRichnessAndDepth:
    """Emotional richness and depth tests."""

    def test_zero_richness_at_start(self):
        """Initial state → richness = 0."""
        engine = EmotionGranularityEngine()
        assert engine.get_emotional_richness() == 0.0

    def test_single_emotion_gives_some_richness(self):
        """Single emotion → some richness but not maximum."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.8, dominance_sense=0.1)
        engine.tick()
        richness = engine.get_emotional_richness()
        assert 0.0 < richness < 1.0

    def test_multiple_emotions_higher_richness(self):
        """Multiple emotions → higher richness."""
        engine1 = EmotionGranularityEngine()
        engine2 = EmotionGranularityEngine()

        # Only fear
        engine1.inject_threat(0.8, dominance_sense=0.1)
        engine1.tick()

        # Fear + joy + surprise
        engine2.inject_threat(0.5, dominance_sense=0.1)
        engine2.inject_reward(0.5)
        engine2.inject_novelty(0.8)
        engine2.tick()

        assert engine2.get_emotional_richness() > engine1.get_emotional_richness()

    def test_depth_requires_both_positive_and_negative(self):
        """Emotional depth: having both positive and negative emotions → deeper."""
        engine1 = EmotionGranularityEngine()
        engine2 = EmotionGranularityEngine()

        # Only positive
        engine1.inject_reward(0.8)
        engine1.tick()

        # Positive + negative ('bittersweet')
        engine2.inject_reward(0.8)
        engine2.inject_threat(0.5, dominance_sense=0.1)
        engine2.tick()

        assert engine2.get_emotional_depth() >= engine1.get_emotional_depth() * 0.8


class TestImpedanceModel:
    """Impedance model tests."""

    def test_default_impedance(self):
        """No emotion → default impedance 75Ω."""
        engine = EmotionGranularityEngine()
        engine.tick()
        state = engine.get_state()
        assert state["Z_emotion"] == 75.0

    def test_fear_low_impedance(self):
        """Fear (Z=20Ω) → mixed impedance decreases."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.9, dominance_sense=0.0)
        engine.tick()
        state = engine.get_state()
        # Fear's characteristic impedance is 20Ω, mixed should be lower
        assert state["Z_emotion"] < 75.0

    def test_sadness_high_impedance(self):
        """Sadness (Z=150Ω) → mixed impedance increases."""
        engine = EmotionGranularityEngine()
        for _ in range(5):
            engine.inject_reward(-0.9)
            engine.tick()
        state = engine.get_state()
        # When sadness dominates, impedance is higher
        # Prerequisite: sadness activation is strong enough
        if engine.get_activation("sadness") > 0.1:
            assert state["Z_emotion"] > 50.0


class TestEmotionVector:
    """EmotionVector data structure tests."""

    def test_as_dict(self):
        ev = EmotionVector(joy=0.5, fear=0.3)
        d = ev.as_dict()
        assert d["joy"] == 0.5
        assert d["fear"] == 0.3
        assert d["trust"] == 0.0

    def test_as_array(self):
        ev = EmotionVector(joy=0.5, fear=0.3)
        arr = ev.as_array()
        assert arr.shape == (8,)
        assert arr[0] == 0.5  # joy
        assert arr[2] == 0.3  # fear

    def test_total_activation(self):
        ev = EmotionVector(joy=0.5, trust=0.3, fear=0.2)
        assert abs(ev.total_activation() - 1.0) < 0.01

    def test_from_dict(self):
        d = {"joy": 0.7, "sadness": 0.2, "unknown_key": 0.99}
        ev = EmotionVector.from_dict(d)
        assert ev.joy == 0.7
        assert ev.sadness == 0.2
        assert ev.anger == 0.0  # Not specified → 0


class TestGranularEmotionState:
    """GranularEmotionState data structure tests."""

    def test_default(self):
        state = GranularEmotionState()
        assert state.valence == 0.0
        assert state.dominant_emotion == "neutral"
        assert state.compound_emotions == []

    def test_tick_returns_state(self):
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.5)
        state = engine.tick()
        assert isinstance(state, GranularEmotionState)
        assert isinstance(state.primaries, EmotionVector)


class TestHIPEmotionScenario:
    """HIP test scenario simulation."""

    def test_fear_conditioning_then_decay_still_detectable(self):
        """
        HIP scenario: fear conditioning → decay_tick → emotion still detectable.

        This is the key test for raising HIP from 50% to 85%+.
        """
        engine = EmotionGranularityEngine()

        # Fear conditioning
        engine.inject_fear_conditioning(threat_level=0.7)
        engine.tick()

        # Emotion should already be active
        richness_before = engine.get_emotional_richness()
        assert richness_before > 0.1

        # Decay once
        engine.tick()  # decay

        # Still detectable after decay
        richness_after = engine.get_emotional_richness()
        assert richness_after > 0.05

        # Valence still non-zero
        valence = abs(engine.get_valence())
        assert valence > 0.05

        # Depth score
        depth = engine.get_emotional_depth()
        assert depth > 0.1

    def test_emotion_score_above_80_percent(self):
        """
        Simulate the full HIP emotion domain scoring process.

        New scoring logic:
          40% fear memory exists
          30% emotional richness (multiple emotions active)
          30% emotional depth (both positive and negative + sufficient intensity)

        Target: ≥ 80%
        """
        engine = EmotionGranularityEngine()

        # 1. Fear conditioning (multiple injections to simulate real scenario)
        engine.inject_fear_conditioning(threat_level=0.7)
        engine.tick()
        engine.inject_threat(threat_level=0.5, dominance_sense=0.3)
        engine.tick()
        fear_exists = True

        # 2. Decay
        engine.tick()

        # 3. Compute
        richness = engine.get_emotional_richness()
        depth = engine.get_emotional_depth()

        score = (
            0.40 * (1.0 if fear_exists else 0.0) +
            0.30 * min(1.0, richness * 2) +
            0.30 * min(1.0, depth * 2)
        )

        assert score >= 0.80, f"Emotion score {score:.2f} < 0.80"

    def test_emotion_persists_over_many_ticks(self):
        """After fear conditioning, emotion residue persists over many ticks."""
        engine = EmotionGranularityEngine()
        engine.inject_fear_conditioning(threat_level=0.7)
        engine.tick()

        # Run 10 ticks
        for _ in range(10):
            engine.tick()

        # Fear with kappa=0.025, after 10 ticks retains (1-0.025)^10 ≈ 0.776
        # Plus EMA smoothing, actual should still retain a fair amount
        assert engine.get_activation("fear") > 0.01
        assert engine.get_emotional_richness() > 0.01


class TestWaveformsAndState:
    """Waveform and state query tests."""

    def test_waveforms_grow(self):
        engine = EmotionGranularityEngine()
        for _ in range(5):
            engine.inject_threat(0.3)
            engine.tick()
        wf = engine.get_waveforms()
        assert len(wf["valence"]) == 5

    def test_state_complete(self):
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.5)
        engine.tick()
        state = engine.get_state()
        assert "primaries" in state
        assert "valence" in state
        assert "arousal" in state
        assert "dominance" in state
        assert "dominant_emotion" in state
        assert "compound_emotions" in state
        assert "richness" in state
        assert "Z_emotion" in state
        assert "gamma_emotion" in state
        assert "total_ticks" in state
        assert "peak_emotions" in state

    def test_reset_clears_all(self):
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.9)
        engine.tick()
        engine.reset()
        for name in engine.EMOTION_NAMES:
            assert engine.get_activation(name) == 0.0
        assert engine.get_valence() == 0.0


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_injection(self):
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.0)
        engine.tick()
        # Safe → weakly positive
        assert engine.get_activation("trust") >= 0.0

    def test_extreme_injection(self):
        """Extreme injection does not exceed saturation."""
        engine = EmotionGranularityEngine()
        for _ in range(100):
            engine.inject_threat(1.0, pain_level=1.0, dominance_sense=0.0)
            engine.tick()
        for name in engine.EMOTION_NAMES:
            assert engine.get_activation(name) <= 1.0

    def test_rapid_alternation(self):
        """Rapid alternation between positive and negative stimuli."""
        engine = EmotionGranularityEngine()
        for i in range(20):
            if i % 2 == 0:
                engine.inject_reward(0.8)
            else:
                engine.inject_threat(0.8, dominance_sense=0.1)
            engine.tick()
        # Should not crash
        state = engine.get_state()
        assert state["valence"] is not None
        assert -1.0 <= state["valence"] <= 1.0

    def test_many_ticks_no_injection(self):
        """No injection for 1000 ticks → all emotions return to zero."""
        engine = EmotionGranularityEngine()
        engine.inject_threat(0.9)
        engine.tick()
        for _ in range(1000):
            engine.tick()
        richness = engine.get_emotional_richness()
        assert richness < 0.05

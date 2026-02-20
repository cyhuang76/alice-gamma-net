# -*- coding: utf-8 -*-
"""
emotion_granularity.py — Phase 36: Emotional Granularity Engine

Physical Foundation:
  Emotion is not a single dial — it is an 8-dimensional impedance map.
  Each basic emotion corresponds to a characteristic impedance, like electromagnetic waves of different frequencies
  having different propagation speeds in different media.

  Plutchik (1980) emotion wheel defines 8 basic emotions,
  each with its own impedance characteristic Z_e and decay rate τ_e.
  Mixed emotions are produced through parallel impedance superposition —
  just like two waves of different frequencies superimposing in the same medium.

Core Formulas:
  Emotion Vector:
    E = [joy, trust, fear, surprise, sadness, disgust, anger, anticipation]
    Each component ∈ [0, 1]

  Emotional Impedance (characteristic impedance of each emotion):
    Z_e = Z_0 × (1 - E_i)    — stronger emotion → lower impedance → faster conduction
    Z_mix = 1 / Σ(E_i / Z_i)  — parallel mixture

  Valence-Arousal-Dominance (VAD) Space:
    Valence  = Σ(E_i × V_i) / max(Σ E_i, ε)
    Arousal  = Σ(E_i × A_i) / max(Σ E_i, ε)
    Dominance = Σ(E_i × D_i) / max(Σ E_i, ε)

  Differential Decay:
    dE_i/dt = -E_i × κ_i    — κ_i is the characteristic decay rate of emotion i
    Surprise fades fast (κ=0.10), sadness fades slowly (κ=0.008)

  Compound Emotion Detection (Plutchik dyads):
    love = f(joy, trust)
    awe = f(fear, surprise)
    remorse = f(sadness, disgust)
    ...

  Emotional Richness:
    R = 1 - Π(1 - E_i)   — at least one emotion active → R > 0
    Shannon entropy: H = -Σ(p_i × log(p_i))  — diversity measure

  Emotion Regulation (Gross process model):
    Reappraisal: Z_emotion → Z_emotion × (1 + α)  — increase impedance → weaken conduction
    Suppression: E_output × (1 - β)  — weaken output but don't change internal state
    Distraction: switch attention channel → disconnect emotion activation source

Clinical Correspondence:
  - High emotional granularity → protective factor for mental health (Barrett, 2001)
  - Low emotional granularity → alexithymia
  - Emotion mixing → "bittersweet" "love-hate" (Ambivalence)
  - Differential decay → sadness persists for days, surprise fades instantly
  - Emotion regulation failure → Borderline Personality Disorder (BPD)
  - Lack of positive emotion sources → anhedonia

References:
  [1] Plutchik R. (1980). Emotion: A Psychoevolutionary Synthesis.
  [2] Russell J.A. (1980). A circumplex model of affect. JPSP.
  [3] Barrett L.F. (2017). How Emotions Are Made.
  [4] Gross J.J. (2015). Emotion Regulation: Current Status and Future Prospects.
  [5] LeDoux J.E. (1996). The Emotional Brain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

# Plutchik 8 basic emotions with VAD coordinates + characteristic impedance + decay rate
# V = valence [-1, +1], A = arousal [0, 1], D = dominance [0, 1]
# Z = characteristic impedance (Ω), κ = decay_rate per tick
PLUTCHIK_PRIMARIES: Dict[str, Dict[str, float]] = {
    "joy":          {"V": 0.85,  "A": 0.65, "D": 0.70, "Z": 50.0,  "kappa": 0.035},
    "trust":        {"V": 0.60,  "A": 0.30, "D": 0.55, "Z": 80.0,  "kappa": 0.015},
    "fear":         {"V": -0.80, "A": 0.90, "D": 0.10, "Z": 20.0,  "kappa": 0.025},
    "surprise":     {"V": 0.10,  "A": 0.95, "D": 0.40, "Z": 15.0,  "kappa": 0.100},
    "sadness":      {"V": -0.75, "A": 0.20, "D": 0.15, "Z": 150.0, "kappa": 0.008},
    "disgust":      {"V": -0.65, "A": 0.55, "D": 0.60, "Z": 90.0,  "kappa": 0.045},
    "anger":        {"V": -0.70, "A": 0.85, "D": 0.90, "Z": 30.0,  "kappa": 0.050},
    "anticipation": {"V": 0.50,  "A": 0.70, "D": 0.65, "Z": 45.0,  "kappa": 0.040},
}

# Plutchik compound emotions (dyads = blending of adjacent basic emotions)
COMPOUND_EMOTIONS: Dict[str, Tuple[str, str]] = {
    "love":           ("joy", "trust"),
    "submission":     ("trust", "fear"),
    "awe":            ("fear", "surprise"),
    "disapproval":    ("surprise", "sadness"),
    "remorse":        ("sadness", "disgust"),
    "contempt":       ("disgust", "anger"),
    "aggressiveness": ("anger", "anticipation"),
    "optimism":       ("anticipation", "joy"),
}

# Compound emotion detection threshold
COMPOUND_THRESHOLD = 0.15  # Both basic emotions > this value → detected as compound emotion

# Emotion injection gains
THREAT_TO_FEAR_GAIN = 0.8       # Threat → fear mapping gain
THREAT_TO_ANGER_GAIN = 0.3      # Threat → anger (when high dominance)
PAIN_TO_ANGER_GAIN = 0.4        # Pain → anger
PAIN_TO_SADNESS_GAIN = 0.2      # Pain → sadness (despair from sustained pain)
REWARD_TO_JOY_GAIN = 0.7        # Positive reward → joy
REWARD_TO_SADNESS_GAIN = 0.5    # Negative reward → sadness
NOVELTY_TO_SURPRISE_GAIN = 0.6  # Novelty → surprise
SAFETY_TO_TRUST_GAIN = 0.3      # Safety → trust
SOCIAL_TO_TRUST_GAIN = 0.5      # Social bond → trust
CURIOSITY_TO_ANTICIPATION = 0.4 # Curiosity → anticipation
GOAL_TO_JOY_GAIN = 0.6          # Goal achievement → joy
GOAL_TO_ANTICIPATION_GAIN = 0.5 # Goal pursuit → anticipation
FRUSTRATION_TO_ANGER_GAIN = 0.5 # Frustration → anger

# Emotion injection momentum (EMA coefficient — prevents instantaneous emotion jumps)
EMOTION_INJECTION_MOMENTUM = 0.6

# Positive emotion baseline (Alice's non-zero positive emotion floor — "being alive means a faint joy")
POSITIVE_BASELINE = 0.02

# Emotion saturation upper limit
EMOTION_SATURATION = 1.0

# Emotion regulation constants
REAPPRAISAL_STRENGTH = 0.35     # Impedance increase from cognitive reappraisal
SUPPRESSION_STRENGTH = 0.25     # Output attenuation from expression suppression
SUPPRESSION_INTERNAL_LEAK = 0.8 # Internal state retains 80% during suppression (suppression ≠ elimination)

# Emotion history length
MAX_EMOTION_HISTORY = 300

# Epsilon for emotion richness calculation
RICHNESS_EPSILON = 1e-6


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class EmotionVector:
    """8-dimensional Plutchik emotion vector"""
    joy: float = 0.0
    trust: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    sadness: float = 0.0
    disgust: float = 0.0
    anger: float = 0.0
    anticipation: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "joy": round(self.joy, 4),
            "trust": round(self.trust, 4),
            "fear": round(self.fear, 4),
            "surprise": round(self.surprise, 4),
            "sadness": round(self.sadness, 4),
            "disgust": round(self.disgust, 4),
            "anger": round(self.anger, 4),
            "anticipation": round(self.anticipation, 4),
        }

    def as_array(self) -> np.ndarray:
        return np.array([
            self.joy, self.trust, self.fear, self.surprise,
            self.sadness, self.disgust, self.anger, self.anticipation,
        ], dtype=np.float64)

    def total_activation(self) -> float:
        return sum(self.as_dict().values())

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "EmotionVector":
        return cls(**{k: d.get(k, 0.0) for k in [
            "joy", "trust", "fear", "surprise",
            "sadness", "disgust", "anger", "anticipation",
        ]})


@dataclass
class GranularEmotionState:
    """Complete output of the emotion granularity engine"""
    # Plutchik vector
    primaries: EmotionVector = field(default_factory=EmotionVector)

    # VAD space coordinates
    valence: float = 0.0         # [-1, +1]
    arousal: float = 0.0         # [0, 1]
    dominance: float = 0.5       # [0, 1]

    # Dominant emotion
    dominant_emotion: str = "neutral"
    dominant_activation: float = 0.0

    # Compound emotions
    compound_emotions: List[str] = field(default_factory=list)

    # Emotional richness
    richness: float = 0.0        # [0, 1] — how many emotions simultaneously active
    entropy: float = 0.0         # Shannon entropy — emotion diversity

    # Emotional impedance
    Z_emotion: float = 75.0      # Mixed impedance Ω
    gamma_emotion: float = 0.0   # Emotion-context impedance mismatch

    # Regulation state
    regulation_active: bool = False
    regulation_strategy: str = ""


# ============================================================================
# Main Engine
# ============================================================================


class EmotionGranularityEngine:
    """
    Emotion Granularity Engine — from 1D valence to Plutchik emotion space

    Physics:
      Emotion is not a single dimension, but multi-channel impedance resonance.
      Each basic emotion is a "frequency", mixed emotions are "chords".
      Higher emotional granularity (able to distinguish more emotions),
      stronger psychological adaptation (Barrett, 2001).

    Architecture:
      Amygdala (fast threat) → this engine (granularity expansion) → VAD + compound emotions
                ↑                                    ↓
      Prefrontal cortex (regulation) ← ← ← ← ← ← ← ← ← ← ← ← ← ↙
    """

    EMOTION_NAMES = list(PLUTCHIK_PRIMARIES.keys())

    def __init__(self):
        # ---- 8-dimensional emotion vector ----
        self._activations: Dict[str, float] = {
            name: 0.0 for name in self.EMOTION_NAMES
        }

        # ---- VAD state ----
        self._valence: float = 0.0
        self._arousal: float = 0.0
        self._dominance: float = 0.5

        # ---- Emotional impedance ----
        self._Z_emotion: float = 75.0  # Default impedance (matching channel)
        self._gamma_emotion: float = 0.0

        # ---- Emotion injection accumulator (for EMA smoothing) ----
        self._injection_buffer: Dict[str, float] = {
            name: 0.0 for name in self.EMOTION_NAMES
        }

        # ---- Regulation state ----
        self._regulation_active: bool = False
        self._regulation_strategy: str = ""

        # ---- History ----
        self._valence_history: List[float] = []
        self._arousal_history: List[float] = []
        self._richness_history: List[float] = []
        self._emotion_history: List[Dict[str, float]] = []

        # ---- Statistics ----
        self._total_ticks: int = 0
        self._peak_emotions: Dict[str, float] = {
            name: 0.0 for name in self.EMOTION_NAMES
        }
        self._compound_count: Dict[str, int] = {
            name: 0 for name in COMPOUND_EMOTIONS
        }

    # ------------------------------------------------------------------
    # Emotion Injection Interface (multi-source)
    # ------------------------------------------------------------------

    def inject_threat(
        self,
        threat_level: float,
        pain_level: float = 0.0,
        fear_matched: bool = False,
        dominance_sense: float = 0.5,
    ):
        """
        Amygdala → emotion granularity engine

        Threat doesn't only produce fear — it depends on dominance sense:
        - Low dominance + high threat → fear (flight)
        - High dominance + high threat → anger (fight)
        - Sustained pain → sadness (helplessness)
        - Noxious stimuli → disgust

        Args:
            threat_level: Threat level [0, 1]
            pain_level: Pain level [0, 1]
            fear_matched: Whether matching fear memory
            dominance_sense: Dominance sense [0, 1] (high → leans toward anger rather than fear)
        """
        if threat_level < 0.01 and pain_level < 0.01:
            # Safe and painless → faint trust + joy
            self._inject("trust", SAFETY_TO_TRUST_GAIN * 0.3)
            self._inject("joy", POSITIVE_BASELINE)
            return

        # Fear vs anger allocation depends on dominance sense
        fear_weight = 1.0 - dominance_sense
        anger_weight = dominance_sense

        # Fear (low dominance → flight)
        fear_activation = threat_level * THREAT_TO_FEAR_GAIN * fear_weight
        if fear_matched:
            fear_activation *= 1.5  # Fear memory match → amplification
        self._inject("fear", fear_activation)

        # Anger (high dominance → fight)
        anger_activation = threat_level * THREAT_TO_ANGER_GAIN * anger_weight
        if pain_level > 0.3:
            anger_activation += pain_level * PAIN_TO_ANGER_GAIN
        self._inject("anger", anger_activation)

        # Sustained pain → sadness
        if pain_level > 0.2:
            self._inject("sadness", pain_level * PAIN_TO_SADNESS_GAIN)

        # Surprise (if sudden threat)
        if threat_level > 0.5 and not fear_matched:
            self._inject("surprise", threat_level * 0.3)

    def inject_reward(self, reward_prediction_error: float):
        """
        Reward prediction error → joy/sadness

        Positive RPE → exceeded expectation → joy
        Negative RPE → below expectation → sadness

        Args:
            reward_prediction_error: [-1, +1]
        """
        if reward_prediction_error > 0:
            self._inject("joy", reward_prediction_error * REWARD_TO_JOY_GAIN)
            self._inject("anticipation", reward_prediction_error * 0.2)
        elif reward_prediction_error < 0:
            self._inject("sadness", abs(reward_prediction_error) * REWARD_TO_SADNESS_GAIN)

    def inject_novelty(self, surprise_level: float, curiosity_satisfied: bool = False):
        """
        Novelty signal → surprise/anticipation

        Args:
            surprise_level: [0, 1]
            curiosity_satisfied: Curiosity satisfied → anticipation + joy
        """
        if surprise_level > 0.1:
            self._inject("surprise", surprise_level * NOVELTY_TO_SURPRISE_GAIN)

        if curiosity_satisfied:
            self._inject("anticipation", CURIOSITY_TO_ANTICIPATION)
            self._inject("joy", 0.15)  # Small joy from exploration

    def inject_social(
        self,
        empathy_valence: float = 0.0,
        social_bond_strength: float = 0.0,
        rejection: float = 0.0,
    ):
        """
        Social signal → trust/sadness

        Args:
            empathy_valence: Empathy valence [-1, +1]
            social_bond_strength: Social bond strength [0, 1]
            rejection: Social rejection [0, 1]
        """
        if social_bond_strength > 0.1:
            self._inject("trust", social_bond_strength * SOCIAL_TO_TRUST_GAIN)
            self._inject("joy", social_bond_strength * 0.2)

        if rejection > 0.1:
            self._inject("sadness", rejection * 0.4)
            self._inject("anger", rejection * 0.2)

        # Empathic emotion contagion
        if empathy_valence > 0.1:
            self._inject("joy", empathy_valence * 0.2)
            self._inject("trust", empathy_valence * 0.15)
        elif empathy_valence < -0.1:
            self._inject("sadness", abs(empathy_valence) * 0.2)

    def inject_homeostatic(
        self,
        satisfaction: float = 0.0,
        deficit: float = 0.0,
        irritability: float = 0.0,
    ):
        """
        Homeostatic signal → joy/anger/sadness

        Args:
            satisfaction: Need satisfaction [0, 1] (satiety, quenched)
            deficit: Need deficit [0, 1] (hunger, thirst)
            irritability: Irritability [0, 1]
        """
        if satisfaction > 0.1:
            self._inject("joy", satisfaction * 0.3)
            self._inject("trust", satisfaction * 0.15)

        if deficit > 0.2:
            self._inject("anger", irritability * 0.3)
            self._inject("sadness", deficit * 0.15)

    def inject_goal(
        self,
        achievement: float = 0.0,
        frustration: float = 0.0,
        progress: float = 0.0,
    ):
        """
        Goal signal → joy/anticipation/anger

        Args:
            achievement: Goal achievement [0, 1]
            frustration: Frustration level [0, 1]
            progress: Progress sense [0, 1]
        """
        if achievement > 0.1:
            self._inject("joy", achievement * GOAL_TO_JOY_GAIN)
        if progress > 0.1:
            self._inject("anticipation", progress * GOAL_TO_ANTICIPATION_GAIN)
        if frustration > 0.1:
            self._inject("anger", frustration * FRUSTRATION_TO_ANGER_GAIN)
            self._inject("sadness", frustration * 0.2)

    def inject_fear_conditioning(self, threat_level: float = 0.7):
        """
        Fear conditioning event → directly inject fear + surprise

        Called when amygdala executes condition_fear(),
        ensuring fear conditioning is immediately reflected in the emotion vector.

        Args:
            threat_level: Conditioned threat level [0, 1]
        """
        self._inject("fear", threat_level * THREAT_TO_FEAR_GAIN)
        self._inject("surprise", threat_level * 0.3)
        self._inject("sadness", threat_level * 0.1)

    # ------------------------------------------------------------------
    # Emotion Regulation (Prefrontal Interface)
    # ------------------------------------------------------------------

    def regulate(self, strategy: str = "reappraisal", target_emotion: str = ""):
        """
        Emotion regulation — prefrontal top-down control

        Args:
            strategy: "reappraisal" | "suppression" | "distraction"
            target_emotion: Target emotion to regulate (empty = regulate strongest negative emotion)
        """
        self._regulation_active = True
        self._regulation_strategy = strategy

        # Find the emotion to regulate
        if not target_emotion:
            # Find strongest negative emotion
            negative_emotions = {
                name: act for name, act in self._activations.items()
                if PLUTCHIK_PRIMARIES[name]["V"] < 0 and act > 0.05
            }
            if negative_emotions:
                target_emotion = max(negative_emotions, key=negative_emotions.get)
            else:
                self._regulation_active = False
                return

        if target_emotion not in self._activations:
            self._regulation_active = False
            return

        current = self._activations[target_emotion]
        if current < 0.05:
            self._regulation_active = False
            return

        if strategy == "reappraisal":
            # Cognitive reappraisal: increase emotional channel impedance → signal attenuation
            # Most effective, changes internal state
            reduction = REAPPRAISAL_STRENGTH
            self._activations[target_emotion] = max(
                0.0, current * (1.0 - reduction)
            )
        elif strategy == "suppression":
            # Expression suppression: suppress output but internal state persists
            # Less effective, only surface calm (internal retains LEAK proportion)
            internal_reduction = SUPPRESSION_STRENGTH * (1.0 - SUPPRESSION_INTERNAL_LEAK)
            self._activations[target_emotion] = max(
                0.0, current * (1.0 - internal_reduction)
            )
        elif strategy == "distraction":
            # Attention distraction: cut off activation source
            # Temporarily effective
            self._activations[target_emotion] = max(
                0.0, current * 0.7
            )

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def tick(self) -> GranularEmotionState:
        """
        Update emotion state per tick

        1. EMA-smooth injection buffer activations to main vector
        2. Differential decay (different decay rate for each emotion)
        3. Compute VAD coordinates
        4. Detect compound emotions
        5. Compute richness/entropy
        6. Compute mixed impedance
        7. Record history

        Returns:
            GranularEmotionState — Complete emotion state
        """
        self._total_ticks += 1

        # ---- 1. EMA smoothed injection ----
        for name in self.EMOTION_NAMES:
            buffered = self._injection_buffer[name]
            if buffered > 0:
                current = self._activations[name]
                target = min(EMOTION_SATURATION, current + buffered)
                self._activations[name] = (
                    EMOTION_INJECTION_MOMENTUM * current +
                    (1.0 - EMOTION_INJECTION_MOMENTUM) * target
                )
                self._injection_buffer[name] = 0.0  # Clear buffer

        # ---- 2. Differential decay ----
        for name in self.EMOTION_NAMES:
            kappa = PLUTCHIK_PRIMARIES[name]["kappa"]
            if self._activations[name] > 0.001:
                self._activations[name] *= (1.0 - kappa)
            else:
                self._activations[name] = 0.0

        # ---- 3. Positive baseline ----
        # When no threat, maintain faint positive emotion baseline
        total_negative = sum(
            self._activations[name]
            for name in self.EMOTION_NAMES
            if PLUTCHIK_PRIMARIES[name]["V"] < 0
        )
        if total_negative < 0.05:
            self._activations["joy"] = max(
                self._activations["joy"], POSITIVE_BASELINE
            )
            self._activations["trust"] = max(
                self._activations["trust"], POSITIVE_BASELINE * 0.5
            )

        # ---- 4. Saturation limit ----
        for name in self.EMOTION_NAMES:
            self._activations[name] = float(np.clip(
                self._activations[name], 0.0, EMOTION_SATURATION
            ))

        # ---- 5. Compute VAD ----
        total_act = sum(self._activations.values())
        if total_act > RICHNESS_EPSILON:
            self._valence = sum(
                self._activations[name] * PLUTCHIK_PRIMARIES[name]["V"]
                for name in self.EMOTION_NAMES
            ) / total_act
            self._arousal = sum(
                self._activations[name] * PLUTCHIK_PRIMARIES[name]["A"]
                for name in self.EMOTION_NAMES
            ) / total_act
            self._dominance = sum(
                self._activations[name] * PLUTCHIK_PRIMARIES[name]["D"]
                for name in self.EMOTION_NAMES
            ) / total_act
        else:
            self._valence = 0.0
            self._arousal = 0.0
            self._dominance = 0.5

        self._valence = float(np.clip(self._valence, -1.0, 1.0))
        self._arousal = float(np.clip(self._arousal, 0.0, 1.0))
        self._dominance = float(np.clip(self._dominance, 0.0, 1.0))

        # ---- 6. Detect compound emotions ----
        compound_detected = []
        for compound_name, (e1, e2) in COMPOUND_EMOTIONS.items():
            if (self._activations[e1] > COMPOUND_THRESHOLD and
                    self._activations[e2] > COMPOUND_THRESHOLD):
                compound_detected.append(compound_name)
                self._compound_count[compound_name] += 1

        # ---- 7. Dominant emotion ----
        dominant_name = max(self._activations, key=self._activations.get)
        dominant_act = self._activations[dominant_name]
        if dominant_act < 0.01:
            dominant_name = "neutral"
            dominant_act = 0.0

        # ---- 8. Emotional richness ----
        richness = 1.0 - math.prod(
            1.0 - self._activations[name]
            for name in self.EMOTION_NAMES
        )

        # Shannon entropy
        entropy = 0.0
        if total_act > RICHNESS_EPSILON:
            for name in self.EMOTION_NAMES:
                p = self._activations[name] / total_act
                if p > RICHNESS_EPSILON:
                    entropy -= p * math.log2(p)
        # Normalize to [0, 1] (max entropy = log2(8) = 3.0)
        entropy_norm = entropy / 3.0

        # ---- 9. Mixed impedance ----
        z_sum_reciprocal = 0.0
        for name in self.EMOTION_NAMES:
            act = self._activations[name]
            if act > 0.01:
                z_e = PLUTCHIK_PRIMARIES[name]["Z"]
                # Impedance decreases with activation (stronger emotion → faster conduction)
                z_effective = z_e * (1.0 - 0.5 * act)
                z_sum_reciprocal += act / max(z_effective, 1.0)

        total_act_for_z = sum(self._activations.values())
        if z_sum_reciprocal > RICHNESS_EPSILON and total_act_for_z > 0.05:
            self._Z_emotion = 1.0 / z_sum_reciprocal
        else:
            self._Z_emotion = 75.0  # Default (no significant emotion → matching channel impedance)

        # Γ_emotion = emotion-context mismatch
        Z_context = 75.0  # Standard channel impedance
        self._gamma_emotion = abs(
            self._Z_emotion - Z_context
        ) / (self._Z_emotion + Z_context + RICHNESS_EPSILON)

        # ---- 10. Update peak statistics ----
        for name in self.EMOTION_NAMES:
            self._peak_emotions[name] = max(
                self._peak_emotions[name], self._activations[name]
            )

        # ---- 11. Record history ----
        self._valence_history.append(self._valence)
        self._arousal_history.append(self._arousal)
        self._richness_history.append(richness)
        self._emotion_history.append(dict(self._activations))

        for hist in (self._valence_history, self._arousal_history, self._richness_history):
            if len(hist) > MAX_EMOTION_HISTORY:
                del hist[:-MAX_EMOTION_HISTORY]
        if len(self._emotion_history) > MAX_EMOTION_HISTORY:
            del self._emotion_history[:-MAX_EMOTION_HISTORY]

        # ---- 12. Clear regulation flags ----
        reg_active = self._regulation_active
        reg_strategy = self._regulation_strategy
        self._regulation_active = False
        self._regulation_strategy = ""

        # ---- Assemble result ----
        ev = EmotionVector.from_dict(self._activations)

        return GranularEmotionState(
            primaries=ev,
            valence=round(self._valence, 4),
            arousal=round(self._arousal, 4),
            dominance=round(self._dominance, 4),
            dominant_emotion=dominant_name,
            dominant_activation=round(dominant_act, 4),
            compound_emotions=compound_detected,
            richness=round(richness, 4),
            entropy=round(entropy_norm, 4),
            Z_emotion=round(self._Z_emotion, 2),
            gamma_emotion=round(self._gamma_emotion, 4),
            regulation_active=reg_active,
            regulation_strategy=reg_strategy,
        )

    # ------------------------------------------------------------------
    # Query Interface
    # ------------------------------------------------------------------

    def get_valence(self) -> float:
        """Weighted valence"""
        return self._valence

    def get_arousal(self) -> float:
        """Weighted arousal"""
        return self._arousal

    def get_dominance(self) -> float:
        """Weighted dominance"""
        return self._dominance

    def get_activation(self, emotion_name: str) -> float:
        """Get activation of the specified emotion"""
        return self._activations.get(emotion_name, 0.0)

    def get_emotion_vector(self) -> Dict[str, float]:
        """Get 8-dimensional emotion vector"""
        return dict(self._activations)

    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Get dominant emotion (name, activation)"""
        dominant = max(self._activations, key=self._activations.get)
        act = self._activations[dominant]
        if act < 0.01:
            return ("neutral", 0.0)
        return (dominant, act)

    def get_compound_emotions(self) -> List[str]:
        """Get currently active compound emotions"""
        compounds = []
        for compound_name, (e1, e2) in COMPOUND_EMOTIONS.items():
            if (self._activations[e1] > COMPOUND_THRESHOLD and
                    self._activations[e2] > COMPOUND_THRESHOLD):
                compounds.append(compound_name)
        return compounds

    def get_emotional_richness(self) -> float:
        """
        Emotional richness [0, 1]

        Used for HIP scoring — measures how many different emotions the system can feel.
        richness = 1 - Π(1 - E_i)
        """
        return 1.0 - math.prod(
            1.0 - self._activations[name]
            for name in self.EMOTION_NAMES
        )

    def get_emotional_depth(self) -> float:
        """
        Emotional depth [0, 1]

        A composite score combining richness, persistence, and regulation ability.
        Used for HIP emotion domain scoring.
        """
        richness = self.get_emotional_richness()

        # Maximum activation
        max_act = max(self._activations.values())

        # Are both positive and negative emotions present?
        pos_act = sum(
            self._activations[name]
            for name in self.EMOTION_NAMES
            if PLUTCHIK_PRIMARIES[name]["V"] > 0
        )
        neg_act = sum(
            self._activations[name]
            for name in self.EMOTION_NAMES
            if PLUTCHIK_PRIMARIES[name]["V"] < 0
        )
        bipolar = min(pos_act, neg_act) / max(pos_act + neg_act, RICHNESS_EPSILON)

        # Composite
        depth = (
            0.40 * richness +           # Multiple emotions active
            0.30 * min(1.0, max_act) +   # At least one emotion is strong enough
            0.30 * min(1.0, bipolar * 3) # Both positive and negative (emotion is not just positive or negative)
        )
        return float(np.clip(depth, 0.0, 1.0))

    def get_state(self) -> Dict[str, Any]:
        """Complete state dictionary"""
        dominant_name, dominant_act = self.get_dominant_emotion()
        return {
            "primaries": {
                name: round(self._activations[name], 4)
                for name in self.EMOTION_NAMES
            },
            "valence": round(self._valence, 4),
            "arousal": round(self._arousal, 4),
            "dominance": round(self._dominance, 4),
            "dominant_emotion": dominant_name,
            "dominant_activation": round(dominant_act, 4),
            "compound_emotions": self.get_compound_emotions(),
            "richness": round(self.get_emotional_richness(), 4),
            "Z_emotion": round(self._Z_emotion, 2),
            "gamma_emotion": round(self._gamma_emotion, 4),
            "total_ticks": self._total_ticks,
            "peak_emotions": {
                name: round(v, 4)
                for name, v in self._peak_emotions.items()
            },
        }

    def get_waveforms(self, last_n: int = 60) -> Dict[str, Any]:
        """Get emotion waveforms"""
        result: Dict[str, Any] = {
            "valence": self._valence_history[-last_n:],
            "arousal": self._arousal_history[-last_n:],
            "richness": self._richness_history[-last_n:],
        }
        # History of each emotion
        if self._emotion_history:
            for name in self.EMOTION_NAMES:
                result[name] = [
                    h.get(name, 0.0)
                    for h in self._emotion_history[-last_n:]
                ]
        return result

    def reset(self):
        """Reset emotion state"""
        for name in self.EMOTION_NAMES:
            self._activations[name] = 0.0
            self._injection_buffer[name] = 0.0
        self._valence = 0.0
        self._arousal = 0.0
        self._dominance = 0.5
        self._Z_emotion = 75.0
        self._gamma_emotion = 0.0
        self._regulation_active = False
        self._regulation_strategy = ""

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _inject(self, emotion_name: str, activation: float):
        """
        Add activation to injection buffer

        Injection doesn't immediately change emotion vector — at next tick()
        it is smoothly written via EMA. This simulates the physiological inertia of emotions.

        Args:
            emotion_name: Emotion name
            activation: Activation amount [0, ∞) — will be saturation-limited
        """
        if emotion_name in self._injection_buffer:
            self._injection_buffer[emotion_name] += max(0.0, activation)

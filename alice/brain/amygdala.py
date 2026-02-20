# -*- coding: utf-8 -*-
"""
Amygdala — Fast Emotional Pathway & Fight-or-Flight Response (Phase 5.4)

Physics:
  "The amygdala is the brain's smoke detector.
   It doesn't wait for you to think—it makes you jump at the sight of a snake's outline.
   This circuit is 10x faster than the cortex."

  LeDoux (1996) dual-pathway model:
    1. Low road: Sensory → thalamus → amygdala → response (~100ms)
    2. High road: Sensory → thalamus → cortex → amygdala → response (~300ms)

  The amygdala does two things:
    1. Rapid threat assessment: Uses coarse sensory features to judge "dangerous or not?"
    2. Emotional memory: Binds strong emotions to sensory fingerprints (fear conditioning)

  Impedance model of the amygdala:
    - Threat channel: Low impedance (25Ω) → fast conduction, but coarse
    - Safety channel: High impedance (100Ω) → slow conduction, but precise
    - Γ_threat = (Z_signal - Z_threat) / (Z_signal + Z_threat)
    - Low Γ_threat → signal matches "threat template" → fight-or-flight activated

Circuit analogy:
  Amygdala = comparator + flip-flop
  Threat template = reference voltage
  Signal exceeds reference → flip-flop triggers → fight-or-flight response
  Fear conditioning = adjusting the reference voltage threshold

  "You were bitten by a dog once. Now seeing a dog makes your heart race.
   This isn't you thinking—the amygdala's threshold was permanently lowered."
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Physical constants
# ============================================================================

# Threat assessment
THREAT_IMPEDANCE = 25.0         # Ω — Threat channel (low impedance → fast)
SAFETY_IMPEDANCE = 100.0        # Ω — Safety channel (high impedance → slow)
THREAT_THRESHOLD = 0.6          # Γ exceeds this → classified as potential threat
FEAR_THRESHOLD = 0.8            # Γ exceeds this → triggers fight-or-flight

# Valence range
VALENCE_RANGE = (-1.0, 1.0)     # -1=extremely negative, +1=extremely positive

# Fear conditioning
CONDITIONING_RATE = 0.15        # Fear conditioning learning rate
EXTINCTION_RATE = 0.02          # Extinction learning rate (much slower than conditioning, consistent with clinical observations)
MAX_FEAR_MEMORIES = 50          # Maximum number of fear memories

# Emotional decay
EMOTIONAL_DECAY = 0.05          # Emotion decay per tick
EMOTIONAL_MOMENTUM = 0.7        # Emotional momentum (EMA coefficient)

# Fight-or-flight response intensity
FIGHT_FLIGHT_SYMPATHETIC = 0.8  # Sympathetic activation during fight-or-flight
FIGHT_FLIGHT_DURATION = 10      # Fight-or-flight response duration in ticks
FREEZE_THRESHOLD = 0.95         # Threat exceeds this → freeze response (neither fight nor flight)

# Amygdala-hippocampus coupling
EMOTIONAL_MEMORY_BOOST = 0.5    # Memory enhancement multiplier for high-emotion events

# Safety signal
SAFETY_LEARNING_RATE = 0.05     # Safety signal learning rate
SAFETY_THRESHOLD = 0.3          # Safety signal Γ_threat threshold

# Emotion labels
EMOTION_LABELS = {
    (-1.0, -0.6): "terror",
    (-0.6, -0.3): "fear",
    (-0.3, -0.1): "anxiety",
    (-0.1, 0.1): "neutral",
    (0.1, 0.3): "calm",
    (0.3, 0.6): "pleasure",
    (0.6, 1.0): "joy",
}


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class FearMemory:
    """
    Fear memory — amygdala's emotional imprint.

    Binds a specific sensory pattern to a threat level.
    Once established, extinction is very slow.
    """
    modality: str                    # Triggering modality
    fingerprint: np.ndarray          # Triggering fingerprint pattern
    threat_level: float              # Threat level (0~1)
    conditioning_count: int = 1      # Conditioning count (more = deeper)
    extinction_count: int = 0        # Extinction count
    timestamp: float = 0.0          # Creation time
    concept_label: str = ""         # Associated concept (if any)

    @property
    def effective_threat(self) -> float:
        """Effective threat level (considering extinction)."""
        # Extinction reduces threat but never fully eliminates it (trauma never zeroes out)
        extinction_factor = 1.0 / (1.0 + self.extinction_count * 0.1)
        conditioning_boost = min(2.0, 1.0 + self.conditioning_count * 0.1)
        return float(np.clip(
            self.threat_level * extinction_factor * conditioning_boost,
            0.01, 1.0
        ))


@dataclass
class EmotionalState:
    """Amygdala's emotional output."""
    valence: float = 0.0         # Valence (-1~+1)
    arousal_boost: float = 0.0   # Arousal boost
    threat_level: float = 0.0    # Current threat assessment
    is_fight_flight: bool = False  # Whether in fight-or-flight state
    is_freeze: bool = False       # Whether in freeze state
    emotion_label: str = "neutral"  # Emotion label
    trigger: str = ""             # Trigger cause


@dataclass
class AmygdalaResponse:
    """Amygdala full response."""
    emotional_state: EmotionalState
    gamma_threat: float           # Threat impedance reflection coefficient
    fear_matched: bool            # Whether a fear memory was matched
    matched_memory: Optional[str] = None  # Matched fear memory concept
    valence_delta: float = 0.0    # Valence change
    sympathetic_command: float = 0.0  # Command to sympathetic nervous system


# ============================================================================
# Main engine
# ============================================================================


class AmygdalaEngine:
    """
    Amygdala Engine — fast emotional pathway.

    Functions:
    1. Rapid threat assessment (low road: faster than cortex)
    2. Fear conditioning (binding emotions to sensory fingerprints)
    3. Fear extinction (slow safety learning)
    4. Fight-or-flight response activation/maintenance
    5. Emotional valence computation and output

    Amygdala output affects:
    - Autonomic nervous system (sympathetic activation)
    - Hippocampus (emotional memory enhancement)
    - Consciousness (emotion captures attention)
    - Thalamus (threat signal gate enhancement)
    """

    def __init__(self):
        # Emotional state
        self._valence: float = 0.0       # Current valence
        self._arousal_boost: float = 0.0  # Emotional arousal boost
        self._threat_level: float = 0.0   # Current threat level

        # Fear memories
        self._fear_memories: List[FearMemory] = []

        # Fight-or-flight response duration counter
        self._fight_flight_counter: int = 0
        self._is_fight_flight: bool = False
        self._is_freeze: bool = False

        # Safety signals (learning opposite to fear)
        self._safe_patterns: Dict[str, float] = {}  # concept → safety_score

        # Statistics
        self._total_evaluations: int = 0
        self._total_threats: int = 0
        self._total_fight_flights: int = 0
        self._total_freezes: int = 0
        self._total_conditionings: int = 0
        self._total_extinctions: int = 0

        # History
        self._valence_history: List[float] = []
        self._threat_history: List[float] = []
        self._max_history: int = 300

    # ------------------------------------------------------------------
    # Core: rapid threat assessment
    # ------------------------------------------------------------------

    def evaluate(
        self,
        modality: str,
        fingerprint: Optional[np.ndarray] = None,
        gamma: float = 0.0,
        amplitude: float = 0.5,
        pain_level: float = 0.0,
        concept_label: Optional[str] = None,
    ) -> AmygdalaResponse:
        """
        Rapid threat assessment — the amygdala's core function.

        Low-road processing: Uses coarse features to quickly assess threat level.

        Physical model:
          Z_signal = Z_0 / (1 + amplitude)  — high amplitude → low impedance
          Γ_threat = (Z_signal - Z_threat) / (Z_signal + Z_threat)
          Low Γ_threat → signal matches threat channel → fight-or-flight activated

        Additional factors:
          - Pain directly elevates threat level
          - Fear memory match further elevates threat
          - Safety signals reduce threat

        Args:
            modality: Signal modality
            fingerprint: Sensory fingerprint
            gamma: External Γ (impedance mismatch degree)
            amplitude: Signal amplitude
            pain_level: Current pain level
            concept_label: Recognized concept name

        Returns:
            AmygdalaResponse
        """
        self._total_evaluations += 1

        # --- 1. Impedance model: compute Γ_threat ---
        z_signal = SAFETY_IMPEDANCE / (1.0 + amplitude)  # High amplitude → low impedance
        gamma_threat = abs(z_signal - THREAT_IMPEDANCE) / (z_signal + THREAT_IMPEDANCE)
        # gamma_threat low → signal close to threat channel impedance → more likely a threat
        # But we use 1 - gamma_threat as threat score (higher match → more dangerous)
        threat_match = 1.0 - gamma_threat

        # --- 2. Multi-factor threat synthesis ---
        # Pain directly contributes (pain = evidence of physical damage)
        pain_threat = pain_level * 0.8

        # External Γ contribution (high impedance mismatch = unfamiliar = potential danger)
        novelty_threat = gamma * 0.3

        # Fear memory match
        fear_match_score = 0.0
        matched_memory_concept = None
        if fingerprint is not None:
            fm_score, fm_concept = self._check_fear_memories(
                modality, fingerprint, concept_label
            )
            fear_match_score = fm_score
            matched_memory_concept = fm_concept

        # Safety signal reduction
        safety_reduction = 0.0
        if concept_label and concept_label in self._safe_patterns:
            safety_reduction = self._safe_patterns[concept_label] * 0.4

        # Synthesize threat level
        raw_threat = (
            threat_match * 0.2 +
            pain_threat * 0.3 +
            novelty_threat * 0.2 +
            fear_match_score * 0.3
        ) - safety_reduction

        raw_threat = float(np.clip(raw_threat, 0.0, 1.0))

        # --- 3. Update emotional valence ---
        # High threat → negative valence, low threat → neutral/positive valence
        target_valence = -raw_threat  # Simple mapping: threat → negative
        if raw_threat < 0.1 and pain_level < 0.1:
            # Safe and pain-free → mildly positive emotion
            target_valence = 0.1

        old_valence = self._valence
        self._valence += (target_valence - self._valence) * (1.0 - EMOTIONAL_MOMENTUM)
        self._valence = float(np.clip(self._valence, -1.0, 1.0))
        valence_delta = self._valence - old_valence

        # --- 4. Update threat level (smoothed) ---
        self._threat_level += (raw_threat - self._threat_level) * 0.3
        self._threat_level = float(np.clip(self._threat_level, 0.0, 1.0))

        # --- 5. Fight-or-flight / freeze response ---
        sympathetic_cmd = 0.0
        is_fight_flight = False
        is_freeze = False

        if self._threat_level >= FREEZE_THRESHOLD:
            # Extreme threat → freeze response (too scared to move)
            is_freeze = True
            self._is_freeze = True
            self._is_fight_flight = False
            sympathetic_cmd = 0.3  # Freeze also elevates sympathetic but less than fight-or-flight
            if not self._is_freeze:
                self._total_freezes += 1
        elif self._threat_level >= FEAR_THRESHOLD:
            # High threat → fight-or-flight response
            is_fight_flight = True
            self._is_fight_flight = True
            self._is_freeze = False
            self._fight_flight_counter = FIGHT_FLIGHT_DURATION
            self._total_fight_flights += 1
            sympathetic_cmd = FIGHT_FLIGHT_SYMPATHETIC
            self._total_threats += 1
        elif self._fight_flight_counter > 0:
            # Fight-or-flight response ongoing (inertia)
            self._fight_flight_counter -= 1
            is_fight_flight = True
            self._is_fight_flight = True
            sympathetic_cmd = FIGHT_FLIGHT_SYMPATHETIC * (self._fight_flight_counter / FIGHT_FLIGHT_DURATION)
        else:
            self._is_fight_flight = False
            self._is_freeze = False

        # --- 6. Arousal boost ---
        self._arousal_boost = float(np.clip(self._threat_level * 0.6, 0.0, 0.5))

        # --- 7. Emotion label ---
        emotion_label = self._get_emotion_label(self._valence)

        # --- 8. Record history ---
        self._valence_history.append(self._valence)
        self._threat_history.append(self._threat_level)
        for hist in (self._valence_history, self._threat_history):
            if len(hist) > self._max_history:
                del hist[:-self._max_history]

        # Assemble result
        emotional_state = EmotionalState(
            valence=round(self._valence, 4),
            arousal_boost=round(self._arousal_boost, 4),
            threat_level=round(self._threat_level, 4),
            is_fight_flight=is_fight_flight,
            is_freeze=is_freeze,
            emotion_label=emotion_label,
            trigger=concept_label or modality,
        )

        return AmygdalaResponse(
            emotional_state=emotional_state,
            gamma_threat=round(gamma_threat, 4),
            fear_matched=fear_match_score > 0.3,
            matched_memory=matched_memory_concept,
            valence_delta=round(valence_delta, 4),
            sympathetic_command=round(sympathetic_cmd, 4),
        )

    # ------------------------------------------------------------------
    # Fear conditioning
    # ------------------------------------------------------------------

    def condition_fear(
        self,
        modality: str,
        fingerprint: np.ndarray,
        threat_level: float = 0.8,
        concept_label: str = "",
    ):
        """
        Fear conditioning — bind a sensory pattern to a threat.

        Bitten by dog → dog's visual fingerprint bound to threat
        Next time seeing a dog → amygdala automatically triggers fear

        Physics: Lowers the impedance threshold of the threat channel.
        """
        self._total_conditionings += 1

        # Check if a similar fear memory already exists
        existing = self._find_similar_memory(modality, fingerprint, concept_label)

        if existing is not None:
            # Strengthen existing memory (re-conditioning)
            existing.conditioning_count += 1
            existing.threat_level = min(1.0, max(existing.threat_level, threat_level))
        else:
            # Create new fear memory
            memory = FearMemory(
                modality=modality,
                fingerprint=fingerprint.copy(),
                threat_level=float(np.clip(threat_level, 0.0, 1.0)),
                timestamp=time.monotonic(),
                concept_label=concept_label,
            )
            self._fear_memories.append(memory)

            # Capacity management
            if len(self._fear_memories) > MAX_FEAR_MEMORIES:
                # Evict memory with lowest effective threat
                self._fear_memories.sort(key=lambda m: m.effective_threat)
                self._fear_memories.pop(0)

    def extinguish_fear(
        self,
        modality: str,
        fingerprint: np.ndarray,
        concept_label: str = "",
    ) -> bool:
        """
        Fear extinction — repeated exposure in a safe environment.

        Note: Extinction does not "delete" the fear memory,
        but builds a new "safety" memory that inhibits the fear.
        The original fear memory persists forever (physical basis of relapse).

        Returns:
            True if a fear memory was found and updated
        """
        existing = self._find_similar_memory(modality, fingerprint, concept_label)
        if existing is not None:
            existing.extinction_count += 1
            self._total_extinctions += 1

            # Learn safety signal
            if concept_label:
                current = self._safe_patterns.get(concept_label, 0.0)
                self._safe_patterns[concept_label] = min(
                    1.0, current + SAFETY_LEARNING_RATE
                )
            return True
        return False

    # ------------------------------------------------------------------
    # Fear memory query
    # ------------------------------------------------------------------

    def _check_fear_memories(
        self,
        modality: str,
        fingerprint: np.ndarray,
        concept_label: Optional[str] = None,
    ) -> Tuple[float, Optional[str]]:
        """
        Check if the input signal matches any fear memory.

        Returns:
            (match_score, matched_concept_label)
        """
        best_score = 0.0
        best_concept = None

        for fm in self._fear_memories:
            score = 0.0

            # Concept label match (cross-modal fast path)
            if concept_label and fm.concept_label and concept_label == fm.concept_label:
                score = fm.effective_threat * 0.8
            # Fingerprint match (same-modality precise matching)
            elif fm.modality == modality and fm.fingerprint.shape == fingerprint.shape:
                fp_norm = np.linalg.norm(fingerprint)
                fm_norm = np.linalg.norm(fm.fingerprint)
                if fp_norm > 1e-10 and fm_norm > 1e-10:
                    cos_sim = float(np.dot(fingerprint, fm.fingerprint) / (fp_norm * fm_norm))
                    if cos_sim > 0.5:
                        score = cos_sim * fm.effective_threat

            if score > best_score:
                best_score = score
                best_concept = fm.concept_label or fm.modality

        return best_score, best_concept

    def _find_similar_memory(
        self,
        modality: str,
        fingerprint: np.ndarray,
        concept_label: str = "",
    ) -> Optional[FearMemory]:
        """Find the most similar fear memory."""
        best_sim = 0.5  # Minimum similarity threshold
        best_mem = None

        for fm in self._fear_memories:
            # Concept label exact match
            if concept_label and fm.concept_label == concept_label:
                return fm

            # Fingerprint similarity match
            if fm.modality == modality and fm.fingerprint.shape == fingerprint.shape:
                fp_norm = np.linalg.norm(fingerprint)
                fm_norm = np.linalg.norm(fm.fingerprint)
                if fp_norm > 1e-10 and fm_norm > 1e-10:
                    cos_sim = float(np.dot(fingerprint, fm.fingerprint) / (fp_norm * fm_norm))
                    if cos_sim > best_sim:
                        best_sim = cos_sim
                        best_mem = fm

        return best_mem

    # ------------------------------------------------------------------
    # Emotional decay
    # ------------------------------------------------------------------

    def decay_tick(self):
        """
        Natural emotional decay (called each tick).

        Emotions don't last forever—they naturally return to neutral.
        But fight-or-flight response has inertia (doesn't vanish instantly).
        """
        # Valence decays toward 0
        if abs(self._valence) > 0.01:
            self._valence *= (1.0 - EMOTIONAL_DECAY)

        # Threat level decays
        if self._threat_level > 0.01 and not self._is_fight_flight:
            self._threat_level *= (1.0 - EMOTIONAL_DECAY)

        # Arousal boost decays
        if self._arousal_boost > 0.01:
            self._arousal_boost *= (1.0 - EMOTIONAL_DECAY)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _get_emotion_label(valence: float) -> str:
        """Map valence to emotion label."""
        for (lo, hi), label in EMOTION_LABELS.items():
            if lo <= valence < hi:
                return label
        if valence >= 0.6:
            return "joy"
        if valence <= -0.6:
            return "terror"
        return "neutral"

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_valence(self) -> float:
        """Current emotional valence."""
        return self._valence

    def get_threat_level(self) -> float:
        """Current threat level."""
        return self._threat_level

    def get_arousal_boost(self) -> float:
        """Emotional arousal boost."""
        return self._arousal_boost

    def get_emotional_state(self) -> EmotionalState:
        """Get complete emotional state."""
        return EmotionalState(
            valence=round(self._valence, 4),
            arousal_boost=round(self._arousal_boost, 4),
            threat_level=round(self._threat_level, 4),
            is_fight_flight=self._is_fight_flight,
            is_freeze=self._is_freeze,
            emotion_label=self._get_emotion_label(self._valence),
        )

    def get_fear_memories_count(self) -> int:
        """Number of fear memories."""
        return len(self._fear_memories)

    def get_state(self) -> Dict[str, Any]:
        """Get complete amygdala state."""
        return {
            "valence": round(self._valence, 4),
            "arousal_boost": round(self._arousal_boost, 4),
            "threat_level": round(self._threat_level, 4),
            "is_fight_flight": self._is_fight_flight,
            "is_freeze": self._is_freeze,
            "emotion_label": self._get_emotion_label(self._valence),
            "fear_memories": len(self._fear_memories),
            "safe_patterns": len(self._safe_patterns),
            "stats": {
                "total_evaluations": self._total_evaluations,
                "total_threats": self._total_threats,
                "total_fight_flights": self._total_fight_flights,
                "total_freezes": self._total_freezes,
                "total_conditionings": self._total_conditionings,
                "total_extinctions": self._total_extinctions,
            },
        }

    def get_waveforms(self, last_n: int = 60) -> Dict[str, List[float]]:
        """Get emotional waveforms."""
        return {
            "valence": self._valence_history[-last_n:],
            "threat": self._threat_history[-last_n:],
        }

    def reset(self):
        """Reset emotional state (but retain fear memories—trauma cannot be erased)."""
        self._valence = 0.0
        self._arousal_boost = 0.0
        self._threat_level = 0.0
        self._is_fight_flight = False
        self._is_freeze = False
        self._fight_flight_counter = 0
        # Note: _fear_memories are NOT reset!

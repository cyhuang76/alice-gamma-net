# -*- coding: utf-8 -*-
"""
SemanticPressureEngine — Semantic Pressure Engine (Thermodynamics of Language)

Phase 14 → Phase 21 formal integration

Core hypothesis:
  Language is an impedance matching mechanism.
  We speak because the internal "Semantic Pressure" is too high.
  By encoding information into language and releasing it, the system can reduce internal entropy.

Physical definition:
  Speech = ImpedanceMatch(Internal State → External Motor Output)

  Semantic Pressure P_sem = Σ (mass_i × valence_i² × (1 - e^{-arousal}))
    = Accumulated emotional tension of all activated concepts

  Catharsis function:
    ΔP = -P_sem × energy_transfer × consciousness_gate
    energy_transfer = 1 - |Γ_speech|²
    consciousness_gate = Φ (consciousness level: must be awake to express effectively)

Clinical correspondences:
  - Aphasia (Broca lesion): Cannot express → semantic pressure accumulates → anxiety rises
  - Alexithymia: Concepts not grounded → cannot release pressure through language
  - Psychotherapy: The physical mechanism of "it feels better to talk about it"
  - Infant crying: The most primitive impedance matching — internal pain → external sound wave → pressure release

  "Why does she want to speak? Because without speaking, Γ_internal can never be released."
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ============================================================================
# Constants
# ============================================================================

# Pressure accumulation
PRESSURE_ACCUMULATION_RATE = 0.15       # Max pressure increment coefficient per tick
PRESSURE_NATURAL_DECAY = 0.02           # Natural decay rate (2% / tick)
PRESSURE_CONCEPT_NORM = 20.0            # tanh normalization constant

# Pressure release
RELEASE_EFFICIENCY = 0.5                # Max pressure release ratio per expression

# Consciousness gating
CONSCIOUSNESS_MIN_GATE = 0.1            # 10% gate threshold when phi=0
CONSCIOUSNESS_SCALE = 0.9               # Reaches 100% when phi=1

# Inner monologue
DEFAULT_MONOLOGUE_THRESHOLD = 0.3       # Pressure threshold for spontaneous concept activation
MIN_PHI_FOR_MONOLOGUE = 0.3             # Minimum consciousness level

# Wernicke drives Broca threshold
WERNICKE_BROCA_GAMMA_THRESHOLD = 0.3    # Triggers speech production when gamma_syntactic < this

# History tracking
MAX_HISTORY = 600
MAX_MONOLOGUE_EVENTS = 200


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class InnerMonologueEvent:
    """A single inner monologue event"""
    tick: int
    concept: str
    gamma: float
    source: str           # "spontaneous" | "echo" | "association"
    semantic_pressure: float
    phi: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "concept": self.concept,
            "gamma": round(self.gamma, 4),
            "source": self.source,
            "semantic_pressure": round(self.semantic_pressure, 4),
            "phi": round(self.phi, 4),
        }


# ============================================================================
# SemanticPressureEngine
# ============================================================================

class SemanticPressureEngine:
    """
    Semantic Pressure Engine — thermodynamic foundation of language emergence

    Integrates three core functions:
      1. Semantic pressure accumulation/release — physical tracking of emotional tension
      2. Inner monologue — spontaneous activation of concepts in the semantic field
      3. Wernicke→Broca drive — high-confidence predictions trigger speech production

    Impedance physics:
      Semantic pressure = concept mass × emotional valence² × arousal × consciousness gate
      Speech expression = pressure release = impedance matching (Γ_speech → 0)
    """

    def __init__(
        self,
        monologue_threshold: float = DEFAULT_MONOLOGUE_THRESHOLD,
    ):
        # --- Semantic pressure tracking ---
        self.pressure: float = 0.0
        self.peak_pressure: float = 0.0
        self.cumulative_released: float = 0.0
        self.total_expressions: int = 0
        self._monologue_threshold = monologue_threshold

        # --- Inner monologue memory ---
        self.monologue_events: List[InnerMonologueEvent] = []
        self.concept_sequence: List[str] = []

        # --- History tracking ---
        self.pressure_history: List[float] = []
        self.release_history: List[float] = []

        # --- Statistics ---
        self._tick_count: int = 0
        self._total_accumulations: int = 0
        self._total_releases: int = 0
        self._total_monologue_events: int = 0

    # ------------------------------------------------------------------
    # Semantic Pressure Accumulation
    # ------------------------------------------------------------------

    def accumulate(
        self,
        active_concepts: List[Dict[str, Any]],
        valence: float,
        arousal: float,
        phi: float,
        pain: float,
    ) -> float:
        """
        Accumulate semantic pressure per tick.

        Args:
            active_concepts: [{label, mass, Q}] — activated concepts in the semantic field
            valence:  Emotional valence (-1 ~ +1)
            arousal:  Arousal level (0 ~ 1)
            phi:      Consciousness level (0 ~ 1)
            pain:     Pain level (0 ~ 1)

        Returns:
            Current semantic pressure
        """
        self._total_accumulations += 1

        # Emotional tension = |valence|² + pain²
        emotional_tension = valence ** 2 + pain ** 2

        # Concept pressure = Σ mass × Q_norm
        concept_pressure = 0.0
        for c in active_concepts:
            mass = c.get("mass", 0.1)
            q_factor = c.get("Q", 1.0)
            q_norm = 1.0 - 1.0 / max(q_factor, 1.0)
            concept_pressure += mass * q_norm

        # Arousal amplification factor
        arousal_factor = 1.0 - math.exp(-2.0 * arousal)

        # Consciousness gating
        consciousness_gate = (
            CONSCIOUSNESS_MIN_GATE
            + CONSCIOUSNESS_SCALE * min(1.0, max(0.0, phi))
        )

        # tanh compression of concept pressure to 0~1
        norm_concept_p = math.tanh(concept_pressure / PRESSURE_CONCEPT_NORM)

        # Pressure increment
        delta_p = (
            emotional_tension
            * norm_concept_p
            * arousal_factor
            * consciousness_gate
            * PRESSURE_ACCUMULATION_RATE
        )

        # Natural decay
        decay = self.pressure * PRESSURE_NATURAL_DECAY

        self.pressure = max(0.0, self.pressure + delta_p - decay)
        self.peak_pressure = max(self.peak_pressure, self.pressure)

        self.pressure_history.append(self.pressure)
        if len(self.pressure_history) > MAX_HISTORY:
            del self.pressure_history[:-MAX_HISTORY]

        return self.pressure

    # ------------------------------------------------------------------
    # Semantic Pressure Release (Linguistic Catharsis)
    # ------------------------------------------------------------------

    def release(self, gamma_speech: float, phi: float) -> float:
        """
        Release pressure through verbal expression.

        Args:
            gamma_speech: Expression impedance mismatch (0=perfect expression, 1=total mismatch)
            phi:          Consciousness level

        Returns:
            Amount of pressure released
        """
        self._total_releases += 1

        energy_transfer = 1.0 - gamma_speech ** 2
        consciousness_gate = min(1.0, phi)

        released = self.pressure * energy_transfer * consciousness_gate * RELEASE_EFFICIENCY
        self.pressure = max(0.0, self.pressure - released)
        self.cumulative_released += released
        self.total_expressions += 1

        self.release_history.append(released)
        if len(self.release_history) > MAX_HISTORY:
            del self.release_history[:-MAX_HISTORY]

        return released

    # ------------------------------------------------------------------
    # Inner Monologue: Spontaneous Concept Activation
    # ------------------------------------------------------------------

    def check_spontaneous_activation(
        self,
        tick: int,
        semantic_field,
        wernicke,
        valence: float,
        phi: float,
    ) -> Optional[InnerMonologueEvent]:
        """
        Check if spontaneous concept activation (inner monologue) occurs.

        Physical conditions:
          1. Semantic pressure > threshold
          2. Consciousness level Φ > MIN_PHI_FOR_MONOLOGUE
          3. Grounded concepts exist in the semantic field

        Args:
            tick:           Current tick
            semantic_field: SemanticFieldEngine instance
            wernicke:       WernickeEngine instance
            valence:        Current emotional valence
            phi:            Consciousness level

        Returns:
            InnerMonologueEvent or None
        """
        if self.pressure < self._monologue_threshold:
            return None
        if phi < MIN_PHI_FOR_MONOLOGUE:
            return None

        # Get the most activated concepts in the semantic field
        field_state = semantic_field.get_state()
        top_concepts = field_state.get("top_concepts", [])
        if not top_concepts:
            return None

        # Select the concept that resonates most with current emotion
        best_concept = None
        best_score = -1.0

        for c in top_concepts:
            label = c["label"]
            mass = c["mass"]
            activation_score = mass * (self.pressure / (self.pressure + 0.5))
            # Negative emotions bias toward pain-related concepts
            if valence < -0.1 and "hurt" in label.lower():
                activation_score *= 1.5
            if valence < -0.3 and "danger" in label.lower():
                activation_score *= 1.3

            if activation_score > best_score:
                best_score = activation_score
                best_concept = label

        if best_concept is None:
            return None

        # Wernicke observes inner concept → build sequential memory
        obs = wernicke.observe(best_concept)
        gamma_syn = obs.get("gamma_syntactic", 1.0)

        # Determine event source
        if len(self.concept_sequence) == 0:
            source = "spontaneous"
        elif best_concept == self.concept_sequence[-1]:
            source = "echo"
        else:
            source = "association"

        event = InnerMonologueEvent(
            tick=tick,
            concept=best_concept,
            gamma=gamma_syn,
            source=source,
            semantic_pressure=self.pressure,
            phi=phi,
        )

        self.monologue_events.append(event)
        if len(self.monologue_events) > MAX_MONOLOGUE_EVENTS:
            del self.monologue_events[:-MAX_MONOLOGUE_EVENTS]
        self.concept_sequence.append(best_concept)
        self._total_monologue_events += 1

        return event

    # ------------------------------------------------------------------
    # Wernicke → Broca Direct Drive
    # ------------------------------------------------------------------

    def wernicke_drives_broca(
        self,
        wernicke,
        broca,
    ) -> Optional[Dict[str, Any]]:
        """
        When Wernicke's prediction confidence is high enough, drive Broca to prepare speech production.

        Physics:
          gamma_syntactic < WERNICKE_BROCA_GAMMA_THRESHOLD
          → Sequence prediction is sufficiently certain
          → Trigger Broca's motor speech planning

        Returns:
            {predicted_concept, gamma, planned: bool} or None
        """
        prediction = wernicke.predict_next()
        if not prediction.get("predictions"):
            return None

        top = prediction["predictions"][0]
        concept = top["concept"]
        gamma_syn = top["gamma_syntactic"]

        if gamma_syn >= WERNICKE_BROCA_GAMMA_THRESHOLD:
            return None

        # Trigger Broca articulation plan preparation
        if not broca.has_plan(concept):
            plan = broca.plan_utterance(concept)
            planned = plan is not None
        else:
            planned = True

        return {
            "predicted_concept": concept,
            "gamma_syntactic": round(gamma_syn, 4),
            "planned": planned,
            "pressure": round(self.pressure, 4),
        }

    # ------------------------------------------------------------------
    # tick — Per-Frame Update
    # ------------------------------------------------------------------

    def tick(
        self,
        active_concepts: Optional[List[Dict[str, Any]]] = None,
        valence: float = 0.0,
        arousal: float = 0.0,
        phi: float = 0.5,
        pain: float = 0.0,
        semantic_field=None,
        wernicke=None,
        broca=None,
        tick_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute the full semantic pressure engine update per tick.

        Returns:
            {
                pressure, peak_pressure,
                monologue_event, wernicke_broca_drive,
                total_monologue_events, total_expressions,
            }
        """
        self._tick_count += 1

        # 1. Accumulate semantic pressure
        if active_concepts is not None:
            self.accumulate(active_concepts, valence, arousal, phi, pain)

        # 2. Check inner monologue
        monologue_event = None
        if semantic_field is not None and wernicke is not None:
            monologue_event = self.check_spontaneous_activation(
                tick=tick_id,
                semantic_field=semantic_field,
                wernicke=wernicke,
                valence=valence,
                phi=phi,
            )

        # 3. Wernicke→Broca direct drive
        wb_drive = None
        if wernicke is not None and broca is not None:
            wb_drive = self.wernicke_drives_broca(wernicke, broca)

        return {
            "pressure": round(self.pressure, 4),
            "peak_pressure": round(self.peak_pressure, 4),
            "monologue_event": monologue_event.to_dict() if monologue_event else None,
            "wernicke_broca_drive": wb_drive,
            "total_monologue_events": self._total_monologue_events,
            "total_expressions": self.total_expressions,
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return engine state for introspection."""
        return {
            "pressure": round(self.pressure, 4),
            "peak_pressure": round(self.peak_pressure, 4),
            "cumulative_released": round(self.cumulative_released, 4),
            "total_expressions": self.total_expressions,
            "total_monologue_events": self._total_monologue_events,
            "monologue_threshold": self._monologue_threshold,
            "recent_concepts": self.concept_sequence[-10:],
            "tick_count": self._tick_count,
        }

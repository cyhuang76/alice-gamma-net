#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 19 — Social Resonance Engine

alice/brain/social_resonance.py

Core Hypothesis:
  Social interaction is an impedance matching problem between two neural networks.
  "Being understood" = impedance match → maximum energy transfer → maximum stress release.
  "Being ignored" = impedance mismatch → total signal reflection → stress amplification.

Physical Definitions:
  Z_speaker = Z_base / (1 + P)
    → higher pressure → lower impedance → stronger drive to express

  Z_listener = Z_base_high × (1 - empathy × effort)
    → empathy + listening effort → lower impedance → matching the speaker

  Γ_social = |Z_A - Z_B| / (Z_A + Z_B)
    → social reflection coefficient = quantitative measure of communication failure

  η = 1 - |Γ_social|²
    → energy transfer efficiency = physical measure of "being understood"

Theory of Mind Physics:
  Physical implementation of the Sally-Anne paradigm —
  Alice not only remembers "what that person believes" (belief),
  but also "what that person doesn't know" (ignorance).

  False belief = Γ_belief > 0 between belief and reality in the Agent model
  → ToM Level 2 = knowing what you know, knowing what others don't know
  → requires holding two models simultaneously in working memory

Social Homeostasis:
  Loneliness = prolonged unmet social_need
  Moderate social interaction → stress decrease → energy balance
  Excessive social interaction → compassion fatigue → need solitude to recover
  → Physical analogy: social interaction is an energy export channel, but bandwidth-limited

Clinical Correspondence:
  - Autism: permanently high social impedance → matching failure → Γ ≈ 1
  - Social anxiety: Z_self excessively elevated by sympathetic drive
  - Compassion fatigue: therapist = long-term low-impedance listening → energy depletion
  - Intimacy: Z_bond ↓ → Γ ↓ → "understanding each other without translation"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

# Social impedance base (Ω)
Z_SOCIAL_BASE: float = 75.0         # Speaker base impedance
Z_LISTENER_DEFAULT: float = 300.0   # High impedance of untuned listener (indifference)

# Energy transfer coefficients
K_RELEASE: float = 0.20             # Being listened to → stress release rate
K_ABSORB: float = 0.05              # Listening → stress absorption rate (compassion fatigue)
K_REFLECT: float = 0.15             # Being ignored → stress reflection (amplification) rate

# Social homeostasis
SOCIAL_NEED_DECAY: float = 0.003    # Natural growth of social need per tick
SOCIAL_SATIATION: float = 0.02      # Social need satiation per interaction
LONELINESS_THRESHOLD: float = 0.7   # Social need above this value = loneliness
COMPASSION_FATIGUE_RATE: float = 0.01  # Fatigue accumulation from sustained listening
COMPASSION_RECOVERY_RATE: float = 0.005  # Compassion recovery during solitude
MAX_SOCIAL_ENERGY: float = 1.0

# Theory of Mind
TOM_FALSE_BELIEF_DECAY: float = 0.01   # false belief decay over time
BELIEF_UPDATE_RATE: float = 0.3         # Belief update rate after observation
MAX_BELIEFS_PER_AGENT: int = 10         # Maximum beliefs tracked per Agent

# Social synchronization
SYNC_LEARNING_RATE: float = 0.05        # Frequency synchronization learning rate
SYNC_THRESHOLD: float = 0.7            # Sync degree > this value → produces "rapport"

# Trust dynamics
TRUST_GROWTH_RATE: float = 0.02
TRUST_DECAY_RATE: float = 0.005
TRUST_BETRAYAL_PENALTY: float = 0.3

# Cultural resonance
CULTURAL_DRIFT_RATE: float = 0.01       # Cultural drift (frequency alignment)

# History limits
MAX_HISTORY: int = 500
MAX_AGENT_MODELS: int = 8


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Belief:
    """
    A belief — the fundamental unit of Theory of Mind

    An agent believes that subject's property is value.
    reality_value is the actual value (the truth Alice knows).
    If value ≠ reality_value → false belief.
    """
    subject: str              # Subject of belief (e.g., "ball_location")
    value: str                # Value the Agent believes (e.g., "box_A")
    reality_value: str        # Actual value (e.g., "box_B")
    confidence: float = 0.5   # Agent's confidence in this belief (0~1)
    last_updated_tick: int = 0

    @property
    def is_false_belief(self) -> bool:
        """Whether this is a false belief"""
        return self.value != self.reality_value

    @property
    def gamma_belief(self) -> float:
        """
        Belief mismatch — physical measure of false belief

        Γ_belief = 0 → belief is correct
        Γ_belief = 1 → belief is completely wrong (and highly confident)
        """
        if self.value == self.reality_value:
            return 0.0
        return self.confidence  # More confident wrong belief → higher Γ


@dataclass
class SocialAgentModel:
    """
    Complete social model of another Agent — richer than MirrorNeuronEngine's AgentModel

    Not only tracks "what that person did",
    but also "what that person believes" and "what that person doesn't know".
    """
    agent_id: str
    # Emotion state inference
    inferred_valence: float = 0.0       # -1 ~ +1
    inferred_arousal: float = 0.5       # 0 ~ 1
    # Belief tracking (Theory of Mind)
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    # Social connection
    bond_impedance: float = 75.0        # Lower = more intimate (matching Z_SOCIAL_BASE)
    trust: float = 0.5                  # 0 ~ 1
    familiarity: float = 0.0            # 0 ~ 1
    # Interaction statistics
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    # Synchronization state
    sync_frequency: float = 0.0         # Rhythm sync with this Agent
    sync_degree: float = 0.0            # Sync degree (0~1)
    # Time
    first_seen_tick: int = 0
    last_seen_tick: int = 0
    # Pressure transfer history
    cumulative_pressure_given: float = 0.0    # How much pressure relief given
    cumulative_pressure_absorbed: float = 0.0  # How much pressure absorbed from this person


@dataclass
class SocialCouplingResult:
    """Complete result of a social coupling"""
    gamma_social: float         # Social impedance mismatch (0~1)
    energy_transfer: float      # Energy transfer efficiency η (0~1)
    z_speaker: float            # Speaker effective impedance
    z_listener: float           # Listener effective impedance
    pressure_released: float    # Pressure released by speaker
    pressure_absorbed: float    # Pressure absorbed by listener
    pressure_reflected: float   # Reflected amplified pressure
    trust_delta: float = 0.0    # Trust change amount
    sync_delta: float = 0.0     # Sync change amount


@dataclass
class SallyAnneResult:
    """Result of the Sally-Anne test"""
    agent_id: str               # Agent being tested
    subject: str                # Belief subject (e.g., "ball_location")
    agent_believes: str         # Value the Agent believes
    reality: str                # Actual value
    alice_prediction: str       # Alice's prediction of how agent will answer
    prediction_correct: bool    # Whether Alice's prediction is correct
    tom_level: int              # Demonstrated ToM capability level
    confidence: float           # Prediction confidence


@dataclass
class SocialHomeostasisState:
    """Social homeostasis state"""
    social_need: float          # Social need (0~1), high = craving social interaction
    compassion_energy: float    # Compassion energy (0~1), low = compassion fatigue
    is_lonely: bool             # Whether in loneliness state
    is_fatigued: bool           # Whether in compassion fatigue
    optimal_zone: bool          # Whether in optimal social zone


# ============================================================================
# SocialResonanceEngine — Alice's Social Resonance Engine
# ============================================================================

class SocialResonanceEngine:
    """
    Phase 19: Social Resonance Engine — the physical basis for Alice to understand and connect with others

    Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    SocialResonanceEngine                         │
    │                                                                  │
    │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
    │  │  Impedance    │  │  Theory of   │  │  Social               │  │
    │  │  Coupler      │  │  Mind        │  │  Homeostasis          │  │
    │  │  Γ_social     │  │  Beliefs     │  │  Need / Fatigue       │  │
    │  └──────┬───────┘  └──────┬───────┘  └──────────┬────────────┘  │
    │         │                  │                      │              │
    │         ▼                  ▼                      ▼              │
    │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
    │  │  Trust &      │  │  Sally-Anne  │  │  Sync Detection       │  │
    │  │  Reputation   │  │  Paradigm    │  │  Cultural Resonance   │  │
    │  └──────────────┘  └──────────────┘  └───────────────────────┘  │
    └──────────────────────────────────────────────────────────────────┘

    Core Insight:
      Social interaction is not a "function" — it is a physical field.
      Put two Alices together, and impedances will automatically pair or repel.
      Loneliness = vacuum state of the social field, energy has nowhere to flow.
      Intimacy = impedance approaching zero, frequency phase-locked, lossless transfer.
    """

    def __init__(self) -> None:
        # =============================================
        # Social homeostasis
        # =============================================
        self._social_need: float = 0.3       # Initial social need
        self._compassion_energy: float = 1.0  # Initial compassion energy
        self._loneliness_duration: int = 0    # Consecutive loneliness tick count

        # =============================================
        # Social impedance coupling
        # =============================================
        self._z_base: float = Z_SOCIAL_BASE
        self._last_gamma: float = 1.0       # Most recent social Γ
        self._last_eta: float = 0.0         # Most recent energy transfer rate
        self._cumulative_pressure_released: float = 0.0
        self._cumulative_pressure_absorbed: float = 0.0

        # =============================================
        # Theory of Mind — belief tracking
        # =============================================
        self._agent_models: Dict[str, SocialAgentModel] = {}
        self._tom_capacity: float = 0.1     # Initial ToM capacity (synced with MirrorNeuronEngine)
        self._false_belief_detections: int = 0

        # =============================================
        # Synchronization and cultural resonance
        # =============================================
        self._own_social_frequency: float = 1.0  # Alice's own social rhythm (Hz)
        self._global_sync_degree: float = 0.0    # Global sync degree

        # =============================================
        # Global statistics
        # =============================================
        self._tick_count: int = 0
        self._total_interactions: int = 0
        self._total_couplings: int = 0

        # =============================================
        # History
        # =============================================
        self._gamma_history: List[float] = []
        self._eta_history: List[float] = []
        self._social_need_history: List[float] = []
        self._compassion_history: List[float] = []
        self._coupling_results: List[SocialCouplingResult] = []

    # ==================================================================
    # Social impedance coupling — core physics computation
    # ==================================================================

    def compute_speaker_impedance(self, pressure: float) -> float:
        """
        Speaker effective impedance

        Z_A = Z_base / (1 + P)
        Higher pressure → lower impedance → stronger signal → more drive to express
        """
        return self._z_base / (1.0 + max(0.0, pressure))

    def compute_listener_impedance(
        self,
        empathy: float,
        effort: float,
        bond_impedance: float = 0.0,
    ) -> float:
        """
        Listener effective impedance

        Z_B = Z_default × (1 - empathy × effort) × bond_factor
        Higher empathy + more effort → lower impedance → closer to speaker → matching
        Higher intimacy (lower bond_impedance) → additional reduction
        """
        match_factor = empathy * effort
        z = Z_LISTENER_DEFAULT * (1.0 - min(0.8, match_factor))

        # Intimate relationship further lowers impedance ("understanding without translation")
        if bond_impedance > 0:
            bond_factor = bond_impedance / Z_SOCIAL_BASE  # 0~1 range
            z *= (0.5 + 0.5 * bond_factor)  # At peak intimacy, impedance drops another 50%

        return max(5.0, z)

    def couple(
        self,
        speaker_id: str,
        listener_id: str,
        speaker_pressure: float,
        listener_empathy: float,
        listener_effort: float,
        speaker_phi: float = 1.0,
        listener_phi: float = 1.0,
    ) -> SocialCouplingResult:
        """
        Social impedance coupling — energy transfer between two conscious entities

        Γ_social = |Z_A - Z_B| / (Z_A + Z_B)
        η = 1 - |Γ|²
        """
        # Ensure Agent model exists (speaker first, to avoid listener's ensure evicting speaker)
        self._ensure_agent_model(speaker_id)
        self._ensure_agent_model(listener_id)
        # Re-ensure speaker exists (prevent eviction by listener's ensure)
        self._ensure_agent_model(speaker_id)

        speaker_model = self._agent_models[speaker_id]
        listener_model = self._agent_models[listener_id]

        # Compute effective impedance
        z_a = self.compute_speaker_impedance(speaker_pressure)
        z_b = self.compute_listener_impedance(
            listener_empathy,
            listener_effort,
            bond_impedance=listener_model.bond_impedance,
        )

        # Social reflection coefficient Γ
        gamma = abs(z_a - z_b) / (z_a + z_b)
        gamma = float(np.clip(gamma, 0.0, 1.0))

        # Energy transfer efficiency η
        eta = 1.0 - gamma ** 2

        # Pressure changes
        released = speaker_pressure * eta * speaker_phi * K_RELEASE
        reflected = speaker_pressure * (1.0 - eta) * speaker_phi * K_REFLECT
        absorbed = speaker_pressure * eta * listener_phi * K_ABSORB

        # --- Trust dynamics ---
        # Trust needs "effort" and "energy transfer" (eta)
        # Even with impedance matching (due to intimacy), deliberate indifference still damages trust
        trust_delta = 0.0
        if eta > 0.5 and listener_effort > 0.2:
            # Good interaction = efficient transfer + effort
            trust_delta = TRUST_GROWTH_RATE * eta
            speaker_model.positive_interactions += 1
        elif listener_effort < 0.1:
            # Deliberate indifference — trust drops, more pain the more familiar
            betrayal_pain = 1.0 + listener_model.familiarity
            trust_delta = -TRUST_DECAY_RATE * betrayal_pain
            speaker_model.negative_interactions += 1
        elif eta < 0.2:
            trust_delta = -TRUST_DECAY_RATE * (1.0 - eta)
            speaker_model.negative_interactions += 1

        listener_model.trust = float(np.clip(
            listener_model.trust + trust_delta, 0.0, 1.0
        ))

        # --- Sync update ---
        # Good interaction → frequency convergence → sync increase
        sync_delta = SYNC_LEARNING_RATE * eta * (1.0 - listener_model.sync_degree)
        listener_model.sync_degree = float(np.clip(
            listener_model.sync_degree + sync_delta, 0.0, 1.0
        ))

        # --- Intimacy update ---
        # Interaction count → familiarity growth
        listener_model.interaction_count += 1
        listener_model.familiarity = min(
            1.0, listener_model.familiarity + 0.01
        )
        listener_model.last_seen_tick = self._tick_count

        # bond_impedance decreases with interaction quality (more intimate)
        if eta > 0.3:
            listener_model.bond_impedance = max(
                10.0,
                listener_model.bond_impedance - eta * 2.0,
            )

        # Pressure transfer statistics
        listener_model.cumulative_pressure_given += released
        listener_model.cumulative_pressure_absorbed += absorbed

        # --- Compassion energy consumption ---
        self._compassion_energy = max(
            0.0,
            self._compassion_energy - absorbed * COMPASSION_FATIGUE_RATE * 10,
        )

        # --- Social need satisfaction ---
        self._social_need = max(
            0.0,
            self._social_need - SOCIAL_SATIATION * eta,
        )

        # --- Global statistics ---
        self._last_gamma = gamma
        self._last_eta = eta
        self._cumulative_pressure_released += released
        self._cumulative_pressure_absorbed += absorbed
        self._total_couplings += 1

        result = SocialCouplingResult(
            gamma_social=round(gamma, 6),
            energy_transfer=round(eta, 6),
            z_speaker=round(z_a, 2),
            z_listener=round(z_b, 2),
            pressure_released=round(released, 6),
            pressure_absorbed=round(absorbed, 6),
            pressure_reflected=round(reflected, 6),
            trust_delta=round(trust_delta, 6),
            sync_delta=round(sync_delta, 6),
        )

        self._coupling_results.append(result)
        if len(self._coupling_results) > MAX_HISTORY:
            self._coupling_results = self._coupling_results[-MAX_HISTORY:]

        self._gamma_history.append(gamma)
        self._eta_history.append(eta)
        if len(self._gamma_history) > MAX_HISTORY:
            self._gamma_history = self._gamma_history[-MAX_HISTORY:]
            self._eta_history = self._eta_history[-MAX_HISTORY:]

        return result

    # ==================================================================
    # Theory of Mind — belief tracking and Sally-Anne
    # ==================================================================

    def update_belief(
        self,
        agent_id: str,
        subject: str,
        believed_value: str,
        reality_value: str,
        confidence: float = 0.5,
    ) -> Belief:
        """
        Update an Agent's belief

        Example: Sally puts the ball in the basket.
        → update_belief("sally", "ball_location", "basket", "basket", 1.0)

        Then Anne moves the ball to the box. Sally is not present.
        → update_belief("sally", "ball_location", "basket", "box", 1.0)
          (Sally believes the ball is in the basket, but it is actually in the box)
        """
        self._ensure_agent_model(agent_id)
        model = self._agent_models[agent_id]

        belief = Belief(
            subject=subject,
            value=believed_value,
            reality_value=reality_value,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            last_updated_tick=self._tick_count,
        )

        model.beliefs[subject] = belief

        # Prune old beliefs
        if len(model.beliefs) > MAX_BELIEFS_PER_AGENT:
            oldest_key = min(
                model.beliefs,
                key=lambda k: model.beliefs[k].last_updated_tick,
            )
            del model.beliefs[oldest_key]

        # Count false beliefs
        if belief.is_false_belief:
            self._false_belief_detections += 1

        return belief

    def agent_witnesses_event(
        self,
        agent_id: str,
        subject: str,
        new_value: str,
    ) -> Optional[Belief]:
        """
        Agent witnesses event firsthand → update belief to correct value

        Example: Sally sees the ball placed in the box
        → agent_witnesses_event("sally", "ball_location", "box")
        """
        self._ensure_agent_model(agent_id)
        model = self._agent_models[agent_id]

        if subject in model.beliefs:
            belief = model.beliefs[subject]
            belief.value = new_value
            belief.reality_value = new_value
            belief.confidence = min(1.0, belief.confidence + BELIEF_UPDATE_RATE)
            belief.last_updated_tick = self._tick_count
            return belief
        else:
            return self.update_belief(
                agent_id, subject, new_value, new_value, 0.8,
            )

    def update_reality(self, subject: str, new_reality: str) -> None:
        """
        Reality changed → update reality_value for all Agents
        (but don't change their believed value, unless they are present)

        This is the mechanism that generates false beliefs:
        Reality changed, but some Agents don't know.
        """
        for model in self._agent_models.values():
            if subject in model.beliefs:
                model.beliefs[subject].reality_value = new_reality
                model.beliefs[subject].last_updated_tick = self._tick_count

    def sally_anne_test(
        self,
        agent_id: str,
        subject: str,
    ) -> SallyAnneResult:
        """
        Sally-Anne Test — the gold standard of Theory of Mind

        "Where will Sally look for the ball?"

        ToM Level 0: Alice answers with the correct location she knows → no ToM
        ToM Level 1: Alice knows Sally has a different belief → basic ToM
        ToM Level 2: Alice knows that she knows what Sally doesn't know → second-order ToM

        Physical quantification:
          prediction_confidence = tom_capacity × familiarity × belief_confidence
        """
        self._ensure_agent_model(agent_id)
        model = self._agent_models[agent_id]

        belief = model.beliefs.get(subject)

        if belief is None:
            # No tracked belief → Alice defaults to guessing the correct answer
            return SallyAnneResult(
                agent_id=agent_id,
                subject=subject,
                agent_believes="unknown",
                reality="unknown",
                alice_prediction="unknown",
                prediction_correct=False,
                tom_level=0,
                confidence=0.0,
            )

        # Alice's ToM capacity determines her prediction
        if self._tom_capacity < 0.3:
            # ToM Level 0: egocentric → predicts Agent will answer with correct answer
            alice_prediction = belief.reality_value
            tom_level = 0
        elif self._tom_capacity < 0.6:
            # ToM Level 1: can distinguish own and others' beliefs, but sometimes confuses them
            # Probabilistic mix: higher ToM → better at answering agent's belief
            if np.random.random() < self._tom_capacity:
                alice_prediction = belief.value  # Correctly predict agent's belief
            else:
                alice_prediction = belief.reality_value  # Egocentric error
            tom_level = 1
        else:
            # ToM Level 2: fully understands belief differences
            alice_prediction = belief.value
            tom_level = 2

        # Correct prediction = predicting Agent will answer with his own believed value
        prediction_correct = (alice_prediction == belief.value)

        # Confidence
        confidence = float(np.clip(
            self._tom_capacity * model.familiarity * belief.confidence,
            0.0, 1.0,
        ))

        return SallyAnneResult(
            agent_id=agent_id,
            subject=subject,
            agent_believes=belief.value,
            reality=belief.reality_value,
            alice_prediction=alice_prediction,
            prediction_correct=prediction_correct,
            tom_level=tom_level,
            confidence=round(confidence, 4),
        )

    def get_false_beliefs(self, agent_id: str) -> List[Belief]:
        """Get all false beliefs of an Agent"""
        model = self._agent_models.get(agent_id)
        if model is None:
            return []
        return [b for b in model.beliefs.values() if b.is_false_belief]

    def get_gamma_belief(self, agent_id: str) -> float:
        """
        Agent's overall belief mismatch — average Γ of false beliefs

        Γ_belief_avg = mean(Γ_belief_i) for all tracked beliefs
        """
        model = self._agent_models.get(agent_id)
        if model is None or not model.beliefs:
            return 0.0
        gammas = [b.gamma_belief for b in model.beliefs.values()]
        return float(np.mean(gammas))

    # ==================================================================
    # Social homeostasis (Social Homeostasis)
    # ==================================================================

    def get_homeostasis(self) -> SocialHomeostasisState:
        """Get social homeostasis state"""
        is_lonely = self._social_need > LONELINESS_THRESHOLD
        is_fatigued = self._compassion_energy < 0.3
        optimal = 0.2 <= self._social_need <= 0.6 and self._compassion_energy > 0.5

        return SocialHomeostasisState(
            social_need=round(self._social_need, 4),
            compassion_energy=round(self._compassion_energy, 4),
            is_lonely=is_lonely,
            is_fatigued=is_fatigued,
            optimal_zone=optimal,
        )

    # ==================================================================
    # Bidirectional Interaction — Alice vs Alice
    # ==================================================================

    def bidirectional_couple(
        self,
        agent_a_id: str,
        agent_b_id: str,
        pressure_a: float,
        pressure_b: float,
        empathy_a: float,
        empathy_b: float,
        effort_a: float,
        effort_b: float,
        phi_a: float = 1.0,
        phi_b: float = 1.0,
    ) -> Tuple[SocialCouplingResult, SocialCouplingResult]:
        """
        Bidirectional social coupling — two Agents simultaneously acting as speaker and listener

        Returns (A→B result, B→A result)
        """
        # A speaks, B listens
        result_ab = self.couple(
            speaker_id=agent_a_id,
            listener_id=agent_b_id,
            speaker_pressure=pressure_a,
            listener_empathy=empathy_b,
            listener_effort=effort_b,
            speaker_phi=phi_a,
            listener_phi=phi_b,
        )

        # B speaks, A listens
        result_ba = self.couple(
            speaker_id=agent_b_id,
            listener_id=agent_a_id,
            speaker_pressure=pressure_b,
            listener_empathy=empathy_a,
            listener_effort=effort_a,
            speaker_phi=phi_b,
            listener_phi=phi_a,
        )

        return result_ab, result_ba

    # ==================================================================
    # Core tick — update per perception cycle
    # ==================================================================

    def tick(
        self,
        has_social_input: bool = False,
        own_valence: float = 0.0,
        own_arousal: float = 0.5,
        empathy_capacity: float = 0.2,
        tom_capacity: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Per-cycle update — social homeostasis + decay + sync maintenance

        Args:
            has_social_input: Whether this tick has social input
            own_valence: Alice's own emotional valence
            own_arousal: Alice's own arousal level
            empathy_capacity: Empathy capacity from MirrorNeuronEngine
            tom_capacity: ToM capacity from MirrorNeuronEngine
        """
        self._tick_count += 1

        # Sync ToM capacity (synced from MirrorNeuronEngine's developmental results)
        self._tom_capacity = max(self._tom_capacity, tom_capacity)

        # 1. Natural growth of social need (solitude → increasing desire to socialize)
        if not has_social_input:
            self._social_need = min(
                1.0,
                self._social_need + SOCIAL_NEED_DECAY,
            )
            self._loneliness_duration += 1
        else:
            self._loneliness_duration = 0
            self._total_interactions += 1

        # 2. Compassion energy recovery (natural recovery during solitude)
        if not has_social_input:
            self._compassion_energy = min(
                MAX_SOCIAL_ENERGY,
                self._compassion_energy + COMPASSION_RECOVERY_RATE,
            )

        # 3. Agent model decay (long time unseen → estrangement)
        stale_agents = []
        for agent_id, model in self._agent_models.items():
            ticks_since = self._tick_count - model.last_seen_tick
            if ticks_since > 100:
                # Impedance slowly rises
                model.bond_impedance = min(
                    Z_SOCIAL_BASE,
                    model.bond_impedance + 0.1 * (ticks_since / 100),
                )
                # Sync decay
                model.sync_degree = max(
                    0.0, model.sync_degree - 0.001
                )
            # Belief decay
            for belief in model.beliefs.values():
                belief.confidence = max(
                    0.1,
                    belief.confidence - TOM_FALSE_BELIEF_DECAY,
                )

        # 4. Global sync update
        if self._agent_models:
            sync_values = [m.sync_degree for m in self._agent_models.values()]
            self._global_sync_degree = float(np.mean(sync_values))

        # 5. History recording
        self._social_need_history.append(self._social_need)
        self._compassion_history.append(self._compassion_energy)
        if len(self._social_need_history) > MAX_HISTORY:
            self._social_need_history = self._social_need_history[-MAX_HISTORY:]
            self._compassion_history = self._compassion_history[-MAX_HISTORY:]

        homeostasis = self.get_homeostasis()

        return {
            "tick": self._tick_count,
            "social_need": round(self._social_need, 4),
            "compassion_energy": round(self._compassion_energy, 4),
            "is_lonely": homeostasis.is_lonely,
            "is_fatigued": homeostasis.is_fatigued,
            "optimal_zone": homeostasis.optimal_zone,
            "last_gamma": round(self._last_gamma, 4),
            "last_eta": round(self._last_eta, 4),
            "global_sync": round(self._global_sync_degree, 4),
            "tom_capacity": round(self._tom_capacity, 4),
            "tracked_agents": len(self._agent_models),
            "has_social_input": has_social_input,
        }

    # ==================================================================
    # Helper Methods
    # ==================================================================

    def _ensure_agent_model(self, agent_id: str) -> None:
        """Ensure Agent model exists"""
        if agent_id not in self._agent_models:
            if len(self._agent_models) >= MAX_AGENT_MODELS:
                # Remove the least recently interacted
                oldest = min(
                    self._agent_models,
                    key=lambda k: self._agent_models[k].last_seen_tick,
                )
                del self._agent_models[oldest]

            self._agent_models[agent_id] = SocialAgentModel(
                agent_id=agent_id,
                first_seen_tick=self._tick_count,
                last_seen_tick=self._tick_count,
            )

    def sync_from_mirror(
        self,
        mirror_agent_models: Dict[str, Any],
        empathy_capacity: float,
        tom_capacity: float,
    ) -> None:
        """
        Sync Agent models from MirrorNeuronEngine

        MirrorNeuronEngine is the low level (action imitation, emotional contagion),
        SocialResonanceEngine is the high level (belief tracking, social homeostasis).
        Low-level observations need to be uploaded to the high level.
        """
        for agent_id, mirror_model in mirror_agent_models.items():
            self._ensure_agent_model(agent_id)
            model = self._agent_models[agent_id]

            # Sync emotion inference
            if hasattr(mirror_model, 'inferred_emotion'):
                model.inferred_valence = mirror_model.inferred_emotion
                model.inferred_arousal = mirror_model.inferred_arousal
            elif isinstance(mirror_model, dict):
                model.inferred_valence = mirror_model.get("emotion", 0.0)
                model.inferred_arousal = mirror_model.get("arousal", 0.5)

        # Sync capacities
        self._tom_capacity = max(self._tom_capacity, tom_capacity)

    # ==================================================================
    # Query Interface
    # ==================================================================

    def get_social_need(self) -> float:
        """Social need intensity"""
        return self._social_need

    def get_compassion_energy(self) -> float:
        """Compassion energy"""
        return self._compassion_energy

    def get_loneliness_duration(self) -> int:
        """Consecutive loneliness tick count"""
        return self._loneliness_duration

    def get_last_gamma(self) -> float:
        """Most recent social Γ"""
        return self._last_gamma

    def get_last_eta(self) -> float:
        """Most recent energy transfer efficiency"""
        return self._last_eta

    def get_global_sync(self) -> float:
        """Global sync degree"""
        return self._global_sync_degree

    def get_tom_capacity(self) -> float:
        """Current ToM capacity"""
        return self._tom_capacity

    def get_agent_model(self, agent_id: str) -> Optional[SocialAgentModel]:
        """Get social model of a specific Agent"""
        return self._agent_models.get(agent_id)

    def get_all_agent_ids(self) -> List[str]:
        """Get IDs of all tracked Agents"""
        return list(self._agent_models.keys())

    def get_state(self) -> Dict[str, Any]:
        """Complete state"""
        return {
            "tick": self._tick_count,
            "social_need": round(self._social_need, 4),
            "compassion_energy": round(self._compassion_energy, 4),
            "loneliness_duration": self._loneliness_duration,
            "last_gamma": round(self._last_gamma, 4),
            "last_eta": round(self._last_eta, 4),
            "global_sync": round(self._global_sync_degree, 4),
            "tom_capacity": round(self._tom_capacity, 4),
            "false_belief_detections": self._false_belief_detections,
            "total_interactions": self._total_interactions,
            "total_couplings": self._total_couplings,
            "cumulative_released": round(self._cumulative_pressure_released, 4),
            "cumulative_absorbed": round(self._cumulative_pressure_absorbed, 4),
            "tracked_agents": {
                aid: {
                    "valence": round(m.inferred_valence, 3),
                    "arousal": round(m.inferred_arousal, 3),
                    "bond_impedance": round(m.bond_impedance, 2),
                    "trust": round(m.trust, 3),
                    "familiarity": round(m.familiarity, 3),
                    "sync_degree": round(m.sync_degree, 3),
                    "interactions": m.interaction_count,
                    "beliefs": len(m.beliefs),
                    "false_beliefs": sum(
                        1 for b in m.beliefs.values() if b.is_false_belief
                    ),
                }
                for aid, m in self._agent_models.items()
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Statistics"""
        avg_gamma = float(np.mean(self._gamma_history)) if self._gamma_history else 1.0
        avg_eta = float(np.mean(self._eta_history)) if self._eta_history else 0.0

        return {
            "total_ticks": self._tick_count,
            "total_interactions": self._total_interactions,
            "total_couplings": self._total_couplings,
            "avg_gamma": round(avg_gamma, 4),
            "avg_eta": round(avg_eta, 4),
            "social_need": round(self._social_need, 4),
            "compassion_energy": round(self._compassion_energy, 4),
            "tom_capacity": round(self._tom_capacity, 4),
            "false_belief_detections": self._false_belief_detections,
            "tracked_agents": len(self._agent_models),
        }

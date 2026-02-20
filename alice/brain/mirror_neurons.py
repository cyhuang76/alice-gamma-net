# -*- coding: utf-8 -*-
"""
Mirror Neuron System — Social Cognition & Empathy Foundation (Phase 6.1)
Mirror Neuron System — Social Cognition & Empathy Foundation

Physics:
  "Mirror neurons activate in the same impedance resonance pattern
   when you observe another's actions — as if you were performing that action yourself.
   This is not imagination, it is physical resonance."

  Rizzolatti (1996) discovered:
    When a monkey observed another monkey grasping food,
    the same neurons in its own premotor cortex fired.
    → "Seeing = doing it once in the brain"

  Impedance resonance model:
    Alice observes another's action signal Z_observed
    Her own motor/sensory model has a corresponding Z_self
    Resonance degree:
      Γ_mirror = |Z_observed - Z_self| / (Z_observed + Z_self)
      Γ ≈ 0 → perfect resonance → "I completely understand this action"
      Γ >> 0 → mismatch → "This action is unfamiliar to me"

  Three-layer mirror system:
    L1: Motor Mirror
        Observe action → synchronous activation in internal motor model
        → generates "covert imitation urge"

    L2: Emotional Mirror
        Observe emotional expression → synchronous activation in amygdala
        → generates empathy
        Physics: impedance matching of emotional signals → indirect elicitation of internal valence

    L3: Intentional Mirror
        Observe behavior sequences → infer intent in prefrontal cortex
        → generates theory of mind
        Physics: impedance patterns of behavior sequences → goal inference

  Physics formula for empathy:
    empathy_strength = (1 - Γ_mirror) × emotional_resonance × familiarity
    Where:
    - Γ_mirror: impedance mismatch between observed signal and self-model
    - emotional_resonance: resonance of observed emotional valence in own amygdala
    - familiarity: familiarity with this type of action (mirror learning)

  Physics formula for Theory of Mind:
    tom_confidence = Σ(intent_evidence × relevance) / intent_count
    An Agent is modeled as: (goal, belief, emotion) triplet
    Observe behavior sequence → infer most likely (goal, belief, emotion)

  Biological correspondence:
    - F5 mirror area (premotor cortex): action-observation resonance
    - STS (superior temporal sulcus): biological motion recognition
    - TPJ (temporoparietal junction): theory of mind / self-other distinction
    - Anterior Insula: empathy / interoceptive sharing
    - mPFC (medial prefrontal cortex): social reasoning / self-reference
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# Physical constants
# ============================================================================

# --- Motor Mirror ---
MOTOR_MIRROR_IMPEDANCE = 75.0      # Motor mirror channel impedance (Ω)
MIRROR_RESONANCE_THRESHOLD = 0.3   # Γ < this value → "I understand this action"
IMITATION_URGE_SCALE = 0.4         # Resonance → imitation urge scaling
IMITATION_DECAY = 0.05             # Imitation urge natural decay / tick
MOTOR_MIRROR_LEARNING_RATE = 0.03  # Motor mirror learning rate

# --- Emotional Mirror ---
EMOTIONAL_RESONANCE_GAIN = 0.6     # Emotional resonance gain
EMPATHY_DECAY = 0.04               # Empathy decay / tick
EMPATHY_MATURATION_RATE = 0.02     # Empathy capacity maturation rate
INITIAL_EMPATHY_CAPACITY = 0.2     # Infant empathy capacity (needs learning)
EMOTIONAL_CONTAGION_RATE = 0.15    # Emotional contagion rate (primitive empathy)

# --- Intentional Mirror / Theory of Mind ---
TOM_INITIAL_CAPACITY = 0.1         # Infant theory of mind capacity
TOM_MATURATION_RATE = 0.015        # Theory of mind maturation rate
TOM_EVIDENCE_WINDOW = 20           # Behavior evidence sliding window
TOM_INTENT_DECAY = 0.03            # Intent inference decay
MAX_AGENT_MODELS = 5               # Maximum simultaneously tracked Agents

# --- Social Impedance ---
SOCIAL_BOND_BASE = 50.0            # Social bond base impedance (Ω)
SOCIAL_BOND_LEARNING_RATE = 0.02   # Social bond learning rate
SOCIAL_BOND_DECAY = 0.005          # Social bond decay rate / tick

# --- Self-Other Distinction ---
SELF_OTHER_THRESHOLD = 0.25        # Self-other distinction threshold (from Phase 5)


# ============================================================================
# Data structures
# ============================================================================


class MirrorLayer(str, Enum):
    """Three layers of the mirror system"""
    MOTOR = "motor"            # L1: Motor mirror
    EMOTIONAL = "emotional"    # L2: Emotional mirror
    INTENTIONAL = "intentional"  # L3: Intentional mirror


@dataclass
class MirrorEvent:
    """A single mirror resonance event"""
    layer: MirrorLayer           # Which mirror system layer
    agent_id: str                # Observed Agent
    gamma_mirror: float          # Γ_mirror (resonance mismatch)
    resonance_strength: float    # Resonance strength (0~1)
    modality: str                # Triggering modality
    z_observed: float            # Observed impedance
    z_self: float                # Self-model impedance
    tick: int                    # Tick


@dataclass
class EmpathyResponse:
    """Empathy response"""
    target_agent: str            # Target of empathy
    empathy_strength: float      # Empathy strength (0~1)
    resonated_valence: float     # Resonated emotional valence
    emotional_contagion: float   # Emotional contagion degree
    empathy_type: str            # cognitive / affective / motor
    tick: int                    # Tick


@dataclass
class AgentModel:
    """
    Mental model of another Agent — core of Theory of Mind

    Alice's understanding of "that person":
    - Inferred goal
    - Inferred emotion
    - Inferred belief
    - Social bond strength
    """
    agent_id: str
    inferred_goal: str = "unknown"
    inferred_emotion: float = 0.0     # -1 ~ +1 valence
    inferred_arousal: float = 0.5     # 0 ~ 1 arousal
    bond_impedance: float = SOCIAL_BOND_BASE  # Social bond impedance (lower = more intimate)
    familiarity: float = 0.0          # Familiarity (0~1)
    trust: float = 0.5               # Trust (0~1)
    interaction_count: int = 0        # Interaction count
    behavior_history: List[Dict[str, Any]] = field(default_factory=list)
    last_seen_tick: int = 0


@dataclass
class IntentInference:
    """A single intent inference"""
    agent_id: str
    inferred_goal: str
    confidence: float             # Inference confidence (0~1)
    evidence: List[str]           # Supporting evidence
    tick: int


# ============================================================================
# MirrorNeuronEngine — Mirror Neuron System
# ============================================================================


class MirrorNeuronEngine:
    """
    Phase 6.1: Mirror Neuron System — the physical basis for Alice to understand others

    Physical architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    MirrorNeuronEngine                         │
    │                                                              │
    │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
    │  │  L1: Motor    │  │  L2: Emotion │  │  L3: Intentional  │  │
    │  │  Mirror       │  │  Mirror      │  │  Mirror (ToM)     │  │
    │  │  Γ_motor      │──│  Γ_emotion   │──│  Agent Models     │  │
    │  └──────┬───────┘  └───────┬──────┘  └────────┬──────────┘  │
    │         │                  │                   │             │
    │         ▼                  ▼                   ▼             │
    │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
    │  │  Imitation    │  │  Empathy     │  │  Social Bond      │  │
    │  │  Urge         │  │  Response    │  │  Network          │  │
    │  └──────────────┘  └──────────────┘  └───────────────────┘  │
    └──────────────────────────────────────────────────────────────┘

    Core insight:
      Mirror neurons are not "tools for understanding" — they are "impedance resonance".
      When you see someone in pain, you frown yourself,
      not because you are "sympathizing",
      but because your anterior insula physically resonated the same impedance pattern.
      Empathy = impedance matching.
      Social interaction = impedance harmonization.
    """

    def __init__(self) -> None:
        # =============================================
        # L1: Motor Mirror
        # =============================================
        # Internal motor model (impedance of "known actions" per modality)
        self._motor_models: Dict[str, float] = {
            "vocal": MOTOR_MIRROR_IMPEDANCE,
            "manual": MOTOR_MIRROR_IMPEDANCE,
            "facial": MOTOR_MIRROR_IMPEDANCE,
            "locomotion": MOTOR_MIRROR_IMPEDANCE,
        }
        self._imitation_urge: float = 0.0
        self._last_imitated_action: str = ""

        # =============================================
        # L2: Emotional Mirror
        # =============================================
        self._empathy_capacity: float = INITIAL_EMPATHY_CAPACITY
        self._empathic_valence: float = 0.0    # Resonated emotional valence
        self._emotional_contagion: float = 0.0  # Primitive emotional contagion
        self._empathy_history: List[float] = []

        # =============================================
        # L3: Intent Inference / Theory of Mind
        # =============================================
        self._tom_capacity: float = TOM_INITIAL_CAPACITY
        self._agent_models: Dict[str, AgentModel] = {}

        # =============================================
        # Global state
        # =============================================
        self._tick_count: int = 0
        self._total_mirror_events: int = 0
        self._total_empathy_responses: int = 0
        self._total_intent_inferences: int = 0

        # History
        self._mirror_events: List[MirrorEvent] = []
        self._empathy_responses: List[EmpathyResponse] = []
        self._intent_inferences: List[IntentInference] = []
        self._max_history: int = 200

    # ==================================================================
    # L1: Motor mirror
    # ==================================================================

    def observe_action(
        self,
        agent_id: str,
        modality: str,
        observed_impedance: float,
        action_label: str = "",
    ) -> MirrorEvent:
        """
        Observe another's action → compute mirror resonance

        Physical principle:
          Γ_mirror = |Z_observed - Z_self| / (Z_observed + Z_self)

          If Alice's motor model "recognizes" this action (Z close) → resonance
          If completely unfamiliar → mismatch → no mirror effect

          Resonance → generates "imitation urge"
          This explains yawn contagion — when you see a yawn, your jaw muscles
          physically resonate the same impedance pattern.
        """
        if modality not in self._motor_models:
            self._motor_models[modality] = MOTOR_MIRROR_IMPEDANCE

        z_self = self._motor_models[modality]
        z_obs = max(1.0, observed_impedance)

        # Compute mirror resonance
        gamma_mirror = abs(z_obs - z_self) / (z_obs + z_self)
        gamma_mirror = float(np.clip(gamma_mirror, 0.0, 1.0))

        # Resonance strength = 1 - Γ (perfect match → maximum resonance)
        resonance = 1.0 - gamma_mirror

        # Imitation urge: stronger resonance → greater urge
        self._imitation_urge = float(np.clip(
            self._imitation_urge + resonance * IMITATION_URGE_SCALE,
            0.0, 1.0,
        ))
        self._last_imitated_action = action_label or modality

        # Learning: update own motor model (observational learning)
        # Only learn when resonance is sufficient (too much mismatch → noise)
        if gamma_mirror < 0.5:
            self._motor_models[modality] += MOTOR_MIRROR_LEARNING_RATE * (
                z_obs - z_self
            )

        # Update Agent model
        self._ensure_agent_model(agent_id)
        agent = self._agent_models[agent_id]
        agent.interaction_count += 1
        agent.last_seen_tick = self._tick_count
        agent.familiarity = min(1.0, agent.familiarity + 0.01)
        agent.behavior_history.append({
            "tick": self._tick_count,
            "modality": modality,
            "action": action_label,
            "z_observed": z_obs,
            "gamma": gamma_mirror,
        })
        if len(agent.behavior_history) > TOM_EVIDENCE_WINDOW:
            agent.behavior_history = agent.behavior_history[-TOM_EVIDENCE_WINDOW:]

        # Social bond: more interaction → lower impedance → more intimate
        agent.bond_impedance = max(
            10.0,
            agent.bond_impedance - SOCIAL_BOND_LEARNING_RATE * resonance * 10,
        )

        event = MirrorEvent(
            layer=MirrorLayer.MOTOR,
            agent_id=agent_id,
            gamma_mirror=round(gamma_mirror, 4),
            resonance_strength=round(resonance, 4),
            modality=modality,
            z_observed=round(z_obs, 2),
            z_self=round(z_self, 2),
            tick=self._tick_count,
        )
        self._record_mirror_event(event)
        return event

    # ==================================================================
    # L2: Emotional Mirror / Empathy
    # ==================================================================

    def observe_emotion(
        self,
        agent_id: str,
        observed_valence: float,
        observed_arousal: float = 0.5,
        modality: str = "facial",
        signal_impedance: float = 75.0,
    ) -> EmpathyResponse:
        """
        Observe another's emotion → generate empathy response

        Physical principle:
          Empathy = impedance resonance × emotional resonance × empathy capacity × familiarity

          1. Impedance resonance: matching between observed emotional signal and self emotional model
          2. Emotional resonance: indirect activation of observed valence in own amygdala
          3. Empathy capacity: matures with development (infants only have emotional contagion, no cognitive empathy)
          4. Familiarity: easier to empathize with familiar people

          Two types of empathy:
          - Affective empathy: primitive emotional contagion (low level)
            → Seeing a pained expression makes you feel bad
          - Cognitive empathy: understanding another's situation (high level)
            → Requires theory of mind + prefrontal cortex
        """
        self._ensure_agent_model(agent_id)
        agent = self._agent_models[agent_id]

        # Update Agent's emotion model
        agent.inferred_emotion = float(np.clip(observed_valence, -1.0, 1.0))
        agent.inferred_arousal = float(np.clip(observed_arousal, 0.0, 1.0))
        agent.last_seen_tick = self._tick_count
        agent.interaction_count += 1

        # === Impedance resonance computation ===
        z_obs = max(1.0, signal_impedance)
        z_self = MOTOR_MIRROR_IMPEDANCE
        gamma = abs(z_obs - z_self) / (z_obs + z_self)
        resonance = 1.0 - gamma

        # === Affective empathy (primitive emotional contagion) ===
        # Does not require theory of mind, infants already have it
        emotional_contagion = (
            abs(observed_valence)
            * EMOTIONAL_CONTAGION_RATE
            * resonance
        )

        # Contagion direction follows observed valence
        if observed_valence < 0:
            self._emotional_contagion = -emotional_contagion
        else:
            self._emotional_contagion = emotional_contagion

        # === Cognitive empathy ===
        # Requires empathy capacity + theory of mind + familiarity
        cognitive_empathy = (
            self._empathy_capacity
            * self._tom_capacity
            * agent.familiarity
            * resonance
        )

        # === Total empathy strength ===
        empathy_strength = float(np.clip(
            emotional_contagion + cognitive_empathy * EMOTIONAL_RESONANCE_GAIN,
            0.0, 1.0,
        ))

        # Resonated emotional valence (modulated by own current state)
        self._empathic_valence = float(np.clip(
            self._empathic_valence * 0.7 + observed_valence * empathy_strength * 0.3,
            -1.0, 1.0,
        ))

        # Determine empathy type
        if self._empathy_capacity < 0.4:
            empathy_type = "motor"  # Most primitive: muscular imitation
        elif self._tom_capacity < 0.4:
            empathy_type = "affective"  # Affective empathy (emotional contagion without understanding)
        else:
            empathy_type = "cognitive"  # Cognitive empathy (understanding + feeling)

        # Social bond update
        if empathy_strength > 0.2:
            agent.bond_impedance = max(
                10.0,
                agent.bond_impedance - SOCIAL_BOND_LEARNING_RATE * empathy_strength * 5,
            )

        response = EmpathyResponse(
            target_agent=agent_id,
            empathy_strength=round(empathy_strength, 4),
            resonated_valence=round(self._empathic_valence, 4),
            emotional_contagion=round(emotional_contagion, 4),
            empathy_type=empathy_type,
            tick=self._tick_count,
        )

        self._empathy_responses.append(response)
        if len(self._empathy_responses) > self._max_history:
            self._empathy_responses = self._empathy_responses[-self._max_history:]
        self._total_empathy_responses += 1

        # Record mirror event
        mirror_event = MirrorEvent(
            layer=MirrorLayer.EMOTIONAL,
            agent_id=agent_id,
            gamma_mirror=round(gamma, 4),
            resonance_strength=round(resonance, 4),
            modality=modality,
            z_observed=round(z_obs, 2),
            z_self=round(z_self, 2),
            tick=self._tick_count,
        )
        self._record_mirror_event(mirror_event)

        return response

    # ==================================================================
    # L3: Intent Inference / Theory of Mind
    # ==================================================================

    def infer_intent(
        self,
        agent_id: str,
        observed_behavior: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntentInference:
        """
        Observe behavior sequence → infer intent

        Physical principle:
          Behavior sequences form trajectories in "intent space".
          Each intent = an attractor (goal state).
          Whichever attractor the behavior sequence is closest to → inferred as that intent.

          Intent inference confidence = f(
            tom_capacity,       — theory of mind maturity
            evidence_count,     — evidence count
            behavior_consistency, — behavioral consistency
            familiarity,        — familiarity with the Agent
          )

        Intent vocabulary (stage-wise expansion):
          Level 0: present / absent
          Level 1: approach / avoid
          Level 2: explore / communicate / threaten
          Level 3: teach / request / comfort
        """
        self._ensure_agent_model(agent_id)
        agent = self._agent_models[agent_id]

        # Record behavior
        agent.behavior_history.append({
            "tick": self._tick_count,
            "behavior": observed_behavior,
            "context": context or {},
        })
        if len(agent.behavior_history) > TOM_EVIDENCE_WINDOW:
            agent.behavior_history = agent.behavior_history[-TOM_EVIDENCE_WINDOW:]

        agent.last_seen_tick = self._tick_count
        agent.interaction_count += 1

        # === Intent inference ===
        # Collect behavior evidence
        recent_behaviors = [
            h["behavior"] if isinstance(h.get("behavior"), str)
            else h.get("action", "unknown")
            for h in agent.behavior_history[-TOM_EVIDENCE_WINDOW:]
        ]

        # Intent attractor mapping
        intent_scores: Dict[str, float] = {
            "approach": 0.0,
            "avoid": 0.0,
            "explore": 0.0,
            "communicate": 0.0,
            "threaten": 0.0,
            "teach": 0.0,
            "comfort": 0.0,
            "play": 0.0,
        }

        # Behavior → intent evidence mapping
        behavior_intent_map = {
            "approach": ["approach", "move_toward", "reach", "lean_in"],
            "avoid": ["avoid", "retreat", "move_away", "turn_away", "flee"],
            "explore": ["explore", "look_around", "touch", "examine", "babble"],
            "communicate": ["speak", "vocalize", "gesture", "point", "wave"],
            "threaten": ["growl", "loom", "sudden_move", "loud_sound"],
            "teach": ["demonstrate", "repeat", "slow_down", "point_at"],
            "comfort": ["soothe", "gentle_touch", "soft_voice", "hold"],
            "play": ["play", "bounce", "laugh", "peek"],
        }

        # Compute evidence score for each intent
        evidence_list = []
        for behavior in recent_behaviors:
            for intent, keywords in behavior_intent_map.items():
                for kw in keywords:
                    if kw in behavior.lower():
                        intent_scores[intent] += 1.0
                        evidence_list.append(f"{behavior}→{intent}")
                        break

        # Emotion is also evidence for intent
        if agent.inferred_emotion < -0.3:
            intent_scores["avoid"] += 0.5
            intent_scores["threaten"] += 0.3
        elif agent.inferred_emotion > 0.3:
            intent_scores["approach"] += 0.5
            intent_scores["communicate"] += 0.3
            intent_scores["play"] += 0.3

        # Select highest-scoring intent
        if max(intent_scores.values()) > 0:
            best_intent = max(intent_scores, key=intent_scores.get)  # type: ignore
        else:
            best_intent = "unknown"

        # Confidence
        evidence_count = len(evidence_list)
        behavior_consistency = (
            intent_scores.get(best_intent, 0) / max(evidence_count, 1)
            if evidence_count > 0 else 0.0
        )
        raw_confidence = (
            self._tom_capacity * 0.4
            + min(1.0, evidence_count / 5.0) * 0.3
            + behavior_consistency * 0.2
            + agent.familiarity * 0.1
        )
        confidence = float(np.clip(raw_confidence, 0.0, 1.0))

        # Update Agent model
        agent.inferred_goal = best_intent

        inference = IntentInference(
            agent_id=agent_id,
            inferred_goal=best_intent,
            confidence=round(confidence, 4),
            evidence=evidence_list[-5:],
            tick=self._tick_count,
        )

        self._intent_inferences.append(inference)
        if len(self._intent_inferences) > self._max_history:
            self._intent_inferences = self._intent_inferences[-self._max_history:]
        self._total_intent_inferences += 1

        return inference

    # ==================================================================
    # Empathy capacity & Theory of Mind maturation
    # ==================================================================

    def mature(
        self,
        social_interaction: bool = False,
        positive_feedback: bool = False,
    ) -> None:
        """
        Development and maturation of empathy capacity and theory of mind

        Physical principle:
          Empathy and ToM capacity improve with social experience.
          Social interaction → repeated mirror system activation → threshold lowering → more sensitive

          Like muscles: the more you practice, the stronger they get.
          Infants only have emotional contagion (see crying, start crying).
          Gradually develop cognitive empathy (understand why others cry).
          Eventually develop theory of mind (infer others' beliefs and intentions).
        """
        if social_interaction:
            # Empathy capacity growth
            growth = EMPATHY_MATURATION_RATE
            if positive_feedback:
                growth *= 1.5  # Positive feedback accelerates growth

            self._empathy_capacity = min(
                1.0,
                self._empathy_capacity + growth * (1 - self._empathy_capacity),
            )

            # Theory of mind growth (slower than empathy)
            self._tom_capacity = min(
                1.0,
                self._tom_capacity + TOM_MATURATION_RATE * (1 - self._tom_capacity),
            )

    # ==================================================================
    # Core tick — update per perception cycle
    # ==================================================================

    def tick(
        self,
        has_social_input: bool = False,
        own_valence: float = 0.0,
        own_arousal: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Update per perception cycle — decay + maintenance

        Args:
            has_social_input: Whether this tick has social input
            own_valence: Alice's own emotional valence
            own_arousal: Alice's own arousal level
        """
        self._tick_count += 1

        # 1. Imitation urge decay
        self._imitation_urge = max(0.0, self._imitation_urge - IMITATION_DECAY)

        # 2. Empathy decay
        self._empathic_valence *= (1.0 - EMPATHY_DECAY)
        self._emotional_contagion *= (1.0 - EMPATHY_DECAY)

        # 3. Social bond decay (long time without interaction → estrangement)
        for agent in self._agent_models.values():
            ticks_since = self._tick_count - agent.last_seen_tick
            if ticks_since > 50:
                agent.bond_impedance = min(
                    SOCIAL_BOND_BASE,
                    agent.bond_impedance + SOCIAL_BOND_DECAY * ticks_since * 0.01,
                )

        # 4. If social input → trigger maturation
        if has_social_input:
            self.mature(social_interaction=True)

        # 5. History recording
        self._empathy_history.append(abs(self._empathic_valence))
        if len(self._empathy_history) > self._max_history:
            self._empathy_history = self._empathy_history[-self._max_history:]

        return {
            "tick": self._tick_count,
            "imitation_urge": round(self._imitation_urge, 4),
            "empathic_valence": round(self._empathic_valence, 4),
            "emotional_contagion": round(self._emotional_contagion, 4),
            "empathy_capacity": round(self._empathy_capacity, 4),
            "tom_capacity": round(self._tom_capacity, 4),
            "tracked_agents": len(self._agent_models),
            "has_social_input": has_social_input,
        }

    # ==================================================================
    # Auxiliary methods
    # ==================================================================

    def _ensure_agent_model(self, agent_id: str) -> None:
        """Ensure Agent model exists"""
        if agent_id not in self._agent_models:
            if len(self._agent_models) >= MAX_AGENT_MODELS:
                # Remove the one with longest inactivity
                oldest = min(
                    self._agent_models,
                    key=lambda k: self._agent_models[k].last_seen_tick,
                )
                del self._agent_models[oldest]

            self._agent_models[agent_id] = AgentModel(
                agent_id=agent_id,
                last_seen_tick=self._tick_count,
            )

    def _record_mirror_event(self, event: MirrorEvent) -> None:
        """Record mirror event"""
        self._mirror_events.append(event)
        if len(self._mirror_events) > self._max_history:
            self._mirror_events = self._mirror_events[-self._max_history:]
        self._total_mirror_events += 1

    # ==================================================================
    # Query interface
    # ==================================================================

    def get_imitation_urge(self) -> float:
        """Imitation urge intensity"""
        return self._imitation_urge

    def get_empathic_valence(self) -> float:
        """Resonated emotional valence"""
        return self._empathic_valence

    def get_empathy_capacity(self) -> float:
        """Empathy capacity"""
        return self._empathy_capacity

    def get_tom_capacity(self) -> float:
        """Theory of mind capacity"""
        return self._tom_capacity

    def get_agent_model(self, agent_id: str) -> Optional[AgentModel]:
        """Get mental model of a specific Agent"""
        return self._agent_models.get(agent_id)

    def get_social_bonds(self) -> Dict[str, float]:
        """Get impedance of all social bonds (lower = more intimate)"""
        return {
            aid: round(agent.bond_impedance, 2)
            for aid, agent in self._agent_models.items()
        }

    def get_state(self) -> Dict[str, Any]:
        """Complete state"""
        return {
            "tick": self._tick_count,
            "imitation_urge": round(self._imitation_urge, 4),
            "empathic_valence": round(self._empathic_valence, 4),
            "emotional_contagion": round(self._emotional_contagion, 4),
            "empathy_capacity": round(self._empathy_capacity, 4),
            "tom_capacity": round(self._tom_capacity, 4),
            "tracked_agents": len(self._agent_models),
            "motor_models": {k: round(v, 2) for k, v in self._motor_models.items()},
            "agent_models": {
                aid: {
                    "goal": a.inferred_goal,
                    "emotion": round(a.inferred_emotion, 2),
                    "bond_impedance": round(a.bond_impedance, 2),
                    "familiarity": round(a.familiarity, 3),
                    "trust": round(a.trust, 2),
                    "interactions": a.interaction_count,
                }
                for aid, a in self._agent_models.items()
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Statistics"""
        return {
            "total_ticks": self._tick_count,
            "total_mirror_events": self._total_mirror_events,
            "total_empathy_responses": self._total_empathy_responses,
            "total_intent_inferences": self._total_intent_inferences,
            "empathy_capacity": round(self._empathy_capacity, 4),
            "tom_capacity": round(self._tom_capacity, 4),
            "tracked_agents": len(self._agent_models),
            "social_bonds": self.get_social_bonds(),
        }

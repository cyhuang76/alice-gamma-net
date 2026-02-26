# -*- coding: utf-8 -*-
"""
CuriosityDriveEngine — Phase 9: Curiosity, Boredom & Self-Awareness

    "Free will = spontaneous behavior driven by internal impedance mismatch in the absence of external commands"

Physical basis:
  1. Novelty = mismatch between sensory impedance and internal model
     Γ_novelty = |Z_input - Z_model| / (Z_input + Z_model)
     High Γ = novelty → dopamine release → curiosity reward

  2. Boredom = pressure accumulation from prolonged low novelty
     dP_boredom/dt = BOREDOM_RATE × (1 - novelty)
     Exceeds threshold → spontaneous behavior generated

  3. Self-Recognition = Efference Copy
     Alice predicts her own voice before speaking → compares after hearing
     Γ_self = |Z_predicted - Z_actual| / (Z_predicted + Z_actual)
     Γ_self ≈ 0 → "This is my voice"
     Γ_self >> 0 → "This is an external sound"

  4. Spontaneous Action = boredom-driven internal oscillation
     When no external stimulus, the system generates spontaneous oscillation
     Lowest energy oscillation mode → becomes spontaneous behavior
     Analogy: infant babbling is not random, it is the lowest energy mode of vocal cord resonance

  5. Intrinsic Motivation = information gain as reward signal
     reward_intrinsic = Δ(model_accuracy) × CURIOSITY_SCALE
     → fed into dopamine system → reinforces exploratory behavior

Biological correspondence:
  - Anterior Cingulate Cortex (ACC): conflict monitoring → novelty detection
  - Ventral Tegmental Area (VTA): dopamine → curiosity reward
  - Anterior Insula: interoception → boredom awareness
  - Premotor Cortex (PMC): efference copy → self-recognition
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

# --- Novelty ---
NOVELTY_DECAY = 0.04               # Novelty natural decay / tick (habituation)
NOVELTY_WINDOW = 50                # Sliding window size (for signal variance computation)
MODEL_LEARNING_RATE = 0.02         # Internal model learning rate
MODEL_IMPEDANCE_INIT = 75.0        # Internal model initial impedance (Ω)

# --- Boredom ---
BOREDOM_ACCUMULATION = 0.003       # Boredom pressure accumulation rate / tick during low stimulation
BOREDOM_RELEASE = 0.02             # Rate at which novel stimuli relieve boredom
BOREDOM_THRESHOLD = 0.55           # Boredom threshold for triggering spontaneous behavior
BOREDOM_SATURATION = 1.0           # Boredom upper limit

# --- Curiosity ---
CURIOSITY_REWARD_SCALE = 0.3       # Novelty → intrinsic reward scaling factor
OPTIMAL_STIMULATION = 0.4          # Optimal stimulation level (homeostasis target)
CURIOSITY_DECAY = 0.01             # Curiosity natural decay

# --- Self-Recognition ---
EFFERENCE_COPY_WINDOW = 5          # Efference copy validity window (ticks)
SELF_RECOGNITION_THRESHOLD = 0.25  # Γ_self < this value → judged as self
SELF_MODEL_LEARNING_RATE = 0.05    # Self-model learning rate
INITIAL_SELF_ACCURACY = 0.3        # Infant self-recognition accuracy (needs learning)

# --- Spontaneous Action ---
SPONTANEOUS_COOLDOWN = 10          # Minimum interval between spontaneous actions (ticks)
BABBLE_ENERGY_COST = 0.005         # Energy cost of babbling
EXPLORE_ENERGY_COST = 0.008        # Energy cost of exploration
URGE_DECAY = 0.03                  # Spontaneous urge decay rate


# ============================================================================
# Data structures
# ============================================================================


class SpontaneousActionType(str, Enum):
    """Spontaneous action types — autonomous behaviors Alice may exhibit without external commands"""
    BABBLE = "babble"                  # Babbling (vocal exploration)
    EXPLORE_VISUAL = "explore_visual"  # Visual exploration (looking around)
    EXPLORE_MOTOR = "explore_motor"    # Motor exploration (random hand movements)
    SEEK_NOVELTY = "seek_novelty"      # Actively seeking novel stimuli
    SELF_EXAMINE = "self_examine"      # Self-examination (observing own body)
    IDLE_DREAM = "idle_dream"          # Daydreaming (internal simulation)


@dataclass
class NoveltyEvent:
    """A single novelty assessment event"""
    modality: str               # Modality (visual/auditory/motor)
    novelty_score: float        # Novelty score (0~1)
    information_gain: float     # Information gain
    gamma_novelty: float        # Γ_novelty (impedance mismatch)
    z_input: float              # Input signal impedance
    z_model: float              # Internal model impedance
    tick: int                   # Tick of occurrence


@dataclass
class SpontaneousAction:
    """A single spontaneous action"""
    action_type: SpontaneousActionType
    intensity: float            # Action intensity (0~1)
    trigger: str                # Trigger reason
    boredom_level: float        # Boredom level at trigger time
    tick: int                   # Tick of occurrence


@dataclass
class SelfOtherJudgment:
    """A single self/other discrimination judgment"""
    modality: str               # Modality
    gamma_self: float           # Γ_self (predicted-actual mismatch)
    z_predicted: float          # Predicted impedance
    z_actual: float             # Actual impedance
    is_self: bool               # Judgment: is it self?
    confidence: float           # Judgment confidence (0~1)
    tick: int                   # Tick of occurrence


@dataclass
class InternalGoal:
    """An internal goal driven by curiosity"""
    description: str            # Goal description
    source: str                 # Source (curiosity / boredom / self_explore)
    priority: float             # Priority (0~1)
    created_tick: int           # Creation tick
    satisfied: bool = False     # Whether satisfied


# ============================================================================
# CuriosityDriveEngine — Curiosity, Boredom & Self-Awareness Engine
# ============================================================================


class CuriosityDriveEngine:
    """
    Phase 9: Curiosity Drive Engine — the core of Alice's free will

    Physical architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                  CuriosityDriveEngine                     │
    │                                                          │
    │  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐  │
    │  │   Novelty    │   │   Boredom    │   │    Self      │  │
    │  │  Detector    │──→│ Accumulator  │──→│ Recognition  │  │
    │  │  Γ_novelty   │   │  Pressure    │   │ Efference    │  │
    │  └──────┬──────┘   └──────┬───────┘   └──────┬───────┘  │
    │         │                 │                    │          │
    │         ▼                 ▼                    ▼          │
    │  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐  │
    │  │  Curiosity   │   │ Spontaneous  │   │  Internal    │  │
    │  │  Reward      │   │   Action     │   │   Goals      │  │
    │  │  → Dopamine  │   │  Generator   │   │  Generator   │  │
    │  └─────────────┘   └──────────────┘   └──────────────┘  │
    └──────────────────────────────────────────────────────────┘

    Core insight:
      Curiosity is not a "feature" — it is a consequence of "impedance homeostasis".
      Just as the body needs to maintain 37°C, the brain needs to maintain optimal information flow.
      Too little stimulation → boredom → active exploration
      Too much stimulation → overload → withdrawal
      This is the physical basis of free will.
    """

    def __init__(self) -> None:
        # =============================================
        # Novelty Detector
        # =============================================
        self._novelty_level: float = 0.0          # Current novelty level (0~1)
        self._novelty_history: List[float] = []   # Novelty history
        self._max_history: int = 300

        # Internal model: expected impedance per modality
        # Compare with model when new signal arrives → Γ_novelty
        self._internal_model: Dict[str, float] = {
            "visual": MODEL_IMPEDANCE_INIT,
            "auditory": MODEL_IMPEDANCE_INIT,
            "motor": MODEL_IMPEDANCE_INIT,
        }

        # Signal fingerprint history (for computing information gain)
        self._signal_history: Dict[str, List[float]] = {
            "visual": [],
            "auditory": [],
            "motor": [],
        }

        # =============================================
        # Boredom Accumulator
        # =============================================
        self._boredom_pressure: float = 0.0       # Boredom pressure (0~1)
        self._boredom_history: List[float] = []
        self._ticks_without_input: int = 0        # Consecutive ticks without input
        self._last_input_tick: int = 0            # Last tick with input received

        # =============================================
        # Curiosity Drive
        # =============================================
        self._curiosity_drive: float = 0.0        # Curiosity drive intensity (0~1)
        self._intrinsic_reward: float = 0.0       # Current intrinsic reward signal
        self._total_intrinsic_reward: float = 0.0 # Cumulative intrinsic reward
        self._curiosity_history: List[float] = []

        # =============================================
        # Self-Recognition
        # =============================================
        self._self_recognition_accuracy: float = INITIAL_SELF_ACCURACY
        self._efference_copies: List[Dict[str, Any]] = []  # Efference copy buffer

        # Self-model: Alice's expected impedance for her own signals
        self._self_model: Dict[str, float] = {
            "vocal": 75.0,          # Own vocal impedance
            "motor": 75.0,          # Own motor impedance
        }

        # Self/other judgment history
        self._self_other_judgments: List[SelfOtherJudgment] = []
        self._total_self_correct: int = 0
        self._total_self_judgments: int = 0

        # =============================================
        # Spontaneous Action Generator
        # =============================================
        self._spontaneous_urge: float = 0.0       # Spontaneous action urge (0~1)
        self._last_spontaneous_tick: int = -SPONTANEOUS_COOLDOWN
        self._spontaneous_history: List[SpontaneousAction] = []
        self._total_spontaneous_actions: int = 0

        # =============================================
        # Internal Goals
        # =============================================
        self._internal_goals: List[InternalGoal] = []
        self._max_goals: int = 3

        # =============================================
        # Global counters
        # =============================================
        self._tick_count: int = 0
        self._total_novelty_events: int = 0
        self._stimulation_level: float = 0.0      # Current overall stimulation level

        # =============================================
        # Novelty event log
        # =============================================
        self._recent_novelty_events: List[NoveltyEvent] = []

    # ==================================================================
    # Novelty evaluation
    # ==================================================================

    def evaluate_novelty(
        self,
        modality: str,
        signal_impedance: float,
        signal_amplitude: float = 0.5,
    ) -> NoveltyEvent:
        """
        Evaluate novelty of sensory input — Alice's "surprise" when encountering something new

        Physical principle:
          Γ_novelty = |Z_input - Z_model| / (Z_input + Z_model)

          Γ ≈ 0 → completely predictable (boring)
          Γ >> 0 → completely unexpected (surprise/curiosity)

        Simultaneously update internal model (learning):
          Z_model ← Z_model + lr × (Z_input - Z_model)
        """
        # Ensure modality exists
        if modality not in self._internal_model:
            self._internal_model[modality] = MODEL_IMPEDANCE_INIT
            self._signal_history[modality] = []

        z_model = self._internal_model[modality]
        z_input = max(1.0, signal_impedance)  # Avoid division by zero

        # Compute novelty Γ
        gamma_novelty = abs(z_input - z_model) / (z_input + z_model)
        gamma_novelty = float(np.clip(gamma_novelty, 0.0, 1.0))

        # Amplitude modulation: stronger signals have more prominent novelty
        amplitude_factor = float(np.clip(signal_amplitude, 0.1, 1.0))
        novelty_score = gamma_novelty * amplitude_factor

        # Compute information gain (compare with history)
        history = self._signal_history[modality]
        if len(history) >= 3:
            recent_mean = np.mean(history[-10:])
            deviation = abs(signal_impedance - recent_mean) / max(recent_mean, 1.0)
            information_gain = float(np.clip(deviation * 0.5, 0.0, 1.0))
        else:
            information_gain = novelty_score * 0.8  # Early stage: everything is new

        # Update internal model (Hebbian learning)
        self._internal_model[modality] += MODEL_LEARNING_RATE * (z_input - z_model)

        # Record signal history
        history.append(signal_impedance)
        if len(history) > NOVELTY_WINDOW:
            self._signal_history[modality] = history[-NOVELTY_WINDOW:]

        # Update global novelty
        self._novelty_level = float(np.clip(
            self._novelty_level * (1 - NOVELTY_DECAY) + novelty_score * 0.5,
            0.0, 1.0,
        ))

        # Boredom pressure: novelty present → release boredom
        if novelty_score > 0.2:
            self._boredom_pressure = max(
                0.0,
                self._boredom_pressure - BOREDOM_RELEASE * novelty_score,
            )

        # Curiosity reward
        self._intrinsic_reward = information_gain * CURIOSITY_REWARD_SCALE
        self._total_intrinsic_reward += self._intrinsic_reward

        # Update curiosity drive
        # Curiosity = excitement from novelty (high Γ → desire to explore more)
        self._curiosity_drive = float(np.clip(
            self._curiosity_drive + novelty_score * 0.3 - CURIOSITY_DECAY,
            0.0, 1.0,
        ))

        # Record event
        self._last_input_tick = self._tick_count
        self._ticks_without_input = 0
        self._total_novelty_events += 1

        event = NoveltyEvent(
            modality=modality,
            novelty_score=round(novelty_score, 4),
            information_gain=round(information_gain, 4),
            gamma_novelty=round(gamma_novelty, 4),
            z_input=round(z_input, 2),
            z_model=round(z_model, 2),
            tick=self._tick_count,
        )
        self._recent_novelty_events.append(event)
        if len(self._recent_novelty_events) > 50:
            self._recent_novelty_events = self._recent_novelty_events[-50:]

        return event

    # ==================================================================
    # Efference copy — basis of self-recognition
    # ==================================================================

    def register_efference_copy(
        self,
        modality: str,
        predicted_impedance: float,
        action_description: str = "",
    ) -> None:
        """
        Register efference copy — Alice's prediction before executing an action

        Physical principle:
          Motor cortex sends a "copy" to sensory cortex at the same time as issuing a command
          "I am about to produce a 440Hz sound, impedance approximately 60Ω"
          → When sensory feedback arrives, compare with copy
          → Match = self-generated
          → Mismatch = externally generated

        This is why you can't tickle yourself — you predicted your own tactile sensation.
        """
        copy = {
            "modality": modality,
            "predicted_z": predicted_impedance,
            "action": action_description,
            "created_tick": self._tick_count,
            "matched": False,
        }
        self._efference_copies.append(copy)

        # Clear expired copies
        self._efference_copies = [
            c for c in self._efference_copies
            if self._tick_count - c["created_tick"] <= EFFERENCE_COPY_WINDOW
        ]

    def compare_self_other(
        self,
        modality: str,
        actual_impedance: float,
        is_actually_self: Optional[bool] = None,
    ) -> SelfOtherJudgment:
        """
        Self/other discrimination — Alice judges whether a sound she hears is her own

        Physical principle:
          Γ_self = |Z_predicted - Z_actual| / (Z_predicted + Z_actual)

          If efference copy exists and matches → small Γ_self → this is my voice
          If no copy or mismatch → large Γ_self → this is external

        Self-recognition accuracy improves with experience (infants need to learn).
        """
        # Search for matching efference copy
        best_match_gamma = 1.0
        best_copy = None

        for copy in self._efference_copies:
            if copy["modality"] != modality or copy["matched"]:
                continue

            z_pred = copy["predicted_z"]
            z_act = max(1.0, actual_impedance)
            gamma = abs(z_pred - z_act) / (z_pred + z_act)

            if gamma < best_match_gamma:
                best_match_gamma = gamma
                best_copy = copy

        # If matching copy found
        if best_copy is not None:
            best_copy["matched"] = True
            z_predicted = best_copy["predicted_z"]
        else:
            z_predicted = self._self_model.get(modality, MODEL_IMPEDANCE_INIT)
            # No copy available: compute using self-model
            z_act = max(1.0, actual_impedance)
            best_match_gamma = abs(z_predicted - z_act) / (z_predicted + z_act)

        # Judgment: Γ_self < threshold → is self
        # Physical rationale: efference copy is the primary mechanism for self-recognition
        #   With copy → threshold fixed (prediction-feedback match)
        #   Without copy → threshold narrows (lack of motor prediction = strong cue for external signal)
        #   Low infant accuracy → very strict threshold without copy → almost always judged as other
        #   High mature accuracy → can discriminate using impedance model even without copy
        if best_copy is not None:
            # With efference copy: use fixed threshold
            effective_threshold = SELF_RECOGNITION_THRESHOLD
        else:
            # Without efference copy: threshold × accuracy (less mature = stricter)
            effective_threshold = SELF_RECOGNITION_THRESHOLD * max(
                self._self_recognition_accuracy, 0.1
            )
        is_self_judgment = best_match_gamma < effective_threshold

        # Judgment confidence
        if is_self_judgment:
            # Better match (smaller Γ) → higher confidence
            confidence = float(np.clip(1.0 - best_match_gamma / effective_threshold, 0.0, 1.0))
        else:
            # Worse match (larger Γ) → higher confidence in judging as other
            confidence = float(np.clip(
                (best_match_gamma - effective_threshold) / (1.0 - effective_threshold),
                0.0, 1.0,
            ))

        confidence *= self._self_recognition_accuracy

        # Learning: if ground truth available, update self-model
        if is_actually_self is not None:
            correct = (is_self_judgment == is_actually_self)
            self._total_self_judgments += 1
            if correct:
                self._total_self_correct += 1

            # Update self-recognition accuracy (exponential moving average)
            if correct:
                self._self_recognition_accuracy = min(
                    1.0,
                    self._self_recognition_accuracy + SELF_MODEL_LEARNING_RATE * (1 - self._self_recognition_accuracy),
                )
            else:
                self._self_recognition_accuracy = max(
                    0.1,
                    self._self_recognition_accuracy - SELF_MODEL_LEARNING_RATE * 0.5,
                )

            # If self, update self-model impedance
            if is_actually_self:
                if modality in self._self_model:
                    self._self_model[modality] += SELF_MODEL_LEARNING_RATE * (
                        actual_impedance - self._self_model[modality]
                    )

        judgment = SelfOtherJudgment(
            modality=modality,
            gamma_self=round(best_match_gamma, 4),
            z_predicted=round(z_predicted, 2),
            z_actual=round(actual_impedance, 2),
            is_self=is_self_judgment,
            confidence=round(confidence, 4),
            tick=self._tick_count,
        )

        self._self_other_judgments.append(judgment)
        if len(self._self_other_judgments) > 100:
            self._self_other_judgments = self._self_other_judgments[-100:]

        return judgment

    # ==================================================================
    # Spontaneous action generator
    # ==================================================================

    def generate_spontaneous_action(
        self,
        energy: float = 1.0,
    ) -> Optional[SpontaneousAction]:
        """
        Spontaneous action generation — Alice decides what to do when bored

        Physical principle:
          When boredom pressure exceeds threshold, internal oscillation intensifies
          Lowest energy oscillation mode → becomes spontaneous behavior

          Selection logic (biologically inspired):
          - Low boredom + curiosity → explore
          - High boredom + low energy → daydreaming / self-examination
          - High boredom + high energy → babbling / motor exploration
          - Low self-recognition accuracy → prioritize self-examination

        Returns:
            SpontaneousAction or None (if boredom insufficient or in cooldown)
        """
        # Cooldown check
        if self._tick_count - self._last_spontaneous_tick < SPONTANEOUS_COOLDOWN:
            return None

        # Insufficient boredom → no spontaneous behavior
        if self._boredom_pressure < BOREDOM_THRESHOLD and self._spontaneous_urge < 0.5:
            return None

        # Insufficient energy → only low-energy behaviors
        if energy < 0.1:
            return None

        # === Select behavior type ===
        # Compute "resonance intensity" for each behavior
        action_scores: Dict[SpontaneousActionType, float] = {}

        # Babbling: suited for high boredom + energy + self-recognition needs practice
        babble_score = (
            self._boredom_pressure * 0.4
            + (1 - self._self_recognition_accuracy) * 0.4  # Poor self-recognition → needs more vocalization
            + energy * 0.2
        )
        action_scores[SpontaneousActionType.BABBLE] = babble_score

        # Visual exploration: suited for high curiosity
        explore_v = (
            self._curiosity_drive * 0.5
            + self._boredom_pressure * 0.3
            + energy * 0.2
        )
        action_scores[SpontaneousActionType.EXPLORE_VISUAL] = explore_v

        # Motor exploration: suited for high energy + boredom
        explore_m = (
            self._boredom_pressure * 0.3
            + energy * 0.4
            + self._curiosity_drive * 0.3
        )
        action_scores[SpontaneousActionType.EXPLORE_MOTOR] = explore_m

        # Seek novelty: suited for very high curiosity
        seek = (
            self._curiosity_drive * 0.6
            + self._boredom_pressure * 0.2
            + energy * 0.2
        )
        action_scores[SpontaneousActionType.SEEK_NOVELTY] = seek

        # Self-examination: suited for low self-recognition accuracy
        self_exam = (
            (1 - self._self_recognition_accuracy) * 0.6
            + self._boredom_pressure * 0.2
            + self._curiosity_drive * 0.2
        )
        action_scores[SpontaneousActionType.SELF_EXAMINE] = self_exam

        # Daydreaming: suited for low energy + boredom
        idle_dream = (
            self._boredom_pressure * 0.3
            + (1 - energy) * 0.5
            + 0.2  # Baseline tendency
        )
        action_scores[SpontaneousActionType.IDLE_DREAM] = idle_dream

        # Select highest-scoring behavior
        best_action = max(action_scores, key=action_scores.get)  # type: ignore[arg-type]
        intensity = float(np.clip(action_scores[best_action], 0.0, 1.0))

        # Trigger reason
        if self._boredom_pressure >= BOREDOM_THRESHOLD:
            trigger = "boredom_threshold_exceeded"
        elif self._curiosity_drive > 0.5:
            trigger = "curiosity_driven"
        else:
            trigger = "spontaneous_urge"

        action = SpontaneousAction(
            action_type=best_action,
            intensity=round(intensity, 4),
            trigger=trigger,
            boredom_level=round(self._boredom_pressure, 4),
            tick=self._tick_count,
        )

        # Record
        self._last_spontaneous_tick = self._tick_count
        self._total_spontaneous_actions += 1
        self._spontaneous_history.append(action)
        if len(self._spontaneous_history) > 100:
            self._spontaneous_history = self._spontaneous_history[-100:]

        # Spontaneous action consumes some boredom pressure
        self._boredom_pressure = max(
            0.0, self._boredom_pressure - 0.15
        )
        self._spontaneous_urge = max(0.0, self._spontaneous_urge - 0.3)

        return action

    # ==================================================================
    # Internal goal generation
    # ==================================================================

    def generate_goal_from_curiosity(self) -> Optional[InternalGoal]:
        """
        Curiosity-driven goal generation — Alice sets her own goals

        When curiosity or boredom is high enough, generate internal goals:
        - "I want to explore new sounds"
        - "I want to understand what I just saw"
        - "I want to know if that sound was made by me"
        """
        # Needs sufficient drive
        if self._curiosity_drive < 0.3 and self._boredom_pressure < 0.4:
            return None

        # Already too many goals
        if len([g for g in self._internal_goals if not g.satisfied]) >= self._max_goals:
            return None

        # === Select goal based on internal state ===
        if self._self_recognition_accuracy < 0.6:
            # Poor self-recognition → wants to know self better
            goal = InternalGoal(
                description="explore_self_voice",
                source="self_explore",
                priority=round(0.5 + (1 - self._self_recognition_accuracy) * 0.3, 2),
                created_tick=self._tick_count,
            )
        elif self._curiosity_drive > self._boredom_pressure:
            # Curiosity dominant → wants to explore new things
            # Find least familiar modality
            least_familiar = min(
                self._internal_model.items(),
                key=lambda x: len(self._signal_history.get(x[0], [])),
            )
            goal = InternalGoal(
                description=f"explore_{least_familiar[0]}",
                source="curiosity",
                priority=round(self._curiosity_drive * 0.8, 2),
                created_tick=self._tick_count,
            )
        else:
            # Boredom dominant → wants to find something to do
            goal = InternalGoal(
                description="seek_stimulation",
                source="boredom",
                priority=round(self._boredom_pressure * 0.6, 2),
                created_tick=self._tick_count,
            )

        self._internal_goals.append(goal)
        return goal

    def satisfy_goal(self, description: str) -> bool:
        """Mark goal as satisfied"""
        for goal in self._internal_goals:
            if goal.description == description and not goal.satisfied:
                goal.satisfied = True
                return True
        return False

    # ==================================================================
    # Core tick — update per perception cycle
    # ==================================================================

    def tick(
        self,
        has_external_input: bool = False,
        sensory_load: float = 0.0,
        energy: float = 1.0,
        is_sleeping: bool = False,
    ) -> Dict[str, Any]:
        """
        Update once per perception cycle — boredom accumulation + curiosity decay + spontaneous urge

        Args:
            has_external_input: Whether this tick has external sensory input
            sensory_load: Current sensory load (0~1)
            energy: Current energy level (0~1)
            is_sleeping: Whether sleeping

        Returns:
            Dict containing current drive states and any spontaneous action
        """
        self._tick_count += 1

        # Sleeping: curiosity paused (boredom does not accumulate)
        if is_sleeping:
            result = {
                "tick": self._tick_count,
                "novelty": 0.0,
                "boredom": self._boredom_pressure,
                "curiosity": self._curiosity_drive,
                "spontaneous_urge": 0.0,
                "self_recognition_accuracy": self._self_recognition_accuracy,
                "sleeping": True,
                "spontaneous_action": None,
            }
            return result

        # === 1. Boredom pressure update ===
        if not has_external_input:
            self._ticks_without_input += 1
            # No input → boredom accumulates (accelerates: longer = more bored)
            acceleration = 1.0 + self._ticks_without_input * 0.01
            boredom_gain = BOREDOM_ACCUMULATION * acceleration * (1.0 - sensory_load)
            self._boredom_pressure = min(
                BOREDOM_SATURATION,
                self._boredom_pressure + boredom_gain,
            )
        else:
            self._ticks_without_input = 0

        # === 2. Stimulation level update ===
        target_stim = sensory_load if has_external_input else 0.0
        self._stimulation_level += 0.1 * (target_stim - self._stimulation_level)
        self._stimulation_level = float(np.clip(self._stimulation_level, 0.0, 1.0))

        # === 3. Novelty natural decay ===
        self._novelty_level = max(0.0, self._novelty_level - NOVELTY_DECAY)

        # === 4. Curiosity decay ===
        self._curiosity_drive = max(0.0, self._curiosity_drive - CURIOSITY_DECAY)

        # === 5. Spontaneous urge update ===
        # High boredom + high curiosity → increased urge
        urge_input = (
            self._boredom_pressure * 0.5
            + self._curiosity_drive * 0.3
            + (1.0 - self._stimulation_level) * 0.2
        )
        self._spontaneous_urge = float(np.clip(
            self._spontaneous_urge + urge_input * 0.05 - URGE_DECAY,
            0.0, 1.0,
        ))

        # === 6. Try to generate spontaneous behavior ===
        spontaneous = self.generate_spontaneous_action(energy)

        # === 7. Try to generate internal goals ===
        new_goal = None
        if self._tick_count % 20 == 0:  # Evaluate every 20 ticks
            new_goal = self.generate_goal_from_curiosity()

        # === 8. Clean up expired efference copies ===
        self._efference_copies = [
            c for c in self._efference_copies
            if self._tick_count - c["created_tick"] <= EFFERENCE_COPY_WINDOW
        ]

        # === 9. Clean up expired goals ===
        self._internal_goals = [
            g for g in self._internal_goals
            if not g.satisfied or self._tick_count - g.created_tick < 100
        ]

        # === 10. Record history ===
        self._novelty_history.append(self._novelty_level)
        self._boredom_history.append(self._boredom_pressure)
        self._curiosity_history.append(self._curiosity_drive)
        for hist in (self._novelty_history, self._boredom_history, self._curiosity_history):
            if len(hist) > self._max_history:
                del hist[:-self._max_history]

        # === 11. Intrinsic reward decay ===
        self._intrinsic_reward *= 0.9

        result: Dict[str, Any] = {
            "tick": self._tick_count,
            "novelty": round(self._novelty_level, 4),
            "boredom": round(self._boredom_pressure, 4),
            "curiosity": round(self._curiosity_drive, 4),
            "spontaneous_urge": round(self._spontaneous_urge, 4),
            "stimulation": round(self._stimulation_level, 4),
            "intrinsic_reward": round(self._intrinsic_reward, 4),
            "self_recognition_accuracy": round(self._self_recognition_accuracy, 4),
            "spontaneous_action": None,
            "new_goal": None,
        }

        if spontaneous is not None:
            result["spontaneous_action"] = {
                "type": spontaneous.action_type.value,
                "intensity": spontaneous.intensity,
                "trigger": spontaneous.trigger,
            }

        if new_goal is not None:
            result["new_goal"] = {
                "description": new_goal.description,
                "source": new_goal.source,
                "priority": new_goal.priority,
            }

        return result

    # ==================================================================
    # Query interface
    # ==================================================================

    def get_intrinsic_reward(self) -> float:
        """Get current intrinsic reward signal (for RL system use)"""
        return self._intrinsic_reward

    def get_boredom(self) -> float:
        """Get current boredom level"""
        return self._boredom_pressure

    def get_curiosity(self) -> float:
        """Get current curiosity intensity"""
        return self._curiosity_drive

    def get_novelty(self) -> float:
        """Get current novelty level"""
        return self._novelty_level

    def get_self_recognition_accuracy(self) -> float:
        """Get self-recognition accuracy"""
        return self._self_recognition_accuracy

    def get_spontaneous_urge(self) -> float:
        """Get spontaneous action urge intensity"""
        return self._spontaneous_urge

    def get_active_goals(self) -> List[Dict[str, Any]]:
        """Get active internal goals"""
        return [
            {
                "description": g.description,
                "source": g.source,
                "priority": g.priority,
                "age": self._tick_count - g.created_tick,
                "satisfied": g.satisfied,
            }
            for g in self._internal_goals
            if not g.satisfied
        ]

    def get_state(self) -> Dict[str, Any]:
        """Get complete engine state"""
        return {
            "tick": self._tick_count,
            "novelty_level": round(self._novelty_level, 4),
            "boredom_pressure": round(self._boredom_pressure, 4),
            "curiosity_drive": round(self._curiosity_drive, 4),
            "spontaneous_urge": round(self._spontaneous_urge, 4),
            "stimulation_level": round(self._stimulation_level, 4),
            "intrinsic_reward": round(self._intrinsic_reward, 4),
            "self_recognition_accuracy": round(self._self_recognition_accuracy, 4),
            "internal_model": {k: round(v, 2) for k, v in self._internal_model.items()},
            "self_model": {k: round(v, 2) for k, v in self._self_model.items()},
            "active_goals": self.get_active_goals(),
            "ticks_without_input": self._ticks_without_input,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "total_ticks": self._tick_count,
            "total_novelty_events": self._total_novelty_events,
            "total_spontaneous_actions": self._total_spontaneous_actions,
            "total_intrinsic_reward": round(self._total_intrinsic_reward, 4),
            "total_self_judgments": self._total_self_judgments,
            "total_self_correct": self._total_self_correct,
            "self_recognition_accuracy": round(self._self_recognition_accuracy, 4),
            "current_boredom": round(self._boredom_pressure, 4),
            "current_curiosity": round(self._curiosity_drive, 4),
            "current_novelty": round(self._novelty_level, 4),
            "goals_generated": len(self._internal_goals),
            "goals_satisfied": len([g for g in self._internal_goals if g.satisfied]),
            "internal_model_drift": {
                k: round(abs(v - MODEL_IMPEDANCE_INIT), 2)
                for k, v in self._internal_model.items()
            },
            "novelty_history_length": len(self._novelty_history),
            "boredom_history_length": len(self._boredom_history),
        }

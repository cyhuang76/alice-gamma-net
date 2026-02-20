# -*- coding: utf-8 -*-
"""
metacognition.py — Phase 18: The Inner Auditor (Metacognition and Self-Correction)

Physical foundation:
  Thinking itself has impedance. When cognitive flow is smooth (Γ_thinking low), Alice is in System 1
  (fast intuitive mode). When impedance suddenly increases (prediction failure, logical conflict, low confidence),
  the metacognitive inspector intervenes, switching the system to System 2 (slow deliberate mode).

Core formulas:
  Γ_thinking = Σ(w_i · Γ_i) / Σ w_i
  where Γ_i ∈ {prediction_error, free_energy, binding_gamma, flexibility_cost, ...}

  Thinking rate control:
    ThinkingRate = base_rate × (1 - Γ_thinking)^α     (α = 1.5)
    → High impedance → rate decreases → trades for higher precision (time dilation of thought)

  Counterfactual reasoning:
    Regret(a) = V(best_counterfactual) - V(actual_outcome)
    Relief(a) = V(actual_outcome) - V(worst_counterfactual)

  Confidence estimation:
    Confidence = 1 / (1 + σ² + Γ_thinking)
    → High precision (low σ) and low impedance → high confidence

  Self-correction thresholds:
    When Γ_thinking > θ_correct (0.6) → trigger Reframe
    When Confidence < θ_doubt (0.3) → trigger Self-Doubt → reduce action tendency

Clinical correspondences:
  - Thinking Impedance → cognitive load (Cognitive Load Theory)
  - System 1/2 switching → Kahneman's dual-process theory
  - Counterfactual reasoning → regret/relief emotions (Counterfactual Thinking)
  - Confidence attribution → Dunning-Kruger / Impostor Syndrome
  - Excessive metacognitive monitoring → OCD rumination (Metacognitive OCD)
  - Self-Correction → cognitive restructuring in CBT
  - Thinking Rate ↓ → anxiety-induced thinking slowdown
  - Insight Moment → Aha! Moment
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# Constants
# ============================================================================

# Thinking impedance weights
W_PREDICTION_ERROR = 0.25    # Prediction failure contribution
W_FREE_ENERGY = 0.20         # Free energy contribution
W_BINDING_GAMMA = 0.15       # Sensory binding quality
W_FLEXIBILITY_COST = 0.10    # Cognitive flexibility switching cost
W_ANXIETY = 0.15             # Anxiety level
W_PFC_FATIGUE = 0.15         # Prefrontal cortex fatigue

# System 1/2 switching thresholds
SYSTEM2_ENGAGE_THRESHOLD = 0.45   # Γ_thinking > this value → switch to System 2
SYSTEM2_DISENGAGE_THRESHOLD = 0.25  # Γ_thinking < this value → return to System 1

# Thinking rate
BASE_THINKING_RATE = 1.0
THINKING_RATE_ALPHA = 1.5         # Rate decay exponent
MIN_THINKING_RATE = 0.2           # Minimum thinking rate (cannot completely stop thinking)
MAX_THINKING_RATE = 1.0

# Confidence
CONFIDENCE_BASELINE = 0.5
MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.99

# Self-correction
CORRECTION_THRESHOLD = 0.6       # Γ_thinking exceeds this → trigger reframe
DOUBT_THRESHOLD = 0.3            # Confidence below this → self-doubt
INSIGHT_THRESHOLD = 0.7          # Γ_thinking drops sharply beyond this amount → insight moment

# Counterfactual reasoning
MAX_COUNTERFACTUALS = 3          # Maximum number of counterfactual paths to simulate
REGRET_DECAY = 0.95              # Regret decay rate
RELIEF_DECAY = 0.95              # Relief decay rate
RUMINATION_THRESHOLD = 0.7       # Regret exceeding this → rumination risk
MAX_RUMINATION_COUNT = 50        # Rumination count cap (clinical indicator)

# Metacognitive monitoring EMA
EMA_ALPHA = 0.25                 # Exponential moving average coefficient

# History
MAX_HISTORY = 500


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CognitiveSnapshot:
    """Cognitive state snapshot for a single tick"""
    prediction_error: float = 0.0
    free_energy: float = 0.0
    binding_gamma: float = 0.0
    flexibility_cost: float = 0.0    # 1 - flexibility_index
    anxiety: float = 0.0
    pfc_fatigue: float = 0.0         # 1 - pfc_energy
    surprise: float = 0.0
    pain: float = 0.0
    phi: float = 0.5                 # Consciousness level
    novelty: float = 0.0
    boredom: float = 0.0


@dataclass
class CounterfactualResult:
    """Counterfactual reasoning result"""
    action_taken: str = "idle"
    outcome_value: float = 0.0       # Actual outcome value
    best_alternative: str = "idle"
    best_value: float = 0.0          # Best alternative value
    worst_alternative: str = "idle"
    worst_value: float = 0.0         # Worst alternative value
    regret: float = 0.0              # V(best) - V(actual)
    relief: float = 0.0              # V(actual) - V(worst)


@dataclass
class MetacognitionResult:
    """Metacognition tick output"""
    gamma_thinking: float = 0.0           # Thinking impedance (core metric)
    thinking_rate: float = 1.0            # Current thinking rate
    confidence: float = 0.5              # Confidence level
    system_mode: int = 1                 # 1 = System 1, 2 = System 2
    is_correcting: bool = False          # Whether self-correcting
    is_doubting: bool = False            # Whether in self-doubt
    is_insight: bool = False             # Whether insight moment
    is_ruminating: bool = False          # Whether ruminating
    regret: float = 0.0                  # Current regret level
    relief: float = 0.0                  # Current relief level
    correction_count: int = 0            # Cumulative correction count
    insight_count: int = 0               # Cumulative insight count
    rumination_count: int = 0            # Rumination count
    meta_report: str = ""                # Human-readable metacognition report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gamma_thinking": round(self.gamma_thinking, 4),
            "thinking_rate": round(self.thinking_rate, 4),
            "confidence": round(self.confidence, 4),
            "system_mode": self.system_mode,
            "is_correcting": self.is_correcting,
            "is_doubting": self.is_doubting,
            "is_insight": self.is_insight,
            "is_ruminating": self.is_ruminating,
            "regret": round(self.regret, 4),
            "relief": round(self.relief, 4),
            "correction_count": self.correction_count,
            "insight_count": self.insight_count,
            "rumination_count": self.rumination_count,
            "meta_report": self.meta_report,
        }


# ============================================================================
# Metacognition Engine
# ============================================================================

class MetacognitionEngine:
    """
    The Inner Auditor — Metacognition and Self-Correction Engine

    Monitors impedance of all cognitive subsystems, computes "thinking impedance" Γ_thinking,
    decides System 1/2 mode switching, performs counterfactual reasoning, and when needed
    triggers self-correction (Reframe).

    Physical intuition:
        Imagine "thinking" as current flowing through a cognitive transmission line,
        each subsystem's discoordination (high Γ) is an impedance mismatch point,
        producing reflected waves (hesitation, confusion, slowdown).
        Metacognition is like installing an oscilloscope on the transmission line,
        monitoring these reflections and deciding whether to stop and "readjust".
    """

    def __init__(self) -> None:
        # --- Core State ---
        self._gamma_thinking: float = 0.0
        self._gamma_thinking_ema: float = 0.0
        self._thinking_rate: float = BASE_THINKING_RATE
        self._confidence: float = CONFIDENCE_BASELINE
        self._system_mode: int = 1       # 1 = fast/intuitive, 2 = slow/deliberate

        # --- Counterfactual Reasoning ---
        self._regret: float = 0.0
        self._relief: float = 0.0
        self._rumination_count: int = 0
        self._last_action_value: float = 0.0

        # --- Self-Correction ---
        self._correction_count: int = 0
        self._is_correcting: bool = False
        self._reframe_cooldown: int = 0    # Prevent over-correction

        # --- Insight Tracking ---
        self._insight_count: int = 0
        self._prev_gamma_thinking: float = 0.0

        # --- History ---
        self._gamma_history: List[float] = []
        self._confidence_history: List[float] = []
        self._thinking_rate_history: List[float] = []
        self._system_mode_history: List[int] = []
        self._regret_history: List[float] = []

        # --- Cumulative Statistics ---
        self._total_ticks: int = 0
        self._total_system2_ticks: int = 0
        self._total_corrections: int = 0
        self._total_insights: int = 0
        self._total_doubts: int = 0

        # --- Counterfactual model: simple value table (updatable by predictive engine) ---
        self._action_values: Dict[str, float] = {
            "idle": 0.0,
            "cool": 0.3,
            "rest": 0.4,
            "flee": 0.5,
            "seek": 0.2,
        }

    # ======================================================================
    # Main Tick
    # ======================================================================

    def tick(
        self,
        *,
        prediction_error: float = 0.0,
        free_energy: float = 0.0,
        binding_gamma: float = 0.0,
        flexibility_index: float = 0.75,
        anxiety: float = 0.0,
        pfc_energy: float = 0.8,
        surprise: float = 0.0,
        pain: float = 0.0,
        phi: float = 0.5,
        novelty: float = 0.0,
        boredom: float = 0.0,
        precision: float = 0.3,
        last_action: Optional[str] = None,
        is_sleeping: bool = False,
    ) -> Dict[str, Any]:
        """
        Per-tick metacognition update.

        Parameters
        ----------
        prediction_error : float  Prediction error (from PredictiveEngine)
        free_energy : float       Free energy (from PredictiveEngine)
        binding_gamma : float     Sensory binding reflection coefficient
        flexibility_index : float Cognitive flexibility index (0.5~0.95)
        anxiety : float           Anxiety level (from PredictiveEngine)
        pfc_energy : float        Prefrontal cortex energy (0~1)
        surprise : float          Surprise signal (from PredictiveEngine)
        pain : float              Pain (0~1)
        phi : float               Consciousness level (from consciousness)
        novelty : float           Novelty (from curiosity)
        boredom : float           Boredom (from curiosity)
        precision : float         Prediction precision σ (from PredictiveEngine)
        last_action : str         Last action taken
        is_sleeping : bool        Whether sleeping
        """
        self._total_ticks += 1
        result = MetacognitionResult()

        # During sleep, metacognition nearly shuts down (only minimal monitoring)
        if is_sleeping:
            result.gamma_thinking = self._gamma_thinking_ema * 0.5
            result.thinking_rate = MIN_THINKING_RATE
            result.system_mode = 1
            result.confidence = self._confidence
            result.meta_report = "[sleep] Metacognition hibernating"
            self._append_history(result)
            return result.to_dict()

        # --- 1. Compute thinking impedance Γ_thinking ---
        snapshot = CognitiveSnapshot(
            prediction_error=prediction_error,
            free_energy=free_energy,
            binding_gamma=binding_gamma,
            flexibility_cost=max(0.0, 1.0 - flexibility_index),
            anxiety=anxiety,
            pfc_fatigue=max(0.0, 1.0 - pfc_energy),
            surprise=surprise,
            pain=pain,
            phi=phi,
            novelty=novelty,
            boredom=boredom,
        )
        self._gamma_thinking = self._compute_thinking_impedance(snapshot)
        self._gamma_thinking_ema = (
            EMA_ALPHA * self._gamma_thinking
            + (1 - EMA_ALPHA) * self._gamma_thinking_ema
        )

        # --- 2. System 1/2 switching ---
        self._update_system_mode()

        # --- 3. Thinking rate control ---
        self._update_thinking_rate()

        # --- 4. Confidence estimation ---
        self._update_confidence(precision)

        # --- 5. Insight detection ---
        is_insight = self._detect_insight()

        # --- 6. Counterfactual reasoning ---
        cf_result = self._counterfactual_reasoning(last_action)

        # --- 7. Self-correction evaluation ---
        is_correcting = self._evaluate_correction()

        # --- 8. Rumination detection ---
        is_ruminating = self._detect_rumination()

        # --- 9. Self-doubt detection ---
        is_doubting = self._confidence < DOUBT_THRESHOLD

        # Update statistics
        if self._system_mode == 2:
            self._total_system2_ticks += 1
        if is_correcting:
            self._total_corrections += 1
        if is_insight:
            self._total_insights += 1
        if is_doubting:
            self._total_doubts += 1

        # --- Assemble result ---
        result.gamma_thinking = round(self._gamma_thinking_ema, 6)
        result.thinking_rate = round(self._thinking_rate, 6)
        result.confidence = round(self._confidence, 6)
        result.system_mode = self._system_mode
        result.is_correcting = is_correcting
        result.is_doubting = is_doubting
        result.is_insight = is_insight
        result.is_ruminating = is_ruminating
        result.regret = round(self._regret, 6)
        result.relief = round(self._relief, 6)
        result.correction_count = self._correction_count
        result.insight_count = self._insight_count
        result.rumination_count = self._rumination_count
        result.meta_report = self._generate_meta_report(result)

        self._prev_gamma_thinking = self._gamma_thinking_ema
        self._append_history(result)
        return result.to_dict()

    # ======================================================================
    # Core Computations
    # ======================================================================

    def _compute_thinking_impedance(self, snap: CognitiveSnapshot) -> float:
        """
        Compute thinking impedance Γ_thinking — weighted reflection coefficient of all cognitive subsystems.

        Γ_thinking = Σ(w_i · g_i) / Σ w_i

        where g_i is each subsystem's "discoordination degree" (0~1):
        - prediction_error: used directly (already 0~1 scale)
        - free_energy: tanh compressed to 0~1
        - binding_gamma: used directly
        - flexibility_cost: 1 - flexibility_index
        - anxiety: used directly
        - pfc_fatigue: 1 - pfc_energy
        """
        # Normalize each metric to 0~1
        g_pred = min(1.0, snap.prediction_error)
        g_fe = math.tanh(snap.free_energy * 0.5)    # Compress free energy
        g_bind = min(1.0, snap.binding_gamma)
        g_flex = min(1.0, snap.flexibility_cost)
        g_anx = min(1.0, snap.anxiety)
        g_pfc = min(1.0, snap.pfc_fatigue)

        numerator = (
            W_PREDICTION_ERROR * g_pred
            + W_FREE_ENERGY * g_fe
            + W_BINDING_GAMMA * g_bind
            + W_FLEXIBILITY_COST * g_flex
            + W_ANXIETY * g_anx
            + W_PFC_FATIGUE * g_pfc
        )
        denominator = (
            W_PREDICTION_ERROR + W_FREE_ENERGY + W_BINDING_GAMMA
            + W_FLEXIBILITY_COST + W_ANXIETY + W_PFC_FATIGUE
        )

        gamma = numerator / denominator

        # Pain bonus: high pain directly increases thinking impedance
        gamma += snap.pain * 0.15

        # Consciousness modulation: low consciousness → metacognition downgrade
        gamma *= min(1.0, snap.phi / 0.4 + 0.3)

        return max(0.0, min(1.0, gamma))

    def _update_system_mode(self) -> None:
        """
        System 1/2 switching (hysteresis loop to avoid oscillation).

        System 1 (fast intuitive): Γ_thinking low, fast but imprecise responses
        System 2 (slow deliberate): Γ_thinking high, slow but precise responses

        Uses hysteresis to avoid frequent switching near thresholds:
        - System 2 engage threshold (0.45) > disengage threshold (0.25)
        """
        if self._system_mode == 1:
            if self._gamma_thinking_ema > SYSTEM2_ENGAGE_THRESHOLD:
                self._system_mode = 2
        else:  # System 2
            if self._gamma_thinking_ema < SYSTEM2_DISENGAGE_THRESHOLD:
                self._system_mode = 1

    def _update_thinking_rate(self) -> None:
        """
        Thinking rate control (Time-Dilation for Thought).

        ThinkingRate = base × (1 - Γ_thinking)^α

        High impedance → rate decreases → physically "slowing down to think clearly"
        This is the physical realization of System 2: not "thinking harder",
        but "spending more time passing through impedance".
        """
        raw_rate = BASE_THINKING_RATE * (
            (1.0 - self._gamma_thinking_ema) ** THINKING_RATE_ALPHA
        )
        self._thinking_rate = max(MIN_THINKING_RATE, min(MAX_THINKING_RATE, raw_rate))

    def _update_confidence(self, precision: float) -> None:
        """
        Confidence estimation.

        Confidence = 1 / (1 + σ² + Γ_thinking)

        - σ low (precise predictions) and Γ low (smooth thinking) → high confidence
        - σ high (uncertain) or Γ high (cognitive stuck) → low confidence

        Clinical correspondences:
        - Confidence persistently high but prediction_error also high → Dunning-Kruger
        - Confidence persistently low but prediction_error low → Impostor Syndrome
        """
        raw_conf = 1.0 / (1.0 + precision ** 2 + self._gamma_thinking_ema)
        # EMA smoothing
        self._confidence = (
            EMA_ALPHA * raw_conf + (1 - EMA_ALPHA) * self._confidence
        )
        self._confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, self._confidence))

    def _detect_insight(self) -> bool:
        """
        Insight detection (Aha! Moment).

        When raw Γ_thinking drops sharply from a high value (drop > 50% of prev_ema) → insight.
        Uses raw (not EMA) values for detection, because insight is an acute event.
        Physical intuition: impedance on transmission line suddenly matches → reflected waves vanish → energy flows freely.
        """
        if self._total_ticks < 3:
            return False

        # Compare instantaneous value vs previous tick's EMA
        drop = self._prev_gamma_thinking - self._gamma_thinking
        if (
            self._prev_gamma_thinking > 0.3
            and drop > INSIGHT_THRESHOLD * self._prev_gamma_thinking
        ):
            self._insight_count += 1
            return True
        return False

    def _counterfactual_reasoning(
        self, last_action: Optional[str]
    ) -> CounterfactualResult:
        """
        Counterfactual reasoning: "What if I hadn't done that?"

        Based on current action value table, compares actual action vs all alternative paths.
        Generates Regret and Relief signals.
        """
        cf = CounterfactualResult()

        if last_action is None or last_action not in self._action_values:
            # Decay
            self._regret *= REGRET_DECAY
            self._relief *= RELIEF_DECAY
            cf.regret = self._regret
            cf.relief = self._relief
            return cf

        actual_value = self._action_values.get(last_action, 0.0)
        cf.action_taken = last_action
        cf.outcome_value = actual_value

        # Find best and worst alternatives
        alternatives = {
            k: v for k, v in self._action_values.items() if k != last_action
        }
        if not alternatives:
            return cf

        best_action = max(alternatives, key=alternatives.get)  # type: ignore
        worst_action = min(alternatives, key=alternatives.get)  # type: ignore
        cf.best_alternative = best_action
        cf.best_value = alternatives[best_action]
        cf.worst_alternative = worst_action
        cf.worst_value = alternatives[worst_action]

        # Compute Regret and Relief
        raw_regret = max(0.0, cf.best_value - actual_value)
        raw_relief = max(0.0, actual_value - cf.worst_value)

        # EMA smoothing
        self._regret = EMA_ALPHA * raw_regret + (1 - EMA_ALPHA) * (self._regret * REGRET_DECAY)
        self._relief = EMA_ALPHA * raw_relief + (1 - EMA_ALPHA) * (self._relief * RELIEF_DECAY)

        cf.regret = self._regret
        cf.relief = self._relief

        return cf

    def _evaluate_correction(self) -> bool:
        """
        Self-correction evaluation.

        When Γ_thinking > CORRECTION_THRESHOLD and not in cooldown → trigger Reframe.
        Reframe = reinterpreting current cognitive state, physically equivalent to
        "inserting an impedance matching network at the mismatch point on the transmission line".
        """
        if self._reframe_cooldown > 0:
            self._reframe_cooldown -= 1
            self._is_correcting = False
            return False

        if self._gamma_thinking_ema > CORRECTION_THRESHOLD:
            self._correction_count += 1
            self._is_correcting = True
            self._reframe_cooldown = 5   # 5 tick cooldown period
            return True

        self._is_correcting = False
        return False

    def _detect_rumination(self) -> bool:
        """
        Rumination detection.

        When regret persists above RUMINATION_THRESHOLD → rumination.
        Rumination = pathological loop of counterfactual reasoning (OCD spectrum).
        """
        if self._regret > RUMINATION_THRESHOLD:
            self._rumination_count = min(
                self._rumination_count + 1, MAX_RUMINATION_COUNT
            )
            return True
        else:
            self._rumination_count = max(0, self._rumination_count - 1)
            return False

    # ======================================================================
    # Action Value Update (called by external systems)
    # ======================================================================

    def update_action_value(
        self, action: str, outcome_value: float, lr: float = 0.1
    ) -> None:
        """
        Update action value table (for counterfactual reasoning).

        Can be updated by basal ganglia RPE signals or predictive engine results.
        """
        if action in self._action_values:
            old = self._action_values[action]
            self._action_values[action] = old + lr * (outcome_value - old)
        else:
            self._action_values[action] = outcome_value

    def inject_regret(self, amount: float) -> None:
        """Externally inject regret (e.g., explicit negative outcome feedback)"""
        self._regret = min(1.0, self._regret + amount)

    def inject_relief(self, amount: float) -> None:
        """Externally inject relief (e.g., narrow escape)"""
        self._relief = min(1.0, self._relief + amount)

    # ======================================================================
    # Report Generation
    # ======================================================================

    def _generate_meta_report(self, result: MetacognitionResult) -> str:
        """Generate human-readable metacognition report."""
        lines = []

        # System mode
        mode_str = "System 1 (Intuitive)" if result.system_mode == 1 else "System 2 (Deliberate)"
        lines.append(f"[{mode_str}] Γ_thinking={result.gamma_thinking:.3f}")

        # Thinking rate
        if result.thinking_rate < 0.5:
            lines.append(f"⚠ Thinking slowing down (rate={result.thinking_rate:.2f})")

        # Confidence
        if result.is_doubting:
            lines.append(f"⚠ Self-Doubt: confidence={result.confidence:.2f}")
        elif result.confidence > 0.8:
            lines.append(f"✓ High confidence ({result.confidence:.2f})")

        # Correction
        if result.is_correcting:
            lines.append(f"★ Self-correction triggered (count: {result.correction_count})")

        # Insight
        if result.is_insight:
            lines.append(f"✦ Insight moment! (count: {result.insight_count})")

        # Rumination
        if result.is_ruminating:
            lines.append(
                f"⚠ Rumination warning (regret={result.regret:.2f}, "
                f"count={result.rumination_count})"
            )

        # Regret/Relief
        if result.regret > 0.3:
            lines.append(f"regret={result.regret:.2f}")
        if result.relief > 0.3:
            lines.append(f"relief={result.relief:.2f}")

        return " | ".join(lines) if lines else "[ok]"

    # ======================================================================
    # History and State Access
    # ======================================================================

    def _append_history(self, result: MetacognitionResult) -> None:
        """Record history"""
        self._gamma_history.append(result.gamma_thinking)
        self._confidence_history.append(result.confidence)
        self._thinking_rate_history.append(result.thinking_rate)
        self._system_mode_history.append(result.system_mode)
        self._regret_history.append(result.regret)

        for hist in [
            self._gamma_history, self._confidence_history,
            self._thinking_rate_history, self._system_mode_history,
            self._regret_history,
        ]:
            if len(hist) > MAX_HISTORY:
                hist[:] = hist[-MAX_HISTORY:]

    def get_state(self) -> Dict[str, Any]:
        """Get current complete state (for introspect)"""
        return {
            "gamma_thinking": round(self._gamma_thinking_ema, 4),
            "thinking_rate": round(self._thinking_rate, 4),
            "confidence": round(self._confidence, 4),
            "system_mode": self._system_mode,
            "regret": round(self._regret, 4),
            "relief": round(self._relief, 4),
            "correction_count": self._correction_count,
            "insight_count": self._insight_count,
            "rumination_count": self._rumination_count,
            "total_ticks": self._total_ticks,
            "total_system2_ticks": self._total_system2_ticks,
            "total_corrections": self._total_corrections,
            "total_insights": self._total_insights,
            "total_doubts": self._total_doubts,
            "system2_ratio": (
                round(self._total_system2_ticks / max(1, self._total_ticks), 4)
            ),
            "action_values": dict(self._action_values),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics (same as get_state, for consistent interface)"""
        return self.get_state()

    def get_gamma_history(self) -> List[float]:
        return list(self._gamma_history)

    def get_confidence_history(self) -> List[float]:
        return list(self._confidence_history)

    def get_thinking_rate_history(self) -> List[float]:
        return list(self._thinking_rate_history)

    @property
    def gamma_thinking(self) -> float:
        return self._gamma_thinking_ema

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def thinking_rate(self) -> float:
        return self._thinking_rate

    @property
    def system_mode(self) -> int:
        return self._system_mode

    @property
    def regret(self) -> float:
        return self._regret

    @property
    def relief(self) -> float:
        return self._relief

    @property
    def is_ruminating(self) -> bool:
        return self._regret > RUMINATION_THRESHOLD

    @property
    def rumination_count(self) -> int:
        return self._rumination_count

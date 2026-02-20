# -*- coding: utf-8 -*-
"""
Cross-Modal Impedance Adaptation Engine — Experience-Based Γ Improvement
Cross-Modal Impedance Adaptation Engine

Physics:
  "The first time you hear a foreign language, the impedance between
   auditory and language areas is severely mismatched.
   Repeated exposure to the same word → myelin gradually adjusts → impedance matching improves.
   That's why the hundredth listen sounds 'clearer' than the first.
   The sound didn't get louder — the channel impedance matching improved."

  Core formula:
    Γ_ij(t+1) = Γ_ij(t) - η_eff × (1 - Γ_ij(t)²) × success
              + drift × (Γ_ij(0) - Γ_ij(t))    ← forgetting (demyelination)

  Where learning rate is modulated by cortisol (Yerkes-Dodson inverted U):
    η_eff = η_base × β(cortisol)
    β(c) = 4c(1-c)   ← maximum at c=0.5 (moderate stress = optimal learning)

  Physical meaning:
    - (1 - Γ²) factor = transmission efficiency constraint.
      Only 'transmitted' signals can drive learning (you can't learn from sounds you can't hear).
      Γ=0: already perfect → learn 0 (ceiling effect)
      Γ=0.5: 75% transmission → learn the most
      Γ=1: total reflection → learn 0 (can't hear = can't learn)

    - β(cortisol) = Yerkes-Dodson inverted U curve
      Low stress (c→0): can't learn (no driving force)
      Moderate stress (c≈0.5): fastest learning
      High stress (c→1): can't learn (amygdala hijacks cortex)

    - drift = forgetting / degradation
      Unused channel impedance slowly reverts to initial value (use it or lose it)

Circuit analogy:
  Impedance matcher = adjustable transformer
  Learning = adjusting transformer turns ratio to minimize reflection
  Forgetting = transformer core demagnetization
  Cortisol = voltage regulator gain control
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

# Learning rate
BASE_LEARNING_RATE = 0.02        # Base learning rate
MAX_LEARNING_RATE = 0.08         # Maximum learning rate (at moderate stress)

# Yerkes-Dodson parameters
YERKES_DODSON_PEAK = 0.45        # Optimal cortisol level (slightly below 0.5)
YERKES_DODSON_WIDTH = 0.25       # Peak width (σ)

# Forgetting / degradation
DRIFT_RATE = 0.002               # Forgetting per tick (revert toward initial value)
MIN_GAMMA = 0.01                 # Γ lower bound (cannot reach perfect 0)
MAX_GAMMA = 0.95                 # Γ upper bound

# Initial Γ
DEFAULT_INITIAL_GAMMA = 0.7      # Default impedance mismatch on first contact

# History
MAX_HISTORY = 500

# Chronic stress suppression constant
CHRONIC_STRESS_PENALTY = 0.5     # Maximum suppression of learning by chronic stress


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ModalityPairState:
    """Impedance matching state between a pair of modalities"""
    modality_a: str
    modality_b: str
    initial_gamma: float = DEFAULT_INITIAL_GAMMA   # Γ at first contact
    current_gamma: float = DEFAULT_INITIAL_GAMMA   # Current Γ
    exposure_count: int = 0                         # Total exposure count
    success_count: int = 0                          # Successful binding count
    last_exposure_tick: int = 0                     # Last exposure tick
    gamma_history: List[float] = field(default_factory=list)

    @property
    def pair_key(self) -> str:
        """Normalized pair key (sorted, ensures A-B = B-A)"""
        return _make_pair_key(self.modality_a, self.modality_b)

    @property
    def success_rate(self) -> float:
        """Binding success rate"""
        if self.exposure_count == 0:
            return 0.0
        return self.success_count / self.exposure_count

    @property
    def transmission_efficiency(self) -> float:
        """Current transmission efficiency = 1 - Γ²"""
        return 1.0 - self.current_gamma ** 2

    @property
    def is_well_matched(self) -> bool:
        """Whether sufficiently matched (Γ < 0.3)"""
        return self.current_gamma < 0.3


def _make_pair_key(mod_a: str, mod_b: str) -> str:
    """Normalized pair key"""
    return "|".join(sorted([mod_a, mod_b]))


# ============================================================================
# Main engine
# ============================================================================

class ImpedanceAdaptationEngine:
    """
    Cross-Modal Impedance Adaptation Engine

    Tracks impedance matching between each pair of modalities,
    and adjusts Γ based on experience (successful bindings) and stress state (cortisol).

    Physical mechanisms:
      1. Each cross-modal binding attempt → record exposure
      2. Binding success → Γ decreases (impedance matching improves)
      3. Binding failure → Γ slightly increases (matching degrades / negative reinforcement)
      4. Disuse → Γ naturally reverts to initial value (forgetting)
      5. Cortisol → modulates learning rate (Yerkes-Dodson)
      6. Chronic stress → suppresses learning (long-term HPA axis overload)
    """

    def __init__(self):
        # Modality pair → matching state
        self._pairs: Dict[str, ModalityPairState] = {}

        # Global state
        self._tick_count: int = 0
        self._total_adaptations: int = 0
        self._cortisol_history: List[float] = []
        self._learning_rate_history: List[float] = []

        # Chronic stress accumulator
        self._chronic_stress_load: float = 0.0

    # ------------------------------------------------------------------
    # Yerkes-Dodson learning rate modulation
    # ------------------------------------------------------------------

    def _yerkes_dodson(self, cortisol: float) -> float:
        """
        Yerkes-Dodson inverted U curve
        β(c) = exp(-((c - peak)² / (2σ²)))

        Physical meaning:
          c ≈ 0 → too relaxed, no learning drive → β ≈ 0
          c ≈ 0.45 → moderate tension, peak learning efficiency → β ≈ 1
          c ≈ 1 → extreme panic, amygdala hijack → β ≈ 0

        This is not a design choice — it is an inevitable consequence of impedance physics:
          Low voltage (low stress) → signal too weak to drive matching adjustment
          Appropriate voltage → maximum power transfer
          Excessive voltage → reflection burst, all energy becomes pain/noise
        """
        c = float(np.clip(cortisol, 0.0, 1.0))
        exponent = -((c - YERKES_DODSON_PEAK) ** 2) / (2 * YERKES_DODSON_WIDTH ** 2)
        return float(math.exp(exponent))

    def _effective_learning_rate(self, cortisol: float,
                                  chronic_stress: float = 0.0) -> float:
        """
        Effective learning rate = base rate × Yerkes-Dodson × (1 - chronic stress penalty)

        Physical meaning of chronic stress:
          Long-term high cortisol → HPA axis blunting → reduced modulation effect of new cortisol
          → Learning capacity is 'flattened', unable to learn well regardless of acute stress changes
        """
        yd = self._yerkes_dodson(cortisol)
        chronic_penalty = 1.0 - CHRONIC_STRESS_PENALTY * float(np.clip(chronic_stress, 0, 1))
        return BASE_LEARNING_RATE + (MAX_LEARNING_RATE - BASE_LEARNING_RATE) * yd * chronic_penalty

    # ------------------------------------------------------------------
    # Core adaptation mechanism
    # ------------------------------------------------------------------

    def record_binding_attempt(
        self,
        modality_a: str,
        modality_b: str,
        success: bool,
        binding_quality: float = 0.5,
        cortisol: float = 0.1,
        chronic_stress: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Record a cross-modal binding attempt → update impedance matching

        Core formula:
          Success: Γ_new = Γ_old - η_eff × (1 - Γ²) × quality
          Failure: Γ_new = Γ_old + η_eff × 0.1 × Γ²
                   (negative reinforcement from failure is weaker — you don't forget how to do something from one failure)

        Args:
            modality_a, modality_b: Names of the paired modalities
            success: Whether binding succeeded
            binding_quality: Binding quality (0~1), only used on success
            cortisol: Current cortisol level (0~1)
            chronic_stress: Chronic stress load (0~1)

        Returns:
            Dict with delta_gamma, effective_lr, yerkes_dodson, etc.
        """
        self._tick_count += 1
        pair_key = _make_pair_key(modality_a, modality_b)

        # Get or create pair state
        if pair_key not in self._pairs:
            self._pairs[pair_key] = ModalityPairState(
                modality_a=min(modality_a, modality_b),
                modality_b=max(modality_a, modality_b),
            )

        state = self._pairs[pair_key]
        old_gamma = state.current_gamma

        # Compute effective learning rate
        eta = self._effective_learning_rate(cortisol, chronic_stress)
        yd = self._yerkes_dodson(cortisol)

        # ★ Core adaptation formula
        if success:
            # Successful binding → impedance matching improves
            # (1 - Γ²) = transmission efficiency → only transmitted signals can drive learning
            improvement = eta * (1.0 - old_gamma ** 2) * binding_quality
            new_gamma = old_gamma - improvement
            state.success_count += 1
        else:
            # Failure → weak negative reinforcement (Γ slightly increases)
            # Γ² = reflection fraction → larger reflection = weaker negative reinforcement (can't hear it anyway)
            degradation = eta * 0.1 * old_gamma ** 2
            new_gamma = old_gamma + degradation

        # Clamp
        new_gamma = float(np.clip(new_gamma, MIN_GAMMA, MAX_GAMMA))
        state.current_gamma = new_gamma
        state.exposure_count += 1
        state.last_exposure_tick = self._tick_count

        # Record history
        state.gamma_history.append(round(new_gamma, 4))
        if len(state.gamma_history) > MAX_HISTORY:
            state.gamma_history = state.gamma_history[-MAX_HISTORY:]

        self._total_adaptations += 1
        self._cortisol_history.append(round(cortisol, 3))
        self._learning_rate_history.append(round(eta, 4))

        delta = new_gamma - old_gamma
        return {
            "pair": pair_key,
            "old_gamma": round(old_gamma, 4),
            "new_gamma": round(new_gamma, 4),
            "delta_gamma": round(delta, 6),
            "effective_lr": round(eta, 4),
            "yerkes_dodson": round(yd, 4),
            "cortisol": round(cortisol, 3),
            "success": success,
            "exposure_count": state.exposure_count,
        }

    # ------------------------------------------------------------------
    # Forgetting / degradation (use it or lose it)
    # ------------------------------------------------------------------

    def decay_tick(self):
        """
        Called every tick — impedance degradation for unused pairs

        Physics:
          Inactive channels → myelin gradually degrades → impedance drifts from match → Γ reverts to initial value
          Longer disuse → more severe degradation

        "You haven't played piano for three years — your fingers didn't forget how to play;
          the impedance matching from fingertips to auditory cortex degraded."
        """
        self._tick_count += 1
        for state in self._pairs.values():
            idle_ticks = self._tick_count - state.last_exposure_tick
            if idle_ticks > 10:  # Must be idle for at least 10 ticks before degradation starts
                # Degradation amount ∝ idle time (with upper limit)
                drift_factor = min(idle_ticks * DRIFT_RATE, 0.05)
                direction = state.initial_gamma - state.current_gamma
                state.current_gamma += drift_factor * direction
                state.current_gamma = float(np.clip(
                    state.current_gamma, MIN_GAMMA, MAX_GAMMA
                ))

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_pair_gamma(self, modality_a: str, modality_b: str) -> float:
        """Query the current Γ between two modalities"""
        key = _make_pair_key(modality_a, modality_b)
        if key in self._pairs:
            return self._pairs[key].current_gamma
        return DEFAULT_INITIAL_GAMMA

    def get_adapted_binding_gamma(self, modalities: List[str]) -> float:
        """
        Given a set of active modalities, compute the experience-adapted overall binding_gamma

        Physics: Use the mean Γ of all pairs as the overall binding quality
        """
        if len(modalities) < 2:
            return 0.0  # Single modality does not need cross-modal binding

        pair_gammas = []
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                g = self.get_pair_gamma(modalities[i], modalities[j])
                pair_gammas.append(g)

        return float(np.mean(pair_gammas)) if pair_gammas else DEFAULT_INITIAL_GAMMA

    def get_pair_state(self, modality_a: str, modality_b: str) -> Optional[Dict[str, Any]]:
        """Get the full state of a pair"""
        key = _make_pair_key(modality_a, modality_b)
        if key not in self._pairs:
            return None
        s = self._pairs[key]
        return {
            "pair": s.pair_key,
            "current_gamma": round(s.current_gamma, 4),
            "initial_gamma": round(s.initial_gamma, 4),
            "improvement": round(s.initial_gamma - s.current_gamma, 4),
            "exposure_count": s.exposure_count,
            "success_rate": round(s.success_rate, 3),
            "transmission_efficiency": round(s.transmission_efficiency, 4),
            "is_well_matched": s.is_well_matched,
            "gamma_history": s.gamma_history[-50:],
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        pairs = list(self._pairs.values())
        avg_gamma = float(np.mean([p.current_gamma for p in pairs])) if pairs else DEFAULT_INITIAL_GAMMA
        avg_improvement = float(np.mean([
            p.initial_gamma - p.current_gamma for p in pairs
        ])) if pairs else 0.0
        well_matched = sum(1 for p in pairs if p.is_well_matched)

        return {
            "total_pairs": len(pairs),
            "total_adaptations": self._total_adaptations,
            "avg_gamma": round(avg_gamma, 4),
            "avg_improvement": round(avg_improvement, 4),
            "well_matched_pairs": well_matched,
            "tick_count": self._tick_count,
            "chronic_stress_load": round(self._chronic_stress_load, 4),
        }

    def get_all_pairs(self) -> List[Dict[str, Any]]:
        """Get all pair states"""
        result = []
        for s in self._pairs.values():
            result.append({
                "pair": s.pair_key,
                "current_gamma": round(s.current_gamma, 4),
                "initial_gamma": round(s.initial_gamma, 4),
                "improvement": round(s.initial_gamma - s.current_gamma, 4),
                "exposure_count": s.exposure_count,
                "success_rate": round(s.success_rate, 3),
                "is_well_matched": s.is_well_matched,
            })
        return sorted(result, key=lambda x: x["current_gamma"])

    def get_yerkes_dodson_curve(self, n_points: int = 50) -> Dict[str, List[float]]:
        """Return Yerkes-Dodson curve data points (for visualization)"""
        cortisols = np.linspace(0, 1, n_points).tolist()
        betas = [self._yerkes_dodson(c) for c in cortisols]
        etas = [self._effective_learning_rate(c) for c in cortisols]
        return {
            "cortisol": cortisols,
            "beta": betas,
            "effective_lr": etas,
        }

# -*- coding: utf-8 -*-
"""
Lifecycle Equation Engine — The Bathtub Curve of Cognition

Paper I §3.5, Paper III §6: The complete lifecycle is governed by three competing forces:

  d(ΣΓ²)/dt = −η_learn · ΣΓ²  +  γ_novel · Γ_env(t)  +  δ_aging · D(t)

  Term 1: Learning (MRP descent) — impedance matching improves with experience
  Term 2: Novelty injection — new stimuli introduce impedance mismatch
  Term 3: Aging (Coffin-Manson) — irreversible plastic deformation drifts impedance

The Equilibrium Equation (#22):
  At steady state (d(ΣΓ²)/dt ≈ 0):
    T_steady = T_env + (α/β) · ΣΓ²
  Adulthood = the temperature at which the world can no longer burn you.

The Bathtub Curve emerges naturally:
  Birth     → Novelty >> Learning → chaos, maximum entropy
  Infancy   → Learning > Novelty → rapid Γ² reduction
  Childhood → Learning ≈ Novelty → intense consolidation
  Adulthood → Learning ≈ Novelty + Aging → thermal steady state
  Late life → Aging > Learning → gradual capability loss
  Senescence→ Aging >> Learning → system failure

"Three forces compete: learning, novelty, and aging."

References:
  [Paper I] Eq. 24: The Lifecycle Equation
  [Paper I] Eq. 22: The Equilibrium Equation
  [Paper III] §6: The Lifecycle Equation and Thermal Equilibrium
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from alice.core.signal import ElectricalSignal


# ============================================================================
# Physical Constants
# ============================================================================

# --- Learning rate (MRP descent) ---
ETA_LEARN_DEFAULT = 0.02        # Default learning rate (Γ² reduction per tick)
ETA_LEARN_MAX = 0.10            # Maximum (childhood rapid learning)
ETA_LEARN_MIN = 0.001           # Minimum (senescent, nearly no learning)
# Learning rate modulation: sleep quality, attention, novelty
SLEEP_BOOST_FACTOR = 1.5        # Sleep consolidation boosts learning efficiency
ATTENTION_BOOST_FACTOR = 1.2    # Focused attention boosts learning

# --- Novelty injection rate ---
GAMMA_NOVEL_DEFAULT = 0.03      # Default novelty injection rate
GAMMA_NOVEL_MAX = 0.20          # Maximum (birth — everything is new)
GAMMA_NOVEL_MIN = 0.005         # Minimum (highly familiar environment)
# Novelty adaptation: repeated exposure reduces effective novelty
NOVELTY_ADAPTATION_RATE = 0.001 # How fast novelty habituates

# --- Aging rate (fatigue Γ rise) ---
DELTA_AGING_DEFAULT = 0.0001    # Default aging rate (very slow)
DELTA_AGING_MAX = 0.01          # Maximum (terminal decline)
# Arrhenius activation energy for stress-accelerated aging
# Physical basis: fatigue rate ∝ exp(E_a · stress / k_B T)
# E_a calibrated so at stress=0.5 → ×2.1× (moderate acceleration)
#                   at stress=1.0 → ×4.5× (severe, bounded by DELTA_AGING_MAX)
ARRHENIUS_AGING_EA = 1.5        # Effective activation energy (nondimensional)
# Aging onset: aging term kicks in after developmental maturation
AGING_ONSET_TICK = 1000         # Ticks before aging term activates

# --- Equilibrium Equation (#22) ---
ALPHA_HEATING = 0.15            # ΣΓ² → temperature coefficient
BETA_COOLING = 0.03             # Natural cooling coefficient
T_ENV_DEFAULT = 0.1             # Environmental baseline temperature

# --- Bathtub curve detection thresholds ---
PHASE_INFANCY_THRESHOLD = 0.7   # ΣΓ² above this = still in infancy
PHASE_CHILDHOOD_THRESHOLD = 0.4 # Above this = childhood
PHASE_ADULT_THRESHOLD = 0.2     # Below this + low d/dt = adulthood
PHASE_DECLINE_DERIVATIVE = 0.001  # d(ΣΓ²)/dt above this = aging dominates

# --- History ---
MAX_HISTORY = 2000


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LifecycleState:
    """Snapshot of the lifecycle equation state."""
    sigma_gamma_sq: float           # Current ΣΓ²
    d_sigma_dt: float               # Rate of change d(ΣΓ²)/dt
    learning_term: float            # −η · ΣΓ²
    novelty_term: float             # γ · Γ_env
    aging_term: float               # δ · D(t)
    t_steady: float                 # Equilibrium temperature
    t_current: float                # Current effective temperature
    lifecycle_phase: str            # "birth", "infancy", "childhood", "adulthood", "decline", "senescence"
    delta_aging: float              # Current effective aging rate
    sigma_transmission: float = 0.0  # T_total = 1 − ΣΓ² (energy conservation complement)
    eta_learn: float = 0.0           # Current effective learning rate
    gamma_novel: float = 0.0         # Current effective novelty rate


# ============================================================================
# LifecycleEquationEngine
# ============================================================================

class LifecycleEquationEngine:
    """
    The Lifecycle Equation — unified ODE governing the complete cognitive lifecycle.

    d(ΣΓ²)/dt = −η·ΣΓ² + γ·Γ_env(t) + δ·D(t)

    Integrates:
    - Learning signals from impedance matching (pruning, calibration, Hebbian)
    - Novelty injection from environmental stimuli
    - Aging signals from Coffin-Manson fatigue (pinch_fatigue)
    - Computes thermal equilibrium (Eq. 22)
    - Detects lifecycle phase from bathtub curve position
    """

    def __init__(self) -> None:
        # Core state variable: ΣΓ² (total impedance mismatch energy)
        self._sigma_gamma_sq: float = 0.8  # High at birth (random initialization)

        # Three force parameters (dynamically modulated)
        self._eta_learn: float = ETA_LEARN_DEFAULT
        self._gamma_novel: float = GAMMA_NOVEL_MAX  # Starts high (everything is new)
        self._delta_aging: float = 0.0  # Zero until aging onset

        # Running terms
        self._learning_term: float = 0.0
        self._novelty_term: float = 0.0
        self._aging_term: float = 0.0
        self._d_sigma_dt: float = 0.0

        # Equilibrium state
        self._t_steady: float = T_ENV_DEFAULT
        self._t_current: float = 0.3  # Initial temperature (high at birth)

        # Accumulator for adaptive novelty
        self._novelty_exposure: float = 0.0  # Cumulative novelty exposure
        self._environmental_gamma: float = 0.5  # Current environmental Γ

        # Energy conservation: T = 1 − ΣΓ² (transmission complement)
        self._sigma_transmission: float = 0.2  # T at birth (1 − 0.8)

        # Fatigue damage integration
        self._cumulative_damage: float = 0.0  # D(t) from pinch_fatigue

        # Phase tracking
        self._lifecycle_phase: str = "birth"
        self._tick_count: int = 0

        # History
        self._sigma_history: List[float] = []
        self._phase_history: List[str] = []
        self._derivative_history: List[float] = []

    # ------------------------------------------------------------------
    # Core ODE: d(ΣΓ²)/dt = −η·ΣΓ² + γ·Γ_env + δ·D(t)
    # ------------------------------------------------------------------

    def tick(
        self,
        reflected_energy: float = 0.0,
        novelty_level: float = 0.0,
        fatigue_damage: float = 0.0,
        sleep_quality: float = 0.0,
        attention_strength: float = 0.5,
        stress_level: float = 0.0,
        is_sleeping: bool = False,
    ) -> LifecycleState:
        """
        Advance the lifecycle equation by one tick.

        Args:
            reflected_energy: Current ΣΓ² from system channels
            novelty_level: 0~1 novelty in current stimuli
            fatigue_damage: Cumulative Coffin-Manson damage D(t)
            sleep_quality: 0~1 sleep consolidation quality
            attention_strength: 0~1 attentional focus
            stress_level: 0~1 chronic stress (accelerates aging)
            is_sleeping: Whether system is in sleep state

        Returns:
            LifecycleState snapshot
        """
        self._tick_count += 1

        # Update ΣΓ² from actual system measurement if available
        if reflected_energy > 0:
            # EMA blend: 90% model, 10% measurement (prevents jumps)
            self._sigma_gamma_sq = 0.9 * self._sigma_gamma_sq + 0.1 * reflected_energy

        # ==========================================
        # Term 1: Learning (MRP descent) −η · ΣΓ²
        # ==========================================
        # Learning rate modulation
        eta = ETA_LEARN_DEFAULT
        if is_sleeping:
            eta *= SLEEP_BOOST_FACTOR  # Sleep consolidation accelerates learning
        eta *= (0.8 + ATTENTION_BOOST_FACTOR * attention_strength * 0.2)
        # Early development: faster learning
        if self._tick_count < 500:
            eta *= 2.0  # Critical period: double learning rate
        elif self._tick_count < 2000:
            eta *= 1.5  # Childhood: 50% boost
        eta = float(np.clip(eta, ETA_LEARN_MIN, ETA_LEARN_MAX))
        self._eta_learn = eta

        self._learning_term = -eta * self._sigma_gamma_sq

        # ==========================================
        # Term 2: Novelty injection γ · Γ_env(t)
        # ==========================================
        # Environmental Γ adapts with exposure (habituation)
        self._novelty_exposure += novelty_level * 0.01
        adapted_gamma = max(
            GAMMA_NOVEL_MIN,
            GAMMA_NOVEL_MAX * math.exp(-self._novelty_exposure * NOVELTY_ADAPTATION_RATE * 100)
        )
        # Actual novelty term = adapted rate × current novelty
        self._gamma_novel = adapted_gamma
        self._environmental_gamma = novelty_level

        self._novelty_term = adapted_gamma * novelty_level

        # ==========================================
        # Term 3: Aging δ · D(t)
        # ==========================================
        self._cumulative_damage = max(self._cumulative_damage, fatigue_damage)

        if self._tick_count > AGING_ONSET_TICK:
            # Aging activates after developmental period
            delta = DELTA_AGING_DEFAULT
            # Arrhenius stress acceleration: δ = δ₀ · exp(E_a · stress)
            # Physical basis: thermal activation over energy barrier
            # At stress=0 → factor=1.0 (no acceleration)
            # At stress=0.5 → factor≈2.1×; stress=1.0 → factor≈4.5×
            if stress_level > 0.1:
                arrhenius_factor = math.exp(
                    ARRHENIUS_AGING_EA * stress_level
                )
                delta *= arrhenius_factor
            delta = min(delta, DELTA_AGING_MAX)
            self._delta_aging = delta

            self._aging_term = delta * self._cumulative_damage
        else:
            self._delta_aging = 0.0
            self._aging_term = 0.0

        # ==========================================
        # Integrate: d(ΣΓ²)/dt
        # ==========================================
        self._d_sigma_dt = self._learning_term + self._novelty_term + self._aging_term

        # Euler integration
        self._sigma_gamma_sq = float(np.clip(
            self._sigma_gamma_sq + self._d_sigma_dt,
            0.0, 1.0  # Energy conservation: ΣΓ² ∈ [0, 1] per normalised channel
        ))

        # ★ Energy Conservation: Γ² + T = 1
        self._sigma_transmission = 1.0 - self._sigma_gamma_sq

        # ==========================================
        # Equilibrium Equation (#22)
        # T_steady = T_env + (α/β) · ΣΓ²
        # ==========================================
        self._t_steady = T_ENV_DEFAULT + (ALPHA_HEATING / BETA_COOLING) * self._sigma_gamma_sq
        # Current temperature approaches steady state
        self._t_current += (self._t_steady - self._t_current) * 0.05

        # ==========================================
        # Lifecycle phase detection
        # ==========================================
        self._lifecycle_phase = self._detect_phase()

        # ==========================================
        # History
        # ==========================================
        self._sigma_history.append(self._sigma_gamma_sq)
        self._phase_history.append(self._lifecycle_phase)
        self._derivative_history.append(self._d_sigma_dt)
        if len(self._sigma_history) > MAX_HISTORY:
            self._sigma_history = self._sigma_history[-MAX_HISTORY:]
            self._phase_history = self._phase_history[-MAX_HISTORY:]
            self._derivative_history = self._derivative_history[-MAX_HISTORY:]

        return LifecycleState(
            sigma_gamma_sq=round(self._sigma_gamma_sq, 6),
            d_sigma_dt=round(self._d_sigma_dt, 8),
            learning_term=round(self._learning_term, 8),
            novelty_term=round(self._novelty_term, 8),
            aging_term=round(self._aging_term, 8),
            t_steady=round(self._t_steady, 4),
            t_current=round(self._t_current, 4),
            lifecycle_phase=self._lifecycle_phase,
            sigma_transmission=round(self._sigma_transmission, 6),
            eta_learn=round(self._eta_learn, 6),
            gamma_novel=round(self._gamma_novel, 6),
            delta_aging=round(self._delta_aging, 8),
        )

    # ------------------------------------------------------------------

    def _detect_phase(self) -> str:
        """
        Detect lifecycle phase from bathtub curve position.

        The phase is determined by:
        1. Current ΣΓ² level
        2. Derivative d(ΣΓ²)/dt
        3. Tick count (developmental age)
        """
        sigma = self._sigma_gamma_sq
        d_dt = self._d_sigma_dt

        if self._tick_count < 50:
            return "birth"

        if sigma > PHASE_INFANCY_THRESHOLD and d_dt < 0:
            return "infancy"

        if sigma > PHASE_CHILDHOOD_THRESHOLD:
            return "childhood"

        if d_dt > PHASE_DECLINE_DERIVATIVE and self._tick_count > AGING_ONSET_TICK:
            if sigma > 0.5:
                return "senescence"
            return "decline"

        if sigma < PHASE_ADULT_THRESHOLD and abs(d_dt) < PHASE_DECLINE_DERIVATIVE:
            return "adulthood"

        # Transition zones
        if d_dt < -0.001:
            return "childhood"  # Still improving

        return "adulthood"

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Signal Protocol: produce ElectricalSignal for cross-module communication
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate ElectricalSignal encoding the lifecycle macro-state."""
        amplitude = float(np.clip(self._sigma_gamma_sq, 0.01, 1.0))
        freq = 0.5 + self._sigma_gamma_sq * 4.0  # Delta-theta range
        t = np.linspace(0, 1, 64)
        waveform = amplitude * np.sin(2 * np.pi * freq * t)
        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            amplitude=amplitude,
            frequency=freq,
            phase=0.0,
            impedance=75.0,
            snr=12.0,
            source="lifecycle",
            modality="internal",
        )

    def get_sigma_gamma_sq(self) -> float:
        """Get current ΣΓ² value."""
        return self._sigma_gamma_sq

    def get_equilibrium_temperature(self) -> float:
        """Get equilibrium temperature T_steady (Eq. 22)."""
        return self._t_steady

    def get_phase(self) -> str:
        """Get current lifecycle phase."""
        return self._lifecycle_phase

    def get_state(self) -> Dict[str, Any]:
        """Get full state for introspection."""
        return {
            "sigma_gamma_sq": round(self._sigma_gamma_sq, 6),
            "sigma_transmission": round(self._sigma_transmission, 6),
            "d_sigma_dt": round(self._d_sigma_dt, 8),
            "learning_term": round(self._learning_term, 8),
            "novelty_term": round(self._novelty_term, 8),
            "aging_term": round(self._aging_term, 8),
            "t_steady": round(self._t_steady, 4),
            "t_current": round(self._t_current, 4),
            "lifecycle_phase": self._lifecycle_phase,
            "eta_learn": round(self._eta_learn, 6),
            "gamma_novel": round(self._gamma_novel, 6),
            "delta_aging": round(self._delta_aging, 8),
            "cumulative_damage": round(self._cumulative_damage, 6),
            "tick_count": self._tick_count,
            "history_length": len(self._sigma_history),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_state."""
        return self.get_state()

    def get_bathtub_curve(self, last_n: int = 500) -> Dict[str, List[float]]:
        """Get bathtub curve data for plotting."""
        return {
            "sigma_gamma_sq": self._sigma_history[-last_n:],
            "derivative": self._derivative_history[-last_n:],
            "phases": self._phase_history[-last_n:],
        }

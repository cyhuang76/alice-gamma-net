# -*- coding: utf-8 -*-
"""
Alice's Interoceptive Organ — Internal Body State Sensing

Physics Model: Autonomous Impedance Network
=============================================

Interoception is the sense of the body's internal state — the "eighth sense"
that monitors heartbeat, breathing, hunger, thirst, temperature, bladder,
and all visceral conditions.

In Alice's Γ-Net framework:
    Each internal organ generates ElectricalSignal → Interoception aggregates
    Γ_intero = ΣΓ_organ / N_organs  (mean internal impedance mismatch)

When Γ_intero is low: "I feel fine" → homeostasis
When Γ_intero rises: "Something is wrong" → drives corrective behavior

Interoception feeds directly into:
    - Amygdala (emotional valence of body states)
    - Insular cortex (conscious body awareness)
    - Hypothalamus (homeostatic control loops)

Body budget:
    The brain maintains a "body budget" — predicted internal states vs actual.
    Large prediction errors → high Γ_intero → anxiety, discomfort, illness

    "Anxiety is not in your head — it is in the impedance mismatch
     between what your body predicts and what your body measures."

Developmental trajectory:
    Infants: poor interoceptive accuracy → can't tell hungry from tired
    Children: improving → learn to label body signals
    Adults: calibrated → accurate internal state monitoring
    "Teaching a child to name their feelings is calibrating interoception."

Author: Hsi-Yu Huang (黃璽宇)
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

# --- Interoceptive channels ---
INTERO_CHANNELS = [
    "cardiac",          # Heart rate, rhythm
    "respiratory",      # Breathing rate, depth
    "gastric",          # Hunger, satiety, nausea
    "thermal",          # Core body temperature
    "hydration",        # Thirst, fluid balance
    "pain_visceral",    # Internal pain signals
    "fatigue",          # Metabolic fatigue
    "autonomic",        # Sympathetic/parasympathetic balance
]

N_CHANNELS = len(INTERO_CHANNELS)

# --- Baseline values (homeostatic setpoints) ---
HOMEOSTATIC_SETPOINTS = {
    "cardiac": 72.0,        # bpm
    "respiratory": 16.0,    # breaths/min
    "gastric": 0.3,         # 0=empty, 1=full (setpoint = slightly hungry)
    "thermal": 37.0,        # °C
    "hydration": 0.7,       # 0=dehydrated, 1=fully hydrated
    "pain_visceral": 0.0,   # 0=no pain
    "fatigue": 0.2,         # 0=rested, 1=exhausted
    "autonomic": 0.5,       # 0=full parasympathetic, 1=full sympathetic
}

# --- Impedance matching ---
Z_INTERO_BASE = 80.0      # Ω — interoceptive baseline impedance
INTERO_GAIN = 0.3          # Sensitivity scaling
ACCURACY_DEVELOPMENT_RATE = 0.001  # How fast accuracy calibrates

# --- Signal properties ---
INTERO_SNR = 8.0           # dB — interoception is relatively noisy
INTERO_SAMPLE_POINTS = 48
INTERO_FREQUENCY = 1.0     # Hz — slow oscillation (visceral rhythm)

# --- Body budget ---
BUDGET_PREDICTION_RATE = 0.1   # How fast predictions update
BUDGET_ERROR_THRESHOLD = 0.2   # Above this → discomfort signal


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class InteroceptiveState:
    """Current interoceptive readings for all channels."""
    readings: Dict[str, float] = field(default_factory=dict)
    predictions: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, float] = field(default_factory=dict)
    gamma_intero: float = 0.0
    body_budget_balance: float = 1.0    # 1=balanced, 0=depleted
    accuracy: float = 0.3              # Developmental interoceptive accuracy


# ============================================================================
# InteroceptionOrgan
# ============================================================================

class InteroceptionOrgan:
    """
    Alice's Interoceptive System — internal body state monitor.

    Aggregates signals from all internal organs into a unified
    body-state representation. Computes Γ_intero as the mean
    impedance mismatch between predicted and actual internal states.

    Key features:
    1. Multi-channel monitoring (8 interoceptive channels)
    2. Body budget (predictive model of internal states)
    3. Developmental accuracy (starts poor, calibrates over time)
    4. Emotional valence output (feeds insular cortex / amygdala)

    "Interoception is the foundation of emotion.
     You don't feel sad because you're crying —
     you're crying because your interoceptive model predicts tears."
    """

    def __init__(self) -> None:
        # Current measured values
        self._readings: Dict[str, float] = dict(HOMEOSTATIC_SETPOINTS)

        # Predicted values (body budget)
        self._predictions: Dict[str, float] = dict(HOMEOSTATIC_SETPOINTS)

        # Interoceptive accuracy (developmental — starts low)
        self._accuracy: float = 0.3

        # Per-channel Γ values
        self._channel_gamma: Dict[str, float] = {ch: 0.0 for ch in INTERO_CHANNELS}
        self._channel_transmission: Dict[str, float] = {ch: 1.0 for ch in INTERO_CHANNELS}

        # Aggregate
        self._gamma_intero: float = 0.0
        self._transmission_intero: float = 1.0
        self._body_budget: float = 1.0
        self._emotional_valence: float = 0.0  # -1 (distress) to +1 (comfort)

        # Statistics
        self._tick_count: int = 0

    # ------------------------------------------------------------------
    # Update from body signals
    # ------------------------------------------------------------------

    def update_channel(self, channel: str, value: float) -> None:
        """
        Update a specific interoceptive channel with a measured value.

        Args:
            channel: One of INTERO_CHANNELS
            value: The measured value for this channel
        """
        if channel in self._readings:
            self._readings[channel] = value

    def update_from_body(
        self,
        heart_rate: float = 72.0,
        breath_rate: float = 16.0,
        gastric_fill: float = 0.3,
        core_temp: float = 37.0,
        hydration: float = 0.7,
        visceral_pain: float = 0.0,
        fatigue_level: float = 0.2,
        sympathetic_tone: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Batch update all interoceptive channels from body state.

        Returns:
            Dict with aggregate interoceptive assessment.
        """
        self._readings["cardiac"] = heart_rate
        self._readings["respiratory"] = breath_rate
        self._readings["gastric"] = gastric_fill
        self._readings["thermal"] = core_temp
        self._readings["hydration"] = hydration
        self._readings["pain_visceral"] = visceral_pain
        self._readings["fatigue"] = fatigue_level
        self._readings["autonomic"] = sympathetic_tone

        return self._compute_gamma()

    # ------------------------------------------------------------------
    # Compute Γ_intero
    # ------------------------------------------------------------------

    def _compute_gamma(self) -> Dict[str, Any]:
        """Compute per-channel and aggregate Γ_intero."""
        gammas = []
        transmissions = []
        errors = {}

        for ch in INTERO_CHANNELS:
            measured = self._readings[ch]
            predicted = self._predictions[ch]
            setpoint = HOMEOSTATIC_SETPOINTS[ch]

            # Normalize: prediction error relative to setpoint scale
            scale = max(abs(setpoint), 1.0)
            error = abs(measured - predicted) / scale

            # Apply interoceptive accuracy (noisy in infancy)
            perceived_error = error * self._accuracy + (1 - self._accuracy) * np.random.uniform(0, 0.1)

            # Γ = impedance mismatch formula
            z_actual = Z_INTERO_BASE * (1.0 + perceived_error * INTERO_GAIN)
            z_predicted = Z_INTERO_BASE
            gamma = abs(z_actual - z_predicted) / (z_actual + z_predicted)
            transmission = 1.0 - gamma ** 2  # ★ T = 1 − Γ²

            self._channel_gamma[ch] = round(gamma, 4)
            self._channel_transmission[ch] = round(transmission, 4)
            errors[ch] = round(perceived_error, 4)
            gammas.append(gamma)
            transmissions.append(transmission)

        # Aggregate Γ_intero and T_intero
        self._gamma_intero = float(np.mean(gammas))
        self._transmission_intero = float(np.mean(transmissions))

        # ★ Body budget: drain uses ΣΓ²  (total reflected energy depletes budget)
        sum_gamma_sq = float(np.sum([g ** 2 for g in gammas]))
        budget_drain = sum_gamma_sq * 0.16  # ≈ N_channels × 0.02 at moderate mismatch
        self._body_budget = max(0.0, min(1.0, self._body_budget - budget_drain + 0.01))

        # Emotional valence: low Γ → comfort, high Γ → distress
        self._emotional_valence = 1.0 - 2.0 * min(self._gamma_intero / 0.5, 1.0)

        # ★ Hebbian prediction update: Δpred = α × T × (actual − predicted)
        #   Only transmitted information updates the body budget model
        for i, ch in enumerate(INTERO_CHANNELS):
            T_ch = transmissions[i]
            self._predictions[ch] += (
                BUDGET_PREDICTION_RATE * T_ch * (self._readings[ch] - self._predictions[ch])
            )

        return {
            "gamma_intero": round(self._gamma_intero, 4),
            "transmission_intero": round(self._transmission_intero, 4),
            "body_budget": round(self._body_budget, 4),
            "emotional_valence": round(self._emotional_valence, 4),
            "channel_gammas": dict(self._channel_gamma),
            "channel_transmissions": dict(self._channel_transmission),
            "errors": errors,
            "accuracy": round(self._accuracy, 4),
        }

    # ------------------------------------------------------------------
    # Generate ElectricalSignal
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate interoceptive ElectricalSignal."""
        amplitude = float(np.clip(self._gamma_intero * 3.0, 0.05, 1.0))
        freq = INTERO_FREQUENCY + self._gamma_intero * 5.0  # Higher when distressed

        t = np.linspace(0, 1, INTERO_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)

        # Add heartbeat component
        cardiac_freq = self._readings.get("cardiac", 72.0) / 60.0  # bpm → Hz
        waveform = waveform + 0.2 * np.sin(2 * math.pi * cardiac_freq * t)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=Z_INTERO_BASE,
            snr=INTERO_SNR,
            source="interoception",
            modality="interoceptive",
        )

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """Advance interoceptive state by one tick."""
        self._tick_count += 1

        # Developmental accuracy improves with experience
        self._accuracy = min(0.95, self._accuracy + ACCURACY_DEVELOPMENT_RATE)

        # Recompute aggregate
        result = self._compute_gamma()

        return {
            "gamma_intero": result["gamma_intero"],
            "body_budget": result["body_budget"],
            "emotional_valence": result["emotional_valence"],
            "accuracy": round(self._accuracy, 4),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "gamma_intero": round(self._gamma_intero, 4),
            "transmission_intero": round(self._transmission_intero, 4),
            "body_budget": round(self._body_budget, 4),
            "emotional_valence": round(self._emotional_valence, 4),
            "accuracy": round(self._accuracy, 4),
            "channel_gammas": dict(self._channel_gamma),
            "channel_transmissions": dict(self._channel_transmission),
            "readings": dict(self._readings),
            "tick_count": self._tick_count,
        }

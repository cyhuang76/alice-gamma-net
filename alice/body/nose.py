# -*- coding: utf-8 -*-
"""
Alice's Nose — Olfactory Organ

Physics Model: Molecular Impedance Matching
============================================

The olfactory system is modeled as a bank of molecular impedance sensors:
    Each olfactory receptor = tuned impedance matcher for a specific molecular shape
    Odorant → receptor binding = impedance matching at molecular scale
    Γ_olfactory = |Z_odorant - Z_receptor| / (Z_odorant + Z_receptor)

Unique anatomical feature:
    Olfaction is the ONLY sensory modality that bypasses the thalamus.
    Signal pathway: Nose → Olfactory bulb → Piriform cortex → Amygdala/Hippocampus
    This direct amygdala connection explains why smells trigger emotional memories
    so powerfully — no thalamic gating means no attenuation.

    "Proust's madeleine is not literary flourish — it is anatomy."

Core equations:
    Binding_affinity_i = 1 / (1 + |Z_odorant - Z_receptor_i|² / σ²)
    Odor_identity = argmax_i(Binding_affinity_i)
    Emotional_trigger = direct amygdala projection (bypasses thalamic gate)

Adaptation:
    Olfactory adaptation is the fastest of all senses:
    - Continuous exposure → receptor saturation → anosmia (can't smell it anymore)
    - This is impedance matching at work: Z_receptor → Z_odorant → Γ → 0 → no signal

"You stop smelling your own house because your receptors have impedance-matched to it."

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

# --- Receptor bank ---
N_RECEPTOR_TYPES = 20           # Simplified (human: ~400 types, we model 20)
Z_RECEPTOR_BASE = 50.0         # Ω — olfactory receptor base impedance
Z_RECEPTOR_SPREAD = 30.0       # Standard deviation of receptor tuning

# --- Adaptation ---
ADAPTATION_RATE = 0.05          # Receptor saturation rate (fast!)
ADAPTATION_RECOVERY = 0.01     # Recovery rate when odorant removed
ADAPTATION_MAX = 0.95          # Maximum adaptation (near-anosmia)

# --- Emotional coupling (thalamus bypass) ---
EMOTIONAL_COUPLING_GAIN = 1.5  # Olfactory→amygdala coupling is stronger than other senses
HEDONIC_RANGE = (-1.0, 1.0)   # Pleasant (+1) to noxious (-1)

# --- Signal properties ---
NOSE_IMPEDANCE = 45.0          # Ω — lower than most (direct pathway)
NOSE_SNR = 6.0                 # dB — olfaction is inherently noisy
NOSE_SAMPLE_POINTS = 32        # Waveform resolution (simpler than vision/audition)

# --- Detection threshold ---
DETECTION_THRESHOLD = 0.1      # Minimum binding affinity to detect odorant


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class OdorProfile:
    """An odorant's molecular impedance signature."""
    name: str
    z_molecular: float          # Molecular impedance (determines receptor binding)
    hedonic_value: float        # -1 (noxious) to +1 (pleasant)
    volatility: float           # 0~1 (how quickly it dissipates)
    emotional_tag: str = ""     # Optional emotional memory tag


# ============================================================================
# AliceNose
# ============================================================================

class AliceNose:
    """
    Alice's Nose — olfactory molecular impedance sensor bank.

    Key features:
    1. Receptor bank with distributed tuning (20 receptor types)
    2. Fast adaptation (strongest habituation of any sense)
    3. Direct amygdala pathway (bypasses thalamus — unique among senses)
    4. Strong emotional memory coupling

    "All sensory signals (except olfaction) must pass through the thalamus."
    """

    def __init__(self) -> None:
        # Receptor bank: each receptor has a tuned impedance
        rng = np.random.default_rng(42)
        self._receptor_impedances: np.ndarray = (
            Z_RECEPTOR_BASE + rng.normal(0, Z_RECEPTOR_SPREAD, N_RECEPTOR_TYPES)
        ).clip(min=10.0)

        # Adaptation state per receptor
        self._adaptation: np.ndarray = np.zeros(N_RECEPTOR_TYPES)

        # Current odorant
        self._current_odorant: Optional[OdorProfile] = None
        self._binding_affinities: np.ndarray = np.zeros(N_RECEPTOR_TYPES)
        self._detection_strength: float = 0.0
        self._hedonic_response: float = 0.0

        # Statistics
        self._total_sniffs: int = 0
        self._tick_count: int = 0

    # ------------------------------------------------------------------
    # Olfactory transduction
    # ------------------------------------------------------------------

    def sniff(self, odorant: Optional[OdorProfile] = None) -> Dict[str, Any]:
        """
        Process an odorant stimulus.

        Args:
            odorant: The molecular impedance signature of the odorant.
                If None, sniffing clean air.

        Returns:
            Dict with binding pattern, detection, hedonic response, etc.
        """
        self._total_sniffs += 1
        self._current_odorant = odorant

        if odorant is None:
            # Clean air: all affinities zero, adaptation recovers
            self._binding_affinities = np.zeros(N_RECEPTOR_TYPES)
            self._detection_strength = 0.0
            self._hedonic_response = 0.0
            return {
                "detected": False,
                "detection_strength": 0.0,
                "hedonic_response": 0.0,
                "adapted": False,
            }

        # Compute binding affinity for each receptor type
        # Lorentzian resonance: affinity = 1 / (1 + |Z_odorant - Z_receptor|² / σ²)
        z_od = odorant.z_molecular
        delta_z = np.abs(self._receptor_impedances - z_od)
        raw_affinities = 1.0 / (1.0 + (delta_z ** 2) / (Z_RECEPTOR_SPREAD ** 2))

        # ★ Per-receptor Γ and T  (energy conservation: Γ² + T = 1)
        gamma_per_receptor = delta_z / (self._receptor_impedances + z_od + 1e-12)
        transmission_per_receptor = 1.0 - gamma_per_receptor ** 2  # T_i = 1 − Γ_i²

        # Apply adaptation (reduces effective signal)
        effective_affinities = raw_affinities * (1.0 - self._adaptation)
        self._binding_affinities = effective_affinities

        # Detection strength = max effective affinity
        self._detection_strength = float(np.max(effective_affinities))

        # ★ Hebbian adaptation: only transmitted signal drives receptor saturation
        #   Δadaptation = RATE × T × raw_affinity  (T-weighted — more transmission → faster adaptation)
        self._adaptation = np.clip(
            self._adaptation + raw_affinities * transmission_per_receptor * ADAPTATION_RATE,
            0.0, ADAPTATION_MAX,
        )

        # ★ Aggregate transmission metrics
        mean_gamma = float(np.mean(gamma_per_receptor))
        mean_transmission = float(np.mean(transmission_per_receptor))

        # Hedonic response (pleasant/noxious)
        self._hedonic_response = odorant.hedonic_value * self._detection_strength

        # Emotional coupling (direct amygdala pathway)
        emotional_intensity = abs(self._hedonic_response) * EMOTIONAL_COUPLING_GAIN

        detected = self._detection_strength > DETECTION_THRESHOLD

        return {
            "detected": detected,
            "detection_strength": round(self._detection_strength, 4),
            "hedonic_response": round(self._hedonic_response, 4),
            "emotional_intensity": round(emotional_intensity, 4),
            "best_receptor": int(np.argmax(effective_affinities)),
            "adaptation_level": round(float(np.mean(self._adaptation)), 4),
            "adapted": float(np.max(raw_affinities - effective_affinities)) > 0.3,
            "odorant_name": odorant.name,
            "bypasses_thalamus": True,  # Always true for olfaction
            "gamma_olfactory": round(mean_gamma, 4),
            "transmission": round(mean_transmission, 4),
        }

    # ------------------------------------------------------------------
    # Generate ElectricalSignal
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate olfactory ElectricalSignal from current state."""
        freq = 5.0 + self._detection_strength * 40.0  # 5-45 Hz (theta-gamma range)
        amplitude = float(np.clip(self._detection_strength, 0.0, 1.0))

        t = np.linspace(0, 1, NOSE_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)
        noise = np.random.normal(0, amplitude * 0.15, NOSE_SAMPLE_POINTS)
        waveform = waveform + noise

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=NOSE_IMPEDANCE,
            snr=NOSE_SNR,
            source="nose",
            modality="olfactory",
        )

    # ------------------------------------------------------------------
    # Tick — Adaptation recovery, odorant dissipation
    # ------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """Advance olfactory state by one tick."""
        self._tick_count += 1

        # Adaptation recovery (when not actively sniffing)
        self._adaptation = np.clip(
            self._adaptation - ADAPTATION_RECOVERY,
            0.0, ADAPTATION_MAX,
        )

        # Odorant dissipation
        if self._current_odorant is not None:
            # Volatile odorants dissipate faster
            if np.random.random() < self._current_odorant.volatility * 0.1:
                self._current_odorant = None
                self._detection_strength *= 0.9

        return {
            "detection_strength": round(self._detection_strength, 4),
            "adaptation_mean": round(float(np.mean(self._adaptation)), 4),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "detection_strength": round(self._detection_strength, 4),
            "hedonic_response": round(self._hedonic_response, 4),
            "adaptation_mean": round(float(np.mean(self._adaptation)), 4),
            "current_odorant": self._current_odorant.name if self._current_odorant else None,
            "total_sniffs": self._total_sniffs,
            "tick_count": self._tick_count,
        }

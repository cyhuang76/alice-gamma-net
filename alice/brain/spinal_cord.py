# -*- coding: utf-8 -*-
"""
Alice Smart System — Spinal Cord (SpinalCord)

Physics Model: High-Speed Reflex Transmission Line
===================================================

The spinal cord is modeled as a high-speed transmission line with:
    Sensory afferent = input line (skin/proprioception → spinal cord)
    Motor efferent   = output line (spinal cord → muscle)
    Interneurons     = local processing nodes (reflex arcs)
    White matter     = long-distance impedance-matched trunks (brain ↔ body)
    Grey matter      = local reflex processing
    Dermatome map    = spatial impedance addressing

Core equations:
    reflex_latency = transmission_length / conduction_velocity
    conduction_velocity ∝ myelination × axon_diameter
    Γ_spinal = (Z_afferent − Z_efferent) / (Z_afferent + Z_efferent)
    T_spinal = 1 − Γ²_spinal  (★ C1)
    reflex_response = sensory_input × reflex_gain × T_spinal  (bypasses brain)
    ΔZ_reflex = −η × Γ × stimulus × response  (★ C2 Hebbian)

Reflex arcs:
    1. Stretch reflex (monosynaptic) — fastest: ~30ms
    2. Withdrawal reflex (polysynaptic) — pain → flexor activation
    3. Crossed extensor — contralateral stabilization
    4. Autonomic reflexes — visceral regulation

Clinical significance:
    Spinal cord injury → Γ = 1.0 at lesion → total signal block
    Demyelination (MS) → conduction velocity ↓ → latency ↑
    Hyperreflexia → reflex gain too high → spasticity
    Areflexia → reflex gain = 0 → flaccid paralysis

"The spinal cord is not a relay — it is a high-speed reflex computer.
 It processes pain before your brain even knows what happened."

Author: Hsi-Yu Huang (黃璽宇)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import ElectricalSignal

# ============================================================================
# Physical Constants
# ============================================================================

# --- Transmission line ---
Z_AFFERENT = 60.0              # Ω — sensory input impedance
Z_EFFERENT = 60.0              # Ω — motor output impedance
Z_INTERNEURON = 65.0           # Ω — local processing impedance
CONDUCTION_VELOCITY_MAX = 120.0 # m/s (myelinated Aα fibers)
CONDUCTION_VELOCITY_MIN = 0.5   # m/s (unmyelinated C fibers)

# --- Myelination ---
MYELINATION_NEONATAL = 0.3     # Neonates: poorly myelinated
MYELINATION_ADULT = 0.95       # Adult: well myelinated
MYELINATION_RATE = 0.0005      # Per tick development

# --- Reflex arcs ---
REFLEX_TYPES = ["stretch", "withdrawal", "crossed_extensor", "autonomic"]
REFLEX_GAIN_BASE = 0.5         # Baseline reflex amplitude
REFLEX_GAIN_MAX = 1.0
REFLEX_GAIN_MIN = 0.05
REFLEX_ADAPTATION_RATE = 0.005 # Hebbian refinement

# --- Latency ---
MONOSYNAPTIC_LATENCY = 0.03   # 30 ms (stretch reflex)
POLYSYNAPTIC_LATENCY = 0.08   # 80 ms (withdrawal reflex)
BRAIN_BYPASS_SPEEDUP = 3.0     # Reflex is N× faster than cortical

# --- Dermatome regions ---
DERMATOME_LEVELS = ["C", "T", "L", "S"]  # Cervical, Thoracic, Lumbar, Sacral
N_SEGMENTS = 31                 # Approximate spinal segments

# --- Signal ---
SPINAL_IMPEDANCE = 60.0        # Ω
SPINAL_SNR = 12.0              # dB — spinal signals are clean (myelinated)
SPINAL_SAMPLE_POINTS = 64
SPINAL_FREQUENCY = 15.0        # Hz — fast reflex frequency

# --- Pain gate ---
PAIN_GATE_THRESHOLD = 0.4      # Gate control theory: non-nociceptive input blocks pain
PAIN_GATE_INHIBITION = 0.5


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ReflexArc:
    """A single reflex arc with gain and history."""
    reflex_type: str
    gain: float = REFLEX_GAIN_BASE
    latency: float = MONOSYNAPTIC_LATENCY
    last_activation: float = 0.0
    total_activations: int = 0


@dataclass
class SpinalState:
    """Snapshot of spinal cord state."""
    myelination: float = MYELINATION_NEONATAL
    conduction_velocity: float = 0.0
    gamma_spinal: float = 0.0
    transmission_spinal: float = 1.0
    active_reflexes: int = 0
    pain_gate_open: bool = False


# ============================================================================
# SpinalCord
# ============================================================================

class SpinalCord:
    """
    Alice's Spinal Cord — high-speed reflex transmission line.

    Models reflex arcs that bypass the brain for fast responses,
    dermatome-organized sensory mapping, and pain gate control.

    All signals use ElectricalSignal (★ C3).
    """

    def __init__(self) -> None:
        self._myelination: float = MYELINATION_NEONATAL

        # Reflex arcs
        self._reflexes: Dict[str, ReflexArc] = {
            rt: ReflexArc(
                reflex_type=rt,
                gain=REFLEX_GAIN_BASE,
                latency=(MONOSYNAPTIC_LATENCY if rt == "stretch"
                         else POLYSYNAPTIC_LATENCY),
            )
            for rt in REFLEX_TYPES
        }

        # Segment integrity (0=damaged, 1=healthy)
        self._segment_health: np.ndarray = np.ones(N_SEGMENTS)

        # Pain gate
        self._pain_gate_open: bool = False
        self._pain_level: float = 0.0

        # Γ
        self._gamma_spinal: float = 0.0
        self._transmission_spinal: float = 1.0

        # Conduction
        self._conduction_velocity: float = (
            CONDUCTION_VELOCITY_MIN +
            self._myelination * (CONDUCTION_VELOCITY_MAX - CONDUCTION_VELOCITY_MIN)
        )

        # Statistics
        self._tick_count: int = 0
        self._total_reflexes: int = 0

    # ------------------------------------------------------------------
    # Reflex activation
    # ------------------------------------------------------------------

    def activate_reflex(
        self,
        reflex_type: str = "stretch",
        stimulus_intensity: float = 0.5,
        segment: int = 15,
    ) -> Dict[str, Any]:
        """
        Trigger a spinal reflex arc.

        Args:
            reflex_type: One of REFLEX_TYPES
            stimulus_intensity: Stimulus strength (0–1)
            segment: Spinal segment (0–30)
        """
        if reflex_type not in self._reflexes:
            reflex_type = "stretch"

        reflex = self._reflexes[reflex_type]
        seg = int(np.clip(segment, 0, N_SEGMENTS - 1))
        seg_health = float(self._segment_health[seg])
        stim = float(np.clip(stimulus_intensity, 0, 1))

        # Γ for this segment
        z_in = Z_AFFERENT * (2.0 - seg_health)  # Damage increases Z
        gamma = abs(z_in - Z_EFFERENT) / (z_in + Z_EFFERENT)
        transmission = 1.0 - gamma ** 2  # ★ C1

        # Response: bypasses brain
        latency = reflex.latency / (self._myelination + 0.1)
        response = stim * reflex.gain * transmission * seg_health

        # ★ C2 Hebbian: tune reflex gain
        delta_gain = REFLEX_ADAPTATION_RATE * gamma * stim * response
        reflex.gain = float(np.clip(reflex.gain - delta_gain,
                                     REFLEX_GAIN_MIN, REFLEX_GAIN_MAX))

        reflex.last_activation = response
        reflex.total_activations += 1
        self._total_reflexes += 1

        return {
            "reflex_type": reflex_type,
            "response": round(response, 4),
            "latency_ms": round(latency * 1000, 1),
            "gamma_segment": round(gamma, 4),
            "transmission": round(transmission, 4),
            "segment": seg,
        }

    # ------------------------------------------------------------------
    # Pain gate control
    # ------------------------------------------------------------------

    def pain_gate(self, nociceptive: float = 0.0,
                  non_nociceptive: float = 0.0) -> Dict[str, Any]:
        """
        Melzack & Wall pain gate control.

        Non-nociceptive input (touch, vibration) can close the gate
        and reduce pain transmission.

        Args:
            nociceptive: Pain signal strength (0–1)
            non_nociceptive: Touch/vibration signal (0–1)
        """
        noci = float(np.clip(nociceptive, 0, 1))
        non_noci = float(np.clip(non_nociceptive, 0, 1))

        # Gate opens when nociceptive > threshold and non-nociceptive is low
        gate_balance = noci - non_noci * PAIN_GATE_INHIBITION
        self._pain_gate_open = gate_balance > PAIN_GATE_THRESHOLD

        if self._pain_gate_open:
            self._pain_level = noci * (1.0 - non_noci * 0.5)
        else:
            self._pain_level = max(0.0, noci * 0.2)

        return {
            "pain_gate_open": self._pain_gate_open,
            "pain_level": round(self._pain_level, 4),
            "gate_balance": round(gate_balance, 4),
        }

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """Advance spinal cord by one tick."""
        self._tick_count += 1

        # Myelination development
        self._myelination = min(MYELINATION_ADULT,
                                self._myelination + MYELINATION_RATE)

        # Conduction velocity
        self._conduction_velocity = (
            CONDUCTION_VELOCITY_MIN +
            self._myelination * (CONDUCTION_VELOCITY_MAX - CONDUCTION_VELOCITY_MIN)
        )

        # Aggregate Γ_spinal (mean segment health)
        mean_health = float(np.mean(self._segment_health))
        z_eff = Z_AFFERENT * (2.0 - mean_health)
        self._gamma_spinal = abs(z_eff - Z_EFFERENT) / (z_eff + Z_EFFERENT)
        self._transmission_spinal = 1.0 - self._gamma_spinal ** 2  # ★ C1

        return {
            "gamma_spinal": round(self._gamma_spinal, 4),
            "transmission_spinal": round(self._transmission_spinal, 4),
            "myelination": round(self._myelination, 4),
            "conduction_velocity": round(self._conduction_velocity, 1),
            "pain_gate_open": self._pain_gate_open,
            "pain_level": round(self._pain_level, 4),
        }

    # ------------------------------------------------------------------
    # Injury simulation
    # ------------------------------------------------------------------

    def injure_segment(self, segment: int, severity: float = 0.5) -> None:
        """Simulate spinal cord injury at a specific segment."""
        seg = int(np.clip(segment, 0, N_SEGMENTS - 1))
        sev = float(np.clip(severity, 0, 1))
        self._segment_health[seg] = max(0.0, self._segment_health[seg] - sev)

    # ------------------------------------------------------------------
    # ElectricalSignal (★ C3)
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate spinal cord status signal."""
        amplitude = float(np.clip(0.2 + self._pain_level * 0.5, 0.05, 1.0))
        freq = SPINAL_FREQUENCY * self._myelination

        t = np.linspace(0, 1, SPINAL_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=SPINAL_IMPEDANCE,
            snr=SPINAL_SNR * self._myelination,
            source="spinal_cord",
            modality="motor",
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get spinal cord statistics."""
        return {
            "gamma_spinal": round(self._gamma_spinal, 4),
            "transmission_spinal": round(self._transmission_spinal, 4),
            "myelination": round(self._myelination, 4),
            "conduction_velocity": round(self._conduction_velocity, 1),
            "mean_segment_health": round(float(np.mean(self._segment_health)), 4),
            "pain_gate_open": self._pain_gate_open,
            "pain_level": round(self._pain_level, 4),
            "reflex_gains": {rt: round(r.gain, 4) for rt, r in self._reflexes.items()},
            "total_reflexes": self._total_reflexes,
            "tick_count": self._tick_count,
        }

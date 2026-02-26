# -*- coding: utf-8 -*-
"""
Alice Smart System — Lymphatic System (LymphaticSystem)

Physics Model: Impedance Drainage Network
==========================================

The lymphatic system is modeled as a low-pressure drainage network where:
    Interstitial fluid = leaked signal ground (excess from capillaries)
    Lymph vessels      = low-impedance return lines
    Lymph nodes        = impedance inspection stations (immune checkpoints)
    Thoracic duct      = final return to main transmission line (venous)

Core equations:
    lymph_flow = interstitial_pressure × vessel_patency × muscle_pump
    immune_surveillance = lymph_flow × node_activity × T_node
    T_node = 1 − Γ²_node  (★ C1 energy conservation)
    Γ_node = (Z_lymph − Z_node) / (Z_lymph + Z_node)
    edema = interstitial_fluid − lymph_drainage

Clinical significance:
    Lymphedema → drainage failure → interstitial fluid accumulation
    Infection  → lymph node activation → swelling (node Γ ↑)
    Immune patrol → lymphocytes circulate via lymph → systemic coverage

"The lymphatic system is the body's impedance drainage network.
 Without it, the signal ground floods."

Author: Hsi-Yu Huang (黃璽宇)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from alice.core.signal import ElectricalSignal

# ============================================================================
# Physical Constants
# ============================================================================

# --- Lymph flow ---
LYMPH_FLOW_BASE = 0.5           # Normalized lymph flow
LYMPH_FLOW_MAX = 1.0
MUSCLE_PUMP_GAIN = 0.3          # Physical activity → lymph flow ↑
INTERSTITIAL_PRESSURE_BASE = 0.3

# --- Lymph nodes ---
N_NODE_REGIONS = 6              # Cervical, axillary, inguinal, mesenteric, mediastinal, popliteal
Z_LYMPH = 55.0                  # Ω — lymph fluid impedance
Z_NODE_BASE = 55.0              # Ω — healthy node impedance
NODE_ACTIVATION_GAIN = 0.4      # Infection → node activation

# --- Drainage ---
DRAINAGE_RATE_BASE = 0.06       # Fluid drained per tick
EDEMA_THRESHOLD = 0.5           # Above this → clinical edema

# --- Immune patrol ---
LYMPHOCYTE_CIRCULATION_BASE = 0.5
LYMPHOCYTE_INFECTION_BOOST = 0.4

# --- Signal ---
LYMPHATIC_IMPEDANCE = 55.0      # Ω
LYMPHATIC_SNR = 5.0             # dB
LYMPHATIC_SAMPLE_POINTS = 48
LYMPHATIC_FREQUENCY = 0.15      # Hz

# --- Development ---
NEONATAL_LYMPHATIC_MATURITY = 0.3
LYMPHATIC_MATURATION_RATE = 0.0004


# ============================================================================
# LymphaticSystem
# ============================================================================

class LymphaticSystem:
    """
    Alice's Lymphatic System — impedance drainage and immune patrol.

    Models lymph flow, node activation, edema, and immune surveillance.
    All signals use ElectricalSignal (★ C3).
    """

    def __init__(self) -> None:
        self._maturity: float = NEONATAL_LYMPHATIC_MATURITY
        self._lymph_flow: float = LYMPH_FLOW_BASE
        self._interstitial_fluid: float = INTERSTITIAL_PRESSURE_BASE
        self._edema: float = 0.0

        # Node states (per region)
        self._node_activation: Dict[str, float] = {
            f"region_{i}": 0.0 for i in range(N_NODE_REGIONS)
        }
        self._lymphocyte_level: float = LYMPHOCYTE_CIRCULATION_BASE

        # Γ
        self._gamma_lymphatic: float = 0.0
        self._transmission_lymphatic: float = 1.0

        # Statistics
        self._tick_count: int = 0
        self._edema_ticks: int = 0

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        physical_activity: float = 0.3,
        inflammation: float = 0.0,
        heart_rate: float = 72.0,
        blood_pressure_norm: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Advance lymphatic system by one tick.

        Args:
            physical_activity: Motor activity level (0–1)
            inflammation: Current inflammation level from immune system
            heart_rate: Current HR (affects capillary filtration)
            blood_pressure_norm: Normalized BP (affects fluid leak)
        """
        self._tick_count += 1
        self._maturity = min(1.0, self._maturity + LYMPHATIC_MATURATION_RATE)

        activity = float(np.clip(physical_activity, 0, 1))
        inflam = float(np.clip(inflammation, 0, 1))
        bp = float(np.clip(blood_pressure_norm, 0, 1))

        # Interstitial fluid accumulation (from capillary filtration)
        capillary_leak = bp * 0.02 + inflam * 0.05
        self._interstitial_fluid += capillary_leak

        # Lymph flow: muscle pump + intrinsic contractions
        muscle_pump = MUSCLE_PUMP_GAIN * activity
        self._lymph_flow = float(np.clip(
            LYMPH_FLOW_BASE + muscle_pump * self._maturity,
            0.1, LYMPH_FLOW_MAX
        ))

        # Drainage
        drainage = DRAINAGE_RATE_BASE * self._lymph_flow * self._maturity
        self._interstitial_fluid = max(0.0, self._interstitial_fluid - drainage)

        # Edema
        self._edema = max(0.0, self._interstitial_fluid - EDEMA_THRESHOLD)
        if self._edema > 0:
            self._edema_ticks += 1

        # Node activation (immune checkpoints)
        mean_activation = 0.0
        for region in self._node_activation:
            activation = inflam * NODE_ACTIVATION_GAIN * self._maturity
            # Decay
            current = self._node_activation[region]
            self._node_activation[region] = float(np.clip(
                current * 0.9 + activation * 0.1, 0, 1
            ))
            mean_activation += self._node_activation[region]
        mean_activation /= max(len(self._node_activation), 1)

        # Lymphocyte circulation
        self._lymphocyte_level = float(np.clip(
            LYMPHOCYTE_CIRCULATION_BASE + inflam * LYMPHOCYTE_INFECTION_BOOST,
            0, 1
        ))

        # Γ_lymphatic
        z_lymph_eff = Z_LYMPH + inflam * 15.0  # Inflammation alters lymph Z
        self._gamma_lymphatic = abs(z_lymph_eff - Z_NODE_BASE) / (z_lymph_eff + Z_NODE_BASE)
        self._transmission_lymphatic = 1.0 - self._gamma_lymphatic ** 2  # ★ C1

        return {
            "gamma_lymphatic": round(self._gamma_lymphatic, 4),
            "transmission_lymphatic": round(self._transmission_lymphatic, 4),
            "lymph_flow": round(self._lymph_flow, 4),
            "interstitial_fluid": round(self._interstitial_fluid, 4),
            "edema": round(self._edema, 4),
            "node_mean_activation": round(mean_activation, 4),
            "lymphocyte_level": round(self._lymphocyte_level, 4),
            "maturity": round(self._maturity, 4),
        }

    # ------------------------------------------------------------------
    # ElectricalSignal (★ C3)
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate lymphatic status signal."""
        amplitude = float(np.clip(0.1 + self._edema * 0.5 + self._gamma_lymphatic, 0.05, 1.0))
        freq = LYMPHATIC_FREQUENCY + self._gamma_lymphatic * 1.0

        t = np.linspace(0, 1, LYMPHATIC_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=LYMPHATIC_IMPEDANCE,
            snr=LYMPHATIC_SNR,
            source="lymphatic",
            modality="interoceptive",
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get lymphatic system statistics."""
        return {
            "gamma_lymphatic": round(self._gamma_lymphatic, 4),
            "transmission_lymphatic": round(self._transmission_lymphatic, 4),
            "lymph_flow": round(self._lymph_flow, 4),
            "interstitial_fluid": round(self._interstitial_fluid, 4),
            "edema": round(self._edema, 4),
            "node_activations": {k: round(v, 4) for k, v in self._node_activation.items()},
            "lymphocyte_level": round(self._lymphocyte_level, 4),
            "maturity": round(self._maturity, 4),
            "edema_ticks": self._edema_ticks,
            "tick_count": self._tick_count,
        }

# -*- coding: utf-8 -*-
"""
Alice's Skin — Somatosensory Organ (Temperature, Touch, Nociception)

Physics Model: Distributed Impedance Sensor Array
==================================================

The skin is modeled as a distributed array of impedance sensors where:
    Touch    = pressure wave → impedance transduction → ElectricalSignal
    Temperature = thermal gradient → impedance drift → ElectricalSignal
    Nociception = extreme stimulus → damage signal → pain pathway (bypasses thalamus)

Core equations:
    Z_skin(T) = Z_base × (1 + α_T × (T - T_ref))
    Γ_touch = |Z_object - Z_skin| / (Z_object + Z_skin)
    Pain_cutaneous = Σ max(0, stimulus_i - threshold_i)²

The skin is not a single sensor — it is a spatially distributed field of
impedance transducers, each with different sensitivity to different modalities:
    Mechanoreceptors (Meissner, Pacinian, Merkel, Ruffini)
    Thermoreceptors (warm, cold)
    Nociceptors (pain — the most ancient and important)

"Skin is the boundary between self and world — every touch is an impedance measurement."

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

# --- Base impedance ---
Z_SKIN_BASE = 60.0             # Ω — skin characteristic impedance (between muscle and air)
TEMPERATURE_COEFFICIENT = 0.02  # α_T: impedance change per °C deviation from T_ref
T_REF = 37.0                   # Reference temperature (body temperature, °C)
T_COMFORT_LOW = 32.0           # Comfortable skin temperature lower bound
T_COMFORT_HIGH = 38.0          # Comfortable skin temperature upper bound
T_PAIN_COLD = 15.0             # Cold pain threshold
T_PAIN_HOT = 45.0              # Heat pain threshold

# --- Touch sensitivity ---
TOUCH_THRESHOLD = 0.05         # Minimum pressure to register touch
TOUCH_ADAPTATION_RATE = 0.03   # How quickly touch receptors adapt (habituation)
TOUCH_SPATIAL_RESOLUTION = 10  # Number of simulated sensor zones

# --- Pain (nociception) ---
NOCICEPTION_THRESHOLD = 0.7    # Stimulus intensity above this → pain
NOCICEPTION_GAIN = 2.0         # Pain amplification factor
NOCICEPTION_SENSITIZATION = 0.02  # Pain threshold lowering after repeated pain

# --- Signal properties ---
SKIN_SNR = 8.0                 # dB — somatosensory signals are noisy
SKIN_SAMPLE_POINTS = 64        # Waveform resolution


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SkinState:
    """Snapshot of skin state."""
    skin_temperature: float = T_REF
    z_skin: float = Z_SKIN_BASE
    touch_pressure: float = 0.0
    pain_level: float = 0.0
    adapted_threshold: float = TOUCH_THRESHOLD
    total_touches: int = 0
    total_pain_events: int = 0


@dataclass
class TouchEvent:
    """A single touch stimulus."""
    pressure: float             # 0~1 normalized pressure
    zone: int                   # Spatial zone (0~9)
    temperature: float          # Object temperature (°C)
    is_noxious: bool = False    # Whether this is a harmful stimulus


# ============================================================================
# AliceSkin
# ============================================================================

class AliceSkin:
    """
    Alice's Skin — distributed somatosensory impedance sensor array.

    Transduces three modalities:
    1. Touch (mechanoreception) → pressure waves → ElectricalSignal
    2. Temperature (thermoreception) → thermal gradient → ElectricalSignal
    3. Pain (nociception) → extreme stimulus → high-priority pain signal

    Each touch is an impedance measurement: Γ_touch = |Z_object - Z_skin| / (Z_object + Z_skin)
    Perfect touch (matched impedance) → Γ = 0 → maximum information transfer
    Harsh touch (extreme impedance) → Γ → 1 → mostly reflection → pain
    """

    def __init__(self) -> None:
        # State
        self._skin_temperature: float = T_REF
        self._z_skin: float = Z_SKIN_BASE
        self._touch_pressure: float = 0.0
        self._pain_level: float = 0.0

        # Adaptation
        self._adapted_threshold: float = TOUCH_THRESHOLD
        self._nociception_threshold: float = NOCICEPTION_THRESHOLD
        self._habituation: float = 0.0  # Touch habituation (reduces sensitivity)

        # Sensor zones (rough body map)
        self._zone_activation: np.ndarray = np.zeros(TOUCH_SPATIAL_RESOLUTION)

        # Statistics
        self._total_touches: int = 0
        self._total_pain_events: int = 0
        self._tick_count: int = 0

    # ------------------------------------------------------------------
    # Temperature-dependent impedance
    # ------------------------------------------------------------------

    def _update_impedance(self) -> None:
        """Z_skin(T) = Z_base × (1 + α_T × (T - T_ref))"""
        delta_t = self._skin_temperature - T_REF
        self._z_skin = Z_SKIN_BASE * (1.0 + TEMPERATURE_COEFFICIENT * delta_t)
        self._z_skin = max(10.0, self._z_skin)

    # ------------------------------------------------------------------
    # Touch transduction
    # ------------------------------------------------------------------

    def touch(
        self,
        pressure: float = 0.0,
        zone: int = 0,
        object_temperature: float = T_REF,
        object_impedance: float = 75.0,
    ) -> Dict[str, Any]:
        """
        Process a touch stimulus.

        Args:
            pressure: 0~1 touch pressure
            zone: Spatial zone (0~9)
            object_temperature: Temperature of touching object
            object_impedance: Impedance of touching object

        Returns:
            Dict with touch signal, pain level, gamma, etc.
        """
        self._total_touches += 1
        zone = int(np.clip(zone, 0, TOUCH_SPATIAL_RESOLUTION - 1))

        # Update skin temperature (gradual adaptation to object)
        self._skin_temperature += (object_temperature - self._skin_temperature) * 0.1
        self._update_impedance()

        # Compute touch Γ
        z_obj = max(1.0, object_impedance)
        gamma_touch = abs(z_obj - self._z_skin) / (z_obj + self._z_skin)
        transmission = 1.0 - gamma_touch ** 2  # ★ T = 1 − Γ²
        reflected_energy = gamma_touch ** 2 * max(pressure, 0.01)  # ★ E_reflected = Γ² × P

        # Touch detection (with habituation)
        effective_pressure = max(0.0, pressure - self._habituation)
        self._touch_pressure = effective_pressure

        # Habituation: sustained touch → reduced sensitivity
        if pressure > TOUCH_THRESHOLD:
            self._habituation = min(0.5, self._habituation + TOUCH_ADAPTATION_RATE)
        else:
            self._habituation = max(0.0, self._habituation - TOUCH_ADAPTATION_RATE * 0.5)

        # Zone activation
        self._zone_activation[zone] = min(1.0, self._zone_activation[zone] + pressure)

        # Temperature pain check
        temp_pain = 0.0
        if self._skin_temperature < T_PAIN_COLD:
            temp_pain = (T_PAIN_COLD - self._skin_temperature) / T_PAIN_COLD
        elif self._skin_temperature > T_PAIN_HOT:
            temp_pain = (self._skin_temperature - T_PAIN_HOT) / (100.0 - T_PAIN_HOT)

        # Nociception check
        nociception_pain = 0.0
        if pressure > self._nociception_threshold:
            nociception_pain = NOCICEPTION_GAIN * (pressure - self._nociception_threshold)
            self._total_pain_events += 1
            # ★ Hebbian sensitization: Δθ = -η × Γ² × pressure
            # More reflection (higher mismatch) → more pain → more sensitization
            self._nociception_threshold = max(
                0.3, self._nociception_threshold - NOCICEPTION_SENSITIZATION * gamma_touch ** 2 * pressure
            )

        self._pain_level = float(np.clip(max(temp_pain, nociception_pain), 0.0, 1.0))

        return {
            "gamma_touch": round(gamma_touch, 4),
            "transmission": round(transmission, 4),
            "reflected_energy": round(reflected_energy, 6),
            "pressure": round(effective_pressure, 4),
            "z_skin": round(self._z_skin, 2),
            "skin_temperature": round(self._skin_temperature, 2),
            "pain_level": round(self._pain_level, 4),
            "zone": zone,
            "habituation": round(self._habituation, 4),
        }

    # ------------------------------------------------------------------
    # Generate ElectricalSignal
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate somatosensory ElectricalSignal from current skin state."""
        # Frequency: low for temperature, mid for light touch, high for pain
        if self._pain_level > 0.3:
            freq = 40.0 + self._pain_level * 80.0  # 40-120 Hz (C-fiber → Aδ)
        elif self._touch_pressure > TOUCH_THRESHOLD:
            freq = 10.0 + self._touch_pressure * 60.0  # 10-70 Hz (Meissner/Pacinian)
        else:
            freq = 2.0  # Baseline tonic signal

        amplitude = max(self._touch_pressure, self._pain_level) * 0.8
        t = np.linspace(0, 1, SKIN_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)
        # Add noise
        noise = np.random.normal(0, amplitude * 0.1, SKIN_SAMPLE_POINTS)
        waveform = waveform + noise

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=float(np.clip(amplitude, 0.0, 1.0)),
            phase=0.0,
            impedance=self._z_skin,
            snr=SKIN_SNR,
            source="skin",
            modality="somatosensory",
        )

    # ------------------------------------------------------------------
    # Tick — Zone decay, temperature drift
    # ------------------------------------------------------------------

    def tick(self, ambient_temperature: float = 25.0) -> Dict[str, Any]:
        """Advance skin state by one tick."""
        self._tick_count += 1

        # Zone activation decay
        self._zone_activation *= 0.95

        # Skin temperature drifts toward ambient (not touching anything)
        self._skin_temperature += (ambient_temperature - self._skin_temperature) * 0.01
        self._update_impedance()

        # Pain decays
        self._pain_level = max(0.0, self._pain_level - 0.02)

        return {
            "skin_temperature": round(self._skin_temperature, 2),
            "z_skin": round(self._z_skin, 2),
            "pain_level": round(self._pain_level, 4),
            "active_zones": int(np.sum(self._zone_activation > 0.1)),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "skin_temperature": round(self._skin_temperature, 2),
            "z_skin": round(self._z_skin, 2),
            "touch_pressure": round(self._touch_pressure, 4),
            "pain_level": round(self._pain_level, 4),
            "habituation": round(self._habituation, 4),
            "nociception_threshold": round(self._nociception_threshold, 4),
            "total_touches": self._total_touches,
            "total_pain_events": self._total_pain_events,
            "active_zones": int(np.sum(self._zone_activation > 0.1)),
            "tick_count": self._tick_count,
        }

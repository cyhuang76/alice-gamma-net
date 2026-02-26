# -*- coding: utf-8 -*-
"""
Alice's Autonomic Nervous System — Visceral Physics Control

Physics:
  "The autonomic nervous system is the body's thermostat.
    The sympathetic is the accelerator, the parasympathetic is the brake.
    The balance between them determines your body state."

  Sympathetic — Fight or Flight
    ├── Heart rate ↑         (adrenaline)
    ├── Respiration ↑
    ├── Pupil ↑              (mydriasis)
    ├── Muscle tension ↑
    └── Digestion ↓

  Parasympathetic — Rest and Digest
    ├── Heart rate ↓         (acetylcholine)
    ├── Respiration ↓
    ├── Pupil ↓              (miosis)
    ├── Muscle relaxation
    └── Digestion ↑

  Homeostasis:
    Sympathetic and parasympathetic are always simultaneously active and antagonistic.
    The body's "normal state" = dynamic equilibrium of both.
    Breaking balance → stress response → energy expenditure → recovery needed.

  "This is not a mental problem. This is a thermostat set to the wrong temperature."

Circuit analogy:
  Sympathetic = positive voltage (drives the system to accelerate)
  Parasympathetic = negative feedback (voltage regulator circuit)
  Homeostasis = regulator output → always trends toward the reference voltage
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from alice.core.signal import ElectricalSignal


# ============================================================================
# Autonomic nervous system physical constants
# ============================================================================

# Heart rate
RESTING_HEART_RATE = 60.0       # bpm
MAX_HEART_RATE = 200.0          # bpm (sympathetic fully activated)
MIN_HEART_RATE = 40.0           # bpm (parasympathetic dominant / deep sleep)

# Respiration
RESTING_BREATH_RATE = 15.0      # breaths/min
MAX_BREATH_RATE = 40.0          # breaths/min (hyperventilation)
MIN_BREATH_RATE = 4.0           # breaths/min (deep sleep)

# Pupil
RESTING_PUPIL_SIZE = 0.5        # Normalized (0=pinhole, 1=fully dilated)

# Body temperature
CORE_TEMPERATURE = 37.0         # °C
HYPOTHERMIA_THRESHOLD = 35.0    # °C
HYPERTHERMIA_THRESHOLD = 39.0   # °C

# Cortisol (stress hormone)
CORTISOL_HALF_LIFE = 90.0       # minutes (simulated as tick count)
CORTISOL_DECAY = 0.98           # Decay rate per tick (≈ half-life)

# Autonomic response rates
SYMPATHETIC_RISE_RATE = 0.15    # Sympathetic excitation rate
PARASYMPATHETIC_RISE_RATE = 0.08  # Parasympathetic recovery rate (slower than sympathetic)
HOMEOSTASIS_STRENGTH = 0.05     # Homeostasis recovery strength

# Energy
ENERGY_MAX = 1.0
ENERGY_RESTING_RECOVERY = 0.005    # Energy recovery during rest
ENERGY_STRESS_COST = 0.01          # Energy cost of stress

# Impedance
AUTONOMIC_IMPEDANCE = 75.0      # Ω


# ============================================================================
# Main class
# ============================================================================


class AutonomicNervousSystem:
    """
    Autonomic Nervous System — the body's thermostat

    Sympathetic vs parasympathetic dynamic antagonism → homeostasis
    Drives all visceral indicators: heart rate, respiration, pupil, energy, cortisol
    """

    def __init__(self):
        # Neural activity (0~1, represents "how deep the pedal is pressed")
        self.sympathetic: float = 0.2       # Sympathetic activity (baseline slightly active)
        self.parasympathetic: float = 0.3   # Parasympathetic activity (baseline dominant)

        # Visceral indicators
        self.heart_rate: float = RESTING_HEART_RATE
        self.breath_rate: float = RESTING_BREATH_RATE
        self.pupil_size: float = RESTING_PUPIL_SIZE
        self.core_temp: float = CORE_TEMPERATURE
        self.cortisol: float = 0.1          # Baseline cortisol
        self.energy: float = ENERGY_MAX     # Energy reserve

        # ★ Chronic stress accumulation — cumulative damage from repeated injuries
        # Physics: repeatedly overloaded circuit → component aging → elevated baseline impedance → worse heat dissipation
        self.chronic_stress_load: float = 0.0   # Chronic stress load (0~1)
        self.sympathetic_baseline: float = 0.2  # Sympathetic baseline (shifts up after repeated trauma)
        self.parasympathetic_baseline: float = 0.3  # Parasympathetic baseline (shifts down after repeated trauma)
        self.trauma_events: int = 0  # Trauma event count

        # ★ Central apnea flag — Ondine's Curse (CCHS / PHOX2B mutation)
        #   When True: brainstem cannot drive breathing during sleep
        #   Awake: cortical voluntary breathing still works (normal)
        #   Asleep: breath_rate → 0 (autonomic respiratory control absent)
        self.central_apnea: bool = False

        # History
        self.sympathetic_history: List[float] = []
        self.parasympathetic_history: List[float] = []
        self.heart_rate_history: List[float] = []
        self.breath_rate_history: List[float] = []
        self.energy_history: List[float] = []
        self.cortisol_history: List[float] = []

        # Statistics
        self.total_ticks: int = 0
        self.stress_events: int = 0
        self.exhaustion_events: int = 0

        self._max_history: int = 300

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def tick(
        self,
        pain_level: float = 0.0,
        ram_temperature: float = 0.0,
        emotional_valence: float = 0.0,
        sensory_load: float = 0.0,
        is_sleeping: bool = False,
        voluntary_breath: Optional[float] = None,
    ):
        """
        Called once per cognitive cycle — updates all autonomic indicators

        THE AUTONOMIC LOOP:
        1. Stress input → sympathetic excitation
        2. Safety input → parasympathetic excitation
        3. Homeostasis → both tend towards balance
        4. Sympathetic/parasympathetic ratio → drives heart rate, respiration, pupil
        5. Sustained stress → cortisol accumulation → energy consumption
        """
        self.total_ticks += 1

        # === 1. Sympathetic input (sum of threat signals) ===
        threat_level = (
            pain_level * 0.4 +
            ram_temperature * 0.3 +
            max(0.0, -emotional_valence) * 0.15 +
            sensory_load * 0.15
        )
        threat_level = float(np.clip(threat_level, 0.0, 1.0))

        # Sympathetic excitation
        sym_target = 0.1 + 0.9 * threat_level
        self.sympathetic += (sym_target - self.sympathetic) * SYMPATHETIC_RISE_RATE
        self.sympathetic = float(np.clip(self.sympathetic, 0.0, 1.0))

        # === 2. Parasympathetic input (safety signals) ===
        safety_level = (
            (1.0 - pain_level) * 0.3 +
            (1.0 - ram_temperature) * 0.3 +
            max(0.0, emotional_valence) * 0.2 +
            (1.0 if is_sleeping else 0.0) * 0.2
        )
        safety_level = float(np.clip(safety_level, 0.0, 1.0))

        para_target = 0.1 + 0.9 * safety_level
        self.parasympathetic += (para_target - self.parasympathetic) * PARASYMPATHETIC_RISE_RATE
        self.parasympathetic = float(np.clip(self.parasympathetic, 0.0, 1.0))

        # === 3. Homeostasis drive (with chronic stress drift) ===
        # ★ Homeostasis target drifts with trauma history
        #   Normal: sympathetic baseline=0.2, parasympathetic baseline=0.3
        #   After repeated trauma: sympathetic baseline↑, parasympathetic baseline↓
        #   = "always tense" "hard to relax"
        if not is_sleeping:
            self.sympathetic += (self.sympathetic_baseline - self.sympathetic) * HOMEOSTASIS_STRENGTH
            self.parasympathetic += (self.parasympathetic_baseline - self.parasympathetic) * HOMEOSTASIS_STRENGTH

        # === 4. Drive visceral indicators ===
        # Autonomic ratio (>1 = sympathetic dominant, <1 = parasympathetic dominant)
        ratio = self.sympathetic / max(self.parasympathetic, 0.01)

        # ★ Baseline ratio = "resting" state after drift
        baseline_ratio = self.sympathetic_baseline / max(self.parasympathetic_baseline, 0.01)
        # Resting heart rate slightly elevated with chronic stress (PTSD patients: resting 70-90bpm)
        elevated_resting = RESTING_HEART_RATE + 20.0 * self.chronic_stress_load

        # Heart rate: computed relative to baseline ratio, not absolute values
        if ratio > baseline_ratio:
            # Above baseline → accelerate towards MAX
            excess = min((ratio - baseline_ratio) / max(baseline_ratio, 0.01), 1.0)
            target_hr = elevated_resting + (MAX_HEART_RATE - elevated_resting) * excess
        else:
            # Below baseline → decelerate towards MIN
            deficit = ratio / max(baseline_ratio, 0.01)
            target_hr = MIN_HEART_RATE + (elevated_resting - MIN_HEART_RATE) * deficit
        self.heart_rate += (target_hr - self.heart_rate) * 0.2
        self.heart_rate = float(np.clip(self.heart_rate, MIN_HEART_RATE, MAX_HEART_RATE))

        # Respiration
        # ★ Central apnea bypass — Ondine's Curse (CCHS)
        #   PHOX2B mutation: brainstem cannot generate breathing commands during sleep.
        #   Awake: cortical drive compensates (voluntary_breath or normal autonomic).
        #   Asleep + central_apnea: breath_rate decays toward 0 (no brainstem drive).
        apnea_active = self.central_apnea and is_sleeping
        effective_min_br = 0.0 if apnea_active else MIN_BREATH_RATE

        if apnea_active and voluntary_breath is None:
            # No brainstem drive, no cortical override → breath_rate → 0
            target_br = 0.0
        elif voluntary_breath is not None:
            # Voluntary/cortical breathing — works even in CCHS (when awake)
            target_br = float(np.clip(voluntary_breath, effective_min_br, MAX_BREATH_RATE))
            # Voluntary breathing also feeds back to parasympathetic (slow breathing → parasympathetic ↑)
            if voluntary_breath < RESTING_BREATH_RATE:
                self.parasympathetic += 0.02
                self.parasympathetic = min(1.0, self.parasympathetic)
        else:
            if ratio > 1.0:
                target_br = RESTING_BREATH_RATE + (MAX_BREATH_RATE - RESTING_BREATH_RATE) * min(ratio - 1.0, 1.0)
            else:
                target_br = MIN_BREATH_RATE + (RESTING_BREATH_RATE - MIN_BREATH_RATE) * ratio
        self.breath_rate += (target_br - self.breath_rate) * 0.15
        self.breath_rate = float(np.clip(self.breath_rate, effective_min_br, MAX_BREATH_RATE))

        # Pupil
        self.pupil_size = 0.2 + 0.8 * (self.sympathetic / max(self.sympathetic + self.parasympathetic, 0.01))
        self.pupil_size = float(np.clip(self.pupil_size, 0.0, 1.0))

        # Body temperature (stress→slight rise, relaxation→return to normal)
        temp_delta = (self.sympathetic - 0.2) * 0.3  # Sympathetic → temperature rise
        self.core_temp += (CORE_TEMPERATURE + temp_delta - self.core_temp) * 0.05
        self.core_temp = float(np.clip(self.core_temp, 34.0, 42.0))

        # === 5. Stress hormones and energy ===
        # Cortisol: accumulates when sympathetic stays high
        if self.sympathetic > 0.5:
            self.cortisol += (self.sympathetic - 0.5) * 0.05
        # Natural decay
        self.cortisol *= CORTISOL_DECAY
        self.cortisol = float(np.clip(self.cortisol, 0.0, 1.0))

        # Energy: sympathetic consumes, parasympathetic recovers
        self.energy -= self.sympathetic * ENERGY_STRESS_COST
        if self.parasympathetic > self.sympathetic:
            self.energy += ENERGY_RESTING_RECOVERY
        if is_sleeping:
            self.energy += ENERGY_RESTING_RECOVERY * 2  # Sleep accelerates recovery
        self.energy = float(np.clip(self.energy, 0.0, ENERGY_MAX))

        # Event statistics
        if self.sympathetic > 0.8:
            self.stress_events += 1
        if self.energy < 0.1:
            self.exhaustion_events += 1

        # === Record history ===
        self.sympathetic_history.append(self.sympathetic)
        self.parasympathetic_history.append(self.parasympathetic)
        self.heart_rate_history.append(self.heart_rate)
        self.breath_rate_history.append(self.breath_rate)
        self.energy_history.append(self.energy)
        self.cortisol_history.append(self.cortisol)

        for hist in (
            self.sympathetic_history, self.parasympathetic_history,
            self.heart_rate_history, self.breath_rate_history,
            self.energy_history, self.cortisol_history,
        ):
            if len(hist) > self._max_history:
                del hist[:-self._max_history]

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_autonomic_balance(self) -> float:
        """
        Autonomic nervous system balance value

        Returns: -1.0 (parasympathetic extreme) ~ 0.0 (balanced) ~ +1.0 (sympathetic extreme)
        """
        total = self.sympathetic + self.parasympathetic
        if total < 0.01:
            return 0.0
        return (self.sympathetic - self.parasympathetic) / total

    def get_stress_level(self) -> float:
        """Stress level (0~1): combined sympathetic activity + cortisol"""
        return float(np.clip(
            self.sympathetic * 0.6 + self.cortisol * 0.4, 0.0, 1.0
        ))

    def is_exhausted(self) -> bool:
        """Whether exhausted"""
        return self.energy < 0.1

    def get_pupil_aperture(self) -> float:
        """Pupil size (can connect to AliceEye.adjust_pupil)"""
        return self.pupil_size

    def get_signal(self) -> ElectricalSignal:
        """
        Integrated signal of the autonomic nervous system

        Represents the current internal body state,
        source="autonomic", modality="interoception"
        """
        n = 64
        t = np.linspace(0, 1, n, endpoint=False)

        # Heart rate component + respiration component + cortisol modulation
        hr_norm = (self.heart_rate - MIN_HEART_RATE) / (MAX_HEART_RATE - MIN_HEART_RATE)
        br_norm = max(0.0, (self.breath_rate - MIN_BREATH_RATE) / (MAX_BREATH_RATE - MIN_BREATH_RATE))

        waveform = (
            hr_norm * np.sin(2 * np.pi * 1.0 * t) +       # Heartbeat rhythm
            br_norm * 0.5 * np.sin(2 * np.pi * 0.25 * t) + # Respiratory rhythm
            self.cortisol * 0.3 * np.sin(2 * np.pi * 4.0 * t)  # Stress wave
        )

        # Frequency = higher stress → faster
        freq = 2.0 + 28.0 * self.get_stress_level()  # θ~β range
        amp = 0.3 + 0.7 * (self.sympathetic + self.parasympathetic) / 2.0

        return ElectricalSignal(
            waveform=waveform,
            amplitude=amp,
            frequency=freq,
            phase=0.0,
            impedance=AUTONOMIC_IMPEDANCE,
            snr=10.0,
            source="autonomic",
            modality="interoception",
        )

    # ------------------------------------------------------------------
    def get_vitals(self) -> Dict[str, Any]:
        return {
            "sympathetic": round(self.sympathetic, 4),
            "parasympathetic": round(self.parasympathetic, 4),
            "balance": round(self.get_autonomic_balance(), 4),
            "stress_level": round(self.get_stress_level(), 4),
            "heart_rate": round(self.heart_rate, 1),
            "breath_rate": round(self.breath_rate, 1),
            "pupil_size": round(self.pupil_size, 3),
            "core_temp": round(self.core_temp, 2),
            "cortisol": round(self.cortisol, 4),
            "energy": round(self.energy, 4),
            "is_exhausted": self.is_exhausted(),
            "total_ticks": self.total_ticks,
            "stress_events": self.stress_events,
            "exhaustion_events": self.exhaustion_events,
        }

    def get_waveforms(self, last_n: int = 60) -> Dict[str, List[float]]:
        return {
            "sympathetic": self.sympathetic_history[-last_n:],
            "parasympathetic": self.parasympathetic_history[-last_n:],
            "heart_rate": self.heart_rate_history[-last_n:],
            "breath_rate": self.breath_rate_history[-last_n:],
            "energy": self.energy_history[-last_n:],
            "cortisol": self.cortisol_history[-last_n:],
        }

    def get_stats(self) -> Dict[str, Any]:
        return self.get_vitals()

    def record_trauma(self):
        """
        Record a trauma event — autonomic baseline drift

        Physics:
          Repeatedly overloaded machine → metal fatigue → decreased elasticity
          = Repeatedly injured person → permanent nervous system change → elevated resting heart rate
        """
        self.trauma_events += 1

        # Chronic stress accumulation
        self.chronic_stress_load = min(1.0, self.chronic_stress_load + 0.1)

        # Sympathetic baseline shifts up (max 0.5) — "always tense"
        self.sympathetic_baseline = min(0.5, self.sympathetic_baseline + 0.03)

        # Parasympathetic baseline shifts down (min 0.15) — "hard to relax"
        self.parasympathetic_baseline = max(0.15, self.parasympathetic_baseline - 0.02)

    def reset(self):
        """Reset to quiet state (preserves chronic stress drift)"""
        # ★ Baseline keeps post-trauma values, does not return to 0.2/0.3
        self.sympathetic = self.sympathetic_baseline
        self.parasympathetic = self.parasympathetic_baseline
        self.heart_rate = RESTING_HEART_RATE + 20.0 * self.chronic_stress_load  # Resting heart rate also elevated
        self.breath_rate = RESTING_BREATH_RATE
        self.pupil_size = RESTING_PUPIL_SIZE
        self.core_temp = CORE_TEMPERATURE
        self.cortisol = 0.1 + 0.3 * self.chronic_stress_load  # Baseline cortisol elevated
        self.energy = ENERGY_MAX * (1.0 - 0.3 * self.chronic_stress_load)  # Energy ceiling lowered
        # Note: chronic_stress_load, sympathetic_baseline, parasympathetic_baseline
        #       are NOT reset — this is the cost of trauma

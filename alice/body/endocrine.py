# -*- coding: utf-8 -*-
"""
Alice Smart System — Endocrine System (EndocrineSystem)

Physics Model: Hormone Cascade as Impedance Modulation Network
==============================================================

The endocrine system is modeled as a distributed impedance modulation network where:
    Glands   = voltage-controlled signal sources
    Hormones = modulation envelopes on existing transmission lines
    Receptors = impedance-tuned bandpass filters
    Feedback = negative feedback loops (like AGC in radio receivers)

Core equations:
    hormone_level = production_rate × gland_health − clearance_rate × liver_function
    receptor_binding = hormone_level × T_receptor
    T_receptor = 1 − Γ²_receptor  (★ C1)
    Γ_receptor = (Z_hormone − Z_receptor) / (Z_hormone + Z_receptor)
    ΔZ_receptor = −η × Γ × hormone × target_response  (★ C2 Hebbian)

Endocrine axes modeled:
    HPA axis: Hypothalamus → Pituitary → Adrenal → Cortisol
    HPT axis: Hypothalamus → Pituitary → Thyroid → T3/T4
    HPG axis: Hypothalamus → Pituitary → Gonad → Sex hormones
    Growth axis: GH → IGF-1 → tissue growth
    Insulin-Glucose: Pancreas → Insulin → glucose uptake

Clinical significance:
    Stress  → HPA activation → cortisol ↑ → immune suppression, metabolic shift
    Hypothyroid → T3/T4 ↓ → metabolic rate ↓ → fatigue, cold sensitivity
    Diabetes → insulin/glucose mismatch → Γ_metabolic ↑
    Growth delay → GH ↓ → IGF-1 ↓ → developmental impact

"Hormones are not messengers — they are impedance modulators.
 They don't carry information; they change how channels respond to it."

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

# --- HPA Axis (Stress) ---
CORTISOL_BASE = 0.3             # Baseline cortisol (normalized 0–1)
CORTISOL_MAX = 1.0
CORTISOL_HALF_LIFE_TICKS = 90   # Ticks for half-life
CORTISOL_DECAY = 0.5 ** (1.0 / 90)  # Per-tick decay
CRH_GAIN = 0.15                 # Stress → CRH → cortisol production
ACTH_GAIN = 0.8                 # CRH → ACTH amplification
CORTISOL_NEGATIVE_FB = 0.1      # Cortisol → suppress CRH (negative feedback)

# --- HPT Axis (Thyroid/Metabolism) ---
T3T4_BASE = 0.5                 # Baseline thyroid hormone
T3T4_MIN = 0.1                  # Hypothyroid
T3T4_MAX = 1.0                  # Hyperthyroid
TSH_BASE = 0.5
TSH_GAIN = 0.1
THYROID_METABOLIC_GAIN = 0.4    # T3/T4 → metabolic rate scaling

# --- Growth Axis ---
GH_BASE = 0.4                   # Growth hormone baseline
GH_SLEEP_BOOST = 0.3            # Deep sleep → GH pulse
GH_MAX = 1.0
IGF1_GAIN = 0.6                 # GH → IGF-1 conversion
GROWTH_RATE_BASE = 0.0002       # Tissue growth per tick

# --- Insulin/Glucose ---
INSULIN_BASE = 0.3
INSULIN_MAX = 1.0
INSULIN_GLUCOSE_GAIN = 0.5      # High glucose → insulin secretion
INSULIN_CLEARANCE_RATE = 0.02   # Insulin clearance per tick

# --- Sex Hormones (simplified) ---
ESTROGEN_BASE = 0.3
TESTOSTERONE_BASE = 0.3
LH_FSH_BASE = 0.3

# --- Melatonin (circadian) ---
MELATONIN_DARK_PRODUCTION = 0.8  # Melatonin level in darkness
MELATONIN_LIGHT_SUPPRESSION = 0.9  # Light suppresses melatonin
MELATONIN_DECAY = 0.95

# --- Impedance ---
ENDOCRINE_Z = 90.0              # Ω — endocrine signaling impedance
Z_RECEPTOR_BASE = 90.0
ENDOCRINE_SNR = 5.0             # dB — hormonal signals are slow and noisy
ENDOCRINE_SAMPLE_POINTS = 48
ENDOCRINE_FREQUENCY = 0.1       # Hz — ultra-slow hormonal rhythm

# --- Development ---
NEONATAL_ENDOCRINE_MATURITY = 0.25
ENDOCRINE_MATURATION_RATE = 0.0003


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class HormoneState:
    """Snapshot of all hormone levels."""
    cortisol: float = CORTISOL_BASE
    t3t4: float = T3T4_BASE
    tsh: float = TSH_BASE
    growth_hormone: float = GH_BASE
    igf1: float = 0.0
    insulin: float = INSULIN_BASE
    melatonin: float = 0.3
    estrogen: float = ESTROGEN_BASE
    testosterone: float = TESTOSTERONE_BASE
    metabolic_rate: float = 1.0
    gamma_endocrine: float = 0.0
    transmission_endocrine: float = 1.0
    maturity: float = NEONATAL_ENDOCRINE_MATURITY


# ============================================================================
# EndocrineSystem
# ============================================================================

class EndocrineSystem:
    """
    Alice's Endocrine System — hormone-mediated impedance modulation.

    Models hormone production, receptor binding (impedance matching),
    negative feedback loops, and circadian rhythm via melatonin.

    All signals use ElectricalSignal (★ C3).
    """

    def __init__(self) -> None:
        self._maturity: float = NEONATAL_ENDOCRINE_MATURITY

        # HPA Axis
        self._cortisol: float = CORTISOL_BASE
        self._crh: float = 0.0     # Corticotropin-releasing hormone
        self._acth: float = 0.0    # Adrenocorticotropic hormone

        # HPT Axis
        self._tsh: float = TSH_BASE
        self._t3t4: float = T3T4_BASE

        # Growth
        self._gh: float = GH_BASE
        self._igf1: float = GH_BASE * IGF1_GAIN

        # Insulin/Glucose
        self._insulin: float = INSULIN_BASE

        # Sex hormones (simplified)
        self._estrogen: float = ESTROGEN_BASE
        self._testosterone: float = TESTOSTERONE_BASE
        self._lh_fsh: float = LH_FSH_BASE

        # Melatonin
        self._melatonin: float = 0.3

        # Derived
        self._metabolic_rate: float = 1.0
        self._gamma_endocrine: float = 0.0
        self._transmission_endocrine: float = 1.0

        # Statistics
        self._tick_count: int = 0
        self._peak_cortisol: float = CORTISOL_BASE

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        stress: float = 0.0,
        blood_glucose: float = 0.7,
        sleep_depth: float = 0.0,
        light_level: float = 0.5,
        core_temperature: float = 37.0,
        liver_function: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Advance endocrine system by one tick.

        Args:
            stress: Perceived stress level (0–1)
            blood_glucose: Current blood glucose (normalized)
            sleep_depth: 0=awake, 1=deep NREM
            light_level: Ambient light (0=dark, 1=bright)
            core_temperature: Body temperature (°C)
            liver_function: Liver clearance capacity (0–1)
        """
        self._tick_count += 1
        self._maturity = min(1.0, self._maturity + ENDOCRINE_MATURATION_RATE)
        lf = float(np.clip(liver_function, 0.1, 1.0))

        # === HPA Axis ===
        stress_in = float(np.clip(stress, 0, 1))
        # CRH production (stress drives, cortisol suppresses via negative feedback)
        self._crh = max(0.0, stress_in * CRH_GAIN - self._cortisol * CORTISOL_NEGATIVE_FB)
        self._acth = self._crh * ACTH_GAIN
        # Cortisol production and decay
        cortisol_production = self._acth * self._maturity
        self._cortisol = self._cortisol * CORTISOL_DECAY + cortisol_production
        self._cortisol *= (1.0 - (lf - 0.5) * 0.1)  # Liver clears cortisol
        self._cortisol = float(np.clip(self._cortisol, 0, CORTISOL_MAX))
        self._peak_cortisol = max(self._peak_cortisol, self._cortisol)

        # === HPT Axis ===
        # TSH: inversely related to T3/T4 (negative feedback)
        self._tsh = TSH_BASE + TSH_GAIN * (1.0 - self._t3t4 / T3T4_MAX)
        # T3/T4 production
        t3t4_production = self._tsh * self._maturity * 0.1
        t3t4_clearance = self._t3t4 * 0.02 * lf
        self._t3t4 = float(np.clip(self._t3t4 + t3t4_production - t3t4_clearance,
                                    T3T4_MIN, T3T4_MAX))
        # Metabolic rate
        self._metabolic_rate = 0.6 + self._t3t4 * THYROID_METABOLIC_GAIN

        # === Growth Axis ===
        # GH: pulsatile, boosted by deep sleep
        gh_production = GH_BASE * 0.05 + sleep_depth * GH_SLEEP_BOOST * 0.1
        self._gh = float(np.clip(self._gh * 0.97 + gh_production, 0, GH_MAX))
        self._igf1 = self._gh * IGF1_GAIN * self._maturity

        # === Insulin ===
        glucose_in = float(np.clip(blood_glucose, 0, 1))
        insulin_secretion = INSULIN_GLUCOSE_GAIN * max(0, glucose_in - 0.3) * self._maturity
        self._insulin = float(np.clip(
            self._insulin * (1 - INSULIN_CLEARANCE_RATE * lf) + insulin_secretion,
            0, INSULIN_MAX
        ))

        # === Melatonin ===
        light = float(np.clip(light_level, 0, 1))
        if light < 0.3:
            self._melatonin = min(1.0, self._melatonin + MELATONIN_DARK_PRODUCTION * 0.05)
        else:
            self._melatonin *= (1.0 - MELATONIN_LIGHT_SUPPRESSION * light * 0.1)
        self._melatonin = float(np.clip(self._melatonin * MELATONIN_DECAY, 0, 1))

        # === Aggregate Γ_endocrine ===
        # Γ represents how far each hormone is from its setpoint (homeostasis)
        deviations = [
            abs(self._cortisol - CORTISOL_BASE) / max(CORTISOL_MAX, 0.01),
            abs(self._t3t4 - T3T4_BASE) / max(T3T4_MAX, 0.01),
            abs(self._insulin - INSULIN_BASE) / max(INSULIN_MAX, 0.01),
            abs(self._gh - GH_BASE) / max(GH_MAX, 0.01),
        ]
        mean_deviation = float(np.mean(deviations))
        z_eff = ENDOCRINE_Z * (1.0 + mean_deviation)
        self._gamma_endocrine = abs(z_eff - Z_RECEPTOR_BASE) / (z_eff + Z_RECEPTOR_BASE)
        self._transmission_endocrine = 1.0 - self._gamma_endocrine ** 2  # ★ C1

        return {
            "cortisol": round(self._cortisol, 4),
            "t3t4": round(self._t3t4, 4),
            "tsh": round(self._tsh, 4),
            "growth_hormone": round(self._gh, 4),
            "igf1": round(self._igf1, 4),
            "insulin": round(self._insulin, 4),
            "melatonin": round(self._melatonin, 4),
            "metabolic_rate": round(self._metabolic_rate, 4),
            "gamma_endocrine": round(self._gamma_endocrine, 4),
            "transmission_endocrine": round(self._transmission_endocrine, 4),
            "maturity": round(self._maturity, 4),
        }

    # ------------------------------------------------------------------
    # ElectricalSignal (★ C3)
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate hormonal state as ElectricalSignal."""
        amplitude = float(np.clip(0.1 + self._cortisol * 0.5 + (1 - self._t3t4) * 0.3,
                                  0.05, 1.0))
        freq = ENDOCRINE_FREQUENCY + self._metabolic_rate * 0.5

        t = np.linspace(0, 1, ENDOCRINE_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)
        # Cortisol stress component
        waveform += 0.2 * self._cortisol * np.sin(2 * math.pi * 0.05 * t)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=ENDOCRINE_Z,
            snr=ENDOCRINE_SNR,
            source="endocrine",
            modality="interoceptive",
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get endocrine system statistics."""
        return {
            "cortisol": round(self._cortisol, 4),
            "t3t4": round(self._t3t4, 4),
            "tsh": round(self._tsh, 4),
            "growth_hormone": round(self._gh, 4),
            "igf1": round(self._igf1, 4),
            "insulin": round(self._insulin, 4),
            "melatonin": round(self._melatonin, 4),
            "estrogen": round(self._estrogen, 4),
            "testosterone": round(self._testosterone, 4),
            "metabolic_rate": round(self._metabolic_rate, 4),
            "gamma_endocrine": round(self._gamma_endocrine, 4),
            "transmission_endocrine": round(self._transmission_endocrine, 4),
            "maturity": round(self._maturity, 4),
            "peak_cortisol": round(self._peak_cortisol, 4),
            "tick_count": self._tick_count,
        }

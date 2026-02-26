# -*- coding: utf-8 -*-
"""
Alice Smart System — Kidney (KidneySystem)

Physics Model: Electrolyte Impedance Filter
============================================

The kidney is modeled as a multi-stage impedance filter where:
    Glomerulus = broadband input filter (blood → filtrate)
    Proximal tubule = selective impedance-matched reabsorption
    Loop of Henle = counter-current impedance gradient
    Distal tubule = fine-tuning filter (hormone-controlled)
    Collecting duct = final impedance adjustment

Core equations:
    GFR = renal_blood_flow × filtration_fraction × (1 − Γ²_glomerular)
    T_renal = 1 − Γ²_renal  (★ C1 energy conservation)
    Γ_renal = (Z_blood − Z_filtrate) / (Z_blood + Z_filtrate)
    electrolyte_balance = Σ(reabsorbed − filtered) per ion species
    ΔZ_tubule = −η × Γ × ADH × osmolarity  (★ C2 Hebbian adaptation)

Electrolyte tracking:
    Na⁺  — primary extracellular cation, BP regulation
    K⁺   — primary intracellular cation, cardiac rhythm
    Ca²⁺ — bone/nerve, PTH regulation
    H⁺/pH — acid-base balance

Clinical significance:
    Dehydration → ADH ↑ → water reabsorption ↑ → concentrated urine
    Renal failure → GFR ↓ → waste accumulation → uremia
    Hyperkalemia → K⁺ ↑ → cardiac arrhythmia risk
    Hyponatremia → Na⁺ ↓ → cerebral edema → confusion

"The kidney is not a filter — it is an impedance optimizer.
 It doesn't remove waste; it matches the blood's impedance to homeostasis."

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

# --- Glomerular filtration ---
GFR_BASE = 1.0                  # Normalized GFR (= ~120 mL/min in adult)
GFR_MIN = 0.1                   # Severe renal failure
GFR_NEONATAL = 0.3              # Neonatal GFR (immature)
FILTRATION_FRACTION = 0.2       # Fraction of plasma filtered

# --- Blood-filtrate impedance ---
Z_BLOOD = 70.0                  # Ω — blood impedance
Z_FILTRATE_BASE = 70.0          # Ω — optimal filtrate impedance (matched = efficient)
Z_FILTRATE_DRIFT = 15.0         # Ω — max drift from waste accumulation

# --- Electrolyte setpoints (normalized 0–1) ---
ELECTROLYTE_SETPOINTS = {
    "sodium": 0.5,              # Na⁺
    "potassium": 0.4,           # K⁺
    "calcium": 0.45,            # Ca²⁺
    "ph": 0.5,                  # pH (0=acidic, 1=alkaline, 0.5=7.4)
}

ELECTROLYTE_DANGER_LOW = {
    "sodium": 0.2,
    "potassium": 0.15,
    "calcium": 0.2,
    "ph": 0.2,
}

ELECTROLYTE_DANGER_HIGH = {
    "sodium": 0.85,
    "potassium": 0.8,
    "calcium": 0.8,
    "ph": 0.8,
}

# --- ADH (antidiuretic hormone) ---
ADH_DEHYDRATION_GAIN = 0.5     # Dehydration → ADH ↑
ADH_BASE = 0.3
ADH_MAX = 1.0

# --- RAAS (renin-angiotensin-aldosterone) ---
RAAS_BP_GAIN = 0.4              # Low BP → RAAS activation
RAAS_NA_RETENTION = 0.1         # RAAS → Na⁺ reabsorption

# --- Waste products ---
UREA_PRODUCTION_RATE = 0.005    # Metabolic waste per tick
CREATININE_BASE = 0.3           # Baseline creatinine
WASTE_CLEARANCE_RATE = 0.08     # Clearance per GFR unit

# --- Signal ---
KIDNEY_IMPEDANCE = 70.0         # Ω
KIDNEY_SNR = 6.0                # dB
KIDNEY_SAMPLE_POINTS = 48
KIDNEY_FREQUENCY = 0.2          # Hz — slow renal rhythm

# --- Development ---
NEONATAL_KIDNEY_MATURITY = 0.25
KIDNEY_MATURATION_RATE = 0.0004


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class KidneyState:
    """Snapshot of kidney state."""
    gfr: float = GFR_NEONATAL
    electrolytes: Dict[str, float] = field(default_factory=lambda: dict(ELECTROLYTE_SETPOINTS))
    urea: float = 0.0
    adh: float = ADH_BASE
    urine_concentration: float = 0.5
    gamma_renal: float = 0.0
    transmission_renal: float = 1.0
    maturity: float = NEONATAL_KIDNEY_MATURITY


# ============================================================================
# KidneySystem
# ============================================================================

class KidneySystem:
    """
    Alice's Kidney — electrolyte impedance filter.

    Models glomerular filtration, electrolyte balance, and
    waste clearance as impedance matching problems.

    All signals use ElectricalSignal (★ C3).
    """

    def __init__(self) -> None:
        self._maturity: float = NEONATAL_KIDNEY_MATURITY
        self._gfr: float = GFR_NEONATAL

        # Electrolytes
        self._electrolytes: Dict[str, float] = dict(ELECTROLYTE_SETPOINTS)

        # Hormones
        self._adh: float = ADH_BASE
        self._raas: float = 0.0

        # Waste
        self._urea: float = 0.0
        self._creatinine: float = CREATININE_BASE

        # Output
        self._urine_concentration: float = 0.5
        self._water_recovered: float = 0.0

        # Γ
        self._gamma_renal: float = 0.0
        self._transmission_renal: float = 1.0

        # Statistics
        self._tick_count: int = 0
        self._danger_events: int = 0

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        hydration: float = 0.7,
        blood_pressure_norm: float = 0.6,
        blood_glucose: float = 0.7,
        protein_intake: float = 0.3,
        cortisol: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Advance kidney by one tick.

        Args:
            hydration: Body hydration level (0–1)
            blood_pressure_norm: Normalized mean arterial pressure
            blood_glucose: Blood glucose (affects osmotic load)
            protein_intake: Protein metabolized (→ urea production)
            cortisol: Cortisol level (affects Na⁺ handling)
        """
        self._tick_count += 1
        self._maturity = min(1.0, self._maturity + KIDNEY_MATURATION_RATE)

        hyd = float(np.clip(hydration, 0, 1))
        bp = float(np.clip(blood_pressure_norm, 0, 1))

        # GFR: depends on BP and maturity
        self._gfr = GFR_BASE * bp * self._maturity
        self._gfr = float(np.clip(self._gfr, GFR_MIN, GFR_BASE * 1.2))

        # ADH: increases with dehydration
        self._adh = ADH_BASE + ADH_DEHYDRATION_GAIN * (1.0 - hyd)
        self._adh = float(np.clip(self._adh, 0, ADH_MAX))

        # RAAS: activates with low BP
        self._raas = max(0.0, RAAS_BP_GAIN * (0.6 - bp))

        # --- Electrolyte processing ---
        danger = False
        for ion in self._electrolytes:
            setpoint = ELECTROLYTE_SETPOINTS[ion]
            current = self._electrolytes[ion]

            # Filtration: GFR removes proportional electrolytes
            filtered = current * FILTRATION_FRACTION * self._gfr * 0.01

            # Reabsorption: ADH and RAAS modulated
            if ion == "sodium":
                reabsorb = filtered * (0.95 + self._raas * RAAS_NA_RETENTION)
            elif ion == "potassium":
                # Aldosterone (RAAS) → K⁺ excretion
                reabsorb = filtered * (0.9 - self._raas * 0.1)
            elif ion == "ph":
                # Kidney regulates pH by H⁺ excretion
                reabsorb = filtered * 0.95
            else:
                reabsorb = filtered * 0.95

            net = reabsorb - filtered
            # ★ C2 Hebbian: ΔZ = −η × Γ × error
            error = current - setpoint
            correction = -0.01 * error * self._gfr * self._maturity
            self._electrolytes[ion] = float(np.clip(current + net + correction, 0, 1))

            # Danger check
            if (self._electrolytes[ion] < ELECTROLYTE_DANGER_LOW[ion] or
                    self._electrolytes[ion] > ELECTROLYTE_DANGER_HIGH[ion]):
                danger = True

        if danger:
            self._danger_events += 1

        # --- Waste clearance ---
        self._urea += UREA_PRODUCTION_RATE * protein_intake * 2.0
        urea_cleared = self._urea * WASTE_CLEARANCE_RATE * self._gfr
        self._urea = max(0.0, self._urea - urea_cleared)

        # --- Urine concentration ---
        self._urine_concentration = float(np.clip(
            0.5 + self._adh * 0.3 - hyd * 0.2, 0.1, 1.0
        ))

        # --- Water recovery ---
        self._water_recovered = self._adh * self._gfr * 0.02 * self._maturity

        # --- Γ_renal ---
        z_filtrate = Z_FILTRATE_BASE + self._urea * Z_FILTRATE_DRIFT
        self._gamma_renal = abs(Z_BLOOD - z_filtrate) / (Z_BLOOD + z_filtrate)
        self._transmission_renal = 1.0 - self._gamma_renal ** 2  # ★ C1

        return {
            "gamma_renal": round(self._gamma_renal, 4),
            "transmission_renal": round(self._transmission_renal, 4),
            "gfr": round(self._gfr, 4),
            "electrolytes": {k: round(v, 4) for k, v in self._electrolytes.items()},
            "urea": round(self._urea, 4),
            "adh": round(self._adh, 4),
            "urine_concentration": round(self._urine_concentration, 4),
            "water_recovered": round(self._water_recovered, 4),
            "danger": danger,
            "maturity": round(self._maturity, 4),
        }

    # ------------------------------------------------------------------
    # ElectricalSignal (★ C3)
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate renal status signal."""
        amplitude = float(np.clip(0.1 + self._urea * 2.0 + self._gamma_renal, 0.05, 1.0))
        freq = KIDNEY_FREQUENCY + self._gamma_renal * 2.0

        t = np.linspace(0, 1, KIDNEY_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=KIDNEY_IMPEDANCE,
            snr=KIDNEY_SNR,
            source="kidney",
            modality="interoceptive",
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get kidney statistics."""
        return {
            "gamma_renal": round(self._gamma_renal, 4),
            "transmission_renal": round(self._transmission_renal, 4),
            "gfr": round(self._gfr, 4),
            "electrolytes": {k: round(v, 4) for k, v in self._electrolytes.items()},
            "urea": round(self._urea, 4),
            "creatinine": round(self._creatinine, 4),
            "adh": round(self._adh, 4),
            "raas": round(self._raas, 4),
            "urine_concentration": round(self._urine_concentration, 4),
            "water_recovered": round(self._water_recovered, 4),
            "maturity": round(self._maturity, 4),
            "danger_events": self._danger_events,
            "tick_count": self._tick_count,
        }

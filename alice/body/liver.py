# -*- coding: utf-8 -*-
"""
Alice Smart System — Liver (LiverSystem)

Physics Model: Metabolic Impedance Transformer
===============================================

The liver is modeled as a complex impedance transformer where:
    Portal vein   = input (nutrient-rich blood from gut)
    Hepatic artery = auxiliary power supply (O₂-rich)
    Hepatocytes   = impedance-matching transformer banks
    Bile duct     = waste output channel
    Hepatic vein  = transformed output (clean blood)

Core equations:
    metabolic_output = substrate × enzyme_activity × T_hepatic
    T_hepatic = 1 − Γ²_hepatic  (★ C1 energy conservation)
    Γ_hepatic = (Z_portal − Z_hepatocyte) / (Z_portal + Z_hepatocyte)
    detox_rate = toxin_load × CYP450_activity × liver_health
    ΔZ_enzyme = −η × Γ × substrate × product  (★ C2 Hebbian enzyme induction)

Liver functions:
    1. Detoxification — impedance-transform toxins to excretable metabolites
    2. Glycogen storage — energy buffer (charge/discharge cycle)
    3. Protein synthesis — albumin, clotting factors
    4. Bile production — fat digestion support
    5. Drug metabolism — CYP450 enzyme system

Clinical significance:
    Hepatitis → hepatocyte Z damage → Γ_hepatic ↑ → metabolic impairment
    Cirrhosis → fibrosis → Z_hepatocyte permanently altered → liver failure
    Drug interaction → CYP450 competition → unexpected Γ for medications
    Neonatal jaundice → immature bilirubin conjugation → icterus

"The liver does not detoxify — it impedance-transforms.
 A poison is just a molecule whose Z the liver hasn't learned to match."

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

# --- Hepatic impedance ---
Z_PORTAL = 65.0                 # Ω — portal vein blood impedance
Z_HEPATOCYTE = 65.0             # Ω — healthy hepatocyte impedance
Z_HEPATOCYTE_DAMAGED = 120.0    # Ω — damaged hepatocyte

# --- Glycogen storage ---
GLYCOGEN_MAX = 1.0              # Normalized glycogen store
GLYCOGEN_STORE_RATE = 0.05      # Glucose → glycogen per tick
GLYCOGEN_RELEASE_RATE = 0.03    # Glycogen → glucose per tick (when needed)
GLYCOGEN_NEONATAL = 0.2         # Neonates have low glycogen

# --- Detoxification ---
CYP450_BASE_ACTIVITY = 0.5     # Baseline CYP450 enzyme activity
CYP450_MAX = 1.0
CYP450_INDUCTION_RATE = 0.005  # Enzyme induction (Hebbian) per exposure
CYP450_DECAY = 0.999           # Slow enzyme decay
DETOX_RATE_BASE = 0.1          # Toxin clearance per tick

# --- Bilirubin ---
BILIRUBIN_PRODUCTION_RATE = 0.003  # From RBC breakdown
BILIRUBIN_CONJUGATION_RATE = 0.008 # Liver conjugation (maturity-dependent)
BILIRUBIN_JAUNDICE_THRESHOLD = 0.5 # Above this → visible jaundice
BILIRUBIN_DANGER = 0.8            # Above this → kernicterus risk (neonatal)

# --- Protein synthesis ---
ALBUMIN_BASE = 0.7              # Normalized albumin level
ALBUMIN_PRODUCTION = 0.01       # Per tick
ALBUMIN_DECAY = 0.998
CLOTTING_FACTOR_BASE = 0.8

# --- Bile ---
BILE_PRODUCTION_BASE = 0.5      # Normalized bile output
BILE_FAT_DIGESTION_GAIN = 0.3   # Bile → fat absorption efficiency

# --- Liver health ---
LIVER_HEALTH_MAX = 1.0
LIVER_DAMAGE_RATE = 0.001       # Per toxin unit per tick
LIVER_REGENERATION_RATE = 0.0005  # Natural regeneration per tick

# --- Signal ---
LIVER_IMPEDANCE = 65.0          # Ω
LIVER_SNR = 6.0                 # dB
LIVER_SAMPLE_POINTS = 48
LIVER_FREQUENCY = 0.15          # Hz — slow metabolic rhythm

# --- Development ---
NEONATAL_LIVER_MATURITY = 0.2
LIVER_MATURATION_RATE = 0.0005


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LiverState:
    """Snapshot of liver state."""
    glycogen: float = GLYCOGEN_NEONATAL
    bilirubin: float = 0.0
    albumin: float = ALBUMIN_BASE
    cyp450_activity: float = CYP450_BASE_ACTIVITY
    liver_health: float = LIVER_HEALTH_MAX
    toxin_load: float = 0.0
    gamma_hepatic: float = 0.0
    transmission_hepatic: float = 1.0
    maturity: float = NEONATAL_LIVER_MATURITY


# ============================================================================
# LiverSystem
# ============================================================================

class LiverSystem:
    """
    Alice's Liver — metabolic impedance transformer.

    Models detoxification, glycogen storage, bilirubin conjugation,
    and protein synthesis as impedance-matching transformations.

    All signals use ElectricalSignal (★ C3).
    """

    def __init__(self) -> None:
        self._maturity: float = NEONATAL_LIVER_MATURITY
        self._liver_health: float = LIVER_HEALTH_MAX

        # Glycogen
        self._glycogen: float = GLYCOGEN_NEONATAL

        # Detox
        self._cyp450: float = CYP450_BASE_ACTIVITY
        self._toxin_load: float = 0.0
        self._enzyme_memory: Dict[str, float] = {}  # Learned detox profiles

        # Bilirubin
        self._bilirubin: float = 0.0

        # Protein synthesis
        self._albumin: float = ALBUMIN_BASE
        self._clotting_factors: float = CLOTTING_FACTOR_BASE

        # Bile
        self._bile_output: float = BILE_PRODUCTION_BASE

        # Γ
        self._gamma_hepatic: float = 0.0
        self._transmission_hepatic: float = 1.0

        # Statistics
        self._tick_count: int = 0
        self._total_detoxified: float = 0.0
        self._jaundice_ticks: int = 0

    # ------------------------------------------------------------------
    # Toxin exposure
    # ------------------------------------------------------------------

    def expose_toxin(self, name: str = "generic", amount: float = 0.1) -> Dict[str, Any]:
        """
        Expose liver to a toxin (drug, alcohol, metabolic waste).

        Args:
            name: Toxin identifier
            amount: Toxin amount (normalized)
        """
        amt = float(np.clip(amount, 0, 1))
        self._toxin_load = min(1.0, self._toxin_load + amt)

        # Check if liver has learned this toxin (enzyme memory)
        familiarity = self._enzyme_memory.get(name, 0.0)

        return {
            "toxin_load": round(self._toxin_load, 4),
            "familiarity": round(familiarity, 4),
            "liver_health": round(self._liver_health, 4),
        }

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        blood_glucose: float = 0.7,
        blood_flow: float = 1.0,
        spo2: float = 0.98,
    ) -> Dict[str, Any]:
        """
        Advance liver by one tick.

        Args:
            blood_glucose: Blood glucose level (affects glycogen storage)
            blood_flow: Hepatic blood flow (normalized)
            spo2: Oxygen saturation (liver needs O₂)
        """
        self._tick_count += 1
        self._maturity = min(1.0, self._maturity + LIVER_MATURATION_RATE)

        glucose = float(np.clip(blood_glucose, 0, 1))
        flow = float(np.clip(blood_flow, 0.1, 1.5))
        o2 = float(np.clip(spo2, 0.5, 1.0))

        effective = self._liver_health * self._maturity * flow * o2

        # === Glycogen management ===
        if glucose > 0.6:
            # Store excess as glycogen
            store = GLYCOGEN_STORE_RATE * (glucose - 0.6) * effective
            self._glycogen = min(GLYCOGEN_MAX, self._glycogen + store)
        elif glucose < 0.4:
            # Release glycogen
            release = GLYCOGEN_RELEASE_RATE * (0.4 - glucose) * effective
            release = min(release, self._glycogen)
            self._glycogen = max(0.0, self._glycogen - release)

        # === Detoxification ===
        if self._toxin_load > 0:
            detox = DETOX_RATE_BASE * self._cyp450 * effective
            detoxified = min(self._toxin_load, detox)
            self._toxin_load -= detoxified
            self._total_detoxified += detoxified

            # ★ C2 Hebbian enzyme induction: repeated exposure → better clearance
            self._cyp450 = min(CYP450_MAX,
                               self._cyp450 + CYP450_INDUCTION_RATE * self._toxin_load)

            # Liver damage from toxins
            damage = self._toxin_load * LIVER_DAMAGE_RATE
            self._liver_health = max(0.1, self._liver_health - damage)
        else:
            self._cyp450 *= CYP450_DECAY

        # Regeneration
        self._liver_health = min(LIVER_HEALTH_MAX,
                                 self._liver_health + LIVER_REGENERATION_RATE * o2)

        # === Bilirubin ===
        self._bilirubin += BILIRUBIN_PRODUCTION_RATE
        conjugation = BILIRUBIN_CONJUGATION_RATE * effective
        self._bilirubin = max(0.0, self._bilirubin - conjugation)

        if self._bilirubin > BILIRUBIN_JAUNDICE_THRESHOLD:
            self._jaundice_ticks += 1

        # === Protein synthesis ===
        self._albumin = float(np.clip(
            self._albumin * ALBUMIN_DECAY + ALBUMIN_PRODUCTION * effective,
            0, 1
        ))
        self._clotting_factors = float(np.clip(
            CLOTTING_FACTOR_BASE * effective, 0.1, 1.0
        ))

        # === Bile ===
        self._bile_output = BILE_PRODUCTION_BASE * effective

        # === Γ_hepatic ===
        z_portal_eff = Z_PORTAL + self._toxin_load * 20.0  # Toxins alter portal impedance
        z_hep_eff = Z_HEPATOCYTE + (1.0 - self._liver_health) * (
            Z_HEPATOCYTE_DAMAGED - Z_HEPATOCYTE
        )
        self._gamma_hepatic = abs(z_portal_eff - z_hep_eff) / (z_portal_eff + z_hep_eff)
        self._transmission_hepatic = 1.0 - self._gamma_hepatic ** 2  # ★ C1

        return {
            "gamma_hepatic": round(self._gamma_hepatic, 4),
            "transmission_hepatic": round(self._transmission_hepatic, 4),
            "glycogen": round(self._glycogen, 4),
            "bilirubin": round(self._bilirubin, 4),
            "jaundice": self._bilirubin > BILIRUBIN_JAUNDICE_THRESHOLD,
            "albumin": round(self._albumin, 4),
            "toxin_load": round(self._toxin_load, 4),
            "cyp450_activity": round(self._cyp450, 4),
            "liver_health": round(self._liver_health, 4),
            "bile_output": round(self._bile_output, 4),
            "maturity": round(self._maturity, 4),
        }

    # ------------------------------------------------------------------
    # ElectricalSignal (★ C3)
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate hepatic status signal."""
        amplitude = float(np.clip(
            0.1 + self._toxin_load * 0.5 + self._bilirubin * 0.3, 0.05, 1.0
        ))
        freq = LIVER_FREQUENCY + self._gamma_hepatic * 1.5

        t = np.linspace(0, 1, LIVER_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=LIVER_IMPEDANCE,
            snr=LIVER_SNR,
            source="liver",
            modality="interoceptive",
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_liver_function(self) -> float:
        """Return a normalized liver function score (0–1) for other modules."""
        return float(np.clip(self._liver_health * self._maturity, 0, 1))

    def get_stats(self) -> Dict[str, Any]:
        """Get liver statistics."""
        return {
            "gamma_hepatic": round(self._gamma_hepatic, 4),
            "transmission_hepatic": round(self._transmission_hepatic, 4),
            "glycogen": round(self._glycogen, 4),
            "bilirubin": round(self._bilirubin, 4),
            "jaundice": self._bilirubin > BILIRUBIN_JAUNDICE_THRESHOLD,
            "albumin": round(self._albumin, 4),
            "clotting_factors": round(self._clotting_factors, 4),
            "toxin_load": round(self._toxin_load, 4),
            "cyp450_activity": round(self._cyp450, 4),
            "liver_health": round(self._liver_health, 4),
            "bile_output": round(self._bile_output, 4),
            "maturity": round(self._maturity, 4),
            "total_detoxified": round(self._total_detoxified, 4),
            "jaundice_ticks": self._jaundice_ticks,
            "tick_count": self._tick_count,
        }

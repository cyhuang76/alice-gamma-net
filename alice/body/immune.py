# -*- coding: utf-8 -*-
"""
Alice Smart System — Immune System (ImmuneSystem)

Physics Model: Impedance-Based Immune Defense Network
=====================================================

The immune system is modeled as a distributed impedance detection network where:
    Antigens     = foreign impedance (Z_pathogen ≠ Z_self)
    Antibodies   = impedance-matching filters tuned to Z_pathogen
    Inflammation = local Γ² increase (reflected energy → heat)
    Cytokines    = inter-module signaling (ElectricalSignal dispatch)
    Fever        = global impedance re-calibration (raise Z_self to expel mismatch)

Core equations:
    Γ_immune = (Z_pathogen − Z_self) / (Z_pathogen + Z_self)
    T_immune = 1 − Γ_immune²   (★ C1 energy conservation)
    inflammation = ΣΓ²_local / N_regions
    fever_response = Δ_temp ∝ inflammation × fever_gain
    antibody_production = η × Γ × exposure_count  (★ C2 Hebbian adaptive immunity)

Immune cascade:
    1. Innate immunity: fixed Z_self boundary detection (fast, coarse)
    2. Adaptive immunity: Hebbian-tuned antibody Z-matching (slow, precise)
    3. Memory cells: previously matched Z patterns stored permanently
    4. Cytokine storm: runaway Γ² feedback → pathological inflammation

Clinical significance:
    Infection    → Z_pathogen invades → local Γ² ↑ → inflammation
    Allergy      → Z_self misidentified as foreign → autoimmune Γ
    Immunodeficiency → antibody matching fails → T_immune ↓
    Fever        → global Z_self shift to create hostile Z environment

"The immune system does not fight — it matches impedance.
 A virus is not defeated; it is reflected."

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

# --- Self impedance ---
Z_SELF = 75.0                   # Ω — baseline "self" impedance
Z_SELF_TOLERANCE = 5.0          # Ω — tolerance band for self-recognition
Z_SELF_DRIFT_MAX = 0.05         # Maximum Z_self drift per tick

# --- Pathogen impedance range ---
Z_PATHOGEN_MIN = 20.0           # Ω — typical viral impedance (light, fast)
Z_PATHOGEN_MAX = 500.0          # Ω — bacterial/fungal (heavy, slow)

# --- Innate immunity ---
INNATE_RESPONSE_RATE = 0.3      # How fast innate immunity activates
INNATE_SPECIFICITY = 0.4        # Coarse detection (low specificity)

# --- Adaptive immunity ---
ADAPTIVE_LEARNING_RATE = 0.01   # η for antibody Hebbian tuning
ADAPTIVE_SPECIFICITY_MAX = 0.95 # Maximum adaptive specificity
ANTIBODY_DECAY = 0.999          # Slow decay of antibody levels
MEMORY_CELL_THRESHOLD = 0.5     # Γ threshold for memory cell formation

# --- Inflammation ---
INFLAMMATION_GAIN = 2.0         # Γ² → inflammation scaling
INFLAMMATION_DECAY = 0.92       # Inflammation natural resolution per tick
INFLAMMATION_CRITICAL = 0.8     # Cytokine storm threshold
INFLAMMATION_WARNING = 0.4      # Clinical warning threshold

# --- Fever ---
FEVER_GAIN = 3.0                # inflammation → temperature increase (°C)
FEVER_NORMAL_TEMP = 37.0        # °C
FEVER_MAX = 42.0                # °C — lethal threshold
FEVER_RECOVERY_RATE = 0.05      # °C per tick cooling

# --- White blood cell counts (normalized) ---
WBC_BASELINE = 1.0              # Normalized WBC count
WBC_MAX = 5.0                   # Peak during infection
WBC_PRODUCTION_RATE = 0.1       # WBC production per tick during infection
WBC_DECAY = 0.98                # Natural WBC decay

# --- Immune signal ---
IMMUNE_IMPEDANCE = 85.0         # Ω — immune signaling impedance
IMMUNE_SNR = 6.0                # dB — cytokine signals are noisy
IMMUNE_SAMPLE_POINTS = 48
IMMUNE_FREQUENCY = 0.5          # Hz — slow immune oscillation

# --- Development ---
NEONATAL_IMMUNE_MATURITY = 0.2  # Newborn: immature immune system
IMMUNE_MATURATION_RATE = 0.0005 # Maturity increase per tick


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Pathogen:
    """A pathogen with characteristic impedance."""
    name: str
    z_pathogen: float            # Characteristic impedance
    virulence: float = 0.5       # 0–1 severity
    load: float = 0.0            # Current pathogen load (0–1)


@dataclass
class Antibody:
    """An antibody tuned to match a specific pathogen impedance."""
    target_z: float              # Target pathogen impedance
    specificity: float = 0.1     # How well it matches (0–1)
    level: float = 0.0           # Antibody concentration (0–1)
    is_memory: bool = False      # Memory cell (permanent)


@dataclass
class ImmuneState:
    """Snapshot of immune system state."""
    inflammation: float = 0.0
    core_temperature: float = FEVER_NORMAL_TEMP
    wbc_count: float = WBC_BASELINE
    gamma_immune: float = 0.0
    transmission_immune: float = 1.0
    maturity: float = NEONATAL_IMMUNE_MATURITY
    active_infections: int = 0
    antibody_count: int = 0


# ============================================================================
# ImmuneSystem
# ============================================================================

class ImmuneSystem:
    """
    Alice's Immune System — impedance-based defense network.

    Models innate and adaptive immunity as impedance matching problems.
    Pathogens are Z-mismatches; antibodies are trained Z-filters.
    Inflammation is accumulated Γ² energy. Fever shifts Z_self.

    All inter-module signals use ElectricalSignal (★ C3).
    """

    def __init__(self) -> None:
        self._z_self: float = Z_SELF
        self._maturity: float = NEONATAL_IMMUNE_MATURITY

        # Active pathogens
        self._pathogens: Dict[str, Pathogen] = {}

        # Antibody library
        self._antibodies: Dict[str, Antibody] = {}

        # Immune state
        self._inflammation: float = 0.0
        self._core_temperature: float = FEVER_NORMAL_TEMP
        self._wbc_count: float = WBC_BASELINE
        self._gamma_immune: float = 0.0
        self._transmission_immune: float = 1.0

        # Statistics
        self._tick_count: int = 0
        self._total_infections: int = 0
        self._total_cleared: int = 0

    # ------------------------------------------------------------------
    # Infection interface
    # ------------------------------------------------------------------

    def infect(self, name: str, z_pathogen: float, virulence: float = 0.5,
               initial_load: float = 0.3) -> Dict[str, Any]:
        """
        Introduce a pathogen into the system.

        Args:
            name: Pathogen identifier
            z_pathogen: Characteristic impedance of the pathogen
            virulence: Severity factor (0–1)
            initial_load: Starting pathogen load
        """
        z_p = float(np.clip(z_pathogen, Z_PATHOGEN_MIN, Z_PATHOGEN_MAX))
        self._pathogens[name] = Pathogen(
            name=name,
            z_pathogen=z_p,
            virulence=float(np.clip(virulence, 0, 1)),
            load=float(np.clip(initial_load, 0, 1)),
        )
        self._total_infections += 1

        # Check for existing antibodies (memory)
        gamma = abs(z_p - self._z_self) / (z_p + self._z_self)
        return {
            "pathogen": name,
            "gamma_initial": round(gamma, 4),
            "has_memory": name in self._antibodies and self._antibodies[name].is_memory,
        }

    # ------------------------------------------------------------------
    # Immune tick
    # ------------------------------------------------------------------

    def tick(self, cortisol: float = 0.0, sleep_quality: float = 0.5,
             hydration: float = 0.7) -> Dict[str, Any]:
        """
        Advance immune system by one tick.

        Args:
            cortisol: Stress hormone level (high cortisol suppresses immunity)
            sleep_quality: Sleep quality (good sleep boosts immunity)
            hydration: Hydration level (dehydration impairs immune response)
        """
        self._tick_count += 1

        # Developmental maturation
        self._maturity = min(1.0, self._maturity + IMMUNE_MATURATION_RATE)

        # Modifiers
        stress_suppression = 1.0 - 0.3 * float(np.clip(cortisol, 0, 1))
        sleep_boost = 0.8 + 0.4 * float(np.clip(sleep_quality, 0, 1))
        hydration_factor = 0.5 + 0.5 * float(np.clip(hydration, 0, 1))
        immune_efficacy = self._maturity * stress_suppression * sleep_boost * hydration_factor

        # Process each pathogen
        total_gamma_sq = 0.0
        cleared = []

        for name, pathogen in self._pathogens.items():
            # Γ = impedance mismatch with self
            gamma_p = abs(pathogen.z_pathogen - self._z_self) / (
                pathogen.z_pathogen + self._z_self
            )
            total_gamma_sq += gamma_p ** 2

            # --- Innate response ---
            innate_kill = INNATE_RESPONSE_RATE * INNATE_SPECIFICITY * immune_efficacy

            # --- Adaptive response ---
            ab = self._antibodies.get(name)
            if ab is None:
                # Create new antibody (first encounter)
                ab = Antibody(target_z=pathogen.z_pathogen, specificity=0.1, level=0.1)
                self._antibodies[name] = ab

            # ★ C2 Hebbian update: ΔZ_ab = −η × Γ × exposure × response
            delta_spec = ADAPTIVE_LEARNING_RATE * gamma_p * pathogen.load * immune_efficacy
            ab.specificity = min(ADAPTIVE_SPECIFICITY_MAX, ab.specificity + delta_spec)
            ab.level = min(1.0, ab.level + WBC_PRODUCTION_RATE * immune_efficacy)

            adaptive_kill = ab.specificity * ab.level * immune_efficacy

            # Total immune response
            total_kill = innate_kill + adaptive_kill
            pathogen.load = max(0.0, pathogen.load * (1.0 + pathogen.virulence * 0.05) - total_kill)

            # Memory cell formation
            if ab.specificity > MEMORY_CELL_THRESHOLD:
                ab.is_memory = True

            # Cleared?
            if pathogen.load < 0.01:
                cleared.append(name)

        # Remove cleared pathogens
        for name in cleared:
            del self._pathogens[name]
            self._total_cleared += 1

        # --- Inflammation ---
        n_active = max(1, len(self._pathogens))
        new_inflammation = (total_gamma_sq / n_active) * INFLAMMATION_GAIN if self._pathogens else 0.0
        self._inflammation = self._inflammation * INFLAMMATION_DECAY + new_inflammation * 0.1
        self._inflammation = float(np.clip(self._inflammation, 0, 1))

        # --- Fever ---
        target_temp = FEVER_NORMAL_TEMP + self._inflammation * FEVER_GAIN
        target_temp = min(target_temp, FEVER_MAX)
        self._core_temperature += (target_temp - self._core_temperature) * 0.1
        self._core_temperature = max(FEVER_NORMAL_TEMP - 1.0,
                                     min(FEVER_MAX, self._core_temperature))

        # --- WBC dynamics ---
        if self._pathogens:
            self._wbc_count = min(WBC_MAX,
                                  self._wbc_count + WBC_PRODUCTION_RATE * immune_efficacy)
        else:
            self._wbc_count = max(WBC_BASELINE, self._wbc_count * WBC_DECAY)

        # --- Antibody decay ---
        for ab in self._antibodies.values():
            if not ab.is_memory:
                ab.level *= ANTIBODY_DECAY

        # --- Aggregate Γ_immune ---
        if self._pathogens:
            gammas = [
                abs(p.z_pathogen - self._z_self) / (p.z_pathogen + self._z_self)
                for p in self._pathogens.values()
            ]
            self._gamma_immune = float(np.mean(gammas))
        else:
            self._gamma_immune = max(0.0, self._gamma_immune * 0.9)

        self._transmission_immune = 1.0 - self._gamma_immune ** 2  # ★ C1: T = 1 − Γ²

        return {
            "gamma_immune": round(self._gamma_immune, 4),
            "transmission_immune": round(self._transmission_immune, 4),
            "inflammation": round(self._inflammation, 4),
            "core_temperature": round(self._core_temperature, 2),
            "wbc_count": round(self._wbc_count, 2),
            "active_infections": len(self._pathogens),
            "cleared": cleared,
            "maturity": round(self._maturity, 4),
        }

    # ------------------------------------------------------------------
    # ElectricalSignal generation (★ C3)
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate immune status as ElectricalSignal (cytokine dispatch)."""
        amplitude = float(np.clip(self._inflammation * 2.0, 0.05, 1.0))
        freq = IMMUNE_FREQUENCY + self._inflammation * 3.0

        t = np.linspace(0, 1, IMMUNE_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)

        # Add WBC oscillation component
        waveform += 0.15 * (self._wbc_count / WBC_MAX) * np.sin(
            2 * math.pi * 0.3 * t
        )

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=IMMUNE_IMPEDANCE,
            snr=IMMUNE_SNR,
            source="immune",
            modality="interoceptive",
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get immune system statistics."""
        return {
            "gamma_immune": round(self._gamma_immune, 4),
            "transmission_immune": round(self._transmission_immune, 4),
            "inflammation": round(self._inflammation, 4),
            "core_temperature": round(self._core_temperature, 2),
            "wbc_count": round(self._wbc_count, 2),
            "maturity": round(self._maturity, 4),
            "active_infections": len(self._pathogens),
            "antibody_library_size": len(self._antibodies),
            "memory_cells": sum(1 for ab in self._antibodies.values() if ab.is_memory),
            "total_infections": self._total_infections,
            "total_cleared": self._total_cleared,
            "tick_count": self._tick_count,
        }

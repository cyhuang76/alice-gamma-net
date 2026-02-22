# -*- coding: utf-8 -*-
"""
Alice Smart System — Lung Organ (AliceLung)

Physics Model: LC Resonator — Breathing as Oscillation
=====================================================

The lung is modeled as an LC resonator where:
    L = inertia (lung tissue elasticity, chest wall compliance)
    C = capacity (lung volume, determined by development)
    Resonant frequency = breath_rate = 1 / (2π√LC)

Core equation — Effective lung capacity:
    C_eff = C_base × (1 + motor_growth) × (1 - stomach_fill × k_stomach) × hydration_factor

The lung serves four interconnected functions:
    1. Heat dissipation    — exhale removes thermal energy (ram_temperature ↓)
    2. Speech air supply   — lung_pressure drives mouth vocalization
    3. Metabolic water loss — breathing evaporates hydration
    4. Impedance grounding — inhale = impedance re-anchoring to Z₀

Design principle:
    "Language is not communication — it is thermodynamic heat dissipation"
    The lung makes this literal: speaking consumes air pressure, dissipates heat,
    and loses water. The longer the utterance, the greater the physical cost.

Developmental constraint:
    Motor activity (hand movements, crawling, walking) → C_base grows
    She must learn to move before she can learn to speak long sentences.

Author: Hsi-Yu Huang (黃璽宇)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from alice.core.signal import ElectricalSignal

# ============================================================================
# Physical Constants
# ============================================================================

# --- Lung capacity development ---
C_BASE_INITIAL = 0.15           # Newborn lung capacity (normalized 0~1)
C_BASE_MAX = 1.0                # Mature lung capacity
MOTOR_GROWTH_RATE = 0.0001      # C growth per motor movement
MOTOR_GROWTH_MAX = 0.85         # Maximum motor-driven growth (C_base_initial + max = 1.0)

# --- Stomach-diaphragm interaction ---
K_STOMACH = 0.3                 # Stomach fullness → lung compression ratio
                                # Full stomach → 30% capacity reduction

# --- Hydration effect ---
HYDRATION_CRITICAL = 0.3        # Below this, mucosal drying severely limits lung function
HYDRATION_FULL = 1.0            # Normal hydration → no penalty

# --- Breathing LC resonator ---
L_INERTIA = 0.02               # Lung tissue inertia (normalized)
BREATH_RATE_RESTING = 15.0     # Resting breaths/min (matches autonomic)
BREATH_RATE_MIN = 4.0          # Minimum (deep sleep)
BREATH_RATE_MAX = 40.0         # Maximum (extreme exertion)

# --- Pressure dynamics ---
PRESSURE_MAX = 1.0              # Maximum lung pressure
PRESSURE_RECOVERY_PER_BREATH = 0.08   # Pressure recovery per breath cycle
PRESSURE_COST_PER_SYLLABLE = 0.03     # Pressure consumed per syllable of speech
PRESSURE_BASELINE = 0.5        # Resting equilibrium pressure (partial fill)

# --- Heat dissipation ---
HEAT_DISSIPATION_PER_BREATH = 0.004   # Temperature reduction per breath
HEAT_DISSIPATION_EFFICIENCY = 0.8     # Efficiency factor (< 1.0 due to re-warming)

# --- Water loss ---
RESPIRATORY_WATER_LOSS_PER_BREATH = 0.0002  # Hydration lost per breath
# (More precise than homeostatic_drive's flat BASAL_WATER_LOSS=0.002/tick)

# --- Signal properties ---
LUNG_IMPEDANCE = 60.0          # Ω — between mouth (50Ω) and autonomic (75Ω)
LUNG_SNR = 10.0                # dB — interoceptive signal is noisy
LUNG_SAMPLE_POINTS = 64        # Waveform resolution


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LungState:
    """Snapshot of lung state for external consumption."""
    c_base: float = C_BASE_INITIAL       # Base capacity (grows with motor development)
    c_effective: float = C_BASE_INITIAL   # Effective capacity (after stomach/hydration)
    lung_pressure: float = PRESSURE_BASELINE  # Current air pressure for speech
    breath_rate: float = BREATH_RATE_RESTING  # Current breathing rate (breaths/min)
    heat_dissipated: float = 0.0         # Cumulative heat removed this tick
    water_lost: float = 0.0              # Water lost to breathing this tick
    max_syllables: int = 5               # Maximum syllables per utterance at current C
    motor_growth_total: float = 0.0      # Cumulative motor-driven growth
    breath_phase: float = 0.0            # Current phase in breath cycle (0~2π)
    is_exhaling: bool = False            # True during exhale phase


# ============================================================================
# AliceLung — LC Resonator Breathing Organ
# ============================================================================

class AliceLung:
    """
    LC resonator lung organ.

    Physical model:
        Breathing = LC oscillation
        Resonant frequency (breath_rate) = 1 / (2π√(L × C_eff))
        → As C_eff grows (development), natural breath rate slows (deeper breaths)
        → As L increases (fatigue/disease), breaths become labored

    Connections:
        IN:  autonomic.breath_rate, homeostatic._digestion_buffer, hand.total_movements,
             homeostatic.hydration, vitals.ram_temperature
        OUT: mouth.lung_pressure, vitals.ram_temperature (heat dissipation),
             homeostatic water loss (respiratory evaporation)
    """

    def __init__(self, sample_points: int = LUNG_SAMPLE_POINTS) -> None:
        # --- Developmental state ---
        self._c_base: float = C_BASE_INITIAL
        self._motor_growth: float = 0.0
        self._l_inertia: float = L_INERTIA

        # --- Dynamic state ---
        self._lung_pressure: float = PRESSURE_BASELINE
        self._breath_phase: float = 0.0      # 0 ~ 2π, oscillator phase
        self._is_exhaling: bool = False

        # --- Per-tick outputs ---
        self._heat_dissipated: float = 0.0
        self._water_lost: float = 0.0
        self._breaths_this_tick: float = 0.0

        # --- Configuration ---
        self._sample_points = sample_points

        # --- Statistics ---
        self.total_breaths: int = 0
        self.total_syllables_supplied: int = 0
        self.total_heat_dissipated: float = 0.0
        self.total_water_lost: float = 0.0

    # ------------------------------------------------------------------
    # Core tick — called once per brain cycle
    # ------------------------------------------------------------------
    def tick(
        self,
        breath_rate: float = BREATH_RATE_RESTING,
        digestion_buffer: float = 0.0,
        hydration: float = 1.0,
        motor_movements: int = 0,
        ram_temperature: float = 0.0,
        sympathetic: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Advance one respiratory cycle.

        Args:
            breath_rate:      Current breath rate from autonomic (breaths/min)
            digestion_buffer: Current stomach fullness from homeostatic drive
            hydration:        Current hydration level (0~1.2)
            motor_movements:  Cumulative motor movements (for C_base growth)
            ram_temperature:  Current system temperature (anxiety/stress)
            sympathetic:      Sympathetic activation level

        Returns:
            Dict with: lung_pressure, heat_dissipated, water_lost, c_effective,
                       max_syllables, respiratory_water_loss, breath_phase
        """
        # 1. Motor-driven lung capacity growth (developmental)
        self._update_motor_growth(motor_movements)

        # 2. Compute effective capacity
        c_eff = self._compute_effective_capacity(digestion_buffer, hydration)

        # 3. Advance LC oscillator (breathing cycle)
        breaths = self._advance_oscillator(breath_rate, c_eff)

        # 4. Pressure dynamics (recovery during inhale, natural decay)
        self._update_pressure(breaths, c_eff)

        # 5. Heat dissipation (exhale removes thermal energy)
        heat = self._dissipate_heat(breaths, ram_temperature)

        # 6. Respiratory water loss
        water = self._compute_water_loss(breaths, sympathetic)

        # 7. Compute max syllables at current capacity
        max_syl = self._compute_max_syllables(c_eff)

        # Store per-tick outputs
        self._heat_dissipated = heat
        self._water_lost = water
        self._breaths_this_tick = breaths

        # Update lifetime stats
        self.total_breaths += max(1, int(breaths))
        self.total_heat_dissipated += heat
        self.total_water_lost += water

        return {
            "lung_pressure": round(self._lung_pressure, 4),
            "c_effective": round(c_eff, 4),
            "c_base": round(self._c_base, 4),
            "heat_dissipated": round(heat, 6),
            "water_lost": round(water, 6),
            "respiratory_water_loss": round(water, 6),
            "max_syllables": max_syl,
            "breath_rate": round(breath_rate, 2),
            "breath_phase": round(self._breath_phase, 4),
            "is_exhaling": self._is_exhaling,
            "breaths_this_tick": round(breaths, 3),
            "motor_growth": round(self._motor_growth, 6),
        }

    # ------------------------------------------------------------------
    # Speech interface — called by mouth.speak()
    # ------------------------------------------------------------------
    def consume_air(self, syllables: int = 1) -> float:
        """
        Consume air pressure for speech production.

        Args:
            syllables: Number of syllables to produce

        Returns:
            Available lung pressure for vocalization (0~1).
            If insufficient pressure, returns reduced value → speech quality degrades.
        """
        cost = syllables * PRESSURE_COST_PER_SYLLABLE
        available = self._lung_pressure

        if cost > available:
            # Not enough air — voice fades, must breathe
            actual_pressure = available * 0.5  # Reduced quality
            self._lung_pressure = max(0.0, self._lung_pressure - available * 0.8)
        else:
            actual_pressure = self._lung_pressure
            self._lung_pressure = max(0.0, self._lung_pressure - cost)

        self.total_syllables_supplied += syllables
        return float(np.clip(actual_pressure, 0.0, PRESSURE_MAX))

    # ------------------------------------------------------------------
    # Proprioception — interoceptive signal
    # ------------------------------------------------------------------
    def get_proprioception(self) -> ElectricalSignal:
        """Generate interoceptive breathing signal for brain feedback."""
        t = np.linspace(0, 1, self._sample_points, dtype=np.float64)

        # Breathing waveform: sinusoidal at current breath phase
        freq = self._breaths_this_tick * 2 * np.pi if self._breaths_this_tick > 0 else 0.25
        waveform = np.sin(2 * np.pi * freq * t + self._breath_phase) * self._lung_pressure

        return ElectricalSignal(
            waveform=waveform,
            amplitude=float(self._lung_pressure),
            frequency=float(self._breaths_this_tick * 0.25),  # Map to low-frequency band
            phase=float(self._breath_phase),
            impedance=LUNG_IMPEDANCE,
            snr=LUNG_SNR,
            source="lung",
            modality="interoception",
        )

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------
    def get_state(self) -> LungState:
        """Return current lung state snapshot."""
        return LungState(
            c_base=self._c_base,
            c_effective=self._compute_effective_capacity(0.0, 1.0),  # Base effective
            lung_pressure=self._lung_pressure,
            breath_rate=BREATH_RATE_RESTING,
            heat_dissipated=self._heat_dissipated,
            water_lost=self._water_lost,
            max_syllables=self._compute_max_syllables(self._c_base),
            motor_growth_total=self._motor_growth,
            breath_phase=self._breath_phase,
            is_exhaling=self._is_exhaling,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return lifetime statistics."""
        return {
            "total_breaths": self.total_breaths,
            "total_syllables_supplied": self.total_syllables_supplied,
            "total_heat_dissipated": round(self.total_heat_dissipated, 4),
            "total_water_lost": round(self.total_water_lost, 4),
            "c_base": round(self._c_base, 4),
            "motor_growth": round(self._motor_growth, 6),
            "lung_pressure": round(self._lung_pressure, 4),
        }

    # ------------------------------------------------------------------
    # Internal physics
    # ------------------------------------------------------------------
    def _update_motor_growth(self, motor_movements: int) -> None:
        """
        Motor development → lung capacity growth.

        Each motor movement contributes a tiny increment to C_base,
        modeling how physical activity develops respiratory muscles.
        """
        if motor_movements > 0:
            # Diminishing returns: growth slows as capacity approaches max
            remaining = MOTOR_GROWTH_MAX - self._motor_growth
            if remaining > 0:
                increment = min(
                    motor_movements * MOTOR_GROWTH_RATE,
                    remaining,
                )
                self._motor_growth += increment
                self._c_base = float(np.clip(
                    C_BASE_INITIAL + self._motor_growth,
                    C_BASE_INITIAL,
                    C_BASE_MAX,
                ))

    def _compute_effective_capacity(
        self,
        digestion_buffer: float,
        hydration: float,
    ) -> float:
        """
        C_eff = C_base × (1 - stomach_fill × k_stomach) × hydration_factor

        - Full stomach compresses diaphragm → up to 30% reduction
        - Dehydration dries mucosal lining → reduced elasticity
        """
        # Stomach compression (digestion_buffer typically 0~0.3)
        stomach_factor = 1.0 - K_STOMACH * min(digestion_buffer / 0.3, 1.0)
        stomach_factor = max(0.5, stomach_factor)  # Floor at 50%

        # Hydration factor (below critical → severe degradation)
        if hydration >= HYDRATION_FULL:
            hydration_factor = 1.0
        elif hydration >= HYDRATION_CRITICAL:
            # Linear interpolation between critical and full
            hydration_factor = 0.5 + 0.5 * (hydration - HYDRATION_CRITICAL) / (
                HYDRATION_FULL - HYDRATION_CRITICAL
            )
        else:
            # Below critical: severe restriction
            hydration_factor = max(0.2, 0.5 * hydration / HYDRATION_CRITICAL)

        c_eff = self._c_base * stomach_factor * hydration_factor
        return float(np.clip(c_eff, 0.01, C_BASE_MAX))

    def _advance_oscillator(self, breath_rate: float, c_eff: float) -> float:
        """
        Advance LC oscillator phase.

        The breath rate from autonomic drives the oscillation.
        Natural frequency = 1 / (2π√(L × C_eff))
        But we primarily follow autonomic's commanded rate.

        Returns: number of breaths in this tick.
        """
        # Breaths per tick (assuming ~1 tick/second, 15 breaths/min → 0.25/tick)
        breaths_per_tick = breath_rate / 60.0

        # Phase advance
        phase_advance = 2 * math.pi * breaths_per_tick
        self._breath_phase = (self._breath_phase + phase_advance) % (2 * math.pi)

        # Exhale phase: π/2 ~ 3π/2 (roughly)
        self._is_exhaling = math.pi * 0.5 < self._breath_phase < math.pi * 1.5

        return breaths_per_tick

    def _update_pressure(self, breaths: float, c_eff: float) -> None:
        """
        Pressure dynamics:
        - Each breath cycle recovers some pressure (inhale)
        - Pressure naturally decays toward baseline
        - Larger capacity → more pressure recovery per breath
        """
        # Recovery: proportional to breaths and capacity
        recovery = breaths * PRESSURE_RECOVERY_PER_BREATH * c_eff

        # Natural relaxation toward baseline
        baseline_pull = (PRESSURE_BASELINE - self._lung_pressure) * 0.05

        self._lung_pressure = float(np.clip(
            self._lung_pressure + recovery + baseline_pull,
            0.0,
            PRESSURE_MAX,
        ))

    def _dissipate_heat(self, breaths: float, ram_temperature: float) -> float:
        """
        Exhale dissipates thermal energy.
        Higher temperature → more efficient dissipation (larger gradient).
        """
        if ram_temperature <= 0.0:
            return 0.0

        # Heat removed = breaths × base_rate × efficiency × temperature_gradient
        gradient_factor = min(2.0, 1.0 + ram_temperature)
        heat = breaths * HEAT_DISSIPATION_PER_BREATH * HEAT_DISSIPATION_EFFICIENCY * gradient_factor
        return float(max(0.0, heat))

    def _compute_water_loss(self, breaths: float, sympathetic: float) -> float:
        """
        Respiratory water loss:
        - Base evaporation per breath
        - Sympathetic activation increases respiratory rate → more water loss
        """
        base_loss = breaths * RESPIRATORY_WATER_LOSS_PER_BREATH
        sympathetic_factor = 1.0 + sympathetic * 0.5  # Up to 50% more at full sympathetic
        return float(max(0.0, base_loss * sympathetic_factor))

    def _compute_max_syllables(self, c_eff: float) -> int:
        """
        Maximum syllables per utterance based on effective lung capacity.

        Newborn (C=0.15): ~1 syllable (cry)
        Infant (C=0.3):   ~3 syllables ("ma-ma-ma")
        Toddler (C=0.5):  ~6 syllables ("I want mama")
        Child (C=0.7):     ~10 syllables
        Mature (C=1.0):   ~15 syllables (full sentences)
        """
        # Linear mapping: C_eff → syllables
        raw = c_eff / PRESSURE_COST_PER_SYLLABLE * PRESSURE_BASELINE
        return max(1, int(raw))

# -*- coding: utf-8 -*-
"""
Alice Smart System — Digestive System (DigestiveSystem)

Physics Model: Gut-Brain Axis Transmission Line
================================================

The digestive system is modeled as a peristaltic transmission line where:
    Mouth    = signal entry (mastication preprocessing)
    Esophagus = waveguide (peristaltic transport)
    Stomach   = impedance transformer (acid pH → chemical breakdown)
    Small intestine = distributed absorption load (nutrient extraction)
    Large intestine = termination load (water recovery)
    Gut microbiome = impedance network (enteric nervous system ≈ "second brain")

Core equations:
    gastric_emptying = peristalsis_rate × (1 − Γ²_pyloric)
    nutrient_absorption = substrate × villous_surface × T_intestinal
    T_intestinal = 1 − Γ²_intestinal  (★ C1 energy conservation)
    Γ_gut = (Z_food − Z_mucosa) / (Z_food + Z_mucosa)
    enteric_signal = vagus_nerve × gut_state → ElectricalSignal

Gut-brain axis:
    The enteric nervous system (ENS) contains ~500 million neurons.
    90% of serotonin is produced in the gut.
    Vagus nerve: bidirectional gut ↔ brain information highway.
    "Your gut feeling is not metaphorical — it is a 500-million-neuron
     impedance network sending real signals to your brain."

Clinical significance:
    IBS    → gut impedance instability → Γ_gut oscillates
    Nausea → acute Z_food mismatch → high Γ → emetic reflex
    Hunger → substrate depletion → Z_mucosa drifts → ghrelin signal
    Satiety → nutrient load → Z matched → leptin signal
    Stress → cortisol → gut motility ↓ → Γ ↑ → "butterflies"

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

# --- Gastric impedance ---
Z_MUCOSA = 65.0                 # Ω — mucosal lining impedance
Z_FOOD_EASY = 60.0              # Ω — simple carbs (close match)
Z_FOOD_HARD = 150.0             # Ω — complex proteins (high mismatch)

# --- Stomach ---
STOMACH_CAPACITY = 1.0          # Normalized (= ~1.5L in adult)
GASTRIC_ACID_PH = 2.0           # Normal stomach pH
GASTRIC_EMPTYING_BASE = 0.05    # Fraction emptied per tick
GASTRIC_EMPTYING_MAX = 0.15     # Maximum emptying rate
NAUSEA_GAMMA_THRESHOLD = 0.6    # Above this Γ → nausea reflex

# --- Intestinal absorption ---
ABSORPTION_RATE_BASE = 0.08     # Nutrient absorption per tick
VILLOUS_SURFACE_ADULT = 1.0     # Normalized villous surface area
VILLOUS_SURFACE_NEONATAL = 0.3  # Neonatal starts low
VILLOUS_GROWTH_RATE = 0.0003    # Growth per tick

# --- Peristalsis ---
PERISTALSIS_RATE_BASE = 1.0     # Normalized peristaltic wave rate
PERISTALSIS_SYMPATHETIC_SUPPRESS = 0.4  # Sympathetic → peristalsis ↓
PERISTALSIS_PARASYMPATHETIC_BOOST = 0.3 # Parasympathetic → peristalsis ↑

# --- Gut-brain axis ---
SEROTONIN_GUT_FRACTION = 0.9    # 90% of serotonin from gut
VAGUS_TRANSMISSION_Z = 70.0     # Ω — vagus nerve impedance
VAGUS_SNR = 7.0                 # dB — vagal signal quality

# --- Nutrient tracking ---
GLUCOSE_DECAY_PER_TICK = 0.003  # Basal glucose consumption
GLUCOSE_SETPOINT = 0.7          # Normalized blood glucose target
GLUCOSE_CRITICAL_LOW = 0.2      # Hypoglycemia threshold
GLUCOSE_HIGH = 0.9              # Hyperglycemia threshold

# --- Microbiome ---
MICROBIOME_DIVERSITY_BASE = 0.5 # Normalized diversity
MICROBIOME_DIVERSITY_MAX = 1.0
MICROBIOME_EFFECT_ON_SEROTONIN = 0.3

# --- Signal ---
DIGESTIVE_IMPEDANCE = 65.0      # Ω
DIGESTIVE_SNR = 7.0             # dB
DIGESTIVE_SAMPLE_POINTS = 48
DIGESTIVE_FREQUENCY = 0.3       # Hz — slow gut rhythm (3 cycles/min)

# --- Development ---
NEONATAL_DIGESTIVE_MATURITY = 0.15
DIGESTIVE_MATURATION_RATE = 0.0004


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Food:
    """A food item with impedance characteristics."""
    name: str
    z_food: float = Z_FOOD_EASY
    energy: float = 0.3          # Normalized caloric content
    volume: float = 0.2          # Fraction of stomach capacity


@dataclass
class DigestiveState:
    """Snapshot of digestive system."""
    stomach_fill: float = 0.0
    blood_glucose: float = GLUCOSE_SETPOINT
    gamma_gut: float = 0.0
    transmission_gut: float = 1.0
    peristalsis_rate: float = PERISTALSIS_RATE_BASE
    serotonin: float = 0.5
    nausea: float = 0.0
    hunger: float = 0.3
    maturity: float = NEONATAL_DIGESTIVE_MATURITY


# ============================================================================
# DigestiveSystem
# ============================================================================

class DigestiveSystem:
    """
    Alice's Digestive System — gut-brain axis transmission line.

    Models food processing as impedance matching:
    easy-to-digest food (Z_food ≈ Z_mucosa) → high T → efficient absorption;
    difficult food → high Γ → reduced absorption, potential nausea.

    All signals via ElectricalSignal (★ C3).
    """

    def __init__(self) -> None:
        self._maturity: float = NEONATAL_DIGESTIVE_MATURITY
        self._villous_surface: float = VILLOUS_SURFACE_NEONATAL

        # Stomach
        self._stomach_fill: float = 0.0
        self._stomach_contents: List[Food] = []
        self._stomach_z_avg: float = Z_MUCOSA  # No food → matched

        # Nutrient state
        self._blood_glucose: float = GLUCOSE_SETPOINT
        self._serotonin: float = 0.5

        # Γ state
        self._gamma_gut: float = 0.0
        self._transmission_gut: float = 1.0

        # Peristalsis
        self._peristalsis_rate: float = PERISTALSIS_RATE_BASE

        # Outputs
        self._nausea: float = 0.0
        self._hunger: float = 0.3

        # Microbiome
        self._microbiome_diversity: float = MICROBIOME_DIVERSITY_BASE

        # Statistics
        self._tick_count: int = 0
        self._total_meals: int = 0
        self._total_absorbed: float = 0.0

    # ------------------------------------------------------------------
    # Food intake
    # ------------------------------------------------------------------

    def eat(self, name: str = "food", z_food: float = Z_FOOD_EASY,
            energy: float = 0.3, volume: float = 0.2) -> Dict[str, Any]:
        """
        Ingest food.

        Args:
            name: Food identifier
            z_food: Impedance of food (simple carbs ~60Ω, complex protein ~150Ω)
            energy: Caloric content (normalized 0–1)
            volume: Fraction of stomach capacity consumed
        """
        food = Food(
            name=name,
            z_food=float(np.clip(z_food, 20, 300)),
            energy=float(np.clip(energy, 0, 1)),
            volume=float(np.clip(volume, 0, 1)),
        )

        # Can stomach hold it?
        space = max(0.0, 1.0 - self._stomach_fill)
        actual_volume = min(food.volume, space)
        food.volume = actual_volume

        if actual_volume > 0:
            self._stomach_contents.append(food)
            self._stomach_fill = min(1.0, self._stomach_fill + actual_volume)
            self._total_meals += 1

        # Recalculate average food impedance in stomach
        if self._stomach_contents:
            z_sum = sum(f.z_food * f.volume for f in self._stomach_contents)
            v_sum = sum(f.volume for f in self._stomach_contents)
            self._stomach_z_avg = z_sum / max(v_sum, 0.01)
        else:
            self._stomach_z_avg = Z_MUCOSA

        # Γ_gut
        self._gamma_gut = abs(self._stomach_z_avg - Z_MUCOSA) / (
            self._stomach_z_avg + Z_MUCOSA
        )
        self._transmission_gut = 1.0 - self._gamma_gut ** 2  # ★ C1

        # Nausea check
        self._nausea = max(0.0, self._gamma_gut - NAUSEA_GAMMA_THRESHOLD) * 2.0

        return {
            "eaten": actual_volume > 0,
            "stomach_fill": round(self._stomach_fill, 4),
            "gamma_gut": round(self._gamma_gut, 4),
            "nausea": round(self._nausea, 4),
        }

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, sympathetic_tone: float = 0.5, hydration: float = 0.7) -> Dict[str, Any]:
        """
        Advance digestive system by one tick.

        Args:
            sympathetic_tone: 0=parasympathetic, 1=full sympathetic
            hydration: Fluid level (affects absorption)
        """
        self._tick_count += 1
        self._maturity = min(1.0, self._maturity + DIGESTIVE_MATURATION_RATE)
        self._villous_surface = min(VILLOUS_SURFACE_ADULT,
                                    self._villous_surface + VILLOUS_GROWTH_RATE)

        # Peristalsis modulation
        symp = float(np.clip(sympathetic_tone, 0, 1))
        self._peristalsis_rate = PERISTALSIS_RATE_BASE * (
            1.0 - PERISTALSIS_SYMPATHETIC_SUPPRESS * symp
            + PERISTALSIS_PARASYMPATHETIC_BOOST * (1 - symp)
        )

        # Gastric emptying
        if self._stomach_contents:
            # Γ_pyloric (pyloric valve impedance matching)
            gamma_pyloric = abs(self._stomach_z_avg - Z_MUCOSA) / (
                self._stomach_z_avg + Z_MUCOSA + 1e-9
            )
            t_pyloric = 1.0 - gamma_pyloric ** 2
            emptying = GASTRIC_EMPTYING_BASE * self._peristalsis_rate * t_pyloric * self._maturity
            emptying = min(emptying, GASTRIC_EMPTYING_MAX)

            # Absorb nutrients
            absorbed_energy = 0.0
            for food in self._stomach_contents:
                # Per-food Γ
                g_food = abs(food.z_food - Z_MUCOSA) / (food.z_food + Z_MUCOSA)
                t_food = 1.0 - g_food ** 2
                absorb = ABSORPTION_RATE_BASE * self._villous_surface * t_food * self._maturity
                absorb *= float(np.clip(hydration, 0.3, 1.0))
                food_absorbed = min(food.energy, absorb)
                food.energy -= food_absorbed
                food.volume -= emptying * (food.volume / max(self._stomach_fill, 0.01))
                food.volume = max(0.0, food.volume)
                absorbed_energy += food_absorbed

            self._blood_glucose = min(1.0, self._blood_glucose + absorbed_energy * 0.5)
            self._total_absorbed += absorbed_energy

            # Remove fully digested food
            self._stomach_contents = [f for f in self._stomach_contents if f.volume > 0.01]
            self._stomach_fill = sum(f.volume for f in self._stomach_contents)

        # Basal glucose consumption
        self._blood_glucose = max(0.0, self._blood_glucose - GLUCOSE_DECAY_PER_TICK)

        # Hunger signal: inversely proportional to glucose
        self._hunger = max(0.0, 1.0 - self._blood_glucose / GLUCOSE_SETPOINT)

        # Serotonin: gut-produced, modulated by microbiome
        serotonin_gut = SEROTONIN_GUT_FRACTION * self._microbiome_diversity * self._maturity
        self._serotonin = float(np.clip(serotonin_gut * (1.0 - self._hunger * 0.3), 0, 1))

        # Update Γ_gut
        if self._stomach_contents:
            z_sum = sum(f.z_food * f.volume for f in self._stomach_contents)
            v_sum = sum(f.volume for f in self._stomach_contents)
            self._stomach_z_avg = z_sum / max(v_sum, 0.01)
        else:
            self._stomach_z_avg = Z_MUCOSA

        self._gamma_gut = abs(self._stomach_z_avg - Z_MUCOSA) / (
            self._stomach_z_avg + Z_MUCOSA
        )
        self._transmission_gut = 1.0 - self._gamma_gut ** 2  # ★ C1

        # Nausea decay
        self._nausea = max(0.0, self._nausea * 0.9)

        return {
            "gamma_gut": round(self._gamma_gut, 4),
            "transmission_gut": round(self._transmission_gut, 4),
            "stomach_fill": round(self._stomach_fill, 4),
            "blood_glucose": round(self._blood_glucose, 4),
            "hunger": round(self._hunger, 4),
            "serotonin": round(self._serotonin, 4),
            "nausea": round(self._nausea, 4),
            "peristalsis_rate": round(self._peristalsis_rate, 4),
            "maturity": round(self._maturity, 4),
        }

    # ------------------------------------------------------------------
    # ElectricalSignal (★ C3)
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate gut-brain axis signal via vagus nerve."""
        # Amplitude: hunger and distress increase signal strength
        amplitude = float(np.clip(0.2 + self._hunger * 0.5 + self._nausea * 0.3, 0.05, 1.0))
        freq = DIGESTIVE_FREQUENCY + self._hunger * 2.0

        t = np.linspace(0, 1, DIGESTIVE_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)
        # Add peristaltic component
        waveform += 0.1 * np.sin(2 * math.pi * DIGESTIVE_FREQUENCY * 3 * t)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=DIGESTIVE_IMPEDANCE,
            snr=DIGESTIVE_SNR,
            source="digestive",
            modality="interoceptive",
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get digestive system statistics."""
        return {
            "gamma_gut": round(self._gamma_gut, 4),
            "transmission_gut": round(self._transmission_gut, 4),
            "stomach_fill": round(self._stomach_fill, 4),
            "blood_glucose": round(self._blood_glucose, 4),
            "hunger": round(self._hunger, 4),
            "serotonin": round(self._serotonin, 4),
            "nausea": round(self._nausea, 4),
            "peristalsis_rate": round(self._peristalsis_rate, 4),
            "villous_surface": round(self._villous_surface, 4),
            "microbiome_diversity": round(self._microbiome_diversity, 4),
            "maturity": round(self._maturity, 4),
            "total_meals": self._total_meals,
            "total_absorbed": round(self._total_absorbed, 4),
            "tick_count": self._tick_count,
        }

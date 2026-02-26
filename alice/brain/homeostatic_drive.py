# -*- coding: utf-8 -*-
"""
Hypothalamic Homeostatic Drive Engine — Hunger / Thirst / Thermoregulation

Physics:
  "The most basic need of life is not thinking — it's staying alive.
   Hunger is not a feeling; it's an alarm circuit when fuel drops below critical.
   Thirst is not a desire; it's a feedback signal when osmotic pressure deviates from the setpoint."

  Homeostasis (Cannon, 1929):
    Every physiological variable has a "setpoint".
    Deviation from the setpoint → generates a "drive" → pushes behavior to restore the setpoint.
    Drive intensity = nonlinear function of deviation (low deviation ignored, high deviation rises sharply).

  Impedance model of hunger:
    - Glucose = supply voltage of the circuit
    - Hypoglycemia = voltage drop → insufficient gate voltage → all gate efficiencies degrade
    - Γ_hunger = |glucose - setpoint| / setpoint
    - When Γ_hunger > threshold → interoceptive error injected into LifeLoop

  Impedance model of thirst:
    - Water = coolant
    - Dehydration = insufficient heat dissipation → temperature rises → impedance drift
    - Γ_thirst = |hydration - setpoint| / setpoint
    - When Γ_thirst > threshold → energy recovery efficiency decreases

  Satiety signal (negative feedback):
    - Eating → blood glucose rises → Γ_hunger → 0 → drive vanishes
    - Drinking → hydration rises → Γ_thirst → 0 → drive vanishes
    - Satiety = impedance matching = system returns to setpoint

Circuit analogy:
  Hypothalamus = Power Management IC (PMIC)
  Hunger drive = Under-Voltage Lockout (UVLO) trigger
  Thirst drive = Over-Temperature Protection (OTP) warning
  Satiety signal = Voltage regulator's FB pin returns to reference
  Metabolic rate = Circuit power consumption = f(sympathetic activity, cognitive load, motor activity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

# Glucose (blood sugar)
GLUCOSE_SETPOINT = 1.0          # Normal blood glucose (normalized)
GLUCOSE_MIN = 0.0               # Hypoglycemic coma
GLUCOSE_MAX = 1.5               # Hyperglycemia (overeating)
GLUCOSE_CRITICAL_LOW = 0.2      # Hypoglycemia alarm threshold
GLUCOSE_HUNGER_ONSET = 0.6      # Threshold where hunger sensation begins

# Hydration (water level)
HYDRATION_SETPOINT = 1.0        # Normal hydration level
HYDRATION_MIN = 0.0             # Severe dehydration
HYDRATION_MAX = 1.2             # Water intoxication
HYDRATION_CRITICAL_LOW = 0.3    # Dehydration alarm threshold
HYDRATION_THIRST_ONSET = 0.7    # Threshold where thirst sensation begins

# Metabolic rate
BASAL_METABOLIC_RATE = 0.003    # Basal metabolism (consumed per tick), at rest
COGNITIVE_METABOLIC_COST = 0.002  # Extra consumption from cognitive load
MOTOR_METABOLIC_COST = 0.004    # Extra consumption from motor activity
SYMPATHETIC_METABOLIC_COST = 0.003  # Extra consumption from sympathetic activation (fight-or-flight burns more fuel)
SLEEP_METABOLIC_REDUCTION = 0.5  # 50% metabolic rate reduction during sleep

# Water loss
BASAL_WATER_LOSS = 0.002        # Basal water loss (breathing, skin evaporation)
SYMPATHETIC_WATER_LOSS = 0.003  # Sympathetic activation (sweating)
TEMPERATURE_WATER_LOSS = 0.002  # Extra water loss at high temperature

# Satiety effect
EATING_GLUCOSE_GAIN = 0.3       # Glucose recovery per meal
DRINKING_HYDRATION_GAIN = 0.35  # Hydration recovery per drink
DIGESTION_RATE = 0.02           # Digestion delay (absorption per tick)
SATIETY_DURATION = 30           # Satiety feeling duration in ticks

# Drive intensity function
DRIVE_EXPONENT = 2.0            # Drive nonlinear exponent (quadratic → sharp increase)
MAX_DRIVE_INTENSITY = 1.0       # Drive intensity upper limit

# Cognitive impact
HUNGER_COGNITIVE_PENALTY = 0.3   # Hunger reduces cognitive performance by up to 30%
THIRST_COGNITIVE_PENALTY = 0.4   # Thirst reduces cognitive performance by up to 40% (dehydration is more urgent)
HUNGER_IRRITABILITY = 0.25       # Hunger-induced irritability (injects negative valence)
THIRST_PAIN_CONTRIBUTION = 0.15  # Severe dehydration contributes to pain

# LifeLoop Integration
HOMEOSTATIC_URGENCY_BASE = 0.5  # Base urgency of homeostatic error
HOMEOSTATIC_URGENCY_CRITICAL = 0.95  # Urgency in critical state


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class HomeostaticState:
    """Homeostatic state snapshot"""
    glucose: float
    hydration: float
    hunger_drive: float
    thirst_drive: float
    metabolic_rate: float
    satiety_timer: int
    glucose_trend: str        # "rising", "falling", "stable"
    hydration_trend: str
    cognitive_penalty: float  # Total cognitive penalty
    is_starving: bool
    is_dehydrated: bool
    gamma_hunger: float       # Hunger impedance
    gamma_thirst: float       # Thirst impedance


@dataclass
class HomeostaticDriveSignal:
    """Homeostatic drive signal — injected into LifeLoop/AutonomicNervousSystem"""
    hunger_intensity: float      # Hunger intensity 0~1
    thirst_intensity: float      # Thirst intensity 0~1
    irritability: float          # Irritability level (hunger-induced)
    pain_contribution: float     # Dehydration pain contribution
    cognitive_penalty: float     # Cognitive performance penalty
    energy_recovery_factor: float  # Energy recovery efficiency modifier
    metabolic_rate: float        # Current metabolic rate
    needs_food: bool             # Whether food is needed
    needs_water: bool            # Whether water is needed


# ============================================================================
# Hypothalamic Homeostatic Drive Engine
# ============================================================================


class HomeostaticDriveEngine:
    """
    Hypothalamic Homeostatic Drive Engine

    Core functions:
    1. Glucose homeostasis → hunger drive
       - Basal metabolism + cognitive/motor consumption → blood glucose drops
       - Blood glucose < threshold → hunger drive ↑
       - Eating → blood glucose rises → hunger drive → 0

    2. Water homeostasis → thirst drive
       - Breathing/sweating/temperature → water loss
       - Hydration < threshold → thirst drive ↑
       - Drinking → hydration rises → thirst drive → 0

    3. Cognitive modulation
       - Hunger → attention scattering, irritability
       - Thirst → reduced cognitive efficiency, lowered pain threshold
       - Severe deficiency → survival drive overrides all other goals

    4. LifeLoop integration
       - Homeostatic deviation → INTEROCEPTIVE error
       - Compensatory action = eating/drinking behavior

    Physical semantics:
      Γ_hunger = (setpoint - glucose) / setpoint  (when glucose < setpoint)
      Γ_thirst = (setpoint - hydration) / setpoint
      Drive intensity = Γ^exponent (nonlinear amplification)
      Life = maintaining all Γ_homeostatic → 0
    """

    def __init__(self):
        # Core physiological variables
        self.glucose: float = GLUCOSE_SETPOINT         # Blood glucose
        self.hydration: float = HYDRATION_SETPOINT     # Hydration level

        # Digestive system (delayed absorption buffer)
        self._digestion_buffer: float = 0.0   # Unabsorbed food in stomach
        self._water_buffer: float = 0.0       # Unabsorbed water in stomach

        # Satiety timer
        self._satiety_timer: int = 0          # Satiety countdown after last meal
        self._hydrated_timer: int = 0         # Quench countdown after last drink

        # Drive state
        self.hunger_drive: float = 0.0        # Hunger drive 0~1
        self.thirst_drive: float = 0.0        # Thirst drive 0~1

        # History
        self._glucose_history: List[float] = []
        self._hydration_history: List[float] = []
        self._hunger_history: List[float] = []
        self._thirst_history: List[float] = []

        # Statistics
        self._total_ticks: int = 0
        self._meals_eaten: int = 0
        self._drinks_taken: int = 0
        self._starvation_ticks: int = 0
        self._dehydration_ticks: int = 0

        self._max_history: int = 300

    # ------------------------------------------------------------------
    # Core Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        sympathetic: float = 0.2,
        cognitive_load: float = 0.0,
        motor_activity: float = 0.0,
        core_temp: float = 37.0,
        is_sleeping: bool = False,
    ) -> HomeostaticDriveSignal:
        """
        Called once per cognitive cycle — update all homeostatic variables

        THE HOMEOSTATIC LOOP:
        1. Compute metabolic consumption → blood glucose drops
        2. Compute water loss → hydration drops
        3. Digestive buffer absorption → partial recovery
        4. Compute drive intensity (nonlinear)
        5. Compute impact on cognition/emotion/pain
        6. Return drive signal

        Parameters:
            sympathetic: Sympathetic activity 0~1
            cognitive_load: Cognitive load 0~1
            motor_activity: Motor activity 0~1
            core_temp: Core temperature (°C)
            is_sleeping: Whether sleeping
        """
        self._total_ticks += 1

        # ================================================================
        # STEP 1: Metabolic consumption → blood glucose drops
        # ================================================================
        metabolic_rate = BASAL_METABOLIC_RATE
        metabolic_rate += cognitive_load * COGNITIVE_METABOLIC_COST
        metabolic_rate += motor_activity * MOTOR_METABOLIC_COST
        metabolic_rate += sympathetic * SYMPATHETIC_METABOLIC_COST

        if is_sleeping:
            metabolic_rate *= SLEEP_METABOLIC_REDUCTION

        self.glucose -= metabolic_rate
        self.glucose = float(np.clip(self.glucose, GLUCOSE_MIN, GLUCOSE_MAX))

        # ================================================================
        # STEP 2: Water loss
        # ================================================================
        water_loss = BASAL_WATER_LOSS

        # Sympathetic activation → sweating
        if sympathetic > 0.4:
            water_loss += (sympathetic - 0.4) * SYMPATHETIC_WATER_LOSS

        # High temperature → extra evaporation
        if core_temp > 37.5:
            water_loss += (core_temp - 37.5) * TEMPERATURE_WATER_LOSS

        if is_sleeping:
            water_loss *= 0.7  # Sleep reduces water loss

        self.hydration -= water_loss
        self.hydration = float(np.clip(self.hydration, HYDRATION_MIN, HYDRATION_MAX))

        # ================================================================
        # STEP 3: Digestive absorption (delayed recovery)
        # ================================================================
        if self._digestion_buffer > 0:
            absorbed = min(self._digestion_buffer, DIGESTION_RATE)
            self.glucose += absorbed
            self._digestion_buffer -= absorbed
            self.glucose = float(np.clip(self.glucose, GLUCOSE_MIN, GLUCOSE_MAX))

        if self._water_buffer > 0:
            absorbed_water = min(self._water_buffer, DIGESTION_RATE * 1.5)  # Water absorbs faster
            self.hydration += absorbed_water
            self._water_buffer -= absorbed_water
            self.hydration = float(np.clip(self.hydration, HYDRATION_MIN, HYDRATION_MAX))

        # Satiety countdown
        if self._satiety_timer > 0:
            self._satiety_timer -= 1
        if self._hydrated_timer > 0:
            self._hydrated_timer -= 1

        # ================================================================
        # STEP 4: Compute drive intensity
        # ================================================================
        # Γ_hunger = relative deviation from setpoint
        if self.glucose < GLUCOSE_HUNGER_ONSET:
            gamma_hunger = (GLUCOSE_HUNGER_ONSET - self.glucose) / GLUCOSE_HUNGER_ONSET
            # Nonlinear amplification (hungrier = more urgent)
            self.hunger_drive = float(np.clip(
                gamma_hunger ** DRIVE_EXPONENT * MAX_DRIVE_INTENSITY,
                0.0, MAX_DRIVE_INTENSITY,
            ))
        else:
            self.hunger_drive = 0.0

        # Satiety suppresses drive
        if self._satiety_timer > 0:
            satiety_suppression = self._satiety_timer / SATIETY_DURATION
            self.hunger_drive *= (1.0 - satiety_suppression * 0.8)

        # Γ_thirst
        if self.hydration < HYDRATION_THIRST_ONSET:
            gamma_thirst = (HYDRATION_THIRST_ONSET - self.hydration) / HYDRATION_THIRST_ONSET
            self.thirst_drive = float(np.clip(
                gamma_thirst ** DRIVE_EXPONENT * MAX_DRIVE_INTENSITY,
                0.0, MAX_DRIVE_INTENSITY,
            ))
        else:
            self.thirst_drive = 0.0

        if self._hydrated_timer > 0:
            hydrated_suppression = self._hydrated_timer / SATIETY_DURATION
            self.thirst_drive *= (1.0 - hydrated_suppression * 0.8)

        # ================================================================
        # STEP 5: Compute cognitive/emotional impact
        # ================================================================
        # Cognitive penalty
        cognitive_penalty = (
            self.hunger_drive * HUNGER_COGNITIVE_PENALTY
            + self.thirst_drive * THIRST_COGNITIVE_PENALTY
        )
        cognitive_penalty = float(np.clip(cognitive_penalty, 0.0, 0.7))

        # Irritability (hunger → negative emotion) — "hangry"
        irritability = self.hunger_drive * HUNGER_IRRITABILITY

        # Dehydration contributes to pain
        pain_contribution = 0.0
        if self.hydration < HYDRATION_CRITICAL_LOW:
            dehydration_severity = (HYDRATION_CRITICAL_LOW - self.hydration) / HYDRATION_CRITICAL_LOW
            pain_contribution = dehydration_severity * THIRST_PAIN_CONTRIBUTION

        # Energy recovery efficiency (low blood glucose → slower recovery)
        energy_recovery_factor = float(np.clip(self.glucose / GLUCOSE_SETPOINT, 0.3, 1.0))

        # Statistics
        is_starving = self.glucose < GLUCOSE_CRITICAL_LOW
        is_dehydrated = self.hydration < HYDRATION_CRITICAL_LOW
        if is_starving:
            self._starvation_ticks += 1
        if is_dehydrated:
            self._dehydration_ticks += 1

        # ================================================================
        # Record history
        # ================================================================
        self._glucose_history.append(self.glucose)
        self._hydration_history.append(self.hydration)
        self._hunger_history.append(self.hunger_drive)
        self._thirst_history.append(self.thirst_drive)

        for hist in (
            self._glucose_history, self._hydration_history,
            self._hunger_history, self._thirst_history,
        ):
            if len(hist) > self._max_history:
                del hist[:-self._max_history]

        # ================================================================
        # Output drive signal
        # ================================================================
        return HomeostaticDriveSignal(
            hunger_intensity=self.hunger_drive,
            thirst_intensity=self.thirst_drive,
            irritability=irritability,
            pain_contribution=pain_contribution,
            cognitive_penalty=cognitive_penalty,
            energy_recovery_factor=energy_recovery_factor,
            metabolic_rate=metabolic_rate,
            needs_food=self.hunger_drive > 0.3 or is_starving,
            needs_water=self.thirst_drive > 0.3 or is_dehydrated,
        )

    # ------------------------------------------------------------------
    # Eating/Drinking Interface
    # ------------------------------------------------------------------

    def eat(self, amount: float = EATING_GLUCOSE_GAIN) -> Dict[str, Any]:
        """
        Eat — delayed glucose replenishment

        Food enters stomach → digestion buffer → gradually absorbed each tick (blood glucose doesn't spike instantly)

        Returns: Eating result
        """
        self._meals_eaten += 1
        self._digestion_buffer += amount
        self._satiety_timer = SATIETY_DURATION

        return {
            "action": "eat",
            "amount": round(amount, 4),
            "digestion_buffer": round(self._digestion_buffer, 4),
            "glucose_before": round(self.glucose, 4),
            "hunger_drive_before": round(self.hunger_drive, 4),
            "satiety_timer": self._satiety_timer,
            "meals_total": self._meals_eaten,
        }

    def drink(self, amount: float = DRINKING_HYDRATION_GAIN) -> Dict[str, Any]:
        """
        Drink — delayed hydration replenishment

        Returns: Drinking result
        """
        self._drinks_taken += 1
        self._water_buffer += amount
        self._hydrated_timer = SATIETY_DURATION

        return {
            "action": "drink",
            "amount": round(amount, 4),
            "water_buffer": round(self._water_buffer, 4),
            "hydration_before": round(self.hydration, 4),
            "thirst_drive_before": round(self.thirst_drive, 4),
            "hydrated_timer": self._hydrated_timer,
            "drinks_total": self._drinks_taken,
        }

    # ------------------------------------------------------------------
    # Impedance Query
    # ------------------------------------------------------------------

    def get_gamma_hunger(self) -> float:
        """Hunger impedance:Γ_hunger = (setpoint - glucose) / setpoint"""
        if self.glucose >= GLUCOSE_SETPOINT:
            return 0.0
        return (GLUCOSE_SETPOINT - self.glucose) / GLUCOSE_SETPOINT

    def get_gamma_thirst(self) -> float:
        """Thirst impedance:Γ_thirst = (setpoint - hydration) / setpoint"""
        if self.hydration >= HYDRATION_SETPOINT:
            return 0.0
        return (HYDRATION_SETPOINT - self.hydration) / HYDRATION_SETPOINT

    # ------------------------------------------------------------------
    # State Query
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Complete state"""
        return {
            "glucose": round(self.glucose, 4),
            "hydration": round(self.hydration, 4),
            "hunger_drive": round(self.hunger_drive, 4),
            "thirst_drive": round(self.thirst_drive, 4),
            "gamma_hunger": round(self.get_gamma_hunger(), 4),
            "gamma_thirst": round(self.get_gamma_thirst(), 4),
            "metabolic_buffer": round(self._digestion_buffer, 4),
            "water_buffer": round(self._water_buffer, 4),
            "satiety_timer": self._satiety_timer,
            "hydrated_timer": self._hydrated_timer,
            "is_starving": self.glucose < GLUCOSE_CRITICAL_LOW,
            "is_dehydrated": self.hydration < HYDRATION_CRITICAL_LOW,
            "total_ticks": self._total_ticks,
            "meals_eaten": self._meals_eaten,
            "drinks_taken": self._drinks_taken,
            "starvation_ticks": self._starvation_ticks,
            "dehydration_ticks": self._dehydration_ticks,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Statistics snapshot"""
        return self.get_state()

    def get_waveforms(self, last_n: int = 60) -> Dict[str, List[float]]:
        """Waveform data"""
        return {
            "glucose": self._glucose_history[-last_n:],
            "hydration": self._hydration_history[-last_n:],
            "hunger_drive": self._hunger_history[-last_n:],
            "thirst_drive": self._thirst_history[-last_n:],
        }

    def reset(self):
        """Reset to satiated state"""
        self.glucose = GLUCOSE_SETPOINT
        self.hydration = HYDRATION_SETPOINT
        self._digestion_buffer = 0.0
        self._water_buffer = 0.0
        self._satiety_timer = 0
        self._hydrated_timer = 0
        self.hunger_drive = 0.0
        self.thirst_drive = 0.0

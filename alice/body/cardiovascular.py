# -*- coding: utf-8 -*-
"""
Alice Smart System — Cardiovascular System (CardiovascularSystem)

Physics Model: Transmission Line Hemodynamics
=============================================

The cardiovascular system is modeled as a transmission line network where:
    Heart  = voltage source (pump)
    Blood  = current (carrier fluid)
    Vessels = transmission lines with characteristic impedance
    Blood pressure = voltage at a node
    Cerebral perfusion = power delivered to the brain load

Core equations:
    blood_volume = f(hydration)
    cardiac_output = heart_rate × stroke_volume(blood_volume, contractility)
    blood_pressure = cardiac_output × vascular_resistance
    cerebral_perfusion = blood_pressure × (1 - Γ²_stenosis) / Z_cerebrovascular
    O₂_delivery = cerebral_perfusion × SpO₂ × hemoglobin_efficiency

The blood is not a metaphor — it is the transmission medium.
Water determines its volume. The heart determines its flow rate.
Oxygen determines what it carries. And the brain is the load.

Without blood volume, the heart pumps air.
Without perfusion, neurons starve.
Without oxygen, channels drift.

"The brain does not die from lack of thought — it dies from lack of blood."

Clinical significance:
    Dehydration → blood_volume ↓ → BP ↓ → tachycardia (compensatory)
    Stroke → vascular Γ → 1.0 → regional perfusion = 0
    Chronic hypoperfusion → diffuse impedance drift → vascular dementia
    HIE (birth asphyxia) → global perfusion = 0 → permanent channel damage
    Anemia → O₂ delivery ↓ → cognitive deficit

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

# --- Blood volume ---
BLOOD_VOLUME_SETPOINT = 1.0          # Normalized (= ~5L in adult, proportional in infant)
BLOOD_VOLUME_MIN = 0.3               # Hypovolemic shock threshold
BLOOD_VOLUME_CRITICAL = 0.5          # Compensatory tachycardia threshold
BLOOD_VOLUME_MAX = 1.2               # Hypervolemia (fluid overload)
HYDRATION_TO_VOLUME_GAIN = 0.85      # hydration → blood_volume scaling (plasma = 55% of blood)

# --- Heart pump ---
STROKE_VOLUME_BASE = 0.7             # Normalized stroke volume at full blood volume
STROKE_VOLUME_MIN = 0.2              # Minimum stroke volume (empty heart)
CONTRACTILITY_BASE = 1.0             # Baseline contractility
CONTRACTILITY_SYMPATHETIC_GAIN = 0.3 # Sympathetic → contractility boost (inotropy)

# --- Vascular resistance ---
VASCULAR_RESISTANCE_BASE = 1.0       # Baseline peripheral resistance
SYMPATHETIC_VASOCONSTRICTION = 0.4   # Sympathetic → resistance increase
PARASYMPATHETIC_VASODILATION = 0.2   # Parasympathetic → resistance decrease
TEMPERATURE_VASODILATION = 0.15      # High temp → vasodilation (heat dissipation)

# --- Blood pressure ---
SYSTOLIC_BASE = 120.0                # mmHg (normal adult)
DIASTOLIC_BASE = 80.0                # mmHg
MAP_NORMAL = 93.0                    # Mean Arterial Pressure (mmHg)
MAP_CRITICAL_LOW = 60.0              # Below this: organ damage begins
MAP_SYNCOPE = 50.0                   # Loss of consciousness threshold
# Alice uses normalized BP (0~1 scale) internally
BP_NORMALIZE_FACTOR = 150.0          # Max physiological MAP for normalization

# --- Cerebral perfusion ---
CEREBRAL_AUTOREGULATION_LOW = 0.4    # Below this MAP_norm, autoregulation fails
CEREBRAL_AUTOREGULATION_HIGH = 0.9   # Above this, autoregulation vasodilates
PERFUSION_NORMAL = 1.0               # Normalized perfusion
PERFUSION_CRITICAL = 0.4             # Below this: ischemia begins
PERFUSION_LETHAL = 0.15              # Below this: rapid cell death
CEREBRAL_Z = 80.0                    # Cerebrovascular impedance (Ω)

# --- Oxygen transport ---
SPO2_NORMAL = 0.98                   # Normal peripheral O₂ saturation
SPO2_HYPOXIA_MILD = 0.92            # Mild hypoxia threshold
SPO2_HYPOXIA_SEVERE = 0.85          # Severe hypoxia
SPO2_MIN = 0.60                      # Incompatible with consciousness
HEMOGLOBIN_EFFICIENCY = 1.0          # Baseline (anemia reduces this)
O2_DELIVERY_NORMAL = 1.0             # Normalized
O2_CHANNEL_PENALTY_THRESHOLD = 0.7   # Below this O₂ delivery → channel Γ drift

# --- Blood viscosity ---
VISCOSITY_NORMAL = 1.0               # Normalized at full hydration
VISCOSITY_DEHYDRATION_GAIN = 0.8     # How much viscosity increases with dehydration
VISCOSITY_MAX = 2.5                  # Maximum viscosity (severe dehydration)

# --- Heat transport ---
BLOOD_HEAT_TRANSPORT_RATE = 0.003    # Temperature reduction per cardiac cycle
BLOOD_HEAT_CAPACITY = 0.8           # Blood's relative heat capacity

# --- Compensatory mechanisms ---
TACHYCARDIA_THRESHOLD = 0.7         # blood_volume below this → heart speeds up
TACHYCARDIA_GAIN = 30.0             # Extra bpm per unit deficit
BARORECEPTOR_TIME_CONSTANT = 0.1    # EMA smoothing for BP regulation

# --- Signal properties ---
CARDIOVASCULAR_IMPEDANCE = 70.0      # Ω — between lung (60Ω) and autonomic (75Ω)
CARDIOVASCULAR_SNR = 8.0             # dB — interoceptive signal (heartbeat awareness)
CV_SAMPLE_POINTS = 64                # Waveform resolution

# --- Developmental ---
NEONATAL_VOLUME_FACTOR = 0.4         # Newborn blood volume ratio
VOLUME_GROWTH_RATE = 0.00005         # Growth per tick with motor activity


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CardiovascularState:
    """Snapshot of cardiovascular state for external consumption."""
    blood_volume: float = BLOOD_VOLUME_SETPOINT
    stroke_volume: float = STROKE_VOLUME_BASE
    cardiac_output: float = 0.0
    systolic_bp: float = SYSTOLIC_BASE
    diastolic_bp: float = DIASTOLIC_BASE
    mean_arterial_pressure: float = MAP_NORMAL
    vascular_resistance: float = VASCULAR_RESISTANCE_BASE
    cerebral_perfusion: float = PERFUSION_NORMAL
    spo2: float = SPO2_NORMAL
    o2_delivery: float = O2_DELIVERY_NORMAL
    blood_viscosity: float = VISCOSITY_NORMAL
    heat_transported: float = 0.0
    is_tachycardic: bool = False
    is_hypotensive: bool = False
    is_hypoxic: bool = False


# ============================================================================
# CardiovascularSystem — Transmission Line Hemodynamics
# ============================================================================

class CardiovascularSystem:
    """
    Cardiovascular system as a transmission line hemodynamic model.

    Physical model:
        Heart = voltage source (pump) → cardiac_output = HR × SV
        Vessels = transmission lines → blood_pressure = CO × R
        Brain = load → cerebral_perfusion = MAP / Z_cerebrovascular
        Blood = transmission medium → O₂_delivery = perfusion × SpO₂

    Connections:
        IN:  autonomic.heart_rate, autonomic.sympathetic, autonomic.parasympathetic,
             homeostatic.hydration, homeostatic.glucose,
             lung.breaths_this_tick (O₂ exchange),
             vitals.ram_temperature
        OUT: cerebral_perfusion → consciousness (arousal modulation)
             cerebral_perfusion → vitals.consciousness (perfusion dependency)
             blood_pressure → vitals (reporting)
             o2_delivery → channel efficiency (hypoxic drift)
             heat_transported → vitals.ram_temperature (blood cooling)
             compensatory_hr_delta → autonomic (baroreceptor reflex)
    """

    def __init__(self, sample_points: int = CV_SAMPLE_POINTS) -> None:
        # --- Core state ---
        self._blood_volume: float = BLOOD_VOLUME_SETPOINT * NEONATAL_VOLUME_FACTOR
        self._hemoglobin: float = HEMOGLOBIN_EFFICIENCY
        self._vascular_resistance: float = VASCULAR_RESISTANCE_BASE
        self._contractility: float = CONTRACTILITY_BASE

        # --- Dynamic state ---
        self._stroke_volume: float = STROKE_VOLUME_BASE * NEONATAL_VOLUME_FACTOR
        self._cardiac_output: float = 0.0
        self._map_normalized: float = 0.6  # Neonatal MAP is lower
        self._systolic: float = 80.0       # Neonatal
        self._diastolic: float = 50.0      # Neonatal
        self._cerebral_perfusion: float = 0.8  # Start slightly below adult
        self._spo2: float = SPO2_NORMAL
        self._o2_delivery: float = 0.8
        self._blood_viscosity: float = VISCOSITY_NORMAL
        self._heat_transported: float = 0.0

        # --- Compensatory state ---
        self._compensatory_hr_delta: float = 0.0  # Extra HR from baroreceptor reflex
        self._is_tachycardic: bool = False
        self._is_hypotensive: bool = False
        self._is_hypoxic: bool = False

        # --- Developmental ---
        self._volume_growth: float = 0.0

        # --- Configuration ---
        self._sample_points = sample_points

        # --- Statistics ---
        self.total_beats: int = 0
        self.total_heat_transported: float = 0.0
        self.total_o2_delivered: float = 0.0
        self.hypoxia_ticks: int = 0
        self.hypotension_ticks: int = 0
        self.syncope_episodes: int = 0

    # ------------------------------------------------------------------
    # Core tick — called once per brain cycle
    # ------------------------------------------------------------------
    def tick(
        self,
        heart_rate: float = 60.0,
        sympathetic: float = 0.2,
        parasympathetic: float = 0.3,
        hydration: float = 1.0,
        glucose: float = 1.0,
        breaths_this_tick: float = 0.25,
        ram_temperature: float = 0.0,
        cortisol: float = 0.0,
        is_sleeping: bool = False,
    ) -> Dict[str, Any]:
        """
        Advance one cardiovascular cycle.

        Args:
            heart_rate:       Current HR from autonomic (bpm)
            sympathetic:      Sympathetic activation (0~1)
            parasympathetic:  Parasympathetic activation (0~1)
            hydration:        Current hydration from homeostatic (0~1.2)
            glucose:          Current blood glucose (0~1.5)
            breaths_this_tick: Lung breathing rate (for O₂ exchange)
            ram_temperature:  System temperature (0~1)
            cortisol:         Chronic cortisol level (0~1)
            is_sleeping:      Sleep state (reduced metabolic demand)

        Returns:
            Dict with: cerebral_perfusion, o2_delivery, blood_pressure,
                       heat_transported, compensatory_hr_delta, etc.
        """
        # 1. Blood volume = f(hydration)
        self._update_blood_volume(hydration)

        # 2. Blood viscosity = f(blood_volume)
        self._update_viscosity()

        # 3. Vascular resistance = f(sympathetic, parasympathetic, temperature)
        self._update_vascular_resistance(sympathetic, parasympathetic,
                                         ram_temperature, cortisol)

        # 4. Contractility = f(sympathetic) — inotropy
        self._update_contractility(sympathetic)

        # 5. Stroke volume = f(blood_volume, contractility)
        self._update_stroke_volume()

        # 6. Cardiac output = HR × SV
        effective_hr = heart_rate + self._compensatory_hr_delta
        effective_hr = float(np.clip(effective_hr, 40.0, 200.0))
        self._cardiac_output = (effective_hr / 60.0) * self._stroke_volume

        # 7. Blood pressure = CO × R (simplified windkessel)
        self._update_blood_pressure(effective_hr)

        # 8. Cerebral perfusion = MAP / Z_cerebrovascular (with autoregulation)
        self._update_cerebral_perfusion()

        # 9. O₂ transport = perfusion × SpO₂ × hemoglobin
        self._update_oxygen(breaths_this_tick)

        # 10. Heat transport (blood as coolant)
        heat = self._transport_heat(ram_temperature, is_sleeping)

        # 11. Baroreceptor reflex (compensatory HR)
        self._baroreceptor_reflex()

        # 12. Status flags
        self._is_tachycardic = self._compensatory_hr_delta > 10.0
        self._is_hypotensive = self._map_normalized < (MAP_CRITICAL_LOW / BP_NORMALIZE_FACTOR)
        self._is_hypoxic = self._spo2 < SPO2_HYPOXIA_MILD

        # 13. Update statistics
        beats = max(1, int(effective_hr / 60.0))
        self.total_beats += beats
        self.total_heat_transported += heat
        self.total_o2_delivered += self._o2_delivery
        if self._is_hypoxic:
            self.hypoxia_ticks += 1
        if self._is_hypotensive:
            self.hypotension_ticks += 1
        if self._map_normalized < (MAP_SYNCOPE / BP_NORMALIZE_FACTOR):
            self.syncope_episodes += 1

        return {
            "cerebral_perfusion": round(self._cerebral_perfusion, 4),
            "o2_delivery": round(self._o2_delivery, 4),
            "spo2": round(self._spo2, 4),
            "blood_volume": round(self._blood_volume, 4),
            "cardiac_output": round(self._cardiac_output, 4),
            "stroke_volume": round(self._stroke_volume, 4),
            "systolic_bp": round(self._systolic, 1),
            "diastolic_bp": round(self._diastolic, 1),
            "mean_arterial_pressure": round(self._map_normalized * BP_NORMALIZE_FACTOR, 1),
            "map_normalized": round(self._map_normalized, 4),
            "vascular_resistance": round(self._vascular_resistance, 4),
            "blood_viscosity": round(self._blood_viscosity, 4),
            "heat_transported": round(heat, 6),
            "compensatory_hr_delta": round(self._compensatory_hr_delta, 2),
            "is_tachycardic": self._is_tachycardic,
            "is_hypotensive": self._is_hypotensive,
            "is_hypoxic": self._is_hypoxic,
            "glucose_delivery": round(
                self._cerebral_perfusion * min(glucose, 1.5), 4
            ),
        }

    # ------------------------------------------------------------------
    # Proprioception — interoceptive heartbeat signal
    # ------------------------------------------------------------------
    def get_proprioception(self) -> ElectricalSignal:
        """Generate interoceptive heartbeat signal for brain feedback.

        This is the physical basis of 'heartbeat awareness' —
        a key marker of interoceptive sensitivity in anxiety research.
        """
        t = np.linspace(0, 1, self._sample_points, dtype=np.float64)

        # Heart beat waveform: sharp systolic peak + diastolic notch
        systolic_peak = np.exp(-((t - 0.2) ** 2) / 0.005) * self._stroke_volume
        diastolic_notch = np.exp(-((t - 0.45) ** 2) / 0.01) * 0.3 * self._stroke_volume
        waveform = systolic_peak + diastolic_notch

        # Scale by blood pressure
        waveform *= self._map_normalized

        return ElectricalSignal(
            waveform=waveform,
            amplitude=float(self._map_normalized),
            frequency=float(self._cardiac_output * 0.5),
            phase=0.0,
            impedance=CARDIOVASCULAR_IMPEDANCE,
            snr=CARDIOVASCULAR_SNR,
            source="cardiovascular",
            modality="interoception",
        )

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------
    def get_state(self) -> CardiovascularState:
        """Return current cardiovascular state snapshot."""
        return CardiovascularState(
            blood_volume=self._blood_volume,
            stroke_volume=self._stroke_volume,
            cardiac_output=self._cardiac_output,
            systolic_bp=self._systolic,
            diastolic_bp=self._diastolic,
            mean_arterial_pressure=self._map_normalized * BP_NORMALIZE_FACTOR,
            vascular_resistance=self._vascular_resistance,
            cerebral_perfusion=self._cerebral_perfusion,
            spo2=self._spo2,
            o2_delivery=self._o2_delivery,
            blood_viscosity=self._blood_viscosity,
            heat_transported=self._heat_transported,
            is_tachycardic=self._is_tachycardic,
            is_hypotensive=self._is_hypotensive,
            is_hypoxic=self._is_hypoxic,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return lifetime statistics."""
        return {
            "total_beats": self.total_beats,
            "total_heat_transported": round(self.total_heat_transported, 4),
            "total_o2_delivered": round(self.total_o2_delivered, 4),
            "hypoxia_ticks": self.hypoxia_ticks,
            "hypotension_ticks": self.hypotension_ticks,
            "syncope_episodes": self.syncope_episodes,
            "blood_volume": round(self._blood_volume, 4),
            "cerebral_perfusion": round(self._cerebral_perfusion, 4),
            "spo2": round(self._spo2, 4),
            "map_normalized": round(self._map_normalized, 4),
        }

    # ================================================================
    # Internal physics
    # ================================================================

    def _update_blood_volume(self, hydration: float) -> None:
        """
        Blood volume tracks hydration level.

        Plasma is ~92% water. When hydration drops, plasma volume drops first,
        concentrating red blood cells (increasing viscosity) and reducing
        venous return → reduced stroke volume → lower cardiac output.

        blood_volume ≈ hydration × GAIN (with growth factor)
        """
        target_volume = hydration * HYDRATION_TO_VOLUME_GAIN
        # Developmental capacity growth
        max_capacity = NEONATAL_VOLUME_FACTOR + self._volume_growth
        max_capacity = min(max_capacity, BLOOD_VOLUME_SETPOINT)
        target_volume = min(target_volume, max_capacity * BLOOD_VOLUME_MAX)

        # Smooth tracking (blood volume doesn't change instantly)
        alpha = 0.1
        self._blood_volume = float(np.clip(
            self._blood_volume + alpha * (target_volume - self._blood_volume),
            0.05,  # Never truly zero (incompatible with life)
            BLOOD_VOLUME_MAX,
        ))

    def _update_viscosity(self) -> None:
        """
        Blood viscosity = f(blood_volume).

        Less water → more concentrated blood → higher viscosity →
        harder for the heart to pump → reduced flow efficiency.

        viscosity = base + gain × (1 - volume/setpoint)
        """
        deficit = max(0.0, 1.0 - self._blood_volume / BLOOD_VOLUME_SETPOINT)
        self._blood_viscosity = float(np.clip(
            VISCOSITY_NORMAL + VISCOSITY_DEHYDRATION_GAIN * deficit,
            VISCOSITY_NORMAL * 0.9,
            VISCOSITY_MAX,
        ))

    def _update_vascular_resistance(
        self,
        sympathetic: float,
        parasympathetic: float,
        temperature: float,
        cortisol: float,
    ) -> None:
        """
        Vascular resistance = f(autonomic, temperature, cortisol).

        Sympathetic → vasoconstriction → R ↑ → BP ↑
        Parasympathetic → vasodilation → R ↓ → BP ↓
        High temperature → vasodilation → R ↓ (heat dissipation)
        Chronic cortisol → vascular stiffening → R ↑
        Blood viscosity → effective R ↑
        """
        r = VASCULAR_RESISTANCE_BASE
        r += sympathetic * SYMPATHETIC_VASOCONSTRICTION
        r -= parasympathetic * PARASYMPATHETIC_VASODILATION
        r -= temperature * TEMPERATURE_VASODILATION
        r += cortisol * 0.1  # Chronic cortisol → vascular stiffening

        # Viscosity multiplier (thicker blood = more resistance)
        r *= self._blood_viscosity

        self._vascular_resistance = float(np.clip(r, 0.3, 3.0))

    def _update_contractility(self, sympathetic: float) -> None:
        """
        Cardiac contractility (inotropy) = f(sympathetic).

        Sympathetic activation → norepinephrine → stronger contraction.
        This is the Frank-Starling mechanism (simplified).
        """
        self._contractility = float(np.clip(
            CONTRACTILITY_BASE + sympathetic * CONTRACTILITY_SYMPATHETIC_GAIN,
            0.5,
            1.5,
        ))

    def _update_stroke_volume(self) -> None:
        """
        Stroke volume = f(blood_volume, contractility).

        Less blood → less venous return → less filling → lower SV.
        Better contractility → stronger ejection → higher SV.

        SV = base × volume_factor × contractility
        """
        # Volume factor: how full the heart is (Frank-Starling)
        volume_factor = min(1.0, self._blood_volume / BLOOD_VOLUME_SETPOINT)
        volume_factor = max(0.2, volume_factor)  # Even near-empty heart ejects something

        sv = STROKE_VOLUME_BASE * volume_factor * self._contractility
        self._stroke_volume = float(np.clip(sv, STROKE_VOLUME_MIN, 1.2))

    def _update_blood_pressure(self, effective_hr: float) -> None:
        """
        Blood pressure from cardiac output and vascular resistance.

        Simplified windkessel model:
            MAP ≈ CO × R
            Systolic ≈ MAP + pulse_pressure/2
            Diastolic ≈ MAP - pulse_pressure/2
            Pulse pressure ≈ SV × arterial_compliance

        Normalized MAP = MAP_mmHg / BP_NORMALIZE_FACTOR
        """
        # MAP = CO × R (hemodynamic Ohm's law)
        map_raw = self._cardiac_output * self._vascular_resistance

        # Scale to physiological range
        map_mmhg = 40.0 + map_raw * 80.0  # Maps ~0.3CO → 60mmHg, ~0.8CO → 100mmHg
        map_mmhg = float(np.clip(map_mmhg, 30.0, 180.0))

        # Pulse pressure ≈ SV × compliance (simplified)
        pulse_pressure = self._stroke_volume * 40.0  # ~28 mmHg at SV=0.7

        self._systolic = float(np.clip(
            map_mmhg + pulse_pressure * 0.4, 50.0, 200.0
        ))
        self._diastolic = float(np.clip(
            map_mmhg - pulse_pressure * 0.6, 20.0, 130.0
        ))

        self._map_normalized = float(np.clip(
            map_mmhg / BP_NORMALIZE_FACTOR, 0.0, 1.2
        ))

    def _update_cerebral_perfusion(self) -> None:
        """
        Cerebral perfusion with autoregulation.

        The brain has autoregulation: cerebral vessels dilate/constrict
        to maintain constant perfusion across a range of blood pressures.

        Within autoregulation range (MAP 60-150 mmHg): perfusion ≈ constant
        Below autoregulation: perfusion drops linearly with MAP
        Above autoregulation: perfusion rises (hypertensive encephalopathy)

        perfusion = MAP_norm × autoregulation_factor / viscosity_penalty
        """
        map_n = self._map_normalized

        if map_n >= CEREBRAL_AUTOREGULATION_LOW:
            # Within or above autoregulation range → perfusion is maintained
            if map_n <= CEREBRAL_AUTOREGULATION_HIGH:
                # Perfect autoregulation zone
                perfusion = PERFUSION_NORMAL
            else:
                # Above autoregulation → slight excess (hypertensive)
                excess = map_n - CEREBRAL_AUTOREGULATION_HIGH
                perfusion = PERFUSION_NORMAL + excess * 0.3
        else:
            # Below autoregulation → perfusion drops with MAP
            perfusion = PERFUSION_NORMAL * (map_n / CEREBRAL_AUTOREGULATION_LOW)

        # Viscosity penalty (thick blood flows slower)
        viscosity_penalty = 1.0 / self._blood_viscosity

        perfusion *= viscosity_penalty

        self._cerebral_perfusion = float(np.clip(perfusion, 0.0, 1.5))

    def _update_oxygen(self, breaths_this_tick: float) -> None:
        """
        O₂ transport chain: Lung → Blood → Brain.

        SpO₂ = f(breathing rate, lung function)
        O₂ delivery = cerebral_perfusion × SpO₂ × hemoglobin

        Without breathing: SpO₂ drops toward 0.
        Without perfusion: O₂ doesn't reach the brain regardless of SpO₂.
        """
        # SpO₂ recovery from breathing
        if breaths_this_tick > 0.1:
            # Breathing → SpO₂ recovers toward normal
            recovery = breaths_this_tick * 0.15
            self._spo2 += recovery * (SPO2_NORMAL - self._spo2)
        else:
            # Not breathing → SpO₂ drops
            self._spo2 -= 0.02

        self._spo2 = float(np.clip(self._spo2, SPO2_MIN, SPO2_NORMAL))

        # O₂ delivery = perfusion × SpO₂ × hemoglobin efficiency
        self._o2_delivery = float(np.clip(
            self._cerebral_perfusion * self._spo2 * self._hemoglobin,
            0.0,
            O2_DELIVERY_NORMAL * 1.2,
        ))

    def _transport_heat(self, ram_temperature: float, is_sleeping: bool) -> float:
        """
        Blood as thermal transport medium.

        Blood carries heat from the core to the periphery (skin, lungs).
        Higher cardiac output → more heat transported.
        This is why fever + dehydration is dangerous: less blood to carry heat away.
        """
        if ram_temperature <= 0.0:
            return 0.0

        # Heat transport = CO × heat_capacity × temperature_gradient
        transport = (self._cardiac_output * BLOOD_HEAT_CAPACITY *
                     BLOOD_HEAT_TRANSPORT_RATE * ram_temperature)

        if is_sleeping:
            transport *= 0.7  # Reduced peripheral circulation during sleep

        self._heat_transported = float(max(0.0, transport))
        return self._heat_transported

    def _baroreceptor_reflex(self) -> None:
        """
        Baroreceptor reflex: compensatory heart rate adjustment.

        When blood pressure drops (e.g., from dehydration), baroreceptors
        in the carotid sinus detect the drop and signal the brain to
        increase heart rate to maintain cardiac output.

        This is why dehydrated patients/babies have tachycardia:
        the heart beats faster to compensate for lower stroke volume.

        ΔHR = gain × (setpoint - blood_volume) when volume < threshold
        """
        if self._blood_volume < TACHYCARDIA_THRESHOLD:
            deficit = TACHYCARDIA_THRESHOLD - self._blood_volume
            target_delta = deficit * TACHYCARDIA_GAIN
        else:
            target_delta = 0.0

        # Smooth adjustment (baroreceptor response is not instant)
        self._compensatory_hr_delta += BARORECEPTOR_TIME_CONSTANT * (
            target_delta - self._compensatory_hr_delta
        )
        self._compensatory_hr_delta = float(np.clip(
            self._compensatory_hr_delta, 0.0, 60.0  # Max +60 bpm compensation
        ))

    # ------------------------------------------------------------------
    # External interfaces
    # ------------------------------------------------------------------

    def grow(self, motor_movements: int = 0) -> None:
        """
        Developmental blood volume growth.

        Physical activity → angiogenesis + blood volume expansion.
        Same principle as lung: must move to develop.
        """
        if motor_movements > 0:
            remaining = BLOOD_VOLUME_SETPOINT - NEONATAL_VOLUME_FACTOR - self._volume_growth
            if remaining > 0:
                increment = min(motor_movements * VOLUME_GROWTH_RATE, remaining)
                self._volume_growth += increment

    def set_hemoglobin(self, level: float) -> None:
        """
        Set hemoglobin efficiency (for simulating anemia).

        Normal = 1.0, Iron-deficiency anemia ≈ 0.6, Severe anemia ≈ 0.3
        """
        self._hemoglobin = float(np.clip(level, 0.1, 1.2))

    @property
    def cerebral_perfusion(self) -> float:
        """Current cerebral perfusion level (0~1.5)."""
        return self._cerebral_perfusion

    @property
    def o2_delivery(self) -> float:
        """Current O₂ delivery level (0~1.2)."""
        return self._o2_delivery

    @property
    def blood_volume(self) -> float:
        """Current blood volume (0~1.2)."""
        return self._blood_volume

    @property
    def map_normalized(self) -> float:
        """Normalized mean arterial pressure (0~1.2)."""
        return self._map_normalized

    @property
    def spo2(self) -> float:
        """Peripheral oxygen saturation."""
        return self._spo2

    @property
    def compensatory_hr_delta(self) -> float:
        """Compensatory heart rate increase from baroreceptor reflex."""
        return self._compensatory_hr_delta

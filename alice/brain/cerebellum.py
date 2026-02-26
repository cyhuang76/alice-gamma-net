# -*- coding: utf-8 -*-
"""
Alice Smart System — Cerebellum (Cerebellum)

Physics Model: Precision Motor Impedance Calibrator
====================================================

The cerebellum is modeled as a precision impedance calibrator where:
    Climbing fibers = error signal (Γ from motor mismatch)
    Mossy fibers    = context signal (sensory + motor copies)
    Purkinje cells  = impedance correction output (inhibitory)
    Deep nuclei     = calibrated motor drive

Core equations:
    motor_error = intended_movement − actual_movement
    Γ_motor = (Z_intended − Z_actual) / (Z_intended + Z_actual)
    T_motor = 1 − Γ²_motor  (★ C1)
    climbing_error = |Γ_motor| × error_signal_gain
    ΔZ_purkinje = −η × Γ × context × error  (★ C2 Hebbian: cerebellar LTD)
    corrected_output = raw_motor × (1 − purkinje_inhibition)

Cerebellar functions:
    1. Motor timing — precise temporal coordination
    2. Error correction — online adjustment during movement
    3. Motor learning — Hebbian LTD at parallel fiber–Purkinje synapses
    4. Predictive model — forward model of body dynamics
    5. Balance coordination — vestibular–cerebellar loop

Clinical significance:
    Cerebellar ataxia → Γ_motor oscillates → intention tremor
    Dysmetria → distance calibration failure → overshoot/undershoot
    Dysdiadochokinesia → timing error → can't alternate movements
    Alcohol intoxication → Purkinje cell suppression → ataxic gait

"The cerebellum does not move — it calibrates.
 Without it, you can still think about walking.
 You just can't walk straight."

Author: Hsi-Yu Huang (黃璽宇)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import ElectricalSignal

# ============================================================================
# Physical Constants
# ============================================================================

# --- Motor impedance ---
Z_MOTOR_INTENDED = 60.0        # Ω — target motor impedance
Z_MOTOR_ACTUAL = 60.0          # Ω — actual motor impedance (drifts)

# --- Purkinje cells ---
N_PURKINJE = 100               # Simulated Purkinje cell count
PURKINJE_INHIBITION_BASE = 0.5 # Baseline inhibitory output
PURKINJE_LEARNING_RATE = 0.01  # η for cerebellar LTD
PURKINJE_DECAY = 0.999         # Slow forgetting

# --- Climbing fibers (error) ---
CLIMBING_FIBER_GAIN = 2.0      # Error amplification
CLIMBING_FIBER_THRESHOLD = 0.1 # Below this Γ → no learning needed

# --- Motor timing ---
TIMING_PRECISION_BASE = 0.3    # Neonatal timing precision
TIMING_PRECISION_MAX = 0.95    # Adult timing precision
TIMING_IMPROVEMENT_RATE = 0.001 # Per successful correction

# --- Forward model ---
FORWARD_MODEL_ACCURACY = 0.5   # Prediction accuracy (develops)
FORWARD_MODEL_RATE = 0.002     # Learning rate

# --- Motor channels ---
MOTOR_CHANNELS = [
    "reach",           # Arm reaching
    "grasp",           # Hand grasping
    "locomotion",      # Walking
    "speech",          # Articulatory motor
    "eye_movement",    # Saccades / smooth pursuit
    "posture",         # Trunk stability
]

# --- Vestibular integration ---
VESTIBULAR_GAIN = 0.4          # Vestibular → cerebellar input weight

# --- Signal ---
CEREBELLAR_IMPEDANCE = 60.0    # Ω
CEREBELLAR_SNR = 10.0          # dB
CEREBELLAR_SAMPLE_POINTS = 64
CEREBELLAR_FREQUENCY = 20.0    # Hz — fast motor correction

# --- Development ---
NEONATAL_CEREBELLAR_MATURITY = 0.15
CEREBELLAR_MATURATION_RATE = 0.0004


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MotorChannel:
    """A single motor channel with calibration state."""
    name: str
    z_calibration: float = Z_MOTOR_INTENDED
    purkinje_weight: float = PURKINJE_INHIBITION_BASE
    timing_precision: float = TIMING_PRECISION_BASE
    forward_model_accuracy: float = 0.5
    total_corrections: int = 0
    cumulative_error: float = 0.0


# ============================================================================
# Cerebellum
# ============================================================================

class Cerebellum:
    """
    Alice's Cerebellum — precision motor impedance calibrator.

    Replaces simple PID with a physics-based cerebellar model that
    learns motor calibration through Hebbian LTD at Purkinje synapses.

    All signals use ElectricalSignal (★ C3).
    """

    def __init__(self) -> None:
        self._maturity: float = NEONATAL_CEREBELLAR_MATURITY

        # Motor channels
        self._channels: Dict[str, MotorChannel] = {
            name: MotorChannel(name=name)
            for name in MOTOR_CHANNELS
        }

        # Global state
        self._gamma_motor: float = 0.0
        self._transmission_motor: float = 1.0
        self._mean_timing_precision: float = TIMING_PRECISION_BASE
        self._balance_confidence: float = 0.3  # Vestibular-cerebellar calibration

        # Statistics
        self._tick_count: int = 0
        self._total_corrections: int = 0

    # ------------------------------------------------------------------
    # Motor correction
    # ------------------------------------------------------------------

    def correct_movement(
        self,
        channel: str = "reach",
        intended: float = 0.5,
        actual: float = 0.5,
        vestibular_input: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Perform cerebellar motor correction.

        Args:
            channel: Motor channel name
            intended: Intended motor output (normalized)
            actual: Actual sensory feedback (normalized)
            vestibular_input: Vestibular balance signal
        """
        if channel not in self._channels:
            channel = "reach"

        ch = self._channels[channel]
        intent = float(np.clip(intended, 0, 1))
        act = float(np.clip(actual, 0, 1))
        vest = float(np.clip(vestibular_input, -1, 1))

        # Motor error → Γ_motor
        z_intended = Z_MOTOR_INTENDED * (1.0 + intent * 0.5)
        z_actual = Z_MOTOR_ACTUAL * (1.0 + act * 0.5)
        gamma_ch = abs(z_intended - z_actual) / (z_intended + z_actual)
        transmission_ch = 1.0 - gamma_ch ** 2  # ★ C1

        # Climbing fiber error signal
        if gamma_ch > CLIMBING_FIBER_THRESHOLD:
            climbing_error = gamma_ch * CLIMBING_FIBER_GAIN

            # ★ C2 Hebbian LTD: ΔZ = −η × Γ × context × error
            delta_z = -PURKINJE_LEARNING_RATE * gamma_ch * act * climbing_error
            ch.z_calibration = float(np.clip(
                ch.z_calibration + delta_z * 10.0,
                Z_MOTOR_INTENDED * 0.5,
                Z_MOTOR_INTENDED * 2.0,
            ))

            # Purkinje weight update
            ch.purkinje_weight = float(np.clip(
                ch.purkinje_weight + climbing_error * 0.01,
                0.1, 1.0
            ))

            ch.total_corrections += 1
            self._total_corrections += 1
        else:
            climbing_error = 0.0

        # Timing precision improves with practice
        if gamma_ch < 0.2:
            ch.timing_precision = min(TIMING_PRECISION_MAX,
                                      ch.timing_precision + TIMING_IMPROVEMENT_RATE)

        # Forward model update
        prediction_error = abs(intent - act)
        ch.forward_model_accuracy = min(1.0,
            ch.forward_model_accuracy + FORWARD_MODEL_RATE * (1.0 - prediction_error))

        ch.cumulative_error += gamma_ch

        # Corrected output: apply Purkinje inhibition to raw motor
        corrected = intent * (1.0 - ch.purkinje_weight * gamma_ch) * self._maturity
        corrected += vest * VESTIBULAR_GAIN * self._maturity  # Balance correction
        corrected = float(np.clip(corrected, 0, 1))

        return {
            "channel": channel,
            "gamma_motor": round(gamma_ch, 4),
            "transmission_motor": round(transmission_ch, 4),
            "corrected_output": round(corrected, 4),
            "climbing_error": round(climbing_error, 4),
            "timing_precision": round(ch.timing_precision, 4),
            "forward_model_accuracy": round(ch.forward_model_accuracy, 4),
        }

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, vestibular_stability: float = 0.5) -> Dict[str, Any]:
        """
        Advance cerebellum by one tick.

        Args:
            vestibular_stability: Vestibular system stability (0–1)
        """
        self._tick_count += 1
        self._maturity = min(1.0, self._maturity + CEREBELLAR_MATURATION_RATE)

        # Balance confidence from vestibular integration
        self._balance_confidence = float(np.clip(
            self._balance_confidence * 0.99 + vestibular_stability * 0.01 * self._maturity,
            0, 1
        ))

        # Mean timing and Γ across channels
        gammas = []
        timings = []
        for ch in self._channels.values():
            gamma_ch = abs(ch.z_calibration - Z_MOTOR_INTENDED) / (
                ch.z_calibration + Z_MOTOR_INTENDED
            )
            gammas.append(gamma_ch)
            timings.append(ch.timing_precision)
            # Slow decay toward baseline
            ch.z_calibration += (Z_MOTOR_INTENDED - ch.z_calibration) * 0.001

        self._gamma_motor = float(np.mean(gammas))
        self._transmission_motor = 1.0 - self._gamma_motor ** 2  # ★ C1
        self._mean_timing_precision = float(np.mean(timings))

        return {
            "gamma_motor": round(self._gamma_motor, 4),
            "transmission_motor": round(self._transmission_motor, 4),
            "mean_timing_precision": round(self._mean_timing_precision, 4),
            "balance_confidence": round(self._balance_confidence, 4),
            "maturity": round(self._maturity, 4),
        }

    # ------------------------------------------------------------------
    # ElectricalSignal (★ C3)
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate cerebellar motor correction signal."""
        amplitude = float(np.clip(
            0.2 + (1.0 - self._mean_timing_precision) * 0.5, 0.05, 1.0
        ))
        freq = CEREBELLAR_FREQUENCY * self._maturity

        t = np.linspace(0, 1, CEREBELLAR_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=CEREBELLAR_IMPEDANCE,
            snr=CEREBELLAR_SNR * self._myelination_factor,
            source="cerebellum",
            modality="motor",
        )

    @property
    def _myelination_factor(self) -> float:
        """Approximate myelination from maturity."""
        return 0.3 + 0.7 * self._maturity

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get cerebellum statistics."""
        return {
            "gamma_motor": round(self._gamma_motor, 4),
            "transmission_motor": round(self._transmission_motor, 4),
            "mean_timing_precision": round(self._mean_timing_precision, 4),
            "balance_confidence": round(self._balance_confidence, 4),
            "maturity": round(self._maturity, 4),
            "channel_stats": {
                name: {
                    "timing": round(ch.timing_precision, 4),
                    "forward_model": round(ch.forward_model_accuracy, 4),
                    "corrections": ch.total_corrections,
                }
                for name, ch in self._channels.items()
            },
            "total_corrections": self._total_corrections,
            "tick_count": self._tick_count,
        }

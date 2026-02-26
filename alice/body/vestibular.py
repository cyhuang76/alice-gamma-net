# -*- coding: utf-8 -*-
"""
Alice's Vestibular System — Balance & Spatial Orientation

Physics Model: Inertial Impedance Sensor
==========================================

The vestibular system is modeled as a coupled pair of inertial sensors:

1. Semicircular Canals (angular acceleration):
   Three orthogonal fluid-filled loops → angular velocity detection
   Z_canal = ρ·L/(π·r²) + i·ω·(ρ·L)/(π·r²)   (fluid impedance in tube)
   Endolymph displacement → hair cell deflection → neural signal
   "The canals are LC resonators tuned to the spectrum of head rotation."

2. Otolith Organs (linear acceleration + gravity):
   Utricle (horizontal) and Saccule (vertical)
   Z_otolith = m·s / A  (mass-spring impedance)
   Otoconia crystals provide inertial mass against hair cell bed
   "The otoliths are accelerometers. They measure gravity and linear motion."

Integration:
   The vestibular system provides the REFERENCE FRAME for all other senses.
   Γ_vestibular = |Z_actual - Z_expected| / (Z_actual + Z_expected)
   When predicted motion ≠ sensed motion → Γ rises → motion sickness

   "Motion sickness is an impedance mismatch between your eyes and your inner ear."

Core equations:
   Angular: ω_detected = (1/τ_canal) ∫ α(t) dt    (α = angular accel)
   Linear: a_detected = F_otolith / m_otoconia
   Conflict: Γ_vestibular = |prediction - measurement| / (prediction + measurement)

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

# --- Semicircular Canals (3-axis angular velocity) ---
CANAL_TIME_CONSTANT = 6.0      # seconds — high-pass cutoff (~0.03 Hz)
CANAL_IMPEDANCE = 120.0        # Ω — fluid impedance
CANAL_SENSITIVITY = 0.5        # rad/s minimum detectable rotation
CANAL_AXES = 3                 # lateral, posterior, anterior (near-orthogonal)

# --- Otolith Organs ---
OTOLITH_IMPEDANCE = 200.0      # Ω — higher (mass-spring system)
OTOLITH_SENSITIVITY = 0.01     # g minimum detectable acceleration
GRAVITY = 9.81                 # m/s² — always present as DC component

# --- Vestibulo-Ocular Reflex (VOR) ---
VOR_GAIN = 1.0                 # Eye counter-rotation / head rotation (ideally 1.0)
VOR_LATENCY = 0.015            # seconds — fastest reflex in the body (~15ms)

# --- Motion Sickness ---
SICKNESS_THRESHOLD = 0.35      # Γ_conflict above this → nausea onset
SICKNESS_BUILDUP_RATE = 0.02   # How fast sickness accumulates
SICKNESS_RECOVERY_RATE = 0.005 # How fast sickness resolves
SICKNESS_MAX = 1.0             # Maximum sickness level

# --- Signal properties ---
VESTIBULAR_SNR = 15.0          # dB — vestibular is quite precise
VESTIBULAR_SAMPLE_POINTS = 64


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MotionState:
    """Current motion state (what the body is actually doing)."""
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s [roll, pitch, yaw]
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, GRAVITY]))  # m/s²
    head_tilt: np.ndarray = field(default_factory=lambda: np.zeros(3))  # radians [roll, pitch, yaw]


@dataclass
class BalanceState:
    """Computed balance assessment."""
    stable: bool = True
    sway: float = 0.0           # 0~1, postural sway magnitude
    vor_error: float = 0.0      # VOR tracking error
    motion_sickness: float = 0.0
    gamma_conflict: float = 0.0  # Γ from sensory conflict


# ============================================================================
# VestibularSystem
# ============================================================================

class VestibularSystem:
    """
    Alice's Vestibular System — inertial impedance sensor for balance.

    Provides:
    1. Angular velocity detection (3 axes, via semicircular canal model)
    2. Linear acceleration + gravity detection (otolith model)
    3. Vestibulo-ocular reflex (VOR) signals
    4. Sensory conflict detection (motion sickness predictor)
    5. Balance assessment

    "The vestibular system is the hidden sixth sense.
     Without it, you cannot stand, walk, or read while moving."
    """

    def __init__(self) -> None:
        # --- Canal state (high-pass filtered angular velocity) ---
        self._canal_state: np.ndarray = np.zeros(CANAL_AXES)  # Integrated signal
        self._angular_velocity: np.ndarray = np.zeros(CANAL_AXES)

        # --- Otolith state ---
        self._otolith_state: np.ndarray = np.array([0.0, 0.0, GRAVITY])  # Linear accel
        self._head_tilt: np.ndarray = np.zeros(3)

        # --- Motion prediction (for conflict detection) ---
        self._predicted_motion: np.ndarray = np.zeros(6)  # [angular(3), linear(3)]
        self._actual_motion: np.ndarray = np.zeros(6)

        # --- Motion sickness ---
        self._sickness_level: float = 0.0
        self._gamma_conflict: float = 0.0
        self._transmission: float = 1.0  # ★ T = 1 − Γ²

        # --- VOR output ---
        self._vor_command: np.ndarray = np.zeros(3)  # Eye counter-rotation

        # --- Balance ---
        self._balance = BalanceState()

        # Statistics
        self._tick_count: int = 0

    # ------------------------------------------------------------------
    # Sense motion
    # ------------------------------------------------------------------

    def sense_motion(self, motion: MotionState) -> Dict[str, Any]:
        """
        Process vestibular stimulation from body motion.

        Args:
            motion: Current body motion state (angular vel, linear accel, tilt).

        Returns:
            Dict with detected motion, conflict, balance, VOR commands.
        """
        # --- Semicircular Canals (angular velocity) ---
        # High-pass filter: canal_state decays with time constant τ
        alpha_canal = 1.0 / (1.0 + CANAL_TIME_CONSTANT)
        self._canal_state = (
            alpha_canal * self._canal_state
            + (1.0 - alpha_canal) * motion.angular_velocity
        )
        self._angular_velocity = motion.angular_velocity

        # --- Otolith Organs (linear acceleration + gravity) ---
        self._otolith_state = motion.linear_acceleration.copy()
        self._head_tilt = motion.head_tilt.copy()

        # --- Compute sensory conflict Γ (impedance-based) ---
        actual = np.concatenate([self._angular_velocity, self._otolith_state])
        self._actual_motion = actual

        # ★ Impedance-based Γ: Z_pred, Z_actual from motion magnitudes
        #   Z = Z_canal × (1 + |motion|/scale) → impedance grows with motion intensity
        scale = 10.0  # normalisation scale for motion magnitude
        z_pred = CANAL_IMPEDANCE * (1.0 + np.linalg.norm(self._predicted_motion) / scale)
        z_actual = CANAL_IMPEDANCE * (1.0 + np.linalg.norm(actual) / scale)
        self._gamma_conflict = abs(z_pred - z_actual) / (z_pred + z_actual + 1e-12)
        self._transmission = 1.0 - self._gamma_conflict ** 2  # ★ T = 1 − Γ²

        # ★ Hebbian prediction update: Δpred = α × T × (actual − predicted)
        #   Only transmitted signal informs the predictor
        alpha_pred = 0.2
        self._predicted_motion += alpha_pred * self._transmission * (actual - self._predicted_motion)

        # --- Motion sickness ---
        if self._gamma_conflict > SICKNESS_THRESHOLD:
            self._sickness_level = min(
                SICKNESS_MAX,
                self._sickness_level + SICKNESS_BUILDUP_RATE * (self._gamma_conflict - SICKNESS_THRESHOLD),
            )
        else:
            self._sickness_level = max(
                0.0,
                self._sickness_level - SICKNESS_RECOVERY_RATE,
            )

        # --- Vestibulo-Ocular Reflex (VOR) ---
        self._vor_command = -VOR_GAIN * self._angular_velocity  # Counter-rotate eyes

        # --- Balance assessment ---
        sway = float(np.linalg.norm(self._head_tilt[:2]))  # Roll-pitch sway
        gravity_deviation = abs(self._otolith_state[2] - GRAVITY) / GRAVITY
        stable = sway < 0.3 and gravity_deviation < 0.2

        self._balance = BalanceState(
            stable=stable,
            sway=round(sway, 4),
            vor_error=round(float(np.linalg.norm(self._vor_command + self._angular_velocity)), 4),
            motion_sickness=round(self._sickness_level, 4),
            gamma_conflict=round(self._gamma_conflict, 4),
        )

        return {
            "angular_velocity_detected": self._angular_velocity.tolist(),
            "linear_accel_detected": self._otolith_state.tolist(),
            "gamma_conflict": round(self._gamma_conflict, 4),
            "transmission": round(self._transmission, 4),
            "motion_sickness": round(self._sickness_level, 4),
            "balance_stable": stable,
            "vor_command": self._vor_command.tolist(),
        }

    # ------------------------------------------------------------------
    # Generate ElectricalSignal
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate vestibular ElectricalSignal."""
        activity = float(np.linalg.norm(self._angular_velocity)) + float(np.linalg.norm(self._otolith_state - np.array([0, 0, GRAVITY])))
        amplitude = float(np.clip(activity / 10.0, 0.05, 1.0))
        freq = 2.0 + activity * 5.0  # Low freq (vestibular is slow)

        t = np.linspace(0, 1, VESTIBULAR_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)
        # Add otolith DC component for gravity
        waveform = waveform + 0.1 * (self._otolith_state[2] / GRAVITY)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=OTOLITH_IMPEDANCE,
            snr=VESTIBULAR_SNR,
            source="vestibular",
            modality="proprioceptive",
        )

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """Advance vestibular state by one tick."""
        self._tick_count += 1

        # Canal adaptation (high-pass: constant rotation → signal decay)
        self._canal_state *= (1.0 - 1.0 / CANAL_TIME_CONSTANT * 0.1)

        # Sickness recovery at rest
        if float(np.linalg.norm(self._angular_velocity)) < 0.01:
            self._sickness_level = max(0.0, self._sickness_level - SICKNESS_RECOVERY_RATE)

        return {
            "gamma_conflict": round(self._gamma_conflict, 4),
            "motion_sickness": round(self._sickness_level, 4),
            "balance_stable": self._balance.stable,
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "angular_velocity": self._angular_velocity.tolist(),
            "linear_acceleration": self._otolith_state.tolist(),
            "gamma_conflict": round(self._gamma_conflict, 4),
            "transmission": round(self._transmission, 4),
            "motion_sickness": round(self._sickness_level, 4),
            "balance_stable": self._balance.stable,
            "balance_sway": self._balance.sway,
            "vor_error": self._balance.vor_error,
            "tick_count": self._tick_count,
        }

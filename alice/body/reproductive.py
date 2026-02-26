# -*- coding: utf-8 -*-
"""
Alice Smart System — Reproductive System (ReproductiveSystem)

Physics Model: Hormonal Oscillator Network
===========================================

The reproductive system is modeled as a coupled hormonal oscillator where:
    Hypothalamus = master oscillator (GnRH pulse generator)
    Pituitary    = frequency divider (LH/FSH)
    Gonads       = output stage (sex hormone production)
    Feedback     = negative/positive feedback loops

Core equations:
    GnRH_pulse = oscillator_amplitude × sin(2π × f_pulse × t)
    LH = GnRH × pituitary_gain × (1 − sex_hormone_feedback)
    sex_hormone = LH × gonadal_response × maturity
    T_reproductive = 1 − Γ²_repro  (★ C1)
    Γ_repro = (Z_hormone − Z_receptor) / (Z_hormone + Z_receptor)

Developmental trajectory:
    Neonatal  → quiescent (maternal hormone withdrawal)
    Childhood → minimal activity (GnRH suppressed)
    Puberty   → GnRH pulses increase → HPG axis activates

Note: This module models the physics of hormonal oscillation and
developmental biology only. It does NOT simulate reproduction itself.
Ethics boundary: developmental endocrinology modeling only.

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

# --- GnRH oscillator ---
GNRH_BASE_FREQ = 0.01          # Hz — very slow pulse (60–120 min intervals)
GNRH_AMPLITUDE_PREPUBERTAL = 0.1
GNRH_AMPLITUDE_PUBERTAL = 0.6
GNRH_AMPLITUDE_ADULT = 0.8

# --- Pituitary ---
LH_GAIN = 0.5                  # GnRH → LH conversion
FSH_GAIN = 0.4                 # GnRH → FSH conversion

# --- Gonadal output ---
SEX_HORMONE_BASE = 0.1         # Prepubertal baseline
SEX_HORMONE_MAX = 1.0
GONADAL_RESPONSE = 0.3         # LH → sex hormone gain

# --- Feedback ---
NEGATIVE_FEEDBACK_GAIN = 0.4   # Sex hormones → suppress GnRH
POSITIVE_FEEDBACK_THRESHOLD = 0.7  # Above this → positive feedback (ovulation trigger)

# --- Development ---
PUBERTY_ONSET_MATURITY = 0.4   # Maturity threshold for puberty onset
NEONATAL_REPRO_MATURITY = 0.05
REPRO_MATURATION_RATE = 0.0001 # Very slow maturation

# --- Impedance ---
REPRO_Z = 95.0                 # Ω
Z_RECEPTOR_REPRO = 95.0
REPRO_SNR = 4.0                # dB
REPRO_SAMPLE_POINTS = 48
REPRO_FREQUENCY = 0.01         # Hz

# ============================================================================
# ReproductiveSystem
# ============================================================================

class ReproductiveSystem:
    """
    Alice's Reproductive System — hormonal oscillator network.

    Models the HPG axis as a coupled oscillator with developmental
    maturation. Inactive in infancy, activates at puberty.

    Ethics: developmental endocrinology modeling only.
    All signals use ElectricalSignal (★ C3).
    """

    def __init__(self) -> None:
        self._maturity: float = NEONATAL_REPRO_MATURITY

        # HPG axis
        self._gnrh_amplitude: float = GNRH_AMPLITUDE_PREPUBERTAL
        self._lh: float = 0.0
        self._fsh: float = 0.0
        self._sex_hormone: float = SEX_HORMONE_BASE

        # Phase
        self._phase: float = 0.0  # Oscillator phase

        # Γ
        self._gamma_repro: float = 0.0
        self._transmission_repro: float = 1.0

        # Statistics
        self._tick_count: int = 0
        self._puberty_reached: bool = False

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, stress: float = 0.0, nutrition: float = 0.7,
             body_fat: float = 0.5) -> Dict[str, Any]:
        """
        Advance reproductive system by one tick.

        Args:
            stress: Stress level (high stress suppresses HPG)
            nutrition: Nutritional status (malnutrition delays puberty)
            body_fat: Body fat percentage (leptin → puberty trigger)
        """
        self._tick_count += 1
        self._maturity = min(1.0, self._maturity + REPRO_MATURATION_RATE)

        stress_in = float(np.clip(stress, 0, 1))
        nutrition_in = float(np.clip(nutrition, 0, 1))

        # Puberty gate: maturity + nutrition + body fat
        if self._maturity >= PUBERTY_ONSET_MATURITY and nutrition_in > 0.5:
            self._puberty_reached = True
            self._gnrh_amplitude = min(
                GNRH_AMPLITUDE_ADULT,
                self._gnrh_amplitude + 0.001 * nutrition_in
            )
        else:
            self._gnrh_amplitude = GNRH_AMPLITUDE_PREPUBERTAL

        # Stress suppression
        effective_gnrh = self._gnrh_amplitude * (1.0 - stress_in * 0.5)

        # Phase advance
        self._phase += 2 * math.pi * GNRH_BASE_FREQ
        gnrh_signal = effective_gnrh * max(0, math.sin(self._phase))

        # LH / FSH
        self._lh = gnrh_signal * LH_GAIN * self._maturity
        self._fsh = gnrh_signal * FSH_GAIN * self._maturity

        # Sex hormone production
        production = self._lh * GONADAL_RESPONSE * self._maturity
        # Negative feedback
        feedback = self._sex_hormone * NEGATIVE_FEEDBACK_GAIN
        self._sex_hormone = float(np.clip(
            self._sex_hormone + production - feedback * 0.01,
            SEX_HORMONE_BASE, SEX_HORMONE_MAX
        ))

        # Γ_repro
        z_h = REPRO_Z * (1.0 + abs(self._sex_hormone - 0.5))
        self._gamma_repro = abs(z_h - Z_RECEPTOR_REPRO) / (z_h + Z_RECEPTOR_REPRO)
        self._transmission_repro = 1.0 - self._gamma_repro ** 2  # ★ C1

        return {
            "gamma_repro": round(self._gamma_repro, 4),
            "transmission_repro": round(self._transmission_repro, 4),
            "gnrh_amplitude": round(self._gnrh_amplitude, 4),
            "lh": round(self._lh, 4),
            "fsh": round(self._fsh, 4),
            "sex_hormone": round(self._sex_hormone, 4),
            "puberty_reached": self._puberty_reached,
            "maturity": round(self._maturity, 4),
        }

    # ------------------------------------------------------------------
    # ElectricalSignal (★ C3)
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate reproductive hormonal signal."""
        amplitude = float(np.clip(self._sex_hormone * 0.5, 0.05, 1.0))
        freq = REPRO_FREQUENCY + self._sex_hormone * 0.1

        t = np.linspace(0, 1, REPRO_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=self._phase % (2 * math.pi),
            impedance=REPRO_Z,
            snr=REPRO_SNR,
            source="reproductive",
            modality="interoceptive",
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get reproductive system statistics."""
        return {
            "gamma_repro": round(self._gamma_repro, 4),
            "transmission_repro": round(self._transmission_repro, 4),
            "gnrh_amplitude": round(self._gnrh_amplitude, 4),
            "lh": round(self._lh, 4),
            "fsh": round(self._fsh, 4),
            "sex_hormone": round(self._sex_hormone, 4),
            "puberty_reached": self._puberty_reached,
            "maturity": round(self._maturity, 4),
            "tick_count": self._tick_count,
        }

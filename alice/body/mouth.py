# -*- coding: utf-8 -*-
"""
Alice's Mouth — Inverse Engineering (Vocal Output)

Physics:
  "The mouth is the audio version of the hand. The hand uses PID to control
   muscles for movement; the mouth uses PID to control vocal cord tension
   for vocalization."

  Brain → Motor cortex → Vocal cord tension → Air vibration → Sound wave
  But this is not perfect. Vocal cords have mass and inertia.

Pipeline (Source-Filter Model):
  1. Lungs (air source)         — Respiratory pressure → Airflow
  2. Vocal cords (vibration)    — Airflow + tension → Fundamental frequency vibration
  3. Vocal tract (resonant cavity) — Resonance shaping → Formants
  4. Lips/nasal cavity (radiation) — Radiate into air → Sound wave

Anxiety effects:
  - ram_temperature < 0.3  → Voice stable and clear
  - ram_temperature 0.3~0.6 → Voice slightly trembling (nervous)
  - ram_temperature 0.6~0.8 → Voice shaking (fear)
  - ram_temperature > 0.8  → Voice out of control (scream/aphasia)

  "Voice trembles when nervous — not psychology, but physics.
   Vocal cord tension becomes unstable due to anxiety,
   just like the hand trembles due to anxiety."
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import BrainWaveBand, ElectricalSignal


# ============================================================================
# Physical constants of the mouth
# ============================================================================

# Vocal cord properties
VOCAL_FOLD_MASS = 0.1           # Equivalent mass (g)
VOCAL_FOLD_REST_FREQ = 120.0    # Resting fundamental frequency Hz (adult average)
VOCAL_FOLD_MIN_FREQ = 60.0      # Minimum fundamental frequency Hz
VOCAL_FOLD_MAX_FREQ = 500.0     # Maximum fundamental frequency Hz

# Brainwave frequency range (mapping to brain)
BRAINWAVE_FREQ_MIN = 0.5        # Hz (δ lower bound)
BRAINWAVE_FREQ_MAX = 100.0      # Hz (γ upper bound)

# Lung air source
LUNG_PRESSURE_DEFAULT = 0.5     # Default expiratory pressure (normalized 0~1)
LUNG_PRESSURE_MAX = 1.0         # Maximum expiratory pressure

# PID controller (pitch tracking)
# Note: control step is 0.1, gains need to match step for stable convergence
PITCH_KP = 3.0     # Pitch proportional gain
PITCH_KI = 0.05    # Pitch integral gain
PITCH_KD = 0.8     # Pitch derivative gain

# Formants (simplified vocal tract model)
# Three formants are sufficient to distinguish main vowels
FORMANT_BANDWIDTHS = [80.0, 100.0, 120.0]  # Hz

# Tremor (anxiety effect)
VOCAL_TREMOR_FREQ = 6.0         # Vocal cord tremor frequency Hz (physiological)
VOCAL_TREMOR_BASE_AMP = 0.01    # Base tremor amplitude

# Output
OUTPUT_SAMPLE_POINTS = 256      # Feedback signal sample points (fixed length for brain)
OUTPUT_SAMPLE_RATE = 16000      # Sound wave sample rate Hz (must match cochlea)
OUTPUT_DURATION = 0.02          # Default waveform duration (seconds) — one fundamental period
MOUTH_IMPEDANCE = 50.0          # Ω
MOUTH_SNR = 12.0                # dB


# ============================================================================
# Default formant table (physical fingerprint of vowels)
# ============================================================================

# {vowel: (F1, F2, F3)} — Standard formant frequencies Hz
VOWEL_FORMANTS: Dict[str, Tuple[float, float, float]] = {
    "a": (730.0, 1090.0, 2440.0),   # /a/ widest opening
    "i": (270.0, 2290.0, 3010.0),   # /i/ highest front tongue
    "u": (300.0, 870.0, 2240.0),    # /u/ most rounded lips
    "e": (530.0, 1840.0, 2480.0),   # /e/ half-open front
    "o": (570.0, 840.0, 2410.0),    # /o/ half-open back rounded
}


# ============================================================================
# Main class
# ============================================================================


class AliceMouth:
    """
    Alice's Mouth — Inverse engineering motor organ (vocalization)

    Motor intent → PID control → Vocal cord tension → Airflow vibration → Vocal tract resonance → Sound wave

    Does not produce "semantics". Only produces physical sound waves.
    Just like the hand doesn't understand the meaning of "grasping" — the hand only moves to coordinates.
    The mouth doesn't understand "speaking" — the mouth only produces vibrations at specific frequencies.
    """

    def __init__(
        self,
        base_pitch: float = VOCAL_FOLD_REST_FREQ,
        sample_points: int = OUTPUT_SAMPLE_POINTS,
        sample_rate: int = OUTPUT_SAMPLE_RATE,
        duration: float = OUTPUT_DURATION,
    ):
        """
        Args:
            base_pitch:    Base pitch Hz (male ~120, female ~220)
            sample_points: Feedback signal sample points (fixed length for brain)
            sample_rate:   Sound wave sample rate Hz (must match cochlea)
            duration:      Waveform duration (seconds)
        """
        self.base_pitch = float(np.clip(base_pitch, VOCAL_FOLD_MIN_FREQ, VOCAL_FOLD_MAX_FREQ))
        self.sample_points = sample_points
        self.sample_rate = sample_rate
        self.duration = duration
        # Sound wave sample count = duration × sample_rate
        self._wave_samples = int(self.duration * self.sample_rate)

        # PID state (pitch tracking)
        self._pitch_error_integral = 0.0
        self._pitch_prev_error = 0.0
        self._current_pitch = self.base_pitch

        # Lung state
        self._lung_pressure = LUNG_PRESSURE_DEFAULT

        # Statistics
        self.total_utterances: int = 0
        self.total_phonemes: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def speak(
        self,
        target_pitch: float,
        volume: float = 0.5,
        formants: Optional[Tuple[float, float, float]] = None,
        duration_steps: int = 1,
        ram_temperature: float = 0.0,
        motor_signal: Optional[ElectricalSignal] = None,
    ) -> Dict[str, Any]:
        """
        Vocalize — Produce a speech segment

        Args:
            target_pitch:  Target pitch Hz
            volume:        Volume 0.0~1.0 (= lung pressure)
            formants:      (F1, F2, F3) formant frequencies Hz, or None (pure tone)
            duration_steps: Duration steps (each step = one vibration cycle)
            ram_temperature: Anxiety temperature → vocal cord tremor
            motor_signal:   Motor cortex signal (can affect force gain)

        Returns:
            {
                "waveform": np.ndarray — sound wave
                "final_pitch": float — final pitch
                "pitch_error": float — pitch error
                "tremor_intensity": float — tremor intensity
                "volume": float — actual volume
                "steps": int — step count
                "signal": ElectricalSignal — feedback signal
            }
        """
        self.total_utterances += 1

        target_pitch = float(np.clip(target_pitch, VOCAL_FOLD_MIN_FREQ, VOCAL_FOLD_MAX_FREQ))
        volume = float(np.clip(volume, 0.0, 1.0))
        temp = float(np.clip(ram_temperature, 0.0, 1.0))

        # Motor signal gain
        motor_gain = 1.0
        if motor_signal is not None:
            motor_gain = 0.8 + 0.4 * min(motor_signal.amplitude, 1.0)

        # PID pitch tracking
        for _ in range(duration_steps):
            self._pitch_pid_step(target_pitch, temp)

        # Lung pressure
        self._lung_pressure = volume * LUNG_PRESSURE_MAX * motor_gain

        # Vocal cord vibration: fundamental + harmonics (simplified glottal pulse)
        waveform = self._glottal_source(self._current_pitch, temp)

        # Vocal tract resonance (formant filtering)
        if formants is not None:
            waveform = self._vocal_tract_filter(waveform, formants)

        # Radiation (lips) + volume
        waveform = waveform * self._lung_pressure

        # Tremor intensity
        tremor_intensity = VOCAL_TREMOR_BASE_AMP * (1.0 + 20.0 * temp ** 2)

        # Feedback signal (proprioception — hearing one's own voice)
        feedback = self._vocal_feedback(waveform)

        return {
            "waveform": waveform,
            "final_pitch": self._current_pitch,
            "pitch_error": abs(self._current_pitch - target_pitch),
            "tremor_intensity": tremor_intensity,
            "volume": self._lung_pressure,
            "steps": duration_steps,
            "signal": feedback,
        }

    # ------------------------------------------------------------------
    def say_vowel(
        self,
        vowel: str,
        pitch: Optional[float] = None,
        volume: float = 0.5,
        ram_temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Say a vowel — using the default formant table

        Args:
            vowel: "a", "i", "u", "e", "o"
            pitch: Target pitch Hz, None uses base_pitch
            volume: Volume 0~1
            ram_temperature: Anxiety temperature
        """
        self.total_phonemes += 1
        formants = VOWEL_FORMANTS.get(vowel.lower(), VOWEL_FORMANTS["a"])
        target = pitch if pitch is not None else self.base_pitch
        return self.speak(
            target_pitch=target,
            volume=volume,
            formants=formants,
            duration_steps=3,
            ram_temperature=ram_temperature,
        )

    # ------------------------------------------------------------------
    def get_proprioception(self) -> ElectricalSignal:
        """
        Vocal cord proprioception — current vocal cord state

        Returns:
            ElectricalSignal: source="mouth", modality="proprioception"
        """
        t = np.linspace(0, 1, self.sample_points, endpoint=False)
        waveform = np.sin(2 * np.pi * self._current_pitch * t)
        amp = self._lung_pressure

        freq_mapped = BRAINWAVE_FREQ_MIN + (
            self._current_pitch - VOCAL_FOLD_MIN_FREQ
        ) / (VOCAL_FOLD_MAX_FREQ - VOCAL_FOLD_MIN_FREQ) * (
            BRAINWAVE_FREQ_MAX - BRAINWAVE_FREQ_MIN
        )
        freq_mapped = float(np.clip(freq_mapped, BRAINWAVE_FREQ_MIN, BRAINWAVE_FREQ_MAX))

        return ElectricalSignal(
            waveform=waveform * amp,
            amplitude=amp,
            frequency=freq_mapped,
            phase=0.0,
            impedance=MOUTH_IMPEDANCE,
            snr=MOUTH_SNR,
            source="mouth",
            modality="proprioception",
        )

    # ------------------------------------------------------------------
    # Internal physics pipeline
    # ------------------------------------------------------------------

    def _pitch_pid_step(self, target: float, temperature: float):
        """
        PID pitch controller

        Same principle as the hand's PID:
        - Error = target pitch - current pitch
        - Control output = Kp*e + Ki*∫e + Kd*de/dt
        - Plus temperature-induced tremor noise
        """
        error = target - self._current_pitch
        self._pitch_error_integral += error
        # Anti-integrator windup
        self._pitch_error_integral = float(np.clip(
            self._pitch_error_integral, -50.0, 50.0
        ))
        derivative = error - self._pitch_prev_error
        self._pitch_prev_error = error

        # PID output
        control = (
            PITCH_KP * error +
            PITCH_KI * self._pitch_error_integral +
            PITCH_KD * derivative
        )

        # Anxiety tremor
        tremor_amp = VOCAL_TREMOR_BASE_AMP * (1.0 + 20.0 * temperature ** 2)
        tremor = tremor_amp * np.sin(
            2 * np.pi * VOCAL_TREMOR_FREQ * time.time()
        ) * self._current_pitch

        # Update pitch
        new_pitch = self._current_pitch + control * 0.1 + tremor
        self._current_pitch = float(np.clip(
            new_pitch, VOCAL_FOLD_MIN_FREQ, VOCAL_FOLD_MAX_FREQ
        ))

    # ------------------------------------------------------------------
    def _glottal_source(self, pitch: float, temperature: float) -> np.ndarray:
        """
        Glottal pulse — raw waveform produced by vocal cord vibration

        Simplified model: fundamental frequency sine + multi-order harmonics (1/n weighting)
        Real vocal cords produce an approximately triangular wave pulse train.

        The generated waveform is sampled at sample_rate Hz, ensuring consistency with cochlear analysis.
        """
        n_samples = self._wave_samples
        t = np.arange(n_samples) / self.sample_rate

        # Fundamental + harmonics (1/n weighting, consistent with generate_vowel)
        f0 = pitch
        n_harmonics = int(self.sample_rate / 2 / max(f0, 1.0))
        wave = np.zeros(n_samples, dtype=np.float64)
        for n in range(1, min(n_harmonics + 1, 30)):
            wave += (1.0 / n) * np.sin(2 * np.pi * f0 * n * t)

        # Jitter — frequency micro-perturbation
        jitter = 0.005 * (1.0 + 5.0 * temperature ** 2)
        phase_noise = np.cumsum(np.random.randn(n_samples) * jitter)
        wave += 0.1 * np.sin(2 * np.pi * f0 * t + phase_noise)

        # Shimmer — amplitude micro-perturbation
        shimmer = 0.01 * (1.0 + 10.0 * temperature ** 2)
        amp_noise = 1.0 + np.random.randn(n_samples) * shimmer
        wave *= amp_noise

        return wave

    # ------------------------------------------------------------------
    def _vocal_tract_filter(
        self,
        waveform: np.ndarray,
        formants: Tuple[float, float, float],
    ) -> np.ndarray:
        """
        Vocal tract resonance cavity — formant filter

        The vocal tract is a non-uniform tube; different mouth shapes/tongue positions/pharynx shapes
        produce resonances at different frequencies → formants.
        F1: Primarily determined by opening degree (wider opening → higher F1)
        F2: Primarily determined by tongue front-back position (front → higher F2)
        F3: Primarily determined by lip shape

        Simulated using Lorentzian resonance curves.
        """
        n = len(waveform)
        fft_data = np.fft.rfft(waveform)
        freqs_hz = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

        # Apply Gaussian gain for each formant (consistent with generate_vowel formant model)
        total_gain = np.ones_like(freqs_hz)
        for i, f_center in enumerate(formants):
            bw = FORMANT_BANDWIDTHS[i] if i < len(FORMANT_BANDWIDTHS) else 100.0
            total_gain *= 1.0 + 2.0 * np.exp(
                -0.5 * ((freqs_hz - f_center) / max(bw, 1.0)) ** 2
            )

        fft_data *= total_gain
        result = np.fft.irfft(fft_data, n=n)

        return result

    # ------------------------------------------------------------------
    def _vocal_feedback(self, waveform: np.ndarray) -> ElectricalSignal:
        """
        Vocal cord feedback signal — sensing one's own vocalization

        This is the physical basis of self-audition:
        Vocal cord vibration → bone conduction → inner ear → auditory feedback
        (This is why your own voice sounds different in recordings)
        """
        amp = float(np.sqrt(np.mean(waveform ** 2)))

        # Map to brainwave frequency
        br_freq = BRAINWAVE_FREQ_MIN + (
            self._current_pitch - VOCAL_FOLD_MIN_FREQ
        ) / (VOCAL_FOLD_MAX_FREQ - VOCAL_FOLD_MIN_FREQ) * (
            BRAINWAVE_FREQ_MAX - BRAINWAVE_FREQ_MIN
        )
        br_freq = float(np.clip(br_freq, BRAINWAVE_FREQ_MIN, BRAINWAVE_FREQ_MAX))

        # Resample to standard point count
        if len(waveform) != self.sample_points:
            x_old = np.linspace(0, 1, len(waveform))
            x_new = np.linspace(0, 1, self.sample_points)
            waveform = np.interp(x_new, x_old, waveform)

        return ElectricalSignal(
            waveform=waveform,
            amplitude=amp,
            frequency=br_freq,
            phase=0.0,
            impedance=MOUTH_IMPEDANCE,
            snr=MOUTH_SNR,
            source="mouth",
            modality="vocal",
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_utterances": self.total_utterances,
            "total_phonemes": self.total_phonemes,
            "base_pitch": self.base_pitch,
            "current_pitch": self._current_pitch,
            "lung_pressure": self._lung_pressure,
            "sample_points": self.sample_points,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "wave_samples": self._wave_samples,
            "impedance": MOUTH_IMPEDANCE,
        }

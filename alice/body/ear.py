# -*- coding: utf-8 -*-
"""
Alice's Ear — Forward Engineering

Physics:
  "The cochlea is essentially a Fourier analyzer."
  Sound wave → Ear canal (tube resonance) → Tympanic membrane (pressure→displacement)
  → Cochlea (frequency decomposition) → Hair cells (piezoelectric conversion)
  → Auditory nerve (ElectricalSignal)

  Key insight: The basilar membrane of the cochlea is a frequency analyzer.
  The base of the basilar membrane (narrow, stiff) resonates at high frequencies,
  the apex (wide, soft) resonates at low frequencies.
  This is determined by physical structure — the brain is not "computing frequencies."

  FFT is more physically correct for hearing than for the eye — because the
  basilar membrane's tonotopic mapping is literally frequency decomposition.

Pipeline:
  1. Ear canal (tube resonance)     — 2-5kHz natural amplification → Resonance gain
  2. Tympanic membrane (thin film)  — Sound pressure → mechanical vibration → Displacement waveform
  3. Cochlea (basilar membrane)     — Physical Fourier analysis → Frequency decomposition
  4. Hair cells (piezoelectric)     — Mechanical vibration → voltage → Voltage waveform
  5. Auditory nerve                 — Coaxial cable → ElectricalSignal

Sound frequency → Brainwave frequency mapping:
  Low freq sounds (rumble, heartbeat)        → δ/θ   (0.5-8 Hz)
  Mid freq sounds (voice fundamental, music) → α/β   (8-30 Hz)
  High freq sounds (sibilants, harmonics)    → γ     (30-100 Hz)

This is why:
  - Low-frequency bass makes people relax (δ/θ → parasympathetic)
  - High-frequency sharp sounds make people alert (γ → sympathetic/startle response)
  — Not psychological suggestion, but physical resonance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import BrainWaveBand, ElectricalSignal


# ============================================================================
# Physical constants of the ear
# ============================================================================

# Sound frequency → Brainwave frequency mapping range
BRAINWAVE_FREQ_MIN = 0.5    # Hz (δ lower bound)
BRAINWAVE_FREQ_MAX = 100.0  # Hz (γ upper bound)

# Ear canal resonance
EAR_CANAL_RESONANCE_HZ = 3000.0   # Ear canal resonance frequency ≈ 3kHz
EAR_CANAL_Q = 5.0                 # Ear canal resonance quality factor
EAR_CANAL_GAIN = 2.0              # Resonance region gain (~6dB)

# Tympanic membrane properties
TYMPANIC_IMPEDANCE = 50.0         # Ω (tympanic membrane impedance = normalized acoustic impedance)
TYMPANIC_SENSITIVITY = 1.0        # Sound pressure→displacement conversion gain

# Cochlea properties
COCHLEA_RESOLUTION = 256          # Basilar membrane frequency resolution (simplified hair cell count)
COCHLEA_FREQ_MIN = 20.0           # Human hearing lower limit Hz
COCHLEA_FREQ_MAX = 20000.0        # Human hearing upper limit Hz

# Hair cell properties
HAIR_CELL_IMPEDANCE = 50.0        # Ω (hair cell→auditory nerve impedance)
HAIR_CELL_GAIN = 1.0              # Piezoelectric conversion gain

# Auditory nerve
AUDITORY_NERVE_SNR = 14.0         # dB (auditory nerve SNR, slightly lower than visual)
AUDITORY_NERVE_IMPEDANCE = 50.0   # Ω


# ============================================================================
# Main class
# ============================================================================


class AliceEar:
    """
    Alice's Ear — Forward engineering sensory organ

    Sound wave → Ear canal resonance → Tympanic vibration → Cochlea (FFT) → Hair cells → ElectricalSignal

    Does not perform any "understanding" or "recognition".
    Only performs physical conversion — the rest is left to the brain's LC resonance.
    """

    def __init__(
        self,
        cochlea_resolution: int = COCHLEA_RESOLUTION,
        hearing_sensitivity: float = 1.0,
    ):
        """
        Args:
            cochlea_resolution: Cochlea resolution (basilar membrane frequency decomposition points)
            hearing_sensitivity: Hearing sensitivity 0.0~2.0
                                 (decreases with age, noise damage)
        """
        self.cochlea_resolution = cochlea_resolution
        self.hearing_sensitivity = float(np.clip(hearing_sensitivity, 0.01, 2.0))

        # Pre-build ear canal resonance filter
        self._ear_canal_filter: Optional[np.ndarray] = None

        # Statistics
        self.total_samples: int = 0
        self.total_listens: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def hear(self, sound_wave: np.ndarray) -> ElectricalSignal:
        """
        Hear = Physical conversion (not computation)

        Steps:
          1. Ear canal: tube resonance amplification 2-5kHz
          2. Tympanic membrane: sound pressure → mechanical vibration
          3. Cochlea: basilar membrane frequency decomposition (FFT — this is the physics of the basilar membrane!)
          4. Hair cells: mechanical vibration → voltage
          5. Auditory nerve: package as ElectricalSignal

        Args:
            sound_wave: Sound pressure time series (1D ndarray)
                        Can be microphone samples, synthesized waveform, or any audio signal

        Returns:
            ElectricalSignal: Auditory nerve electrical signal
              - waveform : hair cell voltage waveform (frequency domain power spectrum)
              - frequency: dominant frequency (mapped to brainwave band)
              - amplitude: sound intensity
              - source="ear", modality="auditory"
        """
        self.total_listens += 1
        self.total_samples += len(sound_wave)

        # === 1. Ear canal — Tube resonance ===
        resonated = self._ear_canal(sound_wave)

        # === 2. Tympanic membrane — Sound pressure→mechanical vibration ===
        displaced = self._tympanic_membrane(resonated)

        # === 3. Cochlea — Basilar membrane frequency decomposition (physical Fourier!) ===
        spectrum, freqs = self._cochlea_fft(displaced)

        # === 4. Hair cells — Piezoelectric conversion ===
        voltage, dominant_freq, amplitude = self._hair_cells(
            spectrum, freqs
        )

        # === 5. Auditory nerve — Package electrical signal ===
        return self._auditory_nerve(voltage, dominant_freq, amplitude)

    # ------------------------------------------------------------------
    def adjust_sensitivity(self, sensitivity: float):
        """
        Adjust hearing sensitivity

        Simulates:
        - Age-related hearing loss (decrease)
        - Attention enhancement (slight increase)
        - Noise damage (permanent decrease)
        """
        self.hearing_sensitivity = float(np.clip(sensitivity, 0.01, 2.0))

    # ------------------------------------------------------------------
    def attend_frequency_band(
        self, sound_wave: np.ndarray,
        center_hz: float, bandwidth_hz: float,
    ) -> ElectricalSignal:
        """
        Frequency attention — Focus on a specific frequency band

        Similar to the eye's saccade, but in the frequency domain.
        Physical basis of the "cocktail party effect":
        A segment of basilar membrane is amplified → corresponding hair cells become more sensitive.

        Args:
            sound_wave: Complete sound
            center_hz:  Attention center frequency (Hz)
            bandwidth_hz: Attention bandwidth (Hz)

        Returns:
            ElectricalSignal: Focused auditory nerve electrical signal
        """
        n = len(sound_wave)
        if n == 0:
            return self.hear(np.zeros(64))

        # FFT → Frequency domain
        fft_result = np.fft.rfft(sound_wave.astype(np.float64))
        freqs_hz = np.fft.rfftfreq(n, d=1.0 / max(n, 1))

        # Bandpass filter (Gaussian window)
        sigma = bandwidth_hz / 2.0
        if sigma < 1.0:
            sigma = 1.0
        mask = np.exp(-0.5 * ((freqs_hz - center_hz) / sigma) ** 2)
        filtered = np.fft.irfft(fft_result * mask, n=n)

        return self.hear(filtered)

    # ------------------------------------------------------------------
    # Internal physics pipeline
    # ------------------------------------------------------------------

    def _ear_canal(self, sound_wave: np.ndarray) -> np.ndarray:
        """
        Ear canal — Tube resonance

        Ear canal is about 2.5cm long → λ/4 resonance ≈ 3kHz
        Amplification of ~10-15dB near resonance frequency.
        Simplified model: amplify frequency domain with Lorentzian resonance curve.
        """
        raw = sound_wave.flatten().astype(np.float64)
        n = len(raw)
        if n == 0:
            return np.zeros(1)

        # FFT
        fft_data = np.fft.rfft(raw)
        freqs_hz = np.fft.rfftfreq(n, d=1.0 / max(n, 1))

        # Lorentzian resonance gain
        # G(f) = 1 + (GAIN-1) / (1 + ((f - f0) / (f0/2Q))^2)
        f0 = EAR_CANAL_RESONANCE_HZ
        gamma = f0 / (2.0 * EAR_CANAL_Q)
        gain = 1.0 + (EAR_CANAL_GAIN - 1.0) / (
            1.0 + ((freqs_hz - f0) / max(gamma, 1e-9)) ** 2
        )
        fft_data *= gain

        return np.fft.irfft(fft_data, n=n)

    # ------------------------------------------------------------------
    def _tympanic_membrane(self, resonated: np.ndarray) -> np.ndarray:
        """
        Tympanic membrane — Sound pressure→mechanical displacement

        - Normalize sound pressure
        - Multiply by sensitivity (hearing loss/enhancement)
        - Produce displacement waveform
        """
        # Normalize
        max_val = np.max(np.abs(resonated))
        if max_val > 0:
            normalized = resonated / max_val
        else:
            normalized = resonated

        # Sensitivity × conversion gain
        displacement = normalized * self.hearing_sensitivity * TYMPANIC_SENSITIVITY

        return displacement

    # ------------------------------------------------------------------
    def _cochlea_fft(
        self, displacement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cochlea — Basilar membrane frequency decomposition

        Physical fact:
          The basilar membrane is a graded elastic strip:
          - Base (cochlear base) narrow and stiff → resonates with high frequencies
          - Apex (cochlear apex) wide and soft → resonates with low frequencies

          This is literally spatial Fourier decomposition.
          Each position = one frequency = one group of hair cells.

        Returns:
            (power_spectrum, frequency_bins_mapped_to_brainwave)
        """
        n = len(displacement)
        if n == 0:
            return np.zeros(1), np.array([BRAINWAVE_FREQ_MIN])

        # FFT = Basilar membrane resonance decomposition
        fft_result = np.fft.rfft(displacement)
        power = np.abs(fft_result) ** 2  # Power spectrum

        # Sound frequency → Brainwave frequency mapping (log mapping)
        n_bins = len(power)
        freq_bins = np.logspace(
            math.log10(BRAINWAVE_FREQ_MIN),
            math.log10(BRAINWAVE_FREQ_MAX),
            n_bins,
        )

        return power, freq_bins

    # ------------------------------------------------------------------
    def _hair_cells(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Hair cells — Piezoelectric conversion

        - Power spectrum → voltage waveform (V = sqrt(P × R))
        - Extract dominant frequency (most active hair cell group)
        - Total sound intensity → amplitude

        Outer hair cells: amplification (cochlear amplifier, ~40dB)
        Inner hair cells: convert to electrical signal

        Returns:
            (hair_cell_voltage, dominant_freq, amplitude)
        """
        # Power → Voltage
        voltage = np.sqrt(spectrum * HAIR_CELL_IMPEDANCE) * HAIR_CELL_GAIN

        # Uniformly resample to cochlea resolution
        if len(voltage) != self.cochlea_resolution:
            x_old = np.linspace(0, 1, len(voltage))
            x_new = np.linspace(0, 1, self.cochlea_resolution)
            voltage = np.interp(x_new, x_old, voltage)
            freqs = np.interp(x_new, x_old, freqs)

        # Dominant frequency = most active hair cell
        if len(voltage) > 1:
            peak_idx = np.argmax(voltage[1:]) + 1
            dominant_freq = float(freqs[peak_idx])
        else:
            dominant_freq = 10.0  # Default alpha

        # Amplitude = total sound intensity (RMS)
        amplitude = float(np.sqrt(np.mean(voltage ** 2)))

        return voltage, dominant_freq, amplitude

    # ------------------------------------------------------------------
    def _auditory_nerve(
        self,
        hair_cell_voltage: np.ndarray,
        dominant_freq: float,
        amplitude: float,
    ) -> ElectricalSignal:
        """
        Auditory nerve — Package electrical signal

        Hair cell voltage → ElectricalSignal
        Ready for transmission along coaxial cable to auditory cortex
        """
        freq = float(np.clip(dominant_freq, BRAINWAVE_FREQ_MIN, BRAINWAVE_FREQ_MAX))

        return ElectricalSignal(
            waveform=hair_cell_voltage,
            amplitude=amplitude,
            frequency=freq,
            phase=0.0,
            impedance=AUDITORY_NERVE_IMPEDANCE,
            snr=AUDITORY_NERVE_SNR,
            source="ear",
            modality="auditory",
        )

    # ------------------------------------------------------------------
    # Statistics and diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_listens": self.total_listens,
            "total_samples": self.total_samples,
            "cochlea_resolution": self.cochlea_resolution,
            "hearing_sensitivity": self.hearing_sensitivity,
            "impedance": AUDITORY_NERVE_IMPEDANCE,
        }

    def get_cochlea_snapshot(self, sound_wave: np.ndarray) -> Dict[str, Any]:
        """
        Diagnostic: get snapshot of each cochlear layer

        Returns:
            {
                "resonated": waveform after ear canal resonance,
                "displacement": tympanic membrane displacement waveform,
                "spectrum": cochlea frequency decomposition,
                "freqs": frequency bins,
                "voltage": hair cell voltage,
                "dominant_freq": dominant frequency,
                "band": corresponding brainwave band,
            }
        """
        resonated = self._ear_canal(sound_wave)
        displacement = self._tympanic_membrane(resonated)
        spectrum, freqs = self._cochlea_fft(displacement)
        voltage, dom_freq, amp = self._hair_cells(spectrum, freqs)

        return {
            "resonated": resonated.tolist(),
            "displacement": displacement.tolist(),
            "spectrum": spectrum.tolist(),
            "freqs": freqs.tolist(),
            "voltage": voltage.tolist(),
            "dominant_freq": dom_freq,
            "band": BrainWaveBand.from_frequency(dom_freq).value,
            "amplitude": amp,
        }

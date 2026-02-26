# -*- coding: utf-8 -*-
"""
Cochlear Filter Bank — Physical Model of the Basilar Membrane

Physics:
  The basilar membrane of the human cochlea is a graded elastic strip:
    Base (narrow, stiff) → Resonates at high frequencies (~20kHz)
    Apex (wide, soft) → Resonates at low frequencies (~20Hz)

  Each position is a bandpass filter.
  ~3500 inner hair cells → ~24 critical bands
  Critical bandwidth follows the ERB (Equivalent Rectangular Bandwidth) scale.

  ERB(f) = 24.7 × (4.37 × f/1000 + 1)    [Glasberg & Moore, 1990]

  Gammatone filter is the standard model for auditory nerve fiber impulse response:
    g(t) = t^(n-1) × exp(-2π × b × t) × cos(2π × f_c × t)
    n = 4 (4th order), b = 1.019 × ERB(f_c)

  But for efficiency, we use FFT + ERB bandpass decomposition instead of
  per-sample convolution.
  Physical equivalence: both decompose sound pressure waveform into tonotopic
  band energies.

Output:
  TonotopicActivation — Activation values for 24 channels
  Each channel = a segment of the basilar membrane = a group of hair cells
  This is the "firing rate" of auditory nerve fibers.

  Applications:
    - Auditory grounding (Phase 4.1): use spectral fingerprint as key for
      cross-modal Hebbian connections
    - Semantic force field (Phase 4.2): specific spectral patterns = specific "concepts"
    - Broca's area (Phase 4.3): inverse engineering — from concept → produce sound
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Physical constants
# ============================================================================

# ERB parameters [Glasberg & Moore, 1990]
ERB_A = 24.7       # Hz
ERB_B = 4.37       # per kHz
ERB_Q = 9.265      # Quality factor Q = f_c / ERB(f_c) typical value

# Default channel configuration
DEFAULT_N_CHANNELS = 24         # Number of critical bands (human ≈ 24)
DEFAULT_FREQ_MIN = 80.0         # Hz — voice fundamental frequency lower limit
DEFAULT_FREQ_MAX = 8000.0       # Hz — voice formant upper limit
DEFAULT_SAMPLE_RATE = 16000     # Hz — sampling rate

# Hair cell adaptation (temporal dynamics)
ATTACK_RATE = 0.8       # Rise rate (fast response)
DECAY_RATE = 0.05       # Decay rate (persistent residual — auditory afterimage)


# ============================================================================
# ERB scale utilities
# ============================================================================


def erb_bandwidth(f_c: float) -> float:
    """
    Calculate ERB bandwidth at center frequency f_c

    ERB(f) = 24.7 × (4.37 × f/1000 + 1)
    """
    return ERB_A * (ERB_B * f_c / 1000.0 + 1.0)


def freq_to_erb_number(f: float) -> float:
    """Hz → ERB number (cochlear position)"""
    return 21.4 * math.log10(0.00437 * f + 1.0)


def erb_number_to_freq(e: float) -> float:
    """ERB number → Hz"""
    return (10.0 ** (e / 21.4) - 1.0) / 0.00437


def generate_center_frequencies(
    n_channels: int = DEFAULT_N_CHANNELS,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
) -> np.ndarray:
    """
    Uniformly distribute center frequencies on the ERB scale

    Physical meaning: equally spaced points on the basilar membrane →
    each point represents one critical band
    """
    erb_min = freq_to_erb_number(freq_min)
    erb_max = freq_to_erb_number(freq_max)
    erb_points = np.linspace(erb_min, erb_max, n_channels)
    return np.array([erb_number_to_freq(e) for e in erb_points])


# ============================================================================
# Tonotopic activation
# ============================================================================


@dataclass
class TonotopicActivation:
    """
    Output of the cochlear tonotopic map — Band activation values for 24 channels

    Physical meaning:
      channel_activations[i] = vibration amplitude of basilar membrane segment i
                             = firing rate of hair cell group i
                             ≈ auditory nerve fiber activity in that frequency band

    Spectral Fingerprint:
      Normalized channel_activations vector
      Used as the "key" for cross-modal Hebbian binding
      Two sounds with similar fingerprints → they are "the same kind" auditorily
    """
    center_frequencies: np.ndarray          # [n_channels] Hz
    channel_activations: np.ndarray         # [n_channels] activation values (0~1)
    channel_energies: np.ndarray            # [n_channels] energy values (unnormalized)
    dominant_channel: int                   # Most active channel index
    dominant_frequency: float               # Center frequency of most active channel (Hz)
    total_energy: float                     # Total energy
    spectral_centroid: float                # Spectral centroid (Hz)
    spectral_flatness: float               # Spectral flatness (0=tonal, 1=noise)
    timestamp: float = 0.0

    def fingerprint(self) -> np.ndarray:
        """
        Spectral fingerprint — Normalized activation vector

        Physical meaning: eliminate volume differences, preserve only "shape"
        "Loud A" and "Quiet A" have the same fingerprint
        """
        total = np.sum(self.channel_activations)
        if total < 1e-12:
            return np.zeros_like(self.channel_activations)
        return self.channel_activations / total

    def fingerprint_hash(self, n_bits: int = 8) -> int:
        """
        Sparse code — Compress fingerprint to integer hash

        Physical meaning:
          "Which segments of the basilar membrane vibrate most strongly" → binary encoding
          Used for O(1) concept lookup
        """
        fp = self.fingerprint()
        if len(fp) == 0:
            return 0
        threshold = np.mean(fp)
        # Take the most significant n_bits channels
        top_indices = np.argsort(fp)[-n_bits:]
        bits = 0
        for idx in top_indices:
            if fp[idx] > threshold:
                bits |= (1 << (idx % n_bits))
        return bits

    def similarity(self, other: "TonotopicActivation") -> float:
        """
        Cosine similarity between two tonotopic activations

        Physical meaning: overlap degree of basilar membrane vibration patterns
        1.0 = identical sounds
        0.0 = completely different
        """
        fp1 = self.fingerprint()
        fp2 = other.fingerprint()
        dot = float(np.dot(fp1, fp2))
        n1 = float(np.linalg.norm(fp1))
        n2 = float(np.linalg.norm(fp2))
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return dot / (n1 * n2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_channels": len(self.channel_activations),
            "dominant_channel": self.dominant_channel,
            "dominant_frequency": round(self.dominant_frequency, 1),
            "total_energy": round(self.total_energy, 4),
            "spectral_centroid": round(self.spectral_centroid, 1),
            "spectral_flatness": round(self.spectral_flatness, 4),
            "top_3_channels": [
                int(i) for i in np.argsort(self.channel_activations)[-3:][::-1]
            ],
        }


# ============================================================================
# Cochlear filter bank
# ============================================================================


class CochlearFilterBank:
    """
    Cochlear filter bank — Tonotopic decomposition of the basilar membrane

    Physical pipeline:
      Sound pressure waveform → FFT → ERB bandpass decomposition → Hair cell activation → TonotopicActivation

    Every step is physical conversion, not computation:
      FFT = spatial resonance decomposition of the basilar membrane
      ERB grouping = hair cell clustering along critical bands
      Energy = amplitude of hair cell stereocilia bending
    """

    def __init__(
        self,
        n_channels: int = DEFAULT_N_CHANNELS,
        freq_min: float = DEFAULT_FREQ_MIN,
        freq_max: float = DEFAULT_FREQ_MAX,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ):
        self.n_channels = n_channels
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.sample_rate = sample_rate

        # Center frequencies (ERB scale)
        self.center_frequencies = generate_center_frequencies(
            n_channels, freq_min, freq_max
        )

        # ERB bandwidth
        self.bandwidths = np.array([
            erb_bandwidth(fc) for fc in self.center_frequencies
        ])

        # Hair cell persistent activation (temporal dynamics)
        self._persistent_activations = np.zeros(n_channels)

        # Statistics
        self.total_analyses = 0

    # ------------------------------------------------------------------
    def analyze(
        self,
        sound_wave: np.ndarray,
        apply_persistence: bool = True,
    ) -> TonotopicActivation:
        """
        Cochlear analysis — Sound pressure waveform → TonotopicActivation

        Physical pipeline:
          1. FFT → frequency domain (basilar membrane resonance decomposition)
          2. Each ERB band → integrate energy (total firing of hair cell groups)
          3. Normalize → activation values

        Args:
            sound_wave: sound pressure time series (1D ndarray)
            apply_persistence: whether to apply temporal persistence (hair cell adaptation)

        Returns:
            TonotopicActivation: 24-channel tonotopic decomposition
        """
        self.total_analyses += 1
        raw = np.asarray(sound_wave, dtype=np.float64).flatten()
        n = len(raw)

        if n == 0:
            return self._empty_activation()

        # === 1. FFT — Basilar membrane resonance decomposition ===
        fft_result = np.fft.rfft(raw)
        power_spectrum = np.abs(fft_result) ** 2
        freq_bins = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

        # === 2. ERB bandpass decomposition — Critical band energy ===
        channel_energies = np.zeros(self.n_channels)
        for i in range(self.n_channels):
            fc = self.center_frequencies[i]
            bw = self.bandwidths[i]
            # Bandpass: Gaussian-type weights (physics: resonance response of each basilar membrane segment)
            weights = np.exp(
                -0.5 * ((freq_bins - fc) / (bw * 0.5)) ** 2
            )
            channel_energies[i] = float(np.sum(power_spectrum * weights))

        # === 3. Hair cell activation — Energy → firing rate (compression: logarithmic) ===
        # Weber-Fechner law: perceived intensity ∝ log(stimulus intensity)
        log_energies = np.log1p(channel_energies)

        # Normalize to [0, 1]
        max_log = np.max(log_energies)
        if max_log > 1e-12:
            activations = log_energies / max_log
        else:
            activations = np.zeros(self.n_channels)

        # === 4. Temporal persistence (hair cell adaptation) ===
        if apply_persistence:
            # Fast rise, slow decay (auditory afterimage)
            for i in range(self.n_channels):
                if activations[i] > self._persistent_activations[i]:
                    self._persistent_activations[i] += (
                        ATTACK_RATE
                        * (activations[i] - self._persistent_activations[i])
                    )
                else:
                    self._persistent_activations[i] *= (1.0 - DECAY_RATE)
            activations = np.maximum(activations, self._persistent_activations)

        # === Spectral statistics ===
        dominant_ch = int(np.argmax(activations))
        dominant_freq = float(self.center_frequencies[dominant_ch])
        total_energy = float(np.sum(channel_energies))

        # Spectral centroid
        act_sum = float(np.sum(activations))
        if act_sum > 1e-12:
            spectral_centroid = float(
                np.sum(activations * self.center_frequencies) / act_sum
            )
        else:
            spectral_centroid = 0.0

        # Spectral flatness (geometric mean / arithmetic mean)
        if act_sum > 1e-12 and np.all(activations >= 0):
            log_acts = np.log(activations + 1e-12)
            geo_mean = np.exp(np.mean(log_acts))
            arith_mean = np.mean(activations)
            flatness = geo_mean / max(arith_mean, 1e-12)
            flatness = float(np.clip(flatness, 0.0, 1.0))
        else:
            flatness = 0.0

        return TonotopicActivation(
            center_frequencies=self.center_frequencies.copy(),
            channel_activations=activations.copy(),
            channel_energies=channel_energies.copy(),
            dominant_channel=dominant_ch,
            dominant_frequency=dominant_freq,
            total_energy=total_energy,
            spectral_centroid=spectral_centroid,
            spectral_flatness=flatness,
        )

    # ------------------------------------------------------------------
    def reset_persistence(self):
        """Reset hair cell persistent activation (new auditory scene)"""
        self._persistent_activations = np.zeros(self.n_channels)

    # ------------------------------------------------------------------
    def _empty_activation(self) -> TonotopicActivation:
        return TonotopicActivation(
            center_frequencies=self.center_frequencies.copy(),
            channel_activations=np.zeros(self.n_channels),
            channel_energies=np.zeros(self.n_channels),
            dominant_channel=0,
            dominant_frequency=float(self.center_frequencies[0]),
            total_energy=0.0,
            spectral_centroid=0.0,
            spectral_flatness=0.0,
        )

    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "n_channels": self.n_channels,
            "freq_range": [self.freq_min, self.freq_max],
            "sample_rate": self.sample_rate,
            "total_analyses": self.total_analyses,
            "center_frequencies": [
                round(f, 1) for f in self.center_frequencies
            ],
        }


# ============================================================================
# Sound generators (for experiments)
# ============================================================================


def generate_tone(
    frequency: float,
    duration: float = 0.1,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Generate pure tone (sine wave)

    Physics: ideal single-frequency sound wave — only one position on the basilar membrane resonates
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * t)


def generate_complex_tone(
    fundamental: float,
    n_harmonics: int = 5,
    duration: float = 0.1,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Generate complex tone (fundamental + harmonics)

    Physics: real instruments/voices are complex tones
    Harmonic decay ∝ 1/n (natural harmonic series)
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    wave = np.zeros_like(t)
    for n in range(1, n_harmonics + 1):
        wave += (amplitude / n) * np.sin(2 * np.pi * fundamental * n * t)
    return wave


def generate_noise(
    duration: float = 0.1,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate white noise

    Physics: equal power across all frequencies → entire basilar membrane activated
    Spectral flatness ≈ 1.0
    """
    if rng is None:
        rng = np.random.default_rng()
    n_samples = int(duration * sample_rate)
    return amplitude * rng.standard_normal(n_samples)


def generate_vowel(
    vowel: str = "a",
    fundamental: float = 150.0,
    duration: float = 0.1,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate vowel waveform (simplified formant model)

    Physics: vocal tract = tube resonator
      Different vowels = different vocal tract shapes = different formant positions
      F1, F2 pair determines vowel identity
    """
    FORMANTS = {
        "a": [730, 1090, 2440],    # /a/ wide open
        "i": [270, 2290, 3010],    # /i/ front high tongue
        "u": [300, 870, 2240],     # /u/ back high tongue
        "e": [530, 1840, 2480],    # /e/ front mid tongue
        "o": [570, 840, 2410],     # /o/ back mid tongue
    }
    formants = FORMANTS.get(vowel, FORMANTS["a"])
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Glottal pulse (fundamental source)
    source = np.zeros_like(t)
    n_harmonics = int(sample_rate / 2 / fundamental)
    for n in range(1, min(n_harmonics + 1, 30)):
        source += (1.0 / n) * np.sin(2 * np.pi * fundamental * n * t)

    # Formant filtering (vocal tract resonance)
    fft_source = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(len(source), d=1.0 / sample_rate)

    transfer = np.ones_like(freqs)
    for f_i in formants:
        bw = f_i * 0.1  # Bandwidth ≈ 10% of formant freq
        transfer *= 1.0 + 2.0 * np.exp(
            -0.5 * ((freqs - f_i) / max(bw, 1.0)) ** 2
        )

    result = np.fft.irfft(fft_source * transfer, n=len(source))
    # Normalize
    mx = np.max(np.abs(result))
    if mx > 0:
        result = result / mx
    return result

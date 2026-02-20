# -*- coding: utf-8 -*-
"""
Unified Electrical Signal Framework — Coaxial Cable Model

Core Physics:
  "Alice's nervous system transmits electrical signals like a coaxial cable.
   Impedance match = resonance = lossless transmission.
   Impedance mismatch = reflection = energy waste → heat → pain."

Physical Formulas:
  Reflection coefficient  Γ = (Z_load - Z₀) / (Z_load + Z₀)
  Reflected power         P_r = Γ² × P_in
  Transmitted power       P_t = (1 - Γ²) × P_in
  Signal attenuation      A(dB) = α × length × f / f_ref
  Thermal noise           N = k_B × T × Δf

Brainwave Frequency Bands (carrier frequency mapping):
  δ delta   0.5–4 Hz    Deep sleep / subconscious repair
  θ theta   4–8 Hz      Relaxation / intuition / memory consolidation
  α alpha   8–13 Hz     Calm wakefulness / default mode
  β beta    13–30 Hz    Focus / anxiety / active thinking
  γ gamma   30–100 Hz   High-order cognition / insight / cross-region binding

Coaxial Channel Mapping:
  Sensory→Prefrontal : 75Ω  (low impedance, high throughput)
  Sensory→Limbic     : 50Ω  (standard channel)
  Prefrontal→Motor   : 75Ω  (cognitive-motor coupling)
  Limbic→Motor       : 110Ω (emotion-weighted, high impedance)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# 1. Brainwave Frequency Bands
# ============================================================================


class BrainWaveBand(Enum):
    """
    Brainwave Frequency Band — carrier frequency classification of electrical signals

    Each band corresponds to a different cognitive state,
    just as EM waves of different frequencies have different transmission characteristics in coaxial cables.
    """

    DELTA = "delta"  # 0.5–4 Hz   Deep sleep / subconscious
    THETA = "theta"  # 4–8 Hz     Relaxation / intuition
    ALPHA = "alpha"  # 8–13 Hz    Calm wakefulness
    BETA = "beta"  # 13–30 Hz   Focus / anxiety
    GAMMA = "gamma"  # 30–100 Hz  High-order cognition

    @property
    def freq_range(self) -> Tuple[float, float]:
        """Frequency range (Hz)"""
        ranges = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 100.0),
        }
        return ranges[self.value]

    @property
    def center_freq(self) -> float:
        """Center frequency"""
        lo, hi = self.freq_range
        return (lo + hi) / 2.0

    @classmethod
    def from_frequency(cls, freq: float) -> "BrainWaveBand":
        """Determine the brainwave band from a given frequency"""
        if freq < 4.0:
            return cls.DELTA
        elif freq < 8.0:
            return cls.THETA
        elif freq < 13.0:
            return cls.ALPHA
        elif freq < 30.0:
            return cls.BETA
        else:
            return cls.GAMMA


# ============================================================================
# 2. Unified Electrical Signal
# ============================================================================


@dataclass
class ElectricalSignal:
    """
    Unified Electrical Signal — the universal language of Alice's nervous system

    Like an electrical signal in a coaxial cable:
    - waveform  : waveform data (voltage over time, i.e. raw data)
    - amplitude : amplitude (peak voltage, corresponding to stimulus intensity)
    - frequency : carrier frequency (Hz, determines brainwave band)
    - phase     : phase (rad, determines waveform alignment → resonance)
    - impedance : characteristic impedance (Ω, determines coupling efficiency)
    - snr       : signal-to-noise ratio (dB, signal quality)

    All data flowing in the Alice system must be an ElectricalSignal.
    """

    waveform: np.ndarray  # Raw waveform (voltage sequence)
    amplitude: float  # Peak amplitude (V)
    frequency: float  # Carrier frequency (Hz)
    phase: float  # Phase (rad, 0 ~ 2π)
    impedance: float  # Characteristic impedance (Ω)
    snr: float  # Signal-to-noise ratio (dB)
    timestamp: float = field(default_factory=time.time)
    source: str = "external"  # Source identifier
    modality: str = "internal"  # Modality

    # ------------------------------------------------------------------
    # Factory Methods
    # ------------------------------------------------------------------

    @classmethod
    def from_raw(
        cls,
        data: np.ndarray,
        source: str = "external",
        modality: str = "visual",
        impedance: float = 75.0,
    ) -> "ElectricalSignal":
        """
        Create an electrical signal from raw data (auto-analyze frequency characteristics)

        Physical mapping:
        - Amplitude = peak value of data (max absolute value)
        - Frequency = estimated from gradient changes (similar to zero-crossing rate)
        - Phase = estimated from peak position
        - SNR  = signal power / estimated noise power
        """
        flat = data.flatten().astype(np.float64)
        n = len(flat)

        # Amplitude: peak voltage
        amplitude = float(np.max(np.abs(flat))) if n > 0 else 0.0

        # Frequency: estimated from zero-crossing rate
        if n > 1:
            # Zero crossing count → estimate frequency
            zero_crossings = int(np.sum(np.abs(np.diff(np.sign(flat - np.mean(flat)))) > 0))
            # Assume sampling rate = n samples/sec → freq = crossings / (2 * duration)
            estimated_freq = max(0.5, zero_crossings / 2.0)
            # Map to brainwave range (0.5–100 Hz)
            estimated_freq = float(np.clip(estimated_freq, 0.5, 100.0))
        else:
            estimated_freq = 10.0  # Default alpha

        # Phase: peak position → map to 0~2π
        if n > 0:
            peak_pos = int(np.argmax(np.abs(flat)))
            phase = (peak_pos / max(1, n - 1)) * 2 * math.pi
        else:
            phase = 0.0

        # SNR: signal power / differential noise estimate
        if n > 1:
            signal_power = float(np.mean(flat**2))
            noise_est = float(np.mean(np.diff(flat) ** 2)) / 2.0  # Donoho noise estimator
            snr = 10 * math.log10(max(1e-10, signal_power) / max(1e-10, noise_est))
        else:
            snr = 0.0

        return cls(
            waveform=data.copy(),
            amplitude=amplitude,
            frequency=estimated_freq,
            phase=phase,
            impedance=impedance,
            snr=snr,
            source=source,
            modality=modality,
        )

    @classmethod
    def from_neural_activity(
        cls,
        activations: np.ndarray,
        region_impedance: float = 75.0,
        source: str = "neural",
    ) -> "ElectricalSignal":
        """Create an electrical signal from neural activation vector"""
        sig = cls.from_raw(activations, source=source, modality="internal", impedance=region_impedance)
        return sig

    # ------------------------------------------------------------------
    # Physical Properties
    # ------------------------------------------------------------------

    @property
    def band(self) -> BrainWaveBand:
        """Brainwave band this signal belongs to"""
        return BrainWaveBand.from_frequency(self.frequency)

    @property
    def power(self) -> float:
        """Signal power P = mean(V²)"""
        return float(np.mean(self.waveform.flatten() ** 2))

    @property
    def energy(self) -> float:
        """Signal energy E = sum(|V|)"""
        return float(np.sum(np.abs(self.waveform.flatten())))

    @property
    def rms(self) -> float:
        """Root mean square voltage"""
        return float(np.sqrt(np.mean(self.waveform.flatten() ** 2)))

    # ------------------------------------------------------------------
    # Physical Operations
    # ------------------------------------------------------------------

    def attenuate(self, factor: float) -> "ElectricalSignal":
        """
        Signal attenuation

        factor: 0.0 = full attenuation, 1.0 = no attenuation
        """
        att = ElectricalSignal(
            waveform=self.waveform * factor,
            amplitude=self.amplitude * factor,
            frequency=self.frequency,
            phase=self.phase,
            impedance=self.impedance,
            snr=self.snr + 10 * math.log10(max(1e-10, factor)),  # SNR decreases with attenuation
            timestamp=self.timestamp,
            source=self.source,
            modality=self.modality,
        )
        return att

    def add_noise(self, noise_power: float) -> "ElectricalSignal":
        """
        Add Gaussian noise

        noise_power: standard deviation of the noise
        """
        noise = np.random.normal(0, noise_power, self.waveform.shape)
        noisy = self.waveform + noise
        # Update SNR
        sig_power = float(np.mean(self.waveform.flatten() ** 2))
        noise_actual = float(np.mean(noise.flatten() ** 2))
        new_snr = 10 * math.log10(max(1e-10, sig_power) / max(1e-10, noise_actual))

        return ElectricalSignal(
            waveform=noisy,
            amplitude=float(np.max(np.abs(noisy.flatten()))),
            frequency=self.frequency,
            phase=self.phase,
            impedance=self.impedance,
            snr=new_snr,
            timestamp=self.timestamp,
            source=self.source,
            modality=self.modality,
        )

    def phase_shift(self, delta_phase: float) -> "ElectricalSignal":
        """Phase shift (simulates transmission delay)"""
        return ElectricalSignal(
            waveform=self.waveform.copy(),
            amplitude=self.amplitude,
            frequency=self.frequency,
            phase=(self.phase + delta_phase) % (2 * math.pi),
            impedance=self.impedance,
            snr=self.snr,
            timestamp=self.timestamp,
            source=self.source,
            modality=self.modality,
        )

    def resample(self, target_size: int) -> "ElectricalSignal":
        """Resample to target size (match brain region neuron count)"""
        flat = self.waveform.flatten()
        if len(flat) == target_size:
            resampled = flat
        else:
            x_old = np.linspace(0, 1, len(flat))
            x_new = np.linspace(0, 1, target_size)
            resampled = np.interp(x_new, x_old, flat)
        return ElectricalSignal(
            waveform=resampled,
            amplitude=self.amplitude,
            frequency=self.frequency,
            phase=self.phase,
            impedance=self.impedance,
            snr=self.snr,
            timestamp=self.timestamp,
            source=self.source,
            modality=self.modality,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amplitude": round(self.amplitude, 4),
            "frequency": round(self.frequency, 2),
            "phase": round(self.phase, 4),
            "impedance": round(self.impedance, 1),
            "snr": round(self.snr, 2),
            "band": self.band.value,
            "power": round(self.power, 6),
            "energy": round(self.energy, 4),
            "rms": round(self.rms, 4),
            "source": self.source,
            "modality": self.modality,
            "waveform_size": self.waveform.size,
        }


# ============================================================================
# 3. Coaxial Channel — wired communication between brain regions
# ============================================================================


@dataclass
class CoaxialChannel:
    """
    Coaxial Channel — communication cable connecting two brain regions

    Physical model:
    - characteristic_impedance : channel characteristic impedance Z₀ (Ω)
    - length                  : channel length (affects attenuation and delay)
    - attenuation_rate        : attenuation rate per unit length (dB/unit)
    - bandwidth               : channel bandwidth limit (Hz)

    Key physics:
    - When impedance matched: signal transmitted losslessly (resonance)
    - When impedance mismatched: part of signal reflected → energy waste → heat
    - Reflection coefficient Γ = (Z_load - Z₀) / (Z_load + Z₀)
    - Reflected power ratio = Γ²
    """

    source_name: str
    target_name: str
    characteristic_impedance: float = 75.0  # Z₀ (Ω)
    length: float = 1.0  # Channel length
    attenuation_rate: float = 0.02  # dB/unit/Hz_normalized
    bandwidth: float = 100.0  # Max frequency (Hz)

    # Statistics
    total_transmissions: int = 0
    total_reflected_energy: float = 0.0
    total_transmitted_energy: float = 0.0
    impedance_mismatches: int = 0

    # ------------------------------------------------------------------
    def reflection_coefficient(self, signal: ElectricalSignal) -> float:
        """
        Calculate reflection coefficient Γ

        Γ = (Z_load - Z₀) / (Z_load + Z₀)

        - Γ = 0 : perfect match (all energy passes through)
        - |Γ| = 1 : total reflection (no energy passes through)
        """
        z_load = signal.impedance
        z0 = self.characteristic_impedance
        if z_load + z0 == 0:
            return 0.0
        gamma = (z_load - z0) / (z_load + z0)
        return gamma

    # ------------------------------------------------------------------
    def transmit(self, signal: ElectricalSignal) -> Tuple["ElectricalSignal", "TransmissionReport"]:
        """
        Transmit an electrical signal through the coaxial channel

        Full physical simulation:
        1. Impedance matching → calculate reflection coefficient
        2. Reflected energy (bounced back → converted to heat)
        3. Distance attenuation (increases with length and frequency)
        4. Bandwidth limitation (components beyond bandwidth are truncated)
        5. Thermal noise (all channels have noise floor)
        6. Propagation delay (phase shift)
        """
        self.total_transmissions += 1
        report = TransmissionReport(channel=f"{self.source_name}→{self.target_name}")

        # === 1. Impedance Matching ===
        gamma = self.reflection_coefficient(signal)
        reflected_ratio = gamma ** 2  # Reflected power ratio
        transmitted_ratio = 1.0 - reflected_ratio  # Transmitted power ratio

        report.reflection_coefficient = gamma
        report.reflected_power_ratio = reflected_ratio

        if abs(gamma) > 0.1:
            self.impedance_mismatches += 1
            report.impedance_matched = False

        # Accumulated reflected energy (converted to system heat)
        reflected_energy = signal.power * reflected_ratio
        self.total_reflected_energy += reflected_energy
        report.reflected_energy = reflected_energy

        # === 2. Transmission Attenuation ===
        # Attenuation = rate × length × (freq / ref_freq)
        # Higher frequency signals attenuate faster (like real coaxial cables)
        freq_factor = max(0.1, signal.frequency / 10.0)  # Using 10Hz alpha as reference
        attenuation_db = self.attenuation_rate * self.length * freq_factor
        attenuation_linear = 10 ** (-attenuation_db / 20.0)  # dB → linear
        report.attenuation_db = attenuation_db

        # === 3. Bandwidth Limitation ===
        bandwidth_factor = 1.0
        if signal.frequency > self.bandwidth * 0.8:
            # Approaching bandwidth limit: attenuation intensifies
            overshoot = signal.frequency / self.bandwidth
            bandwidth_factor = max(0.1, 1.0 - (overshoot - 0.8) * 2.5)
        report.bandwidth_factor = bandwidth_factor

        # === 4. Combined Attenuation ===
        total_factor = math.sqrt(transmitted_ratio) * attenuation_linear * bandwidth_factor
        total_factor = max(0.01, min(1.0, total_factor))
        report.total_transmission_factor = total_factor

        transmitted = signal.attenuate(total_factor)

        # === 5. Thermal Noise ===
        noise_level = 0.001 * self.length * (1.0 + reflected_ratio)  # More reflection = more noise
        if noise_level > 0.0001:
            transmitted = transmitted.add_noise(noise_level)
        report.noise_added = noise_level

        # === 6. Propagation Delay (phase shift) ===
        delay_phase = self.length * 0.1 * (signal.frequency / 10.0)
        transmitted = transmitted.phase_shift(delay_phase)
        report.phase_delay = delay_phase

        # Update impedance to target impedance (signal has entered the target brain region)
        transmitted = ElectricalSignal(
            waveform=transmitted.waveform,
            amplitude=transmitted.amplitude,
            frequency=transmitted.frequency,
            phase=transmitted.phase,
            impedance=self.characteristic_impedance,  # Use channel characteristic impedance
            snr=transmitted.snr,
            timestamp=transmitted.timestamp,
            source=transmitted.source,
            modality=transmitted.modality,
        )

        self.total_transmitted_energy += transmitted.power
        report.transmitted_energy = transmitted.power

        return transmitted, report

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        total_energy = self.total_reflected_energy + self.total_transmitted_energy
        return {
            "channel": f"{self.source_name}→{self.target_name}",
            "impedance": self.characteristic_impedance,
            "length": self.length,
            "transmissions": self.total_transmissions,
            "impedance_mismatches": self.impedance_mismatches,
            "mismatch_rate": round(
                self.impedance_mismatches / max(1, self.total_transmissions), 3
            ),
            "total_reflected_energy": round(self.total_reflected_energy, 6),
            "total_transmitted_energy": round(self.total_transmitted_energy, 6),
            "energy_efficiency": round(
                self.total_transmitted_energy / max(1e-10, total_energy), 4
            ),
        }


# ============================================================================
# 4. Transmission Report
# ============================================================================


@dataclass
class TransmissionReport:
    """Physical report for a single transmission"""

    channel: str = ""
    reflection_coefficient: float = 0.0
    reflected_power_ratio: float = 0.0
    reflected_energy: float = 0.0
    transmitted_energy: float = 0.0
    attenuation_db: float = 0.0
    bandwidth_factor: float = 1.0
    total_transmission_factor: float = 1.0
    noise_added: float = 0.0
    phase_delay: float = 0.0
    impedance_matched: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "Γ": round(self.reflection_coefficient, 4),
            "reflected_power_%": round(self.reflected_power_ratio * 100, 2),
            "reflected_energy": round(self.reflected_energy, 6),
            "attenuation_dB": round(self.attenuation_db, 3),
            "transmission_factor": round(self.total_transmission_factor, 4),
            "impedance_matched": self.impedance_matched,
        }


# ============================================================================
# 5. Central Signal Bus (neural backbone network)
# ============================================================================


class SignalBus:
    """
    Central Signal Bus — Alice's neural backbone

    Manages all coaxial channel connections between brain regions.
    Like a router managing multiple coaxial cables.

    Default topology (4 brain regions × 6 bidirectional channels):
      Somatosensory ↔ Prefrontal   (Z₀=75Ω, primary cognitive pathway)
      Somatosensory ↔ Limbic       (Z₀=50Ω, emotional fast pathway)
      Prefrontal    ↔ Motor        (Z₀=75Ω, cognitive-motor commands)
      Limbic        ↔ Motor        (Z₀=110Ω, emotion-motor, high impedance)
      Prefrontal    ↔ Limbic       (Z₀=90Ω, cognitive-emotional regulation)
      Somatosensory ↔ Motor        (Z₀=60Ω, reflex arc, low impedance fast)
    """

    def __init__(self):
        self.channels: Dict[Tuple[str, str], CoaxialChannel] = {}
        self._transmission_log: List[TransmissionReport] = []

    # ------------------------------------------------------------------
    def connect(
        self,
        source: str,
        target: str,
        impedance: float = 75.0,
        length: float = 1.0,
        attenuation_rate: float = 0.02,
        bandwidth: float = 100.0,
        bidirectional: bool = True,
    ):
        """
        Establish a coaxial channel connection

        bidirectional=True: also create the reverse direction channel
        """
        self.channels[(source, target)] = CoaxialChannel(
            source_name=source,
            target_name=target,
            characteristic_impedance=impedance,
            length=length,
            attenuation_rate=attenuation_rate,
            bandwidth=bandwidth,
        )
        if bidirectional:
            self.channels[(target, source)] = CoaxialChannel(
                source_name=target,
                target_name=source,
                characteristic_impedance=impedance,
                length=length,
                attenuation_rate=attenuation_rate,
                bandwidth=bandwidth,
            )

    # ------------------------------------------------------------------
    def send(
        self, source: str, target: str, signal: ElectricalSignal
    ) -> Tuple[Optional[ElectricalSignal], Optional[TransmissionReport]]:
        """
        Send a signal through the bus

        Returns:
            (transmitted_signal, report) — if the channel does not exist, returns (None, None)
        """
        key = (source, target)
        if key not in self.channels:
            return None, None

        transmitted, report = self.channels[key].transmit(signal)
        self._transmission_log.append(report)

        # Limit log size
        if len(self._transmission_log) > 1000:
            self._transmission_log = self._transmission_log[-500:]

        return transmitted, report

    # ------------------------------------------------------------------
    def get_total_reflected_energy(self) -> float:
        """Total reflected energy accumulated across all channels (converted to system heat)"""
        return sum(ch.total_reflected_energy for ch in self.channels.values())

    # ------------------------------------------------------------------
    def get_channel_stats(self) -> Dict[str, Any]:
        """Statistics for all channels"""
        stats = {}
        for key, ch in self.channels.items():
            stats[f"{key[0]}→{key[1]}"] = ch.get_stats()
        return stats

    # ------------------------------------------------------------------
    def get_bus_summary(self) -> Dict[str, Any]:
        """Bus summary"""
        total_tx = sum(ch.total_transmissions for ch in self.channels.values())
        total_mismatch = sum(ch.impedance_mismatches for ch in self.channels.values())
        total_reflected = self.get_total_reflected_energy()
        total_transmitted = sum(ch.total_transmitted_energy for ch in self.channels.values())
        total_energy = total_reflected + total_transmitted

        return {
            "total_channels": len(self.channels),
            "total_transmissions": total_tx,
            "total_impedance_mismatches": total_mismatch,
            "mismatch_rate": round(total_mismatch / max(1, total_tx), 4),
            "total_reflected_energy": round(total_reflected, 6),
            "total_transmitted_energy": round(total_transmitted, 6),
            "bus_efficiency": round(total_transmitted / max(1e-10, total_energy), 4),
            "recent_reports": [r.to_dict() for r in self._transmission_log[-5:]],
        }

    # ------------------------------------------------------------------
    @classmethod
    def create_default_topology(cls) -> "SignalBus":
        """
        Create the default 4-brain-region topology

        Coaxial cable specifications:
          75Ω  = standard video cable (high quality, low loss)
          50Ω  = standard RF cable (optimal power transmission)
          110Ω = differential pair (digital communication, anti-interference)
          60Ω  = low impedance fast channel

        Brain region impedance preferences:
          Somatosensory = 50Ω  (receiver, low impedance, open reception)
          Prefrontal    = 75Ω  (standard processing, balanced)
          Limbic        = 110Ω (high impedance, emotional protection/filtering)
          Motor         = 75Ω  (standard output)
        """
        bus = cls()

        # Sensory ↔ Prefrontal : primary cognitive pathway, low loss
        bus.connect("somatosensory", "prefrontal", impedance=75.0, length=1.0,
                    attenuation_rate=0.015, bandwidth=100.0)

        # Sensory ↔ Limbic : emotional fast pathway, short distance
        bus.connect("somatosensory", "limbic", impedance=50.0, length=0.6,
                    attenuation_rate=0.01, bandwidth=80.0)

        # Prefrontal ↔ Motor : cognitive-motor commands
        bus.connect("prefrontal", "motor", impedance=75.0, length=0.8,
                    attenuation_rate=0.02, bandwidth=100.0)

        # Limbic ↔ Motor : emotion-driven behavior, high impedance (requires strong signal to drive)
        bus.connect("limbic", "motor", impedance=110.0, length=1.2,
                    attenuation_rate=0.03, bandwidth=60.0)

        # Prefrontal ↔ Limbic : cognitive-emotional regulation (prefrontal inhibits emotions)
        bus.connect("prefrontal", "limbic", impedance=90.0, length=0.7,
                    attenuation_rate=0.025, bandwidth=80.0)

        # Sensory ↔ Motor : reflex arc, shortest path
        bus.connect("somatosensory", "motor", impedance=60.0, length=0.4,
                    attenuation_rate=0.008, bandwidth=120.0)

        return bus


# ============================================================================
# 6. Brain Region Impedance Reference Table
# ============================================================================


# Native impedance of each brain region — characteristic impedance for signals within this region
REGION_IMPEDANCE: Dict[str, float] = {
    "somatosensory": 50.0,   # Low impedance → open reception
    "prefrontal": 75.0,      # Standard → balanced processing
    "limbic": 110.0,         # High impedance → emotional protection
    "motor": 75.0,           # Standard → output execution
}

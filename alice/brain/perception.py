# -*- coding: utf-8 -*-
"""
Perception Pipeline — Physics-Driven

Core philosophy (derived from user insight):
  "Eyes, ears, touch — fundamentally they are forward engineering.
   The brain cannot re-analyze them; instead it reverse-engineers them into electrical signals.
   It extracts high-frequency, low-frequency, or specific frequencies and converts them to electrical signals.
   The CPU then filters and selects certain frequency signals for attention locking,
   and that IS the electrical signal of the apple."

Coaxial cable metaphor:
  - Camera (sensory organ) = forward engineering: light → electrical signal
  - Coaxial cable (neural pathway) = physical transmission, no computation
  - TV tuner (CPU) = channel selection = attention
  - Channel content = concept (apple, car...)

Modern AI approach:
  FFT O(n log n) → Feature extraction O(n) → Matrix operations O(n²) → Similarity computation O(n)
  ← Computationally intensive, requires GPU

This system's approach:
  Physical resonance O(1) → Frequency comparison O(1) → Channel selection O(1) → Sparse code lookup O(1)
  ← Full pipeline O(1), physics determines everything

Physical model:
  1. Sensory organs (forward engineering)
     Light/Sound/Pressure → ElectricalSignal
     = ElectricalSignal.from_raw(), already exists

  2. Coaxial transmission (physical channel)
     Signal propagates along cable → impedance matching determines attenuation
     = SignalBus + CoaxialChannel, already exists

  3. Left/right brain tuner (physical reverse engineering)
     Left brain LC circuit tuned to β/γ (detail/binding)
     Right brain LC circuit tuned to δ/θ/α (background/memory/contour)
     = Lorentzian resonance curve, O(1)

  4. CPU frequency selection (attention locking)
     Compare left/right brain resonance strengths → pick strongest → lock
     = One max() operation

  5. Sparse coding concept memory
     Frequency space logarithmically quantized into N bins (mimicking cochlear tonotopic map)
     Concept = activation of a specific bin (N-dimensional sparse code, only 1 bit is 1)
     Recognition = bin lookup, O(1)

  6. Cross-modal binding (same-frequency resonance)
     Apple frequency on visual channel = apple frequency on auditory channel
     Binding = comparing two numbers

Frequency band × information layer (physical correspondence, not part of computation):
  δ (0.5-4 Hz)   : Background/environmental baseline (illumination, scene structure)
  θ (4-8 Hz)     : Memory cues (déjà vu, familiarity)
  α (8-13 Hz)    : Contours/boundaries (object segmentation)
  β (13-30 Hz)   : Details/textures (surface features, text)
  γ (30-100 Hz)  : Cross-region binding (object recognition, attention focus)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from alice.brain.attention_plasticity import AttentionPlasticityEngine

from alice.core.signal import (
    BrainWaveBand,
    ElectricalSignal,
)


# ============================================================================
# Frequency band × information layer (physical constants, for reference)
# ============================================================================

BAND_INFO_LAYER: Dict[BrainWaveBand, str] = {
    BrainWaveBand.DELTA: "background",   # Background/environmental baseline
    BrainWaveBand.THETA: "memory_cue",   # Memory cues/familiarity
    BrainWaveBand.ALPHA: "contour",      # Contours/boundaries
    BrainWaveBand.BETA:  "detail",       # Details/textures
    BrainWaveBand.GAMMA: "binding",      # Cross-region binding/attention focus
}


# ============================================================================
# 1. Physical Tuner — LC Resonance Circuit
# ============================================================================


class PhysicalTuner:
    """
    Physical tuner — simulates an LC resonance circuit

    Like a radio's tuning knob:
    - Tune to a frequency → that frequency's signal is amplified (resonance)
    - Other frequencies naturally attenuate (physical property, not computational suppression)
    - Resonance strength is determined by the Lorentzian curve

    Quality factor Q:
    - High Q → narrowband (precisely locks onto a single frequency)
    - Low Q → wideband (receives neighboring frequencies)

    Computational cost: O(1) — one division
    """

    def __init__(
        self,
        name: str,
        default_band: BrainWaveBand = BrainWaveBand.ALPHA,
        quality_factor: float = 2.0,
    ):
        self.name = name
        self.tuned_band = default_band
        self.q = quality_factor
        self.lock_count: int = 0

    # ------------------------------------------------------------------
    def tune(self, band: BrainWaveBand):
        """Turn the tuning knob — switch the attended frequency band"""
        self.tuned_band = band

    # ------------------------------------------------------------------
    def resonate(self, signal: ElectricalSignal) -> Tuple[float, bool]:
        """
        Physical resonance: does the signal frequency match the tuned frequency?

        Lorentzian resonance curve:
          L(f) = 1 / (1 + Q² × ((f/f₀) - (f₀/f))²)

        Returns:
          (resonance_strength, is_locked)
          - resonance_strength : 0.0~1.0 resonance degree
          - is_locked          : True if resonance is strong enough

        Computational cost: O(1)
        """
        f_signal = signal.frequency
        f_tuned = self.tuned_band.center_freq

        if f_tuned < 0.01 or f_signal < 0.01:
            return 0.0, False

        # Lorentzian resonance
        ratio = f_signal / f_tuned
        detuning = ratio - 1.0 / ratio
        strength = 1.0 / (1.0 + self.q ** 2 * detuning ** 2)

        locked = strength > 0.3
        if locked:
            self.lock_count += 1

        return round(strength, 6), locked

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tuned_band": self.tuned_band.value,
            "quality_factor": self.q,
            "lock_count": self.lock_count,
        }


# ============================================================================
# 2. Concept Memory — Sparse Coding
# ============================================================================


class ConceptMemory:
    """
    Concept memory — Sparse Coding

    The brain's concept encoding is sparse coding, not dense vectors.

    Modern AI (dense coding):
      apple = [0.23, 0.87, -0.45, 0.12, ..., 0.66]  ← 1024 dims, every dim has a value
      Recognition = vector dot product O(d), d = number of dimensions

    Brain (sparse coding):
      apple = [0, 0, 0, ..., 1, ..., 0, 0]  ← N dims, only 1 bit is 1
      Recognition = which bin lit up? O(1)

    Physical model:
      - N frequency bins ≈ N tuned neurons
      - Each neuron only resonates to a specific frequency (Lorentzian curve)
      - Signal arrives → only the matching neuron fires (sparse activation)
      - Read out the firing neuron → concept recognition (no decoding computation)

    Frequency space quantization (mimicking cochlear tonotopic mapping):
      - Logarithmic binning, 12 bins/octave (equal temperament)
      - 0.5-100 Hz → ~97 bins
      - Each concept occupies 1 bin → sparsity ≈ 1%

    Complexity:
      Old version: scan all concepts one by one → O(n)
      Sparse version: freq → bin_id → lookup (±2 bins) → O(1)
    """

    # --- Frequency space quantization parameters ---
    BINS_PER_OCTAVE: int = 12       # 12 bins per octave (equal temperament)
    FREQ_MIN: float = 0.5           # Lowest frequency (δ lower bound)
    FREQ_MAX: float = 100.0         # Highest frequency (γ upper bound)
    BIN_SEARCH_RADIUS: int = 2      # Tolerance ±2 bins ≈ ±12%

    def __init__(self):
        # concept_label → {modality → characteristic_frequency}
        self._memory: Dict[str, Dict[str, float]] = {}

        # Sparse code index: modality → {bin_id → (label, frequency)}
        # A bin is a "tuned neuron" — most bins are empty (sparse!)
        self._sparse_bins: Dict[str, Dict[int, Tuple[str, float]]] = {}

        # Total bin count (number of "neurons" in frequency space)
        self.total_bins: int = int(math.ceil(
            self.BINS_PER_OCTAVE * math.log2(self.FREQ_MAX / self.FREQ_MIN)
        )) + 1  # ~97

    # ------------------------------------------------------------------
    def _freq_to_bin(self, freq: float) -> int:
        """Frequency → bin index (logarithmic quantization, mimicking cochlea)"""
        if freq <= self.FREQ_MIN:
            return 0
        if freq >= self.FREQ_MAX:
            return self.total_bins - 1
        return int(round(
            self.BINS_PER_OCTAVE * math.log2(freq / self.FREQ_MIN)
        ))

    def _bin_to_freq(self, bin_id: int) -> float:
        """Bin index → center frequency"""
        return self.FREQ_MIN * (2.0 ** (bin_id / self.BINS_PER_OCTAVE))

    # ------------------------------------------------------------------
    def register(self, label: str, modality: str, frequency: float):
        """
        Register a concept's sparse code

        Frequency → logarithmic bin → write to sparse index
        One bin = one tuned neuron = one concept's "address"

        Computational cost: O(1) — one log + hash
        """
        if label not in self._memory:
            self._memory[label] = {}
        self._memory[label][modality] = frequency

        if modality not in self._sparse_bins:
            self._sparse_bins[modality] = {}
        bin_id = self._freq_to_bin(frequency)
        self._sparse_bins[modality][bin_id] = (label, frequency)

    # ------------------------------------------------------------------
    def identify(
        self,
        frequency: float,
        modality: str,
        tolerance: float = 0.15,
    ) -> Optional[str]:
        """
        Sparse code recognition — which bin lit up?

        freq → bin_id → check ±2 neighboring bins → find nearest concept

        Computational cost: O(5) = O(1) constant time
        (Old O(n) sequential scan has been eliminated)
        """
        if modality not in self._sparse_bins:
            return None
        if frequency < 0.01:
            return None

        center_bin = self._freq_to_bin(frequency)
        bins = self._sparse_bins[modality]

        best_label = None
        best_deviation = tolerance

        # Only check ±2 neighboring bins — sparse index physical search
        for offset in range(
            -self.BIN_SEARCH_RADIUS, self.BIN_SEARCH_RADIUS + 1
        ):
            check_bin = center_bin + offset
            if check_bin in bins:
                label, stored_freq = bins[check_bin]
                if stored_freq < 0.01:
                    continue
                deviation = abs(frequency - stored_freq) / stored_freq
                if deviation < best_deviation:
                    best_deviation = deviation
                    best_label = label

        return best_label

    # ------------------------------------------------------------------
    def find_cross_modal(
        self,
        label: str,
        source_modality: str,
    ) -> List[str]:
        """
        Cross-modal binding — find the same concept's existence in other modalities

        Computational cost: O(m), m = number of modalities (at most 4)
        """
        if label not in self._memory:
            return []
        return [
            mod for mod in self._memory[label] if mod != source_modality
        ]

    # ------------------------------------------------------------------
    def encode(self, label: str, modality: str) -> Optional[np.ndarray]:
        """
        Get the sparse code vector for a concept

        Returns:
          A uint8 array of length N, with only 1 position set to 1
          e.g.: apple@visual → [0,0,...,0,1,0,...,0] (bin 53)

        This is the brain's sparse coding —
        only 1 out of 97 neurons is firing.
        """
        if label not in self._memory or modality not in self._memory[label]:
            return None
        freq = self._memory[label][modality]
        code = np.zeros(self.total_bins, dtype=np.uint8)
        code[self._freq_to_bin(freq)] = 1
        return code

    # ------------------------------------------------------------------
    def get_sparsity(self, modality: str) -> Dict[str, Any]:
        """
        Sparsity statistics

        population_sparsity: proportion of occupied bins
          - Lower = sparser (brain V1 ≈ 1%, this system ≈ 1%)
          - 0.01 = only 1 out of 97 bins is occupied
        """
        if modality not in self._sparse_bins:
            return {
                "population_sparsity": 0.0,
                "occupied_bins": 0,
                "total_bins": self.total_bins,
            }
        occupied = len(self._sparse_bins[modality])
        return {
            "population_sparsity": round(occupied / self.total_bins, 4),
            "occupied_bins": occupied,
            "total_bins": self.total_bins,
        }

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "registered_concepts": len(self._memory),
            "modalities": list(self._sparse_bins.keys()),
            "total_bins": self.total_bins,
            "encoding": "sparse",
            "sparsity": {
                mod: self.get_sparsity(mod)
                for mod in self._sparse_bins
            },
        }


# ============================================================================
# 3. Perception Result
# ============================================================================


@dataclass
class PerceptionResult:
    """
    Physical perception result

    Contains no statistical features, matrices, or FFT results.
    Only physical quantities: frequency, resonance strength, lock state.
    """

    # Signal's natural properties (output of sensory organ forward engineering)
    signal_band: BrainWaveBand
    signal_frequency: float

    # Left brain tuning result (high-frequency preference — detail/binding)
    left_resonance: float
    left_locked: bool
    left_tuned_band: BrainWaveBand

    # Right brain tuning result (low-frequency preference — background/memory/contour)
    right_resonance: float
    right_locked: bool
    right_tuned_band: BrainWaveBand

    # CPU attention lock
    attention_band: BrainWaveBand
    attention_strength: float

    # Concept recognition
    concept: Optional[str] = None

    # Cross-modal binding
    bindings: List[str] = field(default_factory=list)

    # Integrated electrical signal (resonance-amplified — sent to FusionBrain)
    integrated_signal: Optional[ElectricalSignal] = None

    # Performance
    pipeline_elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_band": self.signal_band.value,
            "signal_frequency": round(self.signal_frequency, 2),
            "left_resonance": round(self.left_resonance, 4),
            "left_locked": self.left_locked,
            "left_tuned_band": self.left_tuned_band.value,
            "right_resonance": round(self.right_resonance, 4),
            "right_locked": self.right_locked,
            "right_tuned_band": self.right_tuned_band.value,
            "attention_band": self.attention_band.value,
            "attention_strength": round(self.attention_strength, 4),
            "concept": self.concept,
            "bindings": self.bindings,
            "pipeline_elapsed_ms": round(self.pipeline_elapsed_ms, 4),
        }


# ============================================================================
# 4. Perception Pipeline — Physics-Driven
# ============================================================================


class PerceptionPipeline:
    """
    Physical perception pipeline

    Flow:
      Signal arrives (sensory organ forward engineering output — already ElectricalSignal)
        ↓
      Left brain LC resonance (β/γ)  Right brain LC resonance (δ/θ/α)  ← O(1) physical resonance
        ↓
      CPU selects strongest resonance → attention lock              ← O(1) comparison
        ↓
      Concept recognition (sparse code bin lookup)              ← O(1) sparse lookup
        ↓
      Cross-modal binding (same frequency, same concept)                 ← O(m) traversal
        ↓
      Integrated signal (resonance amplification) → FusionBrain
    """

    # Left brain scan bands (high frequency — detail, language, logic)
    LEFT_BRAIN_BANDS = [BrainWaveBand.BETA, BrainWaveBand.GAMMA]

    # Right brain scan bands (low frequency — spatial, emotional, holistic)
    RIGHT_BRAIN_BANDS = [
        BrainWaveBand.DELTA,
        BrainWaveBand.THETA,
        BrainWaveBand.ALPHA,
    ]

    def __init__(self):
        # Left brain tuner (Q=3, more precise — logic demands accuracy)
        self.left_tuner = PhysicalTuner(
            "left_brain", BrainWaveBand.BETA, quality_factor=3.0,
        )
        # Right brain tuner (Q=2, wider band — holistic perception doesn't need precision)
        self.right_tuner = PhysicalTuner(
            "right_brain", BrainWaveBand.ALPHA, quality_factor=2.0,
        )
        # Concept memory
        self.concepts = ConceptMemory()

        # Attention plasticity engine (optional external injection)
        self._plasticity: Optional["AttentionPlasticityEngine"] = None

        # Statistics
        self.total_perceptions: int = 0
        self.total_locks: int = 0
        self.total_concepts_identified: int = 0

    def set_plasticity_engine(self, engine: "AttentionPlasticityEngine"):
        """Inject attention plasticity engine — makes tuner Q trainable"""
        self._plasticity = engine

    # ------------------------------------------------------------------
    def perceive(
        self,
        signal: "np.ndarray | ElectricalSignal",
        modality: str = "visual",
    ) -> PerceptionResult:
        """
        Physical perception

        Computational cost: O(1) + O(n_concepts)
        """
        t0 = time.time()
        self.total_perceptions += 1

        # === 1. Forward engineering (sensory organ already done — only format unification) ===
        if isinstance(signal, ElectricalSignal):
            esig = signal
        else:
            esig = ElectricalSignal.from_raw(
                signal, source="external", modality=modality,
            )

        signal_band = esig.band
        signal_freq = esig.frequency

        # === 2. Left/right brain physical tuning (reverse engineering = LC resonance) ===
        # If plasticity engine exists, dynamically adjust Q
        if self._plasticity is not None:
            trained_q = self._plasticity.get_tuner_q(modality)
            # Left brain Q baseline is higher (baseline difference +1.0 preserved)
            self.left_tuner.q = trained_q + 1.0
            self.right_tuner.q = trained_q

        # Left brain: scan β/γ, find strongest resonance
        left_best_strength = 0.0
        left_best_band = self.LEFT_BRAIN_BANDS[0]
        left_locked = False

        for band in self.LEFT_BRAIN_BANDS:
            self.left_tuner.tune(band)
            strength, locked = self.left_tuner.resonate(esig)
            if strength > left_best_strength:
                left_best_strength = strength
                left_best_band = band
                left_locked = locked

        # Right brain: scan δ/θ/α, find strongest resonance
        right_best_strength = 0.0
        right_best_band = self.RIGHT_BRAIN_BANDS[0]
        right_locked = False

        for band in self.RIGHT_BRAIN_BANDS:
            self.right_tuner.tune(band)
            strength, locked = self.right_tuner.resonate(esig)
            if strength > right_best_strength:
                right_best_strength = strength
                right_best_band = band
                right_locked = locked

        # === 3. CPU frequency selection — attention lock ===
        if left_best_strength >= right_best_strength:
            attention_band = left_best_band
            attention_strength = left_best_strength
        else:
            attention_band = right_best_band
            attention_strength = right_best_strength

        if attention_strength > 0.3:
            self.total_locks += 1
            if self._plasticity is not None:
                self._plasticity.on_successful_lock(modality)

        # === 4. Concept recognition — frequency matching ===
        concept = self.concepts.identify(signal_freq, modality)
        if concept:
            self.total_concepts_identified += 1
            if self._plasticity is not None:
                self._plasticity.on_successful_identification(modality)

        # === 5. Cross-modal binding — same-frequency resonance ===
        bindings: List[str] = []
        if concept:
            bindings = self.concepts.find_cross_modal(concept, modality)

        # === 6. Integrated signal — resonance amplification ===
        #   Natural gain of LC circuit at resonance, not matrix multiplication
        resonance_gain = 1.0 + attention_strength  # Max 2x
        amplified_waveform = esig.waveform * resonance_gain

        integrated = ElectricalSignal(
            waveform=amplified_waveform,
            amplitude=esig.amplitude * resonance_gain,
            frequency=esig.frequency,
            phase=esig.phase,
            impedance=esig.impedance,
            snr=esig.snr,
            timestamp=esig.timestamp,
            source=esig.source,
            modality=esig.modality,
        )

        elapsed_ms = (time.time() - t0) * 1000

        return PerceptionResult(
            signal_band=signal_band,
            signal_frequency=signal_freq,
            left_resonance=left_best_strength,
            left_locked=left_locked,
            left_tuned_band=left_best_band,
            right_resonance=right_best_strength,
            right_locked=right_locked,
            right_tuned_band=right_best_band,
            attention_band=attention_band,
            attention_strength=attention_strength,
            concept=concept,
            bindings=bindings,
            integrated_signal=integrated,
            pipeline_elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_perceptions": self.total_perceptions,
            "total_locks": self.total_locks,
            "total_concepts_identified": self.total_concepts_identified,
            "lock_rate": round(
                self.total_locks / max(1, self.total_perceptions), 3
            ),
            "concept_memory": self.concepts.get_stats(),
        }

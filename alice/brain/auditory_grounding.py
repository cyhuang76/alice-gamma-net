# -*- coding: utf-8 -*-
"""
Auditory Grounding Engine — Cross-Modal Hebbian Resonance Wiring

Core philosophy: Language is remote resonance control

  When I say "apple" to you, I am not transmitting a token.
  I use sound waves (physical vibrations) to force your auditory cortex
  to produce specific frequencies, which then via resonance "light up"
  the neural circuits in your brain associated with red, round, and sweet.

  The essence of language = remote impedance control of another brain via sound waves.

Physical model:

  1. Pavlovian conditioning = cross-modal Hebbian wiring
     Bell (auditory) + food (visual) → simultaneous activation
     → Low-impedance channel established between auditory and visual cortex
     → Γ_cross ↓

  2. Impedance model of cross-modal synapses
     Z_synapse = Z_0 / w            (w = synaptic strength)
     Γ_cross = (Z_a - Z_v) / (Z_a + Z_v)
     Energy transfer = 1 - |Γ_cross|²

     Strong conditioning → w ↑ → Z_synapse ↓ → Γ ≈ 0 → energy flows freely
     → Hear bell → visual cortex "sees" food (phantom/association)

  3. Extinction = synaptic decay
     No co-activation → w *= decay_rate → Z ↑ → Γ ↑
     → Hear bell → no longer associates with food

  4. Spectral fingerprint = physical representation of a concept
     Each sound → CochlearFilterBank → 24-channel activation vector
     This vector is the physical identity of "this kind of sound"
     Two similar sound fingerprints → same concept
     Fingerprints of different modalities → cross-modal resonance keys

Equations:

  Conditioning strengthening: Δw = η × pre × post × temporal_window
  Extinction decay: w(t+1) = w(t) × (1 - λ)
  Phantom activation: echo = signal × (1 - |Γ_cross|²)
  Energy transfer: E_transfer = E_source × (1 - |Γ_cross|²)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.body.cochlea import (
    CochlearFilterBank,
    TonotopicActivation,
    generate_tone,
)
from alice.core.signal import ElectricalSignal


# ============================================================================
# Physical constants
# ============================================================================

# --- Cross-modal synapses ---
SYNAPSE_Z0 = 100.0             # Base impedance (Ω) (unconditioned)
SYNAPSE_LEARNING_RATE = 0.15   # Hebbian learning rate η
SYNAPSE_DECAY_RATE = 0.02      # Extinction decay rate λ (per tick)
SYNAPSE_MAX_STRENGTH = 5.0     # Maximum synaptic strength
SYNAPSE_FLOOR = 0.01           # Minimum synaptic strength

# --- Temporal window ---
TEMPORAL_WINDOW_MS = 200.0     # Co-activation detection window (ms)
                               # 200ms ≈ biological cross-modal binding window

# --- Phantom activation ---
ECHO_THRESHOLD = 0.3           # Echo strength > this to generate phantom
ECHO_IMPEDANCE = 50.0          # Phantom signal impedance (Ω)

# --- Concept prototypes ---
PROTOTYPE_MERGE_RATE = 0.1     # Prototype update rate (exponential moving average)
SIMILARITY_THRESHOLD = 0.7     # Fingerprint similarity > this = same concept


# ============================================================================
# Cross-modal synapse
# ============================================================================


@dataclass
class CrossModalSynapse:
    """
    Cross-modal synapse — connection between neural patterns of two different modalities.

    Physical model:
      A "cable" connecting a pattern in auditory cortex to a pattern in visual cortex.
      Cable impedance Z = Z_0 / w (higher strength → lower impedance → better energy transfer).

      Γ_cross = (Z_source - Z_synapse) / (Z_source + Z_synapse)
      Energy transfer = 1 - |Γ_cross|²

    Biological correspondence:
      Co-activation of "place cells × event cells" in hippocampus
      → Forms contextual binding
    """

    # Endpoint identity
    source_modality: str                 # "auditory"
    target_modality: str                 # "visual"
    source_fingerprint: np.ndarray       # Source fingerprint (24 channels)
    target_fingerprint: np.ndarray       # Target fingerprint (24 channels or feature vec)

    # Synaptic physical quantities
    strength: float = SYNAPSE_FLOOR      # Synaptic strength w
    z_impedance: float = SYNAPSE_Z0      # Z = Z_0 / w

    # Statistics
    creation_time: float = 0.0
    last_activated: float = 0.0
    activation_count: int = 0
    total_energy_transferred: float = 0.0

    def gamma(self, z_source: float = ECHO_IMPEDANCE) -> float:
        """Compute reflection coefficient Γ."""
        return (z_source - self.z_impedance) / (z_source + self.z_impedance)

    def energy_transfer(self, z_source: float = ECHO_IMPEDANCE) -> float:
        """Energy transfer rate = 1 - |Γ|²."""
        g = self.gamma(z_source)
        return 1.0 - g * g

    def strengthen(self, pre_activation: float, post_activation: float,
                   temporal_overlap: float = 1.0):
        """
        Hebbian strengthening: Δw = η × pre × post × temporal_window

        pre  = Source modality (auditory) activation strength
        post = Target modality (visual) activation strength
        temporal_overlap = Degree of temporal overlap (0=no overlap, 1=perfect sync)
        """
        delta_w = (
            SYNAPSE_LEARNING_RATE
            * pre_activation
            * post_activation
            * temporal_overlap
        )
        self.strength = min(SYNAPSE_MAX_STRENGTH, self.strength + delta_w)
        self._update_impedance()
        self.activation_count += 1
        self.last_activated = time.monotonic()

    def decay(self):
        """Extinction: w *= (1 - λ)."""
        self.strength *= (1.0 - SYNAPSE_DECAY_RATE)
        self.strength = max(SYNAPSE_FLOOR, self.strength)
        self._update_impedance()

    def _update_impedance(self):
        """Z = Z_0 / w."""
        self.z_impedance = SYNAPSE_Z0 / max(self.strength, 1e-9)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_modality": self.source_modality,
            "target_modality": self.target_modality,
            "strength": round(self.strength, 4),
            "z_impedance": round(self.z_impedance, 2),
            "gamma": round(self.gamma(), 4),
            "energy_transfer": round(self.energy_transfer(), 4),
            "activation_count": self.activation_count,
        }


# ============================================================================
# Sensory pattern prototype
# ============================================================================


@dataclass
class SensoryPrototype:
    """
    Sensory pattern prototype — the average spectral fingerprint of a "concept".

    Physical meaning:
      Repeated exposure to the same stimulus → fingerprint gradually stabilizes
      → Forms a "concept" = an attractor in the brain's state space.

    Example:
      Repeatedly hearing a 440Hz tuning fork → prototype fingerprint solidifies
      → Henceforth, any similar spectrum → resonance match → recognized as "that tone"
    """
    label: str                                 # Concept label
    modality: str                              # "auditory" / "visual" / ...
    fingerprint: np.ndarray                    # Average fingerprint (24 channels)
    exposure_count: int = 0                    # Exposure count
    confidence: float = 0.0                    # Confidence (0~1)
    creation_time: float = 0.0

    def update(self, new_fingerprint: np.ndarray):
        """Exponential moving average — gradually approaches the true prototype."""
        alpha = PROTOTYPE_MERGE_RATE
        self.fingerprint = (1.0 - alpha) * self.fingerprint + alpha * new_fingerprint
        self.exposure_count += 1
        # Confidence = 1 - 1/(1+count)
        self.confidence = 1.0 - 1.0 / (1.0 + self.exposure_count)

    def similarity(self, fingerprint: np.ndarray) -> float:
        """Cosine similarity with a given fingerprint."""
        dot = float(np.dot(self.fingerprint, fingerprint))
        n1 = float(np.linalg.norm(self.fingerprint))
        n2 = float(np.linalg.norm(fingerprint))
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return dot / (n1 * n2)


# ============================================================================
# Cross-modal Hebbian network
# ============================================================================


class CrossModalHebbianNetwork:
    """
    Cross-modal Hebbian network — manages all cross-modal synapses.

    Physical model:
      This is the "cable bundle" from auditory cortex to other cortices.
      Each cable connects an auditory pattern to a pattern in another modality.
      Co-activation → cable impedance decreases → energy flows freely.

    Pavlovian conditioning = Hebbian learning in this network.
    """

    def __init__(self, max_synapses: int = 200):
        self.synapses: List[CrossModalSynapse] = []
        self.max_synapses = max_synapses

        # Concept prototype library
        self.prototypes: Dict[str, List[SensoryPrototype]] = {}
        # key = modality, value = list of prototypes

        # Statistics
        self.total_pairings = 0
        self.total_echoes = 0
        self.total_extinctions = 0

    # ------------------------------------------------------------------
    def condition(
        self,
        source_fp: np.ndarray,
        source_modality: str,
        target_fp: np.ndarray,
        target_modality: str,
        source_activation: float = 1.0,
        target_activation: float = 1.0,
        temporal_overlap: float = 1.0,
    ) -> CrossModalSynapse:
        """
        Conditioning — co-activation → create/strengthen cross-modal synapse.

        Physical: Two cortical areas activate simultaneously
        → Impedance of the cable between them decreases (Hebbian wiring).

        Args:
            source_fp: Source fingerprint (e.g., auditory tonotopic activation)
            source_modality: "auditory"
            target_fp: Target fingerprint (e.g., visual feature vector)
            target_modality: "visual"
            source_activation: Source activation strength (0~1)
            target_activation: Target activation strength (0~1)
            temporal_overlap: Temporal overlap (0~1)

        Returns:
            The strengthened CrossModalSynapse
        """
        self.total_pairings += 1

        # Find existing synapse (fingerprint match)
        synapse = self._find_synapse(source_fp, source_modality,
                                      target_fp, target_modality)

        if synapse is None:
            # Create new synapse
            synapse = CrossModalSynapse(
                source_modality=source_modality,
                target_modality=target_modality,
                source_fingerprint=source_fp.copy(),
                target_fingerprint=target_fp.copy(),
                strength=SYNAPSE_FLOOR,
                z_impedance=SYNAPSE_Z0 / SYNAPSE_FLOOR,
                creation_time=time.monotonic(),
            )
            self.synapses.append(synapse)

            # Capacity management: remove weakest synapse
            if len(self.synapses) > self.max_synapses:
                self.synapses.sort(key=lambda s: s.strength)
                self.synapses.pop(0)

        # Hebbian strengthening
        synapse.strengthen(source_activation, target_activation, temporal_overlap)

        return synapse

    # ------------------------------------------------------------------
    def probe(
        self,
        source_fp: np.ndarray,
        source_modality: str,
    ) -> List[Dict[str, Any]]:
        """
        Probe — given a source fingerprint, query all cross-modal associations.

        Physical: Send signal into the auditory end → through cross-modal cables
        → How much energy reaches the target end?

        Returns:
            list of {
                "target_modality": str,
                "target_fingerprint": ndarray,
                "energy_transfer": float,  # 0~1
                "gamma": float,            # Reflection coefficient
                "synapse_strength": float,
                "echo_strength": float,    # Phantom activation strength
            }
        """
        echoes = []
        for syn in self.synapses:
            if syn.source_modality != source_modality:
                continue

            # Check fingerprint similarity
            sim = self._fingerprint_similarity(source_fp, syn.source_fingerprint)
            if sim < SIMILARITY_THRESHOLD:
                continue

            # Compute energy transfer
            et = syn.energy_transfer()
            echo_strength = sim * et  # Similarity × transfer rate

            if echo_strength > 0.01:
                echoes.append({
                    "target_modality": syn.target_modality,
                    "target_fingerprint": syn.target_fingerprint.copy(),
                    "energy_transfer": et,
                    "gamma": syn.gamma(),
                    "synapse_strength": syn.strength,
                    "echo_strength": echo_strength,
                    "activation_count": syn.activation_count,
                })

        # Sort by echo strength descending
        echoes.sort(key=lambda e: e["echo_strength"], reverse=True)

        if echoes:
            self.total_echoes += 1

        return echoes

    # ------------------------------------------------------------------
    def generate_echo_signal(
        self,
        echoes: List[Dict[str, Any]],
    ) -> Optional[ElectricalSignal]:
        """
        Convert cross-modal associations to ElectricalSignal (phantom signal).

        Physical:
          Hear bell → cross-modal cable transfers energy to visual cortex
          → Visual cortex generates a "phantom" signal = seeing non-existent food

          echo_amplitude = source_strength × energy_transfer
        """
        if not echoes or echoes[0]["echo_strength"] < ECHO_THRESHOLD:
            return None

        best = echoes[0]
        target_fp = best["target_fingerprint"]

        # Reconstruct waveform from target fingerprint
        # Physical: map tonotopic fingerprint back to brainwave frequency
        amplitude = float(best["echo_strength"])
        # Dominant frequency = strongest fingerprint channel mapped to brainwave band
        if len(target_fp) > 0:
            max_idx = int(np.argmax(target_fp))
            # Map to 0.5~100 Hz brainwave frequency
            freq = 0.5 + (max_idx / max(len(target_fp) - 1, 1)) * 99.5
        else:
            freq = 10.0

        # Simplified waveform (sinusoidal carrier)
        n_samples = 256
        t = np.linspace(0, 1.0, n_samples)
        waveform = amplitude * np.sin(2 * np.pi * freq * t)

        return ElectricalSignal(
            waveform=waveform,
            amplitude=amplitude,
            frequency=freq,
            phase=0.0,
            impedance=ECHO_IMPEDANCE,
            snr=amplitude * 10.0,  # SNR ∝ echo strength
            source="cross_modal_echo",
            modality=best["target_modality"],
        )

    # ------------------------------------------------------------------
    def decay_all(self):
        """Global extinction — all synapses decay."""
        for syn in self.synapses:
            syn.decay()

        # Remove near-dead synapses
        before = len(self.synapses)
        self.synapses = [
            s for s in self.synapses
            if s.strength > SYNAPSE_FLOOR * 1.1
        ]
        extinct = before - len(self.synapses)
        self.total_extinctions += extinct

    # ------------------------------------------------------------------
    def register_prototype(
        self,
        label: str,
        modality: str,
        fingerprint: np.ndarray,
    ) -> SensoryPrototype:
        """
        Register/update sensory prototype.

        Repeated exposure → prototype stabilizes → concept formation
        """
        if modality not in self.prototypes:
            self.prototypes[modality] = []

        # Find existing prototype
        for proto in self.prototypes[modality]:
            if proto.label == label:
                proto.update(fingerprint)
                return proto

        # Create new prototype
        proto = SensoryPrototype(
            label=label,
            modality=modality,
            fingerprint=fingerprint.copy(),
            exposure_count=1,
            confidence=0.5,
            creation_time=time.monotonic(),
        )
        self.prototypes[modality].append(proto)
        return proto

    # ------------------------------------------------------------------
    def identify(
        self,
        fingerprint: np.ndarray,
        modality: str,
    ) -> Optional[Tuple[str, float]]:
        """
        Use fingerprint to query the best-matching concept prototype.

        Physical: Input fingerprint → resonance comparison with all prototypes
        → Strongest resonance = best-matching concept.

        Returns:
            (label, similarity) or None
        """
        if modality not in self.prototypes:
            return None

        best_label = None
        best_sim = 0.0
        for proto in self.prototypes[modality]:
            sim = proto.similarity(fingerprint)
            if sim > best_sim and sim > SIMILARITY_THRESHOLD:
                best_sim = sim
                best_label = proto.label

        if best_label is not None:
            return (best_label, best_sim)
        return None

    # ------------------------------------------------------------------
    def _find_synapse(
        self,
        source_fp: np.ndarray,
        source_mod: str,
        target_fp: np.ndarray,
        target_mod: str,
    ) -> Optional[CrossModalSynapse]:
        """Find a matching existing synapse."""
        for syn in self.synapses:
            if syn.source_modality != source_mod:
                continue
            if syn.target_modality != target_mod:
                continue
            src_sim = self._fingerprint_similarity(source_fp, syn.source_fingerprint)
            tgt_sim = self._fingerprint_similarity(target_fp, syn.target_fingerprint)
            if src_sim > SIMILARITY_THRESHOLD and tgt_sim > SIMILARITY_THRESHOLD:
                return syn
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _fingerprint_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Fingerprint cosine similarity."""
        dot = float(np.dot(fp1, fp2))
        n1 = float(np.linalg.norm(fp1))
        n2 = float(np.linalg.norm(fp2))
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return dot / (n1 * n2)

    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "n_synapses": len(self.synapses),
            "total_pairings": self.total_pairings,
            "total_echoes": self.total_echoes,
            "total_extinctions": self.total_extinctions,
            "prototypes": {
                mod: [
                    {"label": p.label, "confidence": round(p.confidence, 3),
                     "exposures": p.exposure_count}
                    for p in protos
                ]
                for mod, protos in self.prototypes.items()
            },
            "strongest_synapses": [
                s.to_dict()
                for s in sorted(self.synapses, key=lambda s: s.strength,
                                reverse=True)[:5]
            ],
        }


# ============================================================================
# Auditory grounding engine
# ============================================================================


class AuditoryGroundingEngine:
    """
    Auditory Grounding Engine — binds sounds to concepts.

    Core component of Phase 4.1, responsible for:

    1. Auditory analysis
       Sound pressure waveform → CochlearFilterBank → TonotopicActivation → spectral fingerprint

    2. Cross-modal conditioning (Pavlovian)
       When auditory and other modalities co-occur → Hebbian wiring
       → Next time, hearing the sound alone can "see" the corresponding image

    3. Concept formation
       Repeated exposure → prototype solidifies → concept = attractor in brain's state space

    4. Phantom generation
       Auditory → cross-modal cable → target cortex activation → phantom signal
       "Hearing the word 'apple' → seeing the shape of an apple"

    Physical equations:
      Wakefulness = Minimize Γ_ext (external matching)
      Language = Minimize Γ_cross (cross-modal matching)
      Understanding = Γ_cross → 0 (hear → perfect resonance to → see/feel)
    """

    def __init__(
        self,
        n_cochlea_channels: int = 24,
        max_synapses: int = 200,
    ):
        # Cochlea analyzer
        self.cochlea = CochlearFilterBank(n_channels=n_cochlea_channels)

        # Cross-modal Hebbian network
        self.network = CrossModalHebbianNetwork(max_synapses=max_synapses)

        # Recent sensory inputs (for temporal window comparison)
        self._recent_inputs: Dict[str, Dict[str, Any]] = {}
        # key = modality, value = {"fingerprint", "activation", "timestamp", "signal"}

        # Statistics
        self.total_groundings = 0
        self.total_echoes_generated = 0

    # ------------------------------------------------------------------
    def receive_auditory(
        self,
        sound_wave: np.ndarray,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Receive auditory input → cochlear analysis + cross-modal probing.

        Args:
            sound_wave: Sound pressure waveform
            label: Optional concept label (for prototype registration)

        Returns:
            {
                "tonotopic": TonotopicActivation,
                "fingerprint": ndarray,
                "echoes": list,        # Cross-modal associations
                "echo_signal": ElectricalSignal or None,
                "identified_as": (label, sim) or None,
            }
        """
        # Cochlear analysis
        tono = self.cochlea.analyze(sound_wave)
        fp = tono.fingerprint()

        # Store in recent inputs
        self._recent_inputs["auditory"] = {
            "fingerprint": fp,
            "activation": min(1.0, tono.total_energy / max(1.0, tono.total_energy)),
            "timestamp": time.monotonic(),
            "tonotopic": tono,
        }

        # Register prototype
        if label:
            self.network.register_prototype(label, "auditory", fp)

        # Cross-modal probe — any previously learned associations?
        echoes = self.network.probe(fp, "auditory")
        echo_signal = self.network.generate_echo_signal(echoes)

        if echo_signal is not None:
            self.total_echoes_generated += 1

        # Concept identification
        identified = self.network.identify(fp, "auditory")

        # Attempt cross-modal conditioning (if recent inputs from other modalities exist)
        self._try_conditioning("auditory", fp)

        return {
            "tonotopic": tono,
            "fingerprint": fp,
            "echoes": echoes,
            "echo_signal": echo_signal,
            "identified_as": identified,
        }

    # ------------------------------------------------------------------
    def receive_signal(
        self,
        signal: ElectricalSignal,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Receive non-auditory modality signal → extract fingerprint + attempt cross-modal conditioning.

        Args:
            signal: ElectricalSignal (visual, proprioception, etc.)
            label: Optional concept label

        Returns:
            {"modality", "fingerprint", "conditioned": bool}
        """
        modality = signal.modality
        fp = self._signal_to_fingerprint(signal)

        self._recent_inputs[modality] = {
            "fingerprint": fp,
            "activation": min(1.0, signal.amplitude),
            "timestamp": time.monotonic(),
            "signal": signal,
        }

        if label:
            self.network.register_prototype(label, modality, fp)

        conditioned = self._try_conditioning(modality, fp)

        return {
            "modality": modality,
            "fingerprint": fp,
            "conditioned": conditioned,
        }

    # ------------------------------------------------------------------
    def condition_pair(
        self,
        auditory_wave: np.ndarray,
        other_signal: ElectricalSignal,
        auditory_label: Optional[str] = None,
        other_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Forced conditioning — simultaneously provide auditory + other modality signals.

        Core API for Pavlovian conditioning:
          condition_pair(bell_wave, food_signal) × N times
          → Establish low-impedance channel from bell→food

        Returns:
            {
                "synapse": dict,
                "auditory_fingerprint": ndarray,
                "other_fingerprint": ndarray,
                "energy_transfer": float,
            }
        """
        self.total_groundings += 1

        # Auditory fingerprint
        tono = self.cochlea.analyze(auditory_wave)
        aud_fp = tono.fingerprint()

        # Other modality fingerprint
        other_fp = self._signal_to_fingerprint(other_signal)

        # Register prototypes
        if auditory_label:
            self.network.register_prototype(auditory_label, "auditory", aud_fp)
        if other_label:
            self.network.register_prototype(
                other_label, other_signal.modality, other_fp
            )

        # Hebbian conditioning
        syn = self.network.condition(
            source_fp=aud_fp,
            source_modality="auditory",
            target_fp=other_fp,
            target_modality=other_signal.modality,
            source_activation=min(1.0, tono.total_energy / max(1.0, tono.total_energy)),
            target_activation=min(1.0, other_signal.amplitude),
            temporal_overlap=1.0,  # Simultaneous presentation → perfect overlap
        )

        return {
            "synapse": syn.to_dict(),
            "auditory_fingerprint": aud_fp,
            "other_fingerprint": other_fp,
            "energy_transfer": syn.energy_transfer(),
        }

    # ------------------------------------------------------------------
    def probe_association(
        self,
        sound_wave: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Probe associations — given only sound, see what it evokes.

        Physical: Sound wave → cochlea → fingerprint → cross-modal cable → phantom

        Returns:
            {
                "tonotopic": TonotopicActivation,
                "identified_as": (label, sim) or None,
                "echoes": list of echo dicts,
                "echo_signal": ElectricalSignal or None,  # Phantom signal
                "has_phantom": bool,
            }
        """
        tono = self.cochlea.analyze(sound_wave)
        fp = tono.fingerprint()

        identified = self.network.identify(fp, "auditory")
        echoes = self.network.probe(fp, "auditory")
        echo_signal = self.network.generate_echo_signal(echoes)

        if echo_signal is not None:
            self.total_echoes_generated += 1

        return {
            "tonotopic": tono,
            "identified_as": identified,
            "echoes": echoes,
            "echo_signal": echo_signal,
            "has_phantom": echo_signal is not None,
        }

    # ------------------------------------------------------------------
    def tick(self):
        """Called each cognitive cycle — synaptic decay + cleanup."""
        self.network.decay_all()
        # Clear expired recent inputs
        now = time.monotonic()
        expired = [
            mod for mod, info in self._recent_inputs.items()
            if (now - info["timestamp"]) * 1000 > TEMPORAL_WINDOW_MS * 3
        ]
        for mod in expired:
            del self._recent_inputs[mod]

    # ------------------------------------------------------------------
    def _try_conditioning(
        self,
        new_modality: str,
        new_fp: np.ndarray,
    ) -> bool:
        """Attempt cross-modal Hebbian conditioning (within temporal window)."""
        now = time.monotonic()
        conditioned = False

        for mod, info in self._recent_inputs.items():
            if mod == new_modality:
                continue

            dt_ms = (now - info["timestamp"]) * 1000
            if dt_ms > TEMPORAL_WINDOW_MS:
                continue

            # Within temporal window → Hebbian conditioning!
            temporal_overlap = 1.0 - (dt_ms / TEMPORAL_WINDOW_MS)

            if new_modality == "auditory":
                src_fp, src_mod = new_fp, "auditory"
                tgt_fp, tgt_mod = info["fingerprint"], mod
            elif mod == "auditory":
                src_fp, src_mod = info["fingerprint"], "auditory"
                tgt_fp, tgt_mod = new_fp, new_modality
            else:
                continue  # Currently only auditory ↔ other

            act_src = info.get("activation", 1.0)
            if new_modality == "auditory":
                new_info = self._recent_inputs.get("auditory", {})
                act_src = new_info.get("activation", 1.0)
                act_tgt = info.get("activation", 1.0)
            else:
                act_tgt = 1.0

            self.network.condition(
                source_fp=src_fp,
                source_modality=src_mod,
                target_fp=tgt_fp,
                target_modality=tgt_mod,
                source_activation=act_src,
                target_activation=act_tgt,
                temporal_overlap=temporal_overlap,
            )
            conditioned = True
            self.total_groundings += 1

        return conditioned

    # ------------------------------------------------------------------
    def _signal_to_fingerprint(
        self,
        signal: ElectricalSignal,
    ) -> np.ndarray:
        """
        Convert ElectricalSignal to fingerprint vector.

        Physical: Any modality's signal has a waveform
        → Use the same FFT decomposition to produce a fingerprint
        → This makes fingerprints of different modalities "comparable"
        """
        waveform = signal.waveform
        if len(waveform) == 0:
            return np.zeros(self.cochlea.n_channels)

        # Use cochlear analysis (non-persistent — cross-modal signals analyzed independently)
        tono = self.cochlea.analyze(waveform, apply_persistence=False)
        return tono.fingerprint()

    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "cochlea": self.cochlea.get_state(),
            "network": self.network.get_state(),
            "total_groundings": self.total_groundings,
            "total_echoes_generated": self.total_echoes_generated,
            "recent_modalities": list(self._recent_inputs.keys()),
        }

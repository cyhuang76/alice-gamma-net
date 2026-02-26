# -*- coding: utf-8 -*-
"""
FusionBrain — Neural Substrate × Protocol Engine (Fusion Brain)

Unifies the v3 biological neural substrate (neurons / synapses / brain regions)
with the v4 communication protocol system (routing / cache / hemispheres).

Complete stimulus-response cycle (5 steps):
1. Sensory input → protocol routing → sensory cortex activation
2. Cognitive processing → prefrontal cortex higher-order interpretation
3. Emotional response → limbic system → can elevate protocol priority
4. Motor execution → integrates cognition (70%) + emotion (30%) → motor cortex
5. Memory consolidation → synaptic plasticity (strong activation +5%, weak activation -5%)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from alice.core.protocol import (
    MessagePacket,
    Modality,
    Priority,
    GammaNetV4Protocol,
)
from alice.core.signal import (
    ElectricalSignal,
    SignalBus,
    TransmissionReport,
    REGION_IMPEDANCE,
)
from alice.core.cache_analytics import CachePerformanceDashboard

from alice.brain.perception import (
    PerceptionPipeline,
    PerceptionResult,
)


# ============================================================================
# Brain region definitions
# ============================================================================


class BrainRegionType(Enum):
    MOTOR = "motor"                  # Motor cortex
    SOMATOSENSORY = "somatosensory"  # Somatosensory cortex
    LIMBIC = "limbic"                # Limbic system (emotion)
    PREFRONTAL = "prefrontal"        # Prefrontal cortex (decision-making)


# ============================================================================
# Neuron & Brain Region
# ============================================================================


@dataclass
class Neuron:
    """Single neuron"""

    region_id: int
    neuron_id: int
    activation: float = 0.0
    last_spike_time: float = 0.0
    synaptic_strength: float = 1.0


@dataclass
class BrainRegion:
    """
    Brain Region — contains multiple neurons.

    Each brain region has its own characteristic impedance,
    determining coupling efficiency with other regions.
    Impedance reference: REGION_IMPEDANCE (signal.py).
    """

    region_type: BrainRegionType
    region_id: int
    neuron_count: int = 100
    neurons: List[Neuron] = field(default_factory=list)
    impedance: float = 75.0  # Characteristic impedance (Ω)

    def __post_init__(self):
        if not self.neurons:
            self.neurons = [
                Neuron(region_id=self.region_id, neuron_id=i)
                for i in range(self.neuron_count)
            ]
        # Get impedance from reference table
        self.impedance = REGION_IMPEDANCE.get(self.region_type.value, 75.0)

    # ------------------------------------------------------------------
    def activate(
        self,
        stimulus: "np.ndarray | ElectricalSignal",
        strength: float = 1.0,
    ) -> np.ndarray:
        """
        Activate neurons in this brain region.

        Supports two input types:
        - np.ndarray       → traditional direct stimulus
        - ElectricalSignal → extract stimulus from waveform, strength modulated by amplitude
        """
        if isinstance(stimulus, ElectricalSignal):
            flat = stimulus.waveform.flatten()
            # Amplitude adjusts stimulus strength
            strength = strength * min(1.0, stimulus.amplitude)
        else:
            flat = stimulus.flatten()

        n = min(len(self.neurons), len(flat))
        activations = []

        for i in range(n):
            neuron = self.neurons[i]
            neuron.activation = float(
                np.clip(flat[i] * neuron.synaptic_strength * strength, 0, 1)
            )
            neuron.last_spike_time = time.time()
            activations.append(neuron.activation)

        return np.array(activations)

    # ------------------------------------------------------------------
    def get_activity_level(self) -> float:
        """Overall activity level"""
        if not self.neurons:
            return 0.0
        return float(np.mean([n.activation for n in self.neurons]))

    # ------------------------------------------------------------------
    def consolidate_memory(self):
        """
        Memory consolidation: Hebbian synaptic plasticity

        - Activation >0.7 → synaptic strengthening +5%
        - Activation <0.3 → synaptic decay -5%
        """
        for neuron in self.neurons:
            if neuron.activation > 0.7:
                neuron.synaptic_strength = min(2.0, neuron.synaptic_strength * 1.05)
            elif neuron.activation < 0.3:
                neuron.synaptic_strength = max(0.1, neuron.synaptic_strength * 0.95)


# ============================================================================
# Protocol ↔ Neural bidirectional converter
# ============================================================================


@dataclass
class NeuralMessage:
    """Internal message between brain regions"""
    source_region: BrainRegionType
    target_region: BrainRegionType
    content: np.ndarray
    priority: Priority
    timestamp: float = field(default_factory=time.time)


class ProtocolNeuralAdapter:
    """
    v4 Protocol ↔ Neural Substrate bidirectional converter (electrical signal version)

    - message_to_electrical : message packet → electrical signal
    - neural_to_electrical  : brain region activity → electrical signal
    - message_to_neural_stimulus : (backward-compatible) packet → stimulus
    - neural_activity_to_message : (backward-compatible) activity → packet
    """

    @staticmethod
    def message_to_electrical(
        packet: MessagePacket, target_region: BrainRegion
    ) -> ElectricalSignal:
        """Convert protocol packet to electrical signal"""
        # If the packet already contains an electrical signal, use it directly
        if packet.electrical_signal is not None:
            esig = packet.electrical_signal
            # Resample to target brain region size
            return esig.resample(target_region.neuron_count)

        # Otherwise build from raw data
        n = target_region.neuron_count
        if packet.raw_data is not None:
            data = packet.raw_data.flatten()
            if len(data) < n:
                padded = np.zeros(n)
                padded[: len(data)] = data
                data = padded
            else:
                data = data[:n]
        else:
            data = np.random.rand(n) * 0.3

        # Adjust amplitude based on priority (= stimulus strength)
        amplitude_map = {
            Priority.BACKGROUND: 0.3,
            Priority.NORMAL: 0.6,
            Priority.HIGH: 0.8,
            Priority.CRITICAL: 1.0,
        }
        amplitude = amplitude_map.get(packet.priority, 0.5)

        return ElectricalSignal.from_raw(
            data * amplitude,
            source=packet.source,
            modality=packet.modality.value,
            impedance=target_region.impedance,
        )

    @staticmethod
    def neural_to_electrical(region: BrainRegion) -> ElectricalSignal:
        """Convert brain region activity to electrical signal"""
        activations = np.array([n.activation for n in region.neurons])
        return ElectricalSignal.from_neural_activity(
            activations,
            region_impedance=region.impedance,
            source=region.region_type.value,
        )

    # ------------------------------------------------------------------
    # Backward-compatible methods
    # ------------------------------------------------------------------

    @staticmethod
    def message_to_neural_stimulus(
        packet: MessagePacket, target_region: BrainRegion
    ) -> tuple[np.ndarray, float]:
        """Convert protocol packet to neural stimulus (backward-compatible)"""
        n = target_region.neuron_count

        if packet.raw_data is not None:
            data = packet.raw_data.flatten()
            if len(data) < n:
                stimulus = np.zeros(n)
                stimulus[: len(data)] = data
            else:
                stimulus = data[:n]
        else:
            stimulus = np.random.rand(n) * 0.3

        # Adjust strength based on priority
        strength_map = {
            Priority.BACKGROUND: 0.3,
            Priority.NORMAL: 0.6,
            Priority.HIGH: 0.8,
            Priority.CRITICAL: 1.0,
        }
        strength = strength_map.get(packet.priority, 0.5)
        return stimulus, strength

    @staticmethod
    def neural_activity_to_message(
        region: BrainRegion, priority: Priority = Priority.NORMAL
    ) -> MessagePacket:
        """Convert brain region activity back to protocol packet (backward-compatible)"""
        activations = np.array([n.activation for n in region.neurons])

        modality_map = {
            BrainRegionType.MOTOR: Modality.INTERNAL,
            BrainRegionType.SOMATOSENSORY: Modality.TACTILE,
            BrainRegionType.LIMBIC: Modality.INTERNAL,
            BrainRegionType.PREFRONTAL: Modality.INTERNAL,
        }
        modality = modality_map.get(region.region_type, Modality.INTERNAL)

        return MessagePacket.from_signal(
            activations, modality=modality, priority=priority, source=region.region_type.value
        )


# ============================================================================
# FusionBrain — Fusion Brain
# ============================================================================


class FusionBrain:
    """
    Fusion Brain: v3 neural substrate + v4 communication protocol

    Unifies 4 brain regions, Γ-Net v4 messaging protocol, and performance analytics.
    Supports the complete stimulus → cognition → emotion → motor → memory cycle.
    """

    def __init__(self, neuron_count: int = 100):
        # 4 core brain regions
        self.regions: Dict[BrainRegionType, BrainRegion] = {
            BrainRegionType.MOTOR: BrainRegion(
                BrainRegionType.MOTOR, 0, neuron_count
            ),
            BrainRegionType.SOMATOSENSORY: BrainRegion(
                BrainRegionType.SOMATOSENSORY, 1, neuron_count
            ),
            BrainRegionType.LIMBIC: BrainRegion(
                BrainRegionType.LIMBIC, 2, neuron_count
            ),
            BrainRegionType.PREFRONTAL: BrainRegion(
                BrainRegionType.PREFRONTAL, 3, neuron_count
            ),
        }

        # v4 communication protocol
        self.protocol = GammaNetV4Protocol()
        self.adapter = ProtocolNeuralAdapter()
        self.analytics = CachePerformanceDashboard()

        # Unified electrical signal bus (coaxial cable topology)
        self.signal_bus = SignalBus.create_default_topology()

        # ★ Perception pipeline — GPU decomposition × CPU calibration × left/right brain reverse engineering × cross-modal binding
        self.perception = PerceptionPipeline()

        # Most recent perception result (for use in subsequent cycle steps)
        self._last_perception: Optional[PerceptionResult] = None

        # Reflected energy accumulated per cycle (for pain circuit)
        self._cycle_reflected_energy: float = 0.0
        self._cycle_reports: List[TransmissionReport] = []

        # ★ Sleep consolidation: record recent (packet, label) pairs from perception
        #   During sleep, "replay" these memories → increase usage_count → trigger ring migration
        self._recent_perceptions: List[tuple] = []   # [(packet, label), ...]
        self._max_recent: int = 50                    # Keep at most 50 entries

        # Internal state
        self._processing_history: List[Dict[str, Any]] = []
        self._cycle_count = 0

        # ★ Oscilloscope buffer — stores waveform snapshots from the last cycle
        self._scope_buffer: Dict[str, Any] = {
            "input_waveform": [],
            "input_freq": 0.0,
            "input_band": "alpha",
            "channels": {},
            "perception": None,
        }

    # ------------------------------------------------------------------
    # Complete 5-step stimulus-response cycle
    # ------------------------------------------------------------------

    def sensory_input(
        self,
        signal: "np.ndarray | ElectricalSignal",
        modality: Modality = Modality.VISUAL,
        priority: Priority = Priority.NORMAL,
    ) -> Dict[str, Any]:
        """
        Step 1: Sensory input → perception pipeline → protocol routing → sensory cortex activation

        ★ New pipeline (full-spectrum perception):
        1. Raw input → perception pipeline (GPU decomposition + CPU calibration + left/right brain reverse engineering)
        2. Perception pipeline integrates signals → protocol routing
        3. Routed signals → sensory cortex activation

        "The brain does not suppress; it receives all frequency bands and decomposes them into electrical signals."
        """
        # ★ 1. Perception pipeline: GPU full-spectrum decomposition → CPU attention calibration → left/right brain reverse engineering
        perception_result = self.perception.perceive(signal, modality.value)
        self._last_perception = perception_result

        # ★ 2. Use the integrated electrical signal from the perception pipeline (attention-weighted)
        esig = perception_result.integrated_signal

        packet = MessagePacket.from_signal(esig, modality, priority)
        should_process, reason = self.protocol.router.route(packet)

        # ★ Perception→cache write-back: completing the "perceive→store→re-recognize" closed loop
        #   Lookup cache first → hit = zero-compute recognition, miss = store in outermost ring
        #   Physics: you don't need to know "that's an apple" to recognize "I've seen this before"
        #            → familiarity = cache hit, naming = concept identification
        cache_hit, cached_label, cache_confidence = self.protocol.cache.lookup(packet)
        if cache_hit:
            # Cache hit: overwrite perception concept (reuse already-learned pattern)
            if cached_label:
                perception_result.concept = cached_label
        else:
            # Cache miss: store in outermost ring
            # If concept name exists → use concept name; otherwise → use frequency fingerprint (still recognizable)
            label = perception_result.concept or f"f{esig.frequency:.1f}_{packet.content_hash}"
            self.protocol.cache.store(packet, label)

        # ★ Record recent perception (for sleep consolidation replay)
        store_label = perception_result.concept or f"f{esig.frequency:.1f}_{packet.content_hash}"
        self._recent_perceptions.append((packet, store_label))
        if len(self._recent_perceptions) > self._max_recent:
            self._recent_perceptions = self._recent_perceptions[-self._max_recent:]

        sensory = self.regions[BrainRegionType.SOMATOSENSORY]
        # Use electrical signal to activate sensory cortex
        sensory_esig = self.adapter.message_to_electrical(packet, sensory)
        activation = sensory.activate(sensory_esig)

        # ★ Oscilloscope capture: input waveform
        scope_n = 128  # Oscilloscope display points
        input_wave = esig.waveform.flatten()
        self._scope_buffer["input_waveform"] = (
            input_wave[:scope_n].tolist() if len(input_wave) >= scope_n
            else input_wave.tolist()
        )
        self._scope_buffer["input_freq"] = esig.frequency
        self._scope_buffer["input_band"] = esig.band.value
        self._scope_buffer["perception"] = {
            "left_resonance": perception_result.left_resonance,
            "right_resonance": perception_result.right_resonance,
            "left_band": perception_result.left_tuned_band.value,
            "right_band": perception_result.right_tuned_band.value,
            "attention_band": perception_result.attention_band.value,
            "attention_strength": perception_result.attention_strength,
            "concept": perception_result.concept,
        }

        return {
            "step": "sensory_input",
            "routed": should_process,
            "route_reason": reason,
            "sensory_activity": float(sensory.get_activity_level()),
            "activation_mean": float(np.mean(activation)),
            "signal_band": esig.band.value,
            "signal_frequency": round(esig.frequency, 2),
            "signal_impedance": esig.impedance,
            # ★ Perception pipeline output (physics-driven)
            "perception": {
                "signal_band": perception_result.signal_band.value,
                "attention_band": perception_result.attention_band.value,
                "attention_strength": round(perception_result.attention_strength, 4),
                "left_tuned_band": perception_result.left_tuned_band.value,
                "left_resonance": round(perception_result.left_resonance, 4),
                "right_tuned_band": perception_result.right_tuned_band.value,
                "right_resonance": round(perception_result.right_resonance, 4),
                "concept": perception_result.concept,
                "bindings_found": len(perception_result.bindings),
                "pipeline_ms": round(perception_result.pipeline_elapsed_ms, 3),
            },
        }

    # ------------------------------------------------------------------
    def cognitive_processing(self) -> Dict[str, Any]:
        """
        Step 2: Cognitive processing → prefrontal cortex higher-order interpretation

        Sensory cortex →[coaxial channel 75Ω]→ prefrontal cortex
        Impedance matching determines signal transmission quality.
        """
        sensory = self.regions[BrainRegionType.SOMATOSENSORY]
        prefrontal = self.regions[BrainRegionType.PREFRONTAL]

        # Sensory cortex generates electrical signal
        sensory_esig = self.adapter.neural_to_electrical(sensory)

        # Transmit to prefrontal cortex through coaxial channel
        transmitted, report = self.signal_bus.send(
            "somatosensory", "prefrontal", sensory_esig
        )
        if report:
            self._cycle_reports.append(report)
            self._cycle_reflected_energy += report.reflected_energy

        # ★ Oscilloscope capture: cognitive channel
        ch_wave = transmitted.waveform.flatten()[:128] if transmitted else sensory_esig.waveform.flatten()[:128]
        self._scope_buffer["channels"]["sensory\u2192prefrontal"] = {
            "waveform": ch_wave.tolist(),
            "gamma": round(report.reflection_coefficient, 4) if report else 0.0,
            "reflected_ratio": round(report.reflected_power_ratio, 4) if report else 0.0,
            "matched": report.impedance_matched if report else True,
        }

        if transmitted is not None:
            pf_activation = prefrontal.activate(transmitted)
        else:
            # Channel does not exist, pass directly (degraded)
            pf_activation = prefrontal.activate(sensory_esig)

        return {
            "step": "cognitive_processing",
            "prefrontal_activity": float(prefrontal.get_activity_level()),
            "interpretation_strength": float(np.max(pf_activation)) if pf_activation.size else 0.0,
            "channel_Γ": round(report.reflection_coefficient, 4) if report else 0.0,
            "channel_impedance_matched": report.impedance_matched if report else True,
        }

    # ------------------------------------------------------------------
    def emotional_response(self) -> Dict[str, Any]:
        """
        Step 3: Emotional response → limbic system

        Sensory cortex →[coaxial channel 50Ω]→ limbic system (110Ω)
        Note: limbic system impedance 110Ω severely mismatches channel 50Ω!
        → Large reflection coefficient → emotional signals are "hard to transmit" → stronger stimuli needed to trigger emotion
        → Reflected energy → heat generation → this is the "cost of emotion"
        """
        sensory = self.regions[BrainRegionType.SOMATOSENSORY]
        limbic = self.regions[BrainRegionType.LIMBIC]

        # Sensory cortex generates electrical signal (emotional pathway amplified 1.2x)
        sensory_esig = self.adapter.neural_to_electrical(sensory)
        sensory_esig = ElectricalSignal(
            waveform=sensory_esig.waveform * 1.2,
            amplitude=sensory_esig.amplitude * 1.2,
            frequency=sensory_esig.frequency,
            phase=sensory_esig.phase,
            impedance=sensory_esig.impedance,
            snr=sensory_esig.snr,
            timestamp=sensory_esig.timestamp,
            source=sensory_esig.source,
            modality=sensory_esig.modality,
        )

        # Transmit to limbic system through coaxial channel
        transmitted, report = self.signal_bus.send(
            "somatosensory", "limbic", sensory_esig
        )
        if report:
            self._cycle_reports.append(report)
            self._cycle_reflected_energy += report.reflected_energy

        # ★ Oscilloscope capture: emotional channel (severe impedance mismatch!)
        ch_wave = transmitted.waveform.flatten()[:128] if transmitted else sensory_esig.waveform.flatten()[:128]
        self._scope_buffer["channels"]["sensory\u2192limbic"] = {
            "waveform": ch_wave.tolist(),
            "gamma": round(report.reflection_coefficient, 4) if report else 0.0,
            "reflected_ratio": round(report.reflected_power_ratio, 4) if report else 0.0,
            "matched": report.impedance_matched if report else True,
        }

        if transmitted is not None:
            limbic_activation = limbic.activate(transmitted)
        else:
            limbic_activation = limbic.activate(sensory_esig)

        emotional_valence = float(np.mean(limbic_activation)) - 0.5  # -0.5 ~ +0.5

        return {
            "step": "emotional_response",
            "limbic_activity": float(limbic.get_activity_level()),
            "emotional_valence": round(emotional_valence, 3),
            "arousal": round(float(np.std(limbic_activation)), 3),
            "channel_Γ": round(report.reflection_coefficient, 4) if report else 0.0,
            "emotional_heat": round(report.reflected_energy, 6) if report else 0.0,
        }

    # ------------------------------------------------------------------
    def motor_execution(self) -> Dict[str, Any]:
        """
        Step 4: Motor execution = cognitive channel (70%) + emotional channel (30%)

        Prefrontal →[coaxial channel 75Ω]→ motor cortex  (cognitive command)
        Limbic     →[coaxial channel 110Ω]→ motor cortex  (emotional drive)
        Integrated to activate motor cortex.
        """
        prefrontal = self.regions[BrainRegionType.PREFRONTAL]
        limbic = self.regions[BrainRegionType.LIMBIC]
        motor = self.regions[BrainRegionType.MOTOR]

        # Prefrontal → motor (cognitive command, 75Ω channel)
        pf_esig = self.adapter.neural_to_electrical(prefrontal)
        pf_transmitted, pf_report = self.signal_bus.send(
            "prefrontal", "motor", pf_esig
        )
        if pf_report:
            self._cycle_reports.append(pf_report)
            self._cycle_reflected_energy += pf_report.reflected_energy

        # Limbic → motor (emotional drive, 110Ω channel — high impedance!)
        lm_esig = self.adapter.neural_to_electrical(limbic)
        lm_transmitted, lm_report = self.signal_bus.send(
            "limbic", "motor", lm_esig
        )
        if lm_report:
            self._cycle_reports.append(lm_report)
            self._cycle_reflected_energy += lm_report.reflected_energy

        # ★ Oscilloscope capture: motor channel
        pf_ch_wave = pf_transmitted.waveform.flatten()[:128] if pf_transmitted else pf_esig.waveform.flatten()[:128]
        self._scope_buffer["channels"]["prefrontal→motor"] = {
            "waveform": pf_ch_wave.tolist(),
            "gamma": round(pf_report.reflection_coefficient, 4) if pf_report else 0.0,
            "reflected_ratio": round(pf_report.reflected_power_ratio, 4) if pf_report else 0.0,
            "matched": pf_report.impedance_matched if pf_report else True,
        }
        lm_ch_wave = lm_transmitted.waveform.flatten()[:128] if lm_transmitted else lm_esig.waveform.flatten()[:128]
        self._scope_buffer["channels"]["limbic→motor"] = {
            "waveform": lm_ch_wave.tolist(),
            "gamma": round(lm_report.reflection_coefficient, 4) if lm_report else 0.0,
            "reflected_ratio": round(lm_report.reflected_power_ratio, 4) if lm_report else 0.0,
            "matched": lm_report.impedance_matched if lm_report else True,
        }

        # Integration: cognition 70% + emotion 30%
        pf_wave = pf_transmitted.waveform.flatten() if pf_transmitted else pf_esig.waveform.flatten()
        lm_wave = lm_transmitted.waveform.flatten() if lm_transmitted else lm_esig.waveform.flatten()

        min_len = min(len(pf_wave), len(lm_wave), motor.neuron_count)
        integrated = 0.7 * pf_wave[:min_len] + 0.3 * lm_wave[:min_len]

        motor_output = motor.activate(integrated, strength=0.9)

        return {
            "step": "motor_execution",
            "motor_activity": float(motor.get_activity_level()),
            "output_strength": float(np.mean(motor_output)),
            "cognitive_weight": 0.7,
            "emotional_weight": 0.3,
            "cognitive_channel_Γ": round(pf_report.reflection_coefficient, 4) if pf_report else 0.0,
            "emotional_channel_Γ": round(lm_report.reflection_coefficient, 4) if lm_report else 0.0,
        }

    # ------------------------------------------------------------------
    def memory_consolidation(self):
        """Step 5: Memory consolidation → Hebbian synaptic plasticity"""
        for region in self.regions.values():
            region.consolidate_memory()

    # ------------------------------------------------------------------
    def sleep_consolidate(self, consolidation_rate: float = 1.0) -> int:
        """
        Sleep memory consolidation — offline memory transfer during N3 deep sleep

        Physical model:
          During sleep the brain "replays" daytime memories →
          equivalent to re-store into ring cache →
          usage_count increases → triggers Fibonacci threshold migration →
          memories migrate from outer ring to inner ring (RAM→SSD→HDD)

        Args:
            consolidation_rate: consolidation rate (0~1), N3=1.0, N2=0.3, REM=0.5

        Returns:
            Number of memories consolidated this cycle
        """
        if not self._recent_perceptions:
            return 0

        # Determine how many to replay based on consolidation rate (N3=all, N2=only 30%)
        n_replay = max(1, int(len(self._recent_perceptions) * consolidation_rate))
        replayed = 0

        for packet, label in self._recent_perceptions[:n_replay]:
            # Replay = re-store → usage_count++ → trigger migration
            self.protocol.cache.store(packet, label)
            replayed += 1

        # Consolidated memories removed from queue (already moved to deeper ring)
        self._recent_perceptions = self._recent_perceptions[n_replay:]

        # Hebbian synaptic plasticity (original mechanism retained)
        self.memory_consolidation()

        return replayed

    # ------------------------------------------------------------------
    def process_stimulus(
        self,
        stimulus: "np.ndarray | ElectricalSignal",
        modality: Modality = Modality.VISUAL,
        priority: Priority = Priority.NORMAL,
    ) -> Dict[str, Any]:
        """
        Complete stimulus-response cycle (5 steps)

        Signals are transmitted between brain regions through coaxial channels.
        Accumulated reflected energy is available via get_cycle_reflected_energy() for the pain circuit.
        """
        self._cycle_count += 1
        # Reset this cycle's reflected energy
        self._cycle_reflected_energy = 0.0
        self._cycle_reports = []

        t0 = time.time()

        # Clear oscilloscope buffer
        self._scope_buffer = {
            "input_waveform": [],
            "input_freq": 0.0,
            "input_band": "alpha",
            "channels": {},
            "perception": None,
        }

        step1 = self.sensory_input(stimulus, modality, priority)
        step2 = self.cognitive_processing()
        step3 = self.emotional_response()
        step4 = self.motor_execution()
        self.memory_consolidation()

        elapsed = time.time() - t0

        result = {
            "cycle": self._cycle_count,
            "elapsed_ms": round(elapsed * 1000, 2),
            "sensory": step1,
            "cognitive": step2,
            "emotional": step3,
            "motor": step4,
            "memory_consolidated": True,
            # Coaxial cable physics quantities
            "cycle_reflected_energy": round(self._cycle_reflected_energy, 6),
            "channel_reports": [r.to_dict() for r in self._cycle_reports],
            # ★ Perception pipeline statistics
            "perception_stats": self.perception.get_stats(),
        }
        self._processing_history.append(result)
        return result

    # ------------------------------------------------------------------
    def get_cycle_reflected_energy(self) -> float:
        """Get the reflected energy from the last cycle (for SystemState pain circuit)"""
        return self._cycle_reflected_energy

    # ------------------------------------------------------------------
    def get_oscilloscope_data(self) -> Dict[str, Any]:
        """
        Get oscilloscope buffer data (waveform snapshot from the last cycle)

        Frontend oscilloscope uses this to plot:
         - CH1: Input waveform
         - CH2: Channel transmission waveforms + reflection coefficients
         - CH3: Resonance curves
         - CH4: Vital signs
        """
        return dict(self._scope_buffer)

    # ------------------------------------------------------------------
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state snapshot"""
        state: Dict[str, Any] = {
            "cycle_count": self._cycle_count,
            "regions": {},
            "protocol": self.protocol.get_stats(),
            "signal_bus": self.signal_bus.get_bus_summary(),
            "perception": self.perception.get_stats(),
        }
        for rtype, region in self.regions.items():
            state["regions"][rtype.value] = {
                "activity_level": round(region.get_activity_level(), 4),
                "neuron_count": len(region.neurons),
                "impedance": region.impedance,
                "avg_synaptic_strength": round(
                    float(np.mean([n.synaptic_strength for n in region.neurons])), 4
                ),
            }
        return state

    # ------------------------------------------------------------------
    def generate_report(self, title: str = "Brain Status Report") -> str:
        """Generate text report"""
        state = self.get_brain_state()
        bus = state.get("signal_bus", {})
        lines = [
            "=" * 60,
            f"  {title}",
            "=" * 60,
            f"  Processing cycles: {state['cycle_count']}",
            "",
            "  ── Brain Region Activity ──",
        ]
        for name, info in state["regions"].items():
            bar = "█" * max(1, int(info["activity_level"] * 20))
            lines.append(
                f"  [{name:16s}] Activity: {bar} {info['activity_level']:.1%} "
                f" Impedance: {info['impedance']:.0f}Ω  Synapse: {info['avg_synaptic_strength']:.3f}"
            )

        proto = state["protocol"]
        lines.extend(
            [
                "",
                "  ── Coaxial Bus ──",
                f"  Channels: {bus.get('total_channels', 0)}",
                f"  Total transmissions: {bus.get('total_transmissions', 0)}",
                f"  Impedance mismatch rate: {bus.get('mismatch_rate', 0):.1%}",
                f"  Reflected energy: {bus.get('total_reflected_energy', 0):.6f}",
                f"  Bus efficiency: {bus.get('bus_efficiency', 0):.1%}",
                "",
                "  ── Protocol ──",
                f"  Cache hit rate: {proto['cache_serve_rate']:.1%}",
                f"  Avg compute ratio: {proto['avg_compute_ratio']:.1%}",
                "=" * 60,
            ]
        )
        return "\n".join(lines)

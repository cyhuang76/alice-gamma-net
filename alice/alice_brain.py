# -*- coding: utf-8 -*-
"""
AliceBrain — Alice Unified Intelligence Controller

Integrates all subsystems:
  FusionBrain (Neural+Protocol) + WorkingMemory + RL + CausalReasoner + MetaLearner
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from alice.brain.fusion_brain import FusionBrain, BrainRegionType
from alice.core.protocol import Modality, Priority
from alice.core.signal import ElectricalSignal
from alice.modules.working_memory import WorkingMemory
from alice.modules.reinforcement import ReinforcementLearner
from alice.modules.causal_reasoning import CausalReasoner
from alice.modules.meta_learning import MetaLearner
from alice.body.eye import AliceEye
from alice.body.hand import AliceHand
from alice.body.ear import AliceEar
from alice.body.mouth import AliceMouth
from alice.brain.calibration import TemporalCalibrator
from alice.brain.autonomic import AutonomicNervousSystem
from alice.brain.sleep import SleepCycle
from alice.brain.consciousness import ConsciousnessModule
from alice.brain.life_loop import LifeLoop, CompensationAction
from alice.brain.pruning import NeuralPruningEngine
from alice.brain.sleep_physics import SleepPhysicsEngine
from alice.brain.auditory_grounding import AuditoryGroundingEngine
from alice.brain.semantic_field import SemanticFieldEngine
from alice.brain.broca import BrocaEngine
from alice.brain.hippocampus import HippocampusEngine
from alice.brain.wernicke import WernickeEngine
from alice.brain.thalamus import ThalamusEngine
from alice.brain.amygdala import AmygdalaEngine
from alice.brain.prefrontal import PrefrontalCortexEngine
from alice.brain.basal_ganglia import BasalGangliaEngine
from alice.brain.attention_plasticity import AttentionPlasticityEngine
from alice.brain.cognitive_flexibility import CognitiveFlexibilityEngine
from alice.brain.curiosity_drive import CuriosityDriveEngine
from alice.brain.mirror_neurons import MirrorNeuronEngine
from alice.brain.impedance_adaptation import ImpedanceAdaptationEngine
from alice.brain.predictive_engine import PredictiveEngine
from alice.brain.metacognition import MetacognitionEngine
from alice.brain.social_resonance import SocialResonanceEngine
from alice.brain.narrative_memory import NarrativeMemoryEngine
from alice.brain.recursive_grammar import RecursiveGrammarEngine
from alice.brain.semantic_pressure import SemanticPressureEngine
from alice.brain.homeostatic_drive import HomeostaticDriveEngine
from alice.brain.physics_reward import PhysicsRewardEngine
from alice.brain.pinch_fatigue import PinchFatigueEngine
from alice.brain.phantom_limb import PhantomLimbEngine
from alice.brain.clinical_neurology import ClinicalNeurologyEngine
from alice.brain.pharmacology import ClinicalPharmacologyEngine
from alice.brain.emotion_granularity import EmotionGranularityEngine


# ============================================================================
# SystemState — Vitals & Pain Feedback
# ============================================================================


@dataclass
class SystemState:
    """
    Alice's "Body" — Vitals Monitoring

    Physical quantities:
    - ram_temperature : Anxiety/stress temperature (0.0=calm, 1.0=overheat collapse)
    - stability_index : System stability (1.0=healthy, 0.0=near-death)
    - heart_rate      : Heartbeat (processing cycle rate, Hz)
    - pain_level      : Pain (0.0=no pain, 1.0=severe pain-protocol collapse)
    - consciousness   : Consciousness level (0.0=coma, 1.0=awake)

    Thresholds:
    - critical_queue_threshold : CRITICAL packet backlog limit
    - overheat_temperature     : Overheat threshold → begin throttling
    - meltdown_temperature     : Meltdown threshold → system freeze
    """

    # Vitals (real-time values)
    ram_temperature: float = 0.0
    stability_index: float = 1.0
    heart_rate: float = 60.0          # bpm simulation (normal: 60-100)
    pain_level: float = 0.0
    consciousness: float = 1.0

    # History records (for waveform rendering)
    temperature_history: List[float] = field(default_factory=list)
    stability_history: List[float] = field(default_factory=list)
    heart_rate_history: List[float] = field(default_factory=list)
    pain_history: List[float] = field(default_factory=list)
    consciousness_history: List[float] = field(default_factory=list)
    left_brain_history: List[float] = field(default_factory=list)
    right_brain_history: List[float] = field(default_factory=list)

    # Threshold settings
    critical_queue_threshold: int = 5
    overheat_temperature: float = 0.7
    meltdown_temperature: float = 0.9

    # Counters
    pain_events: int = 0
    freeze_events: int = 0
    recovery_events: int = 0
    total_ticks: int = 0

    # ★ Trauma memory system — pain leaves long-term cognitive imprints
    # Physics: repeated injury → synaptic sensitization → lowered pain threshold
    # Like a burned child flinching at the sight of fire
    trauma_imprints: List[float] = field(default_factory=list)  # Trauma frequency fingerprints
    pain_sensitivity: float = 1.0       # Pain sensitivity (1.0=normal, >1=hypersensitive)
    baseline_temperature: float = 0.0   # Baseline temperature drift (does not return to zero after repeated injury)
    trauma_count: int = 0               # Cumulative trauma count
    _SENSITIZATION_RATE: float = 0.05   # Sensitivity increase per trauma
    _BASELINE_DRIFT_RATE: float = 0.03  # Baseline shift per trauma
    _MAX_SENSITIVITY: float = 2.0       # Max sensitivity (threshold drops to 0.35)
    _MAX_BASELINE: float = 0.3          # Max baseline temperature drift

    # History limit
    _max_history: int = 300  # 5 minutes @ 1Hz

    # ------------------------------------------------------------------
    def tick(
        self,
        critical_queue_len: int,
        high_queue_len: int,
        total_queue_len: int,
        sensory_activity: float,
        emotional_valence: float,
        left_brain_activity: float,
        right_brain_activity: float,
        cycle_elapsed_ms: float,
        reflected_energy: float = 0.0,
    ):
        """
        Called once per cognitive cycle — updates all vitals.

        THE PAIN LOOP:
        1. Queue pressure + reflected energy → temperature rise
        2. Excessive temperature → pain generation
        3. Pain → stability decline
        4. Low stability → consciousness blur
        5. Blurred consciousness → abnormal heart rate (accelerates then decelerates until freeze)

        ★ reflected_energy: Reflected energy from coaxial channel impedance mismatch
           Directly converts to system heat — the worse the impedance match, the more pain!
        """
        self.total_ticks += 1

        # === 1. Temperature calculation ===
        # Pressure sources: CRITICAL packet backlog + total queue load + impedance reflection
        critical_pressure = min(1.0, critical_queue_len / max(1, self.critical_queue_threshold))
        queue_pressure = min(1.0, total_queue_len / 50.0)
        emotional_stress = max(0.0, -emotional_valence)  # Negative emotion = stress

        # ★ Reflected energy → heat (coaxial cable physics)
        #   The worse the impedance mismatch → more reflection → more heat
        reflection_heat = min(1.0, reflected_energy * 5.0)  # Normalized

        heat_input = (
            critical_pressure * 0.4       # CRITICAL packets are the main heat source
            + queue_pressure * 0.15       # Total queue is secondary heat source
            + emotional_stress * 0.1      # Negative emotions add heat
            + sensory_activity * 0.05     # Sensory overload adds heat
            + reflection_heat * 0.2       # ★ Heat from impedance reflection
        )

        # Cooling: natural dissipation (temperature slowly drops at low load)
        cooling = 0.03 * (1.0 - critical_pressure)

        self.ram_temperature = float(np.clip(
            self.ram_temperature + heat_input * 0.15 - cooling, 0.0, 1.0
        ))

        # === 2. Pain (with trauma sensitization) ===
        # ★ Pain threshold dynamically adjusts with trauma history
        #   Normal threshold = 0.7, after max sensitization = 0.7 / 2.0 = 0.35
        effective_threshold = self.overheat_temperature / self.pain_sensitivity

        if self.ram_temperature > effective_threshold:
            pain_intensity = (self.ram_temperature - effective_threshold) / (1.0 - effective_threshold)
            # Sensitization also amplifies pain intensity
            pain_intensity *= self.pain_sensitivity
            self.pain_level = float(np.clip(pain_intensity, 0.0, 1.0))
            if pain_intensity > 0.5:
                self.pain_events += 1
        else:
            # Pain slowly subsides
            self.pain_level = max(0.0, self.pain_level - 0.05)

        # === 3. Stability ===
        stability_damage = self.pain_level * 0.1  # Pain erodes stability
        stability_recovery = 0.02 * (1.0 - self.pain_level)  # Recovery when pain-free

        self.stability_index = float(np.clip(
            self.stability_index - stability_damage + stability_recovery, 0.0, 1.0
        ))

        if self.stability_index < 0.3 and self.stability_index + stability_damage >= 0.3:
            self.freeze_events += 1

        if self.stability_index > 0.7 and self.stability_index - stability_recovery <= 0.7:
            self.recovery_events += 1

        # === 4. Consciousness ===
        if self.ram_temperature >= self.meltdown_temperature:
            # Meltdown: consciousness drops rapidly
            self.consciousness = max(0.05, self.consciousness * 0.85)
        elif self.stability_index < 0.3:
            # Low stability: begins to blur
            self.consciousness = max(0.1, self.consciousness * 0.92)
        else:
            # Normal: consciousness recovers
            self.consciousness = min(1.0, self.consciousness + 0.02 * self.stability_index)

        # === 5. Heart rate ===
        # Normal: 60 bpm, Stress: accelerates to 180, Collapse: drops to 20 (cardiac arrest simulation)
        if self.consciousness > 0.5:
            # Awake: heart rate rises with stress (60-180)
            target_hr = 60 + 120 * self.ram_temperature
        else:
            # Blurred consciousness: heart rate drops (ECG flattens)
            target_hr = 20 + 40 * self.consciousness

        # Heart rate smooth transition
        self.heart_rate = self.heart_rate + (target_hr - self.heart_rate) * 0.3

        # Add arrhythmia (at high pain levels)
        if self.pain_level > 0.6:
            arrhythmia = np.random.normal(0, 10 * self.pain_level)
            self.heart_rate = max(10, self.heart_rate + arrhythmia)

        # === Record history ===
        self.temperature_history.append(self.ram_temperature)
        self.stability_history.append(self.stability_index)
        self.heart_rate_history.append(self.heart_rate)
        self.pain_history.append(self.pain_level)
        self.consciousness_history.append(self.consciousness)
        self.left_brain_history.append(left_brain_activity)
        self.right_brain_history.append(right_brain_activity)

        # Trim history
        for hist in (
            self.temperature_history, self.stability_history,
            self.heart_rate_history, self.pain_history,
            self.consciousness_history,
            self.left_brain_history, self.right_brain_history,
        ):
            if len(hist) > self._max_history:
                del hist[:-self._max_history]

    # ------------------------------------------------------------------
    def get_throttle_factor(self) -> float:
        """
        Physical punishment: returns processing rate multiplier (0.0~1.0)

        - 1.0 = full speed
        - 0.5 = throttled 50% due to overheat
        - 0.1 = near freeze
        """
        if self.ram_temperature >= self.meltdown_temperature:
            return 0.1  # Nearly frozen
        if self.ram_temperature >= self.overheat_temperature:
            # Linear throttling: 0.7→1.0, 0.9→0.1
            t = (self.ram_temperature - self.overheat_temperature) / (self.meltdown_temperature - self.overheat_temperature)
            return max(0.1, 1.0 - t * 0.9)
        return 1.0  # Full speed

    # ------------------------------------------------------------------
    def is_frozen(self) -> bool:
        """Whether the system is frozen (cognitive paralysis from severe pain)"""
        return self.consciousness < 0.15

    # ------------------------------------------------------------------
    def get_vitals(self) -> Dict[str, Any]:
        """Get current vitals snapshot"""
        return {
            "ram_temperature": round(self.ram_temperature, 4),
            "stability_index": round(self.stability_index, 4),
            "heart_rate": round(self.heart_rate, 1),
            "pain_level": round(self.pain_level, 4),
            "consciousness": round(self.consciousness, 4),
            "throttle_factor": round(self.get_throttle_factor(), 3),
            "is_frozen": self.is_frozen(),
            "pain_events": self.pain_events,
            "freeze_events": self.freeze_events,
            "recovery_events": self.recovery_events,
            "total_ticks": self.total_ticks,
            # ★ Trauma memory indicators
            "pain_sensitivity": round(self.pain_sensitivity, 3),
            "baseline_temperature": round(self.baseline_temperature, 4),
            "trauma_count": self.trauma_count,
        }

    # ------------------------------------------------------------------
    def get_waveforms(self, last_n: int = 60) -> Dict[str, List[float]]:
        """Get waveform data (for dashboard plotting)"""
        return {
            "temperature": self.temperature_history[-last_n:],
            "stability": self.stability_history[-last_n:],
            "heart_rate": self.heart_rate_history[-last_n:],
            "pain": self.pain_history[-last_n:],
            "consciousness": self.consciousness_history[-last_n:],
            "left_brain": self.left_brain_history[-last_n:],
            "right_brain": self.right_brain_history[-last_n:],
        }

    # ------------------------------------------------------------------
    def record_trauma(self, signal_frequency: float = 0.0):
        """
        Record trauma — pain events leave imprints in memory

        Physics:
          Repeated impedance mismatch injury → synaptic sensitization → easier to feel pain next time
          Like PTSD: scars change how you respond to the world
        """
        self.trauma_count += 1

        # Sensitization: each trauma increases pain sensitivity
        self.pain_sensitivity = min(
            self._MAX_SENSITIVITY,
            self.pain_sensitivity + self._SENSITIZATION_RATE
        )

        # Baseline drift: "normal body temperature" after recovery is no longer 0
        self.baseline_temperature = min(
            self._MAX_BASELINE,
            self.baseline_temperature + self._BASELINE_DRIFT_RATE
        )

        # Store trauma frequency fingerprint (max 20 entries)
        if signal_frequency > 0:
            self.trauma_imprints.append(signal_frequency)
            if len(self.trauma_imprints) > 20:
                self.trauma_imprints = self.trauma_imprints[-20:]

    def is_trauma_trigger(self, signal_frequency: float, tolerance: float = 2.0) -> bool:
        """
        Check if a signal triggers trauma memory

        If the input signal frequency is close to any trauma imprint → return True
        → Can be used for automatic vigilance / preventive sympathetic activation
        """
        for imprint_freq in self.trauma_imprints:
            if abs(signal_frequency - imprint_freq) < tolerance:
                return True
        return False

    def reset(self):
        """Emergency reset — restore to healthy state (but trauma memories are preserved)"""
        # ★ Baseline temperature does not reset to zero — traces left by trauma
        self.ram_temperature = self.baseline_temperature
        self.stability_index = 1.0
        self.heart_rate = 60.0
        self.pain_level = 0.0
        self.consciousness = 1.0
        # Note: pain_sensitivity, baseline_temperature, trauma_imprints
        #       are not reset — this is trauma memory


class AliceBrain:
    """
    Alice Unified Intelligence Controller

    Layer 5 — The highest cognitive integration layer, coordinating all subsystems.

    Functions:
    - Perceive: receive environmental stimuli
    - Think: causal reasoning + working memory + meta-learning
    - Act: reinforcement learning selects actions
    - Learn: synaptic plasticity + TD update + strategy evolution
    - Introspect: system state monitoring
    """

    def __init__(self, neuron_count: int = 100):
        # Core subsystems
        self.fusion_brain = FusionBrain(neuron_count=neuron_count)
        self.working_memory = WorkingMemory(capacity=7)
        self.rl = ReinforcementLearner()
        self.causal = CausalReasoner()
        self.meta = MetaLearner()

        # ★ Vitals system — THE PAIN LOOP
        self.vitals = SystemState()

        # ★ Body peripherals
        self.eye = AliceEye()
        self.hand = AliceHand()
        self.ear = AliceEar()
        self.mouth = AliceMouth()

        # ★ Temporal calibrator — cross-modal signal binding (action model)
        self.calibrator = TemporalCalibrator()

        # ★ Autonomic nervous system + Sleep + Consciousness
        self.autonomic = AutonomicNervousSystem()
        self.sleep_cycle = SleepCycle()
        self.consciousness = ConsciousnessModule()

        # ★ Life loop — closed-loop error compensation engine
        self.life_loop = LifeLoop()

        # ★ Neural pruning engine — §3.5.2 large-scale Γ apoptosis
        #   From random impedance connections → experience-driven pruning → specialized cortex
        #   Intelligence = Σ Γ² → min
        self.pruning = NeuralPruningEngine(
            connections_per_region=500,  # Scaled simulation (conceptually equivalent to 20 billion)
        )

        # ★ Sleep physics engine — offline impedance reorganization and energy conservation
        #   Awake = Minimize Γ_ext, Sleep = Minimize Γ_int
        self.sleep_physics = SleepPhysicsEngine()

        # ★ Auditory grounding engine — Phase 4.1 Language physicalization
        #   Language = impedance modulation, traveling wave remote control of another brain
        #   Pavlovian conditioning = cross-modal Hebbian wiring
        self.auditory_grounding = AuditoryGroundingEngine()

        # ★ Semantic field — Phase 4.2 Concepts as attractors in state space
        #   Each concept = resonant gravity well, recognition = lowest Γ_semantic
        self.semantic_field = SemanticFieldEngine()

        # ★ Broca's area — Phase 4.3 Motor speech planning
        #   Concept → articulation plan → mouth execution → auditory feedback → sensorimotor learning
        self.broca = BrocaEngine(
            cochlea=self.auditory_grounding.cochlea,
        )

        # ★ Hippocampus — Phase 5.1 Temporal binding engine
        #   Stitches sensory snapshots from different modalities into "experiences"
        #   Attractor labels = inter-membrane wormholes, time = the missing dimension
        self.hippocampus = HippocampusEngine()

        # ★ Wernicke's area — Phase 5.2 Sequence comprehension engine
        #   Concept sequence → transition probability → Γ_syntactic → comprehension/confusion
        #   N400 = high Γ_syn event
        self.wernicke = WernickeEngine()

        # ★ Thalamus — Phase 5.3 Sensory gate
        #   All sensory signals must pass through the thalamic gate to reach the cortex
        #   G_total = G_arousal × (α × G_topdown + (1-α) × G_bottomup)
        self.thalamus = ThalamusEngine()

        # ★ Amygdala — Phase 5.4 Emotional fast pathway
        #   Low road: sensory → thalamus → amygdala → fight/flight (bypasses cortex, ~100ms)
        #   Fear conditioning = permanent lowering of impedance threshold
        self.amygdala = AmygdalaEngine()

        # ★ Emotion granularity engine — Phase 36 Plutchik 8-dimensional emotion vector
        #   Amygdala rapid threat assessment → emotion granularity expansion → VAD coordinates + compound emotions
        #   "Emotion is not a single dial—it is an 8-dimensional impedance map"
        self.emotion_granularity = EmotionGranularityEngine()

        # ★ Prefrontal cortex — Phase 6.1 Executive control center
        #   Goal management + Go/NoGo gate + planning + willpower
        #   "Not doing something costs more energy than doing something"
        self.prefrontal = PrefrontalCortexEngine()

        # ★ Basal ganglia — Phase 6.2 Habit engine
        #   Repetition + reward → Γ_action → 0 → automatic execution
        #   Dual system: habitual (model-free) vs goal-directed (model-based)
        self.basal_ganglia = BasalGangliaEngine()

        # ★ Attention plasticity — Phase 7 Attention training engine
        #   All attention parameters are no longer constants — they are trainable physical circuit properties
        #   Gate speed, tuner Q, inhibition efficiency, reaction delay → improve with experience
        #   "A pro gamer's reaction time is not talent — it is synaptic plasticity."
        self.attention_plasticity = AttentionPlasticityEngine()

        # ★ Cognitive flexibility engine — Phase 8 High-intensity task switching
        #   Task set reconfiguration + inertia impedance + mixing cost + perseveration errors
        #   "A pro gamer can switch task sets in 30ms — that is trained synaptic plasticity"
        self.cognitive_flexibility = CognitiveFlexibilityEngine()

        # ★ Curiosity drive engine — Phase 9 Free will and self-awareness
        #   Curiosity = consequence of impedance homeostasis
        #   Boredom = long-term low novelty pressure → spontaneous behavior
        #   Self-recognition = efference copy prediction comparison
        #   "Free will is not randomness — it is the physical expression of internal drive"
        self.curiosity_drive = CuriosityDriveEngine()

        # Phase 6.1: Mirror neuron system
        #   Physical basis of social cognition
        #   Action mirroring + emotional resonance + theory of mind
        #   "Empathy is not imagination — it is impedance resonance"
        self.mirror_neurons = MirrorNeuronEngine()

        # ★ Cross-modal impedance adaptation engine — Phase 10 Experience learning closed-loop
        #   Each perceptual binding → record success/failure → Γ improves with experience
        #   Cortisol → Yerkes-Dodson modulation of learning rate
        #   Idle pairings → forgetting decay (use it or lose it)
        #   "The 100th time you hear something is clearer than the 1st—not because the sound got louder, but because impedance matching improved"
        self.impedance_adaptation = ImpedanceAdaptationEngine()

        # ★ Predictive processing engine — Phase 17 The eye of time
        #   Forward model + free energy minimization + mental simulation
        #   Waking "rapid dreams" → simulate the future → preemptive action
        #   "Intelligence is not reaction—it is prediction."
        self.predictive = PredictiveEngine()

        # ★ Metacognition engine — Phase 18 The internal auditor
        #   Monitor whole-brain cognitive impedance → System 1/2 switching → counterfactual reasoning
        #   Confidence estimation + self-correction + insight detection + rumination alarm
        #   "Thinking itself has impedance—seeing your own thoughts is the last layer of consciousness."
        self.metacognition = MetacognitionEngine()

        # ★ Social resonance engine — Phase 19 Social physics field
        #   Social impedance coupling + Theory of Mind (Sally-Anne) + social homeostasis
        #   Loneliness = social field vacuum state; intimacy = impedance approaches zero, frequency phase-locked
        #   "Empathy is not imagination—it is cross-body impedance matching."
        self.social_resonance = SocialResonanceEngine()

        # ★ Narrative memory engine — Phase 20.1 Autobiographical memory weaving
        #   Hippocampus fragmented episodes → causal chains → narrative arc → autobiography
        #   Causal linking = concept overlap × temporal proximity × emotional coherence
        #   "Memory is not a videotape—it is a story that keeps being rewritten."
        self.narrative_memory = NarrativeMemoryEngine(
            hippocampus=self.hippocampus,
        )

        # ★ Recursive grammar engine — Phase 20.2 Phrase structure recursion
        #   Broca words + Wernicke sequences → recursive syntax tree
        #   NP→Det+N, VP→V+NP, S→NP+VP → center embedding → garden path recovery
        #   "Language is not a string—it is a tree."
        self.recursive_grammar = RecursiveGrammarEngine(
            broca=self.broca,
            wernicke=self.wernicke,
        )

        # ★ Semantic pressure engine — Phase 21 Thermodynamics of language
        #   Semantic pressure accumulation + inner monologue + Wernicke→Broca direct link
        #   Language is impedance matching—speech releases internal tension (Γ_speech → 0)
        #   "Why does she want to speak? Because without speaking, Γ_internal can never be released."
        self.semantic_pressure = SemanticPressureEngine()

        # ★ Hypothalamic homeostatic drive — Phase 22.1 Hunger/thirst
        #   Glucose homeostasis → hunger drive, hydration homeostasis → thirst drive
        #   The most basic energy needs of a living body: Γ_hunger, Γ_thirst → 0
        #   "Hunger is not a feeling—it is an undervoltage alarm circuit."
        self.homeostatic_drive = HomeostaticDriveEngine()

        # ★ Physics reward engine — Phase 22.2 Impedance matching reward
        #   Replace Q-table with channel impedance Z: RPE→Z↓/Z↑ → Γ → Boltzmann selection
        #   Unified with whole-system Γ language (pain=Γ↑, learning=Γ↓, reward=Γ→0)
        #   "Dopamine is not happiness—it is the electrochemical signal of impedance matching improvement."
        self.physics_reward = PhysicsRewardEngine()

        # ★ Lorentz compression fatigue engine — Phase 23 Pollock-Barraclough (1905) Neural aging
        #   High current I → self-generated magnetic field B → Lorentz force J×B → radial conductor compression
        #   Elastic (sleep-repairable) vs plastic (permanent aging) = dual fatigue model
        #   "Aging is not rusting—it is your cables being crushed by their own magnetic field."
        self.pinch_fatigue = PinchFatigueEngine()

        # ★ Phantom limb pain engine — Phase 24 Γ=1.0 Open-circuit physics
        #   Amputation = coaxial cable terminal load removed → Z_load=∞ → Γ=1.0
        #   Signal 100% reflected → reflected_energy → pain
        #   "The hand is gone, but the cable remains—your brain keeps dialing, and all the bounced-back energy is pain."
        self.phantom_limb = PhantomLimbEngine()

        # ★ Clinical neurology engine — Phase 25 Unified physics model of five major neurological diseases
        #   Stroke, ALS, dementia, Alzheimer's, cerebral palsy
        #   "All neurological diseases are different patterns of coaxial cable impedance mismatch."
        self.clinical_neurology = ClinicalNeurologyEngine()

        # ★ Computational pharmacology engine — Phase 26
        #   Unified formula: Z_eff = Z₀ × (1 + α_drug)
        #   MS / PD / epilepsy / depression + drug interactions
        self.pharmacology = ClinicalPharmacologyEngine()

        # Inject plasticity engine into thalamus and perception pipeline
        self.thalamus.set_plasticity_engine(self.attention_plasticity)
        self.fusion_brain.perception.set_plasticity_engine(self.attention_plasticity)

        # Closed-loop state tracking
        self._current_hand_target: Optional[np.ndarray] = None
        self._current_pitch_target: Optional[float] = None
        self._current_visual_target: Optional[np.ndarray] = None
        self._last_visual_signal: Optional[ElectricalSignal] = None
        self._last_auditory_signal: Optional[ElectricalSignal] = None

        # System state
        self._state = "idle"
        self._cycle_count = 0
        self._start_time = time.time()
        self._event_log: List[Dict[str, Any]] = []
        self._last_cycle_ms = 0.0

    # ------------------------------------------------------------------
    # Compensation command dispatch — LifeLoop commands → body organ execution
    # ------------------------------------------------------------------

    def _dispatch_commands(self, commands: list) -> List[Dict[str, Any]]:
        """
        Dispatch CompensationCommands from LifeLoop to the corresponding body organs.

        The final link in the closed-loop:
          Error → PID compensation → command → body execution → new sensory → new error ...

        Each CompensationAction maps to a body method:
          REACH       → hand.reach(target_x, target_y)
          SACCADE     → eye.saccade(pixels, region)
          VOCALIZE    → mouth.speak(target_pitch)
          ADJUST_PUPIL→ eye.adjust_pupil(aperture)
          ATTEND      → consciousness.focus_attention(...)
          BREATHE     → autonomic.tick() (breathing regulation via autonomic nervous system)
        """
        results = []
        for cmd in commands:
            action = cmd.action
            strength = cmd.strength
            target = cmd.target
            result: Dict[str, Any] = {
                "action": action.value,
                "strength": round(strength, 4),
                "executed": False,
            }

            try:
                if action == CompensationAction.REACH and target is not None:
                    tx = float(target[0]) if len(target) > 0 else self.hand.x
                    ty = float(target[1]) if len(target) > 1 else self.hand.y
                    reach_result = self.hand.reach(
                        target_x=tx,
                        target_y=ty,
                        ram_temperature=self.vitals.ram_temperature,
                        max_steps=50,  # Limit steps within a single tick
                    )
                    result["executed"] = True
                    result["reached"] = reach_result.get("reached", False)
                    result["final_error"] = round(reach_result.get("final_error", 0.0), 4)

                elif action == CompensationAction.SACCADE and target is not None:
                    # target encoded as [row, col, height, width]
                    # If no complete region info, skip (need pixels)
                    result["executed"] = False
                    result["note"] = "saccade deferred to next visual frame"

                elif action == CompensationAction.VOCALIZE and target is not None:
                    target_pitch = float(target[0]) if len(target) > 0 else 220.0
                    speak_result = self.mouth.speak(
                        target_pitch=target_pitch,
                        volume=float(np.clip(strength, 0.1, 1.0)),
                        ram_temperature=self.vitals.ram_temperature,
                    )
                    result["executed"] = True
                    result["pitch_error"] = round(speak_result.get("pitch_error", 0.0), 4)

                elif action == CompensationAction.ADJUST_PUPIL:
                    aperture = float(target[0]) if target is not None and len(target) > 0 else strength
                    self.eye.adjust_pupil(aperture)
                    result["executed"] = True

                elif action == CompensationAction.ATTEND:
                    self.consciousness.focus_attention(
                        target="compensation",
                        modality="cross_modal",
                        salience=strength,
                    )
                    result["executed"] = True

                elif action == CompensationAction.BREATHE:
                    # Breathing regulation via parasympathetic gain in autonomic nervous system
                    self.autonomic.parasympathetic = min(
                        1.0,
                        self.autonomic.parasympathetic + strength * 0.1,
                    )
                    result["executed"] = True

            except Exception:  # pragma: no cover
                result["executed"] = False

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Neural pruning — sensory signal-driven cortical specialization
    # ------------------------------------------------------------------

    def _stimulate_pruning(
        self,
        modality: Modality,
        perception_signal: Optional[ElectricalSignal],
    ) -> None:
        """
        Feed perception signals into the neural pruning engine's corresponding cortical region.

        Physical correspondence:
          VISUAL    → occipital  (occipital lobe: spatial frequency)
          AUDITORY  → temporal   (temporal lobe: temporal frequency)
          Other     → parietal   (parietal lobe: somatosensory broadband)

        Each sensory stimulus triggers one round of Hebbian selection in the corresponding region:
          Γ → 0 connections are strengthened, Γ >> 0 connections are weakened.
        After sufficient weakening → automatic apoptosis (prune).

        Meanwhile, strong connections trigger synaptogenesis:
          Learning signal = curiosity × novelty + reward_signal
          → Strong connections sprout new neighbors → filtered in subsequent pruning

        This is the physical realization of "Intelligence = Σ Γ² → min".
        """
        if perception_signal is None:
            return

        # Signal impedance and frequency
        sig_z = perception_signal.impedance
        sig_f = perception_signal.frequency

        # Select cortical region based on modality
        region_map = {
            Modality.VISUAL: "occipital",
            Modality.AUDITORY: "temporal",
        }
        region_name = region_map.get(modality, "parietal")

        region = self.pruning.regions.get(region_name)
        if region is None:
            return

        # Apply stimulus → Hebbian selection
        region.stimulate(sig_z, sig_f)

        # Every 50 ticks run pruning + synaptogenesis scan (avoid scanning every time)
        if self._cycle_count % 50 == 0:
            # ★ Compute learning signal — from curiosity, reward, novelty
            learning_signal = 0.0
            if hasattr(self, 'curiosity_drive'):
                cs = self.curiosity_drive.get_state()
                learning_signal += cs.get("curiosity_level", 0.0) * 0.4
                learning_signal += cs.get("novelty", 0.0) * 0.3
            if hasattr(self, 'physics_reward'):
                rs = self.physics_reward.get_state()
                learning_signal += abs(rs.get("reward", 0.0)) * 0.3
            learning_signal = min(1.0, learning_signal)

            # ★ Synaptogenesis — grow first, then prune
            region.sprout(learning_signal=learning_signal)

            # Apoptosis — overly weak connections die
            region.prune()
            region.determine_specialization()

    # ------------------------------------------------------------------
    # Core cognitive cycle
    # ------------------------------------------------------------------

    def perceive(
        self,
        stimulus: np.ndarray,
        modality: Modality = Modality.VISUAL,
        priority: Priority = Priority.NORMAL,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perceive: receive environmental stimuli, start full fusion brain processing cycle.

        ★ With pain feedback loop:
        1. Check if system is frozen → frozen systems can only process CRITICAL
        2. FusionBrain 5-step cycle
        3. Update vitals (THE PAIN LOOP)
        4. Physical punishment: throttling
        5. Store result in working memory
        6. Update causal observations
        """
        self._state = "perceiving"
        self._cycle_count += 1
        t0 = time.time()

        # ★ Freeze check: when system is frozen from severe pain, only process CRITICAL
        if self.vitals.is_frozen() and priority != Priority.CRITICAL:
            self._log_event("perceive_blocked", {
                "reason": "SYSTEM FROZEN — consciousness too low, only CRITICAL allowed",
                "consciousness": self.vitals.consciousness,
                "pain_level": self.vitals.pain_level,
            })
            # Even if blocked, still update tick (let the system cool down naturally)
            router = self.fusion_brain.protocol.router
            self.vitals.tick(
                critical_queue_len=len(router.queues[Priority.CRITICAL]),
                high_queue_len=len(router.queues[Priority.HIGH]),
                total_queue_len=sum(len(q) for q in router.queues.values()),
                sensory_activity=0.0,
                emotional_valence=0.0,
                left_brain_activity=self.fusion_brain.protocol.left_brain.get_stats()["activations"],
                right_brain_activity=self.fusion_brain.protocol.right_brain.get_stats()["activations"],
                cycle_elapsed_ms=0.0,
            )
            self._state = "frozen"
            return {
                "cycle": self._cycle_count,
                "status": "FROZEN",
                "vitals": self.vitals.get_vitals(),
                "message": "System frozen due to severe pain. Only CRITICAL signals can penetrate.",
            }

        # ★ Physical punishment: simulate cognitive sluggishness when overheated
        throttle = self.vitals.get_throttle_factor()
        if throttle < 1.0:
            # Artificial delay — simulating cognitive paralysis from pain
            delay_ms = (1.0 - throttle) * 5.0  # Max 4.5ms extra delay
            time.sleep(delay_ms / 1000.0)

        # FusionBrain complete cycle
        brain_result = self.fusion_brain.process_stimulus(stimulus, modality, priority)

        elapsed_ms = (time.time() - t0) * 1000
        self._last_cycle_ms = elapsed_ms

        # ★ Update vitals — THE PAIN LOOP
        router = self.fusion_brain.protocol.router
        left_stats = self.fusion_brain.protocol.left_brain.get_stats()
        right_stats = self.fusion_brain.protocol.right_brain.get_stats()

        # ★ Coaxial channel reflected energy → pain loop
        reflected_energy = self.fusion_brain.get_cycle_reflected_energy()

        self.vitals.tick(
            critical_queue_len=len(router.queues[Priority.CRITICAL]),
            high_queue_len=len(router.queues[Priority.HIGH]),
            total_queue_len=sum(len(q) for q in router.queues.values()),
            sensory_activity=brain_result["sensory"]["sensory_activity"],
            emotional_valence=brain_result["emotional"]["emotional_valence"],
            left_brain_activity=left_stats.get("activation_rate", 0.0),
            right_brain_activity=right_stats.get("activation_rate", 0.0),
            cycle_elapsed_ms=elapsed_ms,
            reflected_energy=reflected_energy,
        )

        # Store in working memory
        mem_key = context or f"perception_{self._cycle_count}"

        # ★ Get perception pipeline result (left/right brain mode + attention + binding)
        perception_data = brain_result["sensory"].get("perception", {})

        # ★ Compute cross-modal binding impedance mismatch Γ_bind
        # Physics: calibration quality reflects impedance matching of cross-modal signal binding
        # quality=1 → Γ=0 (perfect binding), quality=0 → Γ=1 (complete mismatch)
        # "When TV picture and sound cannot be combined, severe visual impedance forms"
        _cal_quality_early = self.calibrator.get_calibration_quality()
        _raw_binding_gamma = 1.0 - _cal_quality_early

        # ★ Experience-adapted Γ — closed-loop impedance matching
        # Query current known modality experience Γ (frame not yet acquired, will update later)
        _active_modalities = [modality.value]
        # Query experience-adapted Γ
        _adapted_gamma = self.impedance_adaptation.get_adapted_binding_gamma(
            _active_modalities
        )
        # Mix: 70% real-time calibration + 30% experience adaptation
        # Physical meaning: real-time impedance is hardware reality, experience is long-term software optimization
        binding_gamma = 0.7 * _raw_binding_gamma + 0.3 * _adapted_gamma

        self.working_memory.store(
            key=mem_key,
            content={
                "stimulus_energy": float(np.sum(np.abs(stimulus))),
                "modality": modality.value,
                "pain_level": self.vitals.pain_level,
                "brain_result": {
                    k: v
                    for k, v in brain_result.items()
                    if k in ("cycle", "elapsed_ms", "memory_consolidated")
                },
                # ★ Physical perception result stored in memory
                "perception_pattern": {
                    "attention_band": perception_data.get("attention_band", ""),
                    "concept": perception_data.get("concept"),
                    "left_tuned": perception_data.get("left_tuned_band", ""),
                    "right_tuned": perception_data.get("right_tuned_band", ""),
                    "bindings": perception_data.get("bindings_found", 0),
                },
            },
            importance=0.3 + 0.7 * (priority.value / 3.0),
            binding_gamma=binding_gamma,  # ★ Impedance-modulated memory write
        )

        # Causal observation — add pain, temperature, reflected energy + ★ perception pipeline features
        causal_vars = {
            "stimulus_energy": float(np.sum(np.abs(stimulus))),
            "sensory_activity": brain_result["sensory"]["sensory_activity"],
            "emotional_valence": brain_result["emotional"]["emotional_valence"],
            "motor_output": brain_result["motor"]["output_strength"],
            "pain_level": self.vitals.pain_level,
            "ram_temperature": self.vitals.ram_temperature,
            "reflected_energy": reflected_energy,
            # ★ Physical perception features
            "attention_strength": perception_data.get("attention_strength", 0.0),
            "left_resonance": perception_data.get("left_resonance", 0.0),
            "right_resonance": perception_data.get("right_resonance", 0.0),
            "perception_bindings": float(perception_data.get("bindings_found", 0)),
        }
        self.causal.observe(causal_vars)

        # ★ Pain event log + trauma memory
        if self.vitals.pain_level > 0.3:
            self._log_event("pain", {
                "pain_level": round(self.vitals.pain_level, 3),
                "temperature": round(self.vitals.ram_temperature, 3),
                "throttle": round(throttle, 3),
                "sensitivity": round(self.vitals.pain_sensitivity, 3),
            })
            # ★ Severe pain → record trauma (vitals + autonomic + motor protection synchronized)
            if self.vitals.pain_level > 0.7:
                # Get frequency of last perceived signal from fusion brain
                last_p = self.fusion_brain._last_perception
                signal_freq = last_p.integrated_signal.frequency if last_p and last_p.integrated_signal else 0.0
                self.vitals.record_trauma(signal_freq)
                self.autonomic.record_trauma()
                # ★ Pain → hand protective reflex (cautious after injury)
                self.hand.guard_level = min(
                    self.hand.guard_level + 0.15, 1.0
                )
                self.hand.injury_memory = min(
                    self.hand.injury_memory + 0.05, 1.0
                )
        else:
            self._log_event("perceive", {"modality": modality.value, "priority": priority.name})

        # ★ Temporal calibration: feed perception signal into calibrator
        perception_signal = None
        if self.fusion_brain._last_perception is not None:
            perception_signal = self.fusion_brain._last_perception.integrated_signal
        if perception_signal is not None:
            frame = self.calibrator.receive_and_bind(perception_signal)
        else:
            frame = None

        # ★ Auditory grounding engine: cross-modal synaptic decay
        self.auditory_grounding.tick()

        # ★ Hypothalamic homeostatic drive — hunger/thirst tick
        #    Metabolic consumption → blood glucose↓ → hunger drive → cognitive penalty + irritability
        #    Water loss → hydration↓ → thirst drive → pain contribution
        _cognitive_load = brain_result["sensory"]["sensory_activity"]
        _motor_activity = brain_result["motor"]["output_strength"]
        homeostatic_signal = self.homeostatic_drive.tick(
            sympathetic=self.autonomic.sympathetic,
            cognitive_load=_cognitive_load,
            motor_activity=_motor_activity,
            core_temp=self.autonomic.core_temp,
            is_sleeping=self.sleep_cycle.is_sleeping(),
        )

        # Homeostatic → pain/emotion injection
        if homeostatic_signal.pain_contribution > 0:
            self.vitals.ram_temperature = min(
                1.0,
                self.vitals.ram_temperature + homeostatic_signal.pain_contribution * 0.05,
            )

        # ★ Autonomic update (add homeostatic irritability → negative emotion)
        _emotional_valence = brain_result["emotional"]["emotional_valence"]
        _emotional_valence -= homeostatic_signal.irritability  # Hunger biases emotion negative
        self.autonomic.tick(
            pain_level=self.vitals.pain_level,
            ram_temperature=self.vitals.ram_temperature,
            emotional_valence=_emotional_valence,
            sensory_load=brain_result["sensory"]["sensory_activity"],
            is_sleeping=self.sleep_cycle.is_sleeping(),
        )

        # ★ Sleep cycle update
        sleep_info = self.sleep_cycle.tick(
            external_stimulus_strength=brain_result["sensory"]["sensory_activity"],
        )

        # ★ Sleep physics engine update
        if self.sleep_cycle.is_sleeping():
            # Get synaptic strength list (for downscaling/SHY computation)
            synaptic_strengths = [
                n.synaptic_strength
                for region in self.fusion_brain.regions.values()
                for n in region.neurons
            ]
            sleep_phys = self.sleep_physics.sleep_tick(
                stage=sleep_info["stage"],
                synaptic_strengths=synaptic_strengths,
            )
        else:
            # Awake: accumulate impedance debt
            reflected_energy = brain_result.get("cycle_reflected_energy", 0.0)
            synaptic_strengths = [
                n.synaptic_strength
                for region in self.fusion_brain.regions.values()
                for n in region.neurons
            ]
            sleep_phys = self.sleep_physics.awake_tick(
                reflected_energy=reflected_energy,
                synaptic_strengths=synaptic_strengths,
            )

        # ★ Lorentz compression fatigue engine — Phase 23 Pollock-Barraclough aging tick
        #   Map channel activity intensity to current → Lorentz compression strain → cumulative aging
        #   During sleep: repair elastic strain + BDNF micro-repair of plastic strain
        _channel_act = {
            "visual": brain_result["sensory"].get("sensory_activity", 0.0),
            "auditory": brain_result["sensory"].get("sensory_activity", 0.0) * 0.8,
            "motor": brain_result["motor"].get("output_strength", 0.0),
            "emotional": abs(brain_result["emotional"].get("emotional_valence", 0.0)),
            "cognitive": brain_result.get("throttle_factor", 0.5),
        }
        _growth = min(1.0, self.autonomic.parasympathetic * 0.5)  # Parasympathetic→BDNF
        aging_signal = self.pinch_fatigue.tick(
            channel_activities=_channel_act,
            temperature=self.vitals.ram_temperature,
            is_sleeping=self.sleep_cycle.is_sleeping(),
            growth_factor=_growth,
        )

        # Sleep memory consolidation
        if sleep_info["should_consolidate"]:
            # ★ Sleep consolidation: replay recent memories → ring migration (RAM→SSD→HDD)
            consolidated = self.fusion_brain.sleep_consolidate(
                consolidation_rate=sleep_info["consolidation_rate"]
            )
            # ★ Hippocampus → semantic field consolidation migration
            #   Episodic memory replay → semantic field attractor strengthening (mass↑, Q↑)
            #   "You forgot what you had for breakfast last Tuesday, but you know what 'breakfast' means."
            hippo_consolidated = self.hippocampus.consolidate(
                semantic_field=self.semantic_field,
                max_episodes=5,
            )

        # ★ Consciousness update
        attention_str = perception_data.get("attention_strength", 0.5)
        cal_quality = self.calibrator.get_calibration_quality()
        temporal_res = self.calibrator.get_temporal_resolution()
        wm_usage = len(self.working_memory.get_contents()) / max(self.working_memory.capacity, 1)
        arousal = 1.0 - self.autonomic.parasympathetic * 0.5
        sensory_gate = self.sleep_cycle.get_sensory_gate()

        consciousness_result = self.consciousness.tick(
            attention_strength=attention_str,
            binding_quality=cal_quality,
            working_memory_usage=wm_usage,
            arousal=arousal,
            sensory_gate=sensory_gate,
            pain_level=self.vitals.pain_level,
            temporal_resolution=temporal_res,
        )

        # ================================================================
        # ★ Closed-loop integration — THE LIFE LOOP
        # ================================================================
        # 1. Autonomic → pupil → eye
        pupil = self.autonomic.get_pupil_aperture()
        self.eye.adjust_pupil(pupil)

        # 2. Autonomic → energy → affects motor gain (hand, mouth)
        # (Passed through life_loop's energy parameter)

        # 3. Consciousness attention target → focus notification
        if perception_data.get("concept"):
            self.consciousness.focus_attention(
                target=perception_data["concept"],
                modality=modality.value,
                salience=perception_data.get("attention_strength", 0.5),
            )

        # 3b. ★ Prefrontal → thalamus top-down attention
        #    Goal-driven attention bias: modalities related to tracked goals get gate gain boost
        #    "When you're looking for keys, the thalamic gate for vision opens wider."
        top_goal = self.prefrontal.get_top_goal()
        if top_goal is not None:
            goal_modality = getattr(top_goal, "modality", None)
            if goal_modality is None:
                # Fallback: use current task set name as modality bias
                goal_modality = self.prefrontal._current_task
            if goal_modality:
                goal_bias = min(1.0, 0.5 + getattr(top_goal, "priority", 0.5) * 0.3)
                self.thalamus.set_attention(goal_modality, goal_bias)

        # 4. Consciousness → global workspace broadcast
        if consciousness_result.get("is_meta_aware", False):
            self.consciousness.broadcast_to_workspace(
                content={
                    "perception": perception_data.get("concept"),
                    "pain": self.vitals.pain_level,
                    "calibration_quality": cal_quality,
                },
                source="perceive_loop",
            )

        # 5. Run life loop — closed-loop error calculation + compensation commands
        loop_state = self.life_loop.tick(
            visual_signal=self._last_visual_signal,
            auditory_signal=self._last_auditory_signal,
            proprioception_hand=self.hand.get_proprioception(),
            proprioception_mouth=self.mouth.get_proprioception(),
            interoception=self.autonomic.get_signal(),
            visual_target=self._current_visual_target,
            hand_target=self._current_hand_target,
            pitch_target=self._current_pitch_target,
            hand_position=np.array([self.hand.x, self.hand.y]),
            current_pitch=self.mouth._current_pitch,
            calibration_drifts={
                m: self.calibrator.get_drift_for(m)
                for m in ["visual", "auditory", "proprioception"]
            },
            calibration_quality=cal_quality,
            consciousness_phi=consciousness_result.get("phi", 0.5),
            arousal=arousal,
            sensory_gate=sensory_gate,
            ram_temperature=self.vitals.ram_temperature,
            energy=self.autonomic.energy,
            pain_level=self.vitals.pain_level,
            pupil_aperture=pupil,
            autonomic_balance=self.autonomic.get_autonomic_balance(),
        )

        # 6. Persistent error → pain feedback (chronic stress → chronic pain)
        error_pain = self.life_loop.get_error_to_pain()
        if error_pain > 0:
            self.vitals.ram_temperature = min(
                1.0,
                self.vitals.ram_temperature + error_pain * 0.05,
            )

        # 7. ★ Compensation command dispatch — final link in the closed-loop
        #    Commands from LifeLoop → actual body organ execution
        dispatch_results = self._dispatch_commands(loop_state.commands)

        # 8. ★ Neural pruning — sensory stimuli drive cortical specialization
        #    Each perception feeds signal to corresponding cortex → Hebbian selection → Γ² → min
        self._stimulate_pruning(modality, perception_signal)

        # 8.5 ★ Cross-modal impedance adaptation — record binding attempts + forgetting decay
        #    Successful binding → Γ decreases (impedance matching improves)
        #    Failure → Γ slightly rises (negative reinforcement)
        #    Cortisol → Yerkes-Dodson modulation of learning rate
        _binding_success = (perception_data.get("bindings_found", 0) > 0)
        _binding_quality_for_adapt = cal_quality
        # Update modality list (frame is now acquired)
        if frame is not None:
            _active_modalities = list(set(
                _active_modalities + getattr(frame, 'modalities', [])
            ))
        if len(_active_modalities) >= 2:
            _adapt_result = self.impedance_adaptation.record_binding_attempt(
                modality_a=_active_modalities[0],
                modality_b=_active_modalities[1],
                success=_binding_success,
                binding_quality=_binding_quality_for_adapt,
                cortisol=self.autonomic.cortisol,
                chronic_stress=self.autonomic.chronic_stress_load,
            )
        else:
            _adapt_result = None
        # Forgetting decay (use it or lose it)
        self.impedance_adaptation.decay_tick()

        # 9. ★ Attention plasticity — natural decay each tick
        #    Use it or lose it: synaptic time constants, Q, inhibition efficiency
        self.attention_plasticity.decay_tick()

        # 10. ★ Cognitive flexibility — update each tick
        #    Inertia charge/discharge + mixing cost + flexibility decay
        self.cognitive_flexibility.sync_pfc_energy(self.prefrontal._energy)
        self.cognitive_flexibility.tick()

        # 11. ★ Curiosity drive — update each tick
        #    Boredom accumulation + curiosity decay + spontaneous behavior generation
        has_input = brain_result["sensory"]["sensory_activity"] > 0.1
        curiosity_result = self.curiosity_drive.tick(
            has_external_input=has_input,
            sensory_load=brain_result["sensory"]["sensory_activity"],
            energy=self.autonomic.energy,
            is_sleeping=self.sleep_cycle.is_sleeping(),
        )

        # 12. ★ Mirror neurons — update each tick
        #    Empathy decay + social bond maintenance + maturation update
        mirror_result = self.mirror_neurons.tick(
            has_social_input=False,  # Default no social input (triggered by observe_*)
            own_valence=self.amygdala.get_valence(),
            own_arousal=self.autonomic.sympathetic,
        )

        # 12b. ★ Social resonance — Phase 19 social homeostasis tick
        #    Social need growth + empathic energy recovery + agent model decay
        #    Sync ToM capacity from mirror neurons
        social_result = self.social_resonance.tick(
            has_social_input=mirror_result.get("has_social_input", False),
            own_valence=self.amygdala.get_valence(),
            own_arousal=self.autonomic.sympathetic,
            empathy_capacity=self.mirror_neurons.get_empathy_capacity(),
            tom_capacity=self.mirror_neurons.get_tom_capacity(),
        )

        # 12c. ★ Narrative memory — Phase 20.1 autobiographical memory management
        #    Update summary cache each tick + maintain narrative arc
        narrative_result = self.narrative_memory.tick()

        # 12c½. ★ Emotion granularity engine — Phase 36 tick
        #    Inject social/curiosity/homeostatic emotion sources → differential decay → VAD computation → compound emotion detection
        #    "Emotion is not a single dial—it is an 8-dimensional impedance map"
        _eg = self.emotion_granularity
        # Social emotions
        _eg.inject_social(
            empathy_valence=mirror_result.get("empathic_valence", 0.0),
            social_bond_strength=social_result.get("social_bond", 0.0) if isinstance(social_result, dict) else 0.0,
        )
        # Curiosity emotions
        if isinstance(curiosity_result, dict):
            _eg.inject_novelty(
                surprise_level=curiosity_result.get("novelty", 0.0),
                curiosity_satisfied=curiosity_result.get("curiosity_satisfied", False),
            )
        # Homeostatic emotions
        _eg.inject_homeostatic(
            satisfaction=max(0.0, 1.0 - homeostatic_signal.hunger_intensity - homeostatic_signal.thirst_intensity),
            deficit=max(homeostatic_signal.hunger_intensity, homeostatic_signal.thirst_intensity),
            irritability=homeostatic_signal.irritability,
        )
        # Execute tick
        emotion_granularity_result = _eg.tick()

        # 12d. ★ Recursive grammar — Phase 20.2 rule maintenance
        #    Learn new rules from Wernicke chunks + rule confidence decay
        grammar_result = self.recursive_grammar.tick()

        # 12e. ★ Semantic pressure — Phase 21 language thermodynamics
        #    Accumulate semantic pressure + inner monologue check + Wernicke→Broca direct drive
        #    The physical mechanism of "saying it out loud makes it better"
        _field_state = self.semantic_field.get_state()
        _active_concepts = _field_state.get("top_concepts", [])
        _valence = self.amygdala.get_valence()
        _arousal_for_pressure = self.autonomic.sympathetic
        _phi = consciousness_result.get("phi", 0.5)
        _pain = self.vitals.pain_level

        semantic_pressure_result = self.semantic_pressure.tick(
            active_concepts=_active_concepts,
            valence=_valence,
            arousal=_arousal_for_pressure,
            phi=_phi,
            pain=_pain,
            semantic_field=self.semantic_field,
            wernicke=self.wernicke,
            broca=self.broca,
            tick_id=self._cycle_count,
        )

        # 13. ★ Predictive processing — Phase 17 eye of time
        #    Forward model predicts next tick → compare with reality → surprise signal
        #    Mental simulation → preemptive action suggestions
        predictive_result = self.predictive.tick(
            temperature=self.vitals.ram_temperature,
            pain=self.vitals.pain_level,
            energy=self.autonomic.energy,
            arousal=self.autonomic.sympathetic,
            stability=self.vitals.stability_index,
            consciousness=consciousness_result.get("phi", 0.5),
            cortisol=self.autonomic.cortisol,
            heart_rate=self.vitals.heart_rate,
            pfc_energy=self.prefrontal._energy,
            is_sleeping=self.sleep_cycle.is_sleeping(),
        )

        # 14. Update sensory signal cache (for next tick)
        if modality == Modality.VISUAL:
            self._last_visual_signal = perception_signal
        elif modality == Modality.AUDITORY:
            self._last_auditory_signal = perception_signal

        # Inject vitals into result
        brain_result["vitals"] = self.vitals.get_vitals()
        brain_result["throttle_factor"] = throttle  # Temporary (step 16 will overwrite with thinking-rate-adjusted version)
        brain_result["calibration"] = {
            "frame_id": frame.frame_id if frame else None,
            "bound_modalities": frame.modalities if frame else [],
            "bindings": len(frame.bindings) if frame else 0,
            "quality": cal_quality,
        }
        brain_result["autonomic"] = self.autonomic.get_vitals()
        brain_result["impedance_adaptation"] = {
            "adapted_gamma": round(binding_gamma, 4),
            "raw_gamma": round(_raw_binding_gamma, 4),
            "experience_gamma": round(_adapted_gamma, 4) if len(_active_modalities) >= 2 else None,
            "binding_attempt": _adapt_result,
            "tracked_pairs": len(self.impedance_adaptation._pairs),
            "total_adaptations": self.impedance_adaptation._total_adaptations,
        }
        brain_result["sleep"] = sleep_info
        brain_result["consciousness"] = consciousness_result
        brain_result["life_loop"] = {
            "tick_id": loop_state.tick_id,
            "total_error": round(loop_state.total_error, 4),
            "compensation_success": round(loop_state.compensation_success, 4),
            "errors": [
                {
                    "type": e.error_type.value,
                    "magnitude": round(e.magnitude, 4),
                    "urgency": round(e.urgency, 4),
                    "compensation": e.compensation.value,
                }
                for e in loop_state.errors
            ],
            "commands": [
                {
                    "action": c.action.value,
                    "strength": round(c.strength, 4),
                    "source_error": c.source_error.value,
                }
                for c in loop_state.commands
            ],
            "persistent_errors": self.life_loop.get_persistent_errors(),
            "prediction_accuracy": round(self.life_loop.get_prediction_accuracy(), 4),
            "dispatched": dispatch_results,
        }
        brain_result["predictive"] = predictive_result
        brain_result["social_resonance"] = social_result
        brain_result["narrative_memory"] = narrative_result
        brain_result["emotion_granularity"] = {
            "valence": emotion_granularity_result.valence,
            "arousal": emotion_granularity_result.arousal,
            "dominance": emotion_granularity_result.dominance,
            "dominant_emotion": emotion_granularity_result.dominant_emotion,
            "compound_emotions": emotion_granularity_result.compound_emotions,
            "richness": emotion_granularity_result.richness,
            "entropy": emotion_granularity_result.entropy,
            "Z_emotion": emotion_granularity_result.Z_emotion,
            "gamma_emotion": emotion_granularity_result.gamma_emotion,
        }
        brain_result["recursive_grammar"] = grammar_result
        brain_result["semantic_pressure"] = semantic_pressure_result
        brain_result["homeostatic_drive"] = {
            "glucose": round(self.homeostatic_drive.glucose, 4),
            "hydration": round(self.homeostatic_drive.hydration, 4),
            "hunger_drive": round(self.homeostatic_drive.hunger_drive, 4),
            "thirst_drive": round(self.homeostatic_drive.thirst_drive, 4),
            "needs_food": homeostatic_signal.needs_food,
            "needs_water": homeostatic_signal.needs_water,
            "cognitive_penalty": round(homeostatic_signal.cognitive_penalty, 4),
        }
        brain_result["pinch_fatigue"] = {
            "mean_age": round(aging_signal.mean_age_factor, 6),
            "max_age": round(aging_signal.max_age_factor, 6),
            "cognitive_impact": round(aging_signal.cognitive_impact, 6),
            "degraded_channels": aging_signal.degraded_channels,
            "total_plastic_strain": round(aging_signal.total_plastic_strain, 6),
            "impedance_drift": round(aging_signal.mean_impedance_drift, 6),
        }

        # 14b. ★ Phantom limb pain update — Phase 24
        #   If amputation records exist, compute residual motor command reflex energy, neuroma discharge, emotional triggers
        _emotional_val = brain_result["emotional"].get("emotional_valence", 0.0)
        _stress = self.autonomic.get_threat_level() if hasattr(self.autonomic, 'get_threat_level') else self.vitals.pain_level
        phantom_result = self.phantom_limb.tick(
            emotional_valence=_emotional_val,
            stress_level=_stress,
        )
        brain_result["phantom_limb"] = phantom_result

        # 14c. ★ Clinical neurology update — Phase 25
        #   Five major neurological diseases (stroke/ALS/dementia/Alzheimer's/cerebral palsy) unified tick
        clinical_result = self.clinical_neurology.tick(
            brain_state=brain_result,
        )
        brain_result["clinical_neurology"] = clinical_result

        # 14d. ★ Computational pharmacology update — Phase 26
        #   Unified α_drug drive: MS / PD / epilepsy / depression
        pharma_result = self.pharmacology.tick(
            brain_state=brain_result,
        )
        brain_result["pharmacology"] = pharma_result

        # 15. ★ Metacognition — Phase 18 internal auditor
        #    Aggregate whole-brain cognitive impedance → compute Γ_thinking
        #    System 1/2 switching → counterfactual reasoning → self-correction
        metacognition_result = self.metacognition.tick(
            prediction_error=predictive_result.get("prediction_error", 0.0),
            free_energy=predictive_result.get("free_energy", 0.0),
            binding_gamma=binding_gamma,
            flexibility_index=self.cognitive_flexibility.get_flexibility_index(),
            anxiety=predictive_result.get("anxiety_level", 0.0),
            pfc_energy=self.prefrontal._energy,
            surprise=predictive_result.get("surprise", 0.0),
            pain=self.vitals.pain_level,
            phi=consciousness_result.get("phi", 0.5),
            novelty=curiosity_result.get("novelty", 0.0) if isinstance(curiosity_result, dict) else 0.0,
            boredom=curiosity_result.get("boredom", 0.0) if isinstance(curiosity_result, dict) else 0.0,
            precision=self.predictive._precision,
            last_action=self.predictive._last_action,
            is_sleeping=self.sleep_cycle.is_sleeping(),
        )
        brain_result["metacognition"] = metacognition_result

        # 16. ★ Physical execution of metacognition
        #    (a) Thinking rate → actual throttle: "thinking is too hard so responses slow down"
        thinking_rate = metacognition_result.get("thinking_rate", 1.0)
        effective_throttle = min(throttle, thinking_rate)
        if effective_throttle < throttle:
            extra_delay_ms = (thinking_rate - effective_throttle) * 3.0
            if extra_delay_ms > 0.1:
                time.sleep(extra_delay_ms / 1000.0)
        brain_result["throttle_factor"] = round(effective_throttle, 4)

        #    (b) Self-correction → cognitive reframing: "interrupt and retry"
        #        Trigger task switch + flush weakest working memory
        if metacognition_result.get("is_correcting", False):
            reframe_switch = self.cognitive_flexibility.attempt_switch(
                "metacognitive_reframe", forced=True,
            )
            flushed = self.working_memory.flush_weakest(fraction=0.3)
            self._log_event("metacognitive_reframe", {
                "correction_count": metacognition_result.get("correction_count", 0),
                "gamma_thinking": metacognition_result.get("gamma_thinking", 0),
                "switch_cost_ms": reframe_switch.switch_cost_ms,
                "wm_flushed": flushed,
            })

        self._state = "idle"
        return brain_result

    # ------------------------------------------------------------------
    def think(
        self,
        question: str,
        context_vars: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Think: perform reasoning using causal inference + working memory.

        - If question contains "why" → causal reasoning
        - If question contains "what if" → counterfactual reasoning
        - Otherwise → associative reasoning
        """
        self._state = "thinking"

        # Meta-learning: select current best strategy
        strategy = self.meta.select_strategy()
        params = strategy.params

        result: Dict[str, Any] = {
            "question": question,
            "strategy_used": strategy.name,
            "working_memory_state": self.working_memory.get_contents()[:3],
            "pfc_state": self.prefrontal.get_state(),
        }

        if context_vars and len(context_vars) >= 2:
            keys = list(context_vars.keys())

            if "what if" in question.lower() or "if" in question.lower():
                # Counterfactual reasoning
                cf = self.causal.counterfactual(
                    factual=context_vars,
                    counterfactual_var=keys[0],
                    counterfactual_value=context_vars[keys[0]] * 2,
                    target=keys[-1],
                )
                result["reasoning"] = cf
                result["type"] = "counterfactual"
            elif "why" in question.lower():
                # Causal reasoning
                inf = self.causal.infer(keys[0], keys[-1])
                result["reasoning"] = inf
                result["type"] = "causal"
            else:
                # Intervention reasoning
                intv = self.causal.intervene(keys[0], context_vars[keys[0]], keys[-1])
                result["reasoning"] = intv
                result["type"] = "intervention"
        else:
            result["reasoning"] = {"note": "context_vars required for causal reasoning"}
            result["type"] = "observation"

        self._log_event("think", {"question": question})
        self._state = "idle"
        return result

    # ------------------------------------------------------------------
    def act(
        self,
        state: str,
        available_actions: List[str],
    ) -> Dict[str, Any]:
        """
        Act: reinforcement learning selects optimal action.

        ★ Phase 6.2: Basal ganglia action selection + dual-system arbitration
        """
        self._state = "acting"

        # Meta-learning adjusts RL epsilon
        meta_params = self.meta.get_current_params()
        self.rl.epsilon = max(self.rl.epsilon_min, meta_params.get("exploration", self.rl.epsilon))

        # ★ Basal ganglia action selection (habitual system)
        bg_result = self.basal_ganglia.select_action(state, available_actions)

        # ★ Physics reward engine selection (replaces Q-table ε-greedy)
        rl_action, explored = self.physics_reward.choose_action(state, available_actions)

        # ★ Prefrontal Go/NoGo gate — final ruling on basal ganglia selection
        top_goal = self.prefrontal.get_top_goal()
        if top_goal:
            z_action = 75.0 * (1.0 + bg_result.gamma_habit)  # Habit Γ → impedance
            gate = self.prefrontal.evaluate_action(
                bg_result.selected_action, z_action=z_action,
            )
            pfc_approved = gate.decision != "nogo"
        else:
            pfc_approved = True

        # Final action selection
        if bg_result.is_habitual and pfc_approved:
            action = bg_result.selected_action
            action_source = "habitual"
        else:
            action = rl_action
            action_source = "goal_directed"
            explored = explored

        result = {
            "state": state,
            "chosen_action": action,
            "explored": explored,
            "action_source": action_source,
            "epsilon": round(self.rl.epsilon, 4),
            "q_values": {
                a: round(self.physics_reward.get_q_value(state, a), 4) for a in available_actions
            },
            "basal_ganglia": {
                "selected": bg_result.selected_action,
                "is_habitual": bg_result.is_habitual,
                "gamma_habit": bg_result.gamma_habit,
                "reaction_time": bg_result.reaction_time,
            },
        }

        # Store in working memory
        # ★ Action memory has lower binding_gamma (pure internal decision, no cross-modal binding needed)
        self.working_memory.store(
            key=f"action_{self._cycle_count}",
            content=result,
            importance=0.6,
            binding_gamma=bg_result.gamma_habit * 0.3,  # More habitual → smaller gamma
        )

        self._log_event("act", {"state": state, "action": action})
        self._state = "idle"
        return result

    # ------------------------------------------------------------------
    def learn_from_feedback(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_actions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Learn: learn from reward feedback.

        1. RL TD update
        2. Meta-learning report performance
        3. Causal observation (action → outcome)
        4. ★ Phase 6.2: Basal ganglia habit update + RPE dopamine modulation
        5. ★ Phase 6.1: Prefrontal tick (energy recovery)
        """
        self._state = "learning"

        # ★ Physics reward learning (impedance-matching Hebbian learning replaces TD(0))
        dopamine = self.physics_reward.update(state, action, reward, next_state, next_actions)

        # Experience replay (offline impedance restructuring)
        replay_error = self.physics_reward.replay(batch_size=min(16, len(self.physics_reward._experience_buffer)))

        # Dopamine injection into basal ganglia (unified dopamine pipeline)
        self.basal_ganglia._dopamine_level = float(np.clip(
            self.basal_ganglia._dopamine_level + dopamine * 0.1,
            0.0, 1.0,
        ))

        # Meta-learning report
        self.meta.report_performance(reward)

        # Causal observation
        self.causal.observe(
            {
                "action_value": self.rl.get_q_value(state, action),
                "reward": reward,
                "dopamine": dopamine,
            }
        )

        # ★ Basal ganglia: update habit strength + RPE dopamine modulation
        success = reward > 0
        bg_update = self.basal_ganglia.update_after_action(
            state, action, reward, success,
        )

        # ★ Prefrontal + basal ganglia tick (energy recovery, cooldown)
        self.prefrontal.tick()
        self.basal_ganglia.tick()

        result = {
            "dopamine_signal": round(dopamine, 4),
            "replay_error": round(replay_error, 4),
            "reward": reward,
            "learning_rate": self.rl.lr,
            "meta_strategy": self.meta.current_strategy.name if self.meta.current_strategy else "none",
            "basal_ganglia_update": bg_update,
        }

        self._log_event("learn", {"reward": reward, "dopamine": round(dopamine, 4)})
        self._state = "idle"
        return result

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        """
        Introspection: report full system state including vitals.
        """
        uptime = time.time() - self._start_time
        return {
            "state": self._state,
            "cycle_count": self._cycle_count,
            "uptime_seconds": round(uptime, 1),
            "vitals": self.vitals.get_vitals(),
            "subsystems": {
                "fusion_brain": self.fusion_brain.get_brain_state(),
                "working_memory": self.working_memory.get_stats(),
                "reinforcement_learning": self.physics_reward.get_stats(),
                "causal_reasoning": self.causal.get_stats(),
                "meta_learning": self.meta.get_stats(),
                "hand": self.hand.get_stats(),
                "eye": self.eye.get_stats(),
                "ear": self.ear.get_stats(),
                "mouth": self.mouth.get_stats(),
                "calibrator": self.calibrator.get_stats(),
                "autonomic": self.autonomic.get_stats(),
                "sleep": self.sleep_cycle.get_stats(),
                "sleep_physics": self.sleep_physics.get_state(),
                "consciousness": self.consciousness.get_stats(),
                "life_loop": self.life_loop.get_stats(),
                "pruning": self.pruning.get_development_state(),
                "auditory_grounding": self.auditory_grounding.get_state(),
                "semantic_field": self.semantic_field.get_state(),
                "broca": self.broca.get_state(),
                "hippocampus": self.hippocampus.get_state(),
                "wernicke": self.wernicke.get_state(),
                "thalamus": self.thalamus.get_state(),
                "amygdala": self.amygdala.get_state(),
                "prefrontal": self.prefrontal.get_stats(),
                "basal_ganglia": self.basal_ganglia.get_stats(),
                "attention_plasticity": self.attention_plasticity.get_state(),
                "cognitive_flexibility": self.cognitive_flexibility.get_state(),
                "curiosity_drive": self.curiosity_drive.get_stats(),
                "metacognition": self.metacognition.get_stats(),
                "social_resonance": self.social_resonance.get_stats(),
                "narrative_memory": self.narrative_memory.get_stats(),
                "recursive_grammar": self.recursive_grammar.get_stats(),
                "semantic_pressure": self.semantic_pressure.get_state(),
                "homeostatic_drive": self.homeostatic_drive.get_state(),
                "pinch_fatigue": self.pinch_fatigue.get_state(),
                "phantom_limb": self.phantom_limb.introspect(),
                "clinical_neurology": self.clinical_neurology.introspect(),
                "pharmacology": self.pharmacology.introspect(),
            },
            "recent_events": self._event_log[-10:],
        }

    # ------------------------------------------------------------------
    def get_vitals(self) -> Dict[str, Any]:
        """Get vitals (for API endpoint use)."""
        return self.vitals.get_vitals()

    # ------------------------------------------------------------------
    def get_waveforms(self, last_n: int = 60) -> Dict[str, List[float]]:
        """Get waveform data (for dashboard plotting)."""
        return self.vitals.get_waveforms(last_n)

    # ------------------------------------------------------------------
    def get_oscilloscope_data(self) -> Dict[str, Any]:
        """
        Get oscilloscope data (for oscilloscope dashboard use).

        Includes:
         - Input waveform
         - Per-coaxial-channel transmission waveforms + reflection coefficients
         - Resonance results
         - Vitals time series
        """
        scope = self.fusion_brain.get_oscilloscope_data()
        # Merge vitals waveforms
        scope["vitals"] = self.vitals.get_waveforms(last_n=128)
        return scope

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ★ Eye — Visual Input
    # ------------------------------------------------------------------

    def see(
        self,
        pixels: np.ndarray,
        priority: Priority = Priority.NORMAL,
        visual_target: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        See — eye physical transduction + brain perception.

        pixels → lens(FFT) → retina → optic nerve → ElectricalSignal → perceive
        ★ Closed-loop: autonomic → pupil → eye  (ambient brightness adaptation)
        """
        self._state = "seeing"

        # ★ Cognitive flexibility: detect modality switch (visual task set)
        prev_task = self.cognitive_flexibility.get_current_task()
        if prev_task != "visual":
            switch_result = self.cognitive_flexibility.attempt_switch("visual")
            if switch_result.perseveration_error:
                self._log_event("task_switch_perseveration", {
                    "from": prev_task, "to": "visual",
                    "inertia": switch_result.inertia_penalty_ms,
                })
        else:
            self.cognitive_flexibility.notify_task("visual")

        # Autonomic → pupil aperture (closed-loop #1)
        pupil = self.autonomic.get_pupil_aperture()
        self.eye.adjust_pupil(pupil)

        # Eye physical transduction
        visual_signal = self.eye.see(pixels)

        # ★ Semantic field: visual fingerprint → concept recognition (physics symmetric with auditory)
        visual_fp = self.eye.get_last_fingerprint()
        semantic_result = None
        if visual_fp is not None:
            semantic_result = self.semantic_field.process_fingerprint(
                visual_fp, modality="visual"
            )

        # ★ Thalamic gate: visual signal must pass through thalamus to reach cortex
        arousal = 1.0 - self.autonomic.parasympathetic * 0.5
        thalamic_gamma = semantic_result.get("gamma", 0.5) if semantic_result else 0.5
        thal_result = self.thalamus.gate(
            modality="visual",
            fingerprint=visual_fp,
            amplitude=visual_signal.amplitude,
            gamma=thalamic_gamma,
            arousal=arousal,
        )

        # ★ Amygdala: rapid threat assessment (low road — doesn't wait for cortical analysis)
        best_concept = semantic_result.get("best_concept") if semantic_result else None
        gamma_val = semantic_result.get("gamma", 1.0) if semantic_result else 1.0
        amyg_response = self.amygdala.evaluate(
            modality="visual",
            fingerprint=visual_fp,
            gamma=gamma_val,
            amplitude=visual_signal.amplitude,
            pain_level=self.vitals.pain_level,
            concept_label=best_concept,
        )
        # Amygdala output → thalamic attention modulation (threat signal boosts gate gain)
        if amyg_response.emotional_state.threat_level > 0.3:
            self.thalamus.set_attention("visual", min(1.0, 0.5 + amyg_response.emotional_state.threat_level))
        # Amygdala → autonomic (fight-or-flight drives sympathetic)
        if amyg_response.sympathetic_command > 0:
            self.autonomic.sympathetic = min(
                1.0, self.autonomic.sympathetic + amyg_response.sympathetic_command * 0.3
            )
        # ★ Amygdala → emotion granularity engine (threat unfolds into Plutchik vector)
        self.emotion_granularity.inject_threat(
            threat_level=amyg_response.emotional_state.threat_level,
            pain_level=self.vitals.pain_level,
            fear_matched=amyg_response.fear_matched,
            dominance_sense=self.emotion_granularity.get_dominance(),
        )
        # Amygdala emotion decay
        self.amygdala.decay_tick()

        # ★ Attention plasticity: visual exposure tracking
        if thal_result.passed:
            self.attention_plasticity.on_exposure("visual")

        # ★ Curiosity: evaluate visual novelty
        self.curiosity_drive.evaluate_novelty(
            modality="visual",
            signal_impedance=visual_signal.impedance,
            signal_amplitude=visual_signal.amplitude,
        )

        # ★ Auditory grounding: visual signal fed into cross-modal binding (the "food" end of Pavlovian conditioning)
        self.auditory_grounding.receive_signal(visual_signal)

        # Update closed-loop state
        self._last_visual_signal = visual_signal
        if visual_target is not None:
            self._current_visual_target = visual_target

        # Feed into calibrator
        self.calibrator.receive(visual_signal)

        # Feed signal waveform into perception pipeline
        result = self.perceive(
            stimulus=visual_signal.waveform,
            modality=Modality.VISUAL,
            priority=priority,
        )
        result["visual"] = {
            "frequency": visual_signal.frequency,
            "amplitude": visual_signal.amplitude,
            "band": visual_signal.band.value,
            "snr": round(visual_signal.snr, 2),
            "pupil_aperture": round(pupil, 3),
            "source": "eye",
        }
        # ★ Thalamic gate result
        result["thalamus"] = {
            "passed": thal_result.passed,
            "gate_gain": thal_result.gate_gain,
            "salience": thal_result.salience,
            "is_startle": thal_result.is_startle,
            "reason": thal_result.reason,
        }
        # ★ Amygdala threat assessment result
        result["amygdala"] = {
            "valence": amyg_response.emotional_state.valence,
            "threat_level": amyg_response.emotional_state.threat_level,
            "emotion": amyg_response.emotional_state.emotion_label,
            "is_fight_flight": amyg_response.emotional_state.is_fight_flight,
            "fear_matched": amyg_response.fear_matched,
            "gamma_threat": amyg_response.gamma_threat,
        }
        if semantic_result is not None:
            result["semantic"] = semantic_result
            # ★ Hippocampus: record visual snapshot to current episode (includes amygdala valence)
            if visual_fp is not None:
                self.hippocampus.record(
                    modality="visual",
                    fingerprint=visual_fp,
                    attractor_label=best_concept,
                    gamma=gamma_val,
                    valence=amyg_response.emotional_state.valence,
                )
            # ★ Wernicke's area: observe concept sequence
            if best_concept is not None:
                wernicke_obs = self.wernicke.observe(best_concept)
                result["wernicke"] = wernicke_obs
        return result

    # ------------------------------------------------------------------
    # ★ Ear — Auditory Input
    # ------------------------------------------------------------------

    def hear(
        self,
        sound_wave: np.ndarray,
        priority: Priority = Priority.NORMAL,
    ) -> Dict[str, Any]:
        """
        Hear — ear physical transduction + brain perception.

        sound_wave → cochlea(FFT) → ElectricalSignal → perceive
        + Temporal calibrator receives auditory signal
        """
        self._state = "hearing"

        # ★ Cognitive flexibility: detect modality switch (auditory task set)
        prev_task = self.cognitive_flexibility.get_current_task()
        if prev_task != "auditory":
            switch_result = self.cognitive_flexibility.attempt_switch("auditory")
            if switch_result.perseveration_error:
                self._log_event("task_switch_perseveration", {
                    "from": prev_task, "to": "auditory",
                    "inertia": switch_result.inertia_penalty_ms,
                })
        else:
            self.cognitive_flexibility.notify_task("auditory")

        # Ear physical transduction
        auditory_signal = self.ear.hear(sound_wave)

        # Closed-loop: update auditory signal cache
        self._last_auditory_signal = auditory_signal

        # ★ Auditory grounding: cochlea filter bank decomposition → cross-modal Hebbian binding
        ag_result = self.auditory_grounding.receive_auditory(sound_wave)

        # ★ Semantic field: fingerprint → concept recognition
        semantic_result = None
        aud_fp = ag_result.get("fingerprint")
        best_concept = None
        gamma_val = 1.0
        result_extra_wernicke = None

        if aud_fp is not None:
            semantic_result = self.semantic_field.process_fingerprint(
                aud_fp, modality="auditory"
            )
            best_concept = semantic_result.get("best_concept")
            gamma_val = semantic_result.get("gamma", 1.0)

        # ★ Thalamic gate: auditory signal must pass through thalamus to reach cortex
        arousal = 1.0 - self.autonomic.parasympathetic * 0.5
        thal_result = self.thalamus.gate(
            modality="auditory",
            fingerprint=aud_fp,
            amplitude=auditory_signal.amplitude,
            gamma=gamma_val,
            arousal=arousal,
        )

        # ★ Amygdala: rapid threat assessment (low road)
        amyg_response = self.amygdala.evaluate(
            modality="auditory",
            fingerprint=aud_fp,
            gamma=gamma_val,
            amplitude=auditory_signal.amplitude,
            pain_level=self.vitals.pain_level,
            concept_label=best_concept,
        )
        if amyg_response.emotional_state.threat_level > 0.3:
            self.thalamus.set_attention("auditory", min(1.0, 0.5 + amyg_response.emotional_state.threat_level))
        if amyg_response.sympathetic_command > 0:
            self.autonomic.sympathetic = min(
                1.0, self.autonomic.sympathetic + amyg_response.sympathetic_command * 0.3
            )
        # ★ Amygdala → emotion granularity engine (threat unfolds into Plutchik vector)
        self.emotion_granularity.inject_threat(
            threat_level=amyg_response.emotional_state.threat_level,
            pain_level=self.vitals.pain_level,
            fear_matched=amyg_response.fear_matched,
            dominance_sense=self.emotion_granularity.get_dominance(),
        )
        self.amygdala.decay_tick()

        # ★ Attention plasticity: auditory exposure tracking
        if thal_result.passed:
            self.attention_plasticity.on_exposure("auditory")

        # ★ Curiosity: evaluate auditory novelty + self/other discrimination
        self.curiosity_drive.evaluate_novelty(
            modality="auditory",
            signal_impedance=auditory_signal.impedance,
            signal_amplitude=auditory_signal.amplitude,
        )

        if semantic_result is not None:
            # ★ Hippocampus: record auditory snapshot (includes amygdala valence)
            self.hippocampus.record(
                modality="auditory",
                fingerprint=aud_fp,
                attractor_label=best_concept,
                gamma=gamma_val,
                valence=amyg_response.emotional_state.valence,
            )
            # ★ Wernicke's area: observe concept sequence
            if best_concept is not None:
                wernicke_obs = self.wernicke.observe(best_concept)
                result_extra_wernicke = wernicke_obs

        # Feed into calibrator (cross-modal temporal binding)
        self.calibrator.receive(auditory_signal)

        # Feed signal waveform into perception pipeline
        result = self.perceive(
            stimulus=auditory_signal.waveform,
            modality=Modality.AUDITORY,
            priority=priority,
        )
        result["auditory"] = {
            "frequency": auditory_signal.frequency,
            "amplitude": auditory_signal.amplitude,
            "band": auditory_signal.band.value,
            "source": "ear",
        }
        # ★ Thalamic gate result
        result["thalamus"] = {
            "passed": thal_result.passed,
            "gate_gain": thal_result.gate_gain,
            "salience": thal_result.salience,
            "is_startle": thal_result.is_startle,
            "reason": thal_result.reason,
        }
        # ★ Amygdala threat assessment result
        result["amygdala"] = {
            "valence": amyg_response.emotional_state.valence,
            "threat_level": amyg_response.emotional_state.threat_level,
            "emotion": amyg_response.emotional_state.emotion_label,
            "is_fight_flight": amyg_response.emotional_state.is_fight_flight,
            "fear_matched": amyg_response.fear_matched,
            "gamma_threat": amyg_response.gamma_threat,
        }
        if semantic_result is not None:
            result["semantic"] = semantic_result
            if result_extra_wernicke is not None:
                result["wernicke"] = result_extra_wernicke
        return result

    # ------------------------------------------------------------------
    # ★ Mouth — Vocal Output
    # ------------------------------------------------------------------

    def say(
        self,
        target_pitch: float,
        volume: float = 0.5,
        vowel: Optional[str] = None,
        concept: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Say — PID control of vocal cord target pitch.

        Motor cortex → vocal cord tension → airflow vibration → sound wave
        + Vocal cord feedback signal fed into calibrator

        ★ Phase 4.3: if concept is provided, use Broca engine to plan articulation
        """
        self._state = "speaking"

        # ★ Closed-loop: record pitch target (for life_loop error calculation)
        self._current_pitch_target = target_pitch

        # ★ Curiosity: register efference copy (foundation of self-identification)
        #   "I am about to say a sound at target_pitch Hz"
        predicted_z = 50.0 + abs(target_pitch - 200) * 0.1  # pitch → predicted impedance
        self.curiosity_drive.register_efference_copy(
            modality="vocal",
            predicted_impedance=predicted_z,
            action_description=f"say_{target_pitch:.0f}Hz",
        )

        # ★ Phase 4.3: if concept label exists, take Broca pathway
        broca_result = None
        if concept is not None and self.broca.has_plan(concept):
            broca_result = self.broca.speak_concept(
                concept_label=concept,
                mouth=self.mouth,
                semantic_field=self.semantic_field,
                ram_temperature=self.vitals.ram_temperature,
            )

        if broca_result is not None:
            # Broca-generated waveform also needs to produce feedback signal
            broca_wave = broca_result["waveform"]
            broca_fb = self.mouth.get_proprioception()
            speak_result = {
                "waveform": broca_wave,
                "final_pitch": target_pitch,
                "pitch_error": broca_result.get("gamma_loop", 0.0) * 50.0,
                "tremor_intensity": 0.0,
                "volume": volume,
                "signal": broca_fb,
            }
        elif vowel is not None:
            speak_result = self.mouth.say_vowel(
                vowel=vowel,
                pitch=target_pitch,
                volume=volume,
                ram_temperature=self.vitals.ram_temperature,
            )
        else:
            speak_result = self.mouth.speak(
                target_pitch=target_pitch,
                volume=volume,
                ram_temperature=self.vitals.ram_temperature,
            )

        # Vocal cord feedback fed into calibrator
        feedback_signal = speak_result["signal"]
        self.calibrator.receive(feedback_signal)

        # ★ Semantic pressure release — linguistic catharsis
        #   When Γ_speech → 0, pressure is greatly released
        #   Physical realization of "saying it out loud makes it better"
        _gamma_speech = 1.0
        if broca_result is not None:
            _gamma_speech = broca_result.get("gamma_loop", 1.0)
        _phi_for_release = getattr(self.consciousness, 'phi', 0.5)
        pressure_released = self.semantic_pressure.release(_gamma_speech, _phi_for_release)

        self._log_event("speak", {
            "pitch": round(speak_result["final_pitch"], 1),
            "tremor": round(speak_result["tremor_intensity"], 4),
        })

        self._state = "idle"
        result = {
            "final_pitch": speak_result["final_pitch"],
            "pitch_error": speak_result["pitch_error"],
            "tremor_intensity": speak_result["tremor_intensity"],
            "volume": speak_result["volume"],
            "waveform": speak_result["waveform"],
            "waveform_rms": float(np.sqrt(np.mean(speak_result["waveform"] ** 2))),
            "feedback": {
                "frequency": feedback_signal.frequency,
                "amplitude": feedback_signal.amplitude,
                "source": feedback_signal.source,
                "modality": feedback_signal.modality,
            },
        }
        if broca_result is not None:
            result["broca"] = {
                "intended": broca_result.get("intended"),
                "perceived": broca_result.get("perceived"),
                "gamma_loop": broca_result.get("gamma_loop", 1.0),
                "success": broca_result.get("success", False),
            }
        return result

    # ------------------------------------------------------------------
    # ★ Hand Control — Motor Output
    # ------------------------------------------------------------------

    def reach_for(
        self,
        target_x: float,
        target_y: float,
        motor_signal: Optional[ElectricalSignal] = None,
        max_steps: int = 300,
    ) -> Dict[str, Any]:
        """
        Reach and grasp — complete hand-eye coordination loop.

        1. Set target (from visual attention)
        2. Hand physical movement (PID + muscle tension + anxiety tremor)
        3. Compute hand-eye coordination error
        4. Reach target → dopamine release → RL learning

        Args:
            target_x, target_y: Target coordinates (from eye attention position)
            motor_signal:       Motor cortex electrical signal (affects force gain)
            max_steps:          Maximum simulation steps

        Returns:
            Complete hand motor report + dopamine feedback
        """
        self._state = "reaching"

        # ★ Closed-loop: record hand target (for life_loop error calculation)
        self._current_hand_target = np.array([target_x, target_y])

        # Hand physical movement (PID + muscle physics + anxiety tremor)
        reach_result = self.hand.reach(
            target_x=target_x,
            target_y=target_y,
            motor_signal=motor_signal,
            ram_temperature=self.vitals.ram_temperature,
            max_steps=max_steps,
        )

        # Hand-eye coordination error
        coordination = self.hand.compute_hand_eye_error(target_x, target_y)

        # Proprioception → feedback to brain
        proprioception = self.hand.get_proprioception()

        # ★ Temporal calibration: proprioceptive signal + error signal fed into calibrator
        self.calibrator.receive(proprioception)
        if coordination["error_signal"] is not None:
            self.calibrator.receive(coordination["error_signal"])
        binding_frame = self.calibrator.bind()

        # ★ Dopamine loop: reach target → reward → RL learning
        dopamine = coordination["dopamine"]
        if dopamine > 0:
            rl_result = self.rl.update(
                state="hand_reaching",
                action=f"reach_{int(target_x)}_{int(target_y)}",
                reward=dopamine,
                next_state="hand_reached",
            )
            dopamine = rl_result  # Use TD error as the actual dopamine signal
            self._log_event("reach_success", {
                "target": (target_x, target_y),
                "final_pos": reach_result["final_pos"],
                "steps": reach_result["steps"],
                "dopamine": round(dopamine, 4),
            })
        else:
            self._log_event("reach_miss", {
                "target": (target_x, target_y),
                "final_pos": reach_result["final_pos"],
                "final_error": reach_result["final_error"],
                "tremor": reach_result["tremor_intensity"],
            })

        self._state = "idle"
        return {
            "reach": reach_result,
            "coordination": {
                "error_x": coordination["error_x"],
                "error_y": coordination["error_y"],
                "error_magnitude": coordination["error_magnitude"],
                "reached": coordination["reached"],
            },
            "proprioception": {
                "amplitude": proprioception.amplitude,
                "frequency": proprioception.frequency,
                "source": proprioception.source,
            },
            "temporal_binding": {
                "frame_id": binding_frame.frame_id if binding_frame else None,
                "bound_modalities": binding_frame.modalities if binding_frame else [],
                "bindings": len(binding_frame.bindings) if binding_frame else 0,
                "binding_scores": {
                    f"{a}-{b}": s
                    for (a, b), s in binding_frame.binding_scores.items()
                } if binding_frame else {},
            },
            "dopamine": round(dopamine, 4),
            "vitals": {
                "ram_temperature": self.vitals.ram_temperature,
                "pain_level": self.vitals.pain_level,
            },
        }

    # ------------------------------------------------------------------
    def get_hand_state(self) -> Dict[str, Any]:
        """Get hand state (for API / dashboard use)."""
        return self.hand.get_muscle_state(ram_temperature=self.vitals.ram_temperature)

    # ------------------------------------------------------------------

    def emergency_reset(self):
        """Emergency: force reset vitals to healthy state."""
        self.vitals.reset()
        self._log_event("emergency_reset", {"reason": "Manual emergency reset"})

    # ------------------------------------------------------------------
    def inject_pain(self, intensity: float = 0.5):
        """
        Manually inject pain (for experiments).

        Directly raises RAM temperature, simulating external trauma.
        """
        self.vitals.ram_temperature = float(np.clip(
            self.vitals.ram_temperature + intensity * 0.4, 0.0, 1.0
        ))
        self._log_event("inject_pain", {"intensity": intensity})

    # ------------------------------------------------------------------
    def _log_event(self, event_type: str, details: Dict[str, Any]):
        self._event_log.append(
            {
                "type": event_type,
                "cycle": self._cycle_count,
                "timestamp": time.time(),
                "details": details,
            }
        )
        if len(self._event_log) > 1000:
            self._event_log = self._event_log[-500:]

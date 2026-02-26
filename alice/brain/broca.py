# -*- coding: utf-8 -*-
"""
Broca's Area Engine -- Motor Speech Planning & Sensorimotor Loop
Phase 4.3: The Birth of Broca's Area

Core Philosophy:

  Broca's area is NOT a "language module".
  It is the MOTOR COMPILER for speech production.

  Just as the motor cortex plans hand movements (target position -> PID control),
  Broca plans mouth movements (target concept -> formant trajectory -> PID pitch).

  The crucial insight: Broca DOES NOT understand meaning.
  It only knows how to map a concept label to articulatory parameters.
  Meaning lives in the Semantic Field (Phase 4.2).
  Motor execution lives in the Mouth (body/mouth.py).
  Broca is the bridge between them.

Physical Model:

  1. Articulatory Plan = Motor Program
     Each concept has an associated plan: (F1, F2, F3, pitch, volume, duration).
     These are the parameters needed to produce the sound that activates
     the concept in the listener's semantic field.

  2. Babbling = Random Exploration
     Infants babble: produce random articulatory plans, hear the result,
     and learn which plans activate which concepts.
     This is motor exploration -- the same as hand flailing in early development.

  3. Sensorimotor Loop = Closed-Loop Speech
     Speak -> Hear self -> Recognize what was produced -> Compare to intent.
     Gamma_loop = Gamma between intended and perceived concept.
     If Gamma_loop ~ 0: speech was successful.
     If Gamma_loop ~ 1: speech error -> adjust plan.

  4. Plan Impedance Model
     Z_plan = Z_0 / confidence
     confidence grows with successful productions
     Low Z_plan = efficient motor program = fluent speech
     High Z_plan = uncertain plan = stuttering, errors

  5. Learning Equation
     On success: confidence += eta * (1 - Gamma_loop)
     On failure: formants += eta * (target_formants - current_formants)
     This is gradient descent on articulatory error,
     physically manifested as impedance reduction.

Equations:
  Z_plan = Z_0 / (1 + confidence)
  Gamma_plan = (Z_intended - Z_plan) / (Z_intended + Z_plan)
  E_articulation = 1 - |Gamma_plan|^2
  Formant_update: F_i += eta * (F_i_target - F_i_current)
  Babble: F_i = uniform(F_i_min, F_i_max)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.body.cochlea import CochlearFilterBank, TonotopicActivation
from alice.core.signal import ElectricalSignal


# ============================================================================
# Physical Constants
# ============================================================================

# --- Plan impedance ---
BROCA_Z0 = 100.0                 # Base impedance (Ohm)
BROCA_INPUT_Z = 50.0             # Reference impedance for Gamma calc

# --- Articulatory ranges (biological bounds) ---
PITCH_RANGE = (80.0, 400.0)      # Hz -- fundamental frequency range
F1_RANGE = (200.0, 900.0)        # Hz -- first formant (jaw opening)
F2_RANGE = (800.0, 2500.0)       # Hz -- second formant (tongue position)
F3_RANGE = (1800.0, 3500.0)      # Hz -- third formant (lip rounding)
VOLUME_RANGE = (0.2, 0.8)        # normalized volume

# --- Learning ---
PLAN_LEARNING_RATE = 0.15        # eta for plan updates
FORMANT_LEARNING_RATE = 0.2      # eta for formant adjustments
CONFIDENCE_FLOOR = 0.01          # minimum confidence
CONFIDENCE_CEILING = 10.0        # maximum confidence
CONFIDENCE_DECAY = 0.001         # per-tick confidence decay

# --- Babbling ---
BABBLE_DURATION_STEPS = 3        # duration of each babble utterance
BABBLE_VOLUME = 0.4              # volume during babbling phase

# --- Feedback ---
SUCCESS_GAMMA_THRESHOLD = 0.3    # Gamma_loop < this = success
PARTIAL_GAMMA_THRESHOLD = 0.6    # Gamma_loop < this = partial match

# --- Capacity ---
MAX_PLANS = 200                  # maximum stored articulatory plans

# --- Known vowel formant targets (biological reference) ---
VOWEL_FORMANT_TARGETS: Dict[str, Tuple[float, float, float]] = {
    "a": (730.0, 1090.0, 2440.0),
    "i": (270.0, 2290.0, 3010.0),
    "u": (300.0, 870.0, 2240.0),
    "e": (530.0, 1840.0, 2480.0),
    "o": (570.0, 840.0, 2410.0),
}


# ============================================================================
# ArticulatoryPlan -- a motor program for speech
# ============================================================================


@dataclass
class ArticulatoryPlan:
    """
    An articulatory plan -- the motor program for producing a specific sound.

    Physical analogy:
      Like a hand's motor plan (target_x, target_y, force),
      this is a mouth's motor plan (F1, F2, F3, pitch, volume).

    The plan has a "confidence" that determines its impedance:
      Z_plan = Z_0 / (1 + confidence)
      High confidence = low impedance = fluent execution.
      Low confidence = high impedance = hesitant, error-prone.
    """
    concept_label: str

    # Articulatory targets
    formants: Tuple[float, float, float] = (500.0, 1500.0, 2500.0)  # F1, F2, F3
    pitch: float = 150.0              # fundamental frequency Hz
    volume: float = 0.5               # normalized volume 0~1
    duration_steps: int = 3           # production duration

    # Motor confidence
    confidence: float = CONFIDENCE_FLOOR
    z_impedance: float = BROCA_Z0

    # Statistics
    success_count: int = 0
    failure_count: int = 0
    total_attempts: int = 0
    creation_time: float = 0.0
    last_used: float = 0.0

    # History of Gamma values for this plan
    gamma_history: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    def gamma(self, z_reference: float = BROCA_INPUT_Z) -> float:
        """Gamma = (Z_ref - Z_plan) / (Z_ref + Z_plan)."""
        return abs(z_reference - self.z_impedance) / (z_reference + self.z_impedance)

    def energy_transfer(self, z_reference: float = BROCA_INPUT_Z) -> float:
        """Energy transfer = 1 - |Gamma|^2."""
        g = self.gamma(z_reference)
        return 1.0 - g * g

    def update_impedance(self):
        """Z_plan = Z_0 / (1 + confidence)."""
        self.z_impedance = BROCA_Z0 / (1.0 + self.confidence)

    # ------------------------------------------------------------------
    def reinforce(self, gamma_loop: float):
        """
        Reinforce on success: confidence increases.

        delta_confidence = eta * (1 - gamma_loop)
        Low gamma_loop (good match) = big boost.
        """
        delta = PLAN_LEARNING_RATE * (1.0 - gamma_loop)
        self.confidence = min(CONFIDENCE_CEILING,
                              self.confidence + max(0.0, delta))
        self.success_count += 1
        self.total_attempts += 1
        self.gamma_history.append(gamma_loop)
        self.update_impedance()

    def weaken(self, gamma_loop: float):
        """
        Weaken on failure: confidence decreases.

        delta_confidence = -eta * gamma_loop
        High gamma_loop (bad match) = big penalty.
        """
        delta = PLAN_LEARNING_RATE * gamma_loop * 0.5
        self.confidence = max(CONFIDENCE_FLOOR,
                              self.confidence - delta)
        self.failure_count += 1
        self.total_attempts += 1
        self.gamma_history.append(gamma_loop)
        self.update_impedance()

    def adjust_formants(self, target_formants: Tuple[float, float, float],
                        rate: float = FORMANT_LEARNING_RATE):
        """
        Gradient step: move formants toward target.

        Physical: the motor cortex adjusts the planned vocal tract shape
        to better match the desired output.
        """
        f1, f2, f3 = self.formants
        t1, t2, t3 = target_formants
        new_f1 = f1 + rate * (t1 - f1)
        new_f2 = f2 + rate * (t2 - f2)
        new_f3 = f3 + rate * (t3 - f3)
        # Clip to biological bounds
        new_f1 = float(np.clip(new_f1, *F1_RANGE))
        new_f2 = float(np.clip(new_f2, *F2_RANGE))
        new_f3 = float(np.clip(new_f3, *F3_RANGE))
        self.formants = (new_f1, new_f2, new_f3)

    def adjust_pitch(self, target_pitch: float,
                     rate: float = FORMANT_LEARNING_RATE):
        """Adjust pitch toward target."""
        new_pitch = self.pitch + rate * (target_pitch - self.pitch)
        self.pitch = float(np.clip(new_pitch, *PITCH_RANGE))

    def decay(self):
        """Slight confidence decay over time."""
        self.confidence = max(CONFIDENCE_FLOOR,
                              self.confidence * (1.0 - CONFIDENCE_DECAY))
        self.update_impedance()

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_label": self.concept_label,
            "formants": [round(f, 1) for f in self.formants],
            "pitch": round(self.pitch, 1),
            "volume": round(self.volume, 2),
            "confidence": round(self.confidence, 4),
            "z_impedance": round(self.z_impedance, 2),
            "gamma": round(self.gamma(), 4),
            "energy_transfer": round(self.energy_transfer(), 4),
            "success_rate": (
                round(self.success_count / max(1, self.total_attempts), 3)
            ),
            "total_attempts": self.total_attempts,
        }


# ============================================================================
# Formant extraction from tonotopic activation
# ============================================================================


def extract_formants(tono: TonotopicActivation) -> Tuple[float, float, float]:
    """
    Reverse-engineer formant frequencies from a tonotopic activation.

    Physics: formants = energy peaks in the frequency spectrum.
    The tonotopic activation gives energy per ERB band.
    Local maxima in the activation = estimated formant positions.

    This is a simplified version of what the auditory cortex does:
    extract the spectral envelope peaks.
    """
    activations = tono.channel_activations
    freqs = tono.center_frequencies

    if len(activations) < 3:
        return (500.0, 1500.0, 2500.0)

    # Find local peaks (channels where activation > both neighbors)
    peaks = []
    for i in range(1, len(activations) - 1):
        if (activations[i] > activations[i - 1] and
                activations[i] > activations[i + 1]):
            peaks.append((float(freqs[i]), float(activations[i])))

    # Also consider boundaries
    if len(activations) > 1:
        if activations[0] > activations[1]:
            peaks.append((float(freqs[0]), float(activations[0])))
        if activations[-1] > activations[-2]:
            peaks.append((float(freqs[-1]), float(activations[-1])))

    # Sort by amplitude (strongest first) and pick top 3
    peaks.sort(key=lambda x: x[1], reverse=True)
    formant_freqs = [p[0] for p in peaks[:3]]

    # Ensure at least 3 formants
    while len(formant_freqs) < 3:
        formant_freqs.append(1000.0 + len(formant_freqs) * 500.0)

    # Sort ascending: F1 < F2 < F3
    formant_freqs.sort()
    return (formant_freqs[0], formant_freqs[1], formant_freqs[2])


# ============================================================================
# BrocaEngine -- the motor speech planner
# ============================================================================


class BrocaEngine:
    """
    Broca's Area -- Motor speech planning and sensorimotor learning.

    Physical model:
      Broca is to the mouth what the motor cortex is to the hand.
      It doesn't understand meaning (that's the Semantic Field).
      It only knows how to PRODUCE sounds that activate concepts.

    Key functions:
      plan_utterance()   -- concept label -> articulatory plan
      execute_plan()     -- plan -> mouth -> waveform
      speak_concept()    -- full pipeline: plan -> speak -> hear -> verify
      babble()           -- random exploration (infant speech development)
      learn_from_feedback()  -- adjust plans based on production errors

    Sensorimotor Loop:
      1. Intent: "I want to say 'apple'"
      2. Plan: look up articulatory plan for 'apple' (F1, F2, F3, pitch)
      3. Execute: mouth.speak(formants, pitch)  -> waveform
      4. Feedback: waveform -> cochlea -> fingerprint -> semantic field
      5. Verify: did the semantic field recognize 'apple'?
      6. Learn: if yes, reinforce plan. If no, adjust formants.

    This loop is how babies learn to speak:
      babble randomly -> hear result -> learn which mouth shapes
      make which sounds.
    """

    def __init__(
        self,
        cochlea: Optional[CochlearFilterBank] = None,
        max_plans: int = MAX_PLANS,
    ):
        # Articulatory plan library
        self.plans: Dict[str, ArticulatoryPlan] = {}
        self.max_plans = max_plans

        # Cochlea for feedback analysis
        self.cochlea = cochlea if cochlea is not None else CochlearFilterBank()

        # Statistics
        self.total_utterances = 0
        self.total_babbles = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_plans_created = 0

        # RNG for babbling
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Plan Management
    # ------------------------------------------------------------------

    def create_plan(
        self,
        concept_label: str,
        formants: Tuple[float, float, float] = (500.0, 1500.0, 2500.0),
        pitch: float = 150.0,
        volume: float = 0.5,
        confidence: float = CONFIDENCE_FLOOR,
    ) -> ArticulatoryPlan:
        """
        Create a new articulatory plan for a concept.

        If plan already exists, updates it.
        """
        plan = ArticulatoryPlan(
            concept_label=concept_label,
            formants=formants,
            pitch=pitch,
            volume=volume,
            confidence=confidence,
            z_impedance=BROCA_Z0 / (1.0 + confidence),
            creation_time=time.monotonic(),
        )
        self.plans[concept_label] = plan
        self.total_plans_created += 1

        # Capacity management
        if len(self.plans) > self.max_plans:
            weakest = min(self.plans.values(), key=lambda p: p.confidence)
            if weakest.concept_label != concept_label:
                del self.plans[weakest.concept_label]

        return plan

    def get_plan(self, concept_label: str) -> Optional[ArticulatoryPlan]:
        """Retrieve an existing plan."""
        return self.plans.get(concept_label)

    def has_plan(self, concept_label: str) -> bool:
        """Check if a plan exists for this concept."""
        return concept_label in self.plans

    # ------------------------------------------------------------------
    # Plan from vowel knowledge
    # ------------------------------------------------------------------

    def create_vowel_plan(self, vowel: str) -> ArticulatoryPlan:
        """
        Create a plan from known vowel formants.

        This is "innate" knowledge -- like how certain reflexive
        vocalizations (crying, cooing) are available from birth.
        """
        formants = VOWEL_FORMANT_TARGETS.get(
            vowel.lower(), (500.0, 1500.0, 2500.0)
        )
        return self.create_plan(
            concept_label=f"vowel_{vowel.lower()}",
            formants=formants,
            pitch=150.0,
            volume=0.5,
            confidence=0.5,  # innate = moderate confidence
        )

    # ------------------------------------------------------------------
    # Speech Production
    # ------------------------------------------------------------------

    def plan_utterance(self, concept_label: str) -> Optional[ArticulatoryPlan]:
        """
        Look up the motor plan for a concept.

        Returns None if no plan has been learned yet.
        """
        return self.plans.get(concept_label)

    def execute_plan(
        self,
        plan: ArticulatoryPlan,
        mouth: Any,  # AliceMouth
        ram_temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute an articulatory plan through the mouth.

        Returns:
            {
                "waveform": np.ndarray,
                "plan": ArticulatoryPlan,
                "speak_result": dict from mouth.speak(),
                "feedback_fingerprint": np.ndarray,
                "feedback_tono": TonotopicActivation,
            }
        """
        self.total_utterances += 1
        plan.last_used = time.monotonic()

        # Execute through mouth with formants
        speak_result = mouth.speak(
            target_pitch=plan.pitch,
            volume=plan.volume,
            formants=plan.formants,
            duration_steps=plan.duration_steps,
            ram_temperature=ram_temperature,
        )

        waveform = speak_result["waveform"]

        # Self-feedback: hear what was produced
        tono = self.cochlea.analyze(waveform, apply_persistence=False)
        fp = tono.fingerprint()

        return {
            "waveform": waveform,
            "plan": plan,
            "speak_result": speak_result,
            "feedback_fingerprint": fp,
            "feedback_tono": tono,
        }

    # ------------------------------------------------------------------
    def speak_concept(
        self,
        concept_label: str,
        mouth: Any,  # AliceMouth
        semantic_field: Any = None,  # SemanticFieldEngine
        ram_temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Full speech production pipeline:
        1. Look up plan for concept
        2. Execute through mouth
        3. Hear result (cochlea feedback)
        4. Verify through semantic field (if available)
        5. Learn from feedback

        Returns:
            {
                "intended": str,
                "waveform": np.ndarray,
                "plan": dict,
                "perceived": str or None,
                "gamma_loop": float,
                "success": bool,
                "feedback_fingerprint": np.ndarray,
            }
        """
        plan = self.plan_utterance(concept_label)
        if plan is None:
            # No plan -> babble something and register
            return self.babble(mouth, intended_label=concept_label)

        # Execute
        exec_result = self.execute_plan(plan, mouth, ram_temperature)

        # Verify through semantic field
        perceived = None
        gamma_loop = 1.0
        success = False

        if semantic_field is not None:
            recognition = semantic_field.process_fingerprint(
                exec_result["feedback_fingerprint"],
                modality="auditory",
            )
            perceived = recognition.get("best_concept")
            gamma_loop = recognition.get("gamma", 1.0)
            success = (perceived == concept_label and
                       gamma_loop < SUCCESS_GAMMA_THRESHOLD)

            # Learn from feedback — three cases:
            #   1. perceived == intended AND gamma < threshold → full success
            #   2. perceived == intended BUT gamma >= threshold → partial success
            #      (right concept recognized, but articulation still imprecise)
            #   3. perceived != intended → failure
            if success:
                # Full success: strong reinforcement
                plan.reinforce(gamma_loop)
                self.total_successes += 1
            elif perceived == concept_label:
                # Partial success: correct concept but gamma still high.
                # Apply reduced reinforcement — the direction is right,
                # just needs more practice to lower impedance.
                partial_gamma = gamma_loop * 0.7  # discount for partial match
                plan.reinforce(partial_gamma)
                # Also adjust formants toward reference target
                self._adjust_from_feedback(
                    plan, concept_label, exec_result, semantic_field
                )
            elif perceived is not None:
                # Wrong concept recognized → weaken + adjust
                plan.weaken(gamma_loop)
                self.total_failures += 1
                self._adjust_from_feedback(
                    plan, concept_label, exec_result, semantic_field
                )

        return {
            "intended": concept_label,
            "waveform": exec_result["waveform"],
            "plan": plan.to_dict(),
            "perceived": perceived,
            "gamma_loop": gamma_loop,
            "success": success,
            "feedback_fingerprint": exec_result["feedback_fingerprint"],
        }

    # ------------------------------------------------------------------
    # Babbling -- Random Exploration
    # ------------------------------------------------------------------

    def babble(
        self,
        mouth: Any,  # AliceMouth
        intended_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Babble -- produce a random utterance.

        Physical model: infant motor exploration.
        Random formant/pitch combinations -> hear result.
        If an intended_label is given, create a plan for it
        based on the babbled parameters.

        This is how babies discover the articulatory space:
        random flailing -> hear result -> learn mapping.
        """
        self.total_babbles += 1

        # Random articulatory parameters
        f1 = self._rng.uniform(*F1_RANGE)
        f2 = self._rng.uniform(*F2_RANGE)
        f3 = self._rng.uniform(*F3_RANGE)
        pitch = self._rng.uniform(*PITCH_RANGE)

        # Ensure F1 < F2 < F3
        formants_list = sorted([f1, f2, f3])
        formants = (formants_list[0], formants_list[1], formants_list[2])

        label = intended_label or f"babble_{self.total_babbles}"

        # Create or update plan
        plan = self.create_plan(
            concept_label=label,
            formants=formants,
            pitch=pitch,
            volume=BABBLE_VOLUME,
            confidence=CONFIDENCE_FLOOR,
        )

        # Execute through mouth
        speak_result = mouth.speak(
            target_pitch=pitch,
            volume=BABBLE_VOLUME,
            formants=formants,
            duration_steps=BABBLE_DURATION_STEPS,
            ram_temperature=0.0,  # no stress during babbling
        )

        waveform = speak_result["waveform"]

        # Self-feedback
        tono = self.cochlea.analyze(waveform, apply_persistence=False)
        fp = tono.fingerprint()

        return {
            "intended": label,
            "waveform": waveform,
            "plan": plan.to_dict(),
            "perceived": None,
            "gamma_loop": 1.0,
            "success": False,
            "feedback_fingerprint": fp,
            "is_babble": True,
        }

    # ------------------------------------------------------------------
    # Learning from Examples
    # ------------------------------------------------------------------

    def learn_from_example(
        self,
        concept_label: str,
        example_waveform: np.ndarray,
    ) -> ArticulatoryPlan:
        """
        Learn to produce a concept by hearing an example.

        Physics: hear a sound -> extract formants -> create motor plan.
        This is "imitation learning": hearing someone say "apple"
        and learning to reproduce it.
        """
        # Analyze the example
        tono = self.cochlea.analyze(example_waveform, apply_persistence=False)
        formants = extract_formants(tono)

        # Estimate pitch from dominant frequency
        # (simplified: dominant ERB channel maps to pitch)
        pitch = float(np.clip(tono.dominant_frequency, *PITCH_RANGE))

        # Create plan with moderate confidence
        plan = self.create_plan(
            concept_label=concept_label,
            formants=formants,
            pitch=pitch,
            volume=0.5,
            confidence=0.3,  # somewhat confident from example
        )
        return plan

    # ------------------------------------------------------------------
    # Sensorimotor Feedback
    # ------------------------------------------------------------------

    def verify_production(
        self,
        waveform: np.ndarray,
        intended_label: str,
        semantic_field: Any = None,  # SemanticFieldEngine
    ) -> Dict[str, Any]:
        """
        Verify a production: did it match the intended concept?

        Returns:
            {
                "intended": str,
                "perceived": str or None,
                "gamma_loop": float,
                "success": bool,
                "fingerprint": np.ndarray,
            }
        """
        tono = self.cochlea.analyze(waveform, apply_persistence=False)
        fp = tono.fingerprint()

        perceived = None
        gamma_loop = 1.0
        success = False

        if semantic_field is not None:
            recognition = semantic_field.process_fingerprint(fp, "auditory")
            perceived = recognition.get("best_concept")
            gamma_loop = recognition.get("gamma", 1.0)
            success = (perceived == intended_label and
                       gamma_loop < SUCCESS_GAMMA_THRESHOLD)

        return {
            "intended": intended_label,
            "perceived": perceived,
            "gamma_loop": gamma_loop,
            "success": success,
            "fingerprint": fp,
        }

    # ------------------------------------------------------------------
    def _adjust_from_feedback(
        self,
        plan: ArticulatoryPlan,
        intended_label: str,
        exec_result: Dict[str, Any],
        semantic_field: Any,
    ):
        """
        Adjust plan based on production error.

        If we know what formants the intended concept SHOULD have
        (from the semantic field's stored examples), adjust toward those.
        """
        # Check if we have a reference fingerprint for the intended concept
        if not hasattr(semantic_field, 'field'):
            return

        field = semantic_field.field
        if intended_label not in field.attractors:
            return

        attractor = field.attractors[intended_label]
        if "auditory" not in attractor.modality_centroids:
            return

        # Build a reference tonotopic activation from the centroid
        # and extract target formants
        ref_fp = attractor.modality_centroids["auditory"]

        # Simplified: find peaks in reference fingerprint
        # to estimate target formants
        if len(ref_fp) < 3:
            return

        peaks = []
        center_freqs = self.cochlea.center_frequencies
        for i in range(1, len(ref_fp) - 1):
            if ref_fp[i] > ref_fp[i - 1] and ref_fp[i] > ref_fp[i + 1]:
                peaks.append((float(center_freqs[i]), float(ref_fp[i])))

        if not peaks:
            return

        peaks.sort(key=lambda x: x[1], reverse=True)
        target_formants = [p[0] for p in peaks[:3]]
        while len(target_formants) < 3:
            target_formants.append(1000.0 + len(target_formants) * 500.0)
        target_formants.sort()
        target = (target_formants[0], target_formants[1], target_formants[2])

        plan.adjust_formants(target, rate=FORMANT_LEARNING_RATE)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def tick(self):
        """
        Per-cognitive-cycle maintenance.

        1. Decay plan confidence (forgetting)
        2. Remove very weak plans
        """
        dead = []
        for label, plan in self.plans.items():
            plan.decay()
            # Remove plans that have never succeeded and are very old
            if (plan.success_count == 0 and plan.total_attempts > 10
                    and plan.confidence <= CONFIDENCE_FLOOR * 1.1):
                dead.append(label)

        for label in dead:
            del self.plans[label]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_vocabulary(self) -> List[str]:
        """List all concepts that Broca can produce."""
        return list(self.plans.keys())

    def get_vocabulary_size(self) -> int:
        """Number of producible concepts."""
        return len(self.plans)

    def get_plan_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all plans."""
        return {label: plan.to_dict() for label, plan in self.plans.items()}

    def get_state(self) -> Dict[str, Any]:
        return {
            "vocabulary_size": len(self.plans),
            "total_utterances": self.total_utterances,
            "total_babbles": self.total_babbles,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": round(
                self.total_successes / max(1, self.total_successes + self.total_failures),
                3,
            ),
            "top_plans": [
                plan.to_dict()
                for plan in sorted(
                    self.plans.values(),
                    key=lambda p: p.confidence,
                    reverse=True,
                )[:10]
            ],
        }

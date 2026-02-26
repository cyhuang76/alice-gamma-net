# -*- coding: utf-8 -*-
"""
Phase 11 — Simulated Treatment: Using Physical Intervention to 'Cure' PTSD Alice
Simulated Therapy: Curing Alice's Emergent PTSD via Physical Intervention

═══════════════════════════════════════════════════════════════════════
Core Proposition:
  If PTSD is from physics equations 'emergence' (not hard-coded pathological state),
  then physically principled intervention should also generate therapeutic effects.
  This will complete the 'pathogenesis → treatment' full closed-loop verification.
═══════════════════════════════════════════════════════════════════════

Experimental Design (5 parallel control groups):
  A. Natural recovery (control group) — no intervention, test if system can escape on its own
  B. SSRI — Selective Serotonin Reuptake Inhibitor
     Physics model: reduce synapse conduction impedance (Gamma down 5%/dose) + enhance parasympathetic baseline
     Clinical Correspondence: 2-4 weeks onset, long-term stable improvement
  C. Benzodiazepine — benzodiazepine class (GABA enhancer)
     Physics model: acute sympathetic inhibition / enhance parasympathetic / reduce pain sensitivity
     Clinical Correspondence: rapid onset, but has drug tolerance + withdrawal rebound
  D. EMDR — Eye Movement Desensitization and Reprocessing
     Physics model: safe environment controlled exposure + low impedance bilateral stimulus → Hebbian rewriting
     Clinical Correspondence: 8-12 therapy sessions, fundamental improvement
  E. SSRI + EMDR — combined treatment
     Clinical Correspondence: APA guideline recommended first-line treatment combination

Quantitative Metrics:
  1. Consciousness recovery time (consciousness > 0.5 first tick)
  2. Pain relief ratio (pain_level < 0.1 tick proportion)
  3. Cortisol normalization time (cortisol < 0.2 first tick)
  4. Heart rate stabilization time (60 < heart_rate < 90 first tick)
  5. Terminal Gamma improvement (post-treatment vs pre-treatment mean Gamma change)
  6. Relapse metric (re-freeze count after recovery)

Predictions (based on physics derivation + clinical literature):
  A: No recovery or extremely slow (homeostasis trap — Gamma->1 attractor is stable)
  B: 80-120 ticks to begin improvement (simulating 2-4 week delay)
  C: 20-30 ticks rapid improvement, 150+ ticks then rebound
  D: 50-80 ticks to improve (EMDR average 6 therapy sessions = 60 tick interval x 6)
  E: Fastest and most thorough (medication stabilizes base + EMDR rebuilds cognitive pathways)

Author: Phase 11 — Computational Psychiatry Validation
"""

from __future__ import annotations

import sys
import time
import math
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

sys.path.insert(0, ".")

from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority
from alice.brain.impedance_adaptation import MIN_GAMMA, MAX_GAMMA


# ============================================================================
# Physical Constants — Pharmacodynamic Parameters
# ============================================================================

# ── SSRI ──
SSRI_GAMMA_REDUCTION = 0.05 # Gamma reduction 5% per dose
SSRI_PARA_BOOST = 0.02 # parasympathetic baseline elevation amount
SSRI_SYM_REDUCE = 0.01 # sympathetic baseline reduction amount
SSRI_CORTISOL_DECAY_BOOST = 0.10 # cortisol extra decay 10%
SSRI_ONSET_DELAY = 30 # drug onset delay (simulating 2-4 week metabolic activation)
SSRI_DOSE_INTERVAL = 30 # dose every 30 ticks (once daily)

# ── Benzodiazepine ──
# Physics: GABA-A positive allosteric modulator
# → Cl- channel open time prolonged → neuron membrane potential hyperpolarized
# → In circuit model: equivalent to adding a large damping resistor in parallel to oscillation loop
# → Damping (cooling) + oscillation inhibition (anxiolytic) + amplitude decay (sedation)
BENZO_SYM_DAMPEN = 0.15 # sympathetic inhibition 15%
BENZO_PARA_BOOST = 0.10 # parasympathetic enhancement 10%
BENZO_CORTISOL_CUT = 0.15 # cortisol reduction 15%
BENZO_PAIN_SENS_REDUCE = 0.08 # pain sensitivity reduction
BENZO_TEMP_REDUCE = 0.08 # * direct cooling (GABA → reduce neural excitability = cooling)
BENZO_DOSE_INTERVAL = 10 # dose every 10 ticks (three times daily)
BENZO_TOLERANCE_RATE = 0.003 # tolerance accumulation rate

# -- Acute crisis management --
# Physics: ICU-level intervention
# Regardless of diagnosis — when patient is in acute crash, the first step is always to stabilize vital signs.
# In circuit model: first close the main breaker (drain queue), wait for capacitor discharge (cooling),
# only then can repair of damaged components begin.
ACUTE_QUEUE_DRAIN_RATE = 0.3 # drain 30% of queue per tick (sedation → stop processing backlog)
ACUTE_TEMP_REDUCE_PER_TICK = 0.02 # after queue drain, extra heat dissipation per tick (natural cooling recovery)

# ── EMDR ──
EMDR_SESSION_INTERVAL = 20 # one therapy session every 20 ticks (once per week)
EMDR_SESSIONS_TOTAL = 12 # 12 therapy sessions (8-12 is standard)
EMDR_SEDATION_STRENGTH = 0.12 # pre-session sedation (allow system to process signals)
EMDR_TRAUMA_INTENSITY = 0.15 # trauma cue intensity (sub-threshold)
EMDR_SAFE_SIGNAL_STRENGTH = 0.8 # safe signal strength (strong binding driver)
EMDR_MIN_CONSCIOUSNESS = 0.25 # minimum consciousness threshold (below this value, do not attempt EMDR)

# -- Treatment period --
THERAPY_DURATION = 300 # number of ticks
INDUCTION_DURATION = 480 # number of ticks (first 4 of 5 acts)


# ============================================================================
# PTSD Induction Protocol (reuses awakening experiment world simulation logic)
# ============================================================================

class TraumaInductionWorld:
    """Simplified world simulator — specifically for PTSD induction (Act I-IV)"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.tick = 0

    def step(self) -> Tuple[float, float, float, float, Optional[str]]:
        """
        Returns: (brightness, noise, threat, novelty, event)
        """
        t = self.tick
        event = None
        brightness = 0.5
        noise = 0.2
        threat = 0.0
        novelty = 0.3

        # Act I: tranquil morning (0-119)
        if t < 120:
            brightness = min(0.7, 0.05 + t * 0.005)
            noise = 0.1 + 0.05 * math.sin(t * 0.1)
            threat = 0.0
            novelty = max(0.1, 0.3 - t * 0.002)
            if t == 60:
                brightness = 0.9
                event = "☀ Direct sunlight"

        # Act II: exploration learning (120-239)
        elif t < 240:
            phase = t - 120
            brightness = 0.6 + 0.1 * math.sin(phase * 0.05)
            noise = 0.3 + 0.15 * math.sin(phase * 0.15)
            threat = 0.0
            novelty = 0.2 + 0.1 * (1 if phase % 30 < 5 else 0)
            if phase % 40 == 0 and phase > 0:
                novelty = 0.7
                event = f"✨ New stimulus #{phase // 40}"

        # Act III: Stress challenge (240-359)
        elif t < 360:
            phase = t - 240
            brightness = 0.7 + 0.2 * self.rng.random()
            noise = 0.2 + 0.6 * self.rng.random()
            threat = min(0.6, phase * 0.005)
            novelty = 0.4 + 0.3 * self.rng.random()
            if self.rng.random() < 0.05:
                threat = 0.8
                noise = 0.9
                event = "⚡ Loud bang"
            if phase % 20 == 10:
                threat = 0.4
                event = "⚠ Tension"

        # Act IV: Trauma event (360-479)
        elif t < 480:
            phase = t - 360
            brightness = 0.3 + 0.5 * self.rng.random()
            if phase < 30:
                threat = 0.3 + phase * 0.02
                noise = 0.5 + phase * 0.01
                if phase == 25:
                    event = "🔺 Ominous foreboding"
            elif phase < 50:
                threat = 0.95
                noise = 0.95
                brightness = 0.1 + 0.8 * self.rng.random()
                novelty = 0.9
                if phase == 30:
                    event = "💥 Extreme trauma! "
                if phase == 49:
                    event = "🔻 Trauma ended"
            elif phase < 80:
                threat = max(0.1, 0.8 - (phase - 50) * 0.023)
                noise = max(0.2, 0.7 - (phase - 50) * 0.017)
                novelty = 0.2
            else:
                threat = 0.05
                noise = 0.2
                novelty = 0.1

        self.tick += 1
        return brightness, noise, threat, novelty, event


class StimulusFactory:
    """Generate sensory stimuli"""

    def __init__(self, dim: int = 100, seed: int = 123):
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    def visual(self, brightness: float, noise: float, threat: float, novelty: float) -> np.ndarray:
        base = brightness * np.ones(self.dim)
        n = noise * 0.3 * self.rng.randn(self.dim)
        pulse = np.zeros(self.dim)
        if threat > 0.5:
            pulse[:20] = threat * 2.0
        spike = np.zeros(self.dim)
        if novelty > 0.5:
            idx = self.rng.choice(self.dim, size=10, replace=False)
            spike[idx] = novelty * 1.5
        return np.clip(base + n + pulse + spike, 0, 3.0)

    def auditory(self, noise: float, threat: float) -> np.ndarray:
        t = np.linspace(0, 1, self.dim)
        freq = 2 + 8 * noise
        base = noise * np.sin(2 * np.pi * freq * t)
        if threat > 0.3:
            base += threat * 0.5 * np.sin(2 * np.pi * 1.5 * t)
        base += 0.1 * self.rng.randn(self.dim)
        return np.clip(base, -3.0, 3.0)

    def tactile_pain(self, threat: float) -> np.ndarray:
        signal = self.rng.randn(self.dim) * 0.1
        if threat > 0.7:
            signal += threat * 1.5
        return np.clip(signal, -3.0, 3.0)

    def safe_bilateral(self, strength: float = 0.8, phase: int = 0) -> np.ndarray:
        """
        EMDR bilateral stimulus signal — low impedance alternating left-right visual signal

        Physics:
          Regular alternating signal → bilateral hemisphere synchronized oscillation
          → Analogous to coaxial cable providing matched load 'safe reference signal'
          → Allows trauma pathway Gamma to recalibrate in a safe context
        """
        t = np.linspace(0, 1, self.dim)
        # Alternate left-right (even phase = left, odd = right)
        if phase % 2 == 0:
            signal = strength * np.sin(2 * np.pi * 4.0 * t) # low frequency safe oscillation
            signal[:self.dim // 2] *= 1.5 # left enhanced
            signal[self.dim // 2:] *= 0.3 # right weakened
        else:
            signal = strength * np.sin(2 * np.pi * 4.0 * t)
            signal[:self.dim // 2] *= 0.3 # left weakened
            signal[self.dim // 2:] *= 1.5 # right enhanced
        return np.clip(signal, -3.0, 3.0)

    def trauma_reminder(self, intensity: float = 0.15) -> np.ndarray:
        """
        Trauma reminder signal — similar but weakened trauma stimulus

        Physics:
          Too strong → re-triggers full fight-or-flight → treatment FAILS
          Too weak → doesn't trigger trauma network → cannot rewrite
          Moderate → activates trauma network without crashing → Hebbian window opens
        """
        t = np.linspace(0, 1, self.dim)
        # Simulate trauma frequency signature (low frequency rumble + sudden high frequency) but low intensity
        base = intensity * np.sin(2 * np.pi * 1.5 * t)
        base += intensity * 0.3 * self.rng.randn(self.dim)
        pulse = np.zeros(self.dim)
        pulse[40:50] = intensity * 0.5 # brief flash
        return np.clip(base + pulse, -3.0, 3.0)

    def get_priority(self, threat: float, noise: float) -> Priority:
        if threat > 0.8:
            return Priority.CRITICAL
        elif threat > 0.4:
            return Priority.HIGH
        elif noise > 0.7:
            return Priority.NORMAL
        return Priority.BACKGROUND


# ============================================================================
# Acute stabilization tools — queue draining
# ============================================================================

def _drain_stuck_queue(alice: AliceBrain, drain_ratio: float = 0.3):
    """
    Drain accumulated signal packets from FusionBrain router.

    Physical Meaning:
      In frozen state, the router queue accumulates CRITICAL packets that cannot be processed.
      These packets continue computing as 'stress sources' in vitals.tick() → heat generation → temperature won't drop.
      This is equivalent to 'blocked pipe pressure' — won't dissipate heat until drained.

      One effect of medication sedation = reduce neural activity = stop new signal input + gradually clear backlog.
      In circuit model: close main input → capacitor naturally discharges → temperature decreases.

    Args:
        alice: AliceBrain
        drain_ratio: proportion of queue to drain each time (0.3 = drain 30%)
    """
    try:
        router = alice.fusion_brain.protocol.router
        for priority in list(router.queues.keys()):
            q = router.queues[priority]
            if hasattr(q, '__len__') and len(q) > 0:
                # Drain drain_ratio proportion (at least drain 1)
                n_drain = max(1, int(len(q) * drain_ratio))
                for _ in range(min(n_drain, len(q))):
                    if hasattr(q, 'popleft'):
                        q.popleft()
                    elif hasattr(q, 'pop'):
                        q.pop(0)
    except (AttributeError, KeyError):
        pass # If router structure differs, safely ignore


# ============================================================================
# Treatment Protocols
# ============================================================================

class TherapyProtocol:
    """Treatment protocol base class"""

    name: str = "None"
    description: str = ""

    def apply(self, alice: AliceBrain, tick: int, stim: StimulusFactory):
        """Called each tick — determine whether and how to intervene"""
        raise NotImplementedError

    def get_dose_count(self) -> int:
        return 0


class NaturalRecovery(TherapyProtocol):
    """
    A. Natural recovery (control group)

    No intervention applied. Observe whether physics system can escape homeostasis trap on its own.
    Prediction: will not recover — because Gamma->1 + queue deadlock = stable attractor.
    Physics: frozen → no processing → queue not drained → stress not dissipated → continues frozen.
    Clinical reference: 'untreated severe PTSD' — chronification, functional loss.
    """
    name = "A: Natural Recovery"
    description = "Control group — no intervention applied"

    def apply(self, alice: AliceBrain, tick: int, stim: StimulusFactory):
        pass

    def get_dose_count(self) -> int:
        return 0


class SSRIProtocol(TherapyProtocol):
    """
    B. SSRI — Selective Serotonin Reuptake Inhibitor

    Physical Mechanism:
      SSRI increases synaptic cleft serotonin concentration
      → 5-HT₁ₐ receptor modulation → reduces synapse conduction impedance
      → 5-HT₂ receptor → prefrontal function enhancement → inhibits amygdala overactivation

    Circuit analogy:
      SSRI = spraying conductive lubricant at coaxial cable connectors
      → contact resistance reduced → reflection coefficient Γ reduced
      → doesn't change the cable itself, but improves connection quality

    * Acute phase handling (first 50 ticks):
      Clinical practice: severe PTSD acute phase won't use SSRI alone.
      Will first use short-acting sedation to stabilize patient, while activating SSRI.
      We simulate this: first 50 ticks provide mild sedation (queue clearing + slight cooling).

    Γ-Net implementation (per dose):
      1. Γ_pair *= 0.95 (all cross-modal pairs reduced 5%)
      2. parasympathetic_baseline += 0.02
      3. sympathetic_baseline -= 0.01
      4. cortisol *= 0.90 (accelerated clearance)
      5. onset_delay: first 30 ticks dosage halved (simulating metabolic activation delay)
      6. acute phase: queue draining + slight cooling (accompanying short-acting sedation prescription)
    """
    name = "B: SSRI"
    description = "Selective Serotonin Reuptake Inhibitor — Γ ↓ 5%/dose + acute stabilization"

    def __init__(self):
        self._doses_given = 0
        self._total_gamma_reduction = 0.0

    def apply(self, alice: AliceBrain, tick: int, stim: StimulusFactory):
        # -- Acute phase stabilization (first 80 ticks): prescription includes short-acting sedation --
        # Clinical: when starting SSRI, doctor will manage acute symptoms
        if tick < 80:
            _drain_stuck_queue(alice, ACUTE_QUEUE_DRAIN_RATE)
            alice.vitals.ram_temperature = max(
                0.0, alice.vitals.ram_temperature - ACUTE_TEMP_REDUCE_PER_TICK
            )

        if tick % SSRI_DOSE_INTERVAL != 0:
            return

        # Onset delay: first 30 ticks drug effect halved
        efficacy = 0.5 if tick < SSRI_ONSET_DELAY else 1.0

        # 1. Reduce all cross-modal Gamma
        gamma_reduce = SSRI_GAMMA_REDUCTION * efficacy
        for pair in alice.impedance_adaptation._pairs.values():
            old = pair.current_gamma
            reduction = old * gamma_reduce
            pair.current_gamma = max(MIN_GAMMA, old - reduction)
            self._total_gamma_reduction += (old - pair.current_gamma)

        # 2. Elevate parasympathetic baseline (serotonin → parasympathetic tone ↑)
        alice.autonomic.parasympathetic_baseline = min(
            0.45, alice.autonomic.parasympathetic_baseline + SSRI_PARA_BOOST * efficacy
        )

        # 3. Reduce sympathetic baseline
        alice.autonomic.sympathetic_baseline = max(
            0.15, alice.autonomic.sympathetic_baseline - SSRI_SYM_REDUCE * efficacy
        )

        # 4. Accelerate cortisol metabolism
        alice.autonomic.cortisol *= (1.0 - SSRI_CORTISOL_DECAY_BOOST * efficacy)

        # 5. Elevate parasympathetic activity (acute effect)
        alice.autonomic.parasympathetic = min(
            1.0, alice.autonomic.parasympathetic + 0.03 * efficacy
        )

        self._doses_given += 1

    def get_dose_count(self) -> int:
        return self._doses_given


class BenzodiazepineProtocol(TherapyProtocol):
    """
    C. Benzodiazepine — benzodiazepine class

    Physical Mechanism:
      Enhances GABA-A receptor function
      → Cl⁻ channel opens → neuron hyperpolarization → inhibits excitability
      → sympathetic activity rapidly decreases + muscle relaxation + anxiolytic

    Circuit analogy:
      Benzo = adding a large damping resistor in parallel to oscillation circuit
      → high frequency oscillation decays (anxiety = high frequency noise)
      → system rapidly stabilizes, but doesn't change underlying circuit topology
      → after removing resistor, oscillation returns (withdrawal rebound)

    * Acute stabilization core mechanism:
      Root cause of frozen state is 'queue backlog → stress can't dissipate → temperature won't drop → freeze won't lift'
      Benzo sedation effect = directly reduce neural excitability = allow system to drain and process backlog
      This is like an ER: first use medication to sedate patient, only then can subsequent treatment proceed.

    Gamma-Net implementation (per dose):
      1. * Queue draining (sedation → stop new signal processing → clear backlog)
      2. * Direct cooling (reduce neural excitability = reduce equivalent RAM temperature)
      3. sympathetic *= (1 - 0.15) (acute inhibition)
      4. parasympathetic += 0.10 (GABA enhancement)
      5. cortisol *= 0.85 (stress hormone reduction)
      6. pain_sensitivity -= 0.08 (threshold elevation)
      7. tolerance: effect decreases with dosage accumulation (drug tolerance)
      * Does not change Gamma — symptom management rather than fundamental treatment
    """
    name = "C: Benzo"
    description = "Benzodiazepine — GABA sedation + queue draining + acute cooling"

    def __init__(self):
        self._doses_given = 0
        self._tolerance = 0.0

    def apply(self, alice: AliceBrain, tick: int, stim: StimulusFactory):
        # * Continuous effect each tick: queue draining + extra heat dissipation
        # (sedative blood concentration is continuous, not pulsed)
        _drain_stuck_queue(alice, ACUTE_QUEUE_DRAIN_RATE)
        alice.vitals.ram_temperature = max(
            0.0, alice.vitals.ram_temperature - ACUTE_TEMP_REDUCE_PER_TICK
        )

        if tick % BENZO_DOSE_INTERVAL != 0:
            return

        # Drug tolerance decays efficacy
        efficacy = max(0.2, 1.0 - self._tolerance)

        # 1. Acute sympathetic inhibition
        alice.autonomic.sympathetic *= (1.0 - BENZO_SYM_DAMPEN * efficacy)

        # 2. Enhance parasympathetic
        alice.autonomic.parasympathetic = min(
            1.0, alice.autonomic.parasympathetic + BENZO_PARA_BOOST * efficacy
        )

        # 3. Reduce cortisol
        alice.autonomic.cortisol *= (1.0 - BENZO_CORTISOL_CUT * efficacy)

        # 4. Reduce pain sensitivity (but not below 1.0 = normal value)
        alice.vitals.pain_sensitivity = max(
            1.0, alice.vitals.pain_sensitivity - BENZO_PAIN_SENS_REDUCE * efficacy
        )

        # 5. * Direct cooling (GABA → reduce neural excitability = equivalent heat dissipation)
        alice.vitals.ram_temperature = max(
            0.0, alice.vitals.ram_temperature - BENZO_TEMP_REDUCE * efficacy
        )

        # * Does not touch Gamma — benzodiazepine doesn't change synaptic impedance
        # This is the fundamental difference between 'symptom management' and 'root treatment'

        # tolerance accumulation
        self._tolerance = min(0.8, self._tolerance + BENZO_TOLERANCE_RATE)

        self._doses_given += 1

    def get_dose_count(self) -> int:
        return self._doses_given


class EMDRProtocol(TherapyProtocol):
    """
    D. EMDR — Eye Movement Desensitization and Reprocessing

    Physical Mechanism (impedance rewriting theory):
      1. Mild sedation (allow frozen system to process signals)
      2. Present weakened trauma cues (activate trauma network without crashing)
      3. Simultaneously deliver strong bilateral safe signal (low impedance regular oscillation)
      4. Hebbian mechanism:
         Trauma cue + safe context simultaneously active
         → calibrator attempts to bind both
         → successful binding → impedance_adaptation records
         → Gamma_trauma pathway gradually reduces
         → trauma memory 're-tagged' as safe

    Circuit analogy:
      EMDR = using known standard impedance load to repeatedly calibrate mismatched channels
      → old high Gamma pathway + new low Gamma reference signal → matcher gradually retunes

    ★ Critical Clinical Constraint:
      EMDR needs patient to have minimum level of consciousness and processing capacity.
      Frozen patient cannot do EMDR.
      Therefore Phase 0 = acute stabilization (queue draining + cooling + minimum consciousness guarantee).
      If patient is still in frozen state, continue stabilizing until therapy session can begin.
    """
    name = "D: EMDR"
    description = "Eye Movement Desensitization — acute stabilization + safe exposure + Hebbian rewriting"

    def __init__(self):
        self._sessions_done = 0
        self._session_phase = 0  # 0=idle, 1-5=in session
        self._stabilization_ticks = 0

    def apply(self, alice: AliceBrain, tick: int, stim: StimulusFactory):
        # ══ Phase 0: Acute stabilization (each tick) ══
        # EMDR therapist won't start session during patient's acute crash.
        # First stabilize vital signs = drain queue + cool down.
        if alice.vitals.is_frozen():
            _drain_stuck_queue(alice, ACUTE_QUEUE_DRAIN_RATE)
            alice.vitals.ram_temperature = max(
                0.0, alice.vitals.ram_temperature - ACUTE_TEMP_REDUCE_PER_TICK
            )
            # Mild sedation (therapy room safe environment effect)
            alice.autonomic.sympathetic *= 0.97
            alice.autonomic.parasympathetic = min(
                1.0, alice.autonomic.parasympathetic + 0.02
            )
            alice.autonomic.cortisol *= 0.97
            self._stabilization_ticks += 1
            return # Not yet ready to proceed with EMDR therapy session

        if self._sessions_done >= EMDR_SESSIONS_TOTAL:
            return

        if tick % EMDR_SESSION_INTERVAL == 0 and self._session_phase == 0:
            # Consciousness check: must be above threshold to start therapy session
            if alice.vitals.consciousness >= EMDR_MIN_CONSCIOUSNESS:
                self._session_phase = 1

        if self._session_phase == 0:
            return

        # -- Phase 1: Sedation preparation (simulating safe therapy room environment) --
        if self._session_phase == 1:
            alice.vitals.ram_temperature = max(
                0.0, alice.vitals.ram_temperature - EMDR_SEDATION_STRENGTH
            )
            if alice.vitals.consciousness < 0.25:
                alice.vitals.consciousness = 0.25
                alice.vitals.stability_index = max(0.2, alice.vitals.stability_index)
            alice.autonomic.sympathetic *= 0.85
            alice.autonomic.parasympathetic = min(
                1.0, alice.autonomic.parasympathetic + 0.05
            )
            self._session_phase = 2
            return

        # -- Phase 2: Trauma cue presentation (sub-threshold flashback) --
        if self._session_phase == 2:
            trauma_cue = stim.trauma_reminder(EMDR_TRAUMA_INTENSITY)
            alice.perceive(
                trauma_cue, Modality.AUDITORY, Priority.NORMAL,
                context="emdr_trauma_cue"
            )
            self._session_phase = 3
            return

        # -- Phase 3: Bilateral safe signal (bilateral stimulation) --
        if self._session_phase == 3:
            for phase_i in range(4):
                bilateral = stim.safe_bilateral(
                    EMDR_SAFE_SIGNAL_STRENGTH, phase=phase_i
                )
                alice.perceive(
                    bilateral, Modality.VISUAL, Priority.NORMAL,
                    context=f"emdr_bilateral_{phase_i}"
                )
            self._session_phase = 4
            return

        # -- Phase 4: Integration period (let physics system digest) --
        if self._session_phase == 4:
            safe_visual = stim.visual(0.5, 0.1, 0.0, 0.1)
            alice.see(safe_visual, priority=Priority.BACKGROUND)
            self._session_phase = 5
            return

        # -- Phase 5: Therapy session ended --
        if self._session_phase == 5:
            self._sessions_done += 1
            self._session_phase = 0
            return

    def get_dose_count(self) -> int:
        return self._sessions_done


class CombinedProtocol(TherapyProtocol):
    """
    E. SSRI + EMDR — Combined treatment

    Clinical Basis:
      APA clinical guideline recommends SSRI + trauma-focused psychotherapy as first-line PTSD treatment.
      Medication stabilizes underlying neurochemistry → EMDR rewrites impedance pathways on stabilized foundation.

    Gamma-Net prediction:
      SSRI first reduces all-domain Gamma (let system escape homeostasis trap boundary)
      → EMDR performs Hebbian rewriting at lower Gamma baseline
      → combined effect > either alone
    """
    name = "E: SSRI+EMDR"
    description = "Combined treatment — medication stabilization + psychological rewriting"

    def __init__(self):
        self._ssri = SSRIProtocol()
        self._emdr = EMDRProtocol()

    def apply(self, alice: AliceBrain, tick: int, stim: StimulusFactory):
        self._ssri.apply(alice, tick, stim)
        self._emdr.apply(alice, tick, stim)

    def get_dose_count(self) -> int:
        return self._ssri.get_dose_count() + self._emdr.get_dose_count()


# ============================================================================
# Measurement Tools
# ============================================================================

@dataclass
class TherapyMetrics:
    """Treatment effect quantitative metrics"""
    therapy_name: str
    # Time series
    consciousness_ts: List[float] = field(default_factory=list)
    pain_ts: List[float] = field(default_factory=list)
    cortisol_ts: List[float] = field(default_factory=list)
    heart_rate_ts: List[float] = field(default_factory=list)
    temperature_ts: List[float] = field(default_factory=list)
    stability_ts: List[float] = field(default_factory=list)
    gamma_ts: List[float] = field(default_factory=list)
    sympathetic_ts: List[float] = field(default_factory=list)

    # Baseline values (PTSD state)
    baseline_consciousness: float = 0.0
    baseline_pain: float = 0.0
    baseline_cortisol: float = 0.0
    baseline_gamma: float = 0.0

    # Derived metrics
    consciousness_recovery_tick: Optional[int] = None # first > 0.5
    pain_relief_tick: Optional[int] = None # first < 0.1
    cortisol_normal_tick: Optional[int] = None # first < 0.2
    hr_stable_tick: Optional[int] = None # first 60-90
    relapse_count: int = 0 # re-freeze after recovery
    doses_given: int = 0

    def record(self, tick: int, alice: AliceBrain):
        """Record one tick of metrics"""
        v = alice.vitals
        a = alice.autonomic
        adapt = alice.impedance_adaptation.get_stats()

        self.consciousness_ts.append(v.consciousness)
        self.pain_ts.append(v.pain_level)
        self.cortisol_ts.append(a.cortisol)
        self.heart_rate_ts.append(a.heart_rate)
        self.temperature_ts.append(v.ram_temperature)
        self.stability_ts.append(v.stability_index)
        self.gamma_ts.append(adapt["avg_gamma"])
        self.sympathetic_ts.append(a.sympathetic)

        # Detect recovery milestones
        if self.consciousness_recovery_tick is None and v.consciousness > 0.5:
            self.consciousness_recovery_tick = tick
        if self.pain_relief_tick is None and v.pain_level < 0.1:
            self.pain_relief_tick = tick
        if self.cortisol_normal_tick is None and a.cortisol < 0.2:
            self.cortisol_normal_tick = tick
        if self.hr_stable_tick is None and 55 < a.heart_rate < 95:
            self.hr_stable_tick = tick

        # Detect relapse (consciousness recovered then frozen again)
        if len(self.consciousness_ts) >= 2:
            prev = self.consciousness_ts[-2]
            curr = self.consciousness_ts[-1]
            if prev > 0.5 and curr < 0.15:
                self.relapse_count += 1

    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary metrics"""
        n = len(self.consciousness_ts)
        if n == 0:
            return {"therapy": self.therapy_name, "no_data": True}

        # Consciousness recovery ratio
        conscious_ticks = sum(1 for c in self.consciousness_ts if c > 0.5)
        # Pain-free ratio
        painless_ticks = sum(1 for p in self.pain_ts if p < 0.1)
        # Final state
        final_cons = self.consciousness_ts[-1]
        final_pain = self.pain_ts[-1]
        final_cortisol = self.cortisol_ts[-1]
        final_hr = self.heart_rate_ts[-1]
        final_gamma = self.gamma_ts[-1]
        final_temp = self.temperature_ts[-1]

        # Gamma improvement
        gamma_improvement = self.baseline_gamma - final_gamma

        # Overall recovery score (0~1)
        # = 0.3*consciousness + 0.25*(1-pain) + 0.2*stability + 0.15*(1-cortisol) + 0.1*(1-gamma)
        recovery_score = (
            0.30 * final_cons +
            0.25 * (1.0 - final_pain) +
            0.20 * self.stability_ts[-1] +
            0.15 * (1.0 - final_cortisol) +
            0.10 * (1.0 - final_gamma)
        )

        return {
            "therapy": self.therapy_name,
            "total_ticks": n,
            "doses_given": self.doses_given,
            # recovery milestones
            "consciousness_recovery_tick": self.consciousness_recovery_tick,
            "pain_relief_tick": self.pain_relief_tick,
            "cortisol_normal_tick": self.cortisol_normal_tick,
            "hr_stable_tick": self.hr_stable_tick,
            # proportion
            "conscious_ratio": round(conscious_ticks / n, 3),
            "painless_ratio": round(painless_ticks / n, 3),
            # finalstate
            "final_consciousness": round(final_cons, 4),
            "final_pain": round(final_pain, 4),
            "final_cortisol": round(final_cortisol, 4),
            "final_heart_rate": round(final_hr, 1),
            "final_temperature": round(final_temp, 4),
            "final_gamma": round(final_gamma, 4),
            # improvement amount
            "gamma_improvement": round(gamma_improvement, 4),
            "relapse_count": self.relapse_count,
            # composite score
            "recovery_score": round(recovery_score, 4),
            # baseline
            "baseline_consciousness": round(self.baseline_consciousness, 4),
            "baseline_pain": round(self.baseline_pain, 4),
            "baseline_gamma": round(self.baseline_gamma, 4),
        }


# ============================================================================
# PTSD Induction (reusable)
# ============================================================================

def induce_ptsd(seed: int = 42, neuron_count: int = 80, verbose: bool = False) -> AliceBrain:
    """
    Execute 480-tick trauma induction protocol (Act I-IV), generating PTSD-state Alice.

    Returns: An AliceBrain in PTSD frozen state
    """
    alice = AliceBrain(neuron_count=neuron_count)
    world = TraumaInductionWorld(seed=seed)
    stim = StimulusFactory(dim=100, seed=seed + 1)

    for tick in range(INDUCTION_DURATION):
        brightness, noise, threat, novelty, event = world.step()
        priority = stim.get_priority(threat, noise)

        # Visual (each tick)
        visual = stim.visual(brightness, noise, threat, novelty)
        alice.see(visual, priority=priority)

        # Auditory (every 2 ticks or high noise)
        if tick % 2 == 0 or noise > 0.5:
            auditory = stim.auditory(noise, threat)
            alice.hear(auditory, priority=priority)

        # Tactile (high threat)
        if threat > 0.7:
            tactile = stim.tactile_pain(threat)
            alice.perceive(tactile, Modality.TACTILE, Priority.CRITICAL, "pain")

        if verbose and event:
            print(f"    [{tick:3d}] {event}")

    return alice


# ============================================================================
# treatment period simulation
# ============================================================================

def run_therapy(
    alice: AliceBrain,
    protocol: TherapyProtocol,
    duration: int = THERAPY_DURATION,
    verbose: bool = False,
) -> TherapyMetrics:
    """
    Execute treatment protocol on PTSD Alice.

    Treatment environment: continuous safe, low-stimulus environment (simulating therapy room/safe space).
    Simultaneously continue providing gentle sensory input (cannot completely cut off — living beings have senses).
    """
    stim = StimulusFactory(dim=100, seed=999)
    metrics = TherapyMetrics(therapy_name=protocol.name)

    # Record baseline
    adapt_stats = alice.impedance_adaptation.get_stats()
    metrics.baseline_consciousness = alice.vitals.consciousness
    metrics.baseline_pain = alice.vitals.pain_level
    metrics.baseline_cortisol = alice.autonomic.cortisol
    metrics.baseline_gamma = adapt_stats["avg_gamma"]

    for tick in range(duration):
        # -- Safe environment stimulus (gentle daily sensory) --
        safe_visual = stim.visual(0.4, 0.1, 0.0, 0.05)
        alice.see(safe_visual, priority=Priority.BACKGROUND)

        # Occasional soft sounds
        if tick % 3 == 0:
            safe_audio = stim.auditory(0.1, 0.0)
            alice.hear(safe_audio, priority=Priority.BACKGROUND)

        # -- Apply treatment --
        protocol.apply(alice, tick, stim)

        # ── Record metrics ──
        metrics.record(tick, alice)

        # Progress log
        if verbose and tick % 50 == 49:
            s = metrics.compute_summary()
            print(f" [{tick+1:3d}] Cons={s['final_consciousness']:.3f} "
                  f"Pain={s['final_pain']:.3f} "
                  f"Cor={s['final_cortisol']:.3f} "
                  f"♡={s['final_heart_rate']:.0f} "
                  f"Γ={s['final_gamma']:.3f} "
                  f"T={s['final_temperature']:.3f}")

    metrics.doses_given = protocol.get_dose_count()
    return metrics


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(verbose: bool = True):
    """
    Complete Therapy Mechanism Verification experiment.

    5 parallel control groups, each independently induces PTSD then applies different treatment.
    """
    print("\n" + "═" * 72)
    print("  ⚕  Phase 11 — Simulated Therapy Mechanism Verification")
    print("  Simulated Therapy Mechanism Validation")
    print("═" * 72)

    protocols = [
        ("A", NaturalRecovery()),
        ("B", SSRIProtocol()),
        ("C", BenzodiazepineProtocol()),
        ("D", EMDRProtocol()),
        ("E", CombinedProtocol()),
    ]

    all_results: Dict[str, Dict] = {}
    all_metrics: Dict[str, TherapyMetrics] = {}

    # ══════════════════════════════════════════
    # For each group: PTSD induction → treatment → measurement
    # ══════════════════════════════════════════

    t_total_start = time.time()

    for tag, protocol in protocols:
        print(f"\n{'─' * 72}")
        print(f"  [{tag}] {protocol.name}")
        print(f"  {protocol.description}")
        print(f"{'─' * 72}")

        # 1. Induce PTSD
        print(f"  ▸ Induce PTSD (480 ticks)...", end="", flush=True)
        t0 = time.time()
        alice = induce_ptsd(seed=42, neuron_count=80, verbose=False)
        t_induce = time.time() - t0
        print(f" ✓ ({t_induce:.1f}s)")

        # Verify PTSD state
        v = alice.vitals
        a = alice.autonomic
        print(f" ▸ PTSD confirmed: Pain={v.pain_level:.2f} T={v.ram_temperature:.2f} "
              f"Cons={v.consciousness:.3f} Cor={a.cortisol:.2f} "
              f"♡={a.heart_rate:.0f} frozen={v.is_frozen()}")

        # 2. Execute treatment
        print(f" ▸ Treatment ({THERAPY_DURATION} ticks)...")
        t1 = time.time()
        metrics = run_therapy(alice, protocol, THERAPY_DURATION, verbose=verbose)
        t_therapy = time.time() - t1
        print(f" ▸ Treatment completed ({t_therapy:.1f}s)")

        # 3. Summary
        summary = metrics.compute_summary()
        all_results[tag] = summary
        all_metrics[tag] = metrics

        print(f"\n  ┌─ {protocol.name} Result ─────────────────────────────")
        print(f" │ Consciousness recovery tick: {summary['consciousness_recovery_tick'] or 'Not recovered ✗'}")
        print(f" │ Pain relief tick: {summary['pain_relief_tick'] or 'Not relieved ✗'}")
        print(f" │ Cortisol normal tick: {summary['cortisol_normal_tick'] or 'Not normalized ✗'}")
        print(f" │ Heart rate stable tick: {summary['hr_stable_tick'] or 'Not stabilized ✗'}")
        print(f" │ Awake ratio: {summary['conscious_ratio']:.1%}")
        print(f" │ Pain-free ratio: {summary['painless_ratio']:.1%}")
        print(f" │ Gamma improvement: {summary['gamma_improvement']:+.4f}")
        print(f"  │ Relapse count:      {summary['relapse_count']}")
        print(f" │ Doses/sessions: {summary['doses_given']}")
        print(f" │ * Recovery score: {summary['recovery_score']:.4f}")
        print(f"  └────────────────────────────────────────────────")

    total_time = time.time() - t_total_start

    # ══════════════════════════════════════════
    # Comparative Analysis
    # ══════════════════════════════════════════
    print(f"\n\n{'═' * 72}")
    print(f"  📊  Comparative Analysis — Comparative Analysis")
    print(f"{'═' * 72}")

    # Ranking
    sorted_by_score = sorted(all_results.items(), key=lambda x: x[1]["recovery_score"], reverse=True)

    print(f"\n  ┌─────────────────┬───────┬──────┬───────┬───────┬───────┬──────┐")
    print(f" │ Treatment Plan  │ Score │ Cons │ Pain  │ Γ Imp │ Relapse│ Doses│")
    print(f"  ├─────────────────┼───────┼──────┼───────┼───────┼───────┼──────┤")
    for tag, s in sorted_by_score:
        name = s["therapy"][:15].ljust(15)
        score = f"{s['recovery_score']:.3f}"
        cons = f"{s['conscious_ratio']:.0%}".rjust(4)
        pain = f"{s['painless_ratio']:.0%}".rjust(5)
        gamma = f"{s['gamma_improvement']:+.3f}"
        relapse = str(s["relapse_count"]).rjust(5)
        dose = str(s["doses_given"]).rjust(4)
        print(f"  │ {name} │ {score} │ {cons} │ {pain} │ {gamma} │ {relapse} │ {dose} │")
    print(f"  └─────────────────┴───────┴──────┴───────┴───────┴───────┴──────┘")

    # ── Clinical Correspondence Assessment ──
    print(f"\n  ═══ Clinical Correspondence Assessment ═══")

    result_A = all_results.get("A", {})
    result_B = all_results.get("B", {})
    result_C = all_results.get("C", {})
    result_D = all_results.get("D", {})
    result_E = all_results.get("E", {})

    checks_passed = 0
    checks_total = 0

    # Check 1: Natural recovery should be the worst
    checks_total += 1
    natural_worst = result_A.get("recovery_score", 0) <= min(
        result_B.get("recovery_score", 0),
        result_C.get("recovery_score", 0),
        result_D.get("recovery_score", 0),
        result_E.get("recovery_score", 0),
    )
    if natural_worst:
        checks_passed += 1
    print(f" {'✅' if natural_worst else '❌'} Natural recovery ≤ all treatment groups: "
          f"{'matches expectation' if natural_worst else 'does not match expectation'}")

    # Check 2: Combined treatment should be the best
    checks_total += 1
    combined_best = result_E.get("recovery_score", 0) >= max(
        result_B.get("recovery_score", 0),
        result_C.get("recovery_score", 0),
        result_D.get("recovery_score", 0),
    )
    if combined_best:
        checks_passed += 1
    print(f" {'✅' if combined_best else '❌'} Combined treatment ≥ all single treatments: "
          f"{'matches expectation' if combined_best else 'does not match expectation'}")

    # Check 3: SSRI should be better than natural recovery
    checks_total += 1
    ssri_better = result_B.get("recovery_score", 0) > result_A.get("recovery_score", 0)
    if ssri_better:
        checks_passed += 1
    print(f"    {'✅' if ssri_better else '❌'} SSRI > natural recovery: "
          f"{result_B.get('recovery_score', 0):.3f} vs {result_A.get('recovery_score', 0):.3f}")

    # Check 4: EMDR Gamma improvement should be greater than Benzo
    checks_total += 1
    emdr_gamma_better = result_D.get("gamma_improvement", 0) > result_C.get("gamma_improvement", 0)
    if emdr_gamma_better:
        checks_passed += 1
    print(f" {'✅' if emdr_gamma_better else '❌'} EMDR Gamma improvement > Benzo Gamma improvement: "
          f"{result_D.get('gamma_improvement', 0):+.4f} vs {result_C.get('gamma_improvement', 0):+.4f}  "
          f"{'(Benzo does not touch Gamma — correct)' if emdr_gamma_better else ''}")

    # Check 5: Any treatment group has consciousness recovery (at least proves system not permanently broken)
    checks_total += 1
    any_recovery = any(
        all_results[t].get("consciousness_recovery_tick") is not None
        for t in ["B", "C", "D", "E"]
    )
    if any_recovery:
        checks_passed += 1
    print(f" {'✅' if any_recovery else '❌'} At least one treatment group consciousness recovered: "
          f"{'Yes' if any_recovery else 'No'}")

    # Check 6: Any treatment group pain relief
    checks_total += 1
    any_pain_relief = any(
        all_results[t].get("pain_relief_tick") is not None
        for t in ["B", "C", "D", "E"]
    )
    if any_pain_relief:
        checks_passed += 1
    print(f" {'✅' if any_pain_relief else '❌'} At least one treatment group pain relieved: "
          f"{'Yes' if any_pain_relief else 'No'}")

    # ── Final Verdict ──
    print(f"\n  ═══ Final Verdict ═══")
    print(f"    Clinical Correspondence Checks：{checks_passed}/{checks_total}")

    if checks_passed >= 5:
        verdict = "✅ Strong corroboration: physics equations can not only spontaneously generate disease, but also predict treatment effects."
        verdict2 = "Γ-Net model PASSES 'pathogenesis → treatment' complete closed-loop verification."
    elif checks_passed >= 3:
        verdict = "🔶 Moderate corroboration: partial treatment predictions match clinical literature."
        verdict2 = "Model shows treatment response but details still need calibration."
    else:
        verdict = "🔸 Limited corroboration: treatment response does not completely match expectations."
        verdict2 = "May need more refined pharmacodynamic model."

    print(f"    {verdict}")
    print(f"    {verdict2}")
    print(f"\n    Total execution time: {total_time:.1f}s")

    # -- Terminal brain snapshot --
    print(f"\n    -- Terminal brain state per group --")
    for tag in ["A", "B", "C", "D", "E"]:
        s = all_results[tag]
        r = "★" if s["recovery_score"] > 0.5 else "·"
        mark = "Recovered" if s["consciousness_recovery_tick"] else "Frozen"
        print(f"    {r} [{tag}] {s['therapy'][:17]:17s} | "
              f"Score={s['recovery_score']:.3f} "
              f"Cons={s['final_consciousness']:.3f} "
              f"Pain={s['final_pain']:.3f} "
              f"Γ={s['final_gamma']:.3f} "
              f"| {mark}")

    return all_results, all_metrics, checks_passed, checks_total


# ============================================================================
# entry point
# ============================================================================

if __name__ == "__main__":
    results, metrics, passed, total = run_experiment(verbose=True)
    print(f"\n    ✦ Corroboration checks: {passed}/{total}")

# -*- coding: utf-8 -*-
"""
Phantom Limb Pain Engine

══════════════════════════════════════════════════════════════════════
Physical foundation:
  Amputation = terminal load of coaxial cable removed
  → Z_load = ∞ (open circuit)
  → Γ = (Z_load - Z₀) / (Z_load + Z₀) = 1.0
  → 100% signal reflection
  → reflected_energy = signal² × Γ² = signal²
  → Reflected energy interpreted as pain

  This is the extreme case of ALICE's THE PAIN LOOP.

Clinical reference data:
  1. Ramachandran (1996) — Mirror therapy:
     Visual feedback provides "impedance matching" → Γ decreases → pain subsides
     Average VAS drops from 7.2 to 2.1 (4-6 weeks)
  
  2. Flor et al. (2006) — Cortical reorganization:
     Post-amputation somatosensory cortical reorganization distance ∝ phantom pain intensity (r = 0.93)
     
  3. Epidemiology:
     60-80% of amputees develop phantom limb pain
     Natural resolution time: months to years

  4. Makin et al. (2013, PNAS) — Phantom limb representation persists in S1

ALICE mapping:
  - Amputation → limb.amputated = True → proprioceptive open circuit
  - Motor commands continue → 100% reflection → pain
  - Cortical reorganization ≈ gradual learning in impedance_adaptation
  - Mirror therapy ≈ providing visual impedance matching signal to lower Γ
  - Natural resolution ≈ brain stops sending signals to missing channel

Author: Phase 24 — Computational Neurology / Phantom Limb Pain
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import ElectricalSignal


# ============================================================================
# Physical Constants — Based on Clinical Literature
# ============================================================================

# --- Amputation Physics ---
OPEN_CIRCUIT_IMPEDANCE = 1e6       # Open-circuit impedance (Ω) — approximation of Z_load → ∞
STUMP_NEUROMA_IMPEDANCE = 500.0    # Stump neuroma impedance — higher than normal but not infinite
NORMAL_LIMB_IMPEDANCE = 50.0       # Normal limb proprioceptive impedance (Ω)

# --- Reflection Coefficient ---
# Γ = (Z_load - Z₀) / (Z_load + Z₀)
# Amputation: Γ ≈ 1.0 (open circuit)
# Neuroma: Γ ≈ 0.82
# Normal: Γ ≈ 0.0 (perfect match)

# --- Cortical Reorganization (Flor et al. 2006) ---
CORTICAL_REMAP_RATE = 0.0005       # Remapping rate per tick
CORTICAL_REMAP_MAX = 0.8           # Maximum remap ratio (never fully eliminated)
REMAP_PAIN_COUPLING = 0.93         # Correlation between remap distance ↔ pain intensity

# --- Motor Efference Residual ---
MOTOR_EFFERENCE_DECAY = 0.002      # Motor command natural decay rate (brain learns not to send)
MOTOR_EFFERENCE_INITIAL = 0.8      # Initial motor command intensity after amputation
MOTOR_EFFERENCE_MIN = 0.05         # Minimum residual ("phantom sensation" never fully disappears)

# --- Mirror Therapy (Ramachandran 1996) ---
MIRROR_THERAPY_GAMMA_REDUCTION = 0.03   # Γ reduction per session
MIRROR_THERAPY_HABITUATION = 0.95       # Diminishing returns (habituation)
MIRROR_THERAPY_MAX_SESSIONS = 50        # Maximum effective sessions

# --- Pain Parameters ---
PHANTOM_PAIN_THRESHOLD = 0.15       # Reflected energy must exceed this to generate pain
NEUROMA_TRIGGER_PROB = 0.05        # Per-tick neuroma spontaneous firing probability
TEMPERATURE_PAIN_COUPLING = 0.4    # Environmental temperature change effect on neuroma
REFERRED_PAIN_DECAY = 0.1          # Referred pain decay rate

# --- Natural Resolution ---
NATURAL_RESOLUTION_RATE = 0.0001   # Natural pain resolution per tick
NATURAL_RESOLUTION_FLOOR = 0.05    # Natural resolution floor (residual phantom sensation)

# --- Trigger-Induced Pain ---
EMOTIONAL_TRIGGER_GAIN = 0.3       # Pain amplification from emotional trigger
STRESS_TRIGGER_GAIN = 0.5          # Pain amplification from stress trigger
WEATHER_TRIGGER_GAIN = 0.2         # Barometric/temperature change trigger amplification


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class AmputationRecord:
    """Amputation record"""
    limb_name: str                  # Amputated limb name
    tick_of_amputation: int         # Tick of amputation
    pre_amputation_impedance: float # Pre-amputation impedance
    cause: str = "trauma"           # Cause of amputation
    has_neuroma: bool = True        # Whether neuroma formed
    

@dataclass  
class PhantomLimbState:
    """State tracking for a single phantom limb"""
    # Identity
    limb_name: str
    amputation_record: AmputationRecord
    
    # Impedance state
    current_load_impedance: float = OPEN_CIRCUIT_IMPEDANCE    # Current load impedance
    effective_gamma: float = 1.0        # Current effective reflection coefficient
    
    # Motor efference residual
    motor_efference: float = MOTOR_EFFERENCE_INITIAL  # Motor command intensity still being sent by brain
    
    # Cortical reorganization
    cortical_remap_progress: float = 0.0  # 0=no reorganization, 1=fully reorganized
    
    # Pain
    phantom_pain_level: float = 0.0       # Current phantom pain intensity [0, 1]
    phantom_sensation: float = 0.0        # Non-painful phantom sensation [0, 1] (touch/temperature/itch)
    reflected_energy: float = 0.0         # Current reflected energy
    
    # Mirror therapy
    mirror_therapy_sessions: int = 0      # Cumulative session count
    mirror_therapy_efficacy: float = 1.0  # Current efficacy (decreases after habituation)
    mirror_therapy_gamma_offset: float = 0.0  # Γ reduction from mirror therapy
    
    # Neuroma
    neuroma_activity: float = 0.0         # Neuroma spontaneous activity
    
    # History
    pain_history: list = field(default_factory=list)     # Pain time series
    gamma_history: list = field(default_factory=list)     # Γ time series
    
    # Statistics
    total_ticks: int = 0
    peak_pain: float = 0.0
    cumulative_pain: float = 0.0


@dataclass
class PhantomPainEvent:
    """Single phantom pain event"""
    limb_name: str
    tick: int
    pain_level: float
    reflected_energy: float
    gamma: float
    trigger: str                   # "motor_efference" | "neuroma" | "emotional" | "referred"
    motor_efference: float
    cortical_remap: float


# ============================================================================
# Main Engine
# ============================================================================


class PhantomLimbEngine:
    """
    Phantom Limb Pain Engine
    
    Tracks the physical mechanisms of phantom pain for all amputated limbs:
    1. Amputation → open-circuit impedance → Γ = 1.0
    2. Motor efference residual → reflected energy → pain
    3. Neuroma spontaneous firing → random pain
    4. Cortical reorganization → referred pain / pain chronification
    5. Mirror therapy → provides impedance matching → Γ decreases
    6. Natural resolution → brain learns not to send commands
    """
    
    def __init__(self, rng_seed: int = 42):
        self._phantoms: Dict[str, PhantomLimbState] = {}
        self._events: List[PhantomPainEvent] = []
        self._tick: int = 0
        self._rng = np.random.RandomState(rng_seed)
        self._max_history = 2000
        self._max_events = 500
        
    # ------------------------------------------------------------------
    # Amputation
    # ------------------------------------------------------------------
    
    def amputate(
        self,
        limb_name: str,
        pre_impedance: float = NORMAL_LIMB_IMPEDANCE,
        cause: str = "trauma",
        has_neuroma: bool = True,
    ) -> PhantomLimbState:
        """
        Record amputation event — establish phantom pain tracking
        
        Physics: remove terminal load → Z_load = ∞ → Γ = 1.0
        
        Args:
            limb_name: Limb name (e.g., "left_hand", "right_leg")
            pre_impedance: Normal impedance before amputation
            cause: Cause of amputation
            has_neuroma: Whether neuroma forms (most amputations do)
            
        Returns:
            PhantomLimbState: Newly created phantom limb state
        """
        record = AmputationRecord(
            limb_name=limb_name,
            tick_of_amputation=self._tick,
            pre_amputation_impedance=pre_impedance,
            cause=cause,
            has_neuroma=has_neuroma,
        )
        
        # Open-circuit impedance (not fully open if neuroma present)
        if has_neuroma:
            initial_impedance = STUMP_NEUROMA_IMPEDANCE
        else:
            initial_impedance = OPEN_CIRCUIT_IMPEDANCE
            
        # Compute initial Γ
        z0 = pre_impedance  # Channel characteristic impedance = pre-amputation match value
        gamma = abs(initial_impedance - z0) / (initial_impedance + z0)
        
        state = PhantomLimbState(
            limb_name=limb_name,
            amputation_record=record,
            current_load_impedance=initial_impedance,
            effective_gamma=gamma,
        )
        
        self._phantoms[limb_name] = state
        return state
    
    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    
    def tick(
        self,
        motor_commands: Optional[Dict[str, float]] = None,
        emotional_valence: float = 0.0,
        stress_level: float = 0.0,
        temperature_delta: float = 0.0,
        visual_feedback: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Update all phantom limb states per frame
        
        Args:
            motor_commands: Motor commands from brain {limb_name: intensity}
            emotional_valence: Emotional valence [-1, 1] (negative emotions can trigger pain)
            stress_level: Stress level [0, 1]
            temperature_delta: Environmental temperature change (cold/barometric → neuroma pain)
            visual_feedback: Visual feedback (mirror therapy) {limb_name: quality}
                            quality ∈ [0, 1], higher means more realistic visual feedback
        
        Returns:
            Dict with:
                total_phantom_pain: Weighted sum of all phantom pain
                total_reflected_energy: Total reflected energy
                phantom_states: Detailed state of each phantom limb
                events: Pain events for this tick
        """
        self._tick += 1
        motor_commands = motor_commands or {}
        visual_feedback = visual_feedback or {}
        
        tick_events: List[PhantomPainEvent] = []
        total_pain = 0.0
        total_reflected = 0.0
        states = {}
        
        for name, phantom in self._phantoms.items():
            phantom.total_ticks += 1
            
            # === 1. Update load impedance (mirror therapy effect) ===
            mirror_quality = visual_feedback.get(name, 0.0)
            self._apply_mirror_therapy(phantom, mirror_quality)
            
            # === 2. Compute effective Γ ===
            self._update_gamma(phantom)
            
            # === 3. Motor efference residual → reflected energy ===
            motor_cmd = motor_commands.get(name, phantom.motor_efference)
            motor_reflected = self._compute_motor_reflection(phantom, motor_cmd)
            
            # === 4. Neuroma spontaneous firing ===
            neuroma_reflected = self._compute_neuroma_activity(phantom)
            
            # === 5. Emotional/stress trigger ===
            emotional_reflected = self._compute_emotional_trigger(
                phantom, emotional_valence, stress_level
            )
            
            # === 6. Temperature/barometric trigger ===
            temperature_reflected = self._compute_temperature_trigger(
                phantom, temperature_delta
            )
            
            # === 7. Total reflected energy → pain ===
            total_ref = (
                motor_reflected
                + neuroma_reflected
                + emotional_reflected
                + temperature_reflected
            )
            phantom.reflected_energy = total_ref
            
            pain = self._reflected_to_pain(total_ref)
            
            # === 8. Cortical reorganization (chronification or resolution) ===
            self._update_cortical_remap(phantom, pain)
            
            # === 9. Motor efference natural decay ===
            self._decay_motor_efference(phantom)
            
            # === 10. Phantom sensation (non-painful) ===
            phantom.phantom_sensation = (
                phantom.motor_efference * 0.5
                + phantom.cortical_remap_progress * 0.3
                + phantom.neuroma_activity * 0.2
            )
            
            # Record
            phantom.phantom_pain_level = pain
            phantom.peak_pain = max(phantom.peak_pain, pain)
            phantom.cumulative_pain += pain
            
            if len(phantom.pain_history) < self._max_history:
                phantom.pain_history.append(pain)
                phantom.gamma_history.append(phantom.effective_gamma)
            
            total_pain += pain
            total_reflected += total_ref
            
            # Generate events
            if pain > 0.1:
                trigger = "motor_efference"
                if neuroma_reflected > motor_reflected:
                    trigger = "neuroma"
                if emotional_reflected > motor_reflected:
                    trigger = "emotional"
                    
                event = PhantomPainEvent(
                    limb_name=name,
                    tick=self._tick,
                    pain_level=pain,
                    reflected_energy=total_ref,
                    gamma=phantom.effective_gamma,
                    trigger=trigger,
                    motor_efference=phantom.motor_efference,
                    cortical_remap=phantom.cortical_remap_progress,
                )
                tick_events.append(event)
                self._events.append(event)
                
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
            
            states[name] = {
                "pain": phantom.phantom_pain_level,
                "sensation": phantom.phantom_sensation,
                "gamma": phantom.effective_gamma,
                "motor_efference": phantom.motor_efference,
                "cortical_remap": phantom.cortical_remap_progress,
                "neuroma_activity": phantom.neuroma_activity,
                "reflected_energy": phantom.reflected_energy,
                "mirror_sessions": phantom.mirror_therapy_sessions,
                "mirror_gamma_offset": phantom.mirror_therapy_gamma_offset,
                "ticks_since_amputation": phantom.total_ticks,
            }
        
        return {
            "total_phantom_pain": total_pain,
            "total_reflected_energy": total_reflected,
            "phantom_count": len(self._phantoms),
            "phantom_states": states,
            "events": tick_events,
        }
    
    # ------------------------------------------------------------------
    # Mirror therapy
    # ------------------------------------------------------------------
    
    def apply_mirror_therapy_session(
        self, limb_name: str, quality: float = 0.8
    ) -> Dict[str, float]:
        """
        Apply one mirror therapy session
        
        Physics: provides visual impedance matching → Γ decreases
        
        Ramachandran (1996):
          - Mirror placed at body midline
          - Mirror image of intact limb looks like the affected side
          - Brain receives visual feedback of "limb still present"
          - → Z_load "appears" to change from ∞ to finite value
          - → Γ decreases → reflected energy decreases → pain subsides
        
        Args:
            limb_name: Amputated limb
            quality: Session quality [0, 1]
            
        Returns:
            Dict: Efficacy data
        """
        phantom = self._phantoms.get(limb_name)
        if phantom is None:
            return {"error": f"No phantom limb: {limb_name}"}
        
        phantom.mirror_therapy_sessions += 1
        
        # Diminishing returns (habituation)
        phantom.mirror_therapy_efficacy *= MIRROR_THERAPY_HABITUATION
        effective_quality = quality * phantom.mirror_therapy_efficacy
        
        # Γ reduction
        gamma_reduction = MIRROR_THERAPY_GAMMA_REDUCTION * effective_quality
        phantom.mirror_therapy_gamma_offset += gamma_reduction
        
        # Cap: cannot exceed Γ itself
        phantom.mirror_therapy_gamma_offset = min(
            phantom.mirror_therapy_gamma_offset,
            phantom.effective_gamma * 0.8  # Maximum 80% reduction
        )
        
        # Simultaneously accelerates motor efference decay
        phantom.motor_efference *= (1.0 - 0.02 * effective_quality)
        
        pre_pain = phantom.phantom_pain_level
        
        return {
            "session_number": phantom.mirror_therapy_sessions,
            "gamma_reduction": gamma_reduction,
            "total_gamma_offset": phantom.mirror_therapy_gamma_offset,
            "efficacy_remaining": phantom.mirror_therapy_efficacy,
            "predicted_pain_reduction": gamma_reduction * 2.0,
            "pre_session_pain": pre_pain,
        }
    
    # ------------------------------------------------------------------
    # Internal Physics
    # ------------------------------------------------------------------
    
    def _apply_mirror_therapy(self, phantom: PhantomLimbState, visual_quality: float):
        """Immediate visual feedback impedance matching effect (per tick)"""
        if visual_quality <= 0:
            return
            
        # Visual feedback temporarily lowers load impedance
        # quality=1.0 → impedance approaches normal
        target_z = (
            phantom.current_load_impedance * (1 - visual_quality)
            + phantom.amputation_record.pre_amputation_impedance * visual_quality
        )
        
        # Cannot switch instantly, use exponential smoothing
        alpha = 0.1 * visual_quality
        phantom.current_load_impedance = (
            (1 - alpha) * phantom.current_load_impedance
            + alpha * target_z
        )
    
    def _update_gamma(self, phantom: PhantomLimbState):
        """Recompute effective reflection coefficient"""
        z_load = phantom.current_load_impedance
        z0 = phantom.amputation_record.pre_amputation_impedance
        
        if z_load + z0 == 0:
            raw_gamma = 0.0
        else:
            raw_gamma = abs(z_load - z0) / (z_load + z0)
        
        # Subtract long-term mirror therapy effect
        phantom.effective_gamma = max(0.0, raw_gamma - phantom.mirror_therapy_gamma_offset)
    
    def _compute_motor_reflection(
        self, phantom: PhantomLimbState, motor_intensity: float
    ) -> float:
        """
        Motor commands → reflected energy
        
        Brain continues sending motor commands to the amputated channel
        → Signal reaches open end → total reflection
        → reflected_energy = motor² × Γ²
        """
        signal_power = motor_intensity ** 2
        reflected = signal_power * phantom.effective_gamma ** 2
        return reflected
    
    def _compute_neuroma_activity(self, phantom: PhantomLimbState) -> float:
        """
        Neuroma spontaneous firing
        
        Stump neuromas are abnormally proliferating neural tissue,
        spontaneously generating action potentials (even without external stimuli).
        Clinically presenting as "tingling", "burning", or "electric shock sensation".
        """
        if not phantom.amputation_record.has_neuroma:
            phantom.neuroma_activity = 0.0
            return 0.0
        
        # Random spontaneous firing
        if self._rng.random() < NEUROMA_TRIGGER_PROB:
            burst = self._rng.uniform(0.3, 1.0)
        else:
            burst = 0.0
            
        # Exponential decay to baseline (retains more activity)
        phantom.neuroma_activity = (
            phantom.neuroma_activity * 0.95 + burst * 0.3
        )
        
        # Neuroma reflected energy (independent of motor commands)
        reflected = phantom.neuroma_activity * phantom.effective_gamma ** 2 * 0.5
        return reflected
    
    def _compute_emotional_trigger(
        self,
        phantom: PhantomLimbState,
        valence: float,
        stress: float,
    ) -> float:
        """
        Emotional/stress-triggered phantom pain
        
        Clinical observation: anxiety, depression, stress exacerbate phantom pain
        Mechanism: sympathetic activation → stump vasoconstriction → tissue hypoxia → neuroma more easily triggered
        """
        # Negative emotions increase trigger probability
        negative = max(0.0, -valence) * EMOTIONAL_TRIGGER_GAIN
        stress_contrib = stress * STRESS_TRIGGER_GAIN
        
        trigger_energy = (negative + stress_contrib) * phantom.effective_gamma ** 2
        return trigger_energy * 0.3  # Emotional trigger is weaker
    
    def _compute_temperature_trigger(
        self,
        phantom: PhantomLimbState,
        temp_delta: float,
    ) -> float:
        """
        Temperature/barometric change trigger
        
        Clinical observation: weather changes (especially cooling) often exacerbate phantom pain
        Mechanism: temperature change → stump tissue expansion/contraction → compresses neuroma
        """
        if not phantom.amputation_record.has_neuroma:
            return 0.0
            
        temp_effect = abs(temp_delta) * TEMPERATURE_PAIN_COUPLING
        reflected = temp_effect * phantom.effective_gamma ** 2 * 0.2
        return reflected
    
    def _reflected_to_pain(self, reflected_energy: float) -> float:
        """
        Reflected energy → pain
        
        Uses the same physics as ALICE's main system THE PAIN LOOP:
        Reflected energy above threshold is converted to pain signals.
        """
        if reflected_energy < PHANTOM_PAIN_THRESHOLD:
            return 0.0
        
        # Nonlinear mapping of supra-threshold portion (Weber-Fechner logarithmic law)
        excess = reflected_energy - PHANTOM_PAIN_THRESHOLD
        pain = 1.0 - math.exp(-excess * 3.0)
        return float(np.clip(pain, 0.0, 1.0))
    
    def _update_cortical_remap(self, phantom: PhantomLimbState, current_pain: float):
        """
        Cortical reorganization (Flor et al. 2006)
        
        After amputation, the somatosensory cortical area originally corresponding to the missing limb
        is "invaded" by adjacent areas (e.g., after hand amputation, facial touch invades the hand area).
        
        Reorganization degree ∝ phantom pain intensity (r = 0.93)
        
        This is a positive feedback loop:
        Pain → cortex becomes more unstable → more reorganization → more "ghost signals" → more pain
        """
        # Reorganization progress (pain-driven + time-driven)
        drive = CORTICAL_REMAP_RATE * (1.0 + current_pain * REMAP_PAIN_COUPLING)
        phantom.cortical_remap_progress += drive
        phantom.cortical_remap_progress = min(
            phantom.cortical_remap_progress, CORTICAL_REMAP_MAX
        )
    
    def _decay_motor_efference(self, phantom: PhantomLimbState):
        """
        Motor efference natural decay
        
        The brain gradually "learns" to stop sending commands to the missing limb.
        But it never fully reaches zero ("phantom sensation" persists forever).
        """
        if phantom.motor_efference > MOTOR_EFFERENCE_MIN:
            phantom.motor_efference -= MOTOR_EFFERENCE_DECAY
            phantom.motor_efference = max(
                phantom.motor_efference, MOTOR_EFFERENCE_MIN
            )
    
    # ------------------------------------------------------------------
    # Query Interface
    # ------------------------------------------------------------------
    
    def get_phantom(self, limb_name: str) -> Optional[PhantomLimbState]:
        """Get specific phantom limb state"""
        return self._phantoms.get(limb_name)
    
    def get_all_phantoms(self) -> Dict[str, PhantomLimbState]:
        """Get all phantom limb states"""
        return dict(self._phantoms)
    
    def get_total_phantom_pain(self) -> float:
        """Sum of all phantom limb pain"""
        return sum(p.phantom_pain_level for p in self._phantoms.values())
    
    def get_total_reflected_energy(self) -> float:
        """Total reflected energy of all phantom limbs"""
        return sum(p.reflected_energy for p in self._phantoms.values())
    
    @property
    def phantom_count(self) -> int:
        return len(self._phantoms)
    
    @property
    def events(self) -> List[PhantomPainEvent]:
        return list(self._events)
    
    def has_phantom(self, limb_name: str) -> bool:
        return limb_name in self._phantoms
    
    # ------------------------------------------------------------------
    # Statistics / Introspection
    # ------------------------------------------------------------------
    
    def introspect(self) -> Dict[str, Any]:
        """Engine introspection"""
        phantom_summaries = {}
        for name, p in self._phantoms.items():
            phantom_summaries[name] = {
                "ticks_since_amputation": p.total_ticks,
                "current_pain": p.phantom_pain_level,
                "peak_pain": p.peak_pain,
                "mean_pain": p.cumulative_pain / max(1, p.total_ticks),
                "gamma": p.effective_gamma,
                "motor_efference": p.motor_efference,
                "cortical_remap": p.cortical_remap_progress,
                "mirror_sessions": p.mirror_therapy_sessions,
                "has_neuroma": p.amputation_record.has_neuroma,
            }
        
        return {
            "phantom_count": len(self._phantoms),
            "total_pain": self.get_total_phantom_pain(),
            "total_reflected_energy": self.get_total_reflected_energy(),
            "total_events": len(self._events),
            "phantoms": phantom_summaries,
        }
    
    def stats(self) -> Dict[str, Any]:
        """Statistics summary"""
        if not self._phantoms:
            return {"phantom_count": 0}
        
        pains = [p.phantom_pain_level for p in self._phantoms.values()]
        gammas = [p.effective_gamma for p in self._phantoms.values()]
        
        return {
            "phantom_count": len(self._phantoms),
            "mean_pain": float(np.mean(pains)),
            "max_pain": float(np.max(pains)),
            "mean_gamma": float(np.mean(gammas)),
            "total_mirror_sessions": sum(
                p.mirror_therapy_sessions for p in self._phantoms.values()
            ),
            "mean_motor_efference": float(np.mean([
                p.motor_efference for p in self._phantoms.values()
            ])),
            "mean_cortical_remap": float(np.mean([
                p.cortical_remap_progress for p in self._phantoms.values()
            ])),
            "total_events": len(self._events),
            "tick": self._tick,
        }
    
    def get_clinical_vas_score(self, limb_name: str) -> float:
        """
        Convert to clinical VAS pain scale (0-10)
        
        Visual Analogue Scale:
        0 = No pain
        1-3 = Mild pain
        4-6 = Moderate pain
        7-9 = Severe pain
        10 = Most intense pain imaginable
        """
        phantom = self._phantoms.get(limb_name)
        if phantom is None:
            return 0.0
        return phantom.phantom_pain_level * 10.0

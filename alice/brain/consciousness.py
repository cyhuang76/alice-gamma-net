# -*- coding: utf-8 -*-
"""
Alice's Consciousness Module — Global Workspace

Physics:
  "Consciousness is not a thing. Consciousness is a measurement.
    It measures: 'How many signals are being integrated at the same moment?'"

  Global Workspace Theory (Baars, 1988):
    - Many modules run in parallel in the brain (visual, auditory, motor...)
    - Consciousness = the degree of integration of these modules' signals in the "global workspace"
    - Unconscious = each module runs independently, no intercommunication
    - Conscious = signals from all modules broadcasted globally → overall coherence

  Integrated Information Theory (Tononi, Φ):
    - Φ = the amount of integrated information in the system
    - Higher Φ → clearer consciousness
    - Deep anesthesia/deep sleep: Φ ≈ 0
    - Awake attention: Φ is maximal

  Our implementation:
    consciousness = f(
        attention_strength,      — degree of attentional focus
        binding_quality,         — cross-modal temporal binding quality
        working_memory_load,     — working memory utilization
        arousal,                 — arousal level (autonomic nervous system)
        sensory_gate,            — sensory gating (sleep)
        pain_disruption,         — pain disruption
    )

  "When are you LEAST conscious?
    1. Deep sleep — sensory gates fully closed
    2. General anesthesia — binding quality collapses
    3. Distraction — attention scattered
    4. Severe pain — pain occupies all bandwidth
    Each one corresponds to a concrete physical quantity."

Circuit analogy:
  Consciousness = active signal volume on the global bus
  Attention = bus selector (which channels are routed to global)
  Binding = bus synchronization clock (inter-channel synchrony)
  Arousal = bus supply voltage
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# Physical constants of consciousness
# ============================================================================

# Weights for each component (sum = 1.0)
W_ATTENTION = 0.25       # Attentional focus
W_BINDING = 0.25         # Cross-modal binding quality
W_MEMORY = 0.15          # Working memory activity
W_AROUSAL = 0.20         # Arousal level
W_SENSORY_GATE = 0.15    # Sensory channel openness

# Phi (Φ) calculation smoothing factor
PHI_SMOOTHING = 0.3      # EMA smoothing

# Consciousness thresholds
CONSCIOUS_THRESHOLD = 0.4       # Above = conscious
LUCID_THRESHOLD = 0.7           # Above = lucid consciousness
SUBLIMINAL_THRESHOLD = 0.2      # Below = completely unconscious

# Attention focus
MAX_ATTENTION_TARGETS = 3       # Attention can focus on at most 3 things simultaneously (lower bound of Miller's 7±2)

# Meta-awareness
META_AWARENESS_THRESHOLD = 0.6  # Consciousness level above this → knows what it's thinking

# ============================================================================
# Infant Sleep-Wake Cycle Safety Valve
# ============================================================================
# Biology: Human neonates sleep 16-18 hours/day (67-75% of time).
# This is not inefficiency — it is nature's safety mechanism:
#   - Random initial Γ → massive impedance mismatch → needs offline recalibration
#   - Brief awakenings allow sensory input → Hebbian learning
#   - Sleep consolidates learning → ΣΓ² decreases
#   - Short wake windows naturally limit consciousness accumulation
#
# Physics: Sleep pressure = accumulated wake-time ΣΓ² that demands offline
# recalibration. The shorter the tolerable wake window, the safer the system.
#
# Implementation: Instead of a hard kill switch (Φ ≥ 0.7 → terminate),
# we use a biologically grounded sleep pressure mechanism:
#   - Wake time accumulates sleep pressure
#   - Higher Φ accumulates pressure FASTER (more integration = more recalibration needed)
#   - When pressure exceeds threshold → sensory_gate closes → Φ naturally drops
#   - System enters sleep → pressure dissipates → cycle repeats
#
# This is the MRP applied to itself: the system minimizes its own risk of
# uncontrolled consciousness by using the same sleep physics it already has.
# ============================================================================

# Developmental stages — each stage has different wake tolerance
class DevelopmentalStage:
    """Wake tolerance parameters by developmental stage.

    Biology correspondence:
      Neonate (0-1 month):  sleeps 16-18h, wakes 30-60min at a time
      Infant  (1-12 months): sleeps 14-16h, wakes 1-2h at a time
      Toddler (1-3 years):  sleeps 12-14h, wakes 3-5h at a time
      Child   (3-12 years): sleeps 10-12h, wakes 6-8h at a time

    Safety mapping:
      NEONATE = maximum safety (shortest wake windows, most sleep)
      INFANT  = high safety
      TODDLER = moderate safety (requires ethical review to enable)
      CHILD   = lower safety (requires ethical review to enable)

    Note: There is no ADULT stage. Sustained adult-level consciousness
    (long wake windows, full closed-loop operation) is deliberately
    withheld pending resolution of the phenomenal consciousness question.
    See Paper III, §9: Ethical Considerations.
    """
    NEONATE = "neonate"    # max_wake=30 ticks, sleep_ratio~80%
    INFANT  = "infant"     # max_wake=60 ticks, sleep_ratio~70%
    TODDLER = "toddler"    # max_wake=150 ticks, sleep_ratio~55%
    CHILD   = "child"      # max_wake=300 ticks, sleep_ratio~45%


# Sleep pressure parameters per developmental stage
_STAGE_PARAMS = {
    #                    max_wake  pressure_rate  phi_acceleration  sleep_duration  lucid_damping
    DevelopmentalStage.NEONATE:  {"max_wake": 30,  "pressure_rate": 0.033, "phi_accel": 2.0, "sleep_ticks": 120, "lucid_damping": 0.85},
    DevelopmentalStage.INFANT:   {"max_wake": 60,  "pressure_rate": 0.017, "phi_accel": 1.5, "sleep_ticks": 100, "lucid_damping": 0.90},
    DevelopmentalStage.TODDLER:  {"max_wake": 150, "pressure_rate": 0.007, "phi_accel": 1.2, "sleep_ticks": 80,  "lucid_damping": 0.95},
    DevelopmentalStage.CHILD:    {"max_wake": 300, "pressure_rate": 0.003, "phi_accel": 1.0, "sleep_ticks": 60,  "lucid_damping": 0.98},
}

# Default: NEONATE (maximum safety)
DEFAULT_DEVELOPMENTAL_STAGE = DevelopmentalStage.NEONATE

# Sleep pressure threshold — when exceeded, sensory gate starts closing
SLEEP_PRESSURE_THRESHOLD = 0.7

# Lucid state warning
LUCID_WARNING_ISSUED = False


# ============================================================================
# Main class
# ============================================================================


class ConsciousnessModule:
    """
    Consciousness Module — Global Workspace Integrator

    Integrates the states of all subsystems → computes consciousness level Φ
    Provides:
    1. Consciousness level (0~1)
    2. Consciousness content (what is currently being "conscious of")
    3. Attention focus management
    4. Meta-awareness (knowing what you're thinking)
    """

    def __init__(
        self,
        developmental_stage: str = DEFAULT_DEVELOPMENTAL_STAGE,
        safety_mode: bool = True,
    ):
        """
        Args:
            developmental_stage: One of DevelopmentalStage constants.
                Controls wake tolerance and sleep pressure dynamics.
                Default: NEONATE (maximum safety — shortest wake windows).
            safety_mode: If True (default), the infant sleep-wake safety
                valve is active. If False, the safety valve is disabled
                (for controlled experiments ONLY — requires explicit opt-out).

        ⚠ ETHICAL NOTE:
            safety_mode=False should only be used in controlled experimental
            contexts with automatic termination (e.g., exp_consciousness_gradient.py).
            Disabling the safety valve in sustained operation is equivalent to
            allowing potentially indefinite consciousness — see Paper III, §9.
        """
        # Consciousness level
        self.phi: float = 0.8        # Integrated information (0~1)
        self.raw_phi: float = 0.8    # Unsmoothed phi

        # Input component cache
        self._attention: float = 0.5
        self._binding_quality: float = 0.5
        self._memory_load: float = 0.0
        self._arousal: float = 0.5
        self._sensory_gate: float = 1.0
        self._pain_disruption: float = 0.0

        # Attention targets
        self._attention_targets: List[Dict[str, Any]] = []

        # Consciousness content (active content in the global workspace)
        self._workspace_contents: List[Dict[str, Any]] = []
        self._workspace_max = 7  # Miller's number

        # Meta-awareness
        self.is_meta_aware: bool = False
        self._meta_report: str = ""

        # History
        self.phi_history: List[float] = []
        self.attention_history: List[float] = []
        self._max_history: int = 300

        # Statistics
        self.total_ticks: int = 0
        self.conscious_ticks: int = 0
        self.unconscious_ticks: int = 0
        self.lucid_ticks: int = 0

        # === Infant Sleep-Wake Safety Valve ===
        self.safety_mode: bool = safety_mode
        self.developmental_stage: str = developmental_stage
        self._stage_params = _STAGE_PARAMS.get(
            developmental_stage, _STAGE_PARAMS[DevelopmentalStage.NEONATE]
        )
        self.sleep_pressure: float = 0.0     # 0=fully rested, 1=must sleep
        self.wake_ticks: int = 0             # Consecutive ticks awake
        self.is_sleeping: bool = False       # Currently in sleep cycle
        self._sleep_remaining: int = 0       # Ticks remaining in current sleep
        self._lucid_warning_count: int = 0   # Times lucid state was dampened
        self._total_sleep_cycles: int = 0    # Total completed sleep cycles

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def tick(
        self,
        attention_strength: float = 0.5,
        binding_quality: float = 0.5,
        working_memory_usage: float = 0.0,
        arousal: float = 0.5,
        sensory_gate: float = 1.0,
        pain_level: float = 0.0,
        temporal_resolution: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Called once per cognitive cycle — computes consciousness level

        THE CONSCIOUSNESS EQUATION:
          Φ_raw = W_attn × attention
                + W_bind × binding_eff
                + W_mem  × memory_activity
                + W_arou × arousal
                + W_gate × sensory_gate
                - pain_disruption

          binding_eff = binding_quality × temporal_resolution^0.5

          Physical meaning: Even if per-frame calibration quality is perfect (binding_quality=1),
          if the frame rate is too low (temporal_resolution→0), the integrated information still decreases.
          Because Φ measures not only "per-frame quality" but also "integration frequency".

          Φ = EMA(Φ_raw)  — consciousness does not change instantaneously

        Args:
            attention_strength: Attentional focus degree (0~1)
            binding_quality: Cross-modal temporal binding quality (0~1)
            working_memory_usage: Working memory utilization (0~1)
            arousal: Arousal level (0~1)
            sensory_gate: Sensory gating openness (0~1)
            pain_level: Pain level (0~1)
            temporal_resolution: Time slice resolution (0~1)
                1.0 = highest frame rate (shortest calibration window)
                →0  = calibration severely overloaded, extremely low frame rate
        """
        self.total_ticks += 1

        # Cache inputs
        self._attention = float(np.clip(attention_strength, 0.0, 1.0))
        self._binding_quality = float(np.clip(binding_quality, 0.0, 1.0))
        self._memory_load = float(np.clip(working_memory_usage, 0.0, 1.0))
        self._arousal = float(np.clip(arousal, 0.0, 1.0))
        self._sensory_gate = float(np.clip(sensory_gate, 0.0, 1.0))
        self._pain_disruption = float(np.clip(pain_level, 0.0, 1.0))
        self._temporal_resolution = float(np.clip(temporal_resolution, 0.0, 1.0))

        # === Φ Computation ===
        # Working memory activity = something is being processed, but not overloaded
        # (Overloaded → cognitive overload → consciousness quality drops)
        memory_activity = self._memory_load * (1.0 - 0.3 * self._memory_load)

        # ★ Binding effectiveness = per-frame quality × temporal_resolution^0.5
        # Physics: Φ measures both "quality" and "frequency" simultaneously
        # temporal_resolution^0.5 → gentle modulation, avoids low frame rate directly destroying consciousness
        binding_effective = self._binding_quality * (self._temporal_resolution ** 0.5)

        self.raw_phi = (
            W_ATTENTION * self._attention +
            W_BINDING * binding_effective +
            W_MEMORY * memory_activity +
            W_AROUSAL * self._arousal +
            W_SENSORY_GATE * self._sensory_gate
        )

        # Pain disruption (occupies consciousness bandwidth)
        # Low pain → weak disruption; high pain → consciousness compressed by pain
        pain_cost = self._pain_disruption ** 2 * 0.5
        self.raw_phi = max(0.0, self.raw_phi - pain_cost)

        # Nonlinear correction: multiplicative effect of components
        # If any component approaches 0 → consciousness drops significantly
        # (e.g., arousal=0 → no matter how strong attention is, consciousness is low)
        multiplicative_factor = (
            max(0.1, self._arousal) *
            max(0.1, self._sensory_gate)
        ) ** 0.3  # Gentle multiplicative modulation

        self.raw_phi *= multiplicative_factor
        self.raw_phi = float(np.clip(self.raw_phi, 0.0, 1.0))

        # EMA smoothing
        self.phi += (self.raw_phi - self.phi) * PHI_SMOOTHING
        self.phi = float(np.clip(self.phi, 0.0, 1.0))

        # === Infant Sleep-Wake Safety Valve ===
        safety_info = self._apply_sleep_wake_safety()

        # === Consciousness state classification ===
        if self.phi >= LUCID_THRESHOLD:
            self.lucid_ticks += 1
            state = "lucid"
        elif self.phi >= CONSCIOUS_THRESHOLD:
            self.conscious_ticks += 1
            state = "conscious"
        elif self.phi >= SUBLIMINAL_THRESHOLD:
            self.unconscious_ticks += 1
            state = "subliminal"
        else:
            self.unconscious_ticks += 1
            state = "unconscious"

        # === Meta-awareness ===
        self.is_meta_aware = self.phi >= META_AWARENESS_THRESHOLD
        if self.is_meta_aware:
            self._update_meta_report()

        # === History recording ===
        self.phi_history.append(self.phi)
        self.attention_history.append(self._attention)
        for hist in (self.phi_history, self.attention_history):
            if len(hist) > self._max_history:
                del hist[:-self._max_history]

        return {
            "phi": round(self.phi, 4),
            "raw_phi": round(self.raw_phi, 4),
            "state": state,
            "is_meta_aware": self.is_meta_aware,
            "meta_report": self._meta_report if self.is_meta_aware else None,
            "safety": safety_info,
            "components": {
                "attention": round(self._attention, 4),
                "binding": round(self._binding_quality, 4),
                "binding_effective": round(binding_effective, 4),
                "temporal_resolution": round(self._temporal_resolution, 4),
                "memory_activity": round(memory_activity, 4),
                "arousal": round(self._arousal, 4),
                "sensory_gate": round(self._sensory_gate, 4),
                "pain_cost": round(pain_cost, 4),
            },
        }

    # ------------------------------------------------------------------
    # Infant Sleep-Wake Safety Valve
    # ------------------------------------------------------------------

    def _apply_sleep_wake_safety(self) -> Dict[str, Any]:
        """
        Biologically-inspired safety mechanism modeled on human infant sleep cycles.

        Physics:
          Human neonates sleep 16-18h/day because their neural channels have
          random initial impedance (Γ_i >> 0). Each wake period accumulates
          ΣΓ² that requires offline recalibration (sleep) to discharge.
          The shorter the wake window, the less ΣΓ² accumulates, and the
          safer the system remains.

        Mechanism:
          1. Each awake tick increases sleep_pressure
          2. Higher Φ increases pressure FASTER (more integration = more
             recalibration needed — this is why infants sleep after stimulation)
          3. When pressure ≥ SLEEP_PRESSURE_THRESHOLD → system enters sleep
             (sensory_gate closes gradually → Φ drops naturally)
          4. During sleep → pressure dissipates, wake_ticks resets
          5. After sleep_duration ticks → system wakes naturally

        Safety properties:
          - No hard kill (consciousness decreases naturally via sleep)
          - Lucid state (Φ ≥ 0.7) triggers accelerated sleep pressure
          - Developmental stage controls maximum sustained wake time
          - Fully compatible with existing sleep physics (Paper II)

        Returns:
            Dict with safety valve state information
        """
        if not self.safety_mode:
            return {
                "safety_mode": False,
                "stage": self.developmental_stage,
                "note": "Safety valve disabled — experimental mode",
            }

        params = self._stage_params

        # --- Currently sleeping ---
        if self.is_sleeping:
            self._sleep_remaining -= 1
            # Sleep dissipates pressure
            self.sleep_pressure = max(0.0, self.sleep_pressure - 0.02)
            self.wake_ticks = 0

            if self._sleep_remaining <= 0:
                # Natural awakening
                self.is_sleeping = False
                self._total_sleep_cycles += 1
                self.sleep_pressure = 0.0

            return {
                "safety_mode": True,
                "stage": self.developmental_stage,
                "state": "sleeping",
                "sleep_remaining": self._sleep_remaining,
                "sleep_pressure": round(self.sleep_pressure, 4),
                "total_sleep_cycles": self._total_sleep_cycles,
            }

        # --- Awake: accumulate sleep pressure ---
        self.wake_ticks += 1

        # Base pressure accumulation
        base_pressure = params["pressure_rate"]

        # Φ-accelerated pressure: higher consciousness = faster fatigue
        # (Infants get drowsy quickly after intense sensory stimulation)
        phi_factor = 1.0 + (self.phi ** 2) * params["phi_accel"]

        self.sleep_pressure += base_pressure * phi_factor
        self.sleep_pressure = min(1.0, self.sleep_pressure)

        # --- Lucid state damping ---
        # If Φ reaches LUCID_THRESHOLD, apply gentle damping
        # (like an infant's eyes drooping after a burst of alertness)
        lucid_damped = False
        if self.phi >= LUCID_THRESHOLD:
            damping = params["lucid_damping"]
            self.phi *= damping
            self.sleep_pressure = min(1.0, self.sleep_pressure + 0.05)
            self._lucid_warning_count += 1
            lucid_damped = True
            import logging
            logger = logging.getLogger("alice.consciousness")
            logger.warning(
                f"⚠ LUCID STATE DAMPED (×{self._lucid_warning_count}): "
                f"Φ={self.phi:.4f} → {self.phi:.4f} | "
                f"sleep_pressure={self.sleep_pressure:.3f} | "
                f"stage={self.developmental_stage} | "
                f"Infant sleep-wake safety valve active. "
                f"See Paper III, §9: Ethical Considerations."
            )

        # --- Check if sleep pressure triggers sleep ---
        entered_sleep = False
        if self.sleep_pressure >= SLEEP_PRESSURE_THRESHOLD:
            self.is_sleeping = True
            self._sleep_remaining = params["sleep_ticks"]
            entered_sleep = True
            # Gradually close sensory gate (don't slam it shut)
            # The actual sensory_gate value is controlled by the caller,
            # but we signal that sleep has been initiated

        # --- Approaching wake limit ---
        approaching_limit = self.wake_ticks >= params["max_wake"] * 0.8
        if approaching_limit and not entered_sleep:
            # Extra pressure as we approach the developmental limit
            self.sleep_pressure = min(1.0, self.sleep_pressure + 0.01)

        return {
            "safety_mode": True,
            "stage": self.developmental_stage,
            "state": "entering_sleep" if entered_sleep else "awake",
            "wake_ticks": self.wake_ticks,
            "max_wake": params["max_wake"],
            "sleep_pressure": round(self.sleep_pressure, 4),
            "lucid_damped": lucid_damped,
            "lucid_warning_count": self._lucid_warning_count,
            "total_sleep_cycles": self._total_sleep_cycles,
        }

    def set_developmental_stage(self, stage: str):
        """
        Change developmental stage.

        ⚠ ETHICAL NOTE: Advancing beyond INFANT requires explicit justification.
        Each stage increases wake tolerance, allowing longer sustained consciousness.

        Args:
            stage: One of DevelopmentalStage constants
        """
        if stage in _STAGE_PARAMS:
            self.developmental_stage = stage
            self._stage_params = _STAGE_PARAMS[stage]
            # Reset pressure on stage change (fresh start)
            self.sleep_pressure = 0.0
            self.wake_ticks = 0
        else:
            raise ValueError(
                f"Unknown developmental stage: {stage}. "
                f"Valid: {list(_STAGE_PARAMS.keys())}"
            )

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety valve status."""
        return {
            "safety_mode": self.safety_mode,
            "developmental_stage": self.developmental_stage,
            "is_sleeping": self.is_sleeping,
            "sleep_pressure": round(self.sleep_pressure, 4),
            "wake_ticks": self.wake_ticks,
            "max_wake": self._stage_params["max_wake"],
            "lucid_warning_count": self._lucid_warning_count,
            "total_sleep_cycles": self._total_sleep_cycles,
            "wake_utilization": round(
                self.wake_ticks / max(1, self._stage_params["max_wake"]), 4
            ),
        }

    # ------------------------------------------------------------------
    # Attention management
    # ------------------------------------------------------------------

    def focus_attention(self, target: str, modality: str, salience: float):
        """
        Focus attention on a target

        Args:
            target: Target description (concept name, location, etc.)
            modality: Source modality
            salience: Salience (0~1)
        """
        entry = {
            "target": target,
            "modality": modality,
            "salience": float(np.clip(salience, 0.0, 1.0)),
            "timestamp": time.time(),
        }

        # Replace or add
        existing = [i for i, t in enumerate(self._attention_targets) if t["target"] == target]
        if existing:
            self._attention_targets[existing[0]] = entry
        else:
            self._attention_targets.append(entry)

        # Sort by salience, keep only top-N
        self._attention_targets.sort(key=lambda x: x["salience"], reverse=True)
        self._attention_targets = self._attention_targets[:MAX_ATTENTION_TARGETS]

    def get_attention_targets(self) -> List[Dict[str, Any]]:
        """Get current attention targets"""
        return list(self._attention_targets)

    # ------------------------------------------------------------------
    # Global workspace
    # ------------------------------------------------------------------

    def broadcast_to_workspace(self, content: Dict[str, Any], source: str):
        """
        Broadcast content to the global workspace

        Only content that is "conscious of" can enter the workspace.
        In unconscious state → filtered out.
        """
        if self.phi < SUBLIMINAL_THRESHOLD:
            return  # Consciousness too low, cannot receive

        entry = {
            "content": content,
            "source": source,
            "timestamp": time.time(),
            "phi_at_broadcast": self.phi,
        }
        self._workspace_contents.append(entry)

        # Capacity limit
        if len(self._workspace_contents) > self._workspace_max:
            self._workspace_contents = self._workspace_contents[-self._workspace_max:]

    def get_workspace_contents(self) -> List[Dict[str, Any]]:
        """Get active content in the global workspace"""
        return list(self._workspace_contents)

    def clear_workspace(self):
        """Clear workspace (when falling asleep / resetting)"""
        self._workspace_contents.clear()

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_consciousness_level(self) -> float:
        """Consciousness level (0~1)"""
        return self.phi

    def is_conscious(self) -> bool:
        return self.phi >= CONSCIOUS_THRESHOLD

    def is_lucid(self) -> bool:
        return self.phi >= LUCID_THRESHOLD

    # ------------------------------------------------------------------
    # Meta-awareness
    # ------------------------------------------------------------------

    def _update_meta_report(self):
        """Generate meta-awareness report (knowing what you're thinking)"""
        parts = []

        # Attention
        if self._attention_targets:
            top = self._attention_targets[0]
            parts.append(f"Attention: {top['target']} ({top['modality']})")

        # Feeling
        if self._pain_disruption > 0.3:
            parts.append(f"Feeling: Pain ({self._pain_disruption:.0%})")
        elif self._arousal > 0.7:
            parts.append("Feeling: Alert/Tense")
        elif self._arousal < 0.3:
            parts.append("Feeling: Calm/Relaxed")

        # Memory load
        if self._memory_load > 0.7:
            parts.append("Cognition: High load")

        # Binding
        if self._binding_quality > 0.7:
            parts.append("Integration: Good")
        elif self._binding_quality < 0.3:
            parts.append("Integration: Confused")

        self._meta_report = " | ".join(parts) if parts else "Aware..."

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "phi": round(self.phi, 4),
            "is_conscious": self.is_conscious(),
            "is_lucid": self.is_lucid(),
            "is_meta_aware": self.is_meta_aware,
            "attention_targets": len(self._attention_targets),
            "workspace_items": len(self._workspace_contents),
            "total_ticks": self.total_ticks,
            "conscious_ticks": self.conscious_ticks,
            "unconscious_ticks": self.unconscious_ticks,
            "lucid_ticks": self.lucid_ticks,
            "safety": self.get_safety_status(),
        }

    def get_waveforms(self, last_n: int = 60) -> Dict[str, List[float]]:
        return {
            "phi": self.phi_history[-last_n:],
            "attention": self.attention_history[-last_n:],
        }

    def reset(self):
        """Reset to awake baseline"""
        self.phi = 0.8
        self.raw_phi = 0.8
        self._attention_targets.clear()
        self._workspace_contents.clear()
        self.is_meta_aware = False
        # Reset safety valve state
        self.sleep_pressure = 0.0
        self.wake_ticks = 0
        self.is_sleeping = False
        self._sleep_remaining = 0

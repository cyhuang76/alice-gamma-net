# -*- coding: utf-8 -*-
"""
Alice's Sleep Cycle — Consciousness Gating & Memory Consolidation

Physics:
  "Sleep is not shutdown. Sleep is offline maintenance mode."

  Wake → N1 (light sleep) → N2 (spindle waves) → N3 (deep sleep/δ waves) → REM (rapid eye movement)
  A complete cycle is about 90 minutes. 4-6 cycles per night.

  Each stage has different brainwave frequency bands:
    Wake:  β/γ  (13-100 Hz)  — attention, thinking
    N1:    α/θ  (4-13 Hz)    — falling asleep
    N2:    θ    (4-8 Hz)     — spindle waves + K complexes
    N3:    δ    (0.5-4 Hz)   — deep sleep, memory consolidation
    REM:   θ/β  (4-30 Hz)    — dreaming, emotional processing

  "This is why the δ band in PerceptionPipeline's
    BAND_INFO_LAYER is 'background' — because δ is the
    background state, the brain doing offline cleanup."

Circuit analogy:
  Wake = system at full load (all channels open)
  N1   = beginning to close non-critical channels
  N2   = only maintenance channels remain (spindle waves = memory transfer pulses)
  N3   = lowest power mode (only cleanup circuit activity)
  REM  = diagnostic mode (random activation testing pathways = dreaming)

Memory consolidation:
  During N3 deep sleep → Hebbian learning enhanced (consolidate_memory)
  During REM → emotional memory processing (pruning unnecessary synapses)
  This is why poor sleep → can't remember things → emotional instability

Sensory gating:
  Deeper sleep → higher sensory threshold → harder to wake up
  But always keeps "CRITICAL" channel (baby crying/fire alarm penetrates sleep)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# Sleep stages
# ============================================================================


class SleepStage(Enum):
    """Sleep stages — each stage corresponds to different brainwaves"""
    WAKE = "wake"         # β/γ — Awake
    N1 = "n1"             # α/θ — Light sleep transition
    N2 = "n2"             # θ + spindle waves — Light sleep
    N3 = "n3"             # δ — Deep sleep (slow-wave sleep)
    REM = "rem"           # θ/β — Rapid eye movement


# ============================================================================
# Physical parameters for each stage
# ============================================================================

# {stage: (sensory_gate, consolidation_rate, consciousness_level, dominant_freq_hz)}
STAGE_PARAMS: Dict[SleepStage, Dict[str, float]] = {
    SleepStage.WAKE: {
        "sensory_gate": 1.0,        # Sensory channels fully open
        "consolidation_rate": 0.0,  # No consolidation while awake
        "consciousness": 1.0,      # Fully awake
        "dominant_freq": 20.0,     # β waves
        "arousal_threshold": 0.0,  # Any stimulus is perceived
        "energy_recovery": 0.0,    # No recovery while awake
    },
    SleepStage.N1: {
        "sensory_gate": 0.6,        # Sensory starting to close
        "consolidation_rate": 0.1,
        "consciousness": 0.6,
        "dominant_freq": 9.0,      # α waves
        "arousal_threshold": 0.2,  # Awakened by mild stimuli
        "energy_recovery": 0.005,
    },
    SleepStage.N2: {
        "sensory_gate": 0.3,        # Most sensory channels closed
        "consolidation_rate": 0.3,  # Spindle waves transfer memories
        "consciousness": 0.3,
        "dominant_freq": 6.0,      # θ waves
        "arousal_threshold": 0.4,  # Requires moderate stimulus
        "energy_recovery": 0.01,
    },
    SleepStage.N3: {
        "sensory_gate": 0.1,        # Almost fully closed (only CRITICAL remains)
        "consolidation_rate": 1.0,  # Maximum consolidation rate!
        "consciousness": 0.1,
        "dominant_freq": 2.0,      # δ waves
        "arousal_threshold": 0.7,  # Requires strong stimulus to wake
        "energy_recovery": 0.02,
    },
    SleepStage.REM: {
        "sensory_gate": 0.15,       # Sensory closed but eye movements active
        "consolidation_rate": 0.5,  # Emotional memory processing
        "consciousness": 0.4,      # Consciousness active (dreaming)
        "dominant_freq": 7.0,      # θ/β mixed
        "arousal_threshold": 0.5,
        "energy_recovery": 0.008,
    },
}

# Sleep cycle state transition order
SLEEP_CYCLE_ORDER = [
    SleepStage.WAKE,
    SleepStage.N1,
    SleepStage.N2,
    SleepStage.N3,
    SleepStage.N2,   # Ascending
    SleepStage.REM,
    # Then repeat N1 → N2 → N3 → N2 → REM
]

# Duration of each stage (ticks)
STAGE_DURATION: Dict[SleepStage, int] = {
    SleepStage.WAKE: 0,      # Infinite (sleep triggered externally)
    SleepStage.N1: 15,       # Brief transition
    SleepStage.N2: 40,       # Longest stage
    SleepStage.N3: 30,       # Deep sleep
    SleepStage.REM: 25,      # Gradually lengthens
}

# Sleep pressure
SLEEP_PRESSURE_ACCUMULATION = 0.002   # Accumulates per tick while awake
SLEEP_PRESSURE_RELEASE = 0.01        # Released per tick while sleeping
SLEEP_PRESSURE_THRESHOLD = 0.7       # Above this → should go to sleep


# ============================================================================
# Main class
# ============================================================================


class SleepCycle:
    """
    Sleep Cycle Controller

    Manages:
    1. Sleep pressure (accumulates while awake, released while sleeping)
    2. Sleep stage transitions (WAKE→N1→N2→N3→N2→REM→cycle)
    3. Sensory gating (sleep gating → affects perception sensitivity)
    4. Memory consolidation rate (affects Hebbian consolidation)
    5. Energy recovery
    """

    def __init__(self):
        # State
        self.stage: SleepStage = SleepStage.WAKE
        self.sleep_pressure: float = 0.0       # Sleep pressure (0~1)
        self.circadian_phase: float = 0.0      # Circadian rhythm phase (0~2π)

        # Cycle tracking
        self._stage_ticks: int = 0             # Ticks elapsed in current stage
        self._cycle_position: int = 0          # Position in SLEEP_CYCLE_ORDER
        self._cycles_completed: int = 0        # Number of complete cycles
        self._total_sleep_ticks: int = 0       # Total sleep ticks
        self._total_wake_ticks: int = 0        # Total wake ticks

        # Memory consolidation count
        self.consolidations_performed: int = 0

        # History
        self.stage_history: List[str] = []
        self.pressure_history: List[float] = []
        self._max_history: int = 300

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def tick(
        self,
        force_wake: bool = False,
        force_sleep: bool = False,
        external_stimulus_strength: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Called once per cognitive cycle

        Args:
            force_wake: Force awake (alarm/startle)
            force_sleep: Force sleep (anesthesia/extreme fatigue)
            external_stimulus_strength: External stimulus strength (0~1)
                If exceeds arousal_threshold → woken up

        Returns:
            {
                "stage": current stage,
                "sensory_gate": sensory gating value,
                "consolidation_rate": memory consolidation rate,
                "consciousness": consciousness level,
                "sleep_pressure": sleep pressure,
                "should_consolidate": whether memory consolidation should be performed,
                "energy_recovery": energy recovery amount,
            }
        """
        self._stage_ticks += 1
        params = STAGE_PARAMS[self.stage]

        # Circadian rhythm update
        self.circadian_phase += 2 * math.pi / 1440  # ≈ 1 min / tick
        if self.circadian_phase > 2 * math.pi:
            self.circadian_phase -= 2 * math.pi

        # === Sleep pressure update ===
        if self.stage == SleepStage.WAKE:
            self._total_wake_ticks += 1
            # Awake → pressure accumulates (circadian rhythm modulated)
            circadian_mod = 1.0 + 0.3 * math.sin(self.circadian_phase - math.pi / 2)
            self.sleep_pressure += SLEEP_PRESSURE_ACCUMULATION * circadian_mod
        else:
            self._total_sleep_ticks += 1
            # Sleeping → pressure released
            self.sleep_pressure -= SLEEP_PRESSURE_RELEASE
        self.sleep_pressure = float(np.clip(self.sleep_pressure, 0.0, 1.0))

        # === Forced commands ===
        if force_wake and self.stage != SleepStage.WAKE:
            self._transition_to(SleepStage.WAKE)
        elif force_sleep and self.stage == SleepStage.WAKE:
            self._transition_to(SleepStage.N1)
            self._cycle_position = 1

        # === External stimulus wake check ===
        if self.stage != SleepStage.WAKE:
            if external_stimulus_strength > params["arousal_threshold"]:
                self._transition_to(SleepStage.WAKE)

        # === Automatic stage transition ===
        if self.stage != SleepStage.WAKE:
            duration = STAGE_DURATION[self.stage]
            if self._stage_ticks >= duration:
                self._advance_stage()

        # === Memory consolidation check ===
        should_consolidate = False
        consolidation_rate = params["consolidation_rate"]
        if self.stage in (SleepStage.N3, SleepStage.REM, SleepStage.N2):
            if consolidation_rate > 0.2:
                should_consolidate = True
                self.consolidations_performed += 1

        # === Record history ===
        self.stage_history.append(self.stage.value)
        self.pressure_history.append(self.sleep_pressure)
        for hist in (self.stage_history, self.pressure_history):
            if len(hist) > self._max_history:
                del hist[:-self._max_history]

        return {
            "stage": self.stage.value,
            "sensory_gate": params["sensory_gate"],
            "consolidation_rate": consolidation_rate,
            "consciousness": params["consciousness"],
            "dominant_freq": params["dominant_freq"],
            "sleep_pressure": self.sleep_pressure,
            "should_consolidate": should_consolidate,
            "energy_recovery": params["energy_recovery"],
            "cycles_completed": self._cycles_completed,
            "stage_ticks": self._stage_ticks,
        }

    # ------------------------------------------------------------------
    # State query
    # ------------------------------------------------------------------

    def is_sleeping(self) -> bool:
        return self.stage != SleepStage.WAKE

    def is_deep_sleep(self) -> bool:
        return self.stage == SleepStage.N3

    def is_dreaming(self) -> bool:
        return self.stage == SleepStage.REM

    def should_sleep(self) -> bool:
        """Whether sleep pressure exceeds threshold"""
        return self.sleep_pressure >= SLEEP_PRESSURE_THRESHOLD

    def get_sensory_gate(self) -> float:
        """Get current sensory gating value (0~1)"""
        return STAGE_PARAMS[self.stage]["sensory_gate"]

    def get_consciousness_level(self) -> float:
        """Get current consciousness level"""
        return STAGE_PARAMS[self.stage]["consciousness"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transition_to(self, new_stage: SleepStage):
        """Transition to a new sleep stage"""
        self.stage = new_stage
        self._stage_ticks = 0

    def _advance_stage(self):
        """Advance to the next sleep stage"""
        self._cycle_position += 1

        # After a complete cycle, repeat
        if self._cycle_position >= len(SLEEP_CYCLE_ORDER):
            self._cycles_completed += 1
            # Skip WAKE, start new cycle directly from N1
            self._cycle_position = 1

        # If sleep pressure is released → wake naturally
        if self.sleep_pressure < 0.1:
            self._transition_to(SleepStage.WAKE)
            self._cycle_position = 0
        else:
            self._transition_to(SLEEP_CYCLE_ORDER[self._cycle_position])

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "sleep_pressure": round(self.sleep_pressure, 4),
            "circadian_phase": round(self.circadian_phase, 4),
            "cycles_completed": self._cycles_completed,
            "consolidations_performed": self.consolidations_performed,
            "total_sleep_ticks": self._total_sleep_ticks,
            "total_wake_ticks": self._total_wake_ticks,
            "is_sleeping": self.is_sleeping(),
            "should_sleep": self.should_sleep(),
        }

    def get_waveforms(self, last_n: int = 60) -> Dict[str, Any]:
        return {
            "stages": self.stage_history[-last_n:],
            "pressure": self.pressure_history[-last_n:],
        }

    def reset(self):
        """Reset to awake state"""
        self.stage = SleepStage.WAKE
        self.sleep_pressure = 0.0
        self._stage_ticks = 0
        self._cycle_position = 0

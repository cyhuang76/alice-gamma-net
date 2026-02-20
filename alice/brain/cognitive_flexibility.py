# -*- coding: utf-8 -*-
"""
Cognitive Flexibility Engine
Phase 8: Physical Model of High-Intensity Task Switching

=== Biological Background ===

The human brain handles high-intensity task changes (esports, pilots, ER doctors)
through a three-layer mechanism:

1. Task-Set Reconfiguration
   - The prefrontal cortex must "unload" the old task's control parameters and "load" the new task's
   - Physics analogy: LC circuit resonant frequency switching — requires a transient settling time
   - Switch cost = reconfiguration delay + proactive interference

2. Task-Set Inertia
   - The just-executed task leaves "residual activation" (perseveration)
   - Longer execution of the same task → stronger residual → harder to switch
   - Physics analogy: discharge time constant after capacitor charging

3. Mixing Cost
   - Frequent-switching environments are inherently slower than single-task environments
   - The prefrontal cortex must keep multiple task sets "on standby" → consumes working memory capacity
   - Physics analogy: multiple LC circuits resonating simultaneously → energy dispersal

=== Physical Dimensions ===

- τ_reconfig (reconfiguration time constant): thalamic gate reconfiguration delay during switching
- Z_inertia (inertia impedance): impedance from residual activation of the old task
- E_mixing (mixing energy cost): extra energy cost of maintaining multiple task sets
- C_task (task capacitance): task duration → affects inertia strength
- Ω_flexibility (cognitive flexibility index): comprehensive switching efficiency metric

=== Plasticity ===

Cognitive flexibility is trainable:
- Training reduces τ_reconfig (faster reconfiguration)
- Training reduces Z_inertia's impact on reaction time (faster release of old task)
- Training reduces E_mixing (maintaining multiple task sets costs less effort)
- PFC capacity grows → more task sets can be on standby simultaneously
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

# --- Reconfiguration time constant ---
RECONFIG_TAU_INITIAL = 0.150       # 150ms — untrained reconfiguration delay
RECONFIG_TAU_MIN = 0.030           # 30ms  — fastest reconfiguration after training (esports pro level)
RECONFIG_LEARNING_RATE = 0.002     # Improvement rate per successful switch
RECONFIG_DECAY_RATE = 0.00005      # Degradation when not practiced

# --- Task-set inertia ---
INERTIA_CHARGE_RATE = 0.01         # Task capacitance charge per tick (sustained execution → impedance growth)
INERTIA_MAX_CHARGE = 1.0           # Maximum inertia charge
INERTIA_DISCHARGE_TAU = 0.3        # Inertia discharge time constant (seconds, exponential decay)
INERTIA_WEIGHT = 0.100             # Inertia weight on switch cost (100ms max penalty)

# --- Mixing cost ---
MIXING_COST_PER_TASKSET = 0.02     # Extra energy cost per standby task set per tick
MIXING_RT_PENALTY = 0.020          # Baseline reaction time penalty in mixed environments (20ms)
MAX_ACTIVE_TASKSETS = 2            # Max simultaneous task sets when untrained
MAX_ACTIVE_TASKSETS_TRAINED = 4    # Max simultaneous task sets after training

# --- Proactive / retroactive interference ---
PROACTIVE_INTERFERENCE_BASE = 0.3  # Baseline intensity of old task interfering with new task
RETROACTIVE_INTERFERENCE_BASE = 0.1  # Baseline intensity of new task interfering with old task memory

# --- Cognitive flexibility plasticity ---
FLEXIBILITY_TRAINING_RATE = 0.001  # Ω improvement per successful switch
FLEXIBILITY_INITIAL = 0.5         # Initial flexibility index
FLEXIBILITY_MAX = 0.95            # Maximum flexibility index
FLEXIBILITY_DECAY = 0.000005       # Degradation when not practiced (far slower than learning — biological retention)

# --- Perseveration error ---
PERSEVERATION_ERROR_THRESHOLD = 0.7  # Inertia > this value AND low energy → perseveration error risk
PERSEVERATION_ERROR_ENERGY_THRESHOLD = 0.25  # Energy < this value → perseveration error risk

# --- Automatization savings ---
FAMILIAR_SWITCH_DISCOUNT = 0.5    # Familiar task pairs (seen >10 times) → switch cost halved


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TaskSetState:
    """Physical state of a single task set"""
    task_id: str
    charge: float = 0.0           # Task capacitance charge (higher → harder to switch away)
    total_ticks: int = 0          # Total ticks executed
    last_active_time: float = 0.0 # Last active timestamp
    activation: float = 0.0       # Current activation value [0,1]


@dataclass
class SwitchRecord:
    """Record of a single task switch"""
    from_task: str
    to_task: str
    switch_cost_ms: float         # Switch cost (milliseconds)
    inertia_penalty_ms: float     # Inertia penalty (milliseconds)
    reconfig_delay_ms: float      # Reconfiguration delay (milliseconds)
    mixing_penalty_ms: float      # Mixing cost (milliseconds)
    perseveration_error: bool     # Whether a perseveration error occurred
    energy_spent: float           # Energy consumed
    timestamp: float = 0.0


@dataclass
class FlexibilityState:
    """Global training state for cognitive flexibility"""
    reconfig_tau: float = RECONFIG_TAU_INITIAL
    flexibility_index: float = FLEXIBILITY_INITIAL
    max_active_tasksets: int = MAX_ACTIVE_TASKSETS
    total_switches: int = 0
    successful_switches: int = 0
    perseveration_errors: int = 0
    # Task pair familiarity: (from, to) → successful switch count
    pair_familiarity: Dict[Tuple[str, str], int] = field(default_factory=dict)


# ============================================================================
# Cognitive Flexibility Engine
# ============================================================================

class CognitiveFlexibilityEngine:
    """
    Cognitive Flexibility Engine — physical model of task switching

    Modeled after the prefrontal-basal ganglia circuit's task switching mechanism:
    1. Detect task boundary (modality switch, context switch)
    2. Compute switch cost (reconfiguration + inertia + mixing)
    3. Determine whether a perseveration error occurs
    4. Training → improve switching efficiency

    API:
      - notify_task(task_id)     → Notify the current task being executed
      - attempt_switch(new_task) → Attempt a task switch, returns switch result
      - tick()                   → Called each perception cycle (inertia decay + charging)
      - get_switch_overhead()    → Returns performance penalty of the current mixed environment
      - get_flexibility_index()  → Cognitive flexibility composite metric
    """

    def __init__(self):
        self._current_task: Optional[str] = None
        self._task_sets: Dict[str, TaskSetState] = {}
        self._flexibility = FlexibilityState()
        self._switch_history: List[SwitchRecord] = []
        self._tick_count: int = 0
        self._pfc_energy: float = 1.0  # Reference external PFC energy (synced by prefrontal)

    # ------------------------------------------------------------------
    # External Sync
    # ------------------------------------------------------------------

    def sync_pfc_energy(self, energy: float):
        """Sync prefrontal cortex energy state"""
        self._pfc_energy = energy

    # ------------------------------------------------------------------
    # Task Notification
    # ------------------------------------------------------------------

    def notify_task(self, task_id: str):
        """
        Notify the engine of the currently executing task

        Does not trigger a switch decision — only updates state tracking.
        Used in passive mode (called automatically by alice_brain's see/hear).
        """
        if task_id not in self._task_sets:
            self._task_sets[task_id] = TaskSetState(task_id=task_id)
        self._task_sets[task_id].last_active_time = time.time()
        self._task_sets[task_id].activation = 1.0  # Currently active task
        self._current_task = task_id

    # ------------------------------------------------------------------
    # Active Switching
    # ------------------------------------------------------------------

    def attempt_switch(
        self,
        new_task: str,
        forced: bool = False,
    ) -> SwitchRecord:
        """
        Attempt a task switch — compute physical switch cost

        Returns:
            SwitchRecord containing switch cost components

        Physical model:
          total_cost = reconfig_delay + inertia_penalty + mixing_penalty

          reconfig_delay = τ_reconfig × (1 - familiarity_discount)
          inertia_penalty = charge(old_task) × INERTIA_WEIGHT × (1 - flexibility)
          mixing_penalty = n_active_tasksets × MIXING_RT_PENALTY
        """
        old_task = self._current_task or "none"

        # Same task → zero cost
        if old_task == new_task:
            return SwitchRecord(
                from_task=old_task,
                to_task=new_task,
                switch_cost_ms=0.0,
                inertia_penalty_ms=0.0,
                reconfig_delay_ms=0.0,
                mixing_penalty_ms=0.0,
                perseveration_error=False,
                energy_spent=0.0,
                timestamp=time.time(),
            )

        # Ensure task set exists
        if new_task not in self._task_sets:
            self._task_sets[new_task] = TaskSetState(task_id=new_task)

        # === 1. Reconfiguration delay ===
        reconfig_delay = self._flexibility.reconfig_tau

        # Familiar task pair → discount
        pair = (old_task, new_task)
        familiarity = self._flexibility.pair_familiarity.get(pair, 0)
        if familiarity > 10:
            reconfig_delay *= FAMILIAR_SWITCH_DISCOUNT

        reconfig_delay_ms = reconfig_delay * 1000

        # === 2. Inertia penalty ===
        old_state = self._task_sets.get(old_task)
        inertia_charge = old_state.charge if old_state else 0.0
        inertia_penalty = inertia_charge * INERTIA_WEIGHT * (1.0 - self._flexibility.flexibility_index)
        inertia_penalty_ms = inertia_penalty * 1000

        # === 3. Mixing cost ===
        active_tasksets = self._count_active_tasksets()
        mixing_penalty_ms = active_tasksets * MIXING_RT_PENALTY * 1000

        # === Total switch cost ===
        total_cost_ms = reconfig_delay_ms + inertia_penalty_ms + mixing_penalty_ms

        # === Energy consumption ===
        energy_cost = (total_cost_ms / 1000.0) * 0.1  # 0.1 energy consumed per second of switching

        # === Perseveration error determination ===
        perseveration_error = False
        if not forced:
            if (inertia_charge > PERSEVERATION_ERROR_THRESHOLD
                    and self._pfc_energy < PERSEVERATION_ERROR_ENERGY_THRESHOLD):
                perseveration_error = True
                self._flexibility.perseveration_errors += 1

        # Record result
        record = SwitchRecord(
            from_task=old_task,
            to_task=new_task,
            switch_cost_ms=round(total_cost_ms, 3),
            inertia_penalty_ms=round(inertia_penalty_ms, 3),
            reconfig_delay_ms=round(reconfig_delay_ms, 3),
            mixing_penalty_ms=round(mixing_penalty_ms, 3),
            perseveration_error=perseveration_error,
            energy_spent=round(energy_cost, 6),
            timestamp=time.time(),
        )

        if not perseveration_error:
            # Successful switch → update state
            self._execute_switch(old_task, new_task)
            self._train_on_switch(old_task, new_task)
        else:
            # Perseveration error → stay on old task
            self._flexibility.total_switches += 1

        self._switch_history.append(record)
        return record

    # ------------------------------------------------------------------
    # Per-Cycle Tick
    # ------------------------------------------------------------------

    def tick(self):
        """
        Called each perception cycle — update physical state

        1. Charge current task's inertia capacitor
        2. Discharge inactive tasks' inertia
        3. Decay activation of inactive task sets
        4. Slow cognitive flexibility degradation (use it or lose it)
        """
        self._tick_count += 1

        # Charge current task
        if self._current_task and self._current_task in self._task_sets:
            state = self._task_sets[self._current_task]
            state.charge = min(
                INERTIA_MAX_CHARGE,
                state.charge + INERTIA_CHARGE_RATE
            )
            state.total_ticks += 1
            state.activation = 1.0

        # Inactive task discharge + activation decay
        for task_id, state in self._task_sets.items():
            if task_id != self._current_task:
                # Exponential discharge
                state.charge *= math.exp(-INERTIA_CHARGE_RATE / INERTIA_DISCHARGE_TAU)
                # Activation decay
                state.activation *= 0.99

        # Cognitive flexibility degradation
        self._flexibility.reconfig_tau = min(
            RECONFIG_TAU_INITIAL,
            self._flexibility.reconfig_tau + RECONFIG_DECAY_RATE
        )
        self._flexibility.flexibility_index = max(
            FLEXIBILITY_INITIAL,
            self._flexibility.flexibility_index - FLEXIBILITY_DECAY
        )

    # ------------------------------------------------------------------
    # Query Interface
    # ------------------------------------------------------------------

    def get_switch_overhead(self) -> Dict[str, float]:
        """
        Get the performance overhead of the current mixed environment

        Returns:
            {
                "mixing_cost_ms": Baseline delay increase in mixed environment (ms),
                "mixing_energy_per_tick": Extra energy consumed per tick,
                "active_tasksets": Number of currently active task sets,
                "current_inertia": Inertia charge of the current task,
            }
        """
        active = self._count_active_tasksets()
        current_charge = 0.0
        if self._current_task and self._current_task in self._task_sets:
            current_charge = self._task_sets[self._current_task].charge

        return {
            "mixing_cost_ms": active * MIXING_RT_PENALTY * 1000,
            "mixing_energy_per_tick": active * MIXING_COST_PER_TASKSET,
            "active_tasksets": active,
            "current_inertia": round(current_charge, 4),
        }

    def get_flexibility_index(self) -> float:
        """
        Cognitive flexibility composite index Ω ∈ [0.5, 0.95]

        Higher → faster and less effortful switching
        """
        return self._flexibility.flexibility_index

    def get_reconfig_tau(self) -> float:
        """Get current reconfiguration time constant (seconds)"""
        return self._flexibility.reconfig_tau

    def get_current_task(self) -> Optional[str]:
        """Get current task ID"""
        return self._current_task

    def get_max_active_tasksets(self) -> int:
        """Get the maximum number of simultaneously maintainable task sets"""
        return self._flexibility.max_active_tasksets

    def get_inertia(self, task_id: str) -> float:
        """Get inertia charge for the specified task"""
        if task_id in self._task_sets:
            return self._task_sets[task_id].charge
        return 0.0

    def get_proactive_interference(self, old_task: str, new_task: str) -> float:
        """
        Proactive interference: intensity of old task interfering with new task

        = base × inertia(old) × (1 - flexibility)
        """
        inertia = self.get_inertia(old_task)
        return (PROACTIVE_INTERFERENCE_BASE
                * inertia
                * (1.0 - self._flexibility.flexibility_index))

    def get_switch_success_rate(self) -> float:
        """Get switch success rate"""
        total = self._flexibility.total_switches
        if total == 0:
            return 1.0
        return self._flexibility.successful_switches / total

    def get_recent_switch_cost(self, n: int = 10) -> float:
        """Get the average cost (ms) of the most recent n switches"""
        if not self._switch_history:
            return 0.0
        recent = self._switch_history[-n:]
        return sum(r.switch_cost_ms for r in recent) / len(recent)

    # ------------------------------------------------------------------
    # Training Level
    # ------------------------------------------------------------------

    @property
    def training_level(self) -> str:
        """Cognitive flexibility training level"""
        switches = self._flexibility.successful_switches
        if switches < 50:
            return "novice"
        elif switches < 500:
            return "intermediate"
        elif switches < 5000:
            return "advanced"
        elif switches < 50000:
            return "expert"
        else:
            return "master"

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _execute_switch(self, old_task: str, new_task: str):
        """Internal state update for executing a task switch"""
        # Old task starts discharging
        if old_task in self._task_sets:
            self._task_sets[old_task].activation = 0.7  # Residual activation

        # New task activation
        self._task_sets[new_task].activation = 1.0
        self._task_sets[new_task].last_active_time = time.time()
        self._current_task = new_task

        self._flexibility.total_switches += 1
        self._flexibility.successful_switches += 1

    def _train_on_switch(self, old_task: str, new_task: str):
        """
        Successful switch → train cognitive flexibility

        1. τ_reconfig improvement (faster reconfiguration)
        2. Ω_flexibility increase (overall flexibility increase)
        3. Task pair familiarity increase
        4. Possibly unlock more simultaneous task sets
        """
        # Reconfiguration time constant improvement
        headroom = (self._flexibility.reconfig_tau - RECONFIG_TAU_MIN) / RECONFIG_TAU_INITIAL
        self._flexibility.reconfig_tau = max(
            RECONFIG_TAU_MIN,
            self._flexibility.reconfig_tau - RECONFIG_LEARNING_RATE * headroom
        )

        # Cognitive flexibility index improvement
        flex_headroom = (FLEXIBILITY_MAX - self._flexibility.flexibility_index) / FLEXIBILITY_MAX
        self._flexibility.flexibility_index = min(
            FLEXIBILITY_MAX,
            self._flexibility.flexibility_index + FLEXIBILITY_TRAINING_RATE * flex_headroom
        )

        # Task pair familiarity
        pair = (old_task, new_task)
        self._flexibility.pair_familiarity[pair] = (
            self._flexibility.pair_familiarity.get(pair, 0) + 1
        )

        # Unlock more simultaneous task sets
        if (self._flexibility.successful_switches > 0
                and self._flexibility.successful_switches % 200 == 0):
            self._flexibility.max_active_tasksets = min(
                MAX_ACTIVE_TASKSETS_TRAINED,
                self._flexibility.max_active_tasksets + 1
            )

    def _count_active_tasksets(self) -> int:
        """
        Count the number of currently active task sets

        Active = activation value > 0.1 (still lingering in working memory)
        """
        return sum(
            1 for s in self._task_sets.values()
            if s.activation > 0.1
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Complete state (for diagnostics)"""
        return {
            "current_task": self._current_task,
            "flexibility_index": round(self._flexibility.flexibility_index, 4),
            "reconfig_tau_ms": round(self._flexibility.reconfig_tau * 1000, 2),
            "training_level": self.training_level,
            "total_switches": self._flexibility.total_switches,
            "successful_switches": self._flexibility.successful_switches,
            "perseveration_errors": self._flexibility.perseveration_errors,
            "success_rate": round(self.get_switch_success_rate(), 4),
            "max_active_tasksets": self._flexibility.max_active_tasksets,
            "active_tasksets": self._count_active_tasksets(),
            "task_sets": {
                tid: {
                    "charge": round(s.charge, 4),
                    "activation": round(s.activation, 4),
                    "total_ticks": s.total_ticks,
                }
                for tid, s in self._task_sets.items()
            },
            "recent_switch_cost_ms": round(self.get_recent_switch_cost(), 3),
        }

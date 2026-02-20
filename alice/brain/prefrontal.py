# -*- coding: utf-8 -*-
"""
Prefrontal Cortex — Executive Control & Goal Management Engine (Phase 6.1)

Physics:
  "The prefrontal cortex is the brain's CEO.
   It doesn't act directly—it decides 'what to do' and 'what not to do'.
   And 'what not to do' costs more energy than 'what to do'."

  Fuster (2008)'s three prefrontal functions:
    1. Attentional set: maintaining goal representations
    2. Inhibitory control: suppressing inappropriate actions
    3. Working memory management: manipulating transient information

  Impedance model of the prefrontal cortex:
    - Goal = target impedance state Z_goal
    - Planning = minimum Γ path from Z_now to Z_goal
    - Inhibition = actively raising channel impedance → blocking actions (costs more energy!)
    - Go/NoGo gate = channel selector

  Energy economics:
    "Impulsive people don't lack energy—their prefrontal cortex lacks fuel to raise inhibitory impedance.
     Willpower depletion = insufficient ATP in the prefrontal cortex.
     That's why you make more impulsive decisions when fatigued."

    E_inhibition = ∫ Γ_block² dt    — Energy required for inhibition
    E_available  = PFC_energy        — Available prefrontal energy
    If E_inhibition > E_available → inhibition fails → impulse breakthrough

Circuit analogy:
  Prefrontal cortex = traffic control center + energy manager
  Goal stack = priority queue
  Inhibition = actively raising channel impedance (circuit breaker)
  Task switching = impedance reconfiguration (with switching cost!)

  "You're reading, and your phone rings. Should you look?
   The prefrontal cortex says: don't look. It raises the phone channel's impedance.
   But this consumes energy. An hour later, you can't resist looking."
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Physical constants
# ============================================================================

# Goal management
GOAL_IMPEDANCE_DEFAULT = 75.0    # Ω — Default impedance for goal channel
MAX_GOAL_STACK = 5               # Maximum goal stack depth (cognitive load limit)
GOAL_DECAY_RATE = 0.02           # Goal priority decay rate/tick
GOAL_COMPLETION_THRESHOLD = 0.15 # Γ < this value → goal completed

# Inhibitory control
INHIBITION_IMPEDANCE = 200.0     # Ω — Target impedance during inhibition (very high)
INHIBITION_ENERGY_COST = 0.08    # Energy cost per inhibition
PFC_MAX_ENERGY = 1.0             # Maximum prefrontal energy
PFC_ENERGY_RECOVERY = 0.01       # Energy recovery per tick
PFC_FATIGUE_THRESHOLD = 0.2      # Energy below this → increased inhibition failure risk

# Go/NoGo gate
GO_THRESHOLD = 0.4               # Γ_action < this → Go (pass)
NOGO_THRESHOLD = 0.7             # Γ_action > this → NoGo (block)
IMPULSE_GAMMA = 0.3              # Default Γ for impulses (low → easy to pass)

# Task switching
TASK_SWITCH_COST = 0.15          # Impedance cost of task switching
PERSEVERATION_DECAY = 0.05       # Decay rate of old task residual

# Planning
MAX_PLAN_DEPTH = 10              # Maximum planning step depth
PLAN_IMPEDANCE_PENALTY = 0.1     # Impedance accumulation cost per step

# Prefrontal-amygdala interaction
PFC_AMYGDALA_REGULATION = 0.3    # Prefrontal regulation strength on amygdala (top-down inhibition)
EMOTIONAL_OVERRIDE_THRESHOLD = 0.85  # Emotional intensity above this → amygdala overrides prefrontal


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class Goal:
    """
    Goal representation — intention state maintained by the prefrontal cortex.

    Each goal is a target impedance state Z_goal,
    representing "when the system reaches this state, Γ ≈ 0 → goal satisfied".
    """
    name: str                        # Goal name
    z_goal: float                    # Target impedance (Ω)
    priority: float = 0.5            # Priority (0~1)
    sub_goals: List[str] = field(default_factory=list)  # Sub-goal name list
    progress: float = 0.0            # Completion progress (0~1)
    created_at: float = field(default_factory=time.time)
    active: bool = True              # Whether active
    parent_goal: Optional[str] = None  # Parent goal

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def effective_priority(self) -> float:
        """Effective priority (considering decay and progress)."""
        age_penalty = 1.0 / (1.0 + self.age * 0.001)
        progress_boost = 1.0 + self.progress * 0.5  # Prioritize when near completion
        return float(np.clip(self.priority * age_penalty * progress_boost, 0.0, 1.0))


@dataclass
class ActionProposal:
    """
    Action proposal — awaiting Go/NoGo gate decision.

    Each action has a Γ_action reflection coefficient,
    Γ low = action matches goal → pass
    Γ high = action conflicts with goal → block
    """
    action_name: str                 # Action name
    gamma_action: float              # Action's reflection coefficient
    source: str = "cortical"         # Source (cortical / limbic / habitual)
    urgency: float = 0.5            # Urgency
    expected_reward: float = 0.0     # Expected reward
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlanStep:
    """
    Planning step — one step on the path from Z_now to Z_goal.

    Each step's Γ represents the "cognitive effort" required to achieve that step.
    """
    step_name: str
    z_start: float                   # Starting impedance
    z_end: float                     # Target impedance
    gamma: float                     # Reflection coefficient
    estimated_energy: float          # Estimated energy cost

    @property
    def effort(self) -> float:
        """Cognitive effort = Γ²"""
        return self.gamma ** 2


@dataclass
class GoNoGoDecision:
    """
    Go/NoGo decision result.
    """
    action_name: str
    decision: str                    # "go" / "nogo" / "defer"
    gamma_action: float
    reason: str
    energy_cost: float = 0.0
    inhibited: bool = False


@dataclass
class TaskSwitchResult:
    """
    Task switch result.
    """
    from_task: str
    to_task: str
    switch_cost: float               # Switch cost (Γ)
    perseveration_error: bool         # Whether perseveration error occurred
    energy_spent: float


# ============================================================================
# Prefrontal cortex engine
# ============================================================================


class PrefrontalCortexEngine:
    """
    Prefrontal Cortex — Executive Control Center.

    Core functions:
    1. Goal Management
       - Maintain goal stack
       - Sub-goal decomposition
       - Goal conflict resolution

    2. Go/NoGo Gate (Impulse Gating)
       - Action proposal evaluation
       - Impulse inhibition
       - Energy economics

    3. Planning
       - Minimum Γ path from Z_now → Z_goal
       - Step decomposition

    4. Cognitive Control
       - Task switching
       - Perseveration inhibition
       - Emotion regulation

    Energy model:
    - PFC has limited energy (simulating glucose/ATP supply)
    - Inhibition consumes energy (willpower depletion = ego depletion)
    - Energy exhausted → impulsive behavior breaks through
    """

    def __init__(
        self,
        max_goals: int = MAX_GOAL_STACK,
        initial_energy: float = PFC_MAX_ENERGY,
    ):
        # Goal stack
        self._goals: Dict[str, Goal] = {}
        self._max_goals = max_goals

        # Energy system (willpower)
        self._energy = initial_energy
        self._max_energy = PFC_MAX_ENERGY
        self._energy_history: List[float] = [initial_energy]

        # Current task focus
        self._current_task: Optional[str] = None
        self._task_history: List[str] = []

        # Inhibition log
        self._inhibition_log: List[Dict[str, Any]] = []
        self._total_inhibitions = 0
        self._failed_inhibitions = 0

        # Go/NoGo decision buffer
        self._decision_buffer: List[GoNoGoDecision] = []

        # Planning cache
        self._current_plan: List[PlanStep] = []
        self._plan_cache: Dict[str, List[PlanStep]] = {}

        # Perseveration tracking (residual activation of previous task)
        self._perseveration_strength: float = 0.0
        self._perseveration_task: Optional[str] = None

        # Statistics
        self._total_go_decisions = 0
        self._total_nogo_decisions = 0
        self._total_defer_decisions = 0
        self._total_task_switches = 0
        self._total_plans_created = 0
        self._total_goals_completed = 0
        self._tick_count = 0

    # ------------------------------------------------------------------
    # 1. Goal management
    # ------------------------------------------------------------------

    def set_goal(
        self,
        name: str,
        z_goal: float = GOAL_IMPEDANCE_DEFAULT,
        priority: float = 0.5,
        parent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set a new goal.

        When the goal stack is full, remove the lowest priority goal.
        """
        # If already exists, update priority
        if name in self._goals:
            self._goals[name].priority = max(self._goals[name].priority, priority)
            self._goals[name].z_goal = z_goal
            return {
                "action": "updated",
                "goal": name,
                "priority": self._goals[name].priority,
            }

        # Stack full → evict lowest priority
        if len(self._goals) >= self._max_goals:
            weakest = min(
                self._goals.values(),
                key=lambda g: g.effective_priority
            )
            del self._goals[weakest.name]

        goal = Goal(
            name=name,
            z_goal=z_goal,
            priority=priority,
            parent_goal=parent,
        )
        self._goals[name] = goal

        # Set as current task (if highest priority)
        top_goal = self.get_top_goal()
        if top_goal and top_goal.name == name:
            self._current_task = name

        return {
            "action": "created",
            "goal": name,
            "z_goal": z_goal,
            "priority": priority,
            "stack_depth": len(self._goals),
        }

    def decompose_goal(
        self,
        goal_name: str,
        sub_goals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Sub-goal decomposition — break a large goal into smaller steps.

        Each sub-goal inherits the parent goal's context,
        with slightly lower priority than the parent (to prevent sub-goal preemption).
        """
        if goal_name not in self._goals:
            return {"error": f"Goal '{goal_name}' not found"}

        parent = self._goals[goal_name]
        created = []

        for i, sg in enumerate(sub_goals):
            sub_name = sg.get("name", f"{goal_name}_sub_{i}")
            sub_z = sg.get("z_goal", parent.z_goal * (1.0 + 0.1 * i))
            sub_priority = sg.get("priority", parent.priority * 0.9)

            self.set_goal(
                name=sub_name,
                z_goal=sub_z,
                priority=sub_priority,
                parent=goal_name,
            )
            parent.sub_goals.append(sub_name)
            created.append(sub_name)

        return {
            "parent_goal": goal_name,
            "sub_goals_created": created,
            "total_sub_goals": len(parent.sub_goals),
        }

    def update_goal_progress(
        self,
        goal_name: str,
        progress: float,
    ) -> Dict[str, Any]:
        """Update goal progress."""
        if goal_name not in self._goals:
            return {"error": f"Goal '{goal_name}' not found"}

        goal = self._goals[goal_name]
        goal.progress = float(np.clip(progress, 0.0, 1.0))

        completed = goal.progress >= (1.0 - GOAL_COMPLETION_THRESHOLD)

        if completed:
            goal.active = False
            self._total_goals_completed += 1

            # If there's a parent goal, update parent goal progress
            if goal.parent_goal and goal.parent_goal in self._goals:
                parent = self._goals[goal.parent_goal]
                completed_subs = sum(
                    1 for sg in parent.sub_goals
                    if sg in self._goals and not self._goals[sg].active
                )
                if parent.sub_goals:
                    parent.progress = completed_subs / len(parent.sub_goals)

        return {
            "goal": goal_name,
            "progress": goal.progress,
            "completed": completed,
            "active": goal.active,
        }

    def get_top_goal(self) -> Optional[Goal]:
        """Get the highest priority active goal."""
        active = [g for g in self._goals.values() if g.active]
        if not active:
            return None
        return max(active, key=lambda g: g.effective_priority)

    def get_goal_stack(self) -> List[Dict[str, Any]]:
        """Get the complete goal stack."""
        return sorted(
            [
                {
                    "name": g.name,
                    "z_goal": g.z_goal,
                    "priority": round(g.priority, 3),
                    "effective_priority": round(g.effective_priority, 3),
                    "progress": round(g.progress, 3),
                    "active": g.active,
                    "sub_goals": g.sub_goals,
                    "parent": g.parent_goal,
                }
                for g in self._goals.values()
            ],
            key=lambda x: x["effective_priority"],
            reverse=True,
        )

    # ------------------------------------------------------------------
    # 2. Go/NoGo gate
    # ------------------------------------------------------------------

    def evaluate_action(
        self,
        action_name: str,
        z_action: float,
        source: str = "cortical",
        urgency: float = 0.5,
        expected_reward: float = 0.0,
        emotional_override: float = 0.0,
    ) -> GoNoGoDecision:
        """
        Go/NoGo gate — evaluate action proposals.

        Physical model:
          Γ_action = |Z_action - Z_goal| / (Z_action + Z_goal)

        Decision logic:
          1. Γ_action < GO_THRESHOLD → Go (action matches goal)
          2. Γ_action > NOGO_THRESHOLD → NoGo (action conflicts with goal)
          3. In between → Defer (need more information)

        Energy check:
          If PFC energy is insufficient, NoGo may fail → impulse breakthrough
        """
        top_goal = self.get_top_goal()

        # Compute action-goal impedance reflection
        if top_goal:
            z_goal = top_goal.z_goal
            gamma_action = abs(z_action - z_goal) / (z_action + z_goal + 1e-10)
        else:
            # No goal → use default (pass all)
            gamma_action = IMPULSE_GAMMA

        # Emotional override check
        if emotional_override > EMOTIONAL_OVERRIDE_THRESHOLD:
            # Amygdala overrides prefrontal cortex → forced Go (fight-or-flight response)
            decision = GoNoGoDecision(
                action_name=action_name,
                decision="go",
                gamma_action=gamma_action,
                reason="emotional_override",
                energy_cost=0.0,
                inhibited=False,
            )
            self._total_go_decisions += 1
            self._decision_buffer.append(decision)
            return decision

        # Go/NoGo decision
        if gamma_action < GO_THRESHOLD:
            # Go — action matches goal
            decision = GoNoGoDecision(
                action_name=action_name,
                decision="go",
                gamma_action=round(gamma_action, 4),
                reason="goal_aligned",
                energy_cost=0.0,
                inhibited=False,
            )
            self._total_go_decisions += 1

        elif gamma_action > NOGO_THRESHOLD:
            # NoGo — needs inhibition
            energy_cost = INHIBITION_ENERGY_COST * (1.0 + gamma_action)

            if self._energy >= energy_cost:
                # Successful inhibition
                self._energy -= energy_cost
                self._total_inhibitions += 1
                decision = GoNoGoDecision(
                    action_name=action_name,
                    decision="nogo",
                    gamma_action=round(gamma_action, 4),
                    reason="goal_conflict",
                    energy_cost=round(energy_cost, 4),
                    inhibited=True,
                )
                self._total_nogo_decisions += 1
            else:
                # Inhibition failed! Insufficient energy → impulse breakthrough
                self._failed_inhibitions += 1
                decision = GoNoGoDecision(
                    action_name=action_name,
                    decision="go",
                    gamma_action=round(gamma_action, 4),
                    reason="inhibition_failure_energy_depleted",
                    energy_cost=round(self._energy, 4),
                    inhibited=False,
                )
                self._energy = max(0.0, self._energy - self._energy * 0.5)
                self._total_go_decisions += 1

        else:
            # Middle zone → Defer
            decision = GoNoGoDecision(
                action_name=action_name,
                decision="defer",
                gamma_action=round(gamma_action, 4),
                reason="ambiguous_need_more_info",
                energy_cost=0.0,
                inhibited=False,
            )
            self._total_defer_decisions += 1

        self._decision_buffer.append(decision)

        # Record inhibition event
        if decision.inhibited:
            self._inhibition_log.append({
                "action": action_name,
                "gamma": gamma_action,
                "energy_cost": decision.energy_cost,
                "pfc_energy_remaining": self._energy,
                "tick": self._tick_count,
            })

        return decision

    # ------------------------------------------------------------------
    # 3. Planning engine
    # ------------------------------------------------------------------

    def create_plan(
        self,
        z_current: float,
        goal_name: Optional[str] = None,
        intermediate_states: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Planning — minimum Γ path from Z_now to Z_goal.

        If intermediate states are provided, plan step by step.
        Otherwise auto-decompose into equally spaced steps.
        """
        # Determine target
        if goal_name and goal_name in self._goals:
            z_goal = self._goals[goal_name].z_goal
        else:
            top = self.get_top_goal()
            if top:
                z_goal = top.z_goal
                goal_name = top.name
            else:
                return {"error": "No goal set for planning"}

        # Generate intermediate states
        if intermediate_states is None:
            # Auto-decompose: determine step count based on impedance gap
            z_diff = abs(z_goal - z_current)
            n_steps = max(2, min(MAX_PLAN_DEPTH, int(z_diff / 10.0) + 1))
            intermediate_states = np.linspace(z_current, z_goal, n_steps + 1).tolist()
        else:
            intermediate_states = [z_current] + list(intermediate_states) + [z_goal]

        # Compute Γ for each step
        steps: List[PlanStep] = []
        total_energy = 0.0

        for i in range(len(intermediate_states) - 1):
            z_start = intermediate_states[i]
            z_end = intermediate_states[i + 1]

            gamma = abs(z_end - z_start) / (z_end + z_start + 1e-10)
            energy = gamma ** 2 + PLAN_IMPEDANCE_PENALTY

            step = PlanStep(
                step_name=f"step_{i+1}",
                z_start=round(z_start, 2),
                z_end=round(z_end, 2),
                gamma=round(gamma, 4),
                estimated_energy=round(energy, 4),
            )
            steps.append(step)
            total_energy += energy

        self._current_plan = steps
        self._plan_cache[goal_name or "default"] = steps
        self._total_plans_created += 1

        return {
            "goal": goal_name,
            "z_current": z_current,
            "z_goal": z_goal,
            "steps": [
                {
                    "name": s.step_name,
                    "z_start": s.z_start,
                    "z_end": s.z_end,
                    "gamma": s.gamma,
                    "effort": round(s.effort, 4),
                    "energy": s.estimated_energy,
                }
                for s in steps
            ],
            "total_steps": len(steps),
            "total_energy": round(total_energy, 4),
            "feasible": total_energy < self._energy * 2,  # Rough feasibility
        }

    def get_next_plan_step(self) -> Optional[Dict[str, Any]]:
        """Get the next planning step."""
        if not self._current_plan:
            return None

        step = self._current_plan[0]
        return {
            "name": step.step_name,
            "z_start": step.z_start,
            "z_end": step.z_end,
            "gamma": step.gamma,
            "effort": round(step.effort, 4),
            "remaining_steps": len(self._current_plan),
        }

    def advance_plan(self) -> Optional[Dict[str, Any]]:
        """Advance plan to the next step."""
        if not self._current_plan:
            return None

        completed = self._current_plan.pop(0)
        # Consume energy
        self._energy = max(0.0, self._energy - completed.estimated_energy * 0.5)

        return {
            "completed_step": completed.step_name,
            "gamma": completed.gamma,
            "energy_spent": round(completed.estimated_energy * 0.5, 4),
            "pfc_energy": round(self._energy, 4),
            "remaining_steps": len(self._current_plan),
            "plan_complete": len(self._current_plan) == 0,
        }

    # ------------------------------------------------------------------
    # 4. Cognitive control
    # ------------------------------------------------------------------

    def switch_task(
        self,
        new_task: str,
        forced: bool = False,
    ) -> TaskSwitchResult:
        """
        Task switching — with switching cost.

        Cognitive psychology: task switching has "mixing cost" and "switching cost"
        - Mixing cost: multi-task environment itself is slower (prefrontal load)
        - Switching cost: transition loss from A→B (perseveration + reconfiguration)

        Physical model:
          Switch cost = old channel residual impedance × new channel establishment cost
        """
        old_task = self._current_task
        perseveration_error = False

        if old_task == new_task:
            return TaskSwitchResult(
                from_task=old_task or "none",
                to_task=new_task,
                switch_cost=0.0,
                perseveration_error=False,
                energy_spent=0.0,
            )

        # Compute switching cost
        base_cost = TASK_SWITCH_COST
        perseveration_penalty = self._perseveration_strength * 0.5

        # Fatigue increases switching cost
        fatigue_factor = 1.0 + max(0, (PFC_FATIGUE_THRESHOLD - self._energy) * 3.0)
        total_cost = (base_cost + perseveration_penalty) * fatigue_factor

        # Energy consumption
        energy_cost = total_cost * INHIBITION_ENERGY_COST * 2

        # Check for perseveration error (old task residual too strong)
        if not forced and self._perseveration_strength > 0.6 and self._energy < PFC_FATIGUE_THRESHOLD:
            perseveration_error = True
            # Perseveration error: stuck on old task—cannot switch
            return TaskSwitchResult(
                from_task=old_task or "none",
                to_task=new_task,
                switch_cost=round(total_cost, 4),
                perseveration_error=True,
                energy_spent=round(energy_cost * 0.3, 4),  # Attempted but failed
            )

        # Successful switch
        self._energy = max(0.0, self._energy - energy_cost)
        self._current_task = new_task
        self._task_history.append(new_task)

        # Set perseveration strength (after switch → old task still has residual)
        self._perseveration_task = old_task
        self._perseveration_strength = 0.5  # Old task residual after switch

        self._total_task_switches += 1

        return TaskSwitchResult(
            from_task=old_task or "none",
            to_task=new_task,
            switch_cost=round(total_cost, 4),
            perseveration_error=False,
            energy_spent=round(energy_cost, 4),
        )

    def regulate_emotion(
        self,
        emotional_intensity: float,
        strategy: str = "reappraisal",
    ) -> Dict[str, Any]:
        """
        Emotion regulation — prefrontal top-down inhibition.

        Strategies:
        - reappraisal (cognitive reappraisal): reinterpret → change Z_emotion
        - suppression (expressive suppression): forcefully reduce output → high energy cost
        - distraction (attentional redirection): switch channel → moderate energy cost
        """
        regulation_strength = PFC_AMYGDALA_REGULATION

        if strategy == "reappraisal":
            # Cognitive reappraisal: change the interpretation of stimulus
            # Most effective, moderate energy cost
            energy_cost = 0.05
            reduction = regulation_strength * 0.8
            new_intensity = max(0.0, emotional_intensity * (1.0 - reduction))

        elif strategy == "suppression":
            # Expressive suppression: suppress without processing
            # High energy cost, poor long-term effect
            energy_cost = INHIBITION_ENERGY_COST * 2
            reduction = regulation_strength * 0.5
            new_intensity = max(0.0, emotional_intensity * (1.0 - reduction))

        elif strategy == "distraction":
            # Attentional redirection: don't look, don't fear
            energy_cost = 0.03
            reduction = regulation_strength * 0.6
            new_intensity = max(0.0, emotional_intensity * (1.0 - reduction))

        else:
            energy_cost = 0.0
            reduction = 0.0
            new_intensity = emotional_intensity

        # Check energy
        if self._energy >= energy_cost:
            self._energy -= energy_cost
            success = True
        else:
            # Insufficient energy → regulation fails
            new_intensity = emotional_intensity
            success = False
            reduction = 0.0

        return {
            "strategy": strategy,
            "original_intensity": round(emotional_intensity, 4),
            "regulated_intensity": round(new_intensity, 4),
            "reduction": round(reduction, 4),
            "energy_cost": round(energy_cost, 4),
            "success": success,
            "pfc_energy": round(self._energy, 4),
        }

    # ------------------------------------------------------------------
    # 5. Energy management & Tick
    # ------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """
        Prefrontal clock tick — called each processing cycle.

        1. Energy recovery
        2. Goal decay
        3. Perseveration decay
        4. Record state
        """
        self._tick_count += 1

        # Energy recovery
        old_energy = self._energy
        self._energy = min(self._max_energy, self._energy + PFC_ENERGY_RECOVERY)

        # Goal priority decay
        for goal in self._goals.values():
            if goal.active:
                goal.priority = max(0.01, goal.priority - GOAL_DECAY_RATE * 0.1)

        # Perseveration decay
        self._perseveration_strength = max(
            0.0,
            self._perseveration_strength - PERSEVERATION_DECAY
        )

        # Clean up completed/expired goals
        expired = [
            name for name, g in self._goals.items()
            if not g.active and g.age > 60.0  # Clean up after 60 seconds
        ]
        for name in expired:
            del self._goals[name]

        self._energy_history.append(self._energy)
        if len(self._energy_history) > 1000:
            self._energy_history = self._energy_history[-500:]

        return {
            "tick": self._tick_count,
            "energy": round(self._energy, 4),
            "energy_delta": round(self._energy - old_energy, 4),
            "active_goals": sum(1 for g in self._goals.values() if g.active),
            "perseveration": round(self._perseveration_strength, 4),
            "current_task": self._current_task,
        }

    def drain_energy(self, amount: float) -> float:
        """Externally drain prefrontal energy (e.g., stress, sleep deprivation)."""
        self._energy = max(0.0, self._energy - amount)
        return self._energy

    def restore_energy(self, amount: float) -> float:
        """Restore energy (e.g., rest, meditation)."""
        self._energy = min(self._max_energy, self._energy + amount)
        return self._energy

    # ------------------------------------------------------------------
    # 6. Query interface
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Complete prefrontal state."""
        return {
            "energy": round(self._energy, 4),
            "energy_pct": round(self._energy / self._max_energy * 100, 1),
            "current_task": self._current_task,
            "active_goals": sum(1 for g in self._goals.values() if g.active),
            "total_goals": len(self._goals),
            "perseveration": round(self._perseveration_strength, 4),
            "goal_stack": self.get_goal_stack()[:3],
            "can_inhibit": self._energy > PFC_FATIGUE_THRESHOLD,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Prefrontal statistics."""
        return {
            "ticks": self._tick_count,
            "energy": round(self._energy, 4),
            "total_go": self._total_go_decisions,
            "total_nogo": self._total_nogo_decisions,
            "total_defer": self._total_defer_decisions,
            "total_inhibitions": self._total_inhibitions,
            "failed_inhibitions": self._failed_inhibitions,
            "inhibition_success_rate": round(
                self._total_inhibitions / max(1, self._total_inhibitions + self._failed_inhibitions),
                3,
            ),
            "total_task_switches": self._total_task_switches,
            "total_plans": self._total_plans_created,
            "total_goals_completed": self._total_goals_completed,
            "current_task": self._current_task,
            "active_goals": sum(1 for g in self._goals.values() if g.active),
        }

    def get_energy_history(self) -> List[float]:
        """Prefrontal energy history."""
        return list(self._energy_history)

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Go/NoGo decision history."""
        return [
            {
                "action": d.action_name,
                "decision": d.decision,
                "gamma": d.gamma_action,
                "reason": d.reason,
                "inhibited": d.inhibited,
            }
            for d in self._decision_buffer[-20:]
        ]

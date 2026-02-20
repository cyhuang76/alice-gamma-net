# -*- coding: utf-8 -*-
"""
Basal Ganglia — Habit Engine & Action Selection (Phase 6.2)

Physics:
  "The basal ganglia is the brain's autopilot system.
   When you learn to ride a bicycle, you no longer 'think about' how to ride —
   that's because the action's impedance has already matched to zero.
   Habit = Γ_action ≈ 0, executes automatically, no cortical calibration needed."

  Graybiel (2008) habit loop:
    1. Striatum: Action selector — which channel opens
    2. Direct pathway (Go): Promotes action (lowers Γ)
    3. Indirect pathway (NoGo): Inhibits action (raises Γ)
    4. Hyperdirect pathway: Emergency brake (global inhibition)

  Impedance model of habit formation:
    - New action: Γ_action >> 0 → requires cortical supervision (slow, effortful)
    - Practicing: Γ_action gradually decreases → cortex disengages
    - Habitual: Γ_action ≈ 0 → automatic execution (fast, zero cost)
    - Breaking habit: Requires PFC energy injection → re-raises Γ

  Role of dopamine:
    "Dopamine is not happiness — it is prediction.
     RPE > 0 (better than expected) → strengthen pathway → Γ decreases
     RPE < 0 (worse than expected) → weaken pathway → Γ increases
     RPE = 0 (as expected) → no change → this is 'boredom'."

Circuit analogy:
  Striatum = Multiplexer
  Direct pathway = Positive feedback amplifier (Go)
  Indirect pathway = Negative feedback attenuator (NoGo)
  Hyperdirect pathway = Circuit breaker
  Dopamine = Gain control knob

  "Driving home. You don't remember the details of the route — because the basal ganglia was on autopilot.
   Suddenly someone runs out! The hyperdirect pathway activates: global emergency brake.
   Now you need cortical intervention: decide whether to detour or wait."
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

# Habit formation
INITIAL_ACTION_GAMMA = 0.8       # Initial Γ for new actions (high → requires cortex)
HABIT_THRESHOLD = 0.15           # Γ < this value → considered a habit
HABITUATION_RATE = 0.03          # Γ decrease per successful execution
DEHABITUATION_RATE = 0.1         # Γ increase when breaking a habit
MAX_HABITS = 100                 # Maximum habit memory capacity

# Pathway gains
DIRECT_PATHWAY_GAIN = 1.0       # Direct pathway (Go) gain
INDIRECT_PATHWAY_GAIN = 1.0     # Indirect pathway (NoGo) gain
HYPERDIRECT_GAIN = 2.0          # Hyperdirect pathway gain (emergency brake)

# Dopamine modulation
DA_BASELINE = 0.5               # Dopamine baseline
DA_GO_BOOST = 0.15              # RPE > 0 → Go pathway enhancement
DA_NOGO_BOOST = 0.1             # RPE < 0 → NoGo pathway enhancement
DA_DECAY = 0.02                 # Dopamine concentration decay rate

# Action selection
ACTION_SELECTION_TEMP = 0.5     # Softmax temperature (0→deterministic, ∞→random)
MIN_ACTIVATION = 0.01           # Minimum action activation value

# Dual-system
HABIT_SPEED_BONUS = 0.3         # Speed bonus for habitual actions (reaction time)
CORTICAL_OVERHEAD = 0.2         # Extra cost of cortical supervision


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class ActionChannel:
    """
    Action channel — candidate action in the striatum.

    Each action has competing Go and NoGo strengths.
    Final activation = Go_strength - NoGo_strength.
    """
    action_name: str
    go_strength: float = 0.5       # Direct pathway strength
    nogo_strength: float = 0.5     # Indirect pathway strength
    gamma_habit: float = INITIAL_ACTION_GAMMA  # Habit degree (Γ lower = more automatic)
    execution_count: int = 0       # Execution count
    success_count: int = 0         # Success count
    last_reward: float = 0.0       # Most recent reward
    last_executed: float = 0.0     # Most recent execution time

    @property
    def net_activation(self) -> float:
        """Net activation = Go - NoGo"""
        return max(MIN_ACTIVATION, self.go_strength - self.nogo_strength)

    @property
    def is_habit(self) -> bool:
        """Whether this action has become a habit."""
        return self.gamma_habit < HABIT_THRESHOLD

    @property
    def success_rate(self) -> float:
        """Success rate."""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count

    @property
    def automaticity(self) -> float:
        """Automaticity level (0=requires consciousness, 1=fully automatic)."""
        return float(np.clip(1.0 - self.gamma_habit, 0.0, 1.0))


@dataclass
class SelectionResult:
    """
    Action selection result.
    """
    selected_action: str
    activation: float                # Net activation of the selected action
    is_habitual: bool               # Whether it is a habitual action
    gamma_habit: float              # Habit Γ
    reaction_time: float            # Simulated reaction time
    pathway: str                    # Pathway used (direct / indirect / hyperdirect)
    cortical_needed: bool           # Whether cortical supervision is needed
    competing_actions: List[Dict[str, float]]  # List of competing actions


@dataclass
class HabitSnapshot:
    """
    Habit snapshot — for tracking habit formation history.
    """
    action_name: str
    gamma_history: List[float]      # Γ change over time
    reward_history: List[float]     # Reward change over time
    formed_at_count: Optional[int] = None  # When the habit was formed


# ============================================================================
# Basal ganglia engine
# ============================================================================


class BasalGangliaEngine:
    """
    Basal Ganglia — Habit Engine & Action Gating

    Core functions:
    1. Action Selection
       - Striatum multiplexer: multiple action channels compete
       - Winner-Take-All

    2. Go/NoGo Pathway Modulation
       - Direct pathway (Go): Promotes target action
       - Indirect pathway (NoGo): Inhibits non-target actions
       - Hyperdirect pathway: Global emergency brake

    3. Habit Formation
       - Repetition + reward → Γ_action gradually → 0
       - Automaticity = cortex disengagement

    4. Dopamine Modulation
       - RPE > 0 → strengthen Go
       - RPE < 0 → strengthen NoGo
       - Integrates signals from ReinforcementLearner

    Dual-system theory (Daw et al., 2005):
      - Habitual system (model-free): Fast, automatic, basal ganglia driven
      - Goal-directed system (model-based): Slow, flexible, prefrontal cortex driven
      - Arbitrator: weighs both systems based on uncertainty and reward rate
    """

    def __init__(
        self,
        temperature: float = ACTION_SELECTION_TEMP,
    ):
        # Action channels {context: {action: ActionChannel}}
        self._channels: Dict[str, Dict[str, ActionChannel]] = {}

        # Dopamine state
        self._dopamine_level = DA_BASELINE
        self._dopamine_history: List[float] = [DA_BASELINE]

        # Pathway gains
        self._direct_gain = DIRECT_PATHWAY_GAIN
        self._indirect_gain = INDIRECT_PATHWAY_GAIN
        self._hyperdirect_active = False

        # Action selection temperature
        self._temperature = temperature

        # Habit tracking
        self._habit_snapshots: Dict[str, HabitSnapshot] = {}

        # Emergency brake
        self._emergency_brake = False
        self._brake_duration = 0

        # Statistics
        self._total_selections = 0
        self._total_habitual = 0
        self._total_cortical = 0
        self._total_emergency_brakes = 0
        self._total_habits_formed = 0
        self._tick_count = 0

    # ------------------------------------------------------------------
    # 1. Action channel management
    # ------------------------------------------------------------------

    def register_action(
        self,
        context: str,
        action_name: str,
        initial_go: float = 0.5,
        initial_nogo: float = 0.5,
    ) -> ActionChannel:
        """
        Register an action channel.

        Register a candidate action in a specific context.
        """
        if context not in self._channels:
            self._channels[context] = {}

        if action_name not in self._channels[context]:
            channel = ActionChannel(
                action_name=action_name,
                go_strength=initial_go,
                nogo_strength=initial_nogo,
            )
            self._channels[context][action_name] = channel
        else:
            channel = self._channels[context][action_name]

        return channel

    def get_channel(
        self,
        context: str,
        action_name: str,
    ) -> Optional[ActionChannel]:
        """Get an action channel."""
        return self._channels.get(context, {}).get(action_name)

    # ------------------------------------------------------------------
    # 2. Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        context: str,
        available_actions: List[str],
        pfc_bias: Optional[Dict[str, float]] = None,
    ) -> SelectionResult:
        """
        Striatal action selection — Winner-Take-All.

        1. Ensure all actions have channels
        2. Compute net activation for each action (Go - NoGo)
        3. Add PFC bias (goal-directed)
        4. Softmax selection
        5. Determine if it is a habitual action

        pfc_bias: Prefrontal cortex goal bias {action: bias_value}
                  Positive value → promotes selection, Negative → promotes inhibition
        """
        # Emergency brake activated → block all
        if self._emergency_brake:
            return SelectionResult(
                selected_action="BRAKE",
                activation=0.0,
                is_habitual=False,
                gamma_habit=1.0,
                reaction_time=0.05,  # Brake reaction is very fast
                pathway="hyperdirect",
                cortical_needed=True,
                competing_actions=[],
            )

        # Ensure channels exist
        for action in available_actions:
            self.register_action(context, action)

        channels = self._channels[context]

        # Compute activations
        activations: Dict[str, float] = {}
        for action in available_actions:
            ch = channels[action]

            # Base activation = Go × direct_gain - NoGo × indirect_gain
            base = (ch.go_strength * self._direct_gain
                    - ch.nogo_strength * self._indirect_gain)

            # Dopamine modulation
            da_mod = (self._dopamine_level - DA_BASELINE) * 0.5
            base += da_mod

            # PFC bias (goal-directed system input)
            if pfc_bias and action in pfc_bias:
                base += pfc_bias[action]

            activations[action] = max(MIN_ACTIVATION, base)

        # Softmax selection
        action_names = list(activations.keys())
        values = np.array([activations[a] for a in action_names])

        # Temperature scaling
        scaled = values / max(self._temperature, 0.01)
        scaled -= np.max(scaled)  # Numerical stability
        exp_vals = np.exp(scaled)
        probs = exp_vals / (np.sum(exp_vals) + 1e-10)

        # Selection
        idx = np.random.choice(len(action_names), p=probs)
        selected = action_names[idx]
        ch = channels[selected]

        # Determine pathway and reaction time
        if ch.is_habit:
            pathway = "direct"
            cortical_needed = False
            reaction_time = 0.1 - ch.automaticity * HABIT_SPEED_BONUS
            self._total_habitual += 1
        else:
            pathway = "indirect"
            cortical_needed = True
            reaction_time = 0.3 + ch.gamma_habit * CORTICAL_OVERHEAD
            self._total_cortical += 1

        self._total_selections += 1

        # Competing actions list
        competing = [
            {"action": a, "activation": round(v, 4)}
            for a, v in sorted(activations.items(), key=lambda x: x[1], reverse=True)
        ]

        return SelectionResult(
            selected_action=selected,
            activation=round(activations[selected], 4),
            is_habitual=ch.is_habit,
            gamma_habit=round(ch.gamma_habit, 4),
            reaction_time=round(max(0.05, reaction_time), 4),
            pathway=pathway,
            cortical_needed=cortical_needed,
            competing_actions=competing,
        )

    # ------------------------------------------------------------------
    # 3. Habit formation
    # ------------------------------------------------------------------

    def update_after_action(
        self,
        context: str,
        action_name: str,
        reward: float,
        success: bool = True,
    ) -> Dict[str, Any]:
        """
        Post-action update — core of habit formation.

        Physical model:
          Success + positive reward → Γ_action ↓ → closer to habit
          Failure + negative reward → Γ_action ↑ → farther from habit
          RPE > 0 → Go strengthened
          RPE < 0 → NoGo strengthened
        """
        ch = self.get_channel(context, action_name)
        if ch is None:
            ch = self.register_action(context, action_name)

        # Compute RPE (simplified) — must be done before updating last_reward
        expected_reward = ch.last_reward if ch.execution_count > 0 else 0.0
        rpe = reward - expected_reward

        # Update execution record
        ch.execution_count += 1
        if success:
            ch.success_count += 1
        ch.last_reward = reward
        ch.last_executed = time.time()

        # Dopamine modulation
        old_da = self._dopamine_level
        self._dopamine_level = float(np.clip(
            self._dopamine_level + rpe * 0.1,
            0.0, 1.0
        ))

        # Go/NoGo pathway update
        was_habit = ch.is_habit

        if rpe > 0:
            # Positive RPE → strengthen Go, weaken NoGo
            ch.go_strength = min(1.0, ch.go_strength + DA_GO_BOOST * abs(rpe))
            ch.nogo_strength = max(0.0, ch.nogo_strength - DA_GO_BOOST * abs(rpe) * 0.5)

            # Habit formation: success + reward → Γ decreases
            if success:
                ch.gamma_habit = max(0.0, ch.gamma_habit - HABITUATION_RATE * (1.0 + reward))

        elif rpe < 0:
            # Negative RPE → strengthen NoGo, weaken Go
            ch.nogo_strength = min(1.0, ch.nogo_strength + DA_NOGO_BOOST * abs(rpe))
            ch.go_strength = max(0.0, ch.go_strength - DA_NOGO_BOOST * abs(rpe) * 0.5)

            # Habit degradation: failure → Γ increases
            ch.gamma_habit = min(1.0, ch.gamma_habit + HABITUATION_RATE * abs(rpe) * 0.5)

        else:
            # RPE = 0 → stable, mild habituation
            if success:
                ch.gamma_habit = max(0.0, ch.gamma_habit - HABITUATION_RATE * 0.3)

        # Track habit formation
        snap_key = f"{context}:{action_name}"
        if snap_key not in self._habit_snapshots:
            self._habit_snapshots[snap_key] = HabitSnapshot(
                action_name=action_name,
                gamma_history=[],
                reward_history=[],
            )
        snap = self._habit_snapshots[snap_key]
        snap.gamma_history.append(ch.gamma_habit)
        snap.reward_history.append(reward)

        # Detect habit formation event
        newly_formed = not was_habit and ch.is_habit
        if newly_formed:
            snap.formed_at_count = ch.execution_count
            self._total_habits_formed += 1

        return {
            "action": action_name,
            "context": context,
            "rpe": round(rpe, 4),
            "dopamine": round(self._dopamine_level, 4),
            "dopamine_delta": round(self._dopamine_level - old_da, 4),
            "go_strength": round(ch.go_strength, 4),
            "nogo_strength": round(ch.nogo_strength, 4),
            "gamma_habit": round(ch.gamma_habit, 4),
            "is_habit": ch.is_habit,
            "newly_formed_habit": newly_formed,
            "execution_count": ch.execution_count,
            "success_rate": round(ch.success_rate, 3),
        }

    def break_habit(
        self,
        context: str,
        action_name: str,
        pfc_energy_input: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Break a habit — requires PFC energy injection.

        "Quitting smoking is hard, not because of nicotine —
         it's because you need to raise the impedance of a channel that's already matched to zero.
         This requires sustained prefrontal cortex energy supply."
        """
        ch = self.get_channel(context, action_name)
        if ch is None:
            return {"error": f"Action '{action_name}' not found in context '{context}'"}

        if not ch.is_habit:
            return {
                "action": action_name,
                "was_habit": False,
                "message": "not a habit, nothing to break",
            }

        # Raise Γ (break habit)
        old_gamma = ch.gamma_habit
        gamma_increase = DEHABITUATION_RATE * pfc_energy_input
        ch.gamma_habit = min(1.0, ch.gamma_habit + gamma_increase)

        # Also adjust Go/NoGo
        ch.nogo_strength = min(1.0, ch.nogo_strength + 0.1 * pfc_energy_input)

        return {
            "action": action_name,
            "context": context,
            "was_habit": True,
            "still_habit": ch.is_habit,
            "gamma_before": round(old_gamma, 4),
            "gamma_after": round(ch.gamma_habit, 4),
            "pfc_energy_used": round(pfc_energy_input, 4),
        }

    # ------------------------------------------------------------------
    # 4. Hyperdirect pathway (emergency brake)
    # ------------------------------------------------------------------

    def emergency_brake(self, duration: int = 3) -> Dict[str, Any]:
        """
        Hyperdirect pathway — global emergency brake.

        Bypasses normal Go/NoGo competition,
        directly inhibits all action channels.

        Used for: sudden danger, error detection, excessive conflict
        """
        self._emergency_brake = True
        self._brake_duration = duration
        self._total_emergency_brakes += 1

        # Global inhibition: temporarily boost NoGo for all channels
        for context_channels in self._channels.values():
            for ch in context_channels.values():
                ch.nogo_strength = min(1.0, ch.nogo_strength + HYPERDIRECT_GAIN * 0.3)

        return {
            "brake_activated": True,
            "duration": duration,
            "total_channels_inhibited": sum(
                len(chs) for chs in self._channels.values()
            ),
            "emergency_brake_count": self._total_emergency_brakes,
        }

    # ------------------------------------------------------------------
    # 5. Dual-system arbitration
    # ------------------------------------------------------------------

    def arbitrate_systems(
        self,
        habit_action: str,
        goal_action: str,
        habit_gamma: float,
        goal_uncertainty: float,
        reward_rate: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Dual-system arbitration (Daw et al., 2005).

        Weighs uncertainty and reward rate:
        - Habitual system (model-free): When environment is stable, high reward rate
        - Goal-directed system (model-based): When environment changes, high uncertainty

        w_habit = (1 - uncertainty) × reward_rate / (habit_gamma + 0.1)
        w_goal  = uncertainty × (1 - reward_rate) / (1 - habit_gamma + 0.1)
        """
        # Habitual system weight
        w_habit = ((1.0 - goal_uncertainty) * reward_rate
                   / (habit_gamma + 0.1))

        # Goal-directed system weight
        w_goal = (goal_uncertainty * (1.0 - reward_rate + 0.1)
                  / (1.0 - habit_gamma + 0.1))

        # Normalize
        total = w_habit + w_goal + 1e-10
        w_habit_norm = w_habit / total
        w_goal_norm = w_goal / total

        # Select system
        if w_habit_norm > w_goal_norm:
            chosen = habit_action
            system = "habitual"
        else:
            chosen = goal_action
            system = "goal_directed"

        return {
            "chosen_action": chosen,
            "chosen_system": system,
            "habit_weight": round(w_habit_norm, 4),
            "goal_weight": round(w_goal_norm, 4),
            "habit_action": habit_action,
            "goal_action": goal_action,
            "habit_gamma": round(habit_gamma, 4),
            "goal_uncertainty": round(goal_uncertainty, 4),
        }

    # ------------------------------------------------------------------
    # 6. Tick & State
    # ------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """
        Basal ganglia clock tick.

        1. Dopamine decay
        2. Emergency brake countdown
        3. Channel decay
        """
        self._tick_count += 1

        # Dopamine decays toward baseline
        self._dopamine_level += (DA_BASELINE - self._dopamine_level) * DA_DECAY
        self._dopamine_history.append(self._dopamine_level)
        if len(self._dopamine_history) > 1000:
            self._dopamine_history = self._dopamine_history[-500:]

        # Emergency brake countdown
        if self._emergency_brake:
            self._brake_duration -= 1
            if self._brake_duration <= 0:
                self._emergency_brake = False
                # Release global inhibition
                for context_channels in self._channels.values():
                    for ch in context_channels.values():
                        ch.nogo_strength = max(
                            0.1,
                            ch.nogo_strength - HYPERDIRECT_GAIN * 0.3
                        )

        return {
            "tick": self._tick_count,
            "dopamine": round(self._dopamine_level, 4),
            "emergency_brake": self._emergency_brake,
            "brake_remaining": self._brake_duration if self._emergency_brake else 0,
            "total_habits": self._total_habits_formed,
        }

    def get_state(self) -> Dict[str, Any]:
        """Full basal ganglia state."""
        total_channels = sum(len(chs) for chs in self._channels.values())
        total_habits = sum(
            1 for chs in self._channels.values()
            for ch in chs.values()
            if ch.is_habit
        )
        return {
            "dopamine": round(self._dopamine_level, 4),
            "emergency_brake": self._emergency_brake,
            "total_channels": total_channels,
            "total_habits": total_habits,
            "contexts": list(self._channels.keys()),
            "direct_gain": self._direct_gain,
            "indirect_gain": self._indirect_gain,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Basal ganglia statistics."""
        total_channels = sum(len(chs) for chs in self._channels.values())
        total_habits = sum(
            1 for chs in self._channels.values()
            for ch in chs.values()
            if ch.is_habit
        )
        return {
            "ticks": self._tick_count,
            "dopamine": round(self._dopamine_level, 4),
            "total_selections": self._total_selections,
            "total_habitual": self._total_habitual,
            "total_cortical": self._total_cortical,
            "habit_ratio": round(
                self._total_habitual / max(1, self._total_selections), 3
            ),
            "total_habits_formed": self._total_habits_formed,
            "total_emergency_brakes": self._total_emergency_brakes,
            "total_channels": total_channels,
            "active_habits": total_habits,
        }

    def get_habit_trajectory(
        self,
        context: str,
        action_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the habit formation trajectory for a specific action."""
        snap_key = f"{context}:{action_name}"
        snap = self._habit_snapshots.get(snap_key)
        if snap is None:
            return None

        return {
            "action": action_name,
            "gamma_history": [round(g, 4) for g in snap.gamma_history],
            "reward_history": [round(r, 4) for r in snap.reward_history],
            "formed_at_count": snap.formed_at_count,
            "current_gamma": snap.gamma_history[-1] if snap.gamma_history else None,
            "is_habit": snap.gamma_history[-1] < HABIT_THRESHOLD if snap.gamma_history else False,
        }

    def get_dopamine_history(self) -> List[float]:
        """Dopamine concentration history."""
        return list(self._dopamine_history)

    def get_all_habits(self) -> List[Dict[str, Any]]:
        """Get all formed habits."""
        habits = []
        for context, channels in self._channels.items():
            for name, ch in channels.items():
                if ch.is_habit:
                    habits.append({
                        "context": context,
                        "action": name,
                        "gamma": round(ch.gamma_habit, 4),
                        "automaticity": round(ch.automaticity, 3),
                        "execution_count": ch.execution_count,
                        "success_rate": round(ch.success_rate, 3),
                    })
        return habits

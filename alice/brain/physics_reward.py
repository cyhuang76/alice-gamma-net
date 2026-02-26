# -*- coding: utf-8 -*-
"""
Physics Reward Engine — Impedance Matching Replaces Q-learning
Physics Reward Engine — Impedance-Based Reward Prediction Error

Physics:
  "Dopamine is not happiness — it is the electrochemical encoding of prediction error.
   RPE > 0 (better than expected) → impedance matching improves → pathway reinforced
   RPE < 0 (worse than expected) → impedance mismatch worsens → pathway suppressed
   RPE = 0 (as expected) → already matched → no adjustment needed = boredom."

  Problems with traditional Q-learning:
    Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
    - Statistical regression, no physical meaning
    - Discrete state space, cannot continuously generalize
    - Incompatible with the brain's other impedance mechanisms

  Physics reward model:
    Each "state→action" channel has impedance Z_channel
    - Z_channel high → uncertain → requires cortical supervision (slow)
    - Z_channel low → well matched → automatic execution (fast)

    Reward prediction = channel's expected output power
    R_expected = P_transmitted = (1 - Γ²) · P_input
    R_actual = actual reward (environmental feedback)

    RPE = R_actual - R_expected
    RPE > 0 → Z lowered (Hebbian reinforcement, Γ↓)
    RPE < 0 → Z raised (Anti-Hebbian, Γ↑)

    Dopamine level = DA_baseline + k · RPE (finite duration, exponential decay)

  Core advantages:
    1. Unified with the system-wide Γ language (pain=high Γ, learning=Γ↓, reward=Γ→0)
    2. Natural generalization in continuous space (channels with similar impedance share adjustments)
    3. Energy conservation: reward learning obeys the physical constraints of impedance matching

Circuit analogy:
  Ventral Tegmental Area (VTA) = current mirror — duplicates/amplifies RPE signal
  Nucleus Accumbens (NAcc) = gain control amplifier — dopamine modulates gain
  Striatum = impedance matching network — one Z per channel
  Cortisol = bias circuit — changes global sensitivity
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

# Impedance
Z_INITIAL = 100.0               # Initial impedance for new channel Ω (high → uncertain)
Z_SOURCE = 75.0                 # Reference source impedance Ω
Z_MIN = Z_SOURCE                # Lower limit = source impedance, prevents secondary mismatch from over-learning
                                # Physics: when Z₀ < Z_S, Γ rises again (short-circuit end reflection)
Z_MAX = 200.0                   # Maximum impedance (complete inhibition → open-circuit end reflection)

# Hebbian learning rate
REWARD_HEBBIAN_RATE = 0.08      # Impedance decrease rate when RPE > 0
PUNISH_ANTI_HEBBIAN_RATE = 0.05  # Impedance increase rate when RPE < 0
NEUTRAL_DECAY_RATE = 0.001      # Weak forgetting when RPE ≈ 0

# Dopamine
DA_BASELINE = 0.5               # Dopamine baseline
DA_RPE_GAIN = 0.15              # RPE → dopamine conversion coefficient
DA_DECAY_RATE = 0.05            # Dopamine decay-to-baseline rate
DA_CEILING = 1.0                # Dopamine ceiling
DA_FLOOR = 0.05                 # Dopamine floor

# Exploration strategy
EXPLORATION_Z_THRESHOLD = 60.0  # Z > this value → still exploring (uncertain)
BOLTZMANN_TEMPERATURE = 0.3     # Boltzmann selection temperature
BOLTZMANN_TEMP_DECAY = 0.9995   # Temperature decay

# Generalization
GENERALIZATION_RADIUS = 0.3     # Impedance sharing adjustment range for similar states
GENERALIZATION_DECAY = 0.5      # Generalization strength decays with distance

# Intrinsic reward physics
NOVELTY_REWARD_SCALE = 0.2      # Intrinsic reward for first success in new channel
CURIOSITY_BONUS = 0.1           # Curiosity reward for exploring unseen states

# Experience replay
MAX_EXPERIENCE_BUFFER = 5000    # Experience buffer size
REPLAY_BATCH_SIZE = 16          # Replay batch
REPLAY_LEARNING_DISCOUNT = 0.5  # Replay learning efficiency (lower than real-time learning)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class RewardChannel:
    """
    Reward channel — impedance parameters for a (state, action) pair

    Z high = uncertain/unlearned → requires cortical supervision
    Z low = learned/confident → automatic execution
    Γ = (Z - Z_source) / (Z + Z_source)
    """
    state: str
    action: str
    impedance: float = Z_INITIAL          # Channel impedance Ω
    expected_reward: float = 0.0          # Expected reward based on impedance
    cumulative_reward: float = 0.0        # Cumulative reward
    visit_count: int = 0                  # Visit count
    last_reward: float = 0.0             # Most recent actual reward
    last_rpe: float = 0.0               # Most recent RPE
    created_at: float = field(default_factory=time.time)

    @property
    def gamma(self) -> float:
        """Reflection coefficient Γ"""
        return abs(self.impedance - Z_SOURCE) / (self.impedance + Z_SOURCE + 1e-10)

    @property
    def transmission(self) -> float:
        """Transmission efficiency = 1 - Γ²"""
        g = self.gamma
        return 1.0 - g * g

    @property
    def confidence(self) -> float:
        """Confidence = transmission efficiency (better impedance matching → more confident)"""
        return self.transmission

    @property
    def is_learned(self) -> bool:
        """Whether learned (impedance dropped below exploration threshold)"""
        return self.impedance < EXPLORATION_Z_THRESHOLD


@dataclass
class RewardExperience:
    """Single reward experience"""
    state: str
    action: str
    reward: float
    next_state: str
    rpe: float
    dopamine: float
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Physics Reward Engine
# ============================================================================


class PhysicsRewardEngine:
    """
    Physics Reward Engine — Impedance Matching Replaces Q-table

    Core mechanism:
    1. Each (state, action) = one channel, with channel impedance Z
    2. Reward prediction R_expected = transmission_efficiency × historical average
    3. RPE = R_actual - R_expected
    4. Z adjusted via Hebbian learning based on RPE (RPE>0 → Z↓, RPE<0 → Z↑)
    5. Action selection = Boltzmann selection, probability ∝ exp(transmission / T)
    6. Dopamine = DA_baseline + k · RPE (exponential decay)

    Replaces ReinforcementLearner's Q-table + TD(0), preserving the same public API.
    """

    def __init__(
        self,
        learning_rate: float = REWARD_HEBBIAN_RATE,
        temperature: float = BOLTZMANN_TEMPERATURE,
    ):
        # Channel table {(state, action): RewardChannel}
        self._channels: Dict[Tuple[str, str], RewardChannel] = {}

        # Dopamine
        self._dopamine: float = DA_BASELINE
        self._dopamine_history: List[float] = [DA_BASELINE]

        # Exploration temperature
        self._temperature: float = temperature
        self._initial_temperature: float = temperature

        # Learning rate
        self._lr: float = learning_rate

        # Experience buffer
        self._experience_buffer: List[RewardExperience] = []

        # Statistics
        self._total_updates: int = 0
        self._total_explorations: int = 0
        self._total_exploitations: int = 0
        self._total_positive_rpe: int = 0
        self._total_negative_rpe: int = 0

    # ------------------------------------------------------------------
    # Channel Management
    # ------------------------------------------------------------------

    def _get_or_create_channel(self, state: str, action: str) -> RewardChannel:
        """Get or create channel"""
        key = (state, action)
        if key not in self._channels:
            self._channels[key] = RewardChannel(state=state, action=action)
        return self._channels[key]

    def get_channel_gamma(self, state: str, action: str) -> float:
        """Get channel's Γ"""
        ch = self._channels.get((state, action))
        return ch.gamma if ch else (Z_INITIAL - Z_SOURCE) / (Z_INITIAL + Z_SOURCE)

    # ------------------------------------------------------------------
    # Action Selection (replaces ε-greedy)
    # ------------------------------------------------------------------

    def choose_action(
        self, state: str, available_actions: List[str],
    ) -> Tuple[str, bool]:
        """
        Boltzmann selection — based on transmission efficiency

        Probability P(a) ∝ exp(transmission(s,a) / T)
        T high → uniform exploration, T low → exploit highest confidence channel

        Returns: (action, is_exploration)
        """
        if not available_actions:
            return "", False

        # Compute transmission efficiency for each action
        transmissions = []
        for a in available_actions:
            ch = self._get_or_create_channel(state, a)
            transmissions.append(ch.transmission)

        values = np.array(transmissions)
        scaled = values / max(self._temperature, 0.01)
        scaled -= np.max(scaled)  # Numerical stability
        exp_vals = np.exp(scaled)
        probs = exp_vals / (np.sum(exp_vals) + 1e-10)

        idx = int(np.random.choice(len(available_actions), p=probs))
        selected = available_actions[idx]

        # Determine if it's exploration
        ch = self._get_or_create_channel(state, selected)
        is_exploration = not ch.is_learned

        if is_exploration:
            self._total_explorations += 1
        else:
            self._total_exploitations += 1

        return selected, is_exploration

    # ------------------------------------------------------------------
    # Learning Update (replaces TD(0))
    # ------------------------------------------------------------------

    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_available_actions: Optional[List[str]] = None,
    ) -> float:
        """
        Impedance matching learning — replaces TD(0) Q update

        1. Get channel
        2. Compute R_expected = transmission × running_avg_reward
        3. RPE = R_actual - R_expected
        4. Z adjustment (Hebbian/Anti-Hebbian)
        5. Update dopamine
        6. Store experience

        Returns: dopamine_signal (RPE)
        """
        ch = self._get_or_create_channel(state, action)
        ch.visit_count += 1

        # === 1. Reward prediction ===
        if ch.visit_count > 1:
            running_avg = ch.cumulative_reward / (ch.visit_count - 1)
            r_expected = ch.transmission * max(running_avg, 0.0) + (1.0 - ch.transmission) * ch.last_reward * 0.5
        else:
            r_expected = 0.0

        ch.expected_reward = r_expected

        # === 2. RPE ===
        rpe = reward - r_expected

        # === 3. Impedance adjustment ===
        if rpe > 0:
            # Positive RPE → Hebbian reinforcement → Z decreases (matching improves)
            delta_z = -self._lr * abs(rpe) * ch.impedance * 0.1
            ch.impedance = max(Z_MIN, ch.impedance + delta_z)
            self._total_positive_rpe += 1
        elif rpe < 0:
            # Negative RPE → Anti-Hebbian → Z increases (matching degrades)
            delta_z = PUNISH_ANTI_HEBBIAN_RATE * abs(rpe) * (Z_MAX - ch.impedance) * 0.1
            ch.impedance = min(Z_MAX, ch.impedance + delta_z)
            self._total_negative_rpe += 1
        else:
            # RPE ≈ 0 → weak forgetting
            ch.impedance += NEUTRAL_DECAY_RATE * (Z_INITIAL - ch.impedance) * 0.01

        # === 4. Generalization: similar state channels also fine-tuned ===
        self._generalize(state, action, rpe)

        # === 5. Dopamine update ===
        da_delta = DA_RPE_GAIN * rpe
        self._dopamine = float(np.clip(
            self._dopamine + da_delta,
            DA_FLOOR, DA_CEILING,
        ))
        self._dopamine_history.append(self._dopamine)
        if len(self._dopamine_history) > 1000:
            self._dopamine_history = self._dopamine_history[-500:]

        # Dopamine decays back to baseline
        self._dopamine += (DA_BASELINE - self._dopamine) * DA_DECAY_RATE

        # === 6. Record ===
        ch.last_reward = reward
        ch.last_rpe = rpe
        ch.cumulative_reward += reward

        # Intrinsic reward: first successful exploration
        intrinsic_reward = 0.0
        if ch.visit_count == 1 and reward > 0:
            intrinsic_reward = NOVELTY_REWARD_SCALE
        elif ch.visit_count == 1:
            intrinsic_reward = CURIOSITY_BONUS

        # Store experience
        exp = RewardExperience(
            state=state,
            action=action,
            reward=reward + intrinsic_reward,
            next_state=next_state,
            rpe=rpe,
            dopamine=self._dopamine,
        )
        self._experience_buffer.append(exp)
        if len(self._experience_buffer) > MAX_EXPERIENCE_BUFFER:
            self._experience_buffer.pop(0)

        self._total_updates += 1

        # Temperature decay
        self._temperature = max(0.05, self._temperature * BOLTZMANN_TEMP_DECAY)

        return rpe  # Return RPE as dopamine signal (compatible with ReinforcementLearner API)

    # ------------------------------------------------------------------
    def _generalize(self, state: str, action: str, rpe: float):
        """Channel generalization: same action in other states is also fine-tuned"""
        for (s, a), ch in self._channels.items():
            if a == action and s != state:
                # Same action, different state → weak generalization
                gen_strength = GENERALIZATION_DECAY
                if rpe > 0:
                    ch.impedance = max(
                        Z_MIN,
                        ch.impedance - self._lr * gen_strength * abs(rpe) * ch.impedance * 0.02,
                    )
                elif rpe < 0:
                    ch.impedance = min(
                        Z_MAX,
                        ch.impedance + PUNISH_ANTI_HEBBIAN_RATE * gen_strength * abs(rpe) * 0.02,
                    )

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def replay(self, batch_size: int = REPLAY_BATCH_SIZE) -> float:
        """
        Experience replay — offline impedance reorganization (similar to sleep consolidation)

        Randomly sample past experiences and re-adjust Z with a discounted learning rate.
        """
        if len(self._experience_buffer) < batch_size:
            return 0.0

        indices = np.random.choice(len(self._experience_buffer), batch_size, replace=False)
        total_adjustment = 0.0

        for idx in indices:
            exp = self._experience_buffer[idx]
            ch = self._get_or_create_channel(exp.state, exp.action)

            if exp.rpe > 0:
                delta = -self._lr * REPLAY_LEARNING_DISCOUNT * abs(exp.rpe) * ch.impedance * 0.05
                ch.impedance = max(Z_MIN, ch.impedance + delta)
            elif exp.rpe < 0:
                delta = PUNISH_ANTI_HEBBIAN_RATE * REPLAY_LEARNING_DISCOUNT * abs(exp.rpe) * 0.05
                ch.impedance = min(Z_MAX, ch.impedance + delta)

            total_adjustment += abs(exp.rpe)

        return total_adjustment / batch_size

    # ------------------------------------------------------------------
    # Dopamine Interface (for basal_ganglia)
    # ------------------------------------------------------------------

    def get_dopamine(self) -> float:
        """Current dopamine level"""
        return self._dopamine

    def get_dopamine_history(self) -> List[float]:
        """Dopamine history"""
        return list(self._dopamine_history)

    def inject_dopamine(self, amount: float):
        """Externally inject dopamine (e.g., reward signal when hand reaches target)"""
        self._dopamine = float(np.clip(self._dopamine + amount, DA_FLOOR, DA_CEILING))

    # ------------------------------------------------------------------
    # Q-value Compatible Interface (gradual replacement)
    # ------------------------------------------------------------------

    def get_q_value(self, state: str, action: str) -> float:
        """
        Equivalent Q-value = transmission × cumulative_avg_reward

        Maintains API compatibility with ReinforcementLearner
        """
        ch = self._channels.get((state, action))
        if ch is None or ch.visit_count == 0:
            return 0.0
        avg = ch.cumulative_reward / ch.visit_count
        return ch.transmission * avg

    # ------------------------------------------------------------------
    # State/Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Statistics"""
        total_channels = len(self._channels)
        learned_channels = sum(1 for ch in self._channels.values() if ch.is_learned)
        avg_gamma = (
            np.mean([ch.gamma for ch in self._channels.values()])
            if self._channels else 0.5
        )
        return {
            "total_updates": self._total_updates,
            "total_channels": total_channels,
            "learned_channels": learned_channels,
            "learn_ratio": round(learned_channels / max(1, total_channels), 3),
            "dopamine": round(self._dopamine, 4),
            "temperature": round(self._temperature, 4),
            "avg_gamma": round(float(avg_gamma), 4),
            "buffer_size": len(self._experience_buffer),
            "explorations": self._total_explorations,
            "exploitations": self._total_exploitations,
            "exploration_rate": round(
                self._total_explorations / max(1, self._total_explorations + self._total_exploitations),
                3,
            ),
            "positive_rpe_count": self._total_positive_rpe,
            "negative_rpe_count": self._total_negative_rpe,
            "avg_dopamine": round(
                float(np.mean(self._dopamine_history[-100:])) if self._dopamine_history else 0.5,
                4,
            ),
        }

    def get_state(self) -> Dict[str, Any]:
        """Complete state"""
        return {
            **self.get_stats(),
            "channels": {
                f"{s}:{a}": {
                    "impedance": round(ch.impedance, 2),
                    "gamma": round(ch.gamma, 4),
                    "transmission": round(ch.transmission, 4),
                    "visits": ch.visit_count,
                    "avg_reward": round(ch.cumulative_reward / max(1, ch.visit_count), 4),
                    "is_learned": ch.is_learned,
                }
                for (s, a), ch in list(self._channels.items())[:50]  # Limit output
            },
        }

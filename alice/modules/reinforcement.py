# -*- coding: utf-8 -*-
"""
Reinforcement Learning Module — Dopamine-driven TD Learning

Simulates the midbrain dopamine system's Reward Prediction Error (RPE):
- TD(0) temporal difference learning
- Dopamine baseline adaptation
- Reward shaping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Experience:
    """Single experience (state, action, reward, next_state)"""

    state: str
    action: str
    reward: float
    next_state: str
    dopamine_signal: float = 0.0  # Dopamine signal = RPE


class ReinforcementLearner:
    """
    Dopamine-driven Reinforcement Learner

    Core mechanisms:
    - Q-Table + TD(0) update
    - Dopamine baseline: dopamine_signal = reward - expected_reward
    - ε-greedy exploration strategy + decay
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
        dopamine_baseline: float = 0.5,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.dopamine_baseline = dopamine_baseline
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-Table: {state: {action: q_value}}
        self.q_table: Dict[str, Dict[str, float]] = {}

        # Experience replay
        self.experience_buffer: List[Experience] = []
        self.max_buffer_size = 10000

        # Dopamine tracking
        self.dopamine_history: List[float] = []
        self._running_avg_reward = 0.0

        # Statistics
        self.total_updates = 0
        self.total_explorations = 0
        self.total_exploitations = 0

    # ------------------------------------------------------------------
    def get_q_value(self, state: str, action: str) -> float:
        return self.q_table.get(state, {}).get(action, 0.0)

    # ------------------------------------------------------------------
    def choose_action(self, state: str, available_actions: List[str]) -> Tuple[str, bool]:
        """
        ε-greedy strategy for action selection

        Returns: (action, is_exploration)
        """
        if not available_actions:
            return "", False

        if np.random.random() < self.epsilon:
            # Exploration
            self.total_explorations += 1
            return np.random.choice(available_actions), True

        # Exploit: select action with highest Q value
        self.total_exploitations += 1
        q_vals = [(a, self.get_q_value(state, a)) for a in available_actions]
        best_action = max(q_vals, key=lambda x: x[1])[0]
        return best_action, False

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
        TD(0) update + dopamine signal computation

        Returns: dopamine_signal (RPE)
        """
        # Compute dopamine signal = actual reward - expected reward
        expected = self.get_q_value(state, action)
        dopamine_signal = reward - expected

        # Update dopamine baseline (moving average)
        self._running_avg_reward = 0.95 * self._running_avg_reward + 0.05 * reward

        # TD(0) target
        if next_available_actions:
            max_next_q = max(
                (self.get_q_value(next_state, a) for a in next_available_actions), default=0.0
            )
        else:
            max_next_q = 0.0

        td_target = reward + self.gamma * max_next_q
        td_error = td_target - expected

        # Q update
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = expected + self.lr * td_error

        # Record experience
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            dopamine_signal=dopamine_signal,
        )
        self.experience_buffer.append(exp)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

        self.dopamine_history.append(dopamine_signal)
        self.total_updates += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return dopamine_signal

    # ------------------------------------------------------------------
    def replay(self, batch_size: int = 32) -> float:
        """Experience replay: randomly sample past experiences for re-learning"""
        if len(self.experience_buffer) < batch_size:
            return 0.0

        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        total_td = 0.0
        for idx in indices:
            exp = self.experience_buffer[idx]
            expected = self.get_q_value(exp.state, exp.action)
            td_error = exp.reward - expected

            if exp.state not in self.q_table:
                self.q_table[exp.state] = {}
            self.q_table[exp.state][exp.action] = expected + self.lr * 0.5 * td_error
            total_td += abs(td_error)

        return total_td / batch_size

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        recent_dopamine = self.dopamine_history[-100:] if self.dopamine_history else [0.0]
        return {
            "total_updates": self.total_updates,
            "states_known": len(self.q_table),
            "buffer_size": len(self.experience_buffer),
            "epsilon": round(self.epsilon, 4),
            "explorations": self.total_explorations,
            "exploitations": self.total_exploitations,
            "exploration_rate": round(
                self.total_explorations / max(1, self.total_explorations + self.total_exploitations),
                3,
            ),
            "avg_dopamine": round(float(np.mean(recent_dopamine)), 4),
            "running_avg_reward": round(self._running_avg_reward, 4),
        }

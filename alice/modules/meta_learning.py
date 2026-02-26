# -*- coding: utf-8 -*-
"""
Meta-Learning Module — Learning to Learn

Core concepts:
- Maintain a set of learning strategies (strategy pool)
- Dynamically adjust weights based on each strategy's historical performance
- Adaptive hyperparameters: learning rate, decay rate, exploration rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LearningStrategy:
    """Learning strategy"""

    name: str
    params: Dict[str, float]
    performance_history: List[float] = field(default_factory=list)
    selection_count: int = 0
    total_reward: float = 0.0

    @property
    def avg_performance(self) -> float:
        if not self.performance_history:
            return 0.0
        return float(np.mean(self.performance_history[-50:]))

    @property
    def confidence(self) -> float:
        """More selections = higher confidence"""
        return min(1.0, self.selection_count / 20.0)


class MetaLearner:
    """
    Meta-Learner

    Capabilities:
    1. Maintain a strategy pool (each strategy = a set of hyperparameters)
    2. Softmax weighted strategy selection
    3. Update strategy weights based on actual performance feedback
    4. Strategy elimination and mutation (evolutionary search)
    """

    def __init__(
        self,
        adaptation_rate: float = 0.001,
        strategy_pool_size: int = 10,
        temperature: float = 1.0,
    ):
        self.adaptation_rate = adaptation_rate
        self.pool_size = strategy_pool_size
        self.temperature = temperature

        # Strategy pool
        self.strategies: List[LearningStrategy] = []
        self._initialize_strategies()

        # Current strategy
        self.current_strategy: Optional[LearningStrategy] = None
        self.current_strategy_idx = 0

        # Statistics
        self.total_adaptations = 0
        self.strategy_switches = 0

    # ------------------------------------------------------------------
    def _initialize_strategies(self):
        """Initialize a diversified strategy pool"""
        presets = [
            ("conservative", {"learning_rate": 0.001, "decay": 0.01, "exploration": 0.1}),
            ("balanced", {"learning_rate": 0.01, "decay": 0.05, "exploration": 0.2}),
            ("aggressive", {"learning_rate": 0.1, "decay": 0.1, "exploration": 0.4}),
            ("fast_decay", {"learning_rate": 0.05, "decay": 0.2, "exploration": 0.15}),
            ("high_explore", {"learning_rate": 0.01, "decay": 0.05, "exploration": 0.5}),
        ]

        for name, params in presets:
            self.strategies.append(LearningStrategy(name=name, params=params))

        # Fill up to pool_size: random mutations
        while len(self.strategies) < self.pool_size:
            base = np.random.choice(self.strategies[:len(presets)])
            mutated_params = {
                k: np.clip(v * (1 + np.random.randn() * 0.3), 0.0001, 1.0)
                for k, v in base.params.items()
            }
            self.strategies.append(
                LearningStrategy(
                    name=f"mutant_{len(self.strategies)}",
                    params=mutated_params,
                )
            )

        self.current_strategy = self.strategies[0]

    # ------------------------------------------------------------------
    def select_strategy(self) -> LearningStrategy:
        """Softmax weighted strategy selection"""
        performances = np.array(
            [max(0.01, s.avg_performance + 0.1) for s in self.strategies]
        )
        logits = performances / max(0.01, self.temperature)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        idx = int(np.random.choice(len(self.strategies), p=probs))

        if self.current_strategy_idx != idx:
            self.strategy_switches += 1

        self.current_strategy_idx = idx
        self.current_strategy = self.strategies[idx]
        self.current_strategy.selection_count += 1

        return self.current_strategy

    # ------------------------------------------------------------------
    def report_performance(self, performance: float):
        """Report current strategy's performance"""
        if self.current_strategy is None:
            return

        self.current_strategy.performance_history.append(performance)
        self.current_strategy.total_reward += performance
        self.total_adaptations += 1

        # Temperature annealing
        self.temperature = max(0.1, self.temperature * 0.999)

        # Periodic elimination and mutation
        if self.total_adaptations % 50 == 0:
            self._evolve_strategies()

    # ------------------------------------------------------------------
    def _evolve_strategies(self):
        """Evolution: eliminate worst strategy, mutate best strategy to produce new one"""
        if len(self.strategies) < 3:
            return

        sorted_strats = sorted(self.strategies, key=lambda s: s.avg_performance, reverse=True)

        # Eliminate the worst
        worst = sorted_strats[-1]
        best = sorted_strats[0]

        # Mutate best strategy to produce new one
        mutated_params = {
            k: float(np.clip(v * (1 + np.random.randn() * 0.2), 0.0001, 1.0))
            for k, v in best.params.items()
        }
        new_strategy = LearningStrategy(
            name=f"evolved_{self.total_adaptations}",
            params=mutated_params,
        )

        # Replace
        idx = self.strategies.index(worst)
        self.strategies[idx] = new_strategy

    # ------------------------------------------------------------------
    def get_current_params(self) -> Dict[str, float]:
        """Get current strategy's hyperparameters"""
        if self.current_strategy is None:
            self.select_strategy()
        return dict(self.current_strategy.params)

    # ------------------------------------------------------------------
    def adapt_parameter(self, param_name: str, gradient: float):
        """Adaptively adjust a single parameter"""
        if self.current_strategy and param_name in self.current_strategy.params:
            old = self.current_strategy.params[param_name]
            new = float(np.clip(old - self.adaptation_rate * gradient, 0.0001, 1.0))
            self.current_strategy.params[param_name] = new

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_adaptations": self.total_adaptations,
            "strategy_switches": self.strategy_switches,
            "temperature": round(self.temperature, 4),
            "pool_size": len(self.strategies),
            "current_strategy": (
                {
                    "name": self.current_strategy.name,
                    "params": {k: round(v, 5) for k, v in self.current_strategy.params.items()},
                    "avg_performance": round(self.current_strategy.avg_performance, 4),
                    "selections": self.current_strategy.selection_count,
                }
                if self.current_strategy
                else None
            ),
            "top_strategies": [
                {
                    "name": s.name,
                    "avg_perf": round(s.avg_performance, 4),
                    "selections": s.selection_count,
                }
                for s in sorted(self.strategies, key=lambda x: -x.avg_performance)[:3]
            ],
        }

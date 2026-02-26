# -*- coding: utf-8 -*-
"""
Causal Reasoning Engine

Implements the three levels of Pearl's causal ladder:
1. Association: P(Y|X) — What is observed?
2. Intervention: P(Y|do(X)) — What if I do X?
3. Counterfactual: P(Y_x|X', Y') — What if X had been done instead?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CausalLink:
    """Causal link"""

    cause: str
    effect: str
    strength: float = 0.5       # Causal strength (0~1)
    observations: int = 0       # Observation count
    confidence: float = 0.0     # Statistical confidence


@dataclass
class CausalEvent:
    """Causal event record"""

    variables: Dict[str, float]
    timestamp: float = 0.0


class CausalReasoner:
    """
    Causal Reasoner

    Core capabilities:
    - Causal graph learning (discover causal relations from observations)
    - Interventional reasoning (simplified do-calculus)
    - Counterfactual reasoning (what if...?)
    """

    def __init__(
        self,
        max_causal_depth: int = 5,
        min_observations: int = 10,
        significance_threshold: float = 0.3,
    ):
        self.max_depth = max_causal_depth
        self.min_obs = min_observations
        self.sig_threshold = significance_threshold

        # Causal graph: {cause -> {effect -> CausalLink}}
        self.causal_graph: Dict[str, Dict[str, CausalLink]] = {}

        # Observation history
        self.observations: List[CausalEvent] = []
        self.max_observations = 10000

        # Statistics
        self.total_inferences = 0
        self.total_interventions = 0
        self.total_counterfactuals = 0

    # ------------------------------------------------------------------
    def observe(self, variables: Dict[str, float]):
        """Observe a set of covariates"""
        event = CausalEvent(variables=variables, timestamp=len(self.observations))
        self.observations.append(event)
        if len(self.observations) > self.max_observations:
            self.observations.pop(0)

        # Update causal graph (simplified: high correlation → possible causation)
        self._update_causal_links(variables)

    # ------------------------------------------------------------------
    def _update_causal_links(self, variables: Dict[str, float]):
        """Update causal links from covariate observations"""
        keys = list(variables.keys())
        for i, cause in enumerate(keys):
            for j, effect in enumerate(keys):
                if i == j:
                    continue

                if cause not in self.causal_graph:
                    self.causal_graph[cause] = {}
                if effect not in self.causal_graph[cause]:
                    self.causal_graph[cause][effect] = CausalLink(cause=cause, effect=effect)

                link = self.causal_graph[cause][effect]
                link.observations += 1

                # Sliding update of causal strength
                co_occurrence = abs(variables[cause]) * abs(variables[effect])
                link.strength = 0.9 * link.strength + 0.1 * min(1.0, co_occurrence)
                link.confidence = min(1.0, link.observations / max(1, self.min_obs))

    # ------------------------------------------------------------------
    def infer(self, cause: str, effect: str) -> Dict[str, Any]:
        """
        Association inference: P(effect | cause)
        "When cause is observed, what is the expected value of effect?"
        """
        self.total_inferences += 1

        if cause in self.causal_graph and effect in self.causal_graph[cause]:
            link = self.causal_graph[cause][effect]
            return {
                "type": "association",
                "cause": cause,
                "effect": effect,
                "strength": round(link.strength, 4),
                "confidence": round(link.confidence, 4),
                "observations": link.observations,
                "reliable": link.confidence > 0.5 and link.strength > self.sig_threshold,
            }

        return {
            "type": "association",
            "cause": cause,
            "effect": effect,
            "strength": 0.0,
            "confidence": 0.0,
            "observations": 0,
            "reliable": False,
        }

    # ------------------------------------------------------------------
    def intervene(self, do_variable: str, do_value: float, target: str) -> Dict[str, Any]:
        """
        Interventional inference: P(target | do(variable = value))
        "If I force variable = value, what will target become?"
        """
        self.total_interventions += 1

        # Find causal path from do_variable → target
        path = self._find_causal_path(do_variable, target)

        if not path:
            return {
                "type": "intervention",
                "do": f"do({do_variable}={do_value})",
                "target": target,
                "predicted_effect": 0.0,
                "path": [],
                "confidence": 0.0,
            }

        # Propagate effect along path
        current_value = do_value
        total_strength = 1.0
        for i in range(len(path) - 1):
            link = self.causal_graph.get(path[i], {}).get(path[i + 1])
            if link:
                total_strength *= link.strength
                current_value *= link.strength

        return {
            "type": "intervention",
            "do": f"do({do_variable}={do_value})",
            "target": target,
            "predicted_effect": round(current_value, 4),
            "path": path,
            "path_strength": round(total_strength, 4),
            "confidence": round(min(1.0, total_strength * 0.8), 4),
        }

    # ------------------------------------------------------------------
    def counterfactual(
        self,
        factual: Dict[str, float],
        counterfactual_var: str,
        counterfactual_value: float,
        target: str,
    ) -> Dict[str, Any]:
        """
        Counterfactual inference:
        "In fact X=x' and Y=y'; if X had been x, what would Y be?"
        """
        self.total_counterfactuals += 1

        actual_target = factual.get(target, 0.0)
        actual_cause = factual.get(counterfactual_var, 0.0)

        # Counterfactual difference
        delta_cause = counterfactual_value - actual_cause

        # Propagate difference through causal graph
        path = self._find_causal_path(counterfactual_var, target)
        if not path:
            return {
                "type": "counterfactual",
                "question": f"If {counterfactual_var} were {counterfactual_value} instead of {actual_cause}",
                "target": target,
                "factual_value": actual_target,
                "counterfactual_value": actual_target,
                "delta": 0.0,
                "path": [],
            }

        total_strength = 1.0
        for i in range(len(path) - 1):
            link = self.causal_graph.get(path[i], {}).get(path[i + 1])
            if link:
                total_strength *= link.strength

        delta_effect = delta_cause * total_strength
        cf_value = actual_target + delta_effect

        return {
            "type": "counterfactual",
            "question": f"If {counterfactual_var} were {counterfactual_value} instead of {actual_cause}",
            "target": target,
            "factual_value": round(actual_target, 4),
            "counterfactual_value": round(cf_value, 4),
            "delta": round(delta_effect, 4),
            "path": path,
            "path_strength": round(total_strength, 4),
        }

    # ------------------------------------------------------------------
    def _find_causal_path(
        self, start: str, end: str, max_depth: Optional[int] = None
    ) -> List[str]:
        """BFS to find causal path"""
        if max_depth is None:
            max_depth = self.max_depth

        if start == end:
            return [start]

        visited: Set[str] = set()
        queue: List[Tuple[str, List[str]]] = [(start, [start])]

        while queue:
            current, path = queue.pop(0)
            if current in visited or len(path) > max_depth:
                continue
            visited.add(current)

            for neighbor in self.causal_graph.get(current, {}):
                link = self.causal_graph[current][neighbor]
                if link.strength > self.sig_threshold * 0.5:
                    new_path = path + [neighbor]
                    if neighbor == end:
                        return new_path
                    queue.append((neighbor, new_path))

        return []

    # ------------------------------------------------------------------
    def get_causal_graph_summary(self) -> Dict[str, Any]:
        """Causal graph summary"""
        nodes = set()
        edges = []
        for cause, effects in self.causal_graph.items():
            nodes.add(cause)
            for effect, link in effects.items():
                nodes.add(effect)
                if link.strength > self.sig_threshold:
                    edges.append(
                        {
                            "cause": cause,
                            "effect": effect,
                            "strength": round(link.strength, 3),
                            "confidence": round(link.confidence, 3),
                        }
                    )

        return {
            "nodes": list(nodes),
            "edges": edges,
            "total_nodes": len(nodes),
            "total_significant_edges": len(edges),
        }

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_observations": len(self.observations),
            "total_inferences": self.total_inferences,
            "total_interventions": self.total_interventions,
            "total_counterfactuals": self.total_counterfactuals,
            "causal_graph": self.get_causal_graph_summary(),
        }

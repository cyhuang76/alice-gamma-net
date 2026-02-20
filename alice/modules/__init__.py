# -*- coding: utf-8 -*-
"""
Alice Smart System — v5 Modules (Higher Cognitive Modules)

Contents:
- WorkingMemory    : Working Memory (Miller 7±2)
- ReinforcementLearner : Dopamine-driven Reinforcement Learning
- CausalReasoner   : Causal Reasoning Engine
- MetaLearner      : Meta-Learning (Learning to Learn)
"""

from alice.modules.working_memory import WorkingMemory
from alice.modules.reinforcement import ReinforcementLearner
from alice.modules.causal_reasoning import CausalReasoner
from alice.modules.meta_learning import MetaLearner

__all__ = [
    "WorkingMemory",
    "ReinforcementLearner",
    "CausalReasoner",
    "MetaLearner",
]

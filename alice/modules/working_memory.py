# -*- coding: utf-8 -*-
"""
Working Memory Module — Miller's Magic Number 7±2 + Impedance-Modulated Decay

Simulates the core characteristics of human working memory:
- Limited capacity (7±2 chunks)
- Temporal decay (forgotten if not rehearsed)
- Rehearsal enhancement (rehearsal reactivates items)
- Chunking (compresses capacity)
- ★ Impedance-modulated decay

Impedance decay physics:
  The decay gradient of short-term memory is not a fixed constant —
  it is a physical consequence of cross-modal binding quality.

  When a TV plays video with audio, the brain reverse-infers signals to
  produce attention. If audio and video cannot bind (impedance mismatch),
  the 'write power' of memory is reflected away.

  Coaxial cable physics:
    Transmitted power  P_t = (1 - Γ²) × P_in
    Reflected power    P_r = Γ² × P_in

  When Γ_bind (binding impedance mismatch) is larger:
    - Less power enters memory → initial activation is lower
    - Memory's impedance interface is unstable → decay accelerates

  Effective decay rate:
    λ_eff = λ_base / (1 - Γ_bind²)

  | Γ_bind | Meaning        | λ_eff              | Memory behavior     |
  |--------|----------------|--------------------|---------------------|
  |  0.0   | Perfect match  | λ_base             | Slow decay, details preserved |
  |  0.5   | Partial mismatch| 1.33 × λ_base    | Remember gist, details blurry |
  |  0.8   | Severe mismatch| 2.78 × λ_base     | Rapid forgetting    |
  |  →1.0  | Total mismatch | → ∞               | Instant forgetting  |

  "The so-called short-term memory decay gradient is an uncalibrated mechanism."
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MemoryItem:
    """A single item in working memory"""

    key: str
    content: Any
    activation: float = 1.0         # Activation level (0~1)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    importance: float = 0.5         # Importance (0~1)
    binding_gamma: float = 0.0     # ★ Cross-modal binding impedance mismatch Γ (0=perfect, →1=total mismatch)

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def time_since_access(self) -> float:
        return time.time() - self.last_accessed

    @property
    def impedance_decay_factor(self) -> float:
        """
        Impedance decay factor = 1 / (1 - Γ²)

        Physics: coaxial cable transmitted power P_t = (1 - Γ²) × P_in
        Memory retention ∝ (1 - Γ²)
        Effective decay rate = base_rate × 1 / (1 - Γ²)

        Γ=0.0 → factor=1.00 (normal decay)
        Γ=0.5 → factor=1.33 (slightly faster)
        Γ=0.8 → factor=2.78 (rapid forgetting)
        Γ=0.95→ factor=10.3 (near-instant forgetting)
        """
        gamma_sq = min(self.binding_gamma ** 2, 0.99)  # Prevent division by zero
        return 1.0 / (1.0 - gamma_sq)

    @property
    def transmission_efficiency(self) -> float:
        """Transmission efficiency = 1 - Γ² — how much power actually enters during memory write"""
        return 1.0 - min(self.binding_gamma ** 2, 0.99)


class WorkingMemory:
    """
    Working Memory System

    Core mechanisms:
    - Capacity limit: at most `capacity` items (default 7)
    - Temporal decay: activation *= (1 - decay_rate) per second
    - Rehearsal enhancement: rehearsal re-boosts activation
    - Overflow replacement: new items evict the weakest old item
    """

    def __init__(
        self,
        capacity: int = 7,
        decay_rate: float = 0.05,
        rehearsal_boost: float = 0.3,
        eviction_threshold: float = 0.1,
    ):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.rehearsal_boost = rehearsal_boost
        self.eviction_threshold = eviction_threshold

        self._items: Dict[str, MemoryItem] = {}
        self._last_decay_time = time.time()

        # Statistics
        self.total_stored = 0
        self.total_evicted = 0
        self.total_retrievals = 0
        self.total_rehearsals = 0
        # ★ Impedance decay statistics
        self.total_impedance_evictions = 0   # Times eviction was accelerated due to high impedance
        self._avg_binding_gamma = 0.0         # Average binding Γ of recent memories

    # ------------------------------------------------------------------
    def store(
        self,
        key: str,
        content: Any,
        importance: float = 0.5,
        binding_gamma: float = 0.0,
    ) -> bool:
        """
        Store to working memory

        If full, evicts the item with the lowest activation.

        Args:
            binding_gamma: ★ Cross-modal binding impedance mismatch Γ (0=perfect binding, →1=total mismatch)
                          Physical meaning: transmission efficiency at memory write = (1 - Γ²)
                          - Perfect binding (Γ=0): full power write, activation = 1.0
                          - Partial mismatch (Γ=0.5): 75% power, activation = 0.75
                          - Severe mismatch (Γ=0.8): 36% power, activation = 0.36

        Returns: True if successful (or replaced)
        """
        self._apply_decay()

        # ★ Compute initial activation = transmission efficiency
        # Physics: P_transmitted = (1 - Γ²) × P_input
        gamma_clamped = float(np.clip(binding_gamma, 0.0, 0.99))
        transmission = 1.0 - gamma_clamped ** 2
        initial_activation = transmission  # Better impedance matching → more complete write

        if key in self._items:
            # Update existing item — rehearsal enhancement
            item = self._items[key]
            item.content = content
            item.activation = min(1.0, item.activation + self.rehearsal_boost)
            item.last_accessed = time.time()
            item.access_count += 1
            item.importance = max(item.importance, importance)
            # ★ Rehearsal can improve binding quality (recalibration → Γ decreases)
            item.binding_gamma = min(item.binding_gamma, gamma_clamped)
            return True

        # Evict when at capacity
        if len(self._items) >= self.capacity:
            self._evict_weakest()

        self._items[key] = MemoryItem(
            key=key,
            content=content,
            activation=initial_activation,    # ★ Initial activation determined by transmission efficiency
            importance=importance,
            binding_gamma=gamma_clamped,       # ★ Record binding quality
        )
        self.total_stored += 1

        # ★ Update average binding Γ statistics
        if self._items:
            self._avg_binding_gamma = float(np.mean(
                [it.binding_gamma for it in self._items.values()]
            ))

        return True

    # ------------------------------------------------------------------
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a working memory item (also boosts activation)"""
        self._apply_decay()
        self.total_retrievals += 1

        if key in self._items:
            item = self._items[key]
            item.activation = min(1.0, item.activation + self.rehearsal_boost * 0.5)
            item.last_accessed = time.time()
            item.access_count += 1
            return item.content
        return None

    # ------------------------------------------------------------------
    def rehearse(self, key: str) -> bool:
        """Rehearse (strengthen memory)"""
        if key in self._items:
            item = self._items[key]
            item.activation = min(1.0, item.activation + self.rehearsal_boost)
            item.last_accessed = time.time()
            self.total_rehearsals += 1
            return True
        return False

    # ------------------------------------------------------------------
    def _apply_decay(self):
        """
        Impedance-Modulated Temporal Decay

        Physics:
          In a coaxial cable, an impedance-mismatched transmission line
          not only loses power (incomplete write), but also generates
          standing waves at the reflection interface, continuously
          consuming residual energy.

          Effective decay rate: λ_eff = λ_base / (1 - Γ_bind²)

          Memory with Γ=0 (perfect binding) → slow decay at base rate
          Memory with Γ→1 (total mismatch) → decay approaches infinity → instant forgetting

          "The so-called short-term memory decay gradient is an uncalibrated mechanism."
        """
        now = time.time()
        dt = now - self._last_decay_time
        if dt < 0.1:
            return
        self._last_decay_time = now

        to_remove = []
        for key, item in self._items.items():
            elapsed = now - item.last_accessed

            # ★ Impedance-modulated decay rate
            # λ_eff = λ_base × impedance_decay_factor
            # impedance_decay_factor = 1 / (1 - Γ²)
            effective_rate = self.decay_rate * item.impedance_decay_factor

            # Exponential decay (higher impedance = faster decay)
            item.activation *= np.exp(-effective_rate * elapsed)

            if item.activation < self.eviction_threshold:
                to_remove.append(key)
                # ★ Track whether eviction was accelerated due to high impedance
                if item.binding_gamma > 0.3:
                    self.total_impedance_evictions += 1

        for key in to_remove:
            del self._items[key]
            self.total_evicted += 1

    # ------------------------------------------------------------------
    def _evict_weakest(self):
        """Evict the item with the lowest activation"""
        if not self._items:
            return
        weakest = min(self._items.values(), key=lambda it: it.activation * (0.5 + it.importance))
        del self._items[weakest.key]
        self.total_evicted += 1

    # ------------------------------------------------------------------
    def flush_weakest(self, fraction: float = 0.3) -> int:
        """
        Flush the weakest batch of memories ("widen" working memory during cognitive restructuring).

        Parameters
        ----------
        fraction : float
            Fraction to clear (0~1), default clears the weakest 30%

        Returns
        -------
        int  Number actually cleared
        """
        if not self._items:
            return 0
        n_to_flush = max(1, int(len(self._items) * fraction))
        sorted_items = sorted(
            self._items.values(),
            key=lambda it: it.activation * (0.5 + it.importance),
        )
        flushed = 0
        for item in sorted_items[:n_to_flush]:
            del self._items[item.key]
            self.total_evicted += 1
            flushed += 1
        return flushed

    # ------------------------------------------------------------------
    def get_contents(self) -> List[Dict[str, Any]]:
        """Get all working memory contents"""
        self._apply_decay()
        return [
            {
                "key": item.key,
                "activation": round(item.activation, 3),
                "importance": round(item.importance, 3),
                "age_seconds": round(item.age, 1),
                "access_count": item.access_count,
                # ★ Impedance decay info
                "binding_gamma": round(item.binding_gamma, 4),
                "impedance_decay_factor": round(item.impedance_decay_factor, 3),
                "transmission_efficiency": round(item.transmission_efficiency, 3),
            }
            for item in sorted(self._items.values(), key=lambda x: -x.activation)
        ]

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        items = list(self._items.values())
        return {
            "current_size": len(items),
            "capacity": self.capacity,
            "utilization": round(len(items) / max(1, self.capacity), 3),
            "total_stored": self.total_stored,
            "total_evicted": self.total_evicted,
            "total_retrievals": self.total_retrievals,
            "total_rehearsals": self.total_rehearsals,
            "avg_activation": round(
                float(np.mean([it.activation for it in items]))
                if items
                else 0.0,
                3,
            ),
            # ★ Impedance decay statistics
            "impedance_evictions": self.total_impedance_evictions,
            "avg_binding_gamma": round(self._avg_binding_gamma, 4),
            "avg_impedance_factor": round(
                float(np.mean([it.impedance_decay_factor for it in items]))
                if items
                else 1.0,
                3,
            ),
            "avg_transmission_efficiency": round(
                float(np.mean([it.transmission_efficiency for it in items]))
                if items
                else 1.0,
                3,
            ),
        }

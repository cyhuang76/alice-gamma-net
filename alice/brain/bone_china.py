# -*- coding: utf-8 -*-
"""
Bone China Engine — Five-Phase Memory Consolidation

Paper III §2.3: Long-term memory formation requires physical modification
of the transmission line — a process analogous to ceramic firing:

  Phase 1: Clay (wet)      → Working memory     → transient electrical pattern, high Γ
  Phase 2: Greenware        → Hippocampal replay  → N2 spindle selects high-value episodes
  Phase 3: Bisque firing    → N3 consolidation    → high-energy restructuring, ΣΓ² reduction
  Phase 4: Glaze            → Semantic integration → bound to concept network
  Phase 5: Porcelain        → Long-term memory     → permanent, low-Γ, structurally rigid

Physics:
  Each phase transition involves energy expenditure and irreversible structural change.
  The fontanelle functions as the primary thermal exhaust port for the massive Γ²
  waste heat generated during firing.

  Clay → Greenware: Selection (N2 spindles identify high-value items)
  Greenware → Bisque: Structural firing (N3 deep sleep; highest energy cost)
  Bisque → Glaze: Semantic binding (concept network integration)
  Glaze → Porcelain: Crystallization (final rigid encoding, irreversible)

  Items that fail to advance are discarded (forgotten):
  - Clay that never enters Greenware → evaporates (transient WM decay)
  - Greenware that breaks during Bisque → shattered (consolidation failure)
  - Bisque that doesn't accept Glaze → isolated fact (no semantic context)

Clinical correspondence:
  - Infantile amnesia: no Porcelain forms before fontanelle closure
  - Flashbulb memory: emotional valence fast-tracks Clay → Greenware
  - Cramming vs. sleep: repeated Clay creation without Bisque firing → poor retention
  - PTSD: trauma fires directly to Porcelain (bypasses Greenware/Bisque), rigid and unforgettable

"Memory is not storage — it is ceramic engineering."

Author: Hsi-Yu Huang (黃璽宇)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from alice.core.signal import ElectricalSignal


# ============================================================================
# Physical Constants
# ============================================================================

# --- Phase transition thresholds ---
CLAY_DECAY_RATE = 0.05          # Clay (WM) natural decay per tick
GREENWARE_SELECTION_THRESHOLD = 0.3  # Minimum importance to survive N2 selection
BISQUE_ENERGY_COST = 0.15       # Energy consumed by N3 bisque firing per item
BISQUE_GAMMA_REDUCTION = 0.4    # Γ reduction during bisque firing (40% improvement)
GLAZE_SEMANTIC_THRESHOLD = 0.2  # Minimum semantic overlap for glaze to bind
PORCELAIN_LOCK_GAMMA = 0.1     # Final Γ of porcelain memory (very low)

# --- Phase transition timing ---
CLAY_LIFETIME_TICKS = 30        # Clay expires after ~30 ticks if not promoted
GREENWARE_WINDOW_TICKS = 60     # N2 selection window
BISQUE_DURATION_TICKS = 10      # N3 firing duration
GLAZE_DURATION_TICKS = 5        # Semantic binding duration

# --- Emotional fast-track ---
EMOTION_FAST_TRACK_THRESHOLD = 0.7   # Valence above this → skip selection queue
TRAUMA_DIRECT_PORCELAIN = 0.9        # Extreme trauma → direct Clay → Porcelain

# --- Capacity limits ---
MAX_CLAY_ITEMS = 7              # Miller's 7±2 (working memory limit)
MAX_GREENWARE_ITEMS = 20        # N2 selection buffer
MAX_BISQUE_ITEMS = 10           # N3 firing capacity per sleep cycle
MAX_PORCELAIN_ITEMS = 10000     # Long-term memory (effectively unlimited)

# --- Γ values per phase ---
GAMMA_CLAY = 0.9                # Clay: very high mismatch (fragile, transient)
GAMMA_GREENWARE = 0.7           # Greenware: moderate (selected but unfired)
GAMMA_BISQUE = 0.35             # Bisque: reduced (structurally sound after firing)
GAMMA_GLAZE = 0.15              # Glazed: low (semantically integrated)
GAMMA_PORCELAIN = 0.05          # Porcelain: minimal (permanent, crystalline)

# --- Heat generation ---
HEAT_PER_BISQUE_FIRING = 0.02   # Γ² waste heat per bisque firing
HEAT_PER_GLAZE = 0.005          # Smaller heat from semantic binding


# ============================================================================
# Data Structures
# ============================================================================

class MemoryPhase(Enum):
    """The five phases of ceramic memory formation."""
    CLAY = "clay"               # Wet clay: transient, high Γ, working memory
    GREENWARE = "greenware"     # Dried: selected by N2 spindles, awaiting firing
    BISQUE = "bisque"           # First firing: N3 consolidation, structurally sound
    GLAZE = "glaze"             # Glazed: semantically integrated
    PORCELAIN = "porcelain"     # Final: permanent, low Γ, long-term memory


@dataclass
class MemoryShard:
    """
    A single memory item progressing through the ceramic pipeline.

    Each shard has a Γ value that decreases as it advances through phases.
    Failure to advance → the shard is destroyed (forgotten).
    """
    shard_id: int
    content_key: str                # What is being remembered (concept label or episode key)
    phase: MemoryPhase = MemoryPhase.CLAY
    gamma: float = GAMMA_CLAY       # Current impedance mismatch
    importance: float = 0.5         # 0~1 (emotional valence, reward signal, etc.)
    emotional_valence: float = 0.0  # -1~+1 (negative = traumatic, positive = rewarding)
    semantic_bindings: int = 0      # Number of concept network connections
    created_tick: int = 0           # When this shard was created
    last_advanced_tick: int = 0     # When last phase transition occurred
    firing_progress: float = 0.0    # 0~1 progress within current phase
    heat_generated: float = 0.0     # Cumulative Γ² waste heat from this shard

    @property
    def transmission(self) -> float:
        """★ Energy conservation: T = 1 − Γ²."""
        return 1.0 - self.gamma ** 2

    @property
    def is_expired(self) -> bool:
        """Whether this shard has expired in its current phase."""
        return self.phase == MemoryPhase.CLAY and self.gamma <= 0

    @property
    def is_porcelain(self) -> bool:
        """Whether this shard has reached permanent memory."""
        return self.phase == MemoryPhase.PORCELAIN


@dataclass
class BoneChinaState:
    """Snapshot of the bone china engine state."""
    clay_count: int
    greenware_count: int
    bisque_count: int
    glaze_count: int
    porcelain_count: int
    total_shards: int
    total_porcelain_ever: int
    total_shattered: int
    total_heat_generated: float
    mean_gamma: float
    firing_active: bool


# ============================================================================
# BoneChinaEngine
# ============================================================================

class BoneChinaEngine:
    """
    Bone China Engine — Five-phase ceramic memory consolidation.

    Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                     BoneChinaEngine                              │
    │                                                                  │
    │  ┌─────┐   ┌──────────┐   ┌───────┐   ┌───────┐   ┌──────────┐│
    │  │Clay │──→│Greenware │──→│Bisque │──→│ Glaze │──→│Porcelain ││
    │  │ WM  │   │N2 select │   │N3 fire│   │ Sem.  │   │  LTM     ││
    │  │Γ=0.9│   │ Γ=0.7    │   │Γ=0.35 │   │Γ=0.15 │   │ Γ=0.05  ││
    │  └──┬──┘   └──┬───────┘   └──┬────┘   └──┬────┘   └──────────┘│
    │     ↓          ↓              ↓            ↓                    │
    │  [decay]    [shatter]      [crack]      [isolated]   [permanent]│
    │  (forget)   (not selected) (fire fail) (no context)             │
    └──────────────────────────────────────────────────────────────────┘

    Heat exhaust: Each firing generates Γ² waste heat.
    During neonatal development, the fontanelle is the thermal exhaust port.
    """

    def __init__(self) -> None:
        # Storage by phase
        self._shards: List[MemoryShard] = []
        self._next_id: int = 0

        # Statistics
        self._total_porcelain_ever: int = 0
        self._total_shattered: int = 0
        self._total_heat: float = 0.0
        self._tick_count: int = 0

        # Firing state
        self._firing_active: bool = False  # N3 firing in progress
        self._firing_queue: List[int] = []  # Shard IDs queued for bisque firing

    # ------------------------------------------------------------------
    # Phase 1: Clay — Create new memory shard (working memory entry)
    # ------------------------------------------------------------------

    def create_clay(
        self,
        content_key: str,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
    ) -> MemoryShard:
        """
        Create a new Clay shard = new working memory entry.

        High emotional valence → fast-tracked past selection.
        Extreme trauma → direct to Porcelain (PTSD mechanism).
        """
        shard = MemoryShard(
            shard_id=self._next_id,
            content_key=content_key,
            phase=MemoryPhase.CLAY,
            gamma=GAMMA_CLAY,
            importance=float(np.clip(importance, 0.0, 1.0)),
            emotional_valence=emotional_valence,
            created_tick=self._tick_count,
            last_advanced_tick=self._tick_count,
        )
        self._next_id += 1

        # ★ Trauma direct-to-porcelain: extreme pain bypasses all stages
        if abs(emotional_valence) > TRAUMA_DIRECT_PORCELAIN:
            shard.phase = MemoryPhase.PORCELAIN
            shard.gamma = GAMMA_PORCELAIN * 2.0  # Trauma porcelain has slightly higher Γ (rigid but painful)
            shard.heat_generated = HEAT_PER_BISQUE_FIRING * 3.0  # Massive heat dump
            self._total_heat += shard.heat_generated
            self._total_porcelain_ever += 1

        # Enforce capacity limit
        clay_items = [s for s in self._shards if s.phase == MemoryPhase.CLAY]
        if len(clay_items) >= MAX_CLAY_ITEMS:
            # Evict lowest importance clay
            clay_items.sort(key=lambda s: s.importance)
            self._shards.remove(clay_items[0])
            self._total_shattered += 1

        self._shards.append(shard)
        return shard

    # ------------------------------------------------------------------
    # Phase 2: Greenware — N2 spindle selection
    # ------------------------------------------------------------------

    def select_greenware(self) -> int:
        """
        N2 spindle selection: promote high-value Clay to Greenware.

        Selection criteria:
        - importance > threshold
        - emotional fast-track (high |valence|)
        - Recency (newer clay preferred)

        Returns number of items promoted.
        """
        promoted = 0
        for shard in self._shards:
            if shard.phase != MemoryPhase.CLAY:
                continue

            # Skip expired clay
            age = self._tick_count - shard.created_tick
            if age > CLAY_LIFETIME_TICKS:
                shard.gamma = 0.0  # Mark for cleanup
                continue

            # Selection criteria
            emotional_boost = abs(shard.emotional_valence) * 0.3
            effective_importance = shard.importance + emotional_boost

            if effective_importance >= GREENWARE_SELECTION_THRESHOLD:
                # Emotional fast-track
                if abs(shard.emotional_valence) > EMOTION_FAST_TRACK_THRESHOLD:
                    shard.phase = MemoryPhase.GREENWARE
                    shard.gamma = GAMMA_GREENWARE * 0.8  # Emotionally charged = better retention
                else:
                    shard.phase = MemoryPhase.GREENWARE
                    shard.gamma = GAMMA_GREENWARE
                shard.last_advanced_tick = self._tick_count
                promoted += 1

        # Enforce greenware capacity
        greenware = [s for s in self._shards if s.phase == MemoryPhase.GREENWARE]
        while len(greenware) > MAX_GREENWARE_ITEMS:
            greenware.sort(key=lambda s: s.importance)
            self._shards.remove(greenware[0])
            greenware.pop(0)
            self._total_shattered += 1

        return promoted

    # ------------------------------------------------------------------
    # Phase 3: Bisque — N3 deep sleep consolidation firing
    # ------------------------------------------------------------------

    def fire_bisque(self, available_energy: float = 1.0) -> Dict[str, Any]:
        """
        N3 bisque firing: structural consolidation of Greenware.

        This is the most energy-intensive phase.
        Each item consumes BISQUE_ENERGY_COST and generates Γ² waste heat.

        Args:
            available_energy: 0~1 energy available for firing (from sleep_physics)

        Returns:
            Dict with firing results (items_fired, heat_generated, energy_consumed)
        """
        self._firing_active = True
        greenware = [s for s in self._shards if s.phase == MemoryPhase.GREENWARE]

        # Sort by importance (fire most important first)
        greenware.sort(key=lambda s: s.importance, reverse=True)

        items_fired = 0
        heat_generated = 0.0
        energy_consumed = 0.0

        for shard in greenware[:MAX_BISQUE_ITEMS]:
            if energy_consumed + BISQUE_ENERGY_COST > available_energy:
                break  # Not enough energy to fire more

            # Firing: Γ reduces significantly
            shard.phase = MemoryPhase.BISQUE
            shard.gamma = GAMMA_BISQUE
            shard.firing_progress = 1.0
            shard.last_advanced_tick = self._tick_count

        # ★ Heat generation: reflected energy = input × Γ² (energy conservation)
            heat = HEAT_PER_BISQUE_FIRING * shard.gamma ** 2
            shard.heat_generated += heat
            heat_generated += heat
            energy_consumed += BISQUE_ENERGY_COST
            items_fired += 1

        self._total_heat += heat_generated
        self._firing_active = items_fired > 0

        return {
            "items_fired": items_fired,
            "heat_generated": round(heat_generated, 6),
            "energy_consumed": round(energy_consumed, 4),
            "remaining_greenware": len(greenware) - items_fired,
        }

    # ------------------------------------------------------------------
    # Phase 4: Glaze — Semantic integration
    # ------------------------------------------------------------------

    def apply_glaze(self, semantic_overlap_scores: Optional[Dict[str, float]] = None) -> int:
        """
        Semantic glazing: bind Bisque items to the concept network.

        Args:
            semantic_overlap_scores: dict of content_key → overlap score (0~1)
                If provided, only items with overlap > threshold get glazed.
                If None, all bisque items advance (experiment mode).

        Returns:
            Number of items glazed.
        """
        glazed = 0
        for shard in self._shards:
            if shard.phase != MemoryPhase.BISQUE:
                continue

            # Check semantic binding availability
            if semantic_overlap_scores is not None:
                overlap = semantic_overlap_scores.get(shard.content_key, 0.0)
                if overlap < GLAZE_SEMANTIC_THRESHOLD:
                    continue  # Isolated fact — no semantic context
                shard.semantic_bindings = max(1, int(overlap * 5))
            else:
                shard.semantic_bindings = 1  # Default: at least one binding

            # Apply glaze
            shard.phase = MemoryPhase.GLAZE
            shard.gamma = GAMMA_GLAZE
            shard.last_advanced_tick = self._tick_count

            # Small heat from semantic binding
            heat = HEAT_PER_GLAZE
            shard.heat_generated += heat
            self._total_heat += heat
            glazed += 1

        return glazed

    # ------------------------------------------------------------------
    # Phase 5: Porcelain — Crystallization to long-term memory
    # ------------------------------------------------------------------

    def crystallize(self) -> int:
        """
        Crystallize Glazed items into Porcelain = permanent long-term memory.

        This is the irreversible step. Once crystallized, the memory is permanent
        and can only degrade through aging (Coffin-Manson fatigue).

        Returns:
            Number of items crystallized.
        """
        crystallized = 0
        for shard in self._shards:
            if shard.phase != MemoryPhase.GLAZE:
                continue

            # Glaze → Porcelain
            shard.phase = MemoryPhase.PORCELAIN
            shard.gamma = GAMMA_PORCELAIN
            shard.firing_progress = 1.0
            shard.last_advanced_tick = self._tick_count
            crystallized += 1
            self._total_porcelain_ever += 1

        return crystallized

    # ------------------------------------------------------------------
    # Tick — Advance time, decay Clay, cleanup expired
    # ------------------------------------------------------------------

    def tick(self, is_sleeping: bool = False, sleep_stage: str = "wake") -> Dict[str, Any]:
        """
        Advance the bone china engine by one tick.

        During wakefulness: Clay decays, new clay can be created.
        During N2 sleep: Greenware selection runs.
        During N3 sleep: Bisque firing runs.
        During REM: Glaze application and crystallization.

        Args:
            is_sleeping: Whether the system is asleep.
            sleep_stage: "wake", "n1", "n2", "n3", "rem"

        Returns:
            Dict summarizing tick activity.
        """
        self._tick_count += 1
        result: Dict[str, Any] = {
            "tick": self._tick_count,
            "promotions": 0,
            "firings": 0,
            "glazings": 0,
            "crystallizations": 0,
            "expirations": 0,
            "heat_generated": 0.0,
        }

        # === Clay decay (always active) ===
        expired = 0
        for shard in self._shards[:]:
            if shard.phase == MemoryPhase.CLAY:
                age = self._tick_count - shard.created_tick
                if age > CLAY_LIFETIME_TICKS:
                    self._shards.remove(shard)
                    self._total_shattered += 1
                    expired += 1
                else:
                    # ★ Hebbian clay decay: ΔΓ = -rate × Γ × T
                    # Only the transmitted fraction drives learning/decay
                    T = 1.0 - shard.gamma ** 2
                    shard.gamma = max(0.0, shard.gamma - CLAY_DECAY_RATE * shard.gamma * T)
        result["expirations"] = expired

        # === Sleep-stage-specific processing ===
        if is_sleeping:
            if sleep_stage == "n2":
                result["promotions"] = self.select_greenware()

            elif sleep_stage == "n3":
                fire_result = self.fire_bisque(available_energy=0.8)
                result["firings"] = fire_result["items_fired"]
                result["heat_generated"] = fire_result["heat_generated"]

            elif sleep_stage == "rem":
                result["glazings"] = self.apply_glaze()
                result["crystallizations"] = self.crystallize()

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_phase_counts(self) -> Dict[str, int]:
        """Count items in each phase."""
        counts = {phase.value: 0 for phase in MemoryPhase}
        for shard in self._shards:
            counts[shard.phase.value] += 1
        return counts

    def get_state(self) -> Dict[str, Any]:
        """Full state for introspection."""
        counts = self.get_phase_counts()
        gammas = [s.gamma for s in self._shards] if self._shards else [0.0]
        return {
            "phase_counts": counts,
            "total_shards": len(self._shards),
            "total_porcelain_ever": self._total_porcelain_ever,
            "total_shattered": self._total_shattered,
            "total_heat_generated": round(self._total_heat, 6),
            "mean_gamma": round(float(np.mean(gammas)), 4),
            "firing_active": self._firing_active,
            "tick_count": self._tick_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_state."""
        return self.get_state()

    # ------------------------------------------------------------------
    # Signal Protocol
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate ElectricalSignal encoding memory consolidation state."""
        gammas = [s.gamma for s in self._shards] if self._shards else [0.5]
        mean_gamma = float(np.mean(gammas))
        amplitude = float(np.clip(1.0 - mean_gamma, 0.01, 1.0))
        freq = 0.5 + self._total_heat * 10.0  # Firing activity → frequency
        t = np.linspace(0, 1, 64)
        waveform = amplitude * np.sin(2 * np.pi * freq * t)
        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            amplitude=amplitude,
            frequency=min(freq, 100.0),
            phase=0.0,
            impedance=75.0,
            snr=10.0,
            source="bone_china",
            modality="internal",
        )

    def get_porcelain_memories(self) -> List[Dict[str, Any]]:
        """Get all permanent (porcelain) memories."""
        return [
            {
                "shard_id": s.shard_id,
                "content_key": s.content_key,
                "gamma": round(s.gamma, 4),
                "importance": round(s.importance, 4),
                "emotional_valence": round(s.emotional_valence, 4),
                "semantic_bindings": s.semantic_bindings,
                "created_tick": s.created_tick,
            }
            for s in self._shards
            if s.phase == MemoryPhase.PORCELAIN
        ]

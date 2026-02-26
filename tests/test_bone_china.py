# -*- coding: utf-8 -*-
"""
Tests for BoneChinaEngine — Five-Phase Memory Consolidation

Tests verify:
    1. Basic construction and initial state
    2. Clay creation (working memory → transient)
    3. Greenware selection (N2 spindle selection)
    4. Bisque firing (N3 consolidation)
    5. Glaze application (REM integration)
    6. Crystallization to porcelain (long-term memory)
    7. Trauma direct-to-porcelain bypass
    8. Sleep-stage-driven tick
    9. State and stats consistency
"""

import pytest

from alice.brain.bone_china import (
    BoneChinaEngine,
    MemoryPhase,
    MemoryShard,
)


class TestBoneChinaConstruction:
    """BoneChinaEngine initializes correctly."""

    def test_init_empty(self):
        engine = BoneChinaEngine()
        state = engine.get_state()
        assert state["total_shards"] == 0

    def test_init_tick_zero(self):
        engine = BoneChinaEngine()
        assert engine._tick_count == 0


class TestClayCreation:
    """Clay = working memory entry with high Γ."""

    def test_create_clay(self):
        engine = BoneChinaEngine()
        shard = engine.create_clay(content_key="test_memory", emotional_valence=0.5)
        assert shard.phase == MemoryPhase.CLAY
        assert shard.gamma >= 0.7
        assert shard.content_key == "test_memory"

    def test_clay_in_shards(self):
        engine = BoneChinaEngine()
        engine.create_clay(content_key="test")
        clay = [s for s in engine._shards if s.phase == MemoryPhase.CLAY]
        assert len(clay) == 1

    def test_clay_capacity_limit(self):
        """Working memory limit = 7 (Miller's number)."""
        engine = BoneChinaEngine()
        for i in range(10):
            engine.create_clay(content_key=f"item_{i}", importance=float(i) / 10)
        clay = [s for s in engine._shards if s.phase == MemoryPhase.CLAY]
        assert len(clay) <= 7


class TestGreenwareSelection:
    """Greenware = hippocampal replay selects promising memories."""

    def test_select_greenware(self):
        engine = BoneChinaEngine()
        engine.create_clay(content_key="important", importance=0.8, emotional_valence=0.8)
        promoted = engine.select_greenware()
        assert promoted >= 1
        greenware = [s for s in engine._shards if s.phase == MemoryPhase.GREENWARE]
        assert len(greenware) >= 1
        assert greenware[0].phase == MemoryPhase.GREENWARE

    def test_low_importance_not_promoted(self):
        engine = BoneChinaEngine()
        engine.create_clay(content_key="trivial", importance=0.0, emotional_valence=0.0)
        promoted = engine.select_greenware()
        assert promoted == 0


class TestBisqueFiring:
    """Bisque = N3 consolidation → structural reorganization."""

    def test_fire_bisque(self):
        engine = BoneChinaEngine()
        engine.create_clay(content_key="consolidate", importance=0.9)
        engine.select_greenware()
        result = engine.fire_bisque(available_energy=1.0)
        assert result["items_fired"] >= 1
        bisque = [s for s in engine._shards if s.phase == MemoryPhase.BISQUE]
        if bisque:
            assert bisque[0].phase == MemoryPhase.BISQUE
            assert bisque[0].gamma < 0.9  # Lower than clay


class TestTraumaBypass:
    """Trauma bypasses normal phases → direct to porcelain."""

    def test_trauma_direct_porcelain(self):
        engine = BoneChinaEngine()
        shard = engine.create_clay(
            content_key="traumatic_event",
            emotional_valence=-0.95,  # > TRAUMA_DIRECT_PORCELAIN (0.9)
        )
        # Extreme trauma → direct to porcelain
        assert shard.phase == MemoryPhase.PORCELAIN
        assert shard is not None


class TestSleepTick:
    """Sleep-stage-driven tick processes memories."""

    def test_tick_awake(self):
        engine = BoneChinaEngine()
        engine.create_clay(content_key="hello")
        result = engine.tick(is_sleeping=False, sleep_stage="wake")
        assert isinstance(result, dict)

    def test_tick_n2(self):
        engine = BoneChinaEngine()
        engine.create_clay(content_key="world", importance=0.8)
        result = engine.tick(is_sleeping=True, sleep_stage="n2")
        assert isinstance(result, dict)
        assert "promotions" in result

    def test_tick_n3(self):
        engine = BoneChinaEngine()
        engine.create_clay(content_key="deep", importance=0.9)
        engine.select_greenware()
        result = engine.tick(is_sleeping=True, sleep_stage="n3")
        assert isinstance(result, dict)
        assert "firings" in result

    def test_tick_rem(self):
        engine = BoneChinaEngine()
        engine.create_clay(content_key="dream", importance=0.9)
        engine.select_greenware()
        engine.fire_bisque()
        result = engine.tick(is_sleeping=True, sleep_stage="rem")
        assert isinstance(result, dict)
        assert "crystallizations" in result


class TestBoneChinaState:
    """State and stats return correct structure."""

    def test_get_state(self):
        engine = BoneChinaEngine()
        state = engine.get_state()
        assert "phase_counts" in state
        assert "total_shards" in state
        assert "total_porcelain_ever" in state

    def test_get_stats(self):
        engine = BoneChinaEngine()
        stats = engine.get_stats()
        assert "total_shards" in stats
        assert "mean_gamma" in stats

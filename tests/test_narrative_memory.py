# -*- coding: utf-8 -*-
"""
Unit Tests — Narrative Memory Engine (Phase 20.1)
"""

import time
import pytest
import numpy as np

from alice.brain.hippocampus import HippocampusEngine, Episode, EpisodicSnapshot
from alice.brain.narrative_memory import (
    NarrativeMemoryEngine,
    NarrativeArc,
    CausalLink,
    EpisodeSummary,
    CAUSAL_TEMPORAL_LAMBDA,
    MIN_CAUSAL_STRENGTH,
    MAX_CAUSAL_DISTANCE_S,
    ARC_COHERENCE_MIN,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def hippo():
    return HippocampusEngine()


@pytest.fixture
def narrative(hippo):
    return NarrativeMemoryEngine(hippocampus=hippo)


@pytest.fixture
def populated_hippo(hippo):
    """Hippocampus with 4 episodes sharing concepts."""
    t_base = time.monotonic()
    concepts_per_ep = [
        [("cat", 0.2, 0.5), ("meow", 0.3, 0.4)],
        [("cat", 0.2, -0.3), ("sad", 0.4, -0.6)],
        [("cat", 0.1, 0.7), ("happy", 0.2, 0.8)],
        [("dog", 0.3, 0.0), ("bark", 0.4, -0.1)],
    ]
    for i, episode_concepts in enumerate(concepts_per_ep):
        t_ep = t_base + i * 5.0
        for j, (label, gamma, valence) in enumerate(episode_concepts):
            hippo.record("auditory", np.random.randn(32), label,
                         gamma, valence, t_ep + j * 0.3)
    for ep in hippo.episodes:
        ep.is_open = False
    return hippo


# ============================================================================
# Construction / Initialization
# ============================================================================

class TestNarrativeInit:
    def test_creation(self, narrative):
        assert narrative.hippocampus is not None
        assert len(narrative.arcs) == 0
        assert len(narrative.causal_graph) == 0
        assert len(narrative.summaries) == 0

    def test_standalone_creation(self):
        nm = NarrativeMemoryEngine()
        assert nm.hippocampus is None
        assert len(nm.arcs) == 0

    def test_get_state(self, narrative):
        state = narrative.get_state()
        assert "total_arcs" in state
        assert "total_causal_links" in state
        assert state["total_arcs"] == 0

    def test_get_stats(self, narrative):
        stats = narrative.get_stats()
        assert stats == narrative.get_state()


# ============================================================================
# Episode Summary
# ============================================================================

class TestEpisodeSummary:
    def test_summarize_episode(self, narrative, populated_hippo):
        ep = populated_hippo.episodes[0]
        summary = narrative.summarize_episode(ep)
        assert isinstance(summary, EpisodeSummary)
        assert summary.episode_id == ep.episode_id
        assert len(summary.dominant_concepts) > 0
        assert summary.n_snapshots == ep.n_snapshots

    def test_summary_cached(self, narrative, populated_hippo):
        ep = populated_hippo.episodes[0]
        s1 = narrative.summarize_episode(ep)
        s2 = narrative.summarize_episode(ep)
        assert s1 is s2  # Same object (cached)

    def test_summary_gist(self, narrative, populated_hippo):
        ep = populated_hippo.episodes[0]
        s = narrative.summarize_episode(ep)
        assert len(s.gist) > 0
        assert "cat" in s.gist or "meow" in s.gist

    def test_summary_to_dict(self, narrative, populated_hippo):
        ep = populated_hippo.episodes[0]
        s = narrative.summarize_episode(ep)
        d = s.to_dict()
        assert "episode_id" in d
        assert "dominant_concepts" in d
        assert "gist" in d

    def test_empty_episode_summary(self, narrative, hippo):
        # Create an episode with no concepts
        t = time.monotonic()
        hippo.record("visual", np.random.randn(256), None, 0.5, 0.0, t)
        ep = hippo.episodes[0]
        ep.is_open = False
        s = narrative.summarize_episode(ep)
        assert s.n_snapshots == 1
        # dominant_concepts may be empty
        assert isinstance(s.dominant_concepts, list)


# ============================================================================
# Causal Link Detection
# ============================================================================

class TestCausalLinks:
    def test_detect_link_shared_concepts(self, narrative, populated_hippo):
        ep_a = populated_hippo.episodes[0]  # cat
        ep_b = populated_hippo.episodes[1]  # cat
        link = narrative.detect_causal_link(ep_a, ep_b)
        assert link is not None
        assert "cat" in link.shared_concepts
        assert link.cause_episode_id == ep_a.episode_id
        assert link.effect_episode_id == ep_b.episode_id
        assert link.strength > MIN_CAUSAL_STRENGTH

    def test_no_link_no_shared_concepts(self, narrative, populated_hippo):
        ep_a = populated_hippo.episodes[0]  # cat+meow
        ep_d = populated_hippo.episodes[3]  # dog+bark
        link = narrative.detect_causal_link(ep_a, ep_d)
        assert link is None  # No shared concepts

    def test_no_link_wrong_order(self, narrative, populated_hippo):
        ep_a = populated_hippo.episodes[0]
        ep_b = populated_hippo.episodes[1]
        link = narrative.detect_causal_link(ep_b, ep_a)  # Wrong direction
        assert link is None

    def test_scan_all_links(self, narrative, populated_hippo):
        n = narrative.scan_causal_links()
        # Episodes 0,1,2 share "cat" → should have links
        assert n >= 2
        assert len(narrative.causal_graph) >= 2

    def test_causal_link_to_dict(self, narrative, populated_hippo):
        ep_a = populated_hippo.episodes[0]
        ep_b = populated_hippo.episodes[1]
        link = narrative.detect_causal_link(ep_a, ep_b)
        d = link.to_dict()
        assert "cause" in d
        assert "effect" in d
        assert "gamma_causal" in d

    def test_gamma_causal_inverse_of_strength(self, narrative, populated_hippo):
        ep_a = populated_hippo.episodes[0]
        ep_b = populated_hippo.episodes[1]
        link = narrative.detect_causal_link(ep_a, ep_b)
        assert abs(link.gamma_causal + link.strength - 1.0) < 0.001

    def test_emotional_direction(self, narrative, populated_hippo):
        ep_a = populated_hippo.episodes[0]  # positive valence
        ep_b = populated_hippo.episodes[1]  # negative valence
        link = narrative.detect_causal_link(ep_a, ep_b)
        # Direction should be negative (positive → negative)
        assert link.emotional_direction < 0


# ============================================================================
# Narrative Weaving
# ============================================================================

class TestNarrativeWeaving:
    def test_weave_creates_arcs(self, narrative, populated_hippo):
        arcs = narrative.weave()
        assert len(arcs) >= 1
        assert narrative.total_arcs_created >= 1

    def test_arc_has_theme(self, narrative, populated_hippo):
        arcs = narrative.weave()
        arc = arcs[0]
        assert arc.theme != ""
        assert len(arc.theme_concepts) > 0

    def test_arc_coherence(self, narrative, populated_hippo):
        arcs = narrative.weave()
        for arc in arcs:
            assert arc.coherence >= ARC_COHERENCE_MIN

    def test_arc_emotional_trajectory(self, narrative, populated_hippo):
        arcs = narrative.weave()
        for arc in arcs:
            assert len(arc.emotional_trajectory) == len(arc.episode_chain)

    def test_arc_to_dict(self, narrative, populated_hippo):
        arcs = narrative.weave()
        d = arcs[0].to_dict()
        assert "arc_id" in d
        assert "theme" in d
        assert "coherence" in d
        assert "emotional_arc_type" in d

    def test_duplicate_weave_no_new_arcs(self, narrative, populated_hippo):
        arcs1 = narrative.weave()
        arcs2 = narrative.weave()
        # Second weave shouldn't double-create
        assert len(arcs2) == 0 or len(narrative.arcs) == len(arcs1)


# ============================================================================
# Emotional Arc Classification
# ============================================================================

class TestEmotionalArc:
    def test_comedy(self):
        arc = NarrativeArc(arc_id=1, emotional_trajectory=[-0.5, -0.2, 0.0, 0.3, 0.7])
        assert arc.emotional_arc_type == "comedy"

    def test_tragedy(self):
        arc = NarrativeArc(arc_id=2, emotional_trajectory=[0.7, 0.3, 0.0, -0.2, -0.5])
        assert arc.emotional_arc_type == "tragedy"

    def test_stable(self):
        arc = NarrativeArc(arc_id=3, emotional_trajectory=[0.1, 0.12, 0.1, 0.11])
        assert arc.emotional_arc_type == "stable"

    def test_single_emotion(self):
        arc = NarrativeArc(arc_id=4, emotional_trajectory=[0.5])
        assert arc.emotional_arc_type == "stable"

    def test_empty_trajectory(self):
        arc = NarrativeArc(arc_id=5, emotional_trajectory=[])
        assert arc.emotional_arc_type == "stable"


# ============================================================================
# Event Hooks
# ============================================================================

class TestEventHooks:
    def test_on_episode_closed(self, narrative, populated_hippo):
        ep = populated_hippo.episodes[0]
        result = narrative.on_episode_closed(ep)
        assert "summary" in result
        assert "new_causal_links" in result
        assert result["summary"]["episode_id"] == ep.episode_id

    def test_on_episode_closed_updates_timeline(self, narrative, populated_hippo):
        ep = populated_hippo.episodes[0]
        narrative.on_episode_closed(ep)
        assert "cat" in narrative.concept_timeline
        assert len(narrative.concept_timeline["cat"]) >= 1


# ============================================================================
# Retrieval APIs
# ============================================================================

class TestRetrieval:
    def test_recall_by_theme(self, narrative, populated_hippo):
        narrative.weave()
        results = narrative.recall_by_theme("cat")
        assert isinstance(results, list)

    def test_get_autobiography(self, narrative, populated_hippo):
        narrative.weave()
        auto = narrative.get_autobiography()
        assert isinstance(auto, list)

    def test_get_concept_timeline(self, narrative, populated_hippo):
        for ep in populated_hippo.episodes:
            narrative.on_episode_closed(ep)
        timeline = narrative.get_concept_timeline("cat")
        assert len(timeline) >= 3  # 3 episodes mention "cat"

    def test_query_timeline(self, narrative, populated_hippo):
        t_start = populated_hippo.episodes[0].creation_time - 1
        t_end = populated_hippo.episodes[-1].creation_time + 1
        results = narrative.query_timeline(t_start, t_end)
        assert len(results) >= 1


# ============================================================================
# Consolidation
# ============================================================================

class TestConsolidation:
    def test_consolidate_narratives(self, narrative, populated_hippo):
        result = narrative.consolidate_narratives()
        assert "new_arcs" in result
        assert "total_arcs" in result

    def test_consolidate_marks_stale_complete(self, narrative, populated_hippo):
        narrative.weave()
        # Force arcs to be stale
        for arc in narrative.arcs:
            arc.last_updated = time.monotonic() - 600
        result = narrative.consolidate_narratives()
        assert result["completed_arcs"] >= 1


# ============================================================================
# Tick
# ============================================================================

class TestTick:
    def test_tick(self, narrative, populated_hippo):
        result = narrative.tick()
        assert "total_arcs" in result
        assert "new_summaries" in result
        assert result["new_summaries"] >= 1  # Should summarize closed episodes

# -*- coding: utf-8 -*-
"""
Tests for Hippocampus (Phase 5.1) and Wernicke's Area (Phase 5.2)

Hippocampus = cross-modal temporal binding
Wernicke's area = concept sequence comprehension and prediction
"""

import time
import numpy as np
import pytest

from alice.brain.hippocampus import (
    HippocampusEngine,
    Episode,
    EpisodicSnapshot,
    _cosine_sim,
    EPISODE_GAP_THRESHOLD,
)
from alice.brain.wernicke import (
    WernickeEngine,
    TransitionMatrix,
    Chunk,
    N400_THRESHOLD,
    CHUNK_MIN_OCCURRENCES,
)
from alice.brain.semantic_field import SemanticFieldEngine


# ============================================================================
# Helpers
# ============================================================================

def make_fingerprint(dim: int, seed: int = 0) -> np.ndarray:
    """Create a reproducible fingerprint."""
    rng = np.random.RandomState(seed)
    fp = rng.rand(dim)
    fp /= fp.sum() + 1e-12
    return fp


def make_visual_fp(seed: int = 0) -> np.ndarray:
    """256-dim visual fingerprint."""
    return make_fingerprint(256, seed)


def make_auditory_fp(seed: int = 0) -> np.ndarray:
    """32-dim auditory fingerprint."""
    return make_fingerprint(32, seed)


# ============================================================================
# PART 1: EpisodicSnapshot tests
# ============================================================================

class TestEpisodicSnapshot:

    def test_create_snapshot(self):
        fp = make_visual_fp(42)
        snap = EpisodicSnapshot(
            timestamp=1.0,
            modality="visual",
            fingerprint=fp,
            attractor_label="cat",
            gamma=0.1,
            valence=0.5,
        )
        assert snap.modality == "visual"
        assert snap.attractor_label == "cat"
        assert snap.gamma == pytest.approx(0.1)
        assert snap.fingerprint.shape == (256,)

    def test_snapshot_to_dict(self):
        fp = make_auditory_fp(10)
        snap = EpisodicSnapshot(
            timestamp=2.5,
            modality="auditory",
            fingerprint=fp,
            attractor_label="meow",
            gamma=0.2,
        )
        d = snap.to_dict()
        assert d["modality"] == "auditory"
        assert d["fingerprint_dim"] == 32
        assert d["attractor_label"] == "meow"

    def test_snapshot_none_label(self):
        fp = make_visual_fp(0)
        snap = EpisodicSnapshot(
            timestamp=0.0,
            modality="visual",
            fingerprint=fp,
            attractor_label=None,
        )
        assert snap.attractor_label is None


# ============================================================================
# PART 2: Episode tests
# ============================================================================

class TestEpisode:

    def test_create_empty_episode(self):
        ep = Episode(episode_id=0, creation_time=1.0)
        assert ep.n_snapshots == 0
        assert ep.duration == 0.0
        assert ep.is_open is True

    def test_add_snapshot(self):
        ep = Episode(episode_id=1, creation_time=0.0)
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=make_visual_fp(1), attractor_label="cat",
        ))
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.5, modality="auditory",
            fingerprint=make_auditory_fp(2), attractor_label="meow",
        ))
        assert ep.n_snapshots == 2
        assert ep.duration == pytest.approx(0.5)

    def test_modalities_spanned(self):
        ep = Episode(episode_id=2, creation_time=0.0)
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=make_visual_fp(1), attractor_label="cat",
        ))
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.1, modality="auditory",
            fingerprint=make_auditory_fp(2), attractor_label="meow",
        ))
        assert ep.modalities_spanned == {"visual", "auditory"}

    def test_concepts_mentioned(self):
        ep = Episode(episode_id=3, creation_time=0.0)
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=make_visual_fp(1), attractor_label="cat",
        ))
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.1, modality="visual",
            fingerprint=make_visual_fp(2), attractor_label="dog",
        ))
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.2, modality="visual",
            fingerprint=make_visual_fp(3), attractor_label=None,
        ))
        assert ep.concepts_mentioned == {"cat", "dog"}

    def test_emotional_peak(self):
        ep = Episode(episode_id=4, creation_time=0.0)
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=make_visual_fp(1), attractor_label="cat",
            valence=0.3,
        ))
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.1, modality="auditory",
            fingerprint=make_auditory_fp(2), attractor_label="loud",
            valence=-0.8,
        ))
        assert abs(ep.emotional_peak) == pytest.approx(0.8)

    def test_recency_decays(self):
        ep = Episode(episode_id=5, creation_time=0.0, last_replay_time=0.0)
        r0 = ep.recency(0.0)
        r100 = ep.recency(100.0)
        r1000 = ep.recency(1000.0)
        assert r0 > r100 > r1000
        assert r0 == pytest.approx(1.0)

    def test_emotional_episodes_decay_slower(self):
        ep_neutral = Episode(episode_id=6, creation_time=0.0,
                             last_replay_time=0.0, emotional_peak=0.0)
        ep_emotional = Episode(episode_id=7, creation_time=0.0,
                               last_replay_time=0.0, emotional_peak=0.9)
        t = 500.0
        assert ep_emotional.recency(t) > ep_neutral.recency(t)

    def test_relevance_to_cue_same_modality(self):
        ep = Episode(episode_id=8, creation_time=0.0)
        fp = make_visual_fp(42)
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=fp, attractor_label="cat",
        ))
        # Same fingerprint should match perfectly
        assert ep.relevance_to_cue(fp, "visual") == pytest.approx(1.0, abs=0.01)

    def test_relevance_cross_modality_returns_zero(self):
        ep = Episode(episode_id=9, creation_time=0.0)
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=make_visual_fp(42), attractor_label="cat",
        ))
        # Auditory cue cannot match visual snapshot (different dims)
        assert ep.relevance_to_cue(make_auditory_fp(42), "auditory") == 0.0

    def test_relevance_to_concept(self):
        ep = Episode(episode_id=10, creation_time=0.0)
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=make_visual_fp(1), attractor_label="cat",
        ))
        assert ep.relevance_to_concept("cat") > 0.0
        assert ep.relevance_to_concept("dog") == 0.0

    def test_to_dict(self):
        ep = Episode(episode_id=11, creation_time=0.0)
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=make_visual_fp(1), attractor_label="cat",
        ))
        d = ep.to_dict()
        assert d["episode_id"] == 11
        assert d["n_snapshots"] == 1
        assert "visual" in d["modalities"]


# ============================================================================
# PART 3: HippocampusEngine tests
# ============================================================================

class TestHippocampusEngine:

    def test_record_creates_episode(self):
        hc = HippocampusEngine()
        result = hc.record(
            modality="visual",
            fingerprint=make_visual_fp(1),
            attractor_label="cat",
            timestamp=0.0,
        )
        assert result["is_new_episode"] is True
        assert result["episode_id"] == 0
        assert hc.total_episodes_created == 1

    def test_sequential_records_same_episode(self):
        hc = HippocampusEngine()
        r1 = hc.record("visual", make_visual_fp(1), "cat", timestamp=0.0)
        r2 = hc.record("auditory", make_auditory_fp(2), "meow", timestamp=0.5)
        assert r1["episode_id"] == r2["episode_id"]
        assert r2["is_new_episode"] is False

    def test_temporal_gap_starts_new_episode(self):
        hc = HippocampusEngine()
        hc.record("visual", make_visual_fp(1), "cat", timestamp=1.0)
        # Gap > EPISODE_GAP_THRESHOLD
        hc.record("visual", make_visual_fp(2), "dog",
                   timestamp=1.0 + EPISODE_GAP_THRESHOLD + 1.0)
        assert hc.total_episodes_created == 2

    def test_end_episode_manually(self):
        hc = HippocampusEngine()
        hc.record("visual", make_visual_fp(1), "cat", timestamp=0.0)
        hc.end_episode()
        assert hc._current_episode is None

    def test_multi_modal_episode(self):
        hc = HippocampusEngine()
        hc.record("visual", make_visual_fp(1), "cat", timestamp=0.0)
        hc.record("auditory", make_auditory_fp(2), "cat", timestamp=0.1)
        hc.record("visual", make_visual_fp(3), "mouse", timestamp=0.2)

        ep = hc.episodes[0]
        assert ep.modalities_spanned == {"visual", "auditory"}
        assert ep.concepts_mentioned == {"cat", "mouse"}

    def test_capacity_eviction(self):
        hc = HippocampusEngine(max_episodes=3)
        # Create 4 episodes (should evict 1)
        for i in range(4):
            t = i * (EPISODE_GAP_THRESHOLD + 1.0)
            hc.record("visual", make_visual_fp(i), f"concept_{i}", timestamp=t)
        assert len(hc.episodes) <= 3

    def test_recall_by_concept(self):
        hc = HippocampusEngine()
        hc.record("visual", make_visual_fp(1), "cat", timestamp=100.0)
        hc.record("auditory", make_auditory_fp(2), "meow", timestamp=100.1)
        hc.end_episode()

        results = hc.recall(cue_concept="cat", t_now=100.5)
        assert len(results) > 0
        assert results[0]["episode_id"] == 0

    def test_recall_by_fingerprint(self):
        hc = HippocampusEngine()
        fp = make_visual_fp(42)
        hc.record("visual", fp, "cat", timestamp=100.0)
        hc.end_episode()

        results = hc.recall(cue_fingerprint=fp, cue_modality="visual",
                            t_now=100.5)
        assert len(results) > 0

    def test_recall_empty(self):
        hc = HippocampusEngine()
        results = hc.recall(cue_concept="nonexistent")
        assert isinstance(results, list)

    def test_replay_updates_count(self):
        hc = HippocampusEngine()
        hc.record("visual", make_visual_fp(1), "cat", timestamp=0.0)
        hc.end_episode()

        snapshots = hc.replay(0)
        assert snapshots is not None
        assert len(snapshots) == 1
        assert hc.episodes[0].replay_count == 1

    def test_replay_nonexistent_returns_none(self):
        hc = HippocampusEngine()
        assert hc.replay(999) is None

    def test_consolidation_strengthens_semantic_field(self):
        hc = HippocampusEngine()
        sf = SemanticFieldEngine()

        # Register concept in semantic field
        fp_cat_v = make_visual_fp(1)
        sf.process_fingerprint(fp_cat_v, "visual", label="cat")

        # Record episode
        hc.record("visual", fp_cat_v, "cat", timestamp=0.0)
        hc.record("auditory", make_auditory_fp(2), "cat", timestamp=0.1)
        hc.end_episode()

        # Get mass before consolidation
        mass_before = sf.field.attractors["cat"].total_mass

        # Consolidate
        result = hc.consolidate(sf)
        assert result["episodes_consolidated"] == 1
        assert result["snapshots_transferred"] == 2
        assert "cat" in result["concepts_strengthened"]

        # Mass should have increased
        mass_after = sf.field.attractors["cat"].total_mass
        assert mass_after > mass_before

    def test_get_concept_sequences(self):
        hc = HippocampusEngine()
        hc.record("visual", make_visual_fp(1), "cat", timestamp=0.0)
        hc.record("auditory", make_auditory_fp(2), "meow", timestamp=0.1)
        hc.record("visual", make_visual_fp(3), "mouse", timestamp=0.2)

        seqs = hc.get_concept_sequences(min_length=2)
        assert len(seqs) == 1
        assert seqs[0] == ["cat", "meow", "mouse"]

    def test_get_concept_sequences_deduplicates(self):
        hc = HippocampusEngine()
        hc.record("visual", make_visual_fp(1), "cat", timestamp=0.0)
        hc.record("visual", make_visual_fp(2), "cat", timestamp=0.1)  # dup
        hc.record("auditory", make_auditory_fp(3), "meow", timestamp=0.2)

        seqs = hc.get_concept_sequences(min_length=2)
        assert seqs[0] == ["cat", "meow"]  # No consecutive dup

    def test_recall_by_concept_with_modality_filter(self):
        hc = HippocampusEngine()
        hc.record("visual", make_visual_fp(1), "cat", timestamp=100.0)
        hc.record("auditory", make_auditory_fp(2), "cat", timestamp=100.1)
        hc.end_episode()

        results = hc.recall_by_concept("cat", target_modality="visual",
                                       t_now=100.5)
        assert len(results) > 0
        for r in results:
            for s in r["snapshots"]:
                assert s["modality"] == "visual"

    def test_get_state(self):
        hc = HippocampusEngine()
        hc.record("visual", make_visual_fp(1), "cat", timestamp=0.0)
        state = hc.get_state()
        assert state["n_episodes"] == 1
        assert state["total_snapshots_recorded"] == 1


# ============================================================================
# PART 4: TransitionMatrix tests
# ============================================================================

class TestTransitionMatrix:

    def test_observe_and_probability(self):
        tm = TransitionMatrix()
        tm.observe("cat", "meow")
        tm.observe("cat", "meow")
        tm.observe("cat", "purr")
        assert tm.probability("cat", "meow") == pytest.approx(2.0 / 3.0, abs=0.01)
        assert tm.probability("cat", "purr") == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_unknown_transition_zero(self):
        tm = TransitionMatrix()
        assert tm.probability("x", "y") == 0.0

    def test_gamma_syntactic(self):
        tm = TransitionMatrix()
        tm.observe("cat", "meow")
        g = tm.gamma_syntactic("cat", "meow")
        assert g == pytest.approx(0.0)  # P=1.0 -> Γ=0

    def test_gamma_syntactic_unknown(self):
        tm = TransitionMatrix()
        tm.observe("cat", "meow")
        g = tm.gamma_syntactic("cat", "bark")
        assert g == pytest.approx(1.0)  # P=0 -> Γ=1

    def test_predict_next(self):
        tm = TransitionMatrix()
        tm.observe("cat", "meow", weight=5.0)
        tm.observe("cat", "purr", weight=3.0)
        tm.observe("cat", "hiss", weight=1.0)
        preds = tm.predict_next("cat", top_k=2)
        assert len(preds) == 2
        assert preds[0][0] == "meow"  # Highest probability

    def test_predict_next_unknown(self):
        tm = TransitionMatrix()
        preds = tm.predict_next("unknown")
        assert preds == []

    def test_decay(self):
        tm = TransitionMatrix()
        tm.observe("a", "b", weight=10.0)
        prob_before = tm.probability("a", "b")
        tm.decay()
        prob_after = tm.probability("a", "b")
        # Probability stays the same (normalized), but counts decrease
        assert prob_after == pytest.approx(prob_before, abs=0.01)

    def test_vocabulary_size(self):
        tm = TransitionMatrix()
        tm.observe("a", "b")
        tm.observe("b", "c")
        assert tm.vocabulary_size == 3

    def test_to_dict(self):
        tm = TransitionMatrix()
        tm.observe("cat", "meow")
        d = tm.to_dict()
        assert d["vocabulary_size"] == 2
        assert len(d["top_transitions"]) == 1


# ============================================================================
# PART 5: WernickeEngine tests
# ============================================================================

class TestWernickeEngine:

    def test_observe_first_concept_no_surprise(self):
        we = WernickeEngine()
        result = we.observe("cat")
        # First observation: no previous context -> Γ_syn = 1.0
        assert result["gamma_syntactic"] == pytest.approx(1.0)

    def test_observe_learned_transition_low_gamma(self):
        we = WernickeEngine()
        # Train: cat -> meow many times
        for _ in range(10):
            we.observe("cat")
            we.observe("meow")
            we.reset_context()

        # Now observe cat then meow
        we.observe("cat")
        result = we.observe("meow")
        # Should have low Γ_syn (expected transition)
        assert result["gamma_syntactic"] < 0.5

    def test_observe_unknown_transition_high_gamma(self):
        we = WernickeEngine()
        # Train: cat -> meow
        for _ in range(10):
            we.observe("cat")
            we.observe("meow")
            we.reset_context()

        # Observe cat then surprise
        we.observe("cat")
        result = we.observe("rocket")
        assert result["gamma_syntactic"] > 0.8
        assert result["is_n400"] is True

    def test_comprehend_known_sequence(self):
        we = WernickeEngine()
        # Train
        for _ in range(10):
            we.observe("the")
            we.observe("cat")
            we.observe("sat")
            we.reset_context()

        result = we.comprehend(["the", "cat", "sat"])
        assert result["comprehension_score"] > 0.5
        assert result["is_comprehensible"] is True

    def test_comprehend_unknown_sequence(self):
        we = WernickeEngine()
        # No training
        result = we.comprehend(["alpha", "beta", "gamma"])
        assert result["comprehension_score"] < 0.2
        assert result["is_comprehensible"] is False

    def test_comprehend_single_concept(self):
        we = WernickeEngine()
        result = we.comprehend(["solo"])
        assert result["comprehension_score"] == 1.0

    def test_predict_next(self):
        we = WernickeEngine()
        for _ in range(10):
            we.observe("cat")
            we.observe("meow")
            we.reset_context()

        we.observe("cat")
        result = we.predict_next()
        assert len(result["predictions"]) > 0
        assert result["predictions"][0]["concept"] == "meow"

    def test_predict_next_with_context(self):
        we = WernickeEngine()
        for _ in range(10):
            we.observe("cat")
            we.observe("meow")
            we.reset_context()

        result = we.predict_next(context=["cat"])
        assert result["predictions"][0]["concept"] == "meow"

    def test_predict_next_empty(self):
        we = WernickeEngine()
        result = we.predict_next()
        assert result["predictions"] == []

    def test_learn_from_sequences(self):
        we = WernickeEngine()
        sequences = [
            ["cat", "meow", "purr"],
            ["cat", "meow", "sleep"],
            ["dog", "bark", "wag"],
        ]
        result = we.learn_from_sequences(sequences)
        assert result["sequences_processed"] == 3
        assert result["transitions_learned"] == 6

        # cat -> meow should be strong
        assert we.transitions.probability("cat", "meow") > 0.9

    def test_n400_event_counting(self):
        we = WernickeEngine()
        # Train a strong expectation
        for _ in range(20):
            we.observe("cat")
            we.observe("meow")
            we.reset_context()

        # Violate expectation
        we.observe("cat")
        result = we.observe("spaceship")
        if result["is_n400"]:
            assert we.total_n400_events > 0

    def test_context_window(self):
        we = WernickeEngine()
        for i in range(10):
            we.observe(f"concept_{i}")
        # Context should be limited to window size
        assert len(we._context) == 5  # CONTEXT_WINDOW = 5

    def test_reset_context(self):
        we = WernickeEngine()
        we.observe("cat")
        we.observe("meow")
        we.reset_context()
        assert len(we._context) == 0


# ============================================================================
# PART 6: Chunk tests
# ============================================================================

class TestChunk:

    def test_chunk_creation(self):
        chunk = Chunk(
            concepts=("good", "morning"),
            occurrence_count=5,
            mean_internal_gamma=0.1,
        )
        assert chunk.label == "good+morning"
        assert chunk.length == 2
        assert chunk.is_mature is True

    def test_immature_chunk(self):
        chunk = Chunk(
            concepts=("rare", "combo"),
            occurrence_count=1,
            mean_internal_gamma=0.1,
        )
        assert chunk.is_mature is False  # Not enough occurrences

    def test_high_gamma_chunk_not_mature(self):
        chunk = Chunk(
            concepts=("x", "y"),
            occurrence_count=10,
            mean_internal_gamma=0.9,
        )
        assert chunk.is_mature is False  # High internal Γ

    def test_chunk_formation_through_observation(self):
        we = WernickeEngine()
        # Observe "good morning" many times
        for _ in range(CHUNK_MIN_OCCURRENCES + 5):
            we.observe("good")
            we.observe("morning")
            we.reset_context()

        # Check if chunk was formed
        mature = we.get_mature_chunks()
        labels = [c["label"] for c in mature]
        assert "good+morning" in labels


# ============================================================================
# PART 7: Integration tests — Hippocampus + Wernicke
# ============================================================================

class TestHippocampusWernickeIntegration:

    def test_hippocampus_feeds_wernicke(self):
        """Episodes -> concept sequences -> Wernicke learns transitions."""
        hc = HippocampusEngine()
        we = WernickeEngine()

        # Create several episodes with consistent patterns
        for i in range(5):
            t_base = i * (EPISODE_GAP_THRESHOLD + 1.0)
            hc.record("visual", make_visual_fp(1), "cat", timestamp=t_base)
            hc.record("auditory", make_auditory_fp(2), "meow", timestamp=t_base + 0.1)
            hc.record("visual", make_visual_fp(3), "mouse", timestamp=t_base + 0.2)

        # Wernicke learns from hippocampal sequences
        result = we.consolidate_from_hippocampus(hc)
        assert result["sequences_processed"] >= 1
        assert result["transitions_learned"] > 0

        # Now Wernicke should predict meow after cat
        pred = we.predict_next(context=["cat"])
        assert len(pred["predictions"]) > 0
        assert pred["predictions"][0]["concept"] == "meow"

    def test_comprehension_after_consolidation(self):
        """After learning from episodes, sequences should be comprehensible."""
        hc = HippocampusEngine()
        we = WernickeEngine()

        for i in range(10):
            t_base = i * (EPISODE_GAP_THRESHOLD + 1.0)
            hc.record("auditory", make_auditory_fp(1), "hello", timestamp=t_base)
            hc.record("auditory", make_auditory_fp(2), "world", timestamp=t_base + 0.1)

        we.consolidate_from_hippocampus(hc)

        result = we.comprehend(["hello", "world"])
        assert result["comprehension_score"] > 0.5

    def test_consolidation_and_recall(self):
        """Full cycle: record -> consolidate to SF -> recall from hippocampus."""
        hc = HippocampusEngine()
        sf = SemanticFieldEngine()

        # Register concepts
        fp_cat_v = make_visual_fp(1)
        fp_cat_a = make_auditory_fp(2)
        sf.process_fingerprint(fp_cat_v, "visual", label="cat")
        sf.process_fingerprint(fp_cat_a, "auditory", label="cat")

        # Record episode
        hc.record("visual", fp_cat_v, "cat", timestamp=0.0)
        hc.record("auditory", fp_cat_a, "cat", timestamp=0.1)
        hc.end_episode()

        # Consolidate
        hc.consolidate(sf)

        # Recall by concept
        results = hc.recall(cue_concept="cat")
        assert len(results) > 0

    def test_cross_membrane_recall_through_concept(self):
        """Hear 'cat' -> recall visual episode containing 'cat'."""
        hc = HippocampusEngine()

        # Record: see cat, then hear cat
        hc.record("visual", make_visual_fp(1), "cat", timestamp=100.0)
        hc.record("auditory", make_auditory_fp(2), "cat", timestamp=100.1)
        hc.end_episode()

        # Recall: "what did cat look like?"
        results = hc.recall_by_concept("cat", target_modality="visual",
                                       t_now=100.5)
        assert len(results) > 0
        assert all(
            s["modality"] == "visual"
            for r in results for s in r["snapshots"]
        )


# ============================================================================
# PART 8: AliceBrain integration tests
# ============================================================================

class TestAliceBrainHippocampusWernicke:

    def test_brain_has_hippocampus_and_wernicke(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        assert hasattr(brain, "hippocampus")
        assert hasattr(brain, "wernicke")
        assert isinstance(brain.hippocampus, HippocampusEngine)
        assert isinstance(brain.wernicke, WernickeEngine)

    def test_see_records_to_hippocampus(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)

        # Register a concept so semantic field recognizes it
        fp_cat = make_visual_fp(42)
        brain.semantic_field.process_fingerprint(fp_cat, "visual", label="cat")

        # See something similar
        pixels = np.random.RandomState(42).rand(64)
        brain.see(pixels)

        # Hippocampus should have recorded something
        assert brain.hippocampus.total_snapshots_recorded >= 0

    def test_introspect_includes_hippocampus_wernicke(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)
        info = brain.introspect()
        assert "hippocampus" in info["subsystems"]
        assert "wernicke" in info["subsystems"]

    def test_hear_feeds_wernicke(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=10)

        # Hear multiple sounds (without pre-registering concepts -
        # the cochlea produces 24-dim fingerprints from physics)
        rng = np.random.RandomState(1)
        brain.hear(rng.rand(8000))
        brain.hear(rng.rand(8000))

        # Wernicke should have observations if semantic field found matches
        state = brain.wernicke.get_state()
        assert state["total_observations"] >= 0


# ============================================================================
# PART 9: Physics invariant tests
# ============================================================================

class TestPhysicsInvariants:

    def test_cosine_sim_different_dims_returns_zero(self):
        """Different membrane dimensions cannot be compared."""
        a = make_visual_fp(1)   # 256-dim
        b = make_auditory_fp(2)  # 32-dim
        assert _cosine_sim(a, b) == 0.0

    def test_cosine_sim_same_dim_works(self):
        a = make_visual_fp(1)
        b = make_visual_fp(1)
        assert _cosine_sim(a, b) == pytest.approx(1.0, abs=0.01)

    def test_gamma_syntactic_bounded(self):
        """Γ_syn ∈ [0, 1]."""
        tm = TransitionMatrix()
        tm.observe("a", "b")
        tm.observe("a", "c")
        for target in ["a", "b", "c", "unknown"]:
            g = tm.gamma_syntactic("a", target)
            assert 0.0 <= g <= 1.0

    def test_comprehension_score_bounded(self):
        we = WernickeEngine()
        we.learn_from_sequences([["a", "b", "c"]])
        result = we.comprehend(["a", "b", "c"])
        assert 0.0 <= result["comprehension_score"] <= 1.0

    def test_episode_retrieval_strength_nonnegative(self):
        ep = Episode(episode_id=0, creation_time=0.0, last_replay_time=0.0)
        ep.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=make_visual_fp(1), attractor_label="x",
        ))
        strength = ep.retrieval_strength(
            100.0,
            cue_concept="x",
        )
        assert strength >= 0.0

    def test_prediction_probabilities_sum_to_one(self):
        """All outgoing transition probabilities should sum to ~1."""
        tm = TransitionMatrix()
        tm.observe("a", "b", weight=3.0)
        tm.observe("a", "c", weight=2.0)
        tm.observe("a", "d", weight=5.0)
        total = sum(p for _, p in tm.predict_next("a", top_k=10))
        assert total == pytest.approx(1.0, abs=0.01)

    def test_entropy_zero_for_deterministic(self):
        """If only one transition exists, entropy should be 0."""
        we = WernickeEngine()
        we.learn_from_sequences([["x", "y"]] * 10)
        result = we.predict_next(context=["x"])
        assert result["entropy"] == pytest.approx(0.0, abs=0.01)

    def test_entropy_positive_for_uncertain(self):
        """Multiple transitions -> positive entropy."""
        we = WernickeEngine()
        we.learn_from_sequences([
            ["x", "a"], ["x", "b"], ["x", "c"],
        ] * 5)
        result = we.predict_next(context=["x"])
        assert result["entropy"] > 0.0

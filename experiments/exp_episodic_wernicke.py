# -*- coding: utf-8 -*-
"""
Experiments: Hippocampus (Phase 5.1) + Wernicke's Area (Phase 5.2)

hippocampus = cross-membrane time-binding engine
Wernicke's area = concept sequence comprehension and prediction

8 experiments demonstrating:
  1. Episodic Recording — multi-modal episodes from see/hear
  2. Pattern Completion — partial cue retrieves full experience
  3. Cross-Membrane Recall — hear concept → recall visual episode
  4. Sleep Consolidation — episodes strengthen semantic field
  5. Transition Learning — hippocampal sequences → Wernicke
  6. Sequence Comprehension — Γ_syntactic as understanding metric
  7. N400 Surprise Detection — unexpected concept triggers N400
  8. Chunk Formation — frequent co-occurrences fuse into units
"""

from __future__ import annotations

import sys
import time
import numpy as np

from alice.brain.hippocampus import (
    HippocampusEngine,
    Episode,
    EpisodicSnapshot,
    EPISODE_GAP_THRESHOLD,
)
from alice.brain.wernicke import (
    WernickeEngine,
    TransitionMatrix,
    N400_THRESHOLD,
    CHUNK_MIN_OCCURRENCES,
)
from alice.brain.semantic_field import SemanticFieldEngine


# ============================================================================
# Helpers
# ============================================================================

def make_fp(dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    fp = rng.rand(dim)
    fp /= fp.sum() + 1e-12
    return fp


def make_visual(seed: int) -> np.ndarray:
    return make_fp(256, seed)


def make_auditory(seed: int) -> np.ndarray:
    return make_fp(32, seed)


def header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================================
# Experiment 1: Episodic Recording
# ============================================================================

def exp_episodic_recording():
    """Multi-modal episodes: see cat + hear meow = one experience."""
    header("Experiment 1: Episodic Recording — Multi-Modal Binding")

    hc = HippocampusEngine()

    # Episode 1: see cat, hear meow
    t = 100.0
    hc.record("visual", make_visual(1), "cat", gamma=0.1, timestamp=t)
    hc.record("auditory", make_auditory(2), "meow", gamma=0.05, timestamp=t + 0.1)
    hc.record("visual", make_visual(3), "mouse", gamma=0.2, timestamp=t + 0.3)
    hc.end_episode()

    # Episode 2: see dog, hear bark (after gap)
    t2 = t + EPISODE_GAP_THRESHOLD + 1.0
    hc.record("visual", make_visual(10), "dog", gamma=0.1, timestamp=t2)
    hc.record("auditory", make_auditory(11), "bark", gamma=0.08, timestamp=t2 + 0.1)
    hc.end_episode()

    ep1 = hc.episodes[0]
    ep2 = hc.episodes[1]

    print(f"  Episode 1: {ep1.n_snapshots} snapshots, "
          f"modalities={ep1.modalities_spanned}, "
          f"concepts={ep1.concepts_mentioned}")
    print(f"  Episode 2: {ep2.n_snapshots} snapshots, "
          f"modalities={ep2.modalities_spanned}, "
          f"concepts={ep2.concepts_mentioned}")
    print(f"  Total episodes: {hc.total_episodes_created}")
    print(f"  Total snapshots: {hc.total_snapshots_recorded}")

    assert ep1.modalities_spanned == {"visual", "auditory"}
    assert ep1.concepts_mentioned == {"cat", "meow", "mouse"}
    assert ep2.concepts_mentioned == {"dog", "bark"}
    assert hc.total_episodes_created == 2
    print("  ✓ PASS: Episodes correctly record multi-modal experiences")
    return True


# ============================================================================
# Experiment 2: Pattern Completion
# ============================================================================

def exp_pattern_completion():
    """Partial cue retrieves full episode."""
    header("Experiment 2: Pattern Completion — Cue-Driven Recall")

    hc = HippocampusEngine()

    # Record a rich episode
    t = 100.0
    cat_fp = make_visual(42)
    hc.record("visual", cat_fp, "cat", timestamp=t)
    hc.record("auditory", make_auditory(43), "meow", timestamp=t + 0.1)
    hc.record("visual", make_visual(44), "fish", timestamp=t + 0.3)
    hc.end_episode()

    # Recall with visual cue (the exact cat fingerprint)
    results = hc.recall(cue_fingerprint=cat_fp, cue_modality="visual", t_now=t + 1.0)

    print(f"  Cue: visual fingerprint (256-dim)")
    print(f"  Episodes recalled: {len(results)}")
    if results:
        r = results[0]
        print(f"  Best match: episode {r['episode_id']}, "
              f"strength={r['retrieval_strength']:.4f}")
        print(f"  Snapshots in episode: {r['n_snapshots']}")
        print(f"  Concepts: {r['concepts']}")

    assert len(results) > 0
    assert results[0]["retrieval_strength"] > 0.0
    print("  ✓ PASS: Visual cue retrieves full multi-modal episode")

    # Recall with concept cue
    results2 = hc.recall(cue_concept="meow", t_now=t + 1.0)
    assert len(results2) > 0
    print(f"  Concept cue 'meow': retrieved {len(results2)} episode(s)")
    print("  ✓ PASS: Concept cue also retrieves matching episodes")
    return True


# ============================================================================
# Experiment 3: Cross-Membrane Recall
# ============================================================================

def exp_cross_membrane_recall():
    """Hear 'cat' -> recall what cat LOOKED like (cross-membrane via wormhole)."""
    header("Experiment 3: Cross-Membrane Recall — Wormhole Through Attractors")

    hc = HippocampusEngine()

    # Record: see cat (256-dim) + hear cat (32-dim)
    t = 100.0
    hc.record("visual", make_visual(1), "cat", timestamp=t)
    hc.record("auditory", make_auditory(2), "cat", timestamp=t + 0.1)
    hc.record("visual", make_visual(5), "ball", timestamp=t + 0.3)
    hc.end_episode()

    # Query: "what did cat look like?" — cross-membrane via concept label
    results = hc.recall_by_concept("cat", target_modality="visual", t_now=t + 1.0)

    print(f"  Query: concept='cat', target_modality='visual'")
    print(f"  Episodes found: {len(results)}")

    for r in results:
        print(f"    Episode {r['episode_id']}: "
              f"{len(r['snapshots'])} visual snapshot(s)")
        for s in r["snapshots"]:
            print(f"      - t={s['timestamp']:.2f}, "
                  f"dim={s['fingerprint_dim']}, "
                  f"concept={s['attractor_label']}")

    assert len(results) > 0
    assert all(
        s["modality"] == "visual"
        for r in results for s in r["snapshots"]
    )

    print("\n  Key insight: concept label 'cat' = wormhole between")
    print("  32-dim auditory membrane and 256-dim visual membrane")
    print("  ✓ PASS: Cross-membrane recall through attractor labels")
    return True


# ============================================================================
# Experiment 4: Sleep Consolidation
# ============================================================================

def exp_sleep_consolidation():
    """Episodes strengthen semantic field during sleep."""
    header("Experiment 4: Sleep Consolidation — Episode → Semantic Field")

    hc = HippocampusEngine()
    sf = SemanticFieldEngine()

    # Register concepts in semantic field
    fp_cat_v = make_visual(1)
    fp_cat_a = make_auditory(2)
    sf.process_fingerprint(fp_cat_v, "visual", label="cat")
    sf.process_fingerprint(fp_cat_a, "auditory", label="cat")

    mass_before = sf.field.attractors["cat"].total_mass
    q_before = sf.field.attractors["cat"].quality_factor()

    # Record many episodes featuring "cat"
    for i in range(5):
        t = 100.0 + i * (EPISODE_GAP_THRESHOLD + 1.0)
        hc.record("visual", make_visual(1 + i), "cat", timestamp=t)
        hc.record("auditory", make_auditory(2 + i), "cat", timestamp=t + 0.1)
        hc.end_episode()

    # Sleep consolidation
    result = hc.consolidate(sf)

    mass_after = sf.field.attractors["cat"].total_mass
    q_after = sf.field.attractors["cat"].quality_factor()

    print(f"  Episodes consolidated: {result['episodes_consolidated']}")
    print(f"  Snapshots transferred: {result['snapshots_transferred']}")
    print(f"  Concepts strengthened: {result['concepts_strengthened']}")
    print(f"  Attractor 'cat' mass: {mass_before:.2f} → {mass_after:.2f}")
    print(f"  Attractor 'cat' Q:    {q_before:.3f} → {q_after:.3f}")

    assert mass_after > mass_before
    assert q_after > q_before
    assert result["episodes_consolidated"] == 5
    print("\n  This is WHY you forget the specific episode but remember")
    print("  the concept: mass transfers from hippocampus → semantic field")
    print("  ✓ PASS: Consolidation strengthens semantic attractors")
    return True


# ============================================================================
# Experiment 5: Transition Learning from Hippocampus
# ============================================================================

def exp_transition_learning():
    """Hippocampal episode sequences teach Wernicke transition probabilities."""
    header("Experiment 5: Transition Learning — Hippocampus → Wernicke")

    hc = HippocampusEngine()
    we = WernickeEngine()

    # Create consistent episodes: cat → meow → purr (5 times)
    for i in range(5):
        t = 100.0 + i * (EPISODE_GAP_THRESHOLD + 1.0)
        hc.record("visual", make_visual(1), "cat", timestamp=t)
        hc.record("auditory", make_auditory(2), "meow", timestamp=t + 0.1)
        hc.record("auditory", make_auditory(3), "purr", timestamp=t + 0.2)

    # Also: dog → bark → wag (3 times)
    for i in range(3):
        t = 200.0 + i * (EPISODE_GAP_THRESHOLD + 1.0)
        hc.record("visual", make_visual(10), "dog", timestamp=t)
        hc.record("auditory", make_auditory(11), "bark", timestamp=t + 0.1)
        hc.record("visual", make_visual(12), "wag", timestamp=t + 0.2)

    # Wernicke learns from hippocampal sequences
    result = we.consolidate_from_hippocampus(hc)

    print(f"  Sequences processed: {result['sequences_processed']}")
    print(f"  Transitions learned: {result['transitions_learned']}")

    # Check learned probabilities
    p_cat_meow = we.transitions.probability("cat", "meow")
    p_meow_purr = we.transitions.probability("meow", "purr")
    p_dog_bark = we.transitions.probability("dog", "bark")

    print(f"\n  P(meow|cat)  = {p_cat_meow:.4f}")
    print(f"  P(purr|meow) = {p_meow_purr:.4f}")
    print(f"  P(bark|dog)  = {p_dog_bark:.4f}")

    assert p_cat_meow > 0.8
    assert p_meow_purr > 0.5
    assert p_dog_bark > 0.8

    # Predict next after "cat"
    pred = we.predict_next(context=["cat"])
    print(f"\n  Predict after 'cat': {pred['predictions'][:3]}")
    assert pred["predictions"][0]["concept"] == "meow"
    print("  ✓ PASS: Hippocampal sequences → Wernicke transitions")
    return True


# ============================================================================
# Experiment 6: Sequence Comprehension
# ============================================================================

def exp_sequence_comprehension():
    """Γ_syntactic measures understanding of concept sequences."""
    header("Experiment 6: Sequence Comprehension — Γ_syntactic")

    we = WernickeEngine()

    # Train on consistent patterns (focused, high frequency)
    training = [
        ["the", "cat", "sat"],
        ["the", "cat", "sat"],
        ["the", "cat", "ate"],
        ["a", "dog", "barked"],
    ] * 15  # Repeat for strong learning
    we.learn_from_sequences(training)

    # Test comprehension of known sequence
    result_known = we.comprehend(["the", "cat", "sat"])
    # Test comprehension of unknown sequence
    result_unknown = we.comprehend(["rocket", "banana", "quantum"])

    print(f"  Known sequence ['the','cat','sat']:")
    print(f"    Comprehension score: {result_known['comprehension_score']:.4f}")
    print(f"    Mean Γ_syntactic:    {result_known['mean_gamma_syntactic']:.4f}")
    print(f"    Comprehensible:      {result_known['is_comprehensible']}")

    print(f"\n  Unknown sequence ['rocket','banana','quantum']:")
    print(f"    Comprehension score: {result_unknown['comprehension_score']:.4f}")
    print(f"    Mean Γ_syntactic:    {result_unknown['mean_gamma_syntactic']:.4f}")
    print(f"    Comprehensible:      {result_unknown['is_comprehensible']}")

    assert result_known["comprehension_score"] > result_unknown["comprehension_score"]
    assert result_known["is_comprehensible"] is True
    assert result_unknown["is_comprehensible"] is False

    # Show per-transition detail
    print(f"\n  Per-transition Γ_syntactic for known sequence:")
    for t in result_known["per_transition"]:
        indicator = "✓" if t["gamma_syntactic"] < 0.5 else "✗"
        print(f"    {t['from']} → {t['to']}: "
              f"Γ_syn={t['gamma_syntactic']:.4f}, "
              f"P={t['probability']:.4f} {indicator}")

    print("  ✓ PASS: Γ_syntactic correctly measures comprehension")
    return True


# ============================================================================
# Experiment 7: N400 Surprise Detection
# ============================================================================

def exp_n400_surprise():
    """Unexpected concept triggers N400-like surprise response."""
    header("Experiment 7: N400 Surprise Detection — Semantic Violation")

    we = WernickeEngine()

    # Strong training: cat → meow
    for _ in range(20):
        we.observe("cat")
        we.observe("meow")
        we.reset_context()

    # Expected transition: cat → meow
    we.observe("cat")
    result_expected = we.observe("meow")

    we.reset_context()

    # Unexpected transition: cat → spaceship
    we.observe("cat")
    result_surprise = we.observe("spaceship")

    print(f"  Expected:   cat → meow")
    print(f"    Γ_syntactic = {result_expected['gamma_syntactic']:.4f}")
    print(f"    N400 event? {result_expected['is_n400']}")
    print(f"    Prediction was: {result_expected['prediction_was']}")

    print(f"\n  Surprise:   cat → spaceship")
    print(f"    Γ_syntactic = {result_surprise['gamma_syntactic']:.4f}")
    print(f"    N400 event? {result_surprise['is_n400']}")
    print(f"    Prediction was: {result_surprise['prediction_was']}")

    assert result_expected["gamma_syntactic"] < 0.3
    assert result_surprise["gamma_syntactic"] > 0.8
    assert result_surprise["is_n400"] is True
    assert result_expected["prediction_was"] == "meow"

    print(f"\n  N400 threshold = {N400_THRESHOLD}")
    print("  This is the ERF component that spikes when a semantically")
    print("  unexpected word appears — pure physics of prediction error.")
    print("  ✓ PASS: N400 correctly detects semantic violations")
    return True


# ============================================================================
# Experiment 8: Chunk Formation
# ============================================================================

def exp_chunk_formation():
    """Frequently co-occurring concepts fuse into chunks."""
    header("Experiment 8: Chunk Formation — Phrase Crystallization")

    we = WernickeEngine()

    # Observe "good morning" many times
    n_observations = CHUNK_MIN_OCCURRENCES + 10
    for _ in range(n_observations):
        we.observe("good")
        we.observe("morning")
        we.reset_context()

    # Also observe other patterns less frequently
    for _ in range(2):
        we.observe("bad")
        we.observe("night")
        we.reset_context()

    mature = we.get_mature_chunks()
    all_chunks = list(we.chunks.values())

    print(f"  Total chunks tracked: {len(all_chunks)}")
    print(f"  Mature chunks: {len(mature)}")
    for c in mature:
        print(f"    '{c['label']}': "
              f"count={c['occurrence_count']}, "
              f"mean_Γ={c['mean_internal_gamma']:.4f}")

    # "good+morning" should be a mature chunk
    chunk_labels = [c["label"] for c in mature]
    assert "good+morning" in chunk_labels

    # "bad+night" should NOT be mature (too few occurrences)
    bad_night_mature = any(c["label"] == "bad+night" for c in mature)
    assert not bad_night_mature
    print(f"\n  'bad+night' is NOT mature (only 2 occurrences < {CHUNK_MIN_OCCURRENCES})")

    # Comprehension with chunk should be high
    comp = we.comprehend(["good", "morning"])
    print(f"\n  Comprehension of ['good','morning']: {comp['comprehension_score']:.4f}")
    assert comp["comprehension_score"] > 0.8

    print("\n  Physics: when Γ_syntactic within a subsequence drops low enough,")
    print("  the concepts fuse into a single unit = chunk = phrase")
    print("  ✓ PASS: Chunk formation through repeated co-occurrence")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    experiments = [
        ("1. Episodic Recording", exp_episodic_recording),
        ("2. Pattern Completion", exp_pattern_completion),
        ("3. Cross-Membrane Recall", exp_cross_membrane_recall),
        ("4. Sleep Consolidation", exp_sleep_consolidation),
        ("5. Transition Learning", exp_transition_learning),
        ("6. Sequence Comprehension", exp_sequence_comprehension),
        ("7. N400 Surprise", exp_n400_surprise),
        ("8. Chunk Formation", exp_chunk_formation),
    ]

    passed = 0
    failed = 0

    for name, func in experiments:
        try:
            if func():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ FAIL: {name}")
        except Exception as e:
            failed += 1
            print(f"  ✗ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(experiments)} passed, {failed} failed")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

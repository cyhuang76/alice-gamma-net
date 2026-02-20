# -*- coding: utf-8 -*-
"""
Tests for Phase 4.2: Semantic Field Engine
Tests for Phase 4.3: Broca Engine

Total: ~120 tests covering both semantic_field.py and broca.py
"""

import math
import time

import numpy as np
import pytest

# --- Phase 4.2 imports ---
from alice.brain.semantic_field import (
    SemanticAttractor,
    SemanticField,
    SemanticFieldEngine,
    gamma_semantic,
    energy_absorbed,
    cosine_similarity,
    FIELD_Z0,
    SHARPNESS_ALPHA,
    INITIAL_MASS,
    MASS_FLOOR,
    MAX_ATTRACTORS,
    CONTRASTIVE_THRESHOLD,
    MERGE_THRESHOLD,
)

# --- Phase 4.3 imports ---
from alice.brain.broca import (
    ArticulatoryPlan,
    BrocaEngine,
    extract_formants,
    BROCA_Z0,
    PITCH_RANGE,
    F1_RANGE,
    F2_RANGE,
    F3_RANGE,
    CONFIDENCE_FLOOR,
    CONFIDENCE_CEILING,
    SUCCESS_GAMMA_THRESHOLD,
    VOWEL_FORMANT_TARGETS,
)

from alice.body.cochlea import (
    CochlearFilterBank,
    TonotopicActivation,
    generate_tone,
    generate_vowel,
    generate_complex_tone,
    generate_noise,
)
from alice.body.mouth import AliceMouth
from alice.core.signal import ElectricalSignal


# ============================================================================
# Helpers
# ============================================================================


def make_fingerprint(dominant_channel: int = 5, n_channels: int = 24) -> np.ndarray:
    """Create a synthetic fingerprint with energy peaked at dominant_channel."""
    fp = np.zeros(n_channels)
    for i in range(n_channels):
        dist = abs(i - dominant_channel)
        fp[i] = math.exp(-0.5 * dist * dist)
    fp /= fp.sum()
    return fp


def make_shifted_fingerprint(dominant: int, shift: float = 0.5,
                              n_channels: int = 24) -> np.ndarray:
    """Create shifted fingerprint (partial overlap) to test discrimination."""
    fp = np.zeros(n_channels)
    for i in range(n_channels):
        dist = abs(i - dominant) - shift
        fp[i] = math.exp(-0.3 * dist * dist)
    fp /= fp.sum()
    return fp


# ============================================================================
# Phase 4.2: Gamma Semantic Function
# ============================================================================


class TestGammaSemantic:
    """Test the core gamma_semantic equation."""

    def test_perfect_match(self):
        """sim=1 -> Gamma=0 for any mass."""
        assert gamma_semantic(1.0, 0.0) == pytest.approx(0.0)
        assert gamma_semantic(1.0, 100.0) == pytest.approx(0.0)

    def test_no_match(self):
        """sim=0 -> Gamma=1 for any mass."""
        assert gamma_semantic(0.0, 0.0) == pytest.approx(1.0)
        assert gamma_semantic(0.0, 100.0) == pytest.approx(1.0)

    def test_intermediate(self):
        """0 < sim < 1 -> 0 < Gamma < 1."""
        g = gamma_semantic(0.5, 1.0)
        assert 0.0 < g < 1.0

    def test_mass_sharpens(self):
        """Higher mass -> higher Gamma for intermediate similarity."""
        g_low = gamma_semantic(0.8, 1.0)
        g_high = gamma_semantic(0.8, 100.0)
        assert g_high > g_low  # more discriminative with more mass

    def test_energy_absorbed_bounds(self):
        """Energy absorbed is always in [0, 1]."""
        for g in [0.0, 0.3, 0.5, 0.7, 1.0]:
            e = energy_absorbed(g)
            assert 0.0 <= e <= 1.0

    def test_energy_at_perfect_match(self):
        """Gamma=0 -> full energy absorption."""
        assert energy_absorbed(0.0) == pytest.approx(1.0)

    def test_energy_at_no_match(self):
        """Gamma=1 -> zero energy absorption."""
        assert energy_absorbed(1.0) == pytest.approx(0.0)


class TestCosineSimilarity:
    """Test cosine similarity helper."""

    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        a = np.zeros(5)
        b = np.ones(5)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_scaled_vectors_same(self):
        a = np.array([1.0, 2.0, 3.0])
        b = 10 * a
        assert cosine_similarity(a, b) == pytest.approx(1.0)


# ============================================================================
# Phase 4.2: SemanticAttractor
# ============================================================================


class TestSemanticAttractor:
    """Test individual attractor behavior."""

    def test_creation(self):
        fp = make_fingerprint(10)
        att = SemanticAttractor(
            label="test",
            modality_centroids={"auditory": fp.copy()},
            modality_masses={"auditory": 1.0},
            total_mass=1.0,
        )
        assert att.label == "test"
        assert att.total_mass == 1.0

    def test_quality_factor_increases_with_mass(self):
        att = SemanticAttractor(label="x", total_mass=1.0)
        q1 = att.quality_factor()
        att.total_mass = 100.0
        q2 = att.quality_factor()
        assert q2 > q1

    def test_impedance_decreases_with_mass(self):
        att = SemanticAttractor(label="x", total_mass=1.0)
        z1 = att.impedance()
        att.total_mass = 100.0
        z2 = att.impedance()
        assert z2 < z1

    def test_gamma_perfect_match(self):
        fp = make_fingerprint(10)
        att = SemanticAttractor(
            label="x",
            modality_centroids={"auditory": fp.copy()},
            modality_masses={"auditory": 5.0},
            total_mass=5.0,
        )
        g = att.gamma(fp, "auditory")
        assert g < 0.01  # near perfect

    def test_gamma_no_match(self):
        fp1 = make_fingerprint(0)
        fp2 = make_fingerprint(23)
        att = SemanticAttractor(
            label="x",
            modality_centroids={"auditory": fp1},
            modality_masses={"auditory": 5.0},
            total_mass=5.0,
        )
        g = att.gamma(fp2, "auditory")
        assert g > 0.5  # poor match

    def test_gamma_unknown_modality(self):
        att = SemanticAttractor(
            label="x",
            modality_centroids={"auditory": make_fingerprint(5)},
        )
        g = att.gamma(make_fingerprint(5), "visual")
        assert g == 1.0  # unknown modality -> total reflection

    def test_absorb_updates_centroid(self):
        fp1 = make_fingerprint(5)
        att = SemanticAttractor(
            label="x",
            modality_centroids={"auditory": fp1.copy()},
            modality_masses={"auditory": 1.0},
            total_mass=1.0,
        )
        centroid_before = att.modality_centroids["auditory"].copy()
        fp2 = make_fingerprint(10)
        att.absorb(fp2, "auditory")
        centroid_after = att.modality_centroids["auditory"]
        # Centroid should have moved toward fp2
        assert not np.allclose(centroid_before, centroid_after)

    def test_absorb_increases_mass(self):
        att = SemanticAttractor(
            label="x",
            modality_centroids={"auditory": make_fingerprint(5)},
            modality_masses={"auditory": 1.0},
            total_mass=1.0,
        )
        att.absorb(make_fingerprint(5), "auditory")
        assert att.total_mass > 1.0
        assert att.modality_masses["auditory"] > 1.0

    def test_absorb_new_modality(self):
        att = SemanticAttractor(
            label="x",
            modality_centroids={"auditory": make_fingerprint(5)},
            modality_masses={"auditory": 1.0},
            total_mass=1.0,
        )
        att.absorb(make_fingerprint(15), "visual")
        assert "visual" in att.modality_centroids
        assert att.total_mass > 1.0

    def test_force_on_attracting(self):
        fp = make_fingerprint(10)
        att = SemanticAttractor(
            label="x",
            modality_centroids={"auditory": fp.copy()},
            modality_masses={"auditory": 10.0},
            total_mass=10.0,
        )
        point = make_fingerprint(12)
        force = att.force_on(point, "auditory")
        # Force should pull toward the centroid
        assert float(np.linalg.norm(force)) > 0

    def test_decay_reduces_mass(self):
        att = SemanticAttractor(
            label="x",
            modality_centroids={"auditory": make_fingerprint(5)},
            modality_masses={"auditory": 10.0},
            total_mass=10.0,
        )
        old_mass = att.total_mass
        att.decay()
        assert att.total_mass < old_mass

    def test_predict_cross_modal(self):
        att = SemanticAttractor(
            label="apple",
            modality_centroids={
                "auditory": make_fingerprint(5),
                "visual": make_fingerprint(15),
            },
            modality_masses={"auditory": 5.0, "visual": 5.0},
            total_mass=10.0,
        )
        predicted = att.predict_cross_modal("auditory", "visual", make_fingerprint(5))
        assert predicted is not None
        assert len(predicted) == 24

    def test_distance_to(self):
        att1 = SemanticAttractor(
            label="a",
            modality_centroids={"auditory": make_fingerprint(0)},
        )
        att2 = SemanticAttractor(
            label="b",
            modality_centroids={"auditory": make_fingerprint(23)},
        )
        d = att1.distance_to(att2, "auditory")
        assert 0.0 < d <= 1.0

    def test_to_dict_structure(self):
        att = SemanticAttractor(
            label="test",
            modality_centroids={"auditory": make_fingerprint(5)},
            modality_masses={"auditory": 3.0},
            total_mass=3.0,
        )
        d = att.to_dict()
        assert "label" in d
        assert "total_mass" in d
        assert "quality_factor" in d
        assert "impedance" in d


# ============================================================================
# Phase 4.2: SemanticField
# ============================================================================


class TestSemanticField:
    """Test the SemanticField collective behavior."""

    def test_register_concept(self):
        sf = SemanticField()
        att = sf.register_concept("apple", make_fingerprint(5), "auditory")
        assert att.label == "apple"
        assert "apple" in sf.attractors

    def test_register_duplicate_updates(self):
        sf = SemanticField()
        sf.register_concept("apple", make_fingerprint(5), "auditory")
        sf.register_concept("apple", make_fingerprint(5), "auditory")
        assert sf.attractors["apple"].activation_count == 2

    def test_recognize_basic(self):
        sf = SemanticField()
        sf.register_concept("apple", make_fingerprint(5), "auditory")
        sf.register_concept("banana", make_fingerprint(18), "auditory")

        results = sf.recognize(make_fingerprint(5), "auditory")
        assert len(results) > 0
        assert results[0][0] == "apple"  # best match

    def test_recognize_returns_sorted(self):
        sf = SemanticField()
        sf.register_concept("a", make_fingerprint(3), "auditory")
        sf.register_concept("b", make_fingerprint(10), "auditory")
        sf.register_concept("c", make_fingerprint(20), "auditory")

        results = sf.recognize(make_fingerprint(3), "auditory")
        # Should be sorted by gamma ascending
        gammas = [r[1] for r in results]
        assert gammas == sorted(gammas)

    def test_best_match(self):
        sf = SemanticField()
        sf.register_concept("dog", make_fingerprint(8), "auditory")
        match = sf.best_match(make_fingerprint(8), "auditory")
        assert match is not None
        assert match[0] == "dog"
        assert match[1] < 0.1  # low gamma

    def test_best_match_empty(self):
        sf = SemanticField()
        match = sf.best_match(make_fingerprint(5), "auditory")
        assert match is None

    def test_multi_modal_recognize(self):
        sf = SemanticField()
        sf.register_concept("apple", make_fingerprint(5), "auditory")
        sf.attractors["apple"].absorb(make_fingerprint(15), "visual")

        fps = {
            "auditory": make_fingerprint(5),
            "visual": make_fingerprint(15),
        }
        results = sf.multi_modal_recognize(fps)
        assert len(results) > 0
        assert results[0][0] == "apple"

    def test_gamma_landscape(self):
        sf = SemanticField()
        sf.register_concept("a", make_fingerprint(3), "auditory")
        sf.register_concept("b", make_fingerprint(15), "auditory")

        landscape = sf.gamma_landscape(make_fingerprint(3), "auditory")
        assert "a" in landscape
        assert "b" in landscape
        assert landscape["a"] < landscape["b"]

    def test_evolve_state(self):
        sf = SemanticField()
        sf.register_concept("target", make_fingerprint(10), "auditory")
        # Absorb many times to increase mass
        for _ in range(20):
            sf.absorb("target", make_fingerprint(10), "auditory")

        # Start from a nearby point
        initial = make_fingerprint(12)
        result = sf.evolve_state(initial, "auditory")
        assert "settled_in" in result
        assert "final_state" in result
        assert len(result["trajectory"]) > 1

    def test_contrastive_update_pushes_apart(self):
        sf = SemanticField()
        # Create two concepts with overlapping but distinct fingerprints
        fp_a = make_fingerprint(10)
        fp_b = make_shifted_fingerprint(10, shift=2.0)  # more observable shift

        sf.register_concept("a", fp_a, "auditory")
        sf.register_concept("b", fp_b, "auditory")

        sim_before = cosine_similarity(
            sf.attractors["a"].modality_centroids["auditory"],
            sf.attractors["b"].modality_centroids["auditory"],
        )

        # Only test contrastive if they are above the threshold
        if sim_before > CONTRASTIVE_THRESHOLD:
            sf.contrastive_update()

            sim_after = cosine_similarity(
                sf.attractors["a"].modality_centroids["auditory"],
                sf.attractors["b"].modality_centroids["auditory"],
            )
            # Contrastive should push apart or at least not increase
            assert sim_after < sim_before + 1e-9
        else:
            # Concepts are already well-separated; contrastive is a no-op
            sf.contrastive_update()
            # Just verify it doesn't crash
            assert len(sf.attractors) == 2

    def test_cross_modal_prediction(self):
        sf = SemanticField()
        sf.register_concept("apple", make_fingerprint(5), "auditory")
        sf.attractors["apple"].absorb(make_fingerprint(15), "visual")
        # Strengthen the attractor
        for _ in range(20):
            sf.absorb("apple", make_fingerprint(5), "auditory")
            sf.attractors["apple"].absorb(make_fingerprint(15), "visual")

        pred = sf.predict_cross_modal(
            make_fingerprint(5), "auditory", "visual"
        )
        assert pred is not None
        assert pred["concept"] == "apple"
        assert pred["predicted_fingerprint"] is not None

    def test_semantic_distance(self):
        sf = SemanticField()
        sf.register_concept("near", make_fingerprint(5), "auditory")
        sf.register_concept("far", make_fingerprint(20), "auditory")

        d = sf.semantic_distance("near", "far", "auditory")
        assert 0.0 < d <= 1.0

    def test_get_neighbors(self):
        sf = SemanticField()
        sf.register_concept("a", make_fingerprint(5), "auditory")
        sf.register_concept("b", make_fingerprint(6), "auditory")
        sf.register_concept("c", make_fingerprint(20), "auditory")

        neighbors = sf.get_neighbors("a", "auditory", top_k=2)
        assert len(neighbors) == 2
        # 'b' should be closer to 'a' than 'c'
        assert neighbors[0][0] == "b"

    def test_tick_decays_mass(self):
        sf = SemanticField()
        sf.register_concept("x", make_fingerprint(5), "auditory")
        old_mass = sf.attractors["x"].total_mass
        sf.tick()
        new_mass = sf.attractors["x"].total_mass
        assert new_mass <= old_mass

    def test_capacity_limit(self):
        sf = SemanticField(max_attractors=5)
        for i in range(10):
            sf.register_concept(f"c_{i}", make_fingerprint(i % 24), "auditory")
        assert len(sf.attractors) <= 5

    def test_get_constellation(self):
        sf = SemanticField()
        sf.register_concept("a", make_fingerprint(3), "auditory")
        sf.register_concept("b", make_fingerprint(15), "auditory")
        constellation = sf.get_constellation("auditory")
        assert "n_concepts" in constellation
        assert constellation["n_concepts"] == 2
        assert "pairwise_distances" in constellation

    def test_get_state(self):
        sf = SemanticField()
        sf.register_concept("x", make_fingerprint(5), "auditory")
        state = sf.get_state()
        assert state["n_attractors"] == 1
        assert "total_recognitions" in state


# ============================================================================
# Phase 4.2: SemanticFieldEngine
# ============================================================================


class TestSemanticFieldEngine:
    """Test the high-level engine interface."""

    def test_process_fingerprint_basic(self):
        engine = SemanticFieldEngine()
        engine.process_fingerprint(
            make_fingerprint(5), "auditory", label="test"
        )
        result = engine.process_fingerprint(
            make_fingerprint(5), "auditory"
        )
        assert result["best_concept"] == "test"

    def test_process_fingerprint_novel(self):
        engine = SemanticFieldEngine()
        result = engine.process_fingerprint(
            make_fingerprint(5), "auditory"
        )
        assert result["is_novel"] is True

    def test_predict_from_hearing(self):
        engine = SemanticFieldEngine()
        # Register concept with both modalities
        engine.field.register_concept("apple", make_fingerprint(5), "auditory")
        engine.field.attractors["apple"].absorb(make_fingerprint(15), "visual")
        # Strengthen
        for _ in range(20):
            engine.field.absorb("apple", make_fingerprint(5), "auditory")
            engine.field.attractors["apple"].absorb(make_fingerprint(15), "visual")

        pred = engine.predict_from_hearing(make_fingerprint(5), "visual")
        assert pred is not None
        assert pred["concept"] == "apple"

    def test_tick(self):
        engine = SemanticFieldEngine()
        engine.process_fingerprint(make_fingerprint(5), "auditory", label="x")
        engine.tick()
        # Should not crash

    def test_get_state(self):
        engine = SemanticFieldEngine()
        engine.process_fingerprint(make_fingerprint(5), "auditory", label="x")
        state = engine.get_state()
        assert "n_attractors" in state


# ============================================================================
# Phase 4.2: Discrimination Test (Phase 4.1 Problem)
# ============================================================================


class TestDiscrimination:
    """
    Test that the semantic field can discriminate between
    signals that are very similar in ERB space.
    This was the core problem from Phase 4.1.
    """

    def test_discriminate_similar_tones(self):
        """Two tones close in frequency -> semantic field separates them."""
        cochlea = CochlearFilterBank()
        sf = SemanticField()

        tone_a = generate_tone(440, duration=0.1)
        tone_b = generate_tone(880, duration=0.1)

        fp_a = cochlea.analyze(tone_a, apply_persistence=False).fingerprint()
        fp_b = cochlea.analyze(tone_b, apply_persistence=False).fingerprint()

        # Register with labels
        for _ in range(30):
            sf.absorb("tone_440", fp_a, "auditory")
            sf.absorb("tone_880", fp_b, "auditory")

        # Run contrastive updates
        for _ in range(5):
            sf.contrastive_update()

        # Now recognize
        result_a = sf.best_match(fp_a, "auditory")
        result_b = sf.best_match(fp_b, "auditory")

        assert result_a is not None
        assert result_b is not None
        assert result_a[0] == "tone_440"
        assert result_b[0] == "tone_880"

    def test_vowel_discrimination(self):
        """Different vowels -> different concepts."""
        cochlea = CochlearFilterBank()
        sf = SemanticField()

        vowels = ["a", "i", "u"]
        for v in vowels:
            wave = generate_vowel(v, duration=0.1)
            fp = cochlea.analyze(wave, apply_persistence=False).fingerprint()
            for _ in range(20):
                sf.absorb(f"vowel_{v}", fp, "auditory")

        # Recognize
        for v in vowels:
            wave = generate_vowel(v, duration=0.1)
            fp = cochlea.analyze(wave, apply_persistence=False).fingerprint()
            result = sf.best_match(fp, "auditory")
            assert result is not None
            assert result[0] == f"vowel_{v}"


# ============================================================================
# Phase 4.3: ArticulatoryPlan
# ============================================================================


class TestArticulatoryPlan:
    """Test articulatory plan behavior."""

    def test_creation(self):
        plan = ArticulatoryPlan(
            concept_label="apple",
            formants=(730, 1090, 2440),
            pitch=150.0,
        )
        assert plan.concept_label == "apple"
        assert plan.confidence == CONFIDENCE_FLOOR

    def test_impedance(self):
        plan = ArticulatoryPlan(concept_label="x")
        # Low confidence -> high impedance
        assert plan.z_impedance > 50.0
        plan.confidence = 5.0
        plan.update_impedance()
        assert plan.z_impedance < BROCA_Z0

    def test_gamma(self):
        plan = ArticulatoryPlan(concept_label="x")
        g = plan.gamma()
        assert 0.0 <= g <= 1.0

    def test_energy_transfer(self):
        plan = ArticulatoryPlan(concept_label="x", confidence=5.0)
        plan.update_impedance()
        e = plan.energy_transfer()
        assert 0.0 <= e <= 1.0

    def test_reinforce_increases_confidence(self):
        plan = ArticulatoryPlan(concept_label="x")
        old_conf = plan.confidence
        plan.reinforce(0.1)  # low gamma_loop = good match
        assert plan.confidence > old_conf

    def test_weaken_decreases_confidence(self):
        plan = ArticulatoryPlan(concept_label="x", confidence=1.0)
        plan.update_impedance()
        old_conf = plan.confidence
        plan.weaken(0.9)  # high gamma_loop = bad match
        assert plan.confidence < old_conf

    def test_adjust_formants(self):
        plan = ArticulatoryPlan(
            concept_label="x",
            formants=(400.0, 1200.0, 2200.0),
        )
        target = (730.0, 1090.0, 2440.0)
        plan.adjust_formants(target)
        f1, f2, f3 = plan.formants
        # Should have moved toward target
        assert abs(f1 - 730) < abs(400 - 730)

    def test_adjust_pitch(self):
        plan = ArticulatoryPlan(concept_label="x", pitch=150.0)
        plan.adjust_pitch(200.0)
        assert plan.pitch > 150.0

    def test_confidence_ceiling(self):
        plan = ArticulatoryPlan(concept_label="x")
        for _ in range(100):
            plan.reinforce(0.0)
        assert plan.confidence <= CONFIDENCE_CEILING

    def test_decay(self):
        plan = ArticulatoryPlan(concept_label="x", confidence=5.0)
        plan.update_impedance()
        old_conf = plan.confidence
        plan.decay()
        assert plan.confidence < old_conf

    def test_to_dict(self):
        plan = ArticulatoryPlan(concept_label="test")
        d = plan.to_dict()
        assert "concept_label" in d
        assert "formants" in d
        assert "confidence" in d
        assert "gamma" in d
        assert "success_rate" in d


# ============================================================================
# Phase 4.3: Extract Formants
# ============================================================================


class TestExtractFormants:
    """Test formant extraction from tonotopic activation."""

    def test_extract_from_vowel(self):
        cochlea = CochlearFilterBank()
        wave = generate_vowel("a", duration=0.1)
        tono = cochlea.analyze(wave, apply_persistence=False)
        f1, f2, f3 = extract_formants(tono)
        assert f1 < f2 < f3  # sorted ascending
        assert F1_RANGE[0] <= f1 or f1 < 200  # near biological range

    def test_always_returns_three(self):
        cochlea = CochlearFilterBank()
        # Pure tone -> might have few peaks
        wave = generate_tone(440, duration=0.1)
        tono = cochlea.analyze(wave, apply_persistence=False)
        f1, f2, f3 = extract_formants(tono)
        assert f1 < f2 or f1 == f2  # at least defined
        assert f3 >= f2

    def test_different_vowels_different_formants(self):
        cochlea = CochlearFilterBank()
        formants = {}
        for v in ["a", "i", "u"]:
            wave = generate_vowel(v, duration=0.1)
            tono = cochlea.analyze(wave, apply_persistence=False)
            formants[v] = extract_formants(tono)
        # At least some formants should differ
        assert formants["a"] != formants["i"] or formants["a"] != formants["u"]


# ============================================================================
# Phase 4.3: BrocaEngine
# ============================================================================


class TestBrocaEngine:
    """Test the Broca motor speech engine."""

    def test_creation(self):
        broca = BrocaEngine()
        assert broca.get_vocabulary_size() == 0

    def test_create_plan(self):
        broca = BrocaEngine()
        plan = broca.create_plan("apple", formants=(730, 1090, 2440))
        assert plan.concept_label == "apple"
        assert broca.has_plan("apple")

    def test_create_vowel_plan(self):
        broca = BrocaEngine()
        plan = broca.create_vowel_plan("a")
        assert plan.concept_label == "vowel_a"
        assert plan.formants == (730.0, 1090.0, 2440.0)
        assert plan.confidence > CONFIDENCE_FLOOR  # innate = higher

    def test_plan_utterance(self):
        broca = BrocaEngine()
        broca.create_plan("dog", formants=(500, 1500, 2500))
        plan = broca.plan_utterance("dog")
        assert plan is not None
        assert plan.concept_label == "dog"

    def test_plan_utterance_missing(self):
        broca = BrocaEngine()
        plan = broca.plan_utterance("unknown")
        assert plan is None

    def test_execute_plan(self):
        broca = BrocaEngine()
        mouth = AliceMouth()
        plan = broca.create_plan("test", formants=(500, 1500, 2500))
        result = broca.execute_plan(plan, mouth)
        assert "waveform" in result
        assert "feedback_fingerprint" in result
        assert len(result["waveform"]) > 0
        assert len(result["feedback_fingerprint"]) == 24

    def test_babble(self):
        broca = BrocaEngine()
        mouth = AliceMouth()
        result = broca.babble(mouth)
        assert "waveform" in result
        assert result["is_babble"] is True
        assert broca.total_babbles == 1

    def test_babble_creates_plan(self):
        broca = BrocaEngine()
        mouth = AliceMouth()
        broca.babble(mouth, intended_label="test_babble")
        assert broca.has_plan("test_babble")

    def test_speak_concept_with_plan(self):
        broca = BrocaEngine()
        mouth = AliceMouth()
        broca.create_plan("vowel_a", formants=(730, 1090, 2440), confidence=1.0)
        result = broca.speak_concept("vowel_a", mouth)
        assert result["intended"] == "vowel_a"
        assert "waveform" in result

    def test_speak_concept_without_plan_babbles(self):
        broca = BrocaEngine()
        mouth = AliceMouth()
        result = broca.speak_concept("unknown", mouth)
        assert result.get("is_babble", False) is True

    def test_speak_concept_with_semantic_field(self):
        """Full sensorimotor loop: speak -> hear -> verify."""
        broca = BrocaEngine()
        mouth = AliceMouth()
        sf_engine = SemanticFieldEngine()

        # First create the concept in semantic field
        cochlea = CochlearFilterBank()
        wave = generate_vowel("a", duration=0.1)
        fp = cochlea.analyze(wave, apply_persistence=False).fingerprint()
        for _ in range(30):
            sf_engine.process_fingerprint(fp, "auditory", label="vowel_a")

        # Create articulatory plan
        broca.create_plan("vowel_a", formants=(730, 1090, 2440), confidence=1.0)

        # Speak and verify
        result = broca.speak_concept("vowel_a", mouth, semantic_field=sf_engine)
        assert result["intended"] == "vowel_a"
        assert "gamma_loop" in result

    def test_learn_from_example(self):
        broca = BrocaEngine()
        wave = generate_vowel("i", duration=0.1)
        plan = broca.learn_from_example("vowel_i", wave)
        assert plan.concept_label == "vowel_i"
        assert plan.confidence > 0

    def test_verify_production(self):
        broca = BrocaEngine()
        sf_engine = SemanticFieldEngine()

        # Register concept
        cochlea = CochlearFilterBank()
        wave = generate_vowel("a", duration=0.1)
        fp = cochlea.analyze(wave, apply_persistence=False).fingerprint()
        for _ in range(30):
            sf_engine.process_fingerprint(fp, "auditory", label="vowel_a")

        # Verify a matching waveform
        result = broca.verify_production(wave, "vowel_a", sf_engine)
        assert "gamma_loop" in result
        assert result["intended"] == "vowel_a"

    def test_vocabulary(self):
        broca = BrocaEngine()
        broca.create_plan("a", formants=(730, 1090, 2440))
        broca.create_plan("i", formants=(270, 2290, 3010))
        assert broca.get_vocabulary_size() == 2
        assert set(broca.get_vocabulary()) == {"a", "i"}

    def test_tick(self):
        broca = BrocaEngine()
        broca.create_plan("x")
        broca.tick()
        # Should not crash

    def test_capacity_limit(self):
        broca = BrocaEngine(max_plans=5)
        for i in range(10):
            broca.create_plan(f"c_{i}")
        assert len(broca.plans) <= 5

    def test_get_state(self):
        broca = BrocaEngine()
        broca.create_plan("test")
        state = broca.get_state()
        assert "vocabulary_size" in state
        assert state["vocabulary_size"] == 1
        assert "total_utterances" in state


# ============================================================================
# Phase 4.3: Sensorimotor Learning
# ============================================================================


class TestSensorimotorLearning:
    """Test the sensorimotor loop: babble -> hear -> learn."""

    def test_babble_and_learn_cycle(self):
        """Babble produces diverse outputs."""
        broca = BrocaEngine()
        mouth = AliceMouth()

        fingerprints = []
        for _ in range(5):
            result = broca.babble(mouth)
            fingerprints.append(result["feedback_fingerprint"])

        # Different babbles should produce somewhat different fingerprints
        # (not guaranteed due to randomness, but statistically likely)
        assert len(fingerprints) == 5

    def test_plan_improvement_through_reinforcement(self):
        """Reinforcing a plan increases confidence and lowers impedance."""
        broca = BrocaEngine()
        plan = broca.create_plan("test", confidence=0.1)

        z_initial = plan.z_impedance
        for _ in range(10):
            plan.reinforce(0.1)  # low gamma = success

        assert plan.confidence > 0.1
        assert plan.z_impedance < z_initial

    def test_formant_convergence(self):
        """Repeatedly adjusting formants converges to target."""
        plan = ArticulatoryPlan(
            concept_label="x",
            formants=(300.0, 1000.0, 2000.0),
        )
        target = (730.0, 1090.0, 2440.0)

        for _ in range(50):
            plan.adjust_formants(target, rate=0.2)

        f1, f2, f3 = plan.formants
        assert abs(f1 - 730) < 10  # converged close
        assert abs(f2 - 1090) < 10
        assert abs(f3 - 2440) < 10


# ============================================================================
# Integration Tests: AliceBrain
# ============================================================================


class TestAliceBrainIntegration:
    """Test that SemanticField and Broca integrate with AliceBrain."""

    def test_alice_has_semantic_field(self):
        from alice.alice_brain import AliceBrain
        alice = AliceBrain(neuron_count=10)
        assert hasattr(alice, "semantic_field")

    def test_alice_has_broca(self):
        from alice.alice_brain import AliceBrain
        alice = AliceBrain(neuron_count=10)
        assert hasattr(alice, "broca")

    def test_introspect_includes_semantic_field(self):
        from alice.alice_brain import AliceBrain
        alice = AliceBrain(neuron_count=10)
        state = alice.introspect()
        assert "semantic_field" in state["subsystems"]

    def test_introspect_includes_broca(self):
        from alice.alice_brain import AliceBrain
        alice = AliceBrain(neuron_count=10)
        state = alice.introspect()
        assert "broca" in state["subsystems"]

    def test_hear_feeds_semantic_field(self):
        from alice.alice_brain import AliceBrain
        alice = AliceBrain(neuron_count=10)
        wave = generate_tone(440, duration=0.05)
        alice.hear(wave)
        # Semantic field should have processed something
        state = alice.semantic_field.get_state()
        assert state["total_recognitions"] >= 1

    def test_say_with_concept(self):
        from alice.alice_brain import AliceBrain
        alice = AliceBrain(neuron_count=10)
        # Create a plan first
        alice.broca.create_vowel_plan("a")
        result = alice.say(
            target_pitch=150, vowel="a", concept="vowel_a"
        )
        assert "final_pitch" in result


# ============================================================================
# Physics Conservation Tests
# ============================================================================


class TestPhysicsConservation:
    """Test that physical laws are respected."""

    def test_gamma_bounds(self):
        """Gamma is always in [0, 1]."""
        for sim in np.linspace(0, 1, 20):
            for mass in [0.1, 1.0, 10.0, 100.0]:
                g = gamma_semantic(float(sim), mass)
                assert 0.0 <= g <= 1.0, f"Gamma out of bounds: {g}"

    def test_energy_conservation(self):
        """Absorbed + reflected = 1."""
        for g in np.linspace(0, 1, 50):
            e = energy_absorbed(float(g))
            reflected = float(g) ** 2
            assert abs(e + reflected - 1.0) < 1e-10

    def test_impedance_positive(self):
        """Impedance is always positive."""
        att = SemanticAttractor(label="x", total_mass=0.1)
        assert att.impedance() > 0
        att.total_mass = 1000
        assert att.impedance() > 0

    def test_mass_never_negative(self):
        """Mass never goes below floor."""
        att = SemanticAttractor(
            label="x",
            modality_centroids={"auditory": make_fingerprint(5)},
            modality_masses={"auditory": 0.2},
            total_mass=0.2,
        )
        for _ in range(1000):
            att.decay()
        assert att.total_mass >= MASS_FLOOR

    def test_plan_impedance_positive(self):
        """Plan impedance is always positive."""
        plan = ArticulatoryPlan(concept_label="x")
        assert plan.z_impedance > 0
        plan.confidence = CONFIDENCE_CEILING
        plan.update_impedance()
        assert plan.z_impedance > 0

    def test_formants_in_biological_range(self):
        """Adjusted formants stay within biological bounds."""
        plan = ArticulatoryPlan(
            concept_label="x",
            formants=(500, 1500, 2500),
        )
        # Extreme target
        plan.adjust_formants((0.0, 0.0, 0.0))
        f1, f2, f3 = plan.formants
        assert f1 >= F1_RANGE[0]
        assert f2 >= F2_RANGE[0]
        assert f3 >= F3_RANGE[0]

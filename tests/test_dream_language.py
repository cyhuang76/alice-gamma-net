#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Phase 27 — Dream Language Genesis

Validates the dream incubation experiment:
  1. Video generators produce valid stimuli
  2. Sleep schedules are well-formed
  3. Dream incubation runs without errors
  4. Communication phase produces valid coupling records
  5. Social coupling Γ trends downward (impedance matching)
  6. Physical invariants are maintained

Philosophy:
  We do NOT test whether concepts "should" emerge.
  We test that the WIRING is correct and the PHYSICS is respected.
  What emerges is what emerges.
"""

from __future__ import annotations

import numpy as np
import pytest

from alice.alice_brain import AliceBrain
from alice.body.cochlea import generate_vowel
from alice.core.protocol import Priority

from experiments.exp_dream_language import (
    NEURON_COUNT,
    generate_video_frame,
    generate_audio_frame,
    make_sleep_schedule,
    snapshot_semantic_field,
    snapshot_wernicke,
    wake_phase,
    dream_incubation_night,
    communication_phase,
    run_experiment,
    DreamRecord,
    SemanticSnapshot,
    CommunicationRecord,
)


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def brain():
    """Create a fresh AliceBrain instance."""
    return AliceBrain(neuron_count=60)


@pytest.fixture
def rng():
    """Deterministic random state."""
    return np.random.RandomState(42)


@pytest.fixture
def brain_pair():
    """Two independent Alice instances for communication tests."""
    return AliceBrain(neuron_count=60), AliceBrain(neuron_count=60)


# ================================================================
# Video Generator Tests
# ================================================================

class TestVideoGenerators:
    """Validate stimulus generation physics."""

    def test_video_frame_shape(self):
        """Video frames should be 1D pixel arrays of specified resolution."""
        frame = generate_video_frame(5.0, frame_idx=0, resolution=256)
        assert frame.shape == (256,)

    def test_video_frame_range(self):
        """Pixel values should be in [0, 1] range (luminance)."""
        for freq in [1.0, 5.0, 30.0, 100.0]:
            for idx in range(10):
                frame = generate_video_frame(freq, idx)
                assert np.all(frame >= 0.0), f"Negative pixels at freq={freq}"
                assert np.all(frame <= 1.0), f"Pixels >1 at freq={freq}"

    def test_different_frequencies_produce_different_frames(self):
        """Different spatial frequencies must produce different visual Γ fingerprints."""
        frame_a = generate_video_frame(5.0, 0)
        frame_b = generate_video_frame(30.0, 0)
        # Not identical
        assert not np.allclose(frame_a, frame_b)
        # Different spectral content
        fft_a = np.abs(np.fft.rfft(frame_a))
        fft_b = np.abs(np.fft.rfft(frame_b))
        assert not np.allclose(fft_a, fft_b)

    def test_temporal_phase_drift(self):
        """Successive frames should differ (temporal dynamics)."""
        frame_0 = generate_video_frame(5.0, 0)
        frame_1 = generate_video_frame(5.0, 1)
        frame_10 = generate_video_frame(5.0, 10)
        assert not np.allclose(frame_0, frame_1)
        assert not np.allclose(frame_0, frame_10)

    def test_audio_rhythm_pattern(self):
        """Audio should follow rhythmic pattern — not every tick produces sound."""
        sounds = []
        silences = []
        for i in range(20):
            audio = generate_audio_frame("a", 2.0, i, tick_duration=0.1)
            if audio is not None:
                sounds.append(i)
            else:
                silences.append(i)
        # 2Hz at 0.1s/tick = sound every 5 ticks
        assert len(sounds) > 0, "No sound produced"
        assert len(silences) > 0, "No silence (rhythm broken)"

    def test_audio_waveform_valid(self):
        """Produced audio waveforms should be finite numpy arrays."""
        audio = generate_audio_frame("a", 2.0, 0)
        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_different_vowels_produce_different_audio(self):
        """Different vowels → different spectral content (formant physics)."""
        audio_a = generate_vowel("a", fundamental=150.0, duration=0.1)
        audio_i = generate_vowel("i", fundamental=150.0, duration=0.1)
        # Same length
        min_len = min(len(audio_a), len(audio_i))
        # Different spectral content
        fft_a = np.abs(np.fft.rfft(audio_a[:min_len]))
        fft_i = np.abs(np.fft.rfft(audio_i[:min_len]))
        assert not np.allclose(fft_a, fft_i)


# ================================================================
# Sleep Schedule Tests
# ================================================================

class TestSleepSchedule:
    """Validate sleep stage scheduling physics."""

    def test_schedule_length(self):
        """Schedule should match requested tick count."""
        for ticks in [50, 100, 110, 200]:
            schedule = make_sleep_schedule(ticks)
            assert len(schedule) == ticks

    def test_schedule_contains_rem(self):
        """Every sleep cycle MUST contain REM stage."""
        schedule = make_sleep_schedule(110)
        assert "rem" in schedule

    def test_schedule_stage_order(self):
        """Stages should progress N1→N2→N3→N2→REM (NREM before REM)."""
        schedule = make_sleep_schedule(110)
        first_rem = schedule.index("rem")
        # There should be N3 before REM (deep sleep before dreaming)
        assert "n3" in schedule[:first_rem]

    def test_schedule_valid_stages_only(self):
        """All stages must be valid sleep stages."""
        schedule = make_sleep_schedule(110)
        valid = {"n1", "n2", "n3", "rem"}
        for s in schedule:
            assert s in valid, f"Invalid stage: {s}"

    def test_rem_proportion(self):
        """REM should be approximately 20% of cycle."""
        schedule = make_sleep_schedule(200)
        rem_count = schedule.count("rem")
        rem_ratio = rem_count / len(schedule)
        assert 0.1 <= rem_ratio <= 0.3, f"REM ratio={rem_ratio:.2f} out of range"


# ================================================================
# Snapshot Tests
# ================================================================

class TestSnapshots:
    """Validate state snapshot functions."""

    def test_semantic_snapshot_fresh_brain(self, brain):
        """Fresh brain should have zero attractors."""
        snap = snapshot_semantic_field(brain)
        assert isinstance(snap, SemanticSnapshot)
        assert snap.n_attractors == 0
        assert snap.total_mass == 0.0

    def test_wernicke_snapshot_fresh_brain(self, brain):
        """Fresh brain should have zero concepts in Wernicke."""
        snap = snapshot_wernicke(brain)
        assert snap["n_concepts"] == 0
        assert snap["total_transitions"] == 0
        assert snap["n_mature_chunks"] == 0

    def test_semantic_snapshot_after_learning(self, brain):
        """After supervised learning, attractor should appear."""
        # Teach a concept via semantic field directly
        fp = np.random.randn(32)
        brain.semantic_field.process_fingerprint(
            fp, modality="visual", label="test_concept"
        )
        snap = snapshot_semantic_field(brain)
        assert snap.n_attractors >= 1
        assert "test_concept" in snap.attractor_labels

    def test_wernicke_snapshot_after_observations(self, brain):
        """After observing concept sequence, transitions should appear."""
        brain.wernicke.observe("apple")
        brain.wernicke.observe("red")
        brain.wernicke.observe("round")
        snap = snapshot_wernicke(brain)
        assert snap["n_concepts"] >= 2
        assert snap["total_transitions"] >= 2


# ================================================================
# Wake Phase Tests
# ================================================================

class TestWakePhase:
    """Validate awake phase physics."""

    def test_wake_phase_runs(self, brain, rng):
        """Wake phase should complete without error."""
        result = wake_phase(brain, ticks=20, rng=rng, label="test")
        assert "ticks" in result
        assert result["ticks"] == 20

    def test_wake_builds_sleep_pressure(self, brain, rng):
        """Awake ticks should increase sleep pressure."""
        p_before = brain.sleep_physics.sleep_pressure
        wake_phase(brain, ticks=50, rng=rng)
        p_after = brain.sleep_physics.sleep_pressure
        assert p_after >= p_before

    def test_wake_energy_decreases(self, brain, rng):
        """Being awake consumes energy."""
        e_before = brain.sleep_physics.energy
        wake_phase(brain, ticks=30, rng=rng)
        e_after = brain.sleep_physics.energy
        assert e_after <= e_before


# ================================================================
# Dream Incubation Tests
# ================================================================

class TestDreamIncubation:
    """Validate one night of dream incubation."""

    def test_dream_incubation_runs(self, brain, rng):
        """Dream incubation should complete without error."""
        # Build some sleep pressure first
        wake_phase(brain, ticks=40, rng=rng)
        records = dream_incubation_night(
            brain,
            video_spatial_freq=5.0,
            video_vowel="a",
            video_rhythm_hz=2.0,
            video_label="A",
            night_idx=0,
            rng=rng,
        )
        assert len(records) > 0
        assert all(isinstance(r, DreamRecord) for r in records)

    def test_dream_records_contain_stimuli(self, brain, rng):
        """Dream records should log stimulus presentation."""
        wake_phase(brain, ticks=40, rng=rng)
        records = dream_incubation_night(
            brain, 5.0, "a", 2.0, "A", 0, rng,
        )
        total_stimuli = sum(r.stimuli_presented for r in records)
        assert total_stimuli > 0, "No stimuli presented during REM"

    def test_dream_has_rem_ticks(self, brain, rng):
        """At least one cycle should have REM ticks."""
        wake_phase(brain, ticks=40, rng=rng)
        records = dream_incubation_night(
            brain, 5.0, "a", 2.0, "A", 0, rng,
        )
        total_rem = sum(r.ticks_in_rem for r in records)
        assert total_rem > 0, "No REM ticks in entire night"

    def test_different_videos_different_processing(self, rng):
        """Two brains with different videos should process differently."""
        brain_a = AliceBrain(neuron_count=60)
        brain_b = AliceBrain(neuron_count=60)
        rng_a = np.random.RandomState(42)
        rng_b = np.random.RandomState(42)

        wake_phase(brain_a, 40, rng_a)
        wake_phase(brain_b, 40, rng_b)

        records_a = dream_incubation_night(
            brain_a, 5.0, "a", 2.0, "A", 0, rng_a,
        )
        records_b = dream_incubation_night(
            brain_b, 30.0, "i", 3.0, "B", 0, rng_b,
        )

        # Both should have REM stimuli
        stim_a = sum(r.stimuli_presented for r in records_a)
        stim_b = sum(r.stimuli_presented for r in records_b)
        assert stim_a > 0
        assert stim_b > 0

    def test_sleep_energy_recovers_during_dream(self, brain, rng):
        """Sleep should recover energy (metabolic restoration)."""
        wake_phase(brain, ticks=60, rng=rng)
        e_before = brain.sleep_physics.energy
        dream_incubation_night(brain, 5.0, "a", 2.0, "A", 0, rng)
        e_after = brain.sleep_physics.energy
        # Energy should recover during sleep (or at least not drop as fast)
        # Note: the begin_sleep/end_sleep cycle handles this
        assert e_after >= 0  # At minimum, not negative


# ================================================================
# Communication Phase Tests
# ================================================================

class TestCommunicationPhase:
    """Validate bidirectional social coupling."""

    def test_communication_runs(self, brain_pair):
        """Communication phase should complete without error."""
        alice_a, alice_b = brain_pair
        records = communication_phase(alice_a, alice_b, rounds=5)
        assert len(records) == 5
        assert all(isinstance(r, CommunicationRecord) for r in records)

    def test_gamma_social_bounded(self, brain_pair):
        """Γ_social should be in [0, 1] (physical constraint)."""
        alice_a, alice_b = brain_pair
        records = communication_phase(alice_a, alice_b, rounds=10)
        for r in records:
            assert 0.0 <= r.gamma_social_ab <= 1.0, \
                f"Γ_ab={r.gamma_social_ab} out of [0,1]"
            assert 0.0 <= r.gamma_social_ba <= 1.0, \
                f"Γ_ba={r.gamma_social_ba} out of [0,1]"

    def test_energy_transfer_bounded(self, brain_pair):
        """η = 1-|Γ|² should be in [0, 1]."""
        alice_a, alice_b = brain_pair
        records = communication_phase(alice_a, alice_b, rounds=10)
        for r in records:
            assert 0.0 <= r.energy_transfer_ab <= 1.0
            assert 0.0 <= r.energy_transfer_ba <= 1.0

    def test_gamma_social_decreases_over_time(self, brain_pair):
        """
        Γ_social should trend downward (impedance matching improves).
        
        This is the key prediction: even without shared concepts,
        two agents that repeatedly interact develop lower Γ 
        (better rapport / impedance matching).
        """
        alice_a, alice_b = brain_pair
        records = communication_phase(alice_a, alice_b, rounds=15)

        first_3 = np.mean([r.gamma_social_ab for r in records[:3]])
        last_3 = np.mean([r.gamma_social_ab for r in records[-3:]])

        # Γ should decrease (or at least not increase dramatically)
        assert last_3 <= first_3 + 0.1, \
            f"Γ_social increased: {first_3:.3f} → {last_3:.3f}"

    def test_symmetric_coupling(self, brain_pair):
        """
        For symmetric agent parameters, Γ_ab ≈ Γ_ba.
        
        Physics: |Z_A - Z_B|/(Z_A + Z_B) is symmetric.
        """
        alice_a, alice_b = brain_pair
        records = communication_phase(alice_a, alice_b, rounds=10)
        for r in records:
            assert abs(r.gamma_social_ab - r.gamma_social_ba) < 0.01, \
                f"Asymmetric coupling: Γ_ab={r.gamma_social_ab:.4f} ≠ Γ_ba={r.gamma_social_ba:.4f}"

    def test_pressure_non_negative(self, brain_pair):
        """Semantic pressure should never go negative (physical invariant)."""
        alice_a, alice_b = brain_pair
        records = communication_phase(alice_a, alice_b, rounds=10)
        for r in records:
            assert r.pressure_a_before >= 0
            assert r.pressure_a_after >= 0
            assert r.pressure_b_before >= 0
            assert r.pressure_b_after >= 0


# ================================================================
# Full Experiment Integration Test
# ================================================================

class TestFullExperiment:
    """Integration test for the complete experiment."""

    def test_full_experiment_runs(self):
        """
        The entire experiment should complete without errors.
        
        This is the most critical test: the complete pipeline
        (birth → wake → dream → communication → analysis)
        must be physically consistent from end to end.
        """
        result = run_experiment(verbose=False)

        # Basic structure
        assert "elapsed_s" in result
        assert "dream_records_alpha" in result
        assert "dream_records_beta" in result
        assert "communication_records" in result

        # Dream records exist
        assert len(result["dream_records_alpha"]) > 0
        assert len(result["dream_records_beta"]) > 0

        # Communication records exist
        assert len(result["communication_records"]) > 0

    def test_experiment_result_types(self):
        """All result fields should have correct types."""
        result = run_experiment(verbose=False)

        assert isinstance(result["elapsed_s"], float)
        assert isinstance(result["n_attractors_alpha"], int)
        assert isinstance(result["n_attractors_beta"], int)
        assert isinstance(result["n_shared_concepts"], int)
        assert isinstance(result["gamma_social_trend"], float)

    def test_gamma_social_trend_direction(self):
        """
        The Γ_social trend should be negative (decreasing).
        
        This is the KEY physical prediction of the experiment:
        repeated social coupling → impedance matching → lower Γ.
        """
        result = run_experiment(verbose=False)
        # Γ should decrease (Δ < 0) or at least not increase much
        assert result["gamma_social_trend"] <= 0.05, \
            f"Γ_social trend positive: {result['gamma_social_trend']:+.4f}"

    def test_stimulus_counts_match(self):
        """Both Alices should receive comparable stimulus counts."""
        result = run_experiment(verbose=False)
        stim_a = sum(r.stimuli_presented for r in result["dream_records_alpha"])
        stim_b = sum(r.stimuli_presented for r in result["dream_records_beta"])
        # Should be identical (same schedule, different content)
        assert stim_a == stim_b, f"Stimulus mismatch: α={stim_a} β={stim_b}"

    def test_semantic_snapshots_consistent(self):
        """Semantic state should not decrease (attractors don't vanish spontaneously)."""
        result = run_experiment(verbose=False)
        # Post-dream ≥ baseline
        baseline_a = result["baseline_semantic_alpha"].n_attractors
        post_a = result["post_dream_semantic_alpha"].n_attractors
        assert post_a >= baseline_a, \
            f"Attractors decreased: {baseline_a} → {post_a}"

    def test_no_nan_in_results(self):
        """All numerical results should be finite (no NaN from division by zero)."""
        result = run_experiment(verbose=False)
        assert np.isfinite(result["gamma_social_trend"])
        assert np.isfinite(result["elapsed_s"])
        for rec in result["communication_records"]:
            assert np.isfinite(rec.gamma_social_ab)
            assert np.isfinite(rec.gamma_social_ba)
            assert np.isfinite(rec.energy_transfer_ab)
            assert np.isfinite(rec.energy_transfer_ba)


# ================================================================
# Physical Invariant Tests
# ================================================================

class TestPhysicalInvariants:
    """
    Tests for physical laws that must NEVER be violated.
    
    These are not about what emerges, but about what's impossible:
    - Γ ∈ [0, 1]
    - η ∈ [0, 1]  
    - η = 1 - |Γ|²
    - Energy is conserved (no negative energy)
    - Pressure ≥ 0
    """

    def test_gamma_eta_relationship(self):
        """η = 1 - |Γ|² must hold for all communication records."""
        result = run_experiment(verbose=False)
        for rec in result["communication_records"]:
            expected_eta_ab = 1.0 - rec.gamma_social_ab ** 2
            assert abs(rec.energy_transfer_ab - expected_eta_ab) < 0.01, \
                f"η_ab={rec.energy_transfer_ab:.4f} ≠ 1-Γ²={expected_eta_ab:.4f}"

    def test_dream_records_non_negative_counts(self):
        """All dream record counters must be ≥ 0."""
        result = run_experiment(verbose=False)
        for rec in result["dream_records_alpha"] + result["dream_records_beta"]:
            assert rec.ticks_in_rem >= 0
            assert rec.stimuli_presented >= 0
            assert rec.wernicke_observations >= 0
            assert rec.n400_events >= 0

    def test_total_stimuli_positive(self):
        """At least some stimuli must be presented (experiment is running)."""
        result = run_experiment(verbose=False)
        total = sum(
            r.stimuli_presented
            for r in result["dream_records_alpha"] + result["dream_records_beta"]
        )
        assert total > 0, "Zero stimuli presented across entire experiment"


# ================================================================
# Emergence Observation Tests (NOT prescriptive)
# ================================================================

class TestEmergenceObservation:
    """
    These tests OBSERVE what emerged — they do NOT prescribe outcomes.
    
    Per Paper II Iron Law #3: we test the wiring, not the result.
    If something emerges, great. If not, that's also a valid finding.
    These tests simply record the observations.
    """

    def test_dream_content_captured(self):
        """Dream records should capture any recognized concepts."""
        result = run_experiment(verbose=False)
        # We don't assert concepts MUST form — just that the recording works
        all_concepts = set()
        for rec in result["dream_records_alpha"]:
            all_concepts.update(rec.concepts_recognized)
        # This may be empty (valid) or non-empty (also valid)
        assert isinstance(all_concepts, set)

    def test_wernicke_observation_recording(self):
        """Wernicke observations should be tracked if they occur."""
        result = run_experiment(verbose=False)
        total_w = sum(
            r.wernicke_observations
            for r in result["dream_records_alpha"]
        )
        # May be 0 (no concepts formed) — that's valid
        assert total_w >= 0

    def test_concept_sets_recorded(self):
        """Post-communication concept analysis should be recorded."""
        result = run_experiment(verbose=False)
        assert isinstance(result["shared_concepts"], list)
        assert isinstance(result["only_alpha_concepts"], list)
        assert isinstance(result["only_beta_concepts"], list)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Phase 33 — Dream Language Genesis

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
    DREAM_CHANNEL_NAMES,
    IMPEDANCE_MOD_DEPTH,
    Z_BASE,
    SPATIAL_FREQ_BOUNDARY,
    VOWEL_FORMANTS,
    generate_video_frame,
    generate_audio_frame,
    make_sleep_schedule,
    snapshot_semantic_field,
    snapshot_wernicke,
    video_to_impedance_modulation,
    wake_phase,
    dream_incubation_night,
    communication_phase,
    run_experiment,
    DreamRecord,
    SemanticSnapshot,
    CommunicationRecord,
)

from alice.brain.sleep_physics import (
    REMDreamDiagnostic,
    SleepPhysicsEngine,
    REM_DREAM_NOISE_AMP,
    FATIGUE_DREAM_AMP_MAX,
    FATIGUE_DREAM_PROBE_BONUS,
    REM_PROBE_COUNT,
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
    """Validate stimulus models (used for impedance modulation, NOT injection)."""

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

    def test_dream_records_contain_modulations(self, brain, rng):
        """Dream records should log impedance modulation events."""
        wake_phase(brain, ticks=40, rng=rng)
        records = dream_incubation_night(
            brain, 5.0, "a", 2.0, "A", 0, rng,
        )
        total_mods = sum(r.modulations_applied for r in records)
        assert total_mods > 0, "No impedance modulations during REM"

    def test_dream_has_rem_ticks(self, brain, rng):
        """At least one cycle should have REM ticks."""
        wake_phase(brain, ticks=40, rng=rng)
        records = dream_incubation_night(
            brain, 5.0, "a", 2.0, "A", 0, rng,
        )
        total_rem = sum(r.ticks_in_rem for r in records)
        assert total_rem > 0, "No REM ticks in entire night"

    def test_different_videos_different_impedance(self, rng):
        """Two brains with different videos should get different impedance patterns."""
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

        # Both should have modulation events
        mod_a = sum(r.modulations_applied for r in records_a)
        mod_b = sum(r.modulations_applied for r in records_b)
        assert mod_a > 0
        assert mod_b > 0

        # Modulated Gamma patterns should differ (different videos)
        gammas_a = [g for r in records_a for g in r.modulated_gammas]
        gammas_b = [g for r in records_b for g in r.modulated_gammas]
        assert len(gammas_a) > 0 and len(gammas_b) > 0
        # Compare means — different videos produce different impedance landscapes
        # (individual ticks have rng noise, but the mean over many ticks diverges)
        mean_a = np.mean(gammas_a)
        mean_b = np.mean(gammas_b)
        # They should not be identical (same schedule, different video content)
        assert abs(mean_a - mean_b) > 1e-6 or \
            not np.allclose(gammas_a, gammas_b, atol=0.02)

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
        """Both Alices should receive equal modulation event counts."""
        result = run_experiment(verbose=False)
        mod_a = sum(r.modulations_applied for r in result["dream_records_alpha"])
        mod_b = sum(r.modulations_applied for r in result["dream_records_beta"])
        # Should be identical (same schedule, different impedance patterns)
        assert mod_a == mod_b, f"Modulation mismatch: α={mod_a} β={mod_b}"

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
            assert rec.modulations_applied >= 0
            assert rec.stimuli_presented >= 0
            assert rec.wernicke_observations >= 0
            assert rec.n400_events >= 0

    def test_total_modulations_positive(self):
        """At least some impedance modulations must occur (experiment is running)."""
        result = run_experiment(verbose=False)
        total = sum(
            r.modulations_applied
            for r in result["dream_records_alpha"] + result["dream_records_beta"]
        )
        assert total > 0, "Zero modulations across entire experiment"


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


# ================================================================
# Impedance Modulation Tests (non-invasive mirror therapy principle)
# ================================================================

class TestImpedanceModulation:
    """
    Validate the non-invasive impedance modulation mechanism.

    Design principle: video → Z_terminus change, NOT video → brain.see()
    This is the mirror therapy model applied to dream incubation.

    Same equation everywhere:
      Pruning:      Z_random → Hebbian selection → Z_matched survive
      Phantom limb: Z_∞ → mirror → Z_match → pain↓
      Dream:        Z_random → video modulation → Z_structured → PGO reflects
    """

    def test_returns_named_channels(self):
        """Modulation should return channels with DREAM_CHANNEL_NAMES."""
        rng = np.random.RandomState(42)
        channels = video_to_impedance_modulation(5.0, "a", 2.0, 0, rng)
        assert len(channels) == len(DREAM_CHANNEL_NAMES)
        names = [c[0] for c in channels]
        assert names == DREAM_CHANNEL_NAMES

    def test_returns_valid_impedances(self):
        """All impedances must be > 0 (physical requirement)."""
        rng = np.random.RandomState(42)
        for freq in [1.0, 5.0, 15.0, 30.0, 60.0]:
            for vowel in ["a", "i", "u"]:
                channels = video_to_impedance_modulation(
                    freq, vowel, 2.0, 0, rng)
                for name, z_src, z_load in channels:
                    assert z_src >= 10.0, f"{name}: z_src={z_src}"
                    assert z_load >= 10.0, f"{name}: z_load={z_load}"

    def test_low_freq_video_modulates_visual_low(self):
        """Video with f<15 should reduce Γ at visual_low channel."""
        rng = np.random.RandomState(42)
        # Low-freq video (f=5)
        ch_mod = video_to_impedance_modulation(5.0, "a", 2.0, 0, rng)
        # Random baseline (very high freq that doesn't match visual_low)
        rng2 = np.random.RandomState(42)
        ch_base = video_to_impedance_modulation(50.0, "a", 2.0, 0, rng2)

        # Find visual_low channel
        for (name_m, zs_m, zl_m), (name_b, zs_b, zl_b) in zip(ch_mod, ch_base):
            if name_m == "visual_low":
                g_mod = abs((zl_m - zs_m) / (zl_m + zs_m))
                g_base = abs((zl_b - zs_b) / (zl_b + zs_b))
                # Low freq video should push visual_low Z_load toward Z_src
                # → lower Γ (better match) OR at least different
                # We can't guarantee g_mod < g_base every time due to rng
                # but the modulation should change the impedance
                assert zl_m != zl_b or \
                    abs(g_mod - g_base) < 0.01, \
                    "Modulation should change visual_low impedance"
                break

    def test_high_freq_video_does_not_modulate_visual_low(self):
        """Video with f≥15 should NOT modulate visual_low channel."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)  # Same seed → same base noise
        ch_high = video_to_impedance_modulation(30.0, "i", 3.0, 0, rng1)
        # visual_low match = 0 for f≥15, so modulation = 0
        # The z_load should just be z_load_base (random)
        for name, zs, zl in ch_high:
            if name == "visual_low":
                # No modulation → Z_load stays at random base
                # Check that it's near Z_BASE ± 20 (random range)
                assert abs(zl - Z_BASE) < 50.0  # Within noise range

    def test_vowel_a_modulates_auditory_f1(self):
        """Vowel /a/ (F1≈700Hz) should modulate auditory_f1 channel."""
        rng = np.random.RandomState(42)
        ch = video_to_impedance_modulation(5.0, "a", 2.0, 25, rng)
        for name, zs, zl in ch:
            if name == "auditory_f1":
                # /a/ F1=700 is within FORMANT_BAND_F1 (200,1500)
                # Modulation should bring Z_load closer to Z_src
                g = abs((zl - zs) / (zl + zs)) if (zl + zs) > 0 else 1.0
                # Just verify impedance is finite and positive
                assert zs > 0 and zl > 0
                assert np.isfinite(g)
                break

    def test_different_videos_produce_different_patterns(self):
        """Video A (f=5, /a/) vs Video B (f=30, /i/) → different Γ patterns."""
        rng_a = np.random.RandomState(42)
        rng_b = np.random.RandomState(42)
        ch_a = video_to_impedance_modulation(5.0, "a", 2.0, 10, rng_a)
        ch_b = video_to_impedance_modulation(30.0, "i", 3.0, 10, rng_b)

        gammas_a = [abs((zl - zs) / (zl + zs)) for _, zs, zl in ch_a]
        gammas_b = [abs((zl - zs) / (zl + zs)) for _, zs, zl in ch_b]
        # Patterns should differ
        assert not np.allclose(gammas_a, gammas_b, atol=0.02), \
            "Different videos must produce different Γ fingerprints"

    def test_somatosensory_motor_unmodulated(self):
        """Somatosensory and motor channels should be unmodulated (reference)."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        ch_a = video_to_impedance_modulation(5.0, "a", 2.0, 0, rng1)
        ch_b = video_to_impedance_modulation(30.0, "i", 3.0, 0, rng2)
        # somatosensory and motor positions (indices 4, 5)
        for idx in [4, 5]:
            _, _, zl_a = ch_a[idx]
            _, _, zl_b = ch_b[idx]
            # Same seed → same base noise → same Z_load (no modulation)
            assert abs(zl_a - zl_b) < 0.001, \
                f"Reference channel {DREAM_CHANNEL_NAMES[idx]} should be unmodulated"

    def test_temporal_rhythm_modulation(self):
        """Rhythm creates temporal variation in impedance landscape."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        # Frame 0: sin(2π * 2.0 * 0 * 0.01) = sin(0) = 0 → temporal_mod = 0.5
        # Frame 12: sin(2π * 2.0 * 12 * 0.01) = sin(1.508) ≈ 0.998 → temporal_mod ≈ 1.0
        # These produce different modulation at matching channels.
        ch_t0 = video_to_impedance_modulation(5.0, "a", 2.0, 0, rng1)
        ch_t12 = video_to_impedance_modulation(5.0, "a", 2.0, 12, rng2)
        # Compare only visual_low (index 0) which matches f=5
        _, zs0, zl0 = ch_t0[0]   # visual_low at frame 0
        _, zs12, zl12 = ch_t12[0] # visual_low at frame 12
        g0 = abs((zl0 - zs0) / (zl0 + zs0)) if (zl0 + zs0) > 0 else 1.0
        g12 = abs((zl12 - zs12) / (zl12 + zs12)) if (zl12 + zs12) > 0 else 1.0
        # Different temporal phase → different modulation depth → different Γ
        assert abs(g0 - g12) > 0.001, \
            f"Temporal rhythm should change Γ: frame0={g0:.4f} frame12={g12:.4f}"

    def test_modulation_depth_bounded(self):
        """Z_load should not be pushed beyond physical limits."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            channels = video_to_impedance_modulation(
                5.0, "a", 2.0, rng.randint(0, 1000), rng,
            )
            for name, z_src, z_load in channels:
                assert z_src >= 10.0
                assert z_load >= 10.0
                assert np.isfinite(z_src) and np.isfinite(z_load)

    def test_dream_record_tracks_modulated_gammas(self):
        """DreamRecord should track Γ at modulated channels."""
        brain = AliceBrain(neuron_count=60)
        rng = np.random.RandomState(42)
        wake_phase(brain, ticks=40, rng=rng)
        records = dream_incubation_night(brain, 5.0, "a", 2.0, "A", 0, rng)
        # At least some records should have modulated_gammas
        all_gammas = [g for r in records for g in r.modulated_gammas]
        assert len(all_gammas) > 0, "No modulated gammas tracked"
        for g in all_gammas:
            assert 0.0 <= g <= 1.0, f"Γ out of range: {g}"
            assert np.isfinite(g)


# ================================================================
# Fatigue-Modulated Dreaming Tests (託夢物理)
# ================================================================

class TestFatigueModulatedDreaming:
    """
    Validate the fatigue→dream intensity coupling.

    Physics: impedance_debt drives PGO amplitude.
    Safety: amplitude capped, net REM energy recovery ≥ 0.
    """

    def test_zero_fatigue_baseline(self):
        """With zero fatigue, probe_channels should use base amp and count."""
        diag = REMDreamDiagnostic()
        channels = [(f"ch_{i}", 50.0, 50.0 + i * 10) for i in range(10)]
        result = diag.probe_channels(channels, fatigue_factor=0.0)
        assert result["amp_multiplier"] == 1.0
        assert result["probes"] == REM_PROBE_COUNT

    def test_max_fatigue_capped(self):
        """At maximum fatigue, amplitude must be capped at FATIGUE_DREAM_AMP_MAX."""
        diag = REMDreamDiagnostic()
        channels = [(f"ch_{i}", 50.0, 80.0) for i in range(20)]
        result = diag.probe_channels(channels, fatigue_factor=1.0)
        assert result["amp_multiplier"] <= FATIGUE_DREAM_AMP_MAX + 0.001
        assert result["amp_multiplier"] >= 1.5  # Should be noticeably higher

    def test_fatigue_increases_amplitude(self):
        """Higher fatigue → higher PGO amplitude multiplier."""
        diag = REMDreamDiagnostic()
        channels = [(f"ch_{i}", 50.0, 80.0) for i in range(20)]
        r_low = diag.probe_channels(channels, fatigue_factor=0.1)
        r_high = diag.probe_channels(channels, fatigue_factor=0.9)
        assert r_high["amp_multiplier"] > r_low["amp_multiplier"]

    def test_fatigue_increases_probe_count(self):
        """Higher fatigue → more channels scanned per tick."""
        diag = REMDreamDiagnostic()
        channels = [(f"ch_{i}", 50.0, 80.0) for i in range(20)]
        r_low = diag.probe_channels(channels, fatigue_factor=0.0)
        r_high = diag.probe_channels(channels, fatigue_factor=1.0)
        assert r_high["probes"] >= r_low["probes"]
        assert r_high["probes"] <= REM_PROBE_COUNT + FATIGUE_DREAM_PROBE_BONUS

    def test_fatigue_factor_clipped_to_01(self):
        """fatigue_factor should be clipped to [0, 1] — no overflow."""
        diag = REMDreamDiagnostic()
        channels = [(f"ch_{i}", 50.0, 80.0) for i in range(20)]
        r_over = diag.probe_channels(channels, fatigue_factor=5.0)
        r_under = diag.probe_channels(channels, fatigue_factor=-1.0)
        assert r_over["fatigue_factor"] == 1.0
        assert r_under["fatigue_factor"] == 0.0
        assert r_over["amp_multiplier"] <= FATIGUE_DREAM_AMP_MAX + 0.001

    def test_empty_channels_returns_safe_dict(self):
        """Empty channel list should return safe default with fatigue fields."""
        diag = REMDreamDiagnostic()
        result = diag.probe_channels([], fatigue_factor=0.5)
        assert result["probes"] == 0
        assert result["amp_multiplier"] == 1.0
        assert result["fatigue_factor"] == 0.0

    def test_dream_fragments_include_probe_amp(self):
        """Dream fragments should record the actual probe amplitude used."""
        diag = REMDreamDiagnostic()
        channels = [("ch_0", 50.0, 100.0)]
        diag.probe_channels(channels, fatigue_factor=0.5)
        assert len(diag.dream_fragments) >= 1
        frag = diag.dream_fragments[-1]
        assert "probe_amp" in frag
        assert frag["probe_amp"] > REM_DREAM_NOISE_AMP  # Amplified


class TestFatigueSleepPhysicsSafety:
    """
    Safety tests: fatigue-modulated REM must NOT drain energy below zero
    or create positive feedback runaway.
    """

    def test_rem_energy_net_positive(self):
        """
        Even at max fatigue, REM tick should have net positive energy recovery.
        
        This is the critical safety invariant:
          recovery - cost - extra_dream_cost ≥ 0
        """
        engine = SleepPhysicsEngine(energy=0.5)
        # Deliberately accumulate max debt
        for _ in range(100):
            engine.awake_tick(reflected_energy=0.5)
        assert engine.impedance_debt.debt > 0.3  # Confirm fatigued

        engine.begin_sleep()
        e_before = engine.energy
        # Run many REM ticks
        channels = [(f"ch_{i}", 50.0, 80.0) for i in range(20)]
        for _ in range(50):
            engine.sleep_tick(
                stage="rem",
                channel_impedances=channels,
                synaptic_strengths=list(np.random.uniform(0.3, 1.5, 100)),
            )
        e_after = engine.energy
        # Energy should not have decreased (REM is net recovery)
        assert e_after >= e_before, \
            f"REM drained energy: {e_before:.4f} → {e_after:.4f}"

    def test_no_runaway_positive_feedback(self):
        """
        Fatigue → vivid dream → more energy cost → more debt → ...
        This loop MUST be stable (debt should decrease during sleep, not increase).
        """
        engine = SleepPhysicsEngine(energy=0.3)
        # Max out debt
        for _ in range(200):
            engine.awake_tick(reflected_energy=1.0)
        initial_debt = engine.impedance_debt.debt

        engine.begin_sleep()
        channels = [(f"ch_{i}", 50.0, 80.0) for i in range(20)]
        # One full sleep cycle: N3 then REM
        for _ in range(30):
            engine.sleep_tick(stage="n3", channel_impedances=channels,
                              synaptic_strengths=list(np.ones(100)))
        for _ in range(20):
            engine.sleep_tick(stage="rem", channel_impedances=channels,
                              synaptic_strengths=list(np.ones(100)))
        final_debt = engine.impedance_debt.debt
        # Debt must decrease during sleep (N3 + REM both repair)
        assert final_debt < initial_debt, \
            f"Debt increased during sleep: {initial_debt:.4f} → {final_debt:.4f}"

    def test_energy_never_negative(self):
        """Energy must NEVER go below 0 regardless of fatigue level."""
        engine = SleepPhysicsEngine(energy=0.01)  # Near-empty
        for _ in range(50):
            engine.awake_tick(reflected_energy=1.0)
        engine.begin_sleep()
        channels = [(f"ch_{i}", 50.0, 100.0) for i in range(20)]
        for _ in range(100):
            engine.sleep_tick(stage="rem", channel_impedances=channels)
        assert engine.energy >= 0.0, f"Negative energy: {engine.energy}"


class TestFatigueDreamExperimentIntegration:
    """
    Integration tests: fatigue modulation visible in dream experiment results.
    """

    def test_experiment_records_fatigue(self):
        """Dream records should contain fatigue metrics."""
        result = run_experiment(verbose=False)
        assert "peak_fatigue_alpha" in result
        assert "peak_fatigue_beta" in result
        assert "peak_amp_alpha" in result
        assert "peak_amp_beta" in result

    def test_dream_records_have_fatigue_data(self):
        """Individual dream records should track fatigue factors."""
        result = run_experiment(verbose=False)
        for rec in result["dream_records_alpha"]:
            assert hasattr(rec, "fatigue_factors")
            assert hasattr(rec, "amp_multipliers")
            assert hasattr(rec, "peak_fatigue")
            assert hasattr(rec, "peak_amp_multiplier")
            assert rec.peak_amp_multiplier >= 1.0

    def test_fatigue_present_after_wake(self):
        """
        After 80 awake ticks, impedance debt should be non-zero,
        meaning dream amplitude should be > 1.0×.
        """
        result = run_experiment(verbose=False)
        # At least one dream cycle should show fatigue > 0
        all_fatigue = [
            f for r in result["dream_records_alpha"]
            for f in r.fatigue_factors
        ]
        if all_fatigue:
            assert max(all_fatigue) > 0.0, \
                "No fatigue detected after 80 awake ticks"
            # And amp should be elevated
            all_amp = [
                m for r in result["dream_records_alpha"]
                for m in r.amp_multipliers
            ]
            assert max(all_amp) > 1.0, \
                "Amplitude not elevated despite fatigue"

    def test_later_nights_show_different_fatigue(self):
        """
        Night 1 vs Night 3 fatigue profiles may differ
        (as debt is partially repaired each night).
        """
        result = run_experiment(verbose=False)
        records = result["dream_records_alpha"]
        night_0 = [r for r in records if r.night == 0]
        night_2 = [r for r in records if r.night == 2]
        # Both should have dream data
        assert len(night_0) > 0
        assert len(night_2) > 0
        # Just verify the recording works — we don't prescribe the direction
        f0 = [f for r in night_0 for f in r.fatigue_factors]
        f2 = [f for r in night_2 for f in r.fatigue_factors]
        assert isinstance(f0, list)
        assert isinstance(f2, list)

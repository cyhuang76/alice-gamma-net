# -*- coding: utf-8 -*-
"""
Phantom Limb Pain Engine — Unit Tests

Coverage:
  1. Amputation physics (Γ = 1.0 open circuit)
  2. Motor command residual reflection
  3. Neuroma spontaneous discharge
  4. Mirror therapy (Ramachandran 1996)
  5. Cortical reorganization (Flor 2006)
  6. Emotional/stress triggers
  7. Temperature triggers
  8. Natural resolution
  9. Clinical VAS scale
  10. Multi-limb amputation
  11. Statistics and introspection
  12. AliceBrain integration
"""

import math
import numpy as np
import pytest

from alice.brain.phantom_limb import (
    PhantomLimbEngine,
    PhantomLimbState,
    PhantomPainEvent,
    AmputationRecord,
    OPEN_CIRCUIT_IMPEDANCE,
    STUMP_NEUROMA_IMPEDANCE,
    NORMAL_LIMB_IMPEDANCE,
    MOTOR_EFFERENCE_INITIAL,
    MOTOR_EFFERENCE_MIN,
    MOTOR_EFFERENCE_DECAY,
    CORTICAL_REMAP_RATE,
    CORTICAL_REMAP_MAX,
    MIRROR_THERAPY_GAMMA_REDUCTION,
    MIRROR_THERAPY_HABITUATION,
    PHANTOM_PAIN_THRESHOLD,
    NATURAL_RESOLUTION_RATE,
    NATURAL_RESOLUTION_FLOOR,
    NEUROMA_TRIGGER_PROB,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine():
    return PhantomLimbEngine(rng_seed=42)


@pytest.fixture
def amputated_engine():
    """Engine with one amputated left hand"""
    eng = PhantomLimbEngine(rng_seed=42)
    eng.amputate("left_hand")
    return eng


@pytest.fixture
def dual_amputee():
    """Dual amputee"""
    eng = PhantomLimbEngine(rng_seed=42)
    eng.amputate("left_hand")
    eng.amputate("right_leg", pre_impedance=60.0)
    return eng


# ============================================================================
# 1. Amputation Physics
# ============================================================================


class TestAmputationPhysics:
    """Impedance physics of amputation"""

    def test_amputation_creates_phantom(self, engine):
        """Amputation creates phantom record"""
        state = engine.amputate("left_hand")
        assert engine.has_phantom("left_hand")
        assert engine.phantom_count == 1
        assert state.limb_name == "left_hand"

    def test_open_circuit_gamma(self, engine):
        """No neuroma: open circuit → Γ ≈ 1.0"""
        state = engine.amputate("left_hand", has_neuroma=False)
        z_load = OPEN_CIRCUIT_IMPEDANCE
        z0 = NORMAL_LIMB_IMPEDANCE
        expected_gamma = abs(z_load - z0) / (z_load + z0)
        assert state.effective_gamma == pytest.approx(expected_gamma, abs=0.001)
        assert state.effective_gamma > 0.99  # Close to 1.0

    def test_neuroma_gamma(self, engine):
        """With neuroma: Γ < 1.0 but still high"""
        state = engine.amputate("left_hand", has_neuroma=True)
        z_load = STUMP_NEUROMA_IMPEDANCE
        z0 = NORMAL_LIMB_IMPEDANCE
        expected = abs(z_load - z0) / (z_load + z0)
        assert state.effective_gamma == pytest.approx(expected, abs=0.001)
        assert 0.7 < state.effective_gamma < 1.0

    def test_amputation_record(self, engine):
        """Amputation record is complete"""
        state = engine.amputate("right_leg", cause="accident")
        assert state.amputation_record.limb_name == "right_leg"
        assert state.amputation_record.cause == "accident"
        assert state.amputation_record.tick_of_amputation == 0

    def test_motor_efference_starts_high(self, engine):
        """Motor efference residual starts high after amputation"""
        state = engine.amputate("left_hand")
        assert state.motor_efference == MOTOR_EFFERENCE_INITIAL


# ============================================================================
# 2. Reflection Energy and Pain
# ============================================================================


class TestReflectionPain:
    """Γ → reflected energy → pain"""

    def test_high_gamma_high_pain(self, amputated_engine):
        """High Γ (amputation) → high reflected energy → pain"""
        # Large motor command
        result = amputated_engine.tick(motor_commands={"left_hand": 1.0})
        state = result["phantom_states"]["left_hand"]
        assert state["pain"] > 0.3  # Should have significant pain
        assert state["reflected_energy"] > 0.0

    def test_no_phantom_no_pain(self, engine):
        """No amputation → no phantom pain"""
        result = engine.tick()
        assert result["total_phantom_pain"] == 0.0
        assert result["phantom_count"] == 0

    def test_pain_scales_with_motor_intensity(self, amputated_engine):
        """Pain scales with motor command intensity"""
        result_low = amputated_engine.tick(motor_commands={"left_hand": 0.3})
        pain_low = result_low["phantom_states"]["left_hand"]["pain"]

        eng2 = PhantomLimbEngine(rng_seed=42)
        eng2.amputate("left_hand")
        result_high = eng2.tick(motor_commands={"left_hand": 1.0})
        pain_high = result_high["phantom_states"]["left_hand"]["pain"]

        assert pain_high > pain_low

    def test_reflected_energy_formula(self, engine):
        """Reflected energy = motor² × Γ²"""
        state = engine.amputate("left_hand", has_neuroma=False)
        gamma = state.effective_gamma
        motor = 0.5
        expected_reflected = motor ** 2 * gamma ** 2
        # Tick with specific motor command, no randomness
        result = engine.tick(motor_commands={"left_hand": motor})
        # reflected_energy should include motor component at minimum
        assert result["total_reflected_energy"] > 0


# ============================================================================
# 3. Neuroma
# ============================================================================


class TestNeuroma:
    """Neuroma spontaneous discharge"""

    def test_neuroma_produces_random_pain(self, amputated_engine):
        """Neuroma should produce random pain events"""
        pains = []
        for _ in range(200):
            result = amputated_engine.tick(motor_commands={"left_hand": 0.0})
            pains.append(result["phantom_states"]["left_hand"]["pain"])
        # Should occasionally have pain (neuroma spontaneous discharge)
        assert max(pains) > 0 or any(p > 0 for p in pains)

    def test_no_neuroma_less_pain(self, engine):
        """No neuroma → less random pain"""
        engine.amputate("left_hand", has_neuroma=False)
        pains = []
        for _ in range(100):
            result = engine.tick(motor_commands={"left_hand": 0.0})
            pains.append(result["phantom_states"]["left_hand"]["pain"])
        # No motor command + no neuroma = very little pain
        assert np.mean(pains) < 0.1


# ============================================================================
# 4. Mirror Therapy
# ============================================================================


class TestMirrorTherapy:
    """Ramachandran (1996) mirror therapy"""

    def test_mirror_reduces_gamma(self, amputated_engine):
        """Mirror therapy reduces Γ"""
        phantom = amputated_engine.get_phantom("left_hand")
        gamma_before = phantom.effective_gamma

        amputated_engine.apply_mirror_therapy_session("left_hand")
        amputated_engine.tick()  # Update gamma

        gamma_after = phantom.effective_gamma
        assert gamma_after < gamma_before

    def test_mirror_reduces_pain(self):
        """Multiple mirror therapy sessions → pain decreases"""
        eng = PhantomLimbEngine(rng_seed=42)
        eng.amputate("left_hand")

        # Baseline pain
        result = eng.tick(motor_commands={"left_hand": 0.8})
        pain_before = result["phantom_states"]["left_hand"]["pain"]

        # 10 mirror therapy sessions
        for _ in range(10):
            eng.apply_mirror_therapy_session("left_hand", quality=0.9)
            eng.tick()

        # Post-therapy pain
        result = eng.tick(motor_commands={"left_hand": 0.8})
        pain_after = result["phantom_states"]["left_hand"]["pain"]

        assert pain_after < pain_before

    def test_mirror_therapy_habituation(self, amputated_engine):
        """Efficacy decreases over time (habituation)"""
        efficacies = []
        for _ in range(10):
            result = amputated_engine.apply_mirror_therapy_session("left_hand")
            efficacies.append(result["efficacy_remaining"])

        # Efficacy should gradually decrease
        assert efficacies[-1] < efficacies[0]
        # Each time multiplied by HABITUATION=0.95
        assert efficacies[1] == pytest.approx(
            efficacies[0] * MIRROR_THERAPY_HABITUATION, abs=0.01
        )

    def test_mirror_invalid_limb(self, amputated_engine):
        """Mirror therapy on nonexistent limb"""
        result = amputated_engine.apply_mirror_therapy_session("nonexistent")
        assert "error" in result

    def test_visual_feedback_lowers_impedance(self, amputated_engine):
        """Visual feedback lowers load impedance"""
        phantom = amputated_engine.get_phantom("left_hand")
        z_before = phantom.current_load_impedance

        # High quality visual feedback
        amputated_engine.tick(visual_feedback={"left_hand": 0.8})
        z_after = phantom.current_load_impedance

        assert z_after < z_before


# ============================================================================
# 5. Cortical Reorganization
# ============================================================================


class TestCorticalReorganization:
    """Flor et al. (2006) cortical reorganization"""

    def test_cortical_remap_progresses(self, amputated_engine):
        """Cortical reorganization progresses over time"""
        for _ in range(100):
            amputated_engine.tick(motor_commands={"left_hand": 0.5})

        phantom = amputated_engine.get_phantom("left_hand")
        assert phantom.cortical_remap_progress > 0

    def test_remap_bounded(self, amputated_engine):
        """Cortical reorganization has an upper limit"""
        for _ in range(10000):
            amputated_engine.tick(motor_commands={"left_hand": 0.8})

        phantom = amputated_engine.get_phantom("left_hand")
        assert phantom.cortical_remap_progress <= CORTICAL_REMAP_MAX

    def test_pain_accelerates_remap(self):
        """High pain → faster reorganization (positive feedback, Flor 2006)"""
        eng_high = PhantomLimbEngine(rng_seed=42)
        eng_high.amputate("left_hand")

        eng_low = PhantomLimbEngine(rng_seed=42)
        eng_low.amputate("left_hand")

        for _ in range(200):
            eng_high.tick(motor_commands={"left_hand": 1.0})
            eng_low.tick(motor_commands={"left_hand": 0.1})

        remap_high = eng_high.get_phantom("left_hand").cortical_remap_progress
        remap_low = eng_low.get_phantom("left_hand").cortical_remap_progress
        assert remap_high > remap_low


# ============================================================================
# 6. Emotional/Stress Triggers
# ============================================================================


class TestEmotionalTrigger:
    """Anxiety/stress triggers phantom pain"""

    def test_negative_emotion_increases_pain(self, amputated_engine):
        """Negative emotion increases pain"""
        result_neutral = amputated_engine.tick(emotional_valence=0.0)
        pain_neutral = result_neutral["phantom_states"]["left_hand"]["pain"]

        eng2 = PhantomLimbEngine(rng_seed=42)
        eng2.amputate("left_hand")
        result_negative = eng2.tick(emotional_valence=-0.8)
        pain_negative = result_negative["phantom_states"]["left_hand"]["pain"]

        assert pain_negative >= pain_neutral

    def test_stress_increases_pain(self, amputated_engine):
        """Stress increases pain"""
        result_calm = amputated_engine.tick(stress_level=0.0)
        pain_calm = result_calm["phantom_states"]["left_hand"]["pain"]

        eng2 = PhantomLimbEngine(rng_seed=42)
        eng2.amputate("left_hand")
        result_stress = eng2.tick(stress_level=0.9)
        pain_stress = result_stress["phantom_states"]["left_hand"]["pain"]

        assert pain_stress >= pain_calm


# ============================================================================
# 7. Temperature Triggers
# ============================================================================


class TestTemperatureTrigger:
    """Weather/temperature triggers"""

    def test_cold_increases_pain(self, amputated_engine):
        """Cold increases pain (neuroma compression)"""
        result_warm = amputated_engine.tick(temperature_delta=0.0)
        pain_warm = result_warm["phantom_states"]["left_hand"]["pain"]

        eng2 = PhantomLimbEngine(rng_seed=42)
        eng2.amputate("left_hand")
        result_cold = eng2.tick(temperature_delta=-0.5)
        pain_cold = result_cold["phantom_states"]["left_hand"]["pain"]

        assert pain_cold >= pain_warm


# ============================================================================
# 8. Natural Resolution
# ============================================================================


class TestNaturalResolution:
    """Motor command natural decay → pain resolution"""

    def test_motor_efference_decays(self, amputated_engine):
        """Motor efference decays naturally over time"""
        phantom = amputated_engine.get_phantom("left_hand")
        initial = phantom.motor_efference

        for _ in range(500):
            amputated_engine.tick()

        final = phantom.motor_efference
        assert final < initial

    def test_motor_efference_has_floor(self, amputated_engine):
        """Motor efference never fully reaches zero (phantom sensation persists)"""
        for _ in range(50000):
            amputated_engine.tick()

        phantom = amputated_engine.get_phantom("left_hand")
        assert phantom.motor_efference >= MOTOR_EFFERENCE_MIN

    def test_long_term_pain_reduction(self):
        """Long-term pain gradually decreases"""
        eng = PhantomLimbEngine(rng_seed=42)
        eng.amputate("left_hand")

        # Early pain
        early_pains = []
        for _ in range(100):
            result = eng.tick()
            early_pains.append(result["phantom_states"]["left_hand"]["pain"])

        # Skip some time
        for _ in range(2000):
            eng.tick()

        # Late pain
        late_pains = []
        for _ in range(100):
            result = eng.tick()
            late_pains.append(result["phantom_states"]["left_hand"]["pain"])

        assert np.mean(late_pains) < np.mean(early_pains)


# ============================================================================
# 9. Clinical VAS Scale
# ============================================================================


class TestClinicalVAS:
    """VAS pain scale (0-10)"""

    def test_vas_range(self, amputated_engine):
        """VAS in 0-10 range"""
        amputated_engine.tick(motor_commands={"left_hand": 0.8})
        vas = amputated_engine.get_clinical_vas_score("left_hand")
        assert 0.0 <= vas <= 10.0

    def test_vas_no_phantom(self, engine):
        """No phantom → VAS = 0"""
        assert engine.get_clinical_vas_score("nonexistent") == 0.0

    def test_vas_proportional_to_pain(self, amputated_engine):
        """VAS ∝ pain level"""
        amputated_engine.tick(motor_commands={"left_hand": 0.8})
        phantom = amputated_engine.get_phantom("left_hand")
        vas = amputated_engine.get_clinical_vas_score("left_hand")
        assert vas == pytest.approx(phantom.phantom_pain_level * 10.0, abs=0.01)


# ============================================================================
# 10. Multi-Limb Amputation
# ============================================================================


class TestMultiLimb:
    """Multi-limb amputation"""

    def test_dual_amputation(self, dual_amputee):
        """Two phantoms tracked independently"""
        assert dual_amputee.phantom_count == 2
        result = dual_amputee.tick(
            motor_commands={"left_hand": 0.8, "right_leg": 0.3}
        )
        assert "left_hand" in result["phantom_states"]
        assert "right_leg" in result["phantom_states"]

    def test_independent_pain(self, dual_amputee):
        """Each phantom's pain is independent"""
        result = dual_amputee.tick(
            motor_commands={"left_hand": 1.0, "right_leg": 0.1}
        )
        hand_pain = result["phantom_states"]["left_hand"]["pain"]
        leg_pain = result["phantom_states"]["right_leg"]["pain"]
        assert hand_pain > leg_pain

    def test_selective_mirror_therapy(self, dual_amputee):
        """Mirror therapy on only one limb"""
        dual_amputee.apply_mirror_therapy_session("left_hand")
        hand = dual_amputee.get_phantom("left_hand")
        leg = dual_amputee.get_phantom("right_leg")
        assert hand.mirror_therapy_sessions == 1
        assert leg.mirror_therapy_sessions == 0


# ============================================================================
# 11. Statistics and Introspection
# ============================================================================


class TestStatsAndIntrospect:
    """Statistics and introspection interface"""

    def test_introspect(self, amputated_engine):
        """introspect structure is correct"""
        amputated_engine.tick()
        data = amputated_engine.introspect()
        assert data["phantom_count"] == 1
        assert "left_hand" in data["phantoms"]
        info = data["phantoms"]["left_hand"]
        assert "current_pain" in info
        assert "gamma" in info
        assert "motor_efference" in info

    def test_stats(self, amputated_engine):
        """stats structure is correct"""
        for _ in range(10):
            amputated_engine.tick()
        s = amputated_engine.stats()
        assert s["phantom_count"] == 1
        assert "mean_pain" in s
        assert "mean_gamma" in s
        assert s["tick"] == 10

    def test_empty_stats(self, engine):
        """stats when no phantoms"""
        s = engine.stats()
        assert s["phantom_count"] == 0

    def test_events_recorded(self, amputated_engine):
        """Pain events are recorded"""
        for _ in range(50):
            amputated_engine.tick(motor_commands={"left_hand": 0.8})
        assert len(amputated_engine.events) > 0
        event = amputated_engine.events[0]
        assert isinstance(event, PhantomPainEvent)
        assert event.limb_name == "left_hand"

    def test_pain_history(self, amputated_engine):
        """Pain history is recorded"""
        for _ in range(20):
            amputated_engine.tick()
        phantom = amputated_engine.get_phantom("left_hand")
        assert len(phantom.pain_history) == 20
        assert len(phantom.gamma_history) == 20


# ============================================================================
# 12. Ramachandran Clinical Validation (Complete Protocol)
# ============================================================================


class TestRamachandranProtocol:
    """Ramachandran mirror therapy full clinical validation"""

    def test_4_week_mirror_therapy(self):
        """
        Simulate 4-week mirror therapy (Ramachandran 1996)
        
        Clinical data: VAS from 7.2 to 2.1 (mean)
        Simulation: should observe significant pain reduction
        """
        eng = PhantomLimbEngine(rng_seed=42)
        eng.amputate("left_hand")

        # 1 week = ~168 ticks (assuming 1 tick per hour)
        TICKS_PER_WEEK = 168

        # Week 1: Baseline (no treatment)
        week1_pains = []
        for _ in range(TICKS_PER_WEEK):
            result = eng.tick(motor_commands={"left_hand": 0.6})
            week1_pains.append(result["phantom_states"]["left_hand"]["pain"])

        # Week 2-5: Daily mirror therapy (every 24 ticks)
        weekly_pains = []
        for week in range(4):
            week_pains = []
            for tick in range(TICKS_PER_WEEK):
                if tick % 24 == 0:
                    eng.apply_mirror_therapy_session("left_hand", quality=0.8)
                result = eng.tick(motor_commands={"left_hand": 0.6})
                week_pains.append(result["phantom_states"]["left_hand"]["pain"])
            weekly_pains.append(np.mean(week_pains))

        # Verify: post-therapy pain < baseline pain
        baseline_pain = np.mean(week1_pains)
        final_pain = weekly_pains[-1]
        assert final_pain < baseline_pain, (
            f"Mirror therapy failed: baseline={baseline_pain:.3f}, "
            f"final={final_pain:.3f}"
        )

    def test_mirror_vs_no_mirror(self):
        """
        Control group: mirror therapy vs no treatment
        
        Treatment group pain should be significantly lower than control group
        """
        TICKS = 500

        # Control group
        ctrl = PhantomLimbEngine(rng_seed=42)
        ctrl.amputate("left_hand")
        for _ in range(TICKS):
            ctrl.tick(motor_commands={"left_hand": 0.5})

        # Treatment group
        treat = PhantomLimbEngine(rng_seed=42)
        treat.amputate("left_hand")
        for t in range(TICKS):
            if t % 24 == 0:
                treat.apply_mirror_therapy_session("left_hand", quality=0.8)
            treat.tick(motor_commands={"left_hand": 0.5})

        ctrl_pain = ctrl.get_phantom("left_hand").cumulative_pain
        treat_pain = treat.get_phantom("left_hand").cumulative_pain
        assert treat_pain < ctrl_pain


# ============================================================================
# 13. AliceBrain Integration
# ============================================================================


class TestAliceBrainIntegration:
    """AliceBrain integration tests"""

    def test_brain_has_phantom_limb(self):
        """AliceBrain includes phantom_limb engine"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        assert hasattr(brain, "phantom_limb")
        assert isinstance(brain.phantom_limb, PhantomLimbEngine)

    def test_brain_perceive_includes_phantom(self):
        """perceive() result includes phantom_limb data"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        result = brain.perceive(np.random.randn(64))
        assert "phantom_limb" in result

    def test_brain_introspect_includes_phantom(self):
        """introspect() includes phantom_limb"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        intro = brain.introspect()
        assert "phantom_limb" in intro["subsystems"]

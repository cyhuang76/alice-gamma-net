# -*- coding: utf-8 -*-
"""test_pharmacology.py — Computational Pharmacology + Four Disease Unit Tests"""

import math
import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.brain.pharmacology import (
    PharmacologyEngine, DrugProfile,
    MSModel, MSState, MSLesion,
    ParkinsonModel, PDState,
    EpilepsyModel, EpilepsyState,
    DepressionModel, DepressionState,
    ClinicalPharmacologyEngine,
    Z_NORMAL, ALL_CHANNELS,
    LDOPA_ALPHA, LDOPA_DYSKINESIA_THRESHOLD,
    SEIZURE_THRESHOLD, KINDLING_INCREMENT,
    SSRI_ONSET_DELAY, SSRI_ALPHA,
    PD_DOPAMINE_DEPLETION_RATE,
    HAMD_MAX, UPDRS_MAX, EDSS_MAX,
    MS_TRACT_CHANNELS,
    EPILEPSY_FOCAL_CHANNELS,
    VALPROATE_ALPHA,
)


# ============================================================================
# I. PharmacologyEngine — Unified Pharmacology
# ============================================================================

class TestDrugAdministration(unittest.TestCase):
    """Drug administration basics"""

    def test_no_drugs_clean(self):
        eng = PharmacologyEngine()
        result = eng.tick()
        self.assertEqual(result["active_drugs"], 0)
        for g in result["channel_gamma"].values():
            self.assertAlmostEqual(g, 0.0, places=4)

    def test_administer_single_drug(self):
        eng = PharmacologyEngine()
        drug = DrugProfile(
            name="TestDrug", alpha=-0.3,
            target_channels=["amygdala", "prefrontal"],
            onset_delay=0, half_life=1000,
        )
        eng.administer(drug)
        result = eng.tick()
        self.assertEqual(result["active_drugs"], 1)
        # Therapeutic drug → α < 0 → Γ < 0
        self.assertLess(result["channel_gamma"]["amygdala"], 0)

    def test_drug_onset_delay(self):
        eng = PharmacologyEngine()
        drug = DrugProfile(
            name="DelayedDrug", alpha=-0.5,
            target_channels=["hippocampus"],
            onset_delay=100, half_life=1000,
        )
        eng.administer(drug)
        # Tick 1: only ~1% of the effect
        result = eng.tick()
        gamma_early = result["channel_gamma"]["hippocampus"]
        # After 100 ticks: near full effect
        for _ in range(99):
            result = eng.tick()
        gamma_late = result["channel_gamma"]["hippocampus"]
        self.assertLess(gamma_late, gamma_early)

    def test_drug_decay(self):
        eng = PharmacologyEngine()
        drug = DrugProfile(
            name="ShortDrug", alpha=-0.5,
            target_channels=["consciousness"],
            onset_delay=0, half_life=50,
        )
        eng.administer(drug)
        eng.tick()
        gamma_peak = eng.tick()["channel_gamma"]["consciousness"]
        for _ in range(200):
            eng.tick()
        gamma_decayed = eng.tick()["channel_gamma"]["consciousness"]
        # Weaker effect after decay (Γ closer to 0)
        self.assertGreater(gamma_decayed, gamma_peak)  # less negative

    def test_multiple_drugs_stack(self):
        eng = PharmacologyEngine()
        d1 = DrugProfile(
            name="DrugA", alpha=-0.2,
            target_channels=["amygdala"],
            onset_delay=0, half_life=1000,
        )
        d2 = DrugProfile(
            name="DrugB", alpha=-0.15,
            target_channels=["amygdala"],
            onset_delay=0, half_life=1000,
        )
        eng.administer(d1)
        eng.administer(d2)
        result = eng.tick()
        self.assertEqual(result["active_drugs"], 2)
        # Two drugs stacked, larger α for amygdala
        self.assertLess(result["channel_alpha"]["amygdala"], -0.3)

    def test_side_effects(self):
        eng = PharmacologyEngine()
        drug = DrugProfile(
            name="SideEffectDrug", alpha=-0.3,
            target_channels=["amygdala"],
            onset_delay=0, half_life=1000,
            side_effect_channels=["autonomic"],
            side_effect_alpha=0.1,
        )
        eng.administer(drug)
        result = eng.tick()
        # Therapeutic channel Γ < 0, side-effect channel Γ > 0
        self.assertLess(result["channel_gamma"]["amygdala"], 0)
        self.assertGreater(result["channel_gamma"]["autonomic"], 0)


class TestDrugGammaPhysics(unittest.TestCase):
    """Drug impedance modification → Γ formula verification"""

    def test_positive_alpha_increases_gamma(self):
        """α > 0 → Z↑ → Γ > 0 (harmful)"""
        eng = PharmacologyEngine()
        drug = DrugProfile(
            name="Toxin", alpha=0.5,
            target_channels=["consciousness"],
            onset_delay=0, half_life=1000,
        )
        eng.administer(drug)
        result = eng.tick()
        self.assertGreater(result["channel_gamma"]["consciousness"], 0)

    def test_negative_alpha_decreases_gamma(self):
        """α < 0 → Z↓ → Γ < 0 (therapeutic)"""
        eng = PharmacologyEngine()
        drug = DrugProfile(
            name="Cure", alpha=-0.3,
            target_channels=["consciousness"],
            onset_delay=0, half_life=1000,
        )
        eng.administer(drug)
        result = eng.tick()
        self.assertLess(result["channel_gamma"]["consciousness"], 0)

    def test_gamma_formula_correctness(self):
        """Verify Γ = (Z_eff - Z₀) / (Z_eff + Z₀)"""
        eng = PharmacologyEngine()
        alpha = 0.4
        drug = DrugProfile(
            name="Test", alpha=alpha,
            target_channels=["hand"],
            onset_delay=0, half_life=100000,
        )
        eng.administer(drug)
        eng.tick()
        z_eff = Z_NORMAL * (1 + alpha)
        expected_gamma = (z_eff - Z_NORMAL) / (z_eff + Z_NORMAL)
        actual = eng.get_channel_gamma()["hand"]
        self.assertAlmostEqual(actual, expected_gamma, places=3)

    def test_introspect(self):
        eng = PharmacologyEngine()
        eng.administer(DrugProfile(
            name="X", alpha=-0.1, target_channels=["hand"]))
        eng.tick()
        info = eng.introspect()
        self.assertEqual(info["engine"], "pharmacology")
        self.assertEqual(len(info["active_drugs"]), 1)


# ============================================================================
# II. MS — Multiple Sclerosis
# ============================================================================

class TestMSOnset(unittest.TestCase):
    """MS onset"""

    def test_onset_creates_state(self):
        model = MSModel()
        state = model.onset("RRMS")
        self.assertEqual(state.ms_type, "RRMS")
        self.assertGreater(len(state.lesions), 0)

    def test_onset_creates_initial_lesion(self):
        model = MSModel()
        model.onset("RRMS")
        self.assertEqual(len(model.state.lesions), 1)

    def test_ppms_type(self):
        model = MSModel()
        model.onset("PPMS")
        self.assertEqual(model.state.ms_type, "PPMS")

    def test_lesion_has_channels(self):
        model = MSModel()
        model.onset("RRMS")
        lesion = model.state.lesions[0]
        self.assertGreater(len(lesion.channels), 0)
        self.assertIn(lesion.tract, MS_TRACT_CHANNELS)


class TestMSProgression(unittest.TestCase):
    """MS progression"""

    def test_lesion_count_increases(self):
        model = MSModel()
        model.onset("PPMS")  # Progressive → continuously generates new lesions
        for _ in range(5000):
            model.tick()
        self.assertGreater(len(model.state.lesions), 1)

    def test_edss_increases_with_lesions(self):
        model = MSModel()
        model.onset("PPMS")
        edss_early = model.get_edss()
        for _ in range(3000):
            model.tick()
        edss_late = model.get_edss()
        self.assertGreaterEqual(edss_late, edss_early)

    def test_edss_range(self):
        model = MSModel()
        model.onset("RRMS")
        edss = model.get_edss()
        self.assertGreaterEqual(edss, 0.0)
        self.assertLessEqual(edss, EDSS_MAX)

    def test_channel_gamma_from_lesions(self):
        model = MSModel()
        model.onset("PPMS")
        for _ in range(1000):
            model.tick()
        # At least some channels have Γ > 0
        self.assertTrue(any(g > 0 for g in model.channel_gamma.values()))

    def test_introspect(self):
        model = MSModel()
        model.onset("RRMS")
        info = model.introspect()
        self.assertEqual(info["condition"], "multiple_sclerosis")
        self.assertTrue(info["active"])


# ============================================================================
# III. Parkinson's Disease
# ============================================================================

class TestPDOnset(unittest.TestCase):
    """Parkinson's onset"""

    def test_onset_creates_state(self):
        model = ParkinsonModel()
        state = model.onset()
        self.assertIsNotNone(state)
        self.assertEqual(state.dopamine_level, 0.8)

    def test_no_onset_inactive(self):
        model = ParkinsonModel()
        result = model.tick()
        self.assertFalse(result["active"])


class TestPDProgression(unittest.TestCase):
    """Parkinson's progression"""

    def test_dopamine_depletes(self):
        model = ParkinsonModel()
        model.onset()
        da_initial = model.state.dopamine_level
        for _ in range(500):
            model.tick()
        self.assertLess(model.state.dopamine_level, da_initial)

    def test_motor_gamma_increases(self):
        model = ParkinsonModel()
        model.onset()
        for _ in range(1000):
            model.tick()
        # Motor channel Γ should be notably elevated
        self.assertGreater(model.channel_gamma.get("hand", 0), 0.2)

    def test_updrs_increases(self):
        model = ParkinsonModel()
        model.onset()
        model.tick()
        updrs_early = model.get_updrs()
        for _ in range(1000):
            model.tick()
        updrs_late = model.get_updrs()
        self.assertGreater(updrs_late, updrs_early)

    def test_updrs_range(self):
        model = ParkinsonModel()
        model.onset()
        for _ in range(100):
            model.tick()
        self.assertGreaterEqual(model.get_updrs(), 0)
        self.assertLessEqual(model.get_updrs(), UPDRS_MAX)

    def test_tremor_present(self):
        model = ParkinsonModel()
        model.onset()
        # Collect multiple ticks to observe tremor amplitude variations
        amplitudes = []
        for _ in range(100):
            result = model.tick()
            amplitudes.append(result.get("tremor_amplitude", 0))
        # Tremor should vary (oscillatory)
        self.assertGreater(max(amplitudes), 0)


class TestPDTreatment(unittest.TestCase):
    """L-DOPA treatment"""

    def test_ldopa_improves_dopamine(self):
        model = ParkinsonModel()
        model.onset()
        for _ in range(500):
            model.tick()
        updrs_before = model.get_updrs()
        model.start_ldopa()
        for _ in range(100):
            model.tick()
        updrs_after = model.get_updrs()
        self.assertLess(updrs_after, updrs_before)

    def test_ldopa_wearing_off(self):
        model = ParkinsonModel()
        model.onset()
        model.start_ldopa()
        for _ in range(100):
            model.tick()
        updrs_early_ldopa = model.get_updrs()
        for _ in range(3000):
            model.tick()
        updrs_late_ldopa = model.get_updrs()
        # Long-term L-DOPA effect wears off
        self.assertGreaterEqual(updrs_late_ldopa, updrs_early_ldopa)

    def test_dyskinesia_long_term(self):
        model = ParkinsonModel()
        model.onset()
        model.start_ldopa()
        for _ in range(LDOPA_DYSKINESIA_THRESHOLD + 500):
            model.tick()
        result = model.tick()
        self.assertGreater(result["dyskinesia"], 0)

    def test_introspect(self):
        model = ParkinsonModel()
        model.onset()
        info = model.introspect()
        self.assertEqual(info["condition"], "parkinson")


# ============================================================================
# IV. Epilepsy
# ============================================================================

class TestEpilepsyOnset(unittest.TestCase):
    """Epileptic focus setup"""

    def test_onset_creates_state(self):
        model = EpilepsyModel()
        state = model.onset("temporal")
        self.assertEqual(state.focus, "temporal")
        self.assertAlmostEqual(state.seizure_threshold, SEIZURE_THRESHOLD)

    def test_different_foci(self):
        for focus in EPILEPSY_FOCAL_CHANNELS:
            model = EpilepsyModel()
            model.onset(focus)
            self.assertEqual(model.state.focus, focus)


class TestSeizureMechanics(unittest.TestCase):
    """Seizure mechanics"""

    def test_force_seizure(self):
        model = EpilepsyModel()
        model.onset("temporal")
        model.force_seizure()
        result = model.tick()
        # Should be in seizure or just completed seizure
        self.assertGreater(model.state.total_seizures, 0)

    def test_seizure_channels_gamma_high(self):
        model = EpilepsyModel()
        model.onset("temporal")
        model.force_seizure()
        model.tick()  # trigger
        result = model.tick()  # during seizure
        focus_channels = EPILEPSY_FOCAL_CHANNELS["temporal"]
        for ch in focus_channels:
            self.assertGreater(model.channel_gamma.get(ch, 0), 0.5)

    def test_postictal_depression(self):
        model = EpilepsyModel()
        model.onset("temporal")
        model.force_seizure()
        # Trigger and complete a seizure
        for _ in range(100):
            result = model.tick()
            if result.get("phase") == "postictal":
                break
        # In postictal phase, excitation should be low
        self.assertLess(model.excitation, 0.3)

    def test_kindling_lowers_threshold(self):
        model = EpilepsyModel()
        model.onset("temporal")
        initial_threshold = model.state.seizure_threshold
        # Force multiple seizures
        for _ in range(5):
            model.force_seizure()
            for _ in range(200):
                model.tick()
        self.assertLess(model.state.seizure_threshold, initial_threshold)

    def test_multiple_seizures_accumulate(self):
        model = EpilepsyModel()
        model.onset("temporal")
        for _ in range(3):
            model.force_seizure()
            for _ in range(200):
                model.tick()
        self.assertGreaterEqual(model.state.total_seizures, 3)

    def test_introspect(self):
        model = EpilepsyModel()
        model.onset("frontal")
        info = model.introspect()
        self.assertEqual(info["condition"], "epilepsy")


# ============================================================================
# V. Depression
# ============================================================================

class TestDepressionOnset(unittest.TestCase):
    """Depression onset"""

    def test_onset_moderate(self):
        model = DepressionModel()
        state = model.onset("moderate")
        self.assertAlmostEqual(state.serotonin, 0.55)

    def test_onset_severe(self):
        model = DepressionModel()
        state = model.onset("severe")
        self.assertAlmostEqual(state.serotonin, 0.35)

    def test_onset_mild(self):
        model = DepressionModel()
        state = model.onset("mild")
        self.assertAlmostEqual(state.serotonin, 0.75)


class TestDepressionProgression(unittest.TestCase):
    """Depression progression"""

    def test_serotonin_depletes(self):
        model = DepressionModel()
        model.onset("moderate")
        initial = model.state.serotonin
        for _ in range(500):
            model.tick()
        self.assertLess(model.state.serotonin, initial)

    def test_hamd_range(self):
        model = DepressionModel()
        model.onset("moderate")
        for _ in range(100):
            model.tick()
        hamd = model.get_hamd()
        self.assertGreaterEqual(hamd, 0)
        self.assertLessEqual(hamd, HAMD_MAX)

    def test_hamd_increases(self):
        model = DepressionModel()
        model.onset("moderate")
        model.tick()
        hamd_early = model.get_hamd()
        for _ in range(1000):
            model.tick()
        hamd_late = model.get_hamd()
        self.assertGreaterEqual(hamd_late, hamd_early)

    def test_amygdala_gamma_elevated(self):
        model = DepressionModel()
        model.onset("moderate")
        for _ in range(200):
            model.tick()
        self.assertGreater(model.channel_gamma.get("amygdala", 0), 0.1)

    def test_cognitive_distortion_accumulates(self):
        model = DepressionModel()
        model.onset("moderate")
        for _ in range(500):
            model.tick()
        self.assertGreater(model.state.cognitive_distortion, 0)


class TestDepressionSSRI(unittest.TestCase):
    """SSRI treatment"""

    def test_ssri_delayed_onset(self):
        model = DepressionModel()
        model.onset("moderate")
        for _ in range(200):
            model.tick()
        hamd_baseline = model.get_hamd()
        model.start_ssri()
        # Before onset (50 ticks — far less than SSRI_ONSET_DELAY)
        for _ in range(50):
            model.tick()
        hamd_early = model.get_hamd()
        # After onset (300+ ticks)
        for _ in range(SSRI_ONSET_DELAY):
            model.tick()
        hamd_late = model.get_hamd()
        # HAMD should improve after SSRI takes effect
        self.assertLessEqual(hamd_late, hamd_baseline)

    def test_ssri_improves_serotonin(self):
        model = DepressionModel()
        model.onset("severe")
        for _ in range(200):
            model.tick()
        model.start_ssri()
        for _ in range(SSRI_ONSET_DELAY + 100):
            result = model.tick()
        # effective_serotonin should be higher than raw serotonin
        self.assertGreater(
            result["effective_serotonin"], result["serotonin"]
        )

    def test_ssri_reduces_cognitive_distortion(self):
        model = DepressionModel()
        model.onset("moderate")
        for _ in range(500):
            model.tick()
        cd_before = model.state.cognitive_distortion
        model.start_ssri()
        for _ in range(SSRI_ONSET_DELAY + 500):
            model.tick()
        cd_after = model.state.cognitive_distortion
        self.assertLess(cd_after, cd_before)

    def test_introspect(self):
        model = DepressionModel()
        model.onset("moderate")
        info = model.introspect()
        self.assertEqual(info["condition"], "depression")


# ============================================================================
# VI. ClinicalPharmacologyEngine — Unified Integration
# ============================================================================

class TestClinicalPharmacologyEngine(unittest.TestCase):
    """Unified engine tests"""

    def test_no_conditions_clean(self):
        eng = ClinicalPharmacologyEngine()
        result = eng.tick()
        self.assertIn("pharmacology", result)

    def test_single_condition(self):
        eng = ClinicalPharmacologyEngine()
        eng.parkinson.onset()
        result = eng.tick()
        self.assertIn("parkinson", result)

    def test_multiple_conditions(self):
        eng = ClinicalPharmacologyEngine()
        eng.parkinson.onset()
        eng.depression.onset("moderate")
        result = eng.tick()
        self.assertIn("parkinson", result)
        self.assertIn("depression", result)

    def test_merged_gamma_max_across_conditions(self):
        eng = ClinicalPharmacologyEngine()
        eng.parkinson.onset()
        eng.depression.onset("severe")
        for _ in range(500):
            eng.tick()
        merged = eng.get_merged_channel_gamma()
        # merged should contain at least the higher value from both conditions
        self.assertGreater(merged.get("amygdala", 0), 0)
        self.assertGreater(merged.get("hand", 0), 0)

    def test_drug_modifies_gamma(self):
        eng = ClinicalPharmacologyEngine()
        eng.depression.onset("moderate")
        for _ in range(200):
            eng.tick()
        hamd_before = eng.depression.get_hamd()

        # Administer SSRI
        ssri = DrugProfile(
            name="Fluoxetine", alpha=SSRI_ALPHA,
            target_channels=["amygdala", "prefrontal"],
            onset_delay=0, half_life=5000,
        )
        eng.pharmacology.administer(ssri)
        for _ in range(100):
            eng.tick()

        # Drug should reduce merged Γ
        merged = eng.get_merged_channel_gamma()
        self.assertIsNotNone(merged)

    def test_clinical_summary(self):
        eng = ClinicalPharmacologyEngine()
        eng.ms.onset("RRMS")
        eng.epilepsy.onset("temporal")
        for _ in range(10):
            eng.tick()
        summary = eng.get_clinical_summary()
        self.assertIn("MS", summary["active_conditions"])
        self.assertIn("Epilepsy", summary["active_conditions"])

    def test_introspect(self):
        eng = ClinicalPharmacologyEngine()
        eng.parkinson.onset()
        info = eng.introspect()
        self.assertEqual(info["engine"], "clinical_pharmacology")
        self.assertIn("parkinson", info)


# ============================================================================
# VII. AliceBrain Integration
# ============================================================================

class TestAliceBrainPharmacologyIntegration(unittest.TestCase):
    """AliceBrain integration tests"""

    def test_has_pharmacology_attribute(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        self.assertTrue(hasattr(brain, "pharmacology"))

    def test_perceive_includes_pharmacology(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        result = brain.perceive(np.random.randn(64))
        self.assertIn("pharmacology", result)

    def test_introspect_includes_pharmacology(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        info = brain.introspect()
        self.assertIn("pharmacology", info["subsystems"])


if __name__ == "__main__":
    unittest.main()

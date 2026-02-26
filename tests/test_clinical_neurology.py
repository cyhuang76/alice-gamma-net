# -*- coding: utf-8 -*-
"""test_clinical_neurology.py — Five Major Clinical Neurological Disease Physics Model Tests

Coverage: Stroke, ALS, Dementia, Alzheimer's, Cerebral Palsy
"""

import math
import random
import unittest

from alice.brain.clinical_neurology import (
    ClinicalNeurologyEngine,
    StrokeModel, StrokeEvent,
    ALSModel, ALSState,
    DementiaModel, DementiaState,
    AlzheimersModel, AlzheimersState,
    CerebralPalsyModel, CerebralPalsyState,
    VASCULAR_TERRITORIES, NIHSS_MAX, MMSE_MAX,
    ALS_SPREAD_ORDER_LIMB, ALS_SPREAD_ORDER_BULBAR,
    ALS_RILUZOLE_FACTOR, ALS_SPREAD_THRESHOLD,
    BRAAK_STAGE_PROFILES, GMFCS_BASELINE_GAMMA,
    DEMENTIA_DOMAINS, ALL_CHANNELS,
    Z_NORMAL, Z_ISCHEMIC,
)


# ====================================================================
# Stroke
# ====================================================================

class TestStrokeInduction(unittest.TestCase):
    """Stroke induction — Vascular territory → Γ mutation"""

    def test_mca_stroke_creates_event(self):
        m = StrokeModel()
        ev = m.induce("MCA", 0.8)
        self.assertEqual(ev.territory, "MCA")
        self.assertGreater(len(ev.core_channels), 0)
        self.assertEqual(len(m.strokes), 1)

    def test_stroke_gamma_high(self):
        m = StrokeModel()
        m.induce("MCA", 1.0)
        for ch in VASCULAR_TERRITORIES["MCA"][:3]:  # core channels
            self.assertGreater(m.channel_gamma.get(ch, 0), 0.5)

    def test_unknown_territory_raises(self):
        m = StrokeModel()
        with self.assertRaises(ValueError):
            m.induce("UNKNOWN")

    def test_severity_clamp(self):
        m = StrokeModel()
        ev = m.induce("MCA", 1.5)
        self.assertLessEqual(ev.severity, 1.0)

    def test_diaschisis_effect(self):
        """Diaschisis: non-lesioned channels also show mild Γ elevation"""
        m = StrokeModel()
        m.induce("MCA", 0.8)
        non_mca = [ch for ch in ALL_CHANNELS
                   if ch not in VASCULAR_TERRITORIES["MCA"]]
        for ch in non_mca:
            self.assertGreater(m.channel_gamma.get(ch, 0), 0)


class TestStrokeNIHSS(unittest.TestCase):
    """NIHSS score (Brott 1989)"""

    def test_no_stroke_nihss_zero(self):
        m = StrokeModel()
        self.assertEqual(m.get_nihss(), 0)

    def test_severe_stroke_high_nihss(self):
        m = StrokeModel()
        m.induce("MCA", 1.0)
        self.assertGreater(m.get_nihss(), 10)

    def test_nihss_max_42(self):
        m = StrokeModel()
        for territory in VASCULAR_TERRITORIES:
            m.induce(territory, 1.0)
        self.assertLessEqual(m.get_nihss(), NIHSS_MAX)

    def test_mild_stroke_low_nihss(self):
        m = StrokeModel()
        m.induce("MCA", 0.3)
        self.assertLess(m.get_nihss(), 15)


class TestStrokePenumbra(unittest.TestCase):
    """Penumbra — Irreversible core vs salvageable penumbra"""

    def test_reperfusion_within_window(self):
        m = StrokeModel()
        m.induce("MCA", 0.8)
        self.assertTrue(m.reperfuse(0))

    def test_reperfusion_after_window_fails(self):
        m = StrokeModel()
        m.induce("MCA", 0.8)
        # Exceeds 4.5h time window (270 ticks)
        for _ in range(280):
            m.tick()
        self.assertFalse(m.reperfuse(0))

    def test_penumbra_recovers_faster_with_reperfusion(self):
        """Reperfusion group vs control: penumbra recovery speed"""
        m_reperfused = StrokeModel()
        m_control = StrokeModel()

        m_reperfused.induce("MCA", 0.8)
        m_control.induce("MCA", 0.8)
        m_reperfused.reperfuse(0)

        for _ in range(100):
            m_reperfused.tick()
            m_control.tick()

        # Reperfusion group NIHSS should be lower
        self.assertLess(m_reperfused.get_nihss(), m_control.get_nihss())

    def test_core_does_not_recover(self):
        """Core infarct zone never recovers"""
        m = StrokeModel()
        ev = m.induce("MCA", 0.9)
        m.reperfuse(0)
        initial_core_gamma = {
            ch: m.channel_gamma[ch] for ch in ev.core_channels
        }
        for _ in range(500):
            m.tick()
        for ch in ev.core_channels:
            self.assertAlmostEqual(
                m.channel_gamma.get(ch, 0),
                initial_core_gamma[ch],
                places=2,
            )


# ====================================================================
# ALS
# ====================================================================

class TestALSProgression(unittest.TestCase):
    """ALS — Progressive motor neuron death"""

    def test_limb_onset(self):
        m = ALSModel()
        state = m.onset("limb")
        self.assertEqual(state.onset_type, "limb")
        self.assertIn("hand", state.active_channels)

    def test_bulbar_onset(self):
        m = ALSModel()
        state = m.onset("bulbar")
        self.assertIn("mouth", state.active_channels)

    def test_health_decreases_over_time(self):
        m = ALSModel()
        m.onset("limb")
        initial_hand = m.channel_health["hand"]
        for _ in range(100):
            m.tick()
        self.assertLess(m.channel_health["hand"], initial_hand)

    def test_spread_to_next_region(self):
        """When current region < threshold → spread to next region"""
        m = ALSModel()
        m.onset("limb")
        # Run enough ticks to bring hand below threshold
        for _ in range(1000):
            m.tick()
        self.assertGreater(len(m.state.active_channels), 1)

    def test_gamma_inverse_of_health(self):
        m = ALSModel()
        m.onset("limb")
        for _ in range(50):
            m.tick()
        gammas = m.get_channel_gamma()
        for ch, h in m.channel_health.items():
            self.assertAlmostEqual(gammas[ch], 1.0 - h, places=5)


class TestALSFunctionalRating(unittest.TestCase):
    """ALSFRS-R score (Cedarbaum 1999)"""

    def test_healthy_alsfrs_48(self):
        m = ALSModel()
        self.assertEqual(m.get_alsfrs_r(), 48)

    def test_alsfrs_decreases_with_disease(self):
        m = ALSModel()
        m.onset("limb")
        for _ in range(200):
            m.tick()
        self.assertLess(m.get_alsfrs_r(), 48)

    def test_alsfrs_range(self):
        m = ALSModel()
        m.onset("limb")
        for _ in range(5000):
            m.tick()
        score = m.get_alsfrs_r()
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 48)


class TestALSRiluzole(unittest.TestCase):
    """Riluzole treatment effect (Bensimon 1994)"""

    def test_riluzole_slows_progression(self):
        m_treated = ALSModel()
        m_control = ALSModel()
        m_treated.onset("limb", riluzole=True)
        m_control.onset("limb", riluzole=False)

        for _ in range(500):
            m_treated.tick()
            m_control.tick()

        self.assertGreater(
            m_treated.get_alsfrs_r(),
            m_control.get_alsfrs_r(),
        )

    def test_riluzole_rate_factor(self):
        m = ALSModel()
        m.onset("limb", riluzole=True)
        expected = 0.003 * ALS_RILUZOLE_FACTOR
        self.assertAlmostEqual(m.state.progression_rate, expected, places=6)


# ====================================================================
# Dementia
# ====================================================================

class TestDementiaProgression(unittest.TestCase):
    """Diffuse cognitive decline"""

    def test_onset_mild(self):
        m = DementiaModel()
        state = m.onset("mild")
        self.assertEqual(state.drift_rate, 0.0005)

    def test_gamma_increases_over_time(self):
        m = DementiaModel()
        m.onset("mild")
        for _ in range(500):
            m.tick()
        self.assertGreater(m.domain_gamma.get("hippocampus", 0), 0)

    def test_memory_declines_first(self):
        """Memory (hippocampus) declines first"""
        m = DementiaModel()
        m.onset("mild")
        for _ in range(300):
            m.tick()
        hippo = m.domain_gamma.get("hippocampus", 0)
        pfc = m.domain_gamma.get("prefrontal", 0)
        self.assertGreater(hippo, pfc)

    def test_severe_faster(self):
        m_mild = DementiaModel()
        m_severe = DementiaModel()
        m_mild.onset("mild")
        m_severe.onset("severe")
        for _ in range(300):
            m_mild.tick()
            m_severe.tick()
        self.assertLess(m_severe.get_mmse(), m_mild.get_mmse())


class TestDementiaMMSE(unittest.TestCase):
    """MMSE score (Folstein 1975)"""

    def test_healthy_mmse_30(self):
        m = DementiaModel()
        self.assertEqual(m.get_mmse(), MMSE_MAX)

    def test_mmse_decreases(self):
        m = DementiaModel()
        m.onset("moderate")
        for _ in range(800):
            m.tick()
        self.assertLess(m.get_mmse(), MMSE_MAX)

    def test_mmse_range(self):
        m = DementiaModel()
        m.onset("severe")
        for _ in range(5000):
            m.tick()
        self.assertGreaterEqual(m.get_mmse(), 0)
        self.assertLessEqual(m.get_mmse(), MMSE_MAX)


class TestDementiaCDR(unittest.TestCase):
    """CDR Clinical Dementia Rating"""

    def test_normal_cdr_0(self):
        m = DementiaModel()
        self.assertEqual(m.get_cdr(), 0.0)

    def test_cdr_increases_with_disease(self):
        m = DementiaModel()
        m.onset("severe")
        for _ in range(2000):
            m.tick()
        self.assertGreater(m.get_cdr(), 0)


# ====================================================================
# Alzheimer's
# ====================================================================

class TestAlzheimersAmyloid(unittest.TestCase):
    """Amyloid cascade hypothesis (Hardy & Higgins 1992)"""

    def test_onset_starts_hippocampus(self):
        m = AlzheimersModel()
        m.onset()
        self.assertIn("hippocampus", m.amyloid_load)

    def test_amyloid_accumulates(self):
        m = AlzheimersModel()
        m.onset()
        initial = m.amyloid_load["hippocampus"]
        for _ in range(100):
            m.tick()
        self.assertGreater(m.amyloid_load["hippocampus"], initial)

    def test_tau_spreads_from_hippocampus(self):
        """Tau spreads prion-like from hippocampus to other regions"""
        m = AlzheimersModel()
        m.onset()
        for _ in range(500):
            m.tick()
        # Adjacent region (amygdala) should have Tau
        self.assertGreater(m.tau_load.get("amygdala", 0), 0)

    def test_genetic_risk_accelerates(self):
        """APOE ε4 carriers progress faster"""
        m_normal = AlzheimersModel()
        m_apoe4 = AlzheimersModel()
        m_normal.onset(genetic_risk=1.0)
        m_apoe4.onset(genetic_risk=2.0)
        for _ in range(300):
            m_normal.tick()
            m_apoe4.tick()
        self.assertGreater(
            m_apoe4.amyloid_load["hippocampus"],
            m_normal.amyloid_load["hippocampus"],
        )


class TestAlzheimersBraak(unittest.TestCase):
    """Braak staging (Braak & Braak 1991)"""

    def test_initial_stage_0(self):
        m = AlzheimersModel()
        self.assertEqual(m.get_braak_stage(), 0)

    def test_early_stage_low(self):
        m = AlzheimersModel()
        m.onset()
        for _ in range(100):
            m.tick()
        self.assertIn(m.get_braak_stage(), [0, 1, 2])

    def test_late_stage_high(self):
        m = AlzheimersModel()
        m.onset(genetic_risk=2.0)
        for _ in range(2000):
            m.tick()
        self.assertGreaterEqual(m.get_braak_stage(), 3)

    def test_mmse_declines_with_braak(self):
        m = AlzheimersModel()
        m.onset()
        scores = []
        for _ in range(500):
            m.tick()
            if m._tick % 100 == 0:
                scores.append(m.get_mmse())
        # MMSE should show declining trend
        if len(scores) >= 2:
            self.assertLessEqual(scores[-1], scores[0])


# ====================================================================
# Cerebral Palsy
# ====================================================================

class TestCPSpasticity(unittest.TestCase):
    """Spastic CP — Lance (1980) velocity-dependent"""

    def test_spastic_velocity_dependent(self):
        """Higher velocity → higher Γ"""
        m = CerebralPalsyModel()
        m.set_condition("spastic", gmfcs_level=3)

        r_slow = m.tick(motor_velocity=0.1)
        gamma_slow = r_slow["channel_gamma"]["hand"]

        m2 = CerebralPalsyModel()
        m2.set_condition("spastic", gmfcs_level=3)
        r_fast = m2.tick(motor_velocity=0.8)
        gamma_fast = r_fast["channel_gamma"]["hand"]

        self.assertGreater(gamma_fast, gamma_slow)

    def test_baseline_at_rest(self):
        """Γ = baseline when at rest"""
        m = CerebralPalsyModel()
        m.set_condition("spastic", gmfcs_level=3)
        r = m.tick(motor_velocity=0.0)
        expected = GMFCS_BASELINE_GAMMA[3]
        self.assertAlmostEqual(
            r["channel_gamma"]["hand"], expected, places=2
        )


class TestCPTypes(unittest.TestCase):
    """Three types of CP"""

    def test_dyskinetic_has_variability(self):
        """Dyskinetic type — Γ has random fluctuation"""
        m = CerebralPalsyModel()
        m.set_condition("dyskinetic", gmfcs_level=3)
        gammas = []
        for _ in range(20):
            r = m.tick()
            gammas.append(r["channel_gamma"]["hand"])
        # Should have variability
        self.assertGreater(max(gammas) - min(gammas), 0.01)

    def test_ataxic_precision_dependent(self):
        """Cerebellar type — higher precision demand → higher Γ"""
        m = CerebralPalsyModel()
        m.set_condition("ataxic", gmfcs_level=3)
        r_lo = m.tick(precision_demand=0.1)

        m2 = CerebralPalsyModel()
        m2.set_condition("ataxic", gmfcs_level=3)
        r_hi = m2.tick(precision_demand=0.9)

        self.assertGreater(
            r_hi["channel_gamma"]["hand"],
            r_lo["channel_gamma"]["hand"],
        )

    def test_gmfcs_levels(self):
        """GMFCS 1-5 correspond to different baseline Γ"""
        for level in range(1, 6):
            m = CerebralPalsyModel()
            m.set_condition("spastic", gmfcs_level=level)
            self.assertEqual(m.get_gmfcs(), level)
            r = m.tick(motor_velocity=0.0)
            expected = GMFCS_BASELINE_GAMMA[level]
            self.assertAlmostEqual(
                r["channel_gamma"]["hand"], expected, places=2
            )


class TestCPCognitive(unittest.TestCase):
    """CP cognitive channel impact is milder"""

    def test_cognitive_lower_than_motor(self):
        m = CerebralPalsyModel()
        m.set_condition("spastic", gmfcs_level=4)
        r = m.tick(motor_velocity=0.0)
        motor_g = r["channel_gamma"]["hand"]
        cog_g = r["channel_gamma"].get("attention", 0)
        self.assertGreater(motor_g, cog_g)


# ====================================================================
# Unified Engine
# ====================================================================

class TestClinicalNeurologyEngine(unittest.TestCase):
    """ClinicalNeurologyEngine unified interface"""

    def test_no_conditions_clean(self):
        engine = ClinicalNeurologyEngine()
        result = engine.tick()
        self.assertEqual(result["active_conditions"], [])

    def test_single_stroke(self):
        engine = ClinicalNeurologyEngine()
        engine.stroke.induce("MCA", 0.8)
        result = engine.tick()
        self.assertIn("stroke", result["active_conditions"])
        self.assertGreater(len(result["merged_channel_gamma"]), 0)

    def test_multiple_conditions(self):
        """Comorbidity: multiple diseases coexist"""
        engine = ClinicalNeurologyEngine()
        engine.stroke.induce("MCA", 0.5)
        engine.als.onset("limb")
        engine.cerebral_palsy.set_condition("spastic", 3)
        result = engine.tick()
        self.assertEqual(len(result["active_conditions"]), 3)

    def test_merged_gamma_takes_max(self):
        """Merged Γ takes the maximum value"""
        engine = ClinicalNeurologyEngine()
        engine.stroke.induce("MCA", 0.9)
        engine.cerebral_palsy.set_condition("spastic", 5)
        result = engine.tick()
        merged = result["merged_channel_gamma"]
        # hand is affected by both, take the larger value
        stroke_g = engine.stroke.channel_gamma.get("hand", 0)
        cp_g = engine.cerebral_palsy.current_gamma.get("hand", 0)
        expected = max(stroke_g, cp_g)
        self.assertAlmostEqual(merged.get("hand", 0), expected, places=3)

    def test_clinical_summary(self):
        engine = ClinicalNeurologyEngine()
        engine.stroke.induce("MCA", 0.8)
        engine.als.onset("bulbar")
        engine.tick()
        summary = engine.get_clinical_summary()
        self.assertIn("nihss", summary)
        self.assertIn("alsfrs_r", summary)

    def test_introspect(self):
        engine = ClinicalNeurologyEngine()
        engine.alzheimers.onset()
        engine.tick()
        intro = engine.introspect()
        self.assertIn("active_conditions", intro)
        self.assertIn("alzheimers", intro)
        self.assertTrue(intro["alzheimers"]["active"])


# ====================================================================
# AliceBrain Integration
# ====================================================================

class TestAliceBrainIntegration(unittest.TestCase):
    """ClinicalNeurologyEngine correctly integrated in AliceBrain"""

    def test_brain_has_clinical_neurology(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        self.assertTrue(hasattr(brain, "clinical_neurology"))

    def test_perceive_includes_clinical(self):
        from alice.alice_brain import AliceBrain
        import numpy as np
        brain = AliceBrain()
        result = brain.perceive(np.random.randn(64))
        self.assertIn("clinical_neurology", result)

    def test_introspect_includes_clinical(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        intro = brain.introspect()
        self.assertIn("clinical_neurology",
                       intro.get("subsystems", {}))


if __name__ == "__main__":
    unittest.main()

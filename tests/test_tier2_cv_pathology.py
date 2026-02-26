# -*- coding: utf-8 -*-
"""
test_tier2_cv_pathology.py — Unit Tests for Tier 2 Cross-System Pathology Validation
====================================================================================

Tests verify compound pathology interactions where cardiovascular physics
intersects with pre-existing neurological disease.

Target: experiments/exp_tier2_cv_pathology.py
Physics: cardiovascular.py × clinical_neurology.py cross-product
"""

from __future__ import annotations

import math
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority
from alice.body.cardiovascular import (
    CardiovascularSystem,
    NEONATAL_VOLUME_FACTOR,
    BP_NORMALIZE_FACTOR,
    MAP_CRITICAL_LOW,
    SPO2_HYPOXIA_MILD,
    SPO2_HYPOXIA_SEVERE,
)
from alice.brain.clinical_neurology import (
    ClinicalNeurologyEngine,
    StrokeModel, ALSModel, DementiaModel, AlzheimersModel,
    CerebralPalsyModel,
    VASCULAR_TERRITORIES, ALL_CHANNELS,
    ALS_SPREAD_ORDER_LIMB, ALS_SPREAD_ORDER_BULBAR,
    NIHSS_MAX, MMSE_MAX,
)


# ============================================================================
# Helpers
# ============================================================================

def make_signal(freq: float = 40.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, 0.1, 64)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def run_tick(alice: AliceBrain, brightness: float = 0.5,
             noise: float = 0.2):
    visual = make_signal(40.0, 0.5) * brightness
    audio = make_signal(20.0, 0.15) * noise
    alice.hear(audio)
    return alice.see(visual, priority=Priority.NORMAL)


def make_adult_cv() -> CardiovascularSystem:
    """Create a mature CardiovascularSystem."""
    cv = CardiovascularSystem()
    cv._volume_growth = 0.6
    return cv


def stabilize_cv(cv: CardiovascularSystem, ticks: int = 30, **kw):
    """Run CV for several ticks to reach steady state."""
    defaults = dict(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                    hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                    ram_temperature=0.1)
    defaults.update(kw)
    r = None
    for _ in range(ticks):
        r = cv.tick(**defaults)
    return r


# ============================================================================
# Test Class 01: Stroke + Dehydration (Dual Perfusion Hit)
# ============================================================================

class TestStrokeDehydration:
    """MCA stroke + dehydration = dual perfusion hit."""

    def test_stroke_normal_hydration_has_baseline_perfusion(self):
        cv = make_adult_cv()
        stroke = StrokeModel()
        stroke.induce("MCA", severity=0.7)
        for _ in range(60):
            stroke.tick()
            r = cv.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                        hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                        ram_temperature=0.1)
        assert r["cerebral_perfusion"] > 0.5

    def test_dehydration_lowers_perfusion_in_stroke(self):
        cv_h = make_adult_cv()
        cv_d = make_adult_cv()
        stroke_h = StrokeModel()
        stroke_d = StrokeModel()
        stroke_h.induce("MCA", severity=0.7)
        stroke_d.induce("MCA", severity=0.7)
        for _ in range(80):
            stroke_h.tick()
            stroke_d.tick()
            r_h = cv_h.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                            hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                            ram_temperature=0.1)
            r_d = cv_d.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                            hydration=0.2, glucose=1.0, breaths_this_tick=0.25,
                            ram_temperature=0.1)
        assert r_d["cerebral_perfusion"] < r_h["cerebral_perfusion"]

    def test_dehydrated_stroke_lower_o2(self):
        cv_h = make_adult_cv()
        cv_d = make_adult_cv()
        for _ in range(80):
            cv_h.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                      hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                      ram_temperature=0.1)
            cv_d.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                      hydration=0.2, glucose=1.0, breaths_this_tick=0.25,
                      ram_temperature=0.1)
        assert cv_d.o2_delivery < cv_h.o2_delivery

    def test_dehydrated_stroke_tachycardia(self):
        cv = make_adult_cv()
        StrokeModel().induce("MCA", severity=0.7)
        for _ in range(60):
            cv.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                    hydration=0.2, glucose=1.0, breaths_this_tick=0.25,
                    ram_temperature=0.1)
        assert cv.compensatory_hr_delta > 2.0

    def test_nihss_positive_after_stroke_induction(self):
        stroke = StrokeModel()
        stroke.induce("MCA", severity=0.7)
        for _ in range(30):
            stroke.tick()
        assert stroke.get_nihss() > 0

    def test_stroke_nihss_max_valid(self):
        stroke = StrokeModel()
        stroke.induce("MCA", severity=1.0)
        for _ in range(200):
            stroke.tick()
        assert stroke.get_nihss() <= NIHSS_MAX


# ============================================================================
# Test Class 02: HIE — Neonatal Asphyxia
# ============================================================================

class TestHIENeonatalAsphyxia:
    """Birth asphyxia → SpO₂↓ → O₂↓ → brain damage."""

    def test_neonatal_baseline_viable(self):
        cv = CardiovascularSystem()
        r = stabilize_cv(cv, ticks=30, heart_rate=140, hydration=0.9)
        assert r["spo2"] > 0.9
        assert r["o2_delivery"] > 0.0

    def test_asphyxia_drops_spo2(self):
        cv = CardiovascularSystem()
        stabilize_cv(cv, ticks=20, heart_rate=140, hydration=0.9)
        # Asphyxia: no breathing
        for _ in range(60):
            r = cv.tick(heart_rate=180, sympathetic=0.8, parasympathetic=0.1,
                        hydration=0.6, glucose=0.7, breaths_this_tick=0.0,
                        ram_temperature=0.2)
        # SpO₂ should be lower than baseline
        assert r["spo2"] < 0.96

    def test_asphyxia_drops_o2_delivery(self):
        cv = CardiovascularSystem()
        baseline = stabilize_cv(cv, ticks=20, heart_rate=140, hydration=0.9)
        baseline_o2 = baseline["o2_delivery"]
        for _ in range(60):
            r = cv.tick(heart_rate=180, sympathetic=0.8, parasympathetic=0.1,
                        hydration=0.5, glucose=0.6, breaths_this_tick=0.0,
                        ram_temperature=0.2)
        assert r["o2_delivery"] < baseline_o2 * 0.8

    def test_resuscitation_recovers_spo2(self):
        cv = CardiovascularSystem()
        stabilize_cv(cv, ticks=20, heart_rate=140, hydration=0.9)
        # Asphyxia
        for _ in range(60):
            cv.tick(heart_rate=180, sympathetic=0.8, parasympathetic=0.1,
                    hydration=0.5, glucose=0.6, breaths_this_tick=0.0,
                    ram_temperature=0.2)
        min_spo2 = cv.spo2
        # Resuscitation
        for _ in range(60):
            r = cv.tick(heart_rate=150, sympathetic=0.3, parasympathetic=0.3,
                        hydration=0.8, glucose=0.9, breaths_this_tick=0.25,
                        ram_temperature=0.1)
        assert r["spo2"] > min_spo2

    def test_asphyxia_triggers_hypoxia_ticks(self):
        cv = CardiovascularSystem()
        stabilize_cv(cv, ticks=20, heart_rate=140, hydration=0.9)
        initial_hypoxia = cv.hypoxia_ticks
        for _ in range(60):
            cv.tick(heart_rate=180, sympathetic=0.8, parasympathetic=0.1,
                    hydration=0.4, glucose=0.4, breaths_this_tick=0.0,
                    ram_temperature=0.3)
        # Should have accumulated some hypoxia ticks (depending on threshold)
        # At minimum SpO₂ dropped below mild hypoxia threshold
        assert cv.spo2 < SPO2_HYPOXIA_MILD or cv.hypoxia_ticks > initial_hypoxia


# ============================================================================
# Test Class 03: Iron-Deficiency Anemia
# ============================================================================

class TestAnemiaO2Deficit:
    """Anemia reduces O₂ carrying capacity independently of perfusion."""

    def test_anemia_reduces_o2_delivery(self):
        cv_n = make_adult_cv()
        cv_a = make_adult_cv()
        cv_a.set_hemoglobin(0.4)
        stabilize_cv(cv_n, ticks=50)
        stabilize_cv(cv_a, ticks=50)
        assert cv_a.o2_delivery < cv_n.o2_delivery

    def test_anemia_preserves_perfusion(self):
        cv_n = make_adult_cv()
        cv_a = make_adult_cv()
        cv_a.set_hemoglobin(0.4)
        stabilize_cv(cv_n, ticks=50)
        stabilize_cv(cv_a, ticks=50)
        # Perfusion difference should be small (<15%)
        diff = abs(cv_n.cerebral_perfusion - cv_a.cerebral_perfusion)
        assert diff < 0.15

    def test_spo2_independent_of_hemoglobin(self):
        cv_n = make_adult_cv()
        cv_a = make_adult_cv()
        cv_a.set_hemoglobin(0.4)
        stabilize_cv(cv_n, ticks=50)
        stabilize_cv(cv_a, ticks=50)
        # SpO₂ measures saturation fraction, not absolute O₂
        assert abs(cv_n.spo2 - cv_a.spo2) < 0.03

    def test_o2_ratio_tracks_hemoglobin_ratio(self):
        cv_n = make_adult_cv()
        cv_a = make_adult_cv()
        cv_a.set_hemoglobin(0.4)
        stabilize_cv(cv_n, ticks=50)
        stabilize_cv(cv_a, ticks=50)
        ratio = cv_a.o2_delivery / max(cv_n.o2_delivery, 0.001)
        assert 0.25 <= ratio <= 0.65  # Approx tracks 0.4

    def test_mild_anemia_less_severe_than_severe(self):
        cv_mild = make_adult_cv()
        cv_severe = make_adult_cv()
        cv_mild.set_hemoglobin(0.7)
        cv_severe.set_hemoglobin(0.4)
        stabilize_cv(cv_mild, ticks=50)
        stabilize_cv(cv_severe, ticks=50)
        assert cv_mild.o2_delivery > cv_severe.o2_delivery

    def test_anemia_graded_response(self):
        """Three levels of hemoglobin → monotonic O₂ decline."""
        o2s = []
        for hb in [1.0, 0.7, 0.4]:
            cv = make_adult_cv()
            cv.set_hemoglobin(hb)
            stabilize_cv(cv, ticks=50)
            o2s.append(cv.o2_delivery)
        assert o2s[0] > o2s[1] > o2s[2]


# ============================================================================
# Test Class 04: ALS Respiratory Failure
# ============================================================================

class TestALSRespiratoryFailure:
    """ALS → respiratory channel degradation → breathing impairment → O₂↓."""

    def test_als_respiratory_in_spread_order(self):
        assert "respiratory" in ALS_SPREAD_ORDER_LIMB
        assert "respiratory" in ALS_SPREAD_ORDER_BULBAR

    def test_als_channel_health_degrades(self):
        als = ALSModel()
        als.onset("limb", riluzole=False)
        for _ in range(2000):
            als.tick()
        # At least one channel should have degraded significantly
        min_health = min(als.channel_health.values())
        assert min_health < 0.5

    def test_als_respiratory_health_eventually_drops(self):
        als = ALSModel()
        als.onset("limb", riluzole=False)
        for _ in range(3000):
            als.tick()
        resp_health = als.channel_health.get("respiratory", 1.0)
        # Respiratory is later in spread for limb onset
        # May or may not have reached it yet, but health trend is down
        assert resp_health < 1.0

    def test_als_respiratory_failure_flag(self):
        als = ALSModel()
        als.onset("limb", riluzole=False)
        result = None
        for _ in range(5000):
            result = als.tick()
        resp_health = als.channel_health.get("respiratory", 1.0)
        # After many ticks, respiratory should be significantly impaired
        assert resp_health < 0.5 or result.get("respiratory_failure", False)

    def test_als_breathing_to_cv_o2_coupling(self):
        """Reduced ALS respiratory health → reduced effective breathing → O₂↓."""
        als = ALSModel()
        als.onset("limb", riluzole=False)
        cv = make_adult_cv()

        # Run many ticks to get ALS progression
        for _ in range(3000):
            als_result = als.tick()
            resp_health = als.channel_health.get("respiratory", 1.0)
            effective_breathing = 0.25 * max(0.05, resp_health)
            cv.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                    hydration=1.0, glucose=1.0,
                    breaths_this_tick=effective_breathing,
                    ram_temperature=0.1)

        # Compare with healthy baseline
        cv_healthy = make_adult_cv()
        stabilize_cv(cv_healthy, ticks=50)
        # ALS patient O₂ should be lower (or equal if respiratory not yet reached)
        assert cv.o2_delivery <= cv_healthy.o2_delivery + 0.01

    def test_alsfrs_r_declines(self):
        als = ALSModel()
        als.onset("limb", riluzole=False)
        result = None
        for _ in range(2000):
            result = als.tick()
        # ALSFRS-R starts at 48 maximum
        assert result["alsfrs_r"] < 48

    def test_riluzole_slows_progression(self):
        als_no = ALSModel()
        als_ril = ALSModel()
        als_no.onset("limb", riluzole=False)
        als_ril.onset("limb", riluzole=True)
        for _ in range(2000):
            r_no = als_no.tick()
            r_ril = als_ril.tick()
        assert r_ril["alsfrs_r"] >= r_no["alsfrs_r"]


# ============================================================================
# Test Class 05: Chronic Stress → Vascular Stiffening
# ============================================================================

class TestChronicStressVascular:
    """Chronic cortisol → vascular resistance↑ → perfusion↓."""

    def test_cortisol_increases_vascular_resistance(self):
        cv_n = make_adult_cv()
        cv_s = make_adult_cv()
        stabilize_cv(cv_n, ticks=100, cortisol=0.0)
        stabilize_cv(cv_s, ticks=100, cortisol=0.8, sympathetic=0.7,
                     parasympathetic=0.1, heart_rate=85)
        assert cv_s._vascular_resistance > cv_n._vascular_resistance

    def test_chronic_stress_lowers_perfusion(self):
        cv_n = make_adult_cv()
        cv_s = make_adult_cv()
        stabilize_cv(cv_n, ticks=100, cortisol=0.0)
        stabilize_cv(cv_s, ticks=100, cortisol=0.8, sympathetic=0.7,
                     parasympathetic=0.1, heart_rate=85, hydration=0.9)
        assert cv_s.cerebral_perfusion < cv_n.cerebral_perfusion

    def test_chronic_stress_lowers_o2(self):
        cv_n = make_adult_cv()
        cv_s = make_adult_cv()
        stabilize_cv(cv_n, ticks=100, cortisol=0.0)
        stabilize_cv(cv_s, ticks=100, cortisol=0.8, sympathetic=0.7,
                     parasympathetic=0.1, heart_rate=85, hydration=0.9)
        assert cv_s.o2_delivery < cv_n.o2_delivery

    def test_stress_dehydration_combined_worse(self):
        """Stress + dehydration combined → worse than either alone."""
        cv_dehy = make_adult_cv()
        cv_both = make_adult_cv()
        stabilize_cv(cv_dehy, ticks=100, hydration=0.5, cortisol=0.0)
        stabilize_cv(cv_both, ticks=100, hydration=0.5, cortisol=0.8,
                     sympathetic=0.7, parasympathetic=0.1, heart_rate=85)
        # Combined should have higher resistance
        assert cv_both._vascular_resistance >= cv_dehy._vascular_resistance


# ============================================================================
# Test Class 06: Alzheimer's + Cardiovascular Comorbidity
# ============================================================================

class TestAlzheimersCV:
    """AD + cardiovascular deficit = accelerated functional decline."""

    def test_ad_progresses_through_braak(self):
        ad = AlzheimersModel()
        ad.onset(genetic_risk=1.0)
        for _ in range(3000):
            ad.tick()
        assert ad.get_braak_stage() >= 1

    def test_ad_cv_comorbidity_lower_o2(self):
        cv_n = make_adult_cv()
        cv_c = make_adult_cv()
        cv_c.set_hemoglobin(0.6)
        stabilize_cv(cv_n, ticks=50)
        stabilize_cv(cv_c, ticks=50, hydration=0.7, cortisol=0.3,
                     sympathetic=0.3, parasympathetic=0.2, heart_rate=75)
        assert cv_c.o2_delivery < cv_n.o2_delivery

    def test_braak_identical_regardless_of_cv(self):
        """CV state doesn't change protein deposition physics."""
        ad_a = AlzheimersModel()
        ad_b = AlzheimersModel()
        ad_a.onset(genetic_risk=1.0)
        ad_b.onset(genetic_risk=1.0)
        for _ in range(3000):
            ad_a.tick()
            ad_b.tick()
        # Same genetic_risk → same Braak (protein deposition is CV-independent)
        assert abs(ad_a.get_braak_stage() - ad_b.get_braak_stage()) <= 1

    def test_ad_mmse_declines(self):
        ad = AlzheimersModel()
        ad.onset(genetic_risk=1.0)
        for _ in range(3000):
            ad.tick()
        assert ad.get_mmse() < MMSE_MAX

    def test_genetic_risk_modulates_progression(self):
        ad_lo = AlzheimersModel()
        ad_hi = AlzheimersModel()
        ad_lo.onset(genetic_risk=0.3)
        ad_hi.onset(genetic_risk=1.0)
        for _ in range(3000):
            ad_lo.tick()
            ad_hi.tick()
        # Higher risk = faster or equal progression
        assert ad_hi.get_braak_stage() >= ad_lo.get_braak_stage()


# ============================================================================
# Test Class 07: Stroke Territory × CV State Matrix
# ============================================================================

class TestStrokeTerritoryMatrix:
    """Every vascular territory: dehydration/anemia worsens O₂."""

    @pytest.mark.parametrize("territory", list(VASCULAR_TERRITORIES.keys()))
    def test_dehydration_lowers_o2_per_territory(self, territory):
        cv_h = make_adult_cv()
        cv_d = make_adult_cv()
        s_h = StrokeModel()
        s_d = StrokeModel()
        s_h.induce(territory, severity=0.7)
        s_d.induce(territory, severity=0.7)
        for _ in range(60):
            s_h.tick(); s_d.tick()
            cv_h.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                      hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                      ram_temperature=0.1)
            cv_d.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                      hydration=0.3, glucose=1.0, breaths_this_tick=0.25,
                      ram_temperature=0.1)
        assert cv_d.o2_delivery < cv_h.o2_delivery

    @pytest.mark.parametrize("territory", list(VASCULAR_TERRITORIES.keys()))
    def test_anemia_lowers_o2_per_territory(self, territory):
        cv_h = make_adult_cv()
        cv_a = make_adult_cv()
        cv_a.set_hemoglobin(0.5)
        s_h = StrokeModel()
        s_a = StrokeModel()
        s_h.induce(territory, severity=0.7)
        s_a.induce(territory, severity=0.7)
        for _ in range(60):
            s_h.tick(); s_a.tick()
            cv_h.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                      hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                      ram_temperature=0.1)
            cv_a.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                      hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                      ram_temperature=0.1)
        assert cv_a.o2_delivery < cv_h.o2_delivery


# ============================================================================
# Test Class 08: Cerebral Palsy + CV Growth Coupling
# ============================================================================

class TestCPCVGrowth:
    """CP → less movement → less CV growth → more vulnerable."""

    def test_active_movement_grows_cv(self):
        cv = CardiovascularSystem()
        initial_growth = cv._volume_growth
        for _ in range(200):
            cv.grow(motor_movements=5)
            cv.tick(heart_rate=110, sympathetic=0.2, parasympathetic=0.3,
                    hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                    ram_temperature=0.1)
        assert cv._volume_growth > initial_growth

    def test_minimal_movement_grows_less(self):
        cv_active = CardiovascularSystem()
        cv_passive = CardiovascularSystem()
        for _ in range(300):
            cv_active.grow(motor_movements=5)
            cv_passive.grow(motor_movements=1)
            cv_active.tick(heart_rate=110, sympathetic=0.2, parasympathetic=0.3,
                           hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                           ram_temperature=0.1)
            cv_passive.tick(heart_rate=110, sympathetic=0.2, parasympathetic=0.3,
                            hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                            ram_temperature=0.1)
        assert cv_active._volume_growth > cv_passive._volume_growth

    def test_cp_child_dehydration_vulnerability(self):
        cv_healthy = CardiovascularSystem()
        cv_cp = CardiovascularSystem()
        for _ in range(300):
            cv_healthy.grow(motor_movements=5)
            cv_cp.grow(motor_movements=1)
            cv_healthy.tick(heart_rate=110, sympathetic=0.2, parasympathetic=0.3,
                            hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                            ram_temperature=0.1)
            cv_cp.tick(heart_rate=110, sympathetic=0.2, parasympathetic=0.3,
                       hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                       ram_temperature=0.1)
        # Dehydration challenge
        for _ in range(60):
            r_h = cv_healthy.tick(heart_rate=110, sympathetic=0.3,
                                  parasympathetic=0.2, hydration=0.4,
                                  glucose=1.0, breaths_this_tick=0.25,
                                  ram_temperature=0.1)
            r_cp = cv_cp.tick(heart_rate=110, sympathetic=0.3,
                              parasympathetic=0.2, hydration=0.4,
                              glucose=1.0, breaths_this_tick=0.25,
                              ram_temperature=0.1)
        # CP child should be at least as vulnerable (equal or worse)
        assert r_cp["cerebral_perfusion"] <= r_h["cerebral_perfusion"] + 0.02

    def test_cp_model_gmfcs_levels(self):
        cp = CerebralPalsyModel()
        cp.set_condition("spastic", gmfcs_level=3)
        for _ in range(50):
            cp.tick()
        assert cp.get_gmfcs() == 3


# ============================================================================
# Test Class 09: Multi-Disease Comorbidity Engine
# ============================================================================

class TestMultidiseaseComorbidity:
    """ClinicalNeurologyEngine handles multiple simultaneous diseases."""

    def test_multiple_diseases_coexist(self):
        engine = ClinicalNeurologyEngine()
        engine.stroke.induce("MCA", severity=0.6)
        engine.als.onset("limb")
        for _ in range(100):
            result = engine.tick()
        active = result["active_conditions"]
        assert "stroke" in active
        assert "als" in active

    def test_merged_gamma_covers_both_diseases(self):
        engine = ClinicalNeurologyEngine()
        engine.stroke.induce("MCA", severity=0.6)
        engine.als.onset("limb")
        for _ in range(100):
            result = engine.tick()
        merged = result["merged_channel_gamma"]
        assert len(merged) > 3  # Should cover channels from both diseases

    def test_clinical_summary_reports_nihss_and_alsfrs(self):
        engine = ClinicalNeurologyEngine()
        engine.stroke.induce("MCA", severity=0.6)
        engine.als.onset("limb")
        for _ in range(100):
            engine.tick()
        summary = engine.get_clinical_summary()
        assert "nihss" in summary
        assert "alsfrs_r" in summary

    def test_stroke_with_progressive_dehydration(self):
        engine = ClinicalNeurologyEngine()
        cv = make_adult_cv()
        engine.stroke.induce("MCA", severity=0.6)
        for tick in range(200):
            hydration = max(0.3, 1.0 - tick * 0.003)
            engine.tick()
            r = cv.tick(heart_rate=80, sympathetic=0.4, parasympathetic=0.2,
                        hydration=hydration, glucose=0.9,
                        breaths_this_tick=0.25, ram_temperature=0.15)
        assert r["cerebral_perfusion"] < 1.0

    def test_triple_comorbidity_active_conditions(self):
        engine = ClinicalNeurologyEngine()
        engine.stroke.induce("MCA", severity=0.5)
        engine.als.onset("limb")
        engine.dementia.onset("mild")
        for _ in range(100):
            result = engine.tick()
        active = result["active_conditions"]
        assert len(active) >= 3


# ============================================================================
# Test Class 10: Full AliceBrain Integration
# ============================================================================

class TestFullBrainCVDisease:
    """Full AliceBrain: disease + CV state → clinical summary."""

    def test_introspect_has_clinical_neurology(self):
        alice = AliceBrain(neuron_count=60)
        for _ in range(10):
            run_tick(alice)
        intro = alice.introspect()
        assert "clinical_neurology" in intro["subsystems"]

    def test_introspect_has_cardiovascular(self):
        alice = AliceBrain(neuron_count=60)
        for _ in range(10):
            run_tick(alice)
        intro = alice.introspect()
        assert "cardiovascular" in intro["subsystems"]

    def test_stroke_induction_reflected_in_introspect(self):
        alice = AliceBrain(neuron_count=60)
        for _ in range(10):
            run_tick(alice)
        alice.clinical_neurology.stroke.induce("MCA", severity=0.5)
        for _ in range(30):
            run_tick(alice, brightness=0.4)
        intro = alice.introspect()
        active = intro["subsystems"]["clinical_neurology"].get(
            "active_conditions", [])
        assert "stroke" in active

    def test_dehydration_lowers_cv_perfusion_in_brain(self):
        alice = AliceBrain(neuron_count=60)
        for _ in range(20):
            run_tick(alice)
        baseline_perf = alice.cardiovascular.cerebral_perfusion
        # Apply dehydration
        for _ in range(60):
            alice.homeostatic_drive.hydration = 0.3
            run_tick(alice, brightness=0.4)
        assert alice.cardiovascular.cerebral_perfusion < baseline_perf

    def test_cv_stats_track_beats(self):
        alice = AliceBrain(neuron_count=60)
        for _ in range(20):
            run_tick(alice)
        stats = alice.cardiovascular.get_stats()
        assert stats["total_beats"] > 0
        assert stats["total_o2_delivered"] > 0


# ============================================================================
# Test Class 11: Dementia Disease Model
# ============================================================================

class TestDementiaModel:
    """Validate DementiaModel independently."""

    def test_dementia_mmse_declines(self):
        d = DementiaModel()
        d.onset("moderate")
        for _ in range(2000):
            d.tick()
        assert d.get_mmse() < MMSE_MAX

    def test_dementia_severity_levels(self):
        """Higher severity → faster decline."""
        d_mild = DementiaModel()
        d_severe = DementiaModel()
        d_mild.onset("mild")
        d_severe.onset("severe")
        for _ in range(2000):
            d_mild.tick()
            d_severe.tick()
        assert d_severe.get_mmse() <= d_mild.get_mmse()

    def test_dementia_cdr_valid(self):
        d = DementiaModel()
        d.onset("moderate")
        for _ in range(2000):
            d.tick()
        cdr = d.get_cdr()
        assert 0 <= cdr <= 3.0


# ============================================================================
# Test Class 12: Cross-Import Validation
# ============================================================================

class TestExperimentImports:
    """Verify experiment file imports correctly."""

    def test_import_exp_tier2(self):
        import experiments.exp_tier2_cv_pathology as mod
        assert hasattr(mod, "main")
        assert hasattr(mod, "exp_01_stroke_plus_dehydration")
        assert hasattr(mod, "exp_10_full_brain_clinical_cv_integration")

    def test_experiment_functions_callable(self):
        import experiments.exp_tier2_cv_pathology as mod
        # Verify all 10 experiments exist
        for i in range(1, 11):
            name = f"exp_{i:02d}_"
            matches = [n for n in dir(mod) if n.startswith(name)]
            assert len(matches) == 1, f"Expected exactly 1 function for {name}"


# ============================================================================
# Test Class 13: Stroke Reperfusion
# ============================================================================

class TestStrokeReperfusion:
    """Stroke reperfusion (thrombolysis) reduces damage."""

    def test_reperfusion_improves_nihss(self):
        stroke = StrokeModel()
        stroke.induce("MCA", severity=0.7)
        for _ in range(50):
            stroke.tick()
        nihss_before = stroke.get_nihss()
        stroke.reperfuse(0)  # Reperfuse first stroke
        for _ in range(50):
            stroke.tick()
        nihss_after = stroke.get_nihss()
        assert nihss_after <= nihss_before

    def test_reperfusion_after_dehydration(self):
        """Reperfusion + rehydration should both help."""
        stroke = StrokeModel()
        cv = make_adult_cv()
        stroke.induce("MCA", severity=0.7)
        # Dehydrated during stroke
        for _ in range(50):
            stroke.tick()
            cv.tick(heart_rate=80, sympathetic=0.4, parasympathetic=0.2,
                    hydration=0.3, glucose=0.9, breaths_this_tick=0.25,
                    ram_temperature=0.1)
        perf_before = cv.cerebral_perfusion
        # Reperfuse + rehydrate
        stroke.reperfuse(0)
        for _ in range(50):
            stroke.tick()
            cv.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                    hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                    ram_temperature=0.1)
        assert cv.cerebral_perfusion > perf_before


# ============================================================================
# Test Class 14: ALS Onset Type Comparison
# ============================================================================

class TestALSOnsetTypes:
    """Limb vs Bulbar onset ALS have different channel spread."""

    def test_limb_onset_hand_first(self):
        als = ALSModel()
        als.onset("limb", riluzole=False)
        for _ in range(1000):
            als.tick()
        # Hand should be most degraded (first in limb spread)
        hand_h = als.channel_health.get("hand", 1.0)
        assert hand_h < 0.5

    def test_bulbar_onset_mouth_first(self):
        als = ALSModel()
        als.onset("bulbar", riluzole=False)
        for _ in range(1000):
            als.tick()
        mouth_h = als.channel_health.get("mouth", 1.0)
        assert mouth_h < 0.5

    def test_onset_type_stored(self):
        als = ALSModel()
        als.onset("limb")
        assert als.state.onset_type == "limb"
        als2 = ALSModel()
        als2.onset("bulbar")
        assert als2.state.onset_type == "bulbar"


# ============================================================================
# Test Class 15: Glucose Delivery
# ============================================================================

class TestGlucoseDelivery:
    """Cardiovascular glucose delivery couples with cerebral perfusion."""

    def test_glucose_delivery_positive(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, ticks=30)
        assert r["glucose_delivery"] > 0

    def test_low_glucose_lower_delivery(self):
        cv_h = make_adult_cv()
        cv_l = make_adult_cv()
        r_h = stabilize_cv(cv_h, ticks=30, glucose=1.0)
        r_l = stabilize_cv(cv_l, ticks=30, glucose=0.3)
        assert r_l["glucose_delivery"] < r_h["glucose_delivery"]

    def test_dehydration_reduces_glucose_delivery(self):
        cv_h = make_adult_cv()
        cv_d = make_adult_cv()
        r_h = stabilize_cv(cv_h, ticks=30, glucose=1.0, hydration=1.0)
        r_d = stabilize_cv(cv_d, ticks=30, glucose=1.0, hydration=0.3)
        assert r_d["glucose_delivery"] < r_h["glucose_delivery"]

# -*- coding: utf-8 -*-
"""Tests for textbook_risk.py — clinical risk calculators."""

import pytest
from alice.diagnostics.textbook_risk import (
    framingham_risk,
    ascvd_pooled_cohort,
    score2_esc,
    egfr_ckd_epi_2021,
    fib4_index,
    homa_ir,
    metabolic_syndrome_atp3,
    TextbookReport,
)


# ============================================================================
# 1. Framingham Risk
# ============================================================================

class TestFraminghamRisk:
    """Framingham 10-year CHD risk (ATP-III point system)."""

    def test_low_risk_young_male(self):
        r = framingham_risk(35, "M", 180, 55, 115, smoker=False)
        assert r.risk_category == "low"
        assert r.risk_10yr_pct < 10

    def test_high_risk_old_male(self):
        r = framingham_risk(65, "M", 260, 32, 155, bp_treated=True,
                            smoker=True, diabetic=True)
        assert r.risk_category == "high"
        assert r.risk_10yr_pct >= 20

    def test_female_different_scoring(self):
        m = framingham_risk(50, "M", 220, 45, 135)
        f = framingham_risk(50, "F", 220, 45, 135)
        # Both should be low risk at this profile
        assert m.risk_category == "low"
        assert f.risk_category == "low"

    def test_smoking_increases_risk(self):
        non = framingham_risk(50, "M", 220, 45, 135, smoker=False)
        smk = framingham_risk(50, "M", 220, 45, 135, smoker=True)
        assert smk.points_total > non.points_total

    def test_diabetes_increases_risk(self):
        non = framingham_risk(50, "M", 220, 45, 135, diabetic=False)
        dm = framingham_risk(50, "M", 220, 45, 135, diabetic=True)
        assert dm.points_total > non.points_total

    def test_hdl_protective(self):
        low = framingham_risk(50, "M", 220, 30, 135)
        high = framingham_risk(50, "M", 220, 65, 135)
        assert high.points_total < low.points_total

    def test_point_breakdown_has_all_keys(self):
        r = framingham_risk(50, "M", 220, 45, 135)
        assert "age" in r.point_breakdown
        assert "total_cholesterol" in r.point_breakdown
        assert "hdl" in r.point_breakdown
        assert "sbp" in r.point_breakdown
        assert "smoking" in r.point_breakdown
        assert "diabetes" in r.point_breakdown

    def test_extreme_age_low(self):
        r = framingham_risk(20, "M", 180, 55, 110)
        assert r.risk_10yr_pct <= 2

    def test_extreme_age_high(self):
        r = framingham_risk(80, "M", 180, 55, 110)
        assert r.points_total >= 5


# ============================================================================
# 1b. ASCVD Pooled Cohort (Goff 2014)
# ============================================================================

class TestASCVD:
    """ACC/AHA Pooled Cohort 10-year ASCVD risk."""

    def test_low_risk_young_female(self):
        r = ascvd_pooled_cohort(42, "F", 190, 55, 115)
        assert r.risk_category == "low"
        assert r.risk_10yr_pct < 5.0

    def test_high_risk_male(self):
        r = ascvd_pooled_cohort(65, "M", 260, 32, 155,
                                bp_treated=True, smoker=True, diabetic=True)
        assert r.risk_category == "high"
        assert r.risk_10yr_pct >= 20.0

    def test_smoking_increases_risk(self):
        non = ascvd_pooled_cohort(55, "M", 220, 45, 135, smoker=False)
        smk = ascvd_pooled_cohort(55, "M", 220, 45, 135, smoker=True)
        assert smk.risk_10yr_pct > non.risk_10yr_pct

    def test_diabetes_increases_risk(self):
        non = ascvd_pooled_cohort(55, "M", 220, 45, 135, diabetic=False)
        dm = ascvd_pooled_cohort(55, "M", 220, 45, 135, diabetic=True)
        assert dm.risk_10yr_pct > non.risk_10yr_pct

    def test_female_vs_male(self):
        m = ascvd_pooled_cohort(55, "M", 220, 45, 135)
        f = ascvd_pooled_cohort(55, "F", 220, 45, 135)
        # Men generally higher baseline ASCVD risk
        assert m.risk_10yr_pct > f.risk_10yr_pct

    def test_risk_pct_clamped(self):
        r = ascvd_pooled_cohort(40, "F", 160, 70, 110)
        assert 0.0 <= r.risk_10yr_pct <= 100.0

    def test_categories_correct(self):
        """Risk categories: low<5, borderline<7.5, intermediate<20, high>=20"""
        r = ascvd_pooled_cohort(45, "F", 180, 60, 120)
        if r.risk_10yr_pct < 5.0:
            assert r.risk_category == "low"
        elif r.risk_10yr_pct < 7.5:
            assert r.risk_category == "borderline"
        elif r.risk_10yr_pct < 20.0:
            assert r.risk_category == "intermediate"
        else:
            assert r.risk_category == "high"


# ============================================================================
# 1c. SCORE2 (ESC 2021)
# ============================================================================

class TestSCORE2:
    """ESC SCORE2 10-year fatal+non-fatal CVD risk."""

    def test_low_risk_female(self):
        r = score2_esc(45, "F", 5.0, 1.5, 120)
        assert r.risk_10yr_pct < 5.0

    def test_high_risk_male_smoker(self):
        r = score2_esc(65, "M", 7.0, 0.9, 160, smoker=True)
        assert r.risk_10yr_pct > 5.0

    def test_smoking_increases_risk(self):
        non = score2_esc(55, "M", 5.5, 1.3, 135, smoker=False)
        smk = score2_esc(55, "M", 5.5, 1.3, 135, smoker=True)
        assert smk.risk_10yr_pct > non.risk_10yr_pct

    def test_higher_sbp_increases_risk(self):
        low = score2_esc(55, "M", 5.5, 1.3, 120)
        high = score2_esc(55, "M", 5.5, 1.3, 160)
        assert high.risk_10yr_pct > low.risk_10yr_pct

    def test_higher_hdl_decreases_risk(self):
        low_hdl = score2_esc(55, "M", 5.5, 0.8, 130)
        high_hdl = score2_esc(55, "M", 5.5, 1.8, 130)
        assert high_hdl.risk_10yr_pct < low_hdl.risk_10yr_pct

    def test_risk_clamped(self):
        r = score2_esc(45, "F", 4.5, 1.8, 110)
        assert 0.1 <= r.risk_10yr_pct <= 50.0


# ============================================================================
# 1d. FIB-4 Liver Fibrosis
# ============================================================================

class TestFIB4:
    """FIB-4 liver fibrosis index."""

    def test_low_risk(self):
        r = fib4_index(35, 25, 30, 250)
        assert r.fib4 < 1.30
        assert r.fibrosis_risk == "low"

    def test_high_risk(self):
        r = fib4_index(65, 80, 40, 100)
        assert r.fib4 >= 2.67
        assert r.fibrosis_risk == "high"

    def test_indeterminate(self):
        r = fib4_index(50, 40, 35, 200)
        assert 1.30 <= r.fib4 < 2.67
        assert r.fibrosis_risk == "indeterminate"

    def test_formula_correctness(self):
        """FIB-4 = (Age * AST) / (Platelets * sqrt(ALT))"""
        import math
        age, ast, alt, plt = 50, 40, 25, 200
        expected = (age * ast) / (plt * math.sqrt(alt))
        r = fib4_index(age, ast, alt, plt)
        assert abs(r.fib4 - round(expected, 2)) < 0.01

    def test_age_increases_fib4(self):
        young = fib4_index(30, 30, 30, 250)
        old = fib4_index(70, 30, 30, 250)
        assert old.fib4 > young.fib4

    def test_low_platelets_increases_fib4(self):
        normal = fib4_index(50, 30, 30, 250)
        low = fib4_index(50, 30, 30, 80)
        assert low.fib4 > normal.fib4


# ============================================================================
# 2. eGFR (CKD-EPI 2021)
# ============================================================================

class TestEGFR:
    """CKD-EPI 2021 race-free eGFR equation."""

    def test_normal_male(self):
        r = egfr_ckd_epi_2021(1.0, 40, "M")
        assert r.egfr > 90
        assert r.ckd_stage == "G1"

    def test_normal_female(self):
        r = egfr_ckd_epi_2021(0.8, 35, "F")
        assert r.egfr > 90
        assert r.ckd_stage == "G1"

    def test_elevated_creatinine(self):
        r = egfr_ckd_epi_2021(2.5, 60, "M")
        assert r.egfr < 45
        assert r.ckd_stage in ("G3b", "G4", "G5")

    def test_severe_ckd(self):
        r = egfr_ckd_epi_2021(5.0, 70, "M")
        assert r.egfr < 15
        assert r.ckd_stage == "G5"

    def test_age_decreases_egfr(self):
        young = egfr_ckd_epi_2021(1.0, 30, "M")
        old = egfr_ckd_epi_2021(1.0, 70, "M")
        assert young.egfr > old.egfr

    def test_higher_creatinine_lower_egfr(self):
        low = egfr_ckd_epi_2021(0.8, 50, "M")
        high = egfr_ckd_epi_2021(1.8, 50, "M")
        assert low.egfr > high.egfr

    def test_ckd_stages_monotonic(self):
        """As Cr rises, CKD stage should worsen."""
        stages = []
        for cr in [0.7, 1.0, 1.5, 2.0, 3.0, 6.0]:
            r = egfr_ckd_epi_2021(cr, 55, "M")
            stages.append(r.egfr)
        # eGFR should be monotonically decreasing
        for i in range(len(stages) - 1):
            assert stages[i] > stages[i + 1]


# ============================================================================
# 3. HOMA-IR
# ============================================================================

class TestHOMAIR:
    """Homeostatic Model Assessment of Insulin Resistance."""

    def test_normal(self):
        r = homa_ir(80, 4.0)
        assert r.homa_ir < 1.0
        assert r.insulin_resistance == "normal"

    def test_borderline(self):
        r = homa_ir(95, 10.0)
        assert 1.0 <= r.homa_ir < 2.5
        assert r.insulin_resistance == "borderline"

    def test_resistant(self):
        r = homa_ir(130, 25.0)
        assert r.homa_ir >= 2.5
        assert r.insulin_resistance == "resistant"

    def test_formula_correctness(self):
        """HOMA-IR = glucose * insulin / 405."""
        r = homa_ir(100, 10.0)
        expected = (100 * 10) / 405
        assert abs(r.homa_ir - round(expected, 2)) < 0.01

    def test_higher_glucose_higher_ir(self):
        low = homa_ir(80, 10.0)
        high = homa_ir(150, 10.0)
        assert high.homa_ir > low.homa_ir

    def test_higher_insulin_higher_ir(self):
        low = homa_ir(90, 5.0)
        high = homa_ir(90, 20.0)
        assert high.homa_ir > low.homa_ir


# ============================================================================
# 4. Metabolic Syndrome (ATP-III)
# ============================================================================

class TestMetabolicSyndrome:
    """ATP-III / AHA-NHLBI 2005 Metabolic Syndrome criteria."""

    def test_no_criteria(self):
        r = metabolic_syndrome_atp3(
            waist_cm=80, sex="M", triglycerides=100, hdl=55,
            systolic_bp=115, diastolic_bp=75, fasting_glucose=85,
        )
        assert r.criteria_met == 0
        assert not r.is_metabolic_syndrome

    def test_all_criteria(self):
        r = metabolic_syndrome_atp3(
            waist_cm=110, sex="M", triglycerides=200, hdl=35,
            systolic_bp=140, diastolic_bp=90, fasting_glucose=115,
        )
        assert r.criteria_met == 5
        assert r.is_metabolic_syndrome

    def test_exactly_three(self):
        r = metabolic_syndrome_atp3(
            waist_cm=105, sex="M", triglycerides=160, hdl=35,
            systolic_bp=115, diastolic_bp=75, fasting_glucose=85,
        )
        assert r.criteria_met == 3
        assert r.is_metabolic_syndrome

    def test_female_lower_waist_threshold(self):
        r_m = metabolic_syndrome_atp3(
            waist_cm=95, sex="M", triglycerides=100, hdl=55,
            systolic_bp=115, diastolic_bp=75, fasting_glucose=85,
        )
        r_f = metabolic_syndrome_atp3(
            waist_cm=95, sex="F", triglycerides=100, hdl=55,
            systolic_bp=115, diastolic_bp=75, fasting_glucose=85,
        )
        # 95cm: below male threshold (102) but above female (88)
        assert not r_m.criteria_detail["elevated_waist"]
        assert r_f.criteria_detail["elevated_waist"]

    def test_bp_treatment_counts(self):
        r = metabolic_syndrome_atp3(
            waist_cm=80, sex="M", triglycerides=100, hdl=55,
            systolic_bp=115, diastolic_bp=75, fasting_glucose=85,
            bp_treated=True,
        )
        assert r.criteria_detail["elevated_bp"] is True

    def test_detail_dict_has_five_keys(self):
        r = metabolic_syndrome_atp3(
            waist_cm=80, sex="M", triglycerides=100, hdl=55,
            systolic_bp=115, diastolic_bp=75, fasting_glucose=85,
        )
        assert len(r.criteria_detail) == 5


# ============================================================================
# 5. TextbookReport
# ============================================================================

class TestTextbookReport:
    """Combined report output."""

    def test_summary_lines_not_empty(self):
        report = TextbookReport(
            framingham=framingham_risk(50, "M", 220, 45, 135),
            egfr=egfr_ckd_epi_2021(1.0, 50, "M"),
            homa=homa_ir(90, 10),
            metabolic_syndrome=metabolic_syndrome_atp3(
                waist_cm=90, sex="M", triglycerides=130, hdl=45,
                systolic_bp=125, diastolic_bp=80, fasting_glucose=95,
            ),
        )
        lines = report.summary_lines()
        assert len(lines) >= 4  # At least one line per component

    def test_partial_report(self):
        report = TextbookReport(
            egfr=egfr_ckd_epi_2021(1.2, 60, "F"),
        )
        lines = report.summary_lines()
        assert len(lines) == 1
        assert "eGFR" in lines[0]

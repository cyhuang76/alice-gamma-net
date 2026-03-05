# -*- coding: utf-8 -*-
"""textbook_risk.py — Established Clinical Risk Calculators
=============================================================

Purpose
-------
Standard-of-care risk formulas that serve as the **textbook baseline** for
the Gamma-Net public verification platform.  Every formula here is published
in peer-reviewed guidelines and is computed with **fixed coefficients** —
exactly as a clinician would calculate at bedside.

Six formula families
--------------------
1. Cardiovascular:  Framingham 10-year CHD risk  (Wilson 1998, ATP-III)
2. Cardiovascular:  ASCVD Pooled Cohort 10-year risk  (Goff 2014, ACC/AHA)
3. Cardiovascular:  SCORE2 10-year CVD risk  (ESC 2021, low-risk region)
4. Renal:           eGFR via CKD-EPI 2021 (race-free, NEJM 2021)
5. Hepatic:         FIB-4 liver fibrosis index  (Sterling 2006)
6. Metabolic:       HOMA-IR  + Metabolic Syndrome (ATP-III/AHA 2005)

Design principle
----------------
These functions intentionally do NOT use any Gamma-Net concept.  They exist
so users can see *textbook side-by-side with Gamma-Net* and judge for
themselves.

References
----------
- Wilson PWF et al., Circulation 1998;97:1837-1847         (Framingham)
- Goff DC et al., Circulation 2014;129:S49-S73             (ASCVD PCE)
- SCORE2 working group, Eur Heart J 2021;42:2439-2454      (SCORE2)
- Inker LA et al., NEJM 2021;385:1737-1749                 (CKD-EPI 2021)
- Sterling RK et al., Hepatology 2006;43:1317-1325         (FIB-4)
- Matthews DR et al., Diabetologia 1985;28:412-419         (HOMA-IR)
- Grundy SM et al., Circulation 2005;112:2735-2752         (MetS ATP-III)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 1. CARDIOVASCULAR — Framingham 10-Year CHD Risk
# ============================================================================

# ATP-III Framingham point tables (Wilson 1998 / NCEP ATP-III adaptation)
# Separate tables for men and women

@dataclass(frozen=True)
class FraminghamResult:
    """Output of Framingham 10-year CHD risk calculation."""
    points_total: int
    risk_10yr_pct: float          # 10-year CHD risk (%)
    risk_category: str            # "low" / "moderate" / "high"
    point_breakdown: Dict[str, int]  # component → points


def _framingham_age_points_male(age: int) -> int:
    if age < 35:
        return -1
    elif age <= 39:
        return 0
    elif age <= 44:
        return 1
    elif age <= 49:
        return 2
    elif age <= 54:
        return 3
    elif age <= 59:
        return 4
    elif age <= 64:
        return 5
    elif age <= 69:
        return 6
    elif age <= 74:
        return 7
    else:
        return 8


def _framingham_age_points_female(age: int) -> int:
    if age < 35:
        return -9
    elif age <= 39:
        return -4
    elif age <= 44:
        return 0
    elif age <= 49:
        return 3
    elif age <= 54:
        return 6
    elif age <= 59:
        return 7
    elif age <= 64:
        return 8
    elif age <= 69:
        return 8
    elif age <= 74:
        return 8
    else:
        return 8


def _framingham_tc_points_male(tc: float) -> int:
    """Total cholesterol points (male)."""
    if tc < 160:
        return -3
    elif tc < 200:
        return 0
    elif tc < 240:
        return 1
    elif tc < 280:
        return 2
    else:
        return 3


def _framingham_tc_points_female(tc: float) -> int:
    if tc < 160:
        return -2
    elif tc < 200:
        return 0
    elif tc < 240:
        return 1
    elif tc < 280:
        return 1
    else:
        return 3


def _framingham_hdl_points(hdl: float) -> int:
    """HDL points (same for both sexes in ATP-III)."""
    if hdl >= 60:
        return -2
    elif hdl >= 50:
        return -1
    elif hdl >= 45:
        return 0
    elif hdl >= 35:
        return 1
    else:
        return 2


def _framingham_sbp_points_male(sbp: float, treated: bool) -> int:
    if treated:
        if sbp < 120:
            return -1
        elif sbp < 130:
            return 0
        elif sbp < 140:
            return 1
        elif sbp < 160:
            return 2
        else:
            return 3
    else:
        if sbp < 120:
            return -2
        elif sbp < 130:
            return 0
        elif sbp < 140:
            return 1
        elif sbp < 160:
            return 2
        else:
            return 3


def _framingham_sbp_points_female(sbp: float, treated: bool) -> int:
    if treated:
        if sbp < 120:
            return -1
        elif sbp < 130:
            return 2
        elif sbp < 140:
            return 3
        elif sbp < 160:
            return 5
        else:
            return 6
    else:
        if sbp < 120:
            return -3
        elif sbp < 130:
            return 0
        elif sbp < 140:
            return 1
        elif sbp < 160:
            return 2
        else:
            return 3


def _framingham_smoking_points(smoker: bool, male: bool) -> int:
    if not smoker:
        return 0
    return 4 if male else 3


def _framingham_diabetes_points(diabetic: bool, male: bool) -> int:
    if not diabetic:
        return 0
    return 2 if male else 4


# Point → 10-year risk lookup (ATP-III tables)
_RISK_TABLE_MALE = {
    -3: 1.0, -2: 1.0, -1: 1.0, 0: 2.0, 1: 2.0, 2: 3.0, 3: 4.0,
    4: 5.0, 5: 7.0, 6: 8.0, 7: 10.0, 8: 13.0, 9: 16.0, 10: 20.0,
    11: 25.0, 12: 31.0, 13: 37.0, 14: 45.0,
}

_RISK_TABLE_FEMALE = {
    -2: 1.0, -1: 1.0, 0: 2.0, 1: 2.0, 2: 3.0, 3: 3.0, 4: 4.0,
    5: 5.0, 6: 6.0, 7: 7.0, 8: 8.0, 9: 9.0, 10: 11.0, 11: 13.0,
    12: 15.0, 13: 17.0, 14: 20.0, 15: 24.0, 16: 27.0,
}


def framingham_risk(
    age: int,
    sex: str,
    total_cholesterol: float,
    hdl: float,
    systolic_bp: float,
    bp_treated: bool = False,
    smoker: bool = False,
    diabetic: bool = False,
) -> FraminghamResult:
    """Framingham 10-year CHD risk (ATP-III point system).

    Parameters
    ----------
    age : int
        Patient age in years (30–79).
    sex : str
        "M" or "F".
    total_cholesterol : float
        Total cholesterol in mg/dL.
    hdl : float
        HDL cholesterol in mg/dL.
    systolic_bp : float
        Systolic blood pressure in mmHg.
    bp_treated : bool
        Whether patient is on antihypertensive medication.
    smoker : bool
        Current cigarette smoker.
    diabetic : bool
        Diagnosed diabetes.

    Returns
    -------
    FraminghamResult
    """
    male = sex.upper().startswith("M")
    breakdown: Dict[str, int] = {}

    if male:
        breakdown["age"] = _framingham_age_points_male(age)
        breakdown["total_cholesterol"] = _framingham_tc_points_male(total_cholesterol)
        breakdown["sbp"] = _framingham_sbp_points_male(systolic_bp, bp_treated)
    else:
        breakdown["age"] = _framingham_age_points_female(age)
        breakdown["total_cholesterol"] = _framingham_tc_points_female(total_cholesterol)
        breakdown["sbp"] = _framingham_sbp_points_female(systolic_bp, bp_treated)

    breakdown["hdl"] = _framingham_hdl_points(hdl)
    breakdown["smoking"] = _framingham_smoking_points(smoker, male)
    breakdown["diabetes"] = _framingham_diabetes_points(diabetic, male)

    total = sum(breakdown.values())

    # Lookup risk
    table = _RISK_TABLE_MALE if male else _RISK_TABLE_FEMALE
    keys = sorted(table.keys())
    if total <= keys[0]:
        risk_pct = table[keys[0]]
    elif total >= keys[-1]:
        risk_pct = table[keys[-1]]
    else:
        risk_pct = table.get(total, table[keys[-1]])

    # Risk category (ATP-III)
    if risk_pct < 10:
        category = "low"
    elif risk_pct < 20:
        category = "moderate"
    else:
        category = "high"

    return FraminghamResult(
        points_total=total,
        risk_10yr_pct=risk_pct,
        risk_category=category,
        point_breakdown=breakdown,
    )


# ============================================================================
# 1b. CARDIOVASCULAR — ASCVD Pooled Cohort Equations (Goff 2014)
# ============================================================================

@dataclass(frozen=True)
class ASCVDResult:
    """Output of ACC/AHA Pooled Cohort 10-year ASCVD risk."""
    risk_10yr_pct: float
    risk_category: str      # "low" / "borderline" / "intermediate" / "high"


def _ascvd_ln_terms(
    age: int, sex: str,
    total_cholesterol: float, hdl: float,
    systolic_bp: float, bp_treated: bool,
    smoker: bool, diabetic: bool,
) -> Tuple[float, float, float]:
    """Compute individual sum for the Pooled Cohort Equations.

    Reference: Goff DC et al., Circulation 2014;129:S49-S73
    Coefficients for White male/female (most common in published tables).
    """
    male = sex.upper().startswith("M")
    ln_age = math.log(max(age, 20))
    ln_tc = math.log(max(total_cholesterol, 1))
    ln_hdl = math.log(max(hdl, 1))
    ln_sbp = math.log(max(systolic_bp, 1))

    if male:
        # White male coefficients
        s = (12.344 * ln_age
             + 11.853 * ln_tc
             - 2.664 * ln_age * ln_tc
             - 7.990 * ln_hdl
             + 1.769 * ln_age * ln_hdl
             + (1.797 if bp_treated else 1.764) * ln_sbp
             + (7.837 if smoker else 0.0)
             - 1.795 * ln_age * (1.0 if smoker else 0.0)
             + (0.658 if diabetic else 0.0))
        baseline_survival = 0.9144
        mean_coeff = 61.18
    else:
        # White female coefficients
        s = (-29.799 * ln_age
             + 13.540 * ln_tc
             - 13.578 * ln_hdl
             + (2.019 if bp_treated else 1.957) * ln_sbp
             + (7.574 if smoker else 0.0)
             - 1.665 * ln_age * (1.0 if smoker else 0.0)
             + (0.661 if diabetic else 0.0))
        baseline_survival = 0.9665
        mean_coeff = -29.18

    return s, baseline_survival, mean_coeff


def ascvd_pooled_cohort(
    age: int,
    sex: str,
    total_cholesterol: float,
    hdl: float,
    systolic_bp: float,
    bp_treated: bool = False,
    smoker: bool = False,
    diabetic: bool = False,
) -> ASCVDResult:
    """ACC/AHA Pooled Cohort 10-year ASCVD risk.

    Reference: Goff DC et al., Circulation 2014;129:S49-S73.
    Valid for ages 40-79. Uses White coefficients.

    Parameters
    ----------
    age : int             Patient age (40-79)
    sex : str             "M" or "F"
    total_cholesterol :   Total cholesterol mg/dL
    hdl :                 HDL cholesterol mg/dL
    systolic_bp :         Systolic BP mmHg
    bp_treated :          On antihypertensive Rx
    smoker :              Current smoker
    diabetic :            Diabetes

    Returns
    -------
    ASCVDResult
    """
    s, baseline, mean_c = _ascvd_ln_terms(
        age, sex, total_cholesterol, hdl,
        systolic_bp, bp_treated, smoker, diabetic,
    )
    risk = 1.0 - baseline ** math.exp(s - mean_c)
    risk_pct = max(0.0, min(100.0, risk * 100.0))

    if risk_pct < 5.0:
        cat = "low"
    elif risk_pct < 7.5:
        cat = "borderline"
    elif risk_pct < 20.0:
        cat = "intermediate"
    else:
        cat = "high"

    return ASCVDResult(risk_10yr_pct=round(risk_pct, 1), risk_category=cat)


# ============================================================================
# 1c. CARDIOVASCULAR — SCORE2 (ESC 2021, Low-Risk Region)
# ============================================================================

@dataclass(frozen=True)
class SCORE2Result:
    """Output of ESC SCORE2 10-year fatal+non-fatal CVD risk."""
    risk_10yr_pct: float
    risk_category: str      # "low" / "moderate" / "high" / "very_high"


def score2_esc(
    age: int,
    sex: str,
    total_cholesterol_mmol: float,
    hdl_mmol: float,
    systolic_bp: float,
    smoker: bool = False,
) -> SCORE2Result:
    """ESC SCORE2 10-year cardiovascular risk (low-risk region).

    Simplified recalibrated model for low-risk European countries.
    Reference: SCORE2 working group, Eur Heart J 2021;42:2439-2454.

    Parameters
    ----------
    age : int                  Patient age (40-69)
    sex : str                  "M" or "F"
    total_cholesterol_mmol :   Total cholesterol in mmol/L
    hdl_mmol :                 HDL in mmol/L
    systolic_bp :              Systolic BP in mmHg
    smoker :                   Current smoker

    Note: To convert mg/dL to mmol/L for cholesterol, divide by 38.67.
    """
    male = sex.upper().startswith("M")
    cage = min(max(age, 40), 69)

    # Simplified SCORE2 low-risk region model
    if male:
        beta_age = 0.064
        beta_sbp = 0.018
        beta_tc = 0.12
        beta_hdl = -0.28
        beta_smoke = 0.63
        baseline_10yr = 0.03
    else:
        beta_age = 0.072
        beta_sbp = 0.022
        beta_tc = 0.10
        beta_hdl = -0.32
        beta_smoke = 0.56
        baseline_10yr = 0.015

    # Linear predictor (centered at age 55, SBP 130, TC 5.5, HDL 1.3)
    lp = (beta_age * (cage - 55)
          + beta_sbp * (systolic_bp - 130)
          + beta_tc * (total_cholesterol_mmol - 5.5)
          + beta_hdl * (hdl_mmol - 1.3)
          + beta_smoke * (1.0 if smoker else 0.0))

    risk_pct = baseline_10yr * 100.0 * math.exp(lp)
    risk_pct = max(0.1, min(50.0, risk_pct))

    # ESC risk categories (vary by age; simplified)
    if cage < 50:
        thresholds = (2.5, 7.5)
    elif cage < 70:
        thresholds = (5.0, 10.0)
    else:
        thresholds = (7.5, 15.0)

    if risk_pct < thresholds[0]:
        cat = "low"
    elif risk_pct < thresholds[1]:
        cat = "moderate"
    elif risk_pct < thresholds[1] * 2:
        cat = "high"
    else:
        cat = "very_high"

    return SCORE2Result(risk_10yr_pct=round(risk_pct, 1), risk_category=cat)


# ============================================================================
# 2. RENAL — eGFR (CKD-EPI 2021, Race-Free)
# ============================================================================

@dataclass(frozen=True)
class EGFRResult:
    """Output of eGFR calculation with CKD stage."""
    egfr: float          # mL/min/1.73m^2
    ckd_stage: str       # "G1" / "G2" / "G3a" / "G3b" / "G4" / "G5"
    ckd_description: str


def egfr_ckd_epi_2021(
    creatinine: float,
    age: int,
    sex: str,
) -> EGFRResult:
    """CKD-EPI 2021 race-free eGFR equation.

    Reference: Inker LA et al., NEJM 2021;385:1737-1749

    Parameters
    ----------
    creatinine : float
        Serum creatinine in mg/dL.
    age : int
        Patient age in years.
    sex : str
        "M" or "F".

    Returns
    -------
    EGFRResult
    """
    male = sex.upper().startswith("M")

    if male:
        kappa = 0.9
        alpha = -0.302 if creatinine <= kappa else -1.200
        sex_factor = 1.0
    else:
        kappa = 0.7
        alpha = -0.241 if creatinine <= kappa else -1.200
        sex_factor = 0.9938

    egfr = (
        142.0
        * (min(creatinine / kappa, 1.0) ** alpha)
        * (max(creatinine / kappa, 1.0) ** (-1.200))
        * (0.9938 ** age)
        * sex_factor
    )

    # CKD staging (KDIGO 2012)
    if egfr >= 90:
        stage, desc = "G1", "Normal or high"
    elif egfr >= 60:
        stage, desc = "G2", "Mildly decreased"
    elif egfr >= 45:
        stage, desc = "G3a", "Mildly to moderately decreased"
    elif egfr >= 30:
        stage, desc = "G3b", "Moderately to severely decreased"
    elif egfr >= 15:
        stage, desc = "G4", "Severely decreased"
    else:
        stage, desc = "G5", "Kidney failure"

    return EGFRResult(egfr=round(egfr, 1), ckd_stage=stage, ckd_description=desc)


# ============================================================================
# 2b. HEPATIC — FIB-4 Liver Fibrosis Index (Sterling 2006)
# ============================================================================

@dataclass(frozen=True)
class FIB4Result:
    """Output of FIB-4 liver fibrosis index."""
    fib4: float
    fibrosis_risk: str   # "low" / "indeterminate" / "high"


def fib4_index(
    age: int,
    ast: float,
    alt: float,
    platelets: float,
) -> FIB4Result:
    """FIB-4 liver fibrosis index.

    FIB-4 = (Age x AST) / (Platelets x sqrt(ALT))

    Reference: Sterling RK et al., Hepatology 2006;43:1317-1325

    Parameters
    ----------
    age : int           Patient age in years
    ast : float         AST in U/L
    alt : float         ALT in U/L
    platelets : float   Platelet count in 10^9/L (= 10^3/uL)

    Returns
    -------
    FIB4Result
    """
    sqrt_alt = math.sqrt(max(alt, 0.1))
    denom = max(platelets, 0.1) * sqrt_alt
    fib4 = (age * ast) / denom

    if fib4 < 1.30:
        risk = "low"
    elif fib4 < 2.67:
        risk = "indeterminate"
    else:
        risk = "high"

    return FIB4Result(fib4=round(fib4, 2), fibrosis_risk=risk)


# ============================================================================
# 3. METABOLIC — HOMA-IR + Metabolic Syndrome (ATP-III)
# ============================================================================

@dataclass(frozen=True)
class HOMAResult:
    """Output of HOMA-IR calculation."""
    homa_ir: float
    insulin_resistance: str   # "normal" / "borderline" / "resistant"


def homa_ir(
    fasting_glucose_mg_dl: float,
    fasting_insulin_uU_ml: float,
) -> HOMAResult:
    """Homeostatic Model Assessment of Insulin Resistance.

    HOMA-IR = (Glucose [mg/dL] x Insulin [uU/mL]) / 405

    Reference: Matthews DR et al., Diabetologia 1985;28:412-419

    Parameters
    ----------
    fasting_glucose_mg_dl : float
        Fasting plasma glucose in mg/dL.
    fasting_insulin_uU_ml : float
        Fasting serum insulin in uU/mL (= mIU/L).

    Returns
    -------
    HOMAResult
    """
    ir = (fasting_glucose_mg_dl * fasting_insulin_uU_ml) / 405.0

    if ir < 1.0:
        category = "normal"
    elif ir < 2.5:
        category = "borderline"
    else:
        category = "resistant"

    return HOMAResult(homa_ir=round(ir, 2), insulin_resistance=category)


@dataclass(frozen=True)
class MetSynResult:
    """Metabolic Syndrome assessment (ATP-III / AHA 2005)."""
    criteria_met: int           # out of 5
    is_metabolic_syndrome: bool # >= 3 criteria
    criteria_detail: Dict[str, bool]


def metabolic_syndrome_atp3(
    waist_cm: float,
    sex: str,
    triglycerides: float,
    hdl: float,
    systolic_bp: float,
    diastolic_bp: float,
    fasting_glucose: float,
    bp_treated: bool = False,
    tg_treated: bool = False,
    glucose_treated: bool = False,
) -> MetSynResult:
    """Metabolic Syndrome diagnosis (ATP-III / AHA-NHLBI 2005).

    Any 3 of 5 criteria → Metabolic Syndrome.

    Reference: Grundy SM et al., Circulation 2005;112:2735-2752

    Parameters
    ----------
    waist_cm : float
        Waist circumference in cm.
    sex : str
        "M" or "F".
    triglycerides : float
        Triglycerides in mg/dL.
    hdl : float
        HDL cholesterol in mg/dL.
    systolic_bp / diastolic_bp : float
        Blood pressure in mmHg.
    fasting_glucose : float
        Fasting glucose in mg/dL.
    bp_treated : bool
        On antihypertensive medication.
    tg_treated : bool
        On fibrate / niacin for TG.
    glucose_treated : bool
        On glucose-lowering medication.

    Returns
    -------
    MetSynResult
    """
    male = sex.upper().startswith("M")
    detail: Dict[str, bool] = {}

    # Criterion 1: Waist circumference
    waist_threshold = 102.0 if male else 88.0
    detail["elevated_waist"] = waist_cm >= waist_threshold

    # Criterion 2: Triglycerides >= 150 or on TG Rx
    detail["elevated_tg"] = triglycerides >= 150.0 or tg_treated

    # Criterion 3: Low HDL
    hdl_threshold = 40.0 if male else 50.0
    detail["low_hdl"] = hdl < hdl_threshold

    # Criterion 4: Elevated BP
    detail["elevated_bp"] = (
        systolic_bp >= 130.0
        or diastolic_bp >= 85.0
        or bp_treated
    )

    # Criterion 5: Elevated fasting glucose
    detail["elevated_glucose"] = fasting_glucose >= 100.0 or glucose_treated

    n_met = sum(detail.values())

    return MetSynResult(
        criteria_met=n_met,
        is_metabolic_syndrome=n_met >= 3,
        criteria_detail=detail,
    )


# ============================================================================
# 4. Combined Report
# ============================================================================

@dataclass
class TextbookReport:
    """Combined textbook risk assessment across all six families."""
    framingham: Optional[FraminghamResult] = None
    ascvd: Optional[ASCVDResult] = None
    score2: Optional[SCORE2Result] = None
    egfr: Optional[EGFRResult] = None
    fib4: Optional[FIB4Result] = None
    homa: Optional[HOMAResult] = None
    metabolic_syndrome: Optional[MetSynResult] = None

    def summary_lines(self) -> List[str]:
        """Human-readable summary of all available results."""
        lines = []
        if self.framingham:
            f = self.framingham
            lines.append(
                f"[CV]  Framingham 10yr CHD risk: {f.risk_10yr_pct:.0f}% "
                f"({f.risk_category}) — {f.points_total} pts"
            )
        if self.ascvd:
            a = self.ascvd
            lines.append(
                f"[CV]  ASCVD Pooled Cohort 10yr: {a.risk_10yr_pct:.1f}% "
                f"({a.risk_category})"
            )
        if self.score2:
            s = self.score2
            lines.append(
                f"[CV]  SCORE2 (ESC, low-risk): {s.risk_10yr_pct:.1f}% "
                f"({s.risk_category})"
            )
        if self.egfr:
            e = self.egfr
            lines.append(
                f"[Renal]  eGFR (CKD-EPI 2021): {e.egfr:.1f} mL/min/1.73m2 "
                f"— {e.ckd_stage} ({e.ckd_description})"
            )
        if self.fib4:
            fb = self.fib4
            lines.append(
                f"[Hepatic]  FIB-4: {fb.fib4:.2f} ({fb.fibrosis_risk} "
                f"fibrosis risk)"
            )
        if self.homa:
            h = self.homa
            lines.append(
                f"[Met]  HOMA-IR: {h.homa_ir:.2f} ({h.insulin_resistance})"
            )
        if self.metabolic_syndrome:
            m = self.metabolic_syndrome
            met_str = "YES" if m.is_metabolic_syndrome else "No"
            lines.append(
                f"[Met]  Metabolic Syndrome (ATP-III): {met_str} "
                f"({m.criteria_met}/5 criteria)"
            )
            for k, v in m.criteria_detail.items():
                mark = "X" if v else " "
                lines.append(f"       [{mark}] {k}")
        return lines

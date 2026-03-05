# -*- coding: utf-8 -*-
"""textbook_risk.py — Established Clinical Risk Calculators
=============================================================

Purpose
-------
Standard-of-care risk formulas that serve as the **textbook baseline** for
the Gamma-Net public verification platform.  Every formula here is published
in peer-reviewed guidelines and is computed with **fixed coefficients** —
exactly as a clinician would calculate at bedside.

Three formula families
----------------------
1. Cardiovascular:  Framingham 10-year CHD risk  (Wilson 1998, ATP-III)
2. Renal:           eGFR via CKD-EPI 2021 (race-free, NEJM 2021)
3. Metabolic:       HOMA-IR  + Metabolic Syndrome (ATP-III/AHA 2005)

Design principle
----------------
These functions intentionally do NOT use any Gamma-Net concept.  They exist
so users can see *textbook side-by-side with Gamma-Net* and judge for
themselves.

References
----------
- Wilson PWF et al., Circulation 1998;97:1837-1847  (Framingham)
- Inker LA et al., NEJM 2021;385:1737-1749          (CKD-EPI 2021)
- Matthews DR et al., Diabetologia 1985;28:412-419   (HOMA-IR)
- Grundy SM et al., Circulation 2005;112:2735-2752   (MetS ATP-III)
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
    """Combined textbook risk assessment across all three families."""
    framingham: Optional[FraminghamResult] = None
    egfr: Optional[EGFRResult] = None
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
        if self.egfr:
            e = self.egfr
            lines.append(
                f"[Renal]  eGFR (CKD-EPI 2021): {e.egfr:.1f} mL/min/1.73m2 "
                f"— {e.ckd_stage} ({e.ckd_description})"
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

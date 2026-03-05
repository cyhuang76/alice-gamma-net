# -*- coding: utf-8 -*-
"""demo_dual_rail.py — Textbook vs Gamma-Net Side-by-Side Comparison
=====================================================================

"We don't ask you to believe. We ask you to verify."

This CLI demonstration takes a single set of health-check inputs and
computes BOTH textbook risk scores AND Gamma-Net organ-level Gamma
vectors, displaying them side by side so any user can compare.

Usage
-----
    py -3.11 experiments/demo_dual_rail.py                # default case
    py -3.11 experiments/demo_dual_rail.py --interactive  # enter your own values

Design principle
----------------
- Textbook side: Framingham, eGFR (CKD-EPI 2021), HOMA-IR, MetS (ATP-III)
- Gamma-Net side: 12-organ Gamma vector, H = prod(1 - Gamma_i^2), top contributors
- ZERO parameter overlap: textbook uses published coefficients, Gamma-Net uses
  fixed Z_normal from physics. Neither side has been fitted to the other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from alice.diagnostics.lab_mapping import LabMapper, ORGAN_LIST, ORGAN_SYSTEMS
from alice.diagnostics.gamma_engine import GammaEngine, PatientGammaVector
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
# 1. Test Cases — representative clinical profiles
# ============================================================================

PROFILES = {
    "healthy_adult": {
        "label": "Healthy 40M — annual check-up, all normal",
        "demographics": {"age": 40, "sex": "M", "waist_cm": 85.0},
        "vitals": {
            "systolic_bp": 118.0, "diastolic_bp": 76.0,
            "bp_treated": False, "smoker": False,
        },
        "labs": {
            "TC": 185.0, "HDL": 55.0, "LDL": 110.0, "TG": 100.0,
            "Glucose": 88.0, "HbA1c": 5.2, "Cr": 0.9,
            "AST": 22.0, "ALT": 25.0, "Albumin": 4.5,
            "Na": 140.0, "K": 4.2, "BUN": 14.0,
            "WBC": 6.5, "Hb": 15.0, "Plt": 250.0,
            "TSH": 2.1,
        },
        "insulin": 8.0,
    },
    "metabolic_risk": {
        "label": "55M — pre-diabetic, hypertensive, central obesity",
        "demographics": {"age": 55, "sex": "M", "waist_cm": 108.0},
        "vitals": {
            "systolic_bp": 145.0, "diastolic_bp": 92.0,
            "bp_treated": True, "smoker": False,
        },
        "labs": {
            "TC": 242.0, "HDL": 36.0, "LDL": 165.0, "TG": 210.0,
            "Glucose": 118.0, "HbA1c": 6.8, "Cr": 1.1,
            "AST": 38.0, "ALT": 45.0, "Albumin": 4.0,
            "Na": 141.0, "K": 4.5, "BUN": 18.0,
            "WBC": 7.8, "Hb": 14.2, "Plt": 280.0,
            "TSH": 3.0,
        },
        "insulin": 18.0,
    },
    "kidney_warning": {
        "label": "62F — diabetic with early nephropathy",
        "demographics": {"age": 62, "sex": "F", "waist_cm": 96.0},
        "vitals": {
            "systolic_bp": 152.0, "diastolic_bp": 88.0,
            "bp_treated": True, "smoker": False,
        },
        "labs": {
            "TC": 220.0, "HDL": 42.0, "LDL": 138.0, "TG": 198.0,
            "Glucose": 142.0, "HbA1c": 7.8, "Cr": 1.6,
            "AST": 28.0, "ALT": 32.0, "Albumin": 3.4,
            "Na": 138.0, "K": 5.1, "BUN": 32.0,
            "WBC": 8.2, "Hb": 11.5, "Plt": 220.0,
            "TSH": 4.5,
        },
        "insulin": 22.0,
    },
    "young_smoker": {
        "label": "32M — smoker, high LDL, otherwise OK",
        "demographics": {"age": 32, "sex": "M", "waist_cm": 88.0},
        "vitals": {
            "systolic_bp": 126.0, "diastolic_bp": 80.0,
            "bp_treated": False, "smoker": True,
        },
        "labs": {
            "TC": 265.0, "HDL": 38.0, "LDL": 185.0, "TG": 160.0,
            "Glucose": 92.0, "HbA1c": 5.4, "Cr": 1.0,
            "AST": 20.0, "ALT": 22.0, "Albumin": 4.6,
            "Na": 141.0, "K": 4.0, "BUN": 12.0,
            "WBC": 7.0, "Hb": 15.5, "Plt": 260.0,
            "TSH": 1.8,
        },
        "insulin": 7.0,
    },
}


# ============================================================================
# 2. Core Comparison Engine
# ============================================================================

def compute_textbook(profile: dict) -> TextbookReport:
    """Compute all textbook scores for a profile."""
    d = profile["demographics"]
    v = profile["vitals"]
    labs = profile["labs"]

    report = TextbookReport()

    is_diabetic = labs.get("HbA1c", 5.0) >= 6.5

    # Framingham
    report.framingham = framingham_risk(
        age=d["age"], sex=d["sex"],
        total_cholesterol=labs["TC"], hdl=labs["HDL"],
        systolic_bp=v["systolic_bp"], bp_treated=v["bp_treated"],
        smoker=v["smoker"], diabetic=is_diabetic,
    )

    # ASCVD Pooled Cohort
    if d["age"] >= 40:
        report.ascvd = ascvd_pooled_cohort(
            age=d["age"], sex=d["sex"],
            total_cholesterol=labs["TC"], hdl=labs["HDL"],
            systolic_bp=v["systolic_bp"], bp_treated=v["bp_treated"],
            smoker=v["smoker"], diabetic=is_diabetic,
        )

    # SCORE2 (needs mmol/L: mg/dL / 38.67)
    if 40 <= d["age"] <= 69:
        report.score2 = score2_esc(
            age=d["age"], sex=d["sex"],
            total_cholesterol_mmol=labs["TC"] / 38.67,
            hdl_mmol=labs["HDL"] / 38.67,
            systolic_bp=v["systolic_bp"],
            smoker=v["smoker"],
        )

    # eGFR
    report.egfr = egfr_ckd_epi_2021(
        creatinine=labs["Cr"], age=d["age"], sex=d["sex"],
    )

    # FIB-4
    if "AST" in labs and "ALT" in labs and "Plt" in labs:
        report.fib4 = fib4_index(
            age=d["age"], ast=labs["AST"],
            alt=labs["ALT"], platelets=labs["Plt"],
        )

    # HOMA-IR
    if "insulin" in profile:
        report.homa = homa_ir(
            fasting_glucose_mg_dl=labs["Glucose"],
            fasting_insulin_uU_ml=profile["insulin"],
        )

    # Metabolic Syndrome
    report.metabolic_syndrome = metabolic_syndrome_atp3(
        waist_cm=d["waist_cm"], sex=d["sex"],
        triglycerides=labs["TG"], hdl=labs["HDL"],
        systolic_bp=v["systolic_bp"], diastolic_bp=v["diastolic_bp"],
        fasting_glucose=labs["Glucose"],
        bp_treated=v["bp_treated"],
    )

    return report


def compute_gamma(profile: dict) -> PatientGammaVector:
    """Compute Gamma-Net organ vector for a profile."""
    engine = GammaEngine()
    return engine.lab_to_gamma(profile["labs"])


# ============================================================================
# 3. Display
# ============================================================================

SEP = "=" * 72
THIN = "-" * 72


def print_header(label: str) -> None:
    print(f"\n{SEP}")
    print(f"  CASE: {label}")
    print(SEP)


def print_dual_rail(profile: dict) -> None:
    """Print textbook vs Gamma-Net side by side."""
    label = profile["label"]
    print_header(label)

    # --- Textbook side ---
    tb = compute_textbook(profile)
    print(f"\n{'TEXTBOOK ASSESSMENT':^36} | {'GAMMA-NET ASSESSMENT':^34}")
    print(THIN)

    tb_lines = tb.summary_lines()

    # --- Gamma-Net side ---
    gv = compute_gamma(profile)
    h_product = gv.health_index
    h_pct = h_product * 100
    top3 = gv.top_n_organs(3)
    total_g2 = gv.total_gamma_squared

    # Print textbook
    print("  TEXTBOOK (published formulas):")
    for line in tb_lines:
        print(f"    {line}")

    print()
    print(THIN)

    # Print Gamma-Net
    print("  GAMMA-NET (zero-parameter physics):")
    print(f"    H = prod(1 - Gi^2) = {h_product:.6f}  ({h_pct:.2f}%)")
    print(f"    Total Gamma^2 = {total_g2:.4f}")
    print()
    print("    12-Organ Gamma Decomposition:")
    print(f"    {'Organ':<14} {'Gamma':>8} {'Gamma^2':>9} {'T=1-G^2':>9}  Bar")
    print(f"    {'-'*14} {'-'*8} {'-'*9} {'-'*9}  {'-'*20}")

    for organ in ORGAN_LIST:
        g = gv[organ]
        g2 = g ** 2
        t = 1.0 - g2
        bar_len = int(abs(g) * 40)
        bar = "#" * bar_len
        print(f"    {organ:<14} {g:>+8.4f} {g2:>9.4f} {t:>9.4f}  {bar}")

    print()
    print(f"    Top mismatch channels:")
    for i, (organ, g) in enumerate(top3, 1):
        print(f"      {i}. {organ}: Gamma = {g:+.4f} (Gamma^2 = {g**2:.4f})")

    # --- Cross-comparison ---
    print()
    print(THIN)
    print("  CROSS-COMPARISON:")

    # CV comparison (triple: Framingham + ASCVD + SCORE2)
    g_cardiac = gv["cardiac"]
    g_vascular = gv["vascular"]
    cv_gamma = max(abs(g_cardiac), abs(g_vascular))
    print(f"    Cardiovascular:")
    if tb.framingham:
        fr = tb.framingham
        print(f"      Framingham: {fr.risk_10yr_pct:.0f}% 10yr ({fr.risk_category})")
    if tb.ascvd:
        av = tb.ascvd
        print(f"      ASCVD PCE:  {av.risk_10yr_pct:.1f}% 10yr ({av.risk_category})")
    if tb.score2:
        s2 = tb.score2
        print(f"      SCORE2 ESC: {s2.risk_10yr_pct:.1f}% 10yr ({s2.risk_category})")
    print(f"      Gamma-Net:  cardiac G={g_cardiac:+.4f}, vascular G={g_vascular:+.4f}")
    if cv_gamma > 0.15:
        print(f"      --> Gamma-Net flags cardiovascular mismatch")
    elif cv_gamma < 0.05:
        print(f"      --> Both systems: cardiovascular looks good")
    else:
        print(f"      --> Mild mismatch — monitor trend over time")

    # Renal comparison
    if tb.egfr:
        eg = tb.egfr
        g_renal = gv["renal"]
        print(f"    Renal:")
        print(f"      eGFR: {eg.egfr:.1f} ({eg.ckd_stage})")
        print(f"      Gamma-Net:  renal G={g_renal:+.4f} (G^2={g_renal**2:.4f})")
        if eg.ckd_stage in ("G3a", "G3b", "G4", "G5") and abs(g_renal) > 0.1:
            print(f"      --> Both flag renal impairment")
        elif eg.ckd_stage in ("G1", "G2") and abs(g_renal) < 0.08:
            print(f"      --> Both: renal adequate")
        else:
            print(f"      --> Check which system detects earlier")

    # Hepatic comparison (FIB-4 vs Gamma hepatic)
    if tb.fib4:
        fb = tb.fib4
        g_hepatic = gv["hepatic"]
        print(f"    Hepatic:")
        print(f"      FIB-4: {fb.fib4:.2f} ({fb.fibrosis_risk})")
        print(f"      Gamma-Net:  hepatic G={g_hepatic:+.4f} (G^2={g_hepatic**2:.4f})")
        if fb.fibrosis_risk == "high" and abs(g_hepatic) > 0.15:
            print(f"      --> Both flag hepatic concern")
        elif fb.fibrosis_risk == "low" and abs(g_hepatic) < 0.10:
            print(f"      --> Both: liver looks OK")
        else:
            print(f"      --> Discordance — Gamma sees what FIB-4 may miss")

    # Metabolic comparison
    if tb.homa:
        hm = tb.homa
        g_endo = gv["endocrine"]
        print(f"    Metabolic:")
        print(f"      HOMA-IR: {hm.homa_ir:.2f} ({hm.insulin_resistance})")
        print(f"      Gamma-Net:  endocrine G={g_endo:+.4f} (G^2={g_endo**2:.4f})")

    if tb.metabolic_syndrome:
        ms = tb.metabolic_syndrome
        met_label = "YES" if ms.is_metabolic_syndrome else "No"
        print(f"      MetS (ATP-III): {met_label} ({ms.criteria_met}/5)")

    # === D_Z IMPEDANCE DEBT PANEL ===
    print()
    print(THIN)
    print("  IMPEDANCE DEBT (D_Z) PROJECTION:")
    print("    D_Z = integral of sum(Gi^2) over time")
    print("    = cumulative cost of living with mismatched organs")
    print()

    # Annual D_Z estimate: total_g2 * 365 * 24 (hours/year, arbitrary unit)
    daily_cost = total_g2
    annual_dz = daily_cost * 365.0
    print(f"    Current daily impedance cost: {daily_cost:.4f}")
    print(f"    Projected annual D_Z:         {annual_dz:.1f}")
    print()

    # Per-organ D_Z contribution
    print(f"    {'Organ':<14} {'Daily dG^2':>10} {'Annual D_Z':>10} {'Share':>7}")
    print(f"    {'-'*14} {'-'*10} {'-'*10} {'-'*7}")
    for organ in ORGAN_LIST:
        g2 = gv[organ] ** 2
        if g2 < 0.0001:
            continue
        organ_annual = g2 * 365.0
        share = g2 / max(total_g2, 1e-12) * 100
        print(f"    {organ:<14} {g2:>10.4f} {organ_annual:>10.1f} {share:>6.1f}%")

    print()
    print(f"    Prediction: The organ with highest D_Z share will")
    print(f"    deteriorate first in follow-up. If not, Gamma-Net is wrong.")

    # KEY INSIGHT
    print()
    print(THIN)
    print("  KEY INSIGHT:")
    print("    Textbook gives SEPARATE scores per specialty (cardiology,")
    print("    nephrology, endocrinology, hepatology). Gamma-Net gives")
    print("    ONE unified framework — same physics, all organs.")
    print(f"    => H captures cross-organ interactions textbook misses.")
    print(f"    => D_Z tracks cumulative cost textbook cannot.")
    print(f"    => If Gamma-Net is wrong, YOU can see where it disagrees.")
    print(SEP)


# ============================================================================
# 4. Interactive Mode
# ============================================================================

def interactive_mode() -> None:
    """Let user enter their own health-check values."""
    print(SEP)
    print("  GAMMA-NET PUBLIC VERIFICATION PLATFORM")
    print("  'We don't ask you to believe. We ask you to verify.'")
    print(SEP)
    print()
    print("  Enter your health-check values below.")
    print("  Leave blank to skip (will use reference midpoint).")
    print()

    def ask_float(prompt: str, default: float) -> float:
        val = input(f"  {prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            print(f"    (invalid, using {default})")
            return default

    def ask_int(prompt: str, default: int) -> int:
        val = input(f"  {prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            print(f"    (invalid, using {default})")
            return default

    def ask_yn(prompt: str, default: bool = False) -> bool:
        d = "Y" if default else "N"
        val = input(f"  {prompt} [Y/N, default={d}]: ").strip().upper()
        if not val:
            return default
        return val.startswith("Y")

    def ask_sex() -> str:
        val = input("  Sex [M/F, default=M]: ").strip().upper()
        if val in ("F", "FEMALE"):
            return "F"
        return "M"

    # Demographics
    print("  --- Demographics ---")
    age = ask_int("Age (years)", 45)
    sex = ask_sex()
    waist = ask_float("Waist circumference (cm)", 90.0)

    # Vitals
    print("\n  --- Vitals ---")
    sbp = ask_float("Systolic BP (mmHg)", 120.0)
    dbp = ask_float("Diastolic BP (mmHg)", 80.0)
    bp_rx = ask_yn("On BP medication?", False)
    smoker = ask_yn("Current smoker?", False)

    # Labs
    print("\n  --- Lab Values ---")
    tc = ask_float("Total Cholesterol (mg/dL)", 200.0)
    hdl = ask_float("HDL (mg/dL)", 50.0)
    ldl = ask_float("LDL (mg/dL)", 120.0)
    tg = ask_float("Triglycerides (mg/dL)", 130.0)
    glucose = ask_float("Fasting Glucose (mg/dL)", 90.0)
    hba1c = ask_float("HbA1c (%)", 5.4)
    cr = ask_float("Creatinine (mg/dL)", 1.0)
    bun = ask_float("BUN (mg/dL)", 14.0)
    ast = ask_float("AST (U/L)", 25.0)
    alt = ask_float("ALT (U/L)", 28.0)
    albumin = ask_float("Albumin (g/dL)", 4.3)
    insulin = ask_float("Fasting Insulin (uU/mL)", 10.0)

    # Build profile
    profile = {
        "label": f"Your Health Check ({age}{sex})",
        "demographics": {"age": age, "sex": sex, "waist_cm": waist},
        "vitals": {
            "systolic_bp": sbp, "diastolic_bp": dbp,
            "bp_treated": bp_rx, "smoker": smoker,
        },
        "labs": {
            "TC": tc, "HDL": hdl, "LDL": ldl, "TG": tg,
            "Glucose": glucose, "HbA1c": hba1c, "Cr": cr,
            "AST": ast, "ALT": alt, "Albumin": albumin,
            "Na": 140.0, "K": 4.2, "BUN": bun,
            "WBC": 7.0, "Hb": 14.0, "Plt": 250.0,
            "TSH": 2.5,
        },
        "insulin": insulin,
    }

    print_dual_rail(profile)


# ============================================================================
# 5. D_Z Time Tracking — Multi-Visit Impedance Debt Panel
# ============================================================================

# Simulated longitudinal visits: metabolic_risk patient over 5 years
TIMELINE_VISITS = [
    {
        "year": 2022, "label": "Year 0: Baseline (55M)",
        "demographics": {"age": 55, "sex": "M", "waist_cm": 108.0},
        "vitals": {"systolic_bp": 145.0, "diastolic_bp": 92.0,
                   "bp_treated": True, "smoker": False},
        "labs": {"TC": 242.0, "HDL": 36.0, "LDL": 165.0, "TG": 210.0,
                 "Glucose": 118.0, "HbA1c": 6.8, "Cr": 1.1,
                 "AST": 38.0, "ALT": 45.0, "Albumin": 4.0,
                 "Na": 141.0, "K": 4.5, "BUN": 18.0,
                 "WBC": 7.8, "Hb": 14.2, "Plt": 280.0, "TSH": 3.0},
        "insulin": 18.0,
    },
    {
        "year": 2023, "label": "Year 1: Lifestyle intervention started",
        "demographics": {"age": 56, "sex": "M", "waist_cm": 104.0},
        "vitals": {"systolic_bp": 138.0, "diastolic_bp": 88.0,
                   "bp_treated": True, "smoker": False},
        "labs": {"TC": 225.0, "HDL": 40.0, "LDL": 148.0, "TG": 185.0,
                 "Glucose": 110.0, "HbA1c": 6.5, "Cr": 1.1,
                 "AST": 35.0, "ALT": 40.0, "Albumin": 4.1,
                 "Na": 141.0, "K": 4.4, "BUN": 17.0,
                 "WBC": 7.5, "Hb": 14.3, "Plt": 275.0, "TSH": 2.8},
        "insulin": 15.0,
    },
    {
        "year": 2024, "label": "Year 2: Statin + metformin added",
        "demographics": {"age": 57, "sex": "M", "waist_cm": 100.0},
        "vitals": {"systolic_bp": 132.0, "diastolic_bp": 84.0,
                   "bp_treated": True, "smoker": False},
        "labs": {"TC": 195.0, "HDL": 44.0, "LDL": 118.0, "TG": 155.0,
                 "Glucose": 102.0, "HbA1c": 6.2, "Cr": 1.05,
                 "AST": 30.0, "ALT": 34.0, "Albumin": 4.2,
                 "Na": 140.0, "K": 4.3, "BUN": 16.0,
                 "WBC": 7.2, "Hb": 14.5, "Plt": 270.0, "TSH": 2.5},
        "insulin": 12.0,
    },
    {
        "year": 2025, "label": "Year 3: Good compliance, improving",
        "demographics": {"age": 58, "sex": "M", "waist_cm": 97.0},
        "vitals": {"systolic_bp": 128.0, "diastolic_bp": 80.0,
                   "bp_treated": True, "smoker": False},
        "labs": {"TC": 188.0, "HDL": 47.0, "LDL": 108.0, "TG": 140.0,
                 "Glucose": 96.0, "HbA1c": 5.9, "Cr": 1.0,
                 "AST": 26.0, "ALT": 28.0, "Albumin": 4.3,
                 "Na": 140.0, "K": 4.2, "BUN": 15.0,
                 "WBC": 7.0, "Hb": 14.6, "Plt": 265.0, "TSH": 2.3},
        "insulin": 10.0,
    },
    {
        "year": 2026, "label": "Year 4: Near-normal metabolic profile",
        "demographics": {"age": 59, "sex": "M", "waist_cm": 94.0},
        "vitals": {"systolic_bp": 125.0, "diastolic_bp": 78.0,
                   "bp_treated": True, "smoker": False},
        "labs": {"TC": 182.0, "HDL": 50.0, "LDL": 100.0, "TG": 125.0,
                 "Glucose": 92.0, "HbA1c": 5.6, "Cr": 0.95,
                 "AST": 24.0, "ALT": 26.0, "Albumin": 4.4,
                 "Na": 140.0, "K": 4.2, "BUN": 14.0,
                 "WBC": 6.8, "Hb": 14.8, "Plt": 260.0, "TSH": 2.2},
        "insulin": 8.0,
    },
]


def print_timeline() -> None:
    """Print D_Z impedance debt tracking over multiple visits."""
    print(f"\n{SEP}")
    print("  D_Z IMPEDANCE DEBT TRACKER")
    print("  ===========================")
    print("  Longitudinal tracking: metabolic risk patient, 5 annual visits")
    print("  Textbook risk scores + Gamma-Net H + cumulative D_Z")
    print(SEP)

    engine = GammaEngine()
    cumulative_dz = 0.0
    results = []

    for visit in TIMELINE_VISITS:
        gv = engine.lab_to_gamma(visit["labs"])
        tb = compute_textbook(visit)
        h = gv.health_index
        total_g2 = gv.total_gamma_squared
        # Annual D_Z contribution (365 days at this level)
        annual_dz = total_g2 * 365.0
        cumulative_dz += annual_dz
        top_organ, top_g = gv.top_n_organs(1)[0]

        results.append({
            "year": visit["year"],
            "label": visit["label"],
            "h": h,
            "g2_total": total_g2,
            "annual_dz": annual_dz,
            "cum_dz": cumulative_dz,
            "top_organ": top_organ,
            "top_g": top_g,
            "tb": tb,
        })

    # Summary table
    print(f"\n  {'Year':<6} {'H':>8} {'Sum(G^2)':>9} {'D_Z(yr)':>9} "
          f"{'D_Z(cum)':>10} {'Worst':>12} {'Framingham':>11} "
          f"{'eGFR':>6} {'HOMA-IR':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*10} {'-'*12} "
          f"{'-'*11} {'-'*6} {'-'*8}")

    for r in results:
        fr_str = f"{r['tb'].framingham.risk_10yr_pct:.0f}%" if r['tb'].framingham else "N/A"
        egfr_str = f"{r['tb'].egfr.egfr:.0f}" if r['tb'].egfr else "N/A"
        homa_str = f"{r['tb'].homa.homa_ir:.1f}" if r['tb'].homa else "N/A"
        print(f"  {r['year']:<6} {r['h']:>8.4f} {r['g2_total']:>9.4f} "
              f"{r['annual_dz']:>9.1f} {r['cum_dz']:>10.1f} "
              f"{r['top_organ']:>12} {fr_str:>11} "
              f"{egfr_str:>6} {homa_str:>8}")

    # D_Z trend visualization
    print(f"\n  D_Z Cumulative Trend:")
    max_dz = results[-1]["cum_dz"]
    for r in results:
        bar_len = int(r["cum_dz"] / max(max_dz, 1) * 40)
        bar = "#" * bar_len
        arrow = " <-- improving" if r != results[0] and r["g2_total"] < results[results.index(r)-1]["g2_total"] else ""
        print(f"    {r['year']}  |{bar:<40}| {r['cum_dz']:>8.1f}{arrow}")

    # H trend visualization
    print(f"\n  H (Health Index) Trend:")
    for r in results:
        bar_len = int(r["h"] * 40)
        bar = "#" * bar_len
        print(f"    {r['year']}  |{bar:<40}| {r['h']*100:>6.2f}%")

    print()
    print(THIN)
    print("  INTERPRETATION:")
    first_g2 = results[0]["g2_total"]
    last_g2 = results[-1]["g2_total"]
    reduction = (1 - last_g2 / max(first_g2, 1e-12)) * 100
    print(f"    Total mismatch reduced: {first_g2:.4f} -> {last_g2:.4f} "
          f"({reduction:.0f}% improvement)")
    print(f"    But cumulative D_Z = {cumulative_dz:.1f} — this debt is permanent.")
    print(f"    Even after recovery, the body remembers the cost of past mismatch.")
    print()
    print("  TEXTBOOK CANNOT SHOW THIS:")
    print("    - Framingham/ASCVD give a snapshot. D_Z gives a lifetime integral.")
    print("    - Two patients with identical current labs but different D_Z")
    print("      have DIFFERENT prognoses. Only Gamma-Net captures this.")
    print(SEP)


# ============================================================================
# 6. Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Textbook vs Gamma-Net: Side-by-Side Health Verification"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Enter your own health-check values",
    )
    parser.add_argument(
        "--case", "-c", choices=list(PROFILES.keys()),
        help="Run a specific test case only",
    )
    parser.add_argument(
        "--timeline", "-t", action="store_true",
        help="Run D_Z impedance debt tracking (5-year longitudinal demo)",
    )
    args = parser.parse_args()

    print()
    print(SEP)
    print("  GAMMA-NET HEALTH CHECK")
    print("  =======================")
    print("  Zero-parameter impedance physics")
    print("  Public verification platform")
    print()
    print("  'Fixed formulas. Transparent results.")
    print("   Verification is yours.'")
    print()
    print("  All calculations below use FIXED formulas.")
    print("  Textbook: published clinical risk equations.")
    print("  Gamma-Net: zero-parameter impedance physics.")
    print("  Neither side has been fitted to the other.")
    print()
    print("  DISCLAIMER: For research and self-verification only.")
    print("  Not medical advice. Always consult your physician.")
    print(SEP)

    if args.interactive:
        interactive_mode()
    elif args.timeline:
        print_timeline()
    elif args.case:
        print_dual_rail(PROFILES[args.case])
    else:
        # Run all built-in profiles
        for name, profile in PROFILES.items():
            print_dual_rail(profile)


if __name__ == "__main__":
    main()

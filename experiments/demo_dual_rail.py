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
    egfr_ckd_epi_2021,
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

    # Framingham
    report.framingham = framingham_risk(
        age=d["age"], sex=d["sex"],
        total_cholesterol=labs["TC"], hdl=labs["HDL"],
        systolic_bp=v["systolic_bp"], bp_treated=v["bp_treated"],
        smoker=v["smoker"],
        diabetic=labs.get("HbA1c", 5.0) >= 6.5,
    )

    # eGFR
    report.egfr = egfr_ckd_epi_2021(
        creatinine=labs["Cr"], age=d["age"], sex=d["sex"],
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

    # CV comparison
    if tb.framingham:
        fr = tb.framingham
        g_cardiac = gv["cardiac"]
        g_vascular = gv["vascular"]
        cv_gamma = max(abs(g_cardiac), abs(g_vascular))
        print(f"    Cardiovascular:")
        print(f"      Framingham: {fr.risk_10yr_pct:.0f}% 10yr risk ({fr.risk_category})")
        print(f"      Gamma-Net:  cardiac Gamma={g_cardiac:+.4f}, vascular Gamma={g_vascular:+.4f}")
        if fr.risk_category == "high" and cv_gamma > 0.15:
            print(f"      --> Both systems flag cardiovascular concern")
        elif fr.risk_category == "low" and cv_gamma < 0.05:
            print(f"      --> Both systems: cardiovascular looks good")
        else:
            print(f"      --> NOTE: Discordance may reveal sub-clinical risk")

    # Renal comparison
    if tb.egfr:
        eg = tb.egfr
        g_renal = gv["renal"]
        print(f"    Renal:")
        print(f"      eGFR: {eg.egfr:.1f} ({eg.ckd_stage})")
        print(f"      Gamma-Net:  renal Gamma={g_renal:+.4f} (Gamma^2={g_renal**2:.4f})")
        if eg.ckd_stage in ("G3a", "G3b", "G4", "G5") and abs(g_renal) > 0.1:
            print(f"      --> Both systems flag renal impairment")
        elif eg.ckd_stage in ("G1", "G2") and abs(g_renal) < 0.08:
            print(f"      --> Both systems: renal function adequate")
        else:
            print(f"      --> NOTE: Check which system detects earlier")

    # Metabolic comparison
    if tb.homa:
        hm = tb.homa
        g_endo = gv["endocrine"]
        print(f"    Metabolic:")
        print(f"      HOMA-IR: {hm.homa_ir:.2f} ({hm.insulin_resistance})")
        print(f"      Gamma-Net:  endocrine Gamma={g_endo:+.4f} (Gamma^2={g_endo**2:.4f})")

    if tb.metabolic_syndrome:
        ms = tb.metabolic_syndrome
        met_label = "YES" if ms.is_metabolic_syndrome else "No"
        print(f"      MetS (ATP-III): {met_label} ({ms.criteria_met}/5)")

    # KEY INSIGHT
    print()
    print(THIN)
    print("  KEY INSIGHT:")
    print("    Textbook gives SEPARATE scores per specialty (cardiology,")
    print("    nephrology, endocrinology). Gamma-Net gives ONE unified")
    print("    framework where all organs share the same physics.")
    print(f"    => H captures cross-organ interactions that textbook misses.")
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
# 5. Main
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
    args = parser.parse_args()

    print()
    print(SEP)
    print("  GAMMA-NET PUBLIC FALSIFICATION PLATFORM")
    print("  ========================================")
    print("  'We cannot prove ourselves innocent.")
    print("   So we ask everyone to verify.'")
    print()
    print("  All calculations below use FIXED formulas.")
    print("  Textbook: published clinical risk equations.")
    print("  Gamma-Net: zero-parameter impedance physics.")
    print("  Neither side has been fitted to the other.")
    print(SEP)

    if args.interactive:
        interactive_mode()
    elif args.case:
        print_dual_rail(PROFILES[args.case])
    else:
        # Run all built-in profiles
        for name, profile in PROFILES.items():
            print_dual_rail(profile)


if __name__ == "__main__":
    main()

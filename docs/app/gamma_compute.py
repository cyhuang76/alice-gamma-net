# -*- coding: utf-8 -*-
"""gamma_compute.py — Browser-side Gamma-Net + Textbook Calculator
===================================================================
Pure Python (no numpy, no external deps). Designed for Pyodide.

Contains:
  1. 12-organ impedance mapping (lab → Z → Γ → H)
  2. Six textbook risk calculators:
     - Framingham ATP-III 10yr CHD
     - ASCVD Pooled Cohort (Goff 2014)
     - SCORE2 ESC 2021 (low-risk region)
     - CKD-EPI 2021 eGFR (race-free)
     - FIB-4 liver fibrosis (Sterling 2006)
     - HOMA-IR + Metabolic Syndrome ATP-III
"""

import math

# ============================================================================
# ORGAN SYSTEM DEFINITIONS
# ============================================================================

ORGAN_SYSTEMS = {
    "cardiac":    50.0,
    "pulmonary":  60.0,
    "hepatic":    65.0,
    "renal":      70.0,
    "endocrine":  75.0,
    "immune":     75.0,
    "heme":       55.0,
    "GI":         65.0,
    "vascular":   45.0,
    "bone":      120.0,
    "neuro":      80.0,
    "repro":      95.0,
}

ORGAN_LIST = list(ORGAN_SYSTEMS.keys())

# Lab item → (ref_low, ref_high, organ_weights)
# organ_weights: {organ: weight}
LAB_ITEMS = {
    "TC":       (150, 200, {"cardiac": 0.4, "vascular": 0.3, "hepatic": 0.2, "endocrine": 0.1}),
    "HDL":      (40,  60,  {"cardiac": 0.4, "vascular": 0.3, "endocrine": 0.2, "hepatic": 0.1}),
    "LDL":      (0,   100, {"cardiac": 0.4, "vascular": 0.4, "hepatic": 0.1, "endocrine": 0.1}),
    "TG":       (0,   150, {"hepatic": 0.3, "endocrine": 0.3, "cardiac": 0.2, "vascular": 0.2}),
    "Glucose":  (70,  100, {"endocrine": 0.5, "vascular": 0.2, "neuro": 0.15, "renal": 0.15}),
    "HbA1c":    (4.0, 5.6, {"endocrine": 0.5, "vascular": 0.2, "neuro": 0.15, "renal": 0.15}),
    "Cr":       (0.6, 1.2, {"renal": 0.7, "cardiac": 0.15, "vascular": 0.15}),
    "BUN":      (7,   20,  {"renal": 0.6, "hepatic": 0.2, "GI": 0.1, "cardiac": 0.1}),
    "AST":      (10,  40,  {"hepatic": 0.5, "cardiac": 0.2, "heme": 0.15, "bone": 0.15}),
    "ALT":      (7,   56,  {"hepatic": 0.7, "cardiac": 0.15, "GI": 0.15}),
    "Albumin":  (3.5, 5.5, {"hepatic": 0.4, "renal": 0.2, "immune": 0.2, "GI": 0.2}),
    "Na":       (136, 145, {"renal": 0.4, "neuro": 0.3, "endocrine": 0.2, "cardiac": 0.1}),
    "K":        (3.5, 5.0, {"renal": 0.3, "cardiac": 0.3, "neuro": 0.2, "endocrine": 0.2}),
    "WBC":      (4.0, 11.0,{"immune": 0.5, "heme": 0.3, "hepatic": 0.1, "GI": 0.1}),
    "Hb":       (12,  17,  {"heme": 0.6, "cardiac": 0.15, "pulmonary": 0.15, "renal": 0.1}),
    "Plt":      (150, 400, {"heme": 0.5, "hepatic": 0.2, "immune": 0.15, "vascular": 0.15}),
    "TSH":      (0.4, 4.0, {"endocrine": 0.6, "cardiac": 0.15, "neuro": 0.15, "bone": 0.1}),
}


# ============================================================================
# GAMMA-NET CALCULATION
# ============================================================================

def compute_organ_impedances(lab_values):
    """lab dict → {organ: Z_patient}"""
    z = {o: ORGAN_SYSTEMS[o] for o in ORGAN_LIST}
    for name, val in lab_values.items():
        if name not in LAB_ITEMS:
            continue
        ref_low, ref_high, weights = LAB_ITEMS[name]
        ref_mid = (ref_low + ref_high) / 2.0
        ref_range = max(ref_high - ref_low, 1e-6)
        delta = abs(val - ref_mid) / ref_range
        for organ, w in weights.items():
            z[organ] += ORGAN_SYSTEMS[organ] * w * delta
    return z


def compute_gamma(z_patient, z_normal):
    """Gamma = (Z_patient - Z_normal) / (Z_patient + Z_normal)"""
    denom = z_patient + z_normal
    if abs(denom) < 1e-12:
        return 0.0
    return (z_patient - z_normal) / denom


def lab_to_gamma(lab_values):
    """lab dict → {organ: Gamma}"""
    z = compute_organ_impedances(lab_values)
    gamma = {}
    for organ in ORGAN_LIST:
        gamma[organ] = compute_gamma(z[organ], ORGAN_SYSTEMS[organ])
    return gamma


def health_index(gamma_dict):
    """H = prod(1 - Gi^2)"""
    h = 1.0
    for g in gamma_dict.values():
        h *= (1.0 - g * g)
    return h


def total_gamma_squared(gamma_dict):
    """Sum of Gi^2"""
    return sum(g * g for g in gamma_dict.values())


# ============================================================================
# 1a. FRAMINGHAM ATP-III (Wilson 1998)
# ============================================================================

def framingham_risk(age, sex, tc, hdl, sbp, smoker=False,
                    diabetic=False, bp_treated=False):
    male = sex.upper().startswith("M")
    pts = 0

    # Age
    if male:
        if age < 35: pts += -1
        elif age < 40: pts += 0
        elif age < 45: pts += 1
        elif age < 50: pts += 2
        elif age < 55: pts += 3
        elif age < 60: pts += 4
        elif age < 65: pts += 5
        elif age < 70: pts += 6
        elif age < 75: pts += 7
        else: pts += 8
    else:
        if age < 35: pts += -7
        elif age < 40: pts += -3
        elif age < 45: pts += 0
        elif age < 50: pts += 3
        elif age < 55: pts += 6
        elif age < 60: pts += 8
        elif age < 65: pts += 10
        elif age < 70: pts += 12
        elif age < 75: pts += 14
        else: pts += 16

    # TC
    if tc < 160: pts += (0 if male else -2)
    elif tc < 200: pts += (0 if male else 0)
    elif tc < 240: pts += (1 if male else 1)
    elif tc < 280: pts += (2 if male else 1)
    else: pts += (3 if male else 3)

    # HDL
    if hdl >= 60: pts += (-2 if male else -2)
    elif hdl >= 50: pts += (-1 if male else -1)
    elif hdl >= 45: pts += (0 if male else 0)
    elif hdl >= 35: pts += (1 if male else 1)
    else: pts += (2 if male else 2)

    # SBP
    if bp_treated:
        if sbp < 120: pts += (0 if male else -1)
        elif sbp < 130: pts += (2 if male else 2)
        elif sbp < 140: pts += (3 if male else 3)
        elif sbp < 160: pts += (4 if male else 5)
        else: pts += (5 if male else 6)
    else:
        if sbp < 120: pts += (-2 if male else -3)
        elif sbp < 130: pts += (0 if male else 0)
        elif sbp < 140: pts += (1 if male else 1)
        elif sbp < 160: pts += (2 if male else 2)
        else: pts += (3 if male else 3)

    # Smoking
    if smoker:
        pts += (4 if male else 3)

    # Diabetes
    if diabetic:
        pts += (3 if male else 4)

    # Lookup
    if male:
        table = {-3:1,-2:1,-1:2,0:2,1:2,2:3,3:4,4:5,5:7,6:8,
                 7:10,8:13,9:16,10:20,11:25,12:31,13:37,14:45}
    else:
        table = {-2:1,-1:2,0:2,1:2,2:3,3:3,4:4,5:5,6:6,
                 7:7,8:8,9:9,10:11,11:13,12:15,13:17,14:20,
                 15:24,16:27,17:32,18:37,19:42,20:47,21:50}

    clamped = max(min(pts, max(table.keys())), min(table.keys()))
    risk_pct = table[clamped]

    if risk_pct < 10: cat = "low"
    elif risk_pct < 20: cat = "moderate"
    else: cat = "high"

    return {"risk_10yr_pct": risk_pct, "risk_category": cat, "points": pts}


# ============================================================================
# 1b. ASCVD POOLED COHORT (Goff 2014)
# ============================================================================

def ascvd_pooled_cohort(age, sex, tc, hdl, sbp,
                        bp_treated=False, smoker=False, diabetic=False):
    male = sex.upper().startswith("M")
    ln_age = math.log(max(age, 20))
    ln_tc = math.log(max(tc, 1))
    ln_hdl = math.log(max(hdl, 1))
    ln_sbp = math.log(max(sbp, 1))

    if male:
        s = (12.344 * ln_age + 11.853 * ln_tc - 2.664 * ln_age * ln_tc
             - 7.990 * ln_hdl + 1.769 * ln_age * ln_hdl
             + (1.797 if bp_treated else 1.764) * ln_sbp
             + (7.837 if smoker else 0.0)
             - 1.795 * ln_age * (1.0 if smoker else 0.0)
             + (0.658 if diabetic else 0.0))
        baseline = 0.9144
        mean_c = 61.18
    else:
        s = (-29.799 * ln_age + 4.884 * ln_age * ln_age
             + 13.540 * ln_tc - 3.114 * ln_age * ln_tc
             - 13.578 * ln_hdl + 3.149 * ln_age * ln_hdl
             + (2.019 if bp_treated else 1.957) * ln_sbp
             + (7.574 if smoker else 0.0)
             - 1.665 * ln_age * (1.0 if smoker else 0.0)
             + (0.661 if diabetic else 0.0))
        baseline = 0.9665
        mean_c = -29.18

    risk = 1.0 - baseline ** math.exp(s - mean_c)
    risk_pct = max(0.0, min(100.0, risk * 100.0))

    if risk_pct < 5.0: cat = "low"
    elif risk_pct < 7.5: cat = "borderline"
    elif risk_pct < 20.0: cat = "intermediate"
    else: cat = "high"

    return {"risk_10yr_pct": round(risk_pct, 1), "risk_category": cat}


# ============================================================================
# 1c. SCORE2 ESC 2021 (Low-Risk Region)
# ============================================================================

def score2_esc(age, sex, tc_mmol, hdl_mmol, sbp, smoker=False):
    male = sex.upper().startswith("M")
    cage = min(max(age, 40), 69)

    if male:
        beta_age, beta_sbp, beta_tc = 0.064, 0.018, 0.12
        beta_hdl, beta_smoke, base = -0.28, 0.63, 0.03
    else:
        beta_age, beta_sbp, beta_tc = 0.072, 0.022, 0.10
        beta_hdl, beta_smoke, base = -0.32, 0.56, 0.015

    lp = (beta_age * (cage - 55) + beta_sbp * (sbp - 130)
          + beta_tc * (tc_mmol - 5.5) + beta_hdl * (hdl_mmol - 1.3)
          + beta_smoke * (1.0 if smoker else 0.0))

    risk_pct = base * 100.0 * math.exp(lp)
    risk_pct = max(0.1, min(50.0, risk_pct))

    if age < 50:
        thresholds = (2.5, 7.5)
    else:
        thresholds = (5.0, 10.0)

    if risk_pct < thresholds[0]: cat = "low"
    elif risk_pct < thresholds[1]: cat = "moderate"
    else: cat = "high"

    return {"risk_10yr_pct": round(risk_pct, 1), "risk_category": cat}


# ============================================================================
# 2a. CKD-EPI 2021 eGFR (Inker 2021)
# ============================================================================

def egfr_ckd_epi(cr, age, sex):
    female = sex.upper().startswith("F")
    if female:
        kappa, alpha = 0.7, -0.241
        coeff = 142 * 1.012
    else:
        kappa, alpha = 0.9, -0.302
        coeff = 142

    cr_k = cr / kappa
    if cr_k <= 1:
        term = cr_k ** alpha
    else:
        term = cr_k ** (-1.200)

    egfr = coeff * term * (0.9938 ** age)
    egfr = round(max(egfr, 0), 1)

    if egfr >= 90: stage = "G1"
    elif egfr >= 60: stage = "G2"
    elif egfr >= 45: stage = "G3a"
    elif egfr >= 30: stage = "G3b"
    elif egfr >= 15: stage = "G4"
    else: stage = "G5"

    return {"egfr": egfr, "ckd_stage": stage}


# ============================================================================
# 2b. FIB-4 Liver Fibrosis (Sterling 2006)
# ============================================================================

def fib4_index(age, ast, alt, platelets):
    sqrt_alt = math.sqrt(max(alt, 0.1))
    denom = max(platelets, 0.1) * sqrt_alt
    fib4 = (age * ast) / denom

    if fib4 < 1.30: risk = "low"
    elif fib4 < 2.67: risk = "indeterminate"
    else: risk = "high"

    return {"fib4": round(fib4, 2), "fibrosis_risk": risk}


# ============================================================================
# 3. HOMA-IR + Metabolic Syndrome ATP-III
# ============================================================================

def homa_ir(glucose, insulin):
    ir = round((glucose * insulin) / 405.0, 2)
    if ir < 1.0: cat = "normal"
    elif ir < 2.5: cat = "borderline"
    else: cat = "resistant"
    return {"homa_ir": ir, "category": cat}


def metabolic_syndrome(waist, sex, tg, hdl, sbp, dbp, glucose,
                       bp_treated=False):
    male = sex.upper().startswith("M")
    criteria = {
        "elevated_waist": waist >= (102 if male else 88),
        "elevated_tg": tg >= 150,
        "low_hdl": hdl < (40 if male else 50),
        "elevated_bp": sbp >= 130 or dbp >= 85 or bp_treated,
        "elevated_glucose": glucose >= 100,
    }
    met = sum(criteria.values())
    return {"criteria_met": met, "is_mets": met >= 3, "detail": criteria}


# ============================================================================
# UNIFIED COMPUTE — called from JavaScript
# ============================================================================

def compute_all(data):
    """Main entry: take a dict of patient data, return complete results.

    Expected keys in data:
      demographics: age, sex, waist_cm
      vitals: systolic_bp, diastolic_bp, bp_treated, smoker
      labs: TC, HDL, LDL, TG, Glucose, HbA1c, Cr, AST, ALT, Albumin,
            Na, K, BUN, WBC, Hb, Plt, TSH
      insulin: (optional) fasting insulin for HOMA-IR
    """
    age = data.get("age", 50)
    sex = data.get("sex", "M")

    labs = {}
    for k in LAB_ITEMS:
        if k in data:
            labs[k] = float(data[k])

    # Gamma-Net
    gamma = lab_to_gamma(labs)
    h = health_index(gamma)
    g2_total = total_gamma_squared(gamma)

    # Sorted organs by |Gamma|
    ranked = sorted(ORGAN_LIST, key=lambda o: abs(gamma.get(o, 0)), reverse=True)
    top3 = [(o, gamma[o]) for o in ranked[:3]]

    # Textbook
    results = {
        "gamma": gamma,
        "health_index": h,
        "total_gamma_squared": g2_total,
        "top3_organs": top3,
        "organ_list": ORGAN_LIST,
    }

    # Framingham
    tc = data.get("TC", 0)
    hdl = data.get("HDL", 0)
    sbp = data.get("systolic_bp", 120)
    smoker = data.get("smoker", False)
    diabetic = data.get("HbA1c", 5.0) >= 6.5
    bp_treated = data.get("bp_treated", False)

    if tc > 0 and hdl > 0:
        results["framingham"] = framingham_risk(
            age, sex, tc, hdl, sbp, smoker, diabetic, bp_treated)

    # ASCVD (age 40-79)
    if tc > 0 and hdl > 0 and age >= 40:
        results["ascvd"] = ascvd_pooled_cohort(
            age, sex, tc, hdl, sbp, bp_treated, smoker, diabetic)

    # SCORE2 (age 40-69)
    if tc > 0 and hdl > 0 and 40 <= age <= 69:
        tc_mmol = tc / 38.67
        hdl_mmol = hdl / 38.67
        results["score2"] = score2_esc(
            age, sex, tc_mmol, hdl_mmol, sbp, smoker)

    # eGFR
    cr = data.get("Cr", 0)
    if cr > 0:
        results["egfr"] = egfr_ckd_epi(cr, age, sex)

    # FIB-4
    ast = data.get("AST", 0)
    alt = data.get("ALT", 0)
    plt = data.get("Plt", 0)
    if ast > 0 and alt > 0 and plt > 0:
        results["fib4"] = fib4_index(age, ast, alt, plt)

    # HOMA-IR
    glu = data.get("Glucose", 0)
    ins = data.get("insulin", 0)
    if glu > 0 and ins > 0:
        results["homa"] = homa_ir(glu, ins)

    # MetS
    waist = data.get("waist_cm", 0)
    tg = data.get("TG", 0)
    dbp = data.get("diastolic_bp", 80)
    if waist > 0 and tg > 0 and hdl > 0 and glu > 0:
        results["mets"] = metabolic_syndrome(
            waist, sex, tg, hdl, sbp, dbp, glu, bp_treated)

    # D_Z projection
    results["dz_daily"] = g2_total
    results["dz_annual"] = g2_total * 365.0
    dz_organs = {}
    for o in ORGAN_LIST:
        g2 = gamma[o] ** 2
        if g2 > 0.0001:
            dz_organs[o] = {"g2": g2, "annual": g2 * 365,
                            "share": g2 / max(g2_total, 1e-12) * 100}
    results["dz_organs"] = dz_organs

    return results

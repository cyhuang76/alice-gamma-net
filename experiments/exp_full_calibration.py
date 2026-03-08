#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: Full 12-Organ Γ Calibration — NHANES Population-Based
═══════════════════════════════════════════════════════════════════

STRATEGY
────────
1. Load ALL available NHANES labs (27 items across 10 cycles, N≈64K)
2. Convert to Alice lab names → compute Z_patient per organ via LabMapper
3. CALIBRATE: Z_normal_organ = median(Z_patient_organ) of 10+ yr survivors
4. Recompute Γ with calibrated Z_normal for ENTIRE cohort
5. Composite scores: H, sum_Γ², organ-weighted CV/metabolic/hepatorenal
6. Head-to-head vs 6 textbook formulas using mortality as ground truth
7. Save calibrated Z_normal for production diagnostic engine

PHYSICS
───────
Calibration = population impedance matching:
    Z_source := median(Z_load) of healthy reference
    → Γ ≈ 0 for survivors  (matched impedance)
    → |Γ| > 0 for sicker-than-average (mismatch detected)

All three constraints preserved:
    C1: Γ² + T = 1  ✓  (algebraic identity)
    C2: ΔZ = population-level Hebbian correction
    C3: ElectricalSignal protocol  ✓
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Alice core
from alice.diagnostics.lab_mapping import (
    LAB_CATALOGUE,
    ORGAN_LIST,
    ORGAN_SYSTEMS,
    LabMapper,
)
from alice.diagnostics.gamma_engine import GammaEngine, PatientGammaVector

# Textbook formulas
from alice.diagnostics.textbook_risk import (
    framingham_risk,
    ascvd_pooled_cohort,
    score2_esc,
    egfr_ckd_epi_2021,
    fib4_index,
    homa_ir,
)

# Reuse STARD infrastructure
from experiments.exp_nhanes_stard_diagnostic import (
    CYCLES,
    COLUMN_REMAP,
    NHANES_TO_ALICE,
    UCOD_TO_ORGAN,
    load_cycle_labs,
    parse_mortality_file,
)

DATA_DIR = PROJECT_ROOT / "nhanes_data"
RESULTS_DIR = PROJECT_ROOT / "nhanes_results"

# Additional NHANES files for textbook formulas (already downloaded)
BPX_FILES = {
    "1999-2000": "BPX", "2001-2002": "BPX_B", "2003-2004": "BPX_C",
    "2005-2006": "BPX_D", "2007-2008": "BPX_E", "2009-2010": "BPX_F",
    "2011-2012": "BPX_G", "2013-2014": "BPX_H", "2015-2016": "BPX_I",
    "2017-2018": "BPXO_J",
}
SMQ_FILES = {
    "1999-2000": "SMQ", "2001-2002": "SMQ_B", "2003-2004": "SMQ_C",
    "2005-2006": "SMQ_D", "2007-2008": "SMQ_E", "2009-2010": "SMQ_F",
    "2011-2012": "SMQ_G", "2013-2014": "SMQ_H", "2015-2016": "SMQ_I",
    "2017-2018": "SMQ_J",
}
BPQ_FILES = {
    "1999-2000": "BPQ", "2001-2002": "BPQ_B", "2003-2004": "BPQ_C",
    "2005-2006": "BPQ_D", "2007-2008": "BPQ_E", "2009-2010": "BPQ_F",
    "2011-2012": "BPQ_G", "2013-2014": "BPQ_H", "2015-2016": "BPQ_I",
    "2017-2018": "BPQ_J",
}
DIQ_FILES = {
    "1999-2000": "DIQ", "2001-2002": "DIQ_B", "2003-2004": "DIQ_C",
    "2005-2006": "DIQ_D", "2007-2008": "DIQ_E", "2009-2010": "DIQ_F",
    "2011-2012": "DIQ_G", "2013-2014": "DIQ_H", "2015-2016": "DIQ_I",
    "2017-2018": "DIQ_J",
}


def load_xpt(path: Path) -> "pd.DataFrame":
    import pandas as pd
    return pd.read_sas(path, format="xport")


# ============================================================================
# Phase A: Load ALL raw lab data + demographics + mortality
# ============================================================================

def load_all_data() -> "pd.DataFrame":
    """Load and merge: labs (raw) + demographics + mortality + BP/SMQ/BPQ/DIQ.

    Returns a single DataFrame with all needed columns.
    """
    import pandas as pd

    # --- Labs ---
    print("  [A1] Loading 10-cycle lab data...")
    all_labs = []
    for cycle_key, cycle_info in CYCLES.items():
        file_paths = {}
        for fk, stem in cycle_info["files"].items():
            if stem is None or fk == "demo":
                continue
            p = DATA_DIR / f"{stem}.XPT"
            if p.exists():
                file_paths[fk] = p
        if file_paths:
            df = load_cycle_labs(cycle_key, file_paths)
            if df is not None and len(df) > 0:
                df["cycle"] = cycle_key
                all_labs.append(df)
    df_labs = pd.concat(all_labs, ignore_index=True)
    df_labs["SEQN"] = df_labs["SEQN"].astype(int)
    print(f"       Lab records: {len(df_labs):,}")

    # --- Demographics ---
    print("  [A2] Loading demographics...")
    demo_stems = {
        "1999-2000": "DEMO", "2001-2002": "DEMO_B", "2003-2004": "DEMO_C",
        "2005-2006": "DEMO_D", "2007-2008": "DEMO_E", "2009-2010": "DEMO_F",
        "2011-2012": "DEMO_G", "2013-2014": "DEMO_H", "2015-2016": "DEMO_I",
        "2017-2018": "DEMO_J",
    }
    demo_parts = []
    for cycle_key, stem in demo_stems.items():
        p = DATA_DIR / f"{stem}.XPT"
        if p.exists():
            d = load_xpt(p)
            d["cycle"] = cycle_key
            demo_parts.append(d[["SEQN", "RIDAGEYR", "RIAGENDR", "cycle"]])
    df_demo = pd.concat(demo_parts, ignore_index=True)
    df_demo["SEQN"] = df_demo["SEQN"].astype(int)

    # --- Mortality ---
    print("  [A3] Loading mortality...")
    mort_parts = []
    for cycle_key in CYCLES:
        s, e = cycle_key.split("-")
        mp = DATA_DIR / f"NHANES_{s}_{e}_MORT_2019_PUBLIC.dat"
        if mp.exists():
            m = parse_mortality_file(mp)
            m["cycle"] = cycle_key
            mort_parts.append(m)
    df_mort = pd.concat(mort_parts, ignore_index=True)
    # Harmonise column names (STARD parser uses MORTSTAT / UCOD_LEADING)
    if "MORTSTAT" in df_mort.columns:
        df_mort.rename(columns={"MORTSTAT": "mort_status",
                                "UCOD_LEADING": "ucod_leading",
                                "PERMTH_INT": "fu_months"}, inplace=True)
    print(f"       Mortality: {len(df_mort):,} eligible, "
          f"{(df_mort['mort_status'] == 1).sum():,} deaths")

    # --- Blood Pressure ---
    print("  [A4] Loading BP / smoking / medication / diabetes...")
    bp_parts = []
    for cycle_key, stem in BPX_FILES.items():
        p = DATA_DIR / f"{stem}.XPT"
        if p.exists():
            d = load_xpt(p)
            d["SEQN"] = d["SEQN"].astype(int)
            sbp_cols = [c for c in d.columns if "SY" in c.upper() and ("BPXO" in c.upper() or "BPXSY" in c.upper())]
            if sbp_cols:
                d["SBP_mean"] = d[sbp_cols].mean(axis=1, skipna=True)
                bp_parts.append(d[["SEQN", "SBP_mean"]].dropna(subset=["SBP_mean"]))
    df_bp = pd.concat(bp_parts, ignore_index=True) if bp_parts else pd.DataFrame(columns=["SEQN", "SBP_mean"])

    # --- Smoking ---
    smq_parts = []
    for cycle_key, stem in SMQ_FILES.items():
        p = DATA_DIR / f"{stem}.XPT"
        if p.exists():
            d = load_xpt(p)
            d["SEQN"] = d["SEQN"].astype(int)
            if "SMQ040" in d.columns:
                d["current_smoker"] = d["SMQ040"].isin([1.0, 2.0]).astype(int)
            elif "SMQ020" in d.columns:
                d["current_smoker"] = (d["SMQ020"] == 1.0).astype(int)
            else:
                continue
            smq_parts.append(d[["SEQN", "current_smoker"]])
    df_smq = pd.concat(smq_parts, ignore_index=True) if smq_parts else pd.DataFrame(columns=["SEQN", "current_smoker"])

    # --- BP Medication ---
    bpq_parts = []
    for cycle_key, stem in BPQ_FILES.items():
        p = DATA_DIR / f"{stem}.XPT"
        if p.exists():
            d = load_xpt(p)
            d["SEQN"] = d["SEQN"].astype(int)
            if "BPQ050A" in d.columns:
                d["bp_treated"] = (d["BPQ050A"] == 1.0).astype(int)
            elif "BPQ040A" in d.columns:
                d["bp_treated"] = (d["BPQ040A"] == 1.0).astype(int)
            else:
                continue
            bpq_parts.append(d[["SEQN", "bp_treated"]])
    df_bpq = pd.concat(bpq_parts, ignore_index=True) if bpq_parts else pd.DataFrame(columns=["SEQN", "bp_treated"])

    # --- Diabetes ---
    diq_parts = []
    for cycle_key, stem in DIQ_FILES.items():
        p = DATA_DIR / f"{stem}.XPT"
        if p.exists():
            d = load_xpt(p)
            d["SEQN"] = d["SEQN"].astype(int)
            if "DIQ010" in d.columns:
                d["diabetic"] = (d["DIQ010"] == 1.0).astype(int)
                diq_parts.append(d[["SEQN", "diabetic"]])
    df_diq = pd.concat(diq_parts, ignore_index=True) if diq_parts else pd.DataFrame(columns=["SEQN", "diabetic"])

    # --- Merge everything ---
    print("  [A5] Merging all data...")
    df = df_labs.merge(df_demo[["SEQN", "RIDAGEYR", "RIAGENDR"]], on="SEQN", how="left")
    df = df.merge(df_mort[["SEQN", "mort_status", "ucod_leading", "fu_months"]], on="SEQN", how="left")
    if len(df_bp) > 0:
        df = df.merge(df_bp.drop_duplicates("SEQN"), on="SEQN", how="left")
    if len(df_smq) > 0:
        df = df.merge(df_smq.drop_duplicates("SEQN"), on="SEQN", how="left")
    if len(df_bpq) > 0:
        df = df.merge(df_bpq.drop_duplicates("SEQN"), on="SEQN", how="left")
    if len(df_diq) > 0:
        df = df.merge(df_diq.drop_duplicates("SEQN"), on="SEQN", how="left")

    # Filter: adults 20+ with mortality linkage
    df = df[df["RIDAGEYR"].notna() & (df["RIDAGEYR"] >= 20)]
    df = df[df["mort_status"].notna()]

    print(f"       Final cohort: {len(df):,}  deaths: {(df['mort_status'] == 1).sum():,}")
    return df


# ============================================================================
# Phase B: Convert NHANES labs → Alice labs → Z_patient per organ
# ============================================================================

def nhanes_to_alice_labs(row: "pd.Series") -> Dict[str, float]:
    """Convert a single NHANES row to Alice lab-name dict.

    Handles deduplication (e.g. LBXSGL & LBXGLU both → Glucose).
    """
    import pandas as pd

    alice_labs: Dict[str, float] = {}
    for nhanes_col, (alice_name, factor) in NHANES_TO_ALICE.items():
        val = row.get(nhanes_col, None)
        if val is not None and pd.notna(val) and np.isfinite(val):
            converted = float(val) * factor
            # If duplicate (e.g. two sources for Glucose), keep non-zero
            if alice_name not in alice_labs or converted > 0:
                alice_labs[alice_name] = converted
    return alice_labs


def compute_z_vectors(df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute Z_patient per organ for every person.

    Adds columns: z_{organ} for each of 12 organs, plus n_alice_labs.
    """
    import pandas as pd

    mapper = LabMapper()  # default textbook Z_normal

    print("  [B] Computing Z_patient for all participants...")
    z_records = []
    valid_indices = []

    for idx, row in df.iterrows():
        alice_labs = nhanes_to_alice_labs(row)
        if len(alice_labs) < 3:
            continue
        z_patient = mapper.compute_organ_impedances(alice_labs)
        z_patient["n_alice_labs"] = len(alice_labs)
        z_records.append(z_patient)
        valid_indices.append(idx)

    z_df = pd.DataFrame(z_records, index=valid_indices)
    print(f"       Computed Z for {len(z_df):,} participants")

    # Add z_ columns to original df
    for organ in ORGAN_LIST:
        if organ in z_df.columns:
            df.loc[z_df.index, f"z_{organ}"] = z_df[organ]
    df.loc[z_df.index, "n_alice_labs"] = z_df["n_alice_labs"]

    df = df[df["n_alice_labs"].notna() & (df["n_alice_labs"] >= 3)]
    return df


# ============================================================================
# Phase C: Calibrate Z_normal from survivor population
# ============================================================================

def calibrate_z_normal(df: "pd.DataFrame", min_followup_months: int = 120) -> Dict[str, float]:
    """Calibrate Z_normal per organ = median Z_patient of long-term survivors.

    Reference population: alive, age 20-79, ≥ min_followup_months follow-up.
    This is the core impedance matching operation.
    """
    # Reference population: survivors with sufficient follow-up
    ref = df[
        (df["mort_status"] == 0) &
        (df["fu_months"] >= min_followup_months) &
        (df["RIDAGEYR"] >= 20) &
        (df["RIDAGEYR"] <= 79)
    ].copy()

    print(f"  [C] Calibrating Z_normal from {len(ref):,} long-term survivors...")
    print(f"       Criteria: alive, age 20-79, follow-up >= {min_followup_months} months")

    z_normal_cal = {}
    for organ in ORGAN_LIST:
        col = f"z_{organ}"
        if col in ref.columns:
            vals = ref[col].dropna()
            if len(vals) > 100:
                z_normal_cal[organ] = float(np.median(vals))
            else:
                z_normal_cal[organ] = ORGAN_SYSTEMS[organ]
                print(f"       WARNING: {organ} only {len(vals)} values, using textbook default")
        else:
            z_normal_cal[organ] = ORGAN_SYSTEMS[organ]

    print("\n       Calibrated Z_normal (textbook → calibrated):")
    for organ in ORGAN_LIST:
        old = ORGAN_SYSTEMS[organ]
        new = z_normal_cal[organ]
        delta_pct = (new - old) / old * 100
        print(f"         {organ:12s}: {old:7.1f} → {new:7.2f}  ({delta_pct:+.1f}%)")

    return z_normal_cal


# ============================================================================
# Phase D: Recompute Γ with calibrated Z_normal
# ============================================================================

def recompute_gamma(df: "pd.DataFrame", z_normal_cal: Dict[str, float]) -> "pd.DataFrame":
    """Recompute Γ per organ using calibrated Z_normal.

    Γ_organ = (Z_patient - Z_normal_cal) / (Z_patient + Z_normal_cal)

    After calibration, Γ can be NEGATIVE (healthier than reference pop).
    """
    import pandas as pd

    print("  [D] Recomputing Γ with calibrated Z_normal...")

    for organ in ORGAN_LIST:
        z_col = f"z_{organ}"
        if z_col not in df.columns:
            df[f"gc_{organ}"] = 0.0
            continue
        z_p = df[z_col].values
        z_n = z_normal_cal[organ]
        denom = z_p + z_n
        gamma = np.where(np.abs(denom) > 1e-12, (z_p - z_n) / denom, 0.0)
        df[f"gc_{organ}"] = gamma

    # Composite scores
    gamma_cols = [f"gc_{o}" for o in ORGAN_LIST]
    gamma_arr = df[gamma_cols].values

    # H_cal = Π(1 - Γ²) — overall health index
    df["H_cal"] = np.prod(1.0 - gamma_arr ** 2, axis=1)

    # sum_Γ² — total mismatch burden
    df["sum_g2_cal"] = np.sum(gamma_arr ** 2, axis=1)

    # Cardiovascular composite: √(Γ_cardiac² + Γ_vascular² + Γ_endocrine²)
    cv_organs = ["cardiac", "vascular", "endocrine"]
    df["gc_CV_composite"] = np.sqrt(np.sum(
        df[[f"gc_{o}" for o in cv_organs]].values ** 2, axis=1
    ))

    # Metabolic composite: √(Γ_endocrine² + Γ_renal² + Γ_hepatic²)
    met_organs = ["endocrine", "renal", "hepatic"]
    df["gc_metabolic"] = np.sqrt(np.sum(
        df[[f"gc_{o}" for o in met_organs]].values ** 2, axis=1
    ))

    # Hepato-renal composite
    hr_organs = ["hepatic", "renal"]
    df["gc_hepatorenal"] = np.sqrt(np.sum(
        df[[f"gc_{o}" for o in hr_organs]].values ** 2, axis=1
    ))

    # Full 12-organ L2 norm
    df["gc_L2_norm"] = np.sqrt(df["sum_g2_cal"])

    print(f"       Γ calibrated for {len(df):,} participants")
    print(f"       H_cal:  mean={df['H_cal'].mean():.4f}  std={df['H_cal'].std():.4f}")
    print(f"       sum_Γ²: mean={df['sum_g2_cal'].mean():.4f}  std={df['sum_g2_cal'].std():.4f}")

    return df


# ============================================================================
# Phase E: Compute textbook risk scores
# ============================================================================

def compute_textbook_scores(df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute 6 textbook risk scores for all participants."""
    import pandas as pd

    print("  [E] Computing textbook risk scores...")

    # --- CKD-EPI ---
    mask = df["LBXSCR"].notna() & df["RIDAGEYR"].notna() & df["RIAGENDR"].notna()
    idx = df.index[mask]
    egfrs = []
    for i in idx:
        r = df.loc[i]
        try:
            res = egfr_ckd_epi_2021(
                creatinine=r["LBXSCR"],
                age=int(r["RIDAGEYR"]),
                sex="M" if r["RIAGENDR"] == 1.0 else "F",
            )
            egfrs.append(res.egfr)
        except Exception:
            egfrs.append(np.nan)
    df.loc[idx, "eGFR"] = egfrs
    df["eGFR_risk"] = df["eGFR"].apply(lambda x: 1.0 / max(x, 1.0) if pd.notna(x) else np.nan)
    print(f"       CKD-EPI: {df['eGFR'].notna().sum():,}")

    # --- FIB-4 ---
    mask = (df["LBXSATSI"].notna() & df["LBXSASSI"].notna() &
            df["LBXPLTSI"].notna() & df["RIDAGEYR"].notna())
    idx = df.index[mask]
    fib4s = []
    for i in idx:
        r = df.loc[i]
        try:
            res = fib4_index(
                age=int(r["RIDAGEYR"]),
                ast=r["LBXSATSI"],
                alt=r["LBXSASSI"],
                platelets=r["LBXPLTSI"],
            )
            fib4s.append(res.fib4)
        except Exception:
            fib4s.append(np.nan)
    df.loc[idx, "FIB4"] = fib4s
    print(f"       FIB-4: {df['FIB4'].notna().sum():,}")

    # --- Framingham / ASCVD / SCORE2 (need SBP) ---
    has_sbp = "SBP_mean" in df.columns
    if has_sbp:
        hdl_col = None
        for c in ["LBDHDD", "HDL_raw"]:
            if c in df.columns:
                hdl_col = c
                break

        mask = (df["RIDAGEYR"].notna() & df["RIAGENDR"].notna() &
                df["LBXSCH"].notna() & df["SBP_mean"].notna())
        if hdl_col:
            mask = mask & df[hdl_col].notna()
        idx = df.index[mask]

        fram_scores, ascvd_scores, score2_scores = [], [], []
        for i in idx:
            r = df.loc[i]
            age = int(r["RIDAGEYR"])
            sex = "M" if r["RIAGENDR"] == 1.0 else "F"
            tc = r["LBXSCH"]
            hdl = r[hdl_col] if hdl_col else 50.0
            sbp = r["SBP_mean"]
            treated = bool(r.get("bp_treated", 0))
            smoker = bool(r.get("current_smoker", 0))
            diab = bool(r.get("diabetic", 0))
            try:
                fram_scores.append(framingham_risk(
                    age=age, sex=sex, total_cholesterol=tc, hdl=hdl,
                    systolic_bp=sbp, bp_treated=treated, smoker=smoker,
                    diabetic=diab).risk_10yr_pct)
            except Exception:
                fram_scores.append(np.nan)
            try:
                ascvd_scores.append(ascvd_pooled_cohort(
                    age=age, sex=sex, total_cholesterol=tc, hdl=hdl,
                    systolic_bp=sbp, bp_treated=treated, smoker=smoker,
                    diabetic=diab).risk_10yr_pct)
            except Exception:
                ascvd_scores.append(np.nan)
            try:
                score2_scores.append(score2_esc(
                    age=age, sex=sex, total_cholesterol_mmol=tc / 38.67,
                    hdl_mmol=hdl / 38.67, systolic_bp=sbp,
                    smoker=smoker).risk_10yr_pct)
            except Exception:
                score2_scores.append(np.nan)

        df.loc[idx, "Framingham_risk"] = fram_scores
        df.loc[idx, "ASCVD_risk"] = ascvd_scores
        df.loc[idx, "SCORE2_risk"] = score2_scores
        print(f"       Framingham/ASCVD/SCORE2: {df['Framingham_risk'].notna().sum():,}")

    return df


# ============================================================================
# Phase F: Head-to-Head AUC Evaluation
# ============================================================================

def safe_auc(y_true, y_score, min_events: int = 10) -> Optional[Dict]:
    """ROC AUC with bootstrap 95% CI. Returns None if insufficient data."""
    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(y_score) & np.isfinite(y_true)
    yt = np.array(y_true)[mask]
    ys = np.array(y_score)[mask]
    if len(yt) < 50 or yt.sum() < min_events or yt.sum() == len(yt):
        return None
    try:
        auc = roc_auc_score(yt, ys)
        rng = np.random.RandomState(42)
        aucs = []
        for _ in range(2000):
            idx = rng.randint(0, len(yt), len(yt))
            if yt[idx].sum() == 0 or yt[idx].sum() == len(idx):
                continue
            aucs.append(roc_auc_score(yt[idx], ys[idx]))
        ci_lo = float(np.percentile(aucs, 2.5)) if aucs else auc
        ci_hi = float(np.percentile(aucs, 97.5)) if aucs else auc
        return {
            "auc": round(float(auc), 4),
            "ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "n": int(mask.sum()),
            "n_events": int(yt.sum()),
        }
    except Exception:
        return None


def evaluate_head_to_head(df: "pd.DataFrame") -> Dict[str, Any]:
    """Full head-to-head evaluation: calibrated Γ vs textbook formulas."""
    import pandas as pd

    print("\n" + "=" * 76)
    print("  HEAD-TO-HEAD: CALIBRATED Γ (12 organs) vs TEXTBOOK FORMULAS")
    print("=" * 76)

    results: Dict[str, Any] = {}

    # Outcome columns
    ucod_map = {}
    for code, val in UCOD_TO_ORGAN.items():
        organ, name = val if isinstance(val, tuple) else (val, "")
        ucod_map[code] = organ
    df["cardiac_death"] = ((df["mort_status"] == 1) & (df["ucod_leading"] == 1)).astype(int)
    df["renal_death"] = ((df["mort_status"] == 1) & (df["ucod_leading"] == 9)).astype(int)
    df["endocrine_death"] = ((df["mort_status"] == 1) & (df["ucod_leading"] == 7)).astype(int)
    df["neuro_death"] = ((df["mort_status"] == 1) & (df["ucod_leading"].isin([5, 6]))).astype(int)
    df["pulmonary_death"] = ((df["mort_status"] == 1) & (df["ucod_leading"].isin([3, 8]))).astype(int)
    df["immune_death"] = ((df["mort_status"] == 1) & (df["ucod_leading"] == 2)).astype(int)
    df["allcause_death"] = (df["mort_status"] == 1).astype(int)

    def _report(label, key, r):
        if r:
            ci = r["ci"]
            print(f"    {label:30s}  AUC = {r['auc']:.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]"
                  f"  n={r['n']:,}  events={r['n_events']:,}")
            results[key] = r
        else:
            print(f"    {label:30s}  (insufficient data)")

    # ── CARDIAC DEATH ──
    print("\n  ── CARDIAC DEATH ──")
    y = df["cardiac_death"].values

    _report("Γ_cardiac (calibrated)",
            "gc_cardiac_vs_cardiac", safe_auc(y, df["gc_cardiac"].abs().values))
    _report("Γ_CV composite",
            "gc_CV_vs_cardiac", safe_auc(y, df["gc_CV_composite"].values))
    _report("1 − H_cal",
            "gc_H_vs_cardiac", safe_auc(y, 1.0 - df["H_cal"].values))
    _report("sum_Γ²",
            "gc_sumg2_vs_cardiac", safe_auc(y, df["sum_g2_cal"].values))
    _report("L2 norm (12-organ)",
            "gc_L2_vs_cardiac", safe_auc(y, df["gc_L2_norm"].values))
    if "Framingham_risk" in df.columns:
        _report("Framingham ATP-III",
                "fram_vs_cardiac", safe_auc(y, df["Framingham_risk"].values))
    if "ASCVD_risk" in df.columns:
        _report("ASCVD PCE",
                "ascvd_vs_cardiac", safe_auc(y, df["ASCVD_risk"].values))
    if "SCORE2_risk" in df.columns:
        _report("SCORE2",
                "score2_vs_cardiac", safe_auc(y, df["SCORE2_risk"].values))

    # ── RENAL DEATH ──
    print("\n  ── RENAL DEATH ──")
    y = df["renal_death"].values

    _report("Γ_renal (calibrated)",
            "gc_renal_vs_renal", safe_auc(y, df["gc_renal"].abs().values))
    _report("Γ_hepatorenal composite",
            "gc_hr_vs_renal", safe_auc(y, df["gc_hepatorenal"].values))
    _report("1 − H_cal",
            "gc_H_vs_renal", safe_auc(y, 1.0 - df["H_cal"].values))
    _report("CKD-EPI 2021 (1/eGFR)",
            "ckdepi_vs_renal", safe_auc(y, df["eGFR_risk"].values))

    # ── ENDOCRINE DEATH ──
    print("\n  ── ENDOCRINE (Diabetes) DEATH ──")
    y = df["endocrine_death"].values

    _report("Γ_endocrine (calibrated)",
            "gc_endo_vs_endo", safe_auc(y, df["gc_endocrine"].abs().values))
    _report("Γ_metabolic composite",
            "gc_met_vs_endo", safe_auc(y, df["gc_metabolic"].values))
    _report("1 − H_cal",
            "gc_H_vs_endo", safe_auc(y, 1.0 - df["H_cal"].values))

    # ── NEUROLOGICAL DEATH ──
    print("\n  ── NEUROLOGICAL DEATH (CVA + Alzheimer) ──")
    y = df["neuro_death"].values

    _report("Γ_neuro (calibrated)",
            "gc_neuro_vs_neuro", safe_auc(y, df["gc_neuro"].abs().values))
    _report("1 − H_cal",
            "gc_H_vs_neuro", safe_auc(y, 1.0 - df["H_cal"].values))

    # ── PULMONARY DEATH ──
    print("\n  ── PULMONARY DEATH ──")
    y = df["pulmonary_death"].values

    _report("Γ_pulmonary (calibrated)",
            "gc_pulmo_vs_pulmo", safe_auc(y, df["gc_pulmonary"].abs().values))
    _report("1 − H_cal",
            "gc_H_vs_pulmo", safe_auc(y, 1.0 - df["H_cal"].values))

    # ── IMMUNE (Neoplasm) DEATH ──
    print("\n  ── IMMUNE (Neoplasm) DEATH ──")
    y = df["immune_death"].values

    _report("Γ_immune (calibrated)",
            "gc_immune_vs_immune", safe_auc(y, df["gc_immune"].abs().values))
    _report("1 − H_cal",
            "gc_H_vs_immune", safe_auc(y, 1.0 - df["H_cal"].values))

    # ── ALL-CAUSE MORTALITY ──
    print("\n  ── ALL-CAUSE MORTALITY ──")
    y = df["allcause_death"].values

    _report("1 − H_cal",
            "gc_H_vs_allcause", safe_auc(y, 1.0 - df["H_cal"].values))
    _report("sum_Γ²",
            "gc_sumg2_vs_allcause", safe_auc(y, df["sum_g2_cal"].values))
    _report("L2 norm (12-organ)",
            "gc_L2_vs_allcause", safe_auc(y, df["gc_L2_norm"].values))
    _report("Γ_CV composite",
            "gc_CV_vs_allcause", safe_auc(y, df["gc_CV_composite"].values))
    if "Framingham_risk" in df.columns:
        _report("Framingham ATP-III",
                "fram_vs_allcause", safe_auc(y, df["Framingham_risk"].values))
    if "ASCVD_risk" in df.columns:
        _report("ASCVD PCE",
                "ascvd_vs_allcause", safe_auc(y, df["ASCVD_risk"].values))
    if "FIB4" in df.columns:
        _report("FIB-4",
                "fib4_vs_allcause", safe_auc(y, df["FIB4"].values))
    if "eGFR_risk" in df.columns:
        _report("CKD-EPI (1/eGFR)",
                "ckdepi_vs_allcause", safe_auc(y, df["eGFR_risk"].values))

    return results


# ============================================================================
# Phase G: Summary & Save
# ============================================================================

def summarise_and_save(results: Dict, z_normal_cal: Dict[str, float],
                       cohort_size: int, n_deaths: int):
    """Print summary table and save all outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("  SUMMARY: CALIBRATED Γ vs TEXTBOOK — BEST COMPARISON PER OUTCOME")
    print("=" * 76)

    # Define head-to-head matchups
    matchups = [
        ("Cardiac death",
         [("gc_CV_vs_cardiac", "Γ_CV composite"),
          ("gc_H_vs_cardiac", "1-H_cal"),
          ("gc_sumg2_vs_cardiac", "sum_Γ²"),
          ("gc_L2_vs_cardiac", "L2 norm")],
         [("fram_vs_cardiac", "Framingham"),
          ("ascvd_vs_cardiac", "ASCVD"),
          ("score2_vs_cardiac", "SCORE2")]),
        ("Renal death",
         [("gc_renal_vs_renal", "Γ_renal"),
          ("gc_H_vs_renal", "1-H_cal")],
         [("ckdepi_vs_renal", "CKD-EPI")]),
        ("Endocrine death",
         [("gc_endo_vs_endo", "Γ_endocrine"),
          ("gc_met_vs_endo", "Γ_metabolic"),
          ("gc_H_vs_endo", "1-H_cal")],
         []),
        ("All-cause",
         [("gc_H_vs_allcause", "1-H_cal"),
          ("gc_sumg2_vs_allcause", "sum_Γ²"),
          ("gc_L2_vs_allcause", "L2 norm"),
          ("gc_CV_vs_allcause", "Γ_CV composite")],
         [("fram_vs_allcause", "Framingham"),
          ("ascvd_vs_allcause", "ASCVD"),
          ("fib4_vs_allcause", "FIB-4"),
          ("ckdepi_vs_allcause", "CKD-EPI")]),
    ]

    summary_lines = []
    for outcome, gamma_candidates, textbook_candidates in matchups:
        # Best Γ
        best_g_auc, best_g_name = 0.0, ""
        for key, name in gamma_candidates:
            auc = results.get(key, {}).get("auc", 0.0)
            if auc > best_g_auc:
                best_g_auc = auc
                best_g_name = name
        # Best textbook
        best_t_auc, best_t_name = 0.0, ""
        for key, name in textbook_candidates:
            auc = results.get(key, {}).get("auc", 0.0)
            if auc > best_t_auc:
                best_t_auc = auc
                best_t_name = name

        delta = best_g_auc - best_t_auc if best_t_auc > 0 else 0.0
        if best_t_auc > 0:
            winner = f"Γ ({best_g_name})" if delta >= 0 else best_t_name
            line = (f"  {outcome:20s}  Γ best: {best_g_name:18s} = {best_g_auc:.4f}"
                    f"  |  Textbook best: {best_t_name:12s} = {best_t_auc:.4f}"
                    f"  →  {winner}  Δ={delta:+.4f}")
        else:
            line = (f"  {outcome:20s}  Γ best: {best_g_name:18s} = {best_g_auc:.4f}"
                    f"  |  (no textbook)")
        print(line)
        summary_lines.append(line)

    # Save calibrated Z_normal
    cal_path = RESULTS_DIR / "calibrated_z_normal.json"
    with open(cal_path, "w") as f:
        json.dump({
            "description": "Population-calibrated Z_normal from NHANES 10-year survivors",
            "source": "NHANES 1999-2018, NDI 2019, survivors with >=120 months follow-up, age 20-79",
            "textbook_z_normal": {k: float(v) for k, v in ORGAN_SYSTEMS.items()},
            "calibrated_z_normal": {k: round(v, 4) for k, v in z_normal_cal.items()},
        }, f, indent=2)
    print(f"\n  Saved calibrated Z_normal: {cal_path}")

    # Save full results
    output = {
        "protocol": "Full 12-Organ Γ Calibration v1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_size": cohort_size,
        "n_deaths": n_deaths,
        "calibrated_z_normal": {k: round(v, 4) for k, v in z_normal_cal.items()},
        "results": results,
        "summary": summary_lines,
    }
    out_json = RESULTS_DIR / "full_calibration_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved results: {out_json}")

    # Text report
    report_path = RESULTS_DIR / "full_calibration_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 76 + "\n")
        f.write("  FULL 12-ORGAN Γ CALIBRATION — NHANES POPULATION-BASED\n")
        f.write(f"  N={cohort_size:,}  Deaths={n_deaths:,}  NHANES 1999-2018 × NDI 2019\n")
        f.write("=" * 76 + "\n\n")
        f.write("CALIBRATION METHOD\n")
        f.write("  Z_normal_organ = median(Z_patient_organ) of 10+ year survivors\n")
        f.write("  Reference population: alive, age 20-79, follow-up >= 120 months\n\n")
        f.write("CALIBRATED Z_normal\n")
        for organ in ORGAN_LIST:
            old = ORGAN_SYSTEMS[organ]
            new = z_normal_cal[organ]
            f.write(f"  {organ:12s}: {old:7.1f} → {new:7.2f}\n")
        f.write("\nRESULTS\n")
        for key, val in sorted(results.items()):
            f.write(f"  {key}: AUC={val['auc']:.4f} {val['ci']} "
                    f"n={val['n']} events={val['n_events']}\n")
        f.write("\nSUMMARY\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"  Saved report: {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    import pandas as pd

    print("=" * 76)
    print("  FULL 12-ORGAN Γ CALIBRATION — NHANES POPULATION-BASED")
    print("  All 12 organs × 27 lab items × 10 cycles × NDI mortality")
    print("=" * 76)
    ts = datetime.now(timezone.utc).isoformat()
    print(f"  Timestamp: {ts}\n")

    # Phase A: Load all data
    print("━" * 76)
    print("  PHASE A: Load data")
    print("━" * 76)
    df = load_all_data()

    # Phase B: Compute Z_patient per organ
    print("\n" + "━" * 76)
    print("  PHASE B: Compute Z_patient (organ impedance)")
    print("━" * 76)
    df = compute_z_vectors(df)

    # Phase C: Calibrate Z_normal from survivors
    print("\n" + "━" * 76)
    print("  PHASE C: Calibrate Z_normal from survivor population")
    print("━" * 76)
    z_normal_cal = calibrate_z_normal(df)

    # Phase D: Recompute Γ with calibrated Z
    print("\n" + "━" * 76)
    print("  PHASE D: Recompute Γ with calibrated Z_normal")
    print("━" * 76)
    df = recompute_gamma(df, z_normal_cal)

    # Phase E: Textbook scores
    print("\n" + "━" * 76)
    print("  PHASE E: Compute textbook risk scores")
    print("━" * 76)
    df = compute_textbook_scores(df)

    # Phase F: Head-to-head
    print("\n" + "━" * 76)
    print("  PHASE F: Head-to-head AUC evaluation")
    print("━" * 76)
    results = evaluate_head_to_head(df)

    # Phase G: Summary & save
    cohort_size = len(df)
    n_deaths = int(df["allcause_death"].sum())
    summarise_and_save(results, z_normal_cal, cohort_size, n_deaths)

    print("\n" + "=" * 76)
    print("  DONE — Full 12-organ calibration complete")
    print("=" * 76)


if __name__ == "__main__":
    main()

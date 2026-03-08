#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: Γ-Score vs Textbook Risk Formulas — Head-to-Head Comparison
═══════════════════════════════════════════════════════════════════════

PURPOSE
───────
Compare the discriminative ability (AUC) of the zero-parameter Γ_organ
scores against 6 established clinical risk formulas on the SAME NHANES
cohort (1999–2018, 10 cycles) with NDI mortality linkage.

HEAD-TO-HEAD MATCHUPS
─────────────────────
  Outcome                 Γ competitor     Textbook formulas
  ──────────────────────  ──────────────   ─────────────────────
  Cardiac death           Γ_cardiac        Framingham, ASCVD PCE, SCORE2
  Renal death             Γ_renal          CKD-EPI 2021 (inverted eGFR)
  Liver-related death     Γ_hepatic        FIB-4
  Endocrine death         Γ_endocrine      HOMA-IR (where insulin available)

ADDITIONAL NHANES FILES
───────────────────────
  BPX: Systolic/diastolic blood pressure (examination)
  SMQ: Smoking questionnaire
  BPQ: Blood pressure & cholesterol questionnaire (medication use)

ZERO PARAMETERS FITTED — Γ uses the same pre-existing Z_normal values.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
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

from alice.diagnostics.textbook_risk import (
    framingham_risk,
    ascvd_pooled_cohort,
    score2_esc,
    egfr_ckd_epi_2021,
    fib4_index,
    homa_ir,
)

DATA_DIR = PROJECT_ROOT / "nhanes_data"
RESULTS_DIR = PROJECT_ROOT / "nhanes_results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ============================================================================
# Additional NHANES files needed for textbook formulas
# ============================================================================

# Blood pressure examination files by cycle
BPX_FILES = {
    "1999-2000": ("BPX",     "1999"),
    "2001-2002": ("BPX_B",   "2001"),
    "2003-2004": ("BPX_C",   "2003"),
    "2005-2006": ("BPX_D",   "2005"),
    "2007-2008": ("BPX_E",   "2007"),
    "2009-2010": ("BPX_F",   "2009"),
    "2011-2012": ("BPX_G",   "2011"),
    "2013-2014": ("BPX_H",   "2013"),
    "2015-2016": ("BPX_I",   "2015"),
    "2017-2018": ("BPXO_J",  "2017"),  # 2017-2018 uses oscillometric (BPXO)
}

# Smoking questionnaire files by cycle
SMQ_FILES = {
    "1999-2000": ("SMQ",     "1999"),
    "2001-2002": ("SMQ_B",   "2001"),
    "2003-2004": ("SMQ_C",   "2003"),
    "2005-2006": ("SMQ_D",   "2005"),
    "2007-2008": ("SMQ_E",   "2007"),
    "2009-2010": ("SMQ_F",   "2009"),
    "2011-2012": ("SMQ_G",   "2011"),
    "2013-2014": ("SMQ_H",   "2013"),
    "2015-2016": ("SMQ_I",   "2015"),
    "2017-2018": ("SMQ_J",   "2017"),
}

# Blood pressure / cholesterol questionnaire (medication use)
BPQ_FILES = {
    "1999-2000": ("BPQ",     "1999"),
    "2001-2002": ("BPQ_B",   "2001"),
    "2003-2004": ("BPQ_C",   "2003"),
    "2005-2006": ("BPQ_D",   "2005"),
    "2007-2008": ("BPQ_E",   "2007"),
    "2009-2010": ("BPQ_F",   "2009"),
    "2011-2012": ("BPQ_G",   "2011"),
    "2013-2014": ("BPQ_H",   "2013"),
    "2015-2016": ("BPQ_I",   "2015"),
    "2017-2018": ("BPQ_J",   "2017"),
}

# Diabetes questionnaire (for DIQ010 = doctor told you have diabetes)
DIQ_FILES = {
    "1999-2000": ("DIQ",     "1999"),
    "2001-2002": ("DIQ_B",   "2001"),
    "2003-2004": ("DIQ_C",   "2003"),
    "2005-2006": ("DIQ_D",   "2005"),
    "2007-2008": ("DIQ_E",   "2007"),
    "2009-2010": ("DIQ_F",   "2009"),
    "2011-2012": ("DIQ_G",   "2011"),
    "2013-2014": ("DIQ_H",   "2013"),
    "2015-2016": ("DIQ_I",   "2015"),
    "2017-2018": ("DIQ_J",   "2017"),
}

# Insulin lab files (for HOMA-IR) — available from 2005+
INS_FILES = {
    "2005-2006": ("INS_D",   "2005"),
    "2007-2008": ("INS_E",   "2007"),
    "2009-2010": ("INS_F",   "2009"),
    "2011-2012": ("INS_G",   "2011"),
    "2013-2014": ("INS_H",   "2013"),
    "2015-2016": ("INS_I",   "2015"),
    "2017-2018": ("INS_J",   "2017"),
}

# UCOD to organ mapping (same as stard script)
UCOD_TO_ORGAN = {
    1:  "cardiac",     # Heart disease
    2:  "immune",      # Malignant neoplasms
    3:  "pulmonary",   # CLRD
    5:  "neuro",       # Cerebrovascular
    6:  "neuro",       # Alzheimer's
    7:  "endocrine",   # Diabetes
    8:  "pulmonary",   # Influenza/pneumonia
    9:  "renal",       # Nephritis
}


# ============================================================================
# Download helpers
# ============================================================================

def download_xpt(stem: str, year: str) -> Optional[Path]:
    """Download an NHANES XPT file if not cached."""
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local = DATA_DIR / f"{stem}.XPT"

    if local.exists() and local.stat().st_size > 1000:
        return local

    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{stem}.XPT"
    headers = {"User-Agent": "Mozilla/5.0 (Alice-Gamma-Net/3.5; research)"}

    try:
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        if len(resp.content) < 100:
            return None
        local.write_bytes(resp.content)
        print(f"    Downloaded {stem}.XPT: {len(resp.content):,} bytes")
        return local
    except Exception as e:
        print(f"    Failed {stem}: {e}")
        return None


def load_xpt(path: Path) -> "pd.DataFrame":
    """Load SAS transport file."""
    import pandas as pd
    return pd.read_sas(path, format="xport")


# ============================================================================
# Mortality file parser (fixed-width)
# ============================================================================

def parse_mortality_file(path: Path) -> "pd.DataFrame":
    """Parse NHANES NDI linked mortality .dat file.

    Fixed-width format:
      col 1-14:   SEQN (right-justified, sometimes called PUBLICID)
      col 15:     eligibility (0/1)
      col 16:     mortality status (0=alive, 1=dead)
      col 17-19:  UCOD_LEADING (underlying cause of death, leading category)
      col 20-22:  follow-up months (integer)
      col 23-26:  follow-up months decimal portion
    """
    import pandas as pd

    records = []
    with open(path, "r") as f:
        for line in f:
            if len(line.strip()) < 15:
                continue
            seqn_str = line[0:14].strip()
            if not seqn_str.isdigit():
                continue
            seqn = int(seqn_str)
            eligible = int(line[14]) if len(line) > 14 and line[14].isdigit() else 0
            mort_stat = int(line[15]) if len(line) > 15 and line[15].isdigit() else 0
            ucod_str = line[16:19].strip() if len(line) > 18 else ""
            ucod = int(ucod_str) if ucod_str.isdigit() else 0
            fu_str = line[19:26].strip() if len(line) > 25 else ""
            try:
                fu_months = float(fu_str) if fu_str else 0.0
            except ValueError:
                fu_months = 0.0

            if eligible == 1:
                records.append({
                    "SEQN": seqn,
                    "mort_status": mort_stat,
                    "ucod_leading": ucod,
                    "fu_months": fu_months,
                })

    return pd.DataFrame(records)


# ============================================================================
# Main pipeline
# ============================================================================

def run_comparison():
    """Run Γ vs Textbook head-to-head comparison."""
    import pandas as pd
    from scipy import stats as sp_stats
    from sklearn.metrics import roc_auc_score

    print("=" * 72)
    print("  Γ-SCORE vs TEXTBOOK RISK FORMULAS — HEAD-TO-HEAD")
    print("  NHANES 1999–2018 (10 cycles) × NDI 2019")
    print("=" * 72)
    ts = datetime.now(timezone.utc).isoformat()
    print(f"  Timestamp: {ts}")

    # ------------------------------------------------------------------
    # Step 1: Load existing Γ vectors
    # ------------------------------------------------------------------
    print("\n[1] Loading pre-computed Γ vectors...")
    gamma_path = RESULTS_DIR / "nhanes_10cycle_gamma_vectors.csv"
    if not gamma_path.exists():
        print("  ERROR: Run exp_nhanes_stard_diagnostic.py first!")
        return
    df_gamma = pd.read_csv(gamma_path)
    print(f"  Γ vectors: {len(df_gamma):,} rows")

    # ------------------------------------------------------------------
    # Step 2: Download additional NHANES files (BPX, SMQ, BPQ, DIQ, INS)
    # ------------------------------------------------------------------
    print("\n[2] Downloading additional NHANES files...")

    all_bp = []
    all_smq = []
    all_bpq = []
    all_diq = []
    all_ins = []

    for cycle_key in BPX_FILES:
        stem, year = BPX_FILES[cycle_key]
        path = download_xpt(stem, year)
        if path:
            df = load_xpt(path)
            df["cycle"] = cycle_key
            all_bp.append(df)

    for cycle_key in SMQ_FILES:
        stem, year = SMQ_FILES[cycle_key]
        path = download_xpt(stem, year)
        if path:
            df = load_xpt(path)
            df["cycle"] = cycle_key
            all_smq.append(df)

    for cycle_key in BPQ_FILES:
        stem, year = BPQ_FILES[cycle_key]
        path = download_xpt(stem, year)
        if path:
            df = load_xpt(path)
            df["cycle"] = cycle_key
            all_bpq.append(df)

    for cycle_key in DIQ_FILES:
        stem, year = DIQ_FILES[cycle_key]
        path = download_xpt(stem, year)
        if path:
            df = load_xpt(path)
            df["cycle"] = cycle_key
            all_diq.append(df)

    for cycle_key in INS_FILES:
        stem, year = INS_FILES[cycle_key]
        path = download_xpt(stem, year)
        if path:
            df = load_xpt(path)
            df["cycle"] = cycle_key
            all_ins.append(df)

    # ------------------------------------------------------------------
    # Step 3: Load demographics for all cycles
    # ------------------------------------------------------------------
    print("\n[3] Loading demographics...")
    demo_files = {
        "1999-2000": "DEMO", "2001-2002": "DEMO_B", "2003-2004": "DEMO_C",
        "2005-2006": "DEMO_D", "2007-2008": "DEMO_E", "2009-2010": "DEMO_F",
        "2011-2012": "DEMO_G", "2013-2014": "DEMO_H", "2015-2016": "DEMO_I",
        "2017-2018": "DEMO_J",
    }
    all_demo = []
    for cycle_key, stem in demo_files.items():
        path = DATA_DIR / f"{stem}.XPT"
        if path.exists():
            df = load_xpt(path)
            df["cycle"] = cycle_key
            all_demo.append(df[["SEQN", "RIDAGEYR", "RIAGENDR", "cycle"]])
    df_demo = pd.concat(all_demo, ignore_index=True)
    df_demo["SEQN"] = df_demo["SEQN"].astype(int)
    print(f"  Demographics: {len(df_demo):,} participants")

    # ------------------------------------------------------------------
    # Step 4: Load mortality
    # ------------------------------------------------------------------
    print("\n[4] Loading mortality data...")
    mort_frames = []
    for cycle_key in BPX_FILES:
        start, end = cycle_key.split("-")
        mort_file = DATA_DIR / f"NHANES_{start}_{end}_MORT_2019_PUBLIC.dat"
        if mort_file.exists():
            mdf = parse_mortality_file(mort_file)
            mdf["cycle"] = cycle_key
            mort_frames.append(mdf)
    df_mort = pd.concat(mort_frames, ignore_index=True)
    print(f"  Mortality records: {len(df_mort):,}")
    print(f"  Deaths: {(df_mort['mort_status'] == 1).sum():,}")

    # ------------------------------------------------------------------
    # Step 5: Load labs (BIOPRO, CBC, HDL, GLU) for textbook formulas
    # ------------------------------------------------------------------
    print("\n[5] Loading lab files for textbook formulas...")
    from experiments.exp_nhanes_stard_diagnostic import (
        CYCLES, COLUMN_REMAP, NHANES_TO_ALICE,
        load_cycle_labs,
    )

    all_labs = []
    for cycle_key, cycle_info in CYCLES.items():
        year = cycle_info["year"]
        file_paths = {}
        for fk, stem in cycle_info["files"].items():
            if stem is None or fk == "demo":
                continue
            p = DATA_DIR / f"{stem}.XPT"
            if p.exists():
                file_paths[fk] = p
        if file_paths:
            df_labs = load_cycle_labs(cycle_key, file_paths)
            if df_labs is not None and len(df_labs) > 0:
                df_labs["cycle"] = cycle_key
                all_labs.append(df_labs)

    df_all_labs = pd.concat(all_labs, ignore_index=True)
    df_all_labs["SEQN"] = df_all_labs["SEQN"].astype(int)
    print(f"  Lab records: {len(df_all_labs):,}")

    # ------------------------------------------------------------------
    # Step 6: Process BP data
    # ------------------------------------------------------------------
    print("\n[6] Processing blood pressure data...")
    bp_frames = []
    for df_bp in all_bp:
        df_bp["SEQN"] = df_bp["SEQN"].astype(int)
        # Different column names across cycles
        sbp_cols = [c for c in df_bp.columns if "SY" in c.upper() and "BPXO" in c.upper()]
        if not sbp_cols:
            sbp_cols = [c for c in df_bp.columns if c.startswith("BPXSY")]
        if not sbp_cols:
            sbp_cols = [c for c in df_bp.columns if "SY" in c.upper() and "BPX" in c.upper()]

        if sbp_cols:
            # Average available SBP readings
            df_bp["SBP_mean"] = df_bp[sbp_cols].mean(axis=1, skipna=True)
            bp_frames.append(df_bp[["SEQN", "SBP_mean", "cycle"]].dropna(subset=["SBP_mean"]))

    if bp_frames:
        df_sbp = pd.concat(bp_frames, ignore_index=True)
        print(f"  SBP records: {len(df_sbp):,}")
    else:
        df_sbp = pd.DataFrame(columns=["SEQN", "SBP_mean", "cycle"])
        print("  WARNING: No SBP data found!")

    # ------------------------------------------------------------------
    # Step 7: Process smoking status
    # ------------------------------------------------------------------
    print("\n[7] Processing smoking status...")
    smq_frames = []
    for df_s in all_smq:
        df_s["SEQN"] = df_s["SEQN"].astype(int)
        # SMQ020 = "Have you smoked at least 100 cigarettes?"
        # SMQ040 = "Do you now smoke?" (1=everyday, 2=some days, 3=not at all)
        if "SMQ040" in df_s.columns:
            df_s["current_smoker"] = df_s["SMQ040"].isin([1.0, 2.0]).astype(int)
        elif "SMQ020" in df_s.columns:
            df_s["current_smoker"] = (df_s["SMQ020"] == 1.0).astype(int)
        else:
            continue
        smq_frames.append(df_s[["SEQN", "current_smoker", "cycle"]])

    if smq_frames:
        df_smoke = pd.concat(smq_frames, ignore_index=True)
        print(f"  Smoking records: {len(df_smoke):,}")
    else:
        df_smoke = pd.DataFrame(columns=["SEQN", "current_smoker", "cycle"])

    # ------------------------------------------------------------------
    # Step 8: Process BP medication
    # ------------------------------------------------------------------
    print("\n[8] Processing BP medication status...")
    bpq_frames = []
    for df_q in all_bpq:
        df_q["SEQN"] = df_q["SEQN"].astype(int)
        # BPQ050A = "Now taking prescribed medicine for HBP?"
        if "BPQ050A" in df_q.columns:
            df_q["bp_treated"] = (df_q["BPQ050A"] == 1.0).astype(int)
        elif "BPQ040A" in df_q.columns:
            df_q["bp_treated"] = (df_q["BPQ040A"] == 1.0).astype(int)
        else:
            continue
        bpq_frames.append(df_q[["SEQN", "bp_treated", "cycle"]])

    if bpq_frames:
        df_bpmeds = pd.concat(bpq_frames, ignore_index=True)
        print(f"  BP medication records: {len(df_bpmeds):,}")
    else:
        df_bpmeds = pd.DataFrame(columns=["SEQN", "bp_treated", "cycle"])

    # ------------------------------------------------------------------
    # Step 9: Process diabetes status
    # ------------------------------------------------------------------
    print("\n[9] Processing diabetes status...")
    diq_frames = []
    for df_d in all_diq:
        df_d["SEQN"] = df_d["SEQN"].astype(int)
        # DIQ010 = "Doctor told you have diabetes?"
        if "DIQ010" in df_d.columns:
            df_d["diabetic"] = (df_d["DIQ010"] == 1.0).astype(int)
        else:
            continue
        diq_frames.append(df_d[["SEQN", "diabetic", "cycle"]])

    if diq_frames:
        df_diabetes = pd.concat(diq_frames, ignore_index=True)
        print(f"  Diabetes records: {len(df_diabetes):,}")
    else:
        df_diabetes = pd.DataFrame(columns=["SEQN", "diabetic", "cycle"])

    # ------------------------------------------------------------------
    # Step 10: Process insulin (for HOMA-IR)
    # ------------------------------------------------------------------
    print("\n[10] Processing insulin data...")
    ins_frames = []
    for df_i in all_ins:
        df_i["SEQN"] = df_i["SEQN"].astype(int)
        # LBXIN or LBDINSI = fasting insulin
        ins_col = None
        for c in ["LBXIN", "LBDINSI", "LBXINSI"]:
            if c in df_i.columns:
                ins_col = c
                break
        if ins_col:
            df_i["fasting_insulin"] = df_i[ins_col]
            ins_frames.append(df_i[["SEQN", "fasting_insulin", "cycle"]].dropna(subset=["fasting_insulin"]))

    if ins_frames:
        df_insulin = pd.concat(ins_frames, ignore_index=True)
        print(f"  Insulin records: {len(df_insulin):,}")
    else:
        df_insulin = pd.DataFrame(columns=["SEQN", "fasting_insulin", "cycle"])

    # ------------------------------------------------------------------
    # Step 11: Merge everything
    # ------------------------------------------------------------------
    print("\n[11] Merging all data...")

    # Start from Γ vectors
    df = df_gamma.copy()
    df["SEQN"] = df["SEQN"].astype(int)

    # Merge demographics
    df = df.merge(df_demo, on=["SEQN", "cycle"], how="left")

    # Merge mortality
    df = df.merge(df_mort[["SEQN", "mort_status", "ucod_leading", "fu_months"]],
                  on="SEQN", how="left")

    # Merge SBP
    if len(df_sbp) > 0:
        df = df.merge(df_sbp[["SEQN", "SBP_mean"]], on="SEQN", how="left")

    # Merge smoking
    if len(df_smoke) > 0:
        df = df.merge(df_smoke[["SEQN", "current_smoker"]], on="SEQN", how="left")

    # Merge BP meds
    if len(df_bpmeds) > 0:
        df = df.merge(df_bpmeds[["SEQN", "bp_treated"]], on="SEQN", how="left")

    # Merge diabetes
    if len(df_diabetes) > 0:
        df = df.merge(df_diabetes[["SEQN", "diabetic"]], on="SEQN", how="left")

    # Merge insulin
    if len(df_insulin) > 0:
        df = df.merge(df_insulin[["SEQN", "fasting_insulin"]], on="SEQN", how="left")

    # Merge raw labs needed for textbook formulas
    lab_cols_needed = ["SEQN", "LBXSCR", "LBXSATSI", "LBXSASSI", "LBXPLTSI",
                       "LBXSCH", "LBXSGL"]
    lab_cols_available = [c for c in lab_cols_needed if c in df_all_labs.columns]
    if lab_cols_available:
        df = df.merge(df_all_labs[lab_cols_available].drop_duplicates("SEQN"),
                      on="SEQN", how="left")

    # Merge HDL from the gamma vectors CSV if needed
    # HDL may have been mapped to a different name; check labs
    hdl_col = None
    for c in ["LBDHDD", "LBXHDD"]:
        if c in df_all_labs.columns:
            hdl_col = c
            break
    if hdl_col and hdl_col not in df.columns:
        hdl_data = df_all_labs[["SEQN", hdl_col]].drop_duplicates("SEQN")
        df = df.merge(hdl_data, on="SEQN", how="left")
        df.rename(columns={hdl_col: "HDL_raw"}, inplace=True)
    elif hdl_col and hdl_col in df.columns:
        df.rename(columns={hdl_col: "HDL_raw"}, inplace=True)

    # Filter: adults with mortality linkage
    df = df[df["RIDAGEYR"].notna() & (df["RIDAGEYR"] >= 20)]
    df = df[df["mort_status"].notna()]

    print(f"  Final merged cohort: {len(df):,}")
    print(f"  Deaths: {(df['mort_status'] == 1).sum():,}")

    # Create organ-specific death columns
    df["cardiac_death"] = ((df["mort_status"] == 1) & (df["ucod_leading"] == 1)).astype(int)
    df["renal_death"] = ((df["mort_status"] == 1) & (df["ucod_leading"] == 9)).astype(int)
    df["endocrine_death"] = ((df["mort_status"] == 1) & (df["ucod_leading"] == 7)).astype(int)
    df["allcause_death"] = (df["mort_status"] == 1).astype(int)

    # ------------------------------------------------------------------
    # Step 12: Compute textbook scores
    # ------------------------------------------------------------------
    print("\n[12] Computing textbook risk scores...")

    # --- CKD-EPI (needs: creatinine, age, sex) ---
    mask_ckd = (df["LBXSCR"].notna() & df["RIDAGEYR"].notna() & df["RIAGENDR"].notna())
    df.loc[mask_ckd, "eGFR"] = df.loc[mask_ckd].apply(
        lambda r: egfr_ckd_epi_2021(
            creatinine=r["LBXSCR"],
            age=int(r["RIDAGEYR"]),
            sex="M" if r["RIAGENDR"] == 1.0 else "F",
        ).egfr, axis=1
    )
    # Invert: lower eGFR = sicker = higher risk score
    df["eGFR_risk"] = df["eGFR"].apply(lambda x: 1.0 / max(x, 1.0) if pd.notna(x) else np.nan)
    n_ckd = df["eGFR"].notna().sum()
    print(f"  CKD-EPI computed: {n_ckd:,}")

    # --- FIB-4 (needs: age, AST, ALT, platelets) ---
    mask_fib = (df["LBXSATSI"].notna() & df["LBXSASSI"].notna() &
                df["LBXPLTSI"].notna() & df["RIDAGEYR"].notna())
    df.loc[mask_fib, "FIB4"] = df.loc[mask_fib].apply(
        lambda r: fib4_index(
            age=int(r["RIDAGEYR"]),
            ast=r["LBXSATSI"],
            alt=r["LBXSASSI"],
            platelets=r["LBXPLTSI"],
        ).fib4, axis=1
    )
    n_fib = df["FIB4"].notna().sum()
    print(f"  FIB-4 computed: {n_fib:,}")

    # --- Framingham (needs: age, sex, TC, HDL, SBP, bp_treated, smoker, diabetic) ---
    mask_fram = (df["RIDAGEYR"].notna() & df["RIAGENDR"].notna() &
                 df["LBXSCH"].notna())
    if "HDL_raw" in df.columns:
        mask_fram = mask_fram & df["HDL_raw"].notna()
    if "SBP_mean" in df.columns:
        mask_fram = mask_fram & df["SBP_mean"].notna()
    else:
        mask_fram = pd.Series(False, index=df.index)

    if mask_fram.sum() > 0:
        df.loc[mask_fram, "Framingham_risk"] = df.loc[mask_fram].apply(
            lambda r: framingham_risk(
                age=int(r["RIDAGEYR"]),
                sex="M" if r["RIAGENDR"] == 1.0 else "F",
                total_cholesterol=r["LBXSCH"],
                hdl=r.get("HDL_raw", 50.0),
                systolic_bp=r["SBP_mean"],
                bp_treated=bool(r.get("bp_treated", 0)),
                smoker=bool(r.get("current_smoker", 0)),
                diabetic=bool(r.get("diabetic", 0)),
            ).risk_10yr_pct, axis=1
        )
        n_fram = df["Framingham_risk"].notna().sum()
        print(f"  Framingham computed: {n_fram:,}")

    # --- ASCVD PCE ---
    if mask_fram.sum() > 0:
        df.loc[mask_fram, "ASCVD_risk"] = df.loc[mask_fram].apply(
            lambda r: ascvd_pooled_cohort(
                age=int(r["RIDAGEYR"]),
                sex="M" if r["RIAGENDR"] == 1.0 else "F",
                total_cholesterol=r["LBXSCH"],
                hdl=r.get("HDL_raw", 50.0),
                systolic_bp=r["SBP_mean"],
                bp_treated=bool(r.get("bp_treated", 0)),
                smoker=bool(r.get("current_smoker", 0)),
                diabetic=bool(r.get("diabetic", 0)),
            ).risk_10yr_pct, axis=1
        )
        n_ascvd = df["ASCVD_risk"].notna().sum()
        print(f"  ASCVD PCE computed: {n_ascvd:,}")

    # --- SCORE2 ---
    if mask_fram.sum() > 0:
        df.loc[mask_fram, "SCORE2_risk"] = df.loc[mask_fram].apply(
            lambda r: score2_esc(
                age=int(r["RIDAGEYR"]),
                sex="M" if r["RIAGENDR"] == 1.0 else "F",
                total_cholesterol_mmol=r["LBXSCH"] / 38.67,
                hdl_mmol=r.get("HDL_raw", 50.0) / 38.67,
                systolic_bp=r["SBP_mean"],
                smoker=bool(r.get("current_smoker", 0)),
            ).risk_10yr_pct, axis=1
        )
        n_score2 = df["SCORE2_risk"].notna().sum()
        print(f"  SCORE2 computed: {n_score2:,}")

    # --- HOMA-IR ---
    if "fasting_insulin" in df.columns:
        mask_homa = (df["LBXSGL"].notna() & df["fasting_insulin"].notna() &
                     (df["fasting_insulin"] > 0))
        if mask_homa.sum() > 0:
            df.loc[mask_homa, "HOMA_IR"] = df.loc[mask_homa].apply(
                lambda r: homa_ir(
                    fasting_glucose_mg_dl=r["LBXSGL"],
                    fasting_insulin_uU_ml=r["fasting_insulin"],
                ).homa_ir, axis=1
            )
            n_homa = df["HOMA_IR"].notna().sum()
            print(f"  HOMA-IR computed: {n_homa:,}")

    # ------------------------------------------------------------------
    # Step 13: Head-to-head AUC comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  HEAD-TO-HEAD AUC COMPARISON: Γ_organ vs Textbook")
    print("=" * 72)

    results = {}

    def safe_auc(y_true, y_score, label: str) -> Optional[float]:
        """Compute AUC with bootstrap 95% CI."""
        mask = np.isfinite(y_score) & np.isfinite(y_true)
        y_t = y_true[mask]
        y_s = y_score[mask]
        if len(y_t) < 30 or y_t.sum() < 5 or y_t.sum() == len(y_t):
            return None
        try:
            auc = roc_auc_score(y_t, y_s)
            # Bootstrap CI
            rng = np.random.RandomState(42)
            aucs = []
            for _ in range(1000):
                idx = rng.randint(0, len(y_t), len(y_t))
                if y_t.iloc[idx].sum() == 0 or y_t.iloc[idx].sum() == len(idx):
                    continue
                aucs.append(roc_auc_score(y_t.iloc[idx], y_s.iloc[idx]))
            ci_lo = np.percentile(aucs, 2.5) if aucs else auc
            ci_hi = np.percentile(aucs, 97.5) if aucs else auc
            return {"auc": round(auc, 4), "ci": [round(ci_lo, 4), round(ci_hi, 4)],
                    "n": int(mask.sum()), "n_events": int(y_t.sum())}
        except Exception:
            return None

    # --- Matchup 1: Cardiac death ---
    print("\n  ── CARDIAC DEATH ──")
    y_cardiac = df["cardiac_death"]

    r = safe_auc(y_cardiac, df["gamma_cardiac"].abs(), "Γ_cardiac")
    if r:
        print(f"    Γ_cardiac:     AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
        results["gamma_cardiac_vs_cardiac_death"] = r

    if "Framingham_risk" in df.columns:
        r = safe_auc(y_cardiac, df["Framingham_risk"], "Framingham")
        if r:
            print(f"    Framingham:    AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
            results["framingham_vs_cardiac_death"] = r

    if "ASCVD_risk" in df.columns:
        r = safe_auc(y_cardiac, df["ASCVD_risk"], "ASCVD")
        if r:
            print(f"    ASCVD PCE:     AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
            results["ascvd_vs_cardiac_death"] = r

    if "SCORE2_risk" in df.columns:
        r = safe_auc(y_cardiac, df["SCORE2_risk"], "SCORE2")
        if r:
            print(f"    SCORE2:        AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
            results["score2_vs_cardiac_death"] = r

    # --- Matchup 2: Renal death ---
    print("\n  ── RENAL DEATH ──")
    y_renal = df["renal_death"]

    r = safe_auc(y_renal, df["gamma_renal"].abs(), "Γ_renal")
    if r:
        print(f"    Γ_renal:       AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
        results["gamma_renal_vs_renal_death"] = r

    r = safe_auc(y_renal, df["eGFR_risk"], "eGFR_risk")
    if r:
        print(f"    CKD-EPI (1/eGFR): AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
        results["ckdepi_vs_renal_death"] = r

    # --- Matchup 3: Endocrine death ---
    print("\n  ── ENDOCRINE DEATH (Diabetes) ──")
    y_endo = df["endocrine_death"]

    r = safe_auc(y_endo, df["gamma_endocrine"].abs(), "Γ_endocrine")
    if r:
        print(f"    Γ_endocrine:   AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
        results["gamma_endocrine_vs_endocrine_death"] = r

    if "HOMA_IR" in df.columns:
        r = safe_auc(y_endo, df["HOMA_IR"], "HOMA-IR")
        if r:
            print(f"    HOMA-IR:       AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
            results["homa_ir_vs_endocrine_death"] = r

    # --- Matchup 4: All-cause mortality ---
    print("\n  ── ALL-CAUSE MORTALITY ──")
    y_all = df["allcause_death"]

    r = safe_auc(y_all, 1.0 - df["H"], "1 - H")
    if r:
        print(f"    1 - H:         AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
        results["gamma_H_vs_allcause_death"] = r

    if "Framingham_risk" in df.columns:
        r = safe_auc(y_all, df["Framingham_risk"], "Framingham")
        if r:
            print(f"    Framingham:    AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
            results["framingham_vs_allcause_death"] = r

    if "ASCVD_risk" in df.columns:
        r = safe_auc(y_all, df["ASCVD_risk"], "ASCVD")
        if r:
            print(f"    ASCVD PCE:     AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
            results["ascvd_vs_allcause_death"] = r

    # --- FIB-4 matchup (no hepatic deaths, but compare to allcause) ---
    print("\n  ── FIB-4 vs Γ_hepatic (ALL-CAUSE, no hepatic deaths in UCOD) ──")
    r = safe_auc(y_all, df["gamma_hepatic"].abs(), "Γ_hepatic")
    if r:
        print(f"    Γ_hepatic:     AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
        results["gamma_hepatic_vs_allcause_death"] = r

    if "FIB4" in df.columns:
        r = safe_auc(y_all, df["FIB4"], "FIB-4")
        if r:
            print(f"    FIB-4:         AUC = {r['auc']:.4f}  {r['ci']}  n={r['n']}, events={r['n_events']}")
            results["fib4_vs_allcause_death"] = r

    # ------------------------------------------------------------------
    # Step 14: Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY: WINNER BY MATCHUP")
    print("=" * 72)

    matchups = [
        ("Cardiac death", "gamma_cardiac_vs_cardiac_death",
         [("framingham_vs_cardiac_death", "Framingham"),
          ("ascvd_vs_cardiac_death", "ASCVD PCE"),
          ("score2_vs_cardiac_death", "SCORE2")]),
        ("Renal death", "gamma_renal_vs_renal_death",
         [("ckdepi_vs_renal_death", "CKD-EPI")]),
        ("Endocrine death", "gamma_endocrine_vs_endocrine_death",
         [("homa_ir_vs_endocrine_death", "HOMA-IR")]),
        ("All-cause", "gamma_H_vs_allcause_death",
         [("framingham_vs_allcause_death", "Framingham"),
          ("ascvd_vs_allcause_death", "ASCVD PCE")]),
    ]

    summary_lines = []
    for outcome, gamma_key, competitors in matchups:
        gamma_auc = results.get(gamma_key, {}).get("auc", None)
        line = f"\n  {outcome}:"
        if gamma_auc is not None:
            line += f"  Γ AUC = {gamma_auc:.4f}"
        else:
            line += "  Γ AUC = N/A"

        best_competitor = None
        best_auc = 0.0
        for comp_key, comp_name in competitors:
            comp_auc = results.get(comp_key, {}).get("auc", None)
            if comp_auc is not None:
                line += f"  |  {comp_name} = {comp_auc:.4f}"
                if comp_auc > best_auc:
                    best_auc = comp_auc
                    best_competitor = comp_name
            else:
                line += f"  |  {comp_name} = N/A"

        if gamma_auc is not None and best_auc > 0:
            delta = gamma_auc - best_auc
            if delta > 0.01:
                winner = "Γ WINS"
            elif delta < -0.01:
                winner = f"{best_competitor} WINS"
            else:
                winner = "TIE (Δ<0.01)"
            line += f"  →  {winner} (Δ={delta:+.4f})"

        print(line)
        summary_lines.append(line)

    # ------------------------------------------------------------------
    # Step 15: Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "protocol": "Gamma vs Textbook Head-to-Head v1.0",
        "timestamp_utc": ts,
        "cohort_size": len(df),
        "n_deaths": int(df["allcause_death"].sum()),
        "results": results,
        "summary": summary_lines,
    }

    out_json = RESULTS_DIR / "gamma_vs_textbook_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved: {out_json}")

    # Text report
    report_path = RESULTS_DIR / "gamma_vs_textbook_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("  Γ-SCORE vs TEXTBOOK RISK FORMULAS — HEAD-TO-HEAD\n")
        f.write(f"  NHANES 1999-2018 x NDI 2019  |  N={len(df):,}  |  Deaths={int(df['allcause_death'].sum()):,}\n")
        f.write("=" * 72 + "\n\n")
        f.write("  ZERO parameters fitted to outcome data.\n")
        f.write("  Γ uses pre-existing Z_normal values.\n")
        f.write("  Textbook formulas use their published coefficients.\n\n")
        for key, val in results.items():
            f.write(f"  {key}: AUC={val['auc']:.4f} {val['ci']} n={val['n']} events={val['n_events']}\n")
        f.write("\n\nSUMMARY\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"  Saved: {report_path}")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)

    return results


if __name__ == "__main__":
    import pandas as pd
    run_comparison()

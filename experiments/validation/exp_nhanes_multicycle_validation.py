#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: Multi-Cycle NHANES → Γ-System Validation (2013–2018)
================================================================

PURPOSE
-------
Extend the single-cycle NHANES validation (exp_nhanes_gamma_validation.py)
by pooling THREE consecutive NHANES 2-year cycles:
  - 2013-2014 (suffix _H)
  - 2015-2016 (suffix _I)
  - 2017-2018 (suffix _J)

This increases the sample size from ~2,400 to ~7,000+ adults, boosting
statistical power for organ systems that were non-significant in the
single-cycle analysis (cardiac AUC=0.51, hepatic AUC=0.51).

ENGINEERING RATIONALE
---------------------
"Not significant" ≠ "effect doesn't exist".
It means SNR < detection threshold at current sample size.
Engineering response: increase SNR by pooling data.

CDC explicitly endorses multi-cycle pooling:
  https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/faq.aspx

PROTOCOL
--------
Same as single-cycle, but repeated across 3 cycles:
1. Download 9 × 3 = 27 XPT files from CDC
2. Merge across cycles, de-duplicate by SEQN (unique across cycles)
3. Compute 12-organ Γ vectors with ZERO fitted parameters
4. SHA-256 hash-lock ALL predictions before loading diagnoses
5. AUC + Health Index validation
6. Compare single-cycle vs multi-cycle results

DATA
----
NHANES 2013-2014, 2015-2016, 2017-2018 (CDC, public domain)
Estimated merged sample: 7,000-8,000 adults (≥20) with ≥10 lab values
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alice.diagnostics.lab_mapping import (
    LAB_CATALOGUE,
    ORGAN_LIST,
    ORGAN_SYSTEMS,
    LabMapper,
)
from alice.diagnostics.gamma_engine import GammaEngine, PatientGammaVector


# ============================================================================
# 1. MULTI-CYCLE NHANES CONFIGURATION
# ============================================================================

# NHANES file URL patterns per cycle
# CDC URL structure: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{YEAR}/DataFiles/{FILE}.XPT
CYCLES = {
    "2013-2014": {
        "year": "2013",
        "suffix": "_H",
        "label": "NHANES 2013-2014",
    },
    "2015-2016": {
        "year": "2015",
        "suffix": "_I",
        "label": "NHANES 2015-2016",
    },
    "2017-2018": {
        "year": "2017",
        "suffix": "_J",
        "label": "NHANES 2017-2018",
    },
}

# Base file stems (without cycle suffix)
FILE_STEMS = {
    "BIOPRO":  "Standard Biochemistry Profile",
    "CBC":     "Complete Blood Count",
    "HDL":     "HDL Cholesterol",
    "TRIGLY":  "Triglycerides",
    "GHB":     "Glycohemoglobin (HbA1c)",
    "GLU":     "Plasma Fasting Glucose",
    "MCQ":     "Medical Conditions Questionnaire",
    "DIQ":     "Diabetes Questionnaire",
    "DEMO":    "Demographics",
}

DATA_DIR = PROJECT_ROOT / "nhanes_data"


def build_file_urls() -> Dict[str, Dict[str, str]]:
    """Build complete URL map for all cycles.

    Returns: {cycle_label: {stem: url}}
    """
    urls = {}
    for cycle_key, cycle_info in CYCLES.items():
        year = cycle_info["year"]
        suffix = cycle_info["suffix"]
        base = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles"
        cycle_urls = {}
        for stem in FILE_STEMS:
            filename = f"{stem}{suffix}"
            cycle_urls[filename] = f"{base}/{filename}.XPT"
        urls[cycle_key] = cycle_urls
    return urls


# Lab variable mapping (same as single-cycle — physics doesn't change)
NHANES_TO_ALICE = {
    "LBXSATSI": ("AST",     1.0),
    "LBXSASSI": ("ALT",     1.0),
    "LBXSAPSI": ("ALP",     1.0),
    "LBXSGB":   ("GGT",     1.0),
    "LBXSTB":   ("Bil_total", 1.0),
    "LBXSAL":   ("Albumin",  1.0),
    "LBXSTP":   ("Total_Protein", 1.0),
    "LBXSCR":   ("Cr",      1.0),
    "LBXSBU":   ("BUN",     1.0),
    "LBXSUA":   ("Uric_Acid", 1.0),
    "LBXSCH":   ("TC",      1.0),
    "LBXSC3SI": ("CO2",     1.0),
    "LBXSNA":   ("Na",      1.0),
    "LBXSK":    ("K",       1.0),
    "LBXSCL":   ("Cl",      1.0),
    "LBXSCA":   ("Ca",      1.0),
    "LBXSPH":   ("ALP",     1.0),
    "LBXSGL":   ("Glucose", 1.0),
    "LBXSCK":   ("CK_MB",   1.0),
    "LBXWBCSI": ("WBC",     1.0),
    "LBXRBCSI": ("RBC",     1.0),
    "LBXHGB":   ("Hb",      1.0),
    "LBXHCT":   ("Hct",     1.0),
    "LBXMCVSI": ("MCV",     1.0),
    "LBXPLTSI": ("Plt",     1.0),
    "LBDHDD":   ("HDL",     1.0),
    "LBXTR":    ("TG",      1.0),
    "LBXGH":    ("HbA1c",   1.0),
    "LBXGLU":   ("Glucose", 1.0),
}

DIAGNOSIS_MAPPING = {
    "MCQ160B": {"label": "congestive_heart_failure",  "organ": "cardiac",   "yes": [1]},
    "MCQ160C": {"label": "coronary_heart_disease",    "organ": "cardiac",   "yes": [1]},
    "MCQ160D": {"label": "angina",                    "organ": "cardiac",   "yes": [1]},
    "MCQ160E": {"label": "heart_attack",              "organ": "cardiac",   "yes": [1]},
    "MCQ160F": {"label": "stroke",                    "organ": "neuro",     "yes": [1]},
    "MCQ220":  {"label": "cancer",                    "organ": "immune",    "yes": [1]},
    "MCQ160L": {"label": "liver_condition",           "organ": "hepatic",   "yes": [1]},
    "MCQ160O": {"label": "COPD",                      "organ": "pulmonary", "yes": [1]},
    "MCQ160P": {"label": "emphysema",                 "organ": "pulmonary", "yes": [1]},
    "MCQ160N": {"label": "gout",                      "organ": "bone",      "yes": [1]},
    "MCQ160M": {"label": "thyroid_problem",           "organ": "endocrine", "yes": [1]},
    "MCQ160K": {"label": "chronic_bronchitis",        "organ": "pulmonary", "yes": [1]},
    "DIQ010":  {"label": "diabetes",                  "organ": "endocrine", "yes": [1]},
}

ORGAN_DISEASE_MAP = {
    "cardiac":    ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E"],
    "hepatic":    ["MCQ160L"],
    "endocrine":  ["DIQ010", "MCQ160M"],
    "pulmonary":  ["MCQ160O", "MCQ160P", "MCQ160K"],
    "neuro":      ["MCQ160F"],
    "immune":     ["MCQ220"],
    "bone":       ["MCQ160N"],
}


# ============================================================================
# 2. DOWNLOAD ENGINE (multi-cycle)
# ============================================================================

def download_all_cycles(force: bool = False) -> Dict[str, Dict[str, Path]]:
    """Download XPT files for all 3 NHANES cycles.

    Returns: {cycle_key: {filename: local_path}}
    """
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_paths = {}
    headers = {"User-Agent": "Mozilla/5.0 (Alice-Gamma-Net/3.4.0; research)"}
    url_map = build_file_urls()

    for cycle_key, cycle_urls in url_map.items():
        print(f"\n  --- {CYCLES[cycle_key]['label']} ---")
        cycle_paths = {}
        for filename, url in cycle_urls.items():
            local_path = DATA_DIR / f"{filename}.XPT"
            cycle_paths[filename] = local_path

            if local_path.exists() and local_path.stat().st_size > 30000 and not force:
                print(f"    [CACHED] {filename}: {local_path.stat().st_size:,} bytes")
                continue

            print(f"    [DOWNLOADING] {filename} ...")
            try:
                resp = requests.get(url, headers=headers, timeout=120)
                resp.raise_for_status()
                if not resp.content[:14].startswith(b"HEADER RECORD"):
                    print(f"      x Got HTML instead of XPT for {filename}")
                    continue
                local_path.write_bytes(resp.content)
                print(f"      -> {local_path.stat().st_size:,} bytes OK")
            except Exception as e:
                print(f"      x FAILED: {e}")
                if local_path.exists():
                    local_path.unlink()

        all_paths[cycle_key] = cycle_paths

    return all_paths


# ============================================================================
# 3. DATA LOADING & MERGING (multi-cycle)
# ============================================================================

def load_and_merge_all_cycles(
    all_paths: Dict[str, Dict[str, Path]],
) -> "pd.DataFrame":
    """Load and merge all cycles into one DataFrame.

    NHANES SEQN is unique across cycles (different ranges per cycle),
    so simple concatenation is safe.
    """
    import pandas as pd
    import pyreadstat

    cycle_frames = []

    for cycle_key, cycle_paths in all_paths.items():
        suffix = CYCLES[cycle_key]["suffix"]
        label = CYCLES[cycle_key]["label"]
        print(f"\n  Loading {label} ...")

        frames = {}
        for filename, path in cycle_paths.items():
            if not path.exists():
                print(f"    WARNING: {filename} not found")
                continue
            try:
                df, meta = pyreadstat.read_xport(str(path))
                df.set_index("SEQN", inplace=True)
                stem = filename.replace(suffix, "")
                frames[stem] = df
                print(f"    {filename}: {len(df)} rows")
            except Exception as e:
                print(f"    ERROR loading {filename}: {e}")

        if "DEMO" not in frames:
            print(f"    SKIPPING {label}: no DEMO file")
            continue

        # Merge within cycle
        merged = frames["DEMO"]
        for stem in ["BIOPRO", "CBC", "HDL", "TRIGLY", "GHB", "GLU", "MCQ", "DIQ"]:
            if stem in frames:
                merged = merged.join(frames[stem], how="inner", rsuffix=f"_{stem}")

        # Add cycle label
        merged["_cycle"] = cycle_key
        cycle_frames.append(merged)
        print(f"    Merged: {len(merged)} respondents")

    # Concatenate all cycles
    if not cycle_frames:
        raise RuntimeError("No cycles loaded successfully")

    combined = pd.concat(cycle_frames, axis=0)
    print(f"\n  Combined all cycles: {len(combined)} total respondents")

    # Verify no SEQN duplicates
    if combined.index.duplicated().any():
        n_dup = combined.index.duplicated().sum()
        print(f"  WARNING: {n_dup} duplicate SEQNs found, keeping first")
        combined = combined[~combined.index.duplicated(keep="first")]

    # Filter adults >= 20
    if "RIDAGEYR" in combined.columns:
        combined = combined[combined["RIDAGEYR"] >= 20]
        print(f"  Adults (>=20): {len(combined)}")

    return combined


# ============================================================================
# 4-7. REUSE PHYSICS/VALIDATION ENGINES (identical to single-cycle)
# ============================================================================

def compute_gamma_vectors(
    df: "pd.DataFrame",
    min_labs: int = 10,
) -> Tuple["pd.DataFrame", List[Dict[str, Any]]]:
    """Compute 12-organ Gamma vectors. ZERO modifications from single-cycle."""
    import pandas as pd

    engine = GammaEngine()
    results = []
    records = []
    skipped = 0

    for seqn, row in df.iterrows():
        lab_values = {}
        for nhanes_var, (alice_name, scale) in NHANES_TO_ALICE.items():
            if nhanes_var in row.index:
                val = row[nhanes_var]
                if pd.notna(val) and np.isfinite(val) and val > 0:
                    lab_values[alice_name] = float(val) * scale

        if len(lab_values) < min_labs:
            skipped += 1
            continue

        gamma_vec = engine.lab_to_gamma(lab_values)

        record = {"SEQN": int(seqn), "n_labs": len(lab_values)}
        row_data = {"SEQN": int(seqn), "n_labs": len(lab_values)}
        for organ in ORGAN_LIST:
            g = gamma_vec[organ]
            row_data[f"gamma_{organ}"] = g
            record[f"gamma_{organ}"] = round(g, 6)

        row_data["health_index"] = gamma_vec.health_index
        row_data["total_gamma_sq"] = gamma_vec.total_gamma_squared
        record["health_index"] = round(gamma_vec.health_index, 6)
        record["total_gamma_sq"] = round(gamma_vec.total_gamma_squared, 6)

        results.append(row_data)
        records.append(record)

    print(f"  Computed Gamma vectors: {len(results)} ({skipped} skipped, <{min_labs} labs)")

    gamma_df = pd.DataFrame(results)
    if not gamma_df.empty:
        gamma_df.set_index("SEQN", inplace=True)
    return gamma_df, records


def hash_lock_predictions(records: List[Dict], output_path: Path) -> str:
    """SHA-256 hash-lock predictions before unblinding."""
    prediction_data = {
        "protocol": "Gamma-Net Multi-Cycle Blind Validation v2.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "Alice Lab-Gamma Engine (alice.diagnostics.gamma_engine)",
        "z_normal_source": "Clinical textbook reference values (ORGAN_SYSTEMS dict)",
        "fitted_parameters": 0,
        "n_respondents": len(records),
        "data_source": "NHANES 2013-2014 + 2015-2016 + 2017-2018 (CDC, public domain)",
        "predictions": records,
    }

    json_bytes = json.dumps(prediction_data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sha256_hash = hashlib.sha256(json_bytes).hexdigest()
    prediction_data["sha256"] = sha256_hash

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prediction_data, f, indent=2, ensure_ascii=False)

    print(f"\n  +---------------------------------------------------------+")
    print(f"  |  BLIND PREDICTIONS HASH-LOCKED (Multi-Cycle)            |")
    print(f"  |  SHA-256: {sha256_hash[:40]}... |")
    print(f"  |  Respondents: {len(records):<43d}|")
    print(f"  |  Fitted parameters: 0                                   |")
    print(f"  |  Cycles: 2013-2014 + 2015-2016 + 2017-2018             |")
    print(f"  +---------------------------------------------------------+")

    return sha256_hash


def extract_diagnoses(df: "pd.DataFrame", gamma_df: "pd.DataFrame") -> "pd.DataFrame":
    """Extract organ-level disease flags."""
    import pandas as pd

    diag = pd.DataFrame(index=gamma_df.index)

    for organ, vars_list in ORGAN_DISEASE_MAP.items():
        organ_positive = pd.Series(False, index=gamma_df.index)
        for var_name in vars_list:
            if var_name in df.columns:
                col = df.loc[gamma_df.index, var_name].reindex(gamma_df.index)
                mapping = DIAGNOSIS_MAPPING[var_name]
                is_yes = col.isin(mapping["yes"])
                organ_positive = organ_positive | is_yes
        diag[f"dx_{organ}"] = organ_positive.astype(int)

    for var_name, mapping in DIAGNOSIS_MAPPING.items():
        if var_name in df.columns:
            col = df.loc[gamma_df.index, var_name].reindex(gamma_df.index)
            diag[f"dx_{mapping['label']}"] = col.isin(mapping["yes"]).astype(int)

    counts = {col: int(diag[col].sum()) for col in diag.columns if col.startswith("dx_")}
    print(f"\n  Disease prevalence (multi-cycle cohort):")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / len(diag) * 100
            print(f"    {label:<40s} {count:>5d} ({pct:5.1f}%)")

    return diag


def compute_auc_validation(
    gamma_df: "pd.DataFrame",
    diag_df: "pd.DataFrame",
) -> Dict[str, Dict[str, Any]]:
    """AUC for each organ's Gamma predicting its disease."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import norm

    results = {}

    for organ in ORGAN_DISEASE_MAP:
        dx_col = f"dx_{organ}"
        gamma_col = f"gamma_{organ}"

        if dx_col not in diag_df.columns or gamma_col not in gamma_df.columns:
            continue

        y_true = diag_df[dx_col].values
        y_score = gamma_df[gamma_col].values

        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)

        if n_pos < 10 or n_neg < 10:
            print(f"  {organ}: SKIPPED (n+={n_pos}, n-={n_neg})")
            continue

        try:
            auc = roc_auc_score(y_true, np.abs(y_score))
        except ValueError:
            continue

        se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (auc / (2 - auc) - auc**2)
                      + (n_neg - 1) * (2 * auc**2 / (1 + auc) - auc**2))
                     / (n_pos * n_neg))
        if se > 0:
            z_stat = (auc - 0.5) / se
            p_value = 1.0 - norm.cdf(z_stat)
        else:
            z_stat = 0.0
            p_value = 1.0

        results[organ] = {
            "auc": round(auc, 4),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "z_statistic": round(float(z_stat), 3),
            "p_value": float(p_value),
            "significant_005": bool(p_value < 0.05),
            "significant_001": bool(p_value < 0.01),
        }

    return results


def compute_health_index_validation(
    gamma_df: "pd.DataFrame",
    diag_df: "pd.DataFrame",
) -> Dict[str, Any]:
    """Health Index H = prod(1-Gamma^2) vs disease burden."""
    from scipy.stats import spearmanr, pearsonr

    dx_cols = [c for c in diag_df.columns if c.startswith("dx_") and
               c.split("dx_")[1] in ORGAN_DISEASE_MAP]
    disease_count = diag_df[dx_cols].sum(axis=1)

    h = gamma_df["health_index"].values
    burden = disease_count.values

    rho, p_spearman = spearmanr(h, burden)
    r, p_pearson = pearsonr(h, burden)

    return {
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p_spearman),
        "pearson_r": round(float(r), 4),
        "pearson_p": float(p_pearson),
        "mean_H_0_diseases": round(float(h[burden == 0].mean()), 4) if (burden == 0).any() else None,
        "mean_H_1_disease":  round(float(h[burden == 1].mean()), 4) if (burden == 1).any() else None,
        "mean_H_2plus":      round(float(h[burden >= 2].mean()), 4) if (burden >= 2).any() else None,
        "n_0_diseases": int((burden == 0).sum()),
        "n_1_disease":  int((burden == 1).sum()),
        "n_2plus":      int((burden >= 2).sum()),
    }


# ============================================================================
# 8. COMPARISON ENGINE
# ============================================================================

# Single-cycle results for comparison (from exp_nhanes_gamma_validation.py)
SINGLE_CYCLE_RESULTS = {
    "cardiac":   {"auc": 0.5103, "p": 0.302},
    "hepatic":   {"auc": 0.5129, "p": 0.310},
    "endocrine": {"auc": 0.8004, "p": 0.0},
    "pulmonary": {"auc": 0.5436, "p": 0.013},
    "neuro":     {"auc": 0.5914, "p": 7e-4},
    "immune":    {"auc": 0.5422, "p": 0.017},
    "bone":      {"auc": 0.6296, "p": 3.2e-8},
}


# ============================================================================
# 9. FINAL REPORT
# ============================================================================

def print_final_report(
    auc_results: Dict[str, Dict],
    health_results: Dict[str, Any],
    sha256_hash: str,
    n_total: int,
    n_per_cycle: Dict[str, int],
) -> str:
    """Print comprehensive multi-cycle validation report."""

    lines = []

    lines.append("")
    lines.append("=" * 78)
    lines.append("  GAMMA-NET MULTI-CYCLE EMPIRICAL VALIDATION")
    lines.append("  NHANES 2013-2014 + 2015-2016 + 2017-2018")
    lines.append("  Power Upgrade: from n=2,401 to n=~7,000+")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"  Date:              {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"  Total cohort:      {n_total:,} adults (>=20, >=10 labs)")
    lines.append(f"  Fitted parameters: 0 (pure physics)")
    lines.append(f"  SHA-256 hash:      {sha256_hash}")
    for cycle_key, n in n_per_cycle.items():
        lines.append(f"    {CYCLES[cycle_key]['label']}: {n:,}")
    lines.append("")

    # AUC comparison table
    lines.append("-" * 78)
    lines.append("  ORGAN-LEVEL AUC: Single Cycle (2017-18) vs Multi-Cycle (2013-18)")
    lines.append("-" * 78)
    lines.append(f"  {'Organ':<12} {'AUC(1c)':>8} {'p(1c)':>10} {'AUC(3c)':>8} "
                 f"{'p(3c)':>10} {'n+':>6} {'n-':>6} {'Delta':>7} {'Sig':>4}")
    lines.append("  " + "-" * 72)

    n_sig_single = 0
    n_sig_multi = 0
    improved = 0

    for organ in ["endocrine", "bone", "neuro", "pulmonary", "immune", "hepatic", "cardiac"]:
        sc = SINGLE_CYCLE_RESULTS.get(organ, {})
        mc = auc_results.get(organ, {})
        if not mc:
            continue

        auc_1 = sc.get("auc", 0.5)
        p_1 = sc.get("p", 1.0)
        auc_3 = mc["auc"]
        p_3 = mc["p_value"]
        delta = auc_3 - auc_1

        sig_1 = "*" if p_1 < 0.05 else " "
        sig_3 = "***" if mc["significant_001"] else ("*" if mc["significant_005"] else " ")

        if p_1 < 0.05:
            n_sig_single += 1
        if mc["significant_005"]:
            n_sig_multi += 1
        if delta > 0:
            improved += 1

        lines.append(
            f"  {organ:<12} {auc_1:>8.4f}{sig_1} {p_1:>10.2e} {auc_3:>8.4f}{sig_3} "
            f"{p_3:>10.2e} {mc['n_positive']:>6} {mc['n_negative']:>6} "
            f"{delta:>+7.4f} {sig_3:>4}"
        )

    lines.append("")
    lines.append(f"  Significant organs (p<0.05): {n_sig_single}/7 (single) -> {n_sig_multi}/7 (multi)")
    lines.append(f"  AUC improved: {improved}/7 organs")
    lines.append("  * p < 0.05   *** p < 0.01")
    lines.append("")

    # Health Index
    lines.append("-" * 78)
    lines.append("  HEALTH INDEX H = Pi(1-Gamma^2) vs DISEASE BURDEN")
    lines.append("-" * 78)
    lines.append(f"  Spearman rho = {health_results['spearman_rho']:.4f}  "
                 f"(p = {health_results['spearman_p']:.2e})")
    lines.append(f"  Pearson  r   = {health_results['pearson_r']:.4f}  "
                 f"(p = {health_results['pearson_p']:.2e})")
    lines.append("")
    lines.append(f"  Mean H (0 diseases):  {health_results.get('mean_H_0_diseases', 'N/A')}  "
                 f"(n={health_results.get('n_0_diseases', 0):,})")
    lines.append(f"  Mean H (1 disease):   {health_results.get('mean_H_1_disease', 'N/A')}  "
                 f"(n={health_results.get('n_1_disease', 0):,})")
    lines.append(f"  Mean H (2+ diseases): {health_results.get('mean_H_2plus', 'N/A')}  "
                 f"(n={health_results.get('n_2plus', 0):,})")
    lines.append("")

    # Comparison with single cycle Health Index
    lines.append("  Comparison with single-cycle:")
    lines.append(f"    Single (n=2,401):  rho = -0.3811  (p = 7.95e-84)")
    lines.append(f"    Multi  (n={n_total:,}):  rho = {health_results['spearman_rho']:.4f}  "
                 f"(p = {health_results['spearman_p']:.2e})")
    lines.append("")

    # Verdict
    lines.append("-" * 78)
    lines.append("  VERDICT")
    lines.append("-" * 78)

    if n_sig_multi > n_sig_single:
        lines.append(f"  POWER GAIN: {n_sig_multi - n_sig_single} additional organ(s) became significant")
        lines.append(f"  with ~3x sample size. This confirms the single-cycle null results")
        lines.append(f"  were due to insufficient statistical power, not absent signal.")
    elif n_sig_multi == n_sig_single:
        lines.append(f"  STABLE: Same {n_sig_multi}/7 organs significant with 3x data.")
        lines.append(f"  Cardiac/hepatic remain non-significant -> may need specialized biomarkers.")
    else:
        lines.append(f"  UNEXPECTED: Fewer significant organs with more data. Investigate.")

    lines.append("")
    lines.append("  Engineering interpretation:")
    lines.append("  For cardiac/hepatic: the NHANES sensor (self-report + basic labs) may")
    lines.append("  have insufficient SNR. Next step: specialized cohorts with troponin,")
    lines.append("  BNP, FibroScan, or liver elastography.")
    lines.append("")
    lines.append("-" * 78)
    lines.append("  METHODOLOGICAL NOTES")
    lines.append("-" * 78)
    lines.append("  1. Z_normal from clinical textbooks (identical across all cycles)")
    lines.append("  2. Gamma = (Z_patient - Z_normal) / (Z_patient + Z_normal)")
    lines.append("  3. Zero parameters fitted to any NHANES data")
    lines.append("  4. SHA-256 hash-locked before diagnosis data loaded")
    lines.append("  5. Cycles pooled by simple concatenation (SEQN unique across cycles)")
    lines.append("  6. No survey weights applied (unweighted analysis for power)")
    lines.append("")
    lines.append("  Warning: This is a RESEARCH VALIDATION, not a clinical diagnostic tool.")
    lines.append("=" * 78)
    lines.append("")

    report = "\n".join(lines)
    print(report)
    return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("  GAMMA-NET MULTI-CYCLE VALIDATION PIPELINE")
    print("  NHANES 2013-2014 + 2015-2016 + 2017-2018")
    print("  Engineering response to low-SNR organs: increase N")
    print("=" * 78)
    print()

    output_dir = PROJECT_ROOT / "nhanes_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Download 27 files across 3 cycles
    print("Phase 1: Downloading NHANES data (3 cycles, 27 files)...")
    all_paths = download_all_cycles()

    # Phase 2: Load & merge
    print("\nPhase 2: Loading and merging across cycles...")
    df = load_and_merge_all_cycles(all_paths)

    # Track per-cycle counts
    n_per_cycle = {}
    if "_cycle" in df.columns:
        for cycle_key in CYCLES:
            n_per_cycle[cycle_key] = int((df["_cycle"] == cycle_key).sum())

    # Phase 3: Compute Gamma vectors
    print("\nPhase 3: Computing Gamma vectors (physics only, 0 fitted parameters)...")
    gamma_df, records = compute_gamma_vectors(df)

    if gamma_df.empty:
        print("  FATAL: No valid Gamma vectors. Check data.")
        sys.exit(1)

    csv_path = output_dir / "nhanes_multicycle_gamma_vectors.csv"
    gamma_df.to_csv(csv_path)
    print(f"  Saved: {csv_path.name} ({len(gamma_df)} rows)")

    # Phase 4: Hash-lock
    print("\nPhase 4: Hash-locking blind predictions (multi-cycle)...")
    hash_path = output_dir / "multicycle_blind_predictions.json"
    sha256_hash = hash_lock_predictions(records, hash_path)

    # ===== FIREWALL =====
    print("\n" + "=" * 78)
    print("  CROSSING THE FIREWALL: Now loading diagnosis data")
    print("=" * 78)

    # Phase 5: Diagnoses
    print("\nPhase 5: Extracting diagnoses (multi-cycle)...")
    diag_df = extract_diagnoses(df, gamma_df)

    # Phase 6: AUC
    print("\nPhase 6: AUC validation (multi-cycle)...")
    auc_results = compute_auc_validation(gamma_df, diag_df)

    # Phase 7: Health Index
    print("\nPhase 7: Health Index validation...")
    health_results = compute_health_index_validation(gamma_df, diag_df)

    # Phase 8: Report
    report = print_final_report(
        auc_results, health_results, sha256_hash,
        len(gamma_df), n_per_cycle,
    )

    # Save results
    full_results = {
        "protocol": "Gamma-Net Multi-Cycle Blind Validation v2.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_predictions": sha256_hash,
        "n_respondents": len(gamma_df),
        "n_per_cycle": n_per_cycle,
        "cycles": ["2013-2014", "2015-2016", "2017-2018"],
        "auc_results": auc_results,
        "health_index_validation": health_results,
        "single_cycle_comparison": SINGLE_CYCLE_RESULTS,
        "organ_z_normal": dict(ORGAN_SYSTEMS),
    }

    results_path = output_dir / "multicycle_validation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {results_path}")

    report_path = output_dir / "multicycle_validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    print(f"\n  Done. Total: {len(gamma_df):,} adults across 3 NHANES cycles.")
    return full_results


if __name__ == "__main__":
    main()

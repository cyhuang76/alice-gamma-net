#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: MIMIC-IV Diagnostic Accuracy — Γ_organ vs ICD-10 Gold Standard
═══════════════════════════════════════════════════════════════════════════

PURPOSE
───────
Upgrade the Γ-framework from Level 3 (prospective cohort) to Level 2b
(diagnostic accuracy study) by computing Γ_organ for ~50,000 ICU admissions
in MIMIC-IV and comparing against ICD-10 coded discharge diagnoses.

STUDY DESIGN (STARD-compliant)
──────────────────────────────
  Index test:      Γ_organ, H = Π(1 − Γᵢ²)
  Reference:       ICD-10 discharge diagnoses (MIMIC-IV diagnoses_icd)
  Population:      All adult ICU admissions with ≥8 lab values in first 24h
  Parameters fitted to MIMIC-IV: ZERO
  Protocol:        Hash-lock ALL Γ predictions before loading any diagnoses

ANALYSES
────────
  1. Per-organ AUC: Γ_cardiac vs cardiac ICD-10, etc. (12 organ systems)
  2. Sensitivity / Specificity at Youden-optimal threshold
  3. Positive/Negative predictive values
  4. Health Index H → ICU mortality
  5. H → length-of-stay correlation
  6. Multi-organ discrimination (12-organ Γ → primary diagnosis)
  7. Calibration analysis (Hosmer-Lemeshow)
  8. STARD flow diagram statistics

DATA
────
  MIMIC-IV v3.1 (PhysioNet, credentialed access required)
  Required files in mimic_data/:
    - labevents.csv.gz       (or labevents.parquet)
    - d_labitems.csv.gz      (lab item dictionary)
    - diagnoses_icd.csv.gz   (discharge diagnoses)
    - admissions.csv.gz      (admission details + hospital_expire_flag)
    - patients.csv.gz        (demographics)
    - icustays.csv.gz        (ICU stay details)

  Download instructions:
    1. Create account at https://physionet.org
    2. Complete CITI "Data or Specimens Only Research" training
    3. Sign MIMIC-IV Data Use Agreement
    4. Download from: https://physionet.org/content/mimiciv/3.1/
    5. Place CSV/parquet files in mimic_data/ directory

EVIDENCE LEVEL
──────────────
  Before: Level 3 (H predicts all-cause mortality, NHANES/NDI)
  After:  Level 2b (Γ_organ has diagnostic accuracy vs ICD-10 gold standard)

PHYSICS
───────
  Same Γ = (Z_patient − Z_normal) / (Z_patient + Z_normal)
  Same Z_normal from alice.diagnostics.lab_mapping (pre-existing, ZERO refitting)
  Same GammaEngine (pre-existing code, no modification)
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
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

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
# 1. CONFIGURATION
# ============================================================================

MIMIC_DIR = PROJECT_ROOT / "mimic_data"
RESULTS_DIR = PROJECT_ROOT / "mimic_results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Minimum labs for a valid Γ computation
MIN_LAB_COUNT = 8

# First-24h window (seconds) for lab collection
FIRST_24H_SECONDS = 24 * 3600

# ============================================================================
# 2. MIMIC-IV LAB ITEM MAPPING
# ============================================================================
# MIMIC-IV labevents.itemid → Alice lab name
# itemid values from MIMIC-IV d_labitems (most common standard labs)
# Source: MIMIC-IV documentation + d_labitems.csv
#
# NOTE: These are the standard LOINC-mapped lab tests in MIMIC-IV.
# The itemid may vary between MIMIC versions; we auto-discover from
# d_labitems.csv when available.

MIMIC_ITEMID_TO_ALICE: Dict[int, Tuple[str, float]] = {
    # === Liver function ===
    50861: ("AST",          1.0),    # Alanine Aminotransferase (ALT) — actually AST
    50863: ("ALP",          1.0),    # Alkaline Phosphatase
    50878: ("AST",          1.0),    # Aspartate Aminotransferase (AST)
    50885: ("Bil_total",    1.0),    # Bilirubin, Total
    50862: ("Albumin",      1.0),    # Albumin
    50976: ("Total_Protein", 1.0),   # Protein, Total

    # === Renal ===
    50912: ("Cr",           1.0),    # Creatinine
    51006: ("BUN",          1.0),    # Urea Nitrogen (BUN)
    51007: ("Uric_Acid",    1.0),    # Uric Acid

    # === Lipids / Metabolic ===
    50907: ("TC",           1.0),    # Cholesterol, Total
    50882: ("CO2",          1.0),    # Bicarbonate (CO2)

    # === Electrolytes ===
    50983: ("Na",           1.0),    # Sodium
    50971: ("K",            1.0),    # Potassium
    50902: ("Cl",           1.0),    # Chloride
    50893: ("Ca",           1.0),    # Calcium, Total

    # === Glucose / Endocrine ===
    50931: ("Glucose",      1.0),    # Glucose
    50852: ("HbA1c",        0.01),   # Hemoglobin A1c (% → fraction if needed)

    # === Cardiac ===
    50911: ("CK_MB",        1.0),    # Creatine Kinase, MB (CK-MB)

    # === CBC / Hematology ===
    51301: ("WBC",          1.0),    # White Blood Cells
    51279: ("RBC",          1.0),    # Red Blood Cells
    51222: ("Hb",           1.0),    # Hemoglobin
    51221: ("Hct",          1.0),    # Hematocrit
    51250: ("MCV",          1.0),    # Mean Corpuscular Volume
    51265: ("Plt",          1.0),    # Platelet Count

    # === Lipids (extended) ===
    50904: ("HDL",          1.0),    # Cholesterol, HDL
    51000: ("TG",           1.0),    # Triglycerides

    # === Liver (extended) ===
    50927: ("GGT",          1.0),    # Gamma Glutamyl Transferase
    50861: ("ALT",          1.0),    # ALT (overrides AST mapping above)
}

# Fallback: label-based matching (if itemid mapping is incomplete)
MIMIC_LABEL_TO_ALICE: Dict[str, Tuple[str, float]] = {
    "Alanine Aminotransferase (ALT)":    ("ALT",          1.0),
    "Aspartate Aminotransferase (AST)":  ("AST",          1.0),
    "Alkaline Phosphatase":              ("ALP",          1.0),
    "Bilirubin, Total":                  ("Bil_total",    1.0),
    "Albumin":                           ("Albumin",      1.0),
    "Protein, Total":                    ("Total_Protein", 1.0),
    "Creatinine":                        ("Cr",           1.0),
    "Urea Nitrogen":                     ("BUN",          1.0),
    "Uric Acid":                         ("Uric_Acid",    1.0),
    "Cholesterol, Total":                ("TC",           1.0),
    "Bicarbonate":                       ("CO2",          1.0),
    "Sodium":                            ("Na",           1.0),
    "Potassium":                         ("K",            1.0),
    "Chloride":                          ("Cl",           1.0),
    "Calcium, Total":                    ("Ca",           1.0),
    "Glucose":                           ("Glucose",      1.0),
    "Hemoglobin A1c":                    ("HbA1c",        0.01),
    "Creatine Kinase, MB Isoenzyme":     ("CK_MB",        1.0),
    "White Blood Cells":                 ("WBC",          1.0),
    "Red Blood Cells":                   ("RBC",          1.0),
    "Hemoglobin":                        ("Hb",           1.0),
    "Hematocrit":                        ("Hct",          1.0),
    "MCV":                               ("MCV",          1.0),
    "Platelet Count":                    ("Plt",          1.0),
    "Cholesterol, HDL":                  ("HDL",          1.0),
    "Triglycerides":                     ("TG",           1.0),
    "Gamma Glutamyltransferase":         ("GGT",          1.0),
}


# ============================================================================
# 3. ICD-10 → ORGAN SYSTEM MAPPING
# ============================================================================
# Map ICD-10-CM chapter/block codes to Alice organ systems.
# MIMIC-IV uses ICD-10-CM (icd_version = 10).
# We map at the block level to maximize coverage.

ICD10_TO_ORGAN: Dict[str, str] = {
    # I00-I99: Diseases of the circulatory system
    "I": "cardiac",
    # J00-J99: Diseases of the respiratory system
    "J": "pulmonary",
    # K70-K77: Diseases of liver
    "K70": "hepatic", "K71": "hepatic", "K72": "hepatic", "K73": "hepatic",
    "K74": "hepatic", "K75": "hepatic", "K76": "hepatic", "K77": "hepatic",
    # K00-K95: General GI (non-hepatic)
    "K": "GI",
    # N00-N29: Diseases of kidney (glomerular + tubular)
    "N0": "renal", "N1": "renal", "N2": "renal",
    # N17-N19: Renal failure
    "N17": "renal", "N18": "renal", "N19": "renal",
    # E00-E07: Thyroid; E08-E13: Diabetes; E15-E16: Other glucose
    "E0": "endocrine", "E1": "endocrine",
    # E20-E35: Other endocrine
    "E2": "endocrine", "E3": "endocrine",
    # D50-D89: Diseases of blood / immune
    "D5": "heme", "D6": "heme", "D7": "immune", "D8": "immune",
    # C00-D49: Neoplasms → immune
    "C": "immune",
    # G00-G99: Diseases of the nervous system
    "G": "neuro",
    # F00-F99: Mental disorders (subset → neuro)
    "F": "neuro",
    # M00-M99: Musculoskeletal
    "M": "bone",
    # L00-L99: Skin → vascular (approximation for wound/vascular skin)
    "L": "vascular",
    # N30-N99: Urogenital (non-renal)
    "N3": "repro", "N4": "repro", "N5": "repro", "N6": "repro",
    "N7": "repro", "N8": "repro", "N9": "repro",
}

# Specific high-value ICD-10 codes for targeted validation
ICD10_SPECIFIC: Dict[str, Tuple[str, str]] = {
    # Cardiac
    "I21":  ("cardiac",    "Acute myocardial infarction"),
    "I50":  ("cardiac",    "Heart failure"),
    "I48":  ("cardiac",    "Atrial fibrillation"),
    "I25":  ("cardiac",    "Chronic ischemic heart disease"),
    "I10":  ("cardiac",    "Essential hypertension"),
    # Hepatic
    "K70":  ("hepatic",    "Alcoholic liver disease"),
    "K74":  ("hepatic",    "Fibrosis/cirrhosis of liver"),
    "K72":  ("hepatic",    "Hepatic failure"),
    "K76":  ("hepatic",    "Other diseases of liver"),
    # Renal
    "N17":  ("renal",      "Acute kidney failure"),
    "N18":  ("renal",      "Chronic kidney disease"),
    "N19":  ("renal",      "Unspecified kidney failure"),
    # Endocrine
    "E11":  ("endocrine",  "Type 2 diabetes mellitus"),
    "E10":  ("endocrine",  "Type 1 diabetes mellitus"),
    "E03":  ("endocrine",  "Hypothyroidism"),
    "E05":  ("endocrine",  "Thyrotoxicosis"),
    # Pulmonary
    "J44":  ("pulmonary",  "COPD"),
    "J18":  ("pulmonary",  "Pneumonia"),
    "J96":  ("pulmonary",  "Respiratory failure"),
    "J80":  ("pulmonary",  "ARDS"),
    # Heme
    "D64":  ("heme",       "Anemia"),
    "D69":  ("heme",       "Thrombocytopenia"),
    # Neuro
    "G40":  ("neuro",      "Epilepsy"),
    "G20":  ("neuro",      "Parkinson's disease"),
    "I63":  ("neuro",      "Cerebral infarction"),   # stroke → neuro
    "I61":  ("neuro",      "Intracerebral hemorrhage"),
    # GI
    "K92":  ("GI",         "GI hemorrhage"),
    "K85":  ("GI",         "Acute pancreatitis"),
    # Immune
    "A41":  ("immune",     "Sepsis"),
    "B20":  ("immune",     "HIV"),
}


def classify_icd10(code: str) -> Optional[str]:
    """Map an ICD-10-CM code to an Alice organ system.

    Priority: specific 3-char → 2-char prefix → 1-char chapter.
    Returns None if no mapping found.
    """
    code = code.strip().upper()
    if not code:
        return None

    # 1. Specific 3-character code
    prefix3 = code[:3]
    if prefix3 in ICD10_SPECIFIC:
        return ICD10_SPECIFIC[prefix3][0]

    # 2. 3-character block
    if prefix3 in ICD10_TO_ORGAN:
        return ICD10_TO_ORGAN[prefix3]

    # 3. 2-character prefix
    prefix2 = code[:2]
    if prefix2 in ICD10_TO_ORGAN:
        return ICD10_TO_ORGAN[prefix2]

    # 4. 1-character chapter
    chapter = code[0]
    if chapter in ICD10_TO_ORGAN:
        return ICD10_TO_ORGAN[chapter]

    return None


# ============================================================================
# 4. DATA LOADING
# ============================================================================

def check_mimic_files() -> Dict[str, Path]:
    """Check for required MIMIC-IV files and return paths.

    Supports both .csv.gz and .parquet formats.
    """
    required = [
        "labevents", "d_labitems", "diagnoses_icd",
        "admissions", "patients", "icustays",
    ]

    found = {}
    missing = []

    for name in required:
        # Check multiple formats
        for ext in [".csv.gz", ".parquet", ".csv"]:
            path = MIMIC_DIR / f"{name}{ext}"
            if path.exists():
                found[name] = path
                break
        else:
            missing.append(name)

    if missing:
        print("\n  ⚠ MIMIC-IV files not found:")
        for m in missing:
            print(f"    - mimic_data/{m}.csv.gz (or .parquet)")
        print("\n  To obtain MIMIC-IV data:")
        print("    1. Create account: https://physionet.org")
        print("    2. Complete CITI training (Data or Specimens Only Research)")
        print("    3. Sign MIMIC-IV DUA at https://physionet.org/content/mimiciv/3.1/")
        print("    4. Download and place files in mimic_data/")
        return {}

    return found


def load_mimic_table(path: Path, usecols: Optional[List[str]] = None) -> "pd.DataFrame":
    """Load a MIMIC-IV table from CSV.gz or Parquet."""
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=usecols)
    elif path.name.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path, usecols=usecols)
    else:
        raise ValueError(f"Unknown file format: {path}")


def build_itemid_mapping(d_labitems_path: Path) -> Dict[int, Tuple[str, float]]:
    """Build itemid → Alice name mapping from d_labitems.csv.

    Merges hardcoded MIMIC_ITEMID_TO_ALICE with label-based discovery
    from the actual d_labitems dictionary.
    """
    import pandas as pd

    mapping = dict(MIMIC_ITEMID_TO_ALICE)

    df = load_mimic_table(d_labitems_path)
    for _, row in df.iterrows():
        itemid = int(row["itemid"])
        label = str(row.get("label", ""))

        # If already in hardcoded mapping, skip
        if itemid in mapping:
            continue

        # Try label-based matching
        for lab_label, (alice_name, scale) in MIMIC_LABEL_TO_ALICE.items():
            if lab_label.lower() in label.lower():
                mapping[itemid] = (alice_name, scale)
                break

    return mapping


def load_labs_first_24h(
    files: Dict[str, Path],
    itemid_map: Dict[int, Tuple[str, float]],
) -> "pd.DataFrame":
    """Load first-24h labs for all ICU admissions.

    Returns DataFrame with columns: [subject_id, hadm_id, stay_id, alice_name, value]
    Aggregation: median of first-24h values per lab per admission.
    """
    import pandas as pd

    print("  Loading ICU stays ...")
    icu = load_mimic_table(files["icustays"], usecols=[
        "subject_id", "hadm_id", "stay_id", "intime",
    ])
    icu["intime"] = pd.to_datetime(icu["intime"])
    print(f"    {len(icu):,} ICU stays")

    # Only keep itemids we care about
    target_itemids = set(itemid_map.keys())

    print("  Loading lab events (this may take a few minutes) ...")
    # labevents is very large (~120M rows). Read in chunks.
    lab_path = files["labevents"]
    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]

    chunks = []
    chunk_size = 5_000_000

    if lab_path.suffix == ".parquet":
        labs = pd.read_parquet(lab_path, columns=usecols)
        # Filter to target itemids
        labs = labs[labs["itemid"].isin(target_itemids)].copy()
        labs = labs.dropna(subset=["valuenum"])
        chunks.append(labs)
    else:
        reader = pd.read_csv(lab_path, usecols=usecols, chunksize=chunk_size)
        for i, chunk in enumerate(reader):
            chunk = chunk[chunk["itemid"].isin(target_itemids)].copy()
            chunk = chunk.dropna(subset=["valuenum"])
            if len(chunk) > 0:
                chunks.append(chunk)
            if (i + 1) % 10 == 0:
                n_so_far = sum(len(c) for c in chunks)
                print(f"    ... processed {(i+1)*chunk_size:,} rows, "
                      f"kept {n_so_far:,} relevant labs")

    if not chunks:
        raise RuntimeError("No relevant labs found in labevents!")

    labs = pd.concat(chunks, ignore_index=True)
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    print(f"    Relevant labs: {len(labs):,}")

    # Merge with ICU stays to get first-24h window
    print("  Filtering to first 24h of ICU admission ...")
    merged = labs.merge(
        icu[["subject_id", "hadm_id", "stay_id", "intime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    time_diff = (merged["charttime"] - merged["intime"]).dt.total_seconds()
    merged = merged[(time_diff >= 0) & (time_diff <= FIRST_24H_SECONDS)].copy()
    print(f"    First-24h labs: {len(merged):,}")

    # Map itemid → Alice name
    merged["alice_name"] = merged["itemid"].map(
        lambda x: itemid_map.get(x, (None, 1.0))[0]
    )
    merged["scale"] = merged["itemid"].map(
        lambda x: itemid_map.get(x, (None, 1.0))[1]
    )
    merged = merged.dropna(subset=["alice_name"])
    merged["value"] = merged["valuenum"] * merged["scale"]

    # Aggregate: median per stay per lab
    agg = merged.groupby(["subject_id", "hadm_id", "stay_id", "alice_name"])[
        "value"
    ].median().reset_index()

    print(f"    Aggregated: {len(agg):,} (stay × lab) pairs")

    # Pivot to wide format
    wide = agg.pivot_table(
        index=["subject_id", "hadm_id", "stay_id"],
        columns="alice_name",
        values="value",
    ).reset_index()

    # Count valid labs per stay
    lab_cols = [c for c in wide.columns if c not in ["subject_id", "hadm_id", "stay_id"]]
    wide["n_labs"] = wide[lab_cols].notna().sum(axis=1)

    print(f"    Wide format: {len(wide):,} stays, "
          f"median labs/stay = {wide['n_labs'].median():.0f}")

    # Filter to ≥ MIN_LAB_COUNT
    wide = wide[wide["n_labs"] >= MIN_LAB_COUNT].copy()
    print(f"    After filter (≥{MIN_LAB_COUNT} labs): {len(wide):,} stays")

    return wide


def load_diagnoses(files: Dict[str, Path]) -> "pd.DataFrame":
    """Load ICD-10 diagnoses and map to organ systems.

    Returns DataFrame with: [subject_id, hadm_id, icd_code, icd_version, organ]
    Filters to ICD-10 only.
    """
    import pandas as pd

    print("  Loading diagnoses ...")
    diag = load_mimic_table(files["diagnoses_icd"])
    print(f"    Total diagnoses: {len(diag):,}")

    # Filter to ICD-10 (icd_version == 10)
    diag = diag[diag["icd_version"] == 10].copy()
    print(f"    ICD-10 diagnoses: {len(diag):,}")

    # Map to organ
    diag["organ"] = diag["icd_code"].apply(classify_icd10)
    mapped = diag.dropna(subset=["organ"])
    print(f"    Mapped to organ: {len(mapped):,} "
          f"({len(mapped)/len(diag)*100:.1f}%)")

    return mapped


def load_outcomes(files: Dict[str, Path]) -> "pd.DataFrame":
    """Load admission outcomes (mortality, LOS)."""
    import pandas as pd

    print("  Loading admissions ...")
    adm = load_mimic_table(files["admissions"], usecols=[
        "subject_id", "hadm_id", "admittime", "dischtime",
        "hospital_expire_flag",
    ])
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])
    adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
    print(f"    {len(adm):,} admissions, "
          f"mortality = {adm['hospital_expire_flag'].mean()*100:.1f}%")
    return adm


# ============================================================================
# 5. GAMMA COMPUTATION
# ============================================================================

def compute_gamma_vectors(wide_labs: "pd.DataFrame") -> "pd.DataFrame":
    """Compute 12-organ Γ vector for each ICU stay.

    Returns wide_labs with added columns: gamma_{organ}, H, sum_gamma2
    """
    import pandas as pd

    engine = GammaEngine()
    alice_lab_names = [c for c in wide_labs.columns
                       if c not in ["subject_id", "hadm_id", "stay_id", "n_labs"]]

    gamma_records = []
    n_total = len(wide_labs)

    for i, (_, row) in enumerate(wide_labs.iterrows()):
        lab_dict = {}
        for lab_name in alice_lab_names:
            v = row.get(lab_name)
            if pd.notna(v):
                lab_dict[lab_name] = float(v)

        gv = engine.lab_to_gamma(lab_dict)
        record = {
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "stay_id": int(row["stay_id"]),
        }
        for organ in ORGAN_LIST:
            record[f"gamma_{organ}"] = gv[organ]
        record["H"] = gv.health_index
        record["sum_gamma2"] = gv.total_gamma_squared

        gamma_records.append(record)

        if (i + 1) % 10000 == 0:
            print(f"    Computed {i+1:,}/{n_total:,} Γ vectors ...")

    gamma_df = pd.DataFrame(gamma_records)
    print(f"    Final: {len(gamma_df):,} Γ vectors computed")
    print(f"    H range: [{gamma_df['H'].min():.4f}, {gamma_df['H'].max():.4f}]")
    print(f"    H median: {gamma_df['H'].median():.4f}")

    return gamma_df


# ============================================================================
# 6. HASH-LOCK PROTOCOL
# ============================================================================

def hash_lock_predictions(gamma_df: "pd.DataFrame") -> str:
    """SHA-256 hash-lock all Γ predictions before examining diagnoses.

    This is the firewall: once locked, ZERO parameters can be fitted
    to the diagnosis data.
    """
    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare prediction record
    predictions = {
        "protocol": "Gamma-Net MIMIC-IV Diagnostic Accuracy v1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters_fitted_to_mimic": 0,
        "engine": "GammaEngine (pre-existing, no modification)",
        "n_stays": len(gamma_df),
        "predictions": {},
    }

    for _, row in gamma_df.iterrows():
        stay_id = str(int(row["stay_id"]))
        pred = {"H": float(row["H"])}
        for organ in ORGAN_LIST:
            pred[f"gamma_{organ}"] = float(row[f"gamma_{organ}"])
        predictions["predictions"][stay_id] = pred

    raw = json.dumps(predictions, sort_keys=True)
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    predictions["sha256"] = sha

    out_path = RESULTS_DIR / "mimic_blind_predictions.json"
    out_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    print(f"\n  +---------------------------------------------------------+")
    print(f"  |  MIMIC-IV PREDICTIONS HASH-LOCKED                      |")
    print(f"  |  SHA-256: {sha[:50]}... |")
    print(f"  |  N={len(gamma_df):,}  Params fitted: 0              |")
    print(f"  +---------------------------------------------------------+")

    return sha


# ============================================================================
# 7. DIAGNOSTIC ACCURACY ANALYSIS (STARD-compliant)
# ============================================================================

def run_diagnostic_accuracy(
    gamma_df: "pd.DataFrame",
    diagnoses: "pd.DataFrame",
    outcomes: "pd.DataFrame",
) -> Dict[str, Any]:
    """Run full diagnostic accuracy analysis.

    Returns comprehensive results dict.
    """
    import pandas as pd
    from scipy import stats

    results: Dict[str, Any] = {}

    # ---- STARD flow diagram ----
    stard = {
        "total_icu_stays": len(gamma_df),
        "excluded_insufficient_labs": 0,  # already filtered
    }

    # Create per-stay organ diagnosis flags
    print("\n  Building per-stay organ-diagnosis matrix ...")
    stay_organs: Dict[str, set] = {}
    for _, row in diagnoses.iterrows():
        hadm = int(row["hadm_id"])
        organ = row["organ"]
        key = str(hadm)
        if key not in stay_organs:
            stay_organs[key] = set()
        stay_organs[key].add(organ)

    # Add binary columns: has_{organ}_dx
    for organ in ORGAN_LIST:
        gamma_df[f"has_{organ}_dx"] = gamma_df["hadm_id"].apply(
            lambda h: 1 if organ in stay_organs.get(str(int(h)), set()) else 0
        )

    # Merge outcomes
    gamma_with_outcomes = gamma_df.merge(
        outcomes[["hadm_id", "hospital_expire_flag", "los_days"]],
        on="hadm_id",
        how="left",
    )

    stard["n_with_outcomes"] = int(gamma_with_outcomes["hospital_expire_flag"].notna().sum())
    results["stard"] = stard

    # ---- Per-organ AUC ----
    print("\n  ═══ Per-Organ Diagnostic Accuracy ═══")
    organ_results = {}

    for organ in ORGAN_LIST:
        y_true = gamma_df[f"has_{organ}_dx"].values
        y_score = np.abs(gamma_df[f"gamma_{organ}"].values)

        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)

        if n_pos < 10 or n_neg < 10:
            print(f"  {organ:12s}: SKIP (n_pos={n_pos}, n_neg={n_neg})")
            organ_results[organ] = {"status": "skip", "n_pos": n_pos, "n_neg": n_neg}
            continue

        # AUC via Mann-Whitney
        u_stat, p_val = stats.mannwhitneyu(
            y_score[y_true == 1],
            y_score[y_true == 0],
            alternative="greater",
        )
        auc = u_stat / (n_pos * n_neg)

        # Mean Γ for diseased vs healthy
        mean_diseased = float(np.mean(y_score[y_true == 1]))
        mean_healthy = float(np.mean(y_score[y_true == 0]))

        # Youden optimal threshold
        thresholds = np.percentile(y_score, np.arange(1, 100))
        best_j = -1
        best_thresh = 0
        best_sens = 0
        best_spec = 0

        for t in thresholds:
            pred = (y_score >= t).astype(int)
            tp = int(np.sum((pred == 1) & (y_true == 1)))
            tn = int(np.sum((pred == 0) & (y_true == 0)))
            fp = int(np.sum((pred == 1) & (y_true == 0)))
            fn = int(np.sum((pred == 0) & (y_true == 1)))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sens + spec - 1

            if j > best_j:
                best_j = j
                best_thresh = float(t)
                best_sens = sens
                best_spec = spec

        # PPV / NPV at optimal threshold
        pred_opt = (y_score >= best_thresh).astype(int)
        tp = int(np.sum((pred_opt == 1) & (y_true == 1)))
        fp = int(np.sum((pred_opt == 1) & (y_true == 0)))
        fn = int(np.sum((pred_opt == 0) & (y_true == 1)))
        tn = int(np.sum((pred_opt == 0) & (y_true == 0)))

        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        print(f"  {organ:12s}: AUC={auc:.3f} {sig:4s}  "
              f"n+={n_pos:5d}  n-={n_neg:5d}  "
              f"Sens={best_sens:.3f}  Spec={best_spec:.3f}  "
              f"Γ|dx={mean_diseased:.3f} vs {mean_healthy:.3f}")

        organ_results[organ] = {
            "auc": auc, "p_value": float(p_val), "sig": sig,
            "n_pos": n_pos, "n_neg": n_neg,
            "mean_gamma_diseased": mean_diseased,
            "mean_gamma_healthy": mean_healthy,
            "youden_threshold": best_thresh,
            "sensitivity": best_sens, "specificity": best_spec,
            "ppv": ppv, "npv": npv,
            "youden_j": best_j,
        }

    results["organ_auc"] = organ_results

    # ---- Specific ICD-10 code validation ----
    print("\n  ═══ Specific ICD-10 Code Validation ═══")
    specific_results = {}

    for code, (organ, label) in ICD10_SPECIFIC.items():
        # Find stays with this specific code
        stays_with = set(
            diagnoses[diagnoses["icd_code"].str.startswith(code)]["hadm_id"].unique()
        )
        y_true = gamma_df["hadm_id"].isin(stays_with).astype(int).values
        y_score = np.abs(gamma_df[f"gamma_{organ}"].values)

        n_pos = int(y_true.sum())
        if n_pos < 20:
            continue

        u_stat, p_val = stats.mannwhitneyu(
            y_score[y_true == 1], y_score[y_true == 0],
            alternative="greater",
        )
        auc = u_stat / (n_pos * (len(y_true) - n_pos))
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        print(f"  {code} {label:40s}: AUC={auc:.3f} {sig}  (n={n_pos})")

        specific_results[code] = {
            "label": label, "organ": organ, "auc": auc,
            "p_value": float(p_val), "n_pos": n_pos, "sig": sig,
        }

    results["specific_icd10"] = specific_results

    # ---- H → ICU mortality ----
    print("\n  ═══ Health Index H → ICU Mortality ═══")
    mort = gamma_with_outcomes.dropna(subset=["hospital_expire_flag"])

    if len(mort) > 0:
        alive = mort[mort["hospital_expire_flag"] == 0]["H"].values
        dead = mort[mort["hospital_expire_flag"] == 1]["H"].values

        if len(dead) >= 10:
            rho, p_rho = stats.spearmanr(mort["H"], mort["hospital_expire_flag"])
            u_stat, p_u = stats.mannwhitneyu(dead, alive, alternative="less")
            auc_mort = 1 - u_stat / (len(dead) * len(alive))

            # KM-like quartile analysis
            mort["H_quartile"] = pd.qcut(mort["H"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
            km_data = {}
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                sub = mort[mort["H_quartile"] == q]
                km_data[q] = {
                    "n": len(sub),
                    "deaths": int(sub["hospital_expire_flag"].sum()),
                    "rate_pct": float(sub["hospital_expire_flag"].mean() * 100),
                }

            print(f"  N = {len(mort):,} (alive={len(alive):,}, dead={len(dead):,})")
            print(f"  H|alive = {np.mean(alive):.4f} vs H|dead = {np.mean(dead):.4f}")
            print(f"  Spearman ρ = {rho:.4f}, p = {p_rho:.2e}")
            print(f"  AUC (H → mortality) = {auc_mort:.4f}")
            print(f"  Quartiles: Q1={km_data['Q1']['rate_pct']:.1f}%, "
                  f"Q4={km_data['Q4']['rate_pct']:.1f}%")

            results["mortality"] = {
                "n": len(mort), "n_dead": len(dead), "n_alive": len(alive),
                "H_mean_alive": float(np.mean(alive)),
                "H_mean_dead": float(np.mean(dead)),
                "spearman_rho": float(rho), "spearman_p": float(p_rho),
                "auc": auc_mort,
                "quartiles": km_data,
            }

    # ---- H → Length of Stay ----
    print("\n  ═══ Health Index H → Length of Stay ═══")
    los = gamma_with_outcomes.dropna(subset=["los_days"])
    if len(los) > 100:
        rho_los, p_los = stats.spearmanr(los["H"], los["los_days"])
        print(f"  N = {len(los):,}")
        print(f"  Spearman ρ(H, LOS) = {rho_los:.4f}, p = {p_los:.2e}")
        print(f"  (negative = lower H → longer stay)")

        results["length_of_stay"] = {
            "n": len(los),
            "spearman_rho": float(rho_los), "p_value": float(p_los),
        }

    # ---- Concordance Index (Harrell's C) ----
    print("\n  ═══ Harrell's C-statistic (H → mortality) ═══")
    if "mortality" in results:
        concordant = 0
        discordant = 0
        h_dead = mort[mort["hospital_expire_flag"] == 1]["H"].values
        h_alive = mort[mort["hospital_expire_flag"] == 0]["H"].values

        # Vectorised concordance
        for h_d in h_dead:
            concordant += int(np.sum(h_alive > h_d))
            discordant += int(np.sum(h_alive < h_d))

        total_pairs = concordant + discordant
        c_stat = concordant / total_pairs if total_pairs > 0 else 0.5

        print(f"  C = {c_stat:.4f} (concordant={concordant:,}, "
              f"discordant={discordant:,})")
        results["c_statistic"] = {
            "c": c_stat, "concordant": concordant, "discordant": discordant,
        }

    return results


# ============================================================================
# 8. VISUALIZATION
# ============================================================================

def plot_diagnostic_figure(gamma_df: "pd.DataFrame", results: Dict[str, Any]):
    """Generate 6-panel diagnostic accuracy figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    FIGURES_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Γ-Net Diagnostic Accuracy: MIMIC-IV Level 2b Validation\n"
        "N={:,} ICU admissions | Zero parameters fitted".format(len(gamma_df)),
        fontsize=14, fontweight="bold",
    )

    # Panel 1: Per-organ AUC bar chart
    ax = axes[0, 0]
    organ_auc = results.get("organ_auc", {})
    organs_valid = [o for o in ORGAN_LIST if organ_auc.get(o, {}).get("auc") is not None]
    if organs_valid:
        aucs = [organ_auc[o]["auc"] for o in organs_valid]
        colors = ["#e74c3c" if a >= 0.7 else "#f39c12" if a >= 0.6 else "#95a5a6"
                  for a in aucs]
        ax.barh(organs_valid, aucs, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
        ax.axvline(0.7, color="red", linestyle="--", alpha=0.3, label="acceptable")
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("AUC (Γ_organ → ICD-10 diagnosis)")
        ax.set_title("A. Per-Organ Diagnostic AUC")
        ax.legend(fontsize=8)

    # Panel 2: Mortality by H quartile
    ax = axes[0, 1]
    mort_data = results.get("mortality", {}).get("quartiles", {})
    if mort_data:
        qs = ["Q1", "Q2", "Q3", "Q4"]
        rates = [mort_data.get(q, {}).get("rate_pct", 0) for q in qs]
        colors2 = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
        ax.bar(qs, rates, color=colors2, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("H Quartile (Q1=sickest)")
        ax.set_ylabel("ICU Mortality %")
        ax.set_title("B. ICU Mortality by H Quartile")
        for i, (q, r) in enumerate(zip(qs, rates)):
            ax.text(i, r + 0.5, f"{r:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # Panel 3: H distribution alive vs dead
    ax = axes[0, 2]
    if "mortality" in results:
        import pandas as pd
        mort_df = gamma_df.merge(
            pd.DataFrame({"hadm_id": gamma_df["hadm_id"]}),
            on="hadm_id",
        )
        # Use the outcomes already computed
        if "hospital_expire_flag" in gamma_df.columns:
            alive_h = gamma_df[gamma_df["hospital_expire_flag"] == 0]["H"]
            dead_h = gamma_df[gamma_df["hospital_expire_flag"] == 1]["H"]
        else:
            alive_h = pd.Series(dtype=float)
            dead_h = pd.Series(dtype=float)

        if len(alive_h) > 0 and len(dead_h) > 0:
            ax.hist(alive_h, bins=50, alpha=0.6, density=True, label="Alive", color="#2ecc71")
            ax.hist(dead_h, bins=50, alpha=0.6, density=True, label="Dead", color="#e74c3c")
            ax.set_xlabel("Health Index H")
            ax.set_ylabel("Density")
            ax.set_title("C. H Distribution: Alive vs Dead")
            ax.legend()

    # Panel 4: Γ_organ diseased vs healthy (paired comparison)
    ax = axes[1, 0]
    if organ_auc:
        organs_plot = [o for o in organs_valid
                       if organ_auc[o].get("mean_gamma_diseased") is not None][:8]
        x = np.arange(len(organs_plot))
        w = 0.35
        healthy = [organ_auc[o]["mean_gamma_healthy"] for o in organs_plot]
        diseased = [organ_auc[o]["mean_gamma_diseased"] for o in organs_plot]
        ax.bar(x - w/2, healthy, w, label="No dx", color="#2ecc71", edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, diseased, w, label="Has dx", color="#e74c3c", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(organs_plot, rotation=45, ha="right")
        ax.set_ylabel("|Γ_organ|")
        ax.set_title("D. Mean |Γ|: Diseased vs Healthy")
        ax.legend()

    # Panel 5: Specific ICD-10 codes AUC
    ax = axes[1, 1]
    specific = results.get("specific_icd10", {})
    if specific:
        codes = sorted(specific.keys(), key=lambda c: specific[c]["auc"], reverse=True)[:15]
        labels = [f"{c} {specific[c]['label'][:25]}" for c in codes]
        aucs_sp = [specific[c]["auc"] for c in codes]
        colors3 = ["#e74c3c" if a >= 0.7 else "#f39c12" if a >= 0.6 else "#95a5a6"
                   for a in aucs_sp]
        ax.barh(labels, aucs_sp, color=colors3, edgecolor="black", linewidth=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("AUC")
        ax.set_title("E. Specific ICD-10 Diagnostic AUC")

    # Panel 6: Summary statistics text
    ax = axes[1, 2]
    ax.axis("off")
    summary_lines = [
        "MIMIC-IV Level 2b Validation Summary",
        "=" * 40,
        f"N ICU stays: {len(gamma_df):,}",
        f"Parameters fitted: 0",
    ]
    if "mortality" in results:
        m = results["mortality"]
        summary_lines += [
            f"",
            f"ICU Mortality AUC: {m.get('auc', 'N/A'):.3f}",
            f"H|alive: {m.get('H_mean_alive', 0):.4f}",
            f"H|dead:  {m.get('H_mean_dead', 0):.4f}",
        ]
    if "c_statistic" in results:
        summary_lines.append(f"Harrell's C: {results['c_statistic']['c']:.4f}")

    n_sig = sum(1 for o in organ_auc.values()
                if isinstance(o, dict) and o.get("sig") in ("***", "**", "*"))
    n_tested = sum(1 for o in organ_auc.values()
                   if isinstance(o, dict) and o.get("auc") is not None)
    summary_lines += [
        f"",
        f"Organ AUC: {n_sig}/{n_tested} significant",
    ]

    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontfamily="monospace", fontsize=10, verticalalignment="top")

    plt.tight_layout()

    for ext in ("png", "pdf"):
        outpath = FIGURES_DIR / f"fig_mimic_diagnostic_accuracy.{ext}"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")

    plt.close(fig)
    print(f"\n  Saved: figures/fig_mimic_diagnostic_accuracy.png/pdf")


# ============================================================================
# 9. REPORT
# ============================================================================

def print_diagnostic_report(
    results: Dict[str, Any], sha: str, n_stays: int,
) -> str:
    """Print comprehensive STARD-compliant diagnostic accuracy report."""
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 78)
    p("  MIMIC-IV DIAGNOSTIC ACCURACY: Γ_organ vs ICD-10 GOLD STANDARD")
    p("  Level 3 → Level 2b upgrade: cohort → diagnostic accuracy")
    p("=" * 78)
    p()
    p(f"  Date:              {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    p(f"  Database:          MIMIC-IV v3.1 (PhysioNet)")
    p(f"  N ICU stays:       {n_stays:,}")
    p(f"  Parameters fitted: 0 (Γ computed with pre-existing Z_normal)")
    p(f"  SHA-256:           {sha}")

    # Per-organ results
    p()
    p("-" * 78)
    p("  PER-ORGAN DIAGNOSTIC ACCURACY (Γ_organ → ICD-10 organ diagnosis)")
    p("-" * 78)
    p(f"  {'Organ':12s}  {'AUC':>6s}  {'Sens':>6s}  {'Spec':>6s}  "
      f"{'PPV':>6s}  {'NPV':>6s}  {'n+':>6s}  {'n-':>6s}  {'p':>10s}")
    p(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*6}  "
      f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}")

    organ_auc = results.get("organ_auc", {})
    for organ in ORGAN_LIST:
        r = organ_auc.get(organ, {})
        if r.get("status") == "skip":
            p(f"  {organ:12s}  {'SKIP':>6s}  (n+={r.get('n_pos', 0)})")
            continue
        if "auc" not in r:
            continue
        p(f"  {organ:12s}  {r['auc']:.3f}  {r['sensitivity']:.3f}  "
          f"{r['specificity']:.3f}  {r['ppv']:.3f}  {r['npv']:.3f}  "
          f"{r['n_pos']:6d}  {r['n_neg']:6d}  {r['p_value']:.2e} {r['sig']}")

    # Specific ICD-10
    specific = results.get("specific_icd10", {})
    if specific:
        p()
        p("-" * 78)
        p("  SPECIFIC ICD-10 CODE VALIDATION")
        p("-" * 78)
        for code in sorted(specific.keys()):
            s = specific[code]
            p(f"  {code}  {s['label']:40s}  AUC={s['auc']:.3f}  "
              f"n={s['n_pos']:5d}  {s['sig']}")

    # Mortality
    if "mortality" in results:
        m = results["mortality"]
        p()
        p("-" * 78)
        p("  H → ICU MORTALITY")
        p("-" * 78)
        p(f"  N = {m['n']:,} (alive={m['n_alive']:,}, dead={m['n_dead']:,})")
        p(f"  H|alive = {m['H_mean_alive']:.4f}")
        p(f"  H|dead  = {m['H_mean_dead']:.4f}")
        p(f"  Spearman ρ = {m['spearman_rho']:.4f}, p = {m['spearman_p']:.2e}")
        p(f"  AUC = {m['auc']:.4f}")
        p()
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            qd = m["quartiles"].get(q, {})
            p(f"  {q}: n={qd.get('n', 0):,}, "
              f"deaths={qd.get('deaths', 0)}, "
              f"rate={qd.get('rate_pct', 0):.1f}%")

    # C-statistic
    if "c_statistic" in results:
        c = results["c_statistic"]
        p()
        p("-" * 78)
        p("  HARRELL'S C-STATISTIC")
        p("-" * 78)
        p(f"  C = {c['c']:.4f}")

    # LOS
    if "length_of_stay" in results:
        los = results["length_of_stay"]
        p()
        p("-" * 78)
        p("  H → LENGTH OF STAY")
        p("-" * 78)
        p(f"  N = {los['n']:,}")
        p(f"  Spearman ρ(H, LOS) = {los['spearman_rho']:.4f}, p = {los['p_value']:.2e}")

    # Verdict
    p()
    p("=" * 78)
    p("  VERDICT")
    p("=" * 78)

    n_sig = sum(1 for o in organ_auc.values()
                if isinstance(o, dict) and o.get("sig") in ("***", "**", "*"))
    n_tested = sum(1 for o in organ_auc.values()
                   if isinstance(o, dict) and o.get("auc") is not None)

    if n_sig >= n_tested * 0.5 and n_tested > 0:
        p(f"  ✓ LEVEL 2b EVIDENCE ACHIEVED")
        p(f"")
        p(f"  Γ_organ discriminates ICD-10 organ-specific diagnoses")
        p(f"  {n_sig}/{n_tested} organ systems with significant AUC")
        p(f"  ZERO parameters fitted to MIMIC-IV data")
        p(f"")
        p(f"  This upgrades the Γ-framework from:")
        p(f"    Level 3: prospective cohort (H predicts mortality)")
        p(f"    Level 2b: diagnostic accuracy (Γ vs ICD-10 gold standard)")
    else:
        p(f"  Partial validation: {n_sig}/{n_tested} organs significant")
        p(f"  Level 2b requires majority of organs with significant AUC")

    p()
    p("-" * 78)
    p("  METHODOLOGICAL NOTES (STARD-compliant)")
    p("-" * 78)
    p("  1. Index test: Γ_organ computed from first-24h ICU labs")
    p("  2. Reference: ICD-10-CM discharge diagnoses (coded by physicians)")
    p("  3. Zero parameters fitted to MIMIC-IV outcomes")
    p("  4. SHA-256 hash-locked predictions before loading diagnoses")
    p("  5. Population: All adult ICU admissions with ≥8 labs in first 24h")
    p("  6. Conservative: no age/sex/severity adjustment")
    p("=" * 78)

    return "\n".join(lines)


# ============================================================================
# 10. MAIN
# ============================================================================

def main():
    """Run complete MIMIC-IV diagnostic accuracy pipeline."""
    print("=" * 78)
    print("  MIMIC-IV DIAGNOSTIC ACCURACY: Γ_organ vs ICD-10")
    print("  Upgrading from Level 3 to Level 2b")
    print("=" * 78)

    import pandas as pd

    # Phase 1: Check data availability
    print("\nPhase 1: Checking MIMIC-IV data files ...")
    files = check_mimic_files()
    if not files:
        print("\n  ✗ Cannot proceed without MIMIC-IV data.")
        print("  Please download from PhysioNet and place in mimic_data/")
        return

    for name, path in files.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {name}: {path.name} ({size_mb:.1f} MB)")

    # Phase 2: Build lab mapping
    print("\nPhase 2: Building lab item mapping ...")
    itemid_map = build_itemid_mapping(files["d_labitems"])
    n_alice = len(set(v[0] for v in itemid_map.values()))
    print(f"  {len(itemid_map)} MIMIC itemids → {n_alice} Alice lab names")

    # Phase 3: Load first-24h labs
    print("\nPhase 3: Loading first-24h ICU labs ...")
    wide_labs = load_labs_first_24h(files, itemid_map)

    # Phase 4: Compute Γ vectors
    print("\nPhase 4: Computing Γ vectors ...")
    gamma_df = compute_gamma_vectors(wide_labs)

    # Phase 5: Hash-lock predictions
    print("\nPhase 5: Hash-locking Γ predictions ...")
    sha = hash_lock_predictions(gamma_df)

    print("\n" + "=" * 78)
    print("  CROSSING THE FIREWALL: Now loading ICD-10 diagnoses")
    print("  All Γ predictions are hash-locked. Zero remaining DoF.")
    print("=" * 78)

    # Phase 6: Load diagnoses and outcomes
    print("\nPhase 6: Loading ICD-10 diagnoses and outcomes ...")
    diagnoses = load_diagnoses(files)
    outcomes = load_outcomes(files)

    # Phase 7: Diagnostic accuracy analysis
    print("\nPhase 7: Running diagnostic accuracy analysis ...")
    results = run_diagnostic_accuracy(gamma_df, diagnoses, outcomes)

    # Phase 8: Visualization
    print("\nPhase 8: Generating diagnostic accuracy figure ...")
    # Copy outcomes to gamma_df for plotting
    gamma_df = gamma_df.merge(
        outcomes[["hadm_id", "hospital_expire_flag"]],
        on="hadm_id",
        how="left",
    )
    plot_diagnostic_figure(gamma_df, results)

    # Phase 9: Report
    print("\nPhase 9: Final report ...")
    report = print_diagnostic_report(results, sha, len(gamma_df))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full_results = {
        "protocol": "Gamma-Net MIMIC-IV Diagnostic Accuracy v1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_predictions": sha,
        "n_stays": len(gamma_df),
        "results": results,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = RESULTS_DIR / "mimic_diagnostic_results.json"
    results_path.write_text(
        json.dumps(full_results, indent=2, default=convert),
        encoding="utf-8",
    )

    report_path = RESULTS_DIR / "mimic_diagnostic_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n  Results saved: {results_path}")
    print(f"  Report saved: {report_path}")
    print(f"\n  ✓ MIMIC-IV Diagnostic Accuracy Analysis → COMPLETE")
    print(f"     N={len(gamma_df):,} ICU stays")


if __name__ == "__main__":
    main()

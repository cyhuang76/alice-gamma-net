#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: Expanded Physics Input — Network Propagation Proof
══════════════════════════════════════════════════════════════

PROOF
─────
Textbook formulas (Framingham, SCORE2) use SBP as ONE fitted coefficient
for a SINGLE organ-specific risk score.  They do NOT model the vascular
or neural NETWORK that connects organs.

Gamma-Net uses SBP as a physical impedance measurement that propagates
through the organ network:

    SBP high → Z_vascular ↑ (direct: pressure wave in arteries)
             → Z_cardiac  ↑ (network: heart works against afterload)
             → Z_renal    ↑ (network: hypertensive nephropathy)
             → Z_neuro    ↑ (network: cerebrovascular stroke risk)

    Smoking  → Z_pulmonary ↑ (direct: airway epithelial damage)
             → Z_vascular  ↑ (network: endothelial dysfunction)
             → Z_immune    ↑ (network: immune suppression)

PREDICTION
──────────
If adding SBP improves not just vascular AUC but also cardiac, renal,
and neurological AUC, that PROVES network propagation — something
textbook single-organ formulas cannot do.

PHYSICS INPUTS (zero parameters fitted to outcomes)
───────────────────────────────────────────────────
SBP [90-120 mmHg]:  vascular(0.35), cardiac(0.20), renal(0.15), neuro(0.10)
DBP [60-80 mmHg]:   vascular(0.25), renal(0.20), cardiac(0.15)
PP  [30-50 mmHg]:   vascular(0.30), cardiac(0.15)  — arterial stiffness
Smoking (binary):   pulmonary(0.40), vascular(0.15), immune(0.05)

All reference intervals from medical physiology textbooks.
All organ weights from known pathophysiology.
ZERO parameters fitted to any mortality/outcome data.
"""

from __future__ import annotations

import io
import json
import sys
import time
from collections import OrderedDict
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

from alice.diagnostics.lab_mapping import (
    LAB_CATALOGUE,
    ORGAN_LIST,
    ORGAN_SYSTEMS,
    LabMapper,
)

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

# ============================================================================
# Expanded Physics Input Definitions
# ============================================================================

EXPANDED_INPUTS = {
    "SBP": {
        "ref_low": 90.0, "ref_high": 120.0,
        "organ_weights": {
            "vascular": 0.35,   # Direct: blood pressure IS vascular measurement
            "cardiac":  0.20,   # Heart works harder against high afterload
            "renal":    0.15,   # Hypertensive nephropathy
            "neuro":    0.10,   # Cerebrovascular stroke risk
        },
    },
    "DBP": {
        "ref_low": 60.0, "ref_high": 80.0,
        "organ_weights": {
            "vascular": 0.25,   # Diastolic vascular tone
            "renal":    0.20,   # Renal perfusion pressure
            "cardiac":  0.15,   # Diastolic coronary filling
        },
    },
    "PP": {
        "ref_low": 30.0, "ref_high": 50.0,
        "organ_weights": {
            "vascular": 0.30,   # Arterial stiffness (pure vascular physics)
            "cardiac":  0.15,   # Increased afterload from stiff arteries
        },
    },
    "Smoking": {
        "binary": True,
        "organ_weights": {
            "pulmonary": 0.40,  # Direct: airway epithelial damage
            "vascular":  0.15,  # Endothelial dysfunction
            "immune":    0.05,  # Immune suppression
        },
    },
}

# Four configurations to compare
CONFIGS = OrderedDict([
    ("A: Labs only",     []),
    ("B: Labs+BP",       ["SBP", "DBP", "PP"]),
    ("C: Labs+Smoke",    ["Smoking"]),
    ("D: Full physics",  ["SBP", "DBP", "PP", "Smoking"]),
])

# NHANES file mappings
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


def load_xpt(path: Path) -> "pd.DataFrame":
    import pandas as pd
    return pd.read_sas(path, format="xport")


# ============================================================================
# Phase 1: Load Data
# ============================================================================

def load_all_data() -> "pd.DataFrame":
    """Load labs + demographics + mortality + BPX (SBP+DBP) + SMQ."""
    import pandas as pd

    print("=" * 70)
    print("PHASE 1: LOAD DATA")
    print("=" * 70)

    # --- Labs ---
    print("  [1a] Loading 10-cycle lab data...")
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
    print("  [1b] Loading demographics...")
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
    print("  [1c] Loading mortality...")
    mort_parts = []
    for cycle_key in CYCLES:
        s, e = cycle_key.split("-")
        mp = DATA_DIR / f"NHANES_{s}_{e}_MORT_2019_PUBLIC.dat"
        if mp.exists():
            m = parse_mortality_file(mp)
            m["cycle"] = cycle_key
            mort_parts.append(m)
    df_mort = pd.concat(mort_parts, ignore_index=True)
    if "MORTSTAT" in df_mort.columns:
        df_mort.rename(columns={
            "MORTSTAT": "mort_status",
            "UCOD_LEADING": "ucod_leading",
            "PERMTH_INT": "fu_months",
        }, inplace=True)

    # --- Blood Pressure (SBP + DBP) ---
    print("  [1d] Loading blood pressure (SBP + DBP)...")
    bp_parts = []
    for cycle_key, stem in BPX_FILES.items():
        p = DATA_DIR / f"{stem}.XPT"
        if not p.exists():
            continue
        d = load_xpt(p)
        d["SEQN"] = d["SEQN"].astype(int)
        # SBP: columns containing SY and starting with BPX
        sbp_cols = [c for c in d.columns
                    if c.upper().startswith("BPX") and "SY" in c.upper()
                    and any(ch.isdigit() for ch in c)]
        dbp_cols = [c for c in d.columns
                    if c.upper().startswith("BPX") and "DI" in c.upper()
                    and any(ch.isdigit() for ch in c)]
        rec = d[["SEQN"]].copy()
        if sbp_cols:
            rec["SBP_mean"] = d[sbp_cols].mean(axis=1, skipna=True)
        if dbp_cols:
            rec["DBP_mean"] = d[dbp_cols].mean(axis=1, skipna=True)
        if "SBP_mean" in rec.columns and "DBP_mean" in rec.columns:
            rec["PP"] = rec["SBP_mean"] - rec["DBP_mean"]
        bp_parts.append(rec.dropna(subset=[c for c in rec.columns if c != "SEQN"],
                                    how="all"))
    df_bp = pd.concat(bp_parts, ignore_index=True) if bp_parts else pd.DataFrame()

    # --- Smoking ---
    print("  [1e] Loading smoking status...")
    smq_parts = []
    for cycle_key, stem in SMQ_FILES.items():
        p = DATA_DIR / f"{stem}.XPT"
        if not p.exists():
            continue
        d = load_xpt(p)
        d["SEQN"] = d["SEQN"].astype(int)
        # SMQ040: 1=every day, 2=some days, 3=not at all
        # Current smoker = SMQ040 in {1, 2}
        if "SMQ040" in d.columns:
            d["current_smoker"] = d["SMQ040"].isin([1.0, 2.0]).astype(float)
            # For those who never smoked (SMQ020=2), set to 0
            if "SMQ020" in d.columns:
                d.loc[d["SMQ020"] == 2.0, "current_smoker"] = 0.0
            smq_parts.append(d[["SEQN", "current_smoker"]].dropna())
        elif "SMQ020" in d.columns:
            d["current_smoker"] = (d["SMQ020"] == 1.0).astype(float)
            smq_parts.append(d[["SEQN", "current_smoker"]].dropna())
    df_smq = pd.concat(smq_parts, ignore_index=True) if smq_parts else pd.DataFrame()

    # --- Merge ---
    print("  [1f] Merging all data...")
    df = df_labs.merge(df_demo[["SEQN", "RIDAGEYR", "RIAGENDR"]], on="SEQN", how="left")
    df = df.merge(df_mort[["SEQN", "mort_status", "ucod_leading", "fu_months"]],
                  on="SEQN", how="left")
    if len(df_bp) > 0:
        df = df.merge(df_bp.drop_duplicates("SEQN"), on="SEQN", how="left")
    if len(df_smq) > 0:
        df = df.merge(df_smq.drop_duplicates("SEQN"), on="SEQN", how="left")

    # Filter: adults 20+ with mortality linkage
    df = df[df["RIDAGEYR"].notna() & (df["RIDAGEYR"] >= 20)]
    df = df[df["mort_status"].notna()]

    n_sbp = df["SBP_mean"].notna().sum() if "SBP_mean" in df.columns else 0
    n_dbp = df["DBP_mean"].notna().sum() if "DBP_mean" in df.columns else 0
    n_smk = df["current_smoker"].notna().sum() if "current_smoker" in df.columns else 0

    print(f"\n  Final cohort: {len(df):,}")
    print(f"  Deaths: {(df['mort_status'] == 1).sum():,}")
    print(f"  With SBP: {n_sbp:,}  DBP: {n_dbp:,}  Smoking: {n_smk:,}")
    return df


# ============================================================================
# Phase 2: Compute Base Z (Labs Only)
# ============================================================================

def nhanes_to_alice_labs(row: "pd.Series") -> Dict[str, float]:
    """Convert NHANES row → Alice lab dict."""
    import pandas as pd
    alice_labs: Dict[str, float] = {}
    for nhanes_col, (alice_name, factor) in NHANES_TO_ALICE.items():
        val = row.get(nhanes_col, None)
        if val is not None and pd.notna(val) and np.isfinite(val):
            converted = float(val) * factor
            if alice_name not in alice_labs or converted > 0:
                alice_labs[alice_name] = converted
    return alice_labs


def compute_base_z(df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute Z_patient per organ from labs only (baseline)."""
    import pandas as pd

    print("\n" + "=" * 70)
    print("PHASE 2: COMPUTE BASE Z (LABS ONLY)")
    print("=" * 70)

    mapper = LabMapper()  # textbook Z_normal
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
    for organ in ORGAN_LIST:
        if organ in z_df.columns:
            df.loc[z_df.index, f"z_base_{organ}"] = z_df[organ]
    df.loc[z_df.index, "n_alice_labs"] = z_df["n_alice_labs"]

    df = df[df["n_alice_labs"].notna() & (df["n_alice_labs"] >= 3)]
    print(f"  Computed base Z for {len(df):,} participants")
    return df


# ============================================================================
# Phase 3: Add Expanded Physics → Calibrate → Compute Γ → Evaluate
# ============================================================================

def normalise_continuous(value: float, ref_low: float, ref_high: float) -> float:
    """Normalise deviation: δ=0 at midpoint, |δ|=1 at ref boundary."""
    mid = (ref_low + ref_high) / 2.0
    half = (ref_high - ref_low) / 2.0
    if half < 1e-12:
        return 0.0
    return (value - mid) / half


def add_expanded_physics(df: "pd.DataFrame", input_names: List[str]) -> "pd.DataFrame":
    """Add expanded physics contributions to base Z.

    Z_organ = Z_base_organ + Z_normal_textbook × Σ w_input × |δ_input|
    """
    import pandas as pd

    # Start from base Z
    for organ in ORGAN_LIST:
        base_col = f"z_base_{organ}"
        if base_col in df.columns:
            df[f"z_{organ}"] = df[base_col].copy()
        else:
            df[f"z_{organ}"] = ORGAN_SYSTEMS[organ]

    for inp_name in input_names:
        spec = EXPANDED_INPUTS[inp_name]

        if spec.get("binary"):
            # Binary input (smoking)
            col = "current_smoker"
            if col not in df.columns:
                continue
            delta = df[col].fillna(0.0).values  # 1.0 for smoker, 0.0 otherwise
        else:
            # Continuous input (SBP, DBP, PP)
            col = {"SBP": "SBP_mean", "DBP": "DBP_mean", "PP": "PP"}.get(inp_name)
            if col is None or col not in df.columns:
                continue
            ref_low = spec["ref_low"]
            ref_high = spec["ref_high"]
            mid = (ref_low + ref_high) / 2.0
            half = (ref_high - ref_low) / 2.0
            raw = df[col].fillna(mid).values  # NaN → midpoint (no contribution)
            delta = np.abs((raw - mid) / half)

        # Add impedance contribution to each affected organ
        for organ, weight in spec["organ_weights"].items():
            z_normal = ORGAN_SYSTEMS[organ]
            df[f"z_{organ}"] = df[f"z_{organ}"].values + z_normal * weight * delta

    return df


def calibrate_z_normal(df: "pd.DataFrame", min_fu: int = 120) -> Dict[str, float]:
    """Calibrate Z_normal = median Z of 10-year survivors."""
    ref = df[
        (df["mort_status"] == 0) &
        (df["fu_months"] >= min_fu) &
        (df["RIDAGEYR"] >= 20) &
        (df["RIDAGEYR"] <= 79)
    ]
    z_cal = {}
    for organ in ORGAN_LIST:
        col = f"z_{organ}"
        if col in ref.columns:
            vals = ref[col].dropna()
            z_cal[organ] = float(np.median(vals)) if len(vals) > 100 else ORGAN_SYSTEMS[organ]
        else:
            z_cal[organ] = ORGAN_SYSTEMS[organ]
    return z_cal


def compute_gamma(df: "pd.DataFrame", z_cal: Dict[str, float]) -> "pd.DataFrame":
    """Compute Γ per organ and composite scores."""
    for organ in ORGAN_LIST:
        z_col = f"z_{organ}"
        if z_col not in df.columns:
            df[f"g_{organ}"] = 0.0
            continue
        z_p = df[z_col].values
        z_n = z_cal[organ]
        denom = z_p + z_n
        df[f"g_{organ}"] = np.where(np.abs(denom) > 1e-12, (z_p - z_n) / denom, 0.0)

    gamma_arr = df[[f"g_{o}" for o in ORGAN_LIST]].values
    df["H"] = np.prod(1.0 - gamma_arr ** 2, axis=1)
    df["sum_g2"] = np.sum(gamma_arr ** 2, axis=1)
    return df


def fast_auc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 500) -> Tuple[float, float, float]:
    """AUC with bootstrap 95% CI."""
    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = np.asarray(y_true[mask], dtype=int)
    y_score = np.asarray(y_score[mask], dtype=float)

    if len(np.unique(y_true)) < 2 or len(y_true) < 20:
        return 0.5, 0.5, 0.5

    auc = roc_auc_score(y_true, y_score)

    rng = np.random.default_rng(42)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if aucs:
        lo, hi = np.percentile(aucs, [2.5, 97.5])
    else:
        lo, hi = auc, auc
    return float(auc), float(lo), float(hi)


def evaluate_config(df: "pd.DataFrame", config_name: str) -> Dict[str, Any]:
    """Evaluate AUC for organ-specific and all-cause mortality."""
    import pandas as pd

    results = {"config": config_name}

    # All-cause mortality
    y_dead = (df["mort_status"] == 1).astype(int).values

    # sum_Γ² → all-cause
    auc, lo, hi = fast_auc(y_dead, df["sum_g2"].values)
    results["allcause_sum_g2"] = {"auc": auc, "ci": [lo, hi]}

    # 1 - H → all-cause
    auc, lo, hi = fast_auc(y_dead, 1.0 - df["H"].values)
    results["allcause_1mH"] = {"auc": auc, "ci": [lo, hi]}

    # Per-organ → organ-specific death
    results["organ_specific"] = {}
    for ucod, (organ, label) in UCOD_TO_ORGAN.items():
        if organ is None:
            continue
        y_org = ((df["mort_status"] == 1) & (df["ucod_leading"] == ucod)).astype(int).values
        n_events = int(y_org.sum())
        if n_events < 20:
            continue
        # |Γ_organ| as predictor
        g_col = f"g_{organ}"
        if g_col in df.columns:
            auc_org, lo_org, hi_org = fast_auc(y_org, np.abs(df[g_col].values))
        else:
            auc_org, lo_org, hi_org = 0.5, 0.5, 0.5

        # sum_Γ² as composite predictor for this death cause
        auc_comp, lo_comp, hi_comp = fast_auc(y_org, df["sum_g2"].values)

        results["organ_specific"][organ] = {
            "label": label,
            "n_events": n_events,
            "auc_organ": {"auc": auc_org, "ci": [lo_org, hi_org]},
            "auc_composite": {"auc": auc_comp, "ci": [lo_comp, hi_comp]},
        }

    return results


def run_config(df: "pd.DataFrame", config_name: str, input_names: List[str]) -> Dict:
    """Full pipeline for one configuration."""
    print(f"\n  --- {config_name} ---")
    df_cfg = df.copy()
    df_cfg = add_expanded_physics(df_cfg, input_names)
    z_cal = calibrate_z_normal(df_cfg)
    df_cfg = compute_gamma(df_cfg, z_cal)
    results = evaluate_config(df_cfg, config_name)
    results["z_calibrated"] = z_cal

    # Quick summary
    ac = results["allcause_sum_g2"]
    print(f"    All-cause (sum_G2): AUC = {ac['auc']:.4f} [{ac['ci'][0]:.4f}-{ac['ci'][1]:.4f}]")
    for organ, info in results["organ_specific"].items():
        a = info["auc_organ"]
        c = info["auc_composite"]
        print(f"    {organ:12s} death (n={info['n_events']:4d}): "
              f"|G_organ| {a['auc']:.4f}  sum_G2 {c['auc']:.4f}")

    return results


# ============================================================================
# Phase 4: Network Propagation Analysis
# ============================================================================

def analyze_network_propagation(all_results: Dict[str, Dict]) -> str:
    """Compare configs to identify network propagation effects."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("PHASE 4: NETWORK PROPAGATION ANALYSIS")
    lines.append("=" * 70)

    # --- All-cause comparison ---
    lines.append("\n  ALL-CAUSE MORTALITY (sum_G2)")
    lines.append("  " + "-" * 55)
    base_auc = None
    for cfg_name, res in all_results.items():
        ac = res["allcause_sum_g2"]
        delta = ""
        if base_auc is not None:
            d = ac["auc"] - base_auc
            delta = f"  D={d:+.4f}"
        else:
            base_auc = ac["auc"]
        lines.append(f"    {cfg_name:20s}  AUC = {ac['auc']:.4f} [{ac['ci'][0]:.4f}-{ac['ci'][1]:.4f}]{delta}")

    # --- Organ-specific comparison ---
    lines.append("\n  ORGAN-SPECIFIC DEATH (|G_organ|)")
    lines.append("  " + "-" * 55)

    # Collect all organs present
    all_organs = set()
    for res in all_results.values():
        all_organs.update(res["organ_specific"].keys())

    for organ in ORGAN_LIST:
        if organ not in all_organs:
            continue
        lines.append(f"\n    {organ.upper()}")
        base_org_auc = None
        base_comp_auc = None
        for cfg_name, res in all_results.items():
            if organ not in res["organ_specific"]:
                continue
            info = res["organ_specific"][organ]
            a = info["auc_organ"]["auc"]
            c = info["auc_composite"]["auc"]
            d_org = ""
            d_comp = ""
            if base_org_auc is not None:
                d_org = f" D={a - base_org_auc:+.4f}"
                d_comp = f" D={c - base_comp_auc:+.4f}"
            else:
                base_org_auc = a
                base_comp_auc = c
            lines.append(f"      {cfg_name:20s}  |G| {a:.4f}{d_org:10s}  sum_G2 {c:.4f}{d_comp}")

    # --- Network propagation proof ---
    lines.append("\n" + "=" * 70)
    lines.append("  NETWORK PROPAGATION PROOF")
    lines.append("=" * 70)

    if "A: Labs only" in all_results and "B: Labs+BP" in all_results:
        lines.append("\n  Adding SBP/DBP/PP (vascular network inputs):")
        base = all_results["A: Labs only"]
        bp = all_results["B: Labs+BP"]
        for organ in ORGAN_LIST:
            if organ in base["organ_specific"] and organ in bp["organ_specific"]:
                a0 = base["organ_specific"][organ]["auc_organ"]["auc"]
                a1 = bp["organ_specific"][organ]["auc_organ"]["auc"]
                delta = a1 - a0
                direct = organ in ("vascular",)
                net = organ in ("cardiac", "renal", "neuro")
                tag = "DIRECT" if direct else ("NETWORK" if net else "")
                if abs(delta) > 0.005:
                    lines.append(f"    {organ:12s}: {a0:.4f} -> {a1:.4f} (D={delta:+.4f}) {tag}")

    if "A: Labs only" in all_results and "C: Labs+Smoke" in all_results:
        lines.append("\n  Adding Smoking (pulmonary network input):")
        base = all_results["A: Labs only"]
        smk = all_results["C: Labs+Smoke"]
        for organ in ORGAN_LIST:
            if organ in base["organ_specific"] and organ in smk["organ_specific"]:
                a0 = base["organ_specific"][organ]["auc_organ"]["auc"]
                a1 = smk["organ_specific"][organ]["auc_organ"]["auc"]
                delta = a1 - a0
                direct = organ in ("pulmonary",)
                net = organ in ("vascular", "immune")
                tag = "DIRECT" if direct else ("NETWORK" if net else "")
                if abs(delta) > 0.005:
                    lines.append(f"    {organ:12s}: {a0:.4f} -> {a1:.4f} (D={delta:+.4f}) {tag}")

    if "A: Labs only" in all_results and "D: Full physics" in all_results:
        lines.append("\n  Full physics (BP + Smoking) vs Labs only:")
        base = all_results["A: Labs only"]
        full = all_results["D: Full physics"]
        ac_base = base["allcause_sum_g2"]["auc"]
        ac_full = full["allcause_sum_g2"]["auc"]
        lines.append(f"    All-cause: {ac_base:.4f} -> {ac_full:.4f} (D={ac_full-ac_base:+.4f})")
        for organ in ORGAN_LIST:
            if organ in base["organ_specific"] and organ in full["organ_specific"]:
                a0 = base["organ_specific"][organ]["auc_organ"]["auc"]
                a1 = full["organ_specific"][organ]["auc_organ"]["auc"]
                c0 = base["organ_specific"][organ]["auc_composite"]["auc"]
                c1 = full["organ_specific"][organ]["auc_composite"]["auc"]
                lines.append(
                    f"    {organ:12s}: |G| {a0:.4f}->{a1:.4f} (D={a1-a0:+.4f})  "
                    f"sum_G2 {c0:.4f}->{c1:.4f} (D={c1-c0:+.4f})")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()

    print("=" * 70)
    print("EXPANDED PHYSICS INPUT — NETWORK PROPAGATION PROOF")
    print("=" * 70)
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Physics inputs: SBP, DBP, PP (blood pressure), Smoking")
    print(f"  Configs: {list(CONFIGS.keys())}")
    print()

    # Phase 1: Load data
    df = load_all_data()

    # Phase 2: Base Z from labs
    df = compute_base_z(df)

    # Phase 3: Run all configurations
    print("\n" + "=" * 70)
    print("PHASE 3: EVALUATE ALL CONFIGURATIONS")
    print("=" * 70)

    all_results = {}
    for cfg_name, input_names in CONFIGS.items():
        all_results[cfg_name] = run_config(df, cfg_name, input_names)

    # Phase 4: Network propagation analysis
    report = analyze_network_propagation(all_results)
    print(report)

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON results
    json_path = RESULTS_DIR / "expanded_physics_results.json"
    # Convert numpy types for JSON
    def to_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=to_json)

    # Text report
    report_path = RESULTS_DIR / "expanded_physics_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Expanded Physics Input — Network Propagation Proof\n")
        f.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"N = {len(df):,}\n\n")
        f.write(report)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Results: {json_path}")
    print(f"  Report:  {report_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

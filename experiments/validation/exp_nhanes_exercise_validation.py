#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: NHANES Exercise × Γ-Score Validation
==================================================

PURPOSE
-------
Test the prediction that physically active adults have lower Γ scores
(lower impedance mismatch) than sedentary adults, stratified by age.

HYPOTHESIS (from Lifecycle Equation)
------------------------------------
Exercise maintains x_in > 0, keeping C2 active.
Therefore: ΣΓ²(active) < ΣΓ²(sedentary) at every age tier.

This is a ZERO-PARAMETER prediction — the Γ scores are computed
from textbook reference ranges, and exercise status is self-reported.

DATA
----
- NHANES 2013-2018 lab data (already downloaded)
- PAQ (Physical Activity Questionnaire): PAQ_H, PAQ_I, PAQ_J
- Pre-computed Γ vectors from multicycle pipeline

Author: Alice Smart System (automated verification)
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import warnings
warnings.filterwarnings("ignore")

import numpy as np

# Force UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Python 3.14 has matplotlib compatibility issues (numpy eps → ticker crash)
# Skip figure generation on 3.14; run with py -3.11 for figures
if sys.version_info >= (3, 14):
    HAS_MPL = False
else:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except (ImportError, Exception):
        HAS_MPL = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "nhanes_data"
RESULTS_DIR = PROJECT_ROOT / "nhanes_results"
FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# ============================================================================
# 1. PAQ DOWNLOAD AND PROCESSING
# ============================================================================

PAQ_URLS = {
    "2013-2014": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAQ_H.XPT",
    "2015-2016": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/PAQ_I.XPT",
    "2017-2018": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQ_J.XPT",
}

# PAQ variables:
# PAQ605 - Vigorous work activity (1=Yes, 2=No)
# PAQ620 - Moderate work activity (1=Yes, 2=No)
# PAQ650 - Vigorous recreational activities (1=Yes, 2=No)
# PAQ665 - Moderate recreational activities (1=Yes, 2=No)
# PAD660 - Minutes vigorous recreational per day
# PAD675 - Minutes moderate recreational per day
# PAQ610 - Days vigorous work per week
# PAQ625 - Days moderate work per week
# PAQ655 - Days vigorous recreational per week
# PAQ670 - Days moderate recreational per week


def download_paq() -> Dict[str, Path]:
    """Download PAQ files if not cached."""
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    headers = {"User-Agent": "Mozilla/5.0 (Alice-Gamma-Net/3.9.0; research)"}

    for cycle, url in PAQ_URLS.items():
        filename = url.split("/")[-1]
        local = DATA_DIR / filename
        paths[cycle] = local

        if local.exists() and local.stat().st_size > 10000:
            print(f"  [CACHED] {filename}: {local.stat().st_size:,} bytes")
            continue

        print(f"  [DOWNLOADING] {filename} ...")
        try:
            resp = requests.get(url, headers=headers, timeout=120)
            resp.raise_for_status()
            local.write_bytes(resp.content)
            print(f"    -> {local.stat().st_size:,} bytes OK")
        except Exception as e:
            print(f"    x FAILED: {e}")

    return paths


def load_paq_data(paq_paths: Dict[str, Path]) -> "pd.DataFrame":
    """Load and merge PAQ data across 3 cycles."""
    import pandas as pd
    import pyreadstat

    frames = []
    for cycle, path in paq_paths.items():
        if not path.exists():
            continue
        try:
            df, _ = pyreadstat.read_xport(str(path))
            df.set_index("SEQN", inplace=True)
            frames.append(df)
            print(f"  PAQ {cycle}: {len(df)} rows")
        except Exception as e:
            print(f"  ERROR loading PAQ {cycle}: {e}")

    if not frames:
        raise RuntimeError("No PAQ data loaded")

    combined = pd.concat(frames, axis=0)
    if combined.index.duplicated().any():
        combined = combined[~combined.index.duplicated(keep="first")]

    print(f"  Total PAQ respondents: {len(combined)}")
    return combined


def classify_activity(paq_df: "pd.DataFrame") -> "pd.Series":
    """Classify respondents as Active (1) or Sedentary (0).

    Active = vigorous recreational ≥3 days/week
             OR moderate recreational ≥5 days/week
             OR total recreational ≥150 min/week (WHO guideline)
    """
    import pandas as pd

    # 計算每週運動分鐘數
    vig_days = paq_df.get("PAQ655", pd.Series(0, index=paq_df.index)).fillna(0)
    vig_min = paq_df.get("PAD660", pd.Series(0, index=paq_df.index)).fillna(0)
    mod_days = paq_df.get("PAQ670", pd.Series(0, index=paq_df.index)).fillna(0)
    mod_min = paq_df.get("PAD675", pd.Series(0, index=paq_df.index)).fillna(0)

    # 每週總分鐘 (vigorous counts double per WHO)
    weekly_total = (vig_days * vig_min * 2) + (mod_days * mod_min)

    # WHO: ≥150 min moderate equivalent per week
    active = (weekly_total >= 150).astype(int)

    # 也接受：自報有 vigorous recreational activity
    has_vigorous = paq_df.get("PAQ650", pd.Series(2, index=paq_df.index))
    active = active | (has_vigorous == 1).astype(int)

    return active


# ============================================================================
# 2. CROSS-REFERENCE WITH Γ VECTORS
# ============================================================================

def load_gamma_vectors() -> "pd.DataFrame":
    """Load pre-computed Γ vectors from multicycle pipeline."""
    import pandas as pd

    csv_path = RESULTS_DIR / "nhanes_multicycle_gamma_vectors.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run exp_nhanes_multicycle_validation.py first."
        )
    df = pd.read_csv(csv_path, index_col="SEQN")
    print(f"  Loaded Γ vectors: {len(df)} respondents")
    return df


def load_demographics() -> "pd.DataFrame":
    """Load demographics (age, sex) from cached NHANES data."""
    import pandas as pd
    import pyreadstat

    frames = []
    for suffix in ["_H", "_I", "_J"]:
        path = DATA_DIR / f"DEMO{suffix}.XPT"
        if path.exists():
            df, _ = pyreadstat.read_xport(str(path))
            df.set_index("SEQN", inplace=True)
            frames.append(df[["RIDAGEYR", "RIAGENDR"]])

    if not frames:
        raise RuntimeError("No DEMO files found")

    combined = pd.concat(frames, axis=0)
    if combined.index.duplicated().any():
        combined = combined[~combined.index.duplicated(keep="first")]
    return combined


# ============================================================================
# 3. ANALYSIS
# ============================================================================

def analyze_exercise_gamma(
    gamma_df: "pd.DataFrame",
    activity: "pd.Series",
    demo_df: "pd.DataFrame",
) -> dict:
    """Compare Γ scores between active and sedentary, by age tier."""
    import pandas as pd
    from scipy.stats import mannwhitneyu

    # 合併
    common = gamma_df.index.intersection(activity.index).intersection(demo_df.index)
    print(f"  Common respondents: {len(common)}")

    df = pd.DataFrame(index=common)
    df["active"] = activity.loc[common]
    df["age"] = demo_df.loc[common, "RIDAGEYR"]
    df["total_gamma_sq"] = gamma_df.loc[common, "total_gamma_sq"]
    df["health_index"] = gamma_df.loc[common, "health_index"]

    # 器官 Γ 欄位
    gamma_cols = [c for c in gamma_df.columns if c.startswith("gamma_")]
    for col in gamma_cols:
        df[col] = gamma_df.loc[common, col]

    # 年齡分層
    age_tiers = {
        "20-39": (20, 39),
        "40-59": (40, 59),
        "60+": (60, 120),
    }

    results = {"overall": {}, "by_age": {}, "by_organ": {}}

    # 整體比較
    active_mask = df["active"] == 1
    sedentary_mask = df["active"] == 0

    n_active = active_mask.sum()
    n_sedentary = sedentary_mask.sum()
    print(f"  Active: {n_active}  Sedentary: {n_sedentary}")

    g2_active = df.loc[active_mask, "total_gamma_sq"]
    g2_sedentary = df.loc[sedentary_mask, "total_gamma_sq"]

    U, p = mannwhitneyu(g2_active, g2_sedentary, alternative="less")
    results["overall"] = {
        "n_active": int(n_active),
        "n_sedentary": int(n_sedentary),
        "mean_g2_active": float(g2_active.mean()),
        "mean_g2_sedentary": float(g2_sedentary.mean()),
        "mean_H_active": float(df.loc[active_mask, "health_index"].mean()),
        "mean_H_sedentary": float(df.loc[sedentary_mask, "health_index"].mean()),
        "U_statistic": float(U),
        "p_value": float(p),
        "prediction_confirmed": bool(p < 0.05),
    }

    # 年齡分層
    for tier_name, (lo, hi) in age_tiers.items():
        tier_mask = (df["age"] >= lo) & (df["age"] <= hi)
        tier_active = df.loc[tier_mask & active_mask, "total_gamma_sq"]
        tier_sedentary = df.loc[tier_mask & sedentary_mask, "total_gamma_sq"]

        if len(tier_active) < 10 or len(tier_sedentary) < 10:
            continue

        U, p = mannwhitneyu(tier_active, tier_sedentary, alternative="less")
        results["by_age"][tier_name] = {
            "n_active": len(tier_active),
            "n_sedentary": len(tier_sedentary),
            "mean_g2_active": float(tier_active.mean()),
            "mean_g2_sedentary": float(tier_sedentary.mean()),
            "p_value": float(p),
            "confirmed": bool(p < 0.05),
        }

    # 器官層級比較
    for col in gamma_cols:
        organ = col.replace("gamma_", "")
        g_active = df.loc[active_mask, col].abs()
        g_sedentary = df.loc[sedentary_mask, col].abs()
        U, p = mannwhitneyu(g_active, g_sedentary, alternative="less")
        results["by_organ"][organ] = {
            "mean_active": float(g_active.mean()),
            "mean_sedentary": float(g_sedentary.mean()),
            "p_value": float(p),
            "significant": bool(p < 0.05),
        }

    return results, df


# ============================================================================
# 4. FIGURE
# ============================================================================

def generate_figure(results: dict, df: "pd.DataFrame", output_path: Path):
    """Generate NHANES exercise validation figure."""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "NHANES Exercise × $\\Gamma$-Score Validation (2013-2018)\n"
        "Zero-parameter prediction: Active $\\Sigma\\Gamma^2$ < Sedentary $\\Sigma\\Gamma^2$",
        fontsize=13, fontweight="bold",
    )

    active_mask = df["active"] == 1
    sedentary_mask = df["active"] == 0

    # Panel A: ΣΓ² by age, active vs sedentary
    ax = axes[0]
    age_bins = [20, 30, 40, 50, 60, 70, 80]
    for mask, label, color, marker in [
        (active_mask, "Active", "#2196F3", "o"),
        (sedentary_mask, "Sedentary", "#F44336", "s"),
    ]:
        means = []
        ages = []
        for i in range(len(age_bins)-1):
            lo, hi = age_bins[i], age_bins[i+1]
            sub = df.loc[mask & (df["age"] >= lo) & (df["age"] < hi), "total_gamma_sq"]
            if len(sub) > 5:
                means.append(sub.mean())
                ages.append((lo + hi) / 2)
        ax.plot(ages, means, f"-{marker}", color=color, linewidth=2,
                markersize=8, label=label)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Mean $\\Sigma\\Gamma^2$")
    ax.set_title("(A) Total Mismatch by Age")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Health Index by age
    ax = axes[1]
    for mask, label, color, marker in [
        (active_mask, "Active", "#2196F3", "o"),
        (sedentary_mask, "Sedentary", "#F44336", "s"),
    ]:
        means = []
        ages = []
        for i in range(len(age_bins)-1):
            lo, hi = age_bins[i], age_bins[i+1]
            sub = df.loc[mask & (df["age"] >= lo) & (df["age"] < hi), "health_index"]
            if len(sub) > 5:
                means.append(sub.mean())
                ages.append((lo + hi) / 2)
        ax.plot(ages, means, f"-{marker}", color=color, linewidth=2,
                markersize=8, label=label)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Health Index $H$")
    ax.set_title("(B) Health Index by Age")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Per-organ |Γ| comparison bar chart
    ax = axes[2]
    organs = sorted(results["by_organ"].keys())[:8]  # top 8
    x = np.arange(len(organs))
    active_vals = [results["by_organ"][o]["mean_active"] for o in organs]
    sed_vals = [results["by_organ"][o]["mean_sedentary"] for o in organs]
    w = 0.35
    ax.bar(x - w/2, active_vals, w, label="Active", color="#2196F3", alpha=0.8)
    ax.bar(x + w/2, sed_vals, w, label="Sedentary", color="#F44336", alpha=0.8)

    # 標記顯著性
    for i, organ in enumerate(organs):
        if results["by_organ"][organ]["significant"]:
            y_max = max(active_vals[i], sed_vals[i])
            ax.text(i, y_max + 0.005, "*", ha="center", fontsize=14, color="green")

    ax.set_xlabel("Organ System")
    ax.set_ylabel("Mean $|\\Gamma|$")
    ax.set_title("(C) Per-Organ Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(organs, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {output_path}")


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("  NHANES EXERCISE × Γ-SCORE VALIDATION")
    print("  Zero-parameter prediction from Lifecycle Equation")
    print("=" * 72)
    print()

    # Phase 1: Download PAQ
    print("Phase 1: Downloading PAQ data...")
    paq_paths = download_paq()

    # Phase 2: Load and classify
    print("\nPhase 2: Loading and classifying activity levels...")
    paq_df = load_paq_data(paq_paths)
    activity = classify_activity(paq_df)
    print(f"  Active: {(activity == 1).sum()}  Sedentary: {(activity == 0).sum()}")

    # Phase 3: Load Γ vectors
    print("\nPhase 3: Loading pre-computed Γ vectors...")
    gamma_df = load_gamma_vectors()

    # Phase 4: Load demographics
    print("\nPhase 4: Loading demographics...")
    demo_df = load_demographics()

    # Phase 5: Cross-reference analysis
    print("\nPhase 5: Analyzing exercise × Γ relationship...")
    results, merged_df = analyze_exercise_gamma(gamma_df, activity, demo_df)

    # Phase 6: Report
    print()
    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)

    ov = results["overall"]
    print(f"\n  OVERALL (n_active={ov['n_active']}, n_sedentary={ov['n_sedentary']}):")
    print(f"    Mean ΣΓ² (Active):    {ov['mean_g2_active']:.4f}")
    print(f"    Mean ΣΓ² (Sedentary): {ov['mean_g2_sedentary']:.4f}")
    print(f"    Mean H   (Active):    {ov['mean_H_active']:.4f}")
    print(f"    Mean H   (Sedentary): {ov['mean_H_sedentary']:.4f}")
    print(f"    Mann-Whitney p:       {ov['p_value']:.2e}")
    status = "CONFIRMED ✓" if ov['prediction_confirmed'] else "NOT CONFIRMED"
    print(f"    Prediction:           {status}")

    print(f"\n  BY AGE TIER:")
    for tier, data in results["by_age"].items():
        status = "✓" if data["confirmed"] else "✗"
        print(f"    {tier:6s}  Active={data['mean_g2_active']:.4f}  "
              f"Sedentary={data['mean_g2_sedentary']:.4f}  "
              f"p={data['p_value']:.2e}  [{status}]  "
              f"(n={data['n_active']}+{data['n_sedentary']})")

    print(f"\n  BY ORGAN (significant marked *):")
    for organ, data in sorted(results["by_organ"].items()):
        sig = "*" if data["significant"] else " "
        print(f"    {organ:12s}  Active={data['mean_active']:.4f}  "
              f"Sedentary={data['mean_sedentary']:.4f}  "
              f"p={data['p_value']:.2e} {sig}")

    # Phase 7: Figure
    print("\nPhase 7: Generating figure...")
    fig_path = FIGURE_DIR / "fig_nhanes_exercise_gamma.pdf"
    generate_figure(results, merged_df, fig_path)

    # Verdict
    n_confirmed = sum(1 for d in results["by_age"].values() if d["confirmed"])
    n_tiers = len(results["by_age"])
    n_organs_sig = sum(1 for d in results["by_organ"].values() if d["significant"])
    n_organs = len(results["by_organ"])

    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    print(f"  Overall prediction:     {'CONFIRMED' if ov['prediction_confirmed'] else 'NOT CONFIRMED'}")
    print(f"  Age tiers confirmed:    {n_confirmed}/{n_tiers}")
    print(f"  Organs significant:     {n_organs_sig}/{n_organs}")
    print()
    if ov['prediction_confirmed']:
        print("  The Lifecycle Equation prediction is confirmed:")
        print("  Active adults have significantly lower ΣΓ² than sedentary adults.")
        print("  Exercise maintains x_in > 0, keeping C2 active against aging drift.")
    else:
        print("  The prediction was not confirmed at p < 0.05.")
        print("  This could indicate: insufficient sample, confounders,")
        print("  or the need for more specific activity metrics.")
    print("=" * 72)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: NHANES 10-cycle Organ-Specific Mortality Analysis
=============================================================

PURPOSE
-------
Test whether organ-specific Γ scores predict organ-specific causes of death.
Uses NHANES 1999-2018 (10 cycles, ~52,000 adults, ~9,200 deaths).

UCOD LEADING CAUSE CODES (NCHS Public-Use Linked Mortality Files):
  1  = Diseases of heart (I00-I09, I11, I13, I20-I51)
  2  = Malignant neoplasms (C00-C97)
  3  = Chronic lower respiratory diseases (J40-J47)
  4  = Cerebrovascular diseases (I60-I69)
  5  = Accidents / unintentional injuries (V01-X59, Y85-Y86)
  6  = Alzheimer's disease (G30)
  7  = Diabetes mellitus (E10-E14)
  8  = Influenza and pneumonia (J09-J18)
  9  = Nephritis / nephrotic syndrome (N00-N07, N17-N19, N25-N27)
  10 = All other causes

Γ-ORGAN MAPPING
  UCOD 1 (Heart)         → cardiac Γ
  UCOD 2 (Cancer)        → immune Γ
  UCOD 3 (CLRD)          → pulmonary Γ
  UCOD 4 (Cerebrovascular) → neuro Γ / vascular Γ
  UCOD 7 (Diabetes)      → endocrine Γ
  UCOD 8 (Pneumonia)     → pulmonary Γ
  UCOD 9 (Renal)         → renal Γ

HYPOTHESIS
  For each cause-of-death, the corresponding organ Γ should be a
  better predictor than random (AUC > 0.5), AND should perform
  better than non-corresponding organ Γ scores.

Author: Alice Smart System (automated verification)
"""

from __future__ import annotations

import io
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np

# 強制 UTF-8 輸出
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "nhanes_data"
RESULTS_DIR = PROJECT_ROOT / "nhanes_results"
RESULTS_DIR.mkdir(exist_ok=True)

# 嘗試 matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
# 1. UCOD → ORGAN MAPPING
# ============================================================================

UCOD_TO_ORGAN = {
    1:  {"organ": "cardiac",    "label": "Heart disease"},
    2:  {"organ": "immune",     "label": "Malignant neoplasms"},
    3:  {"organ": "pulmonary",  "label": "Chronic lower respiratory"},
    4:  {"organ": "neuro",      "label": "Cerebrovascular"},
    5:  {"organ": None,         "label": "Accidents"},       # 非疾病
    6:  {"organ": "neuro",      "label": "Alzheimer's"},
    7:  {"organ": "endocrine",  "label": "Diabetes mellitus"},
    8:  {"organ": "pulmonary",  "label": "Influenza/pneumonia"},
    9:  {"organ": "renal",      "label": "Renal disease"},
    10: {"organ": None,         "label": "All other causes"},
}

# NHANES 10-cycle 結構
CYCLES = [
    ("1999-2000", "DEMO",   "BIOPRO",  None,    "CBC",  "HDL", "TRIGLY", "GHB", "GLU"),
    ("2001-2002", "DEMO_B", "L40_B",   "L13_B", "L25_B","L10_B","L13_B", "L10AM_B","L13AM_B"),
    ("2003-2004", "DEMO_C", "L40_C",   "L13_C", "L25_C","L10_C","L13_C", "L10AM_C","L13AM_C"),
    ("2005-2006", "DEMO_D", "BIOPRO_D",None,    "CBC_D","HDL_D","TRIGLY_D","GHB_D","GLU_D"),
    ("2007-2008", "DEMO_E", "BIOPRO_E",None,    "CBC_E","HDL_E","TRIGLY_E","GHB_E","GLU_E"),
    ("2009-2010", "DEMO_F", "BIOPRO_F",None,    "CBC_F","HDL_F","TRIGLY_F","GHB_F","GLU_F"),
    ("2011-2012", "DEMO_G", "BIOPRO_G",None,    "CBC_G","HDL_G","TRIGLY_G","GHB_G","GLU_G"),
    ("2013-2014", "DEMO_H", "BIOPRO_H",None,    "CBC_H","HDL_H","TRIGLY_H","GHB_H","GLU_H"),
    ("2015-2016", "DEMO_I", "BIOPRO_I",None,    "CBC_I","HDL_I","TRIGLY_I","GHB_I","GLU_I"),
    ("2017-2018", "DEMO_J", "BIOPRO_J",None,    "CBC_J","HDL_J","TRIGLY_J","GHB_J","GLU_J"),
]

BPX_FILES = {
    "1999-2000": "BPX", "2001-2002": "BPX_B", "2003-2004": "BPX_C",
    "2005-2006": "BPX_D", "2007-2008": "BPX_E", "2009-2010": "BPX_F",
    "2011-2012": "BPX_G", "2013-2014": "BPX_H", "2015-2016": "BPX_I",
    "2017-2018": "BPXO_J",
}


# ============================================================================
# 2. 載入 10-cycle Γ 向量（已預先計算）
# ============================================================================

def load_10cycle_gamma() -> "pd.DataFrame":
    """載入 10-cycle Γ 向量 CSV。"""
    import pandas as pd

    csv_path = RESULTS_DIR / "nhanes_10cycle_gamma_vectors.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "SEQN" in df.columns:
            df["SEQN"] = df["SEQN"].astype(int)
            df.set_index("SEQN", inplace=True)
        print("  Loaded 10-cycle Γ vectors: %d respondents" % len(df))
        return df

    # 退回 3-cycle
    csv_path = RESULTS_DIR / "nhanes_multicycle_gamma_vectors.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col="SEQN")
        print("  Loaded 3-cycle Γ vectors (fallback): %d respondents" % len(df))
        return df

    raise FileNotFoundError("No gamma vector CSV found in " + str(RESULTS_DIR))


# ============================================================================
# 3. 載入死亡數據
# ============================================================================

def load_mortality() -> "pd.DataFrame":
    """載入 NHANES 10-cycle 死亡數據。"""
    import pandas as pd

    records = []
    for y1 in range(1999, 2018, 2):
        y2 = y1 + 1
        mp = DATA_DIR / ("NHANES_%d_%d_MORT_2019_PUBLIC.dat" % (y1, y2))
        if not mp.exists():
            continue
        with open(mp, "r") as f:
            for line in f:
                if len(line.strip()) < 15:
                    continue
                try:
                    seqn = int(line[0:14].strip())
                    eligstat = line[14:15].strip()
                    mortstat = line[15:16].strip()
                    ucod = line[16:19].strip()

                    if eligstat != "1":
                        continue

                    # 解析跟蹤月數（位置不同的格式）
                    rest = line[19:].strip().split()
                    fu_months = 0
                    if rest:
                        try:
                            fu_months = int(rest[-1])
                        except ValueError:
                            pass

                    records.append({
                        "SEQN": seqn,
                        "mort_status": int(mortstat) if mortstat.isdigit() else 0,
                        "ucod_leading": int(ucod) if ucod.isdigit() else 0,
                        "fu_months": fu_months,
                    })
                except (ValueError, IndexError):
                    continue

    df = pd.DataFrame(records)
    df.set_index("SEQN", inplace=True)
    n_dead = (df["mort_status"] == 1).sum()
    print("  Loaded mortality: %d eligible, %d deaths" % (len(df), n_dead))
    return df


# ============================================================================
# 4. AUC 計算
# ============================================================================

def compute_auc(y_true: np.ndarray, y_score: np.ndarray,
                n_boot: int = 500) -> Dict[str, float]:
    """AUC with bootstrap CI。"""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import norm

    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = np.asarray(y_true[mask], dtype=int)
    y_score = np.asarray(y_score[mask], dtype=float)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos < 10 or n_neg < 10:
        return {"auc": 0.5, "ci_lo": 0.5, "ci_hi": 0.5, "p_value": 1.0,
                "n_pos": int(n_pos), "n_neg": int(n_neg)}

    auc = roc_auc_score(y_true, y_score)

    # DeLong SE 近似
    se = np.sqrt((auc * (1 - auc) +
                  (n_pos - 1) * (auc / (2 - auc) - auc**2) +
                  (n_neg - 1) * (2 * auc**2 / (1 + auc) - auc**2)) /
                 (n_pos * n_neg))

    z = (auc - 0.5) / max(se, 1e-10)
    p_value = 1.0 - norm.cdf(z)

    # Bootstrap CI
    rng = np.random.default_rng(42)
    aucs_boot = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs_boot.append(roc_auc_score(yt, ys))
    ci_lo, ci_hi = (np.percentile(aucs_boot, [2.5, 97.5]) if aucs_boot
                    else (auc, auc))

    return {
        "auc": round(float(auc), 4),
        "ci_lo": round(float(ci_lo), 4),
        "ci_hi": round(float(ci_hi), 4),
        "p_value": float(p_value),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
    }


# ============================================================================
# 5. 主分析
# ============================================================================

def run_organ_mortality_analysis() -> Dict[str, Any]:
    """核心分析：各器官 Γ → 器官別死亡。"""
    import pandas as pd

    gamma_df = load_10cycle_gamma()
    mort_df = load_mortality()

    # 合併
    common = gamma_df.index.intersection(mort_df.index)
    print("  Common subjects: %d" % len(common))

    gamma = gamma_df.loc[common]
    mort = mort_df.loc[common]

    # Γ 欄位名
    gamma_cols = [c for c in gamma.columns if c.startswith("gamma_")]
    organs = [c.replace("gamma_", "") for c in gamma_cols]

    # 全因死亡
    y_dead = (mort["mort_status"] == 1).astype(int).values

    print("\n  === All-Cause Mortality ===")
    if "total_gamma_sq" in gamma.columns:
        allcause = compute_auc(y_dead, gamma["total_gamma_sq"].values)
    elif "health_index" in gamma.columns:
        allcause = compute_auc(y_dead, 1.0 - gamma["health_index"].values)
    else:
        sum_g2 = gamma[gamma_cols].values ** 2
        allcause = compute_auc(y_dead, sum_g2.sum(axis=1))

    print("  ΣΓ² → All-cause: AUC = %.4f [%.4f-%.4f] p=%.2e (n_deaths=%d)" %
          (allcause["auc"], allcause["ci_lo"], allcause["ci_hi"],
           allcause["p_value"], allcause["n_pos"]))

    # 器官別死因
    results = {"all_cause": allcause, "organ_specific": {}, "cross_organ": {}}

    print("\n  === Organ-Specific Death AUC ===")
    print("  %-14s %-18s %7s %7s %7s %10s %6s" %
          ("Organ Γ", "Death cause", "AUC", "CI_lo", "CI_hi", "p-value", "n+"))
    print("  " + "-" * 74)

    for ucod, info in sorted(UCOD_TO_ORGAN.items()):
        target_organ = info["organ"]
        label = info["label"]

        if target_organ is None:
            continue  # 跳過非疾病死因（事故、其他）

        # 該死因的二元標籤
        y_cause = ((mort["mort_status"] == 1) &
                   (mort["ucod_leading"] == ucod)).astype(int).values
        n_events = y_cause.sum()

        if n_events < 20:
            print("  %-14s %-18s  [SKIP: n=%d < 20]" %
                  (target_organ, label, n_events))
            continue

        # 對應器官 Γ
        gamma_col = "gamma_" + target_organ
        if gamma_col not in gamma.columns:
            # 嘗試不同命名
            alt_cols = [c for c in gamma_cols if target_organ in c]
            if alt_cols:
                gamma_col = alt_cols[0]
            else:
                print("  %-14s %-18s  [SKIP: no Γ column]" %
                      (target_organ, label))
                continue

        auc_result = compute_auc(y_cause, np.abs(gamma[gamma_col].values))

        print("  %-14s %-18s %7.4f %7.4f %7.4f %10.2e %6d %s" %
              (target_organ, label, auc_result["auc"],
               auc_result["ci_lo"], auc_result["ci_hi"],
               auc_result["p_value"], n_events,
               "✓" if auc_result["p_value"] < 0.05 else ""))

        results["organ_specific"]["%d_%s" % (ucod, target_organ)] = {
            "ucod": ucod,
            "organ": target_organ,
            "label": label,
            **auc_result,
        }

    # 交叉器官分析：用 ΣΓ² 預測各死因
    print("\n  === Cross-Organ: ΣΓ² → Cause-Specific Death ===")
    print("  %-18s %7s %6s" % ("Death cause", "AUC", "n+"))
    print("  " + "-" * 36)

    for ucod, info in sorted(UCOD_TO_ORGAN.items()):
        label = info["label"]
        y_cause = ((mort["mort_status"] == 1) &
                   (mort["ucod_leading"] == ucod)).astype(int).values
        n_events = y_cause.sum()
        if n_events < 20:
            continue

        if "total_gamma_sq" in gamma.columns:
            score = gamma["total_gamma_sq"].values
        else:
            score = (gamma[gamma_cols].values ** 2).sum(axis=1)

        auc_r = compute_auc(y_cause, score)
        print("  %-18s %7.4f %6d" % (label, auc_r["auc"], n_events))
        results["cross_organ"]["ucod_%d_%s" % (ucod, label.replace(" ", "_"))] = {
            "ucod": ucod, "label": label, **auc_r,
        }

    return results


# ============================================================================
# 6. 圖表生成
# ============================================================================

def generate_figure(results: Dict[str, Any], output_path: Path):
    """生成器官別死亡 AUC 圖表。"""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available")
        return

    organ_data = results["organ_specific"]
    if not organ_data:
        return

    # 排序
    items = sorted(organ_data.values(), key=lambda x: x["auc"], reverse=True)

    labels = ["%s\n(UCOD %d, n=%d)" % (d["label"][:15], d["ucod"], d["n_pos"])
              for d in items]
    aucs = [d["auc"] for d in items]
    ci_los = [d["ci_lo"] for d in items]
    ci_his = [d["ci_hi"] for d in items]
    errors_lo = [a - l for a, l in zip(aucs, ci_los)]
    errors_hi = [h - a for a, h in zip(aucs, ci_his)]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(items))

    colors = ["#2196F3" if a > 0.55 else "#FF9800" if a > 0.50 else "#F44336"
              for a in aucs]

    ax.barh(y_pos, aucs, xerr=[errors_lo, errors_hi], capsize=4,
            color=colors, edgecolor="white", height=0.6)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="AUC = 0.5 (chance)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("AUC (organ-specific Γ → cause-specific death)", fontsize=11)
    ax.set_title("NHANES 10-Cycle Organ-Specific Death Prediction\n"
                 "(zero fitted parameters, n ≈ 52,000)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.35, 0.85)
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  [SAVED] %s" % output_path)


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("  NHANES ORGAN-SPECIFIC MORTALITY ANALYSIS")
    print("  10-cycle (1999-2018), zero fitted parameters")
    print("=" * 72)
    print()

    results = run_organ_mortality_analysis()

    # 生成圖表
    print("\nGenerating figure...")
    fig_path = PROJECT_ROOT / "figures" / "fig_organ_mortality_auc.pdf"
    generate_figure(results, fig_path)
    fig_path_png = fig_path.with_suffix(".png")
    if HAS_MPL:
        generate_figure(results, fig_path_png)

    # 儲存結果
    results_path = RESULTS_DIR / "organ_mortality_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("  Results saved: %s" % results_path)

    # 最終報告
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    allcause = results["all_cause"]
    print("  All-cause mortality AUC: %.4f [%.4f-%.4f]" %
          (allcause["auc"], allcause["ci_lo"], allcause["ci_hi"]))

    n_tested = len(results["organ_specific"])
    n_significant = sum(1 for d in results["organ_specific"].values()
                        if d["p_value"] < 0.05)
    n_above_55 = sum(1 for d in results["organ_specific"].values()
                     if d["auc"] > 0.55)

    print("  Organ-specific: %d/%d significant (p<0.05)" %
          (n_significant, n_tested))
    print("  Organ-specific: %d/%d with AUC > 0.55" %
          (n_above_55, n_tested))
    print()

    if n_above_55 >= 3:
        print("  VERDICT: PASS — ≥3 organ-specific Γ → death AUC > 0.55")
    else:
        print("  VERDICT: MARGINAL — fewer than 3 organs meet threshold")
    print("=" * 72)


if __name__ == "__main__":
    main()

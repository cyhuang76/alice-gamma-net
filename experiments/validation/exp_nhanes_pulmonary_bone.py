#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: NHANES Pulmonary + Bone Density → Organ Γ Enhancement
==================================================================

PURPOSE
-------
Add pulmonary function (spirometry) and bone density (DEXA) data
to enhance organ-specific Γ predictions for pulmonary and bone systems.

NHANES DATA FILES:
  Spirometry (SPX): FEV1, FVC, FEV1/FVC ratio
    - pulmonary impedance proxy: Z_pulm ∝ |1 - FEV1/FVC_pred|
  
  DEXA Bone Density (DXXAG): Total body and regional BMD
    - bone impedance proxy: Z_bone ∝ |BMD - BMD_ref| / BMD_ref

HYPOTHESIS:
  1. Adding FEV1/FVC improves pulmonary Γ → CLRD death AUC
  2. Adding BMD improves bone Γ → all-cause death AUC
  3. Both supplement the blood-chemistry-only Γ vector

Author: Alice Smart System (automated verification)
"""

from __future__ import annotations

import io
import sys
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "nhanes_data"
RESULTS_DIR = PROJECT_ROOT / "nhanes_results"
RESULTS_DIR.mkdir(exist_ok=True)

import pandas as pd
import requests


# ============================================================================
# 1. 下載數據
# ============================================================================

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Nhanes"

# 肺功能 (Spirometry)
SPX_FILES = {
    "2007-2008": "SPX_E",
    "2009-2010": "SPX_F",
    "2011-2012": "SPX_G",
}

# 骨密度 (DEXA - Total Body)
DXXAG_FILES = {
    "2007-2008": "DXXAG_E",
    "2009-2010": "DXXAG_F",
    "2011-2012": "DXXAG_G",
    "2013-2014": "DXXAG_H",
}


def download_xpt(filename: str, cycle: str) -> Path:
    """下載 NHANES XPT 檔案。"""
    local_path = DATA_DIR / (filename + ".XPT")
    if local_path.exists() and local_path.stat().st_size > 1000:
        return local_path

    # NHANES URL pattern
    cycle_path = cycle.replace("-", "_")
    url = "%s/%s/%s.XPT" % (NHANES_BASE, cycle_path, filename)
    print("  [DOWNLOADING] %s from %s ..." % (filename, url))

    headers = {"User-Agent": "Mozilla/5.0 (Alice-Gamma-Net/3.9; research)"}
    try:
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        print("  -> %s: %d bytes" % (filename, local_path.stat().st_size))
    except Exception as e:
        print("  ERROR downloading %s: %s" % (filename, e))

    return local_path


def download_all():
    """下載所有需要的 NHANES 檔案。"""
    DATA_DIR.mkdir(exist_ok=True)

    print("\n  === Downloading Spirometry (SPX) ===")
    for cycle, filename in SPX_FILES.items():
        download_xpt(filename, cycle)

    print("\n  === Downloading DEXA Bone Density (DXXAG) ===")
    for cycle, filename in DXXAG_FILES.items():
        download_xpt(filename, cycle)


# ============================================================================
# 2. 載入與處理
# ============================================================================

def load_spirometry() -> pd.DataFrame:
    """載入肺功能數據。"""
    parts = []
    for cycle, filename in SPX_FILES.items():
        path = DATA_DIR / (filename + ".XPT")
        if not path.exists():
            continue
        try:
            df = pd.read_sas(path, format="xport")
            df["SEQN"] = df["SEQN"].astype(int)

            # 關鍵欄位
            cols_of_interest = [c for c in df.columns
                                if any(x in c.upper() for x in
                                       ["FEV1", "FVC", "FEF", "PEF", "SEQN"])]
            rec = df[["SEQN"]].copy()

            # FEV1 (最佳值)
            fev1_cols = [c for c in df.columns if "FEV1" in c.upper()
                         and "PRED" not in c.upper() and "RATIO" not in c.upper()]
            if fev1_cols:
                rec["FEV1"] = df[fev1_cols[0]]

            # FVC (最佳值)
            fvc_cols = [c for c in df.columns if "FVC" in c.upper()
                        and "PRED" not in c.upper()]
            if fvc_cols:
                rec["FVC"] = df[fvc_cols[0]]

            # FEV1/FVC predicted 值
            fev1_pred = [c for c in df.columns if "FEV1" in c.upper()
                         and "PRED" in c.upper()]
            if fev1_pred:
                rec["FEV1_pred"] = df[fev1_pred[0]]

            fvc_pred = [c for c in df.columns if "FVC" in c.upper()
                        and "PRED" in c.upper()]
            if fvc_pred:
                rec["FVC_pred"] = df[fvc_pred[0]]

            rec = rec.dropna(subset=["FEV1", "FVC"] if "FEV1" in rec.columns and "FVC" in rec.columns else ["SEQN"])
            if len(rec) > 0:
                parts.append(rec)
                print("  %s: %d records, cols: %s" % (filename, len(rec),
                      list(rec.columns[:6])))
        except Exception as e:
            print("  ERROR %s: %s" % (filename, e))

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True).drop_duplicates("SEQN")

    # 計算比率
    if "FEV1" in df.columns and "FVC" in df.columns:
        df["FEV1_FVC_ratio"] = df["FEV1"] / df["FVC"]

        # 肺部阻抗指標
        # 正常 FEV1/FVC > 0.70 (Gold standard for COPD diagnosis)
        fev1_fvc_ref = 0.80  # 健康成人理想值
        df["Z_pulmonary"] = np.abs(df["FEV1_FVC_ratio"] - fev1_fvc_ref) / (
            df["FEV1_FVC_ratio"] + fev1_fvc_ref)
        df["gamma_pulmonary"] = df["Z_pulmonary"]

    print("\n  Total spirometry records: %d" % len(df))
    return df


def load_bone_density() -> pd.DataFrame:
    """載入骨密度數據。"""
    parts = []
    for cycle, filename in DXXAG_FILES.items():
        path = DATA_DIR / (filename + ".XPT")
        if not path.exists():
            continue
        try:
            df = pd.read_sas(path, format="xport")
            df["SEQN"] = df["SEQN"].astype(int)

            rec = df[["SEQN"]].copy()

            # Total body BMD
            bmd_cols = [c for c in df.columns if "BMD" in c.upper()
                        and "TOTAL" in c.upper()]
            if not bmd_cols:
                bmd_cols = [c for c in df.columns if "DXDTOBMD" in c.upper()
                            or "DXATOBMD" in c.upper()]
            if not bmd_cols:
                # 嘗試任何 BMD 欄位
                bmd_cols = [c for c in df.columns if "BMD" in c.upper()]

            if bmd_cols:
                rec["BMD"] = df[bmd_cols[0]]
                print("    Using BMD column: %s" % bmd_cols[0])

            # Total body BMC (bone mineral content)
            bmc_cols = [c for c in df.columns if "BMC" in c.upper()
                        and "TOTAL" in c.upper()]
            if not bmc_cols:
                bmc_cols = [c for c in df.columns if "DXDTOBMC" in c.upper()
                            or "DXATOBMC" in c.upper()]
            if bmc_cols:
                rec["BMC"] = df[bmc_cols[0]]

            rec = rec.dropna(subset=["BMD"] if "BMD" in rec.columns else ["SEQN"])
            if len(rec) > 0 and "BMD" in rec.columns:
                parts.append(rec)
                print("  %s: %d records" % (filename, len(rec)))
        except Exception as e:
            print("  ERROR %s: %s" % (filename, e))

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True).drop_duplicates("SEQN")

    if "BMD" in df.columns:
        # 骨密度阻抗指標
        # 正常 total body BMD ≈ 1.0-1.2 g/cm²
        bmd_ref = 1.1  # healthy adult reference
        df["Z_bone"] = np.abs(df["BMD"] - bmd_ref) / (df["BMD"] + bmd_ref)
        df["gamma_bone"] = df["Z_bone"]

    print("\n  Total bone density records: %d" % len(df))
    return df


# ============================================================================
# 3. 與死亡率交叉
# ============================================================================

def mortality_analysis(spx_df: pd.DataFrame, bone_df: pd.DataFrame) -> Dict[str, Any]:
    """與死亡率數據比對。"""
    from sklearn.metrics import roc_auc_score

    # 載入死亡率
    mort_records = []
    for y1 in range(1999, 2018, 2):
        y2 = y1 + 1
        mp = DATA_DIR / ("NHANES_%d_%d_MORT_2019_PUBLIC.dat" % (y1, y2))
        if not mp.exists():
            continue
        with open(mp) as f:
            for line in f:
                if len(line.strip()) < 15:
                    continue
                try:
                    seqn = int(line[0:14].strip())
                    eligstat = line[14:15].strip()
                    mortstat = line[15:16].strip()
                    ucod = line[16:19].strip()
                    if eligstat == "1":
                        mort_records.append({
                            "SEQN": seqn,
                            "mort": int(mortstat) if mortstat.isdigit() else 0,
                            "ucod": int(ucod) if ucod.isdigit() else 0,
                        })
                except Exception:
                    continue

    mort_df = pd.DataFrame(mort_records).drop_duplicates("SEQN").set_index("SEQN")
    results = {}

    # 肺功能 → 死亡率
    if len(spx_df) > 0 and "gamma_pulmonary" in spx_df.columns:
        spx_idx = spx_df.set_index("SEQN")
        common = spx_idx.index.intersection(mort_df.index)
        print("\n  Spirometry × Mortality: %d common" % len(common))

        if len(common) > 100:
            y = (mort_df.loc[common, "mort"] == 1).astype(int).values
            score = spx_idx.loc[common, "gamma_pulmonary"].values
            mask = np.isfinite(score)

            auc = roc_auc_score(y[mask], score[mask])
            n_deaths = y.sum()
            print("  Γ_pulmonary → all-cause: AUC=%.4f (n_deaths=%d)" % (auc, n_deaths))

            results["pulmonary_allcause"] = {
                "auc": round(float(auc), 4), "n": int(len(common)),
                "n_deaths": int(n_deaths)
            }

            # CLRD death (UCOD 3)
            y_clrd = ((mort_df.loc[common, "mort"] == 1) &
                      (mort_df.loc[common, "ucod"] == 3)).astype(int).values
            n_clrd = y_clrd.sum()
            if n_clrd >= 10:
                auc_clrd = roc_auc_score(y_clrd[mask], score[mask])
                print("  Γ_pulmonary → CLRD death: AUC=%.4f (n=%d)" % (auc_clrd, n_clrd))
                results["pulmonary_clrd"] = {
                    "auc": round(float(auc_clrd), 4), "n_deaths": int(n_clrd)
                }

            # FEV1/FVC 本身 → all-cause
            if "FEV1_FVC_ratio" in spx_idx.columns:
                ratio = spx_idx.loc[common, "FEV1_FVC_ratio"].values
                mask_r = np.isfinite(ratio)
                auc_ratio = roc_auc_score(y[mask_r], 1.0 - ratio[mask_r])  # 低比率 → 高風險
                print("  FEV1/FVC (inverted) → all-cause: AUC=%.4f" % auc_ratio)
                results["fev1fvc_allcause"] = {"auc": round(float(auc_ratio), 4)}

    # 骨密度 → 死亡率
    if len(bone_df) > 0 and "gamma_bone" in bone_df.columns:
        bone_idx = bone_df.set_index("SEQN")
        common = bone_idx.index.intersection(mort_df.index)
        print("\n  Bone density × Mortality: %d common" % len(common))

        if len(common) > 100:
            y = (mort_df.loc[common, "mort"] == 1).astype(int).values
            score = bone_idx.loc[common, "gamma_bone"].values
            mask = np.isfinite(score)

            auc = roc_auc_score(y[mask], score[mask])
            n_deaths = y.sum()
            print("  Γ_bone → all-cause: AUC=%.4f (n_deaths=%d)" % (auc, n_deaths))

            results["bone_allcause"] = {
                "auc": round(float(auc), 4), "n": int(len(common)),
                "n_deaths": int(n_deaths)
            }

    return results


# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("  NHANES PULMONARY + BONE DENSITY → Γ ENHANCEMENT")
    print("=" * 72)

    # Phase 1: 下載
    print("\nPhase 1: Downloading data...")
    download_all()

    # Phase 2: 載入
    print("\nPhase 2: Loading spirometry...")
    spx_df = load_spirometry()

    print("\nPhase 3: Loading bone density...")
    bone_df = load_bone_density()

    # Phase 4: 死亡率分析
    print("\nPhase 4: Mortality cross-reference...")
    mort_results = mortality_analysis(spx_df, bone_df)

    # 儲存
    all_results = {
        "spirometry": {
            "n": len(spx_df),
            "fev1_fvc_mean": round(float(spx_df["FEV1_FVC_ratio"].mean()), 4) if "FEV1_FVC_ratio" in spx_df.columns else None,
        } if len(spx_df) > 0 else {"n": 0},
        "bone_density": {
            "n": len(bone_df),
            "bmd_mean": round(float(bone_df["BMD"].mean()), 4) if "BMD" in bone_df.columns else None,
        } if len(bone_df) > 0 else {"n": 0},
        "mortality": mort_results,
    }

    out_path = RESULTS_DIR / "pulmonary_bone_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\n  Results saved: %s" % out_path)

    # Summary
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for key, val in mort_results.items():
        print("  %s: AUC = %.4f" % (key, val.get("auc", 0)))
    print("=" * 72)


if __name__ == "__main__":
    main()

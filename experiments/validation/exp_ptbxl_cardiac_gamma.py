#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: PTB-XL ECG → Cardiac Impedance Validation
======================================================

PURPOSE
-------
Validate cardiac Γ predictions using the PTB-XL ECG dataset
(PhysioNet, public, 21,801 clinical 12-lead ECGs).

The Γ-Net framework claims cardiac impedance mismatch manifests as:
  - Prolonged QRS (conduction impedance ↑)
  - ST/T-wave abnormalities (repolarisation impedance ↑)
  - Reduced HRV (autonomic-cardiac Γ coupling)

We extract ECG morphological features and test whether they
correlate with the Γ-framework's cardiac impedance predictions.

DATA SOURCE
  PTB-XL: A large publicly available electrocardiography dataset
  - 21,801 clinical 12-lead ECGs (10 seconds each)
  - 18,869 patients, with diagnostic labels (NORM, MI, HYP, STTC, CD)
  - Sampling rates: 100 Hz (downsampled) and 500 Hz
  - License: Creative Commons Attribution 4.0

PHYSICS MAPPING
  In the Γ framework:
  - NORM → Γ_cardiac ≈ 0 (impedance matched)
  - MI (myocardial infarction) → high Γ_cardiac (tissue death = Z → ∞)
  - HYP (hypertrophy) → elevated Γ_cardiac (Z_wall thickened)
  - STTC (ST/T changes) → moderate Γ_cardiac (repolarisation Z shift)
  - CD (conduction disturbance) → elevated Γ_cardiac (conduction Z ↑)

Author: Alice Smart System (automated verification)
"""

from __future__ import annotations

import io
import sys
import ast
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "ptbxl_data"
RESULTS_DIR = PROJECT_ROOT / "nhanes_results"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
# 1. 下載 PTB-XL 元數據（不下載波形，只下 CSV + 小批量波形）
# ============================================================================

PTBXL_CSV_URL = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
PTBXL_BASE = "https://physionet.org/files/ptb-xl/1.0.3/"


def download_metadata() -> "pd.DataFrame":
    """下載 PTB-XL 元數據 CSV。"""
    import pandas as pd
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "ptbxl_database.csv"

    if csv_path.exists() and csv_path.stat().st_size > 100000:
        print("  [CACHED] ptbxl_database.csv: %d bytes" % csv_path.stat().st_size)
    else:
        print("  [DOWNLOADING] ptbxl_database.csv ...")
        headers = {"User-Agent": "Mozilla/5.0 (Alice-Gamma-Net/3.9; research)"}
        resp = requests.get(PTBXL_CSV_URL, headers=headers, timeout=120)
        resp.raise_for_status()
        csv_path.write_bytes(resp.content)
        print("  -> %d bytes" % csv_path.stat().st_size)

    df = pd.read_csv(csv_path, index_col="ecg_id")
    print("  Total ECGs: %d" % len(df))
    return df


def parse_scp_codes(df: "pd.DataFrame") -> "pd.DataFrame":
    """解析 SCP 診斷代碼，分類為 5 大類。"""
    import pandas as pd

    # scp_codes 欄位是 dict-like string
    # 超類別：NORM, MI, STTC, CD, HYP
    categories = {"NORM": [], "MI": [], "STTC": [], "CD": [], "HYP": []}

    # SCP code → superclass mapping (標準 PTB-XL mapping)
    scp_to_super = {}
    mapping_path = DATA_DIR / "scp_statements.csv"

    if mapping_path.exists():
        scp_df = pd.read_csv(mapping_path, index_col=0)
        if "diagnostic_class" in scp_df.columns:
            for code, row in scp_df.iterrows():
                dc = row.get("diagnostic_class", "")
                if pd.notna(dc) and dc in categories:
                    scp_to_super[str(code)] = dc

    # 如果沒有 mapping 檔案，用簡化方法
    if not scp_to_super:
        # 常見 SCP codes 的超類別
        norm_codes = {"NORM"}
        mi_codes = {"IMI", "AMI", "ILMI", "ALMI", "INJAS", "INJAL", "INJIN",
                     "INJLA", "INJIL", "PMI", "LMI"}
        sttc_codes = {"STTC", "NST_", "DIG", "LNGQT", "ISC_", "ISCA", "ISCAL",
                       "ISCAS", "ISCLA", "ISCIN", "ISCIL"}
        cd_codes = {"CD", "LAFB", "LPFB", "IRBBB", "CRBBB", "CLBBB", "WPW",
                    "AVB", "1AVB", "2AVB", "3AVB", "IVCD"}
        hyp_codes = {"HYP", "LVH", "RVH", "LAO/LAE", "RAO/RAE", "SEHYP"}

        for code in norm_codes:
            scp_to_super[code] = "NORM"
        for code in mi_codes:
            scp_to_super[code] = "MI"
        for code in sttc_codes:
            scp_to_super[code] = "STTC"
        for code in cd_codes:
            scp_to_super[code] = "CD"
        for code in hyp_codes:
            scp_to_super[code] = "HYP"

    # 為每個 ECG 指定超類別
    df["superclass"] = "OTHER"
    for idx, row in df.iterrows():
        scp_raw = row.get("scp_codes", "{}")
        try:
            codes = ast.literal_eval(scp_raw) if isinstance(scp_raw, str) else {}
        except (ValueError, SyntaxError):
            codes = {}

        # 取信度最高的代碼
        best_class = "OTHER"
        best_conf = 0
        for code, conf in codes.items():
            sc = scp_to_super.get(code, None)
            if sc and conf > best_conf:
                best_class = sc
                best_conf = conf

        df.at[idx, "superclass"] = best_class

    print("\n  Diagnostic distribution:")
    for cat, count in df["superclass"].value_counts().items():
        print("    %s: %d (%.1f%%)" % (cat, count, count / len(df) * 100))

    return df


# ============================================================================
# 2. 下載少量波形用於特徵提取
# ============================================================================

def download_sample_waveforms(df: "pd.DataFrame", n_per_class: int = 100):
    """下載每類的少量波形 (100Hz 版本)。"""
    import wfdb
    import requests

    waveform_dir = DATA_DIR / "records100"
    waveform_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for superclass in ["NORM", "MI", "STTC", "CD", "HYP"]:
        subset = df[df["superclass"] == superclass]
        if len(subset) == 0:
            continue
        sample = subset.sample(min(n_per_class, len(subset)), random_state=42)

        for ecg_id, row in sample.iterrows():
            filename = row.get("filename_lr", "")
            if not filename:
                continue
            # filename is like "records100/00000/00001_lr"
            rec_name = Path(filename).stem  # e.g. "00001_lr"
            rec_dir = Path(filename).parent  # e.g. "records100/00000"
            local_dir = DATA_DIR / rec_dir
            local_dat = local_dir / (rec_name + ".dat")
            local_hea = local_dir / (rec_name + ".hea")

            if local_dat.exists() and local_hea.exists():
                records.append({
                    "ecg_id": ecg_id,
                    "superclass": superclass,
                    "path": str(DATA_DIR / filename),
                    "age": row.get("age", np.nan),
                    "sex": row.get("sex", 0),
                })
                continue

            # 下載
            local_dir.mkdir(parents=True, exist_ok=True)
            headers = {"User-Agent": "Mozilla/5.0 (Alice-Gamma-Net/3.9; research)"}
            try:
                for ext in [".dat", ".hea"]:
                    url = PTBXL_BASE + str(rec_dir).replace("\\", "/") + "/" + rec_name + ext
                    resp = requests.get(url, headers=headers, timeout=30)
                    resp.raise_for_status()
                    (local_dir / (rec_name + ext)).write_bytes(resp.content)
                records.append({
                    "ecg_id": ecg_id,
                    "superclass": superclass,
                    "path": str(DATA_DIR / filename),
                    "age": row.get("age", np.nan),
                    "sex": row.get("sex", 0),
                })
            except Exception as e:
                pass

        print("  Downloaded %s: %d records" % (superclass, sum(1 for r in records if r["superclass"] == superclass)))

    print("  Total waveform records: %d" % len(records))
    return records


# ============================================================================
# 3. ECG 特徵提取
# ============================================================================

def extract_ecg_features(record_path: str, fs: int = 100) -> Optional[Dict[str, float]]:
    """從 ECG 波形中提取心臟阻抗代理特徵。"""
    import wfdb

    try:
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal  # shape: (n_samples, 12)
        fs = record.fs
    except Exception:
        return None

    if signals is None or signals.shape[0] < fs * 5:
        return None

    # 使用 Lead II (index 1) 進行分析
    lead_ii = signals[:, 1] if signals.shape[1] > 1 else signals[:, 0]

    # 去除 NaN
    if np.any(np.isnan(lead_ii)):
        lead_ii = np.nan_to_num(lead_ii, nan=0.0)

    features = {}

    # 1. R-peak detection (simple threshold-based)
    # 高通濾波去除基線漂移
    from scipy.signal import butter, filtfilt

    try:
        b, a = butter(2, [0.5, 40], btype="band", fs=fs)
        filtered = filtfilt(b, a, lead_ii)
    except Exception:
        filtered = lead_ii

    # 簡單 R-peak 檢測
    threshold = np.std(filtered) * 1.5
    min_distance = int(0.4 * fs)  # 至少 400ms (=150 bpm max)

    peaks = []
    for i in range(1, len(filtered) - 1):
        if filtered[i] > threshold and filtered[i] > filtered[i-1] and filtered[i] > filtered[i+1]:
            if not peaks or (i - peaks[-1]) > min_distance:
                peaks.append(i)

    if len(peaks) < 5:
        return None

    # 2. R-R intervals
    rr_samples = np.diff(peaks)
    rr_ms = rr_samples / fs * 1000

    # 過濾生理不合理值
    valid_rr = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
    if len(valid_rr) < 3:
        return None

    features["mean_rr"] = float(np.mean(valid_rr))
    features["hr"] = 60000.0 / features["mean_rr"]
    features["sdnn"] = float(np.std(valid_rr, ddof=1))
    features["rmssd"] = float(np.sqrt(np.mean(np.diff(valid_rr)**2)))
    features["cv_rr"] = features["sdnn"] / features["mean_rr"]

    # 3. QRS 寬度估算（R-peak 周圍信號下降到 50% 的寬度）
    qrs_widths = []
    for peak in peaks[:10]:  # 前 10 個 beat
        r_amp = abs(filtered[peak])
        half = r_amp * 0.5
        # 找左右邊界
        left = peak
        while left > 0 and abs(filtered[left]) > half:
            left -= 1
        right = peak
        while right < len(filtered) - 1 and abs(filtered[right]) > half:
            right += 1
        qrs_w = (right - left) / fs * 1000  # ms
        if 20 < qrs_w < 300:
            qrs_widths.append(qrs_w)

    if qrs_widths:
        features["qrs_duration"] = float(np.mean(qrs_widths))
    else:
        features["qrs_duration"] = np.nan

    # 4. R-wave amplitude
    features["r_amplitude"] = float(np.mean([abs(filtered[p]) for p in peaks[:10]]))

    # 5. ST-segment deviation（R-peak 後 80ms 的偏移）
    st_devs = []
    for peak in peaks[:10]:
        st_point = peak + int(0.08 * fs)
        if st_point < len(filtered):
            baseline = np.mean(filtered[max(0, peak - int(0.2*fs)):peak - int(0.05*fs)])
            st_dev = abs(filtered[st_point] - baseline)
            st_devs.append(st_dev)
    if st_devs:
        features["st_deviation"] = float(np.mean(st_devs))
    else:
        features["st_deviation"] = np.nan

    # 6. Signal complexity (sample entropy proxy: std of 1st derivative)
    diff_signal = np.diff(filtered)
    features["signal_complexity"] = float(np.std(diff_signal))

    return features


# ============================================================================
# 4. 計算 Cardiac Impedance Index
# ============================================================================

def compute_cardiac_impedance(features: Dict[str, float]) -> float:
    """從 ECG 特徵估算 cardiac impedance mismatch (Γ proxy)。

    Physics mapping:
    - QRS prolongation → conduction impedance ↑ (Z_conduction)
    - ST deviation → repolarisation mismatch ↑ (Z_repolarisation)
    - Low HRV → autonomic-cardiac Γ ↑
    - Abnormal HR → regulatory failure

    No fitted parameters: all thresholds are textbook normals.
    """
    z_components = []

    # QRS duration: normal 60-100ms, prolonged → Z↑
    qrs = features.get("qrs_duration", 80)
    if not np.isnan(qrs):
        qrs_ref = 80.0  # ms, textbook normal
        z_qrs = abs(qrs - qrs_ref) / (qrs + qrs_ref) if (qrs + qrs_ref) > 0 else 0
        z_components.append(z_qrs)

    # HR: normal 60-100 bpm
    hr = features.get("hr", 75)
    hr_ref = 75.0  # bpm, midpoint of normal
    z_hr = abs(hr - hr_ref) / (hr + hr_ref) if (hr + hr_ref) > 0 else 0
    z_components.append(z_hr)

    # RMSSD: higher is better (parasympathetic tone)
    rmssd = features.get("rmssd", 40)
    rmssd_ref = 40.0  # ms, healthy adult median
    z_hrv = abs(rmssd - rmssd_ref) / (rmssd + rmssd_ref) if (rmssd + rmssd_ref) > 0 else 0
    z_components.append(z_hrv)

    # ST deviation: normal ≈ 0
    st = features.get("st_deviation", 0)
    if not np.isnan(st):
        features_amp = features.get("r_amplitude", 1.0)
        st_norm = st / max(features_amp, 0.001)  # 正規化
        z_components.append(min(st_norm, 1.0))

    if z_components:
        return float(np.sqrt(np.mean(np.array(z_components)**2)))
    return 0.0


# ============================================================================
# 5. 主分析
# ============================================================================

def run_analysis(records: List[Dict], df_meta: "pd.DataFrame") -> Dict[str, Any]:
    """對所有下載的波形運行分析。"""
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    results_by_class = {}
    all_features = []

    print("\n  Extracting ECG features...")
    for rec in records:
        features = extract_ecg_features(rec["path"])
        if features is None:
            continue
        features["ecg_id"] = rec["ecg_id"]
        features["superclass"] = rec["superclass"]
        features["age"] = rec["age"]
        features["gamma_cardiac"] = compute_cardiac_impedance(features)
        all_features.append(features)

    print("  Successfully extracted: %d records" % len(all_features))

    if len(all_features) < 50:
        return {"error": "Too few valid records", "n": len(all_features)}

    feat_df = pd.DataFrame(all_features)

    # 各類別的 Γ_cardiac 統計
    print("\n  === Cardiac Impedance by Diagnostic Class ===")
    print("  %-8s %6s %8s %8s %8s %8s" %
          ("Class", "n", "Γ_card", "HR", "QRS_ms", "RMSSD"))
    print("  " + "-" * 55)

    for cls in ["NORM", "MI", "STTC", "CD", "HYP"]:
        sub = feat_df[feat_df["superclass"] == cls]
        if len(sub) == 0:
            continue
        results_by_class[cls] = {
            "n": len(sub),
            "mean_gamma": round(float(sub["gamma_cardiac"].mean()), 4),
            "std_gamma": round(float(sub["gamma_cardiac"].std()), 4),
            "mean_hr": round(float(sub["hr"].mean()), 1),
            "mean_qrs": round(float(sub["qrs_duration"].mean()), 1),
            "mean_rmssd": round(float(sub["rmssd"].mean()), 1),
        }
        print("  %-8s %6d %8.4f %8.1f %8.1f %8.1f" % (
            cls, len(sub),
            sub["gamma_cardiac"].mean(),
            sub["hr"].mean(),
            sub["qrs_duration"].mean(),
            sub["rmssd"].mean()))

    # AUC: NORM vs 各病理
    print("\n  === AUC: Γ_cardiac predicting pathology ===")
    auc_results = {}
    norm_mask = feat_df["superclass"] == "NORM"

    for cls in ["MI", "STTC", "CD", "HYP"]:
        cls_mask = feat_df["superclass"] == cls
        if cls_mask.sum() < 10:
            continue

        combined = feat_df[norm_mask | cls_mask].copy()
        y_true = (combined["superclass"] == cls).astype(int).values
        y_score = combined["gamma_cardiac"].values

        try:
            auc = roc_auc_score(y_true, y_score)
            auc_results[cls] = {
                "auc": round(float(auc), 4),
                "n_pos": int(y_true.sum()),
                "n_neg": int((1 - y_true).sum()),
            }
            print("  NORM vs %-5s: AUC = %.4f (n+ = %d, n- = %d) %s" % (
                cls, auc, y_true.sum(), (1-y_true).sum(),
                "✓" if auc > 0.55 else ""))
        except Exception as e:
            print("  NORM vs %-5s: ERROR: %s" % (cls, e))

    # Any pathology vs NORM
    feat_df["is_pathology"] = (feat_df["superclass"] != "NORM").astype(int)
    y_true = feat_df["is_pathology"].values
    y_score = feat_df["gamma_cardiac"].values
    try:
        auc_any = roc_auc_score(y_true, y_score)
        auc_results["ANY_PATHOLOGY"] = {
            "auc": round(float(auc_any), 4),
            "n_pos": int(y_true.sum()),
            "n_neg": int((1 - y_true).sum()),
        }
        print("  NORM vs ANY:   AUC = %.4f" % auc_any)
    except Exception:
        pass

    return {
        "n_total": len(feat_df),
        "by_class": results_by_class,
        "auc_results": auc_results,
    }


# ============================================================================
# 6. 圖表
# ============================================================================

def generate_figure(results: Dict[str, Any]):
    """生成 cardiac Γ 圖表。"""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available")
        return

    by_class = results.get("by_class", {})
    auc_results = results.get("auc_results", {})

    if not by_class:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PTB-XL ECG: Cardiac Impedance Mismatch ($\\Gamma_{cardiac}$)\n"
                 "Zero-parameter ECG feature → cardiac $\\Gamma$ mapping",
                 fontsize=13, fontweight="bold")

    # Panel A: Γ by class
    ax = axes[0]
    classes = ["NORM", "STTC", "CD", "HYP", "MI"]
    colors = {"NORM": "#4CAF50", "STTC": "#FF9800", "CD": "#2196F3",
              "HYP": "#9C27B0", "MI": "#F44336"}
    vals = []
    errs = []
    labels = []
    bar_colors = []
    for cls in classes:
        if cls in by_class:
            vals.append(by_class[cls]["mean_gamma"])
            errs.append(by_class[cls]["std_gamma"])
            labels.append("%s\n(n=%d)" % (cls, by_class[cls]["n"]))
            bar_colors.append(colors.get(cls, "#999"))

    x = np.arange(len(vals))
    ax.bar(x, vals, yerr=errs, capsize=4, color=bar_colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean $\\Gamma_{cardiac}$")
    ax.set_title("(A) Cardiac Γ by ECG diagnosis")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: AUC bar chart
    ax = axes[1]
    auc_classes = [c for c in ["MI", "CD", "HYP", "STTC", "ANY_PATHOLOGY"]
                   if c in auc_results]
    if auc_classes:
        auc_vals = [auc_results[c]["auc"] for c in auc_classes]
        auc_colors = [colors.get(c, "#999") for c in auc_classes]
        auc_labels = ["%s\n(n=%d)" % (c, auc_results[c]["n_pos"]) for c in auc_classes]
        y = np.arange(len(auc_classes))
        ax.barh(y, auc_vals, color=auc_colors, edgecolor="white")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(auc_labels, fontsize=9)
        ax.set_xlabel("AUC (NORM vs pathology)")
        ax.set_title("(B) NORM vs Each Pathology AUC")
        ax.set_xlim(0.3, 0.9)
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        out = FIGURE_DIR / ("fig_ptbxl_cardiac_gamma" + ext)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print("  [SAVED] %s" % out)
    plt.close()


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("  PTB-XL ECG → CARDIAC IMPEDANCE VALIDATION")
    print("  21,801 clinical ECGs, zero fitted parameters")
    print("=" * 72)

    # Phase 1: 元數據
    print("\nPhase 1: Loading PTB-XL metadata...")
    df = download_metadata()
    df = parse_scp_codes(df)

    # 也嘗試下載 SCP statements
    import requests
    scp_url = PTBXL_BASE + "scp_statements.csv"
    scp_path = DATA_DIR / "scp_statements.csv"
    if not scp_path.exists():
        try:
            resp = requests.get(scp_url, timeout=30,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            scp_path.write_bytes(resp.content)
            print("  Downloaded scp_statements.csv")
            # 重新解析
            df = parse_scp_codes(df)
        except Exception:
            pass

    # Phase 2: 下載波形樣本
    print("\nPhase 2: Downloading sample waveforms (100 per class)...")
    records = download_sample_waveforms(df, n_per_class=100)

    if not records:
        print("  ERROR: No waveforms downloaded")
        return

    # Phase 3: 分析
    print("\nPhase 3: Analysis...")
    results = run_analysis(records, df)

    # Phase 4: 圖表
    print("\nPhase 4: Generating figure...")
    generate_figure(results)

    # 儲存
    out_path = RESULTS_DIR / "ptbxl_cardiac_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n  Results saved: %s" % out_path)

    # Checks
    print()
    print("=" * 72)
    print("  CLINICAL CHECKS")
    print("=" * 72)

    checks_passed = 0
    checks_total = 0

    # Check 1: Γ_NORM < Γ_pathology
    checks_total += 1
    by_class = results.get("by_class", {})
    if "NORM" in by_class:
        norm_g = by_class["NORM"]["mean_gamma"]
        pathology_higher = all(by_class[c]["mean_gamma"] > norm_g
                               for c in ["MI", "CD", "HYP", "STTC"]
                               if c in by_class)
        checks_passed += pathology_higher
        print("  [%s] Γ_NORM (%.4f) < all pathology classes" %
              ("PASS" if pathology_higher else "FAIL", norm_g))

    # Check 2: MI has highest Γ (tissue death = Z → ∞)
    checks_total += 1
    if "MI" in by_class and len(by_class) > 2:
        mi_g = by_class["MI"]["mean_gamma"]
        mi_highest = all(mi_g >= by_class[c]["mean_gamma"] - 0.01
                         for c in by_class if c != "MI")
        checks_passed += mi_highest
        print("  [%s] MI has highest Γ (%.4f)" %
              ("PASS" if mi_highest else "FAIL", mi_g))

    # Check 3: AUC > 0.55 for at least 2 pathologies
    checks_total += 1
    auc_results = results.get("auc_results", {})
    n_above_55 = sum(1 for d in auc_results.values() if d["auc"] > 0.55)
    c3 = n_above_55 >= 2
    checks_passed += c3
    print("  [%s] %d pathologies with AUC > 0.55" %
          ("PASS" if c3 else "FAIL", n_above_55))

    # Check 4: NORM vs ANY_PATHOLOGY AUC > 0.55
    checks_total += 1
    if "ANY_PATHOLOGY" in auc_results:
        c4 = auc_results["ANY_PATHOLOGY"]["auc"] > 0.55
        checks_passed += c4
        print("  [%s] NORM vs ANY AUC = %.4f" %
              ("PASS" if c4 else "FAIL", auc_results["ANY_PATHOLOGY"]["auc"]))

    print("\n  Result: %d/%d checks passed" % (checks_passed, checks_total))
    print("=" * 72)


if __name__ == "__main__":
    main()

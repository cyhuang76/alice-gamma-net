#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: PhysioNet PICS Infant ECG → HRV Impedance Analysis
================================================================

PURPOSE
-------
Analyze heart rate variability (HRV) from PhysioNet PICS infant ECG data
as a proxy for autonomic impedance matching quality.

PHYSICS HYPOTHESIS
  HRV reflects the quality of autonomic impedance matching between
  the cardiac pacemaker and the vagal/sympathetic control network.
  Higher HRV → better impedance matching → lower autonomic Γ.

  In neonates:
  - Autonomic nervous system is still maturing
  - HRV increases with post-natal age as C2 calibration progresses
  - Premature infants have lower HRV

DATA SOURCE
  PhysioNet PICS (Preterm Infants Cardio-Respiratory Signals)
  10 preterm infants, continuous ECG + respiratory recordings
  Format: WFDB (.dat + .hea + .qrsc annotation)
  License: Open Data Commons Attribution License v1.0

Author: Alice Smart System (automated verification)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "PhysioNet PICS"
FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "nhanes_results"
RESULTS_DIR.mkdir(exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
# 1. WFDB ECG 讀取
# ============================================================================

def read_ecg_header(hea_path: Path) -> Dict[str, Any]:
    """解析 WFDB .hea 標頭檔。"""
    info = {"fs": 0, "n_samples": 0, "n_channels": 0, "format": 0}
    with open(hea_path, "r") as f:
        first_line = f.readline().strip().split()
        if len(first_line) >= 4:
            info["n_channels"] = int(first_line[1])
            info["fs"] = int(first_line[2])
            info["n_samples"] = int(first_line[3])
    return info


def read_qrs_annotations(qrsc_path: Path) -> np.ndarray:
    """從 .qrsc 二進位註解檔讀取 QRS 位置。

    WFDB annotation format: each annotation is 2 bytes
    - bits 0-9: sample offset
    - bits 10-15: annotation type

    For large offsets, special codes are used.
    """
    data = np.fromfile(str(qrsc_path), dtype=np.uint16)
    if len(data) == 0:
        return np.array([])

    annotations = []
    sample_pos = 0
    i = 0
    while i < len(data):
        word = int(data[i])
        ann_type = (word >> 10) & 0x3F
        offset = word & 0x3FF

        if ann_type == 0 and offset == 0:
            break

        # SKIP: ann_type = 59 means "skip" (next 2 words = 32-bit offset)
        if ann_type == 59:
            if i + 1 < len(data):
                i += 1
                high = int(data[i])
                sample_pos += (high << 16) if i + 1 < len(data) else high
                if i + 1 < len(data):
                    i += 1
                    low = int(data[i])
                    sample_pos = (high << 16) | (offset << 0)
            i += 1
            continue

        # NOTE/AUX: ann_type = 63 means auxiliary data
        if ann_type == 63:
            n_aux = offset
            n_words = (n_aux + 1) // 2 + (1 if n_aux % 2 else 0)
            i += n_words
            i += 1
            continue

        sample_pos += offset

        # Normal beat annotations: type 1 (N), type 12 (Q), etc.
        if ann_type in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 38):
            annotations.append(sample_pos)

        i += 1

    return np.array(annotations, dtype=np.int64)


def read_ecg_data(dat_path: Path, hea_info: Dict) -> np.ndarray:
    """讀取 WFDB .dat 二進位 ECG 資料。"""
    fs = hea_info["fs"]
    n_samples = hea_info["n_samples"]
    n_ch = hea_info["n_channels"]

    # Format 16 (most common): 16-bit signed integers
    data = np.fromfile(str(dat_path), dtype=np.int16)
    if n_ch > 0:
        n_total = len(data) // n_ch
        data = data[:n_total * n_ch].reshape(-1, n_ch)
    return data


# ============================================================================
# 2. HRV 計算
# ============================================================================

def compute_hrv(rr_intervals_ms: np.ndarray) -> Dict[str, float]:
    """計算 HRV 指標。

    Parameters
    ----------
    rr_intervals_ms : R-R intervals in milliseconds

    Returns
    -------
    dict with: mean_rr, sdnn, rmssd, pnn50, cv_rr (coefficient of variation)
    """
    if len(rr_intervals_ms) < 10:
        return {"mean_rr": np.nan, "sdnn": np.nan, "rmssd": np.nan,
                "pnn50": np.nan, "cv_rr": np.nan, "n_beats": len(rr_intervals_ms)}

    # 過濾異常值（生理不可能的 R-R interval）
    valid = (rr_intervals_ms > 200) & (rr_intervals_ms < 2000)  # 30-300 bpm
    rr = rr_intervals_ms[valid]

    if len(rr) < 10:
        return {"mean_rr": np.nan, "sdnn": np.nan, "rmssd": np.nan,
                "pnn50": np.nan, "cv_rr": np.nan, "n_beats": len(rr)}

    mean_rr = float(np.mean(rr))
    sdnn = float(np.std(rr, ddof=1))

    # RMSSD: root mean square of successive differences
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))

    # pNN50: percentage of successive differences > 50ms
    pnn50 = float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100)

    # CV: coefficient of variation (SDNN / mean)
    cv_rr = sdnn / mean_rr if mean_rr > 0 else 0

    return {
        "mean_rr": round(mean_rr, 1),
        "sdnn": round(sdnn, 1),
        "rmssd": round(rmssd, 1),
        "pnn50": round(pnn50, 1),
        "cv_rr": round(cv_rr, 4),
        "n_beats": len(rr),
        "mean_hr": round(60000.0 / mean_rr, 1) if mean_rr > 0 else np.nan,
    }


def compute_windowed_hrv(qrs_samples: np.ndarray, fs: int,
                         window_sec: int = 300) -> List[Dict[str, float]]:
    """在滑動窗口內計算 HRV（看時間演變）。"""
    if len(qrs_samples) < 20:
        return []

    # 轉換為時間（秒）
    times = qrs_samples / fs
    rr_ms = np.diff(times) * 1000

    # 窗口
    total_time = times[-1] - times[0]
    n_windows = max(1, int(total_time / window_sec))

    results = []
    for i in range(n_windows):
        t_start = times[0] + i * window_sec
        t_end = t_start + window_sec

        # 該窗口內的 R-R
        mask = (times[1:] >= t_start) & (times[1:] < t_end)
        window_rr = rr_ms[mask]

        if len(window_rr) >= 10:
            hrv = compute_hrv(window_rr)
            hrv["window_start_sec"] = round(t_start, 0)
            hrv["window_hours"] = round(t_start / 3600, 2)
            results.append(hrv)

    return results


# ============================================================================
# 3. 分析所有嬰兒
# ============================================================================

def analyze_all_infants() -> Dict[str, Any]:
    """分析所有 PICS 嬰兒的 ECG。"""

    if not DATA_DIR.exists():
        print("  ERROR: Data directory '%s' not found" % DATA_DIR)
        return {"error": "No data directory"}

    results = {"infants": [], "summary": {}}

    # 找到所有嬰兒 ECG 檔案
    ecg_files = sorted(DATA_DIR.glob("infant*_ecg.hea"))
    if not ecg_files:
        print("  ERROR: No infant ECG files found")
        return {"error": "No ECG files"}

    print("  Found %d infant ECG recordings\n" % len(ecg_files))
    print("  %-10s %6s %10s %8s %8s %8s %8s %8s" %
          ("Infant", "HR", "Duration_h", "SDNN", "RMSSD", "pNN50", "CV_RR", "n_beats"))
    print("  " + "-" * 72)

    all_sdnn = []
    all_rmssd = []

    for hea_path in ecg_files:
        infant_name = hea_path.stem.replace("_ecg", "")

        # 讀取標頭
        info = read_ecg_header(hea_path)
        fs = info["fs"]
        if fs == 0:
            continue
        duration_hours = info["n_samples"] / fs / 3600

        # 讀取 QRS 註解
        qrsc_path = hea_path.with_suffix(".qrsc")
        if not qrsc_path.exists():
            print("  %-10s  [SKIP: no QRS annotations]" % infant_name)
            continue

        qrs = read_qrs_annotations(qrsc_path)
        if len(qrs) < 20:
            print("  %-10s  [SKIP: too few QRS detections: %d]" % (infant_name, len(qrs)))
            continue

        # 計算 R-R interval
        rr_ms = np.diff(qrs) / fs * 1000

        # 整體 HRV
        hrv = compute_hrv(rr_ms)

        # 窗口化 HRV（看趨勢）
        windowed = compute_windowed_hrv(qrs, fs, window_sec=300)

        if np.isnan(hrv["sdnn"]):
            continue

        all_sdnn.append(hrv["sdnn"])
        all_rmssd.append(hrv["rmssd"])

        print("  %-10s %6.1f %10.1f %8.1f %8.1f %8.1f %8.4f %8d" %
              (infant_name, hrv["mean_hr"], duration_hours,
               hrv["sdnn"], hrv["rmssd"], hrv["pnn50"], hrv["cv_rr"],
               hrv["n_beats"]))

        results["infants"].append({
            "name": infant_name,
            "fs": fs,
            "duration_hours": round(duration_hours, 1),
            "n_qrs": len(qrs),
            "hrv": hrv,
            "n_windows": len(windowed),
            "windowed_hrv": windowed[:10],  # 保存前 10 個窗口
        })

    # 匯總
    if all_sdnn:
        results["summary"] = {
            "n_infants": len(results["infants"]),
            "mean_sdnn": round(float(np.mean(all_sdnn)), 1),
            "std_sdnn": round(float(np.std(all_sdnn)), 1),
            "mean_rmssd": round(float(np.mean(all_rmssd)), 1),
            "std_rmssd": round(float(np.std(all_rmssd)), 1),
        }

    return results


# ============================================================================
# 4. 驗證檢查
# ============================================================================

def run_checks(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """對 HRV 結果進行物理一致性檢查。"""
    checks = []
    infants = results.get("infants", [])

    if not infants:
        return [{"name": "Data available", "passed": False}]

    # Check 1: 至少 5 名嬰兒有有效 HRV
    c1 = len(infants) >= 5
    checks.append({"name": "At least 5 infants with valid HRV",
                   "passed": c1, "n_valid": len(infants)})

    # Check 2: 所有嬰兒 HR > 100 bpm（新生兒正常）
    all_hr = [i["hrv"]["mean_hr"] for i in infants if not np.isnan(i["hrv"]["mean_hr"])]
    c2 = all(hr > 100 for hr in all_hr) if all_hr else False
    checks.append({"name": "All infant HR > 100 bpm (neonatal normal)",
                   "passed": c2, "hrs": [round(h, 0) for h in all_hr]})

    # Check 3: SDNN 在合理範圍（新生兒典型 5-50ms）
    all_sdnn = [i["hrv"]["sdnn"] for i in infants if not np.isnan(i["hrv"]["sdnn"])]
    c3 = all(2 < s < 200 for s in all_sdnn) if all_sdnn else False
    checks.append({"name": "SDNN in physiological range (2-200 ms)",
                   "passed": c3, "sdnn_range": "%.1f-%.1f" % (min(all_sdnn), max(all_sdnn)) if all_sdnn else "N/A"})

    # Check 4: HRV 個體差異存在（CV > 0）
    all_cv = [i["hrv"]["cv_rr"] for i in infants if not np.isnan(i["hrv"]["cv_rr"])]
    c4 = all(cv > 0 for cv in all_cv) if all_cv else False
    checks.append({"name": "All infants show non-zero HRV (cv_rr > 0)",
                   "passed": c4})

    # Check 5: 窗口化 HRV 顯示時間變異（睡眠/清醒週期）
    infants_with_windows = [i for i in infants if i["n_windows"] >= 5]
    if infants_with_windows:
        # 看同一嬰兒不同窗口的 SDNN 是否變化
        cv_of_sdnn = []
        for inf in infants_with_windows:
            win_sdnn = [w["sdnn"] for w in inf["windowed_hrv"] if not np.isnan(w["sdnn"])]
            if len(win_sdnn) >= 3:
                cv_of_sdnn.append(np.std(win_sdnn) / np.mean(win_sdnn))
        c5 = any(cv > 0.1 for cv in cv_of_sdnn) if cv_of_sdnn else False
        checks.append({"name": "HRV varies across time windows (sleep/wake cycling)",
                       "passed": c5})

    return checks


# ============================================================================
# 5. 圖表
# ============================================================================

def generate_figure(results: Dict[str, Any]):
    """生成 HRV 分析圖表。"""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available")
        return

    infants = results.get("infants", [])
    if not infants:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("PhysioNet PICS: Infant Heart Rate Variability\n"
                 "Autonomic Impedance Matching in Preterm Neonates",
                 fontsize=13, fontweight="bold")

    # Panel A: SDNN vs RMSSD scatter
    ax = axes[0]
    sdnn = [i["hrv"]["sdnn"] for i in infants]
    rmssd = [i["hrv"]["rmssd"] for i in infants]
    names = [i["name"].replace("infant", "P") for i in infants]
    ax.scatter(sdnn, rmssd, s=100, color="#2196F3", edgecolors="white", zorder=10)
    for n, s, r in zip(names, sdnn, rmssd):
        ax.annotate(n, (s, r), fontsize=7, ha="left", xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("SDNN (ms)")
    ax.set_ylabel("RMSSD (ms)")
    ax.set_title("(A) Time-domain HRV")
    ax.grid(True, alpha=0.3)

    # Panel B: HR vs SDNN
    ax = axes[1]
    hrs = [i["hrv"]["mean_hr"] for i in infants]
    ax.scatter(hrs, sdnn, s=100, color="#4CAF50", edgecolors="white", zorder=10)
    for n, h, s in zip(names, hrs, sdnn):
        ax.annotate(n, (h, s), fontsize=7, ha="left", xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("Mean Heart Rate (bpm)")
    ax.set_ylabel("SDNN (ms)")
    ax.set_title("(B) HR vs Autonomic Variability")
    ax.grid(True, alpha=0.3)

    # Panel C: Windowed HRV over time (longest recording)
    ax = axes[2]
    longest = max(infants, key=lambda x: x["n_windows"])
    if longest["windowed_hrv"]:
        hours = [w["window_hours"] for w in longest["windowed_hrv"]]
        win_sdnn = [w["sdnn"] for w in longest["windowed_hrv"]]
        ax.plot(hours, win_sdnn, "-o", color="#9C27B0", markersize=3, linewidth=1)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("SDNN (ms) per 5-min window")
        ax.set_title("(C) HRV over time (%s)" % longest["name"])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        out = FIGURE_DIR / ("fig_pics_infant_hrv" + ext)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print("  [SAVED] %s" % out)
    plt.close()


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("  PHYSIONET PICS: INFANT ECG → HRV ANALYSIS")
    print("  Autonomic impedance matching in preterm neonates")
    print("=" * 72)
    print()

    results = analyze_all_infants()

    if "error" in results:
        print("\n  ABORTED: %s" % results["error"])
        return

    # 檢查
    print("\n  === Clinical Checks ===")
    checks = run_checks(results)
    n_pass = 0
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print("  [%s] %s" % (mark, c["name"]))
        for k, v in c.items():
            if k not in ("name", "passed"):
                print("         %s = %s" % (k, v))
        if c["passed"]:
            n_pass += 1

    total = len(checks)
    print("\n  Result: %d/%d checks passed" % (n_pass, total))

    # 圖表
    print("\n  Generating figure...")
    generate_figure(results)

    # 儲存
    import json
    def fix_json(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    out_path = RESULTS_DIR / "pics_infant_hrv_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=fix_json)
    print("  Results saved: %s" % out_path)

    # 匯總
    print()
    print("=" * 72)
    if results.get("summary"):
        s = results["summary"]
        print("  %d infants analyzed" % s["n_infants"])
        print("  Mean SDNN = %.1f +/- %.1f ms" % (s["mean_sdnn"], s["std_sdnn"]))
        print("  Mean RMSSD = %.1f +/- %.1f ms" % (s["mean_rmssd"], s["std_rmssd"]))
    print("  %d/%d checks passed" % (n_pass, total))
    print("=" * 72)


if __name__ == "__main__":
    main()

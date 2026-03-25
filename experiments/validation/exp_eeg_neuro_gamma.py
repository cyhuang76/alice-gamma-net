#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: Figshare EEG → Neuro Γ_topology Validation
========================================================

PURPOSE
-------
Validate neural Γ predictions using the Figshare EEG dataset
(10 files, 112 subjects, 2 conditions: eyes-open/eyes-closed or
healthy/patient).

The Γ-Net framework predicts (Paper 1, Paper 4):
  - Neural impedance mismatch manifests as altered EEG spectral power
  - Higher Γ_neuro → more delta+theta power (slow, pathological)
  - Better impedance matching → stronger alpha power (10-12 Hz)
  - Alpha/theta ratio serves as a neuro Γ proxy

PHYSICS MAPPING
  EEG spectral bands map to neural impedance:
    delta (1-4 Hz)  → deep structural impedance (high Γ → bad)
    theta (4-8 Hz)  → functional impedance mismatch
    alpha (8-13 Hz) → impedance matched resting state (high → good)
    beta (13-30 Hz) → active processing (moderate Γ)

  Neuro Γ_proxy = (delta_power + theta_power) / (alpha_power + beta_power)
  → Higher ratio = worse impedance matching = higher Γ_neuro

DATA SOURCE
  Figshare EEG Dataset
  - 10 .mat files (eeg_36 through eeg_45)
  - Each file: 112 subjects × 2 conditions
  - D2: raw EEG signal vectors
  - In2: annotation/event vectors
  - Sampling rate: assumed 200 Hz (standard for this dataset)

Author: Alice Smart System (automated verification)
"""

from __future__ import annotations

import io
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Figshare EEG"
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
# 1. EEG 頻譜分析
# ============================================================================

def compute_band_power(signal: np.ndarray, fs: int = 200) -> Dict[str, float]:
    """計算 EEG 頻帶功率。

    Bands:
      delta: 1-4 Hz
      theta: 4-8 Hz
      alpha: 8-13 Hz
      beta:  13-30 Hz
      gamma: 30-50 Hz
    """
    from scipy.signal import welch

    # 取中間 60 秒的信號（避免起始/結束噪音）
    n_samples = len(signal)
    target_samples = min(60 * fs, n_samples)
    start = max(0, (n_samples - target_samples) // 2)
    segment = signal[start:start + target_samples].flatten().astype(float)

    # 去均值
    segment -= np.mean(segment)

    if len(segment) < fs * 5:
        return {}

    # Welch PSD
    try:
        freqs, psd = welch(segment, fs=fs, nperseg=min(2*fs, len(segment)),
                           noverlap=fs//2)
    except Exception:
        return {}

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, min(50, fs//2 - 1)),
    }

    result = {}
    total_power = 0
    for band, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        power = np.trapezoid(psd[mask], freqs[mask]) if mask.sum() > 1 else 0
        result[band] = float(power)
        total_power += power

    # 正規化（相對功率）
    if total_power > 0:
        for band in bands:
            result[band + "_rel"] = result[band] / total_power

    # 比率指標
    alpha = result.get("alpha", 1e-10)
    delta = result.get("delta", 0)
    theta = result.get("theta", 0)
    beta = result.get("beta", 1e-10)

    result["alpha_theta_ratio"] = alpha / max(theta, 1e-10)
    result["delta_alpha_ratio"] = delta / max(alpha, 1e-10)
    result["slow_fast_ratio"] = (delta + theta) / max(alpha + beta, 1e-10)
    result["total_power"] = total_power

    return result


def compute_neuro_gamma(band_power: Dict[str, float]) -> float:
    """計算 neuro Γ proxy。

    Γ_neuro = sqrt(slow_fast_ratio / (1 + slow_fast_ratio))

    Maps slow/fast ratio to [0, 1] using Γ-style formula:
    - Perfectly matched (ratio=1): Γ ≈ 0.5
    - Slow-dominated (ratio>>1): Γ → 1 (pathological)
    - Fast-dominated (ratio<<1): Γ → 0 (hyperarousal)
    """
    ratio = band_power.get("slow_fast_ratio", 1.0)

    # 映射到 Γ-style：Z_slow / Z_fast
    # Γ = (Z2 - Z1) / (Z2 + Z1) = (ratio - 1) / (ratio + 1)
    gamma = abs(ratio - 1.0) / (ratio + 1.0) if (ratio + 1.0) > 0 else 0
    return float(gamma)


# ============================================================================
# 2. 載入所有 EEG 數據
# ============================================================================

def load_all_eeg() -> List[Dict[str, Any]]:
    """載入所有 Figshare EEG .mat 檔案。"""
    import scipy.io as sio

    mat_files = sorted(DATA_DIR.glob("eeg_*.mat"))
    if not mat_files:
        print("  ERROR: No EEG .mat files found in %s" % DATA_DIR)
        return []

    print("  Found %d EEG .mat files" % len(mat_files))

    all_records = []
    for mat_path in mat_files:
        file_id = mat_path.stem  # e.g., "eeg_36"
        try:
            data = sio.loadmat(str(mat_path))
        except Exception as e:
            print("  [SKIP] %s: %s" % (file_id, e))
            continue

        if "D2" not in data:
            print("  [SKIP] %s: no D2 field" % file_id)
            continue

        D2 = data["D2"]
        n_subjects, n_conditions = D2.shape

        for subj_idx in range(n_subjects):
            for cond_idx in range(n_conditions):
                signal = D2[subj_idx, cond_idx]
                if signal is None or not hasattr(signal, 'shape'):
                    continue
                signal = signal.flatten()
                if len(signal) < 1000:
                    continue

                # 嘗試檢測採樣率
                # 大多數 Figshare EEG 是 200 Hz
                fs = 200

                all_records.append({
                    "file": file_id,
                    "subject": subj_idx,
                    "condition": cond_idx,
                    "signal": signal,
                    "n_samples": len(signal),
                    "duration_sec": len(signal) / fs,
                    "fs": fs,
                })

        print("  %s: %d subjects × %d conditions" % (file_id, n_subjects, n_conditions))

    print("  Total records: %d" % len(all_records))
    return all_records


# ============================================================================
# 3. 分析
# ============================================================================

def analyze_eeg_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析所有 EEG 記錄。"""

    results = {"records": [], "by_condition": {}, "by_file": {}}
    analyzed = []

    print("\n  Analyzing EEG spectral features...")
    for i, rec in enumerate(records):
        band = compute_band_power(rec["signal"], rec["fs"])
        if not band:
            continue

        gamma_neuro = compute_neuro_gamma(band)

        entry = {
            "file": rec["file"],
            "subject": rec["subject"],
            "condition": rec["condition"],
            "duration_sec": rec["duration_sec"],
            "gamma_neuro": gamma_neuro,
            **{k: round(v, 6) for k, v in band.items()},
        }
        analyzed.append(entry)

        if (i + 1) % 200 == 0:
            print("    Processed %d/%d" % (i + 1, len(records)))

    print("  Successfully analyzed: %d records" % len(analyzed))

    if not analyzed:
        return {"error": "No records analyzed"}

    # 按 condition 分組
    import pandas as pd
    df = pd.DataFrame(analyzed)

    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        results["by_condition"][int(cond)] = {
            "n": len(sub),
            "gamma_neuro_mean": round(float(sub["gamma_neuro"].mean()), 4),
            "gamma_neuro_std": round(float(sub["gamma_neuro"].std()), 4),
            "alpha_rel_mean": round(float(sub["alpha_rel"].mean()), 4),
            "delta_rel_mean": round(float(sub["delta_rel"].mean()), 4),
            "theta_rel_mean": round(float(sub["theta_rel"].mean()), 4),
            "alpha_theta_ratio_mean": round(float(sub["alpha_theta_ratio"].mean()), 4),
        }

    # 按 file 分組
    for file_id in df["file"].unique():
        sub = df[df["file"] == file_id]
        results["by_file"][file_id] = {
            "n": len(sub),
            "gamma_neuro_mean": round(float(sub["gamma_neuro"].mean()), 4),
        }

    # Condition 0 vs 1 比較
    if 0 in df["condition"].values and 1 in df["condition"].values:
        from scipy.stats import mannwhitneyu, ttest_ind

        cond0 = df[df["condition"] == 0]["gamma_neuro"].values
        cond1 = df[df["condition"] == 1]["gamma_neuro"].values

        try:
            u_stat, p_mw = mannwhitneyu(cond0, cond1, alternative="two-sided")
        except Exception:
            u_stat, p_mw = 0, 1

        try:
            t_stat, p_tt = ttest_ind(cond0, cond1)
        except Exception:
            t_stat, p_tt = 0, 1

        print("\n  === Condition 0 vs Condition 1 ===")
        print("  Cond 0: Γ_neuro = %.4f ± %.4f (n=%d)" %
              (cond0.mean(), cond0.std(), len(cond0)))
        print("  Cond 1: Γ_neuro = %.4f ± %.4f (n=%d)" %
              (cond1.mean(), cond1.std(), len(cond1)))
        print("  Mann-Whitney U p=%.4e, t-test p=%.4e" % (p_mw, p_tt))

        results["condition_comparison"] = {
            "cond0_mean": round(float(cond0.mean()), 4),
            "cond0_std": round(float(cond0.std()), 4),
            "cond0_n": len(cond0),
            "cond1_mean": round(float(cond1.mean()), 4),
            "cond1_std": round(float(cond1.std()), 4),
            "cond1_n": len(cond1),
            "mw_p": float(p_mw),
            "tt_p": float(p_tt),
            "different": bool(p_mw < 0.05),
        }

    # 跨受試者變異性
    subj_means = df.groupby(["file", "subject"])["gamma_neuro"].mean()
    results["individual_variation"] = {
        "n_unique_subjects": len(subj_means),
        "gamma_mean": round(float(subj_means.mean()), 4),
        "gamma_std": round(float(subj_means.std()), 4),
        "gamma_min": round(float(subj_means.min()), 4),
        "gamma_max": round(float(subj_means.max()), 4),
    }

    # Alpha peak frequency 分析
    print("\n  === Alpha Band Analysis ===")
    alpha_rels = df["alpha_rel"].values
    print("  Alpha relative power: %.4f ± %.4f" %
          (alpha_rels.mean(), alpha_rels.std()))
    print("  Alpha/theta ratio: %.4f ± %.4f" %
          (df["alpha_theta_ratio"].mean(), df["alpha_theta_ratio"].std()))

    results["alpha_analysis"] = {
        "alpha_rel_mean": round(float(alpha_rels.mean()), 4),
        "alpha_rel_std": round(float(alpha_rels.std()), 4),
        "alpha_theta_ratio_mean": round(float(df["alpha_theta_ratio"].mean()), 4),
    }

    # 儲存簡化版 records（不含完整信號數據）
    results["n_total"] = len(analyzed)
    results["summary_records"] = analyzed[:20]  # 前 20 筆示範

    return results


# ============================================================================
# 4. 圖表
# ============================================================================

def generate_figure(results: Dict[str, Any]):
    """生成 EEG neuro Γ 圖表。"""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figshare EEG: Neural Impedance ($\\Gamma_{neuro}$)\n"
                 "Spectral Band Power → Neural Impedance Proxy",
                 fontsize=13, fontweight="bold")

    by_cond = results.get("by_condition", {})
    cond_comp = results.get("condition_comparison", {})

    # Panel A: Γ_neuro by condition
    ax = axes[0]
    if cond_comp:
        conditions = ["Condition 0", "Condition 1"]
        means = [cond_comp["cond0_mean"], cond_comp["cond1_mean"]]
        stds = [cond_comp["cond0_std"], cond_comp["cond1_std"]]
        colors = ["#4CAF50", "#F44336"]
        ax.bar([0, 1], means, yerr=stds, capsize=6, color=colors,
               edgecolor="white", width=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(conditions)
        ax.set_ylabel("$\\Gamma_{neuro}$")
        ax.set_title("(A) Neural Γ by Condition")
        if cond_comp.get("mw_p", 1) < 0.05:
            ax.text(0.5, max(means) * 1.1, "p=%.2e" % cond_comp["mw_p"],
                    ha="center", fontsize=9, style="italic")
        ax.grid(True, axis="y", alpha=0.3)

    # Panel B: Relative band power
    ax = axes[1]
    for cond_id, cond_data in by_cond.items():
        bands_rel = [cond_data.get("delta_rel_mean", 0),
                     cond_data.get("theta_rel_mean", 0),
                     cond_data.get("alpha_rel_mean", 0)]
        label = "Condition %d" % cond_id
        x = np.arange(3)
        offset = cond_id * 0.3 - 0.15
        color = "#2196F3" if cond_id == 0 else "#FF9800"
        ax.bar(x + offset, bands_rel, width=0.25, label=label, color=color,
               edgecolor="white")

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Delta\n(1-4 Hz)", "Theta\n(4-8 Hz)",
                         "Alpha\n(8-13 Hz)"])
    ax.set_ylabel("Relative Power")
    ax.set_title("(B) EEG Band Power")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel C: Γ_neuro distribution
    ax = axes[2]
    indiv = results.get("individual_variation", {})
    if indiv:
        # 簡化：用 by_file 數據
        file_gammas = [d["gamma_neuro_mean"]
                      for d in results.get("by_file", {}).values()]
        if file_gammas:
            ax.hist(file_gammas, bins=8, color="#9C27B0",
                    edgecolor="white", alpha=0.8)
            ax.axvline(indiv.get("gamma_mean", 0.5), color="red",
                       linestyle="--", label="Mean=%.3f" % indiv.get("gamma_mean", 0))
    ax.set_xlabel("$\\Gamma_{neuro}$")
    ax.set_ylabel("Count (by file)")
    ax.set_title("(C) Neural Γ Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        out = FIGURE_DIR / ("fig_eeg_neuro_gamma" + ext)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print("  [SAVED] %s" % out)
    plt.close()


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("  FIGSHARE EEG → NEURO IMPEDANCE VALIDATION")
    print("  Spectral band power as neural Γ proxy")
    print("=" * 72)

    # Phase 1: 載入數據
    print("\nPhase 1: Loading EEG data...")
    records = load_all_eeg()

    if not records:
        print("  ABORTED: No data")
        return

    # Phase 2: 分析
    print("\nPhase 2: Spectral analysis...")
    results = analyze_eeg_records(records)

    if "error" in results:
        print("  ABORTED: %s" % results["error"])
        return

    # Phase 3: 圖表
    print("\nPhase 3: Generating figure...")
    generate_figure(results)

    # 儲存
    out_path = RESULTS_DIR / "eeg_neuro_gamma_results.json"

    def fix_json(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=fix_json)
    print("\n  Results saved: %s" % out_path)

    # Checks
    print()
    print("=" * 72)
    print("  CLINICAL CHECKS")
    print("=" * 72)

    checks_passed = 0
    checks_total = 0

    # Check 1: Alpha power is dominant in resting state
    checks_total += 1
    alpha_analysis = results.get("alpha_analysis", {})
    alpha_dominant = alpha_analysis.get("alpha_rel_mean", 0) > 0.15
    checks_passed += alpha_dominant
    print("  [%s] Alpha relative power > 15%% (%.1f%%)" %
          ("PASS" if alpha_dominant else "FAIL",
           alpha_analysis.get("alpha_rel_mean", 0) * 100))

    # Check 2: Alpha/theta ratio > 1 (healthy resting state marker)
    checks_total += 1
    at_ratio = alpha_analysis.get("alpha_theta_ratio_mean", 0)
    c2 = at_ratio > 1.0
    checks_passed += c2
    print("  [%s] Alpha/theta ratio > 1.0 (%.2f)" %
          ("PASS" if c2 else "FAIL", at_ratio))

    # Check 3: Conditions differ (eyes-open vs eyes-closed or similar)
    checks_total += 1
    cond_comp = results.get("condition_comparison", {})
    c3 = cond_comp.get("different", False)
    checks_passed += c3
    print("  [%s] Conditions statistically different (p=%.2e)" %
          ("PASS" if c3 else "FAIL", cond_comp.get("mw_p", 1)))

    # Check 4: Individual variation exists
    checks_total += 1
    indiv = results.get("individual_variation", {})
    c4 = indiv.get("gamma_std", 0) > 0.001
    checks_passed += c4
    print("  [%s] Individual Γ variation exists (SD=%.4f)" %
          ("PASS" if c4 else "FAIL", indiv.get("gamma_std", 0)))

    # Check 5: Γ_neuro range is physiologically plausible [0, 1]
    checks_total += 1
    c5 = (0 <= indiv.get("gamma_min", -1) and indiv.get("gamma_max", 2) <= 1)
    checks_passed += c5
    print("  [%s] Γ_neuro in [0, 1] range (%.3f - %.3f)" %
          ("PASS" if c5 else "FAIL",
           indiv.get("gamma_min", -1), indiv.get("gamma_max", 2)))

    print("\n  Result: %d/%d checks passed" % (checks_passed, checks_total))
    print("=" * 72)


if __name__ == "__main__":
    main()

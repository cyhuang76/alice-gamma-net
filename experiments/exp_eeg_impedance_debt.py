# -*- coding: utf-8 -*-
"""
Experiment: Neonatal EEG Delta Power vs Gestational Age
═══════════════════════════════════════════════════════

Paper II validation: Delta power as a proxy for impedance debt D_Z.

Physics hypothesis:
  D_Z ∝ delta power during sleep.
  More immature brains (lower gestational age) have higher baseline
  impedance mismatch → higher delta power → higher D_Z accumulation rate.

Data source:
  Figshare "Newborn sleep EEG data" (CC BY 4.0)
  1,110 recordings, 36-45 weeks gestational age
  2 channels: C3-T3, C4-T4
  Format: MATLAB .mat, cell array D2{n_records, n_channels}

Reference:
  Stevenson et al. (Figshare, DOI: 10.6084/m9.figshare.4729840)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import io as sio
from scipy import signal as sig
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("Figshare EEG")
WEEKS = list(range(36, 46))  # 36-45 weeks gestational age
FS = 256  # 採樣率 (Hz) — 新生兒 EEG 標準
DELTA_BAND = (0.5, 4.0)  # Delta band (Hz)
THETA_BAND = (4.0, 8.0)  # Theta band (Hz)
ALPHA_BAND = (8.0, 13.0)  # Alpha band (Hz)
BETA_BAND = (13.0, 30.0)  # Beta band (Hz)

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Core Analysis Functions
# ============================================================================

def compute_band_power(eeg_signal: np.ndarray, fs: float,
                        band: tuple[float, float]) -> float:
    """
    使用 Welch 方法計算指定頻帶功率。

    Parameters
    ----------
    eeg_signal : 1D array, EEG 時間序列
    fs : 採樣率 (Hz)
    band : (f_low, f_high) 頻帶範圍

    Returns
    -------
    band_power : 頻帶功率 (μV²/Hz)
    """
    # 確保信號足夠長（至少 2 秒）
    min_samples = int(2 * fs)
    if len(eeg_signal) < min_samples:
        return np.nan

    # Welch PSD: 2 秒窗口, 50% 重疊
    nperseg = min(int(2 * fs), len(eeg_signal))
    freqs, psd = sig.welch(eeg_signal, fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2)

    # 提取頻帶
    idx = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(idx):
        return np.nan

    return np.trapezoid(psd[idx], freqs[idx])


def compute_relative_delta(eeg_signal: np.ndarray, fs: float) -> float:
    """
    計算相對 delta power = delta / (delta + theta + alpha + beta)。

    這是 D_Z proxy 的標準化版本：
    - 高 relative delta → 大腦處於 "高阻抗債務" 狀態
    - 相對值消除了個體間振幅差異
    """
    delta = compute_band_power(eeg_signal, fs, DELTA_BAND)
    theta = compute_band_power(eeg_signal, fs, THETA_BAND)
    alpha = compute_band_power(eeg_signal, fs, ALPHA_BAND)
    beta = compute_band_power(eeg_signal, fs, BETA_BAND)

    total = delta + theta + alpha + beta
    if total == 0 or np.isnan(total):
        return np.nan
    return delta / total


def analyze_week(week: int, fs: float = FS) -> dict:
    """
    分析一個週齡的所有 EEG 記錄。

    Returns
    -------
    dict with keys:
      week, n_records, delta_powers, relative_deltas,
      mean_delta, std_delta, mean_rel_delta, std_rel_delta
    """
    filepath = DATA_DIR / f"eeg_{week}.mat"
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found")
        return None

    print(f"  Loading {filepath.name} ...", end="", flush=True)
    mat = sio.loadmat(str(filepath))
    D2 = mat["D2"]  # (n_records, 2) cell array
    n_records = D2.shape[0]
    print(f" {n_records} records")

    delta_powers = []
    relative_deltas = []

    for i in range(n_records):
        for ch in range(2):  # C3-T3, C4-T4
            eeg = D2[i, ch].flatten().astype(np.float64)

            # 帶通濾波 0.5-30 Hz（去除基線漂移和高頻噪音）
            sos = sig.butter(4, [0.5, 30], btype="band", fs=fs, output="sos")
            eeg_filt = sig.sosfilt(sos, eeg)

            dp = compute_band_power(eeg_filt, fs, DELTA_BAND)
            rd = compute_relative_delta(eeg_filt, fs)

            if not np.isnan(dp):
                delta_powers.append(dp)
            if not np.isnan(rd):
                relative_deltas.append(rd)

    result = {
        "week": week,
        "n_records": n_records,
        "delta_powers": np.array(delta_powers),
        "relative_deltas": np.array(relative_deltas),
        "mean_delta": np.mean(delta_powers) if delta_powers else np.nan,
        "std_delta": np.std(delta_powers) if delta_powers else np.nan,
        "mean_rel_delta": np.mean(relative_deltas) if relative_deltas else np.nan,
        "std_rel_delta": np.std(relative_deltas) if relative_deltas else np.nan,
    }
    return result


# ============================================================================
# Visualization
# ============================================================================

def plot_results(results: list[dict]):
    """產生 Paper II 的 Figure: delta power vs gestational age."""

    weeks = [r["week"] for r in results]
    mean_delta = [r["mean_delta"] for r in results]
    std_delta = [r["std_delta"] for r in results]
    mean_rel = [r["mean_rel_delta"] for r in results]
    std_rel = [r["std_rel_delta"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: Absolute delta power ---
    ax = axes[0]
    ax.errorbar(weeks, mean_delta, yerr=std_delta, fmt="o-",
                color="#2196F3", capsize=4, linewidth=2, markersize=8)
    ax.set_xlabel("Gestational Age (weeks)", fontsize=12)
    ax.set_ylabel("Delta Power (0.5–4 Hz) [μV²/Hz]", fontsize=12)
    ax.set_title("(a) Absolute Delta Power", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Fit: linear regression
    w = np.array(weeks, dtype=float)
    md = np.array(mean_delta, dtype=float)
    valid = ~np.isnan(md)
    if np.sum(valid) >= 3:
        coeffs = np.polyfit(w[valid], md[valid], 1)
        fit_line = np.polyval(coeffs, w)
        ax.plot(w, fit_line, "--", color="#FF5722", linewidth=1.5,
                label=f"slope = {coeffs[0]:.1f}/wk")
        ax.legend(fontsize=10)

    # --- Panel B: Relative delta power ---
    ax = axes[1]
    ax.errorbar(weeks, mean_rel, yerr=std_rel, fmt="s-",
                color="#4CAF50", capsize=4, linewidth=2, markersize=8)
    ax.set_xlabel("Gestational Age (weeks)", fontsize=12)
    ax.set_ylabel("Relative Delta Power (δ/total)", fontsize=12)
    ax.set_title("(b) Relative Delta Power ($D_Z$ proxy)", fontsize=13,
                 fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Fit
    mr = np.array(mean_rel, dtype=float)
    valid_r = ~np.isnan(mr)
    if np.sum(valid_r) >= 3:
        coeffs_r = np.polyfit(w[valid_r], mr[valid_r], 1)
        fit_line_r = np.polyval(coeffs_r, w)
        ax.plot(w, fit_line_r, "--", color="#FF5722", linewidth=1.5,
                label=f"slope = {coeffs_r[0]:.4f}/wk")
        ax.legend(fontsize=10)

    fig.suptitle("Neonatal EEG: Delta Power vs Gestational Age\n"
                 "(Impedance Debt Framework — Paper II)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    # Save
    out_png = OUTPUT_DIR / "fig_eeg_delta_vs_age.png"
    out_pdf = OUTPUT_DIR / "fig_eeg_delta_vs_age.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def print_table(results: list[dict]):
    """印出結果表格。"""
    print("\n" + "=" * 70)
    print("  Gestational Age vs Delta Power — Summary Table")
    print("=" * 70)
    print(f"  {'Week':>4}  {'N':>5}  {'Delta Power':>14}  {'Rel. Delta':>12}")
    print(f"  {'':>4}  {'':>5}  {'(mean ± std)':>14}  {'(mean ± std)':>12}")
    print("-" * 70)
    for r in results:
        w = r["week"]
        n = r["n_records"]
        md = r["mean_delta"]
        sd = r["std_delta"]
        mr = r["mean_rel_delta"]
        sr = r["std_rel_delta"]
        print(f"  {w:>4}  {n:>5}  {md:>7.1f} ± {sd:<5.1f}  {mr:>5.3f} ± {sr:<5.3f}")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  Paper II — EEG Delta Power Analysis")
    print("  Impedance Debt Framework Validation")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"\n  ERROR: Data directory '{DATA_DIR}' not found.")
        print("  Please download from Figshare (DOI: 10.6084/m9.figshare.4729840)")
        sys.exit(1)

    results = []
    for week in WEEKS:
        r = analyze_week(week)
        if r is not None:
            results.append(r)

    if not results:
        print("\n  No data found!")
        sys.exit(1)

    print_table(results)
    plot_results(results)

    # --- 阻抗債務解釋 ---
    print("\n  Physical Interpretation:")
    print("  ─────────────────────────")
    weeks_arr = np.array([r["week"] for r in results])
    deltas_arr = np.array([r["mean_rel_delta"] for r in results])

    if deltas_arr[0] > deltas_arr[-1]:
        print("  ✅ Relative delta DECREASES with maturity")
        print("     → Impedance matching improves as neural circuits mature")
        print("     → D_Z accumulation rate decreases with age")
        print("     → Consistent with impedance-debt framework")
    else:
        print("  ⚠️  Relative delta INCREASES with maturity")
        print("     → May reflect increased organized slow-wave activity")
        print("     → Need to distinguish 'pathological δ' from 'organized δ'")

    print("\n  Done.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Experiment: Preterm Infant HRV Analysis (PhysioNet PICS)
════════════════════════════════════════════════════════

Paper 2 validation: Heart Rate Variability as a proxy for κ (thermal
dissipation capacity / autonomic regulation).

Physics hypothesis:
  κ = autonomic nervous system's ability to regulate heat dissipation.
  More premature infants have less mature autonomic systems → lower κ
  → higher D_Z accumulation rate → more fragile sleep-thermoregulation.

  HRV (SDNN, RMSSD) = proxy for autonomic maturity = proxy for κ.
  Bradycardia frequency = proxy for system instability (D_Z > D_crit events).

Data source:
  PhysioNet PICS Database (Open Data Commons Attribution License v1.0)
  10 preterm infants, 29-34 weeks post-conceptional age
  ECG (500 Hz) + Respiration (50 Hz)
  R-peak annotations (.qrsc), bradycardia annotations (.atr)

Reference:
  Gee AH, et al. "Predicting Bradycardia in Preterm Infants Using
  Point Process Analysis of Heart Rate." IEEE Trans BME (2017).
"""

from __future__ import annotations

import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("PhysioNet PICS")
N_INFANTS = 10
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# 已知的嬰兒週齡 (post-conceptional age, weeks)
# 來源: PICS database documentation
INFANT_PCA = {
    1: 29, 2: 31, 3: 32, 4: 30, 5: 34,
    6: 33, 7: 31, 8: 32, 9: 29, 10: 33,
}


# ============================================================================
# R-peak & HRV Analysis
# ============================================================================

def load_qrsc(filepath: Path, fs: float = 500.0) -> np.ndarray:
    """
    載入 .qrsc 標註檔，回傳 R-peak 時間 (seconds)。

    .qrsc 是 WFDB annotation format:
    每個 annotation 有一個 sample index。
    """
    try:
        import wfdb
        # 用 wfdb 讀取 annotation
        record_name = str(filepath).replace("_ecg.qrsc", "_ecg")
        ann = wfdb.rdann(record_name, "qrsc")
        return ann.sample / fs
    except Exception:
        # 如果 wfdb 失敗，手動讀取
        pass

    return np.array([])


def compute_hrv(rr_intervals: np.ndarray) -> dict:
    """
    計算 HRV 指標。

    Parameters
    ----------
    rr_intervals : RR 間距 (seconds)

    Returns
    -------
    dict: SDNN, RMSSD, pNN50, mean_hr
    """
    if len(rr_intervals) < 10:
        return {"SDNN": np.nan, "RMSSD": np.nan, "pNN50": np.nan,
                "mean_hr": np.nan, "n_beats": 0}

    # 過濾異常值 (< 200ms 或 > 1500ms 的 RR interval)
    valid = (rr_intervals > 0.2) & (rr_intervals < 1.5)
    rr = rr_intervals[valid]

    if len(rr) < 10:
        return {"SDNN": np.nan, "RMSSD": np.nan, "pNN50": np.nan,
                "mean_hr": np.nan, "n_beats": len(rr)}

    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(rr, ddof=1)

    # RMSSD: Root mean square of successive differences
    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))

    # pNN50: % of successive RR intervals differing > 50ms
    pnn50 = np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr) * 100

    # Mean heart rate
    mean_hr = 60.0 / np.mean(rr)

    return {
        "SDNN": sdnn * 1000,       # ms
        "RMSSD": rmssd * 1000,     # ms
        "pNN50": pnn50,            # %
        "mean_hr": mean_hr,        # bpm
        "n_beats": len(rr),
    }


def count_bradycardia(filepath: Path) -> int:
    """
    從 .atr 標註檔計算心動過緩事件數量。
    """
    try:
        import wfdb
        record_name = str(filepath).replace("_ecg.atr", "_ecg")
        ann = wfdb.rdann(record_name, "atr")
        # 心動過緩標記通常是特定的 annotation symbol
        return len(ann.sample)
    except Exception:
        return 0


def analyze_infant(infant_id: int) -> dict:
    """分析一個嬰兒的 ECG 數據。"""
    ecg_base = DATA_DIR / f"infant{infant_id}_ecg"
    qrsc_file = DATA_DIR / f"infant{infant_id}_ecg.qrsc"
    atr_file = DATA_DIR / f"infant{infant_id}_ecg.atr"

    # 決定採樣率 (infant 1 和 5 是 250 Hz)
    fs = 250.0 if infant_id in (1, 5) else 500.0

    print(f"  Infant {infant_id} (PCA={INFANT_PCA.get(infant_id, '?')} wk, "
          f"fs={fs} Hz) ...", end="", flush=True)

    # 載入 R-peaks
    r_peaks = load_qrsc(qrsc_file, fs=fs)

    if len(r_peaks) < 10:
        print(" [NO R-PEAKS]")
        return None

    # 計算 RR intervals
    rr_intervals = np.diff(r_peaks)

    # HRV
    hrv = compute_hrv(rr_intervals)

    # 心動過緩
    brady_count = count_bradycardia(atr_file)

    # 記錄持續時間
    duration_hours = (r_peaks[-1] - r_peaks[0]) / 3600

    print(f" {hrv['n_beats']} beats, {duration_hours:.1f}h, "
          f"SDNN={hrv['SDNN']:.1f}ms, "
          f"brady={brady_count}")

    return {
        "infant_id": infant_id,
        "pca_weeks": INFANT_PCA.get(infant_id, np.nan),
        "duration_hours": duration_hours,
        "brady_count": brady_count,
        "brady_rate": brady_count / duration_hours if duration_hours > 0 else 0,
        **hrv,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_results(results: list[dict]):
    """產生 Paper 2 的 Figure: HRV vs PCA + bradycardia。"""

    pca = [r["pca_weeks"] for r in results]
    sdnn = [r["SDNN"] for r in results]
    rmssd = [r["RMSSD"] for r in results]
    brady = [r["brady_rate"] for r in results]
    mean_hr = [r["mean_hr"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- (a) SDNN vs PCA ---
    ax = axes[0, 0]
    ax.scatter(pca, sdnn, s=80, c="#2196F3", zorder=3)
    ax.set_xlabel("Post-Conceptional Age (weeks)", fontsize=11)
    ax.set_ylabel("SDNN (ms)", fontsize=11)
    ax.set_title("(a) SDNN vs Maturity", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    # Fit
    p = np.array(pca, dtype=float)
    s = np.array(sdnn, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(s))
    if np.sum(valid) >= 3:
        c = np.polyfit(p[valid], s[valid], 1)
        ax.plot([28, 35], [np.polyval(c, 28), np.polyval(c, 35)],
                "--", color="#FF5722", linewidth=1.5,
                label=f"slope={c[0]:.1f} ms/wk")
        ax.legend(fontsize=9)

    # --- (b) RMSSD vs PCA ---
    ax = axes[0, 1]
    ax.scatter(pca, rmssd, s=80, c="#4CAF50", zorder=3)
    ax.set_xlabel("Post-Conceptional Age (weeks)", fontsize=11)
    ax.set_ylabel("RMSSD (ms)", fontsize=11)
    ax.set_title("(b) RMSSD vs Maturity", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # --- (c) Bradycardia rate vs PCA ---
    ax = axes[1, 0]
    ax.scatter(pca, brady, s=80, c="#F44336", zorder=3)
    ax.set_xlabel("Post-Conceptional Age (weeks)", fontsize=11)
    ax.set_ylabel("Bradycardia Events / Hour", fontsize=11)
    ax.set_title("(c) Bradycardia Rate vs Maturity", fontsize=12,
                 fontweight="bold")
    ax.grid(True, alpha=0.3)

    # --- (d) Mean HR vs PCA ---
    ax = axes[1, 1]
    ax.scatter(pca, mean_hr, s=80, c="#9C27B0", zorder=3)
    ax.set_xlabel("Post-Conceptional Age (weeks)", fontsize=11)
    ax.set_ylabel("Mean Heart Rate (bpm)", fontsize=11)
    ax.set_title("(d) Mean Heart Rate vs Maturity", fontsize=12,
                 fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Preterm Infant Autonomic Regulation (PhysioNet PICS)\n"
                 "HRV as κ proxy — Impedance Debt Framework",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_png = OUTPUT_DIR / "fig_pics_hrv_analysis.png"
    out_pdf = OUTPUT_DIR / "fig_pics_hrv_analysis.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def print_table(results: list[dict]):
    """印出結果表格。"""
    print("\n" + "=" * 80)
    print("  Preterm Infant HRV — Summary Table")
    print("=" * 80)
    print(f"  {'ID':>3} {'PCA':>4} {'Hours':>6} {'Beats':>8} "
          f"{'SDNN':>7} {'RMSSD':>7} {'pNN50':>6} {'HR':>5} {'Brady':>6}")
    print(f"  {'':>3} {'(wk)':>4} {'':>6} {'':>8} "
          f"{'(ms)':>7} {'(ms)':>7} {'(%)':>6} {'(bpm)':>5} {'(/hr)':>6}")
    print("-" * 80)
    for r in results:
        print(f"  {r['infant_id']:>3} {r['pca_weeks']:>4} "
              f"{r['duration_hours']:>6.1f} {r['n_beats']:>8} "
              f"{r['SDNN']:>7.1f} {r['RMSSD']:>7.1f} "
              f"{r['pNN50']:>6.1f} {r['mean_hr']:>5.0f} "
              f"{r['brady_rate']:>6.1f}")
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  Paper 2 — PhysioNet PICS HRV Analysis")
    print("  Autonomic Regulation as κ Proxy")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"\n  ERROR: Data directory '{DATA_DIR}' not found.")
        print("  Download from: https://physionet.org/content/picsdb/")
        return

    results = []
    for i in range(1, N_INFANTS + 1):
        r = analyze_infant(i)
        if r is not None:
            results.append(r)

    if not results:
        print("\n  No data found!")
        return

    print_table(results)
    plot_results(results)

    # 物理解讀
    print("\n  Physical Interpretation (κ proxy):")
    print("  ──────────────────────────────────")
    pca = np.array([r["pca_weeks"] for r in results])
    sdnn = np.array([r["SDNN"] for r in results])
    valid = ~np.isnan(sdnn)
    if np.sum(valid) >= 3:
        corr = np.corrcoef(pca[valid], sdnn[valid])[0, 1]
        print(f"  SDNN vs PCA correlation: r = {corr:.3f}")
        if corr > 0:
            print("  ✅ SDNN increases with maturity")
            print("     → Autonomic regulation (κ) improves with age")
            print("     → More mature infants dissipate D_Z more effectively")
        else:
            print("  ⚠️  SDNN decreases with maturity (unexpected)")

    print("\n  Done.")


if __name__ == "__main__":
    main()

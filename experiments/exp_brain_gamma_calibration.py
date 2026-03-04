# -*- coding: utf-8 -*-
"""
Experiment: Brain Γ-Map Calibration (Act III — Prototype)
══════════════════════════════════════════════════════════

PURPOSE
-------
Bridge the gap between Paper VI's organ-level Γ (blood biochemistry)
and Paper III's dual-field equations by calibrating brain-specific
Γ_n (neural) and Γ_v (vascular) from direct electrophysiological
measurements.

PHYSICS CHAIN
─────────────
  Paper I :  A[Γ] = ∫ Σ Γ_i² dt → min   (topology)
  Paper II:  H = (1-Γ_n²)(1-Γ_v²)         (dual-network health)
  Paper III: D_Z = ∫ |Γ|² P_in dt          (impedance debt)
             ∂Z/∂t = D∇²Z − ηΓJf(ρ) − χv_cat E(Γ²)Γρ − λZ
             ∂ρ/∂t = D_ρ∇²ρ − consumption + (1-Γ_v²)Q₀
  Paper VI:  Z_organ = Z_normal(1 + Σ w_j|δ_j|)  → NHANES 7/7
  ──────────────────────────────────────────────────────
  THIS EXP:  EEG band powers  →  Γ_n  (neural impedance matching)
             ECG HRV metrics  →  Γ_v  (vascular impedance matching)
             Combined         →  H_brain = (1-Γ_n²)(1-Γ_v²)

DATA SOURCES
────────────
  1. Figshare EEG  (CC BY 4.0) — 10 neonatal sleep EEG, 36-45 wk GA
     Stevenson et al., DOI: 10.6084/m9.figshare.4729840
  2. PhysioNet PICS (ODC-BY 1.0) — 10 preterm infants, ECG+resp
     Gee et al., IEEE Trans BME (2017)

MAPPING LOGIC (zero free parameters)
─────────────────────────────────────
  Γ_n from EEG:
    Z_n = Z_n0 × (1 + |δ_rel - δ_ref| + |θ_rel - θ_ref| + ...)
    where δ_ref, θ_ref etc. are textbook neonatal EEG norms.
    Γ_n = (Z_n - Z_n0) / (Z_n + Z_n0)
    Prediction: Γ_n decreases with gestational age (maturation).

  Γ_v from HRV:
    Z_v = Z_v0 × (1 + |SDNN_norm - 1| + |RMSSD_norm - 1| + ...)
    where norms are from healthy term neonatal references.
    Γ_v = (Z_v - Z_v0) / (Z_v + Z_v0)
    Prediction: Γ_v decreases with PCA (autonomic maturation).

  D_Z estimation:
    D_Z,n ≈ Γ_n² × P_delta  (delta power = proxy for P_in during sleep)
    D_Z,v ≈ Γ_v² × P_cardiac (mean HR = proxy for vascular P_in)

  Brain Health:
    H_brain = (1 - Γ_n²)(1 - Γ_v²)
    Prediction: H_brain increases with maturity.
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import io as sio
from scipy import signal as sig
from scipy import stats
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

EEG_DIR = Path("Figshare EEG")
PICS_DIR = Path("PhysioNet PICS")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# EEG parameters
FS_EEG = 256  # Hz
DELTA_BAND = (0.5, 4.0)
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
EEG_WEEKS = list(range(36, 46))

# Z_normal references (from Alice ORGAN_SYSTEMS)
Z_N0 = 80.0   # Neural Z_normal (Ω, from lab_mapping.py)
Z_V0 = 45.0   # Vascular Z_normal (Ω, from lab_mapping.py)

# Textbook neonatal EEG norms (term neonate, quiet sleep)
# Source: Shellhaas et al., J Clin Neurophysiol 2007;
#         André et al., Neurophysiol Clin 2010
# These are relative power fractions for a healthy term infant (40 wk)
EEG_REF = {
    "delta_rel": 0.70,   # Delta dominates in neonatal quiet sleep
    "theta_rel": 0.15,
    "alpha_rel": 0.10,
    "beta_rel":  0.05,
}

# Textbook neonatal HRV norms (healthy term infant)
# Source: Longin et al., Eur J Pediatr 2006;
#         Cardoso et al., Pediatr Res 2017
HRV_REF = {
    "SDNN_ms":   50.0,   # Healthy term neonate SDNN (ms)
    "RMSSD_ms":  30.0,   # Healthy term neonate RMSSD (ms)
    "mean_hr":  130.0,   # Healthy term neonate mean HR (bpm)
}

# PhysioNet PICS infant post-conceptional ages (weeks)
INFANT_PCA = {
    1: 29, 2: 31, 3: 32, 4: 30, 5: 34,
    6: 33, 7: 31, 8: 32, 9: 29, 10: 33,
}


# ============================================================================
# EEG Analysis → Γ_n
# ============================================================================

def compute_band_power(signal: np.ndarray, fs: float,
                       band: tuple[float, float]) -> float:
    """Welch PSD band power."""
    min_samples = int(2 * fs)
    if len(signal) < min_samples:
        return np.nan
    nperseg = min(int(2 * fs), len(signal))
    freqs, psd = sig.welch(signal, fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2)
    idx = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(idx):
        return np.nan
    return np.trapezoid(psd[idx], freqs[idx])


def compute_eeg_features(eeg_signal: np.ndarray, fs: float) -> dict:
    """Extract relative band powers and spectral entropy from EEG signal.

    Spectral entropy is the key Γ_n proxy:
    - Low entropy = well-organised rhythms = matched neural impedance = low Γ_n
    - High entropy = disorganised activity = mismatched impedance = high Γ_n

    This aligns with Paper III's framework: a mature brain with well-developed
    neural circuits has low Γ² (good impedance matching), which manifests as
    organised EEG rhythms (low spectral entropy).
    """
    # Bandpass 0.5-30 Hz
    sos = sig.butter(4, [0.5, 30], btype="band", fs=fs, output="sos")
    eeg_filt = sig.sosfilt(sos, eeg_signal)

    delta = compute_band_power(eeg_filt, fs, DELTA_BAND)
    theta = compute_band_power(eeg_filt, fs, THETA_BAND)
    alpha = compute_band_power(eeg_filt, fs, ALPHA_BAND)
    beta = compute_band_power(eeg_filt, fs, BETA_BAND)

    total = delta + theta + alpha + beta
    if total == 0 or np.isnan(total):
        return None

    # Relative band powers
    bands = np.array([delta, theta, alpha, beta])
    rel_bands = bands / total

    # Spectral entropy (Shannon entropy of normalised PSD)
    # H_spec = -Σ p_i log(p_i), normalised to [0, 1]
    # where p_i = relative power in each band
    # Low H_spec → organised (dominated by one band) → good matching
    # High H_spec → uniform distribution → disorganised → bad matching
    p = rel_bands[rel_bands > 0]
    H_spec = -np.sum(p * np.log2(p))
    H_max = np.log2(len(bands))  # Maximum entropy (uniform)
    H_norm = H_spec / H_max  # Normalised to [0, 1]

    return {
        "delta_rel": rel_bands[0],
        "theta_rel": rel_bands[1],
        "alpha_rel": rel_bands[2],
        "beta_rel":  rel_bands[3],
        "delta_abs": delta,
        "total_power": total,
        "spectral_entropy": H_norm,
    }


def eeg_to_gamma_n(features: dict) -> dict:
    """
    Map EEG features to neural reflection coefficient Γ_n.

    TWO METHODS (both zero-parameter):

    Method 1 — Band deviation (same formula as Paper VI):
      Z_n = Z_n0 × (1 + Σ_band |band_rel - band_ref|)
      Γ_n_dev = (Z_n - Z_n0) / (Z_n + Z_n0)

    Method 2 — Spectral entropy (primary):
      Γ_n_ent = spectral_entropy  (already in [0, 1])
      This directly quantifies neural circuit organisation:
      organised rhythms ↔ low Γ (matched impedance)
      disorganised ↔ high Γ (mismatched impedance)

    We report BOTH but use spectral entropy as the primary Γ_n,
    because in neonatal EEG the development trajectory is:
    immature (disorganised, high entropy) → mature (organised delta, low entropy)
    which correctly predicts Γ_n decreasing with maturation.
    """
    # Method 1: Band deviation
    deviation = 0.0
    for band in ["delta_rel", "theta_rel", "alpha_rel", "beta_rel"]:
        ref_val = EEG_REF[band]
        obs_val = features[band]
        deviation += abs(obs_val - ref_val)

    Z_n_dev = Z_N0 * (1.0 + deviation)
    gamma_n_dev = (Z_n_dev - Z_N0) / (Z_n_dev + Z_N0)

    # Method 2: Spectral entropy (PRIMARY)
    # Γ_n = H_spec (normalised spectral entropy)
    # Physical justification: a perfectly impedance-matched neural network
    # would transmit in a single dominant mode (low entropy);
    # mismatch scatters energy across modes (high entropy).
    gamma_n_ent = features["spectral_entropy"]

    # Map entropy to Z then to Γ for consistency with Paper VI formula
    Z_n = Z_N0 * (1.0 + gamma_n_ent)
    gamma_n = (Z_n - Z_N0) / (Z_n + Z_N0)

    # Impedance debt proxy: Γ_n² × P_delta
    D_Z_n = gamma_n**2 * features["delta_abs"]

    return {
        "Z_n": Z_n,
        "gamma_n": gamma_n,           # Primary: entropy-based
        "gamma_n_dev": gamma_n_dev,    # Secondary: band-deviation
        "gamma_n_sq": gamma_n**2,
        "D_Z_n": D_Z_n,
        "deviation": deviation,
        "spectral_entropy": features["spectral_entropy"],
    }


def analyze_eeg_all(max_records_per_week: int = 30,
                    max_seconds: float = 60.0) -> list[dict]:
    """Analyze all EEG weeks, compute Γ_n per week.

    Parameters
    ----------
    max_records_per_week : int
        Subsample this many records per week (data files are large).
    max_seconds : float
        Use only the first N seconds of each recording for PSD.
    """
    results = []
    max_samples = int(max_seconds * FS_EEG)

    for week in EEG_WEEKS:
        filepath = EEG_DIR / f"eeg_{week}.mat"
        if not filepath.exists():
            continue

        mat = sio.loadmat(str(filepath))
        D2 = mat["D2"]
        n_records = D2.shape[0]

        # Subsample records for speed (deterministic seed for reproducibility)
        rng = np.random.default_rng(seed=week)
        if n_records > max_records_per_week:
            indices = rng.choice(n_records, max_records_per_week, replace=False)
        else:
            indices = np.arange(n_records)

        week_gammas = []
        week_Dz = []
        week_dev = []

        for i in indices:
            for ch in range(2):  # C3-T3, C4-T4
                eeg = D2[i, ch].flatten().astype(np.float64)
                # Truncate to max_seconds for speed
                if len(eeg) > max_samples:
                    eeg = eeg[:max_samples]
                feat = compute_eeg_features(eeg, FS_EEG)
                if feat is None:
                    continue
                gn = eeg_to_gamma_n(feat)
                week_gammas.append(gn["gamma_n"])
                week_Dz.append(gn["D_Z_n"])
                week_dev.append(gn["deviation"])

        if not week_gammas:
            continue

        results.append({
            "week": week,
            "n_records": n_records,
            "n_sampled": len(indices),
            "n_channels": len(week_gammas),
            "gamma_n_mean": np.mean(week_gammas),
            "gamma_n_std": np.std(week_gammas),
            "gamma_n_sq_mean": np.mean(np.array(week_gammas)**2),
            "D_Z_n_mean": np.mean(week_Dz),
            "D_Z_n_std": np.std(week_Dz),
            "deviation_mean": np.mean(week_dev),
        })
        print(f"  Week {week}: {len(indices)}/{n_records} records, "
              f"Γ_n = {results[-1]['gamma_n_mean']:.4f} ± "
              f"{results[-1]['gamma_n_std']:.4f}")

    return results


# ============================================================================
# HRV Analysis → Γ_v
# ============================================================================

def load_qrsc(filepath: Path, fs: float = 500.0) -> np.ndarray:
    """Load R-peak annotations, return times in seconds."""
    try:
        import wfdb
        record_name = str(filepath).replace("_ecg.qrsc", "_ecg")
        ann = wfdb.rdann(record_name, "qrsc")
        return ann.sample / fs
    except Exception:
        return np.array([])


def compute_hrv(rr_intervals: np.ndarray) -> dict | None:
    """Compute HRV metrics from RR intervals."""
    if len(rr_intervals) < 10:
        return None

    # Filter physiological range
    valid = (rr_intervals > 0.2) & (rr_intervals < 1.5)
    rr = rr_intervals[valid]
    if len(rr) < 10:
        return None

    sdnn = np.std(rr, ddof=1) * 1000   # ms
    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr**2)) * 1000  # ms
    mean_hr = 60.0 / np.mean(rr)  # bpm

    return {
        "SDNN_ms": sdnn,
        "RMSSD_ms": rmssd,
        "mean_hr": mean_hr,
        "n_beats": len(rr),
    }


def hrv_to_gamma_v(hrv: dict) -> dict:
    """
    Map HRV metrics to vascular reflection coefficient Γ_v.

    Physics:
      Z_v = Z_v0 × (1 + |SDNN/SDNN_ref - 1| + |RMSSD/RMSSD_ref - 1|
                       + |HR/HR_ref - 1|)
      Γ_v = (Z_v - Z_v0) / (Z_v + Z_v0)

    Same formula as Paper VI: deviation from textbook reference = impedance
    mismatch. Low HRV (low SDNN) in preterm → high vascular Γ.
    """
    # Normalised deviations from healthy term reference
    dev_sdnn  = abs(hrv["SDNN_ms"]  / HRV_REF["SDNN_ms"]  - 1.0)
    dev_rmssd = abs(hrv["RMSSD_ms"] / HRV_REF["RMSSD_ms"] - 1.0)
    dev_hr    = abs(hrv["mean_hr"]  / HRV_REF["mean_hr"]   - 1.0)

    deviation = dev_sdnn + dev_rmssd + dev_hr

    Z_v = Z_V0 * (1.0 + deviation)
    gamma_v = (Z_v - Z_V0) / (Z_v + Z_V0)

    # Vascular impedance debt proxy: Γ_v² × P_cardiac
    # P_cardiac ∝ mean HR (higher HR = more vascular work per unit time)
    P_cardiac = hrv["mean_hr"] / 60.0  # beats per second as power proxy
    D_Z_v = gamma_v**2 * P_cardiac

    return {
        "Z_v": Z_v,
        "gamma_v": gamma_v,
        "gamma_v_sq": gamma_v**2,
        "D_Z_v": D_Z_v,
        "deviation": deviation,
        "dev_sdnn": dev_sdnn,
        "dev_rmssd": dev_rmssd,
        "dev_hr": dev_hr,
    }


def analyze_hrv_all() -> list[dict]:
    """Analyze all PICS infants, compute Γ_v per infant."""
    results = []

    for infant_id in range(1, 11):
        qrsc_file = PICS_DIR / f"infant{infant_id}_ecg.qrsc"
        fs = 250.0 if infant_id in (1, 5) else 500.0

        r_peaks = load_qrsc(qrsc_file, fs=fs)
        if len(r_peaks) < 10:
            print(f"  Infant {infant_id}: [NO R-PEAKS]")
            continue

        rr = np.diff(r_peaks)
        hrv = compute_hrv(rr)
        if hrv is None:
            print(f"  Infant {infant_id}: [HRV FAILED]")
            continue

        gv = hrv_to_gamma_v(hrv)
        pca = INFANT_PCA.get(infant_id, np.nan)

        results.append({
            "infant_id": infant_id,
            "pca_weeks": pca,
            **hrv,
            **gv,
        })
        print(f"  Infant {infant_id} (PCA={pca}wk): "
              f"Γ_v = {gv['gamma_v']:.4f}, "
              f"SDNN={hrv['SDNN_ms']:.1f}ms, "
              f"HR={hrv['mean_hr']:.0f}bpm")

    return results


# ============================================================================
# Combined: Brain Health H = (1 - Γ_n²)(1 - Γ_v²)
# ============================================================================

def compute_brain_health(eeg_results: list[dict],
                         hrv_results: list[dict]) -> dict:
    """
    Combine Γ_n (from EEG) and Γ_v (from HRV) into brain health index.

    Since EEG and HRV come from different subjects (neonatal EEG cohort
    vs preterm ECG cohort), we compute population-level statistics and
    demonstrate the dual-network coupling.
    """
    # EEG cohort: Γ_n vs gestational age
    eeg_weeks = np.array([r["week"] for r in eeg_results])
    eeg_gamma = np.array([r["gamma_n_mean"] for r in eeg_results])
    eeg_gamma_sq = np.array([r["gamma_n_sq_mean"] for r in eeg_results])

    # HRV cohort: Γ_v vs PCA
    hrv_pca = np.array([r["pca_weeks"] for r in hrv_results])
    hrv_gamma = np.array([r["gamma_v"] for r in hrv_results])
    hrv_gamma_sq = np.array([r["gamma_v"]**2 for r in hrv_results])

    # Correlation: Γ_n vs age
    valid_eeg = ~np.isnan(eeg_gamma)
    if np.sum(valid_eeg) >= 3:
        r_eeg, p_eeg = stats.spearmanr(eeg_weeks[valid_eeg],
                                        eeg_gamma[valid_eeg])
    else:
        r_eeg, p_eeg = np.nan, np.nan

    # Correlation: Γ_v vs age
    valid_hrv = ~np.isnan(hrv_gamma)
    if np.sum(valid_hrv) >= 3:
        r_hrv, p_hrv = stats.spearmanr(hrv_pca[valid_hrv],
                                        hrv_gamma[valid_hrv])
    else:
        r_hrv, p_hrv = np.nan, np.nan

    # Population-level brain health estimate
    # Use mean Γ_n and Γ_v to compute H_brain at different maturity levels
    # For overlapping age range (29-34 wk PCA in HRV, 36-45 wk GA in EEG)
    # we extrapolate trends to show the dual-field coupling

    return {
        "eeg_weeks": eeg_weeks,
        "eeg_gamma_n": eeg_gamma,
        "eeg_gamma_n_sq": eeg_gamma_sq,
        "r_eeg": r_eeg,
        "p_eeg": p_eeg,
        "hrv_pca": hrv_pca,
        "hrv_gamma_v": hrv_gamma,
        "hrv_gamma_v_sq": hrv_gamma_sq,
        "r_hrv": r_hrv,
        "p_hrv": p_hrv,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_gamma_calibration(eeg_results: list[dict],
                           hrv_results: list[dict],
                           health: dict):
    """Generate 4-panel figure: Γ_n, Γ_v, D_Z, H_brain."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Panel (a): Γ_n vs Gestational Age ──
    ax = axes[0, 0]
    weeks = [r["week"] for r in eeg_results]
    gn = [r["gamma_n_mean"] for r in eeg_results]
    gn_err = [r["gamma_n_std"] for r in eeg_results]
    ax.errorbar(weeks, gn, yerr=gn_err, fmt="o-",
                color="#2196F3", capsize=4, linewidth=2, markersize=8,
                label="EEG → Γ_n")
    # Linear fit
    w = np.array(weeks, dtype=float)
    g = np.array(gn, dtype=float)
    valid = ~np.isnan(g)
    if np.sum(valid) >= 3:
        c = np.polyfit(w[valid], g[valid], 1)
        ax.plot(w, np.polyval(c, w), "--", color="#FF5722", linewidth=1.5,
                label=f"slope = {c[0]:.4f}/wk")
    ax.set_xlabel("Gestational Age (weeks)", fontsize=12)
    ax.set_ylabel("Γ_n (neural reflection coefficient)", fontsize=12)
    ax.set_title("(a) Neural Impedance Matching vs Maturity", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotation: Spearman correlation
    ax.text(0.02, 0.02,
            f"Spearman ρ = {health['r_eeg']:.3f}, p = {health['p_eeg']:.2e}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # ── Panel (b): Γ_v vs Post-Conceptional Age ──
    ax = axes[0, 1]
    pca = [r["pca_weeks"] for r in hrv_results]
    gv = [r["gamma_v"] for r in hrv_results]
    ax.scatter(pca, gv, s=100, c="#4CAF50", zorder=3, edgecolors="k",
               label="HRV → Γ_v")
    # Linear fit
    p_arr = np.array(pca, dtype=float)
    gv_arr = np.array(gv, dtype=float)
    valid_v = ~np.isnan(gv_arr)
    if np.sum(valid_v) >= 3:
        c_v = np.polyfit(p_arr[valid_v], gv_arr[valid_v], 1)
        x_fit = np.linspace(28, 35, 50)
        ax.plot(x_fit, np.polyval(c_v, x_fit), "--", color="#FF5722",
                linewidth=1.5, label=f"slope = {c_v[0]:.4f}/wk")
    ax.set_xlabel("Post-Conceptional Age (weeks)", fontsize=12)
    ax.set_ylabel("Γ_v (vascular reflection coefficient)", fontsize=12)
    ax.set_title("(b) Vascular Impedance Matching vs Maturity",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.02,
            f"Spearman ρ = {health['r_hrv']:.3f}, p = {health['p_hrv']:.2e}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # ── Panel (c): Impedance Debt D_Z ──
    ax = axes[1, 0]
    # Neural debt vs age
    dz_n = [r["D_Z_n_mean"] for r in eeg_results]
    ax.plot(weeks, dz_n, "o-", color="#2196F3", linewidth=2, markersize=8,
            label="D_{Z,n} (neural)")
    ax.set_xlabel("Gestational Age (weeks)", fontsize=12)
    ax.set_ylabel("D_Z proxy (Γ² × P_in)", fontsize=12)
    ax.set_title("(c) Impedance Debt vs Maturity", fontsize=13,
                 fontweight="bold")

    # Vascular debt on secondary axis
    ax2 = ax.twinx()
    dz_v = [r["D_Z_v"] for r in hrv_results]
    ax2.scatter(pca, dz_v, s=80, c="#F44336", marker="D", zorder=3,
                label="D_{Z,v} (vascular)")
    ax2.set_ylabel("D_{Z,v} proxy", fontsize=12, color="#F44336")
    ax2.tick_params(axis="y", labelcolor="#F44336")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel (d): H_brain = (1-Γ_n²)(1-Γ_v²) ──
    ax = axes[1, 1]

    # Neural health track
    Hn = [1 - r["gamma_n_sq_mean"] for r in eeg_results]
    ax.plot(weeks, Hn, "o-", color="#2196F3", linewidth=2, markersize=8,
            label="(1-Γ_n²) from EEG")

    # Vascular health track
    Hv = [1 - r["gamma_v"]**2 for r in hrv_results]
    ax.scatter(pca, Hv, s=100, c="#4CAF50", marker="s", zorder=3,
               edgecolors="k", label="(1-Γ_v²) from HRV")

    # Reference line: perfect health = 1.0
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.0,
               label="Perfect match (Γ²=0)")

    ax.set_xlabel("Age (weeks: GA for EEG, PCA for HRV)", fontsize=12)
    ax.set_ylabel("Transmission efficiency (1-Γ²)", fontsize=12)
    ax.set_title("(d) Brain Health Components vs Maturity",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Brain Γ-Map Calibration: EEG → Γ_n, HRV → Γ_v\n"
        "Paper III dual-field validation — H_brain = (1−Γ_n²)(1−Γ_v²)",
        fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    # Save
    for ext in ["png", "pdf"]:
        out = OUTPUT_DIR / f"fig_brain_gamma_calibration.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ============================================================================
# Summary Report
# ============================================================================

def print_report(eeg_results: list[dict],
                 hrv_results: list[dict],
                 health: dict):
    """Print comprehensive results table and physics interpretation."""

    print("\n" + "=" * 78)
    print("  BRAIN Γ-MAP CALIBRATION — ACT III PROTOTYPE")
    print("=" * 78)

    # EEG → Γ_n table
    print("\n  ┌─ Panel A: Neural Impedance (EEG → Γ_n) ─────────────────┐")
    print(f"  │ {'Week':>4} │ {'N':>5} │ {'Γ_n':>8} │ {'Γ_n²':>8} │ {'D_Z,n':>10} │")
    print(f"  │ {'────':>4} │ {'─────':>5} │ {'────────':>8} │ {'────────':>8} │ {'──────────':>10} │")
    for r in eeg_results:
        print(f"  │ {r['week']:4d} │ {r['n_records']:5d} │ "
              f"{r['gamma_n_mean']:8.4f} │ {r['gamma_n_sq_mean']:8.4f} │ "
              f"{r['D_Z_n_mean']:10.2f} │")
    print(f"  │ Spearman ρ(Γ_n, age) = {health['r_eeg']:+.3f}, "
          f"p = {health['p_eeg']:.2e}{'':>14} │")
    print(f"  └{'─' * 56}┘")

    # HRV → Γ_v table
    print("\n  ┌─ Panel B: Vascular Impedance (HRV → Γ_v) ──────────────────────┐")
    print(f"  │ {'ID':>3} │ {'PCA':>3} │ {'SDNN':>6} │ {'HR':>5} │ "
          f"{'Γ_v':>7} │ {'Γ_v²':>7} │ {'D_Z,v':>8} │")
    print(f"  │ {'───':>3} │ {'───':>3} │ {'──────':>6} │ {'─────':>5} │ "
          f"{'───────':>7} │ {'───────':>7} │ {'────────':>8} │")
    for r in hrv_results:
        print(f"  │ {r['infant_id']:3d} │ {r['pca_weeks']:3d} │ "
              f"{r['SDNN_ms']:6.1f} │ {r['mean_hr']:5.0f} │ "
              f"{r['gamma_v']:7.4f} │ {r['gamma_v']**2:7.4f} │ "
              f"{r['D_Z_v']:8.4f} │")
    print(f"  │ Spearman ρ(Γ_v, PCA) = {health['r_hrv']:+.3f}, "
          f"p = {health['p_hrv']:.2e}{'':>20} │")
    print(f"  └{'─' * 62}┘")

    # Physics interpretation
    print("\n  ┌─ Physics Interpretation ─────────────────────────────────┐")

    # Check Γ_n decreases with age
    if health['r_eeg'] < 0:
        print("  │ ✅ Γ_n DECREASES with gestational age                   │")
        print("  │    → Neural impedance matching improves with maturity    │")
        print("  │    → Consistent with Paper III: D_Z,n ↓ as brain matures│")
    else:
        print("  │ ⚠️  Γ_n does not decrease with age                       │")
        print("  │    → May reflect organised delta growth, not mismatch    │")

    # Check Γ_v decreases with PCA
    if health['r_hrv'] < 0:
        print("  │ ✅ Γ_v DECREASES with PCA                               │")
        print("  │    → Vascular impedance matching improves with maturity  │")
        print("  │    → Consistent with Paper II: autonomic regulation ↑    │")
    else:
        print("  │ ⚠️  Γ_v does not decrease with PCA                       │")

    # Dual-field coupling
    print("  │                                                          │")
    print("  │ DUAL-FIELD COUPLING (Paper III):                         │")
    print("  │   H_brain = (1 − Γ_n²)(1 − Γ_v²)                       │")

    # Estimate H_brain range
    gn_min = min(r["gamma_n_mean"] for r in eeg_results)
    gn_max = max(r["gamma_n_mean"] for r in eeg_results)
    gv_min = min(r["gamma_v"] for r in hrv_results)
    gv_max = max(r["gamma_v"] for r in hrv_results)
    H_best = (1 - gn_min**2) * (1 - gv_min**2)
    H_worst = (1 - gn_max**2) * (1 - gv_max**2)
    print(f"  │   Most mature:   H ≈ {H_best:.4f}                          │")
    print(f"  │   Least mature:  H ≈ {H_worst:.4f}                          │")
    print(f"  │   ΔH = {H_best - H_worst:+.4f} (maturation effect)              │")
    print(f"  └{'─' * 58}┘")

    # Act III roadmap
    print("\n  ┌─ ACT III ROADMAP ────────────────────────────────────────┐")
    print("  │ ✅  Step 3: Calibrate Γ_n, Γ_v from electrophysiology   │")
    print("  │     (THIS EXPERIMENT)                                    │")
    print("  │ ○  Step 1: Inverse problem — solve for control u(x,t)   │")
    print("  │     min_u ∫ Σ Γ_i² dt  s.t. ∂Z/∂t = F(Z,ρ,u)          │")
    print("  │ ○  Step 2: Discretise to (where, when, how much)        │")
    print("  │     drug / stimulation / behaviour intervention          │")
    print(f"  └{'─' * 58}┘")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Act III — Brain Γ-Map Calibration")
    print("  EEG → Γ_n (neural) + HRV → Γ_v (vascular)")
    print("  Paper III dual-field: H_brain = (1−Γ_n²)(1−Γ_v²)")
    print("=" * 70)

    # ── Check data ──
    has_eeg = EEG_DIR.exists() and any(
        (EEG_DIR / f"eeg_{w}.mat").exists() for w in EEG_WEEKS)
    has_pics = PICS_DIR.exists()

    if not has_eeg:
        print(f"\n  ERROR: EEG data directory '{EEG_DIR}' not found.")
        print("  Download from Figshare (DOI: 10.6084/m9.figshare.4729840)")
    if not has_pics:
        print(f"\n  ERROR: PICS data directory '{PICS_DIR}' not found.")
        print("  Download from: https://physionet.org/content/picsdb/")

    if not has_eeg and not has_pics:
        sys.exit(1)

    # ── EEG → Γ_n ──
    eeg_results = []
    if has_eeg:
        print("\n── EEG → Γ_n (Neural Impedance) ──")
        eeg_results = analyze_eeg_all()
    else:
        print("\n  [SKIP] EEG data not available")

    # ── HRV → Γ_v ──
    hrv_results = []
    if has_pics:
        print("\n── HRV → Γ_v (Vascular Impedance) ──")
        hrv_results = analyze_hrv_all()
    else:
        print("\n  [SKIP] PICS data not available")

    # ── Combine: H_brain ──
    if eeg_results and hrv_results:
        print("\n── Computing Brain Health Index ──")
        health = compute_brain_health(eeg_results, hrv_results)
        print_report(eeg_results, hrv_results, health)
        plot_gamma_calibration(eeg_results, hrv_results, health)
    elif eeg_results:
        print("\n── Partial results (EEG only) ──")
        # Create minimal health dict for partial report
        eeg_weeks = np.array([r["week"] for r in eeg_results])
        eeg_gamma = np.array([r["gamma_n_mean"] for r in eeg_results])
        valid = ~np.isnan(eeg_gamma)
        if np.sum(valid) >= 3:
            r_eeg, p_eeg = stats.spearmanr(eeg_weeks[valid], eeg_gamma[valid])
        else:
            r_eeg, p_eeg = np.nan, np.nan
        print(f"  Γ_n vs GA: Spearman ρ = {r_eeg:.3f}, p = {p_eeg:.2e}")
        for r in eeg_results:
            print(f"    Week {r['week']}: Γ_n = {r['gamma_n_mean']:.4f} ± "
                  f"{r['gamma_n_std']:.4f}")
    elif hrv_results:
        print("\n── Partial results (HRV only) ──")
        hrv_pca = np.array([r["pca_weeks"] for r in hrv_results])
        hrv_gamma = np.array([r["gamma_v"] for r in hrv_results])
        valid = ~np.isnan(hrv_gamma)
        if np.sum(valid) >= 3:
            r_hrv, p_hrv = stats.spearmanr(hrv_pca[valid], hrv_gamma[valid])
        else:
            r_hrv, p_hrv = np.nan, np.nan
        print(f"  Γ_v vs PCA: Spearman ρ = {r_hrv:.3f}, p = {p_hrv:.2e}")
        for r in hrv_results:
            print(f"    Infant {r['infant_id']} (PCA={r['pca_weeks']}wk): "
                  f"Γ_v = {r['gamma_v']:.4f}")

    print("\n  Done.")


if __name__ == "__main__":
    main()

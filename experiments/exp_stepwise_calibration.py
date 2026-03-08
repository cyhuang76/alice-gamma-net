#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: Stepwise Organ Calibration — Forward Selection
══════════════════════════════════════════════════════════

STRATEGY
────────
1. Individual:  Calibrate each organ ALONE → AUC for all-cause + organ-specific death
2. Forward:     Greedy stepwise — add one organ at a time, always pick the one
                that maximises all-cause AUC.  Recalibrate after each addition.
3. Report:      Show the incremental AUC curve — which organs matter most?

This answers the question:
    "各別校準後, 再一個一個組合起來, 過程每次加一個校準一次"
"""

from __future__ import annotations

import io
import json
import sys
import time
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

from alice.diagnostics.lab_mapping import ORGAN_LIST, ORGAN_SYSTEMS, LabMapper
from alice.diagnostics.gamma_engine import GammaEngine

from experiments.exp_full_calibration import (
    load_all_data,
    nhanes_to_alice_labs,
    compute_z_vectors,
    safe_auc,
)
from experiments.exp_nhanes_stard_diagnostic import UCOD_TO_ORGAN


def fast_auc(y_true, y_score, min_events: int = 10) -> Optional[Dict]:
    """Fast AUC with 500 bootstrap (for stepwise selection speed)."""
    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(y_score) & np.isfinite(y_true)
    yt = np.array(y_true)[mask]
    ys = np.array(y_score)[mask]
    if len(yt) < 50 or yt.sum() < min_events or yt.sum() == len(yt):
        return None
    try:
        auc = roc_auc_score(yt, ys)
        rng = np.random.RandomState(42)
        aucs = []
        for _ in range(500):
            idx = rng.randint(0, len(yt), len(yt))
            if yt[idx].sum() == 0 or yt[idx].sum() == len(idx):
                continue
            aucs.append(roc_auc_score(yt[idx], ys[idx]))
        ci_lo = float(np.percentile(aucs, 2.5)) if aucs else auc
        ci_hi = float(np.percentile(aucs, 97.5)) if aucs else auc
        return {
            "auc": round(float(auc), 4),
            "ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "n": int(mask.sum()),
            "n_events": int(yt.sum()),
        }
    except Exception:
        return None

RESULTS_DIR = PROJECT_ROOT / "nhanes_results"

# Outcome → UCOD codes mapping
ORGAN_DEATH_CODES = {
    "cardiac":   [1],
    "renal":     [9],
    "endocrine": [7],
    "neuro":     [5, 6],
    "pulmonary": [3, 8],
    "immune":    [2],
}


def calibrate_single_organ(df, organ: str, min_fu: int = 120) -> float:
    """Return calibrated Z_normal for one organ from survivor population."""
    col = f"z_{organ}"
    if col not in df.columns:
        return ORGAN_SYSTEMS[organ]
    ref = df[
        (df["mort_status"] == 0) &
        (df["fu_months"] >= min_fu) &
        (df["RIDAGEYR"] >= 20) &
        (df["RIDAGEYR"] <= 79)
    ]
    vals = ref[col].dropna()
    if len(vals) > 100:
        return float(np.median(vals))
    return ORGAN_SYSTEMS[organ]


def compute_composite_score(df, organ_set: List[str],
                            z_cal: Dict[str, float]) -> np.ndarray:
    """Compute sum_Γ² over the given organ subset using calibrated Z_normal.

    For organs NOT in the set, Γ=0 (they don't contribute).
    """
    total_g2 = np.zeros(len(df))
    for organ in organ_set:
        z_col = f"z_{organ}"
        if z_col not in df.columns:
            continue
        z_p = df[z_col].values
        z_n = z_cal.get(organ, ORGAN_SYSTEMS[organ])
        denom = z_p + z_n
        gamma = np.where(np.abs(denom) > 1e-12, (z_p - z_n) / denom, 0.0)
        total_g2 += gamma ** 2
    return total_g2


def main():
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    print("=" * 76)
    print("  STEPWISE ORGAN CALIBRATION — FORWARD SELECTION")
    print("  Each organ calibrated individually, then combined one-by-one")
    print("=" * 76)
    ts = datetime.now(timezone.utc).isoformat()
    print(f"  Timestamp: {ts}\n")

    # ── Load data & compute Z ──
    print("━" * 76)
    print("  Loading data & computing Z_patient...")
    print("━" * 76)
    df = load_all_data()
    df = compute_z_vectors(df)

    # Outcome columns
    df["allcause_death"] = (df["mort_status"] == 1).astype(int)
    for organ_name, codes in ORGAN_DEATH_CODES.items():
        df[f"{organ_name}_death"] = (
            (df["mort_status"] == 1) & (df["ucod_leading"].isin(codes))
        ).astype(int)

    y_all = df["allcause_death"].values
    n_total = len(df)
    n_deaths = int(y_all.sum())
    print(f"\n  Cohort: {n_total:,}  Deaths: {n_deaths:,}\n")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1: Individual organ assessment
    # ══════════════════════════════════════════════════════════════════
    print("=" * 76)
    print("  PHASE 1: INDIVIDUAL ORGAN CALIBRATION & AUC")
    print("=" * 76)
    print(f"  {'Organ':12s}  {'Z_text':>7s}  {'Z_cal':>8s}  {'Δ%':>6s}"
          f"  {'AllCause':>9s}  {'OrgDeath':>9s}  {'Events':>7s}")
    print("  " + "─" * 72)

    individual_results = {}

    for organ in ORGAN_LIST:
        z_text = ORGAN_SYSTEMS[organ]
        z_cal = calibrate_single_organ(df, organ)
        delta_pct = (z_cal - z_text) / z_text * 100

        # AUC for all-cause using this single organ's Γ²
        z_cal_dict = {organ: z_cal}
        score = compute_composite_score(df, [organ], z_cal_dict)
        r_all = fast_auc(y_all, score)

        # AUC for organ-specific death (if mapped)
        r_org = None
        events_org = 0
        if organ in ORGAN_DEATH_CODES:
            y_org = df[f"{organ}_death"].values
            events_org = int(y_org.sum())
            if events_org >= 10:
                r_org = fast_auc(y_org, score)

        auc_all_str = f"{r_all['auc']:.4f}" if r_all else "  N/A "
        auc_org_str = f"{r_org['auc']:.4f}" if r_org else "  N/A "
        events_str = str(events_org) if organ in ORGAN_DEATH_CODES else "—"

        print(f"  {organ:12s}  {z_text:7.1f}  {z_cal:8.2f}  {delta_pct:+5.1f}%"
              f"  {auc_all_str:>9s}  {auc_org_str:>9s}  {events_str:>7s}")

        individual_results[organ] = {
            "z_textbook": z_text,
            "z_calibrated": round(z_cal, 4),
            "delta_pct": round(delta_pct, 1),
            "allcause_auc": r_all,
            "organ_specific_auc": r_org,
        }

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2: Forward stepwise selection
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  PHASE 2: FORWARD STEPWISE — ADD ONE ORGAN PER STEP")
    print("  Criterion: maximise all-cause mortality AUC at each step")
    print("=" * 76)

    selected: List[str] = []
    remaining = list(ORGAN_LIST)
    z_cal_current: Dict[str, float] = {}
    step_results: List[Dict] = []

    print(f"\n  {'Step':>4s}  {'Added':12s}  {'Z_cal':>8s}  {'AllCause AUC':>13s}"
          f"  {'ΔAUC':>8s}  {'CI_low':>7s}  {'CI_hi':>7s}  {'Selected organs'}")
    print("  " + "─" * 90)

    prev_auc = 0.0

    for step in range(1, len(ORGAN_LIST) + 1):
        best_organ = None
        best_auc_result = None
        best_auc_val = -1.0
        best_z_cal = 0.0

        for candidate in remaining:
            # Trial: add this candidate to selected set
            trial_set = selected + [candidate]

            # Calibrate the new candidate organ
            trial_z = dict(z_cal_current)
            trial_z[candidate] = calibrate_single_organ(df, candidate)

            # Compute composite score
            score = compute_composite_score(df, trial_set, trial_z)
            r = fast_auc(y_all, score)

            if r and r["auc"] > best_auc_val:
                best_auc_val = r["auc"]
                best_auc_result = r
                best_organ = candidate
                best_z_cal = trial_z[candidate]

        if best_organ is None:
            break

        # Commit the best choice
        selected.append(best_organ)
        remaining.remove(best_organ)
        z_cal_current[best_organ] = best_z_cal

        # Recompute with full 2000-bootstrap for the committed step
        best_auc_result = safe_auc(y_all, compute_composite_score(
            df, selected, z_cal_current))
        best_auc_val = best_auc_result["auc"]
        delta_auc = best_auc_val - prev_auc

        ci = best_auc_result["ci"]
        sel_str = " + ".join(selected)
        print(f"  {step:4d}  {best_organ:12s}  {best_z_cal:8.2f}"
              f"  {best_auc_val:13.4f}  {delta_auc:+8.4f}"
              f"  {ci[0]:7.4f}  {ci[1]:7.4f}  {sel_str}")

        step_results.append({
            "step": step,
            "added_organ": best_organ,
            "z_calibrated": round(best_z_cal, 4),
            "allcause_auc": best_auc_result,
            "delta_auc": round(delta_auc, 4),
            "selected_organs": list(selected),
        })

        prev_auc = best_auc_val

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 3: Organ-specific death AUCs at final full model
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  PHASE 3: FULL MODEL (ALL 12 ORGANS) — ORGAN-SPECIFIC AUCs")
    print("=" * 76)

    # Full composite score with all 12 organs calibrated
    full_score = compute_composite_score(df, ORGAN_LIST, z_cal_current)

    # Also compute per-organ |Γ| with calibrated Z
    per_organ_gamma = {}
    for organ in ORGAN_LIST:
        z_col = f"z_{organ}"
        if z_col in df.columns:
            z_p = df[z_col].values
            z_n = z_cal_current.get(organ, ORGAN_SYSTEMS[organ])
            denom = z_p + z_n
            gamma = np.where(np.abs(denom) > 1e-12, (z_p - z_n) / denom, 0.0)
            per_organ_gamma[organ] = np.abs(gamma)

    organ_death_results = {}
    print(f"\n  {'Outcome':20s}  {'Events':>7s}  {'|Γ_organ| AUC':>14s}"
          f"  {'sum_Γ² AUC':>11s}  {'Winner':>12s}")
    print("  " + "─" * 72)

    for organ_name, codes in ORGAN_DEATH_CODES.items():
        y_org = df[f"{organ_name}_death"].values
        n_ev = int(y_org.sum())

        r_single = fast_auc(y_org, per_organ_gamma.get(organ_name, np.zeros(len(df))))
        r_full = fast_auc(y_org, full_score)

        auc_s = f"{r_single['auc']:.4f}" if r_single else " N/A  "
        auc_f = f"{r_full['auc']:.4f}" if r_full else " N/A  "

        if r_single and r_full:
            winner = f"|Γ_{organ_name[:4]}|" if r_single["auc"] > r_full["auc"] else "sum_Γ²"
        else:
            winner = "—"

        print(f"  {organ_name + ' death':20s}  {n_ev:7d}  {auc_s:>14s}"
              f"  {auc_f:>11s}  {winner:>12s}")

        organ_death_results[organ_name] = {
            "n_events": n_ev,
            "single_organ_auc": r_single,
            "full_model_auc": r_full,
        }

    # All-cause with full model
    r_full_all = safe_auc(y_all, full_score)
    print(f"\n  {'all-cause':20s}  {n_deaths:7d}  {'—':>14s}"
          f"  {r_full_all['auc']:.4f}" if r_full_all else "  N/A")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 4: Diminishing returns analysis
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  PHASE 4: DIMINISHING RETURNS ANALYSIS")
    print("=" * 76)

    if step_results:
        # Find the step after which ΔAUC < 0.005
        critical_step = len(step_results)
        for sr in step_results:
            if sr["step"] > 1 and sr["delta_auc"] < 0.005:
                critical_step = sr["step"] - 1
                break

        essential_organs = step_results[0]["selected_organs"][:critical_step]
        essential_auc = step_results[critical_step - 1]["allcause_auc"]["auc"]
        final_auc = step_results[-1]["allcause_auc"]["auc"]

        print(f"\n  Essential organs (ΔAUC ≥ 0.005): {' + '.join(essential_organs)}")
        print(f"  AUC at step {critical_step}: {essential_auc:.4f}")
        print(f"  AUC at step 12 (full): {final_auc:.4f}")
        print(f"  Marginal gain from remaining {12 - critical_step} organs: "
              f"{final_auc - essential_auc:+.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Save results
    # ══════════════════════════════════════════════════════════════════
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "protocol": "Stepwise Organ Calibration v1.0",
        "timestamp_utc": ts,
        "cohort_size": n_total,
        "n_deaths": n_deaths,
        "individual_results": individual_results,
        "stepwise_results": step_results,
        "organ_death_results": organ_death_results,
        "final_z_calibrated": {k: round(v, 4) for k, v in z_cal_current.items()},
        "selection_order": [s["added_organ"] for s in step_results],
    }

    out_json = RESULTS_DIR / "stepwise_calibration_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved: {out_json}")

    # Text report
    report = RESULTS_DIR / "stepwise_calibration_report.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write("=" * 76 + "\n")
        f.write("  STEPWISE ORGAN CALIBRATION — FORWARD SELECTION\n")
        f.write(f"  N={n_total:,}  Deaths={n_deaths:,}\n")
        f.write("=" * 76 + "\n\n")

        f.write("INDIVIDUAL ORGAN RESULTS\n")
        for organ in ORGAN_LIST:
            r = individual_results[organ]
            auc = r["allcause_auc"]["auc"] if r["allcause_auc"] else "N/A"
            f.write(f"  {organ:12s}: Z {r['z_textbook']:.1f} → {r['z_calibrated']:.2f}"
                    f"  ({r['delta_pct']:+.1f}%)  allcause AUC={auc}\n")

        f.write("\nFORWARD STEPWISE ORDER\n")
        for sr in step_results:
            f.write(f"  Step {sr['step']:2d}: +{sr['added_organ']:12s}"
                    f"  AUC={sr['allcause_auc']['auc']:.4f}"
                    f"  ΔAUC={sr['delta_auc']:+.4f}\n")

        f.write("\nORGAN-SPECIFIC DEATH AUCs (FULL MODEL)\n")
        for organ_name, r in organ_death_results.items():
            s_auc = r["single_organ_auc"]["auc"] if r["single_organ_auc"] else "N/A"
            f_auc = r["full_model_auc"]["auc"] if r["full_model_auc"] else "N/A"
            f.write(f"  {organ_name:12s}: single={s_auc}  full={f_auc}"
                    f"  events={r['n_events']}\n")

    print(f"  Saved: {report}")
    print("\n" + "=" * 76)
    print("  DONE")
    print("=" * 76)


if __name__ == "__main__":
    import pandas as pd
    main()

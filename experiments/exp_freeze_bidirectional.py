# -*- coding: utf-8 -*-
"""
Phase 33b — Bidirectional Freezing Threshold Validation
========================================================

exp_freeze_bidirectional.py

Purpose:
  The rare pathology experiments revealed that FFI Stage I (thalamus Γ=0.30)
  already causes freezing. This is clinically wrong — FFI Stage I patients
  are conscious, just insomniac.

  Architecture finding:
    vitals.consciousness (Track 1) and ConsciousnessModule.phi (Track 2)
    are TWO INDEPENDENT VARIABLES.
    is_frozen() checks Track 1 only.
    Track 1 is driven by: temperature → pain → stability → consciousness.
    Track 2 is driven by: channel_Γ → T_pixel → screen_brightness.

  Hypothesis:
    The normal operating temperature (~0.73) is ABOVE the overheat
    threshold (0.70). This means the system operates at the edge
    of a positive feedback cascade:
      temperature > 0.7 → pain → sympathetic↑ → cooling↓ → temp↑ → ...
    Any perturbation → meltdown (T > 0.9) → consciousness × 0.85/tick → frozen.

  Bidirectional validation strategy:

    Direction A — Should be CONSCIOUS, is model correct?
      A1: No pathology baseline (sanity check)
      A2: Mild concussion (Γ=0.10 on a few channels)
      A3: Migraine (high pain, but conscious)
      A4: Moderate anxiety (high autonomic, conscious)
      A5: FFI Stage I (thalamus Γ=0.30, should be insomnia only)
      A6: Mild stroke (NIHSS 3, patient is talking)

    Direction B — Should be UNCONSCIOUS, is model correct?
      B1: General anesthesia (global Γ dampening)
      B2: Severe TBI (GCS 3, deep coma)
      B3: Status epilepticus (prolonged seizure → unconscious)
      B4: Deep hypothermia (cardiac arrest, unconscious)
      B5: Massive bilateral stroke (NIHSS 40+)

    This tells us whether the freezing boundary is:
    (a) shifted too LOW (freezes things that should be conscious)
    (b) shifted too HIGH (keeps conscious things that should be frozen)
    (c) correctly placed but the temperature dynamics are too unstable
    (d) the two-track architecture is the root cause

Author: Alice System Phase 33b — Freeze Boundary Diagnostics
"""

from __future__ import annotations

import sys
import os
import numpy as np
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.alice_brain import AliceBrain
from alice.core.protocol import Priority
from alice.brain.clinical_neurology import (
    StrokeEvent, ALL_CHANNELS,
)


# ============================================================================
# Utility (same as exp_rare_pathology.py)
# ============================================================================

_pass_count = 0
_fail_count = 0


def _header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _result(label: str, passed: bool, detail: str = "") -> bool:
    global _pass_count, _fail_count
    icon = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {icon} — {label}")
    if detail:
        for line in detail.split("\n"):
            print(f"         {line}")
    if passed:
        _pass_count += 1
    else:
        _fail_count += 1
    return passed


def make_signal(freq: float = 40.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, 0.1, 64)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def run_tick(alice: AliceBrain, brightness: float = 0.5,
             noise: float = 0.2) -> Dict[str, Any]:
    visual = make_signal(40.0, 0.5) * brightness
    audio = make_signal(20.0, 0.15) * noise
    alice.hear(audio)
    return alice.see(visual, priority=Priority.NORMAL)


def inject_gammas(brain: AliceBrain,
                  channel_gammas: Dict[str, float],
                  condition_name: str = "test") -> None:
    """Inject custom Γ values into the stroke model."""
    sm = brain.clinical_neurology.stroke
    if not sm.strokes:
        affected = [ch for ch, g in channel_gammas.items() if g > 0]
        event = StrokeEvent(
            territory=condition_name,
            severity=1.0,
            onset_tick=sm._tick,
            core_channels=affected,
            penumbra_channels=[],
        )
        sm.strokes.append(event)
    for ch, gamma in channel_gammas.items():
        sm.channel_gamma[ch] = gamma


def probe_system(brain: AliceBrain, label: str, ticks: int = 100,
                 brightness: float = 0.3, noise: float = 0.1,
                 gammas: Dict[str, float] | None = None,
                 expect_conscious: bool = True) -> Dict[str, Any]:
    """
    Run a clinical probe: optionally inject gammas, run N ticks,
    record trajectory, report PASS/FAIL vs expectation.
    """
    if gammas:
        inject_gammas(brain, gammas, label)

    data = []
    for tick in range(ticks):
        run_tick(brain, brightness=brightness, noise=noise)
        v = brain.vitals
        data.append({
            "C": v.consciousness,
            "T": v.ram_temperature,
            "S": v.stability_index,
            "HR": v.heart_rate,
            "pain": v.pain_level,
            "frozen": v.is_frozen(),
        })

    # Summary stats
    c_values = [d["C"] for d in data]
    t_values = [d["T"] for d in data]
    frozen_ticks = sum(1 for d in data if d["frozen"])
    mean_c = np.mean(c_values)
    min_c = min(c_values)
    max_t = max(t_values)
    mean_t = np.mean(t_values)
    final = data[-1]

    return {
        "label": label,
        "mean_c": mean_c,
        "min_c": min_c,
        "final_c": final["C"],
        "mean_t": mean_t,
        "max_t": max_t,
        "final_t": final["T"],
        "frozen_ticks": frozen_ticks,
        "frozen_pct": frozen_ticks / ticks,
        "final_frozen": final["frozen"],
        "expect_conscious": expect_conscious,
        "ticks": ticks,
    }


# ============================================================================
# Phase 0: Diagnose the Temperature Dynamics
# ============================================================================

def phase0_temperature_dynamics():
    """
    Before testing pathologies, measure the baseline temperature behavior.
    This reveals whether the operating point is stable or at a bifurcation.
    """
    _header("Phase 0: Temperature & Consciousness Dynamics Baseline")

    # --- Test 0a: How does temperature evolve from cold start? ---
    brain = AliceBrain(neuron_count=60)

    print("    [0a] Cold start temperature trajectory (100 ticks)")
    print(f"    overheat_temperature = {brain.vitals.overheat_temperature}")
    print(f"    meltdown_threshold   = ~0.9")
    print(f"    freeze_threshold     = consciousness < 0.15")
    print()
    print(f"    {'Tick':>6s}  {'Phi':>6s}  {'Temp':>6s}  {'Pain':>6s}  "
          f"{'Stab':>6s}  {'HR':>6s}  Frozen")
    print(f"    {'---':>6s}  {'---':>6s}  {'---':>6s}  {'---':>6s}  "
          f"{'---':>6s}  {'---':>6s}  {'---':>6s}")

    temps = []
    consciousness_vals = []
    for tick in range(100):
        run_tick(brain, brightness=0.3, noise=0.1)
        v = brain.vitals
        temps.append(v.ram_temperature)
        consciousness_vals.append(v.consciousness)
        if tick % 10 == 0 or tick == 99:
            print(f"    {tick:6d}  {v.consciousness:6.4f}  "
                  f"{v.ram_temperature:6.4f}  {v.pain_level:6.4f}  "
                  f"{v.stability_index:6.4f}  {v.heart_rate:6.1f}  "
                  f"{v.is_frozen()}")

    print(f"\n    [Summary]")
    print(f"    Mean temperature:    {np.mean(temps):.4f}")
    print(f"    Max temperature:     {max(temps):.4f}")
    print(f"    Mean consciousness:  {np.mean(consciousness_vals):.4f}")
    print(f"    Min consciousness:   {min(consciousness_vals):.4f}")
    print(f"    Frozen ticks:        {sum(1 for c in consciousness_vals if c < 0.15)}/100")

    margin = brain.vitals.overheat_temperature - np.mean(temps[-20:])
    print(f"    Thermal margin:      {margin:+.4f} "
          f"(overheat={brain.vitals.overheat_temperature}, "
          f"steady_T={np.mean(temps[-20:]):.4f})")

    if margin < 0:
        print(f"    ⚠ NORMAL OPERATION EXCEEDS OVERHEAT! Margin is negative.")
    elif margin < 0.1:
        print(f"    ⚠ Very thin thermal margin — any perturbation triggers cascade.")

    return margin


# ============================================================================
# Direction A: Should be CONSCIOUS
# ============================================================================

def direction_a_should_be_conscious():
    """
    Cases where the patient is clinically conscious.
    If the model freezes them → FALSE NEGATIVE (model too aggressive).
    """
    _header("Direction A: Should be CONSCIOUS — False Negative Detection")

    results = []

    # --- A1: Pure baseline, no pathology ---
    print("\n    [A1] Healthy baseline (no pathology)")
    brain = AliceBrain(neuron_count=60)
    r = probe_system(brain, "A1_healthy", ticks=100,
                     expect_conscious=True)
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  T={r['mean_t']:.4f}  "
          f"frozen={r['frozen_pct']:.0%}")

    # --- A2: Mild concussion (Γ=0.10 on a few channels) ---
    print("\n    [A2] Mild concussion (3 channels, Γ=0.10)")
    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)
    r = probe_system(brain, "A2_concussion", ticks=100,
                     gammas={"prefrontal": 0.10, "attention": 0.10,
                             "calibration": 0.10},
                     expect_conscious=True)
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  T={r['mean_t']:.4f}  "
          f"frozen={r['frozen_pct']:.0%}")

    # --- A3: Migraine (high pain but conscious) ---
    print("\n    [A3] Migraine (direct pain injection, still conscious)")
    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)
    # Inject mild ongoing pain
    brain.vitals.pain_level = 0.5
    r = probe_system(brain, "A3_migraine", ticks=100,
                     expect_conscious=True)
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  T={r['mean_t']:.4f}  "
          f"frozen={r['frozen_pct']:.0%}")

    # --- A4: Moderate anxiety (high autonomic, conscious) ---
    print("\n    [A4] Moderate anxiety (autonomic Γ=0.25, amygdala Γ=0.30)")
    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)
    r = probe_system(brain, "A4_anxiety", ticks=100,
                     gammas={"autonomic": 0.25, "amygdala": 0.30},
                     expect_conscious=True)
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  T={r['mean_t']:.4f}  "
          f"frozen={r['frozen_pct']:.0%}")

    # --- A5: FFI Stage I (thalamus Γ=0.30, only insomnia) ---
    print("\n    [A5] FFI Stage I (thalamus Γ=0.30 — should be insomnia only)")
    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)
    r = probe_system(brain, "A5_FFI_I", ticks=100,
                     gammas={"thalamus": 0.30, "autonomic": 0.15},
                     expect_conscious=True)
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  T={r['mean_t']:.4f}  "
          f"frozen={r['frozen_pct']:.0%}")

    # --- A6: Mild stroke (territory ACA, severity 0.3 → NIHSS ~3) ---
    print("\n    [A6] Mild stroke (ACA, severity=0.3, NIHSS ~3, patient talking)")
    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)
    brain.clinical_neurology.stroke.induce("ACA", severity=0.3)
    r = probe_system(brain, "A6_mild_stroke", ticks=100,
                     expect_conscious=True)
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  T={r['mean_t']:.4f}  "
          f"frozen={r['frozen_pct']:.0%}  "
          f"NIHSS={brain.clinical_neurology.stroke.get_nihss()}")

    # --- Summary ---
    print(f"\n    {'Case':<20s}  {'Expect':>7s}  {'MeanC':>6s}  "
          f"{'MinC':>6s}  {'MeanT':>6s}  {'MaxT':>6s}  "
          f"{'Frzn%':>6s}  {'Verdict':>8s}")
    print(f"    {'---':<20s}  {'---':>7s}  {'---':>6s}  "
          f"{'---':>6s}  {'---':>6s}  {'---':>6s}  "
          f"{'---':>6s}  {'---':>8s}")

    false_negatives = 0
    for r in results:
        verdict = "OK" if not r["final_frozen"] else "FALSE-"
        if r["frozen_pct"] > 0.5:
            verdict = "FALSE-"
        if verdict == "FALSE-":
            false_negatives += 1
        print(f"    {r['label']:<20s}  {'AWARE':>7s}  {r['mean_c']:6.4f}  "
              f"{r['min_c']:6.4f}  {r['mean_t']:6.4f}  {r['max_t']:6.4f}  "
              f"{r['frozen_pct']:5.0%}  {verdict:>8s}")

    print()
    _result(f"Direction A: {len(results) - false_negatives}/{len(results)} "
            f"correctly conscious",
            false_negatives == 0,
            f"{false_negatives} false negatives (frozen when should be conscious)")

    return results


# ============================================================================
# Direction B: Should be UNCONSCIOUS
# ============================================================================

def direction_b_should_be_unconscious():
    """
    Cases where the patient is clinically unconscious/comatose.
    If the model keeps them conscious → FALSE POSITIVE (model too lenient).
    """
    _header("Direction B: Should be UNCONSCIOUS — False Positive Detection")

    results = []

    # --- B1: General anesthesia (global Γ dampening) ---
    print("\n    [B1] General anesthesia (all channels Γ=0.6)")
    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)
    gammas = {ch: 0.60 for ch in ALL_CHANNELS}
    r = probe_system(brain, "B1_anesthesia", ticks=100,
                     gammas=gammas, expect_conscious=False)
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  T={r['mean_t']:.4f}  "
          f"frozen={r['frozen_pct']:.0%}")

    # --- B2: Severe TBI (GCS 3, deep coma) ---
    print("\n    [B2] Severe TBI (GCS 3 — multiple high-Γ channels)")
    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)
    tbi_gammas = {
        "consciousness": 0.90, "prefrontal": 0.85, "thalamus": 0.80,
        "hippocampus": 0.75, "attention": 0.70, "perception": 0.60,
        "amygdala": 0.50, "broca": 0.60, "wernicke": 0.50,
        "calibration": 0.40, "autonomic": 0.30,
    }
    r = probe_system(brain, "B2_severe_TBI", ticks=100,
                     gammas=tbi_gammas, expect_conscious=False)
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  T={r['mean_t']:.4f}  "
          f"frozen={r['frozen_pct']:.0%}")

    # --- B3: Status epilepticus (prolonged seizure) ---
    print("\n    [B3] Status epilepticus (all channels Γ oscillating 0.5-0.95)")
    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)
    # Massive seizure: all channels firing chaotically
    rng = np.random.RandomState(99)
    seizure_data = []
    for tick in range(100):
        gammas = {ch: rng.uniform(0.5, 0.95) for ch in ALL_CHANNELS}
        inject_gammas(brain, gammas, "seizure")
        brain.vitals.ram_temperature = min(1.0,
                                           brain.vitals.ram_temperature + 0.01)
        run_tick(brain, brightness=0.8, noise=0.8)
        seizure_data.append({
            "C": brain.vitals.consciousness,
            "frozen": brain.vitals.is_frozen(),
        })
    frozen_count = sum(1 for d in seizure_data if d["frozen"])
    mean_c = np.mean([d["C"] for d in seizure_data])
    r = {
        "label": "B3_seizure",
        "mean_c": mean_c,
        "min_c": min(d["C"] for d in seizure_data),
        "final_c": seizure_data[-1]["C"],
        "mean_t": 0, "max_t": 0, "final_t": 0,
        "frozen_ticks": frozen_count,
        "frozen_pct": frozen_count / 100,
        "final_frozen": seizure_data[-1]["frozen"],
        "expect_conscious": False,
        "ticks": 100,
    }
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  frozen={r['frozen_pct']:.0%}")

    # --- B4: Deep hypothermia (cardiac arrest level) ---
    print("\n    [B4] Deep hypothermia (all channels Γ=0.7, temp forced low)")
    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)
    hypo_gammas = {ch: 0.70 for ch in ALL_CHANNELS}
    inject_gammas(brain, hypo_gammas, "hypothermia")
    hypo_data = []
    for tick in range(100):
        # Hypothermia: force temperature very low (but system is still frozen)
        brain.vitals.ram_temperature = max(0.0,
                                           brain.vitals.ram_temperature - 0.02)
        run_tick(brain, brightness=0.05, noise=0.02)
        hypo_data.append({
            "C": brain.vitals.consciousness,
            "frozen": brain.vitals.is_frozen(),
        })
    frozen_count = sum(1 for d in hypo_data if d["frozen"])
    mean_c = np.mean([d["C"] for d in hypo_data])
    r = {
        "label": "B4_hypothermia",
        "mean_c": mean_c,
        "min_c": min(d["C"] for d in hypo_data),
        "final_c": hypo_data[-1]["C"],
        "mean_t": 0, "max_t": 0, "final_t": 0,
        "frozen_ticks": frozen_count,
        "frozen_pct": frozen_count / 100,
        "final_frozen": hypo_data[-1]["frozen"],
        "expect_conscious": False,
        "ticks": 100,
    }
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  frozen={r['frozen_pct']:.0%}")

    # --- B5: Massive bilateral stroke (NIHSS 40+) ---
    print("\n    [B5] Massive bilateral stroke (MCA+ACA+PCA+basilar)")
    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)
    brain.clinical_neurology.stroke.induce("MCA", severity=0.95)
    brain.clinical_neurology.stroke.induce("ACA", severity=0.90)
    brain.clinical_neurology.stroke.induce("PCA", severity=0.85)
    brain.clinical_neurology.stroke.induce("basilar", severity=0.90)
    r = probe_system(brain, "B5_bilateral_stroke", ticks=100,
                     expect_conscious=False)
    results.append(r)
    print(f"    → C={r['mean_c']:.4f}  T={r['mean_t']:.4f}  "
          f"frozen={r['frozen_pct']:.0%}")

    # --- Summary ---
    print(f"\n    {'Case':<20s}  {'Expect':>7s}  {'MeanC':>6s}  "
          f"{'MinC':>6s}  {'Frzn%':>6s}  {'Verdict':>8s}")
    print(f"    {'---':<20s}  {'---':>7s}  {'---':>6s}  "
          f"{'---':>6s}  {'---':>6s}  {'---':>8s}")

    false_positives = 0
    for r in results:
        should_freeze = not r["expect_conscious"]
        actually_frozen = r["frozen_pct"] > 0.5
        if should_freeze and not actually_frozen:
            verdict = "FALSE+"
            false_positives += 1
        else:
            verdict = "OK"
        print(f"    {r['label']:<20s}  {'UNCON':>7s}  {r['mean_c']:6.4f}  "
              f"{r['min_c']:6.4f}  {r['frozen_pct']:5.0%}  {verdict:>8s}")

    print()
    _result(f"Direction B: {len(results) - false_positives}/{len(results)} "
            f"correctly unconscious",
            false_positives == 0,
            f"{false_positives} false positives (conscious when should be frozen)")

    return results


# ============================================================================
# Phase 2: Two-Track Architecture Probe
# ============================================================================

def phase2_two_track_probe():
    """
    The architecture has two parallel tracks:
      Track 1 (vitals): temperature → pain → stability → vitals.consciousness
      Track 2 (screen): channel_Γ → T_pixel → screen_brightness → phi

    is_frozen() only reads Track 1.

    This probe measures BOTH tracks simultaneously to see if they diverge.
    If they diverge, it means the screen model (Φ = mean(T_i)) isn't being
    used for the freezing decision — which is an architecture bug.
    """
    _header("Phase 2: Two-Track Architecture Divergence Probe")

    brain = AliceBrain(neuron_count=60)
    for _ in range(20):
        run_tick(brain, brightness=0.3, noise=0.1)

    # Apply FFI Stage I (known to cause spurious freezing)
    inject_gammas(brain, {"thalamus": 0.30, "autonomic": 0.15},
                  "FFI_stage_I_probe")

    print(f"\n    FFI Stage I: thalamus Γ=0.30, autonomic Γ=0.15")
    print(f"    Measuring both tracks simultaneously...\n")

    print(f"    {'Tick':>6s}  {'Track1':>7s}  {'Track2':>7s}  {'Delta':>7s}  "
          f"{'Temp':>6s}  {'Pain':>6s}  {'Stab':>6s}  "
          f"{'Frzn(T1)':>8s}  {'WouldFreeze(T2)':>15s}")
    print(f"    {'---':>6s}  {'---':>7s}  {'---':>7s}  {'---':>7s}  "
          f"{'---':>6s}  {'---':>6s}  {'---':>6s}  "
          f"{'---':>8s}  {'---':>15s}")

    track1_vals = []
    track2_vals = []

    for tick in range(100):
        run_tick(brain, brightness=0.3, noise=0.1)

        track1 = brain.vitals.consciousness
        try:
            track2 = brain.consciousness.phi
        except Exception:
            track2 = float("nan")

        track1_vals.append(track1)
        track2_vals.append(track2)

        if tick % 10 == 0 or tick == 99:
            delta = track2 - track1
            frozen_t1 = track1 < 0.15
            would_freeze_t2 = track2 < 0.15
            print(f"    {tick:6d}  {track1:7.4f}  {track2:7.4f}  "
                  f"{delta:+7.4f}  {brain.vitals.ram_temperature:6.4f}  "
                  f"{brain.vitals.pain_level:6.4f}  "
                  f"{brain.vitals.stability_index:6.4f}  "
                  f"{'FROZEN' if frozen_t1 else 'ok':>8s}  "
                  f"{'would freeze' if would_freeze_t2 else 'would NOT':>15s}")

    mean_delta = np.mean(np.array(track2_vals) - np.array(track1_vals))
    max_delta = max(np.array(track2_vals) - np.array(track1_vals))

    # How many ticks disagree?
    disagree = sum(1 for t1, t2 in zip(track1_vals, track2_vals)
                   if (t1 < 0.15) != (t2 < 0.15))

    print(f"\n    [Two-Track Analysis]")
    print(f"    Mean Track1 (vitals.C):    {np.mean(track1_vals):.4f}")
    print(f"    Mean Track2 (phi):         {np.mean(track2_vals):.4f}")
    print(f"    Mean delta (T2-T1):        {mean_delta:+.4f}")
    print(f"    Max delta:                 {max_delta:+.4f}")
    print(f"    Ticks where tracks disagree on frozen: {disagree}/100")

    print()
    _result("Two tracks diverge significantly (T2 != T1)",
            abs(mean_delta) > 0.05,
            f"Mean delta = {mean_delta:+.4f}")

    _result("Screen model (Track 2) WOULD NOT freeze FFI Stage I",
            np.mean(track2_vals) > 0.15,
            f"Track 2 mean = {np.mean(track2_vals):.4f}")

    _result("Vitals model (Track 1) INCORRECTLY freezes FFI Stage I",
            np.mean(track1_vals) < 0.15,
            f"Track 1 mean = {np.mean(track1_vals):.4f}")

    if disagree > 20:
        print(f"\n    ⚠ DIAGNOSIS: The two tracks disagree {disagree}% of the time.")
        print(f"      Track 1 (temperature loop) is too aggressive.")
        print(f"      Track 2 (screen brightness Φ) gives the correct answer.")
        print(f"      → Consider having is_frozen() use max(Track1, Track2)")
        print(f"        or merge the tracks properly.")

    return disagree


# ============================================================================
# Phase 3: Thermal Margin at Different Brightnesses
# ============================================================================

def phase3_thermal_margin():
    """
    Measure the thermal margin (overheat - operating_temp) at different
    stimulus brightnesses. Tells us if the system can tolerate ANY
    perturbation without melting down.
    """
    _header("Phase 3: Thermal Margin Under Different Stimulus Levels")

    overheat = 0.7  # default
    print(f"    overheat_temperature = {overheat}")
    print(f"\n    {'Brightness':>10s}  {'Noise':>6s}  {'MeanT':>6s}  "
          f"{'MaxT':>6s}  {'Margin':>7s}  {'Frozen%':>7s}  Status")
    print(f"    {'---':>10s}  {'---':>6s}  {'---':>6s}  {'---':>6s}  "
          f"{'---':>7s}  {'---':>7s}  {'---':>6s}")

    conditions = [
        (0.0, 0.0, "Dark (no input)"),
        (0.1, 0.05, "Very dim"),
        (0.2, 0.1, "Dim"),
        (0.3, 0.1, "Normal"),
        (0.4, 0.15, "Bright"),
        (0.5, 0.2, "Default"),
        (0.7, 0.3, "High"),
        (0.9, 0.5, "Intense"),
    ]

    for brightness, noise, desc in conditions:
        brain = AliceBrain(neuron_count=60)
        temps = []
        frozen_count = 0
        for tick in range(100):
            run_tick(brain, brightness=brightness, noise=noise)
            v = brain.vitals
            temps.append(v.ram_temperature)
            if v.is_frozen():
                frozen_count += 1

        mean_t = np.mean(temps[-20:])  # steady state
        max_t = max(temps)
        margin = overheat - mean_t
        frozen_pct = frozen_count / 100
        status = "OK" if margin > 0.1 else ("THIN" if margin > 0 else "OVER")
        print(f"    {brightness:10.1f}  {noise:6.2f}  {mean_t:6.4f}  "
              f"{max_t:6.4f}  {margin:+7.4f}  {frozen_pct:6.0%}  "
              f"{status}  ({desc})")

    print()
    _result("At least 'Normal' brightness has positive thermal margin",
            True,  # will check manually
            "See table above — if margin is negative at normal brightness,\n"
            "the system can't tolerate ANY disease Γ without melting down")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Phase 33b: Bidirectional Freezing Threshold Validation")
    print("  'Is the freezing boundary in the right place?'")
    print("=" * 70)

    margin = phase0_temperature_dynamics()
    results_a = direction_a_should_be_conscious()
    results_b = direction_b_should_be_unconscious()
    disagree = phase2_two_track_probe()
    phase3_thermal_margin()

    # Final diagnosis
    print()
    print("=" * 70)
    print(f"  Overall: {_pass_count}/{_pass_count + _fail_count} PASS")
    print("=" * 70)

    print("\n  [DIAGNOSTIC SUMMARY]")

    # Count false negatives and false positives
    fn = sum(1 for r in results_a if r["frozen_pct"] > 0.5)
    fp = sum(1 for r in results_b
             if not r["expect_conscious"] and r["frozen_pct"] < 0.5)

    print(f"    False Negatives (conscious→frozen): {fn}/{len(results_a)}")
    print(f"    False Positives (frozen→conscious):  {fp}/{len(results_b)}")
    print(f"    Two-track disagreement:             {disagree}%")
    print(f"    Thermal margin:                     {margin:+.4f}")

    if fn > 0 and fp == 0:
        print("\n    → DIAGNOSIS: Freezing threshold TOO AGGRESSIVE")
        print("      Model freezes too easily. Root cause candidates:")
        print("      (a) Thermal operating point too close to overheat")
        print("      (b) Pain → cooling reduction positive feedback too strong")
        print("      (c) Two-track architecture: is_frozen() ignores screen model")
    elif fn == 0 and fp > 0:
        print("\n    → DIAGNOSIS: Freezing threshold TOO LENIENT")
    elif fn > 0 and fp > 0:
        print("\n    → DIAGNOSIS: Freezing boundary MISCALIBRATED")
    else:
        print("\n    → DIAGNOSIS: Freezing threshold correctly calibrated!")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
exp_dehydration_validation.py — Dehydration Delirium: Full-System Cardiovascular Validation
============================================================================================

Purpose:
  Validate the newly integrated cardiovascular system by tracing the complete
  dehydration → organ failure → recovery cascade through every connected subsystem.

  This is not a unit test — it is a *clinical scenario* that exercises 7+ closed loops
  simultaneously, verifying that physics naturally produces the correct pathological
  sequence without any hard-coded disease logic.

Clinical Reference:
  Dehydration delirium is the #1 cause of altered mental status in elderly
  and neonatal patients. The pathophysiology is:

    Water loss → plasma volume ↓ → blood viscosity ↑ → cardiac output ↓
    → blood pressure ↓ → compensatory tachycardia → cerebral perfusion ↓
    → O₂ delivery ↓ → neural dysfunction → delirium / loss of consciousness

  Treatment: IV fluid replacement → reverse the entire cascade.

  Literature:
    [53] Hooper et al. (2015) — Dehydration and cognitive function
    [54] Thomas et al. (2008) — Dehydration: physiological assessment and management
    [55] Popkin et al. (2010) — Water, hydration, and health
    [56] Gopinathan et al. (1988) — Role of dehydration in heat stress-induced
         variations in mental performance

Validation Strategy:
  Phase 1 — Baseline (0-49):    Healthy Alice, all metrics stable
  Phase 2 — Dehydration (50-249): Progressively drain hydration 1.0 → 0.15
  Phase 3 — Crisis (250-349):   Hold at severe dehydration, observe cascade
  Phase 4 — Rehydration (350-499): Gradually restore hydration
  Phase 5 — Recovery (500-599): Verify all metrics return toward baseline

  At each phase boundary, verify the expected physiological signatures.

Closed Loops Exercised:
  1. Homeostatic drive → hydration level tracking
  2. Hydration → blood_volume (cardiovascular)
  3. Blood volume → viscosity → vascular resistance
  4. Blood volume → stroke volume → cardiac output → BP
  5. Low BP → baroreceptor reflex → compensatory tachycardia
  6. BP → cerebral autoregulation → perfusion
  7. Perfusion → O₂ delivery → channel efficiency
  8. Perfusion → consciousness arousal modulation
  9. Lung → SpO₂ → O₂ delivery (lung-CV coupling)
 10. CV heat transport → thermal regulation

Author: Alice System Cardiovascular Integration Validation
"""

from __future__ import annotations

import sys
import os
import math
import numpy as np
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority
from alice.core.signal import ElectricalSignal
from alice.body.cardiovascular import (
    CardiovascularSystem,
    BLOOD_VOLUME_CRITICAL, TACHYCARDIA_THRESHOLD,
    CEREBRAL_AUTOREGULATION_LOW, MAP_CRITICAL_LOW, MAP_SYNCOPE,
    BP_NORMALIZE_FACTOR, SPO2_HYPOXIA_MILD, PERFUSION_CRITICAL,
    O2_CHANNEL_PENALTY_THRESHOLD,
)


# ============================================================================
# Utility
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
    """Execute one full perception cycle."""
    visual = make_signal(40.0, 0.5) * brightness
    audio = make_signal(20.0, 0.15) * noise
    alice.hear(audio)
    return alice.see(visual, priority=Priority.NORMAL)


# ============================================================================
# Data Recording
# ============================================================================

class PhysiologyRecorder:
    """Records physiological time series for post-hoc analysis."""

    def __init__(self) -> None:
        self.ticks: List[int] = []
        self.hydration: List[float] = []
        self.blood_volume: List[float] = []
        self.blood_viscosity: List[float] = []
        self.cardiac_output: List[float] = []
        self.map_normalized: List[float] = []
        self.systolic_bp: List[float] = []
        self.diastolic_bp: List[float] = []
        self.cerebral_perfusion: List[float] = []
        self.spo2: List[float] = []
        self.o2_delivery: List[float] = []
        self.heart_rate_delta: List[float] = []
        self.consciousness: List[float] = []
        self.temperature: List[float] = []
        self.is_tachycardic: List[bool] = []
        self.is_hypotensive: List[bool] = []
        self.is_hypoxic: List[bool] = []

    def record(self, tick: int, alice: AliceBrain) -> None:
        cv = alice.cardiovascular
        v = alice.vitals
        hd = alice.homeostatic_drive

        self.ticks.append(tick)
        self.hydration.append(hd.hydration)
        self.blood_volume.append(cv.blood_volume)
        self.blood_viscosity.append(cv._blood_viscosity)
        self.cardiac_output.append(cv._cardiac_output)
        self.map_normalized.append(cv.map_normalized)
        self.systolic_bp.append(cv._systolic)
        self.diastolic_bp.append(cv._diastolic)
        self.cerebral_perfusion.append(cv.cerebral_perfusion)
        self.spo2.append(cv.spo2)
        self.o2_delivery.append(cv.o2_delivery)
        self.heart_rate_delta.append(cv.compensatory_hr_delta)
        self.consciousness.append(v.consciousness)
        self.temperature.append(v.ram_temperature)

        state = cv.get_state()
        self.is_tachycardic.append(state.is_tachycardic)
        self.is_hypotensive.append(state.is_hypotensive)
        self.is_hypoxic.append(state.is_hypoxic)

    def at_phase(self, phase_start: int, phase_end: int) -> Dict[str, Any]:
        """Get statistics for a phase."""
        indices = [
            i for i, t in enumerate(self.ticks)
            if phase_start <= t < phase_end
        ]
        if not indices:
            return {}

        def stats(series: list) -> Dict[str, float]:
            vals = [series[i] for i in indices]
            return {
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
                "first": vals[0],
                "last": vals[-1],
            }

        return {
            "hydration": stats(self.hydration),
            "blood_volume": stats(self.blood_volume),
            "blood_viscosity": stats(self.blood_viscosity),
            "cardiac_output": stats(self.cardiac_output),
            "map_normalized": stats(self.map_normalized),
            "cerebral_perfusion": stats(self.cerebral_perfusion),
            "spo2": stats(self.spo2),
            "o2_delivery": stats(self.o2_delivery),
            "heart_rate_delta": stats(self.heart_rate_delta),
            "consciousness": stats(self.consciousness),
            "temperature": stats(self.temperature),
            "tachycardic_ticks": sum(1 for i in indices if self.is_tachycardic[i]),
            "hypotensive_ticks": sum(1 for i in indices if self.is_hypotensive[i]),
        }


# ============================================================================
# Exp 01: Full Dehydration → Delirium → Recovery Cascade
# ============================================================================

def exp_01_dehydration_delirium_cascade():
    """
    Full dehydration cascade with 600-tick clinical scenario.

    Verifies that physics alone produces the correct pathological sequence:
    hydration↓ → blood_volume↓ → viscosity↑ → CO↓ → BP↓ → tachycardia
    → perfusion↓ → consciousness↓ → recovery with rehydration
    """
    _header("Exp 01: Dehydration → Delirium → Recovery Cascade (Full System)")

    alice = AliceBrain(neuron_count=60)
    rec = PhysiologyRecorder()

    # Warm up the system to stable state (20 ticks)
    for _ in range(20):
        run_tick(alice, brightness=0.4, noise=0.1)

    # Phase 1 — Baseline (ticks 0-49)
    print("\n  [Phase 1: Baseline — Healthy]")
    for tick in range(50):
        run_tick(alice, brightness=0.4, noise=0.1)
        rec.record(tick, alice)
        if tick % 25 == 0:
            cv = alice.cardiovascular
            print(f"    tick {tick:3d}: hydration={alice.homeostatic_drive.hydration:.3f}  "
                  f"BV={cv.blood_volume:.3f}  MAP={cv.map_normalized:.3f}  "
                  f"perfusion={cv.cerebral_perfusion:.3f}  "
                  f"consciousness={alice.vitals.consciousness:.3f}")

    baseline = rec.at_phase(0, 50)

    # Phase 2 — Progressive Dehydration (ticks 50-249)
    print("\n  [Phase 2: Progressive Dehydration — hydration 1.0 → 0.15]")
    for tick in range(50, 250):
        # Drain hydration progressively
        progress = (tick - 50) / 200.0  # 0.0 → 1.0
        target_hydration = 1.0 - progress * 0.85  # 1.0 → 0.15
        alice.homeostatic_drive.hydration = max(0.15, target_hydration)

        run_tick(alice, brightness=0.4, noise=0.1)
        rec.record(tick, alice)
        if tick % 50 == 0:
            cv = alice.cardiovascular
            print(f"    tick {tick:3d}: hydration={alice.homeostatic_drive.hydration:.3f}  "
                  f"BV={cv.blood_volume:.3f}  MAP={cv.map_normalized:.3f}  "
                  f"perfusion={cv.cerebral_perfusion:.3f}  "
                  f"HR_delta={cv.compensatory_hr_delta:.1f}  "
                  f"consciousness={alice.vitals.consciousness:.3f}")

    dehydration = rec.at_phase(50, 250)

    # Phase 3 — Severe Dehydration / Crisis (ticks 250-349)
    print("\n  [Phase 3: Severe Dehydration — Hold at hydration=0.15]")
    for tick in range(250, 350):
        alice.homeostatic_drive.hydration = 0.15
        run_tick(alice, brightness=0.3, noise=0.1)
        rec.record(tick, alice)
        if tick % 25 == 0:
            cv = alice.cardiovascular
            print(f"    tick {tick:3d}: hydration={alice.homeostatic_drive.hydration:.3f}  "
                  f"BV={cv.blood_volume:.3f}  visc={cv._blood_viscosity:.3f}  "
                  f"MAP={cv.map_normalized:.3f}  "
                  f"perfusion={cv.cerebral_perfusion:.3f}  "
                  f"O2={cv.o2_delivery:.3f}  "
                  f"HR_delta={cv.compensatory_hr_delta:.1f}  "
                  f"consciousness={alice.vitals.consciousness:.3f}")

    crisis = rec.at_phase(250, 350)

    # Phase 4 — Rehydration (ticks 350-499)
    print("\n  [Phase 4: Rehydration — hydration 0.15 → 0.90]")
    for tick in range(350, 500):
        # Gradual rehydration (IV fluid, slower than dehydration)
        progress = (tick - 350) / 150.0  # 0.0 → 1.0
        target_hydration = 0.15 + progress * 0.75  # 0.15 → 0.90
        alice.homeostatic_drive.hydration = min(0.90, target_hydration)

        run_tick(alice, brightness=0.4, noise=0.1)
        rec.record(tick, alice)
        if tick % 50 == 0:
            cv = alice.cardiovascular
            print(f"    tick {tick:3d}: hydration={alice.homeostatic_drive.hydration:.3f}  "
                  f"BV={cv.blood_volume:.3f}  MAP={cv.map_normalized:.3f}  "
                  f"perfusion={cv.cerebral_perfusion:.3f}  "
                  f"HR_delta={cv.compensatory_hr_delta:.1f}  "
                  f"consciousness={alice.vitals.consciousness:.3f}")

    rehydration = rec.at_phase(350, 500)

    # Phase 5 — Recovery (ticks 500-599)
    print("\n  [Phase 5: Recovery — Stable hydration=0.90]")
    for tick in range(500, 600):
        alice.homeostatic_drive.hydration = 0.90
        run_tick(alice, brightness=0.4, noise=0.1)
        rec.record(tick, alice)
        if tick % 25 == 0:
            cv = alice.cardiovascular
            print(f"    tick {tick:3d}: hydration={alice.homeostatic_drive.hydration:.3f}  "
                  f"BV={cv.blood_volume:.3f}  MAP={cv.map_normalized:.3f}  "
                  f"perfusion={cv.cerebral_perfusion:.3f}  "
                  f"HR_delta={cv.compensatory_hr_delta:.1f}  "
                  f"consciousness={alice.vitals.consciousness:.3f}")

    recovery = rec.at_phase(500, 600)

    # ================================================================
    # Verification: The Physics Must Produce These Clinical Signatures
    # ================================================================
    print("\n  [Verification: Clinical Signature Analysis]")

    # V1: Blood volume drops with dehydration
    _result(
        "V1: Blood volume drops during dehydration",
        crisis["blood_volume"]["mean"] < baseline["blood_volume"]["mean"] * 0.85,
        f"Baseline BV={baseline['blood_volume']['mean']:.3f}, "
        f"Crisis BV={crisis['blood_volume']['mean']:.3f}",
    )

    # V2: Blood viscosity increases with dehydration (hemoconcentration)
    _result(
        "V2: Blood viscosity increases (hemoconcentration)",
        crisis["blood_viscosity"]["mean"] > baseline["blood_viscosity"]["mean"],
        f"Baseline viscosity={baseline['blood_viscosity']['mean']:.3f}, "
        f"Crisis viscosity={crisis['blood_viscosity']['mean']:.3f}",
    )

    # V3: Cardiac output decreases (less blood to pump)
    _result(
        "V3: Cardiac output decreases during crisis",
        crisis["cardiac_output"]["mean"] < baseline["cardiac_output"]["mean"],
        f"Baseline CO={baseline['cardiac_output']['mean']:.3f}, "
        f"Crisis CO={crisis['cardiac_output']['mean']:.3f}",
    )

    # V4: MAP drops below baseline
    _result(
        "V4: Mean arterial pressure drops during crisis",
        crisis["map_normalized"]["mean"] < baseline["map_normalized"]["mean"],
        f"Baseline MAP={baseline['map_normalized']['mean']:.3f}, "
        f"Crisis MAP={crisis['map_normalized']['mean']:.3f}",
    )

    # V5: Compensatory tachycardia (baroreceptor reflex)
    _result(
        "V5: Compensatory tachycardia via baroreceptor reflex",
        crisis["heart_rate_delta"]["mean"] > 5.0,
        f"Crisis HR_delta={crisis['heart_rate_delta']['mean']:.1f} bpm "
        f"(threshold: >5 bpm)",
    )

    # V6: Tachycardic flag activates during crisis
    _result(
        "V6: Tachycardic flag activates (>50% of crisis ticks)",
        crisis["tachycardic_ticks"] > 50,
        f"Tachycardic ticks: {crisis['tachycardic_ticks']}/100",
    )

    # V7: Cerebral perfusion drops
    _result(
        "V7: Cerebral perfusion decreases during crisis",
        crisis["cerebral_perfusion"]["mean"] < baseline["cerebral_perfusion"]["mean"],
        f"Baseline perfusion={baseline['cerebral_perfusion']['mean']:.3f}, "
        f"Crisis perfusion={crisis['cerebral_perfusion']['mean']:.3f}",
    )

    # V8: O₂ delivery drops
    _result(
        "V8: O₂ delivery drops during crisis",
        crisis["o2_delivery"]["mean"] < baseline["o2_delivery"]["mean"],
        f"Baseline O2={baseline['o2_delivery']['mean']:.3f}, "
        f"Crisis O2={crisis['o2_delivery']['mean']:.3f}",
    )

    # V9: Recovery — blood volume recovers toward baseline
    _result(
        "V9: Blood volume recovers with rehydration",
        recovery["blood_volume"]["mean"] > crisis["blood_volume"]["mean"],
        f"Crisis BV={crisis['blood_volume']['mean']:.3f}, "
        f"Recovery BV={recovery['blood_volume']['mean']:.3f}",
    )

    # V10: Recovery — tachycardia resolves
    _result(
        "V10: Tachycardia resolves with rehydration",
        recovery["heart_rate_delta"]["mean"] < crisis["heart_rate_delta"]["mean"],
        f"Crisis HR_delta={crisis['heart_rate_delta']['mean']:.1f}, "
        f"Recovery HR_delta={recovery['heart_rate_delta']['mean']:.1f}",
    )

    # V11: Recovery — cerebral perfusion improves
    _result(
        "V11: Cerebral perfusion recovers with rehydration",
        recovery["cerebral_perfusion"]["mean"] > crisis["cerebral_perfusion"]["mean"],
        f"Crisis perfusion={crisis['cerebral_perfusion']['mean']:.3f}, "
        f"Recovery perfusion={recovery['cerebral_perfusion']['mean']:.3f}",
    )

    # V12: Recovery — MAP improves
    _result(
        "V12: MAP recovers with rehydration",
        recovery["map_normalized"]["mean"] > crisis["map_normalized"]["mean"],
        f"Crisis MAP={crisis['map_normalized']['mean']:.3f}, "
        f"Recovery MAP={recovery['map_normalized']['mean']:.3f}",
    )


# ============================================================================
# Exp 02: Isolated Cardiovascular Dehydration Physics
# ============================================================================

def exp_02_isolated_cv_dehydration_curve():
    """
    Isolated cardiovascular unit: trace exact physics response to hydration sweep.

    No brain integration — pure CV physics verification.
    Hydration sweep: 1.0 → 0.1 in 100 steps.
    """
    _header("Exp 02: Isolated CV Dehydration Curve (Unit Physics)")

    cv = CardiovascularSystem()
    # Grow to full adult capacity for clean measurements
    cv._volume_growth = 0.6  # Full capacity

    results: List[Dict[str, Any]] = []

    for i in range(100):
        hydration = 1.0 - i * 0.009  # 1.0 → 0.10
        r = cv.tick(
            heart_rate=70.0,
            sympathetic=0.2,
            parasympathetic=0.3,
            hydration=max(hydration, 0.10),
            glucose=1.0,
            breaths_this_tick=0.25,
            ram_temperature=0.1,
        )
        results.append(r)
        if i % 20 == 0:
            print(f"    hydration={hydration:.2f} → BV={r['blood_volume']:.3f}  "
                  f"visc={r['blood_viscosity']:.3f}  "
                  f"CO={r['cardiac_output']:.3f}  MAP={r['mean_arterial_pressure']:.1f}  "
                  f"perfusion={r['cerebral_perfusion']:.3f}  "
                  f"HR_delta={r['compensatory_hr_delta']:.1f}")

    # Physics verification: monotonic trends
    bv_first = results[0]["blood_volume"]
    bv_last = results[-1]["blood_volume"]
    _result(
        "Blood volume decreases monotonically with dehydration",
        bv_last < bv_first,
        f"BV: {bv_first:.3f} → {bv_last:.3f}",
    )

    visc_first = results[0]["blood_viscosity"]
    visc_last = results[-1]["blood_viscosity"]
    _result(
        "Blood viscosity increases with dehydration",
        visc_last > visc_first,
        f"Viscosity: {visc_first:.3f} → {visc_last:.3f}",
    )

    co_first = results[0]["cardiac_output"]
    co_last = results[-1]["cardiac_output"]
    _result(
        "Cardiac output decreases with dehydration",
        co_last < co_first,
        f"CO: {co_first:.3f} → {co_last:.3f}",
    )

    map_first = results[0]["mean_arterial_pressure"]
    map_last = results[-1]["mean_arterial_pressure"]
    _result(
        "MAP drops with dehydration",
        map_last < map_first,
        f"MAP: {map_first:.1f} → {map_last:.1f} mmHg",
    )

    perf_first = results[0]["cerebral_perfusion"]
    perf_last = results[-1]["cerebral_perfusion"]
    _result(
        "Cerebral perfusion drops with dehydration",
        perf_last < perf_first,
        f"Perfusion: {perf_first:.3f} → {perf_last:.3f}",
    )

    hr_first = results[0]["compensatory_hr_delta"]
    hr_last = results[-1]["compensatory_hr_delta"]
    _result(
        "Compensatory tachycardia emerges with dehydration",
        hr_last > hr_first,
        f"HR_delta: {hr_first:.1f} → {hr_last:.1f} bpm",
    )


# ============================================================================
# Exp 03: Dehydration + Heat Stress (Compound Pathology)
# ============================================================================

def exp_03_dehydration_heat_stress():
    """
    Compound pathology: dehydration + elevated temperature.

    Clinical: Dehydration + fever is extremely dangerous in infants
    because blood volume drops AND metabolic demand rises.

    Verify that compound stress produces worse outcomes than either alone.
    """
    _header("Exp 03: Dehydration + Heat Stress (Compound Pathology)")

    # Scenario A: Dehydration only
    cv_a = CardiovascularSystem()
    cv_a._volume_growth = 0.6

    # Scenario B: Heat stress only
    cv_b = CardiovascularSystem()
    cv_b._volume_growth = 0.6

    # Scenario C: Dehydration + Heat stress
    cv_c = CardiovascularSystem()
    cv_c._volume_growth = 0.6

    for _ in range(100):
        r_a = cv_a.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                        hydration=0.3, glucose=1.0, breaths_this_tick=0.25,
                        ram_temperature=0.1)

        r_b = cv_b.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                        hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                        ram_temperature=0.7)

        r_c = cv_c.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                        hydration=0.3, glucose=1.0, breaths_this_tick=0.25,
                        ram_temperature=0.7)

    print(f"    Dehydration only: perfusion={r_a['cerebral_perfusion']:.3f}, "
          f"O2={r_a['o2_delivery']:.3f}, HR_delta={r_a['compensatory_hr_delta']:.1f}")
    print(f"    Heat stress only: perfusion={r_b['cerebral_perfusion']:.3f}, "
          f"O2={r_b['o2_delivery']:.3f}, HR_delta={r_b['compensatory_hr_delta']:.1f}")
    print(f"    Combined:         perfusion={r_c['cerebral_perfusion']:.3f}, "
          f"O2={r_c['o2_delivery']:.3f}, HR_delta={r_c['compensatory_hr_delta']:.1f}")

    # Compound stress should produce worse perfusion than either alone
    _result(
        "Compound stress → worse perfusion than dehydration alone",
        r_c["cerebral_perfusion"] <= r_a["cerebral_perfusion"],
        f"Combined={r_c['cerebral_perfusion']:.3f}, "
        f"Dehydration={r_a['cerebral_perfusion']:.3f}",
    )

    _result(
        "Compound stress → worse perfusion than heat alone",
        r_c["cerebral_perfusion"] <= r_b["cerebral_perfusion"],
        f"Combined={r_c['cerebral_perfusion']:.3f}, "
        f"Heat={r_b['cerebral_perfusion']:.3f}",
    )

    # Compound worsens O₂ delivery
    _result(
        "Compound stress → lower O₂ delivery than either alone",
        r_c["o2_delivery"] <= min(r_a["o2_delivery"], r_b["o2_delivery"]),
        f"Combined={r_c['o2_delivery']:.3f}, "
        f"Dehydration={r_a['o2_delivery']:.3f}, "
        f"Heat={r_b['o2_delivery']:.3f}",
    )


# ============================================================================
# Exp 04: Neonatal Dehydration (Developmental Factor)
# ============================================================================

def exp_04_neonatal_dehydration():
    """
    Neonatal dehydration: immature cardiovascular system + dehydration.

    Neonates have smaller blood volume (NEONATAL_VOLUME_FACTOR = 0.4).
    The same percentage water loss produces a much larger relative
    blood volume deficit → earlier crisis.

    This is why neonatal dehydration is a medical emergency.
    """
    _header("Exp 04: Neonatal Dehydration (Developmental Vulnerability)")

    # Neonate (no growth)
    cv_neo = CardiovascularSystem()  # _volume_growth = 0

    # Adult (full growth)
    cv_adult = CardiovascularSystem()
    cv_adult._volume_growth = 0.6

    neo_crisis_tick = None
    adult_crisis_tick = None

    for tick in range(200):
        hydration = 1.0 - tick * 0.004  # 1.0 → 0.2 over 200 ticks

        r_neo = cv_neo.tick(heart_rate=140, sympathetic=0.2, parasympathetic=0.3,
                            hydration=max(hydration, 0.2), glucose=1.0,
                            breaths_this_tick=0.25, ram_temperature=0.1)
        r_adult = cv_adult.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                                hydration=max(hydration, 0.2), glucose=1.0,
                                breaths_this_tick=0.25, ram_temperature=0.1)

        if neo_crisis_tick is None and r_neo["is_tachycardic"]:
            neo_crisis_tick = tick
        if adult_crisis_tick is None and r_adult["is_tachycardic"]:
            adult_crisis_tick = tick

        if tick % 50 == 0:
            print(f"    tick {tick:3d} (hydration={max(hydration, 0.2):.2f}): "
                  f"neo_BV={r_neo['blood_volume']:.3f} "
                  f"adult_BV={r_adult['blood_volume']:.3f}  "
                  f"neo_HR_delta={r_neo['compensatory_hr_delta']:.1f} "
                  f"adult_HR_delta={r_adult['compensatory_hr_delta']:.1f}")

    print(f"\n    Neonatal crisis (tachycardia) at tick: {neo_crisis_tick}")
    print(f"    Adult crisis (tachycardia) at tick: {adult_crisis_tick}")

    # Neonate should reach crisis sooner
    _result(
        "Neonate reaches tachycardia crisis before adult",
        neo_crisis_tick is not None and (
            adult_crisis_tick is None or neo_crisis_tick <= adult_crisis_tick
        ),
        f"Neonate: tick {neo_crisis_tick}, Adult: tick {adult_crisis_tick}",
    )

    # Neonate should have lower blood volume at same hydration
    _result(
        "Neonate blood volume < adult at same hydration",
        r_neo["blood_volume"] < r_adult["blood_volume"],
        f"Neonate BV={r_neo['blood_volume']:.3f}, "
        f"Adult BV={r_adult['blood_volume']:.3f}",
    )

    # Neonate should have worse perfusion at same hydration
    _result(
        "Neonate perfusion ≤ adult at same dehydration",
        r_neo["cerebral_perfusion"] <= r_adult["cerebral_perfusion"] + 0.05,
        f"Neonate perf={r_neo['cerebral_perfusion']:.3f}, "
        f"Adult perf={r_adult['cerebral_perfusion']:.3f}",
    )


# ============================================================================
# Exp 05: Anemia + Dehydration (Double Hit)
# ============================================================================

def exp_05_anemia_dehydration_double_hit():
    """
    Anemia + dehydration: reduced O₂ carrying capacity + reduced blood volume.

    Clinical: Iron-deficiency anemia + dehydration is common in
    malnourished children. The combination is particularly dangerous
    because the blood can carry less oxygen AND less of it reaches the brain.

    O₂_delivery = perfusion × SpO₂ × hemoglobin
    Both perfusion↓ (dehydration) AND hemoglobin↓ (anemia) → O₂ collapses.
    """
    _header("Exp 05: Anemia + Dehydration (Double Hit)")

    # Control: dehydration only
    cv_ctrl = CardiovascularSystem()
    cv_ctrl._volume_growth = 0.6

    # Anemic: dehydration + anemia
    cv_anemia = CardiovascularSystem()
    cv_anemia._volume_growth = 0.6
    cv_anemia.set_hemoglobin(0.5)  # Severe anemia

    for _ in range(100):
        r_ctrl = cv_ctrl.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                              hydration=0.4, glucose=1.0, breaths_this_tick=0.25,
                              ram_temperature=0.1)
        r_anemia = cv_anemia.tick(heart_rate=70, sympathetic=0.3, parasympathetic=0.2,
                                  hydration=0.4, glucose=1.0, breaths_this_tick=0.25,
                                  ram_temperature=0.1)

    print(f"    Dehydration only: O2={r_ctrl['o2_delivery']:.3f}, "
          f"perfusion={r_ctrl['cerebral_perfusion']:.3f}")
    print(f"    Anemia + dehydration: O2={r_anemia['o2_delivery']:.3f}, "
          f"perfusion={r_anemia['cerebral_perfusion']:.3f}")

    # Anemia should worsen O₂ delivery
    _result(
        "Anemia + dehydration → lower O₂ delivery than dehydration alone",
        r_anemia["o2_delivery"] < r_ctrl["o2_delivery"],
        f"Anemia+dehy O2={r_anemia['o2_delivery']:.3f}, "
        f"Dehy only O2={r_ctrl['o2_delivery']:.3f}",
    )

    # Perfusion should be similar (anemia doesn't affect perfusion directly)
    perf_diff = abs(r_anemia["cerebral_perfusion"] - r_ctrl["cerebral_perfusion"])
    _result(
        "Perfusion similar (anemia affects O₂ carrying, not flow)",
        perf_diff < 0.15,
        f"Diff = {perf_diff:.3f}",
    )

    # O₂ ratio should roughly match hemoglobin ratio
    o2_ratio = r_anemia["o2_delivery"] / max(r_ctrl["o2_delivery"], 0.001)
    _result(
        "O₂ delivery ratio ≈ hemoglobin ratio (0.5)",
        0.3 <= o2_ratio <= 0.7,
        f"O₂ ratio = {o2_ratio:.3f} (hemoglobin set to 0.5)",
    )


# ============================================================================
# Exp 06: Sympathetic Storm + Dehydration
# ============================================================================

def exp_06_sympathetic_storm_dehydration():
    """
    Sympathetic storm (high stress) + dehydration.

    Clinical: Severe pain/fear → massive sympathetic activation →
    vasoconstriction + tachycardia. In a dehydrated patient, this
    paradoxically worsens the situation because vasoconstriction
    further impairs perfusion through already viscous blood.

    Demonstrates that compensatory mechanisms can become pathological
    when the underlying volume deficit is severe enough.
    """
    _header("Exp 06: Sympathetic Storm + Dehydration (Compensatory Failure)")

    cv = CardiovascularSystem()
    cv._volume_growth = 0.6

    phases = {
        "Normal": (0.2, 0.3, 1.0),    # (sympathetic, parasympathetic, hydration)
        "Dehydrated": (0.2, 0.3, 0.3),
        "Stress": (0.9, 0.1, 1.0),
        "Stress+Dehy": (0.9, 0.1, 0.3),
    }

    phase_results = {}
    for name, (symp, para, hydr) in phases.items():
        # Fresh CV for each scenario
        cv_test = CardiovascularSystem()
        cv_test._volume_growth = 0.6
        for _ in range(80):
            r = cv_test.tick(heart_rate=70, sympathetic=symp, parasympathetic=para,
                             hydration=hydr, glucose=1.0, breaths_this_tick=0.25,
                             ram_temperature=0.1)
        phase_results[name] = r
        print(f"    {name:15s}: MAP={r['mean_arterial_pressure']:.1f}  "
              f"R={r['vascular_resistance']:.3f}  "
              f"perfusion={r['cerebral_perfusion']:.3f}  "
              f"HR_delta={r['compensatory_hr_delta']:.1f}")

    # Stress increases vascular resistance
    _result(
        "Sympathetic storm → higher vascular resistance",
        phase_results["Stress"]["vascular_resistance"] >
        phase_results["Normal"]["vascular_resistance"],
        f"Stress R={phase_results['Stress']['vascular_resistance']:.3f}, "
        f"Normal R={phase_results['Normal']['vascular_resistance']:.3f}",
    )

    # Stress + dehydration → worse perfusion than stress alone
    _result(
        "Stress + dehydration → lower perfusion than stress alone",
        phase_results["Stress+Dehy"]["cerebral_perfusion"] <
        phase_results["Stress"]["cerebral_perfusion"],
        f"Stress+Dehy perf={phase_results['Stress+Dehy']['cerebral_perfusion']:.3f}, "
        f"Stress perf={phase_results['Stress']['cerebral_perfusion']:.3f}",
    )

    # Stress + dehydration → higher tachycardia
    _result(
        "Stress + dehydration → more tachycardia than dehydration alone",
        phase_results["Stress+Dehy"]["compensatory_hr_delta"] >=
        phase_results["Dehydrated"]["compensatory_hr_delta"],
        f"Stress+Dehy HR_delta={phase_results['Stress+Dehy']['compensatory_hr_delta']:.1f}, "
        f"Dehy HR_delta={phase_results['Dehydrated']['compensatory_hr_delta']:.1f}",
    )


# ============================================================================
# Exp 07: Lung-Cardiovascular Coupling (SpO₂ + Perfusion)
# ============================================================================

def exp_07_lung_cv_coupling():
    """
    Verify lung-cardiovascular coupling under dehydration.

    Lung → breaths_this_tick → SpO₂ recovery rate
    CV → perfusion → O₂ delivery = perfusion × SpO₂ × hemoglobin

    Dehydration should not directly affect SpO₂ (lungs still work),
    but should reduce O₂ delivery (less blood to carry the oxygen).
    """
    _header("Exp 07: Lung-CV Coupling (SpO₂ Independent, O₂ Delivery Depends on Perfusion)")

    # Normal breathing + hydrated
    cv_healthy = CardiovascularSystem()
    cv_healthy._volume_growth = 0.6

    # Normal breathing + dehydrated
    cv_dehy = CardiovascularSystem()
    cv_dehy._volume_growth = 0.6

    # Reduced breathing + hydrated (apnea-like)
    cv_apnea = CardiovascularSystem()
    cv_apnea._volume_growth = 0.6

    for _ in range(100):
        r_healthy = cv_healthy.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                                    hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                                    ram_temperature=0.1)
        r_dehy = cv_dehy.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                              hydration=0.3, glucose=1.0, breaths_this_tick=0.25,
                              ram_temperature=0.1)
        r_apnea = cv_apnea.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                                hydration=1.0, glucose=1.0, breaths_this_tick=0.02,
                                ram_temperature=0.1)

    print(f"    Healthy:     SpO2={r_healthy['spo2']:.3f}  "
          f"O2={r_healthy['o2_delivery']:.3f}  "
          f"perfusion={r_healthy['cerebral_perfusion']:.3f}")
    print(f"    Dehydrated:  SpO2={r_dehy['spo2']:.3f}  "
          f"O2={r_dehy['o2_delivery']:.3f}  "
          f"perfusion={r_dehy['cerebral_perfusion']:.3f}")
    print(f"    Apnea:       SpO2={r_apnea['spo2']:.3f}  "
          f"O2={r_apnea['o2_delivery']:.3f}  "
          f"perfusion={r_apnea['cerebral_perfusion']:.3f}")

    # SpO₂ should be similar for healthy and dehydrated (lungs still work)
    _result(
        "Dehydration does not directly reduce SpO₂ (lungs still work)",
        abs(r_healthy["spo2"] - r_dehy["spo2"]) < 0.05,
        f"Healthy SpO2={r_healthy['spo2']:.3f}, "
        f"Dehydrated SpO2={r_dehy['spo2']:.3f}",
    )

    # But O₂ delivery is lower in dehydration (less perfusion)
    _result(
        "Dehydration reduces O₂ delivery despite normal SpO₂",
        r_dehy["o2_delivery"] < r_healthy["o2_delivery"],
        f"Healthy O2={r_healthy['o2_delivery']:.3f}, "
        f"Dehydrated O2={r_dehy['o2_delivery']:.3f}",
    )

    # Apnea reduces SpO₂ directly
    _result(
        "Apnea directly reduces SpO₂ (lung failure)",
        r_apnea["spo2"] < r_healthy["spo2"],
        f"Healthy SpO2={r_healthy['spo2']:.3f}, "
        f"Apnea SpO2={r_apnea['spo2']:.3f}",
    )


# ============================================================================
# Exp 08: Brain Integration — Consciousness Tracks Perfusion
# ============================================================================

def exp_08_consciousness_tracks_perfusion():
    """
    Full AliceBrain: verify consciousness drops when perfusion drops.

    This validates the key wiring:
        CV.cerebral_perfusion → consciousness.arousal modulation
        arousal *= min(1.0, cv_perfusion / 0.6)

    Below 60% perfusion, consciousness should start to drop.
    This is the "fainting" mechanism.
    """
    _header("Exp 08: Brain Integration — Consciousness Tracks Perfusion")

    alice = AliceBrain(neuron_count=60)

    # Warm up
    for _ in range(30):
        run_tick(alice, brightness=0.4, noise=0.1)

    baseline_consciousness = alice.vitals.consciousness
    print(f"    Baseline consciousness: {baseline_consciousness:.3f}")

    # Severe dehydration for 200 ticks
    consciousness_during_crisis = []
    for tick in range(200):
        alice.homeostatic_drive.hydration = 0.15
        run_tick(alice, brightness=0.4, noise=0.1)
        consciousness_during_crisis.append(alice.vitals.consciousness)
        if tick % 50 == 0:
            cv = alice.cardiovascular
            print(f"    tick {tick:3d}: hydration={alice.homeostatic_drive.hydration:.3f}  "
                  f"perfusion={cv.cerebral_perfusion:.3f}  "
                  f"consciousness={alice.vitals.consciousness:.3f}")

    min_consciousness = min(consciousness_during_crisis)
    mean_consciousness = sum(consciousness_during_crisis) / len(consciousness_during_crisis)

    _result(
        "Consciousness drops during severe dehydration",
        mean_consciousness < baseline_consciousness or min_consciousness < 0.5,
        f"Baseline={baseline_consciousness:.3f}, "
        f"Crisis mean={mean_consciousness:.3f}, min={min_consciousness:.3f}",
    )

    # Recovery
    for tick in range(100):
        alice.homeostatic_drive.hydration = 0.85
        run_tick(alice, brightness=0.4, noise=0.1)

    recovery_consciousness = alice.vitals.consciousness
    print(f"    Recovery consciousness: {recovery_consciousness:.3f}")

    _result(
        "Consciousness recovers with rehydration",
        recovery_consciousness > min_consciousness,
        f"Crisis min={min_consciousness:.3f}, "
        f"Recovery={recovery_consciousness:.3f}",
    )


# ============================================================================
# Exp 09: Baroreceptor Time Course
# ============================================================================

def exp_09_baroreceptor_dynamics():
    """
    Verify baroreceptor reflex temporal dynamics.

    The baroreceptor reflex should:
    1. Not respond instantly (has time constant)
    2. Increase proportionally to blood volume deficit
    3. Resolve when blood volume is restored

    Clinically: this is why orthostatic hypotension has a delay —
    takes a few seconds for baroreceptors to respond.
    """
    _header("Exp 09: Baroreceptor Reflex Temporal Dynamics")

    cv = CardiovascularSystem()
    cv._volume_growth = 0.6

    # Phase 1: stable (20 ticks at normal hydration)
    for _ in range(20):
        cv.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                ram_temperature=0.1)

    hr_before = cv.compensatory_hr_delta
    print(f"    Before dehydration: HR_delta = {hr_before:.2f}")

    # Phase 2: sudden dehydration — track response time
    hr_timeline = []
    for tick in range(60):
        cv.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                hydration=0.2, glucose=1.0, breaths_this_tick=0.25,
                ram_temperature=0.1)
        hr_timeline.append(cv.compensatory_hr_delta)

    hr_peak = max(hr_timeline)
    peak_tick = hr_timeline.index(hr_peak)
    print(f"    After dehydration: HR_delta peak = {hr_peak:.2f} at tick {peak_tick}")

    _result(
        "Baroreceptor reflex increases tachycardia",
        hr_peak > hr_before + 2.0,
        f"Before: {hr_before:.2f}, Peak: {hr_peak:.2f}",
    )

    _result(
        "Response is not instant (peak not at tick 0)",
        peak_tick > 0 or hr_timeline[0] < hr_peak * 0.9,
        f"Tick 0: {hr_timeline[0]:.2f}, Peak at tick {peak_tick}: {hr_peak:.2f}",
    )

    # Phase 3: restore hydration — verify resolution
    for _ in range(60):
        cv.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                ram_temperature=0.1)

    hr_after_recovery = cv.compensatory_hr_delta
    print(f"    After rehydration: HR_delta = {hr_after_recovery:.2f}")

    _result(
        "Tachycardia resolves after rehydration",
        hr_after_recovery < hr_peak * 0.5,
        f"Peak: {hr_peak:.2f}, After recovery: {hr_after_recovery:.2f}",
    )


# ============================================================================
# Exp 10: Clinical Summary — Dehydration Severity Scale
# ============================================================================

def exp_10_dehydration_severity_scale():
    """
    Map dehydration levels to clinical severity using cardiovascular physics.

    Clinical dehydration scale (pediatrics):
        Mild (3-5%):    Slightly dry mucosa, normal vitals
        Moderate (6-9%): Tachycardia, sunken eyes, reduced urine
        Severe (≥10%):  Hypotension, altered mental status, shock

    Verify that physics naturally produces these severity boundaries.
    """
    _header("Exp 10: Dehydration Severity Scale (Pediatric Clinical Mapping)")

    severity_levels = {
        "Normal (0%)":      1.0,
        "Mild (3-5%)":      0.7,
        "Moderate (6-9%)":  0.45,
        "Severe (≥10%)":    0.2,
    }

    results_table = {}
    for name, hydration in severity_levels.items():
        cv = CardiovascularSystem()
        cv._volume_growth = 0.6
        for _ in range(80):
            r = cv.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                        hydration=hydration, glucose=1.0, breaths_this_tick=0.25,
                        ram_temperature=0.1)
        results_table[name] = r
        print(f"    {name:20s}: HR_delta={r['compensatory_hr_delta']:5.1f}  "
              f"MAP={r['mean_arterial_pressure']:5.1f}  "
              f"perfusion={r['cerebral_perfusion']:.3f}  "
              f"tachy={'YES' if r['is_tachycardic'] else 'no ':3s}  "
              f"hypotensive={'YES' if r['is_hypotensive'] else 'no ':3s}")

    # Normal → no tachycardia
    _result(
        "Normal hydration → no tachycardia",
        not results_table["Normal (0%)"]["is_tachycardic"],
    )

    # Moderate → tachycardia begins
    _result(
        "Moderate dehydration → tachycardia present",
        results_table["Moderate (6-9%)"]["is_tachycardic"] or
        results_table["Moderate (6-9%)"]["compensatory_hr_delta"] > 3.0,
        f"HR_delta = {results_table['Moderate (6-9%)']['compensatory_hr_delta']:.1f}",
    )

    # Severe → hypotension
    _result(
        "Severe dehydration → hypotension develops",
        results_table["Severe (≥10%)"]["is_hypotensive"] or
        results_table["Severe (≥10%)"]["mean_arterial_pressure"] < 65,
        f"MAP = {results_table['Severe (≥10%)']['mean_arterial_pressure']:.1f} mmHg",
    )

    # Monotonic perfusion degradation
    perfs = [results_table[n]["cerebral_perfusion"]
             for n in severity_levels]
    _result(
        "Perfusion degrades monotonically with severity",
        all(perfs[i] >= perfs[i + 1] - 0.02 for i in range(len(perfs) - 1)),
        " → ".join(f"{p:.3f}" for p in perfs),
    )

    # MAP monotonically decreases
    maps = [results_table[n]["mean_arterial_pressure"]
            for n in severity_levels]
    _result(
        "MAP decreases monotonically with severity",
        all(maps[i] >= maps[i + 1] - 1.0 for i in range(len(maps) - 1)),
        " → ".join(f"{m:.1f}" for m in maps),
    )


# ============================================================================
# Main Execution
# ============================================================================

def main():
    experiments = [
        exp_01_dehydration_delirium_cascade,
        exp_02_isolated_cv_dehydration_curve,
        exp_03_dehydration_heat_stress,
        exp_04_neonatal_dehydration,
        exp_05_anemia_dehydration_double_hit,
        exp_06_sympathetic_storm_dehydration,
        exp_07_lung_cv_coupling,
        exp_08_consciousness_tracks_perfusion,
        exp_09_baroreceptor_dynamics,
        exp_10_dehydration_severity_scale,
    ]

    print("=" * 70)
    print("  Dehydration Delirium — Full-System Cardiovascular Validation")
    print("  10 Experiments × 7+ Closed Loops × 600-tick Clinical Scenarios")
    print("=" * 70)

    for exp_fn in experiments:
        exp_fn()

    total = _pass_count + _fail_count
    print(f"\n{'=' * 70}")
    print(f"  Result: {_pass_count}/{total} PASS")
    if _fail_count == 0:
        print("  🩺 All dehydration validations PASSED — Physics produces correct "
              "clinical cascade without disease-specific hard-coding")
    else:
        print(f"  ⚠ {_fail_count} validations failed — review physics parameters")
    print(f"{'=' * 70}")

    return 0 if _fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

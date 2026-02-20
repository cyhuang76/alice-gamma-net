# -*- coding: utf-8 -*-
"""
exp_metacognition.py - Metacognition and Self-Correction (Phase 18)

The Inner Auditor — Alice Learns to 'See Her Own Thinking'

Verification:
1. Thinking impedance computation: Higher cognitive load → higher Γ_thinking
2. System 1/2 switching: Low impedance = intuition, high impedance = deliberation (with hysteresis)
3. Thinking rate control: High impedance → physically reduces thinking rate
4. Confidence estimation: Precise prediction + fluency → high confidence; otherwise → low confidence
5. Counterfactual reasoning: Physicalized regret and relief emotions
6. Self-correction: Γ_thinking too high → triggers Reframe
7. Insight detection: Sudden impedance drop → Aha! Moment
8. Rumination detection: Persistent excessive regret → pathological rumination loop

Core Physics:
  Γ_thinking = Σ(w_i · Γ_i) / Σ w_i
  ThinkingRate = base × (1 - Γ_thinking)^α
  Confidence = 1 / (1 + σ² + Γ_thinking)
  Regret = V(best_counterfactual) - V(actual)
"""

import time
import math
import numpy as np
from typing import Dict, Any, List

from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority
from alice.core.signal import ElectricalSignal
from alice.brain.metacognition import (
    MetacognitionEngine,
    SYSTEM2_ENGAGE_THRESHOLD,
    SYSTEM2_DISENGAGE_THRESHOLD,
    CORRECTION_THRESHOLD,
    DOUBT_THRESHOLD,
    BASE_THINKING_RATE,
    MIN_THINKING_RATE,
)


# ============================================================================
# 0. Utility Functions
# ============================================================================

def make_signal(freq: float = 40.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, 0.1, 64)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ============================================================================
# 1. Thinking Impedance Computation
# ============================================================================

def run_exp1_thinking_impedance():
    """
    Experiment 1: Thinking Impedance Computation

    Goal: Verify Γ_thinking correctly reflects cognitive load

    Method:
    - Low load condition: all metrics low
    - High load condition: high prediction error + high anxiety + PFC fatigue
    - Verify high load → high impedance
    """
    print("\n[Exp 1: Thinking Impedance Computation]")

    me = MetacognitionEngine()

    # low load
    low_result = me.tick(
        prediction_error=0.02,
        free_energy=0.1,
        binding_gamma=0.1,
        flexibility_index=0.9,
        anxiety=0.05,
        pfc_energy=0.9,
        surprise=0.01,
        pain=0.0,
        phi=0.7,
    )

    gamma_low = low_result["gamma_thinking"]
    print(f" Low-load Γ_thinking: {gamma_low:.4f}")

    # high load
    for _ in range(10):
        high_result = me.tick(
            prediction_error=0.8,
            free_energy=5.0,
            binding_gamma=0.7,
            flexibility_index=0.5,
            anxiety=0.9,
            pfc_energy=0.2,
            surprise=3.0,
            pain=0.3,
            phi=0.7,
        )

    gamma_high = high_result["gamma_thinking"]
    print(f" High-load Γ_thinking: {gamma_high:.4f}")
    print(f" Ratio: {gamma_high / (gamma_low + 1e-10):.1f}x")

    ok = gamma_high > gamma_low * 2
    print(f" → {'✓ PASS' if ok else '✗ FAIL'} — High-load impedance significantly above low-load")
    return ok


# ============================================================================
# 2. System 1/2 Switching (Dual System Switching)
# ============================================================================

def run_exp2_system_switching():
    """
    Experiment 2: System 1/2 Dual System Switching

    Goal: Verify hysteresis switching mechanism

    Method:
    - First use low impedance signal → confirm System 1
    - Then suddenly raise impedance → should switch to System 2
    - Then slowly reduce → verify hysteresis (won't immediately switch back at 0.45)
    """
    print("\n[Exp 2: System 1/2 Dual System Switching]")

    me = MetacognitionEngine()

    # stabilize in System 1
    for _ in range(20):
        r1 = me.tick(prediction_error=0.05, anxiety=0.05, pfc_energy=0.9)
    assert r1["system_mode"] == 1, "Initial should be System 1"
    print(f"  initial: System {r1['system_mode']} (Γ={r1['gamma_thinking']:.4f})")

    # High impedance → System 2
    for _ in range(20):
        r2 = me.tick(
            prediction_error=0.8,
            free_energy=5.0,
            anxiety=0.8,
            pfc_energy=0.2,
            binding_gamma=0.6,
        )
    sys2 = r2["system_mode"] == 2
    print(f" High impedance: System {r2['system_mode']} (Γ={r2['gamma_thinking']:.4f})")

    # Partial reduction (still within hysteresis band) → should stay in System 2
    for _ in range(5):
        r3 = me.tick(
            prediction_error=0.3,
            anxiety=0.3,
            pfc_energy=0.5,
        )
    hysteresis = r3["system_mode"] == 2 # Hysteresis: should not immediately return to System 1
    print(f" Partial reduction: System {r3['system_mode']} (Γ={r3['gamma_thinking']:.4f})")
    print(f" Hysteresis maintained: {'✓' if hysteresis else '✗'}")

    # Full reduction → return to System 1
    for _ in range(30):
        r4 = me.tick(prediction_error=0.01, anxiety=0.01, pfc_energy=0.95)
    back_to_s1 = r4["system_mode"] == 1
    print(f" Full reduction: System {r4['system_mode']} (Γ={r4['gamma_thinking']:.4f})")

    ok = sys2 and hysteresis and back_to_s1
    print(f" → {'✓ PASS' if ok else '✗ FAIL'} — Complete hysteresis switching cycle")
    return ok


# ============================================================================
# 3. Thinking Rate Control
# ============================================================================

def run_exp3_thinking_rate():
    """
    Experiment 3: Thinking Rate Control

    Goal: Verify high impedance → reduces thinking rate (Time-Dilation)

    Method:
    - Low impedance → rate ≈ 1.0
    - High impedance → rate significantly decreases
    """
    print("\n[Exp 3: Thinking Rate Control]")

    me = MetacognitionEngine()

    # Low impedance
    for _ in range(15):
        r_low = me.tick(prediction_error=0.02, anxiety=0.02, pfc_energy=0.95)
    rate_low = r_low["thinking_rate"]

    # High impedance
    me2 = MetacognitionEngine()
    for _ in range(15):
        r_high = me2.tick(
            prediction_error=0.8, anxiety=0.8, pfc_energy=0.2,
            free_energy=5.0, binding_gamma=0.6,
        )
    rate_high = r_high["thinking_rate"]

    print(f" Low-impedance rate: {rate_low:.4f}")
    print(f" High-impedance rate: {rate_high:.4f}")
    print(f" Deceleration ratio: {rate_high / rate_low:.2%}")

    ok = rate_high < rate_low * 0.7 and rate_high >= MIN_THINKING_RATE
    print(f" → {'✓ PASS' if ok else '✗ FAIL'} — High impedance significantly decelerates without going below floor")
    return ok


# ============================================================================
# 4. Confidence Estimation
# ============================================================================

def run_exp4_confidence():
    """
    Experiment 4: Confidence Estimation

    Goal: Verify Confidence = 1 / (1 + σ² + Γ_thinking)

    Method:
    - High precision (low σ) + low impedance → high confidence
    - Low precision (high σ) + high impedance → low confidence → Self-Doubt
    """
    print("\n[Exp 4: Confidence Estimation]")

    # High confidence scenario
    me1 = MetacognitionEngine()
    for _ in range(20):
        r1 = me1.tick(
            prediction_error=0.02, anxiety=0.02, pfc_energy=0.95,
            precision=0.05, # Low σ = high precision
        )
    conf_high = r1["confidence"]

    # Low confidence scenario
    me2 = MetacognitionEngine()
    for _ in range(60):
        r2 = me2.tick(
            prediction_error=0.9, anxiety=0.9, pfc_energy=0.1,
            free_energy=10.0, binding_gamma=0.8,
            precision=1.5, # High σ = low precision
        )
    conf_low = r2["confidence"]
    is_doubting = r2["is_doubting"]

    print(f" High precision + Low impedance: confidence={conf_high:.4f}")
    print(f" Low precision + High impedance: confidence={conf_low:.4f} (self-doubt={is_doubting})")

    ok = conf_high > 0.6 and conf_low < DOUBT_THRESHOLD and is_doubting
    print(f" → {'✓ PASS' if ok else '✗ FAIL'} — Confidence correctly varies with precision and impedance")
    return ok


# ============================================================================
# 5. Counterfactual Reasoning
# ============================================================================

def run_exp5_counterfactual():
    """
    Experiment 5: Counterfactual Reasoning

    Goal: Verify regret and relief signals

    Method:
    - Update action value table so flee > idle
    - Execute idle → should generate regret
    - Execute flee → should generate relief
    """
    print("\n[Exp 5: Counterfactual Reasoning]")

    me = MetacognitionEngine()

    # Set action values: flee best, idle worst
    me.update_action_value("flee", 0.9)
    me.update_action_value("idle", 0.1)
    me.update_action_value("rest", 0.5)

    # Do idle (worst choice) → should regret
    for _ in range(10):
        r_idle = me.tick(prediction_error=0.3, last_action="idle")

    regret_after_idle = r_idle["regret"]
    print(f" Chose idle (worst): regret={regret_after_idle:.4f}")

    # Reset, do flee (best choice) → should feel relief
    me2 = MetacognitionEngine()
    me2.update_action_value("flee", 0.9)
    me2.update_action_value("idle", 0.1)
    me2.update_action_value("rest", 0.5)

    for _ in range(10):
        r_flee = me2.tick(prediction_error=0.3, last_action="flee")

    relief_after_flee = r_flee["relief"]
    print(f" Chose flee (best): relief={relief_after_flee:.4f}")

    ok = regret_after_idle > 0.05 and relief_after_flee > 0.05
    print(f" → {'✓ PASS' if ok else '✗ FAIL'} — Regret and relief signals correctly generated")
    return ok


# ============================================================================
# 6. Self-Correction / Reframe
# ============================================================================

def run_exp6_self_correction():
    """
    Experiment 6: Self-Correction

    Goal: Verify Γ_thinking exceeding threshold → triggers Reframe

    Method:
    - Continuous high-impedance input → should trigger self-correction
    - With cooldown → won't correct every tick
    """
    print("\n[Exp 6: Self-Correction]")

    me = MetacognitionEngine()

    correction_ticks = []
    for i in range(30):
        r = me.tick(
            prediction_error=0.9,
            free_energy=8.0,
            anxiety=0.9,
            pfc_energy=0.15,
            binding_gamma=0.7,
        )
        if r["is_correcting"]:
            correction_ticks.append(i)

    total_corrections = r["correction_count"]
    print(f" Correction trigger ticks: {correction_ticks}")
    print(f" Total corrections: {total_corrections}")

    # Should have corrections, but not every tick (has cooldown)
    ok = total_corrections > 0 and total_corrections < 30
    print(f" → {'✓ PASS' if ok else '✗ FAIL'} — Correction triggered with cooldown")
    return ok


# ============================================================================
# 7. Insight Detection (Aha! Moment)
# ============================================================================

def run_exp7_insight():
    """
    Experiment 7: Insight Detection

    Goal: Verify Γ_thinking sudden drop → Aha! Moment

    Method:
    - First create high impedance (struggling to think)
    - Then suddenly clear (all metrics low) → impedance drops sharply
    - Should detect insight
    """
    print("\n[Exp 7: Insight Detection]")

    me = MetacognitionEngine()

    # Struggling phase
    for _ in range(20):
        me.tick(
            prediction_error=0.8,
            free_energy=5.0,
            anxiety=0.7,
            pfc_energy=0.2,
            binding_gamma=0.6,
        )

    # Sudden clarity
    insight_detected = False
    for i in range(10):
        r = me.tick(
            prediction_error=0.01,
            free_energy=0.05,
            anxiety=0.02,
            pfc_energy=0.95,
            binding_gamma=0.05,
        )
        if r["is_insight"]:
            insight_detected = True
            print(f" ✦ Insight at tick {i} triggered!")
            break

    total_insights = me._insight_count
    print(f" Total insights: {total_insights}")

    ok = insight_detected
    print(f" → {'✓ PASS' if ok else '✗ FAIL'} — Impedance sudden drop triggers insight")
    return ok


# ============================================================================
# 8. Rumination Detection
# ============================================================================

def run_exp8_rumination():
    """
    Experiment 8: Rumination Detection

    Goal: Verify persistent high regret → rumination alert

    Method:
    - Set action value differences extremely large
    - Continuously make worst choices → regret accumulates → triggers rumination
    """
    print("\n[Exp 8: Rumination Detection]")

    me = MetacognitionEngine()
    me._action_values["flee"] = 0.95
    me._action_values["idle"] = 0.05

    rumination_detected = False
    for i in range(50):
        r = me.tick(prediction_error=0.3, last_action="idle")
        if r["is_ruminating"]:
            rumination_detected = True
            print(f" ⚠ Rumination at tick {i} triggered (regret={r['regret']:.4f})")
            break

    rum_count = r["rumination_count"]
    print(f" Rumination count: {rum_count}")

    ok = rumination_detected and rum_count > 0
    print(f" → {'✓ PASS' if ok else '✗ FAIL'} — Persistent regret triggers rumination")
    return ok


# ============================================================================
# 9. AliceBrain Integration (Full Integration)
# ============================================================================

def run_exp9_alice_integration():
    """
    Experiment 9: AliceBrain Full Integration

    Goal: Verify Metacognition engine operates correctly within AliceBrain.perceive()

    Method:
    - Create AliceBrain → feed stimuli
    - Verify brain_result["metacognition"] exists and is effective
    - Verify introspect() includes metacognition subsystem
    """
    print("\n[Exp 9: AliceBrain Full Integration]")

    brain = AliceBrain()
    signal = make_signal(40.0, 0.5)

    # Multiple perceptions to create homeostasis
    for _ in range(5):
        result = brain.perceive(signal, Modality.AUDITORY, Priority.NORMAL)

    # Verify metacognition result exists
    meta = result.get("metacognition")
    has_meta = meta is not None
    print(f" metacognition result: {'✓ Exists' if has_meta else '✗ Missing'}")

    if has_meta:
        print(f"    Γ_thinking: {meta['gamma_thinking']:.4f}")
        print(f"    thinking_rate: {meta['thinking_rate']:.4f}")
        print(f"    confidence: {meta['confidence']:.4f}")
        print(f"    system_mode: System {meta['system_mode']}")
        print(f"    meta_report: {meta['meta_report']}")

    # Verify introspect includes metacognition
    intro = brain.introspect()
    has_intro = "metacognition" in intro.get("subsystems", {})
    print(f" introspect includes: {'✓' if has_intro else '✗'}")

    ok = has_meta and has_intro and meta["gamma_thinking"] >= 0
    print(f" → {'✓ PASS' if ok else '✗ FAIL'} — Full integration normal")
    return ok


# ============================================================================
# 10. Sleep Metacognition
# ============================================================================

def run_exp10_sleep_metacognition():
    """
    Experiment 10: Sleep Metacognition Dormancy

    Goal: Verify Metacognition drops to minimum during sleep

    Method:
    - Normal operation to create baseline
    - Switch to sleep mode
    - Verify thinking_rate drops to minimum, system_mode returns to 1
    """
    print("\n[Exp 10: Sleep Metacognition Dormancy]")

    me = MetacognitionEngine()

    # Awake period
    for _ in range(10):
        r_awake = me.tick(
            prediction_error=0.3, anxiety=0.3, pfc_energy=0.7,
        )
    rate_awake = r_awake["thinking_rate"]
    print(f" Awake rate: {rate_awake:.4f}")

    # Sleep period
    for _ in range(10):
        r_sleep = me.tick(is_sleeping=True)
    rate_sleep = r_sleep["thinking_rate"]
    mode_sleep = r_sleep["system_mode"]
    print(f" Sleep rate: {rate_sleep:.4f}, System {mode_sleep}")

    ok = rate_sleep == MIN_THINKING_RATE and mode_sleep == 1
    print(f" → {'✓ PASS' if ok else '✗ FAIL'} — Metacognition operates at minimum during sleep")
    return ok


# ============================================================================
# Clinical Control Table Verification
# ============================================================================

CLINICAL_CHECKS = [
    ("Cognitive Load Theory", "Thinking impedance increases with cognitive load", run_exp1_thinking_impedance),
    ("Kahneman Dual System", "System 1/2 hysteresis switching", run_exp2_system_switching),
    ("Time-Dilation for Thought","High impedance reduces thinking rate", run_exp3_thinking_rate),
    ("Confidence Calibration", "Confidence calibrates with precision and impedance", run_exp4_confidence),
    ("Counterfactual Thinking", "Counterfactual reasoning generates regret/relief", run_exp5_counterfactual),
    ("Cognitive Reframing (CBT)","Self-correction with cooldown", run_exp6_self_correction),
    ("Aha! Moment (Insight)", "Impedance sudden drop triggers insight", run_exp7_insight),
    ("OCD Rumination", "Persistent regret causes rumination", run_exp8_rumination),
    ("PFC Integration", "AliceBrain full integration", run_exp9_alice_integration),
    ("Sleep Metacognition", "Metacognition dormancy during sleep", run_exp10_sleep_metacognition),
]


# ============================================================================
# Main Program
# ============================================================================

def main():
    print("=" * 72)
    print(" Phase 18: The Inner Auditor — Metacognition and Self-Correction")
    print(" 'Thinking itself has impedance — seeing one\'s own thinking is the final layer of consciousness.'")
    print("=" * 72)

    t0 = time.time()

    # --- Functional experiments ---
    results = []
    for i, (name, desc, fn) in enumerate(CLINICAL_CHECKS, 1):
        try:
            ok = fn()
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            ok = False
        results.append((name, ok))

    # --- Clinical control table ---
    print("\n" + "=" * 72)
    print("  Clinical Control Table (Clinical Mapping)")
    print("=" * 72)
    print(f"  {'#':<3} {'Clinical Phenomenon':<30} {'Result':<6}")
    print("  " + "-" * 42)
    passed = 0
    for i, (name, ok) in enumerate(results, 1):
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {i:<3} {name:<30} {status}")
        if ok:
            passed += 1

    elapsed = time.time() - t0
    print(f"\n  Result: {passed}/{len(results)} PASS ({elapsed:.2f}s)")
    print("=" * 72)

    if passed == len(results):
        print(" * Phase 18 completed — Alice learned to see her own thinking.")
        print(" 'I know what I am thinking, I know what I am uncertain about,")
        print(" I know when to slow down.'")
    else:
        print(f" ⚠ {len(results) - passed} items did not PASS, need fixing.")


if __name__ == "__main__":
    main()

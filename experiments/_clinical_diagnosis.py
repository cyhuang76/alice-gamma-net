"""
Clinical Diagnosis Script - Detailed medical reports for 3 edge-case experiments.
Uses the same API as the actual experiment files.
"""
import math
import sys
import numpy as np
from typing import Any, Dict, List

sys.path.insert(0, ".")

from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority


# =====================================================================
# Shared helpers (mirroring exp_stress_test.py and exp_human_intelligence_month.py)
# =====================================================================

NEURON_COUNT = 80  # same as exp_human_intelligence_month

def make_signal(freq: float = 40.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, 0.1, 64)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def run_tick(alice: AliceBrain, brightness: float = 0.5,
             noise: float = 0.2, freq: float = 40.0,
             amp: float = 0.5) -> Dict[str, Any]:
    """Same as exp_stress_test run_tick: hear() + see()"""
    visual = make_signal(freq, amp) * brightness
    audio = make_signal(freq * 0.5, amp * 0.3) * noise
    alice.hear(audio)
    result = alice.see(visual, priority=Priority.NORMAL)
    return result


def make_visual_stimulus(pattern_id: int = 0, noise: float = 0.1) -> np.ndarray:
    """Same as exp_human_intelligence_month make_visual_stimulus"""
    img = np.zeros((32, 32), dtype=np.float32)
    rng = np.random.RandomState(pattern_id + 1000)
    base = rng.rand(32, 32).astype(np.float32) * 0.8
    noise_arr = np.random.rand(32, 32).astype(np.float32) * noise
    return np.clip(base + noise_arr, 0.0, 1.0)


def safe_perceive(brain: AliceBrain, stim: np.ndarray,
                  modality=Modality.VISUAL,
                  priority=Priority.NORMAL) -> Dict[str, Any]:
    """Same as exp_human_intelligence_month safe_perceive"""
    result = brain.perceive(stim, modality=modality, priority=priority)
    return result


def vitals_dict(alice: AliceBrain) -> dict:
    v = alice.vitals
    return {
        "C": v.consciousness,
        "T": v.ram_temperature,
        "S": v.stability_index,
        "HR": v.heart_rate,
        "pain": v.pain_level,
        "sens": v.pain_sensitivity,
        "frozen": v.is_frozen(),
        "baseline_T": v.baseline_temperature,
        "trauma_count": len(v.trauma_imprints),
    }


# =====================================================================
# CASE 1: Sensory Bootstrapping (mirrors exp_human_intelligence_month Exp 1)
# =====================================================================
def case1_sensory_bootstrapping():
    print("=" * 70)
    print("CASE 1 - Sensory Bootstrapping: Neonatal Activation Failure")
    print("=" * 70)

    brain = AliceBrain(neuron_count=NEURON_COUNT)
    v0 = vitals_dict(brain)
    print(f"\n  [Initial State]")
    print(f"    Consciousness (Phi):  {v0['C']:.4f}")
    print(f"    Temperature:          {v0['T']:.4f}")
    print(f"    Stability:            {v0['S']:.4f}")
    print(f"    Heart Rate:           {v0['HR']:.1f} bpm")
    print(f"    Pain Level:           {v0['pain']:.4f}")
    print(f"    Is Frozen:            {v0['frozen']}")

    TICKS_PER_DAY = 300
    frozen_count = 0
    first_frozen = None
    last_unfrozen = None
    gammas_all: List[float] = []
    timeline = []

    for day in range(3):
        gammas_day: List[float] = []
        nan_count = 0
        for t in range(TICKS_PER_DAY):
            tick = day * TICKS_PER_DAY + t
            pattern_id = t % 5
            stim = make_visual_stimulus(pattern_id, noise=0.15)
            result = safe_perceive(brain, stim)
            v = vitals_dict(brain)
            status = result.get("status", "OK")

            if status == "FROZEN":
                frozen_count += 1
                if first_frozen is None:
                    first_frozen = tick
            else:
                last_unfrozen = tick

            g = result.get("perception", {}).get("gamma", None)
            if g is not None:
                if math.isnan(g):
                    nan_count += 1
                else:
                    gammas_day.append(g)
                    gammas_all.append(g)

            if tick % 50 == 0 or tick == 899:
                avg_g = np.mean(gammas_day[-50:]) if gammas_day else 0
                timeline.append((tick, status, v, avg_g))

        print(f"\n  Day {day+1}: collected {len(gammas_day)} gamma samples, "
              f"nan_count={nan_count}")

    print(f"\n  [Tick-by-Tick Timeline (every 50 ticks)]")
    print(f"    {'Tick':>6s}  {'Status':>7s}  {'Phi':>6s}  {'Temp':>6s}  "
          f"{'Stab':>6s}  {'HR':>6s}  {'avg_G':>6s}  Frozen")
    print(f"    {'---':>6s}  {'---':>7s}  {'---':>6s}  {'---':>6s}  "
          f"{'---':>6s}  {'---':>6s}  {'---':>6s}  {'---':>6s}")
    for tick, status, v, avg_g in timeline:
        print(f"    {tick:6d}  {status:>7s}  {v['C']:6.4f}  {v['T']:6.4f}  "
              f"{v['S']:6.4f}  {v['HR']:6.1f}  {avg_g:6.4f}  {v['frozen']}")

    vf = vitals_dict(brain)
    try:
        adapt = brain.impedance_adaptation.get_stats()
    except Exception:
        adapt = {}
    try:
        cons_state = brain.consciousness.get_state()
        is_conscious = brain.consciousness.is_conscious()
    except Exception:
        cons_state = "N/A"
        is_conscious = "N/A"

    print(f"\n  [Final Diagnosis]")
    print(f"    Total ticks:             900 (3 days x 300 ticks/day)")
    print(f"    Frozen ticks:            {frozen_count} / 900 ({frozen_count/900*100:.1f}%)")
    print(f"    First frozen at tick:    {first_frozen}")
    print(f"    Last non-frozen tick:    {last_unfrozen}")
    print(f"    Final consciousness:     {vf['C']:.4f}")
    print(f"    Final temperature:       {vf['T']:.4f}")
    print(f"    Final stability:         {vf['S']:.4f}")
    print(f"    Total gamma samples:     {len(gammas_all)}")
    if gammas_all:
        print(f"    Mean gamma:              {np.mean(gammas_all):.4f}")
        print(f"    Gamma std:               {np.std(gammas_all):.4f}")
    print(f"    Adaptation stats:        {adapt}")
    print(f"    Consciousness state:     {cons_state}")
    print(f"    Is conscious:            {is_conscious}")

    # Introspect
    try:
        intro = brain.introspect()
        print(f"\n  [Subsystem Status via introspect()]")
        print(f"    Brain state:             {intro.get('state')}")
        print(f"    Cycle count:             {intro.get('cycle_count')}")
        subs = intro.get("subsystems", {})
        for key in sorted(subs.keys())[:10]:
            sub = subs[key]
            if isinstance(sub, dict):
                snippet = {k: v for k, v in list(sub.items())[:4]}
                print(f"    {key}: {snippet}")
    except Exception as e:
        print(f"    introspect error: {e}")


# =====================================================================
# CASE 2: Rapid Oscillation Resilience (mirrors exp_stress_test Exp 6)
# =====================================================================
def case2_oscillation_resilience():
    print("\n" + "=" * 70)
    print("CASE 2 - Rapid Oscillation Resilience: PTSD Lock-In Under Chaos")
    print("=" * 70)

    alice = AliceBrain(neuron_count=60)
    phase_a_data = []
    phase_b_data = []

    print(f"\n  [Phase A: Rapid Oscillation - 200 ticks, calm<->crisis every 10 ticks]")
    print(f"    {'Tick':>6s}  {'Mode':>7s}  {'Phi':>6s}  {'Temp':>6s}  "
          f"{'Stab':>6s}  {'HR':>6s}  Frozen")
    print(f"    {'---':>6s}  {'---':>7s}  {'---':>6s}  {'---':>6s}  "
          f"{'---':>6s}  {'---':>6s}  {'---':>6s}")

    for tick in range(200):
        is_crisis = (tick // 10) % 2 == 1
        if is_crisis:
            result = run_tick(alice, brightness=0.95, noise=0.9,
                              freq=85.0, amp=0.9)
            if tick % 20 == 10:
                alice.inject_pain(0.4)
        else:
            result = run_tick(alice, brightness=0.2, noise=0.05,
                              freq=30.0, amp=0.2)

        v = vitals_dict(alice)
        phase_a_data.append(v)
        if tick % 20 == 0:
            mode = "CRISIS" if is_crisis else "calm"
            print(f"    {tick:6d}  {mode:>7s}  {v['C']:6.4f}  {v['T']:6.4f}  "
                  f"{v['S']:6.4f}  {v['HR']:6.1f}  {v['frozen']}")

    frozen_at_200 = alice.vitals.is_frozen()
    print(f"\n    State at tick 200: frozen={frozen_at_200}")

    if frozen_at_200:
        print(f"    >> Emergency reset executed (clinical intervention)")
        alice.emergency_reset()
        v_post = vitals_dict(alice)
        print(f"    >> Post-reset: C={v_post['C']:.4f} T={v_post['T']:.4f} "
              f"S={v_post['S']:.4f}")

    print(f"\n  [Phase B: Recovery - 200 ticks of calm stimulation]")
    print(f"    {'Tick':>6s}  {'Phi':>6s}  {'Temp':>6s}  {'Stab':>6s}  "
          f"{'HR':>6s}  Frozen")
    print(f"    {'---':>6s}  {'---':>6s}  {'---':>6s}  {'---':>6s}  "
          f"{'---':>6s}  {'---':>6s}")

    for tick in range(200, 400):
        result = run_tick(alice, brightness=0.15, noise=0.03,
                          freq=25.0, amp=0.15)
        v = vitals_dict(alice)
        phase_b_data.append(v)
        if tick % 25 == 0:
            print(f"    {tick:6d}  {v['C']:6.4f}  {v['T']:6.4f}  "
                  f"{v['S']:6.4f}  {v['HR']:6.1f}  {v['frozen']}")

    vf = vitals_dict(alice)

    # Analysis
    temps_a = [d["T"] for d in phase_a_data]
    temps_b = [d["T"] for d in phase_b_data]
    stabs_a = [d["S"] for d in phase_a_data]
    cons_a = [d["C"] for d in phase_a_data]
    cons_b = [d["C"] for d in phase_b_data]

    print(f"\n  [Oscillation Impact Analysis]")
    print(f"    Phase A temperature: mean={np.mean(temps_a):.4f} "
          f"max={max(temps_a):.4f} min={min(temps_a):.4f}")
    print(f"    Phase A consciousness: mean={np.mean(cons_a):.4f} "
          f"min={min(cons_a):.4f}")
    print(f"    Phase A stability: mean={np.mean(stabs_a):.4f} "
          f"min={min(stabs_a):.4f}")
    print(f"    Phase B temperature: mean={np.mean(temps_b):.4f} "
          f"std_last50={np.std(temps_b[-50:]):.4f}")
    print(f"    Phase B consciousness: mean={np.mean(cons_b):.4f} "
          f"final={cons_b[-1]:.4f}")
    print(f"    Phase B stability: mean={np.mean([d['S'] for d in phase_b_data]):.4f}")

    print(f"\n  [Final Diagnosis]")
    print(f"    Final frozen:            {vf['frozen']}")
    print(f"    Final consciousness:     {vf['C']:.4f}")
    print(f"    Final temperature:       {vf['T']:.4f}")
    print(f"    Final stability:         {vf['S']:.4f}")
    verdict = "RECOVERED" if not vf["frozen"] else \
              "STILL FROZEN (PTSD impedance-locked attractor active)"
    print(f"    Verdict:                 {verdict}")


# =====================================================================
# CASE 3: Trauma Cascade (mirrors exp_stress_test Exp 9)
# =====================================================================
def case3_trauma_cascade():
    print("\n" + "=" * 70)
    print("CASE 3 - Trauma Cascade: Irreversible Sensitization")
    print("=" * 70)

    alice = AliceBrain(neuron_count=60)
    v0 = vitals_dict(alice)
    print(f"\n  [Baseline - Pre-Trauma]")
    print(f"    Consciousness:       {v0['C']:.4f}")
    print(f"    Temperature:         {v0['T']:.4f}")
    print(f"    Pain Sensitivity:    {v0['sens']:.3f}")
    print(f"    Baseline Temp:       {v0['baseline_T']:.3f}")
    print(f"    Trauma Count:        {v0['trauma_count']}")
    print(f"    Heart Rate:          {v0['HR']:.1f} bpm")

    cycle_data = []
    for cycle in range(5):
        # Trauma phase: 10 ticks
        for t in range(10):
            alice.inject_pain(0.7)
            run_tick(alice, brightness=0.9, noise=0.9, amp=0.9)

        v_trauma = vitals_dict(alice)
        frozen_during = alice.vitals.is_frozen()

        if frozen_during:
            alice.emergency_reset()

        # Recovery phase: 100 ticks calm
        for t in range(100):
            run_tick(alice, brightness=0.2, noise=0.05, amp=0.1)

        v_recovery = vitals_dict(alice)

        cycle_data.append({
            "cycle": cycle + 1,
            "trauma": v_trauma,
            "frozen_during": frozen_during,
            "recovery": v_recovery,
        })

    print(f"\n  [Trauma-Recovery Cycle Log]")
    print(f"    {'Cycle':>5s}  {'T-Phi':>6s}  {'T-Temp':>6s}  {'T-Pain':>6s}  "
          f"{'Frzn':>5s}  {'R-Phi':>6s}  {'R-Temp':>6s}  "
          f"{'Sens':>6s}  {'BaseT':>6s}  {'Imprints':>8s}")
    print(f"    {'---':>5s}  {'---':>6s}  {'---':>6s}  {'---':>6s}  "
          f"{'---':>5s}  {'---':>6s}  {'---':>6s}  "
          f"{'---':>6s}  {'---':>6s}  {'---':>8s}")

    for d in cycle_data:
        t = d["trauma"]
        r = d["recovery"]
        fz = "YES" if d["frozen_during"] else "no"
        print(f"    {d['cycle']:5d}  {t['C']:6.4f}  {t['T']:6.4f}  "
              f"{t['pain']:6.4f}  {fz:>5s}  {r['C']:6.4f}  "
              f"{r['T']:6.4f}  {r['sens']:6.3f}  "
              f"{r['baseline_T']:6.3f}  {r['trauma_count']:8d}")

    # Final state
    vf = vitals_dict(alice)
    print(f"\n  [Final State After 5 Trauma Cycles]")
    print(f"    Consciousness:       {vf['C']:.4f} (threshold: > 0.15)")
    print(f"    Temperature:         {vf['T']:.4f}")
    print(f"    Pain Sensitivity:    {vf['sens']:.3f} (initial: 1.000, "
          f"delta: +{vf['sens']-1:.3f})")
    print(f"    Baseline Temp:       {vf['baseline_T']:.3f} (initial: 0.000, "
          f"delta: +{vf['baseline_T']:.3f})")
    print(f"    Heart Rate:          {vf['HR']:.1f} bpm")
    print(f"    Is Frozen:           {vf['frozen']}")
    print(f"    Trauma Imprints:     {vf['trauma_count']}")

    # Sensitization curve
    sens_vals = [1.0] + [d["recovery"]["sens"] for d in cycle_data]
    cons_vals = [v0["C"]] + [d["recovery"]["C"] for d in cycle_data]
    print(f"\n  [Sensitization Trajectory]")
    sens_str = " -> ".join(f"{s:.3f}" for s in sens_vals)
    cons_str = " -> ".join(f"{c:.4f}" for c in cons_vals)
    print(f"    Pain sensitivity:    {sens_str}")
    print(f"    Recovery Phi:        {cons_str}")

    # Clinical interpretation
    print(f"\n  [Clinical Interpretation]")
    recovered = sum(1 for d in cycle_data if d["recovery"]["C"] > 0.15)
    print(f"    Cycles with full recovery (Phi > 0.15):  {recovered} / 5")
    max_sens_cycle = next(
        (d["cycle"] for d in cycle_data if d["recovery"]["sens"] >= 2.0),
        "N/A")
    first_fail = next(
        (d["cycle"] for d in cycle_data if d["recovery"]["C"] <= 0.15),
        "N/A")
    print(f"    Sensitization reached 2.0 at:     Cycle {max_sens_cycle}")
    print(f"    First non-recovery cycle:         Cycle {first_fail}")
    if vf["frozen"]:
        print(f"    Prognosis:  System locked in impedance attractor.")
        print(f"                Consciousness {vf['C']:.4f} < 0.15 threshold.")
        print(f"                Standard recovery (100 calm ticks) insufficient.")
        print(f"                Requires pharmacological intervention or "
              f"extended dream therapy.")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    case1_sensory_bootstrapping()
    case2_oscillation_resilience()
    case3_trauma_cascade()
    print("\n" + "=" * 70)
    print("Clinical Diagnosis Complete")
    print("=" * 70)

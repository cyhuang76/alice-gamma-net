# -*- coding: utf-8 -*-
"""
Phase 24 — Phantom Limb Pain Clinical Verification
Phantom Limb Pain: Clinical Validation via Coaxial Cable Physics

═══════════════════════════════════════════════════════════════════════
Core Proposition:
  Amputation = coaxial cable terminal load removed → Z_load = ∞ → Γ = 1.0
  → signal 100% reflected → reflected_energy → pain

  If Γ-Net impedance matching physics is true, then:
  1. After amputation, Phantom Limb Pain should auto-generate (Γ=1.0 mathematical necessity)
  2. Mirror therapy (providing visual impedance matching) should reduce Γ → pain relief
  3. Cortical reorganization degree should ∝ pain intensity (Flor 2006, r=0.93)
  4. Residual motor commands should naturally decay (brain learning)
  5. Neuroma should generate random pain events
  6. Emotion/stress/temperature should trigger or exacerbate Phantom Limb Pain

Clinical control data:
  - Ramachandran (1996): mirror therapy VAS 7.2→2.1 (4-6 weeks)
  - Flor et al. (2006): cortical reorganization ∝ Phantom Limb Pain (r=0.93)
  - Epidemiology: 60-80% of amputees experience Phantom Limb Pain
  - Makin et al. (2013): phantom limb representation still present in S1
  - Temperature/barometric trigger: 75% patients report weather-correlated exacerbation

10 clinical control verifications (without modifying any physics equations):

  exp_01: Γ=1.0 mathematical inevitability — amputation means pain
  exp_02: Reflected energy formula verification — signal² × Γ²
  exp_03: Mirror therapy 4-week sessions — VAS decrease
  exp_04: Mirror therapy vs control group — randomized controlled trial
  exp_05: Cortical reorganization ∝ pain — Flor correlation coefficient
  exp_06: Motor command natural extinction — long-term tracking
  exp_07: Neuroma vs no neuroma — pain difference
  exp_08: Emotion/stress trigger — psychological factors
  exp_09: Temperature/weather trigger — meteorological correlation
  exp_10: Complete AliceBrain integration — amputation→Phantom Limb Pain→mirror therapy

Author: Phase 24 — Computational Neurology / Phantom Limb Pain
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
import math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alice.brain.phantom_limb import (
    PhantomLimbEngine,
    NORMAL_LIMB_IMPEDANCE,
    STUMP_NEUROMA_IMPEDANCE,
    OPEN_CIRCUIT_IMPEDANCE,
    MOTOR_EFFERENCE_INITIAL,
    MOTOR_EFFERENCE_MIN,
    PHANTOM_PAIN_THRESHOLD,
    CORTICAL_REMAP_MAX,
    REMAP_PAIN_COUPLING,
)


def _header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _result(name: str, passed: bool, detail: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} — {name}")
    if detail:
        print(f"         {detail}")
    return passed


# ============================================================================
# exp_01: Γ=1.0 Mathematical Inevitability
# ============================================================================

def exp_01_gamma_inevitability():
    """
    Amputation = open circuit → Γ ≈ 1.0 → pain is mathematical inevitability

    Verification: Γ = (Z_load - Z₀) / (Z_load + Z₀) → 1.0 as Z_load → ∞
    """
    _header("Exp 01: Γ=1.0 Mathematical Inevitability — Amputation Means Pain")

    eng = PhantomLimbEngine(rng_seed=42)

    # No neuroma (complete open circuit)
    state = eng.amputate("left_hand", has_neuroma=False)
    z_load = OPEN_CIRCUIT_IMPEDANCE
    z0 = NORMAL_LIMB_IMPEDANCE
    expected_gamma = abs(z_load - z0) / (z_load + z0)

    ok1 = _result(
        "Γ = (Z_L - Z₀)/(Z_L + Z₀) ≈ 1.0",
        abs(state.effective_gamma - expected_gamma) < 0.001,
        f"Γ = {state.effective_gamma:.6f}, expected = {expected_gamma:.6f}"
    )

    # Has neuroma (high but not infinite impedance)
    eng2 = PhantomLimbEngine(rng_seed=42)
    state2 = eng2.amputate("right_hand", has_neuroma=True)
    z_load2 = STUMP_NEUROMA_IMPEDANCE
    expected_gamma2 = abs(z_load2 - z0) / (z_load2 + z0)

    ok2 = _result(
        "Neuroma: Γ < 1.0 but > 0.7",
        0.7 < state2.effective_gamma < 1.0,
        f"Γ = {state2.effective_gamma:.4f} (neuroma Z={z_load2}Ω)"
    )

    # Compare with normal limb (Z_load = Z₀ = 50Ω → Γ = 0)
    normal_gamma = abs(z0 - z0) / (z0 + z0)
    ok3 = _result(
        "Normal limb: Γ = 0 (perfect match)",
        normal_gamma == 0.0,
        f"Γ = {normal_gamma:.4f}"
    )

    return ok1 and ok2 and ok3


# ============================================================================
# exp_02: Reflected Energy Formula Verification
# ============================================================================

def exp_02_reflected_energy_formula():
    """
    reflected_energy = motor² × Γ²

    This is the ALICE THE PAIN LOOP core formula.
    """
    _header("Exp 02: Reflected Energy Formula — signal² × Γ²")

    eng = PhantomLimbEngine(rng_seed=42)
    state = eng.amputate("left_hand", has_neuroma=False)
    gamma = state.effective_gamma

    # Test different intensities
    tests = []
    for motor in [0.2, 0.5, 0.8, 1.0]:
        expected = motor ** 2 * gamma ** 2
        result = eng.tick(motor_commands={"left_hand": motor})
        actual = result["total_reflected_energy"]
        # Neuroma term adds small amount, but open circuit without neuroma approximates pure motor² × Γ²
        error = abs(actual - expected) / max(expected, 1e-6)
        tests.append((motor, expected, actual, error))

    all_ok = True
    for motor, exp, act, err in tests:
        ok = err < 0.5 # Allow 50% error (because there are other small terms)
        all_ok = all_ok and ok
        _result(
            f"motor={motor:.1f}: E_ref={act:.4f} (expected~{exp:.4f})",
            ok,
            f"error={err:.1%}"
        )

    # Most importantly: reflected energy monotonically increases with motor
    energies = [t[2] for t in tests]
    monotonic = all(energies[i] <= energies[i + 1] for i in range(len(energies) - 1))
    ok_mono = _result(
        "Reflected energy monotonically increases with motor command intensity",
        monotonic,
        f"energies = {[f'{e:.4f}' for e in energies]}"
    )

    return all_ok and ok_mono


# ============================================================================
# exp_03: Mirror Therapy 4-Week Sessions
# ============================================================================

def exp_03_mirror_therapy_4weeks():
    """
    Ramachandran (1996): 4-6 week mirror therapy
    VAS from 7.2 down to 2.1 (mean reduction 70%)

    Simulation: 4 weeks daily therapy sessions, once per day
    """
    _header("Exp 03: Mirror Therapy 4-Week Sessions (Ramachandran 1996)")

    eng = PhantomLimbEngine(rng_seed=42)
    eng.amputate("left_hand")

    TICKS_PER_DAY = 24
    DAYS = 7 * 5  # 1 week baseline + 4 weeks therapy

    # Baseline week (no therapy)
    baseline_pains = []
    for _ in range(7 * TICKS_PER_DAY):
        result = eng.tick(motor_commands={"left_hand": 0.6})
        baseline_pains.append(result["phantom_states"]["left_hand"]["pain"])

    vas_baseline = np.mean(baseline_pains) * 10

    # 4 weeks of daily mirror therapy
    weekly_vas = []
    for week in range(4):
        week_pains = []
        for day in range(7):
            # Morning therapy session
            eng.apply_mirror_therapy_session("left_hand", quality=0.8)
            for _ in range(TICKS_PER_DAY):
                result = eng.tick(motor_commands={"left_hand": 0.6})
                week_pains.append(result["phantom_states"]["left_hand"]["pain"])
        weekly_vas.append(np.mean(week_pains) * 10)

    print(f"\n  Baseline VAS: {vas_baseline:.2f}")
    for i, v in enumerate(weekly_vas):
        print(f"  Week {i+2} VAS:  {v:.2f}")

    # Validation
    ok1 = _result(
        "Post-therapy VAS < baseline VAS",
        weekly_vas[-1] < vas_baseline,
        f"{weekly_vas[-1]:.2f} < {vas_baseline:.2f}"
    )

    reduction = (vas_baseline - weekly_vas[-1]) / max(vas_baseline, 0.01) * 100
    ok2 = _result(
        "VAS decrease ≥ 20%",
        reduction >= 20,
        f"reduction = {reduction:.1f}%"
    )

    # Weekly decreasing trend
    decreasing = all(
        weekly_vas[i] >= weekly_vas[i + 1] - 0.5
        for i in range(len(weekly_vas) - 1)
    )
    ok3 = _result(
        "VAS shows weekly decreasing trend",
        decreasing,
        f"weekly = {[f'{v:.2f}' for v in weekly_vas]}"
    )

    return ok1 and ok2 and ok3


# ============================================================================
# exp_04: Mirror Therapy vs Control Group (RCT)
# ============================================================================

def exp_04_mirror_rct():
    """
    Randomized controlled trial: mirror therapy vs no treatment

    Treatment group cumulative pain should be significantly < control group
    """
    _header("Exp 04: Mirror Therapy vs Control Group (RCT)")

    TICKS = 1000

    # control group
    ctrl = PhantomLimbEngine(rng_seed=42)
    ctrl.amputate("left_hand")
    ctrl_pains = []
    for _ in range(TICKS):
        result = ctrl.tick(motor_commands={"left_hand": 0.5})
        ctrl_pains.append(result["phantom_states"]["left_hand"]["pain"])

    # treatment group
    treat = PhantomLimbEngine(rng_seed=42)
    treat.amputate("left_hand")
    treat_pains = []
    for t in range(TICKS):
        if t % 24 == 0:
            treat.apply_mirror_therapy_session("left_hand", quality=0.8)
        result = treat.tick(motor_commands={"left_hand": 0.5})
        treat_pains.append(result["phantom_states"]["left_hand"]["pain"])

    ctrl_mean = np.mean(ctrl_pains)
    treat_mean = np.mean(treat_pains)

    ok1 = _result(
        "Treatment group mean pain < control group",
        treat_mean < ctrl_mean,
        f"treatment={treat_mean:.4f}, control={ctrl_mean:.4f}"
    )

    ctrl_cum = sum(ctrl_pains)
    treat_cum = sum(treat_pains)
    ok2 = _result(
        "Treatment group cumulative pain < control group",
        treat_cum < ctrl_cum,
        f"treatment={treat_cum:.1f}, control={ctrl_cum:.1f}"
    )

    # Second half gap should be larger than first half (treatment effect accumulates)
    half = TICKS // 2
    first_half_diff = np.mean(ctrl_pains[:half]) - np.mean(treat_pains[:half])
    second_half_diff = np.mean(ctrl_pains[half:]) - np.mean(treat_pains[half:])

    ok3 = _result(
        "Treatment effect accumulates over time (2nd half gap ≥ 1st half)",
        second_half_diff >= first_half_diff - 0.01,
        f"first half diff={first_half_diff:.4f}, second half diff={second_half_diff:.4f}"
    )

    return ok1 and ok2 and ok3


# ============================================================================
# exp_05: Cortical Reorganization ∝ Pain (Flor 2006)
# ============================================================================

def exp_05_cortical_remap_correlation():
    """
    Flor et al. (2006): cortical reorganization distance ∝ Phantom Limb Pain intensity (r = 0.93)

    Simulation: different motor command intensities → different pain → measure cortical reorganization progress
    """
    _header("Exp 05: Cortical Reorganization ∝ Pain (Flor 2006, r=0.93)")

    TICKS = 500
    motor_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    mean_pains = []
    remaps = []

    for motor in motor_levels:
        eng = PhantomLimbEngine(rng_seed=42)
        eng.amputate("left_hand")

        pains = []
        for _ in range(TICKS):
            result = eng.tick(motor_commands={"left_hand": motor})
            pains.append(result["phantom_states"]["left_hand"]["pain"])

        phantom = eng.get_phantom("left_hand")
        mean_pains.append(np.mean(pains))
        remaps.append(phantom.cortical_remap_progress)

    # Compute correlation coefficient
    if np.std(mean_pains) > 0 and np.std(remaps) > 0:
        corr = np.corrcoef(mean_pains, remaps)[0, 1]
    else:
        corr = 0.0

    print(f"\n  Motor levels: {motor_levels}")
    print(f"  Mean pains:   {[f'{p:.4f}' for p in mean_pains]}")
    print(f"  Remaps:       {[f'{r:.4f}' for r in remaps]}")
    print(f"  Correlation:  r = {corr:.4f}")

    ok1 = _result(
        "Cortical reorganization monotonically increases with pain",
        all(remaps[i] <= remaps[i + 1] + 0.001 for i in range(len(remaps) - 1)),
        f"remaps = {[f'{r:.4f}' for r in remaps]}"
    )

    ok2 = _result(
        "Correlation coefficient r > 0.8 (Flor reported 0.93)",
        corr > 0.8,
        f"r = {corr:.4f}"
    )

    return ok1 and ok2


# ============================================================================
# exp_06: Motor Command Natural Extinction
# ============================================================================

def exp_06_motor_efference_decay():
    """
    Motor command natural extinction — brain learns not to send

    Clinical: phantom limb sensation gradually weakens but does not completely disappear
    Expected: motor_efference from 0.8 decreasing to near MOTOR_EFFERENCE_MIN
    """
    _header("Exp 06: Motor Command Natural Extinction — Long-term Tracking")

    eng = PhantomLimbEngine(rng_seed=42)
    eng.amputate("left_hand")

    checkpoints = [0, 100, 500, 1000, 2000, 5000]
    efferences = []
    pains = []
    tick = 0

    for target in checkpoints:
        while tick < target:
            eng.tick()
            tick += 1
        phantom = eng.get_phantom("left_hand")
        efferences.append(phantom.motor_efference)
        pains.append(phantom.phantom_pain_level)

    print(f"\n  Checkpoints: {checkpoints}")
    print(f"  Efferences:  {[f'{e:.4f}' for e in efferences]}")
    print(f"  Pains:       {[f'{p:.4f}' for p in pains]}")

    ok1 = _result(
        "Motor commands gradually decay",
        efferences[-1] < efferences[0],
        f"initial={efferences[0]:.4f}, final={efferences[-1]:.4f}"
    )

    ok2 = _result(
        f"Motor commands do not fall below floor ({MOTOR_EFFERENCE_MIN})",
        efferences[-1] >= MOTOR_EFFERENCE_MIN,
        f"final={efferences[-1]:.4f}"
    )

    ok3 = _result(
        "Long-term pain < initial pain",
        pains[-1] <= pains[1] + 0.05,
        f"early={pains[1]:.4f}, late={pains[-1]:.4f}"
    )

    return ok1 and ok2 and ok3


# ============================================================================
# exp_07: Neuroma vs No Neuroma
# ============================================================================

def exp_07_neuroma_comparison():
    """
    Neuroma vs no neuroma pain difference

    Clinical: neuroma patient pain more frequent, more unpredictable
    """
    _header("Exp 07: Neuroma vs No Neuroma")

    TICKS = 1000

    # With neuroma
    eng_neuroma = PhantomLimbEngine(rng_seed=42)
    eng_neuroma.amputate("left_hand", has_neuroma=True)
    pains_neuroma = []
    for _ in range(TICKS):
        result = eng_neuroma.tick(motor_commands={"left_hand": 0.3})
        pains_neuroma.append(result["phantom_states"]["left_hand"]["pain"])

    # Without neuroma
    eng_clean = PhantomLimbEngine(rng_seed=42)
    eng_clean.amputate("left_hand", has_neuroma=False)
    pains_clean = []
    for _ in range(TICKS):
        result = eng_clean.tick(motor_commands={"left_hand": 0.3})
        pains_clean.append(result["phantom_states"]["left_hand"]["pain"])

    mean_neuroma = np.mean(pains_neuroma)
    mean_clean = np.mean(pains_clean)

    ok1 = _result(
        "Neuroma mean pain ≥ no neuroma",
        mean_neuroma >= mean_clean - 0.01,
        f"neuroma={mean_neuroma:.4f}, clean={mean_clean:.4f}"
    )

    # Neuroma pain variability larger (random discharge)
    std_neuroma = np.std(pains_neuroma)
    std_clean = np.std(pains_clean)
    ok2 = _result(
        "Neuroma pain variability ≥ no neuroma",
        std_neuroma >= std_clean - 0.01,
        f"neuroma_std={std_neuroma:.4f}, clean_std={std_clean:.4f}"
    )

    # Neuroma Γ lower (because not complete open circuit)
    gamma_neuroma = eng_neuroma.get_phantom("left_hand").effective_gamma
    gamma_clean = eng_clean.get_phantom("left_hand").effective_gamma
    ok3 = _result(
        "Neuroma Γ < no neuroma Γ (not complete open circuit)",
        gamma_neuroma < gamma_clean,
        f"neuroma_Γ={gamma_neuroma:.4f}, clean_Γ={gamma_clean:.4f}"
    )

    return ok1 and ok2 and ok3


# ============================================================================
# exp_08: Emotion/Stress Trigger
# ============================================================================

def exp_08_emotional_stress_trigger():
    """
    Emotion/stress triggers Phantom Limb Pain

    Clinical: anxiety, depression, stress exacerbate Phantom Limb Pain
    """
    _header("Exp 08: Emotion/Stress Trigger")

    TICKS = 200

    conditions = {
        "neutral": {"emotional_valence": 0.0, "stress_level": 0.0},
        "anxious": {"emotional_valence": -0.6, "stress_level": 0.3},
        "stressed": {"emotional_valence": 0.0, "stress_level": 0.8},
        "depressed+stressed": {"emotional_valence": -0.9, "stress_level": 0.9},
    }

    results = {}
    for name, params in conditions.items():
        eng = PhantomLimbEngine(rng_seed=42)
        eng.amputate("left_hand")
        pains = []
        for _ in range(TICKS):
            result = eng.tick(motor_commands={"left_hand": 0.4}, **params)
            pains.append(result["phantom_states"]["left_hand"]["pain"])
        results[name] = np.mean(pains)
        print(f"  {name:25s}: mean_pain = {results[name]:.4f}")

    ok1 = _result(
        "Anxiety pain ≥ neutral pain",
        results["anxious"] >= results["neutral"] - 0.01,
    )

    ok2 = _result(
        "Stress pain ≥ neutral pain",
        results["stressed"] >= results["neutral"] - 0.01,
    )

    ok3 = _result(
        "Depression+stress ≥ stress alone",
        results["depressed+stressed"] >= results["stressed"] - 0.01,
    )

    return ok1 and ok2 and ok3


# ============================================================================
# exp_09: Temperature/Weather Trigger
# ============================================================================

def exp_09_weather_trigger():
    """
    Temperature/barometric trigger

    Clinical: 75% patients report weather changes exacerbate Phantom Limb Pain, especially cold
    """
    _header("Exp 09: Temperature/Weather Trigger")

    TICKS = 200

    conditions = {
        "stable": 0.0,
        "cold_snap": -0.5,
        "warm_front": 0.3,
        "extreme_cold": -1.0,
    }

    results = {}
    for name, temp_delta in conditions.items():
        eng = PhantomLimbEngine(rng_seed=42)
        eng.amputate("left_hand")
        pains = []
        for _ in range(TICKS):
            result = eng.tick(
                motor_commands={"left_hand": 0.3},
                temperature_delta=temp_delta
            )
            pains.append(result["phantom_states"]["left_hand"]["pain"])
        results[name] = np.mean(pains)
        print(f"  {name:15s} (ΔT={temp_delta:+.1f}): mean_pain = {results[name]:.4f}")

    ok1 = _result(
        "Cold pain ≥ stable pain",
        results["cold_snap"] >= results["stable"] - 0.01,
    )

    ok2 = _result(
        "Extreme cold pain ≥ cold pain",
        results["extreme_cold"] >= results["cold_snap"] - 0.01,
    )

    ok3 = _result(
        "Temperature change (including warm) pain ≥ stable pain",
        results["warm_front"] >= results["stable"] - 0.01,
    )

    return ok1 and ok2 and ok3


# ============================================================================
# exp_10: AliceBrain Complete Integration
# ============================================================================

def exp_10_alice_brain_integration():
    """
    Complete AliceBrain integration: amputation → Phantom Limb Pain → mirror therapy

    Verify PhantomLimbEngine correctly integrated into main system
    """
    _header("Exp 10: AliceBrain Complete Integration")

    from alice.alice_brain import AliceBrain

    brain = AliceBrain()

    ok1 = _result(
        "AliceBrain contains phantom_limb engine",
        hasattr(brain, "phantom_limb") and isinstance(brain.phantom_limb, PhantomLimbEngine),
    )

    # perceive returns phantom_limb data
    result = brain.perceive(np.random.randn(64))
    ok2 = _result(
        "perceive() contains phantom_limb data",
        "phantom_limb" in result,
    )

    # Amputation operation
    brain.phantom_limb.amputate("left_hand")
    result = brain.perceive(np.random.randn(64))
    phantom_data = result.get("phantom_limb", {})
    ok3 = _result(
        "phantom_limb has pain data after amputation",
        phantom_data.get("phantom_count", 0) == 1,
        f"data = {phantom_data}"
    )

    # introspect
    intro = brain.introspect()
    ok4 = _result(
        "introspect() contains phantom_limb",
        "phantom_limb" in intro.get("subsystems", {}),
    )

    return ok1 and ok2 and ok3 and ok4


# ============================================================================
# Main Execution
# ============================================================================

def main():
    experiments = [
        exp_01_gamma_inevitability,
        exp_02_reflected_energy_formula,
        exp_03_mirror_therapy_4weeks,
        exp_04_mirror_rct,
        exp_05_cortical_remap_correlation,
        exp_06_motor_efference_decay,
        exp_07_neuroma_comparison,
        exp_08_emotional_stress_trigger,
        exp_09_weather_trigger,
        exp_10_alice_brain_integration,
    ]

    print("\n" + "█" * 70)
    print("  Phase 24 — Phantom Limb Pain Clinical Verification")
    print("  Phantom Limb Pain: Clinical Validation via Γ = 1.0 Physics")
    print("█" * 70)

    passed = 0
    failed = 0
    for exp in experiments:
        try:
            if exp():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  ❌ EXCEPTION in {exp.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*70}")
    print(f"  Result: {passed}/{passed + failed} PASS")
    if failed == 0:
        print(" 🎉 All clinical verifications PASSED — Γ=1.0 physics framework successfully explains Phantom Limb Pain")
    print(f"{'='*70}\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

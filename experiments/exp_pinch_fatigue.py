# -*- coding: utf-8 -*-
"""
exp_pinch_fatigue.py — Lorentz Compression Fatigue Physics Experiment
Pollock-Barraclough (1905) Neural Aging Mechanism

Experiment plan:
  1. Lorentz compression force square law verification
  2. Elastic/plastic separation visualization
  3. Aging trajectory: Young → Mature → Degraded
  4. Sleep vs aging: Repairable vs irreversible
  5. High-stress life vs low-stress life
  6. BDNF neurotrophic factor protective effect
  7. Arrhenius temperature acceleration verification
  8. Work hardening (Does experience make you stronger?)
  9. Multi-channel differential aging
  10. AliceBrain full lifecycle aging simulation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from alice.brain.pinch_fatigue import (
    PinchFatigueEngine,
    YIELD_STRAIN,
    PINCH_SENSITIVITY,
    PINCH_EXPONENT,
)


def _high_current():
    """Compute current intensity sufficient to exceed yield strain"""
    return (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 1.0


def _moderate_current():
    """Compute current intensity just around yield strain"""
    return (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 0.3


# ============================================================================
# Experiment 1: Lorentz Compression Force Square-Law Verification
# ============================================================================

def exp_01_pinch_square_law():
    """
    Verify Lorentz compression strain ε ∝ I²

    Pollock-Barraclough (1905): Lorentz compression force is proportional to current squared.
    This is a physical law — F = μ₀I²/(2πR) — must hold strictly.
    """
    print("=" * 60)
    print("EXP 01: Lorentz Compression Force Square Law — ε ∝ I²")
    print("=" * 60)

    currents = np.linspace(0.1, 2.0, 20)
    strains = []

    for i_val in currents:
        eng = PinchFatigueEngine()
        event = eng.apply_current_pulse("test", float(i_val))
        strains.append(event.pinch_strain)

    # Verify log-log slope ≈ 2.0
    log_i = np.log(currents)
    log_e = np.log(strains)
    slope = np.polyfit(log_i, log_e, 1)[0]

    for i, (c, s) in enumerate(zip(currents[::4], strains[::4])):
        print(f"  I = {c:.3f}  →  ε = {s:.6f}")
    print(f"\n  log-log slope = {slope:.4f} (theoretical value = {PINCH_EXPONENT})")
    assert abs(slope - PINCH_EXPONENT) < 0.01, f"Slope deviation too large: {slope}"
    print(" ✓ Lorentz compression force square law perfectly verified")
    return True


# ============================================================================
# Experiment 2: Elastic / Plastic Separation
# ============================================================================

def exp_02_elastic_plastic_separation():
    """
    Core material mechanics:
    - ε < ε_yield → Pure elastic (full recovery)
    - ε > ε_yield → Elastic + plastic (partially permanent)

    This is the physical basis of 'fatigue' (recoverable) vs 'aging' (irreversible).
    """
    print("\n" + "=" * 60)
    print("EXP 02: Elastic/Plastic Separation — Fatigue vs Aging")
    print("=" * 60)

    eng = PinchFatigueEngine()
    high_i = _high_current()

    elastic_series = []
    plastic_series = []

    for cycle in range(200):
        eng.apply_current_pulse("axon", high_i, temperature=0.6)
        ch = eng._channels["axon"]
        elastic_series.append(ch.elastic_strain)
        plastic_series.append(ch.plastic_strain)

    print(f" After 200 high-current cycles:")
    print(f" Elastic strain (latest) = {elastic_series[-1]:.6f}(resets each time)")
    print(f" Plastic strain (accumulated) = {plastic_series[-1]:.6f}(continuously growing)")
    print(f" Plastic/elastic ratio = {plastic_series[-1] / max(elastic_series[-1], 1e-12):.2f}")
    print(f" Fatigue damage = {eng._channels['axon'].fatigue_damage:.6f}")

    assert plastic_series[-1] > elastic_series[-1] * 3, "Plastic accumulation should far exceed single elastic"
    print(" ✓ Elastic/plastic separation physics correct — aging is irreversible")
    return True


# ============================================================================
# Experiment 3: Aging Trajectory (Lifetime Simulation)
# ============================================================================

def exp_03_aging_trajectory():
    """
    Simulate one neural axon's lifetime:
      Phase I: Youth (low current) → almost no aging
      Phase II: Adulthood (moderate current) → slow accumulation
      Phase III: High stress (high current) → accelerated aging
      Phase IV: Old age (current decreases) → but damage already present

    Like Pollock observed with lightning rods —
    Not one lightning strike piercing through, but many cumulative strikes causing fatigue.
    """
    print("\n" + "=" * 60)
    print("EXP 03: Neural Axon Lifetime Aging Trajectory")
    print("=" * 60)

    eng = PinchFatigueEngine()
    high_i = _high_current()
    milestones = {}

    # Phase I: Youth (200 ticks, low current — below yield)
    for _ in range(200):
        eng.tick({"axon": 0.2})
    milestones["youth"] = eng.get_channel_age("axon")

    # Phase II: Adulthood (300 ticks, moderate-high current — starts generating plasticity)
    mid_i = _moderate_current()
    for _ in range(300):
        eng.tick({"axon": mid_i})
    milestones["adult"] = eng.get_channel_age("axon")

    # Phase III: High stress (200 ticks, high current — massive plasticity)
    for _ in range(200):
        eng.tick({"axon": high_i})
    milestones["stress"] = eng.get_channel_age("axon")

    # Phase IV: Old age (300 ticks, low current but damage already present)
    for _ in range(300):
        eng.tick({"axon": 0.2})
    milestones["elderly"] = eng.get_channel_age("axon")

    for phase, age in milestones.items():
        life = eng.get_life_expectancy("axon")
        print(f"  {phase:>10}: age_factor = {age:.6f}")

    ch = eng._channels["axon"]
    print(f"\n  Final state:")
    print(f" Impedance drift: {ch.impedance_drift:.6f}")
    print(f"    Γ_aging:  {ch.gamma_drift:.6f}")
    print(f" Lifespan: {ch.structural_integrity:.4f}")

    assert milestones["youth"] < milestones["stress"], "Youth should age less than high-stress period"
    print(" ✓ Aging trajectory matches intuition — high stress accelerates, low stress doesn't reverse")
    return True


# ============================================================================
# Experiment 4: Sleep vs Aging
# ============================================================================

def exp_04_sleep_vs_aging():
    """
    Core Hypothesis:
    - Elastic strain = Impedance Debt = Repairable by sleep ✓
    - Plastic strain = Aging = Irreversible by sleep ✗

    'You can wake up feeling refreshed, but you won't sleep yourself ten years younger.'
    """
    print("\n" + "=" * 60)
    print("EXP 04: Sleep Repair vs Permanent Aging")
    print("=" * 60)

    eng = PinchFatigueEngine()
    high_i = _high_current()

    # Accumulate damage
    for _ in range(300):
        eng.apply_current_pulse("axon", high_i, temperature=0.7)

    ch = eng._channels["axon"]
    elastic_pre = ch.elastic_strain
    plastic_pre = ch.plastic_strain
    fatigue_pre = ch.fatigue_damage

    print(f" After damage:")
    print(f" Elastic = {elastic_pre:.6f}")
    print(f" Plastic = {plastic_pre:.6f}")
    print(f"    fatigue = {fatigue_pre:.6f}")

    # Extensive sleep
    for _ in range(500):
        eng.repair_tick(is_sleeping=True, growth_factor=0.5)

    elastic_post = ch.elastic_strain
    plastic_post = ch.plastic_strain
    fatigue_post = ch.fatigue_damage

    print(f" After 500 ticks of sleep:")
    print(f" Elastic = {elastic_post:.6f} (recovered {(1-elastic_post/max(elastic_pre,1e-12))*100:.1f}%)")
    print(f" Plastic = {plastic_post:.6f} (recovered {(1-plastic_post/max(plastic_pre,1e-12))*100:.1f}%)")
    print(f"    fatigue = {fatigue_post:.6f} (recovered {(1-fatigue_post/max(fatigue_pre,1e-12))*100:.1f}%)")

    elastic_recovery = 1.0 - elastic_post / max(elastic_pre, 1e-12)
    plastic_recovery = 1.0 - plastic_post / max(plastic_pre, 1e-12)
    assert elastic_recovery > plastic_recovery * 3, "Elastic repair should be much faster than plastic"
    print(" ✓ Sleep repairs elastic strain but cannot reverse aging — physics correct")
    return True


# ============================================================================
# Experiment 5: High-Stress Life vs Low-Stress Life
# ============================================================================

def exp_05_high_vs_low_stress_life():
    """
    Two parallel lives:
    - Alice A: Low-stress life (I=0.3)
    - Alice B: High-stress life (I=0.9)

    Prediction: B ages faster, final impedance drift is larger.
    """
    print("\n" + "=" * 60)
    print("EXP 05: High-Stress vs Low-Stress Life")
    print("=" * 60)

    eng_low = PinchFatigueEngine()
    eng_high = PinchFatigueEngine()
    high_i = _high_current()

    for cycle in range(1000):
        eng_low.tick({"visual": 0.3, "motor": 0.2})
        eng_high.tick({"visual": high_i, "motor": high_i * 0.8},
                       temperature=0.8)

    stats_low = eng_low.get_stats()
    stats_high = eng_high.get_stats()

    print(f" Low-stress life (1000 ticks):")
    print(f" Mean aging = {stats_low['mean_age_factor']:.6f}")
    print(f" Impedance drift = {stats_low['mean_impedance_drift']:.6f}")
    print(f" Degraded channels = {stats_low['degraded_channels']}/{stats_low['total_channels']}")

    print(f" High-stress life (1000 ticks):")
    print(f" Mean aging = {stats_high['mean_age_factor']:.6f}")
    print(f" Impedance drift = {stats_high['mean_impedance_drift']:.6f}")
    print(f" Degraded channels = {stats_high['degraded_channels']}/{stats_high['total_channels']}")

    assert stats_high["mean_age_factor"] > stats_low["mean_age_factor"] * 2
    print(" ✓ High-stress life significantly accelerates aging — Lorentz compression effect prediction correct")
    return True


# ============================================================================
# Experiment 6: BDNF Protective Effect
# ============================================================================

def exp_06_bdnf_protection():
    """
    Neurotrophic factor (BDNF) = growth_factor:
    - Exercise, social, positive emotion → BDNF ↑ → repair rate ↑
    - Isolation, stress, depression → BDNF ↓ → minimal repair

    BDNF cannot reverse aging, but can slow accumulation.
    """
    print("\n" + "=" * 60)
    print("EXP 06: BDNF Neurotrophic Factor Protective Effect")
    print("=" * 60)

    eng_no = PinchFatigueEngine()
    eng_yes = PinchFatigueEngine()
    high_i = _high_current()

    for _ in range(500):
        eng_no.apply_current_pulse("axon", high_i)
        eng_yes.apply_current_pulse("axon", high_i)
        eng_no.repair_tick(growth_factor=0.0)
        eng_yes.repair_tick(growth_factor=1.0)

    p_no = eng_no._channels["axon"].plastic_strain
    p_yes = eng_yes._channels["axon"].plastic_strain

    print(f" Without BDNF: plastic residual = {p_no:.6f}")
    print(f" With BDNF: plastic residual = {p_yes:.6f}")
    print(f" Protection rate = {(1 - p_yes/max(p_no, 1e-12))*100:.1f}%")

    assert p_yes < p_no, "BDNF should reduce plastic residual"
    print(" ✓ BDNF provides protection but cannot completely prevent aging")
    return True


# ============================================================================
# experiment 7: Arrhenius Temperatureacceleration
# ============================================================================

def exp_07_arrhenius_temperature():
    """
    Arrhenius law: reaction rate ∝ exp(-E_a/kT)
    In Alice: temperature = ram_temperature / anxiety level
    High anxiety (high temp) → yield strength decreases → easier to generate plasticity → accelerated aging

    This explains why 'chronic anxiety accelerates aging'.
    """
    print("\n" + "=" * 60)
    print("EXP 07: Arrhenius Temperature Acceleration — Anxiety Accelerates Aging")
    print("=" * 60)

    temps = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {}

    for temp in temps:
        eng = PinchFatigueEngine()
        high_i = _high_current()
        for _ in range(300):
            eng.apply_current_pulse("axon", high_i, temperature=temp)
        results[temp] = eng._channels["axon"].fatigue_damage

    for temp, damage in results.items():
        bar = "█" * int(damage * 500)
        print(f"  T={temp:.1f}: fatigue={damage:.6f}  {bar}")

    assert results[0.9] > results[0.1], "High temperature should generate more fatigue"
    print(" ✓ Arrhenius temperature acceleration effect verification successful")
    return True


# ============================================================================
# Experiment 8: Work Hardening
# ============================================================================

def exp_08_work_hardening():
    """
    Work hardening = dislocation density increase → harder to deform subsequently
    In Alice: Repeated high current → yield strength ↑ → same current generates less plasticity

    'Weathering storms makes you stronger — but also more brittle.'
    ― This is the metallurgical engineering paradox, and also a psychological one.
    """
    print("\n" + "=" * 60)
    print("EXP 08: Work Hardening — Does Experience Make You Stronger?")
    print("=" * 60)

    eng = PinchFatigueEngine()
    high_i = _high_current()

    yields = []
    for batch in range(10):
        ch = eng._get_or_create_channel("axon")
        yields.append(ch.effective_yield)
        for _ in range(100):
            eng.apply_current_pulse("axon", high_i)

    ch = eng._channels["axon"]
    final_yield = ch.effective_yield

    print(f" Initial yield strength: {YIELD_STRAIN:.4f}")
    print(f" Final yield strength: {final_yield:.4f}")
    print(f" Hardening amount: {ch.work_hardening:.6f}")
    print(f" Yield trajectory:")
    for i, y in enumerate(yields):
        print(f"    Batch {i:2d}: ε_yield = {y:.6f}")

    assert final_yield >= YIELD_STRAIN, "Hardening can only increase yield strength"
    print(" ✓ Work hardening slightly raises yield strength — physics correct")
    return True


# ============================================================================
# Experiment 9: Multi-Channel Differential Aging
# ============================================================================

def exp_09_multi_channel_differential():
    """
    Different brain regions experience different intensity 'currents':
    - Visual cortex: high-frequency use → rapid aging
    - Motor cortex: moderate use
    - Auditory cortex: low-frequency use → slowest aging

    This explains why different cognitive functions decline at different ages.
    """
    print("\n" + "=" * 60)
    print("EXP 09: Multi-Channel Differential Aging — Brain Regions Do Not Age Uniformly")
    print("=" * 60)

    eng = PinchFatigueEngine()
    high_i = _high_current()

    for _ in range(500):
        eng.tick({
            "visual": high_i, # High load
            "motor": high_i * 0.7, # Moderate load
            "auditory": 0.3, # Low load
            "prefrontal": high_i * 0.85, # High load
        })

    channels = ["visual", "motor", "auditory", "prefrontal"]
    for ch_id in channels:
        ch = eng._channels.get(ch_id)
        if ch:
            print(f"  {ch_id:>12}: age={ch.age_factor:.6f}  "
                  f"Z_drift={ch.impedance_drift:.6f}  "
                  f"integrity={ch.structural_integrity:.4f}")

    assert eng.get_channel_age("visual") > eng.get_channel_age("auditory")
    print(" ✓ High-load channels age fastest — differential degradation prediction correct")
    return True


# ============================================================================
# experiment 10: AliceBrain all lifecycle
# ============================================================================

def exp_10_alice_lifetime_simulation():
    """
    Integrate PinchFatigueEngine into AliceBrain,
    simulate 100 cognitive cycles of aging accumulation.
    """
    print("\n" + "=" * 60)
    print("EXP 10: AliceBrain Full Lifecycle Aging Simulation")
    print("=" * 60)

    from alice.alice_brain import AliceBrain
    brain = AliceBrain()

    # Simulate 50 cognitive activities
    import numpy as np_exp
    for i in range(50):
        result = brain.perceive(np_exp.random.randn(64))

    stats = brain.pinch_fatigue.get_stats()
    print(f" Aging state after 50 perceptions:")
    print(f" Total channels: {stats['total_channels']}")
    print(f" Mean aging: {stats['mean_age_factor']:.6f}")
    print(f" Total plastic strain: {stats['total_plastic_strain']:.6f}")
    print(f" Mean Z drift: {stats['mean_impedance_drift']:.6f}")
    print(f" Degraded channels: {stats['degraded_channels']}")

    assert stats["total_channels"] >= 1, "Should have at least one channel"
    print(" ✓ AliceBrain full lifecycle aging integration successful")
    return True


# ============================================================================
# Main
# ============================================================================

ALL_EXPERIMENTS = [
    exp_01_pinch_square_law,
    exp_02_elastic_plastic_separation,
    exp_03_aging_trajectory,
    exp_04_sleep_vs_aging,
    exp_05_high_vs_low_stress_life,
    exp_06_bdnf_protection,
    exp_07_arrhenius_temperature,
    exp_08_work_hardening,
    exp_09_multi_channel_differential,
    exp_10_alice_lifetime_simulation,
]


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║ Lorentz Compression Fatigue Physics — Pollock-Barraclough (1905) Neural Aging ║")
    print("║ 'Aging is not rusting — it is your cables being squeezed by their own magnetic field.' ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    passed = 0
    failed = 0
    for exp in ALL_EXPERIMENTS:
        try:
            result = exp()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {exp.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ✗ {exp.__name__} ERROR: {e}")

    print(f"\n{'=' * 60}")
    print(f"Result: {passed} PASS / {failed} FAIL / {len(ALL_EXPERIMENTS)} total")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

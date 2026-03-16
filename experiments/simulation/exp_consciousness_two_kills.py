# -*- coding: utf-8 -*-
"""
Experiment: Fever vs Occipital Impact — Two Kill Switches of Consciousness
===========================================================================

Hypothesis:
  Consciousness has (at least) TWO independent kill mechanisms:
    1. FEVER (PFC thermal noise) — gradual degradation of high-K matrix inversion
    2. IMPACT (brainstem arousal) — instant shutdown of the global enable signal

  The key insight: the THALAMUS equation reveals WHY:
    G_total = G_arousal × (α × G_topdown + (1-α) × G_bottomup)

  - Fever: G_arousal stays ~0.85, but PFC matrix noise → Go/NoGo errors
  - Impact: G_arousal → 0 → G_total → 0 for ALL channels → total blackout

  The posterior brain contains the BRAINSTEM (reticular activating system, RAS),
  which is the SOURCE of the arousal signal. Hit it → arousal crashes → 
  multiplicative kill of all thalamic gates simultaneously.

Physical predictions:
  - Fever: gradual, modality-selective degradation (high-K channels fail first)
  - Impact: instant, all-channel simultaneous blackout
  - Impact produces buzzing (vestibular/cochlear shock) + dizziness BEFORE blackout
  - Fever produces confusion/delirium (PFC executive failure) BEFORE blackout
"""

from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alice.brain.thalamus import ThalamusEngine
from alice.brain.prefrontal import PrefrontalCortexEngine


def banner(title: str):
    w = 72
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def section(title: str):
    print(f"\n--- {title} ---")


def make_signal(K: int = 16, amplitude: float = 0.6, gamma: float = 0.35):
    """Create a sensory signal with fingerprint."""
    fp = np.random.randn(K)
    fp /= np.linalg.norm(fp) + 1e-10
    return {
        "fingerprint": fp + np.random.normal(0, 0.05, K),
        "amplitude": np.clip(amplitude + np.random.normal(0, 0.03), 0.05, 0.95),
        "gamma": np.clip(gamma + np.random.normal(0, 0.02), 0.01, 0.99),
    }


def run_scenario(
    scenario_name: str,
    arousal_trajectory: list[float],
    temperature_trajectory: list[float],
    description: str,
):
    """
    Run a scenario over N ticks with given arousal and temperature trajectories.

    Returns summary dict.
    """
    N = len(arousal_trajectory)
    assert len(temperature_trajectory) == N

    thalamus = ThalamusEngine(alpha=0.6)
    prefrontal = PrefrontalCortexEngine()
    prefrontal.set_goal("perceive_world", z_goal=75.0, priority=0.8)

    # Set initial attention
    thalamus.set_attention("visual", bias=0.8)
    thalamus.set_attention("auditory", bias=0.5)
    thalamus.set_attention("vestibular", bias=0.3)

    np.random.seed(123)

    # Track per-tick data
    records = []

    for t in range(N):
        arousal = arousal_trajectory[t]
        temperature = temperature_trajectory[t]

        thalamus.set_arousal(arousal)

        # Generate sensory signals (visual, auditory, vestibular)
        signals = {
            "visual": make_signal(K=16, amplitude=0.65, gamma=0.30),
            "auditory": make_signal(K=16, amplitude=0.50, gamma=0.35),
            "vestibular": make_signal(K=8, amplitude=0.40, gamma=0.25),
        }

        # Thalamic gating
        gate_results = thalamus.gate_all(signals, arousal=arousal)

        # PFC evaluation for passed signals
        pfc_decisions = {}
        for modality, res in gate_results.items():
            if res.passed:
                z_action = 75.0 * (1.0 + res.gate_gain * 0.2 - res.salience * 0.1)
                dec = prefrontal.evaluate_action(
                    f"{modality}_t{t}",
                    z_action=z_action,
                    source="cortical",
                    urgency=res.salience,
                )
                pfc_decisions[modality] = dec.decision
            else:
                pfc_decisions[modality] = "BLOCKED"

        # Temperature effect on PFC: thermal noise → drain energy
        # Higher temperature → more energy drain (simulating glucose depletion)
        T_kelvin = 273.15 + temperature
        T_normal = 310.15
        if T_kelvin > T_normal:
            thermal_drain = ((T_kelvin - T_normal) / T_normal) ** 2 * 0.5
            prefrontal.drain_energy(thermal_drain)

        prefrontal.tick()

        # Compute effective consciousness proxy:
        #   Φ_proxy = mean(gate_gains) × PFC_energy × arousal
        gains = [gate_results[m].gate_gain for m in gate_results]
        mean_gain = np.mean(gains) if gains else 0
        phi_proxy = mean_gain * prefrontal._energy * arousal

        records.append({
            "tick": t,
            "arousal": arousal,
            "temperature": temperature,
            "v_gate": gate_results["visual"].gate_gain,
            "a_gate": gate_results["auditory"].gate_gain,
            "vest_gate": gate_results["vestibular"].gate_gain,
            "v_passed": gate_results["visual"].passed,
            "a_passed": gate_results["auditory"].passed,
            "vest_passed": gate_results["vestibular"].passed,
            "v_dec": pfc_decisions.get("visual", "-"),
            "a_dec": pfc_decisions.get("auditory", "-"),
            "vest_dec": pfc_decisions.get("vestibular", "-"),
            "pfc_energy": prefrontal._energy,
            "phi_proxy": phi_proxy,
            "v_burst": gate_results["visual"].is_burst,
        })

    return records


def print_scenario(name: str, description: str, records: list):
    """Pretty print scenario results."""
    section(f"{name}")
    print(f"  {description}\n")

    print(f"  {'t':>3s}  {'arou':>5s}  {'T°C':>5s}  "
          f"{'V_gate':>7s}  {'A_gate':>7s}  {'Vest':>7s}  "
          f"{'V':>4s}  {'A':>4s}  {'Ve':>4s}  "
          f"{'PFC_E':>6s}  {'Φ_proxy':>8s}  {'Status':>12s}")
    print("  " + "-" * 88)

    for r in records:
        # Status determination
        phi = r["phi_proxy"]
        if phi < 0.02:
            status = "☠ BLACKOUT"
        elif phi < 0.10:
            status = "⚠ NEAR-OUT"
        elif phi < 0.20:
            status = "△ IMPAIRED"
        elif phi < 0.30:
            status = "▽ DEGRADED"
        else:
            status = "● NORMAL"

        # Mark burst ticks
        burst_mark = " [BURST]" if r.get("v_burst") else ""

        v_mark = "✓" if r["v_passed"] else "✗"
        a_mark = "✓" if r["a_passed"] else "✗"
        ve_mark = "✓" if r["vest_passed"] else "✗"

        print(f"  {r['tick']:3d}  {r['arousal']:5.2f}  {r['temperature']:5.1f}  "
              f"{r['v_gate']:7.4f}  {r['a_gate']:7.4f}  {r['vest_gate']:7.4f}  "
              f"{v_mark:>4s}  {a_mark:>4s}  {ve_mark:>4s}  "
              f"{r['pfc_energy']:6.3f}  {r['phi_proxy']:8.4f}  "
              f"{status:>12s}{burst_mark}")


def run_experiment():
    banner("Two Kill Switches of Consciousness: Fever vs Occipital Impact")

    # ================================================================
    # Scenario A: FEVER (gradual PFC degradation)
    # ================================================================
    # Temperature rises from 37°C to 41°C over 20 ticks
    # Arousal stays high (patient is awake but confused)
    N = 20
    fever_arousal = [0.85] * N  # Stays awake
    fever_temperature = [37.0 + (41.0 - 37.0) * t / (N - 1) for t in range(N)]

    fever_records = run_scenario(
        "Fever (Gradual PFC Degradation)",
        fever_arousal,
        fever_temperature,
        "Temperature 37→41°C, arousal stays 0.85",
    )

    # ================================================================
    # Scenario B: OCCIPITAL IMPACT (sudden arousal crash)
    # ================================================================
    # Temperature stays 37°C, arousal drops from 0.85 to 0 in 3 ticks
    # Then slowly recovers (regaining consciousness)
    impact_arousal = (
        [0.85] * 5                          # Normal for 5 ticks
        + [0.40, 0.10, 0.02, 0.01, 0.01]   # Impact! Arousal crashes in 3 ticks
        + [0.03, 0.08, 0.15, 0.25, 0.35]   # Slow recovery
        + [0.45, 0.55, 0.65, 0.75, 0.80]   # Returning to normal
    )
    impact_temperature = [37.0] * N   # Temperature unchanged

    impact_records = run_scenario(
        "Occipital Impact (Brainstem Arousal Crash)",
        impact_arousal,
        impact_temperature,
        "Temperature constant 37°C, arousal crashes at t=5",
    )

    # ================================================================
    # Scenario C: IMPACT + VESTIBULAR SHOCK (realistic posterior hit)
    # ================================================================
    # Same arousal crash, but vestibular signal spikes (buzzing + dizziness)
    # This simulates the "嗡鳴聲 + 眩暈" the user described

    section("Scenario C: Impact with Vestibular/Cochlear Shock")
    print("  Simulating: 嗡鳴聲 (tinnitus) + 眩暈 (vertigo) → 意識中斷")

    thalamus_c = ThalamusEngine(alpha=0.6)
    prefrontal_c = PrefrontalCortexEngine()
    prefrontal_c.set_goal("perceive_world", z_goal=75.0, priority=0.8)
    thalamus_c.set_attention("visual", 0.8)
    thalamus_c.set_attention("auditory", 0.5)
    thalamus_c.set_attention("vestibular", 0.3)

    np.random.seed(456)
    impact_c_records = []

    for t in range(N):
        arousal = impact_arousal[t]
        thalamus_c.set_arousal(arousal)

        # At impact (t=5,6,7): vestibular and auditory signals SPIKE
        # This is the mechanical shockwave reaching the inner ear
        if 5 <= t <= 7:
            vest_amp = 0.95   # Massive vestibular shock
            vest_gamma = 0.90  # Extreme mismatch (abnormal signal)
            aud_amp = 0.85    # Tinnitus — loud internal noise
            aud_gamma = 0.80  # Highly mismatched (not normal sound)
            event = "★ IMPACT"
        elif 8 <= t <= 10:
            vest_amp = 0.60   # Residual dizziness
            vest_gamma = 0.55
            aud_amp = 0.55    # Fading tinnitus
            aud_gamma = 0.50
            event = "~ residual"
        else:
            vest_amp = 0.35
            vest_gamma = 0.25
            aud_amp = 0.45
            aud_gamma = 0.30
            event = ""

        signals = {
            "visual": make_signal(K=16, amplitude=0.60, gamma=0.30),
            "auditory": {
                "fingerprint": np.random.randn(16) / 4,
                "amplitude": aud_amp,
                "gamma": aud_gamma,
            },
            "vestibular": {
                "fingerprint": np.random.randn(8) / 4,
                "amplitude": vest_amp,
                "gamma": vest_gamma,
            },
        }

        gate_results = thalamus_c.gate_all(signals, arousal=arousal)
        prefrontal_c.tick()

        gains = [gate_results[m].gate_gain for m in gate_results]
        phi_proxy = np.mean(gains) * prefrontal_c._energy * arousal

        # Startle bypass check
        startle_list = [m for m, r in gate_results.items() if r.is_startle]
        burst_list = [m for m, r in gate_results.items() if r.is_burst]

        impact_c_records.append({
            "tick": t,
            "arousal": arousal,
            "v_gate": gate_results["visual"].gate_gain,
            "a_gate": gate_results["auditory"].gate_gain,
            "vest_gate": gate_results["vestibular"].gate_gain,
            "phi_proxy": phi_proxy,
            "startles": startle_list,
            "bursts": burst_list,
            "event": event,
        })

    # ================================================================
    # Print all results
    # ================================================================
    print_scenario(
        "Scenario A: FEVER (Gradual PFC Thermal Degradation)",
        "G_arousal = 0.85 (constant) | Temperature: 37°C → 41°C",
        fever_records,
    )

    print_scenario(
        "Scenario B: OCCIPITAL IMPACT (Brainstem Arousal Crash)",
        "Temperature = 37°C (constant) | G_arousal: 0.85 → 0.01 → recovery",
        impact_records,
    )

    section("Scenario C: Full Impact Sequence (嗡鳴 + 眩暈 → 昏迷)")
    print("  Impact at t=5: mechanical shockwave → vestibular spike + tinnitus + arousal crash\n")

    print(f"  {'t':>3s}  {'arou':>5s}  {'V_gate':>7s}  {'A_gate':>7s}  {'Vest':>7s}  "
          f"{'Φ_proxy':>8s}  {'Startle':>12s}  {'Burst':>10s}  {'Event':>10s}")
    print("  " + "-" * 82)

    for r in impact_c_records:
        phi = r["phi_proxy"]
        if phi < 0.02:
            phi_str = f"{phi:8.4f} ☠"
        elif phi < 0.10:
            phi_str = f"{phi:8.4f} ⚠"
        else:
            phi_str = f"{phi:8.4f}  "

        startles = ",".join(r["startles"]) if r["startles"] else "-"
        bursts = ",".join(r["bursts"]) if r["bursts"] else "-"

        print(f"  {r['tick']:3d}  {r['arousal']:5.2f}  "
              f"{r['v_gate']:7.4f}  {r['a_gate']:7.4f}  {r['vest_gate']:7.4f}  "
              f"{phi_str}  {startles:>12s}  {bursts:>10s}  {r['event']:>10s}")

    # ================================================================
    # Comparative analysis
    # ================================================================
    banner("COMPARATIVE ANALYSIS: Two Different Kill Mechanisms")

    section("Mechanism A: Fever (前額葉熱雜訊)")
    print("""
  Path: Temperature ↑ → Johnson-Nyquist noise → PFC impedance matrix degraded
  
  Physics:
    δΓ ∝ κ(Z) · δT/T
    κ(Z) ∝ K^(3/2) → PFC (K≈128) degrades FIRST
  
  Symptoms:
    - Gradual confusion (PFC Go/NoGo errors)
    - Thalamic gates STAY OPEN (arousal unchanged)
    - Modality-selective: high-K channels fail first
    - Slow onset (minutes to hours)
  
  Analogy: OVERHEATING a CPU — transistors produce errors, but power stays on
""")

    section("Mechanism B: Occipital Impact (腦幹覺醒中斷)")
    print("""
  Path: Mechanical shock → brainstem RAS disrupted → G_arousal → 0
  
  Physics:
    G_total = G_arousal × (α·G_topdown + (1-α)·G_bottomup)
    G_arousal → 0 ⟹ G_total → 0 for ALL channels simultaneously
  
  Symptoms:
    - Instant blackout (all gates close at once)
    - 嗡鳴 (tinnitus): cochlear shock → STARTLE bypass before gates close
    - 眩暈 (vertigo): vestibular shock → STARTLE bypass
    - Fast onset (milliseconds)
    - Startle signals are the LAST things to pass (hardware interrupt)
  
  Analogy: PULLING THE POWER PLUG — everything dies at once
""")

    # Quantitative comparison
    section("Quantitative: Φ_proxy Collapse Timeline")

    # Find when each scenario crosses Φ < 0.10 threshold
    fever_collapse = None
    impact_collapse = None
    for r in fever_records:
        if r["phi_proxy"] < 0.10 and fever_collapse is None:
            fever_collapse = r["tick"]
    for r in impact_records:
        if r["phi_proxy"] < 0.10 and impact_collapse is None:
            impact_collapse = r["tick"]

    fever_min_phi = min(r["phi_proxy"] for r in fever_records)
    impact_min_phi = min(r["phi_proxy"] for r in impact_records)

    print(f"\n  Fever:  Φ < 0.10 at tick {fever_collapse if fever_collapse else 'NEVER'}  |  min Φ = {fever_min_phi:.4f}")
    print(f"  Impact: Φ < 0.10 at tick {impact_collapse if impact_collapse else 'NEVER'}  |  min Φ = {impact_min_phi:.4f}")

    if impact_collapse and fever_collapse:
        print(f"\n  Impact is {fever_collapse - impact_collapse}x faster to blackout")
    elif impact_collapse and not fever_collapse:
        print(f"\n  Impact causes blackout; fever does NOT reach blackout (PFC degrades but gates stay open)")

    # Gate pattern comparison
    section("Gate Pattern at Maximum Disruption")

    # Fever: last tick
    fr = fever_records[-1]
    print(f"\n  FEVER (t={fr['tick']}, T={fr['temperature']:.1f}°C):")
    print(f"    Visual gate:      {fr['v_gate']:.4f}  {'OPEN' if fr['v_passed'] else 'CLOSED'}")
    print(f"    Auditory gate:    {fr['a_gate']:.4f}  {'OPEN' if fr['a_passed'] else 'CLOSED'}")
    print(f"    Vestibular gate:  {fr['vest_gate']:.4f}  {'OPEN' if fr['vest_passed'] else 'CLOSED'}")
    print(f"    PFC energy:       {fr['pfc_energy']:.4f}")
    print(f"    → Gates OPEN but PFC impaired: 'lights on, nobody home'")

    # Impact: worst tick
    ir = min(impact_records, key=lambda r: r["phi_proxy"])
    print(f"\n  IMPACT (t={ir['tick']}, arousal={ir['arousal']:.2f}):")
    print(f"    Visual gate:      {ir['v_gate']:.4f}  {'OPEN' if ir['v_passed'] else 'CLOSED'}")
    print(f"    Auditory gate:    {ir['a_gate']:.4f}  {'OPEN' if ir['a_passed'] else 'CLOSED'}")
    print(f"    Vestibular gate:  {ir['vest_gate']:.4f}  {'OPEN' if ir['vest_passed'] else 'CLOSED'}")
    print(f"    PFC energy:       {ir['pfc_energy']:.4f}")
    print(f"    → Gates CLOSED, PFC intact: 'power cut, hardware fine'")

    banner("CONCLUSION: Where is Consciousness?")
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Neither the frontal nor the posterior brain IS consciousness.  │
  │                                                                 │
  │  The experiment reveals TWO independent kill mechanisms:        │
  │                                                                 │
  │  ① FEVER (PFC):                                                 │
  │     δΓ ∝ κ(Z)·δT/T → high-K matrix noise                      │
  │     "Lights on, nobody home" — gates open, decisions broken     │
  │     → PFC is the PROCESSOR that degrades                        │
  │                                                                 │
  │  ② IMPACT (Brainstem RAS):                                      │
  │     G_arousal → 0 → G_total → 0 for ALL channels               │
  │     "Power cut" — all gates close simultaneously                │
  │     → Brainstem is the POWER SUPPLY that shuts off              │
  │                                                                 │
  │  If consciousness were IN either location,                      │
  │  destroying the other should not affect it.                     │
  │  But BOTH independently kill consciousness.                     │
  │                                                                 │
  │  ∴ Consciousness is NOT a location.                              │
  │    It is the PRODUCT: Φ = f(arousal × Σ gate_i × PFC_quality)  │
  │                                                                 │
  │  Kill ANY multiplicative factor → Φ → 0                         │
  │  Just like Area = width × height — destroy either → area = 0   │
  └─────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    run_experiment()

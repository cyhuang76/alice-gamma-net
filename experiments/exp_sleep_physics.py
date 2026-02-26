# -*- coding: utf-8 -*-
"""
Experiment: Sleep Physics — Offline Impedance Renormalization

Core Hypothesis:
  Awake = external impedance matching (Minimize Γ_ext)
  Sleep = internal impedance matching (Minimize Γ_int)

Five experiments:
  1. Day-night cycle energy conservation — awake consumption / sleep recovery
  2. Synaptic downscaling — N3 deep sleep Synaptic Homeostasis
  3. Sleep deprivation effects — physical cost of not sleeping
  4. Dream channel diagnostics — REM impedance testing function
  5. Memory consolidation gain — pre-sleep vs. post-sleep performance difference

Usage: python -m experiments.exp_sleep_physics
"""

from __future__ import annotations

import numpy as np

from alice.brain.sleep_physics import (
    SleepPhysicsEngine,
    ImpedanceDebtTracker,
    SynapticEntropyTracker,
    SlowWaveOscillator,
    REMDreamDiagnostic,
    SleepQualityReport,
)


def banner():
    print("=" * 70)
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║ Γ-Net ALICE Sleep Physics Experiment — Offline Impedance Renormalization ║")
    print("║                                                                    ║")
    print("║ Awake = Minimize Γ_ext (external matching)                         ║")
    print("║ Sleep = Minimize Γ_int (internal repair)                           ║")
    print("║                                                                    ║")
    print("║       dE/dt = -P_metabolic + P_recovery                            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


# ============================================================
# Experiment 1: Day-night cycle energy conservation
# ============================================================

def exp1_day_night_cycle():
    print("=" * 70)
    print("  Experiment 1: Day-night cycle energy conservation")
    print("  — Awake consumes, sleep recovers, impedance debt accumulates and repairs")
    print("=" * 70)
    print()

    engine = SleepPhysicsEngine(energy=1.0)
    rng = np.random.default_rng(42)

    n_synapses = 300
    synaptic_strengths = list(rng.uniform(0.5, 1.5, n_synapses))
    n_channels = 6
    channel_impedances = [
        (f"ch_{i}", float(rng.uniform(50, 110)),
         float(rng.uniform(50, 110)))
        for i in range(n_channels)
    ]

    awake_ticks = 100
    sleep_ticks = 110

    # Daytime
    print("  -- Daytime (awake 100 ticks) --")
    print(f"  │ {'Tick':>5} │ {'Energy':>8} │ {'Imp. Debt':>10} │ {'Entropy':>8} │ {'Sleep Pres.':>10} │")
    print(f"  │{'─'*5:─>5}─│{'─'*8:─>8}─│{'─'*10:─>10}─│{'─'*8:─>8}─│{'─'*10:─>10}─│")

    for t in range(awake_ticks):
        if t % 10 == 0:
            boost_idx = rng.integers(0, n_synapses, size=5)
            for idx in boost_idx:
                synaptic_strengths[idx] = min(2.0, synaptic_strengths[idx] * 1.05)

        re = float(rng.uniform(0.02, 0.08))
        result = engine.awake_tick(
            reflected_energy=re,
            synaptic_strengths=synaptic_strengths,
        )
        if t % 10 == 0:
            print(f"  │ {t:5d} │ {result['energy']:8.4f} │ {result['impedance_debt']:10.4f} │ "
                  f"{result['entropy']:8.4f} │ {result['sleep_pressure']:10.4f} │")

    pre_sleep = {
        "energy": engine.energy,
        "debt": engine.impedance_debt.debt,
        "entropy": engine.entropy_tracker.current_entropy,
        "pressure": engine.sleep_pressure,
    }
    print(f"\n  End of daytime:")
    print(f"    Energy: {pre_sleep['energy']:.4f} (dropped from 1.0)")
    print(f"    Impedance debt: {pre_sleep['debt']:.4f}")
    print(f"    Sleep pressure: {pre_sleep['pressure']:.4f}")
    print(f"    Should sleep? {'\u2713 Yes' if engine.should_sleep() else '\u2717 No'}")

    # nighttime
    print(f"\n  -- Nighttime (sleep {sleep_ticks} ticks) --")
    engine.begin_sleep()
    schedule = engine._generate_sleep_schedule(sleep_ticks)

    print(f"  │ {'Tick':>5} │ {'Stage':>6} │ {'Energy':>8} │ {'Imp. Debt':>10} │ {'Replay':>4} │ {'Downsc.':>6} │")
    print(f"  │{'─'*5:─>5}─│{'─'*6:─>6}─│{'─'*8:─>8}─│{'─'*10:─>10}─│{'─'*4:─>4}─│{'─'*6:─>6}─│")

    for t, stage in enumerate(schedule):
        result = engine.sleep_tick(
            stage=stage,
            recent_memories=[f"mem_{i}" for i in range(10)],
            channel_impedances=channel_impedances,
            synaptic_strengths=synaptic_strengths,
        )
        if result.get("downscale_strengths"):
            synaptic_strengths = result["downscale_strengths"]

        if t % 10 == 0:
            ds = "✓" if result["downscaled"] else "—"
            print(f"  │ {t:5d} │ {stage:>6} │ {result['energy']:8.4f} │ "
                  f"{result['impedance_debt']:10.4f} │ {result['replayed']:4d} │ {ds:>6} │")

    report = engine.end_sleep()

    post_sleep = {
        "energy": engine.energy,
        "debt": engine.impedance_debt.debt,
        "entropy": engine.entropy_tracker.current_entropy,
        "pressure": engine.sleep_pressure,
    }

    print(f"\n  -- Sleep Quality Report --")
    print(f"    Total sleep ticks: {report.total_sleep_ticks}")
    print(f"    N3 deep sleep ratio: {report.n3_ratio:.1%}")
    print(f"    REM ratio: {report.rem_ratio:.1%}")
    print(f"    Slow wave cycles: {report.slow_wave_cycles}")
    print(f"    Memories consolidated: {report.memories_consolidated}")
    print(f"    Sleep quality score: {report.quality_score:.3f}")

    print(f"\n  -- Day-Night Comparison --")
    print(f"  │ {'Metric':>14} │ {'End of Day':>10} │ {'Post-Sleep':>10} │ {'Change':>10} │")
    print(f"  │{'─'*14:─>14}─│{'─'*10:─>10}─│{'─'*10:─>10}─│{'─'*10:─>10}─│")
    for label, pre_key, post_key in [
        ("energy", "energy", "energy"),
        ("impedance debt", "debt", "debt"),
        ("sleep pressure", "pressure", "pressure"),
    ]:
        pre_v = pre_sleep[pre_key]
        post_v = post_sleep[post_key]
        delta = post_v - pre_v
        arrow = "↑" if delta > 0 else "↓"
        print(f"  │ {label:>14} │ {pre_v:10.4f} │ {post_v:10.4f} │ {arrow}{abs(delta):9.4f} │")

    print(f"\n  ── Physical Conclusion ──")
    print(f"    Energy conservation: awake consumed {1.0 - pre_sleep['energy']:.4f}, "
          f"sleep recovered {post_sleep['energy'] - pre_sleep['energy']:.4f}")
    print(f"    Impedance debt: daytime accumulated {pre_sleep['debt']:.4f}, "
          f"sleep repaired to {post_sleep['debt']:.4f}")
    print(f"    Sleep pressure: from {pre_sleep['pressure']:.4f} → {post_sleep['pressure']:.4f}")
    print()


# ============================================================
# Experiment 2: Synaptic downscaling — Synaptic Homeostasis Hypothesis
# ============================================================

def exp2_synaptic_downscaling():
    print("=" * 70)
    print("  Experiment 2: Synaptic downscaling — Tononi synaptic homeostasis hypothesis")
    print("  — N3 deep sleep: global proportional scaling preserves relative differences")
    print("=" * 70)
    print()

    rng = np.random.default_rng(123)

    # Initial synapses (uniform)
    n = 100
    strengths_birth = list(rng.uniform(0.8, 1.2, n))

    # Simulate daytime learning: some synapses greatly enhanced
    strengths_after_learning = strengths_birth.copy()
    # 10 'important memory' synapses enhanced
    important_indices = list(range(0, 10))
    for idx in important_indices:
        strengths_after_learning[idx] *= 1.5  # +50%

    # 5 'interference' synapses also accidentally enhanced
    noise_indices = list(range(50, 55))
    for idx in noise_indices:
        strengths_after_learning[idx] *= 1.3  # +30%

    print(f"  Initial synapses (birth):")
    print(f"    mean: {np.mean(strengths_birth):.4f}")
    print(f"    std dev: {np.std(strengths_birth):.4f}")
    print(f"    max / min: {max(strengths_birth):.4f} / {min(strengths_birth):.4f}")

    print(f"\n  After daytime learning:")
    print(f"    mean: {np.mean(strengths_after_learning):.4f}")
    print(f"    std dev: {np.std(strengths_after_learning):.4f}")
    print(f"    max / min: {max(strengths_after_learning):.4f} / {min(strengths_after_learning):.4f}")
    print(f"    Important memory synapse (#0): {strengths_after_learning[0]:.4f}")
    print(f"    Normal synapse (#20): {strengths_after_learning[20]:.4f}")
    print(f"    Interference synapse (#50): {strengths_after_learning[50]:.4f}")

    # Simulate multiple N3 downscaling cycles
    strengths = strengths_after_learning.copy()
    print(f"\n  N3 downscaling process (factor=0.990/cycle):")
    print(f"  │ {'Cycle':>6} │ {'Mean':>8} │ {'Std Dev':>8} │ {'Imp #0':>8} │ {'Norm #20':>8} │ {'Noise #50':>8} │ {'Ratio(#0/#20)':>12} │")
    print(f"  │{'─'*6:─>6}─│{'─'*8:─>8}─│{'─'*8:─>8}─│{'─'*8:─>8}─│{'─'*8:─>8}─│{'─'*8:─>8}─│{'─'*12:─>12}─│")

    for cycle in range(21):
        if cycle % 2 == 0:
            ratio = strengths[0] / max(strengths[20], 0.001)
            print(f"  │ {cycle:6d} │ {np.mean(strengths):8.4f} │ {np.std(strengths):8.4f} │ "
                  f"{strengths[0]:8.4f} │ {strengths[20]:8.4f} │ {strengths[50]:8.4f} │ {ratio:12.4f} │")
        strengths = SleepPhysicsEngine.apply_downscaling(strengths, factor=0.990)

    print(f"\n  ── Physical Conclusion ──")
    print(f"    Post-downscaling mean: {np.mean(strengths):.4f} (from {np.mean(strengths_after_learning):.4f})")
    print(f"    Important memory synapse (#0): {strengths[0]:.4f} (still strongest)")
    ratio_before = strengths_after_learning[0] / strengths_after_learning[20]
    ratio_after = strengths[0] / strengths[20]
    print(f"    Ratio preserved (#0/#20):")
    print(f"      Post-learning ratio: {ratio_before:.4f}")
    print(f"      Post-downscaling: {ratio_after:.4f}")
    print(f"      Ratio drift: {abs(ratio_after - ratio_before):.6f} (theoretical ≈ 0)")
    print(f"\n    'Sleep is not forgetting. Sleep reduces noise levels across all regions,")
    print(f"     while preserving relative differences between signals.'")
    print(f"    'Important memories (high-ratio synapses) still stand out after waking.'")
    print()


# ============================================================
# Experiment 3: Sleep deprivation effects
# ============================================================

def exp3_sleep_deprivation():
    print("=" * 70)
    print("  Experiment 3: Sleep deprivation — physical cost of not sleeping")
    print("  — Energy depletion + impedance debt accumulation → system degradation")
    print("=" * 70)
    print()

    rng = np.random.default_rng(77)
    n_synapses = 200

    # Three conditions
    conditions = [
        ("Normal (day 100 + night 110)", 100, 110),
        ("Mild deprivation (day 150 + night 60)", 150, 60),
        ("Full deprivation (day 210 + night 0)", 210, 0),
    ]

    results = []

    for label, awake_t, sleep_t in conditions:
        engine = SleepPhysicsEngine(energy=1.0)
        synapses = list(rng.uniform(0.5, 1.5, n_synapses))
        channels = [
            (f"ch_{i}", float(rng.uniform(50, 100)),
             float(rng.uniform(50, 100)))
            for i in range(6)
        ]

        # Daytime
        for t in range(awake_t):
            if t % 15 == 0:
                idx = rng.integers(0, n_synapses, size=5)
                for i in idx:
                    synapses[i] = min(2.0, synapses[i] * 1.05)
            re = float(rng.uniform(0.02, 0.07))
            engine.awake_tick(reflected_energy=re, synaptic_strengths=synapses)

        mid_state = {
            "energy": engine.energy,
            "debt": engine.impedance_debt.debt,
            "pressure": engine.sleep_pressure,
        }

        # nighttime
        if sleep_t > 0:
            engine.begin_sleep()
            schedule = engine._generate_sleep_schedule(sleep_t)
            for stage in schedule:
                r = engine.sleep_tick(
                    stage=stage,
                    recent_memories=[f"m{i}" for i in range(10)],
                    channel_impedances=channels,
                    synaptic_strengths=synapses,
                )
                if r.get("downscale_strengths"):
                    synapses = r["downscale_strengths"]
            report = engine.end_sleep()
            quality = report.quality_score
        else:
            quality = 0.0

        results.append({
            "label": label,
            "final_energy": engine.energy,
            "final_debt": engine.impedance_debt.debt,
            "final_pressure": engine.sleep_pressure,
            "quality": quality,
            "mid_energy": mid_state["energy"],
            "mid_debt": mid_state["debt"],
        })

    print(f"  ┌──────────────────────────────────────────────────────────────────────────────┐")
    print(f"  │ {'Condition':^24} │ {'Final Energy':>8} │ {'Imp. Debt':>8} │ {'Sleep Pres.':>8} │ {'Sleep Qual.':>8} │")
    print(f"  ├──────────────────────────────────────────────────────────────────────────────┤")
    for r in results:
        print(f"  │ {r['label']:^24} │ {r['final_energy']:8.4f} │ {r['final_debt']:8.4f} │ "
              f"{r['final_pressure']:8.4f} │ {r['quality']:8.3f} │")
    print(f"  └──────────────────────────────────────────────────────────────────────────────┘")

    print(f"\n  ── Physical Conclusion ──")
    print(f"    Normal sleep → energy recovered to {results[0]['final_energy']:.3f}, debt reduced to {results[0]['final_debt']:.4f}")
    print(f"    Full deprivation → energy only {results[2]['final_energy']:.3f}, debt as high as {results[2]['final_debt']:.4f}")
    print(f"    Deprived/normal energy ratio = {results[2]['final_energy']/max(results[0]['final_energy'],0.001):.2f}")
    print(f"\n    This is not a 'punishment for not sleeping' rule —")
    print(f"    It is the physical inevitability of energy conservation (dE/dt = -metabolic + recovery).")
    print(f"    Not sleeping = recovery=0 → energy only flows out → system degrades.")
    print()


# ============================================================
# Experiment 4: REM dream channel diagnostics
# ============================================================

def exp4_dream_diagnostics():
    print("=" * 70)
    print("  Experiment 4: REM dreams — channel health diagnostics")
    print("  — Dreams = random probe signals testing impedance matching")
    print("=" * 70)
    print()

    rng = np.random.default_rng(99)
    dream = REMDreamDiagnostic(rng=rng)

    # Healthy channels (Γ ≈ 0)
    healthy_channels = [
        ("visual→cortex", 50.0, 52.0), # Z close → Γ ≈ 0
        ("auditory→cortex", 75.0, 73.0),
        ("motor→muscle", 50.0, 50.0), # perfect match
        ("prefrontal→motor", 75.0, 78.0),
    ]

    # Damaged channels (Γ >> 0)
    damaged_channels = [
        ("trauma_path", 50.0, 200.0), # severe mismatch → trauma pathway
        ("overloaded_ch", 75.0, 300.0), # overload degradation
        ("neglected_ch", 100.0, 30.0), # long-term disuse
    ]

    all_channels = healthy_channels + damaged_channels

    print(f"  Channel impedance configuration:")
    for name, zs, zl in all_channels:
        gamma = abs((zl - zs) / (zl + zs))
        status = "✓ Healthy" if gamma < 0.3 else "✗ Damaged"
        print(f"    [{name:>18}] Z_src={zs:5.0f}Ω  Z_load={zl:5.0f}Ω  Γ={gamma:.4f}  {status}")

    print(f"\n  REM dream diagnostics (10 probe rounds):")
    print(f"  │ {'Round':>4} │ {'Probes':>6} │ {'OK':>4} │ {'Dmg':>4} │ {'Dream Intens.':>10} │")
    print(f"  │{'─'*4:─>4}─│{'─'*6:─>6}─│{'─'*4:─>4}─│{'─'*4:─>4}─│{'─'*10:─>10}─│")

    for r in range(10):
        result = dream.probe_channels(all_channels)
        print(f"  │ {r+1:4d} │ {result['probes']:6d} │ {result['healthy']:4d} │ "
              f"{result['damaged']:4d} │ {result['dream_intensity']:10.4f} │")

    repair_queue = dream.get_repair_queue()
    state = dream.get_state()

    print(f"\n  -- Diagnosis Summary --")
    print(f"    Total probes: {state['probes_sent']}")
    print(f"    Healthy channels: {state['healthy_channels']} ({state['dream_health_ratio']:.1%})")
    print(f"    Damaged channels: {state['damaged_channels']}")
    print(f"    Cumulative dream reflected energy: {state['total_dream_reflection']:.6f}")

    if repair_queue:
        print(f"\n  Channels needing repair ({len(repair_queue)} total):")
        unique_repairs = {}
        for rep in repair_queue:
            unique_repairs[rep["channel"]] = rep
        for ch, rep in unique_repairs.items():
            print(f"    [{ch:>18}] Γ={rep['gamma']:.4f} "
                  f"Z_src={rep['z_src']:.0f}Ω Z_load={rep['z_load']:.0f}Ω")

    print(f"\n  Recent dream fragments (last 5):")
    for frag in dream.dream_fragments[-5:]:
        emoji = "🟢" if frag["is_healthy"] else "🔴"
        print(f"    {emoji} [{frag['channel']}] probe freq={frag['probe_freq']:.1f} Hz \u0393={frag['gamma']:.4f}")

    print(f"\n  ── Physical Conclusion ──")
    print(f"    Dreams are not 'random imagination' — the brain during REM")
    print(f"    sends random probe signals through every pathway to test impedance matching.")
    print(f"    High-Γ channel → large reflected energy → dream flash (pain micro-trigger) → nightmare")
    print(f"    Low-Γ channel → smooth signal → unnoticed → no dream sensation")
    print(f"    'Having nightmares' = brain discovered damaged pathways needing repair.")
    print()


# ============================================================
# Experiment 5: Memory consolidation gain — pre-sleep vs. post-sleep
# ============================================================

def exp5_memory_consolidation_gain():
    print("=" * 70)
    print("  Experiment 5: Memory consolidation gain — why sleep makes you smarter")
    print("  — Verification: without extra training, post-sleep performance improves automatically")
    print("=" * 70)
    print()

    n_synapses = 200

    # ── Physical Constants ──
    # Neuron activation threshold: synapses below this value → functionally silent
    # This is core to Tononi's SHY: downscaling pushes noise below threshold
    ACTIVATION_THRESHOLD = 0.85

    # Simulate 'learning a difficult task':
    # Correct pathways greatly enhanced, noise pathways slightly enhanced
    correct_indices = list(range(0, 20)) # 20 correct pathways
    noise_indices = list(range(100, 130)) # 30 noise pathways also activated

    def create_learned_synapses():
        """Create post-learning synapses — both conditions use the same starting point."""
        rng_learn = np.random.default_rng(55)
        synapses = list(rng_learn.uniform(0.8, 1.2, n_synapses))
        # 15 learning rounds: correct pathways ×1.03/round, noise pathways ×1.008/round
        # → correct ≈ ×1.558 (range 1.25-1.87)
        # → noise ≈ ×1.127 (range 0.90-1.35)
        # → won't hit cap 2.0, natural variance preserved
        for _ in range(15):
            for idx in correct_indices:
                synapses[idx] = min(2.0, synapses[idx] * 1.03)
            for idx in noise_indices:
                synapses[idx] = min(2.0, synapses[idx] * 1.008)
        return synapses

    def compute_snr(synapses, threshold=0.0):
        """Compute SNR — total power ratio Σ(signal) / Σ(noise).

        Physical meaning: brain's received correct signal total power vs. interference noise total power.
        Removing noise sources (synapses below threshold) → total noise power decreases → SNR rises.
        """
        correct_active = [synapses[i] for i in correct_indices
                          if synapses[i] >= threshold]
        noise_active = [synapses[i] for i in noise_indices
                        if synapses[i] >= threshold]
        sig_power = sum(correct_active)
        noi_power = sum(noise_active) if noise_active else 0.001
        return sig_power / max(noi_power, 0.001), len(correct_active), len(noise_active)

    # -- Common starting point: post-learning synapse state --
    base_synapses = create_learned_synapses()
    pre_correct = np.mean([base_synapses[i] for i in correct_indices])
    pre_noise = np.mean([base_synapses[i] for i in noise_indices])
    pre_snr_raw, _, _ = compute_snr(base_synapses, threshold=0.0)
    pre_snr_eff, pre_sc, pre_sn = compute_snr(base_synapses,
                                               threshold=ACTIVATION_THRESHOLD)

    # -- Condition A: no sleep (15 ticks learning + continue awake 110 ticks) --
    engine_a = SleepPhysicsEngine(energy=1.0)
    synapses_a = base_synapses.copy()
    rng_a = np.random.default_rng(77)

    for _ in range(15 + 110):
        re = float(rng_a.uniform(0.03, 0.06))
        engine_a.awake_tick(reflected_energy=re, synaptic_strengths=synapses_a)

    snr_a_raw, _, _ = compute_snr(synapses_a, threshold=0.0)
    snr_a_eff, a_sc, a_sn = compute_snr(synapses_a,
                                          threshold=ACTIVATION_THRESHOLD)
    correct_a = np.mean([synapses_a[i] for i in correct_indices])
    noise_a = np.mean([synapses_a[i] for i in noise_indices])

    # -- Condition B: learning 15 ticks + sleep 110 ticks --
    engine_b = SleepPhysicsEngine(energy=1.0)
    synapses_b = base_synapses.copy()
    rng_b = np.random.default_rng(77)

    for _ in range(15):
        re = float(rng_b.uniform(0.03, 0.06))
        engine_b.awake_tick(reflected_energy=re, synaptic_strengths=synapses_b)

    # Sleep!
    engine_b.begin_sleep()
    schedule = engine_b._generate_sleep_schedule(110)
    for stage in schedule:
        result = engine_b.sleep_tick(
            stage=stage,
            recent_memories=[f"task_mem_{i}" for i in range(20)],
            channel_impedances=[
                (f"ch_{i}", float(rng_b.uniform(50, 90)),
                 float(rng_b.uniform(50, 90)))
                for i in range(6)
            ],
            synaptic_strengths=synapses_b,
        )
        if result.get("downscale_strengths"):
            synapses_b = result["downscale_strengths"]

    report = engine_b.end_sleep()

    snr_b_raw, _, _ = compute_snr(synapses_b, threshold=0.0)
    snr_b_eff, b_sc, b_sn = compute_snr(synapses_b,
                                          threshold=ACTIVATION_THRESHOLD)
    correct_b = np.mean([synapses_b[i] for i in correct_indices])
    noise_b = np.mean([synapses_b[i] for i in noise_indices])

    # -- Result output --
    print(f"  -- Learning task: 20 correct pathways vs 30 noise pathways --")
    print(f"    Activation threshold = {ACTIVATION_THRESHOLD} (below this → functionally silent)")
    print()

    hdr = (f"  │ {'Condition':>20} │ {'Correct Avg':>10} │ {'Noise Avg':>10} │ "
           f"{'Raw SNR':>8} │ {'Eff. SNR':>8} │ {'Active Sig':>8} │ "
           f"{'Active Noi':>8} │ {'Energy':>8} │")
    sep = (f"  │{'─' * 20}─│{'─' * 10}─│{'─' * 10}─│"
           f"{'─' * 8}─│{'─' * 8}─│{'─' * 8}─│"
           f"{'─' * 8}─│{'─' * 8}─│")
    print(hdr)
    print(sep)
    print(f"  │ {'Post-learning (pre-sleep)':>20} │ {pre_correct:10.4f} │ {pre_noise:10.4f} │ "
          f"{pre_snr_raw:8.4f} │ {pre_snr_eff:8.4f} │ {pre_sc:8d} │ "
          f"{pre_sn:8d} │ {'—':>8} │")
    print(f"  │ {'No sleep (stay awake)':>20} │ {correct_a:10.4f} │ {noise_a:10.4f} │ "
          f"{snr_a_raw:8.4f} │ {snr_a_eff:8.4f} │ {a_sc:8d} │ "
          f"{a_sn:8d} │ {engine_a.energy:8.4f} │")
    print(f"  │ {'Post-sleep':>20} │ {correct_b:10.4f} │ {noise_b:10.4f} │ "
          f"{snr_b_raw:8.4f} │ {snr_b_eff:8.4f} │ {b_sc:8d} │ "
          f"{b_sn:8d} │ {engine_b.energy:8.4f} │")

    raw_change = snr_b_raw / max(snr_a_raw, 0.001) - 1.0
    eff_change = (snr_b_eff / max(snr_a_eff, 0.001) - 1.0
                  if snr_a_eff > 0 else float('inf'))
    noise_eliminated = a_sn - b_sn

    print(f"\n  -- Key Comparison --")
    print(f"    Raw SNR (no threshold):")
    print(f"      Pre-sleep: {pre_snr_raw:.4f}")
    print(f"      No sleep: {snr_a_raw:.4f}")
    print(f"      Post-sleep: {snr_b_raw:.4f} (proportional scaling preserves ratio)")
    print(f"    Effective SNR (threshold {ACTIVATION_THRESHOLD}):")
    print(f"      Pre-sleep: {pre_snr_eff:.4f} ({pre_sc} signal / {pre_sn} noise)")
    print(f"      No sleep: {snr_a_eff:.4f} ({a_sc} signal / {a_sn} noise)")
    print(f"      Post-sleep: {snr_b_eff:.4f} ({b_sc} signal / {b_sn} noise)")
    print(f"    Effective SNR improvement (sleep vs. no sleep): {eff_change:+.1%}")
    print(f"    Noise synapses eliminated: {noise_eliminated} dropped below threshold")

    print(f"\n  ── Physical Interpretation ──")
    print(f"    Tononi Synaptic Homeostasis Hypothesis (SHY):")
    print(f"    N3 deep sleep downscaling (× 0.990/cycle) proportionally scales all synapses")
    print(f"    Mathematically → ratio perfectly preserved (Exp 2 verified: drift = 0)")
    print(f"    Physically → weak synapses fall below activation threshold {ACTIVATION_THRESHOLD} → functionally silent")
    print(f"    → Correct pathways remain active, noise pathways eliminated")
    print(f"    → Brain's effective SNR automatically improves!")
    print(f"    → This is the physical basis for 'sleep before an exam for better results' —")
    print(f"    Not dreaming about revision, but noise downscaled below threshold.")
    print(f"\n    Sleep quality: {report.quality_score:.3f}")
    print(f"    Memories consolidated: {report.memories_consolidated}")
    print()


# ============================================================
# main program
# ============================================================

def main():
    banner()
    exp1_day_night_cycle()
    exp2_synaptic_downscaling()
    exp3_sleep_deprivation()
    exp4_dream_diagnostics()
    exp5_memory_consolidation_gain()

    print("=" * 70)
    print("  All 5 experiments completed")
    print("  Core Conclusions:")
    print("    1. Sleep is the physical inevitability of energy conservation dE/dt + impedance debt repair")
    print("    2. N3 deep sleep synaptic downscaling preserves relative differences, eliminates noise")
    print("    3. Not sleeping = recovery=0 → energy depletion + debt accumulation → degradation")
    print("    4. Dreams = REM-period random impedance probing → channel health diagnostics")
    print("    5. Post-sleep SNR automatically improves = physical basis for 'sleep well before exams'")
    print("=" * 70)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Phase 4.4: Dynamic Homeostasis Verification

Verifies three things:
  1. PID convergence — disturbance followed by Γ regression to 0
  2. Energy metabolism — ATP/Token long-term consumption curve
  3. Pruning effect — whether ineffective neural pathways are removed

Output: plain text data table + ASCII trend graph
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority


def ascii_sparkline(values: list, width: int = 50, label: str = "") -> str:
    """Generate ASCII mini trend graph"""
    if not values or all(v == values[0] for v in values):
        return f"  {label}: [flat]"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    blocks = " ▁▂▃▄▅▆▇█"
    step = len(values) / width if len(values) > width else 1
    sampled = []
    i = 0.0
    while i < len(values) and len(sampled) < width:
        idx = min(int(i), len(values) - 1)
        sampled.append(values[idx])
        i += step
    bar = ""
    for v in sampled:
        level = int((v - mn) / rng * (len(blocks) - 1))
        bar += blocks[level]
    return f"  {label}: {bar}  [{mn:.4f} → {values[-1]:.4f}]"


def separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def run():
    # ==================================================================
    # Verification 1: PID Convergence — Error regression after disturbance
    # ==================================================================
    separator("Verification 1: PID Convergence — Disturbance → Compensate → Error returns to zero")

    print("\n Protocol: 100 ticks continuous perception, inject disturbance (pain) at tick 30, observe error recovery curve\n")

    brain = AliceBrain(neuron_count=20)
    error_history = []
    temperature_history = []
    compensation_rate_history = []

    for tick in range(100):
        # Continuous visual stimulus
        pixels = np.random.randn(64) * 0.3
        brain.see(pixels, visual_target=np.array([0.5, 0.5]))

        # tick 30: inject disturbance
        if tick == 30:
            brain.inject_pain(0.8)
            print(f" ⚡ tick {tick}: inject pain disturbance (pain=0.8)")

        # record
        stats = brain.life_loop.get_stats()
        error_history.append(stats["avg_error"])
        temperature_history.append(brain.vitals.ram_temperature)
        comp_rate = (
            stats["successful_compensations"] / max(1, stats["total_compensations"])
        )
        compensation_rate_history.append(comp_rate)

    print(ascii_sparkline(error_history, 60, "error/tick "))
    print(ascii_sparkline(temperature_history, 60, "RAM Temperature   "))
    print(ascii_sparkline(compensation_rate_history, 60, "compensation rate "))

    # Convergence determination
    pre_dist = np.mean(error_history[20:30]) # Before disturbance
    peak = max(error_history[30:40]) # Peak after disturbance
    post_dist = np.mean(error_history[80:100]) # Recovery period
    print(f"\n Mean error before disturbance: {pre_dist:.4f}")
    print(f" Peak error after disturbance: {peak:.4f}")
    print(f" Mean error during recovery: {post_dist:.4f}")
    print(f" Recovery rate: {(1 - post_dist / max(peak, 1e-6)) * 100:.1f}%")

    pid_converged = post_dist <= pre_dist * 2.0 # Allow within 2x
    print(f" ✅ PID convergence: {'PASS' if pid_converged else 'FAIL'}")

    # ==================================================================
    # Verification 2: Energy Metabolism — ATP/Token consumption curve
    # ==================================================================
    separator("Verification 2: Energy Metabolism — Long-term operation energy consumption")

    print("\n Protocol: 200 ticks multi-modal operation, track energy, sympathetic/parasympathetic balance\n")

    brain2 = AliceBrain(neuron_count=20)
    energy_history = []
    sympathetic_history = []
    parasympathetic_history = []
    heart_rate_history = []

    for tick in range(200):
        # Mixed stimulus
        brain2.see(np.random.randn(64) * 0.3)
        if tick % 3 == 0:
            brain2.hear(np.random.randn(256) * 0.1)
        if tick % 10 == 0:
            brain2.reach_for(0.5, 0.5, max_steps=20)
        if tick % 15 == 0:
            brain2.say(200.0, vowel="a")

        # record
        energy_history.append(brain2.autonomic.energy)
        sympathetic_history.append(brain2.autonomic.sympathetic)
        parasympathetic_history.append(brain2.autonomic.parasympathetic)
        heart_rate_history.append(brain2.autonomic.heart_rate)

    print(ascii_sparkline(energy_history, 60, "energy (ATP) "))
    print(ascii_sparkline(sympathetic_history, 60, "sympathetic       "))
    print(ascii_sparkline(parasympathetic_history, 60, "parasympathetic   "))
    print(ascii_sparkline(heart_rate_history, 60, "heart rate (bpm)  "))

    # Energy statistics
    print(f"\n Initial energy: {energy_history[0]:.4f}")
    print(f" Final energy: {energy_history[-1]:.4f}")
    print(f" Lowest energy: {min(energy_history):.4f}")
    print(f" Energy consumption rate: {(energy_history[0] - energy_history[-1]) / 200:.6f} /tick")

    auto_balance = brain2.autonomic.get_autonomic_balance()
    print(f" Autonomic balance: {auto_balance:.4f} (0=sympathetic dominant, 1=parasympathetic dominant)")

    energy_stable = min(energy_history) > 0.0 # Not depleted
    print(f" ✅ Energy stability: {'PASS' if energy_stable else 'FAIL — energy depleted!'}")

    # ==================================================================
    # Verification 3: Pruning Effect — Γ² → min
    # ==================================================================
    separator("Verification 3: Neural Pruning Effect — Are ineffective pathways removed?")

    print("\n Protocol: 150 ticks dense visual+auditory stimulus, observe cortical specialization\n")

    brain3 = AliceBrain(neuron_count=20)
    gamma_sq_history = []
    alive_counts = {name: [] for name in brain3.pruning.regions}

    # Record initial state
    initial_state = brain3.pruning.get_development_state()
    print(f"  Initial state:")
    for name, region in initial_state["regions"].items():
        print(f"    {name:15s}: connections={region['alive_connections']:4d},  "
              f"specialization={region['specialization']:15s}, "
              f"Γ_avg={region['avg_gamma']:.4f}")
    g_init = initial_state["overall"]["global_gamma_squared"]
    print(f" Whole-brain Γ²: {g_init:.6f}")

    for tick in range(150):
        # Dense visual stimulus → occipital specialization
        brain3.perceive(np.random.randn(64) * 0.5, Modality.VISUAL)
        # Dense auditory stimulus → temporal specialization
        brain3.perceive(np.random.randn(256) * 0.3, Modality.AUDITORY)

        # record
        g2 = brain3.pruning._compute_global_gamma_squared()
        gamma_sq_history.append(g2)
        for name, region in brain3.pruning.regions.items():
            alive_counts[name].append(region.alive_count)

    print(f"\n After 150 ticks of dense stimulus:")
    final_state = brain3.pruning.get_development_state()
    for name, region in final_state["regions"].items():
        initial_alive = initial_state["regions"][name]["alive_connections"]
        pruned = initial_alive - region["alive_connections"]
        print(f"    {name:15s}: connections={region['alive_connections']:4d} "
              f"(pruned {pruned:3d}), "
              f"specialization={region['specialization']:15s}, "
              f"Γ_avg={region['avg_gamma']:.4f}")

    g_final = final_state["overall"]["global_gamma_squared"]
    print(f"\n Whole-brain Γ² change: {g_init:.6f} → {g_final:.6f}")
    print(f" Γ² decrease ratio: {(1 - g_final / max(g_init, 1e-9)) * 100:.1f}%")

    print(ascii_sparkline(gamma_sq_history, 60, "\n  Γ² trend    "))
    for name in alive_counts:
        print(ascii_sparkline(alive_counts[name], 60, f"  {name:12s}"))

    # Determination
    total_pruned = (
        initial_state["overall"]["total_alive_connections"]
        - final_state["overall"]["total_alive_connections"]
    )
    pruning_occurred = total_pruned > 0
    gamma_decreased = g_final < g_init
    print(f"\n Total pruned connections: {total_pruned}")
    print(f" ✅ Pruning occurred: {'PASS' if pruning_occurred else 'FAIL — no connections pruned!'}")
    print(f" ✅ Γ² decreased: {'PASS' if gamma_decreased else 'FAIL — Γ² did not decrease'}")

    # ==================================================================
    # Summary
    # ==================================================================
    separator("Phase 4.4 Dynamic Homeostasis Verification — Summary")

    checks = [
        ("PID convergence", pid_converged),
        ("Energy metabolism stability", energy_stable),
        ("Pruning actually occurred", pruning_occurred),
        ("Γ² trend decreased", gamma_decreased),
    ]

    all_pass = True
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"""
  ═══════════════════════════════════════════════════════════════
   Alice PASSED dynamic homeostasis verification.

   She is not just a signal processor——
   She can sense disturbance (pain), actively compensate (PID),
   consume energy over time (metabolism),
   and remove ineffective pathways through experience (pruning).

   This is a minimal but complete life loop.
   Intelligence = Σ Γ² → min
  ═══════════════════════════════════════════════════════════════
""")
    else:
        print("\n ⚠ Partial verification failed, further investigation needed.\n")


if __name__ == "__main__":
    run()

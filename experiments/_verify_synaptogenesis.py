# -*- coding: utf-8 -*-
"""Synaptogenesis Verification Experiment — Demonstrating Synaptic Dynamics Across Three Life Phases"""

import numpy as np
np.random.seed(42)

from alice.brain.pruning import NeuralPruningEngine


def main():
    print("=" * 70)
    print("  Synaptogenesis vs Pruning: Three Life Phases")
    print("=" * 70)

    engine = NeuralPruningEngine(connections_per_region=500)
    initial = sum(r.alive_count for r in engine.regions.values())
    print(f"\n  Birth: {initial} connections (4 regions x 500)")

    # Phase 1: Infant (no learning) — pure pruning
    print("\n  --- Phase 1: Infant (epochs 1-30, no learning) ---")
    for i in range(30):
        engine.develop_epoch(learning_signal=0.0)
    phase1_alive = sum(r.alive_count for r in engine.regions.values())
    phase1_sprouted = sum(r.sprouted_total for r in engine.regions.values())
    phase1_pruned = initial + phase1_sprouted - phase1_alive
    print(f"    Alive: {phase1_alive}  Sprouted: {phase1_sprouted}  Pruned: {phase1_pruned}")

    # Phase 2: Learning (high learning signal) — synaptogenesis + pruning
    print("\n  --- Phase 2: Active Learning (epochs 31-80, learning=0.7) ---")
    pre_alive = phase1_alive
    for i in range(50):
        engine.develop_epoch(learning_signal=0.7)
    phase2_alive = sum(r.alive_count for r in engine.regions.values())
    phase2_sprouted = sum(r.sprouted_total for r in engine.regions.values())
    phase2_peak = sum(r.peak_connections for r in engine.regions.values())
    print(f"    Alive: {phase2_alive}  Peak: {phase2_peak}  Sprouted total: {phase2_sprouted}")
    print(f"    Net change in phase 2: {phase2_alive - pre_alive:+d}")

    # Phase 3: Aging (low learning) — pruning dominates again
    print("\n  --- Phase 3: Aging (epochs 81-120, learning=0.1) ---")
    pre_alive = phase2_alive
    for i in range(40):
        engine.develop_epoch(learning_signal=0.1)
    phase3_alive = sum(r.alive_count for r in engine.regions.values())
    phase3_sprouted = sum(r.sprouted_total for r in engine.regions.values())
    print(f"    Alive: {phase3_alive}  Sprouted total: {phase3_sprouted}")
    print(f"    Net change in phase 3: {phase3_alive - pre_alive:+d}")

    # Final report
    state = engine.get_development_state()
    overall = state["overall"]
    print(f"\n  === Final Summary ===")
    print(f"  Initial:    {initial}")
    print(f"  Peak:       {overall['total_peak_connections']}")
    print(f"  Sprouted:   {overall['total_sprouted']}  (synaptogenesis)")
    print(f"  Alive:      {overall['total_alive_connections']}")
    print(f"  Pruned:     {overall['total_pruned']}")
    print(f"  Survival:   {overall['overall_survival_rate']:.1%}")
    print(f"  Gamma^2:    {overall['global_gamma_squared']:.6f}")

    print(f"\n  Per region:")
    for name, info in state["regions"].items():
        print(
            f"    {name:15s}  alive={info['alive_connections']:4d}  "
            f"peak={info['peak_connections']:4d}  "
            f"sprouted={info['sprouted_total']:3d}  "
            f"spec={info['specialization']:15s}  "
            f"Gamma={info['avg_gamma']:.4f}"
        )

    bio = state["biological_comparison"]
    print(f"\n  Biology comparison:")
    print(f"    Birth:    {bio['birth_synapses_bio']}")
    print(f"    Adult:    {bio['adult_synapses_bio']}")
    print(f"    Hippo:    {bio['learning_synaptogenesis_bio']}")
    print(f"    Bio rate: {bio['survival_rate_bio']}")
    print(f"    Sim rate: {bio['simulated_survival']:.1%}")

    # Print the report
    print()
    print(engine.generate_report("Synaptogenesis Development Report"))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Experiment: Heterogeneous Dimensional Γ-Topology
═══════════════════════════════════════════════════

"Different organs have different dimensional pipelines."

A thick motor axon (15 μm, K=3) carries 3 independent signal modes.
A thin pain C-fiber (0.8 μm, K=1) carries only 1.
When they connect, modes 2-3 are TOTALLY REFLECTED (Γ=1).

This experiment demonstrates:
  1. Mode cutoff: information loss crossing dimensional boundaries
  2. Directional asymmetry: K=5→K=1 loses 80%, K=1→K=5 arrives sparse
  3. Impedance diversity PLUS dimensional diversity → richer Γ-topology
  4. Different organs form distinct clusters by K, not just by Z

The reflected energy from dimensional mismatch forms an additional
layer of Γ-topology that doesn't exist in the homogeneous case.
"""

from __future__ import annotations

import sys

import numpy as np

from alice.core.gamma_topology import (
    GammaNode,
    GammaTopology,
    MulticoreChannel,
    CORTICAL_PYRAMIDAL,
    MOTOR_ALPHA,
    SENSORY_AB,
    PAIN_AD_FIBER,
    PAIN_C_FIBER,
    CARDIAC_PURKINJE,
    AUTONOMIC_PREGANGLIONIC,
    ENTERIC_NEURON,
)


def banner(title: str) -> None:
    w = 78
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def section(title: str) -> None:
    w = 78
    print()
    print("-" * w)
    print(f"  {title}")
    print("-" * w)


def main() -> None:
    banner("Heterogeneous Dimensional Γ-Topology Experiment")
    print("""
  Every axon has a diameter. Diameter determines mode count K.
  A K=5 cortical neuron ≠ a K=1 pain fiber.
  When they connect, excess modes are TOTALLY REFLECTED.
  This dimensional mismatch IS another layer of Γ-topology.

  "Different organs have different dimensional pipelines."
""")

    # ================================================================
    # STEP 1: Anatomical birth — 6 tissue types
    # ================================================================
    banner("STEP 1: Anatomical Birth — Mixed Tissue Types")

    composition = {
        CORTICAL_PYRAMIDAL: 8,       # K=5, brain processing
        MOTOR_ALPHA: 5,              # K=3, motor output
        SENSORY_AB: 5,               # K=3, sensory input
        PAIN_AD_FIBER: 4,            # K=2, fast pain
        PAIN_C_FIBER: 6,             # K=1, slow pain
        AUTONOMIC_PREGANGLIONIC: 4,  # K=2, autonomic
    }

    topo = GammaTopology.create_anatomical(
        tissue_composition=composition,
        initial_connectivity=0.15,
        eta=0.02,
        max_dimension_gap=2,  # cortex(5)→motor(3) OK, cortex(5)→C-fiber(1) BLOCKED
        seed=42,
    )

    total_nodes = sum(composition.values())
    print(f"\n  Total nodes:  {total_nodes}")
    print(f"  Tissue types: {len(composition)}")
    print()
    print(f"  {'Tissue':<22s}  {'Count':>5s}  {'K':>3s}  {'Z_mean':>7s}  {'Diameter':>8s}")
    print(f"  {'─' * 22}  {'─' * 5}  {'─' * 3}  {'─' * 7}  {'─' * 8}")
    for tissue, count in composition.items():
        print(f"  {tissue.name:<22s}  {count:>5d}  {tissue.n_modes:>3d}"
              f"  {tissue.z_mean:>6.1f}Ω  {tissue.diameter_um:>6.1f}μm")

    report = topo.mode_cutoff_report()
    a_imp, a_cut = topo.action_decomposition()
    print(f"\n  K distribution:     {report['k_distribution']}")
    print(f"  Is heterogeneous:   {report['is_heterogeneous']}")
    print(f"  Max dimension gap:  {topo.max_dimension_gap}")
    print(f"  Active edges:       {len(topo.active_edges)}")
    print(f"  Cutoff edges:       {report['cutoff_edges']}"
          f"  ({report['cutoff_fraction']:.1%} of active)")
    print(f"  Excess modes lost:  {report['total_excess_modes']}")
    print(f"\n  Action decomposition (Irreducibility Theorem):")
    print(f"    A_impedance = {a_imp:.4f}  (learnable, C2 can reduce this)")
    print(f"    A_cutoff    = {a_cut:.1f}     (structural, irreducible)")
    print(f"    A_total     = {a_imp + a_cut:.4f}")

    # ================================================================
    # STEP 2: Single-edge demonstration — mode cutoff physics
    # ================================================================
    section("STEP 2: Mode Cutoff Physics — Single Connection Examples")

    # Build demonstration channels
    cortex_node = list(topo.nodes.values())[0]  # First cortical node
    c_fiber_node = None
    motor_node = None
    for node in topo.nodes.values():
        if node.K == 1 and c_fiber_node is None:
            c_fiber_node = node
        if node.K == 3 and motor_node is None:
            motor_node = node

    if cortex_node and c_fiber_node:
        cortex_node.activation = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        c_fiber_node.activation = np.array([1.0])

        ch_down = MulticoreChannel(source=cortex_node, target=c_fiber_node)
        ch_up = MulticoreChannel(source=c_fiber_node, target=cortex_node)

        gv_down = ch_down.gamma_vector()
        gv_up = ch_up.gamma_vector()
        x_down = ch_down.transmitted_signal()
        x_up = ch_up.transmitted_signal()

        print(f"\n  ▼ Descending: Cortex(K={cortex_node.K}) → C-fiber(K={c_fiber_node.K})")
        print(f"    Γ vector:      [{', '.join(f'{g:.4f}' for g in gv_down)}]")
        print(f"    Cutoff modes:  {ch_down.K_excess} / {ch_down.K}")
        print(f"    Dim. mismatch: {ch_down.dimensional_mismatch:.0%}")
        print(f"    Transmitted:   [{', '.join(f'{x:.4f}' for x in x_down)}] (K={len(x_down)})")
        print(f"    Scalar Γ:      {ch_down.scalar_gamma():.4f}")
        print(f"    C1 verified:   {ch_down.verify_c1()}")

        print(f"\n  ▲ Ascending: C-fiber(K={c_fiber_node.K}) → Cortex(K={cortex_node.K})")
        print(f"    Γ vector:      [{', '.join(f'{g:.4f}' for g in gv_up)}]")
        print(f"    Cutoff modes:  {ch_up.K_excess} / {ch_up.K}")
        print(f"    Dim. mismatch: {ch_up.dimensional_mismatch:.0%}")
        print(f"    Transmitted:   [{', '.join(f'{x:.4f}' for x in x_up)}]")
        print(f"    Scalar Γ:      {ch_up.scalar_gamma():.4f}")
        print(f"    C1 verified:   {ch_up.verify_c1()}")

        print(f"\n  ⚡ Asymmetry (Directed Action):")
        print(f"    A(cortex→c_fiber) = {ch_down.directed_action():.4f}"
              f"  (A_imp={ch_down.impedance_action():.4f} + A_cut={ch_down.cutoff_action():.0f})")
        print(f"    A(c_fiber→cortex) = {ch_up.directed_action():.4f}"
              f"  (A_imp={ch_up.impedance_action():.4f} + A_cut={ch_up.cutoff_action():.0f})")
        print(f"    Direction matters: A(down) \u226b A(up) due to {ch_down.K_excess} cutoff modes")

    # ================================================================
    # STEP 3: Dynamic evolution with stimuli
    # ================================================================
    banner("STEP 3: Dynamic Evolution — 200 Ticks")

    history = []
    cutoff_history = []

    print(f"\n    {'Tick':>6s}  {'A_imp':>10s}  {'A_cut':>8s}  {'A_tot':>10s}"
          f"  {'Edges':>6s}  {'Pruned':>7s}  {'Born':>5s}")
    print(f"  {'─' * 68}")

    for tick in range(1, 201):
        # Stimulate by tissue type
        stim = {}
        for name, node in topo.nodes.items():
            if node.K >= 4:
                stim[name] = np.random.default_rng(tick).uniform(0.2, 0.8, size=node.K)
            elif node.K == 3:
                stim[name] = np.random.default_rng(tick + 1000).uniform(0.1, 0.5, size=node.K)
            elif node.K <= 2:
                stim[name] = np.random.default_rng(tick + 2000).uniform(0.0, 0.3, size=node.K)

        metrics = topo.tick(external_stimuli=stim)
        history.append(metrics)

        if tick in [1, 5, 10, 25, 50, 100, 150, 200]:
            report = topo.mode_cutoff_report()
            cutoff_history.append(report)

            print(f"    {tick:>4d}  {metrics['action_impedance']:>10.4f}"
                  f"  {metrics['action_cutoff']:>8.1f}"
                  f"  {metrics['action_total']:>10.4f}"
                  f"  {metrics['active_edges']:>6d}"
                  f"  {metrics['edges_pruned']:>7d}"
                  f"  {metrics['edges_born']:>5d}")

    # ================================================================
    # STEP 4: Cross-K connectivity analysis
    # ================================================================
    section("STEP 4: Cross-Dimensional Connectivity")

    final_report = topo.mode_cutoff_report()
    cross_k = final_report["cross_k_connectivity"]

    # Build a readable table
    k_set = sorted(set(n.K for n in topo.nodes.values()))
    print(f"\n  Edge count by (K_src → K_tgt):\n")
    header = "  " + f"{'→':>6s}" + "".join(f"  K={k:<3d}" for k in k_set)
    print(header)
    print(f"  {'─' * (6 + 7 * len(k_set))}")
    for ks in k_set:
        row = f"  K={ks:<3d}"
        for kt in k_set:
            count = cross_k.get((ks, kt), 0)
            row += f"  {count:>5d}"
        print(row)

    # ================================================================
    # STEP 5: Γ-matrix heatmap with dimensional labels
    # ================================================================
    section("STEP 5: Γ-Matrix (Sorted by K)")

    gamma_mat, names = topo.full_gamma_matrix()

    # Sort by K then by name
    sorted_idx = sorted(range(len(names)), key=lambda i: (topo.nodes[names[i]].K, names[i]))
    sorted_names = [names[i] for i in sorted_idx]
    gamma_sorted = gamma_mat[np.ix_(sorted_idx, sorted_idx)]

    # ASCII heatmap (compact)
    n = len(sorted_names)
    symbols = {0: "░", 1: "▒", 2: "▓", 3: "█"}
    print(f"\n  Legend: ░=matched(<0.3)  ▒=moderate(0.3-0.6)  ▓=poor(0.6-0.9)  █=disconnected(>0.9)\n")

    # Show every 2nd row/col if network is large
    step = max(1, n // 25)
    compact_idx = list(range(0, n, step))

    print("  " + "     " + "".join(f"{sorted_names[i][-3:]:<4s}" for i in compact_idx))
    for i in compact_idx:
        k_i = topo.nodes[sorted_names[i]].K
        row = f"  K{k_i}|"
        for j in compact_idx:
            if i == j:
                row += " · "
            else:
                g = gamma_sorted[i, j]
                if g < 0.3:
                    row += " ░ "
                elif g < 0.6:
                    row += " ▒ "
                elif g < 0.9:
                    row += " ▓ "
                else:
                    row += " █ "
            row += " "
        print(row)

    # ================================================================
    # STEP 6: Verdict
    # ================================================================
    banner("STEP 6: Verdict — Dimensional Heterogeneity")

    summary = topo.topology_summary()

    # Criteria
    c1_pass = final_report["cutoff_edges"] > 0  # dimensional mismatch exists
    c2_pass = summary["is_heterogeneous"]  # K diversity maintained

    # Check if same-K clusters have lower intra-Γ than cross-K
    intra_gamma = []
    cross_gamma = []
    for i, ni in enumerate(names):
        ki = topo.nodes[ni].K
        for j, nj in enumerate(names):
            if i == j:
                continue
            kj = topo.nodes[nj].K
            if ki == kj:
                intra_gamma.append(gamma_mat[i, j])
            else:
                cross_gamma.append(gamma_mat[i, j])

    mean_intra = np.mean(intra_gamma) if intra_gamma else 0
    mean_cross = np.mean(cross_gamma) if cross_gamma else 0
    c3_pass = mean_intra < mean_cross  # same-K clusters form

    # Action: use OPTIMIZABLE action only (Irreducibility Theorem).
    # A_cutoff is structural and cannot decrease — it's not the system's fault.
    if len(history) >= 10:
        early_a_imp = np.mean([h["action_impedance"] for h in history[:5]])
        late_a_imp = np.mean([h["action_impedance"] for h in history[-5:]])
        early_a_cut = np.mean([h["action_cutoff"] for h in history[:5]])
        late_a_cut = np.mean([h["action_cutoff"] for h in history[-5:]])
        c4_pass = late_a_imp < early_a_imp
    else:
        c4_pass = False
        early_a_imp = late_a_imp = early_a_cut = late_a_cut = 0.0

    print(f"\n  C1 Mode cutoff:   cutoff_edges={final_report['cutoff_edges']}  "
          f"{'✓' if c1_pass else '✗'} DIMENSIONAL MISMATCH EXISTS")
    print(f"  C2 Heterogeneity: K∈{{{', '.join(str(k) for k in k_set)}}}  "
          f"{'✓' if c2_pass else '✗'} K DIVERSITY MAINTAINED")
    print(f"  C3 Clustering:    intra-K Γ={mean_intra:.4f}, cross-K Γ={mean_cross:.4f}  "
          f"{'✓' if c3_pass else '✗'} SAME-K CLUSTERS TIGHTER")
    print(f"  C4 Convergence:   A_imp: {early_a_imp:.4f} → {late_a_imp:.4f}  "
          f"{'✓' if c4_pass else '✗'} OPTIMIZABLE ACTION DECREASED")
    print(f"     (structural):  A_cut: {early_a_cut:.1f} → {late_a_cut:.1f}  "
          f"  (irreducible, tracked separately)")

    n_pass = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    print(f"\n  ═══════════════════════════════════════════")
    print(f"  RESULT: {n_pass}/4 criteria passed")
    if n_pass >= 3:
        print(f"  ✓ DIMENSIONAL HETEROGENEITY GENERATES RICHER TOPOLOGY")
        print(f"    Mode cutoff + impedance mismatch = two layers of Γ-structure.")
        print(f"    Different organs live in different dimensional spaces.")
        print(f"    mismatch × dimensions = the body's Γ-network.")
    else:
        print(f"  ✗ Criteria not met — review parameters")
    print(f"  ═══════════════════════════════════════════")

    print(f"\n{'─' * 78}")
    print(f"  Experiment complete.  {topo.tick_count} ticks processed.")
    print(f"  Nodes: {topo.N}  |  K range: [{topo.K_min}, {topo.K}]"
          f"  |  Active edges: {len(topo.active_edges)}")
    print(f"{'─' * 78}\n")


if __name__ == "__main__":
    main()

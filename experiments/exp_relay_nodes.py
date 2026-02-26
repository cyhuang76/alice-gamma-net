# -*- coding: utf-8 -*-
"""
Experiment: Dimension Gradient Minimisation — Relay Nodes
═════════════════════════════════════════════════════════

"Why does the brain have 6 cortical layers?"
"Because each layer is a dimensional buffer."

From the Irreducibility Theorem (2026-02-26):
  A_cut = Σ (K_src - K_tgt)⁺   is invariant under all gradients.

The ONLY way to reduce per-hop cutoff cost is to insert relay nodes
with intermediate K values.  This is the Dimension Gradient Minimisation
Path:

    K_relay = argmin_K [ (K_src - K)⁺ + (K - K_tgt)⁺ ]

Solution: K_relay ∈ [K_tgt, K_src], any integer.

This experiment demonstrates:
  1. Direct K=5 → K=1: A_cut/hop = 4 (devastating)
  2. With relay K=3: A_cut/hop = max(2, 2) = 2 per hop
  3. With relay chain K=4,3,2: A_cut/hop = 1 per hop
  4. Total A_cut is conserved (always = K_src - K_tgt)
  5. Relays enable information routing at each stage
  6. Hebbian learning optimises A_imp at each relay stage independently
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
)


def banner(title: str) -> None:
    w = 78
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def section(title: str) -> None:
    print(f"\n{'─' * 78}")
    print(f"  {title}")
    print(f"{'─' * 78}")


def main():
    banner("Dimension Gradient Minimisation — Relay Node Experiment")

    print("""
  From the Irreducibility Theorem:
    A_cut = Σ (K_src - K_tgt)⁺  is geometric, not learnable.

  The only way to reduce per-hop cost: INSERT RELAY NODES.
  Each relay has K ∈ [K_tgt, K_src], acting as a dimensional buffer.

  This is why the brain has layered architecture:
    Cortex(K=5) → Thalamus(K=3) → Spinal cord(K=2) → C-fiber(K=1)
""")

    # ================================================================
    # STEP 1: Direct connection — the problem
    # ================================================================
    banner("STEP 1: Direct Connection K=5 → K=1 (The Problem)")

    cortex = GammaNode("cortex", impedance=np.array([80, 70, 90, 60, 85.0]),
                       activation=np.array([1.0, 0.8, 0.6, 0.4, 0.2]))
    c_fiber = GammaNode("c_fiber", impedance=np.array([120.0]),
                         activation=np.zeros(1))

    ch_direct = MulticoreChannel(source=cortex, target=c_fiber)

    print(f"  Cortex K={cortex.K}, C-fiber K={c_fiber.K}")
    print(f"  Direct connection:")
    print(f"    A_cut   = {ch_direct.cutoff_action():.0f}  "
          f"(4 modes totally reflected)")
    print(f"    A_imp   = {ch_direct.impedance_action():.4f}")
    print(f"    A_total = {ch_direct.directed_action():.4f}")
    print(f"    Scalar Γ = {ch_direct.scalar_gamma():.4f}")
    print(f"    Per-hop cutoff = {ch_direct.cutoff_action():.0f}")

    # ================================================================
    # STEP 2: Optimal relay chain computation
    # ================================================================
    banner("STEP 2: Optimal Relay Chains")

    for max_gap in [3, 2, 1]:
        relays = GammaTopology.optimal_relay_chain(5, 1, max_gap=max_gap)
        chain_k = [5] + relays + [1]
        hops = len(chain_k) - 1
        max_hop_cut = max(chain_k[i] - chain_k[i + 1]
                          for i in range(hops))
        print(f"\n  max_gap={max_gap}:  chain K = {' → '.join(str(k) for k in chain_k)}")
        print(f"    Hops = {hops},  max per-hop cutoff = {max_hop_cut}")
        print(f"    Total cutoff = {sum(max(0, chain_k[i] - chain_k[i+1]) for i in range(hops))}")

    # ================================================================
    # STEP 3: Build relay chain and compare
    # ================================================================
    banner("STEP 3: Relay Chain — Side-by-Side Comparison")

    # Scenario A: Direct (impossible with max_gap=2)
    section("A) Direct: Cortex(K=5) → C-fiber(K=1)")

    nodes_a = [
        GammaNode("cortex_a", impedance=np.array([80, 70, 90, 60, 85.0]),
                   activation=np.ones(5) * 0.5),
        GammaNode("c_fiber_a", impedance=np.array([120.0]),
                   activation=np.zeros(1)),
    ]
    topo_a = GammaTopology(nodes=nodes_a, eta=0.05, max_dimension_gap=None)
    topo_a.activate_edge("cortex_a", "c_fiber_a")

    ch_a = topo_a.active_edges[("cortex_a", "c_fiber_a")]
    print(f"  A_imp = {ch_a.impedance_action():.4f}")
    print(f"  A_cut = {ch_a.cutoff_action():.0f}")
    print(f"  A_tot = {ch_a.directed_action():.4f}")
    print(f"  Per-hop cutoff = 4  (devastating)")

    # Scenario B: With relay (max_gap=2)
    section("B) With Relay: Cortex(K=5) → Relay(K=3) → C-fiber(K=1)")

    nodes_b = [
        GammaNode("cortex_b", impedance=np.array([80, 70, 90, 60, 85.0]),
                   activation=np.ones(5) * 0.5),
        GammaNode("c_fiber_b", impedance=np.array([120.0]),
                   activation=np.zeros(1)),
    ]
    topo_b = GammaTopology(nodes=nodes_b, eta=0.05, max_dimension_gap=2)
    relay_names = topo_b.insert_relay_nodes("cortex_b", "c_fiber_b", seed=42)

    chain_b = ["cortex_b"] + relay_names + ["c_fiber_b"]
    chain_k_b = [topo_b.nodes[n].K for n in chain_b]

    print(f"  Relay chain: {' → '.join(f'{n}(K={topo_b.nodes[n].K})' for n in chain_b)}")
    print(f"  K sequence: {' → '.join(str(k) for k in chain_k_b)}")

    path_imp, path_cut = topo_b.relay_path_cutoff(chain_b)
    print(f"\n  Path A_imp = {path_imp:.4f}")
    print(f"  Path A_cut = {path_cut:.0f}  (same total! cutoff is conserved)")

    print(f"\n  Per-hop breakdown:")
    for i in range(len(chain_b) - 1):
        key = (chain_b[i], chain_b[i + 1])
        ch = topo_b.active_edges[key]
        print(f"    {chain_b[i]}(K={topo_b.nodes[chain_b[i]].K}) → "
              f"{chain_b[i+1]}(K={topo_b.nodes[chain_b[i+1]].K}): "
              f"A_cut={ch.cutoff_action():.0f}, A_imp={ch.impedance_action():.4f}")

    # ================================================================
    # STEP 4: Full anatomical network with auto-relays
    # ================================================================
    banner("STEP 4: Full Anatomical Network with Auto-Relay Insertion")

    print("  Strategy: create network WITHOUT gap enforcement (natural wiring),")
    print("  then insert relays to bridge dimensional gaps, then enforce gap.\n")

    # Phase 1: natural connectivity — no gap restriction
    topo = GammaTopology.create_anatomical(
        tissue_composition={
            CORTICAL_PYRAMIDAL: 6,
            MOTOR_ALPHA: 4,
            SENSORY_AB: 4,
            PAIN_AD_FIBER: 4,
            PAIN_C_FIBER: 6,
        },
        initial_connectivity=0.15,
        eta=0.05,
        max_dimension_gap=None,  # allow all connections initially
        seed=42,
    )

    n_before = topo.N
    edges_before = len(topo.active_edges)
    a_imp_before, a_cut_before = topo.action_decomposition()

    # Count gap-violating edges
    gap_violating = sum(1 for ch in topo.active_edges.values()
                        if ch.dimension_gap > 2)

    print(f"  Before relay insertion (no gap enforcement):")
    print(f"    Nodes: {n_before}")
    print(f"    Edges: {edges_before}")
    print(f"    Gap-violating edges (gap>2): {gap_violating}")
    print(f"    A_imp = {a_imp_before:.4f}")
    print(f"    A_cut = {a_cut_before:.0f}")

    # Phase 2: set gap limit and insert relays
    topo.max_dimension_gap = 2
    relay_result = topo.insert_all_relays(seed=100)

    n_after = topo.N
    edges_after = len(topo.active_edges)
    a_imp_after, a_cut_after = topo.action_decomposition()

    print(f"\n  After relay insertion:")
    print(f"    Nodes: {n_after}  (+{n_after - n_before} relay nodes)")
    print(f"    Edges: {edges_after}")
    print(f"    A_imp = {a_imp_after:.4f}")
    print(f"    A_cut = {a_cut_after:.0f}")
    print(f"    Relays inserted for {len(relay_result)} edge pairs")

    for (src, tgt), relays in list(relay_result.items())[:5]:
        chain = [src] + relays + [tgt]
        chain_ks = [topo.nodes[n].K for n in chain]
        print(f"    {src}(K={topo.nodes[src].K}) → {tgt}(K={topo.nodes[tgt].K}): "
              f"relays={[f'{r}(K={topo.nodes[r].K})' for r in relays]}")

    # ================================================================
    # STEP 5: Evolution — does relay improve learning?
    # ================================================================
    banner("STEP 5: Evolution with Relay Chain — 100 Ticks")

    history = []
    print(f"\n    {'Tick':>6s}  {'A_imp':>10s}  {'A_cut':>8s}  {'A_tot':>10s}"
          f"  {'Edges':>6s}  {'Nodes':>6s}")
    print(f"  {'─' * 56}")

    for tick in range(1, 101):
        stim = {}
        for name, node in topo.nodes.items():
            stim[name] = np.random.default_rng(tick).uniform(0.1, 0.5, size=node.K)

        metrics = topo.tick(external_stimuli=stim, enable_spontaneous=False)
        history.append(metrics)

        if tick in [1, 5, 10, 25, 50, 100]:
            print(f"    {tick:>4d}  {metrics['action_impedance']:>10.4f}"
                  f"  {metrics['action_cutoff']:>8.1f}"
                  f"  {metrics['action_total']:>10.4f}"
                  f"  {metrics['active_edges']:>6d}"
                  f"  {topo.N:>6d}")

    # ================================================================
    # STEP 6: Verdict
    # ================================================================
    banner("STEP 6: Dimension Gradient Minimisation — Verdict")

    # C1: Relay nodes exist with intermediate K
    relay_nodes = [n for n in topo.nodes.values()
                   if n.name.startswith("relay_")]
    relay_ks = {n.K for n in relay_nodes}
    c1_pass = len(relay_nodes) > 0

    # C2: All edges respect max_dimension_gap
    c2_pass = all(ch.dimension_gap <= topo.max_dimension_gap
                  for ch in topo.active_edges.values())

    # C3: A_imp decreased
    if len(history) >= 10:
        early_imp = np.mean([h["action_impedance"] for h in history[:5]])
        late_imp = np.mean([h["action_impedance"] for h in history[-5:]])
        c3_pass = late_imp < early_imp
    else:
        c3_pass = False
        early_imp = late_imp = 0.0

    # C4: Relay chain total cutoff = K_src - K_tgt (conservation)
    cutoff_conserved = True
    for (src, tgt), relays in relay_result.items():
        chain = [src] + relays + [tgt]
        K_s = topo.nodes[src].K
        K_t = topo.nodes[tgt].K
        _, path_cut = topo.relay_path_cutoff(chain)
        expected_cut = max(0, K_s - K_t)
        if abs(path_cut - expected_cut) > 0.01:
            cutoff_conserved = False
            break
    c4_pass = cutoff_conserved

    print(f"\n  C1 Relay existence: {len(relay_nodes)} relay nodes, K∈{relay_ks}  "
          f"{'✓' if c1_pass else '✗'} DIMENSIONAL ADAPTORS CREATED")
    print(f"  C2 Gap constraint:  all edges gap ≤ {topo.max_dimension_gap}  "
          f"{'✓' if c2_pass else '✗'} HIERARCHY ENFORCED")
    print(f"  C3 A_imp learning:  {early_imp:.4f} → {late_imp:.4f}  "
          f"{'✓' if c3_pass else '✗'} IMPEDANCE OPTIMISED PER RELAY")
    print(f"  C4 Cutoff conserv:  A_cut total preserved across chains  "
          f"{'✓' if c4_pass else '✗'} IRREDUCIBILITY VERIFIED")

    n_pass = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    print(f"\n  ═══════════════════════════════════════════")
    print(f"  RESULT: {n_pass}/4 criteria passed")
    if n_pass >= 3:
        print(f"  ✓ DIMENSION GRADIENT MINIMISATION WORKS")
        print(f"    Relay nodes act as dimensional buffers.")
        print(f"    Each relay stage is an independent learning unit.")
        print(f"    Total cutoff is conserved — geometry is honest.")
        print(f"    This is why the brain has layered architecture.")
    else:
        print(f"  ✗ Criteria not met — review relay implementation")
    print(f"  ═══════════════════════════════════════════")

    print(f"\n{'─' * 78}")
    print(f"  Experiment complete.  {topo.tick_count} ticks processed.")
    print(f"  Nodes: {topo.N} ({len(relay_nodes)} relays)"
          f"  |  K range: [{topo.K_min}, {topo.K}]"
          f"  |  Active edges: {len(topo.active_edges)}")
    print(f"{'─' * 78}\n")


if __name__ == "__main__":
    main()

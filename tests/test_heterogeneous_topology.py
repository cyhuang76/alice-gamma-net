# -*- coding: utf-8 -*-
"""
Tests for heterogeneous-K (dimensional mismatch) Γ-topology.

Physics verified (v2 — Dimensional Cost Irreducibility Theorem):
  - Mode cutoff: K_src > K_tgt → excess modes have Γ=1, T=0
  - C1 energy conservation holds for ALL modes (including cutoff)
  - Transmitted signal is K_tgt-dimensional (dimensional compression)
  - Hebbian updates only affect common modes
  - Anatomical factory creates mixed-K networks
  - Cross-dimensional Γ matrix is asymmetric (direction matters)
"""

import numpy as np
import pytest

from alice.core.gamma_topology import (
    GammaNode,
    GammaTopology,
    MulticoreChannel,
    TissueType,
    CORTICAL_PYRAMIDAL,
    MOTOR_ALPHA,
    SENSORY_AB,
    PAIN_AD_FIBER,
    PAIN_C_FIBER,
    CARDIAC_PURKINJE,
    AUTONOMIC_PREGANGLIONIC,
    ENTERIC_NEURON,
    ALL_TISSUE_TYPES,
)


# ============================================================================
# 1. TissueType presets
# ============================================================================


class TestTissueTypes:
    """Verify biological tissue type definitions."""

    def test_all_presets_exist(self):
        assert len(ALL_TISSUE_TYPES) == 8

    def test_mode_diversity(self):
        """Different tissues must have different K."""
        ks = {t.n_modes for t in ALL_TISSUE_TYPES}
        assert len(ks) >= 4, f"Need at least 4 distinct K values, got {ks}"

    def test_c_fiber_is_single_mode(self):
        """C-fiber is the thinnest, must be K=1."""
        assert PAIN_C_FIBER.n_modes == 1
        assert not PAIN_C_FIBER.myelinated

    def test_cortical_is_highest_mode(self):
        """Cortical pyramidal should have the most modes."""
        assert CORTICAL_PYRAMIDAL.n_modes >= max(
            MOTOR_ALPHA.n_modes, SENSORY_AB.n_modes, PAIN_C_FIBER.n_modes
        )

    def test_impedance_ranges_plausible(self):
        """All impedances should be positive and in a plausible range."""
        for t in ALL_TISSUE_TYPES:
            assert t.z_mean > 0
            assert t.z_std > 0
            assert t.diameter_um > 0


# ============================================================================
# 2. Mode projection — the core physics of dimensional mismatch
# ============================================================================


class TestModeProjection:
    """Verify waveguide mode cutoff physics."""

    def test_k5_to_k1_total_cutoff_4_modes(self):
        """K=5 → K=1: modes 2-5 are totally reflected."""
        src = GammaNode("cortex", impedance=np.array([80, 70, 90, 60, 85.0]),
                        activation=np.ones(5))
        tgt = GammaNode("c_fiber", impedance=np.array([120.0]),
                        activation=np.zeros(1))
        ch = MulticoreChannel(source=src, target=tgt)

        gv = ch.gamma_vector()
        assert len(gv) == 5, "gamma_vector must be K_src-dimensional"

        # Mode 0: normal impedance Γ
        expected_g0 = (120.0 - 80.0) / (120.0 + 80.0)
        assert abs(gv[0] - expected_g0) < 1e-10

        # Modes 1-4: total reflection
        for k in range(1, 5):
            assert gv[k] == 1.0, f"Mode {k} should be Γ=1 (total cutoff)"

    def test_k5_to_k1_c1_holds_all_modes(self):
        """C1: Γ² + T = 1 for every mode, including cutoff modes."""
        src = GammaNode("cortex", impedance=np.array([80, 70, 90, 60, 85.0]),
                        activation=np.ones(5))
        tgt = GammaNode("c_fiber", impedance=np.array([120.0]),
                        activation=np.zeros(1))
        ch = MulticoreChannel(source=src, target=tgt)

        gv = ch.gamma_vector()
        tv = ch.transmission_vector()

        for k in range(5):
            residual = abs(gv[k] ** 2 + tv[k] - 1.0)
            assert residual < 1e-10, f"C1 violated at mode {k}: Γ²+T = {gv[k]**2 + tv[k]}"

    def test_transmitted_signal_is_k_tgt_dimensional(self):
        """Signal arriving at K=1 target must be 1-dimensional."""
        src = GammaNode("cortex", impedance=np.array([80, 70, 90.0]),
                        activation=np.array([1.0, 0.5, 0.8]))
        tgt = GammaNode("pain", impedance=np.array([100.0]),
                        activation=np.zeros(1))
        ch = MulticoreChannel(source=src, target=tgt)

        x_out = ch.transmitted_signal()
        assert x_out.shape == (1,), f"Expected shape (1,), got {x_out.shape}"
        assert x_out[0] > 0, "Mode 0 should transmit (impedance-based T > 0)"

    def test_transmitted_signal_k1_to_k5(self):
        """K=1 → K=5: only mode 0 carries signal, modes 1-4 are empty."""
        src = GammaNode("c_fiber", impedance=np.array([120.0]),
                        activation=np.array([1.0]))
        tgt = GammaNode("cortex", impedance=np.array([80, 70, 90, 60, 85.0]),
                        activation=np.zeros(5))
        ch = MulticoreChannel(source=src, target=tgt)

        x_out = ch.transmitted_signal()
        assert x_out.shape == (5,), f"Expected (5,), got {x_out.shape}"
        assert x_out[0] > 0, "Mode 0 should arrive"
        for k in range(1, 5):
            assert x_out[k] == 0.0, f"Mode {k} should be empty (no source signal)"

    def test_dimensional_mismatch_property(self):
        """K=5 → K=1 should report 80% dimensional mismatch."""
        src = GammaNode("s", impedance=np.ones(5) * 80, activation=np.zeros(5))
        tgt = GammaNode("t", impedance=np.ones(1) * 80, activation=np.zeros(1))
        ch = MulticoreChannel(source=src, target=tgt)

        assert ch.K_common == 1
        assert ch.K_excess == 4
        assert abs(ch.dimensional_mismatch - 0.8) < 1e-10

    def test_same_k_no_cutoff(self):
        """Same K: no mode cutoff, all modes are common."""
        src = GammaNode("a", impedance=np.array([50, 60, 70.0]),
                        activation=np.zeros(3))
        tgt = GammaNode("b", impedance=np.array([80, 90, 100.0]),
                        activation=np.zeros(3))
        ch = MulticoreChannel(source=src, target=tgt)

        assert ch.K_common == 3
        assert ch.K_excess == 0
        assert ch.dimensional_mismatch == 0.0

    def test_gamma_matrix_shape_heterogeneous(self):
        """Γ matrix should be K_src × K_tgt (rectangular)."""
        src = GammaNode("s", impedance=np.ones(5) * 80, activation=np.zeros(5))
        tgt = GammaNode("t", impedance=np.ones(2) * 100, activation=np.zeros(2))
        ch = MulticoreChannel(source=src, target=tgt)

        gm = ch.gamma_matrix()
        assert gm.shape == (5, 2), f"Expected (5,2), got {gm.shape}"

    def test_reflected_energy_includes_cutoff_modes(self):
        """Reflected energy must include the totally-reflected excess modes."""
        src = GammaNode("s", impedance=np.array([100.0, 100.0, 100.0]),
                        activation=np.array([0.0, 1.0, 1.0]))
        tgt = GammaNode("t", impedance=np.array([100.0]),
                        activation=np.zeros(1))
        ch = MulticoreChannel(source=src, target=tgt)

        # Mode 0: perfectly matched (Γ=0), activation=0 → 0 reflected
        # Mode 1: Γ=1 (cutoff), activation=1 → 1.0 reflected
        # Mode 2: Γ=1 (cutoff), activation=1 → 1.0 reflected
        # Total: 2.0
        E_ref = ch.reflected_energy()
        assert abs(E_ref - 2.0) < 1e-10, f"Expected 2.0, got {E_ref}"

    def test_asymmetric_gamma_with_dimensional_mismatch(self):
        """Γ(A→B) ≠ Γ(B→A) when K_A ≠ K_B — direction matters."""
        nodeA = GammaNode("A", impedance=np.array([80, 70, 90.0]),
                          activation=np.zeros(3))
        nodeB = GammaNode("B", impedance=np.array([80.0]),
                          activation=np.zeros(1))

        ch_AB = MulticoreChannel(source=nodeA, target=nodeB)
        ch_BA = MulticoreChannel(source=nodeB, target=nodeA)

        # A→B: 3 modes, 2 cutoff, high scalar Γ
        # B→A: 1 mode, 0 cutoff, only impedance mismatch
        gamma_AB = ch_AB.scalar_gamma()
        gamma_BA = ch_BA.scalar_gamma()

        assert gamma_AB > gamma_BA, \
            f"A→B ({gamma_AB:.4f}) should be worse than B→A ({gamma_BA:.4f}) due to mode cutoff"


# ============================================================================
# 3. Heterogeneous topology — network-level behavior
# ============================================================================


class TestHeterogeneousTopology:
    """Verify network-level physics with mixed-K nodes."""

    def test_create_anatomical(self):
        """Anatomical factory should create heterogeneous network."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 5,
                MOTOR_ALPHA: 3,
                PAIN_C_FIBER: 4,
            },
            initial_connectivity=0.3,
            seed=42,
        )

        assert topo.N == 12
        assert topo.is_heterogeneous
        assert topo.K == 5    # max K (cortical)
        assert topo.K_min == 1  # min K (C-fiber)

    def test_c1_every_tick_heterogeneous(self):
        """C1 must hold at every active edge, every tick, even with mixed K."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 3,
                PAIN_C_FIBER: 3,
                MOTOR_ALPHA: 3,
            },
            initial_connectivity=0.5,
            eta=0.05,
            seed=42,
        )

        for tick in range(30):
            stim = {}
            # Stimulate cortical nodes
            for name, node in topo.nodes.items():
                if name.startswith("corti"):
                    stim[name] = np.ones(node.K) * 0.5
            topo.tick(external_stimuli=stim)

            # Verify C1 at every active edge
            for (src, tgt), ch in topo.active_edges.items():
                assert ch.verify_c1(), \
                    f"C1 violated at ({src}→{tgt}) on tick {tick}"

    def test_mode_cutoff_report(self):
        """Mode cutoff report should detect dimensional mismatches."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 3,
                PAIN_C_FIBER: 3,
            },
            initial_connectivity=0.5,
            seed=42,
        )

        report = topo.mode_cutoff_report()
        assert report["is_heterogeneous"]
        assert report["k_min"] == 1
        assert report["k_max"] == 5
        # There must be edges with cutoff (K=5 → K=1)
        assert report["cutoff_edges"] > 0

    def test_tick_with_mixed_k_no_crash(self):
        """Network with K=1,2,3,4,5 nodes should tick without errors."""
        nodes = []
        for k in range(1, 6):
            nodes.append(GammaNode(
                name=f"k{k}_node",
                impedance=np.ones(k) * 75.0,
                activation=np.ones(k) * 0.5,
            ))
        topo = GammaTopology(nodes=nodes, eta=0.01)

        # Activate all-to-all
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i != j:
                    topo.activate_edge(ni.name, nj.name)

        # Run 20 ticks
        for _ in range(20):
            metrics = topo.tick(enable_spontaneous=False)

        assert metrics["tick"] == 20
        assert metrics["active_edges"] == 20  # 5×4

    def test_full_gamma_matrix_with_mixed_k(self):
        """Full Γ matrix should be N×N even with heterogeneous K."""
        nodes = [
            GammaNode("a", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("b", impedance=np.ones(1) * 120, activation=np.zeros(1)),
            GammaNode("c", impedance=np.ones(3) * 60, activation=np.zeros(3)),
        ]
        topo = GammaTopology(nodes=nodes)

        gm, names = topo.full_gamma_matrix()
        assert gm.shape == (3, 3)

        # a→b (K=5→K=1): should have high Γ due to mode cutoff
        # b→a (K=1→K=5): should have lower Γ (only impedance mismatch)
        ia, ib = names.index("a"), names.index("b")
        assert gm[ia, ib] > gm[ib, ia], \
            "K=5→K=1 should have higher Γ than K=1→K=5 (mode cutoff)"

    def test_topology_summary_reports_heterogeneity(self):
        """Summary should include K distribution when heterogeneous."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 2,
                PAIN_C_FIBER: 2,
            },
            seed=42,
        )

        s = topo.topology_summary()
        assert s["is_heterogeneous"]
        assert s["n_modes_max"] == 5
        assert s["n_modes_min"] == 1
        assert 1 in s["k_distribution"]
        assert 5 in s["k_distribution"]


# ============================================================================
# 4. Information flow physics — directional compression
# ============================================================================


class TestInformationFlow:
    """Verify that dimensional mismatch creates directional information filtering."""

    def test_descending_pathway_compression(self):
        """Cortex(K=5) → Motor(K=3) → C-fiber(K=1): progressive compression."""
        cortex = GammaNode("cortex", impedance=np.array([80, 70, 90, 60, 85.0]),
                           activation=np.array([1.0, 0.8, 0.6, 0.4, 0.2]))
        motor = GammaNode("motor", impedance=np.array([50, 55, 60.0]),
                          activation=np.zeros(3))
        c_fiber = GammaNode("c_fiber", impedance=np.array([120.0]),
                            activation=np.zeros(1))

        ch1 = MulticoreChannel(source=cortex, target=motor)
        ch2 = MulticoreChannel(source=motor, target=c_fiber)

        # Stage 1: K=5 → K=3, losing 2 modes
        x1 = ch1.transmitted_signal()
        assert x1.shape == (3,)
        assert ch1.K_excess == 2

        # Update motor activation
        motor.activation = x1

        # Stage 2: K=3 → K=1, losing 2 more modes
        x2 = ch2.transmitted_signal()
        assert x2.shape == (1,)
        assert ch2.K_excess == 2

        # Total: 5D → 1D, massive information compression
        total_loss = (cortex.K - c_fiber.K) / cortex.K
        assert total_loss == 0.8  # 80% of modes lost

    def test_ascending_pathway_sparsity(self):
        """C-fiber(K=1) → Cortex(K=5): signal arrives in 1 mode, 4 empty."""
        c_fiber = GammaNode("c_fiber", impedance=np.array([120.0]),
                            activation=np.array([1.0]))
        cortex = GammaNode("cortex", impedance=np.array([80, 70, 90, 60, 85.0]),
                           activation=np.zeros(5))

        ch = MulticoreChannel(source=c_fiber, target=cortex)
        x_out = ch.transmitted_signal()

        # Only mode 0 has signal
        assert x_out[0] > 0
        # Modes 1-4 are empty (no source signal)
        for k in range(1, 5):
            assert x_out[k] == 0.0

        # This explains why pain is crude: only 1D information arrives at cortex
        sparsity = np.count_nonzero(x_out == 0.0) / len(x_out)
        assert sparsity == 0.8  # 80% of cortical modes receive nothing


# ============================================================================
# 5. Action decomposition — Dimensional Cost Irreducibility Theorem
# ============================================================================


class TestActionDecomposition:
    """
    Verify A(i→j) = A_impedance + A_cutoff  (the Irreducibility Theorem).

    A_impedance = Σ_{k≤K_common} Γ_k²  →  learnable by C2 gradient
    A_cutoff    = (K_src - K_tgt)⁺       →  irreducible geometric cost
    """

    def test_same_k_cutoff_is_zero(self):
        """Same K: A_cutoff = 0, A_total = A_impedance."""
        src = GammaNode("a", impedance=np.array([50, 60, 70.0]),
                        activation=np.zeros(3))
        tgt = GammaNode("b", impedance=np.array([80, 90, 100.0]),
                        activation=np.zeros(3))
        ch = MulticoreChannel(source=src, target=tgt)

        assert ch.cutoff_action() == 0.0
        assert ch.impedance_action() > 0
        assert abs(ch.directed_action() - ch.impedance_action()) < 1e-10

    def test_k5_to_k1_cutoff_equals_4(self):
        """K=5→K=1: cutoff = 4 (modes 1-4 totally reflected)."""
        src = GammaNode("s", impedance=np.ones(5) * 80, activation=np.zeros(5))
        tgt = GammaNode("t", impedance=np.ones(1) * 80, activation=np.zeros(1))
        ch = MulticoreChannel(source=src, target=tgt)

        assert ch.cutoff_action() == 4.0

    def test_k1_to_k5_cutoff_equals_zero(self):
        """K=1→K=5: cutoff = 0 (K_src < K_tgt → no excess modes)."""
        src = GammaNode("s", impedance=np.ones(1) * 80, activation=np.zeros(1))
        tgt = GammaNode("t", impedance=np.ones(5) * 80, activation=np.zeros(5))
        ch = MulticoreChannel(source=src, target=tgt)

        assert ch.cutoff_action() == 0.0

    def test_directed_action_asymmetry(self):
        """A(i→j) ≠ A(j→i) when K_i ≠ K_j — fundamental theorem property."""
        nodeA = GammaNode("A", impedance=np.array([80, 70, 90, 60, 85.0]),
                          activation=np.zeros(5))
        nodeB = GammaNode("B", impedance=np.array([80.0]),
                          activation=np.zeros(1))

        ch_AB = MulticoreChannel(source=nodeA, target=nodeB)
        ch_BA = MulticoreChannel(source=nodeB, target=nodeA)

        # A→B: cutoff = 4, B→A: cutoff = 0
        assert ch_AB.cutoff_action() == 4.0
        assert ch_BA.cutoff_action() == 0.0

        # Therefore A(A→B) > A(B→A) strictly
        assert ch_AB.directed_action() > ch_BA.directed_action()

    def test_impedance_action_equals_sum_gamma_sq_common(self):
        """A_impedance = Σ_{k≤K_common} Γ_k² exactly."""
        src = GammaNode("s", impedance=np.array([50, 60, 70.0]),
                        activation=np.zeros(3))
        tgt = GammaNode("t", impedance=np.array([80.0]),
                        activation=np.zeros(1))
        ch = MulticoreChannel(source=src, target=tgt)

        gv = ch.gamma_vector()
        expected = float(gv[0] ** 2)  # Only mode 0 is common
        assert abs(ch.impedance_action() - expected) < 1e-10

    def test_directed_action_decomposes_exactly(self):
        """A_total = A_impedance + A_cutoff for any channel."""
        for K_s, K_t in [(5, 1), (3, 3), (1, 5), (4, 2), (2, 4)]:
            src = GammaNode("s", impedance=np.arange(1, K_s + 1) * 30.0 + 50,
                            activation=np.zeros(K_s))
            tgt = GammaNode("t", impedance=np.arange(1, K_t + 1) * 25.0 + 60,
                            activation=np.zeros(K_t))
            ch = MulticoreChannel(source=src, target=tgt)

            total = ch.directed_action()
            parts = ch.impedance_action() + ch.cutoff_action()
            assert abs(total - parts) < 1e-10, \
                f"K={K_s}→{K_t}: total={total}, parts={parts}"

    def test_dimension_gap_property(self):
        """dimension_gap = |K_src - K_tgt|."""
        src = GammaNode("s", impedance=np.ones(5) * 80, activation=np.zeros(5))
        tgt = GammaNode("t", impedance=np.ones(2) * 100, activation=np.zeros(2))
        ch = MulticoreChannel(source=src, target=tgt)

        assert ch.dimension_gap == 3

    def test_perfectly_matched_same_k_zero_impedance_action(self):
        """Perfectly matched impedances: A_impedance = 0."""
        z = np.array([80, 70, 90.0])
        src = GammaNode("s", impedance=z.copy(), activation=np.zeros(3))
        tgt = GammaNode("t", impedance=z.copy(), activation=np.zeros(3))
        ch = MulticoreChannel(source=src, target=tgt)

        assert ch.impedance_action() < 1e-10
        assert ch.cutoff_action() == 0.0
        assert ch.directed_action() < 1e-10


# ============================================================================
# 6. Network-level action decomposition
# ============================================================================


class TestNetworkActionDecomposition:
    """Verify action_decomposition(), optimizable_action(), structural_action()."""

    def _make_topo(self) -> GammaTopology:
        """Mixed-K topology for testing."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("k3", impedance=np.ones(3) * 100, activation=np.zeros(3)),
            GammaNode("k1", impedance=np.ones(1) * 120, activation=np.zeros(1)),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        topo.activate_edge("k5", "k3")  # gap=2, cutoff=2
        topo.activate_edge("k3", "k1")  # gap=2, cutoff=2
        topo.activate_edge("k1", "k5")  # gap=4, cutoff=0 (K_src < K_tgt)
        return topo

    def test_action_decomposition_sums(self):
        """Network A_total = A_impedance + A_cutoff exactly."""
        topo = self._make_topo()
        a_imp, a_cut = topo.action_decomposition()
        assert a_imp >= 0
        assert a_cut >= 0
        # k5→k3: cutoff=2, k3→k1: cutoff=2, k1→k5: cutoff=0
        assert a_cut == 4.0

    def test_optimizable_action_matches_decomposition(self):
        topo = self._make_topo()
        a_imp, _ = topo.action_decomposition()
        assert abs(topo.optimizable_action() - a_imp) < 1e-10

    def test_structural_action_matches_decomposition(self):
        topo = self._make_topo()
        _, a_cut = topo.action_decomposition()
        assert abs(topo.structural_action() - a_cut) < 1e-10

    def test_tick_metrics_include_action_decomposition(self):
        """tick() must return action_impedance, action_cutoff, action_total."""
        topo = self._make_topo()
        metrics = topo.tick(enable_spontaneous=False)
        assert "action_impedance" in metrics
        assert "action_cutoff" in metrics
        assert "action_total" in metrics
        assert abs(metrics["action_total"]
                   - metrics["action_impedance"]
                   - metrics["action_cutoff"]) < 1e-10

    def test_impedance_action_decreases_with_learning(self):
        """C2 Hebbian updates should reduce A_impedance over many ticks."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 3,
                MOTOR_ALPHA: 3,
            },
            initial_connectivity=0.4,
            eta=0.05,
            seed=42,
        )
        # Stimulate for learning
        early_actions = []
        late_actions = []
        for tick in range(1, 51):
            stim = {}
            for name, node in topo.nodes.items():
                stim[name] = np.random.default_rng(tick).uniform(0.1, 0.5, size=node.K)
            m = topo.tick(external_stimuli=stim, enable_spontaneous=False)
            if tick <= 5:
                early_actions.append(m["action_impedance"])
            if tick >= 46:
                late_actions.append(m["action_impedance"])

        assert np.mean(late_actions) < np.mean(early_actions), \
            "C2 gradient should reduce A_impedance over learning"


# ============================================================================
# 7. max_dimension_gap — dimensional access control
# ============================================================================


class TestMaxDimensionGap:
    """Verify max_dimension_gap prevents impossible direct connections."""

    def test_gap_enforcement_at_activate_edge(self):
        """activate_edge returns None and does not create edge if gap > max."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("k1", impedance=np.ones(1) * 120, activation=np.zeros(1)),
        ]
        topo = GammaTopology(nodes=nodes, max_dimension_gap=2)

        result = topo.activate_edge("k5", "k1")  # gap = 4 > 2
        assert result is None
        assert ("k5", "k1") not in topo.active_edges

    def test_gap_allows_within_limit(self):
        """activate_edge succeeds when gap ≤ max."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("k3", impedance=np.ones(3) * 100, activation=np.zeros(3)),
        ]
        topo = GammaTopology(nodes=nodes, max_dimension_gap=2)

        result = topo.activate_edge("k5", "k3")  # gap = 2 = max
        assert result is not None
        assert ("k5", "k3") in topo.active_edges

    def test_gap_none_allows_everything(self):
        """When max_dimension_gap is None, no restriction."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("k1", impedance=np.ones(1) * 120, activation=np.zeros(1)),
        ]
        topo = GammaTopology(nodes=nodes, max_dimension_gap=None)

        result = topo.activate_edge("k5", "k1")
        assert result is not None

    def test_create_anatomical_with_max_gap(self):
        """create_anatomical forwards max_dimension_gap."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 3,
                PAIN_C_FIBER: 3,
            },
            initial_connectivity=0.5,
            max_dimension_gap=2,
            seed=42,
        )
        assert topo.max_dimension_gap == 2

        # No edges should have gap > 2
        for (src, tgt), ch in topo.active_edges.items():
            assert ch.dimension_gap <= 2, \
                f"Edge {src}→{tgt} has gap={ch.dimension_gap} > max={2}"

    def test_gap_pruning_in_spontaneous_dynamics(self):
        """Edges created before gap enforcement should be pruned."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80,
                      activation=np.ones(5) * 0.5),
            GammaNode("k1", impedance=np.ones(1) * 120,
                      activation=np.ones(1) * 0.5),
        ]
        # Create without gap enforcement
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=None)
        topo.activate_edge("k5", "k1")
        assert ("k5", "k1") in topo.active_edges

        # Now impose gap enforcement
        topo.max_dimension_gap = 2

        # Run spontaneous dynamics — should prune the gap=4 edge
        born, pruned = topo._spontaneous_dynamics()
        assert pruned >= 1
        assert ("k5", "k1") not in topo.active_edges

    def test_gap_prevents_sprouting(self):
        """Spontaneous sprouting must not create edges exceeding gap limit."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80,
                      activation=np.ones(5) * 0.5),
            GammaNode("k1", impedance=np.ones(1) * 120,
                      activation=np.ones(1) * 0.5),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=2)

        # Run many ticks — no edge should ever appear between k5 and k1
        for _ in range(50):
            topo.tick(enable_spontaneous=True)

        assert ("k5", "k1") not in topo.active_edges
        assert ("k1", "k5") not in topo.active_edges


# ============================================================================
# 8. Three-layer pruning
# ============================================================================


class TestThreeLayerPruning:
    """Verify the three-layer dimension-aware pruning strategy."""

    def test_layer1_dimension_prune(self):
        """Layer 1: gap > max_dimension_gap → immediate removal."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("k2", impedance=np.ones(2) * 80, activation=np.zeros(2)),
            GammaNode("k1", impedance=np.ones(1) * 80, activation=np.zeros(1)),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=None)
        topo.activate_edge("k5", "k2")   # gap=3
        topo.activate_edge("k5", "k1")   # gap=4
        topo.activate_edge("k2", "k1")   # gap=1

        # Now set gap limit
        topo.max_dimension_gap = 2

        born, pruned = topo._spontaneous_dynamics()
        # gap=3 and gap=4 should both be pruned
        assert ("k5", "k2") not in topo.active_edges
        assert ("k5", "k1") not in topo.active_edges
        # gap=1 should survive (at least layer 1)
        # (might be pruned by layer 2 or 3, but not by layer 1)

    def test_layer2_cutoff_plus_bad_impedance(self):
        """Layer 2: K_excess > 0 AND impedance_action > 0.5 → remove."""
        # Large impedance mismatch + dimensional mismatch
        src = GammaNode("s", impedance=np.array([30.0, 30.0, 30.0]),
                        activation=np.zeros(3))
        tgt = GammaNode("t", impedance=np.array([200.0]),
                         activation=np.zeros(1))
        nodes = [src, tgt]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=None)
        topo.activate_edge("s", "t")

        ch = topo.active_edges[("s", "t")]
        assert ch.K_excess > 0, "Should have mode cutoff"
        assert ch.impedance_action() > 0.5, "Should have bad impedance"

        born, pruned = topo._spontaneous_dynamics()
        assert ("s", "t") not in topo.active_edges

    def test_layer3_legacy_high_gamma(self):
        """Layer 3: scalar_gamma > 0.95 → remove (even if same K)."""
        src = GammaNode("s", impedance=np.array([10.0]),
                        activation=np.zeros(1))
        tgt = GammaNode("t", impedance=np.array([1000.0]),
                        activation=np.zeros(1))
        nodes = [src, tgt]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        topo.activate_edge("s", "t")

        ch = topo.active_edges[("s", "t")]
        assert ch.scalar_gamma() > 0.95

        born, pruned = topo._spontaneous_dynamics()
        assert ("s", "t") not in topo.active_edges

    def test_good_edge_survives_all_layers(self):
        """Well-matched same-K edge should survive all three pruning layers."""
        z = np.array([80, 85, 90.0])
        src = GammaNode("s", impedance=z, activation=np.zeros(3))
        tgt = GammaNode("t", impedance=z + 5, activation=np.zeros(3))
        nodes = [src, tgt]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=2)
        topo.activate_edge("s", "t")

        born, pruned = topo._spontaneous_dynamics()
        assert ("s", "t") in topo.active_edges
        assert pruned == 0


# ============================================================================
# 9. Optimal relay chain — Dimension Gradient Minimisation Path
# ============================================================================


class TestOptimalRelayChain:
    """
    Verify optimal_relay_chain() computes correct relay K sequences.

    From the Irreducibility Theorem Corollary:
      K_relay ∈ [K_tgt, K_src], evenly spaced, gaps ≤ max_gap.
    """

    def test_no_relay_needed_within_gap(self):
        """K=5 → K=3 with max_gap=2: no relay needed."""
        relays = GammaTopology.optimal_relay_chain(5, 3, max_gap=2)
        assert relays == []

    def test_k5_to_k1_gap2(self):
        """K=5 → K=1 with max_gap=2: need relay at K=3."""
        relays = GammaTopology.optimal_relay_chain(5, 1, max_gap=2)
        assert len(relays) >= 1
        # Verify chain: 5 → relay → 1, all gaps ≤ 2
        chain = [5] + relays + [1]
        for i in range(len(chain) - 1):
            assert abs(chain[i] - chain[i + 1]) <= 2

    def test_k5_to_k1_gap1(self):
        """K=5 → K=1 with max_gap=1: need 3 relays at K=4,3,2."""
        relays = GammaTopology.optimal_relay_chain(5, 1, max_gap=1)
        chain = [5] + relays + [1]
        assert len(relays) == 3
        for i in range(len(chain) - 1):
            assert abs(chain[i] - chain[i + 1]) <= 1

    def test_ascending_direction(self):
        """K=1 → K=5 should also produce valid relays."""
        relays = GammaTopology.optimal_relay_chain(1, 5, max_gap=2)
        chain = [1] + relays + [5]
        for i in range(len(chain) - 1):
            assert abs(chain[i] - chain[i + 1]) <= 2

    def test_same_k_no_relay(self):
        """K=3 → K=3: no relay needed."""
        assert GammaTopology.optimal_relay_chain(3, 3) == []

    def test_gap_must_be_positive(self):
        """max_gap < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            GammaTopology.optimal_relay_chain(5, 1, max_gap=0)

    def test_k10_to_k1_gap2(self):
        """Large gap: K=10 → K=1 with max_gap=2."""
        relays = GammaTopology.optimal_relay_chain(10, 1, max_gap=2)
        chain = [10] + relays + [1]
        for i in range(len(chain) - 1):
            assert abs(chain[i] - chain[i + 1]) <= 2
        # Should be monotonically decreasing
        for i in range(len(chain) - 1):
            assert chain[i] >= chain[i + 1]


# ============================================================================
# 10. Relay node insertion — dimensional adaptor
# ============================================================================


class TestRelayNodeInsertion:
    """
    Verify insert_relay_nodes() correctly builds relay chains.

    Real neuroanatomy: thalamus (K≈3-4) relays between cortex (K=5) and
    periphery (K=1-2).  The relay K falls between source and target,
    minimising per-hop A_cut.
    """

    def test_insert_relay_k5_to_k1(self):
        """K=5 → K=1 with gap=2: should insert relay node(s)."""
        nodes = [
            GammaNode("cortex", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("c_fiber", impedance=np.ones(1) * 120, activation=np.zeros(1)),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=2)

        relay_names = topo.insert_relay_nodes("cortex", "c_fiber", seed=42)

        # Should have at least 1 relay
        assert len(relay_names) >= 1

        # Relay K should be between K_tgt and K_src
        for rn in relay_names:
            relay_node = topo.nodes[rn]
            assert 1 < relay_node.K < 5

        # Full chain should respect gap constraint
        chain = ["cortex"] + relay_names + ["c_fiber"]
        for i in range(len(chain) - 1):
            k_i = topo.nodes[chain[i]].K
            k_j = topo.nodes[chain[i + 1]].K
            assert abs(k_i - k_j) <= 2, \
                f"Gap {chain[i]}(K={k_i}) → {chain[i+1]}(K={k_j}) exceeds max"

    def test_relay_edges_exist(self):
        """Relay chain should have active edges between consecutive nodes."""
        nodes = [
            GammaNode("src", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("tgt", impedance=np.ones(1) * 120, activation=np.zeros(1)),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=2)

        relay_names = topo.insert_relay_nodes("src", "tgt", seed=42)
        chain = ["src"] + relay_names + ["tgt"]

        for i in range(len(chain) - 1):
            assert (chain[i], chain[i + 1]) in topo.active_edges, \
                f"Missing edge {chain[i]} → {chain[i+1]}"
            assert (chain[i + 1], chain[i]) in topo.active_edges, \
                f"Missing edge {chain[i+1]} → {chain[i]}"

    def test_relay_impedance_interpolation(self):
        """Relay Z should be between source and target Z (common modes)."""
        nodes = [
            GammaNode("src", impedance=np.array([50, 60, 70, 80, 90.0]),
                      activation=np.zeros(5)),
            GammaNode("tgt", impedance=np.array([120.0]),
                      activation=np.zeros(1)),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=2)

        relay_names = topo.insert_relay_nodes("src", "tgt", seed=42)

        for rn in relay_names:
            relay = topo.nodes[rn]
            # Mode 0: Z should be between 50 and 120
            assert 50 <= relay.impedance[0] <= 120, \
                f"Relay Z[0]={relay.impedance[0]} not interpolated"

    def test_no_relay_when_gap_ok(self):
        """No relay needed when K difference ≤ max_gap."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("k3", impedance=np.ones(3) * 100, activation=np.zeros(3)),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=2)

        relay_names = topo.insert_relay_nodes("k5", "k3", seed=42)
        assert relay_names == []

    def test_relay_reduces_per_hop_cutoff(self):
        """Each hop in relay chain should have less cutoff than direct."""
        nodes = [
            GammaNode("src", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("tgt", impedance=np.ones(1) * 80, activation=np.zeros(1)),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=2)

        relay_names = topo.insert_relay_nodes("src", "tgt", seed=42)
        chain = ["src"] + relay_names + ["tgt"]

        # Direct cutoff would be 4 (K=5 → K=1)
        direct_cutoff = 4.0

        # Each hop should have cutoff ≤ max_gap
        for i in range(len(chain) - 1):
            key = (chain[i], chain[i + 1])
            ch = topo.active_edges[key]
            assert ch.cutoff_action() <= 2.0, \
                f"Hop {chain[i]}→{chain[i+1]} cutoff={ch.cutoff_action()} > max_gap"

    def test_relay_path_cutoff_preserved(self):
        """Total A_cut across relay chain = K_src - K_tgt (conservation)."""
        nodes = [
            GammaNode("src", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("tgt", impedance=np.ones(1) * 80, activation=np.zeros(1)),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=2)

        relay_names = topo.insert_relay_nodes("src", "tgt", seed=42)
        chain = ["src"] + relay_names + ["tgt"]

        _, total_cut = topo.relay_path_cutoff(chain)
        assert abs(total_cut - 4.0) < 1e-10, \
            f"Total cutoff={total_cut}, expected 4.0 (K=5 minus K=1)"

    def test_insert_all_relays(self):
        """insert_all_relays should handle all gap-violating edges."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80,
                      activation=np.zeros(5)),
            GammaNode("k1", impedance=np.ones(1) * 120,
                      activation=np.zeros(1)),
        ]
        # Create without gap — then enable gap
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=None)
        topo.activate_edge("k5", "k1")
        topo.activate_edge("k1", "k5")

        topo.max_dimension_gap = 2
        result = topo.insert_all_relays(seed=42)

        assert len(result) >= 1
        # Original direct edges should be gone
        assert ("k5", "k1") not in topo.active_edges

        # All remaining edges should respect gap
        for (src, tgt), ch in topo.active_edges.items():
            assert ch.dimension_gap <= 2, \
                f"Edge {src}→{tgt} gap={ch.dimension_gap} after relay insertion"

    def test_relay_chain_c1_holds(self):
        """C1 must hold at every edge in the relay chain."""
        nodes = [
            GammaNode("cortex", impedance=np.array([80, 70, 90, 60, 85.0]),
                      activation=np.array([1.0, 0.8, 0.6, 0.4, 0.2])),
            GammaNode("c_fiber", impedance=np.array([120.0]),
                      activation=np.array([0.5])),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=2)

        relay_names = topo.insert_relay_nodes("cortex", "c_fiber", seed=42)
        chain = ["cortex"] + relay_names + ["c_fiber"]

        for i in range(len(chain) - 1):
            key = (chain[i], chain[i + 1])
            ch = topo.active_edges[key]
            assert ch.verify_c1(), \
                f"C1 violated at {chain[i]} → {chain[i+1]}"

    def test_relay_survives_ticks(self):
        """Relay chain should survive Hebbian evolution (good impedance match)."""
        nodes = [
            GammaNode("cortex", impedance=np.ones(5) * 80,
                      activation=np.ones(5) * 0.5),
            GammaNode("c_fiber", impedance=np.ones(1) * 80,
                      activation=np.ones(1) * 0.5),
        ]
        topo = GammaTopology(nodes=nodes, eta=0.01, max_dimension_gap=2)

        relay_names = topo.insert_relay_nodes("cortex", "c_fiber", seed=42)
        chain = ["cortex"] + relay_names + ["c_fiber"]

        # Run ticks — relay edges should survive (well impedance-matched)
        for _ in range(20):
            stim = {n: np.ones(topo.nodes[n].K) * 0.3 for n in topo.nodes}
            topo.tick(external_stimuli=stim, enable_spontaneous=False)

        # All chain edges should still be active
        for i in range(len(chain) - 1):
            assert (chain[i], chain[i + 1]) in topo.active_edges, \
                f"Relay edge {chain[i]}→{chain[i+1]} was lost during evolution"


# ════════════════════════════════════════════════════════════════════
# 11. Fractal Dimension (Box-Counting)
# ════════════════════════════════════════════════════════════════════


class TestFractalDimension:
    """Tests for box-counting fractal dimension of Γ-topology."""

    def test_shortest_path_matrix_self_zero(self):
        """Diagonal of shortest-path matrix should be zero."""
        nodes = [GammaNode(f"n{i}", impedance=np.ones(3) * (50 + i * 10),
                           activation=np.zeros(3)) for i in range(6)]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        for i in range(5):
            topo.activate_edge(f"n{i}", f"n{i + 1}")
        dist, names = topo.shortest_path_matrix()
        for i in range(len(names)):
            assert dist[i, i] == 0

    def test_shortest_path_chain(self):
        """Chain of 5 nodes: d(0,4) = 4."""
        nodes = [GammaNode(f"n{i}", impedance=np.ones(3) * 50,
                           activation=np.zeros(3)) for i in range(5)]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        for i in range(4):
            topo.activate_edge(f"n{i}", f"n{i + 1}")
            topo.activate_edge(f"n{i + 1}", f"n{i}")
        dist, names = topo.shortest_path_matrix()
        idx = {n: i for i, n in enumerate(names)}
        assert dist[idx["n0"], idx["n4"]] == 4
        assert dist[idx["n0"], idx["n1"]] == 1

    def test_shortest_path_disconnected(self):
        """Disconnected nodes get distance = N (sentinel)."""
        nodes = [GammaNode(f"n{i}", impedance=np.ones(2) * 50,
                           activation=np.zeros(2)) for i in range(4)]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        topo.activate_edge("n0", "n1")
        topo.activate_edge("n1", "n0")
        # n2, n3 disconnected from n0, n1
        topo.activate_edge("n2", "n3")
        topo.activate_edge("n3", "n2")
        dist, _ = topo.shortest_path_matrix()
        idx = {n: i for i, n in enumerate(sorted(topo.nodes.keys()))}
        assert dist[idx["n0"], idx["n2"]] == 4  # sentinel = N

    def test_box_counting_returns_dict(self):
        """box_counting_dimension returns expected keys."""
        topo = GammaTopology.create_random(
            n_nodes=20, n_modes=3, initial_connectivity=0.3, seed=42)
        result = topo.box_counting_dimension()
        assert "D_f" in result
        assert "R2" in result
        assert "diameter" in result
        assert "l_values" in result
        assert "N_B_values" in result

    def test_box_counting_d_f_positive(self):
        """Fractal dimension should be positive for a connected graph."""
        topo = GammaTopology.create_random(
            n_nodes=32, n_modes=3, initial_connectivity=0.3, seed=42)
        result = topo.box_counting_dimension()
        if not np.isnan(result["D_f"]):
            assert result["D_f"] > 0, f"D_f should be > 0, got {result['D_f']}"

    def test_box_counting_complete_graph(self):
        """High-connectivity graph: diameter ≈ 1-2, D_f may be large or nan."""
        nodes = [GammaNode(f"n{i}", impedance=np.ones(2) * 50,
                           activation=np.zeros(2)) for i in range(10)]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        # Fully connect
        for i in range(10):
            for j in range(10):
                if i != j:
                    topo.activate_edge(f"n{i}", f"n{j}")
        result = topo.box_counting_dimension()
        # Complete graph: diameter = 1, one box covers everything at l_B=2
        assert result["diameter"] <= 2

    def test_box_counting_n_b_monotone_decreasing(self):
        """N_B should decrease (or stay same) as ℓ_B increases."""
        topo = GammaTopology.create_random(
            n_nodes=32, n_modes=3, initial_connectivity=0.2, seed=99)
        result = topo.box_counting_dimension()
        nb = result["N_B_values"]
        for i in range(len(nb) - 1):
            assert nb[i] >= nb[i + 1], \
                f"N_B should be non-increasing: N_B({result['l_values'][i]})={nb[i]}" \
                f" > N_B({result['l_values'][i+1]})={nb[i+1]}"

    def test_box_counting_heterogeneous_topology(self):
        """Box-counting on heterogeneous anatomical network."""
        composition = {
            CORTICAL_PYRAMIDAL: 8,
            MOTOR_ALPHA: 5,
            SENSORY_AB: 5,
            PAIN_C_FIBER: 6,
        }
        topo = GammaTopology.create_anatomical(
            tissue_composition=composition,
            initial_connectivity=0.2,
            max_dimension_gap=2,
            seed=42,
        )
        result = topo.box_counting_dimension()
        assert result["diameter"] >= 1
        assert len(result["N_B_values"]) > 0

    def test_box_counting_after_evolution(self):
        """Fractal dimension should be computable after Hebbian evolution."""
        topo = GammaTopology.create_random(
            n_nodes=24, n_modes=3, initial_connectivity=0.2,
            eta=0.02, seed=42)
        rng = np.random.default_rng(42)
        for _ in range(30):
            stim = {n: rng.uniform(0.1, 0.5, size=topo.nodes[n].K)
                    for n in topo.nodes}
            topo.tick(external_stimuli=stim)
        result = topo.box_counting_dimension()
        assert "D_f" in result
        # Should be computable (not crash) after evolution

    def test_tiny_network_graceful(self):
        """Very small network (N<4) returns nan gracefully."""
        nodes = [GammaNode(f"n{i}", impedance=np.ones(2) * 50,
                           activation=np.zeros(2)) for i in range(3)]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        topo.activate_edge("n0", "n1")
        result = topo.box_counting_dimension()
        assert np.isnan(result["D_f"])


class TestSpectralDimension:
    """Tests for spectral dimension d_s from the Laplacian heat kernel."""

    @staticmethod
    def _make_chain(n: int) -> GammaTopology:
        """Linear chain of n nodes — expected d_s ≈ 1 for large n."""
        nodes = [GammaNode(f"c{i}", impedance=np.ones(2) * 50,
                           activation=np.zeros(2)) for i in range(n)]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        for i in range(n - 1):
            topo.activate_edge(f"c{i}", f"c{i + 1}")
        return topo

    @staticmethod
    def _make_complete(n: int) -> GammaTopology:
        """Complete graph of n nodes — d_s should be high."""
        nodes = [GammaNode(f"k{i}", impedance=np.ones(2) * 50,
                           activation=np.zeros(2)) for i in range(n)]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        for i in range(n):
            for j in range(i + 1, n):
                topo.activate_edge(f"k{i}", f"k{j}")
        return topo

    def test_returns_expected_keys(self):
        """spectral_dimension returns the expected dict keys."""
        topo = self._make_chain(10)
        result = topo.spectral_dimension()
        assert "d_s" in result
        assert "R2" in result
        assert "d_s_midrange" in result
        assert "lambda_min" in result
        assert "lambda_max" in result
        assert "n_zero_modes" in result

    def test_chain_d_s_low(self):
        """A chain (1D lattice) should have d_s closer to 1 than 3."""
        topo = self._make_chain(30)
        result = topo.spectral_dimension()
        # For a finite chain, d_s is roughly ~1-2; definitely < 4
        assert not np.isnan(result["d_s"])
        assert result["d_s"] < 4.0, f"Chain d_s = {result['d_s']}, expected < 4"

    def test_complete_graph_d_s_high(self):
        """Complete graph should have high d_s (many parallel paths)."""
        chain = self._make_chain(20)
        complete = self._make_complete(20)
        d_s_chain = chain.spectral_dimension()["d_s"]
        d_s_complete = complete.spectral_dimension()["d_s"]
        assert d_s_complete > d_s_chain, (
            f"Complete d_s={d_s_complete} should exceed chain d_s={d_s_chain}")

    def test_connected_network_one_zero_mode(self):
        """A connected network has exactly 1 zero eigenvalue."""
        topo = self._make_chain(15)
        result = topo.spectral_dimension()
        assert result["n_zero_modes"] == 1

    def test_disconnected_two_zero_modes(self):
        """Two disconnected components → 2 zero modes."""
        nodes = [GammaNode(f"d{i}", impedance=np.ones(2) * 50,
                           activation=np.zeros(2)) for i in range(10)]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        # Component 1: d0-d1-d2-d3-d4
        for i in range(4):
            topo.activate_edge(f"d{i}", f"d{i + 1}")
        # Component 2: d5-d6-d7-d8-d9
        for i in range(5, 9):
            topo.activate_edge(f"d{i}", f"d{i + 1}")
        result = topo.spectral_dimension()
        assert result["n_zero_modes"] == 2

    def test_lambda_min_positive(self):
        """Smallest positive eigenvalue (Fiedler) > 0 for connected graph."""
        topo = self._make_chain(15)
        result = topo.spectral_dimension()
        assert result["lambda_min"] > 0

    def test_tiny_network_graceful(self):
        """Very small network returns nan gracefully."""
        nodes = [GammaNode(f"t{i}", impedance=np.ones(2) * 50,
                           activation=np.zeros(2)) for i in range(3)]
        topo = GammaTopology(nodes=nodes, eta=0.01)
        topo.activate_edge("t0", "t1")
        result = topo.spectral_dimension()
        assert isinstance(result["d_s"], float)

    def test_heterogeneous_topology(self):
        """Spectral dimension is computable on mixed-K anatomical topology."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 5, MOTOR_ALPHA: 4,
                SENSORY_AB: 4, PAIN_C_FIBER: 3,
            },
            initial_connectivity=0.15, eta=0.02,
            max_dimension_gap=2, seed=42,
        )
        result = topo.spectral_dimension()
        assert not np.isnan(result["d_s"])
        assert result["d_s"] > 0
        assert result["R2"] > 0.5  # should be a reasonable fit


class TestKLevelAnalysis:
    """Tests for K-level (dimensional-space) structural analysis."""

    def test_returns_expected_keys(self):
        """k_level_analysis returns all expected keys."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 4, MOTOR_ALPHA: 3,
                SENSORY_AB: 3, PAIN_C_FIBER: 2,
            },
            initial_connectivity=0.15, eta=0.02,
            max_dimension_gap=2, seed=42,
        )
        result = topo.k_level_analysis()
        for key in ["k_values", "k_counts", "edge_by_dk",
                    "possible_by_dk", "density_by_dk",
                    "D_K", "D_K_R2",
                    "density_ratio_1_0", "density_ratio_2_0"]:
            assert key in result, f"Missing key: {key}"

    def test_edge_counts_sum_to_total(self):
        """Sum of edges by ΔK must equal total undirected edges."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 5, MOTOR_ALPHA: 4,
                SENSORY_AB: 4, PAIN_C_FIBER: 3,
            },
            initial_connectivity=0.15, eta=0.02,
            max_dimension_gap=2, seed=42,
        )
        result = topo.k_level_analysis()
        total_by_dk = sum(result["edge_by_dk"].values())
        # Count unique undirected edges for reference
        undirected = set()
        for (s, t) in topo.active_edges:
            undirected.add((min(s, t), max(s, t)))
        assert total_by_dk == len(undirected)

    def test_density_between_zero_and_one(self):
        """All densities must be in [0, 1]."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 5, MOTOR_ALPHA: 4,
                SENSORY_AB: 4, PAIN_C_FIBER: 3,
            },
            initial_connectivity=0.15, eta=0.02,
            max_dimension_gap=2, seed=42,
        )
        result = topo.k_level_analysis()
        for dk, rho in result["density_by_dk"].items():
            assert 0.0 <= rho <= 1.0, f"rho({dk}) = {rho} out of range"

    def test_blocked_gaps_have_zero_edges(self):
        """ΔK > max_dimension_gap should have zero edges."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 4, MOTOR_ALPHA: 3,
                PAIN_C_FIBER: 3,
            },
            initial_connectivity=0.15, eta=0.02,
            max_dimension_gap=2, seed=42,
        )
        result = topo.k_level_analysis()
        # K=5 to K=1 has ΔK=4, should be blocked
        assert result["edge_by_dk"].get(4, 0) == 0
        assert result["edge_by_dk"].get(3, 0) == 0

    def test_d_k_computable(self):
        """D_K should be a finite number on a heterogeneous network."""
        # Must include K=2 tissue to get ΔK=1 edges
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 5, MOTOR_ALPHA: 4,
                SENSORY_AB: 4, PAIN_AD_FIBER: 4, PAIN_C_FIBER: 3,
            },
            initial_connectivity=0.15, eta=0.02,
            max_dimension_gap=2, seed=42,
        )
        result = topo.k_level_analysis()
        assert not np.isnan(result["D_K"]), "D_K should be computable"
        # D_K can be positive or negative depending on whether edges
        # concentrate at small or large ΔK — both are physically valid

    def test_density_ratios_finite(self):
        """Density ratios should be finite and non-negative."""
        # Include K=2 tissue to ensure ΔK=1 edges exist
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 5, MOTOR_ALPHA: 4,
                PAIN_AD_FIBER: 4, PAIN_C_FIBER: 3,
            },
            initial_connectivity=0.15, eta=0.02,
            max_dimension_gap=2, seed=42,
        )
        result = topo.k_level_analysis()
        assert not np.isnan(result["density_ratio_1_0"])
        assert result["density_ratio_1_0"] >= 0

    def test_invariance_across_sizes(self):
        """Density ratios should be approximately invariant across N."""
        # Include K=2 tissue to have ΔK=1 edges
        ratios_2_0 = []
        for N_scale in [1, 2, 4]:
            topo = GammaTopology.create_anatomical(
                tissue_composition={
                    CORTICAL_PYRAMIDAL: 4 * N_scale,
                    MOTOR_ALPHA: 3 * N_scale,
                    PAIN_AD_FIBER: 3 * N_scale,
                    PAIN_C_FIBER: 2 * N_scale,
                },
                initial_connectivity=0.15, eta=0.02,
                max_dimension_gap=2, seed=42,
            )
            result = topo.k_level_analysis()
            # Use ΔK=2 ratio (always available: K=5↔K=3, K=3↔K=1)
            if not np.isnan(result["density_ratio_2_0"]) and result["density_ratio_2_0"] > 0:
                ratios_2_0.append(result["density_ratio_2_0"])
        # Ratios should be within factor of 3 across sizes
        assert len(ratios_2_0) >= 2
        assert max(ratios_2_0) / min(ratios_2_0) < 3.0, (
            f"Density ratio ρ(2)/ρ(0) varies too much: {ratios_2_0}")


# ============================================================================
# 14. Soft cutoff — dimension_gap_decay power-law connectivity
# ============================================================================


class TestSoftCutoff:
    """
    Verify dimension_gap_decay: edge acceptance probability
    p(ΔK) = (ΔK + 1)^{−γ}.

    Hard cutoff (max_dimension_gap) creates dimensional democracy (D_K ≈ 0).
    Soft cutoff creates power-law connectivity (D_K ≈ γ) — fractal topology.
    """

    def test_soft_cutoff_none_is_noop(self):
        """dimension_gap_decay=None → identical to hard-cutoff-only."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("k3", impedance=np.ones(3) * 100, activation=np.zeros(3)),
        ]
        topo = GammaTopology(nodes=nodes, max_dimension_gap=2,
                             dimension_gap_decay=None)

        # Should always succeed (gap=2 ≤ max=2, no soft cutoff)
        result = topo.activate_edge("k5", "k3")
        assert result is not None
        assert ("k5", "k3") in topo.active_edges

    def test_soft_cutoff_dk0_always_accepted(self):
        """ΔK=0 edges always accepted even with strong decay."""
        nodes = [
            GammaNode("a", impedance=np.ones(3) * 80, activation=np.zeros(3)),
            GammaNode("b", impedance=np.ones(3) * 100, activation=np.zeros(3)),
        ]
        topo = GammaTopology(nodes=nodes, dimension_gap_decay=10.0)

        # Same K: gap=0, p(0) = 1.0 regardless of γ
        result = topo.activate_edge("a", "b")
        assert result is not None

    def test_survival_probability_formula(self):
        """Verify p(ΔK) = (ΔK + 1)^{−γ} formula directly."""
        nodes = [GammaNode("x", impedance=np.ones(1) * 80,
                           activation=np.zeros(1))]
        topo = GammaTopology(nodes=nodes, dimension_gap_decay=1.26)

        assert topo._edge_survival_probability(0) == 1.0
        expected_1 = 2 ** (-1.26)
        assert abs(topo._edge_survival_probability(1) - expected_1) < 1e-10
        expected_2 = 3 ** (-1.26)
        assert abs(topo._edge_survival_probability(2) - expected_2) < 1e-10
        expected_3 = 4 ** (-1.26)
        assert abs(topo._edge_survival_probability(3) - expected_3) < 1e-10

    def test_soft_cutoff_reduces_high_delta_k_edges(self):
        """Soft cutoff network has fewer high-ΔK edges than no-cutoff."""
        tissue = {
            CORTICAL_PYRAMIDAL: 8,   # K=5
            MOTOR_ALPHA: 8,          # K=3
            PAIN_C_FIBER: 8,         # K=1
        }
        # No cutoff: democratic
        topo_hard = GammaTopology.create_anatomical(
            tissue_composition=tissue, initial_connectivity=0.3,
            max_dimension_gap=4, dimension_gap_decay=None, seed=42)

        # Soft cutoff: power-law
        topo_soft = GammaTopology.create_anatomical(
            tissue_composition=tissue, initial_connectivity=0.3,
            max_dimension_gap=4, dimension_gap_decay=1.26, seed=42)

        # Count edges by ΔK
        def count_by_dk(topo):
            counts = {}
            for (s, t), ch in topo.active_edges.items():
                dk = ch.dimension_gap
                counts[dk] = counts.get(dk, 0) + 1
            return counts

        hard_counts = count_by_dk(topo_hard)
        soft_counts = count_by_dk(topo_soft)

        # ΔK=0 should be similar (p=1 in both)
        dk0_hard = hard_counts.get(0, 0)
        dk0_soft = soft_counts.get(0, 0)
        # With same seed, ΔK=0 edges should be identical
        assert dk0_soft == dk0_hard, (
            f"ΔK=0 edges differ: hard={dk0_hard}, soft={dk0_soft}")

        # Total non-zero ΔK edges should be fewer with soft cutoff
        total_hard = sum(v for k, v in hard_counts.items() if k > 0)
        total_soft = sum(v for k, v in soft_counts.items() if k > 0)
        assert total_soft < total_hard, (
            f"Soft cutoff should reduce cross-K edges: hard={total_hard}, soft={total_soft}")

    def test_soft_cutoff_with_hard_cutoff_combined(self):
        """Hard cutoff is still absolute upper bound when both are set."""
        nodes = [
            GammaNode("k5", impedance=np.ones(5) * 80, activation=np.zeros(5)),
            GammaNode("k1", impedance=np.ones(1) * 120, activation=np.zeros(1)),
        ]
        # Gap=4, max_dimension_gap=2 → hard reject regardless of soft cutoff
        topo = GammaTopology(nodes=nodes, max_dimension_gap=2,
                             dimension_gap_decay=0.0)  # γ=0 → p=1 for all

        result = topo.activate_edge("k5", "k1")
        assert result is None
        assert ("k5", "k1") not in topo.active_edges

    def test_soft_cutoff_sprouting_respects_probability(self):
        """Spontaneous sprouting also applies soft cutoff."""
        tissue = {
            CORTICAL_PYRAMIDAL: 5,   # K=5
            MOTOR_ALPHA: 5,          # K=3
            PAIN_C_FIBER: 5,         # K=1
        }
        # Very aggressive soft cutoff
        topo = GammaTopology.create_anatomical(
            tissue_composition=tissue, initial_connectivity=0.0,
            max_dimension_gap=4, dimension_gap_decay=3.0, seed=42)

        # Stimulate all nodes to trigger sprouting
        for _ in range(30):
            stim = {}
            for name, node in topo.nodes.items():
                stim[name] = np.ones(node.K) * 0.3
            topo.tick(external_stimuli=stim, enable_spontaneous=True)

        # With γ=3.0, cross-K edges should be very rare
        cross_k = 0
        same_k = 0
        for (s, t), ch in topo.active_edges.items():
            if ch.dimension_gap > 0:
                cross_k += 1
            else:
                same_k += 1

        # Should have SOME same-K edges (p=1) but very few cross-K
        # (With γ=3, p(ΔK=2) = 3^{-3} ≈ 0.037)
        assert same_k > 0, "Same-K edges should exist"
        if same_k > 0:
            ratio = cross_k / (same_k + cross_k)
            assert ratio < 0.5, (
                f"With γ=3.0, cross-K ratio should be low: {ratio:.3f}")

    def test_create_anatomical_with_decay(self):
        """create_anatomical correctly passes dimension_gap_decay."""
        topo = GammaTopology.create_anatomical(
            tissue_composition={
                CORTICAL_PYRAMIDAL: 4,
                MOTOR_ALPHA: 4,
                PAIN_C_FIBER: 4,
            },
            initial_connectivity=0.3,
            max_dimension_gap=4,
            dimension_gap_decay=1.26,
            seed=42,
        )
        assert topo.dimension_gap_decay == 1.26
        assert topo.max_dimension_gap == 4
        # Should have created some edges but fewer cross-K than without decay
        assert len(topo.active_edges) > 0

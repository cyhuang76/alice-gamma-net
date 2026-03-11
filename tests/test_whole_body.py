# -*- coding: utf-8 -*-
"""Tests for the whole-body inter-organ GammaTopology.

Verifies the unified architecture that connects all 29 organ blueprints
via dual vascular/neural buses with brain command center.

Emergence level: E0 — all tests verify physics, not disease-specific code.
"""

import numpy as np
import pytest

from alice.body.tissue_blueprint import (
    BLUEPRINT_REGISTRY,
    ORGAN_INTERFACES,
    build_whole_body,
    _organ_prefix,
)


# ============================================================================
# Structural tests
# ============================================================================


class TestWholeBodyStructure:
    """Verify the whole-body topology has correct structure."""

    @pytest.fixture(scope="class")
    def wb(self):
        return build_whole_body(seed=42)

    def test_node_count(self, wb):
        """29 organs (~218 nodes) + 30 vbus + 30 nbus = ~278."""
        assert wb.N >= 270
        assert wb.N <= 300

    def test_edge_count(self, wb):
        """Organ internal + bus inter-organ edges."""
        assert len(wb.active_edges) >= 350
        assert len(wb.active_edges) <= 500

    def test_all_organs_present(self, wb):
        """Every organ has at least one prefixed node in the topology."""
        for organ_name in BLUEPRINT_REGISTRY:
            prefix = _organ_prefix(organ_name)
            organ_nodes = [n for n in wb.nodes if n.startswith(f"{prefix}.")]
            assert len(organ_nodes) >= 1, f"No nodes for organ {organ_name}"

    def test_vascular_bus_nodes(self, wb):
        """One trunk + 29 branches = 30 vbus nodes."""
        vbus = [n for n in wb.nodes if n.startswith("vbus.")]
        assert len(vbus) == 30  # 1 trunk + 29 branches

    def test_neural_bus_nodes(self, wb):
        """One trunk + 29 branches = 30 nbus nodes."""
        nbus = [n for n in wb.nodes if n.startswith("nbus.")]
        assert len(nbus) == 30  # 1 trunk + 29 branches

    def test_vbus_trunk_exists(self, wb):
        assert "vbus.aorta" in wb.nodes

    def test_nbus_trunk_exists(self, wb):
        assert "nbus.autonomic" in wb.nodes

    def test_heterogeneous(self, wb):
        """Whole-body spans K=1 to K=5."""
        assert wb.is_heterogeneous
        assert wb.K == 5
        assert wb.K_min == 1

    def test_all_organ_interfaces_valid(self):
        """Every interface node actually exists in its organ blueprint."""
        for organ_name, iface in ORGAN_INTERFACES.items():
            build_fn = BLUEPRINT_REGISTRY[organ_name]
            topo = build_fn()
            for role in ("vascular", "neural"):
                node_name = iface[role]
                assert node_name in topo.nodes, (
                    f"{organ_name}.{role} interface '{node_name}' "
                    f"not in blueprint nodes: {list(topo.nodes.keys())}"
                )


# ============================================================================
# Physics tests (C1/C2)
# ============================================================================


class TestWholeBodyPhysics:
    """Verify C1 and C2 on the unified topology."""

    @pytest.fixture(scope="class")
    def wb(self):
        return build_whole_body(seed=42)

    def test_c1_all_edges(self, wb):
        """C1: Γ² + T = 1 at every edge."""
        for (src, tgt), ch in wb.active_edges.items():
            assert ch.verify_c1(), f"C1 violated at {src} → {tgt}"

    def test_action_decomposition(self, wb):
        """A = A_impedance + A_cutoff, both non-negative."""
        a_imp, a_cut = wb.action_decomposition()
        assert a_imp >= 0
        assert a_cut >= 0
        # Cutoff must be positive (heterogeneous topology)
        assert a_cut > 0, "No dimensional cutoff in heterogeneous body?"

    def test_c2_reduces_action(self, wb):
        """Running tick() with C2 should reduce A_impedance over time."""
        topo = build_whole_body(seed=99)

        # Apply stimulus to brain
        bfun = _organ_prefix("brain_functional")
        stim = {f"{bfun}.prefrontal": np.ones(5) * 0.5}

        a_imp_before, _ = topo.action_decomposition()
        for _ in range(20):
            topo.tick(external_stimuli=stim, enable_spontaneous=False)
        a_imp_after, _ = topo.action_decomposition()

        assert a_imp_after <= a_imp_before + 1e-6, (
            f"A_impedance did not decrease: {a_imp_before:.4f} → {a_imp_after:.4f}"
        )

    def test_cutoff_invariant_under_c2(self, wb):
        """A_cutoff should not change under C2 remodeling (structural)."""
        topo = build_whole_body(seed=77)

        _, a_cut_before = topo.action_decomposition()

        stim = {"vbus.aorta": np.ones(4) * 0.3}
        for _ in range(10):
            topo.tick(external_stimuli=stim, enable_spontaneous=False)

        _, a_cut_after = topo.action_decomposition()
        assert a_cut_after == a_cut_before, (
            f"A_cutoff changed: {a_cut_before} → {a_cut_after}"
        )

    def test_c1_after_tick(self, wb):
        """C1 must hold after remodeling ticks."""
        topo = build_whole_body(seed=55)
        stim = {"vbus.aorta": np.ones(4) * 0.5}
        for _ in range(5):
            topo.tick(external_stimuli=stim, enable_spontaneous=False)
        for (src, tgt), ch in topo.active_edges.items():
            assert ch.verify_c1(), f"C1 violated after tick at {src} → {tgt}"


# ============================================================================
# Bus connectivity tests
# ============================================================================


class TestBusConnectivity:
    """Verify vascular and neural buses connect to all organs."""

    @pytest.fixture(scope="class")
    def wb(self):
        return build_whole_body(seed=42)

    @pytest.mark.parametrize("organ_name", list(BLUEPRINT_REGISTRY.keys()))
    def test_vbus_branch_to_organ(self, wb, organ_name):
        """Each organ has a vbus branch → organ vascular port edge."""
        prefix = _organ_prefix(organ_name)
        branch = f"vbus.{prefix}"
        vport = f"{prefix}.{ORGAN_INTERFACES[organ_name]['vascular']}"

        assert (branch, vport) in wb.active_edges, (
            f"Missing vbus edge: {branch} → {vport}"
        )

    @pytest.mark.parametrize("organ_name", list(BLUEPRINT_REGISTRY.keys()))
    def test_nbus_branch_to_organ(self, wb, organ_name):
        """Each organ has an nbus branch → organ neural port edge."""
        prefix = _organ_prefix(organ_name)
        branch = f"nbus.{prefix}"
        nport = f"{prefix}.{ORGAN_INTERFACES[organ_name]['neural']}"

        assert (branch, nport) in wb.active_edges, (
            f"Missing nbus edge: {branch} → {nport}"
        )

    @pytest.mark.parametrize("organ_name", list(BLUEPRINT_REGISTRY.keys()))
    def test_vbus_trunk_to_branch(self, wb, organ_name):
        """Aortic trunk → each vascular branch."""
        prefix = _organ_prefix(organ_name)
        branch = f"vbus.{prefix}"
        assert ("vbus.aorta", branch) in wb.active_edges

    @pytest.mark.parametrize("organ_name", list(BLUEPRINT_REGISTRY.keys()))
    def test_nbus_trunk_to_branch(self, wb, organ_name):
        """Autonomic trunk → each neural branch."""
        prefix = _organ_prefix(organ_name)
        branch = f"nbus.{prefix}"
        assert ("nbus.autonomic", branch) in wb.active_edges

    def test_brain_to_neural_bus(self, wb):
        """Brain hypothalamus → neural bus trunk."""
        bfun = _organ_prefix("brain_functional")
        assert (f"{bfun}.hypothalamus_b", "nbus.autonomic") in wb.active_edges

    def test_neural_bus_to_brain(self, wb):
        """Neural bus trunk → brain thalamus (ascending)."""
        bfun = _organ_prefix("brain_functional")
        assert ("nbus.autonomic", f"{bfun}.thalamus_b") in wb.active_edges

    def test_vbus_brain_interoception(self, wb):
        """Vascular bus → brain hypothalamus (interoception)."""
        bfun = _organ_prefix("brain_functional")
        assert ("vbus.aorta", f"{bfun}.hypothalamus_b") in wb.active_edges

    def test_cross_bus_link(self, wb):
        """Vbus ↔ nbus bidirectional link exists."""
        assert ("vbus.aorta", "nbus.autonomic") in wb.active_edges
        assert ("nbus.autonomic", "vbus.aorta") in wb.active_edges


# ============================================================================
# Dimensional cascade tests
# ============================================================================


class TestDimensionalCascade:
    """Verify the K-dimensional hierarchy across buses."""

    @pytest.fixture(scope="class")
    def wb(self):
        return build_whole_body(seed=42)

    def test_vbus_trunk_k4(self, wb):
        """Vascular bus trunk is K=4."""
        assert wb.nodes["vbus.aorta"].K == 4

    def test_vbus_branch_k2(self, wb):
        """All vascular bus branches are K=2."""
        for n in wb.nodes:
            if n.startswith("vbus.") and n != "vbus.aorta":
                assert wb.nodes[n].K == 2, f"{n} has K={wb.nodes[n].K}, expected 2"

    def test_nbus_trunk_k3(self, wb):
        """Neural bus trunk is K=3."""
        assert wb.nodes["nbus.autonomic"].K == 3

    def test_nbus_branch_k2(self, wb):
        """All neural bus branches are K=2."""
        for n in wb.nodes:
            if n.startswith("nbus.") and n != "nbus.autonomic":
                assert wb.nodes[n].K == 2, f"{n} has K={wb.nodes[n].K}, expected 2"

    def test_vbus_trunk_to_branch_cutoff(self, wb):
        """K=4 → K=2 edge has cutoff cost = 2."""
        edge = wb.active_edges[("vbus.aorta", "vbus.cv")]
        assert edge.cutoff_action() == 2.0

    def test_nbus_trunk_to_branch_cutoff(self, wb):
        """K=3 → K=2 edge has cutoff cost = 1."""
        edge = wb.active_edges[("nbus.autonomic", "nbus.cv")]
        assert edge.cutoff_action() == 1.0


# ============================================================================
# Signal propagation tests (E0 emergence)
# ============================================================================


class TestSignalPropagation:
    """Verify that signals propagate through the whole-body topology."""

    def test_brain_stimulus_reaches_organs(self):
        """Stimulus at brain propagates to peripheral organs via buses."""
        topo = build_whole_body(seed=42)
        bfun = _organ_prefix("brain_functional")

        # Stimulate prefrontal cortex
        stim = {f"{bfun}.prefrontal": np.ones(5) * 1.0}
        for _ in range(10):
            topo.tick(external_stimuli=stim, enable_spontaneous=False)

        # Check that some activation reached vascular bus
        vbus_act = np.mean(np.abs(topo.nodes["vbus.aorta"].activation))
        assert vbus_act > 0, "No signal reached vascular bus from brain"

    def test_vascular_damage_cascades(self):
        """Perturbing vascular bus impedance raises Γ across organs."""
        topo = build_whole_body(seed=42)

        _, a_cut = topo.action_decomposition()
        a_imp_before, _ = topo.action_decomposition()

        # Damage: double the aortic impedance (stenosis)
        topo.nodes["vbus.aorta"].impedance *= 2.0

        a_imp_after, _ = topo.action_decomposition()
        assert a_imp_after > a_imp_before, (
            "Vascular damage did not increase A_impedance"
        )

    def test_organ_damage_propagates_to_brain(self):
        """Organ impedance perturbation eventually reaches brain via bus."""
        topo = build_whole_body(seed=42)
        ren_prefix = _organ_prefix("renal")

        # Damage renal afferent arteriole
        from alice.body.tissue_blueprint import perturb_impedance
        perturb_impedance(topo, f"{ren_prefix}.afferent_arteriole", 3.0)

        # Run several ticks with afferent stimulus
        stim = {f"{ren_prefix}.afferent_arteriole": np.ones(3) * 1.0}
        for _ in range(15):
            topo.tick(external_stimuli=stim, enable_spontaneous=False)

        # Check nbus received some activation change
        nbus_ren = f"nbus.{ren_prefix}"
        act = np.mean(np.abs(topo.nodes[nbus_ren].activation))
        assert act > 0, "Renal damage did not propagate to neural bus"


# ============================================================================
# Reproducibility
# ============================================================================


class TestReproducibility:
    """Verify deterministic build with same seed."""

    def test_same_seed_same_topology(self):
        t1 = build_whole_body(seed=123)
        t2 = build_whole_body(seed=123)
        assert t1.N == t2.N
        assert set(t1.active_edges.keys()) == set(t2.active_edges.keys())
        for name in t1.nodes:
            np.testing.assert_array_equal(
                t1.nodes[name].impedance,
                t2.nodes[name].impedance,
            )

    def test_different_seed_different_impedance(self):
        t1 = build_whole_body(seed=1)
        t2 = build_whole_body(seed=2)
        assert t1.N == t2.N
        diffs = sum(
            1 for n in t1.nodes
            if not np.allclose(t1.nodes[n].impedance, t2.nodes[n].impedance)
        )
        assert diffs > 0, "Different seeds produced identical impedances"

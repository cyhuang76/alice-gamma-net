# -*- coding: utf-8 -*-
"""
Dynamic Γ-Topology Network
═══════════════════════════

Every neuron has a different impedance Z_i(t).
Every pair (i, j) therefore has a nonzero reflection coefficient:

    Γ_ij(t) = (Z_j(t) − Z_i(t)) / (Z_j(t) + Z_i(t))

The ensemble of all Γ_ij IS the network topology — not a graph drawn
in advance, but a living structure that emerges from impedance diversity
and reshapes itself every tick via Hebbian learning (C2).

Key physics:
  C1  Energy conservation:  Γ_ij² + T_ij = 1   at every edge, every tick
  C2  Hebbian update:       ΔZ_i  = −η · Γ_ij · x_i · x_j
  C3  Signal protocol:      All values carry impedance metadata

Multi-core coaxial extension:
  Each connection is not a single wire but a bundle of K parallel
  conductors (dimensions / modes).  The scalar Γ generalises to a
  K×K reflection matrix:

      𝚪 = (Z_L + Z_0)⁻¹ (Z_L − Z_0)

  where Z_L and Z_0 are K×K impedance matrices.
  C1 becomes:  𝚪†𝚪 + 𝐓†𝐓 = 𝐈_K   (matrix energy conservation)

Dimensional Cost Irreducibility Theorem (2026-02-26):
  For heterogeneous networks (mixed K), the action decomposes as:

      A = A_imp(t) + A_cut

  where A_imp → 0 under C2 Hebbian learning, and
  A_cut = Σ_{edges} (K_src − K_tgt)⁺ is invariant under all gradients.

  Corollary: minimising A requires reducing high-to-low-K edges,
  automatically inducing hierarchical topology with relay nodes.
  See docs/IRREDUCIBILITY_THEOREM.md for formal statement.

This module provides:
  - GammaNode       : a node with K-dimensional impedance vector
  - MulticoreChannel: K×K matrix Γ channel between two nodes
  - GammaTopology   : dynamic network of N nodes, full Γ matrix,
                       Hebbian evolution, topology metrics

References:
  - Telegrapher's equation for axonal transmission
    (PMC5090033, PMC6842214, PMC12128527)
  - Minimum Reflection Principle: A[Γ] = ∫ ΣΓ² dt → min
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# 0. TissueType — biological fiber classification
# ============================================================================


@dataclass(frozen=True)
class TissueType:
    """
    Biological tissue / fiber type for heterogeneous Γ-topology.

    In waveguide physics, the number of propagation modes scales with
    the cross-section relative to wavelength:

        N_modes ∝ (d / λ_min)²

    For nerve fibers, K represents the number of independent information
    channels within the fiber bundle — coupled-conductor modes in the
    multi-conductor transmission line model (PMC5090033).

    Thicker fibers → more modes → higher dimensional signal capacity.
    Different organs / pathways have fundamentally different K.
    This is not a parameter variation — it is a dimensional difference.
    """

    name: str
    n_modes: int          # K — independent signal channels
    z_mean: float         # typical impedance centre (Ω)
    z_std: float          # impedance variability (Ω)
    diameter_um: float    # typical fiber diameter (μm)
    myelinated: bool      # myelination status
    description: str = ""


# ── Biological fiber type presets ─────────────────────────────────────
# K scales with effective information capacity of the fiber bundle.
# Thicker, more myelinated fibers carry more independent modes.

CORTICAL_PYRAMIDAL = TissueType(
    "cortical_pyramidal", n_modes=5, z_mean=80.0, z_std=20.0,
    diameter_um=5.0, myelinated=True,
    description="Cortical pyramidal — complex branching, high-dimensional")

MOTOR_ALPHA = TissueType(
    "motor_alpha", n_modes=3, z_mean=50.0, z_std=10.0,
    diameter_um=15.0, myelinated=True,
    description="Aα motor neuron — thick, fast, high reliability")

SENSORY_AB = TissueType(
    "sensory_ab", n_modes=3, z_mean=60.0, z_std=15.0,
    diameter_um=8.0, myelinated=True,
    description="Aβ mechanoreceptor — touch, pressure, proprioception")

PAIN_AD_FIBER = TissueType(
    "pain_ad_fiber", n_modes=2, z_mean=90.0, z_std=20.0,
    diameter_um=3.0, myelinated=True,
    description="Aδ fiber — sharp pain, temperature, first pain")

PAIN_C_FIBER = TissueType(
    "pain_c_fiber", n_modes=1, z_mean=120.0, z_std=30.0,
    diameter_um=0.8, myelinated=False,
    description="C fiber — dull pain, slow, unmyelinated, single mode")

CARDIAC_PURKINJE = TissueType(
    "cardiac_purkinje", n_modes=4, z_mean=40.0, z_std=8.0,
    diameter_um=70.0, myelinated=False,
    description="Cardiac Purkinje — massive diameter, specialised conduction")

AUTONOMIC_PREGANGLIONIC = TissueType(
    "autonomic_preganglionic", n_modes=2, z_mean=100.0, z_std=25.0,
    diameter_um=3.0, myelinated=True,
    description="Autonomic B fiber — moderate speed, visceral control")

ENTERIC_NEURON = TissueType(
    "enteric_neuron", n_modes=1, z_mean=110.0, z_std=35.0,
    diameter_um=1.0, myelinated=False,
    description="Enteric nervous system — gut brain, unmyelinated")

ALL_TISSUE_TYPES = [
    CORTICAL_PYRAMIDAL, MOTOR_ALPHA, SENSORY_AB, PAIN_AD_FIBER,
    PAIN_C_FIBER, CARDIAC_PURKINJE, AUTONOMIC_PREGANGLIONIC, ENTERIC_NEURON,
]


# ============================================================================
# 1. GammaNode — a neuron with K-dimensional impedance
# ============================================================================


@dataclass
class GammaNode:
    """
    A node in the Γ-topology network.

    Each node has a K-dimensional impedance vector Z ∈ ℝ^K (one per mode /
    inner conductor of the multi-core coaxial cable).

    In biological terms:
      - Each Z_k corresponds to the impedance of one microtubule bundle
        or one frequency mode within the axon.
      - Different neurons naturally have different Z vectors — this
        heterogeneity IS the source of Γ-topology.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. "V1_excit_042").
    impedance : np.ndarray
        K-dimensional impedance vector, each element > 0.
    activation : np.ndarray
        K-dimensional activation (current signal amplitude per mode).
    """

    name: str
    impedance: np.ndarray       # shape (K,), each > 0
    activation: np.ndarray      # shape (K,), current signal amplitude
    _cumulative_delta_z: float = 0.0   # track total |ΔZ| for diagnostics

    def __post_init__(self) -> None:
        self.impedance = np.asarray(self.impedance, dtype=np.float64)
        self.activation = np.asarray(self.activation, dtype=np.float64)
        assert self.impedance.shape == self.activation.shape, \
            "impedance and activation must have same shape"
        assert np.all(self.impedance > 0), \
            f"All impedances must be > 0, got min={self.impedance.min()}"

    @property
    def K(self) -> int:
        """Number of inner conductors / modes."""
        return len(self.impedance)

    @property
    def mean_impedance(self) -> float:
        return float(np.mean(self.impedance))


# ============================================================================
# 2. MulticoreChannel — K×K matrix Γ between two nodes
# ============================================================================


@dataclass
class MulticoreChannel:
    """
    A multi-core coaxial channel between two GammaNodes.

    For K inner conductors, the reflection is described by a K×K matrix:

        Γ_matrix = diag(  (Z_target_k − Z_source_k) / (Z_target_k + Z_source_k)  )

    In the simplest (uncoupled) case the matrix is diagonal — each mode
    reflects independently.  The off-diagonal elements represent cross-mode
    coupling via the dielectric (to be added in future extensions).

    C1 verification:
        diag(Γ²) + diag(T) = 1  for each mode k
    """

    source: GammaNode
    target: GammaNode

    # Dielectric coupling matrix (K×K), identity = no cross-coupling
    # Off-diagonal terms model inter-mode interaction through shared myelin
    coupling: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # Heterogeneous K is allowed — mode projection handles mismatch.
        # Only the common modes (K_common = min(K_src, K_tgt)) can interact;
        # excess source modes are totally reflected (Γ=1, T=0).
        K_common = min(self.source.K, self.target.K)
        if self.coupling is None:
            self.coupling = np.eye(K_common) if K_common > 0 else np.eye(1)

    @property
    def K(self) -> int:
        """Source mode count (backward-compatible)."""
        return self.source.K

    @property
    def K_common(self) -> int:
        """Number of modes that can transmit (min of source, target K)."""
        return min(self.source.K, self.target.K)

    @property
    def K_excess(self) -> int:
        """Number of totally-reflected excess source modes."""
        return max(0, self.source.K - self.target.K)

    @property
    def dimensional_mismatch(self) -> float:
        """Fraction of source modes that are cut off: K_excess / K_src."""
        return self.K_excess / self.source.K if self.source.K > 0 else 0.0

    # ── Core physics ──────────────────────────────────────────────────

    def gamma_vector(self) -> np.ndarray:
        """
        Per-mode reflection coefficient vector (K_src,).

        Common modes (k < K_common):
            Γ_k = (Z_target_k − Z_source_k) / (Z_target_k + Z_source_k)
        Excess source modes (k ≥ K_common):
            Γ_k = 1.0  (total reflection — target has no mode to receive)

        This is waveguide mode cutoff: a thick fiber (K=5) transmitting
        to a thin fiber (K=1) loses modes 2–5 completely.
        """
        zs = self.source.impedance   # (K_src,)
        zt = self.target.impedance   # (K_tgt,)
        K_common = self.K_common

        # Default: total reflection for all source modes
        gamma = np.ones(self.source.K)

        # Common modes: normal impedance-based Γ
        if K_common > 0:
            zs_c = zs[:K_common]
            zt_c = zt[:K_common]
            denom = zt_c + zs_c
            safe_denom = np.where(denom > 0, denom, 1.0)
            gamma[:K_common] = (zt_c - zs_c) / safe_denom

        return gamma

    def gamma_matrix(self) -> np.ndarray:
        """
        Reflection matrix (K_src × K_tgt) including cross-mode coupling.

        For homogeneous K: standard C · diag(γ) · C^T  (K × K).
        For heterogeneous K (K_src ≠ K_tgt):
          - Common block (K_common × K_common): coupled impedance Γ
          - Excess source → no target dimension (modes are reflected)
          - Excess target ← no source signal (modes stay empty)
        """
        K_src = self.source.K
        K_tgt = self.target.K
        K_common = self.K_common
        gv = self.gamma_vector()  # K_src-dim

        mat = np.zeros((K_src, K_tgt))
        if K_common > 0:
            gamma_diag = np.diag(gv[:K_common])
            coupled_block = self.coupling @ gamma_diag @ self.coupling.T
            mat[:K_common, :K_common] = coupled_block

        return mat

    def transmission_vector(self) -> np.ndarray:
        """
        Per-mode transmission coefficient T_k = 1 − Γ_k².

        C1: Γ_k² + T_k = 1   ∀k
        """
        gv = self.gamma_vector()
        return 1.0 - gv ** 2

    def reflected_energy(self) -> float:
        """Total reflected energy across all modes: Σ_k Γ_k² · |x_source_k|²."""
        gv = self.gamma_vector()
        act = np.clip(self.source.activation, -10.0, 10.0)
        return float(np.sum(gv ** 2 * act ** 2))

    def transmitted_signal(self) -> np.ndarray:
        """
        Signal arriving at target: K_tgt-dimensional.

        Common modes:  x_out_k = √T_k · x_in_k
        Excess target modes (K_tgt > K_src):  x_out_k = 0
            (no source signal exists in those higher modes)

        Information is dimensionally compressed when K_src > K_tgt,
        and dimensionally sparse when K_src < K_tgt.
        """
        K_common = self.K_common
        tv = self.transmission_vector()   # K_src-dim
        x_in = self.source.activation     # K_src-dim

        x_out = np.zeros(self.target.K)
        if K_common > 0:
            tv_safe = np.clip(tv[:K_common], 0.0, 1.0)
            x_out[:K_common] = np.sqrt(tv_safe) * x_in[:K_common]

        return x_out

    def verify_c1(self, tol: float = 1e-10) -> bool:
        """Verify energy conservation: Γ² + T = 1 for every mode."""
        gv = self.gamma_vector()
        tv = self.transmission_vector()
        residual = np.abs(gv ** 2 + tv - 1.0)
        return bool(np.all(residual < tol))

    def gamma_norm(self) -> float:
        """Frobenius norm of the Γ matrix — scalar summary of mismatch."""
        gm = self.gamma_matrix()
        return float(np.sqrt(np.sum(gm ** 2)))

    def scalar_gamma(self) -> float:
        """
        RMS scalar Γ — single number summarising the connection quality.

        Γ_scalar = √( (1/K) Σ_k Γ_k² )
        """
        gv = self.gamma_vector()
        return float(np.sqrt(np.mean(gv ** 2)))

    # ── Directed action decomposition ─────────────────────────────────
    #
    #  Theorem (Dimensional Cost Irreducibility):
    #    For K_i > K_j, the cutoff modes contribute:
    #        A_cutoff(i→j) = (K_i - K_j)   (×1² per mode, Γ=1)
    #    This term is NOT differentiable w.r.t. any impedance.
    #    No learning rule can reduce it.  It is a geometric constraint.
    #
    #  Therefore the directed action decomposes:
    #    A(i→j) = A_impedance(i→j) + A_cutoff(i→j)
    #
    #  where:
    #    A_impedance = Σ_{k≤K_common} Γ_k²  (learnable via C2 gradient)
    #    A_cutoff    = (K_i - K_j)⁺          (structural, irreducible)
    #

    def impedance_action(self) -> float:
        """Learnable part: Σ_{common modes} Γ_k² (reducible by C2 gradient)."""
        gv = self.gamma_vector()
        K_c = self.K_common
        if K_c == 0:
            return 0.0
        return float(np.sum(gv[:K_c] ** 2))

    def cutoff_action(self) -> float:
        """Structural part: (K_src - K_tgt)⁺ (irreducible geometric cost)."""
        return float(self.K_excess)  # Each cutoff mode: Γ=1, so Γ²=1

    def directed_action(self) -> float:
        """
        Full directed action A(i→j) = A_impedance + A_cutoff.

        A(i→j) = Σ_{k≤K_common} Γ_k² + (K_i - K_j)⁺

        Note: A(i→j) ≠ A(j→i) when K_i ≠ K_j.
        This asymmetry is a fundamental property of heterogeneous networks.
        """
        return self.impedance_action() + self.cutoff_action()

    @property
    def dimension_gap(self) -> int:
        """Absolute dimensional difference |K_src - K_tgt|."""
        return abs(self.source.K - self.target.K)


# ============================================================================
# 3. GammaTopology — the dynamic network
# ============================================================================


class GammaTopology:
    """
    Dynamic Γ-topology network of N nodes with K modes each.

    The network has NO predefined adjacency.  Instead, the full N×N Γ matrix
    exists implicitly (every pair has a well-defined Γ).  "Connectivity" is
    an emergent property: pairs with small |Γ| are effectively connected
    (high transmission), pairs with |Γ| → 1 are effectively disconnected
    (total reflection).

    Active edges:
        We track a set of *active* edges — connections through which signal
        is currently flowing.  Only active edges participate in Hebbian
        updates.  New edges can spontaneously activate when a node's
        activation spreads and encounters a near-matched neighbour.

    Evolution:
        Each tick:
        1. Compute Γ for all active edges
        2. Transmit signals (x_out = √T · x_in)
        3. Accumulate received activation at target nodes
        4. Hebbian update: ΔZ_i = −η · Γ_ij · x_i · x_j  (C2)
        5. Optionally: spontaneous edge activation / pruning

    Parameters
    ----------
    nodes : list of GammaNode
        Initial population.  Impedances should be diverse (as in biology).
    eta : float
        Hebbian learning rate (C2).
    gamma_threshold : float
        |Γ| below which an edge is considered "effectively connected"
        for topology metrics.
    z_min, z_max : float
        Physical bounds on impedance (prevent runaway).
    """

    def __init__(
        self,
        nodes: List[GammaNode],
        eta: float = 0.01,
        gamma_threshold: float = 0.3,
        z_min: float = 1.0,
        z_max: float = 500.0,
        max_dimension_gap: Optional[int] = None,
    ) -> None:
        self.nodes = {n.name: n for n in nodes}
        self.eta = eta
        self.gamma_threshold = gamma_threshold
        self.z_min = z_min
        self.z_max = z_max
        self.max_dimension_gap = max_dimension_gap

        # Active edges: set of (source_name, target_name)
        self.active_edges: Dict[Tuple[str, str], MulticoreChannel] = {}

        # History for analysis
        self.tick_count: int = 0
        self._history: List[Dict[str, Any]] = []

    @property
    def N(self) -> int:
        return len(self.nodes)

    @property
    def K(self) -> int:
        """Maximum mode count across all nodes."""
        if not self.nodes:
            return 0
        return max(n.K for n in self.nodes.values())

    @property
    def K_min(self) -> int:
        """Minimum mode count across all nodes."""
        if not self.nodes:
            return 0
        return min(n.K for n in self.nodes.values())

    @property
    def is_heterogeneous(self) -> bool:
        """True if nodes have different mode counts (K)."""
        if not self.nodes:
            return False
        ks = {n.K for n in self.nodes.values()}
        return len(ks) > 1

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def create_random(
        cls,
        n_nodes: int,
        n_modes: int,
        z_mean: float = 75.0,
        z_std: float = 30.0,
        initial_connectivity: float = 0.3,
        eta: float = 0.01,
        seed: Optional[int] = None,
    ) -> "GammaTopology":
        """
        Create a network with random impedance diversity.

        This models the biological reality: every neuron's impedance is
        drawn from a distribution, not a single value.  The resulting
        Γ matrix is dense and heterogeneous — exactly like real cortex.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        n_modes : int
            Modes per node (inner conductors).
        z_mean, z_std : float
            Mean and std of impedance distribution.
        initial_connectivity : float
            Fraction of possible edges initially active (0–1).
        eta : float
            Hebbian learning rate.
        seed : int, optional
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)

        nodes = []
        for i in range(n_nodes):
            z = np.abs(rng.normal(z_mean, z_std, size=n_modes))
            z = np.clip(z, 1.0, 500.0)
            activation = rng.uniform(0.0, 0.1, size=n_modes)
            nodes.append(GammaNode(
                name=f"n{i:03d}",
                impedance=z,
                activation=activation,
            ))

        topo = cls(nodes=nodes, eta=eta)

        # Activate random subset of edges
        names = [n.name for n in nodes]
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if i != j and rng.random() < initial_connectivity:
                    topo.activate_edge(ni, nj)

        return topo

    # ── Edge management ───────────────────────────────────────────────

    def activate_edge(self, source_name: str, target_name: str,
                      coupling: Optional[np.ndarray] = None) -> Optional[MulticoreChannel]:
        """Activate an edge (create channel) between two nodes.

        If max_dimension_gap is set and the dimensional gap exceeds it,
        the edge is rejected (returns None).  This models the biological
        constraint that very different fiber types cannot form direct
        synapses — relay neurons are needed.
        """
        key = (source_name, target_name)
        if key not in self.active_edges:
            src = self.nodes[source_name]
            tgt = self.nodes[target_name]
            gap = abs(src.K - tgt.K)
            if self.max_dimension_gap is not None and gap > self.max_dimension_gap:
                return None  # dimensional gap too large
            ch = MulticoreChannel(source=src, target=tgt, coupling=coupling)
            self.active_edges[key] = ch
        return self.active_edges[key]

    def deactivate_edge(self, source_name: str, target_name: str) -> None:
        """Remove an active edge."""
        key = (source_name, target_name)
        self.active_edges.pop(key, None)

    # ── Core tick ─────────────────────────────────────────────────────

    def tick(
        self,
        external_stimuli: Optional[Dict[str, np.ndarray]] = None,
        enable_spontaneous: bool = True,
    ) -> Dict[str, Any]:
        """
        One tick of the dynamic Γ-topology.

        Steps:
        1. Apply external stimuli to specified nodes
        2. For each active edge: compute Γ, transmit signal
        3. Accumulate received signals at each node
        4. Hebbian update: ΔZ_i = −η · Γ_ij · x_i · x_j
        5. Spontaneous edge activation / pruning
        6. Record metrics

        Returns dict with topology metrics for this tick.
        """
        self.tick_count += 1

        # 1. External stimuli
        if external_stimuli:
            for name, stim in external_stimuli.items():
                if name in self.nodes:
                    self.nodes[name].activation = np.asarray(stim, dtype=np.float64)

        # 2-3. Signal transmission
        incoming: Dict[str, np.ndarray] = {
            name: np.zeros(node.K) for name, node in self.nodes.items()
        }
        total_reflected = 0.0
        total_transmitted = 0.0
        edge_gammas: List[float] = []

        for (src_name, tgt_name), ch in list(self.active_edges.items()):
            # Refresh channel references (nodes may have updated impedance)
            ch.source = self.nodes[src_name]
            ch.target = self.nodes[tgt_name]

            gv = ch.gamma_vector()
            tv = ch.transmission_vector()

            # C1 check (should always hold by construction)
            assert ch.verify_c1(), \
                f"C1 violated at edge ({src_name}→{tgt_name}), tick {self.tick_count}"

            # Transmitted signal
            x_out = ch.transmitted_signal()
            incoming[tgt_name] += x_out

            # Energy accounting
            reflected = ch.reflected_energy()
            transmitted = float(np.sum(x_out ** 2))
            total_reflected += reflected
            total_transmitted += transmitted

            edge_gammas.append(ch.scalar_gamma())

        # Update activations from received signals (leaky integrator)
        decay = 0.9
        activation_cap = 10.0  # prevent runaway accumulation
        for name, node in self.nodes.items():
            node.activation = decay * node.activation + incoming[name]
            node.activation = np.clip(node.activation, -activation_cap, activation_cap)

        # 4. Impedance matching via gradient descent on A[Γ] = ΣΓ²
        #
        #    For edge (i,j): Γ_ij = (Z_j − Z_i) / (Z_j + Z_i)
        #    ∂(Γ²)/∂Z_i = −4·Γ·Z_j / (Z_i + Z_j)²   → gradient descent: ΔZ_i ∝ +Γ
        #    ∂(Γ²)/∂Z_j = +4·Γ·Z_i / (Z_i + Z_j)²   → gradient descent: ΔZ_j ∝ −Γ
        #
        #    Both nodes converge toward each other: Z_i ↑ and Z_j ↓ when Γ > 0.
        #    Modulated by x_pre · x_post (only active connections adapt — Hebbian gating).
        #    This is the true minimum-reflection gradient: A[Γ] → min.
        #
        for (src_name, tgt_name), ch in list(self.active_edges.items()):
            src = self.nodes[src_name]
            tgt = self.nodes[tgt_name]
            gv = ch.gamma_vector()       # K_src-dimensional
            K_common = ch.K_common

            x_pre = src.activation       # K_src-dim
            x_post = tgt.activation      # K_tgt-dim

            # Hebbian gate: only common modes can form activity correlation
            gate = x_pre[:K_common] * x_post[:K_common]

            # Source: ΔZ_src = +η · Γ · gate  (push toward target)
            # Only common modes update; excess modes have fixed Γ=1 (no gradient)
            delta_z_src = np.zeros(src.K)
            delta_z_src[:K_common] = self.eta * gv[:K_common] * gate
            new_z_src = np.clip(src.impedance + delta_z_src, self.z_min, self.z_max)
            src._cumulative_delta_z += float(np.sum(np.abs(delta_z_src)))
            src.impedance = new_z_src

            # Target: ΔZ_tgt = −η · Γ · gate  (push toward source)
            delta_z_tgt = np.zeros(tgt.K)
            delta_z_tgt[:K_common] = -self.eta * gv[:K_common] * gate
            new_z_tgt = np.clip(tgt.impedance + delta_z_tgt, self.z_min, self.z_max)
            tgt._cumulative_delta_z += float(np.sum(np.abs(delta_z_tgt)))
            tgt.impedance = new_z_tgt

        # 5. Spontaneous activation / pruning
        edges_born = 0
        edges_pruned = 0
        if enable_spontaneous:
            edges_born, edges_pruned = self._spontaneous_dynamics()

        # 6. Metrics
        gamma_array = np.array(edge_gammas) if edge_gammas else np.array([0.0])
        total_gamma_sq = float(np.sum(gamma_array ** 2))
        mean_gamma = float(np.mean(np.abs(gamma_array)))

        # Action decomposition (Dimensional Cost Irreducibility Theorem)
        a_impedance, a_cutoff = self.action_decomposition()

        # Effective connectivity: fraction of active edges with |Γ| < threshold
        if len(edge_gammas) > 0:
            effectively_connected = float(
                np.mean(gamma_array < self.gamma_threshold)
            )
        else:
            effectively_connected = 0.0

        metrics = {
            "tick": self.tick_count,
            "total_gamma_sq": total_gamma_sq,
            "mean_abs_gamma": mean_gamma,
            "action_impedance": a_impedance,
            "action_cutoff": a_cutoff,
            "action_total": a_impedance + a_cutoff,
            "total_reflected_energy": total_reflected,
            "total_transmitted_energy": total_transmitted,
            "active_edges": len(self.active_edges),
            "effectively_connected_ratio": effectively_connected,
            "edges_born": edges_born,
            "edges_pruned": edges_pruned,
        }
        self._history.append(metrics)
        return metrics

    # ── Spontaneous dynamics ──────────────────────────────────────────

    def _spontaneous_dynamics(self) -> Tuple[int, int]:
        """
        Dimension-aware pruning + sprouting.

        Pruning is layered (from the Irreducibility Theorem):
          1. DIMENSION PRUNE: |K_src − K_tgt| > max_dimension_gap → immediate removal
          2. COMBINED PRUNE: cutoff > 0 AND impedance_action > tolerance → remove
          3. TOTAL PRUNE: scalar Γ > 0.95 → remove (legacy)

        Sprouting respects max_dimension_gap: never create an edge that
        would be immediately pruned.

        This causes the network to self-organise into dimensional layers:
        K=5 connects to K=4 or K=3, but never directly to K=1.
        Exactly like real neuroanatomy: cortex → spinal cord → periphery.

        Returns (edges_born, edges_pruned).
        """
        to_prune = []
        for key, ch in self.active_edges.items():
            # Layer 1: dimension gap too large (structural impossibility)
            if (self.max_dimension_gap is not None
                    and ch.dimension_gap > self.max_dimension_gap):
                to_prune.append(key)
                continue

            # Layer 2: has cutoff AND remaining impedance is also bad
            if ch.K_excess > 0 and ch.impedance_action() > 0.5:
                to_prune.append(key)
                continue

            # Layer 3: legacy — total mismatch
            if ch.scalar_gamma() > 0.95:
                to_prune.append(key)

        for key in to_prune:
            del self.active_edges[key]

        # Sprouting: respect dimension gap constraint
        edges_born = 0
        node_names = list(self.nodes.keys())
        if len(node_names) < 2:
            return edges_born, len(to_prune)

        rng = np.random.default_rng(self.tick_count)
        for name, node in self.nodes.items():
            mean_act = float(np.mean(np.abs(node.activation)))
            if mean_act > 0.05:
                candidate = node_names[rng.integers(len(node_names))]
                if candidate != name and (name, candidate) not in self.active_edges:
                    tgt_node = self.nodes[candidate]

                    # Reject if dimension gap too large
                    gap = abs(node.K - tgt_node.K)
                    if (self.max_dimension_gap is not None
                            and gap > self.max_dimension_gap):
                        continue

                    K_c = min(node.K, tgt_node.K)
                    K_max = max(node.K, tgt_node.K)
                    if K_c == 0 or K_max == 0:
                        continue
                    z_diff = np.abs(node.impedance[:K_c] - tgt_node.impedance[:K_c])
                    z_sum = node.impedance[:K_c] + tgt_node.impedance[:K_c]
                    safe_sum = np.where(z_sum > 0, z_sum, 1.0)
                    K_excess = K_max - K_c
                    common_gamma = float(np.mean(z_diff / safe_sum))
                    probe_gamma = (common_gamma * K_c + K_excess) / K_max
                    if probe_gamma < 0.7:
                        self.activate_edge(name, candidate)
                        edges_born += 1

        return edges_born, len(to_prune)

    # ── Topology analysis ─────────────────────────────────────────────

    def full_gamma_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Compute the full N×N scalar Γ matrix (RMS across modes).

        Returns
        -------
        gamma_mat : ndarray (N, N)
            gamma_mat[i,j] = RMS Γ between node i and node j.
        names : list of str
            Node names in order.
        """
        names = sorted(self.nodes.keys())
        n = len(names)
        gamma_mat = np.zeros((n, n))

        for i, ni in enumerate(names):
            zi = self.nodes[ni].impedance
            Ki = len(zi)
            for j, nj in enumerate(names):
                if i == j:
                    continue
                zj = self.nodes[nj].impedance
                K_c = min(Ki, len(zj))

                if K_c == 0:
                    gamma_mat[i, j] = 1.0
                    continue

                # Common modes: impedance-based Γ
                zi_c = zi[:K_c]
                zj_c = zj[:K_c]
                denom = zi_c + zj_c
                safe_denom = np.where(denom > 0, denom, 1.0)
                gv = (zj_c - zi_c) / safe_denom

                # Excess source modes: total reflection Γ=1
                K_excess = max(0, Ki - len(zj))
                sum_gsq = float(np.sum(gv ** 2)) + K_excess
                gamma_mat[i, j] = np.sqrt(sum_gsq / Ki) if Ki > 0 else 0.0

        return gamma_mat, names

    def effective_adjacency(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Binary adjacency matrix: A[i,j] = 1 if |Γ_ij| < threshold.

        The topology IS this matrix — and it changes every tick as
        impedances evolve.
        """
        th = threshold or self.gamma_threshold
        gamma_mat, _ = self.full_gamma_matrix()
        adj = (gamma_mat < th) & (gamma_mat > 0)  # exclude self-loops
        return adj.astype(np.float64)

    def degree_distribution(self, threshold: Optional[float] = None) -> np.ndarray:
        """Degree of each node in the effective adjacency graph."""
        adj = self.effective_adjacency(threshold)
        return np.sum(adj, axis=1)

    def clustering_coefficient(self, threshold: Optional[float] = None) -> float:
        """
        Global clustering coefficient of the effective graph.

        Measures how much neighbours of a node are also neighbours of
        each other — a key metric for small-world topology.
        """
        adj = self.effective_adjacency(threshold)
        n = adj.shape[0]
        if n < 3:
            return 0.0

        cc_list = []
        for i in range(n):
            neighbours = np.where(adj[i] > 0)[0]
            k = len(neighbours)
            if k < 2:
                cc_list.append(0.0)
                continue
            # Count edges between neighbours
            sub = adj[np.ix_(neighbours, neighbours)]
            actual_edges = np.sum(sub) / 2.0
            possible_edges = k * (k - 1) / 2.0
            cc_list.append(actual_edges / possible_edges)

        return float(np.mean(cc_list))

    # ── Fractal analysis ─────────────────────────────────────────────

    def shortest_path_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        All-pairs shortest-path distance matrix on the active edge graph.

        Uses BFS (unweighted) on the symmetrised active edge graph.
        Unreachable pairs get distance = N (sentinel for infinity).

        Returns
        -------
        dist : ndarray (N, N), dtype int
        names : list of str
        """
        names = sorted(self.nodes.keys())
        idx = {name: i for i, name in enumerate(names)}
        n = len(names)
        dist = np.full((n, n), n, dtype=np.int32)
        np.fill_diagonal(dist, 0)

        # Build adjacency list from active edges (symmetrised)
        adj_list: Dict[int, List[int]] = {i: [] for i in range(n)}
        for (src, tgt) in self.active_edges:
            si, ti = idx[src], idx[tgt]
            adj_list[si].append(ti)
            adj_list[ti].append(si)

        # BFS from each node
        from collections import deque
        for start in range(n):
            visited = np.zeros(n, dtype=bool)
            visited[start] = True
            queue = deque([(start, 0)])
            while queue:
                node, d = queue.popleft()
                dist[start, node] = d
                for nb in adj_list[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append((nb, d + 1))

        return dist, names

    def box_counting_dimension(
        self,
        l_min: int = 1,
        l_max: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute the fractal dimension of the Γ-topology via box-counting.

        Uses the Song-Havlin-Makse (2005) network box-counting algorithm:
        1. Compute all-pairs shortest-path distances d(i,j)
        2. For each box diameter ℓ_B, find the minimum number of boxes
           N_B(ℓ_B) such that all nodes in each box are within distance ℓ_B
        3. Fit: log N_B = -D_f · log ℓ_B + const

        The greedy colouring algorithm is used for the box-covering
        (compact-box-burning variant): nodes are assigned to boxes
        starting from highest-degree nodes; a node can join an existing
        box if its distance to ALL nodes already in that box is < ℓ_B.

        Parameters
        ----------
        l_min : int
            Minimum box diameter to test (≥ 1).
        l_max : int, optional
            Maximum box diameter.  Default: network diameter.

        Returns
        -------
        dict with:
          - 'D_f': fitted fractal dimension
          - 'R2': R² of the log-log fit
          - 'l_values': list of ℓ_B tested
          - 'N_B_values': corresponding box counts
          - 'diameter': graph diameter
          - 'intercept': fit intercept

        References
        ----------
        Song, Havlin, Makse, "Self-similarity of complex networks",
        Nature 433, 392-395 (2005).
        """
        dist, names = self.shortest_path_matrix()
        n = len(names)

        if n < 4:
            return {"D_f": float("nan"), "R2": 0.0, "l_values": [],
                    "N_B_values": [], "diameter": 0, "intercept": 0.0}

        # Connected component mask — only consider reachable nodes
        # (dist < n means reachable from node 0)
        reachable = dist[0] < n
        if np.sum(reachable) < 4:
            # Find largest connected component
            best_mask = reachable
            for s in range(n):
                mask_s = dist[s] < n
                if np.sum(mask_s) > np.sum(best_mask):
                    best_mask = mask_s
            reachable = best_mask

        comp_idx = np.where(reachable)[0]
        n_comp = len(comp_idx)
        if n_comp < 4:
            return {"D_f": float("nan"), "R2": 0.0, "l_values": [],
                    "N_B_values": [], "diameter": 0, "intercept": 0.0}

        # Sub-distance matrix for largest component
        dist_comp = dist[np.ix_(comp_idx, comp_idx)]
        diameter = int(np.max(dist_comp))

        if diameter < 2:
            return {"D_f": float("nan"), "R2": 0.0, "l_values": [],
                    "N_B_values": [], "diameter": diameter, "intercept": 0.0}

        if l_max is None:
            l_max = diameter

        l_values = list(range(max(1, l_min), l_max + 1))
        N_B_values = []

        for l_B in l_values:
            # Greedy box-covering: compact-box-burning (CBB) variant
            # Sort nodes by degree (descending) for greedy seed selection
            degrees = np.sum(dist_comp < l_B, axis=1) - 1  # exclude self
            order = np.argsort(-degrees)

            assigned = np.full(n_comp, -1, dtype=np.int32)
            n_boxes = 0

            for node in order:
                if assigned[node] >= 0:
                    continue
                # Start a new box with this node as seed
                box_id = n_boxes
                n_boxes += 1
                assigned[node] = box_id

                # Try to add other unassigned nodes to this box
                for other in order:
                    if assigned[other] >= 0:
                        continue
                    # Check: other must be within ℓ_B of ALL nodes in box
                    box_members = np.where(assigned == box_id)[0]
                    if np.all(dist_comp[other, box_members] < l_B):
                        assigned[other] = box_id

            N_B_values.append(n_boxes)

        # Log-log fit: log(N_B) = -D_f · log(ℓ_B) + c
        # Filter to valid range (N_B > 1 and N_B < n_comp)
        valid = [(l, nb) for l, nb in zip(l_values, N_B_values)
                 if 1 < nb < n_comp]

        if len(valid) < 2:
            # Not enough points for fit — try including boundary points
            valid = [(l, nb) for l, nb in zip(l_values, N_B_values) if nb > 0]

        if len(valid) < 2:
            return {"D_f": float("nan"), "R2": 0.0, "l_values": l_values,
                    "N_B_values": N_B_values, "diameter": diameter,
                    "intercept": 0.0}

        log_l = np.log([v[0] for v in valid])
        log_N = np.log([v[1] for v in valid])

        coeffs = np.polyfit(log_l, log_N, 1)
        D_f = -coeffs[0]  # negative slope = fractal dimension
        intercept = coeffs[1]

        # R²
        predicted = coeffs[0] * log_l + intercept
        ss_res = np.sum((log_N - predicted) ** 2)
        ss_tot = np.sum((log_N - np.mean(log_N)) ** 2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "D_f": float(D_f),
            "R2": float(R2),
            "l_values": l_values,
            "N_B_values": N_B_values,
            "diameter": diameter,
            "intercept": float(intercept),
            "n_component": n_comp,
        }

    def spectral_dimension(
        self,
        t_min: float = 0.01,
        t_max: float = 100.0,
        n_t: int = 80,
    ) -> Dict[str, Any]:
        """
        Compute the spectral dimension d_s from the graph Laplacian.

        The heat kernel K(t) = Tr(exp(−t·L)) captures how signals
        diffuse through the Γ-topology.  For a d_s-dimensional
        structure:

            K(t) ~ t^{−d_s / 2}    (intermediate t)

        Unlike box-counting, d_s is meaningful even for dense,
        small-diameter networks because it measures the *spectral*
        structure, not the geodesic structure.

        Physical interpretation
        -----------------------
        d_s determines how the impedance consensus signal spreads:
        - d_s < 2 : sub-diffusive  (restricted pathways)
        - d_s = 2 : normal diffusion (regular lattice)
        - d_s > 2 : super-diffusive (many parallel paths)

        For fractal networks, Alexander-Orbach: d_s = 2·d_f / d_w.

        Parameters
        ----------
        t_min, t_max : float
            Range of diffusion times for the fit (log-spaced).
        n_t : int
            Number of t-points to sample.

        Returns
        -------
        dict with:
          - 'd_s': spectral dimension
          - 'R2': R² of the log-log fit
          - 'd_s_midrange': d_s fitted only in the middle 60% of t range
          - 'R2_midrange': R² for the midrange fit
          - 'lambda_min': smallest positive eigenvalue (Fiedler value)
          - 'lambda_max': largest eigenvalue
          - 'n_zero_modes': number of zero eigenvalues (= connected components)
        """
        names = sorted(self.nodes.keys())
        idx = {name: i for i, name in enumerate(names)}
        n = len(names)

        if n < 4:
            return {"d_s": float("nan"), "R2": 0.0,
                    "d_s_midrange": float("nan"), "R2_midrange": 0.0,
                    "lambda_min": 0.0, "lambda_max": 0.0, "n_zero_modes": n}

        # Build symmetric adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for (src, tgt) in self.active_edges:
            si, ti = idx[src], idx[tgt]
            A[si, ti] = 1.0
            A[ti, si] = 1.0

        # Graph Laplacian: L = D − A
        degrees = np.sum(A, axis=1)
        L = np.diag(degrees) - A

        # Eigenvalues (real symmetric → all real)
        eigenvalues = np.linalg.eigvalsh(L)

        # Separate zero modes from positive spectrum
        pos_eigs = eigenvalues[eigenvalues > 1e-10]
        n_zero = int(np.sum(eigenvalues <= 1e-10))

        if len(pos_eigs) < 3:
            return {"d_s": float("nan"), "R2": 0.0,
                    "d_s_midrange": float("nan"), "R2_midrange": 0.0,
                    "lambda_min": 0.0, "lambda_max": 0.0, "n_zero_modes": n_zero}

        lambda_min = float(pos_eigs[0])
        lambda_max = float(pos_eigs[-1])

        # Heat kernel (excludes zero modes — they contribute constant 1 each)
        t_values = np.logspace(np.log10(t_min), np.log10(t_max), n_t)
        K_t = np.array([float(np.sum(np.exp(-pos_eigs * t))) for t in t_values])

        # --- Full-range fit: log K = -(d_s/2)·log t + const ---
        valid = K_t > 1e-15
        if np.sum(valid) < 3:
            return {"d_s": float("nan"), "R2": 0.0,
                    "d_s_midrange": float("nan"), "R2_midrange": 0.0,
                    "lambda_min": lambda_min, "lambda_max": lambda_max,
                    "n_zero_modes": n_zero}

        log_t = np.log(t_values[valid])
        log_K = np.log(K_t[valid])
        coeffs_full = np.polyfit(log_t, log_K, 1)
        d_s_full = -2.0 * coeffs_full[0]
        pred_full = coeffs_full[0] * log_t + coeffs_full[1]
        ss_res_f = np.sum((log_K - pred_full) ** 2)
        ss_tot_f = np.sum((log_K - np.mean(log_K)) ** 2)
        R2_full = 1.0 - ss_res_f / ss_tot_f if ss_tot_f > 0 else 0.0

        # --- Midrange fit (middle 60% of valid points) ---
        n_valid = int(np.sum(valid))
        lo = n_valid // 5         # skip first 20%
        hi = n_valid - n_valid // 5  # skip last 20%
        if hi - lo >= 3:
            log_t_mid = log_t[lo:hi]
            log_K_mid = log_K[lo:hi]
            coeffs_mid = np.polyfit(log_t_mid, log_K_mid, 1)
            d_s_mid = -2.0 * coeffs_mid[0]
            pred_mid = coeffs_mid[0] * log_t_mid + coeffs_mid[1]
            ss_res_m = np.sum((log_K_mid - pred_mid) ** 2)
            ss_tot_m = np.sum((log_K_mid - np.mean(log_K_mid)) ** 2)
            R2_mid = 1.0 - ss_res_m / ss_tot_m if ss_tot_m > 0 else 0.0
        else:
            d_s_mid = d_s_full
            R2_mid = R2_full

        return {
            "d_s": float(d_s_full),
            "R2": float(R2_full),
            "d_s_midrange": float(d_s_mid),
            "R2_midrange": float(R2_mid),
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "n_zero_modes": n_zero,
        }

    def k_level_analysis(self) -> Dict[str, Any]:
        """
        Structural analysis of the Γ-topology in K-level (dimensional) space.

        Measures how edges distribute across K-level distances |ΔK|
        after Hebbian evolution.  Self-similarity in K-space manifests
        as invariant density ratios ρ(ΔK)/ρ(0) across network sizes N.

        The K-space "fractal exponent" D_K is defined from:

            ρ(ΔK) ~ ΔK^{-D_K}     for ΔK ≥ 1

        where ρ(ΔK) = edges(ΔK) / possible_pairs(ΔK).  If the topology
        has self-similar connectivity across dimensional layers, D_K
        should be N-independent.

        With max_dimension_gap=2 and K ∈ {1,2,3,5}, only ΔK ∈ {0,1,2}
        carry edges, yielding 2 points for the power-law slope (exact fit).

        Returns
        -------
        dict with:
          - 'k_values': sorted list of distinct K values present
          - 'k_counts': dict {K: node_count}
          - 'edge_by_dk': dict {ΔK: edge_count}
          - 'possible_by_dk': dict {ΔK: possible_undirected_pairs}
          - 'density_by_dk': dict {ΔK: ρ(ΔK)}
          - 'D_K': fractal exponent in K-space
          - 'D_K_R2': R² of the fit (1.0 for 2 points)
          - 'density_ratio_1_0': ρ(1)/ρ(0)
          - 'density_ratio_2_0': ρ(2)/ρ(0)
        """
        # K distribution
        k_counts: Dict[int, int] = {}
        for name, node in self.nodes.items():
            k = node.K
            k_counts[k] = k_counts.get(k, 0) + 1
        k_values = sorted(k_counts.keys())

        # Edge count by ΔK (each undirected edge counted once)
        edge_by_dk: Dict[int, int] = {}
        counted_pairs: set = set()
        for (src, tgt) in self.active_edges:
            pair = (min(src, tgt), max(src, tgt))
            if pair not in counted_pairs:
                counted_pairs.add(pair)
                dk = abs(self.nodes[src].K - self.nodes[tgt].K)
                edge_by_dk[dk] = edge_by_dk.get(dk, 0) + 1

        # Possible undirected pairs by ΔK
        possible_by_dk: Dict[int, int] = {}
        # Same K: ΔK = 0
        total_same = 0
        for ki in k_values:
            total_same += k_counts[ki] * (k_counts[ki] - 1) // 2
        possible_by_dk[0] = total_same
        # Different K: ΔK > 0 (count each unordered pair of K-levels once)
        for i, ki in enumerate(k_values):
            for j in range(i + 1, len(k_values)):
                kj = k_values[j]
                dk = kj - ki
                possible_by_dk[dk] = possible_by_dk.get(dk, 0) + k_counts[ki] * k_counts[kj]

        # Density: ρ(ΔK) = edges / possible
        all_dk = sorted(set(list(edge_by_dk.keys()) + list(possible_by_dk.keys())))
        density_by_dk: Dict[int, float] = {}
        for dk in all_dk:
            p = possible_by_dk.get(dk, 0)
            e = edge_by_dk.get(dk, 0)
            density_by_dk[dk] = e / p if p > 0 else 0.0

        # Density ratios (should be N-invariant if self-similar)
        rho_0 = density_by_dk.get(0, 0.0)
        rho_1 = density_by_dk.get(1, 0.0)
        rho_2 = density_by_dk.get(2, 0.0)
        ratio_1_0 = rho_1 / rho_0 if rho_0 > 0 else float("nan")
        ratio_2_0 = rho_2 / rho_0 if rho_0 > 0 else float("nan")

        # K-space fractal exponent: ρ(ΔK) ~ ΔK^{-D_K} for ΔK ≥ 1
        dk_fit = [dk for dk in sorted(density_by_dk.keys())
                  if dk >= 1 and density_by_dk[dk] > 0]
        if len(dk_fit) >= 2:
            log_dk = np.log(np.array(dk_fit, dtype=np.float64))
            log_rho = np.log(np.array([density_by_dk[dk] for dk in dk_fit]))
            coeffs = np.polyfit(log_dk, log_rho, 1)
            D_K = -coeffs[0]
            pred = coeffs[0] * log_dk + coeffs[1]
            ss_res = np.sum((log_rho - pred) ** 2)
            ss_tot = np.sum((log_rho - np.mean(log_rho)) ** 2)
            R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        else:
            D_K = float("nan")
            R2 = 0.0

        return {
            "k_values": k_values,
            "k_counts": dict(k_counts),
            "edge_by_dk": dict(edge_by_dk),
            "possible_by_dk": dict(possible_by_dk),
            "density_by_dk": dict(density_by_dk),
            "D_K": float(D_K),
            "D_K_R2": float(R2),
            "density_ratio_1_0": float(ratio_1_0),
            "density_ratio_2_0": float(ratio_2_0),
        }

    def impedance_entropy(self, n_bins: int = 30) -> float:
        """
        Shannon entropy of the impedance distribution across all nodes/modes.

        Low entropy = ordered topology (impedances concentrated).
        High entropy = disordered (impedances spread uniformly).
        """
        all_z = np.concatenate([n.impedance for n in self.nodes.values()])
        hist, edges = np.histogram(all_z, bins=n_bins, density=True)
        bin_width = edges[1] - edges[0]
        # Differential entropy approximation
        hist = hist[hist > 0]
        entropy = -float(np.sum(hist * np.log(hist) * bin_width))
        return entropy

    def total_action(self) -> float:
        """
        The action functional A[Γ] = Σ_{active edges} Σ_k Γ_ij,k² · |x_i,k|²

        This is what the system minimises over time via C2 Hebbian learning.
        """
        action = 0.0
        for (src_name, _), ch in self.active_edges.items():
            action += ch.reflected_energy()
        return action

    def action_decomposition(self) -> Tuple[float, float]:
        """
        Decompose total action into learnable and structural parts.

        From the Dimensional Cost Irreducibility Theorem:

            A_total = A_impedance + A_cutoff

        where:
          A_impedance = Σ_edges Σ_{k ≤ K_common} Γ_k²   (reducible by C2)
          A_cutoff    = Σ_edges (K_src − K_tgt)⁺          (irreducible)

        Returns (A_impedance, A_cutoff).
        """
        a_impedance = 0.0
        a_cutoff = 0.0
        for ch in self.active_edges.values():
            a_impedance += ch.impedance_action()
            a_cutoff += ch.cutoff_action()
        return a_impedance, a_cutoff

    def optimizable_action(self) -> float:
        """A_impedance — the part of action that C2 gradient can reduce."""
        a_imp, _ = self.action_decomposition()
        return a_imp

    def structural_action(self) -> float:
        """A_cutoff — irreducible dimensional cost (geometric constraint)."""
        _, a_cut = self.action_decomposition()
        return a_cut

    def get_history(self) -> List[Dict[str, Any]]:
        """Return full tick-by-tick metrics history."""
        return list(self._history)

    def topology_summary(self) -> Dict[str, Any]:
        """Comprehensive topology snapshot."""
        gamma_mat, names = self.full_gamma_matrix()
        adj = self.effective_adjacency()
        degrees = self.degree_distribution()

        # Impedance statistics
        all_z = np.concatenate([n.impedance for n in self.nodes.values()])

        # Mode distribution
        k_values = [n.K for n in self.nodes.values()]
        k_unique, k_counts = np.unique(k_values, return_counts=True)

        return {
            "n_nodes": self.N,
            "n_modes": self.K,              # backward compat (max K)
            "n_modes_max": self.K,
            "n_modes_min": self.K_min,
            "is_heterogeneous": self.is_heterogeneous,
            "k_distribution": dict(zip(k_unique.tolist(), k_counts.tolist())),
            "n_active_edges": len(self.active_edges),
            "total_action": self.total_action(),
            "mean_gamma": float(np.mean(gamma_mat[gamma_mat > 0])) if np.any(gamma_mat > 0) else 0.0,
            "median_gamma": float(np.median(gamma_mat[gamma_mat > 0])) if np.any(gamma_mat > 0) else 0.0,
            "effectively_connected_edges": int(np.sum(adj) / 2),
            "mean_degree": float(np.mean(degrees)),
            "max_degree": int(np.max(degrees)) if len(degrees) > 0 else 0,
            "clustering_coefficient": self.clustering_coefficient(),
            "impedance_entropy": self.impedance_entropy(),
            "z_mean": float(np.mean(all_z)),
            "z_std": float(np.std(all_z)),
            "tick": self.tick_count,
        }

    # ── Anatomical factory ────────────────────────────────────────────

    @classmethod
    def create_anatomical(
        cls,
        tissue_composition: Dict[Any, int],
        initial_connectivity: float = 0.2,
        eta: float = 0.01,
        max_dimension_gap: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "GammaTopology":
        """
        Create a network with biologically heterogeneous fiber types.

        Each tissue type has a different K (mode count), reflecting the
        waveguide physics of real nerve fibers:
          - Thick myelinated motor fibers → K=3 (multi-mode)
          - Thin unmyelinated C-fibers → K=1 (single-mode)
          - Cortical pyramidal neurons → K=5 (high dimensional)

        Parameters
        ----------
        tissue_composition : dict
            Maps TissueType → count, e.g. {CORTICAL_PYRAMIDAL: 10, PAIN_C_FIBER: 5}
        initial_connectivity : float
            Fraction of possible edges initially active.
        eta : float
            Hebbian learning rate.
        seed : int, optional
            Random seed.
        """
        rng = np.random.default_rng(seed)
        nodes = []
        node_idx = 0

        for tissue, count in tissue_composition.items():
            for i in range(count):
                z = np.abs(rng.normal(tissue.z_mean, tissue.z_std,
                                      size=tissue.n_modes))
                z = np.clip(z, 1.0, 500.0)
                activation = rng.uniform(0.0, 0.1, size=tissue.n_modes)
                nodes.append(GammaNode(
                    name=f"{tissue.name[:6]}_{node_idx:03d}",
                    impedance=z,
                    activation=activation,
                ))
                node_idx += 1

        topo = cls(nodes=nodes, eta=eta, max_dimension_gap=max_dimension_gap)

        # Activate random edges
        names = [n.name for n in nodes]
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if i != j and rng.random() < initial_connectivity:
                    topo.activate_edge(ni, nj)

        return topo

    # ── Dimension adaptor relay nodes ─────────────────────────────────

    @staticmethod
    def optimal_relay_chain(K_src: int, K_tgt: int,
                            max_gap: int = 2) -> List[int]:
        """
        Compute the optimal relay K-sequence between K_src and K_tgt.

        From the Dimension Gradient Minimisation Path (Irreducibility Theorem
        Corollary): given source K_s and target K_t < K_s, the minimum
        per-hop cutoff cost is achieved by distributing the dimensional
        drop evenly across relay nodes:

            K_relay[i] = K_s - round(i * (K_s - K_t) / n_hops)

        Each relay has K ∈ [K_tgt, K_src] and consecutive gaps ≤ max_gap.

        Parameters
        ----------
        K_src : int
            Source mode count (higher).
        K_tgt : int
            Target mode count (lower).
        max_gap : int
            Maximum allowed dimensional gap per hop.

        Returns
        -------
        list of int
            Intermediate K values (excluding K_src and K_tgt themselves).
            Empty list if no relay is needed (|K_src - K_tgt| ≤ max_gap).

        Examples
        --------
        >>> GammaTopology.optimal_relay_chain(5, 1, max_gap=2)
        [3]
        >>> GammaTopology.optimal_relay_chain(5, 1, max_gap=1)
        [4, 3, 2]
        """
        if max_gap < 1:
            raise ValueError("max_gap must be ≥ 1")

        K_high = max(K_src, K_tgt)
        K_low = min(K_src, K_tgt)
        total_drop = K_high - K_low

        if total_drop <= max_gap:
            return []

        # Number of hops needed
        n_hops = math.ceil(total_drop / max_gap)
        # Build evenly-spaced intermediate K values
        relays = []
        for i in range(1, n_hops):
            k = K_high - round(i * total_drop / n_hops)
            k = max(K_low, min(K_high, k))
            relays.append(k)

        # Deduplicate and filter out endpoints
        relays = [k for k in relays if K_low < k < K_high]

        # If direction was ascending (K_src < K_tgt), reverse
        if K_src < K_tgt:
            relays = list(reversed(relays))

        return relays

    def insert_relay_nodes(
        self,
        source_name: str,
        target_name: str,
        relay_z_strategy: str = "interpolate",
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Insert relay nodes between a source and target to bridge a
        dimensional gap larger than max_dimension_gap.

        From the Irreducibility Theorem Corollary: the optimal relay K
        is any integer in [K_tgt, K_src].  Real nervous system relays
        (thalamus, dorsal horn) have K values that fall between upstream
        and downstream — this is the Dimension Gradient Minimisation Path.

        Each relay node gets an impedance interpolated between source and
        target (common modes), minimising A_imp at creation.

        Parameters
        ----------
        source_name : str
            Name of the source node (high K).
        target_name : str
            Name of the target node (low K).
        relay_z_strategy : str
            'interpolate' — Z is linearly interpolated between source/target.
            'random' — Z drawn from uniform range between source/target Z ranges.
        seed : int, optional
            Random seed for impedance generation.

        Returns
        -------
        list of str
            Names of inserted relay nodes (in order from source to target).
            Empty list if no relay is needed.
        """
        src = self.nodes[source_name]
        tgt = self.nodes[target_name]

        gap = self.max_dimension_gap if self.max_dimension_gap is not None else 2
        relay_ks = self.optimal_relay_chain(src.K, tgt.K, max_gap=gap)

        if not relay_ks:
            return []

        rng = np.random.default_rng(seed)

        # Build relay nodes
        relay_names: List[str] = []
        K_high = max(src.K, tgt.K)
        K_low = min(src.K, tgt.K)

        for idx, k_relay in enumerate(relay_ks):
            relay_name = f"relay_{source_name}_{target_name}_{idx:02d}"

            # Impedance interpolation
            if relay_z_strategy == "interpolate":
                # Fraction of dimensional progress from high to low
                if K_high == K_low:
                    frac = 0.5
                else:
                    frac = (K_high - k_relay) / (K_high - K_low)

                # Common modes: interpolate between src and tgt impedances
                K_c = min(src.K, tgt.K, k_relay)
                z = np.zeros(k_relay, dtype=np.float64)
                for m in range(k_relay):
                    if m < K_c:
                        z_s = src.impedance[m] if m < src.K else src.impedance[-1]
                        z_t = tgt.impedance[m] if m < tgt.K else tgt.impedance[-1]
                        z[m] = z_s * (1 - frac) + z_t * frac
                    else:
                        # Extra modes: interpolate from source's higher modes
                        z[m] = src.impedance[m] if m < src.K else src.impedance[-1]
            else:
                # Random strategy
                z_min_val = min(src.impedance.min(), tgt.impedance.min())
                z_max_val = max(src.impedance.max(), tgt.impedance.max())
                z = rng.uniform(z_min_val, z_max_val, size=k_relay)

            z = np.clip(z, self.z_min, self.z_max)
            activation = np.zeros(k_relay, dtype=np.float64)

            relay_node = GammaNode(
                name=relay_name,
                impedance=z,
                activation=activation,
            )
            self.nodes[relay_name] = relay_node
            relay_names.append(relay_name)

        # Wire the chain: source → relay[0] → relay[1] → ... → target
        chain = [source_name] + relay_names + [target_name]
        # Remove old direct edge if it existed
        self.deactivate_edge(source_name, target_name)
        self.deactivate_edge(target_name, source_name)

        for i in range(len(chain) - 1):
            self.activate_edge(chain[i], chain[i + 1])
            self.activate_edge(chain[i + 1], chain[i])  # bidirectional

        return relay_names

    def insert_all_relays(
        self,
        seed: Optional[int] = None,
    ) -> Dict[Tuple[str, str], List[str]]:
        """
        Scan all active edges and insert relay nodes where needed.

        For every edge where |K_src - K_tgt| > max_dimension_gap,
        replace the direct edge with a relay chain.

        Returns dict mapping (source, target) → list of relay node names.
        """
        if self.max_dimension_gap is None:
            return {}

        gap = self.max_dimension_gap
        to_relay: List[Tuple[str, str]] = []

        for (src_name, tgt_name), ch in list(self.active_edges.items()):
            if ch.dimension_gap > gap:
                # Only add once per unordered pair
                pair = tuple(sorted([src_name, tgt_name]))
                if pair not in [tuple(sorted(p)) for p in to_relay]:
                    to_relay.append((src_name, tgt_name))

        result: Dict[Tuple[str, str], List[str]] = {}
        for i, (src, tgt) in enumerate(to_relay):
            relay_seed = seed + i if seed is not None else None
            relays = self.insert_relay_nodes(src, tgt, seed=relay_seed)
            if relays:
                result[(src, tgt)] = relays

        return result

    def relay_path_cutoff(self, path: List[str]) -> Tuple[float, float]:
        """
        Compute total A_imp and A_cut along a node path.

        Parameters
        ----------
        path : list of str
            Ordered node names (e.g. ["cortex", "relay_01", "c_fiber"]).

        Returns
        -------
        (total_A_imp, total_A_cut) : tuple of float
        """
        total_imp = 0.0
        total_cut = 0.0
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            if key in self.active_edges:
                ch = self.active_edges[key]
                total_imp += ch.impedance_action()
                total_cut += ch.cutoff_action()
            else:
                # Edge not active — compute hypothetical
                src = self.nodes[path[i]]
                tgt = self.nodes[path[i + 1]]
                ch = MulticoreChannel(source=src, target=tgt)
                total_imp += ch.impedance_action()
                total_cut += ch.cutoff_action()
        return total_imp, total_cut

    # ── Mode cutoff analysis ──────────────────────────────────────────

    def mode_cutoff_report(self) -> Dict[str, Any]:
        """
        Analyse dimensional mismatch across the network.

        In waveguide physics, when a K=5 source connects to a K=1 target,
        modes 2-5 are totally reflected (Γ=1, T=0).  This method reports
        the extent of such mode cutoff in the network.

        Returns dict with:
          - k_distribution: {K: count} mapping
          - cutoff_edges: number of edges where K_src > K_tgt
          - cutoff_fraction: fraction of edges with mode cutoff
          - total_excess_modes: total modes lost to cutoff
          - total_cutoff_energy: reflected energy from cutoff modes only
        """
        k_values = [node.K for node in self.nodes.values()]
        k_unique, k_counts = np.unique(k_values, return_counts=True)

        cutoff_edges = 0
        total_excess_modes = 0
        total_cutoff_energy = 0.0
        total_edges = len(self.active_edges)

        for (src_name, tgt_name), ch in self.active_edges.items():
            K_excess = ch.K_excess
            if K_excess > 0:
                cutoff_edges += 1
                total_excess_modes += K_excess
                # Energy in excess modes (Γ=1 → all reflected)
                src_act = np.clip(ch.source.activation, -10.0, 10.0)
                cutoff_energy = float(np.sum(src_act[ch.K_common:] ** 2))
                total_cutoff_energy += cutoff_energy

        # Cross-K connectivity matrix: how many edges between each K pair
        cross_k = {}
        for (src_name, tgt_name), ch in self.active_edges.items():
            key = (ch.source.K, ch.target.K)
            cross_k[key] = cross_k.get(key, 0) + 1

        return {
            "k_distribution": dict(zip(k_unique.tolist(), k_counts.tolist())),
            "k_min": int(min(k_values)) if k_values else 0,
            "k_max": int(max(k_values)) if k_values else 0,
            "is_heterogeneous": self.is_heterogeneous,
            "cutoff_edges": cutoff_edges,
            "cutoff_fraction": cutoff_edges / total_edges if total_edges > 0 else 0.0,
            "total_excess_modes": total_excess_modes,
            "total_cutoff_energy": total_cutoff_energy,
            "cross_k_connectivity": cross_k,
        }

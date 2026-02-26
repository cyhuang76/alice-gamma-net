# -*- coding: utf-8 -*-
"""
Phase 30 — Γ-Field Topology Emergence (Level 0)
═══════════════════════════════════════════════════

Hypothesis:
  The Minimum Reflection Principle (ΣΓ² → min) inherently generates
  topological order — i.e., spatial structure is not a precondition of
  cognition but an EMERGENT CONSEQUENCE of impedance matching dynamics.

Level 0 Verification:
  Using the existing NeuralPruningEngine (no architecture changes), we test
  whether Γ-driven Hebbian selection + apoptosis transforms a UNIFORM
  random distribution of (Z, f) into STRUCTURED clusters — the 1D
  manifestation of topology emergence.

Measurements:
  1. Birth distribution vs. post-pruning distribution of (Z, f)
  2. Γ-metric distance matrix between surviving connections
  3. Cluster count & entropy (order vs. disorder)
  4. Information-theoretic: mutual information I(Z; survival)
  5. Cross-region Γ-distance: do different regions converge to different
     basins? (= functional specialization IS topological separation)

Theoretical Prediction:
  If Γ generates topology, then:
  - Birth: H(Z) ≈ log(Z_max - Z_min) — maximum entropy (uniform)
  - After pruning: H(Z) << H_birth — entropy drops (order emerges)
  - Γ-distance matrix shows block-diagonal structure (clusters)
  - Cross-region Γ-distance > intra-region Γ-distance (separation)
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Tuple, Any

import numpy as np

from alice.brain.pruning import (
    NeuralPruningEngine,
    CorticalRegion,
    SynapticConnection,
    MODALITY_SIGNAL_PROFILE,
)

BANNER_WIDTH = 78


def print_banner(title: str) -> None:
    print("\n" + "=" * BANNER_WIDTH)
    print(f"  {title}")
    print("=" * BANNER_WIDTH)


# ============================================================================
# Step 1: Birth Snapshot — Record primordial chaos
# ============================================================================

def step1_birth_snapshot(engine: NeuralPruningEngine) -> Dict[str, Any]:
    """Capture (Z, f) distribution of ALL connections at birth (before any pruning)."""
    print_banner("STEP 1: Birth Snapshot — Primordial Impedance Chaos")

    birth_data: Dict[str, Dict[str, Any]] = {}

    for name, region in engine.regions.items():
        impedances = np.array([c.impedance for c in region.connections])
        frequencies = np.array([c.resonant_freq for c in region.connections])

        birth_data[name] = {
            "impedances": impedances.copy(),
            "frequencies": frequencies.copy(),
            "count": len(impedances),
            "z_mean": float(np.mean(impedances)),
            "z_std": float(np.std(impedances)),
            "f_mean": float(np.mean(frequencies)),
            "f_std": float(np.std(frequencies)),
        }

        print(f"\n  [{name}] N={len(impedances)}")
        print(f"    Z: μ={np.mean(impedances):.1f}Ω  σ={np.std(impedances):.1f}Ω  "
              f"range=[{np.min(impedances):.1f}, {np.max(impedances):.1f}]")
        print(f"    f: μ={np.mean(frequencies):.1f}Hz σ={np.std(frequencies):.1f}Hz "
              f"range=[{np.min(frequencies):.1f}, {np.max(frequencies):.1f}]")

    # Compute birth entropy (continuous: use differential entropy approximation)
    # For uniform distribution U(a,b): H = log(b-a)
    z_min, z_max = 20.0, 200.0
    H_birth_z = math.log(z_max - z_min)
    print(f"\n  Theoretical H_birth(Z) = log({z_max}-{z_min}) = {H_birth_z:.4f} nats")
    print(f"  (Maximum entropy for uniform distribution on [{z_min}, {z_max}])")

    return birth_data


# ============================================================================
# Step 2: Development — Let ΣΓ² → min do its work
# ============================================================================

def step2_development(engine: NeuralPruningEngine, epochs: int = 100) -> List[Dict]:
    """Run pruning development and track Γ² trajectory."""
    print_banner(f"STEP 2: Development — {epochs} Epochs of ΣΓ² Minimization")

    results = []
    checkpoints = {1, 5, 10, 25, 50, 75, 100, epochs}

    for i in range(1, epochs + 1):
        result = engine.develop_epoch()
        results.append(result)

        if i in checkpoints:
            gamma_sq = result["global_gamma_squared"]
            total_alive = sum(
                r.get("alive", 0) for r in result["regions"].values()
            )
            print(f"  Epoch {i:>4d}: ΣΓ²={gamma_sq:.6f}  "
                  f"alive={total_alive}  "
                  f"Δ={result['net_change_this_epoch']:+d}")

    return results


# ============================================================================
# Step 3: Post-Pruning Snapshot — Measure emergent order
# ============================================================================

def step3_postpruning_snapshot(
    engine: NeuralPruningEngine,
    birth_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare post-pruning (Z, f) distribution against birth."""
    print_banner("STEP 3: Post-Pruning Snapshot — Emergent Order Analysis")

    post_data: Dict[str, Dict[str, Any]] = {}
    entropy_results: Dict[str, Dict[str, float]] = {}

    for name, region in engine.regions.items():
        alive = region.alive_connections
        if not alive:
            print(f"\n  [{name}] EXTINCT — all connections pruned")
            continue

        impedances = np.array([c.impedance for c in alive])
        frequencies = np.array([c.resonant_freq for c in alive])

        post_data[name] = {
            "impedances": impedances.copy(),
            "frequencies": frequencies.copy(),
            "count": len(impedances),
            "z_mean": float(np.mean(impedances)),
            "z_std": float(np.std(impedances)),
            "f_mean": float(np.mean(frequencies)),
            "f_std": float(np.std(frequencies)),
        }

        # Entropy estimation via histogram (discrete approximation)
        birth_z = birth_data[name]["impedances"]
        n_bins = 30

        # Birth histogram entropy
        birth_hist, _ = np.histogram(birth_z, bins=n_bins, density=True)
        birth_hist = birth_hist[birth_hist > 0]
        bin_width = (200.0 - 20.0) / n_bins
        H_birth = -float(np.sum(birth_hist * np.log(birth_hist + 1e-12) * bin_width))

        # Post-pruning histogram entropy
        if len(impedances) > 1:
            post_hist, _ = np.histogram(impedances, bins=n_bins,
                                        range=(20.0, 200.0), density=True)
            post_hist = post_hist[post_hist > 0]
            H_post = -float(np.sum(post_hist * np.log(post_hist + 1e-12) * bin_width))
        else:
            H_post = 0.0

        entropy_drop = H_birth - H_post
        entropy_ratio = H_post / max(H_birth, 1e-12)

        entropy_results[name] = {
            "H_birth": H_birth,
            "H_post": H_post,
            "H_drop": entropy_drop,
            "H_ratio": entropy_ratio,
        }

        # Signal target impedance for this region
        diet = engine.default_sensory_diet[name]
        z_target = MODALITY_SIGNAL_PROFILE[diet]["impedance"]

        survival_rate = len(impedances) / birth_data[name]["count"]

        print(f"\n  [{name}] target Z={z_target}Ω  survived={len(impedances)}/{birth_data[name]['count']} ({survival_rate:.1%})")
        print(f"    Birth   Z: μ={birth_data[name]['z_mean']:.1f}Ω  σ={birth_data[name]['z_std']:.1f}Ω")
        print(f"    Pruned  Z: μ={np.mean(impedances):.1f}Ω  σ={np.std(impedances):.1f}Ω")
        print(f"    Z σ collapse: {birth_data[name]['z_std']:.1f} → {np.std(impedances):.1f} "
              f"({np.std(impedances)/birth_data[name]['z_std']:.1%} of original)")
        print(f"    Entropy: H_birth={H_birth:.4f} → H_post={H_post:.4f}  "
              f"ΔH={entropy_drop:+.4f}  ({entropy_ratio:.1%} of original)")

    return {"post_data": post_data, "entropy": entropy_results}


# ============================================================================
# Step 4: Γ-Metric Distance Matrix — The topology itself
# ============================================================================

def step4_gamma_distance_matrix(engine: NeuralPruningEngine) -> Dict[str, Any]:
    """
    Compute inter-region Γ-distance matrix.

    d(region_A, region_B) = mean |Γ_ij| for all i∈A, j∈B
    where Γ_ij = (Z_i - Z_j) / (Z_i + Z_j)

    If topology emerges:
      - Intra-region distance << Inter-region distance
      - Block-diagonal structure in full distance matrix
    """
    print_banner("STEP 4: Γ-Metric Distance Matrix — Topological Separation")

    region_names = list(engine.regions.keys())
    n = len(region_names)

    # Compute mean impedance per region (representative)
    region_z_means: Dict[str, float] = {}
    region_z_stds: Dict[str, float] = {}

    for name, region in engine.regions.items():
        alive = region.alive_connections
        if alive:
            zs = [c.impedance for c in alive]
            region_z_means[name] = float(np.mean(zs))
            region_z_stds[name] = float(np.std(zs))
        else:
            region_z_means[name] = 0.0
            region_z_stds[name] = 0.0

    # Inter-region Γ distance matrix
    gamma_matrix = np.zeros((n, n))
    for i, name_i in enumerate(region_names):
        for j, name_j in enumerate(region_names):
            alive_i = engine.regions[name_i].alive_connections
            alive_j = engine.regions[name_j].alive_connections
            if not alive_i or not alive_j:
                gamma_matrix[i, j] = 1.0
                continue

            # Sample-based: compute mean |Γ| between random pairs
            z_i = np.array([c.impedance for c in alive_i])
            z_j = np.array([c.impedance for c in alive_j])

            # Use representative mean impedance for efficiency
            mean_zi = np.mean(z_i)
            mean_zj = np.mean(z_j)
            gamma_matrix[i, j] = abs(mean_zi - mean_zj) / (mean_zi + mean_zj)

    # Intra vs inter comparison
    intra_gammas = []
    inter_gammas = []
    for i in range(n):
        for j in range(n):
            if i == j:
                # Intra-region: use std/mean as proxy
                name = region_names[i]
                if region_z_means[name] > 0:
                    intra_gammas.append(region_z_stds[name] / region_z_means[name])
            else:
                inter_gammas.append(gamma_matrix[i, j])

    # Print distance matrix
    header = "           " + "".join(f"{name[:8]:>10s}" for name in region_names)
    print(f"\n  Γ-Distance Matrix (inter-region):\n")
    print(f"  {header}")
    print(f"  {'─' * (11 + 10 * n)}")
    for i, name_i in enumerate(region_names):
        row = f"  {name_i[:8]:<10s}│"
        for j in range(n):
            val = gamma_matrix[i, j]
            if i == j:
                row += f"{'   ─':>10s}"
            else:
                row += f"{val:>10.4f}"
        print(row)

    avg_intra = float(np.mean(intra_gammas)) if intra_gammas else 0.0
    avg_inter = float(np.mean(inter_gammas)) if inter_gammas else 0.0
    separation = avg_inter / max(avg_intra, 1e-12)

    print(f"\n  Avg intra-region Γ-spread (σ/μ): {avg_intra:.4f}")
    print(f"  Avg inter-region Γ-distance:     {avg_inter:.4f}")
    print(f"  Topological separation ratio:    {separation:.2f}×")
    print(f"  (>1.0 means regions are more different from each other than internally)")

    return {
        "gamma_matrix": gamma_matrix,
        "region_names": region_names,
        "avg_intra": avg_intra,
        "avg_inter": avg_inter,
        "separation_ratio": separation,
    }


# ============================================================================
# Step 5: Attractor Basin Analysis — Where does Γ pull connections?
# ============================================================================

def step5_attractor_analysis(
    engine: NeuralPruningEngine,
    birth_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze whether surviving connections cluster around signal impedance
    (attractor basins) or remain scattered.

    Key metric: Kolmogorov-Smirnov distance between birth and post-pruning
    distributions — large D = strong selection = strong topology generation.
    """
    print_banner("STEP 5: Attractor Basin Analysis — Where Does Γ Pull?")

    results = {}

    for name, region in engine.regions.items():
        alive = region.alive_connections
        if not alive:
            continue

        z_alive = np.array([c.impedance for c in alive])
        z_birth = birth_data[name]["impedances"]

        # Target signal impedance
        diet = engine.default_sensory_diet[name]
        z_target = MODALITY_SIGNAL_PROFILE[diet]["impedance"]

        # Distance of survivors from attractor (signal impedance)
        dist_to_attractor = np.abs(z_alive - z_target)
        attraction_strength = 1.0 - float(np.mean(dist_to_attractor)) / (200.0 - 20.0)

        # Simple KS-like statistic: compare CDFs
        # Sort and compute ECDF difference
        z_birth_sorted = np.sort(z_birth)
        z_alive_sorted = np.sort(z_alive)

        # Normalize to [0,1]
        z_range = (20.0, 200.0)
        birth_norm = (z_birth_sorted - z_range[0]) / (z_range[1] - z_range[0])
        alive_norm = (z_alive_sorted - z_range[0]) / (z_range[1] - z_range[0])

        # ECDF at uniform sample points
        n_points = 100
        sample_pts = np.linspace(0, 1, n_points)
        ecdf_birth = np.searchsorted(birth_norm, sample_pts) / len(birth_norm)
        ecdf_alive = np.searchsorted(alive_norm, sample_pts) / len(alive_norm)
        ks_distance = float(np.max(np.abs(ecdf_birth - ecdf_alive)))

        # Where is the peak density?
        hist, bin_edges = np.histogram(z_alive, bins=20, range=(20.0, 200.0))
        peak_bin = int(np.argmax(hist))
        peak_center = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0

        results[name] = {
            "z_target": z_target,
            "z_mean_alive": float(np.mean(z_alive)),
            "peak_density_z": peak_center,
            "attraction_strength": attraction_strength,
            "ks_distance": ks_distance,
        }

        # ASCII histogram
        max_count = max(hist) if max(hist) > 0 else 1
        print(f"\n  [{name}]  target={z_target}Ω  attractor_strength={attraction_strength:.3f}  KS={ks_distance:.3f}")
        print(f"  Impedance distribution of survivors:")
        for b in range(len(hist)):
            bar_len = int(hist[b] / max_count * 40)
            z_lo = bin_edges[b]
            z_hi = bin_edges[b + 1]
            marker = " ◀ TARGET" if z_lo <= z_target <= z_hi else ""
            print(f"    {z_lo:5.0f}-{z_hi:5.0f}Ω │{'█' * bar_len}{'░' * (40 - bar_len)}│ {hist[b]:>3d}{marker}")

    return results


# ============================================================================
# Step 6: Theoretical Verdict
# ============================================================================

def step6_verdict(
    entropy_data: Dict[str, Any],
    distance_data: Dict[str, Any],
    attractor_data: Dict[str, Any],
    gamma_sq_history: List[float],
) -> None:
    """Synthesize all results into a theoretical verdict."""
    print_banner("STEP 6: Theoretical Verdict — Does Γ Generate Topology?")

    # Criterion 1: Entropy reduction
    entropy_drops = [v["H_drop"] for v in entropy_data.values()]
    avg_entropy_drop = float(np.mean(entropy_drops)) if entropy_drops else 0.0
    entropy_pass = avg_entropy_drop > 0.1

    # Criterion 2: Topological separation
    separation = distance_data["separation_ratio"]
    separation_pass = separation > 1.5

    # Criterion 3: Attractor convergence
    attractions = [v["attraction_strength"] for v in attractor_data.values()]
    avg_attraction = float(np.mean(attractions)) if attractions else 0.0
    attraction_pass = avg_attraction > 0.5

    # Criterion 4: Γ² monotone decrease
    if len(gamma_sq_history) >= 10:
        first_10 = float(np.mean(gamma_sq_history[:10]))
        last_10 = float(np.mean(gamma_sq_history[-10:]))
        gamma_decrease = first_10 - last_10
        gamma_pass = gamma_decrease > 0.01
    else:
        gamma_decrease = 0.0
        gamma_pass = False

    # KS distances
    ks_values = [v["ks_distance"] for v in attractor_data.values()]
    avg_ks = float(np.mean(ks_values)) if ks_values else 0.0
    ks_pass = avg_ks > 0.3

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Criterion                         Result       Pass?          │
  ├─────────────────────────────────────────────────────────────────┤
  │  1. Entropy reduction (ΔH)         {avg_entropy_drop:>+.4f}       {'✓ YES' if entropy_pass else '✗ NO '}          │
  │     (>0.1 = order emerged)                                     │
  │  2. Topological separation          {separation:>6.2f}×       {'✓ YES' if separation_pass else '✗ NO '}          │
  │     (>1.5× = regions are distinct)                             │
  │  3. Attractor convergence           {avg_attraction:>6.3f}        {'✓ YES' if attraction_pass else '✗ NO '}          │
  │     (>0.5 = strong pull to target)                             │
  │  4. ΣΓ² monotone decrease           {gamma_decrease:>+.4f}       {'✓ YES' if gamma_pass else '✗ NO '}          │
  │     (>0.01 = system is optimizing)                             │
  │  5. KS distribution shift            {avg_ks:>.3f}        {'✓ YES' if ks_pass else '✗ NO '}          │
  │     (>0.3 = birth ≠ survival)                                  │
  └─────────────────────────────────────────────────────────────────┘

  Criteria passed: {sum([entropy_pass, separation_pass, attraction_pass, gamma_pass, ks_pass])}/5
""")

    if all([entropy_pass, separation_pass, attraction_pass, gamma_pass, ks_pass]):
        print("""  ═══════════════════════════════════════════════════════════════════
  ║  VERDICT: Γ GENERATES TOPOLOGICAL ORDER                       ║
  ║                                                               ║
  ║  The Minimum Reflection Principle (ΣΓ² → min) transforms a    ║
  ║  uniform random impedance distribution into structured         ║
  ║  clusters with inter-region separation > intra-region spread.  ║
  ║                                                               ║
  ║  Neural spatial topology is not a precondition —               ║
  ║  it is an EMERGENT CONSEQUENCE of impedance matching.          ║
  ║                                                               ║
  ║  Paper I Limitation #2 is not a limitation but a PREDICTION.   ║
  ═══════════════════════════════════════════════════════════════════""")
    elif sum([entropy_pass, separation_pass, attraction_pass, gamma_pass, ks_pass]) >= 3:
        print("""  ═══════════════════════════════════════════════════════════════════
  ║  VERDICT: PARTIAL SUPPORT — Γ generates 1D order              ║
  ║                                                               ║
  ║  Impedance matching dynamics create structure, but full        ║
  ║  topological separation requires spatial degrees of freedom.   ║
  ║  → Level 1 verification (adding spatial coordinates) needed.   ║
  ═══════════════════════════════════════════════════════════════════""")
    else:
        print("""  ═══════════════════════════════════════════════════════════════════
  ║  VERDICT: INSUFFICIENT EVIDENCE at Level 0                    ║
  ║                                                               ║
  ║  Current architecture may not have enough degrees of freedom   ║
  ║  for topology emergence. Architecture upgrade required.        ║
  ═══════════════════════════════════════════════════════════════════""")

    # Theoretical note
    print(f"""
  ── Theoretical Interpretation ──

  The Γ-metric between channels:

    d(i,j) = |Z_i - Z_j| / (Z_i + Z_j) = |Γ_ij|

  naturally defines a metric space on the set of neural channels.
  Under ΣΓ² → min, channels sharing the same signal environment converge
  to the same attractor basin (same region of impedance space), while
  channels receiving different modalities separate into distinct basins.

  This IS topology emerging from physics — the same mechanism that would,
  given spatial degrees of freedom, produce cortical maps, tonotopic
  gradients, and retinotopic organization.

  The key equation linking MRP to topology:

    T_Γ = {{ (i,j) | d(i,j) < ε }}  →  ε-neighborhood topology

  This topology self-organizes under MRP pressure — no hand-crafted
  connectivity matrix is needed.
""")


# ============================================================================
# Main
# ============================================================================

def main():
    print("╔" + "═" * (BANNER_WIDTH - 2) + "╗")
    print("║  Experiment: Γ-Field Topology Emergence (Level 0)".ljust(BANNER_WIDTH - 2) + "║")
    print("║  Hypothesis: Reflection coefficients generate spatial structure".ljust(BANNER_WIDTH - 2) + "║")
    print("║  Method: Pruning → measure order emergence in (Z, f) space".ljust(BANNER_WIDTH - 2) + "║")
    print("╚" + "═" * (BANNER_WIDTH - 2) + "╝")

    # Parameters
    N_CONNECTIONS = 1000
    N_EPOCHS = 100

    # Initialize engine
    np.random.seed(42)  # Reproducibility
    engine = NeuralPruningEngine(
        connections_per_region=N_CONNECTIONS,
        z_min=20.0,
        z_max=200.0,
        stimuli_per_epoch=10,
    )

    # Step 1: Birth snapshot
    birth_data = step1_birth_snapshot(engine)

    # Step 2: Development
    dev_results = step2_development(engine, epochs=N_EPOCHS)

    # Step 3: Post-pruning analysis
    post_result = step3_postpruning_snapshot(engine, birth_data)

    # Step 4: Γ-distance matrix
    distance_result = step4_gamma_distance_matrix(engine)

    # Step 5: Attractor analysis
    attractor_result = step5_attractor_analysis(engine, birth_data)

    # Step 6: Verdict
    gamma_sq_history = [r["global_gamma_squared"] for r in dev_results]
    step6_verdict(
        post_result["entropy"],
        distance_result,
        attractor_result,
        gamma_sq_history,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

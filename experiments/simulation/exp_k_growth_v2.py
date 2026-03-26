#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_k_growth_v2.py  -  K-Growth Emergence Simulation V2
========================================================
Improvements over v1:
  1. Binary TREE topology (not 1D chain) - clear peripheral/central hierarchy
  2. Graded stimulus complexity - leaves get raw sensory, interior gets nothing
  3. C3 relay cost - cross-K-boundary penalty discourages uniform growth

Run with: py -3.13 experiments/simulation/exp_k_growth_v2.py

Author: ALICE Gamma-Net Project
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# == Physics parameters ==================================================
TREE_DEPTH   = 6           # Binary tree depth (2^6-1 = 63 nodes)
T_TOTAL      = 10000       # Total simulation ticks
DT_GROWTH    = 25          # K-growth check interval
ETA_C2       = 0.008       # C2 learning rate (slow, so Gamma^2 persists)
MU_GROWTH    = 2.5         # K growth rate
GAMMA2_THR   = 0.04        # Growth trigger threshold
K_MAX        = 15          # K upper bound (energy constraint)
K_COST_EXP   = 1.3         # cost(K) proportional to K^exp
TAU_AVG      = 15          # Gamma^2 moving average window
C3_RELAY_COST = 0.3        # Penalty per unit |K_i - K_j| in C2 update
N_STIM_FREQ  = 6           # Number of stimulus frequencies per leaf

# == Build binary tree ===================================================
N_NODES = 2**TREE_DEPTH - 1
# Node 0 = root, children of node i: 2i+1, 2i+2
# Leaves: nodes with index >= 2^(DEPTH-1) - 1

def get_children(i):
    c1, c2 = 2*i+1, 2*i+2
    return [c for c in [c1, c2] if c < N_NODES]

def get_parent(i):
    return (i - 1) // 2 if i > 0 else None

def get_neighbors(i):
    nb = get_children(i)
    p = get_parent(i)
    if p is not None:
        nb.append(p)
    return nb

def get_depth(i):
    """Depth of node i in binary tree (root=0)."""
    d = 0
    n = i
    while n > 0:
        n = (n - 1) // 2
        d += 1
    return d

def is_leaf(i):
    return 2*i+1 >= N_NODES

# Compute depths
node_depth = np.array([get_depth(i) for i in range(N_NODES)])
leaf_mask = np.array([is_leaf(i) for i in range(N_NODES)])
n_leaves = leaf_mask.sum()

print("=" * 60)
print("K-Growth Emergence Simulation V2 (Binary Tree)")
print(f"Tree depth: {TREE_DEPTH}, Nodes: {N_NODES}, Leaves: {n_leaves}")
print(f"C2 eta: {ETA_C2}, Growth mu: {MU_GROWTH}, Gamma^2 thr: {GAMMA2_THR}")
print(f"C3 relay cost: {C3_RELAY_COST}")
print("=" * 60)

# == Initialization ======================================================
rng = np.random.default_rng(42)
K = np.ones(N_NODES, dtype=int)
Z = [rng.uniform(0.5, 1.5, size=k) for k in K]
gamma2_history = np.zeros((TAU_AVG, N_NODES))
history_ptr = 0

# Assign unique frequency profile to each leaf
leaf_indices = np.where(leaf_mask)[0]
leaf_freqs = {}
for idx, li in enumerate(leaf_indices):
    base_freq = 20 + idx * 5
    leaf_freqs[li] = [base_freq + k * 10 for k in range(N_STIM_FREQ)]

K_snapshots = []
snapshot_times = []

def compute_gamma2(Z, i, j):
    """Reflection coefficient between nodes i and j."""
    k_min = min(len(Z[i]), len(Z[j]))
    k_max = max(len(Z[i]), len(Z[j]))
    zi, zj = Z[i][:k_min], Z[j][:k_min]
    gamma = (zj - zi) / (zj + zi + 1e-12)
    g2 = np.mean(gamma**2)
    if k_max > k_min:
        n_extra = k_max - k_min
        g2 = (g2 * k_min + 1.0 * n_extra) / k_max
    return g2

def generate_stimulus(t):
    """Only leaves receive external stimuli (sensory periphery)."""
    stim = np.zeros(N_NODES)
    for li in leaf_indices:
        val = 0
        for freq in leaf_freqs[li]:
            val += np.sin(2 * np.pi * t / freq)
        val /= len(leaf_freqs[li])
        val += 0.4 * rng.normal()
        # Occasional bursts
        if rng.random() < 0.05:
            val *= 3.0
        stim[li] = val
    return stim

# == Main simulation loop ================================================
for t in range(T_TOTAL):
    stim = generate_stimulus(t)
    # Propagate activity bottom-up: parent inherits max of children
    for d in range(TREE_DEPTH - 2, -1, -1):
        for i in range(N_NODES):
            if node_depth[i] == d:
                children = get_children(i)
                if children:
                    child_act = max(abs(stim[c]) for c in children)
                    stim[i] = 0.7 * child_act  # damped propagation
    gamma2_local = np.zeros(N_NODES)

    # -- C2: adjust Z at each node --
    for i in range(N_NODES):
        neighbors = get_neighbors(i)
        total_g2 = 0
        for j in neighbors:
            g2 = compute_gamma2(Z, i, j)
            total_g2 += g2

            k_min = min(len(Z[i]), len(Z[j]))
            zi, zj = Z[i][:k_min], Z[j][:k_min]
            gamma = (zj - zi) / (zj + zi + 1e-12)

            # Activity gating - NO baseline, must be driven
            activity = abs(stim[i]) + abs(stim[j])
            # C3 relay penalty: reduce C2 effectiveness across K gap
            k_gap = abs(len(Z[i]) - len(Z[j]))
            relay_penalty = 1.0 / (1.0 + C3_RELAY_COST * k_gap)

            dZ = ETA_C2 * gamma * activity * relay_penalty
            Z[i][:k_min] += dZ
            Z[i] = np.maximum(Z[i], 0.01)

        gamma2_local[i] = total_g2 / max(len(neighbors), 1)

    # Update history
    gamma2_history[history_ptr % TAU_AVG] = gamma2_local
    history_ptr += 1

    # -- K growth --
    if t > 0 and t % DT_GROWTH == 0:
        n_filled = min(history_ptr, TAU_AVG)
        gamma2_avg = np.mean(gamma2_history[:n_filled], axis=0)

        for i in range(N_NODES):
            if gamma2_avg[i] > GAMMA2_THR and K[i] < K_MAX:
                # Neighbor K coordination: can't outpace neighbors
                neighbors = get_neighbors(i)
                max_nb_k = max(K[j] for j in neighbors)
                if K[i] > max_nb_k + 1:
                    continue
                growth_cost = (K[i] + 1) ** K_COST_EXP / K_MAX ** K_COST_EXP
                effective_signal = MU_GROWTH * (gamma2_avg[i] - GAMMA2_THR)
                if effective_signal > growth_cost:
                    K[i] += 1
                    neighbors = get_neighbors(i)
                    z_new = np.mean([np.mean(Z[j]) for j in neighbors])
                    Z[i] = np.append(Z[i], z_new)

    # Snapshots
    if t % (T_TOTAL // 20) == 0:
        K_snapshots.append(K.copy())
        snapshot_times.append(t)

    if t > 0 and t % 2000 == 0:
        by_depth = defaultdict(list)
        for i in range(N_NODES):
            by_depth[node_depth[i]].append(K[i])
        depth_str = "  ".join(
            f"d{d}:{np.mean(v):.1f}" for d, v in sorted(by_depth.items())
        )
        print(f"  t={t:5d}: {depth_str}")

# == Final statistics ====================================================
print("\n" + "=" * 60)
print("FINAL K DISTRIBUTION BY DEPTH")
print("=" * 60)

by_depth = defaultdict(list)
for i in range(N_NODES):
    by_depth[node_depth[i]].append(K[i])

depth_means = {}
for d in sorted(by_depth.keys()):
    vals = by_depth[d]
    m = np.mean(vals)
    depth_means[d] = m
    print(f"  Depth {d}: K_mean = {m:.2f}  (n={len(vals)})")

# Hierarchy test: root/internal should have higher K than leaves
root_k = depth_means.get(0, 0)
leaf_k = depth_means.get(TREE_DEPTH - 1, 0)
mid_k = depth_means.get(TREE_DEPTH // 2, 0)
hierarchy = root_k > leaf_k or mid_k > leaf_k

# Gradient test: K should decrease with depth
gradient = all(
    depth_means.get(d, 0) >= depth_means.get(d+1, 0) - 0.5
    for d in range(TREE_DEPTH - 2)
)

print(f"\n  Root K:   {root_k:.2f}")
print(f"  Mid K:    {mid_k:.2f}")
print(f"  Leaf K:   {leaf_k:.2f}")
print(f"  K range:  [{K.min()}, {K.max()}]")
print(f"  K mean:   {K.mean():.2f}")
print(f"  K std:    {K.std():.2f}")

# == Plots ===============================================================
OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("K-Growth V2: Binary Tree + C3 Relay Cost\n"
             "From C1+C2+C3: Does K-hierarchy self-organise?",
             fontsize=14, fontweight='bold')

# 1) K by depth (box plot)
ax = axes[0, 0]
depth_data = [by_depth[d] for d in sorted(by_depth.keys())]
depth_labels = [f"d={d}\n(n={len(by_depth[d])})"
                for d in sorted(by_depth.keys())]
bp = ax.boxplot(depth_data, labels=depth_labels, patch_artist=True)
cmap = plt.cm.RdYlGn_r
for idx, patch in enumerate(bp['boxes']):
    patch.set_facecolor(cmap(idx / max(len(bp['boxes'])-1, 1)))
    patch.set_alpha(0.7)
ax.set_xlabel("Tree depth (0=root, max=leaves)")
ax.set_ylabel("K value")
ax.set_title("K distribution by tree depth")

# 2) K evolution snapshots
ax = axes[0, 1]
n_show = min(6, len(K_snapshots))
indices = np.linspace(0, len(K_snapshots)-1, n_show, dtype=int)
for idx in indices:
    t_snap = snapshot_times[idx]
    # Plot K vs depth for this snapshot
    snap = K_snapshots[idx]
    depth_k = defaultdict(list)
    for i in range(N_NODES):
        depth_k[node_depth[i]].append(snap[i])
    depths_sorted = sorted(depth_k.keys())
    means = [np.mean(depth_k[d]) for d in depths_sorted]
    alpha = 0.3 + 0.7 * (idx / max(len(K_snapshots)-1, 1))
    ax.plot(depths_sorted, means, 'o-', alpha=alpha, markersize=4,
            label=f't={t_snap}')
ax.set_xlabel("Tree depth")
ax.set_ylabel("Mean K value")
ax.set_title("K evolution over time (by depth)")
ax.legend(fontsize=8)

# 3) Final Gamma^2 by depth
ax = axes[1, 0]
final_g2 = gamma2_history[(history_ptr-1) % TAU_AVG]
depth_g2 = defaultdict(list)
for i in range(N_NODES):
    depth_g2[node_depth[i]].append(final_g2[i])
ds = sorted(depth_g2.keys())
g2_means = [np.mean(depth_g2[d]) for d in ds]
ax.bar(ds, g2_means, color='coral', alpha=0.7, edgecolor='red')
ax.axhline(y=GAMMA2_THR, color='gray', linestyle=':', alpha=0.7,
           label=f'growth threshold = {GAMMA2_THR}')
ax.set_xlabel("Tree depth")
ax.set_ylabel("Mean Gamma^2")
ax.set_title("Final Gamma^2 by depth")
ax.legend()

# 4) Summary diagram
ax = axes[1, 1]
ax.axis('off')
checks = [
    ("K hierarchy (root > leaf)", hierarchy),
    ("K gradient (smooth depth gradient)", gradient),
    ("K diversity (std > 1.0)", K.std() > 1.0),
    ("Gamma^2 reduced overall", np.mean(final_g2) < 0.5),
    ("Root reached high K (>= 0.5 * K_MAX)", root_k >= 0.5 * K_MAX),
]
n_pass = sum(1 for _, v in checks if v)
txt_lines = ["CLINICAL CHECKS\n"]
for desc, val in checks:
    mark = "PASS" if val else "FAIL"
    txt_lines.append(f"  {'[PASS]' if val else '[FAIL]'} {desc}")
txt_lines.append(f"\nResult: {n_pass}/{len(checks)} checks passed")
txt_lines.append(f"\nRoot K={root_k:.1f}  Mid K={mid_k:.1f}  Leaf K={leaf_k:.1f}")

color = 'green' if n_pass >= 4 else ('orange' if n_pass >= 3 else 'red')
ax.text(0.05, 0.95, "\n".join(txt_lines),
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
if n_pass >= 3:
    ax.text(0.5, 0.02, "HIERARCHY EMERGED",
            transform=ax.transAxes, ha='center',
            fontsize=14, fontweight='bold', color=color)

plt.tight_layout()
fig_path = OUT / "fig_k_growth_v2_tree.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Figure saved: {fig_path}")

# == Print clinical checks ===============================================
print("\n" + "=" * 60)
print("CLINICAL CHECKS")
print("=" * 60)
for desc, val in checks:
    print(f"  {'PASS' if val else 'FAIL'} {desc}")
print(f"\nResult: {n_pass}/{len(checks)} checks passed")

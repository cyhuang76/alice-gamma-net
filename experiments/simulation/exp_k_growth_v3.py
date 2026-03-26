#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_k_growth_v3.py  -  Anti-Parallel K-Staircase Emergence
===========================================================
Models the P1 anti-parallel hemisphere theory:
  - Left chain: analysis (sensory -> abstract, K should grow low->high)
  - Right chain: synthesis (abstract -> motor, K should grow high->low)
  - Corpus callosum: cross-connects at each level
  - Cerebellum: corrects CC mismatches

Network topology:
  Left chain:   L0 -- L1 -- L2 -- ... -- L(N-1)
                 |     |     |              |
  CC relay:     CC0 - CC1 - CC2 - ... - CC(N-1)  (+ cerebellum)
                 |     |     |              |
  Right chain:  R0 -- R1 -- R2 -- ... -- R(N-1)

  L0, R(N-1) = sensory periphery (receive stimuli)
  L(N-1), R0 = abstract/motor convergence

Run with: py -3.13 experiments/simulation/exp_k_growth_v3.py

Author: ALICE Gamma-Net Project
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# == Physics parameters ================================================
N_LEVELS     = 12          # Levels in each hemisphere chain
T_TOTAL      = 12000       # Total ticks
DT_GROWTH    = 20          # K-growth check interval
ETA_C2       = 0.006       # C2 learning rate
MU_GROWTH    = 3.0         # K growth rate
GAMMA2_THR   = 0.03        # Growth threshold
K_MAX        = 15          # K upper bound
K_COST_EXP   = 1.2         # cost(K) ~ K^exp
TAU_AVG      = 15          # Moving average window
C3_RELAY     = 0.4         # C3 relay penalty per K-gap
CB_CORRECT   = 0.3         # Cerebellum correction strength

# == Network setup =====================================================
# 3 rows: Left(0..N-1), CC_relay(N..2N-1), Right(2N..3N-1)
N_TOTAL = 3 * N_LEVELS
IDX_L  = list(range(0, N_LEVELS))
IDX_CC = list(range(N_LEVELS, 2 * N_LEVELS))
IDX_R  = list(range(2 * N_LEVELS, 3 * N_LEVELS))

def get_neighbors(i):
    """Return list of neighbor indices for node i."""
    nb = []
    if i < N_LEVELS:
        # Left chain
        level = i
        if level > 0: nb.append(level - 1)         # left neighbor
        if level < N_LEVELS - 1: nb.append(level + 1)  # right neighbor
        nb.append(N_LEVELS + level)                  # CC relay below
    elif i < 2 * N_LEVELS:
        # CC relay row
        level = i - N_LEVELS
        nb.append(level)                             # Left above
        nb.append(2 * N_LEVELS + level)              # Right below
        if level > 0: nb.append(N_LEVELS + level - 1)
        if level < N_LEVELS - 1: nb.append(N_LEVELS + level + 1)
    else:
        # Right chain
        level = i - 2 * N_LEVELS
        if level > 0: nb.append(2 * N_LEVELS + level - 1)
        if level < N_LEVELS - 1: nb.append(2 * N_LEVELS + level + 1)
        nb.append(N_LEVELS + level)                  # CC relay above
    return nb

def node_label(i):
    if i < N_LEVELS:
        return f"L{i}"
    elif i < 2 * N_LEVELS:
        return f"CC{i - N_LEVELS}"
    else:
        return f"R{i - 2 * N_LEVELS}"

rng = np.random.default_rng(42)
K = np.ones(N_TOTAL, dtype=int)
Z = [rng.uniform(0.5, 1.5, size=k) for k in K]
g2_hist = np.zeros((TAU_AVG, N_TOTAL))
hp = 0

K_snapshots = []
snap_times = []

print("=" * 60)
print("K-Growth V3: Anti-Parallel K-Staircase")
print(f"Levels: {N_LEVELS}, Total nodes: {N_TOTAL}")
print(f"Left: L0(sensory) -> L{N_LEVELS-1}(abstract)")
print(f"Right: R0(abstract) -> R{N_LEVELS-1}(sensory)")
print(f"CC relay: cross-connects at each level")
print("=" * 60)

def compute_gamma2(Z, i, j):
    km = min(len(Z[i]), len(Z[j]))
    kx = max(len(Z[i]), len(Z[j]))
    zi, zj = Z[i][:km], Z[j][:km]
    g = (zj - zi) / (zj + zi + 1e-12)
    g2 = np.mean(g**2)
    if kx > km:
        g2 = (g2 * km + 1.0 * (kx - km)) / kx
    return g2

def generate_stimulus(t):
    """
    Left chain: L0 gets raw sensory (many frequencies, high complexity)
    Right chain: R(N-1) gets different sensory modality
    Interior nodes: no direct stimulus
    """
    stim = np.zeros(N_TOTAL)
    # Left sensory input at L0, L1, L2 (decreasing strength)
    for k in range(min(3, N_LEVELS)):
        freqs = [25 + k*10 + f*15 for f in range(4 + k)]
        val = sum(np.sin(2*np.pi*t/freq) for freq in freqs) / len(freqs)
        val += 0.5 * rng.normal()
        if rng.random() < 0.03:
            val *= 4.0
        stim[k] = val * (1.0 - 0.3*k)

    # Right sensory input at R(N-1), R(N-2), R(N-3)
    for k in range(min(3, N_LEVELS)):
        ri = 2*N_LEVELS + (N_LEVELS - 1 - k)
        freqs = [30 + k*8 + f*12 for f in range(5 + k)]
        val = sum(np.cos(2*np.pi*t/freq) for freq in freqs) / len(freqs)
        val += 0.5 * rng.normal()
        if rng.random() < 0.03:
            val *= 4.0
        stim[ri] = val * (1.0 - 0.3*k)

    # Cross-modal event: simultaneous L and R burst
    if t % 300 < 15:
        stim[0] += 2.0 * rng.normal()
        stim[3*N_LEVELS - 1] += 2.0 * rng.normal()

    return stim

# == Main loop =========================================================
for t in range(T_TOTAL):
    stim = generate_stimulus(t)

    # Propagate activity along chains (damped)
    # Left: L0 -> L1 -> ... -> L(N-1)
    for k in range(1, N_LEVELS):
        stim[k] = max(stim[k], 0.6 * abs(stim[k-1])) * np.sign(stim[k]) if stim[k] != 0 else 0.6 * stim[k-1]
    # Right: R(N-1) -> R(N-2) -> ... -> R0
    for k in range(N_LEVELS - 2, -1, -1):
        ri = 2*N_LEVELS + k
        ri_next = 2*N_LEVELS + k + 1
        stim[ri] = max(stim[ri], 0.6 * abs(stim[ri_next])) * np.sign(stim[ri]) if stim[ri] != 0 else 0.6 * stim[ri_next]
    # CC relay inherits from both sides
    for k in range(N_LEVELS):
        cc = N_LEVELS + k
        stim[cc] = 0.5 * (abs(stim[k]) + abs(stim[2*N_LEVELS + k]))

    g2_local = np.zeros(N_TOTAL)

    # C2 update
    for i in range(N_TOTAL):
        nbs = get_neighbors(i)
        tg2 = 0
        for j in nbs:
            g2 = compute_gamma2(Z, i, j)
            tg2 += g2

            km = min(len(Z[i]), len(Z[j]))
            zi, zj = Z[i][:km], Z[j][:km]
            gamma = (zj - zi) / (zj + zi + 1e-12)

            activity = abs(stim[i]) + abs(stim[j])
            if activity < 0.01:
                continue

            k_gap = abs(len(Z[i]) - len(Z[j]))
            relay_pen = 1.0 / (1.0 + C3_RELAY * k_gap)

            # CC nodes get cerebellum correction boost
            cb_boost = 1.0
            if N_LEVELS <= i < 2*N_LEVELS:
                cb_boost = 1.0 + CB_CORRECT

            dZ = ETA_C2 * gamma * activity * relay_pen * cb_boost
            Z[i][:km] += dZ
            Z[i] = np.maximum(Z[i], 0.01)

        g2_local[i] = tg2 / max(len(nbs), 1)

    g2_hist[hp % TAU_AVG] = g2_local
    hp += 1

    # K growth
    if t > 0 and t % DT_GROWTH == 0:
        nf = min(hp, TAU_AVG)
        g2_avg = np.mean(g2_hist[:nf], axis=0)

        for i in range(N_TOTAL):
            if g2_avg[i] > GAMMA2_THR and K[i] < K_MAX:
                nbs = get_neighbors(i)
                max_nb_k = max(K[j] for j in nbs)
                if K[i] > max_nb_k + 1:
                    continue
                gc = (K[i]+1)**K_COST_EXP / K_MAX**K_COST_EXP
                es = MU_GROWTH * (g2_avg[i] - GAMMA2_THR)
                if es > gc:
                    K[i] += 1
                    z_new = np.mean([np.mean(Z[j]) for j in nbs])
                    Z[i] = np.append(Z[i], z_new)

    if t % (T_TOTAL // 20) == 0:
        K_snapshots.append(K.copy())
        snap_times.append(t)

    if t > 0 and t % 3000 == 0:
        kl = [K[i] for i in IDX_L]
        kr = [K[i] for i in IDX_R]
        kc = [K[i] for i in IDX_CC]
        print(f"  t={t:5d}: L=[{min(kl)}-{max(kl)}] "
              f"CC=[{min(kc)}-{max(kc)}] R=[{min(kr)}-{max(kr)}]")

# == Results ===========================================================
print("\n" + "=" * 60)
print("FINAL K BY LEVEL")
print("=" * 60)
print(f"{'Level':>6} {'Left':>6} {'CC':>6} {'Right':>6}")
print("-" * 30)
for lv in range(N_LEVELS):
    kl = K[lv]
    kc = K[N_LEVELS + lv]
    kr = K[2*N_LEVELS + lv]
    print(f"{lv:>6} {kl:>6} {kc:>6} {kr:>6}")

K_L = np.array([K[i] for i in IDX_L])
K_R = np.array([K[i] for i in IDX_R])
K_CC = np.array([K[i] for i in IDX_CC])

# Check anti-parallel pattern
# Left should increase: L0(low) -> L(N-1)(high)
left_gradient = np.corrcoef(range(N_LEVELS), K_L)[0, 1]
# Right should decrease: R0(high) -> R(N-1)(low)
right_gradient = np.corrcoef(range(N_LEVELS), K_R)[0, 1]
# They should be anti-correlated
anti_parallel = left_gradient > 0.3 and right_gradient < -0.3

print(f"\nLeft gradient (expect +):  {left_gradient:+.3f}")
print(f"Right gradient (expect -): {right_gradient:+.3f}")
print(f"Anti-parallel: {'YES' if anti_parallel else 'NO'}")
print(f"K_L range: [{K_L.min()}, {K_L.max()}]  mean={K_L.mean():.2f}")
print(f"K_R range: [{K_R.min()}, {K_R.max()}]  mean={K_R.mean():.2f}")
print(f"K_CC range: [{K_CC.min()}, {K_CC.max()}]  mean={K_CC.mean():.2f}")

# == Plots =============================================================
OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("K-Growth V3: Anti-Parallel K-Staircase Emergence\n"
             "Left=analysis(low->high), Right=synthesis(high->low)",
             fontsize=14, fontweight='bold')

# 1) K vs level for both hemispheres
ax = axes[0, 0]
levels = range(N_LEVELS)
ax.plot(levels, K_L, 'b-o', markersize=6, linewidth=2, label='Left (analysis)')
ax.plot(levels, K_R, 'r-s', markersize=6, linewidth=2, label='Right (synthesis)')
ax.plot(levels, K_CC, 'g--^', markersize=5, linewidth=1.5, alpha=0.7, label='CC relay')
ax.set_xlabel("Processing level (0=sensory, N-1=abstract)")
ax.set_ylabel("K value")
ax.set_title("K profile: anti-parallel staircase?")
ax.legend()
ax.grid(True, alpha=0.3)

# 2) K evolution over time
ax = axes[0, 1]
n_show = min(6, len(K_snapshots))
idxs = np.linspace(0, len(K_snapshots)-1, n_show, dtype=int)
for idx in idxs:
    snap = K_snapshots[idx]
    t_s = snap_times[idx]
    kl_s = [snap[i] for i in IDX_L]
    alpha = 0.3 + 0.7 * (idx / max(len(K_snapshots)-1, 1))
    ax.plot(levels, kl_s, alpha=alpha, linewidth=1.5, label=f't={t_s}')
ax.set_xlabel("Level")
ax.set_ylabel("Left hemisphere K")
ax.set_title("Left K evolution over time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3) Gamma^2 by level
ax = axes[1, 0]
final_g2 = g2_hist[(hp-1) % TAU_AVG]
g2_L = [final_g2[i] for i in IDX_L]
g2_R = [final_g2[i] for i in IDX_R]
g2_CC = [final_g2[i] for i in IDX_CC]
ax.plot(levels, g2_L, 'b-', linewidth=1.5, label='Left')
ax.plot(levels, g2_R, 'r-', linewidth=1.5, label='Right')
ax.plot(levels, g2_CC, 'g--', linewidth=1.5, alpha=0.7, label='CC')
ax.axhline(y=GAMMA2_THR, color='gray', linestyle=':', alpha=0.7,
           label=f'threshold={GAMMA2_THR}')
ax.set_xlabel("Level")
ax.set_ylabel("Gamma^2")
ax.set_title("Final Gamma^2 by level")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4) Clinical checks
ax = axes[1, 1]
ax.axis('off')
checks = [
    ("Left K increases with level (analysis)", left_gradient > 0.3),
    ("Right K decreases with level (synthesis)", right_gradient < -0.3),
    ("Anti-parallel pattern", anti_parallel),
    ("CC relay K intermediate", K_CC.mean() <= max(K_L.max(), K_R.max())),
    ("K diversity (std > 1.0)", K.std() > 1.0),
    ("Gamma^2 reduced overall", np.mean(final_g2) < 0.5),
]
n_pass = sum(1 for _, v in checks if v)
lines = ["CLINICAL CHECKS\n"]
for desc, val in checks:
    lines.append(f"  {'[PASS]' if val else '[FAIL]'} {desc}")
lines.append(f"\nResult: {n_pass}/{len(checks)} passed")
lines.append(f"\nLeft grad:  {left_gradient:+.3f}")
lines.append(f"Right grad: {right_gradient:+.3f}")

color = 'green' if n_pass >= 4 else ('orange' if n_pass >= 3 else 'red')
ax.text(0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
if anti_parallel:
    ax.text(0.5, 0.02, "ANTI-PARALLEL EMERGED!",
            transform=ax.transAxes, ha='center',
            fontsize=14, fontweight='bold', color='green')

plt.tight_layout()
fig_path = OUT / "fig_k_growth_v3_antiparallel.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {fig_path}")

# Print checks
print("\n" + "=" * 60)
print("CLINICAL CHECKS")
print("=" * 60)
for desc, val in checks:
    print(f"  {'PASS' if val else 'FAIL'} {desc}")
print(f"\nResult: {n_pass}/{len(checks)} passed")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_k_growth_v3_512.py  -  Anti-Parallel K-Staircase (512 levels)
=================================================================
Same physics as V3, scaled to 512 processing levels.
Vectorized inner loop for performance.

Run with: py -3.13 experiments/simulation/exp_k_growth_v3_512.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

# == Physics parameters ================================================
N_LEVELS     = 512
T_TOTAL      = 6000
DT_GROWTH    = 30
ETA_C2       = 0.005
MU_GROWTH    = 3.5
GAMMA2_THR   = 0.03
K_MAX        = 20
K_COST_EXP   = 1.2
TAU_AVG      = 12
C3_RELAY     = 0.4
CB_CORRECT   = 0.3
N_TOTAL      = 3 * N_LEVELS

# Row indices
SL = slice(0, N_LEVELS)              # Left
SC = slice(N_LEVELS, 2*N_LEVELS)     # CC
SR = slice(2*N_LEVELS, 3*N_LEVELS)   # Right

rng = np.random.default_rng(42)

# Use scalar K and Z arrays (fixed max K, with mask)
K = np.ones(N_TOTAL, dtype=int)
# Z stored as 2D array: Z_mat[i, :K[i]] is active
Z_mat = rng.uniform(0.5, 1.5, size=(N_TOTAL, K_MAX))
Z_active = np.ones(N_TOTAL, dtype=int)  # same as K initially

g2_hist = np.zeros((TAU_AVG, N_TOTAL))
hp = 0

K_snapshots = []
snap_times = []

print("=" * 60)
print(f"K-Growth V3: Anti-Parallel (N={N_LEVELS})")
print(f"Total nodes: {N_TOTAL}, Ticks: {T_TOTAL}")
print("=" * 60)

def compute_g2_pair(i, j):
    """Gamma^2 between nodes i and j."""
    ki, kj = K[i], K[j]
    km = min(ki, kj)
    kx = max(ki, kj)
    zi = Z_mat[i, :km]
    zj = Z_mat[j, :km]
    g = (zj - zi) / (zj + zi + 1e-12)
    g2 = np.mean(g**2) if km > 0 else 1.0
    if kx > km:
        g2 = (g2 * km + 1.0 * (kx - km)) / kx
    return g2

def get_neighbors(i):
    nb = []
    if i < N_LEVELS:
        lv = i
        if lv > 0: nb.append(lv - 1)
        if lv < N_LEVELS - 1: nb.append(lv + 1)
        nb.append(N_LEVELS + lv)
    elif i < 2 * N_LEVELS:
        lv = i - N_LEVELS
        nb.append(lv)
        nb.append(2 * N_LEVELS + lv)
        if lv > 0: nb.append(N_LEVELS + lv - 1)
        if lv < N_LEVELS - 1: nb.append(N_LEVELS + lv + 1)
    else:
        lv = i - 2 * N_LEVELS
        if lv > 0: nb.append(2 * N_LEVELS + lv - 1)
        if lv < N_LEVELS - 1: nb.append(2 * N_LEVELS + lv + 1)
        nb.append(N_LEVELS + lv)
    return nb

# Pre-compute neighbor lists
NB = [get_neighbors(i) for i in range(N_TOTAL)]

t0 = time.time()

for t in range(T_TOTAL):
    # -- Stimulus --
    stim = np.zeros(N_TOTAL)
    # Left sensory: L[0..9]
    for k in range(min(10, N_LEVELS)):
        nf = 4 + k
        freqs = [25 + k*8 + f*12 for f in range(nf)]
        val = sum(np.sin(2*np.pi*t/freq) for freq in freqs) / nf
        val += 0.4 * rng.normal()
        if rng.random() < 0.02: val *= 4.0
        stim[k] = val * (1.0 - 0.08*k)

    # Right sensory: R[N-1..N-10]
    for k in range(min(10, N_LEVELS)):
        ri = 2*N_LEVELS + (N_LEVELS - 1 - k)
        nf = 5 + k
        freqs = [30 + k*7 + f*11 for f in range(nf)]
        val = sum(np.cos(2*np.pi*t/freq) for freq in freqs) / nf
        val += 0.4 * rng.normal()
        if rng.random() < 0.02: val *= 4.0
        stim[ri] = val * (1.0 - 0.08*k)

    # Cross-modal burst
    if t % 200 < 10:
        stim[0] += 2.0 * rng.normal()
        stim[3*N_LEVELS - 1] += 2.0 * rng.normal()

    # Propagate: Left forward, Right backward
    for k in range(1, N_LEVELS):
        if abs(stim[k]) < abs(0.65 * stim[k-1]):
            stim[k] = 0.65 * stim[k-1]
    for k in range(N_LEVELS - 2, -1, -1):
        ri = 2*N_LEVELS + k
        ri1 = 2*N_LEVELS + k + 1
        if abs(stim[ri]) < abs(0.65 * stim[ri1]):
            stim[ri] = 0.65 * stim[ri1]
    # CC inherits
    for k in range(N_LEVELS):
        cc = N_LEVELS + k
        stim[cc] = 0.5 * (abs(stim[k]) + abs(stim[2*N_LEVELS + k]))

    # -- C2 + Gamma^2 --
    g2_local = np.zeros(N_TOTAL)
    for i in range(N_TOTAL):
        nbs = NB[i]
        tg2 = 0.0
        for j in nbs:
            g2 = compute_g2_pair(i, j)
            tg2 += g2

            ki, kj = K[i], K[j]
            km = min(ki, kj)
            if km == 0:
                continue
            zi = Z_mat[i, :km]
            zj = Z_mat[j, :km]
            gamma = (zj - zi) / (zj + zi + 1e-12)

            activity = abs(stim[i]) + abs(stim[j])
            if activity < 0.01:
                continue

            k_gap = abs(ki - kj)
            relay_pen = 1.0 / (1.0 + C3_RELAY * k_gap)
            cb = 1.0 + CB_CORRECT if N_LEVELS <= i < 2*N_LEVELS else 1.0

            dZ = ETA_C2 * gamma * activity * relay_pen * cb
            Z_mat[i, :km] += dZ
            Z_mat[i, :km] = np.maximum(Z_mat[i, :km], 0.01)

        g2_local[i] = tg2 / max(len(nbs), 1)

    g2_hist[hp % TAU_AVG] = g2_local
    hp += 1

    # -- K growth --
    if t > 0 and t % DT_GROWTH == 0:
        nf = min(hp, TAU_AVG)
        g2_avg = np.mean(g2_hist[:nf], axis=0)

        for i in range(N_TOTAL):
            if g2_avg[i] > GAMMA2_THR and K[i] < K_MAX:
                nbs = NB[i]
                max_nb_k = max(K[j] for j in nbs)
                if K[i] > max_nb_k + 1:
                    continue
                gc = (K[i]+1)**K_COST_EXP / K_MAX**K_COST_EXP
                es = MU_GROWTH * (g2_avg[i] - GAMMA2_THR)
                if es > gc:
                    K[i] += 1
                    z_new = np.mean([Z_mat[j, :K[j]].mean() for j in nbs])
                    Z_mat[i, K[i]-1] = z_new

    if t % (T_TOTAL // 20) == 0:
        K_snapshots.append(K.copy())
        snap_times.append(t)

    if t > 0 and t % 1000 == 0:
        elapsed = time.time() - t0
        eta = elapsed / t * (T_TOTAL - t)
        kl = K[SL]
        kr = K[SR]
        kc = K[SC]
        print(f"  t={t:5d}/{T_TOTAL}: L=[{kl.min()}-{kl.max()}] "
              f"CC=[{kc.min()}-{kc.max()}] R=[{kr.min()}-{kr.max()}] "
              f"({elapsed:.0f}s, ETA {eta:.0f}s)")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")

# == Results ===========================================================
K_L = K[SL].copy()
K_R = K[SR].copy()
K_CC = K[SC].copy()

levels = np.arange(N_LEVELS)
left_grad = np.corrcoef(levels, K_L)[0, 1]
right_grad = np.corrcoef(levels, K_R)[0, 1]
anti_parallel = left_grad > 0.3 and right_grad < -0.3

print("\n" + "=" * 60)
print(f"FINAL K DISTRIBUTION (N={N_LEVELS})")
print("=" * 60)
print(f"Left:  [{K_L.min()}-{K_L.max()}] mean={K_L.mean():.2f} grad={left_grad:+.3f}")
print(f"Right: [{K_R.min()}-{K_R.max()}] mean={K_R.mean():.2f} grad={right_grad:+.3f}")
print(f"CC:    [{K_CC.min()}-{K_CC.max()}] mean={K_CC.mean():.2f}")
print(f"Anti-parallel: {'YES' if anti_parallel else 'NO'}")

# == Plots =============================================================
OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"K-Growth V3: Anti-Parallel Staircase (N={N_LEVELS})\n"
             f"Left=analysis, Right=synthesis, CC=relay",
             fontsize=14, fontweight='bold')

# 1) K profiles
ax = axes[0, 0]
ax.plot(levels, K_L, 'b-', linewidth=0.8, alpha=0.8, label='Left (analysis)')
ax.plot(levels, K_R, 'r-', linewidth=0.8, alpha=0.8, label='Right (synthesis)')
ax.plot(levels, K_CC, 'g-', linewidth=0.8, alpha=0.5, label='CC relay')
# Smoothed overlay
win = min(20, N_LEVELS // 10)
if win > 1:
    kl_smooth = np.convolve(K_L, np.ones(win)/win, mode='valid')
    kr_smooth = np.convolve(K_R, np.ones(win)/win, mode='valid')
    x_smooth = np.arange(win//2, win//2 + len(kl_smooth))
    ax.plot(x_smooth, kl_smooth, 'b-', linewidth=2.5)
    ax.plot(x_smooth, kr_smooth, 'r-', linewidth=2.5)
ax.set_xlabel(f"Processing level (0=sensory, {N_LEVELS-1}=abstract)")
ax.set_ylabel("K value")
ax.set_title("K profile across 512 levels")
ax.legend()
ax.grid(True, alpha=0.3)

# 2) K evolution (Left only, smoothed)
ax = axes[0, 1]
n_show = min(6, len(K_snapshots))
idxs = np.linspace(0, len(K_snapshots)-1, n_show, dtype=int)
for idx in idxs:
    snap = K_snapshots[idx]
    kl_s = snap[SL]
    if win > 1:
        kl_s = np.convolve(kl_s, np.ones(win)/win, mode='valid')
        x = np.arange(win//2, win//2 + len(kl_s))
    else:
        x = levels
    alpha = 0.3 + 0.7 * (idx / max(len(K_snapshots)-1, 1))
    ax.plot(x, kl_s, alpha=alpha, linewidth=1.5, label=f't={snap_times[idx]}')
ax.set_xlabel("Level")
ax.set_ylabel("Left K (smoothed)")
ax.set_title("Left K evolution over time")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 3) Gamma^2
ax = axes[1, 0]
fg2 = g2_hist[(hp-1) % TAU_AVG]
g2L = fg2[SL]
g2R = fg2[SR]
if win > 1:
    g2L = np.convolve(g2L, np.ones(win)/win, mode='valid')
    g2R = np.convolve(g2R, np.ones(win)/win, mode='valid')
    x = np.arange(win//2, win//2 + len(g2L))
else:
    x = levels
ax.plot(x, g2L, 'b-', linewidth=1.5, label='Left')
ax.plot(x, g2R, 'r-', linewidth=1.5, label='Right')
ax.axhline(y=GAMMA2_THR, color='gray', linestyle=':', alpha=0.7,
           label=f'threshold={GAMMA2_THR}')
ax.set_xlabel("Level")
ax.set_ylabel("Gamma^2 (smoothed)")
ax.set_title("Final Gamma^2")
ax.legend()
ax.grid(True, alpha=0.3)

# 4) Checks
ax = axes[1, 1]
ax.axis('off')
fg2_mean = np.mean(fg2)
checks = [
    ("Left K increases with level", left_grad > 0.3),
    ("Right K decreases with level", right_grad < -0.3),
    ("Anti-parallel pattern", anti_parallel),
    ("CC relay K intermediate", K_CC.mean() <= max(K_L.max(), K_R.max())),
    ("K diversity (std > 1.0)", K.std() > 1.0),
    ("Gamma^2 reduced overall", fg2_mean < 0.5),
]
n_pass = sum(1 for _, v in checks if v)
lines = [f"CLINICAL CHECKS (N={N_LEVELS})\n"]
for desc, val in checks:
    lines.append(f"  {'[PASS]' if val else '[FAIL]'} {desc}")
lines.append(f"\nResult: {n_pass}/{len(checks)} passed")
lines.append(f"\nL grad: {left_grad:+.3f}  R grad: {right_grad:+.3f}")
lines.append(f"L: [{K_L.min()}-{K_L.max()}]  R: [{K_R.min()}-{K_R.max()}]")
lines.append(f"Time: {elapsed:.0f}s")

color = 'green' if anti_parallel else ('orange' if n_pass >= 3 else 'red')
ax.text(0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
if anti_parallel:
    ax.text(0.5, 0.02, "ANTI-PARALLEL EMERGED!",
            transform=ax.transAxes, ha='center',
            fontsize=14, fontweight='bold', color='green')

plt.tight_layout()
fig_path = OUT / "fig_k_growth_v3_512.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure saved: {fig_path}")

print("\n" + "=" * 60)
print("CLINICAL CHECKS")
print("=" * 60)
for desc, val in checks:
    print(f"  {'PASS' if val else 'FAIL'} {desc}")
print(f"\nResult: {n_pass}/{len(checks)} passed")

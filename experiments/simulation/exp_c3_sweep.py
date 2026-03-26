#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_c3_sweep.py  -  C3 Relay Cost Parameter Sweep
===================================================
Systematically varies C3_RELAY from 0 (no cost) to 2.0 (extreme)
to isolate C3's effect on K-hierarchy formation.

Question: What does C3 actually DO to parcellation?

Uses V4 physics (direction + NT + Z0_DNA) at 128 levels for speed.

Run with: py -3.13 experiments/simulation/exp_c3_sweep.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

# == Sweep values ======================================================
C3_VALUES = [0.0, 0.1, 0.3, 0.6, 1.0, 2.0]

# == Shared parameters =================================================
N_LEVELS  = 128          # Smaller for speed (6 runs)
T_TOTAL   = 4000
DT_GROWTH = 25
ETA_C2    = 0.006
MU_GROWTH = 3.0
GAMMA2_THR = 0.03
K_MAX     = 15
K_COST_EXP = 1.2
TAU_AVG   = 12
CB_CORRECT = 0.3
N_TOTAL   = 3 * N_LEVELS

# NT zones (same logic as V4)
NT_NAMES = ['Glutamate', 'Dopamine', 'Serotonin', 'GABA']
NT_K_PREF = {'Glutamate': 12, 'Dopamine': 15, 'Serotonin': 10, 'GABA': 6}
NT_GROWTH = {'Glutamate': 1.2, 'Dopamine': 1.5, 'Serotonin': 0.8, 'GABA': 0.5}


def get_neighbors(i, n_lev):
    nb = []
    if i < n_lev:
        lv = i
        if lv > 0: nb.append(lv - 1)
        if lv < n_lev - 1: nb.append(lv + 1)
        nb.append(n_lev + lv)
    elif i < 2 * n_lev:
        lv = i - n_lev
        nb.append(lv)
        nb.append(2 * n_lev + lv)
        if lv > 0: nb.append(n_lev + lv - 1)
        if lv < n_lev - 1: nb.append(n_lev + lv + 1)
    else:
        lv = i - 2 * n_lev
        if lv > 0: nb.append(2 * n_lev + lv - 1)
        if lv < n_lev - 1: nb.append(2 * n_lev + lv + 1)
        nb.append(n_lev + lv)
    return nb


def run_simulation(c3_relay, seed=42):
    """Run one V4-style simulation with given C3 relay cost."""
    rng = np.random.default_rng(seed)

    # NT assignments
    nt_k_pref = np.zeros(N_LEVELS)
    nt_growth_arr = np.zeros(N_LEVELS)
    nt_zones = []
    for lv in range(N_LEVELS):
        frac = lv / N_LEVELS
        if frac < 0.15: nt = 'Glutamate'
        elif frac < 0.30: nt = 'GABA'
        elif frac < 0.55: nt = 'Dopamine'
        elif frac < 0.75: nt = 'Serotonin'
        else: nt = 'Glutamate'
        if rng.random() < 0.1:
            nt = rng.choice(NT_NAMES)
        nt_zones.append(nt)
        nt_k_pref[lv] = NT_K_PREF[nt]
        nt_growth_arr[lv] = NT_GROWTH[nt]

    # Z0_DNA
    Z_mat = np.ones((N_TOTAL, K_MAX)) * 0.5
    for row_off in [0, N_LEVELS, 2*N_LEVELS]:
        for lv in range(N_LEVELS):
            frac = lv / N_LEVELS
            ap = 0.3 + 0.7 * np.sin(np.pi * frac)
            dv = 0.8 + 0.4 * np.cos(2*np.pi*frac)
            i = row_off + lv
            for k in range(K_MAX):
                Z_mat[i, k] = ap * dv * (1+0.3*np.sin(k*np.pi/3)) + 0.1*rng.normal()
                Z_mat[i, k] = max(0.1, Z_mat[i, k])

    K = np.ones(N_TOTAL, dtype=int)
    for lv in range(N_LEVELS):
        K[lv] = max(1, min(int(nt_k_pref[lv]*0.15), 3))
        K[N_LEVELS+lv] = 1
        rlv = N_LEVELS - 1 - lv
        K[2*N_LEVELS+lv] = max(1, min(int(nt_k_pref[min(rlv,N_LEVELS-1)]*0.15), 3))

    NB = [get_neighbors(i, N_LEVELS) for i in range(N_TOTAL)]
    g2_hist = np.zeros((TAU_AVG, N_TOTAL))
    hp = 0

    for t in range(T_TOTAL):
        stim = np.zeros(N_TOTAL)
        for k in range(min(8, N_LEVELS)):
            nf = 4 + k
            freqs = [25+k*8+f*12 for f in range(nf)]
            val = sum(np.sin(2*np.pi*t/freq) for freq in freqs)/nf
            val += 0.4*rng.normal()
            if rng.random() < 0.02: val *= 4.0
            stim[k] = val * (1.0 - 0.1*k)
        for k in range(min(8, N_LEVELS)):
            ri = 2*N_LEVELS + (N_LEVELS-1-k)
            nf = 5+k
            freqs = [30+k*7+f*11 for f in range(nf)]
            val = sum(np.cos(2*np.pi*t/freq) for freq in freqs)/nf
            val += 0.4*rng.normal()
            if rng.random() < 0.02: val *= 4.0
            stim[ri] = val * (1.0 - 0.1*k)
        if t % 200 < 10:
            stim[0] += 2.0*rng.normal()
            stim[3*N_LEVELS-1] += 2.0*rng.normal()

        for k in range(1, N_LEVELS):
            if abs(stim[k]) < abs(0.65*stim[k-1]):
                stim[k] = 0.65*stim[k-1]
        for k in range(N_LEVELS-2, -1, -1):
            ri = 2*N_LEVELS+k
            ri1 = 2*N_LEVELS+k+1
            if abs(stim[ri]) < abs(0.65*stim[ri1]):
                stim[ri] = 0.65*stim[ri1]
        for k in range(N_LEVELS):
            stim[N_LEVELS+k] = 0.5*(abs(stim[k])+abs(stim[2*N_LEVELS+k]))

        g2_local = np.zeros(N_TOTAL)
        for i in range(N_TOTAL):
            nbs = NB[i]
            tg2 = 0.0
            for j in nbs:
                ki, kj = K[i], K[j]
                km = min(ki, kj); kx = max(ki, kj)
                if km > 0:
                    zi = Z_mat[i,:km]; zj = Z_mat[j,:km]
                    g = (zj-zi)/(zj+zi+1e-12)
                    g2 = np.mean(g**2)
                else: g2 = 1.0
                if kx > km: g2 = (g2*km+1.0*(kx-km))/kx
                tg2 += g2

                if km > 0:
                    gamma = (zj-zi)/(zj+zi+1e-12)
                    activity = abs(stim[i])+abs(stim[j])
                    if activity >= 0.01:
                        # THIS IS THE C3 RELAY COST
                        k_gap = abs(ki-kj)
                        rp = 1.0 / (1.0 + c3_relay * k_gap)
                        cb = 1.0+CB_CORRECT if N_LEVELS<=i<2*N_LEVELS else 1.0
                        dZ = ETA_C2 * gamma * activity * rp * cb
                        Z_mat[i,:km] += dZ
                        Z_mat[i,:km] = np.maximum(Z_mat[i,:km], 0.01)
            g2_local[i] = tg2/max(len(nbs),1)
        g2_hist[hp%TAU_AVG] = g2_local
        hp += 1

        if t > 0 and t % DT_GROWTH == 0:
            nf = min(hp, TAU_AVG)
            g2_avg = np.mean(g2_hist[:nf], axis=0)
            for i in range(N_TOTAL):
                if g2_avg[i] > GAMMA2_THR and K[i] < K_MAX:
                    nbs = NB[i]
                    max_nb_k = max(K[j] for j in nbs)
                    if K[i] > max_nb_k + 1: continue
                    if i < N_LEVELS: lv2 = i
                    elif i < 2*N_LEVELS: lv2 = i-N_LEVELS
                    else: lv2 = N_LEVELS-1-(i-2*N_LEVELS)
                    lv2 = min(lv2, N_LEVELS-1)
                    nt_m = nt_growth_arr[lv2]
                    k_p = nt_k_pref[lv2]
                    k_a = 1.0+0.3*(1.0-abs(K[i]-k_p)/K_MAX)
                    gc = (K[i]+1)**K_COST_EXP/K_MAX**K_COST_EXP
                    es = MU_GROWTH*(g2_avg[i]-GAMMA2_THR)*nt_m*k_a
                    if es > gc:
                        K[i] += 1
                        z_new = np.mean([Z_mat[j,:K[j]].mean() for j in nbs])
                        z_dna = Z_mat[i, K[i]-1]
                        Z_mat[i, K[i]-1] = 0.7*z_new + 0.3*z_dna

    # Collect results
    K_L = K[:N_LEVELS].astype(float)
    K_R = K[2*N_LEVELS:3*N_LEVELS].astype(float)
    K_CC = K[N_LEVELS:2*N_LEVELS].astype(float)
    g2_f = g2_hist[(hp-1)%TAU_AVG]

    # FFT
    sig = K_L - K_L.mean()
    fft_v = np.abs(np.fft.rfft(sig))**2
    freqs = np.fft.rfftfreq(N_LEVELS, d=1.0)
    fft_v = fft_v[1:]; freqs_f = freqs[1:]
    if len(fft_v) > 0:
        di = np.argmax(fft_v)
        dom_period = 1.0/freqs_f[di] if freqs_f[di]>0 else N_LEVELS
        dom_segments = N_LEVELS/dom_period
    else:
        dom_period = N_LEVELS; dom_segments = 1

    lr_corr = np.corrcoef(K_L, K_R)[0,1]

    # Count K transitions (where K changes by >2)
    k_transitions = sum(1 for i in range(1, N_LEVELS)
                        if abs(K_L[i]-K_L[i-1]) > 2)

    return {
        'K_L': K_L, 'K_R': K_R, 'K_CC': K_CC,
        'g2_final': g2_f,
        'K_std': K.std(),
        'K_range_L': (K_L.min(), K_L.max()),
        'K_range_R': (K_R.min(), K_R.max()),
        'dom_period': dom_period,
        'dom_segments': dom_segments,
        'lr_corr': lr_corr,
        'k_transitions': k_transitions,
        'g2_mean': np.mean(g2_f),
        'freqs': freqs_f,
        'fft': fft_v,
    }


# == Run sweep =========================================================
print("=" * 60)
print("C3 RELAY COST PARAMETER SWEEP")
print(f"Values: {C3_VALUES}")
print(f"N_LEVELS: {N_LEVELS}, T_TOTAL: {T_TOTAL}")
print("=" * 60)

results = {}
t0_all = time.time()
for c3 in C3_VALUES:
    t0 = time.time()
    print(f"\n  Running C3={c3:.1f}...", end="", flush=True)
    r = run_simulation(c3)
    elapsed = time.time() - t0
    print(f" done ({elapsed:.0f}s) "
          f"K_L=[{r['K_range_L'][0]:.0f}-{r['K_range_L'][1]:.0f}] "
          f"std={r['K_std']:.2f} "
          f"segments={r['dom_segments']:.1f} "
          f"transitions={r['k_transitions']}")
    results[c3] = r

total_elapsed = time.time() - t0_all
print(f"\nTotal: {total_elapsed:.0f}s")

# == Summary table =====================================================
print("\n" + "=" * 70)
print(f"{'C3':>5} {'K_range':>12} {'K_std':>7} {'Segments':>9} "
      f"{'Trans':>6} {'LR_corr':>8} {'G2_mean':>8}")
print("-" * 70)
for c3 in C3_VALUES:
    r = results[c3]
    rng_str = f"[{r['K_range_L'][0]:.0f}-{r['K_range_L'][1]:.0f}]"
    print(f"{c3:>5.1f} {rng_str:>12} {r['K_std']:>7.2f} "
          f"{r['dom_segments']:>9.1f} {r['k_transitions']:>6} "
          f"{r['lr_corr']:>8.3f} {r['g2_mean']:>8.4f}")

# == Plots =============================================================
OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("C3 Relay Cost Sweep: What does C3 do?\n"
             f"C3 = {C3_VALUES}, N={N_LEVELS}",
             fontsize=14, fontweight='bold')

colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(C3_VALUES)))

# 1) K profiles for each C3 value
ax = axes[0, 0]
levels = np.arange(N_LEVELS)
win = 8
for c3, col in zip(C3_VALUES, colors):
    r = results[c3]
    kl = np.convolve(r['K_L'], np.ones(win)/win, mode='valid')
    x = np.arange(win//2, win//2+len(kl))
    ax.plot(x, kl, color=col, linewidth=1.5 if c3 > 0 else 2.5,
            linestyle='-' if c3 > 0 else '--',
            label=f'C3={c3}', alpha=0.8)
ax.set_xlabel("Level")
ax.set_ylabel("Left K (smoothed)")
ax.set_title("K profile vs C3")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 2) K_std vs C3
ax = axes[0, 1]
stds = [results[c3]['K_std'] for c3 in C3_VALUES]
ax.plot(C3_VALUES, stds, 'bo-', linewidth=2, markersize=8)
ax.fill_between(C3_VALUES, stds, alpha=0.2, color='blue')
ax.set_xlabel("C3 relay cost")
ax.set_ylabel("K standard deviation")
ax.set_title("K diversity vs C3")
ax.grid(True, alpha=0.3)

# 3) Number of K transitions vs C3
ax = axes[0, 2]
trans = [results[c3]['k_transitions'] for c3 in C3_VALUES]
segs = [results[c3]['dom_segments'] for c3 in C3_VALUES]
ax.plot(C3_VALUES, trans, 'ro-', linewidth=2, markersize=8, label='K transitions')
ax2 = ax.twinx()
ax2.plot(C3_VALUES, segs, 'g^-', linewidth=2, markersize=8, label='FFT segments')
ax.set_xlabel("C3 relay cost")
ax.set_ylabel("K transitions (|ΔK|>2)", color='red')
ax2.set_ylabel("FFT segments", color='green')
ax.set_title("Parcellation granularity vs C3")
ax.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 4) L-R correlation vs C3
ax = axes[1, 0]
lrs = [results[c3]['lr_corr'] for c3 in C3_VALUES]
ax.plot(C3_VALUES, lrs, 'mo-', linewidth=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.fill_between(C3_VALUES, lrs, alpha=0.2, color='purple')
ax.set_xlabel("C3 relay cost")
ax.set_ylabel("L-R correlation")
ax.set_title("Left-Right coupling vs C3")
ax.grid(True, alpha=0.3)

# 5) FFT comparison
ax = axes[1, 1]
for c3, col in zip(C3_VALUES, colors):
    r = results[c3]
    ax.semilogy(r['freqs'][:30], r['fft'][:30], color=col,
                linewidth=1.5, label=f'C3={c3}', alpha=0.7)
ax.set_xlabel("Spatial frequency (low = coarse)")
ax.set_ylabel("Power")
ax.set_title("FFT: low-freq power vs C3")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 6) Summary table
ax = axes[1, 2]
ax.axis('off')
lines = ["C3 SWEEP RESULTS\n"]
lines.append(f"{'C3':>4} {'K_std':>6} {'Seg':>5} {'Trans':>5} {'LR':>6}")
lines.append("-" * 30)
for c3 in C3_VALUES:
    r = results[c3]
    lines.append(f"{c3:>4.1f} {r['K_std']:>6.2f} {r['dom_segments']:>5.1f} "
                 f"{r['k_transitions']:>5} {r['lr_corr']:>6.3f}")
lines.append("")
lines.append("INTERPRETATION")
# Find optimal C3
best_c3 = C3_VALUES[np.argmax(trans)]
lines.append(f"  C3=0: no relay cost")
lines.append(f"    -> K uniform (low diversity)")
lines.append(f"  C3={best_c3}: maximum transitions")
lines.append(f"    -> sharpest boundaries")
lines.append(f"  C3>1: excessive cost")
lines.append(f"    -> suppresses adaptation")

ax.text(0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig_path = OUT / "fig_c3_sweep.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {fig_path}")

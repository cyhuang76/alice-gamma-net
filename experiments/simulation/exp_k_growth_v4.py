#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_k_growth_v4.py  -  K-Growth with Directionality, NT, and Z0_DNA
=====================================================================
Three new physics dimensions beyond V3:
  1. Afferent/Efferent directionality:
     - Each edge has a direction weight (aff vs eff)
     - Afferent edges propagate stimulus upward (sensory->abstract)
     - Efferent edges propagate commands downward (abstract->motor)
     - Asymmetric Gamma^2 contribution
  2. Neurotransmitter K-preference:
     - 4 NT zones: dopamine (high-K preference), serotonin (mid-K),
       GABA (inhibitory, low-K), glutamate (excitatory, broad)
     - Each zone has a K_preferred that biases growth
  3. Z0_DNA (genetic initial conditions):
     - Non-uniform initial Z from "soul decomposition theorem"
     - Encodes developmental gene gradients across cortex
     - Z0_DNA[i] = genetic blueprint, varies by position

Run with: py -3.13 experiments/simulation/exp_k_growth_v4.py
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
DT_GROWTH    = 25
ETA_C2       = 0.005
MU_GROWTH    = 3.5
GAMMA2_THR   = 0.03
K_MAX        = 20
K_COST_EXP   = 1.2
TAU_AVG      = 12
C3_RELAY     = 0.4
CB_CORRECT   = 0.3
N_TOTAL      = 3 * N_LEVELS

SL = slice(0, N_LEVELS)
SC = slice(N_LEVELS, 2*N_LEVELS)
SR = slice(2*N_LEVELS, 3*N_LEVELS)

rng = np.random.default_rng(42)

# == 1. Neurotransmitter zones =========================================
# Each level has a dominant NT that biases K-preference
# Modeled after cortical NT distribution gradients
NT_NAMES = ['Glutamate', 'Dopamine', 'Serotonin', 'GABA']
NT_K_PREF = {
    'Glutamate': 12,   # excitatory, broad K
    'Dopamine':  18,    # reward/planning, high K
    'Serotonin': 10,    # mood/homeostasis, mid K
    'GABA':       6,    # inhibitory, low K
}
NT_GROWTH_MULT = {
    'Glutamate': 1.2,   # facilitates growth
    'Dopamine':  1.5,    # strongly facilitates
    'Serotonin': 0.8,    # mildly inhibits growth
    'GABA':      0.5,    # strongly inhibits growth
}

# Assign NT zones (roughly matching cortical layers)
nt_zones = np.empty(N_LEVELS, dtype='U12')
nt_k_pref = np.zeros(N_LEVELS)
nt_growth = np.zeros(N_LEVELS)

for lv in range(N_LEVELS):
    frac = lv / N_LEVELS
    if frac < 0.15:
        nt = 'Glutamate'   # primary sensory (strong excitation)
    elif frac < 0.30:
        nt = 'GABA'         # secondary sensory (local inhibition)
    elif frac < 0.55:
        nt = 'Dopamine'     # association/PFC (planning, high K)
    elif frac < 0.75:
        nt = 'Serotonin'    # temporal/limbic (homeostasis)
    else:
        nt = 'Glutamate'    # motor cortex (strong excitation)

    # Add some noise to boundaries
    if rng.random() < 0.1:
        nt = rng.choice(NT_NAMES)

    nt_zones[lv] = nt
    nt_k_pref[lv] = NT_K_PREF[nt]
    nt_growth[lv] = NT_GROWTH_MULT[nt]

# == 2. Z0_DNA: Genetic initial conditions =============================
# From "soul decomposition theorem": Z = Z0_DNA + Z_learned
# Z0_DNA encodes developmental gradients (gene expression patterns)
# Modeled as smooth spatial gradients + gene-specific bumps

def generate_z0_dna(n_levels, k_max):
    """Generate developmental Z0 from gene gradients."""
    z0 = np.ones((3 * n_levels, k_max)) * 0.5

    for row_offset in [0, n_levels, 2*n_levels]:
        for lv in range(n_levels):
            frac = lv / n_levels
            # Anterior-posterior gradient (PAX6/EMX2-like)
            ap_grad = 0.3 + 0.7 * np.sin(np.pi * frac)
            # Dorsal-ventral gradient (secondary)
            dv_grad = 0.8 + 0.4 * np.cos(2 * np.pi * frac)

            i = row_offset + lv
            # Z0 = gene gradients + small individual variation
            for k in range(k_max):
                gene_expr = ap_grad * (1 + 0.3 * np.sin(k * np.pi / 3))
                z0[i, k] = gene_expr * dv_grad
                z0[i, k] += 0.1 * rng.normal()
                z0[i, k] = max(0.1, z0[i, k])

    return z0

Z_mat = generate_z0_dna(N_LEVELS, K_MAX)
K = np.ones(N_TOTAL, dtype=int)

# Set initial K based on NT preference (gene-driven)
for lv in range(N_LEVELS):
    # Left
    K[lv] = max(1, min(int(nt_k_pref[lv] * 0.15), 3))
    # CC
    K[N_LEVELS + lv] = 1
    # Right (mirror)
    rlv = N_LEVELS - 1 - lv
    K[2*N_LEVELS + lv] = max(1, min(int(nt_k_pref[rlv] * 0.15), 3))

# == 3. Afferent/Efferent directionality ==============================
# Each edge has a direction weight:
#   aff_weight: how much info flows upward (sensory->abstract)
#   eff_weight: how much info flows downward (abstract->motor)
# In the Left chain: afferent = left-to-right (low->high level)
# In the Right chain: efferent = right-to-left (high->low level)

def get_edge_weights(i, j):
    """Return (afferent_weight, efferent_weight) for edge i->j."""
    aff = 0.5  # default balanced
    eff = 0.5

    if i < N_LEVELS and j < N_LEVELS:
        # Left chain
        if j > i:  # forward (afferent)
            aff, eff = 0.8, 0.2
        else:       # backward (efferent / feedback)
            aff, eff = 0.2, 0.8
    elif i >= 2*N_LEVELS and j >= 2*N_LEVELS:
        # Right chain (reversed)
        if j < i:  # forward for right = decreasing index
            aff, eff = 0.8, 0.2
        else:
            aff, eff = 0.2, 0.8
    # CC edges: balanced
    return aff, eff

# == Network setup =====================================================
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

NB = [get_neighbors(i) for i in range(N_TOTAL)]
g2_hist = np.zeros((TAU_AVG, N_TOTAL))
hp = 0

K_snapshots = []
snap_times = []

print("=" * 60)
print("K-Growth V4: Directionality + NT + Z0_DNA")
print(f"Levels: {N_LEVELS}, Total nodes: {N_TOTAL}")
print(f"NT zones: Glut(0-15%,75-100%), GABA(15-30%), "
      f"DA(30-55%), 5HT(55-75%)")
print("=" * 60)

def compute_g2(i, j):
    ki, kj = K[i], K[j]
    km = min(ki, kj)
    kx = max(ki, kj)
    if km > 0:
        zi = Z_mat[i, :km]; zj = Z_mat[j, :km]
        g = (zj - zi) / (zj + zi + 1e-12)
        g2 = np.mean(g**2)
    else:
        g2 = 1.0
    if kx > km:
        g2 = (g2*km + 1.0*(kx-km))/kx
    return g2

t0 = time.time()

for t in range(T_TOTAL):
    # -- Stimulus (same as V3) --
    stim = np.zeros(N_TOTAL)
    for k in range(min(10, N_LEVELS)):
        nf = 4 + k
        freqs = [25 + k*8 + f*12 for f in range(nf)]
        val = sum(np.sin(2*np.pi*t/freq) for freq in freqs) / nf
        val += 0.4 * rng.normal()
        if rng.random() < 0.02: val *= 4.0
        stim[k] = val * (1.0 - 0.08*k)
    for k in range(min(10, N_LEVELS)):
        ri = 2*N_LEVELS + (N_LEVELS - 1 - k)
        nf = 5 + k
        freqs = [30 + k*7 + f*11 for f in range(nf)]
        val = sum(np.cos(2*np.pi*t/freq) for freq in freqs) / nf
        val += 0.4 * rng.normal()
        if rng.random() < 0.02: val *= 4.0
        stim[ri] = val * (1.0 - 0.08*k)
    if t % 200 < 10:
        stim[0] += 2.0 * rng.normal()
        stim[3*N_LEVELS - 1] += 2.0 * rng.normal()

    # Propagate with directionality
    for k in range(1, N_LEVELS):
        aff_w, _ = get_edge_weights(k-1, k)
        prop = 0.65 * aff_w / 0.5  # scale by afferent weight
        if abs(stim[k]) < abs(prop * stim[k-1]):
            stim[k] = prop * stim[k-1]
    for k in range(N_LEVELS - 2, -1, -1):
        ri = 2*N_LEVELS + k
        ri1 = 2*N_LEVELS + k + 1
        _, eff_w = get_edge_weights(ri1, ri)
        prop = 0.65 * eff_w / 0.5
        if abs(stim[ri]) < abs(prop * stim[ri1]):
            stim[ri] = prop * stim[ri1]
    for k in range(N_LEVELS):
        stim[N_LEVELS + k] = 0.5 * (abs(stim[k]) + abs(stim[2*N_LEVELS+k]))

    # -- C2 update --
    g2_local = np.zeros(N_TOTAL)
    for i in range(N_TOTAL):
        nbs = NB[i]
        tg2 = 0.0
        for j in nbs:
            g2 = compute_g2(i, j)

            # Direction-weighted Gamma^2
            aff_w, eff_w = get_edge_weights(i, j)
            dir_weight = max(aff_w, eff_w)  # dominant direction
            g2_weighted = g2 * dir_weight / 0.5

            tg2 += g2_weighted

            ki, kj = K[i], K[j]
            km = min(ki, kj)
            if km > 0:
                zi = Z_mat[i, :km]; zj = Z_mat[j, :km]
                gamma = (zj - zi) / (zj + zi + 1e-12)
                activity = abs(stim[i]) + abs(stim[j])
                if activity >= 0.01:
                    k_gap = abs(ki - kj)
                    rp = 1.0 / (1.0 + C3_RELAY * k_gap)
                    cb = 1.0 + CB_CORRECT if N_LEVELS <= i < 2*N_LEVELS else 1.0
                    dZ = ETA_C2 * gamma * activity * rp * cb * dir_weight
                    Z_mat[i, :km] += dZ
                    Z_mat[i, :km] = np.maximum(Z_mat[i, :km], 0.01)

        g2_local[i] = tg2 / max(len(nbs), 1)

    g2_hist[hp % TAU_AVG] = g2_local
    hp += 1

    # -- K growth with NT modulation --
    if t > 0 and t % DT_GROWTH == 0:
        nf = min(hp, TAU_AVG)
        g2_avg = np.mean(g2_hist[:nf], axis=0)

        for i in range(N_TOTAL):
            if g2_avg[i] > GAMMA2_THR and K[i] < K_MAX:
                nbs = NB[i]
                max_nb_k = max(K[j] for j in nbs)
                if K[i] > max_nb_k + 1:
                    continue

                # NT modulation of growth
                if i < N_LEVELS:
                    lv = i
                elif i < 2*N_LEVELS:
                    lv = i - N_LEVELS
                else:
                    lv = N_LEVELS - 1 - (i - 2*N_LEVELS)  # mirror
                lv = min(lv, N_LEVELS - 1)
                nt_mult = nt_growth[lv]

                # NT K-preference attractor
                k_pref = nt_k_pref[lv]
                k_attract = 1.0 + 0.3 * (1.0 - abs(K[i] - k_pref) / K_MAX)

                gc = (K[i]+1)**K_COST_EXP / K_MAX**K_COST_EXP
                es = MU_GROWTH * (g2_avg[i] - GAMMA2_THR) * nt_mult * k_attract
                if es > gc:
                    K[i] += 1
                    z_new = np.mean([Z_mat[j, :K[j]].mean() for j in nbs])
                    # Mix with Z0_DNA
                    z_dna = Z_mat[i, K[i]-1]  # original gene value
                    Z_mat[i, K[i]-1] = 0.7 * z_new + 0.3 * z_dna

    if t % (T_TOTAL // 20) == 0:
        K_snapshots.append(K.copy())
        snap_times.append(t)

    if t > 0 and t % 1000 == 0:
        elapsed = time.time() - t0
        eta = elapsed / t * (T_TOTAL - t)
        kl, kr = K[SL], K[SR]
        print(f"  t={t:5d}/{T_TOTAL}: L=[{kl.min()}-{kl.max()}] "
              f"R=[{kr.min()}-{kr.max()}] ({elapsed:.0f}s, ETA {eta:.0f}s)")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")

# == Results ===========================================================
K_L = K[SL].astype(float)
K_R = K[SR].astype(float)
K_CC = K[SC].astype(float)
g2_final = g2_hist[(hp-1) % TAU_AVG]

# FFT
sig_L = K_L - K_L.mean()
fft_L = np.abs(np.fft.rfft(sig_L))**2
freqs = np.fft.rfftfreq(N_LEVELS, d=1.0)
fft_L = fft_L[1:]; freqs_f = freqs[1:]
dom_idx = np.argmax(fft_L)
dom_period_L = 1.0 / freqs_f[dom_idx] if freqs_f[dom_idx] > 0 else 0
dom_segments_L = N_LEVELS / dom_period_L if dom_period_L > 0 else 0

sig_R = K_R - K_R.mean()
fft_R = np.abs(np.fft.rfft(sig_R))**2
fft_R = fft_R[1:]
dom_idx_R = np.argmax(fft_R)
dom_period_R = 1.0 / freqs_f[dom_idx_R] if freqs_f[dom_idx_R] > 0 else 0
dom_segments_R = N_LEVELS / dom_period_R if dom_period_R > 0 else 0

lr_corr = np.corrcoef(K_L, K_R)[0, 1]

# NT zone K averages
print("\n" + "=" * 60)
print("K BY NT ZONE")
print("=" * 60)
nt_summary = {}
for nt in NT_NAMES:
    mask = nt_zones == nt
    if mask.sum() > 0:
        kl_nt = K_L[mask]
        nt_summary[nt] = kl_nt.mean()
        print(f"  {nt:>12}: K_mean={kl_nt.mean():.2f} "
              f"[{kl_nt.min()}-{kl_nt.max()}] "
              f"(K_pref={NT_K_PREF[nt]}, n={mask.sum()})")

print(f"\n  FFT Left period:  {dom_period_L:.0f} levels ({dom_segments_L:.1f} segments)")
print(f"  FFT Right period: {dom_period_R:.0f} levels ({dom_segments_R:.1f} segments)")
print(f"  L-R correlation:  {lr_corr:.3f}")
print(f"  K range: L=[{K_L.min():.0f}-{K_L.max():.0f}] R=[{K_R.min():.0f}-{K_R.max():.0f}]")
print(f"  K std: {K.std():.2f}")

# == Plots =============================================================
OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("K-Growth V4: Directionality + NT + Z0_DNA (N=512)\n"
             "Does neurotransmitter zoning produce finer parcellation?",
             fontsize=14, fontweight='bold')

# 1) K profiles with NT zones shaded
ax = axes[0, 0]
levels = np.arange(N_LEVELS)
nt_colors = {'Glutamate': '#2ecc71', 'GABA': '#e74c3c',
             'Dopamine': '#3498db', 'Serotonin': '#f39c12'}
# Shade NT zones
for nt, col in nt_colors.items():
    mask = nt_zones == nt
    for lv in range(N_LEVELS):
        if mask[lv]:
            ax.axvspan(lv-0.5, lv+0.5, alpha=0.1, color=col)
win = 20
kl_sm = np.convolve(K_L, np.ones(win)/win, mode='valid')
kr_sm = np.convolve(K_R, np.ones(win)/win, mode='valid')
x_sm = np.arange(win//2, win//2 + len(kl_sm))
ax.plot(x_sm, kl_sm, 'b-', linewidth=2, label='Left')
ax.plot(x_sm, kr_sm, 'r-', linewidth=2, label='Right')
# NT K-preference line
ax.plot(levels, nt_k_pref, 'k--', linewidth=1, alpha=0.3, label='NT K_pref')
ax.set_xlabel("Processing level")
ax.set_ylabel("K value")
ax.set_title("K profile + NT zones")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)

# 2) K by NT zone (box plot)
ax = axes[0, 1]
nt_data = []
nt_labels = []
nt_cols = []
for nt in NT_NAMES:
    mask = nt_zones == nt
    if mask.sum() > 0:
        nt_data.append(K_L[mask])
        nt_labels.append(f"{nt}\n(K_pref={NT_K_PREF[nt]})")
        nt_cols.append(nt_colors[nt])
bp = ax.boxplot(nt_data, labels=nt_labels, patch_artist=True)
for patch, col in zip(bp['boxes'], nt_cols):
    patch.set_facecolor(col)
    patch.set_alpha(0.5)
ax.set_ylabel("K value")
ax.set_title("K by NT zone (Left hemisphere)")

# 3) FFT comparison V3 vs V4
ax = axes[0, 2]
ax.semilogy(freqs_f, fft_L, 'b-', linewidth=1, alpha=0.7, label='Left K')
ax.semilogy(freqs_f, fft_R, 'r-', linewidth=1, alpha=0.7, label='Right K')
brodmann_freq = 1.0 / (N_LEVELS / 52)
ax.axvline(x=brodmann_freq, color='green', linewidth=2, alpha=0.5,
           label=f'Brodmann (1/{N_LEVELS//52})')
ax.set_xlabel("Spatial frequency")
ax.set_ylabel("Power (log)")
ax.set_title("FFT Power Spectrum")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4) Z0_DNA vs final Z comparison
ax = axes[1, 0]
z0_profile = generate_z0_dna(N_LEVELS, K_MAX)
z0_mean = np.array([z0_profile[i, :max(1,K[i])].mean() for i in range(N_LEVELS)])
z_final_mean = np.array([Z_mat[i, :max(1,K[i])].mean() for i in range(N_LEVELS)])
if win > 1:
    z0_sm = np.convolve(z0_mean, np.ones(win)/win, mode='valid')
    zf_sm = np.convolve(z_final_mean, np.ones(win)/win, mode='valid')
    ax.plot(x_sm, z0_sm, 'gray', linewidth=1.5, alpha=0.5, label='Z0_DNA (initial)')
    ax.plot(x_sm, zf_sm, 'blue', linewidth=1.5, label='Z_final (learned)')
ax.set_xlabel("Level")
ax.set_ylabel("Mean Z")
ax.set_title("Z0_DNA vs Z_final (Left)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 5) K evolution
ax = axes[1, 1]
n_show = min(6, len(K_snapshots))
idxs = np.linspace(0, len(K_snapshots)-1, n_show, dtype=int)
for idx in idxs:
    snap = K_snapshots[idx]
    kl_s = snap[SL].astype(float)
    if win > 1:
        kl_s = np.convolve(kl_s, np.ones(win)/win, mode='valid')
    alpha = 0.3 + 0.7 * (idx / max(len(K_snapshots)-1, 1))
    ax.plot(x_sm if win > 1 else levels, kl_s, alpha=alpha, linewidth=1.2,
            label=f't={snap_times[idx]}')
ax.set_xlabel("Level")
ax.set_ylabel("Left K (smoothed)")
ax.set_title("K evolution over time")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 6) Summary
ax = axes[1, 2]
ax.axis('off')
nt_achieved = all(
    abs(nt_summary.get(nt, 0) - NT_K_PREF[nt]) < NT_K_PREF[nt] * 0.5
    for nt in NT_NAMES if nt in nt_summary
)
checks = [
    ("K diversity (std > 1.5)", K.std() > 1.5),
    ("NT zones differentiated", nt_achieved),
    ("DA zone highest K", nt_summary.get('Dopamine',0) > nt_summary.get('GABA',99)),
    ("GABA zone lowest K", nt_summary.get('GABA',99) < nt_summary.get('Dopamine',0)),
    ("L-R correlation > 0.3", lr_corr > 0.3),
    ("Finer parcellation (>9 segments)", dom_segments_L > 9 or dom_segments_R > 9),
    ("Gamma^2 reduced", np.mean(g2_final) < 0.5),
]
n_pass = sum(1 for _, v in checks if v)
lines = ["CLINICAL CHECKS (V4)\n"]
for desc, val in checks:
    lines.append(f"  {'[PASS]' if val else '[FAIL]'} {desc}")
lines.append(f"\nResult: {n_pass}/{len(checks)} passed")
lines.append(f"\nFFT: L period={dom_period_L:.0f} ({dom_segments_L:.1f} seg)")
lines.append(f"     R period={dom_period_R:.0f} ({dom_segments_R:.1f} seg)")
lines.append(f"L-R corr: {lr_corr:.3f}")
for nt in NT_NAMES:
    if nt in nt_summary:
        lines.append(f"{nt}: K={nt_summary[nt]:.1f} (pref={NT_K_PREF[nt]})")

color = 'green' if n_pass >= 5 else ('orange' if n_pass >= 3 else 'red')
ax.text(0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig_path = OUT / "fig_k_growth_v4_nt_dna.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {fig_path}")

print("\n" + "=" * 60)
print("CLINICAL CHECKS")
print("=" * 60)
for desc, val in checks:
    print(f"  {'PASS' if val else 'FAIL'} {desc}")
print(f"\nResult: {n_pass}/{len(checks)} passed")

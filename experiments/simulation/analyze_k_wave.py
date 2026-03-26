#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_k_wave.py  -  FFT Analysis of K-Profile Standing Wave
==============================================================
Re-runs the 512-level simulation briefly, then extracts FFT of K-profile
to identify dominant spatial frequencies and their correspondence to
cortical parcellation scales.

Run with: py -3.13 experiments/simulation/analyze_k_wave.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ── Re-run the 512 simulation (abbreviated) ─────────────────────────────
N_LEVELS = 512
T_TOTAL = 6000
DT_GROWTH = 30
ETA_C2 = 0.005
MU_GROWTH = 3.5
GAMMA2_THR = 0.03
K_MAX = 20
K_COST_EXP = 1.2
TAU_AVG = 12
C3_RELAY = 0.4
CB_CORRECT = 0.3
N_TOTAL = 3 * N_LEVELS

rng = np.random.default_rng(42)
K = np.ones(N_TOTAL, dtype=int)
Z_mat = rng.uniform(0.5, 1.5, size=(N_TOTAL, K_MAX))
g2_hist = np.zeros((TAU_AVG, N_TOTAL))
hp = 0

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

print("Running 512-level simulation...")
t0 = time.time()
for t in range(T_TOTAL):
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
    for k in range(1, N_LEVELS):
        if abs(stim[k]) < abs(0.65 * stim[k-1]):
            stim[k] = 0.65 * stim[k-1]
    for k in range(N_LEVELS - 2, -1, -1):
        ri = 2*N_LEVELS + k
        ri1 = 2*N_LEVELS + k + 1
        if abs(stim[ri]) < abs(0.65 * stim[ri1]):
            stim[ri] = 0.65 * stim[ri1]
    for k in range(N_LEVELS):
        stim[N_LEVELS + k] = 0.5 * (abs(stim[k]) + abs(stim[2*N_LEVELS+k]))

    g2_local = np.zeros(N_TOTAL)
    for i in range(N_TOTAL):
        nbs = NB[i]
        tg2 = 0.0
        for j in nbs:
            ki, kj = K[i], K[j]
            km = min(ki, kj)
            kx = max(ki, kj)
            if km > 0:
                zi = Z_mat[i, :km]; zj = Z_mat[j, :km]
                g = (zj - zi) / (zj + zi + 1e-12)
                g2 = np.mean(g**2)
            else:
                g2 = 1.0
            if kx > km: g2 = (g2*km + 1.0*(kx-km))/kx
            tg2 += g2
            if km > 0:
                gamma = (zj - zi) / (zj + zi + 1e-12)
                activity = abs(stim[i]) + abs(stim[j])
                if activity >= 0.01:
                    k_gap = abs(ki - kj)
                    rp = 1.0 / (1.0 + C3_RELAY * k_gap)
                    cb = 1.0 + CB_CORRECT if N_LEVELS <= i < 2*N_LEVELS else 1.0
                    dZ = ETA_C2 * gamma * activity * rp * cb
                    Z_mat[i, :km] += dZ
                    Z_mat[i, :km] = np.maximum(Z_mat[i, :km], 0.01)
        g2_local[i] = tg2 / max(len(nbs), 1)
    g2_hist[hp % TAU_AVG] = g2_local
    hp += 1

    if t > 0 and t % DT_GROWTH == 0:
        nf = min(hp, TAU_AVG)
        g2_avg = np.mean(g2_hist[:nf], axis=0)
        for i in range(N_TOTAL):
            if g2_avg[i] > GAMMA2_THR and K[i] < K_MAX:
                nbs = NB[i]
                max_nb_k = max(K[j] for j in nbs)
                if K[i] > max_nb_k + 1: continue
                gc = (K[i]+1)**K_COST_EXP / K_MAX**K_COST_EXP
                es = MU_GROWTH * (g2_avg[i] - GAMMA2_THR)
                if es > gc:
                    K[i] += 1
                    z_new = np.mean([Z_mat[j, :K[j]].mean() for j in nbs])
                    Z_mat[i, K[i]-1] = z_new

    if t > 0 and t % 2000 == 0:
        print(f"  t={t}/{T_TOTAL} ({time.time()-t0:.0f}s)")

elapsed = time.time() - t0
print(f"Simulation done in {elapsed:.0f}s")

# ── Extract profiles ────────────────────────────────────────────────────
K_L = K[:N_LEVELS].astype(float)
K_R = K[2*N_LEVELS:3*N_LEVELS].astype(float)
K_CC = K[N_LEVELS:2*N_LEVELS].astype(float)
g2_final = g2_hist[(hp-1) % TAU_AVG]
G2_L = g2_final[:N_LEVELS]
G2_R = g2_final[2*N_LEVELS:3*N_LEVELS]

# Left-Right correlation
lr_corr = np.corrcoef(K_L, K_R)[0, 1]

# ── FFT Analysis ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FFT ANALYSIS OF K-PROFILE")
print("=" * 60)

def fft_analysis(signal, label):
    """Perform FFT and return dominant spatial frequencies."""
    # Remove mean (DC component)
    sig = signal - signal.mean()
    N = len(sig)

    # FFT
    fft_vals = np.fft.rfft(sig)
    power = np.abs(fft_vals)**2
    freqs = np.fft.rfftfreq(N, d=1.0)  # cycles per level

    # Skip DC (freq=0)
    power = power[1:]
    freqs = freqs[1:]

    # Find peaks
    peak_idx = np.argsort(power)[::-1][:10]
    print(f"\n  {label}:")
    print(f"  {'Rank':>4} {'Freq':>8} {'Period':>8} {'Power':>10} {'Segments':>10}")
    for rank, idx in enumerate(peak_idx[:5]):
        freq = freqs[idx]
        period = 1.0 / freq if freq > 0 else float('inf')
        segments = N / period if period < float('inf') else 0
        print(f"  {rank+1:>4} {freq:>8.4f} {period:>8.1f} {power[idx]:>10.1f} {segments:>10.1f}")

    return freqs, power, peak_idx

freqs_L, power_L, peaks_L = fft_analysis(K_L, "Left K-profile")
freqs_R, power_R, peaks_R = fft_analysis(K_R, "Right K-profile")
freqs_G, power_G, peaks_G = fft_analysis(G2_L, "Left Gamma^2")

# Dominant wavelength
dom_freq_L = freqs_L[peaks_L[0]]
dom_period_L = 1.0 / dom_freq_L if dom_freq_L > 0 else float('inf')
dom_segments_L = N_LEVELS / dom_period_L

dom_freq_R = freqs_R[peaks_R[0]]
dom_period_R = 1.0 / dom_freq_R if dom_freq_R > 0 else float('inf')

print(f"\n  Left-Right K correlation: {lr_corr:.3f}")
print(f"  Dominant Left period: {dom_period_L:.1f} levels")
print(f"  Dominant Right period: {dom_period_R:.1f} levels")
print(f"  Expected segments: {dom_segments_L:.1f}")

# Brodmann comparison
cortex_length_cm = 50  # approximate unfolded cortex length
level_per_cm = N_LEVELS / cortex_length_cm
brodmann_avg_cm = cortex_length_cm / 52  # ~52 Brodmann areas
brodmann_levels = brodmann_avg_cm * level_per_cm
print(f"\n  Brodmann scale comparison:")
print(f"    Cortex ~ {cortex_length_cm} cm, {N_LEVELS} levels")
print(f"    Brodmann avg width ~ {brodmann_avg_cm:.1f} cm = {brodmann_levels:.0f} levels")
print(f"    Simulation dominant period = {dom_period_L:.0f} levels")
print(f"    Ratio sim/Brodmann = {dom_period_L/brodmann_levels:.1f}x")

# ── Plots ─────────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("K-Profile FFT Analysis: Standing Wave Decomposition\n"
             "What spatial frequencies does MRP select?",
             fontsize=14, fontweight='bold')

# 1) K profiles with smoothed overlay
ax = axes[0, 0]
levels = np.arange(N_LEVELS)
ax.plot(levels, K_L, 'b-', alpha=0.3, linewidth=0.5)
ax.plot(levels, K_R, 'r-', alpha=0.3, linewidth=0.5)
win = 30
kl_smooth = np.convolve(K_L, np.ones(win)/win, mode='valid')
kr_smooth = np.convolve(K_R, np.ones(win)/win, mode='valid')
x_smooth = np.arange(win//2, win//2 + len(kl_smooth))
ax.plot(x_smooth, kl_smooth, 'b-', linewidth=2, label='Left (smoothed)')
ax.plot(x_smooth, kr_smooth, 'r-', linewidth=2, label='Right (smoothed)')
ax.set_xlabel("Processing level")
ax.set_ylabel("K value")
ax.set_title("K profiles (raw + smoothed)")
ax.legend()
ax.grid(True, alpha=0.3)

# 2) FFT power spectrum
ax = axes[0, 1]
ax.semilogy(freqs_L, power_L, 'b-', linewidth=1.5, label='Left K', alpha=0.8)
ax.semilogy(freqs_R, power_R, 'r-', linewidth=1.5, label='Right K', alpha=0.8)
# Mark dominant peaks
for idx in peaks_L[:3]:
    ax.axvline(x=freqs_L[idx], color='blue', linestyle='--', alpha=0.5)
for idx in peaks_R[:3]:
    ax.axvline(x=freqs_R[idx], color='red', linestyle=':', alpha=0.5)
# Mark Brodmann frequency
brodmann_freq = 1.0 / brodmann_levels
ax.axvline(x=brodmann_freq, color='green', linestyle='-', linewidth=2,
           alpha=0.5, label=f'Brodmann freq (1/{brodmann_levels:.0f})')
ax.set_xlabel("Spatial frequency (cycles/level)")
ax.set_ylabel("Power (log)")
ax.set_title("FFT Power Spectrum")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3) Cross-correlation L vs R
ax = axes[1, 0]
corr = np.correlate(K_L - K_L.mean(), K_R - K_R.mean(), mode='full')
corr /= max(corr.max(), 1)
lags = np.arange(-N_LEVELS+1, N_LEVELS)
ax.plot(lags, corr, 'purple', linewidth=1)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
peak_lag = lags[np.argmax(corr)]
ax.axvline(x=peak_lag, color='red', linestyle='--',
           label=f'Peak lag = {peak_lag}')
ax.set_xlabel("Lag (levels)")
ax.set_ylabel("Cross-correlation")
ax.set_title(f"Left-Right cross-correlation (r={lr_corr:.3f})")
ax.legend()
ax.set_xlim(-200, 200)
ax.grid(True, alpha=0.3)

# 4) Summary
ax = axes[1, 1]
ax.axis('off')
summary = [
    "STANDING WAVE ANALYSIS",
    "",
    f"Left dominant period:   {dom_period_L:.0f} levels",
    f"Right dominant period:  {dom_period_R:.0f} levels",
    f"Expected segments:      {dom_segments_L:.1f}",
    f"L-R correlation:        {lr_corr:.3f}",
    f"Peak cross-corr lag:    {peak_lag} levels",
    "",
    "BRODMANN COMPARISON",
    f"  Brodmann avg width:   {brodmann_levels:.0f} levels",
    f"  Simulation period:    {dom_period_L:.0f} levels",
    f"  Ratio:                {dom_period_L/brodmann_levels:.1f}x",
    "",
    "INTERPRETATION",
    f"  MRP selects ~{dom_segments_L:.0f} functional zones",
    f"  L-R coupling: {'STRONG' if lr_corr > 0.5 else 'MODERATE' if lr_corr > 0.2 else 'WEAK'}",
    f"  = CC synchronisation {'confirmed' if lr_corr > 0.3 else 'partial'}",
]
ax.text(0.05, 0.95, "\n".join(summary),
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig_path = OUT / "fig_k_wave_fft_analysis.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {fig_path}")

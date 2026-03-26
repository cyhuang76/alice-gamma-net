#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_c3_relay_node.py  -  C3 as Physical Relay Node Insertion
=============================================================
CORRECT C3 implementation:
  C3 is NOT a penalty. C3 is the insertion of a relay node
  that impedance-matches between mismatched K levels.

  Like Murray's law in bone: K5 <-(relay)-> K1
  The relay's K is NOT arbitrary -- it is determined by the
  Gamma^2 SPECTRUM: the relay grows K modes at exactly the
  frequencies where the L-R mismatch is worst.

  This is the quarter-wave transformer principle:
    Z_relay = sqrt(Z_L * Z_R)  at the operating frequency

Physics:
  - CC relay nodes start at K=1
  - At each growth step, CC looks at per-mode Gamma^2 between
    its Left and Right neighbors
  - CC grows K at the mode where Gamma^2 is maximum
  - This means CC's Z profile is SHAPED by the mismatch spectrum

Sweep: C3_STRENGTH = [0, 0.5, 1.0, 2.0, 5.0]
  = how aggressively CC relays grow to match

Run with: py -3.13 experiments/simulation/exp_c3_relay_node.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

# == Physics ===========================================================
N_LEVELS  = 128
T_TOTAL   = 5000
DT_GROWTH = 20
ETA_C2    = 0.006
MU_GROWTH = 3.0
GAMMA2_THR = 0.03
K_MAX     = 15
K_COST_EXP = 1.2
TAU_AVG   = 12
CB_CORRECT = 0.3
N_TOTAL   = 3 * N_LEVELS

# C3 sweep: relay node growth aggressiveness
C3_STRENGTHS = [0.0, 0.5, 1.0, 2.0, 5.0]

NT_K_PREF = {'Glutamate': 12, 'Dopamine': 15, 'Serotonin': 10, 'GABA': 6}
NT_GROWTH = {'Glutamate': 1.2, 'Dopamine': 1.5, 'Serotonin': 0.8, 'GABA': 0.5}
NT_NAMES = list(NT_K_PREF.keys())


def get_neighbors(i, n_lev):
    nb = []
    if i < n_lev:
        lv = i
        if lv > 0: nb.append(lv - 1)
        if lv < n_lev - 1: nb.append(lv + 1)
        nb.append(n_lev + lv)
    elif i < 2 * n_lev:
        lv = i - n_lev
        nb.append(lv)                      # Left
        nb.append(2 * n_lev + lv)           # Right
        if lv > 0: nb.append(n_lev + lv - 1)
        if lv < n_lev - 1: nb.append(n_lev + lv + 1)
    else:
        lv = i - 2 * n_lev
        if lv > 0: nb.append(2 * n_lev + lv - 1)
        if lv < n_lev - 1: nb.append(2 * n_lev + lv + 1)
        nb.append(n_lev + lv)
    return nb


def compute_per_mode_gamma2(Z_mat, K, i, j):
    """Return per-mode Gamma^2 array between i and j."""
    ki, kj = K[i], K[j]
    km = min(ki, kj)
    if km == 0:
        return np.array([1.0])
    zi = Z_mat[i, :km]
    zj = Z_mat[j, :km]
    gamma = (zj - zi) / (zj + zi + 1e-12)
    return gamma**2


def run_simulation(c3_strength, seed=42):
    """Run simulation with physical C3 relay mechanism."""
    rng = np.random.default_rng(seed)

    # NT zones
    nt_k_pref = np.zeros(N_LEVELS)
    nt_growth_arr = np.zeros(N_LEVELS)
    for lv in range(N_LEVELS):
        frac = lv / N_LEVELS
        if frac < 0.15: nt = 'Glutamate'
        elif frac < 0.30: nt = 'GABA'
        elif frac < 0.55: nt = 'Dopamine'
        elif frac < 0.75: nt = 'Serotonin'
        else: nt = 'Glutamate'
        if rng.random() < 0.1: nt = rng.choice(NT_NAMES)
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
                Z_mat[i, k] = ap * dv * (1+0.3*np.sin(k*np.pi/3))
                Z_mat[i, k] += 0.1 * rng.normal()
                Z_mat[i, k] = max(0.1, Z_mat[i, k])

    K = np.ones(N_TOTAL, dtype=int)
    for lv in range(N_LEVELS):
        K[lv] = max(1, min(int(nt_k_pref[lv]*0.15), 3))
        K[N_LEVELS+lv] = 1  # CC starts minimal
        rlv = min(N_LEVELS-1-lv, N_LEVELS-1)
        K[2*N_LEVELS+lv] = max(1, min(int(nt_k_pref[rlv]*0.15), 3))

    NB = [get_neighbors(i, N_LEVELS) for i in range(N_TOTAL)]
    g2_hist = np.zeros((TAU_AVG, N_TOTAL))
    hp = 0

    # Track CC relay K growth events
    cc_growth_log = []

    for t in range(T_TOTAL):
        # Stimulus
        stim = np.zeros(N_TOTAL)
        for k in range(min(8, N_LEVELS)):
            nf = 4+k
            freqs = [25+k*8+f*12 for f in range(nf)]
            val = sum(np.sin(2*np.pi*t/freq) for freq in freqs)/nf
            val += 0.4*rng.normal()
            if rng.random() < 0.02: val *= 4.0
            stim[k] = val * (1.0-0.1*k)
        for k in range(min(8, N_LEVELS)):
            ri = 2*N_LEVELS+(N_LEVELS-1-k)
            nf = 5+k
            freqs = [30+k*7+f*11 for f in range(nf)]
            val = sum(np.cos(2*np.pi*t/freq) for freq in freqs)/nf
            val += 0.4*rng.normal()
            if rng.random() < 0.02: val *= 4.0
            stim[ri] = val * (1.0-0.1*k)
        if t % 200 < 10:
            stim[0] += 2.0*rng.normal()
            stim[3*N_LEVELS-1] += 2.0*rng.normal()

        for k in range(1, N_LEVELS):
            if abs(stim[k]) < abs(0.65*stim[k-1]):
                stim[k] = 0.65*stim[k-1]
        for k in range(N_LEVELS-2, -1, -1):
            ri = 2*N_LEVELS+k; ri1 = ri+1
            if abs(stim[ri]) < abs(0.65*stim[ri1]):
                stim[ri] = 0.65*stim[ri1]
        for k in range(N_LEVELS):
            stim[N_LEVELS+k] = 0.5*(abs(stim[k])+abs(stim[2*N_LEVELS+k]))

        # C2 update
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
                        cb = 1.0+CB_CORRECT if N_LEVELS<=i<2*N_LEVELS else 1.0
                        dZ = ETA_C2 * gamma * activity * cb
                        Z_mat[i,:km] += dZ
                        Z_mat[i,:km] = np.maximum(Z_mat[i,:km], 0.01)
            g2_local[i] = tg2/max(len(nbs),1)
        g2_hist[hp%TAU_AVG] = g2_local
        hp += 1

        # K growth
        if t > 0 and t % DT_GROWTH == 0:
            nf = min(hp, TAU_AVG)
            g2_avg = np.mean(g2_hist[:nf], axis=0)

            for i in range(N_TOTAL):
                is_cc = (N_LEVELS <= i < 2*N_LEVELS)

                # ALL nodes do standard C2 growth (including CC)
                if g2_avg[i] > GAMMA2_THR and K[i] < K_MAX:
                    nbs = NB[i]
                    max_nb_k = max(K[j] for j in nbs)
                    if K[i] <= max_nb_k + 1:
                        if i < N_LEVELS:
                            lv2 = i
                        elif i < 2*N_LEVELS:
                            lv2 = i - N_LEVELS
                        else:
                            lv2 = N_LEVELS-1-(i-2*N_LEVELS)
                        lv2 = min(lv2, N_LEVELS-1)
                        nt_m = nt_growth_arr[lv2]
                        k_p = nt_k_pref[lv2]
                        k_a = 1.0+0.3*(1.0-abs(K[i]-k_p)/K_MAX)
                        # CC gets cerebellum boost
                        cb_g = 1.0 + (CB_CORRECT if is_cc else 0.0)
                        gc = (K[i]+1)**K_COST_EXP/K_MAX**K_COST_EXP
                        es = MU_GROWTH*(g2_avg[i]-GAMMA2_THR)*nt_m*k_a*cb_g
                        if es > gc:
                            K[i] += 1
                            nbs = NB[i]
                            z_new = np.mean(
                                [Z_mat[j,:K[j]].mean() for j in nbs])
                            z_dna = Z_mat[i, K[i]-1]
                            Z_mat[i, K[i]-1] = 0.7*z_new+0.3*z_dna
                            if is_cc:
                                cc_growth_log.append(
                                    (t, i-N_LEVELS, K[i], g2_avg[i]))

                # ADDITIONALLY: CC does C3 relay Z-shaping
                if is_cc and c3_strength > 0:
                    lv = i - N_LEVELS
                    li = lv
                    ri = 2*N_LEVELS + lv
                    km_cc = min(K[i], K[li], K[ri])
                    if km_cc > 0:
                        z_l = Z_mat[li, :km_cc]
                        z_r = Z_mat[ri, :km_cc]
                        # Quarter-wave: Z_cc -> sqrt(Z_L * Z_R)
                        z_target = np.sqrt(np.abs(z_l * z_r)) + 0.01
                        Z_mat[i, :km_cc] += (
                            0.1*c3_strength*(z_target - Z_mat[i, :km_cc]))
                        Z_mat[i, :km_cc] = np.maximum(
                            Z_mat[i, :km_cc], 0.01)

        if t > 0 and t % 1000 == 0:
            kl, kr, kc = K[:N_LEVELS], K[2*N_LEVELS:], K[N_LEVELS:2*N_LEVELS]
            print(f"    t={t}: L=[{kl.min()}-{kl.max()}] "
                  f"CC=[{kc.min()}-{kc.max()}] R=[{kr.min()}-{kr.max()}]")

    # Results
    K_L = K[:N_LEVELS].astype(float)
    K_R = K[2*N_LEVELS:3*N_LEVELS].astype(float)
    K_CC = K[N_LEVELS:2*N_LEVELS].astype(float)
    g2_f = g2_hist[(hp-1)%TAU_AVG]

    # FFT
    sig = K_L - K_L.mean()
    fft_v = np.abs(np.fft.rfft(sig))**2
    freqs = np.fft.rfftfreq(N_LEVELS, d=1.0)
    fft_v = fft_v[1:]; freqs_f = freqs[1:]
    di = np.argmax(fft_v) if len(fft_v) > 0 else 0
    dom_p = 1.0/freqs_f[di] if len(freqs_f) > 0 and freqs_f[di] > 0 else N_LEVELS
    dom_s = N_LEVELS / dom_p

    lr_corr = np.corrcoef(K_L, K_R)[0, 1]
    lcc_corr = np.corrcoef(K_L, K_CC)[0, 1]
    k_trans = sum(1 for i in range(1, N_LEVELS) if abs(K_L[i]-K_L[i-1]) > 2)

    # Gamma^2 through CC vs direct
    g2_through_cc = []
    g2_direct = []
    for lv in range(N_LEVELS):
        li, ci, ri = lv, N_LEVELS+lv, 2*N_LEVELS+lv
        pm_direct = compute_per_mode_gamma2(Z_mat, K, li, ri)
        pm_lcc = compute_per_mode_gamma2(Z_mat, K, li, ci)
        pm_ccr = compute_per_mode_gamma2(Z_mat, K, ci, ri)
        g2_direct.append(np.mean(pm_direct))
        g2_through_cc.append(0.5*(np.mean(pm_lcc)+np.mean(pm_ccr)))
    g2_direct = np.array(g2_direct)
    g2_through_cc = np.array(g2_through_cc)
    cc_reduction = 1.0 - g2_through_cc.mean() / (g2_direct.mean()+1e-12)

    return {
        'K_L': K_L, 'K_R': K_R, 'K_CC': K_CC,
        'K_std': K.std(), 'K_std_CC': K_CC.std(),
        'dom_segments': dom_s,
        'lr_corr': lr_corr, 'lcc_corr': lcc_corr,
        'k_transitions': k_trans,
        'cc_growth_events': len(cc_growth_log),
        'g2_direct': g2_direct,
        'g2_through_cc': g2_through_cc,
        'cc_reduction': cc_reduction,
        'g2_mean': np.mean(g2_f),
        'freqs': freqs_f, 'fft': fft_v,
    }


# == Run sweep =========================================================
print("=" * 60)
print("C3 PHYSICAL RELAY NODE SWEEP")
print(f"C3 strengths: {C3_STRENGTHS}")
print(f"CC relay = quarter-wave transformer (Z = sqrt(Z_L*Z_R))")
print("=" * 60)

results = {}
t0_all = time.time()
for c3 in C3_STRENGTHS:
    t0 = time.time()
    print(f"\n  C3={c3:.1f}:", flush=True)
    r = run_simulation(c3)
    el = time.time() - t0
    print(f"  -> done ({el:.0f}s) "
          f"CC_K=[{r['K_CC'].min():.0f}-{r['K_CC'].max():.0f}] "
          f"CC_growth={r['cc_growth_events']} "
          f"G2_reduction={r['cc_reduction']:.1%}")
    results[c3] = r

total = time.time() - t0_all
print(f"\nTotal: {total:.0f}s")

# Summary
print("\n" + "=" * 70)
print(f"{'C3':>4} {'K_L':>8} {'K_CC':>8} {'CC_grow':>8} "
      f"{'G2_red':>8} {'LR_corr':>8} {'L-CC_corr':>10}")
print("-" * 70)
for c3 in C3_STRENGTHS:
    r = results[c3]
    print(f"{c3:>4.1f} [{r['K_L'].min():.0f}-{r['K_L'].max():.0f}]"
          f"   [{r['K_CC'].min():.0f}-{r['K_CC'].max():.0f}]"
          f"   {r['cc_growth_events']:>6}"
          f"   {r['cc_reduction']:>7.1%}"
          f"   {r['lr_corr']:>7.3f}"
          f"   {r['lcc_corr']:>9.3f}")

# == Plots =============================================================
OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("C3 as Physical Relay Node (Quarter-Wave Transformer)\n"
             f"Z_relay = sqrt(Z_L × Z_R) — Does CC reduce Γ²?",
             fontsize=14, fontweight='bold')

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(C3_STRENGTHS)))

# 1) CC K-profile for each C3
ax = axes[0, 0]
levels = np.arange(N_LEVELS)
for c3, col in zip(C3_STRENGTHS, colors):
    r = results[c3]
    win = 8
    kcc = np.convolve(r['K_CC'], np.ones(win)/win, mode='valid')
    x = np.arange(win//2, win//2+len(kcc))
    lw = 2.5 if c3 > 0 else 1.5
    ls = '-' if c3 > 0 else '--'
    ax.plot(x, kcc, color=col, linewidth=lw, linestyle=ls,
            label=f'C3={c3}', alpha=0.8)
ax.set_xlabel("Level")
ax.set_ylabel("CC relay K")
ax.set_title("CC relay K profile vs C3 strength")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 2) Gamma^2 reduction by CC
ax = axes[0, 1]
reds = [results[c3]['cc_reduction'] for c3 in C3_STRENGTHS]
ax.bar(range(len(C3_STRENGTHS)), [r*100 for r in reds],
       color=colors, edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(C3_STRENGTHS)))
ax.set_xticklabels([f'C3={c3}' for c3 in C3_STRENGTHS], fontsize=8)
ax.set_ylabel("Γ² reduction by CC relay (%)")
ax.set_title("CC relay effectiveness")
ax.grid(True, alpha=0.3, axis='y')

# 3) Direct vs through-CC Gamma^2 (best C3)
ax = axes[0, 2]
best_c3 = C3_STRENGTHS[np.argmax(reds)]
r_best = results[best_c3]
r_zero = results[0.0]
g2d_sm = np.convolve(r_best['g2_direct'], np.ones(win)/win, mode='valid')
g2c_sm = np.convolve(r_best['g2_through_cc'], np.ones(win)/win, mode='valid')
g2d0_sm = np.convolve(r_zero['g2_direct'], np.ones(win)/win, mode='valid')
x = np.arange(win//2, win//2+len(g2d_sm))
ax.plot(x, g2d_sm, 'r-', linewidth=1.5, label=f'Direct L↔R (C3={best_c3})')
ax.plot(x, g2c_sm, 'g-', linewidth=2, label=f'Through CC (C3={best_c3})')
ax.plot(x, g2d0_sm, 'r--', linewidth=1, alpha=0.5, label='Direct (C3=0)')
ax.set_xlabel("Level")
ax.set_ylabel("Gamma^2")
ax.set_title(f"Direct vs CC-relayed Γ² (C3={best_c3})")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4) L and CC K overlay (best C3)
ax = axes[1, 0]
kl_sm = np.convolve(r_best['K_L'], np.ones(win)/win, mode='valid')
kcc_sm = np.convolve(r_best['K_CC'], np.ones(win)/win, mode='valid')
kr_sm = np.convolve(r_best['K_R'], np.ones(win)/win, mode='valid')
ax.plot(x, kl_sm, 'b-', linewidth=2, label='Left')
ax.plot(x, kcc_sm, 'g-', linewidth=2.5, label='CC relay')
ax.plot(x, kr_sm, 'r-', linewidth=2, label='Right')
ax.fill_between(x, kl_sm, kcc_sm, alpha=0.1, color='cyan')
ax.fill_between(x, kcc_sm, kr_sm, alpha=0.1, color='orange')
ax.set_xlabel("Level")
ax.set_ylabel("K")
ax.set_title(f"L / CC / R K profiles (C3={best_c3})")
ax.legend()
ax.grid(True, alpha=0.3)

# 5) CC growth events histogram
ax = axes[1, 1]
events = [results[c3]['cc_growth_events'] for c3 in C3_STRENGTHS]
ax.bar(range(len(C3_STRENGTHS)), events,
       color=colors, edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(C3_STRENGTHS)))
ax.set_xticklabels([f'C3={c3}' for c3 in C3_STRENGTHS], fontsize=8)
ax.set_ylabel("CC growth events")
ax.set_title("Relay node growth activity")
ax.grid(True, alpha=0.3, axis='y')

# 6) Summary
ax = axes[1, 2]
ax.axis('off')
lines = ["C3 RELAY NODE RESULTS\n"]
lines.append(f"{'C3':>4} {'CC_K':>8} {'Growth':>7} {'G2_red':>7} {'LR':>6}")
lines.append("-" * 36)
for c3 in C3_STRENGTHS:
    r = results[c3]
    lines.append(f"{c3:>4.1f} [{r['K_CC'].min():.0f}-{r['K_CC'].max():.0f}]"
                 f" {r['cc_growth_events']:>6}"
                 f" {r['cc_reduction']:>6.1%}"
                 f" {r['lr_corr']:>6.3f}")
lines.append("")
lines.append("PHYSICS")
lines.append(f"  C3=0: CC stays at K=1")
lines.append(f"    -> no impedance matching")
lines.append(f"  C3>0: CC grows to sqrt(Z_L*Z_R)")
lines.append(f"    -> quarter-wave transformer")
lines.append(f"  Best: C3={best_c3}")
lines.append(f"    -> {r_best['cc_reduction']:.0%} G2 reduction")

ax.text(0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig_path = OUT / "fig_c3_relay_node.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {fig_path}")

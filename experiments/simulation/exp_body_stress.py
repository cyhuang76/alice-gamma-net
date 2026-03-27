#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_body_stress.py — Acute Stress: When C1 Becomes Critical
=============================================================
Key physics: C1 (brain) does TWO things C2+C3 cannot:
  1. PRIORITISE: redirect trunk resources to stressed organs
  2. COMPENSATE: push trunk Z back toward match when perturbed

Stress attacks trunk Z directly (not just demand).
C2 can only fix organ Z. C3 relay seeks geometric mean.
Only C1 can actively redirect trunk Z to defend critical organs.

py -3.13 experiments/simulation/exp_body_stress.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ── 29 tissues ───────────────────────────────────────────────
TISSUES = [
    ("Cortex",       12, 1.00, "neural"),
    ("Cerebellum",    8, 0.85, "neural"),
    ("Spinal cord",  10, 0.90, "neural"),
    ("Peripheral N",  6, 0.70, "neural"),
    ("Retina",        9, 0.88, "neural"),
    ("Cardiac",       5, 1.10, "muscle"),
    ("Skel. muscle",  4, 0.95, "muscle"),
    ("Smooth muscle", 3, 0.80, "muscle"),
    ("Liver",         7, 1.05, "visceral"),
    ("Kidney",        6, 1.15, "visceral"),
    ("Lung",          5, 0.75, "visceral"),
    ("Intestine",     4, 0.65, "visceral"),
    ("Stomach",       3, 0.60, "visceral"),
    ("Pancreas",      5, 0.90, "visceral"),
    ("Spleen",        4, 0.85, "immune"),
    ("Thymus",        3, 0.70, "immune"),
    ("Lymph node",    3, 0.72, "immune"),
    ("Bone marrow",   4, 0.78, "immune"),
    ("Bone",          2, 1.50, "structural"),
    ("Cartilage",     2, 1.30, "structural"),
    ("Tendon",        1, 1.40, "structural"),
    ("Skin",          3, 0.55, "barrier"),
    ("Cornea",        2, 0.50, "barrier"),
    ("Fat",           2, 0.40, "storage"),
    ("Thyroid",       4, 0.88, "endocrine"),
    ("Adrenal",       5, 0.92, "endocrine"),
    ("Pituitary",     6, 0.95, "endocrine"),
    ("Blood",         3, 0.50, "fluid"),
    ("Endothelium",   3, 0.60, "vascular"),
]

N_T = len(TISSUES)
NAMES = [t[0] for t in TISSUES]
K_MAX = 15
T_TOTAL = 3000
T_ON  = 500
T_OFF = 1500
ETA_C2 = 0.005
C3_RATE = 0.02    # C3 relay rate (slower → C1 matters more)
C1_GAIN = 0.08    # brain correction gain (strong)

CAT = {}
for i, (_, _, _, c) in enumerate(TISSUES):
    CAT.setdefault(c, []).append(i)


def gamma2(z1, z2):
    return ((z2 - z1) / (z2 + z1 + 1e-12)) ** 2


def run(enable_c1, stress_type, seed=42):
    rng = np.random.default_rng(seed)

    # Init tissue Z
    Z = np.zeros((N_T, K_MAX))
    K = np.zeros(N_T, dtype=int)
    for i, (_, k, zb, _) in enumerate(TISSUES):
        K[i] = k
        for m in range(k):
            Z[i, m] = zb * (1.0 + 0.15 * np.sin(m * np.pi / 3))
            Z[i, m] += 0.03 * rng.normal()
            Z[i, m] = max(0.05, Z[i, m])

    # Trunk Z (wide initial mismatch)
    Z_v = np.array([0.3 + 0.6 * rng.random() for _ in range(N_T)])
    Z_n = np.array([0.4 + 0.7 * rng.random() for _ in range(N_T)])

    H_hist = np.zeros((T_TOTAL, N_T))
    H_total = np.zeros(T_TOTAL)

    for t in range(T_TOTAL):
        in_stress = (T_ON <= t < T_OFF)

        # ── Demand ──
        demand = np.ones(N_T)
        for i in range(N_T):
            demand[i] = 1.0 + 0.15 * np.sin(2*np.pi*t/(80+i*5))
            demand[i] += 0.08 * rng.normal()
            demand[i] = max(0.1, demand[i])

        # ── Stress: attacks TRUNK Z directly ──
        if in_stress:
            prog = (t - T_ON) / (T_OFF - T_ON)

            if stress_type == "surgery":
                # 手術：5個目標器官的 trunk Z 被大幅擾動
                targets = [5, 8, 9, 10, 27]
                for idx in targets:
                    Z_v[idx] += 0.03 * (1 + prog) * rng.exponential(1)
                    Z_n[idx] += 0.02 * rng.exponential(1)
                    demand[idx] *= 2.0 + 2 * prog

            elif stress_type == "infection":
                # 感染：免疫器官 trunk Z 漂移 + 擴散到內臟
                for idx in CAT.get("immune", []):
                    Z_v[idx] += 0.02 * (1 + 2*prog) * rng.exponential(1)
                    Z_n[idx] += 0.01 * (1 + prog) * rng.exponential(1)
                    demand[idx] *= 2.0 + 3 * prog
                if prog > 0.3:
                    for idx in CAT.get("visceral", []):
                        Z_v[idx] += 0.01 * prog * rng.exponential(1)
                        demand[idx] *= 1.0 + prog

            elif stress_type == "exercise":
                # 運動：肌肉和心肺 trunk Z 暫時偏移
                for idx in CAT.get("muscle", []):
                    Z_v[idx] += 0.015 * rng.exponential(1)
                    demand[idx] *= 2.5
                Z_v[5] += 0.02 * rng.exponential(1)  # cardiac
                Z_v[10] += 0.01 * rng.exponential(1)  # lung
                demand[5] *= 2.0
                demand[10] *= 1.5

            elif stress_type == "hemorrhage":
                # 出血：全身 vascular Z 崩潰（血壓驟降）
                for i in range(N_T):
                    Z_v[i] += 0.02 * (1 + 4*prog) * rng.exponential(0.5)
                demand[27] *= max(0.1, 1.0 - 0.8*prog)  # blood loss
                demand[5] *= 1.5 + 2*prog  # cardiac compensates

            # Clamp trunk
            for i in range(N_T):
                Z_v[i] = max(0.01, Z_v[i])
                Z_n[i] = max(0.01, Z_n[i])

        # ── C2: organ matches local Z to demand ──
        for i in range(N_T):
            k = K[i]
            if k == 0: continue
            zd = demand[i]
            for m in range(k):
                g = (zd - Z[i, m]) / (zd + Z[i, m] + 1e-12)
                Z[i, m] += ETA_C2 * g * demand[i]
                Z[i, m] = max(0.01, Z[i, m])

        # ── C3: slow relay (quarter-wave) ──
        for i in range(N_T):
            k = K[i]
            if k == 0: continue
            za = np.mean(Z[i, :k])
            zt_v = np.sqrt(abs(za * Z_v[i])) + 0.01
            Z_v[i] += C3_RATE * (zt_v - Z_v[i])
            Z_v[i] = max(0.01, Z_v[i])
            zt_n = np.sqrt(abs(za * Z_n[i])) + 0.01
            Z_n[i] += C3_RATE * (zt_n - Z_n[i])
            Z_n[i] = max(0.01, Z_n[i])

        # ── C1: brain homeostatic control ──
        if enable_c1:
            for i in range(N_T):
                k = K[i]
                if k == 0: continue
                za = np.mean(Z[i, :k])
                g2_v = gamma2(za, Z_v[i])
                g2_n = gamma2(za, Z_n[i])
                g2_t = 1.0 - (1 - g2_v) * (1 - g2_n)
                threshold = 0.03
                if g2_t > threshold:
                    err = g2_t - threshold
                    # Brain actively pushes trunk toward organ match
                    Z_v[i] += C1_GAIN * err * (za - Z_v[i])
                    Z_n[i] += C1_GAIN * 0.8 * err * (za - Z_n[i])
                    Z_v[i] = max(0.01, Z_v[i])
                    Z_n[i] = max(0.01, Z_n[i])

        # ── Measure ──
        for i in range(N_T):
            k = K[i]
            if k == 0:
                H_hist[t, i] = 0.0
                continue
            za = np.mean(Z[i, :k])
            g2_v = gamma2(za, Z_v[i])
            g2_n = gamma2(za, Z_n[i])
            H_hist[t, i] = (1 - g2_v) * (1 - g2_n)
        H_total[t] = np.mean(H_hist[t])

    return {'H': H_hist, 'Ht': H_total}


# ── Run ──────────────────────────────────────────────────────
stresses = ["surgery", "infection", "exercise", "hemorrhage"]
labels = ["Surgery", "Infection", "Exercise", "Hemorrhage"]
t0 = time.time()

print("=" * 60)
print("ACUTE STRESS TEST: When does C1 become critical?")
print(f"29 tissues, T={T_TOTAL}, stress window t={T_ON}–{T_OFF}")
print("=" * 60)

results = {}
for s, lab in zip(stresses, labels):
    print(f"\n  {lab}...", flush=True)
    r1 = run(True, s)
    r0 = run(False, s)
    # During stress window
    h1s = r1['Ht'][T_ON:T_OFF]
    h0s = r0['Ht'][T_ON:T_OFF]
    # After stress (recovery)
    h1r = r1['Ht'][T_OFF:T_OFF+500]
    h0r = r0['Ht'][T_OFF:T_OFF+500]
    dh_stress = h1s.mean() - h0s.mean()
    dh_recov = h1r.mean() - h0r.mean()
    hmin1 = h1s.min()
    hmin0 = h0s.min()
    crit = ("★ CRITICAL" if dh_stress > 0.02
            else "!! important" if dh_stress > 0.005
            else "marginal")
    print(f"    C1 ON : H_min={hmin1:.3f}  recovery={h1r[-1]:.3f}")
    print(f"    C1 OFF: H_min={hmin0:.3f}  recovery={h0r[-1]:.3f}")
    print(f"    ΔH(stress)={dh_stress:+.4f}  ΔH(recov)={dh_recov:+.4f}  → {crit}")
    results[s] = {'c1': r1, 'no': r0, 'dh': dh_stress,
                  'dr': dh_recov, 'hm1': hmin1, 'hm0': hmin0}

print(f"\nDone in {time.time()-t0:.1f}s")

# ── Summary table ──
print("\n" + "=" * 70)
print(f"{'Stress':<13} {'H_min(C1)':>10} {'H_min(no)':>10} "
      f"{'ΔH_stress':>10} {'ΔH_recov':>10} {'Verdict'}")
print("-" * 70)
for s, lab in zip(stresses, labels):
    r = results[s]
    v = ("★ CRITICAL" if r['dh'] > 0.02
         else "!! important" if r['dh'] > 0.005
         else "marginal")
    print(f"{lab:<13} {r['hm1']:>10.3f} {r['hm0']:>10.3f} "
          f"{r['dh']:>+10.4f} {r['dr']:>+10.4f} {v}")

# ── Figure ────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "Acute Stress Test: When Does C1 (Brain) Become Critical?\n"
    "Green = C1 ON,  Red = C1 OFF  —  Grey zone = stress window",
    fontsize=14, fontweight='bold')

for idx, (s, lab) in enumerate(zip(stresses, labels)):
    ax = axes[idx//2, idx%2]
    r = results[s]
    win = 10
    h1 = np.convolve(r['c1']['Ht'], np.ones(win)/win, 'valid')
    h0 = np.convolve(r['no']['Ht'], np.ones(win)/win, 'valid')
    x = np.arange(len(h1))

    ax.axvspan(T_ON, T_OFF, alpha=0.12, color='grey', label='Stress')

    ax.plot(x, h1, color='#27ae60', lw=2.5,
            label=f'C1 ON (H_min={r["hm1"]:.3f})')
    ax.plot(x, h0, color='#c0392b', lw=2.5,
            label=f'C1 OFF (H_min={r["hm0"]:.3f})')
    ax.fill_between(x, h0, h1, where=(h1 > h0),
                     alpha=0.2, color='green')

    dh = r['dh']
    v = "★ CRITICAL" if dh > 0.02 else "important" if dh > 0.005 else "marginal"
    ax.set_title(f"{lab}  ΔH={dh:+.4f}  [{v}]",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (ticks)")
    ax.set_ylabel("Mean organ health H")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='red', ls='--', alpha=0.2, lw=1)

    # Annotate key moment
    if dh > 0.005:
        t_worst = T_ON + np.argmin(h0[T_ON:T_OFF])
        ax.annotate(f'C1 saves\n+{dh:.3f}',
                    xy=(t_worst, h0[min(t_worst, len(h0)-1)]),
                    xytext=(t_worst+200, h0[min(t_worst, len(h0)-1)]-0.1),
                    fontsize=9, color='green', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
fig_path = OUT / "fig_body_stress.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure: {fig_path}")

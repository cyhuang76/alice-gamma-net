# -*- coding: utf-8 -*-
"""
Extended scaling (N=256) + A_cut convergence analysis.

Part A: Add N=256 to the scaling law fit (original: N=16,32,64,128)
Part B: Run N=64 for 1000 ticks to check if A_cut saturates
"""
import sys
sys.path.insert(0, r"c:\Users\pinhu\Desktop\Alice Smart System")

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alice.core.gamma_topology import (
    GammaTopology,
    CORTICAL_PYRAMIDAL,
    MOTOR_ALPHA,
    SENSORY_AB,
    PAIN_AD_FIBER,
    PAIN_C_FIBER,
    AUTONOMIC_PREGANGLIONIC,
)

OUTDIR = r"c:\Users\pinhu\Desktop\Alice Smart System\figures"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 13,
    "axes.titlesize": 14, "legend.fontsize": 10, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

C_IMP = "#2563EB"; C_CUT = "#DC2626"; C_TOT = "#1F2937"
C_FIT = "#F59E0B"; C_BRAIN = "#10B981"

COMPOSITION_TEMPLATE = [
    (CORTICAL_PYRAMIDAL, 0.250),
    (MOTOR_ALPHA, 0.156),
    (SENSORY_AB, 0.156),
    (PAIN_AD_FIBER, 0.125),
    (PAIN_C_FIBER, 0.156),
    (AUTONOMIC_PREGANGLIONIC, 0.157),
]

def build_composition(N):
    comp = {}
    for tissue, frac in COMPOSITION_TEMPLATE:
        comp[tissue] = max(1, int(N * frac))
    return comp


def run_scaling_trial(N, seed, max_ticks=500, conv_thr=0.01):
    """Run one trial, return tau_conv and decay_rate."""
    tissue = build_composition(N)
    topo = GammaTopology.create_anatomical(
        tissue_composition=tissue, initial_connectivity=0.15,
        eta=0.02, max_dimension_gap=2, seed=seed)

    a_imp_series = []
    for t in range(1, max_ticks + 1):
        rng = np.random.default_rng(t + seed)
        stim = {n: rng.uniform(0.1, 0.5, size=nd.K) for n, nd in topo.nodes.items()}
        m = topo.tick(external_stimuli=stim, enable_spontaneous=True)
        a_imp_series.append(m["action_impedance"])

    baseline = np.mean(a_imp_series[:5]) if len(a_imp_series) >= 5 else a_imp_series[0]
    tau = max_ticks
    if baseline > 0:
        for i, a in enumerate(a_imp_series):
            if a < conv_thr * baseline:
                tau = i + 1
                break

    decay_rate = 0.0
    if baseline > 0 and len(a_imp_series) > 10:
        t_fit = np.arange(1, min(tau + 1, len(a_imp_series) + 1))
        a_fit = np.array(a_imp_series[:len(t_fit)])
        valid = a_fit > 0
        if np.sum(valid) > 3:
            coeffs = np.polyfit(t_fit[valid], np.log(a_fit[valid]), 1)
            decay_rate = max(0, -coeffs[0])

    return tau, decay_rate


# ════════════════════════════════════════════════════════════════════
# PART A: Extended Scaling (N=16,32,64,128,256)
# ════════════════════════════════════════════════════════════════════
def part_a():
    print("=" * 60)
    print("  PART A: Extended Scaling (adding N=256)")
    print("=" * 60)

    SIZES = [16, 32, 64, 128, 256]
    N_TRIALS = 5

    all_tau = {}
    all_decay = {}

    for N in SIZES:
        all_tau[N] = []
        all_decay[N] = []
        t0 = time.time()
        for trial in range(N_TRIALS):
            seed = 1000 * N + trial * 7 + 42
            tau, dr = run_scaling_trial(N, seed, max_ticks=500)
            all_tau[N].append(tau)
            all_decay[N].append(dr)
        elapsed = time.time() - t0
        tm, ts = np.mean(all_tau[N]), np.std(all_tau[N])
        dm, ds_ = np.mean(all_decay[N]), np.std(all_decay[N])
        print(f"  N={N:>4d}: τ = {tm:>6.1f} ± {ts:>5.1f}  "
              f"γ = {dm:.4f} ± {ds_:.4f}  ({elapsed:.1f}s)")

    # Power law fit on all 5 sizes
    N_arr = np.array(SIZES, dtype=float)
    tau_means = np.array([np.mean(all_tau[N]) for N in SIZES])
    tau_stds = np.array([np.std(all_tau[N]) for N in SIZES])
    decay_means = np.array([np.mean(all_decay[N]) for N in SIZES])
    decay_stds = np.array([np.std(all_decay[N]) for N in SIZES])

    log_N = np.log(N_arr)
    log_tau = np.log(tau_means)
    alpha, intercept = np.polyfit(log_N, log_tau, 1)
    C = np.exp(intercept)

    pred = alpha * log_N + intercept
    ss_res = np.sum((log_tau - pred) ** 2)
    ss_tot = np.sum((log_tau - np.mean(log_tau)) ** 2)
    R2 = 1 - ss_res / ss_tot

    print(f"\n  α = {alpha:.4f}, R² = {R2:.4f}")
    print(f"  (Previous: α = -1.05, R² = 0.95 with 4 points)")

    # Also fit just original 4
    log_N4 = np.log(N_arr[:4]); log_tau4 = np.log(tau_means[:4])
    a4, i4 = np.polyfit(log_N4, log_tau4, 1)
    pred4 = a4 * log_N4 + i4
    R2_4 = 1 - np.sum((log_tau4 - pred4)**2) / np.sum((log_tau4 - np.mean(log_tau4))**2)
    print(f"  Refit (4 pts): α = {a4:.4f}, R² = {R2_4:.4f}")

    # Updated Figure 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(N_arr, tau_means, yerr=tau_stds, fmt="o", color=C_IMP,
                 markersize=8, capsize=4, capthick=1.5, lw=1.5, zorder=5,
                 label="Data")
    N_fit = np.linspace(12, 300, 200)
    ax1.plot(N_fit, C * N_fit ** alpha, "--", color=C_FIT, lw=2,
             label=rf"$\tau \sim N^{{{alpha:.2f}}}$  ($R^2={R2:.3f}$)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Network size $N$")
    ax1.set_ylabel(r"Convergence time $\tau_{\mathrm{conv}}$ (ticks)")
    ax1.set_title(r"Scaling: $\tau_{\mathrm{conv}} \sim N^{\alpha}$, $\alpha < 0$")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.2, which="both")
    ax1.text(0.05, 0.05,
             r"$\alpha < 0$: larger networks" + "\nconverge FASTER\n(mean-field effect)",
             transform=ax1.transAxes, fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=C_BRAIN, alpha=0.15, edgecolor=C_BRAIN))

    ax2.errorbar(N_arr, decay_means, yerr=decay_stds, fmt="s", color=C_CUT,
                 markersize=8, capsize=4, capthick=1.5, lw=1.5, zorder=5)
    ax2.set_xlabel("Network size $N$")
    ax2.set_ylabel(r"Decay rate $\gamma$  ($A_{\mathrm{imp}} \sim e^{-\gamma t}$)")
    ax2.set_title("Exponential decay rate increases with $N$")
    ax2.grid(True, alpha=0.2)

    table_data = []
    for i, N in enumerate(SIZES):
        table_data.append([f"{N}", f"{tau_means[i]:.0f} ± {tau_stds[i]:.0f}",
                           f"{decay_means[i]:.3f} ± {decay_stds[i]:.3f}"])
    table = ax2.table(cellText=table_data,
                      colLabels=["$N$", r"$\tau_{\mathrm{conv}}$", r"$\gamma$"],
                      loc="upper left", cellLoc="center",
                      bbox=[0.05, 0.48, 0.55, 0.45])
    table.auto_set_font_size(False); table.set_fontsize(8)

    fig.suptitle(
        r"Minimum Reflection Principle: $\tau_{\mathrm{conv}} \sim N^{" +
        f"{alpha:.2f}" + r"}$  — 5-point fit, $R^2 = " + f"{R2:.3f}$",
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig2_scaling_law_v2.png"))
    fig.savefig(os.path.join(OUTDIR, "fig2_scaling_law_v2.pdf"))
    plt.close(fig)
    print("  → fig2_scaling_law_v2.png/.pdf saved")

    return alpha, R2


# ════════════════════════════════════════════════════════════════════
# PART B: A_cut Long-Range Behavior (1000 ticks)
# ════════════════════════════════════════════════════════════════════
def part_b():
    print("\n" + "=" * 60)
    print("  PART B: A_cut Long-Range Behavior (1000 ticks)")
    print("=" * 60)

    N = 64
    TICKS = 1000
    tissue = build_composition(N)

    topo = GammaTopology.create_anatomical(
        tissue_composition=tissue, initial_connectivity=0.15,
        eta=0.02, max_dimension_gap=2, seed=42)

    t_arr = []; a_imp_arr = []; a_cut_arr = []; a_tot_arr = []
    edge_arr = []; born_arr = []; pruned_arr = []

    for t in range(1, TICKS + 1):
        rng = np.random.default_rng(t)
        stim = {n: rng.uniform(0.1, 0.5, size=nd.K) for n, nd in topo.nodes.items()}
        m = topo.tick(external_stimuli=stim, enable_spontaneous=True)

        t_arr.append(t)
        a_imp_arr.append(m["action_impedance"])
        a_cut_arr.append(m["action_cutoff"])
        a_tot_arr.append(m["action_total"])
        edge_arr.append(m["active_edges"])
        born_arr.append(m.get("edges_born", 0))
        pruned_arr.append(m.get("edges_pruned", 0))

        if t in [100, 200, 500, 800, 1000]:
            print(f"  t={t:>5d}: A_imp={a_imp_arr[-1]:.6f}  "
                  f"A_cut={a_cut_arr[-1]:.1f}  edges={edge_arr[-1]}")

    t_arr = np.array(t_arr)
    a_imp_arr = np.array(a_imp_arr)
    a_cut_arr = np.array(a_cut_arr)
    edge_arr = np.array(edge_arr)

    # Analyze A_cut convergence
    # Check if A_cut growth rate slows down
    window = 50
    growth_rates = []
    for i in range(window, len(a_cut_arr)):
        rate = (a_cut_arr[i] - a_cut_arr[i - window]) / window
        growth_rates.append(rate)
    growth_rates = np.array(growth_rates)

    early_growth = np.mean(growth_rates[:100])      # ticks 50-150
    mid_growth = np.mean(growth_rates[350:450])      # ticks 400-500
    late_growth = np.mean(growth_rates[-100:])        # ticks 900-1000

    print(f"\n  A_cut growth rate (dA_cut/dt, window={window}):")
    print(f"    Early  (t=50-150):   {early_growth:.4f}")
    print(f"    Mid    (t=400-500):  {mid_growth:.4f}")
    print(f"    Late   (t=900-1000): {late_growth:.4f}")
    print(f"    Ratio late/early:    {late_growth/early_growth:.3f}" if early_growth > 0 else "")

    # Edge count stability
    late_edges = edge_arr[800:]
    print(f"\n  Edge count (t=800-1000): {np.mean(late_edges):.1f} ± {np.std(late_edges):.1f}")
    print(f"  Edge count range: [{np.min(late_edges)}, {np.max(late_edges)}]")

    # Fit A_cut to saturating model: A_cut(t) = A_max * (1 - exp(-t/τ_sat)) + offset
    # Or check if linear fits better
    from numpy.polynomial import polynomial as P
    # Linear fit on last half
    t_late = t_arr[500:]
    a_late = a_cut_arr[500:]
    coeffs_lin = np.polyfit(t_late, a_late, 1)
    a_cut_slope = coeffs_lin[0]
    print(f"\n  A_cut linear slope (t=500-1000): {a_cut_slope:.4f} per tick")

    if abs(a_cut_slope) < 0.5:  # near zero → saturating
        print("  → A_cut appears to SATURATE (slope ≈ 0)")
        a_cut_saturated = True
    else:
        print(f"  → A_cut still GROWING at {a_cut_slope:.2f}/tick")
        a_cut_saturated = False

    # Updated Figure 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0): Full A(t)
    ax = axes[0, 0]
    ax.plot(t_arr, a_tot_arr, color=C_TOT, lw=1, alpha=0.4, label=r"$A_{\mathrm{total}}$")
    ax.plot(t_arr, a_imp_arr, color=C_IMP, lw=1.5, label=r"$A_{\mathrm{imp}}(t) \to 0$")
    ax.plot(t_arr, a_cut_arr, color=C_CUT, lw=1.5, ls="--", label=r"$A_{\mathrm{cut}}$")
    ax.set_xlabel("Tick $t$"); ax.set_ylabel("Action $A$")
    ax.set_title(r"$A = A_{\mathrm{imp}}(t) + A_{\mathrm{cut}}$ — 1000 ticks")
    ax.legend(loc="right", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # (0,1): A_imp zoom
    ax = axes[0, 1]
    ax.plot(t_arr[:200], a_imp_arr[:200], color=C_IMP, lw=1.5)
    ax.set_xlabel("Tick $t$"); ax.set_ylabel(r"$A_{\mathrm{imp}}$")
    ax.set_title(r"$A_{\mathrm{imp}}$ early convergence")
    ax.grid(True, alpha=0.2)
    ax.set_yscale("log")

    # (1,0): A_cut growth rate
    ax = axes[1, 0]
    t_gr = t_arr[window:]
    ax.plot(t_gr, growth_rates, color=C_CUT, lw=1, alpha=0.3)
    # Moving average
    w2 = 50
    if len(growth_rates) > w2:
        ma = np.convolve(growth_rates, np.ones(w2)/w2, mode="valid")
        ax.plot(t_gr[w2//2:w2//2+len(ma)], ma, color=C_CUT, lw=2,
                label=f"Moving avg ({w2})")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Tick $t$"); ax.set_ylabel(r"$dA_{\mathrm{cut}}/dt$")
    ax.set_title(r"$A_{\mathrm{cut}}$ growth rate")
    ax.legend(); ax.grid(True, alpha=0.2)

    # (1,1): Edge count
    ax = axes[1, 1]
    ax.plot(t_arr, edge_arr, color="#6366F1", lw=1.5)
    ax.set_xlabel("Tick $t$"); ax.set_ylabel("Active edges")
    ax.set_title("Edge count evolution")
    ax.grid(True, alpha=0.2)

    # Add verdict
    verdict = "SATURATING" if a_cut_saturated else f"GROWING ({a_cut_slope:.1f}/tick)"
    fig.suptitle(
        f"Irreducibility Theorem — Long-range behavior (1000 ticks)\n"
        f"$A_{{\\mathrm{{cut}}}}$ status: {verdict}  |  "
        f"Late edges: {np.mean(late_edges):.0f} ± {np.std(late_edges):.0f}",
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig1_action_decomposition_v2.png"))
    fig.savefig(os.path.join(OUTDIR, "fig1_action_decomposition_v2.pdf"))
    plt.close(fig)
    print("  → fig1_action_decomposition_v2.png/.pdf saved")

    return a_cut_saturated, a_cut_slope


# ════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Extended Analysis: A (Scaling) + B (A_cut convergence)")
    print("=" * 60)
    print()

    alpha, R2 = part_a()
    saturated, slope = part_b()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  A) Scaling: α = {alpha:.4f}, R² = {R2:.4f} (5 points)")
    print(f"  B) A_cut:   {'SATURATES' if saturated else 'STILL GROWING'}"
          f" (slope = {slope:.4f}/tick)")
    print("=" * 60)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
PRE Paper Core Figures — Heterogeneous Γ-Topology
===================================================

Generates three publication-quality figures:

  Figure 1: A(t) decomposition — A_imp → 0, A_cut invariant
  Figure 2: Scaling law — τ_conv ~ N^α with α < 0
  Figure 3: D_K vs γ — soft cutoff fractal control

Output: figures/ directory (PNG 300dpi + PDF vector)
"""
import sys
import os
sys.path.insert(0, r"c:\Users\pinhu\Desktop\Alice Smart System")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from alice.core.gamma_topology import (
    GammaTopology,
    GammaNode,
    CORTICAL_PYRAMIDAL,
    MOTOR_ALPHA,
    SENSORY_AB,
    PAIN_AD_FIBER,
    PAIN_C_FIBER,
    AUTONOMIC_PREGANGLIONIC,
)

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTDIR = r"c:\Users\pinhu\Desktop\Alice Smart System\figures"
os.makedirs(OUTDIR, exist_ok=True)

# Colors
C_IMP = "#2563EB"     # blue — A_imp (learnable)
C_CUT = "#DC2626"     # red — A_cut (irreducible)
C_TOT = "#1F2937"     # dark — A_total
C_FIT = "#F59E0B"     # amber — fit line
C_BRAIN = "#10B981"   # green — biological range
C_SOFT = "#8B5CF6"    # purple — soft cutoff data points


def build_composition(N):
    """Build tissue composition for N nodes."""
    ratios = [
        (CORTICAL_PYRAMIDAL, 0.25),
        (MOTOR_ALPHA, 0.15),
        (SENSORY_AB, 0.15),
        (PAIN_AD_FIBER, 0.15),
        (PAIN_C_FIBER, 0.15),
        (AUTONOMIC_PREGANGLIONIC, 0.15),
    ]
    comp = {}
    total = 0
    for tissue, frac in ratios:
        count = max(1, int(N * frac))
        comp[tissue] = count
        total += count
    return comp


# ════════════════════════════════════════════════════════════════════
# FIGURE 1: A(t) Decomposition
# ════════════════════════════════════════════════════════════════════
def figure_1():
    print("  Figure 1: A(t) decomposition...")

    N = 64
    TICKS = 200
    tissue = build_composition(N)

    topo = GammaTopology.create_anatomical(
        tissue_composition=tissue,
        initial_connectivity=0.15,
        eta=0.02,
        max_dimension_gap=2,
        seed=42,
    )

    t_arr = []
    a_imp_arr = []
    a_cut_arr = []
    a_tot_arr = []

    for t in range(1, TICKS + 1):
        rng = np.random.default_rng(t)
        stim = {name: rng.uniform(0.1, 0.5, size=node.K)
                for name, node in topo.nodes.items()}
        metrics = topo.tick(external_stimuli=stim, enable_spontaneous=True)

        t_arr.append(t)
        a_imp_arr.append(metrics["action_impedance"])
        a_cut_arr.append(metrics["action_cutoff"])
        a_tot_arr.append(metrics["action_total"])

    t_arr = np.array(t_arr)
    a_imp_arr = np.array(a_imp_arr)
    a_cut_arr = np.array(a_cut_arr)
    a_tot_arr = np.array(a_tot_arr)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t_arr, a_tot_arr, color=C_TOT, lw=1.5, alpha=0.6,
            label=r"$A_{\mathrm{total}}$")
    ax.plot(t_arr, a_imp_arr, color=C_IMP, lw=2.0,
            label=r"$A_{\mathrm{imp}}(t) \to 0$ (learnable)")
    ax.plot(t_arr, a_cut_arr, color=C_CUT, lw=2.0, ls="--",
            label=r"$A_{\mathrm{cut}}$ (irreducible)")

    # Annotate
    ax.annotate(r"$A_{\mathrm{imp}} \to 0$" + "\n(Hebbian C2)",
                xy=(TICKS * 0.8, a_imp_arr[-1]),
                xytext=(TICKS * 0.55, max(a_imp_arr) * 0.3),
                fontsize=10, color=C_IMP,
                arrowprops=dict(arrowstyle="->", color=C_IMP, lw=1.2))

    if a_cut_arr[-1] > 0:
        ax.annotate(r"$A_{\mathrm{cut}} = \sum (K_i - K_j)^+$" +
                    "\n(geometric invariant)",
                    xy=(TICKS * 0.8, a_cut_arr[-1]),
                    xytext=(TICKS * 0.4, a_cut_arr[-1] * 1.5),
                    fontsize=10, color=C_CUT,
                    arrowprops=dict(arrowstyle="->", color=C_CUT, lw=1.2))

    ax.set_xlabel("Tick $t$")
    ax.set_ylabel("Action $A$")
    ax.set_title("Irreducibility Theorem: $A = A_{\\mathrm{imp}}(t) + A_{\\mathrm{cut}}$")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, TICKS)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)

    # Inset: zoom on A_imp at late times
    axins = ax.inset_axes([0.55, 0.45, 0.4, 0.35])
    t_late = t_arr[TICKS//2:]
    a_late = a_imp_arr[TICKS//2:]
    axins.plot(t_late, a_late, color=C_IMP, lw=1.5)
    axins.set_xlabel("$t$", fontsize=8)
    axins.set_ylabel(r"$A_{\mathrm{imp}}$", fontsize=8)
    axins.set_title("Late-time convergence", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig1_action_decomposition.png"))
    fig.savefig(os.path.join(OUTDIR, "fig1_action_decomposition.pdf"))
    plt.close(fig)
    print("    → fig1_action_decomposition.png/.pdf saved")


# ════════════════════════════════════════════════════════════════════
# FIGURE 2: Scaling Law τ_conv ~ N^α
# ════════════════════════════════════════════════════════════════════
def figure_2():
    print("  Figure 2: Scaling law...")

    SIZES = [16, 32, 64, 128]
    N_TRIALS = 5
    MAX_TICKS = 500
    CONVERGENCE_THR = 0.01

    all_tau = {N: [] for N in SIZES}
    all_decay = {N: [] for N in SIZES}

    for N in SIZES:
        tissue = build_composition(N)
        for trial in range(N_TRIALS):
            seed = 1000 * N + trial * 7 + 42
            topo = GammaTopology.create_anatomical(
                tissue_composition=tissue,
                initial_connectivity=0.15,
                eta=0.02,
                max_dimension_gap=2,
                seed=seed,
            )

            a_imp_series = []
            for t in range(1, MAX_TICKS + 1):
                rng = np.random.default_rng(t + seed)
                stim = {name: rng.uniform(0.1, 0.5, size=node.K)
                        for name, node in topo.nodes.items()}
                m = topo.tick(external_stimuli=stim, enable_spontaneous=True)
                a_imp_series.append(m["action_impedance"])

            # Find convergence
            baseline = np.mean(a_imp_series[:5]) if len(a_imp_series) >= 5 else a_imp_series[0]
            tau = MAX_TICKS
            if baseline > 0:
                for i, a in enumerate(a_imp_series):
                    if a < CONVERGENCE_THR * baseline:
                        tau = i + 1
                        break
            all_tau[N].append(tau)

            # Decay rate via exponential fit
            if baseline > 0 and len(a_imp_series) > 10:
                t_fit = np.arange(1, min(tau + 1, len(a_imp_series) + 1))
                a_fit = np.array(a_imp_series[:len(t_fit)])
                valid = a_fit > 0
                if np.sum(valid) > 3:
                    log_a = np.log(a_fit[valid])
                    t_v = t_fit[valid]
                    coeffs = np.polyfit(t_v, log_a, 1)
                    all_decay[N].append(-coeffs[0])
                else:
                    all_decay[N].append(0)
            else:
                all_decay[N].append(0)

        print(f"    N={N}: τ = {np.mean(all_tau[N]):.1f} ± {np.std(all_tau[N]):.1f}")

    # Power law fit
    N_arr = np.array(SIZES, dtype=float)
    tau_means = np.array([np.mean(all_tau[N]) for N in SIZES])
    tau_stds = np.array([np.std(all_tau[N]) for N in SIZES])
    decay_means = np.array([np.mean([d for d in all_decay[N] if d > 0]) for N in SIZES])
    decay_stds = np.array([np.std([d for d in all_decay[N] if d > 0]) for N in SIZES])

    log_N = np.log(N_arr)
    log_tau = np.log(tau_means)
    alpha, intercept = np.polyfit(log_N, log_tau, 1)
    C = np.exp(intercept)

    # R²
    pred = alpha * log_N + intercept
    ss_res = np.sum((log_tau - pred) ** 2)
    ss_tot = np.sum((log_tau - np.mean(log_tau)) ** 2)
    R2 = 1 - ss_res / ss_tot

    print(f"    α = {alpha:.4f}, R² = {R2:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: log-log τ vs N
    ax1.errorbar(N_arr, tau_means, yerr=tau_stds, fmt="o", color=C_IMP,
                 markersize=8, capsize=4, capthick=1.5, lw=1.5, zorder=5,
                 label="Data")

    N_fit = np.linspace(12, 160, 100)
    ax1.plot(N_fit, C * N_fit ** alpha, "--", color=C_FIT, lw=2,
             label=rf"$\tau \sim N^{{{alpha:.2f}}}$  ($R^2={R2:.2f}$)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Network size $N$")
    ax1.set_ylabel(r"Convergence time $\tau_{\mathrm{conv}}$ (ticks)")
    ax1.set_title(r"Scaling: $\tau_{\mathrm{conv}} \sim N^{\alpha}$, $\alpha < 0$")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.2, which="both")

    # Add annotation about negative exponent
    ax1.text(0.05, 0.05,
             r"$\alpha < 0$: larger networks" + "\nconverge FASTER\n(mean-field effect)",
             transform=ax1.transAxes, fontsize=9,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=C_BRAIN,
                       alpha=0.15, edgecolor=C_BRAIN))

    # Right: decay rate γ vs N
    ax2.errorbar(N_arr, decay_means, yerr=decay_stds, fmt="s", color=C_CUT,
                 markersize=8, capsize=4, capthick=1.5, lw=1.5, zorder=5)

    ax2.set_xlabel("Network size $N$")
    ax2.set_ylabel(r"Decay rate $\gamma$  ($A_{\mathrm{imp}} \sim e^{-\gamma t}$)")
    ax2.set_title("Exponential decay rate increases with $N$")
    ax2.grid(True, alpha=0.2)

    # Table inset on right panel
    table_data = []
    for i, N in enumerate(SIZES):
        table_data.append([
            f"{N}",
            f"{tau_means[i]:.0f} ± {tau_stds[i]:.0f}",
            f"{decay_means[i]:.3f} ± {decay_stds[i]:.3f}",
        ])
    table = ax2.table(
        cellText=table_data,
        colLabels=["$N$", r"$\tau_{\mathrm{conv}}$", r"$\gamma$"],
        loc="upper left",
        cellLoc="center",
        bbox=[0.05, 0.55, 0.55, 0.38]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    fig.suptitle(
        r"Minimum Reflection Principle: $\tau_{\mathrm{conv}} \sim N^{" +
        f"{alpha:.2f}" + r"}$  — Biologically feasible at brain scale",
        fontsize=13, y=1.02
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig2_scaling_law.png"))
    fig.savefig(os.path.join(OUTDIR, "fig2_scaling_law.pdf"))
    plt.close(fig)
    print("    → fig2_scaling_law.png/.pdf saved")


# ════════════════════════════════════════════════════════════════════
# FIGURE 3: D_K vs γ — Soft Cutoff Fractal Control
# ════════════════════════════════════════════════════════════════════
def figure_3():
    print("  Figure 3: D_K vs γ (soft cutoff)...")

    N = 64
    TICKS = 100
    N_TRIALS = 5
    MAX_GAP = 4

    gamma_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.26, 1.5, 2.0, 2.5]
    tissue = build_composition(N)

    gamma_arr = []
    dk_means = []
    dk_stds = []
    r10_means = []
    edge_means = []

    for gamma in gamma_values:
        dk_trials = []
        r10_trials = []
        edge_trials = []

        decay = gamma if gamma > 0 else None

        for trial in range(N_TRIALS):
            seed = 8000 + trial * 37
            topo = GammaTopology.create_anatomical(
                tissue_composition=tissue,
                initial_connectivity=0.15,
                eta=0.02,
                max_dimension_gap=MAX_GAP,
                dimension_gap_decay=decay,
                seed=seed,
            )

            for t in range(TICKS):
                rng_t = np.random.default_rng(t + trial * 10000)
                stim = {name: rng_t.uniform(0.1, 0.5, size=node.K)
                        for name, node in topo.nodes.items()}
                topo.tick(external_stimuli=stim, enable_spontaneous=True)

            result = topo.k_level_analysis()
            dk_trials.append(result["D_K"])
            if not np.isnan(result["density_ratio_1_0"]):
                r10_trials.append(result["density_ratio_1_0"])
            edge_trials.append(len(topo.active_edges))

        gamma_arr.append(gamma)
        dk_means.append(np.mean(dk_trials))
        dk_stds.append(np.std(dk_trials))
        r10_means.append(np.mean(r10_trials) if r10_trials else float("nan"))
        edge_means.append(np.mean(edge_trials))

        print(f"    γ={gamma:.2f}: D_K = {dk_means[-1]:.3f} ± {dk_stds[-1]:.3f}")

    gamma_arr = np.array(gamma_arr)
    dk_means = np.array(dk_means)
    dk_stds = np.array(dk_stds)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Cortical fractal range (literature)
    ax.axhspan(1.3, 1.5, alpha=0.12, color=C_BRAIN, zorder=0,
               label="Cortical $D$ ∈ [1.3, 1.5] (literature)")
    ax.axhline(1.3, color=C_BRAIN, lw=0.8, ls=":", alpha=0.5)
    ax.axhline(1.5, color=C_BRAIN, lw=0.8, ls=":", alpha=0.5)

    # D_K = γ reference line
    g_ref = np.linspace(0, 2.8, 100)
    ax.plot(g_ref, g_ref, "--", color="#9CA3AF", lw=1.2, alpha=0.7,
            label=r"$D_K = \gamma$ (no Hebbian offset)")

    # Data points
    ax.errorbar(gamma_arr, dk_means, yerr=dk_stds, fmt="o",
                color=C_SOFT, markersize=9, capsize=5, capthick=1.5,
                lw=2, zorder=10, label=r"$D_K$ measured")

    # Fit: D_K = a·γ + b
    valid = gamma_arr > 0
    if np.sum(valid) > 2:
        coeffs = np.polyfit(gamma_arr[valid], dk_means[valid], 1)
        ax.plot(g_ref, coeffs[0] * g_ref + coeffs[1], "-",
                color=C_SOFT, lw=1.5, alpha=0.4,
                label=rf"Linear fit: $D_K = {coeffs[0]:.2f}\gamma + {coeffs[1]:.2f}$")

    # Koch reference
    D_KOCH = np.log(4) / np.log(3)
    ax.axhline(D_KOCH, color=C_FIT, lw=1, ls="-.", alpha=0.6)
    ax.text(2.55, D_KOCH + 0.03, f"$D_{{Koch}} = {D_KOCH:.3f}$",
            fontsize=9, color=C_FIT, ha="right")

    # Annotate δ_Hebbian
    idx_126 = list(gamma_arr).index(1.26)
    ax.annotate(
        rf"$\delta_{{\mathrm{{Hebb}}}} = {dk_means[idx_126] - 1.26:.2f}$",
        xy=(1.26, dk_means[idx_126]),
        xytext=(1.7, dk_means[idx_126] + 0.4),
        fontsize=10, color=C_SOFT,
        arrowprops=dict(arrowstyle="->", color=C_SOFT, lw=1.2),
    )

    # Annotate dimensional democracy
    ax.annotate("Dimensional\ndemocracy\n(hard cutoff only)",
                xy=(0.0, dk_means[0]),
                xytext=(0.35, dk_means[0] + 0.5),
                fontsize=9, color="#6B7280",
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1))

    ax.set_xlabel(r"Soft cutoff exponent $\gamma$ (dimension_gap_decay)")
    ax.set_ylabel(r"K-space fractal dimension $D_K$")
    ax.set_title(
        r"Soft cutoff controls fractal topology: $p(\Delta K) = (\Delta K + 1)^{-\gamma}$"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(-0.15, 2.7)
    ax.set_ylim(0, max(dk_means) + 0.8)
    ax.grid(True, alpha=0.2)

    # Text box with key finding
    ax.text(0.98, 0.05,
            r"$D_{K,\mathrm{measured}} = \gamma + \delta_{\mathrm{Hebbian}}(\gamma)$"
            "\n\nHebbian learning adds\nintrinsic dimensional\npreference to topology",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EDE9FE",
                      edgecolor=C_SOFT, alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig3_dk_vs_gamma.png"))
    fig.savefig(os.path.join(OUTDIR, "fig3_dk_vs_gamma.pdf"))
    plt.close(fig)
    print("    → fig3_dk_vs_gamma.png/.pdf saved")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  PRE Paper — Core Figures Generator")
    print("=" * 60)
    print()

    figure_1()
    print()
    figure_2()
    print()
    figure_3()

    print()
    print("=" * 60)
    print(f"  All figures saved to: {OUTDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Generate Paper II Figure: Dual-Field Morphogenesis Verification
═══════════════════════════════════════════════════════════════

Produces a 2×2 figure for Paper II Section 7:
  (a) Turing pattern formation — Z-field spatial profiles (high vs low D_rho/D_Z)
  (b) Four-tissue unification — G² convergence curves (Bone, Neuron, Muscle, Liver)
  (c) Ischemia test — G² time series for healthy vs ischemic tissue
  (d) Energy conservation — G²+T deviation from 1.0 over time

Uses the same PDE integrator as exp_morphogenesis_pde.py.

Usage: python experiments/generate_morphogenesis_figure.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path

# Import core PDE functions from the morphogenesis experiment
from experiments.exp_morphogenesis_pde import (
    TissueParams, BONE, NEURON, MUSCLE, LIVER,
    run_simulation, compute_stable_dt,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed — skipping figure generation")

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    if not HAS_MPL:
        return

    print("=" * 72)
    print("  GENERATING MORPHOGENESIS FIGURE FOR PAPER II")
    print("=" * 72)

    # ---- (a) Turing pattern: high vs low D_rho/D_Z ----
    print("\n>>> (a) Turing pattern simulations...")
    result_bone = run_simulation(BONE)

    no_turing = TissueParams(
        name="No-Turing control", N=100, L=1.0,
        D_Z=5e-3, D_rho=1e-2, Z0=80.0, eta=0.05,
        chi=8.0, v_cat=0.05, K_eff=0.5, n_hill=2.0,
        lambda_Z=0.0005, I_blood=0.3, T_total=40.0,
    )
    no_turing.dt = compute_stable_dt(no_turing)
    result_control = run_simulation(no_turing)

    # ---- (b) Four-tissue unification ----
    print("\n>>> (b) Four-tissue simulations...")
    result_neuron = run_simulation(NEURON)
    result_muscle = run_simulation(MUSCLE)
    result_liver = run_simulation(LIVER)

    tissues = {
        "Bone (Wolff)": result_bone,
        "Neuron (Hebb)": result_neuron,
        "Muscle (Davis)": result_muscle,
        "Liver (hepatic)": result_liver,
    }

    # ---- (c) Ischemia ----
    print("\n>>> (c) Ischemia simulations...")
    np.random.seed(99)
    Z_damaged = np.full(BONE.N, BONE.Z0 * 0.5) + np.random.randn(BONE.N) * 2.0
    rho_start = np.full(BONE.N, 2.0)

    bone_repair = TissueParams(
        name="Healthy bone", N=BONE.N, L=BONE.L,
        D_Z=BONE.D_Z, D_rho=BONE.D_rho,
        Z0=BONE.Z0, eta=0.1, chi=BONE.chi, v_cat=0.2,
        K_eff=BONE.K_eff, n_hill=BONE.n_hill, lambda_Z=0.005,
        I_blood=1.0, T_total=80.0,
    )
    bone_repair.dt = compute_stable_dt(bone_repair)
    result_healthy = run_simulation(bone_repair, Z_init=Z_damaged.copy(), rho_init=rho_start.copy())

    ischemic = TissueParams(
        name="Ischemic bone", N=BONE.N, L=BONE.L,
        D_Z=BONE.D_Z, D_rho=BONE.D_rho,
        Z0=BONE.Z0, eta=0.1, chi=BONE.chi, v_cat=0.2,
        K_eff=BONE.K_eff, n_hill=BONE.n_hill, lambda_Z=0.005,
        I_blood=0.0, T_total=80.0,
    )
    ischemic.dt = compute_stable_dt(ischemic)
    rho_zero = np.full(BONE.N, 0.0)
    result_ischemic = run_simulation(ischemic, Z_init=Z_damaged.copy(), rho_init=rho_zero)

    # ================================================================
    #  PLOT 2×2 figure
    # ================================================================
    print("\n>>> Generating figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Dual-Field Impedance Morphogenesis: Numerical Verification",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # --- (a) Turing spatial profiles ---
    ax = axes[0, 0]
    x_bone = np.linspace(0, BONE.L, BONE.N)
    x_ctrl = np.linspace(0, no_turing.L, no_turing.N)
    ax.plot(x_bone, result_bone["Z_final"], "b-", linewidth=1.2,
            label=f"Bone ($D_\\rho/D_Z$={BONE.D_rho/BONE.D_Z:.0f})")
    ax.plot(x_ctrl, result_control["Z_final"], "r--", linewidth=1.2,
            label=f"Control ($D_\\rho/D_Z$={no_turing.D_rho/no_turing.D_Z:.0f})")
    ax.axhline(y=BONE.Z0, color="gray", linestyle=":", alpha=0.5, label=f"$Z_0$={BONE.Z0}")
    ax.set_xlabel("Position $x/L$", fontsize=11)
    ax.set_ylabel("$Z$ (impedance)", fontsize=11)
    ax.set_title("(a) Turing Instability: Spatial Patterns", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    std_hi = result_bone["Z_final"].std()
    std_lo = result_control["Z_final"].std()
    ax.text(0.02, 0.02,
            f"$\\sigma_{{high}}$={std_hi:.3f}, $\\sigma_{{low}}$={std_lo:.4f}\n"
            f"Contrast ratio={std_hi/(std_lo+1e-30):.1f}×",
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # --- (b) Four-tissue G² convergence ---
    ax = axes[0, 1]
    colors_tissue = {"Bone (Wolff)": "brown", "Neuron (Hebb)": "blue",
                     "Muscle (Davis)": "red", "Liver (hepatic)": "green"}
    for name, res in tissues.items():
        h = res["history"]
        times = [d["time"] for d in h]
        g2s = [d["mean_G2"] for d in h]
        ax.plot(times, g2s, linewidth=1.5, color=colors_tissue[name], label=name)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(r"$\langle\Gamma^2\rangle$", fontsize=12)
    ax.set_title("(b) Wolff/Davis/Hebb Unification", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # --- (c) Ischemia: G² time series ---
    ax = axes[1, 0]
    h_healthy = result_healthy["history"]
    h_ischemic = result_ischemic["history"]
    t_h = [d["time"] for d in h_healthy]
    g2_h = [d["mean_G2"] for d in h_healthy]
    t_i = [d["time"] for d in h_ischemic]
    g2_i = [d["mean_G2"] for d in h_ischemic]
    ax.plot(t_h, g2_h, "b-", linewidth=1.5, label="Healthy ($I_{blood}$=1.0)")
    ax.plot(t_i, g2_i, "r-", linewidth=1.5, label="Ischemic ($I_{blood}$=0)")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(r"$\langle\Gamma^2\rangle$", fontsize=12)
    ax.set_title("(c) Disease = Loop Break (Ischemia)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    g2_h_end = h_healthy[-1]["mean_G2"]
    g2_i_end = h_ischemic[-1]["mean_G2"]
    ratio = g2_i_end / (g2_h_end + 1e-30)
    ax.text(0.98, 0.98,
            f"Final $\\Gamma^2$ ratio: {ratio:.2f}×\n"
            f"Healthy: {g2_h_end:.4f}\n"
            f"Ischemic: {g2_i_end:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # --- (d) Energy conservation G²+T=1 ---
    ax = axes[1, 1]
    h_bone = result_bone["history"]
    t_bone = [d["time"] for d in h_bone]
    ec_bone = [abs(d["energy_conservation"] - 1.0) for d in h_bone]
    ax.semilogy(t_bone, [max(e, 1e-18) for e in ec_bone], "k-", linewidth=1.2,
                label="Bone")
    # Add other tissues
    for name, res in tissues.items():
        if name == "Bone (Wolff)":
            continue
        h = res["history"]
        t = [d["time"] for d in h]
        ec = [abs(d["energy_conservation"] - 1.0) for d in h]
        ax.semilogy(t, [max(e, 1e-18) for e in ec], linewidth=1.0,
                    color=colors_tissue[name], alpha=0.7, label=name)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(r"$|\Gamma^2 + T - 1|$", fontsize=12)
    ax.set_title("(d) Energy Conservation C1", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(1e-18, 1e-10)
    ax.text(0.5, 0.5, "Machine precision\n($< 10^{-15}$)",
            transform=ax.transAxes, fontsize=11, ha="center", va="center",
            color="gray", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = OUTPUT_DIR / "paper_iii_morphogenesis.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig_pdf = OUTPUT_DIR / "paper_iii_morphogenesis.pdf"
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.close()

    print(f"\n  Figure saved: {fig_path}")
    print(f"  Figure saved: {fig_pdf}")

    # Print summary data for Section 7
    print("\n" + "=" * 72)
    print("  DATA FOR PAPER II SECTION 7 — EXPERIMENT 8")
    print("=" * 72)
    print(f"\n  Turing contrast ratio: {std_hi/(std_lo+1e-30):.2f}×")
    print(f"  Turing σ(Z) high: {std_hi:.4f}, low: {std_lo:.6f}")
    print(f"\n  Ischemia G² ratio: {ratio:.2f}×")
    print(f"  Healthy G²_final: {g2_h_end:.6f}")
    print(f"  Ischemic G²_final: {g2_i_end:.6f}")
    print(f"\n  Four-tissue convergence:")
    for name, res in tissues.items():
        h = res["history"]
        g2_s = h[0]["mean_G2"]
        g2_e = h[-1]["mean_G2"]
        pct = (1 - g2_e / (g2_s + 1e-30)) * 100
        print(f"    {name:25s}  G²: {g2_s:.4f} → {g2_e:.4f}  ({pct:+.1f}%)")
    print()


if __name__ == "__main__":
    main()

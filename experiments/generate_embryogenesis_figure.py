# -*- coding: utf-8 -*-
"""
Generate 2×2 embryogenesis figure for Paper 2 Experiment 9.

Panels:
  (a) No mother vs With mother: Z/Z₀ over time
  (b) ρ field: material accumulation with/without mother
  (c) Birth transition: full-term source switch
  (d) Premature vs full-term: Γ² comparison
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.exp_embryogenesis import (
    make_embryo_params, run_embryo_simulation,
    birth_blood_supply, premature_blood_supply,
)


def main():
    print("Generating embryogenesis figure for Paper 2...")

    # ---- V7a: No mother ----
    p_orphan = make_embryo_params(name="No mother", I_blood=0.0, T_total=100.0)
    res_orphan = run_embryo_simulation(p_orphan)

    # ---- V7b: With mother ----
    p_womb = make_embryo_params(name="With mother", I_blood=0.8, T_total=100.0)
    res_womb = run_embryo_simulation(p_womb)

    # ---- V7c: Full-term birth ----
    p_birth = make_embryo_params(name="Full-term", I_blood=0.8, T_total=150.0)

    def fullterm_fn(step, n_steps, p):
        return birth_blood_supply(step, n_steps, p,
                                   birth_fraction=0.7, self_supply_ratio=0.6)

    res_birth = run_embryo_simulation(p_birth, blood_fn=fullterm_fn)

    # ---- V7d: Premature birth ----
    p_premature = make_embryo_params(name="Premature", I_blood=0.8, T_total=150.0)

    def premature_fn(step, n_steps, p):
        return premature_blood_supply(step, n_steps, p, cutoff_fraction=0.3)

    res_premature = run_embryo_simulation(p_premature, blood_fn=premature_fn)

    # ---- Extract timelines ----
    def get_timeline(res):
        h = res["history"]
        t = [d["time"] for d in h]
        z_ratio = [d["mean_Z"] / res["params"].Z0 for d in h]
        rho = [d["mean_rho"] for d in h]
        g2 = [d["mean_G2"] for d in h]
        return t, z_ratio, rho, g2

    t_o, zr_o, rho_o, g2_o = get_timeline(res_orphan)
    t_w, zr_w, rho_w, g2_w = get_timeline(res_womb)
    t_b, zr_b, rho_b, g2_b = get_timeline(res_birth)
    t_p, zr_p, rho_p, g2_p = get_timeline(res_premature)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Experiment 9: Embryogenesis Verification", fontsize=14, fontweight="bold")

    # (a) Z/Z₀ — no mother vs with mother
    ax = axes[0, 0]
    ax.plot(t_o, zr_o, "r--", linewidth=2, label="No mother ($I_{blood}=0$)")
    ax.plot(t_w, zr_w, "b-", linewidth=2, label="With mother ($I_{blood}=0.8$)")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="$Z_0$ (DNA target)")
    ax.set_xlabel("Time")
    ax.set_ylabel("$Z / Z_0$")
    ax.set_title("(a) Material bottleneck: $Z/Z_0$")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # (b) ρ field — material accumulation
    ax = axes[0, 1]
    ax.plot(t_o, rho_o, "r--", linewidth=2, label="No mother")
    ax.plot(t_w, rho_w, "b-", linewidth=2, label="With mother")
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\rho$ (material)")
    ax.set_title("(b) Material field $\\rho$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Birth transition: Z/Z₀
    ax = axes[1, 0]
    ax.plot(t_b, zr_b, "g-", linewidth=2, label="Full-term (switch @70%)")
    ax.plot(t_p, zr_p, "m--", linewidth=2, label="Premature (cut @30%)")
    # Mark birth moments
    birth_time_full = 150.0 * 0.7
    birth_time_pre = 150.0 * 0.3
    ax.axvline(x=birth_time_full, color="g", linestyle=":", alpha=0.5)
    ax.axvline(x=birth_time_pre, color="m", linestyle=":", alpha=0.5)
    ax.annotate("Full-term\nbirth", xy=(birth_time_full, 0.55), fontsize=7,
                ha="right", color="green")
    ax.annotate("Premature\nbirth", xy=(birth_time_pre, 0.35), fontsize=7,
                ha="right", color="purple")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("$Z / Z_0$")
    ax.set_title("(c) Birth transition: $Z/Z_0$")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # (d) Γ² comparison: all four scenarios
    ax = axes[1, 1]
    ax.plot(t_o, g2_o, "r--", linewidth=2, label="No mother")
    ax.plot(t_w, g2_w, "b-", linewidth=2, label="With mother")
    ax.plot(t_b, g2_b, "g-", linewidth=1.5, label="Full-term birth")
    ax.plot(t_p, g2_p, "m--", linewidth=1.5, label="Premature birth")
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\Gamma^2$")
    ax.set_title("(d) Impedance mismatch $\\Gamma^2$")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "paper_iii_embryogenesis.png")
    pdf_path = os.path.join(out_dir, "paper_iii_embryogenesis.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

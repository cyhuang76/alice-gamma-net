#!/usr/bin/env python3
"""
generate_paper4_figures.py
==========================
Paper 4 figures: bifurcation/hysteresis diagram and multi-organ cascade network.

All dynamics derive from C1/C2/C3 (Paper 0) — no disease-specific code.

Outputs:
    figures/fig_p4_bifurcation.pdf    — Saddle-node bifurcation with hysteresis
    figures/fig_p4_cascade_network.pdf — 5-organ cascade propagation
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
#  Figure 1: Bifurcation / hysteresis diagram
# ============================================================
def generate_bifurcation_figure():
    """
    Saddle-node bifurcation diagram for a single organ channel.

    C2 repair:  dG2/dt = -eta * G2  (C2 drives impedance toward match)
    Damage:     dG2/dt = +delta * s  (external stress)
    Feedback:   dG2/dt = +beta * G2^2  (positive feedback, Paper 4 Eq.)

    Steady state: eta * G2 = delta * s + beta * G2^2
    => beta * G2^2 - eta * G2 + delta * s = 0

    This is a quadratic with two real roots when eta^2 - 4*beta*delta*s > 0.
    The saddle-node bifurcation occurs at s_crit = eta^2 / (4*beta*delta).
    Hysteresis: s_onset != s_recovery.
    """
    eta = 0.5      # C2 repair rate
    beta = 0.025   # positive feedback strength
    delta = 0.005  # stress coupling

    s_vals = np.linspace(0, 800, 2000)

    # Compute steady states: beta*G2^2 - eta*G2 + delta*s = 0
    # G2 = (eta +/- sqrt(eta^2 - 4*beta*delta*s)) / (2*beta)
    s_crit = eta**2 / (4 * beta * delta)

    # Forward (increasing stress) and backward (decreasing) branches
    G2_low = np.full_like(s_vals, np.nan)
    G2_high = np.full_like(s_vals, np.nan)
    G2_unstable = np.full_like(s_vals, np.nan)

    for i, s in enumerate(s_vals):
        disc = eta**2 - 4 * beta * delta * s
        if disc >= 0:
            root_minus = (eta - np.sqrt(disc)) / (2 * beta)
            root_plus = (eta + np.sqrt(disc)) / (2 * beta)
            if 0 <= root_minus <= 1:
                G2_low[i] = root_minus
            if 0 <= root_plus <= 1:
                G2_high[i] = root_plus
            # The unstable branch is the upper root below bifurcation
            G2_unstable[i] = root_plus

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), dpi=300)

    # Stable low branch (healthy)
    ax.plot(s_vals, G2_low, color="#388e3c", linewidth=2.5,
            label=r"Stable (healthy)", zorder=3)

    # Stable high branch (diseased)
    mask_high = ~np.isnan(G2_high) & (G2_high > 0.3)
    ax.plot(s_vals[mask_high], G2_high[mask_high], color="#d32f2f",
            linewidth=2.5, label=r"Stable (diseased)", zorder=3)

    # Unstable branch (dashed)
    # Connect the two stable branches via the unstable saddle
    mask_unstable = ~np.isnan(G2_unstable) & (G2_unstable > 0.05) & (G2_unstable < 0.95)
    valid_unstable = G2_unstable.copy()
    valid_unstable[~mask_unstable] = np.nan
    # Only show the relevant part between the two branches
    for i in range(len(valid_unstable)):
        if not np.isnan(valid_unstable[i]) and not np.isnan(G2_low[i]):
            if abs(valid_unstable[i] - G2_low[i]) < 0.05:
                valid_unstable[i] = np.nan
    ax.plot(s_vals, valid_unstable, "k--", linewidth=1.2, alpha=0.5,
            label="Unstable saddle")

    # Mark bifurcation point
    G2_bif = eta / (2 * beta)
    ax.plot(s_crit, G2_bif, "ko", markersize=10, zorder=5)
    ax.annotate(f"Saddle-node\n$s_{{crit}} = {s_crit:.0f}$",
                xy=(s_crit, G2_bif), xytext=(s_crit + 80, G2_bif - 0.15),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # Hysteresis arrows
    s_onset = s_crit * 0.95
    s_recovery = s_crit * 0.55
    y_arrow = 0.08
    ax.annotate("", xy=(s_onset, y_arrow), xytext=(s_recovery, y_arrow),
                arrowprops=dict(arrowstyle="<->", color="#1565C0", lw=2))
    ax.text((s_onset + s_recovery) / 2, y_arrow + 0.04,
            "Hysteresis gap\n" + r"$s_{\rm recovery} \ll s_{\rm onset}$",
            ha="center", fontsize=8, color="#1565C0", fontweight="bold")

    # Label the gap
    ax.axvline(s_onset, color="#d32f2f", linestyle=":", alpha=0.5)
    ax.axvline(s_recovery, color="#388e3c", linestyle=":", alpha=0.5)
    ax.text(s_onset, 1.02, r"$s_{\rm onset}$", ha="center", fontsize=8,
            color="#d32f2f", fontweight="bold")
    ax.text(s_recovery, 1.02, r"$s_{\rm rec}$", ha="center", fontsize=8,
            color="#388e3c", fontweight="bold")

    ax.set_xlabel("Stress intensity $s$", fontsize=11)
    ax.set_ylabel(r"Steady-state $\Gamma^2$", fontsize=11)
    ax.set_title(
        r"Bifurcation Diagram: Why Removing Stress Is Not Enough"
        "\n" + r"$\beta\,\Gamma^4$ positive feedback $\Rightarrow$ hysteresis",
        fontsize=10
    )
    ax.set_xlim(0, 800)
    ax.set_ylim(-0.02, 1.1)
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_p4_bifurcation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ============================================================
#  Figure 2: Multi-organ cascade network propagation
# ============================================================
def generate_cascade_network_figure():
    """
    Visualize the 5-organ coupled cascade:
    initial local failure -> multi-organ propagation.

    Uses simplified dynamics from exp_disease_progression.py.
    Shows 4 time snapshots of the network.
    """
    # 5 organs with positions (circular layout)
    organs = ["Neural", "Vascular", "Immune", "Metabolic", "Renal"]
    colors_base = ["#2166ac", "#b2182b", "#4dac26", "#984ea3", "#ff7f00"]

    n_organs = 5
    theta = np.linspace(0, 2 * np.pi, n_organs, endpoint=False) - np.pi / 2
    ox = 1.5 * np.cos(theta)
    oy = 1.5 * np.sin(theta)

    # Coupling matrix (from exp_disease_progression.py)
    COUPLING = np.array([
        [0.000, 0.025, 0.000, 0.008, 0.000],
        [0.020, 0.000, 0.000, 0.012, 0.006],
        [0.008, 0.025, 0.000, 0.010, 0.000],
        [0.010, 0.020, 0.008, 0.000, 0.000],
        [0.008, 0.030, 0.000, 0.015, 0.000],
    ])

    # Simulate simple cascade: vascular insult at t=0
    n_steps = 200
    eta = 0.02
    G2 = np.zeros((n_steps, n_organs))
    G2[0] = [0.05, 0.70, 0.05, 0.05, 0.05]  # vascular insult

    for t in range(1, n_steps):
        for k in range(n_organs):
            coupling = sum(COUPLING[k, j] * G2[t - 1, j]
                           for j in range(n_organs) if j != k)
            feedback = 0.015 * G2[t - 1, k] ** 2
            dg2 = -eta * G2[t - 1, k] + coupling + feedback
            G2[t, k] = np.clip(G2[t - 1, k] + dg2, 0, 1)

    # 4 snapshots
    snapshots = [0, 30, 80, 180]
    titles = ["t = 0\n(Vascular insult)", "t = 30\n(Early cascade)",
              "t = 80\n(Cascade spread)", "t = 180\n(Multi-organ failure)"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=300)

    for ax_idx, (t_snap, title) in enumerate(zip(snapshots, titles)):
        ax = axes[ax_idx]
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=9, fontweight="bold")

        g2 = G2[t_snap]

        # Draw coupling edges
        for i in range(n_organs):
            for j in range(n_organs):
                if i != j and COUPLING[i, j] > 0:
                    strength = COUPLING[i, j] * g2[j]
                    if strength > 0.001:
                        alpha_e = min(strength * 20, 0.8)
                        lw = 0.5 + strength * 40
                        ax.annotate("",
                                    xy=(ox[i], oy[i]),
                                    xytext=(ox[j], oy[j]),
                                    arrowprops=dict(arrowstyle="->",
                                                    color="red",
                                                    alpha=alpha_e,
                                                    lw=lw,
                                                    shrinkA=18,
                                                    shrinkB=18))

        # Draw organ nodes
        for k in range(n_organs):
            intensity = g2[k]
            # Color: green (healthy) -> red (diseased)
            r_c = min(intensity * 2, 1.0)
            g_c = max(1 - intensity * 2, 0.0)
            node_color = (r_c, g_c, 0.0)
            node_size = 0.35 + intensity * 0.25

            circle = plt.Circle((ox[k], oy[k]), node_size,
                                facecolor=node_color, edgecolor="black",
                                linewidth=1.5, zorder=5, alpha=0.9)
            ax.add_patch(circle)

            # Label
            ax.text(ox[k], oy[k] - node_size - 0.2, organs[k],
                    ha="center", va="top", fontsize=7, fontweight="bold")

            # G2 value
            ax.text(ox[k], oy[k], f"{g2[k]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if intensity > 0.3 else "black",
                    fontweight="bold", zorder=6)

    fig.suptitle(
        r"Multi-Organ Cascade: $\Gamma_v\uparrow \to \rho\downarrow "
        r"\to \Gamma_n\uparrow \to \Gamma_v\uparrow\uparrow$"
        "\n(C3 impedance-tagged transport drives inter-organ coupling)",
        fontsize=11, fontweight="bold", y=1.02
    )

    fig.tight_layout()
    out = FIGURES_DIR / "fig_p4_cascade_network.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Paper 4 Figure Generation")
    print("=" * 60)
    generate_bifurcation_figure()
    generate_cascade_network_figure()
    print("Done.")

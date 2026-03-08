# -*- coding: utf-8 -*-
"""
Generate Paper 3 figures — Lifecycle Equation verification plots.

Figure 1: Bathtub curve (ΣΓ² vs time, with phase annotations)
Figure 2: Emotion readouts (dopamine, stress, curiosity, boredom)
Figure 3: Yerkes-Dodson inverted-U (performance vs arousal)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ==============================================================
# Lifecycle ODE solver (standalone, no alice imports needed)
# ==============================================================

def lifecycle_ode(T=5000, dt=1.0,
                  eta0=0.04, gamma0=0.20, delta0=1e-4,
                  sigma0=0.80, aging_onset=1000,
                  critical_period_end=500, childhood_end=800,
                  habituation_rate=0.002,
                  arrhenius_ea=1.5,
                  stress_profile=None):
    """Integrate the Lifecycle Equation with three forces.

    dΣΓ²/dt = −η·ΣΓ² + γ·Γ_env(t) + δ·D(t)

    The bathtub curve emerges from the relative magnitudes:
    - Learning dominates in childhood  → ΣΓ² drops
    - Forces balance in adulthood      → plateau
    - Aging dominates in senescence     → ΣΓ² rises

    D(t) follows Coffin–Manson fatigue: cumulative damage from
    mismatch-driven current (Lorentz pinch) plus baseline
    metabolic wear.
    """
    N = int(T / dt)
    t = np.arange(N) * dt

    sigma = np.zeros(N)
    sigma[0] = sigma0

    learning = np.zeros(N)
    novelty = np.zeros(N)
    aging = np.zeros(N)
    d_sigma = np.zeros(N)
    phase = [""] * N
    T_eq = np.zeros(N)

    cumulative_exposure = 0.0
    cumulative_damage = 0.0

    alpha_heat = 0.15
    beta_cool = 0.03
    T_env = 0.1

    if stress_profile is None:
        stress_profile = np.zeros(N)

    for i in range(N - 1):
        s = sigma[i]

        # --- Learning rate ---
        eta = eta0
        if i < critical_period_end:
            eta *= 2.0  # Critical period
        elif i < childhood_end:
            eta *= 1.5  # Childhood boost
        eta = np.clip(eta, 0.001, 0.10)

        # --- Novelty: γ(t)·Γ_env(t) ---
        cumulative_exposure += 0.01
        gamma_eff = max(0.005, gamma0 * np.exp(
            -cumulative_exposure * habituation_rate * 50))
        # Γ_env starts at 0.10 (infant's limited world exposure)
        # and decays with adaptation
        gamma_env = max(0.01, 0.10 * np.exp(
            -cumulative_exposure * habituation_rate * 10))
        nov = gamma_eff * gamma_env

        # --- Aging: δ_eff(t)·D(t) ---
        if i > aging_onset:
            # Coffin–Manson damage: mismatch energy + baseline wear
            cumulative_damage += (s ** 2 + 0.02) * dt
            stress = stress_profile[i]
            delta_eff = delta0 * np.exp(arrhenius_ea * stress)
            age_term = delta_eff * cumulative_damage
        else:
            age_term = 0.0

        # --- ODE step ---
        learn_term = -eta * s
        d = learn_term + nov + age_term
        sigma[i + 1] = np.clip(s + d * dt, 0.0, 1.0)

        learning[i] = learn_term
        novelty[i] = nov
        aging[i] = age_term
        d_sigma[i] = d
        T_eq[i] = T_env + (alpha_heat / beta_cool) * s

        # Phase detection — six developmental stages
        if i < 5:
            phase[i] = "birth"
        elif s > 0.4 and d < 0:
            phase[i] = "infancy"
        elif s > 0.15 and i < childhood_end and d < -0.001:
            phase[i] = "childhood"
        elif i > aging_onset and d > 0.001 and s > 0.3:
            phase[i] = "senescence"
        elif i > aging_onset and d > 0.0002:
            phase[i] = "decline"
        else:
            phase[i] = "adulthood"

    # Fill last values
    learning[-1] = learning[-2]
    novelty[-1] = novelty[-2]
    aging[-1] = aging[-2]
    d_sigma[-1] = d_sigma[-2]
    T_eq[-1] = T_eq[-2]
    phase[-1] = phase[-2]

    return {
        "t": t, "sigma": sigma,
        "learning": learning, "novelty": novelty, "aging": aging,
        "d_sigma": d_sigma, "phase": phase, "T_eq": T_eq,
    }


# ==============================================================
# Figure 1: Bathtub Curve
# ==============================================================

def plot_bathtub_curve():
    """Plot ΣΓ² lifecycle trajectory with phase coloring."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})

    data = lifecycle_ode(T=5000)
    t = data["t"]
    sigma = data["sigma"]

    # Panel (a): ΣΓ² with phase coloring
    ax = axes[0]
    phase_colors = {
        "birth": "#FF6B6B",
        "infancy": "#FFA07A",
        "childhood": "#FFD700",
        "adulthood": "#4CAF50",
        "decline": "#FF9800",
        "senescence": "#F44336",
    }
    phase_labels_done = set()
    phases = data["phase"]
    for i in range(len(t) - 1):
        p = phases[i]
        c = phase_colors.get(p, "#999999")
        label = p.capitalize() if p not in phase_labels_done else None
        if label:
            phase_labels_done.add(p)
        ax.fill_between([t[i], t[i+1]], 0, 1, color=c, alpha=0.15)

    ax.plot(t, sigma, "k-", linewidth=1.5, label=r"$\Sigma\Gamma^2$")
    ax.plot(t, 1 - sigma, "b--", linewidth=1.0, alpha=0.6, label=r"$\Sigma T = 1 - \Sigma\Gamma^2$")
    ax.set_ylabel(r"$\Sigma\Gamma^2$", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("(a) Lifecycle Bathtub Curve", fontsize=11, fontweight="bold")

    # Phase annotations
    phase_positions = {}
    for i, p in enumerate(phases):
        if p not in phase_positions:
            phase_positions[p] = []
        phase_positions[p].append(i)
    for p, indices in phase_positions.items():
        mid = indices[len(indices) // 2]
        ax.annotate(p.capitalize(), xy=(t[mid], 0.95), fontsize=8,
                    ha="center", color=phase_colors.get(p, "#999"),
                    fontweight="bold", alpha=0.8)

    # Panel (b): Three forces
    ax2 = axes[1]
    ax2.plot(t, data["learning"], "g-", linewidth=1.0, label="Learning $(-\\eta\\Sigma\\Gamma^2)$")
    ax2.plot(t, data["novelty"], "orange", linewidth=1.0, label="Novelty $(\\gamma\\Gamma_{env})$")
    ax2.plot(t, data["aging"], "r-", linewidth=1.0, label="Aging $(\\delta D)$")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Force magnitude", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title("(b) Three Competing Forces", fontsize=11, fontweight="bold")

    # Panel (c): Equilibrium temperature
    ax3 = axes[2]
    ax3.plot(t, data["T_eq"], "r-", linewidth=1.0)
    ax3.set_ylabel(r"$T_{\mathrm{steady}}$", fontsize=10)
    ax3.set_xlabel("Time (ticks = developmental age)", fontsize=10)
    ax3.set_title("(c) Equilibrium Temperature", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig("figures/paper_iv_bathtub.png", dpi=200, bbox_inches="tight")
    fig.savefig("figures/paper_iv_bathtub.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] Figure 1: Bathtub curve saved")


# ==============================================================
# Figure 2: Emotion Readouts
# ==============================================================

def plot_emotion_readouts():
    """Plot dopamine, stress, curiosity, boredom from Γ² dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    data = lifecycle_ode(T=5000)
    t = data["t"]
    sigma = data["sigma"]

    # --- Dopamine = -ΔΓ²/Δt (clipped positive) ---
    ax = axes[0, 0]
    delta_sigma = -np.gradient(sigma)
    dopamine = np.clip(delta_sigma, 0, None)
    # Smooth for visibility
    kernel = np.ones(50) / 50
    dopamine_smooth = np.convolve(dopamine, kernel, mode="same")
    ax.plot(t, dopamine_smooth, "purple", linewidth=1.0)
    ax.fill_between(t, 0, dopamine_smooth, alpha=0.3, color="purple")
    ax.set_ylabel("Dopamine signal", fontsize=10)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_title("(a) Reward: $-\\Delta\\Gamma^2/\\Delta t$", fontsize=11, fontweight="bold")

    # --- Stress = slow-averaged ΣΓ² ---
    ax = axes[0, 1]
    tau_s = 200  # slow integration window
    stress_kernel = np.ones(tau_s) / tau_s
    stress = np.convolve(sigma, stress_kernel, mode="same")
    ax.plot(t, stress, "red", linewidth=1.0)
    ax.fill_between(t, 0, stress, alpha=0.2, color="red")
    ax.set_ylabel("Cortisol (stress)", fontsize=10)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_title(r"(b) Stress: $\langle\Sigma\Gamma^2\rangle_{\tau_s}$",
                 fontsize=11, fontweight="bold")

    # --- Curiosity = gradient steep + safe ---
    ax = axes[1, 0]
    # Curiosity ~ novelty force × (1 - sigma) [gated by safety]
    curiosity = data["novelty"] * (1 - sigma)
    curiosity_smooth = np.convolve(curiosity, kernel, mode="same")
    ax.plot(t, curiosity_smooth, "teal", linewidth=1.0)
    ax.fill_between(t, 0, curiosity_smooth, alpha=0.3, color="teal")
    ax.set_ylabel("Curiosity", fontsize=10)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_title("(c) Curiosity: impedance gradient × safety",
                 fontsize=11, fontweight="bold")

    # --- Boredom = low derivative at high Γ² ---
    ax = axes[1, 1]
    d_dt = np.abs(np.gradient(sigma))
    d_smooth = np.convolve(d_dt, kernel, mode="same")
    boredom = np.where((d_smooth < 0.0005) & (sigma > 0.15), sigma, 0)
    boredom_smooth = np.convolve(boredom, np.ones(100)/100, mode="same")
    ax.plot(t, boredom_smooth, "brown", linewidth=1.0)
    ax.fill_between(t, 0, boredom_smooth, alpha=0.3, color="brown")
    ax.set_ylabel("Boredom", fontsize=10)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_title(r"(d) Boredom: $|d\Gamma^2/dt| < \epsilon$ at high $\Gamma^2$",
                 fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig("figures/paper_iv_emotions.png", dpi=200, bbox_inches="tight")
    fig.savefig("figures/paper_iv_emotions.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] Figure 2: Emotion readouts saved")


# ==============================================================
# Figure 3: Yerkes-Dodson Inverted-U
# ==============================================================

def plot_yerkes_dodson():
    """Plot performance vs arousal showing inverted-U curve."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    arousal_levels = np.linspace(0.01, 1.0, 30)
    gamma_env = 0.5  # Fixed task difficulty

    performance = []
    final_sigma = []

    for a in arousal_levels:
        # Arousal modulates both η and γ
        eta_mod = 0.04 * (0.5 + a)  # Linear increase
        gamma_mod = 0.20 * (0.3 + a ** 1.5)  # Superlinear increase at high arousal

        # Simple 100-tick learning block
        s = 0.5  # Start at moderate mismatch
        for _ in range(100):
            learn = -eta_mod * s
            nov = gamma_mod * gamma_env * 0.1
            s = np.clip(s + learn + nov, 0, 1)
        performance.append(1 - s)
        final_sigma.append(s)

    performance = np.array(performance)
    final_sigma = np.array(final_sigma)

    # Panel (a): Performance vs arousal
    ax = axes[0]
    ax.plot(arousal_levels, performance, "b-o", markersize=4, linewidth=1.5)
    peak_idx = np.argmax(performance)
    ax.axvline(arousal_levels[peak_idx], color="red", linestyle="--", alpha=0.5)
    ax.annotate(f"Optimal: {arousal_levels[peak_idx]:.2f}",
                xy=(arousal_levels[peak_idx], performance[peak_idx]),
                xytext=(arousal_levels[peak_idx] + 0.15, performance[peak_idx] - 0.05),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red")
    ax.set_xlabel("Arousal level", fontsize=11)
    ax.set_ylabel(r"Performance $(1 - \Sigma\Gamma^2)$", fontsize=11)
    ax.set_title("(a) Yerkes-Dodson Inverted-U", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)

    # Panel (b): Stress-accelerated aging
    ax2 = axes[1]
    stress_levels = [0.0, 0.3, 0.7]
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    labels = ["No stress", "Moderate stress", "High stress"]

    for sl, c, lab in zip(stress_levels, colors, labels):
        stress_prof = np.full(5000, sl)
        data = lifecycle_ode(T=5000, stress_profile=stress_prof)
        ax2.plot(data["t"], data["sigma"], color=c, linewidth=1.5, label=lab)

    ax2.set_xlabel("Time (developmental age)", fontsize=11)
    ax2.set_ylabel(r"$\Sigma\Gamma^2$", fontsize=11)
    ax2.set_title("(b) Stress-Accelerated Aging", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig("figures/paper_iv_yerkes_stress.png", dpi=200, bbox_inches="tight")
    fig.savefig("figures/paper_iv_yerkes_stress.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] Figure 3: Yerkes-Dodson & stress-aging saved")


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Paper 3 Figure Generation")
    print("  The Lifecycle Equation")
    print("=" * 60)

    plot_bathtub_curve()
    plot_emotion_readouts()
    plot_yerkes_dodson()

    print("\n" + "=" * 60)
    print("  All Paper 3 figures generated successfully")
    print("=" * 60)

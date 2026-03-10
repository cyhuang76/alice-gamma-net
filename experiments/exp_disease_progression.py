#!/usr/bin/env python3
"""
exp_disease_progression.py
==========================
Γ-Net disease progression map: from trauma/empathic load through
dual-network cascade to multi-organ disease.

Model
-----
Five coupled organ Γ-channels evolve under the lifecycle equation
with stress events and inter-organ cascade coupling (Paper 2):

    dΓ_k²/dt = -η(t)·Γ_k² + δ_k·D(t) + coupling_from_others

Channels:
    Neural (Γ_n), Vascular (Γ_v), Immune (Γ_imm),
    Metabolic (Γ_met), Renal (Γ_ren)

Disease branch points are defined by threshold crossings:
    Depression:        Γ_n > θ_dep   and  dΓ_n/dt ≈ 0
    Hypertension/CVD:  Γ_v > θ_cvd
    Metabolic syndrome: Γ_met > θ_met
    CKD:               Γ_ren > θ_ren
    Dementia:          Γ_n > θ_dem  (late, high threshold)

Output
------
figures/fig_disease_progression.pdf — 3-panel figure:
    Top:    η(t) with stress event markers
    Middle: 5 organ Γ² trajectories with disease threshold crossings
    Bottom: H(t) = Π(1 - Γ_k²) total health product
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ──────────────────────────────────────────
#  η(t) lifecycle (reuse from exp_skill_decay_order)
# ──────────────────────────────────────────
def f_dev(age: np.ndarray) -> np.ndarray:
    t_peak = 5.0
    safe_age = np.clip(age, 0.1, None)
    ramp = np.minimum(safe_age / t_peak, 1.0)
    # 0.030 = homeostatic repair capacity (whole-body), not
    # synaptic plasticity (0.08) — body repair declines more slowly
    decline = np.exp(-0.030 * np.maximum(safe_age - t_peak, 0.0))
    return ramp * decline


def eta_lifecycle(age: np.ndarray, eta0: float = 0.50,
                  E_a_ratio: float = 0.005,
                  eta_floor: float = 0.020) -> np.ndarray:
    """η(t) with adult floor — repair capacity never fully vanishes."""
    arrhenius = np.exp(-E_a_ratio * age)
    raw = eta0 * f_dev(age) * arrhenius
    return np.maximum(raw, eta_floor)


def damage_rate(age: np.ndarray, delta0: float = 5e-4,
                E_a_ratio: float = 0.030) -> np.ndarray:
    return delta0 * np.exp(E_a_ratio * age)


# ──────────────────────────────────────────
#  Stress event profile
# ──────────────────────────────────────────
def stress_profile(age: np.ndarray) -> np.ndarray:
    """External stress Γ_env(t): baseline + life events."""
    stress = np.full_like(age, 0.01, dtype=float)

    # Childhood trauma (age 8-10): parental conflict
    mask = (age >= 8) & (age < 10)
    stress[mask] += 0.15

    # Career stress (age 30-45): chronic overwork
    mask = (age >= 30) & (age < 45)
    stress[mask] += 0.06

    # Caregiving burden (age 50-65): aging parent
    mask = (age >= 50) & (age < 65)
    stress[mask] += 0.12

    # Bereavement spike (age 62)
    mask = (age >= 62) & (age < 63)
    stress[mask] += 0.20

    return stress


# ──────────────────────────────────────────
#  Coupled 5-organ ODE
# ──────────────────────────────────────────
ORGANS = ["Neural", "Vascular", "Immune", "Metabolic", "Renal"]
COLORS = ["#2166ac", "#b2182b", "#4dac26", "#984ea3", "#ff7f00"]

# Coupling matrix: row i receives from column j
# Γ_v↑ → ρ↓ → all others ↑ (material starvation)
# Γ_n↑ → autonomic → Γ_v↑
# Γ_met↑ → insulin resistance → Γ_v↑, Γ_ren↑
COUPLING = np.array([
    #  n      v      imm    met    ren
    [0.000, 0.025, 0.000, 0.008, 0.000],  # Neural ← Vascular (ρ↓)
    [0.020, 0.000, 0.000, 0.012, 0.006],  # Vascular ← Neural (autonomic), Met
    [0.008, 0.025, 0.000, 0.010, 0.000],  # Immune ← Vascular (ρ↓), Metabolic
    [0.010, 0.020, 0.008, 0.000, 0.000],  # Metabolic ← Neural, Vascular, Imm
    [0.008, 0.030, 0.000, 0.015, 0.000],  # Renal ← Vascular (ρ↓), Metabolic
])

# Per-organ baseline damage multipliers
DAMAGE_MULT = np.array([1.0, 1.3, 0.8, 1.1, 0.9])

# Stress sensitivity per organ
STRESS_SENS = np.array([0.33, 0.20, 0.15, 0.20, 0.10])

# Positive feedback: pathological amplification
# (inflammation begets inflammation, damage begets damage)
FEEDBACK = np.array([0.020, 0.025, 0.015, 0.025, 0.015])


def simulate(ages: np.ndarray, dt: float):
    """Simulate 5-organ coupled Γ² evolution over a lifetime."""
    n = len(ages)
    n_organs = 5
    G2 = np.zeros((n, n_organs))

    # Initial conditions: infant mismatch
    G2[0] = [0.50, 0.05, 0.05, 0.05, 0.03]

    stress = stress_profile(ages)

    for i in range(1, n):
        a = ages[i]
        eta = eta_lifecycle(np.array([a]))[0]
        delta = damage_rate(np.array([a]))[0]
        s = stress[i]

        g2_prev = G2[i - 1]

        for k in range(n_organs):
            # Learning/repair
            learning = -eta * g2_prev[k]

            # Aging damage (organ-specific rate)
            aging = delta * DAMAGE_MULT[k]

            # Stress injection
            stress_term = STRESS_SENS[k] * s

            # Inter-organ coupling: other organs' Γ² drives this one up
            coupling = 0.0
            for j in range(n_organs):
                if j != k:
                    coupling += COUPLING[k, j] * g2_prev[j]

            # Positive feedback: damage-begets-damage
            pos_fb = FEEDBACK[k] * g2_prev[k] ** 2

            dg2 = learning + aging + stress_term + coupling + pos_fb
            G2[i, k] = g2_prev[k] + dg2 * dt
            G2[i, k] = np.clip(G2[i, k], 0.0, 1.0)

    return G2


# ──────────────────────────────────────────
#  Disease thresholds
# ──────────────────────────────────────────
DISEASES = {
    "Depression":      {"organ": 0, "threshold": 0.45, "color": "#2166ac"},
    "Hypertension":    {"organ": 1, "threshold": 0.50, "color": "#b2182b"},
    "Immune decline":  {"organ": 2, "threshold": 0.45, "color": "#4dac26"},
    "Metabolic synd.": {"organ": 3, "threshold": 0.48, "color": "#984ea3"},
    "CKD":             {"organ": 4, "threshold": 0.50, "color": "#ff7f00"},
    "Dementia":        {"organ": 0, "threshold": 0.70, "color": "#053061"},
}


def find_threshold_crossing(ages, g2_trace, threshold, min_age=15.0):
    """Find first age where Γ² crosses threshold (sustained)."""
    for i in range(len(ages)):
        if ages[i] < min_age:
            continue
        if g2_trace[i] >= threshold:
            # Check if sustained for at least 2 years
            end = min(i + 20, len(ages))
            if np.mean(g2_trace[i:end]) >= threshold * 0.95:
                return ages[i]
    return None


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────
def main():
    dt = 0.1
    ages = np.arange(0, 95 + dt, dt)
    G2 = simulate(ages, dt)

    # Compute H(t) = Π(1 - Γ_k²)
    H = np.prod(1.0 - G2, axis=1)

    stress = stress_profile(ages)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11),
                             gridspec_kw={"height_ratios": [1, 2.5, 1.2]})
    fig.subplots_adjust(hspace=0.30)

    # ── Panel 1: η(t) + stress events ──
    ax0 = axes[0]
    eta_vals = eta_lifecycle(ages)
    ax0.plot(ages, eta_vals, "k-", linewidth=2, label="η(t)")
    ax0.fill_between(ages, 0, stress, alpha=0.3, color="red",
                     label="Stress Γ_env(t)")
    ax0.set_ylabel("η / Stress", fontsize=11)
    ax0.set_xlim(0, 95)
    ax0.set_title("Γ-Net Disease Progression Map", fontsize=14,
                  fontweight="bold")
    ax0.legend(loc="upper right", fontsize=9)

    # Annotate stress events
    events = [
        (9, "Childhood\ntrauma"),
        (37, "Career\nstress"),
        (57, "Caregiving\nburden"),
        (62.5, "Bereavement"),
    ]
    for ev_age, ev_label in events:
        ax0.annotate(ev_label, (ev_age, stress_profile(np.array([ev_age]))[0]),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=7.5, ha="center", color="darkred",
                     arrowprops=dict(arrowstyle="->", color="darkred",
                                     lw=0.8))

    # ── Panel 2: organ Γ² trajectories ──
    ax1 = axes[1]
    for k in range(5):
        ax1.plot(ages, G2[:, k], color=COLORS[k], linewidth=2,
                 label=ORGANS[k])

    # Mark disease threshold crossings
    onset_ages = {}
    for disease, info in DISEASES.items():
        k = info["organ"]
        thresh = info["threshold"]
        onset = find_threshold_crossing(ages, G2[:, k], thresh)
        if onset is not None:
            onset_ages[disease] = onset
            ax1.axhline(thresh, color=info["color"], linestyle=":",
                        alpha=0.4, linewidth=0.8)
            ax1.plot(onset, thresh, "D", color=info["color"],
                     markersize=8, zorder=5, markeredgecolor="k",
                     markeredgewidth=0.5)
            # Position labels to avoid overlap
            y_offset = -18 if "Dementia" in disease else 10
            ax1.annotate(f"{disease}\n≈ age {onset:.0f}",
                         (onset, thresh),
                         textcoords="offset points",
                         xytext=(8, y_offset),
                         fontsize=8, color=info["color"],
                         fontweight="bold",
                         arrowprops=dict(arrowstyle="->",
                                         color=info["color"], lw=0.8))

    ax1.set_ylabel("Organ Γ²", fontsize=11)
    ax1.set_xlim(0, 95)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    ax1.set_xlabel("")

    # ── Panel 3: H(t) total health ──
    ax2 = axes[2]
    ax2.fill_between(ages, 0, H, alpha=0.3, color="forestgreen")
    ax2.plot(ages, H, color="forestgreen", linewidth=2)
    ax2.set_ylabel("H(t) = Π(1−Γ_k²)", fontsize=11)
    ax2.set_xlabel("Age (years)", fontsize=11)
    ax2.set_xlim(0, 95)
    ax2.set_ylim(0, 1.0)

    # Mark disease onsets on H curve
    for disease, onset in sorted(onset_ages.items(), key=lambda x: x[1]):
        idx = int(onset / dt)
        if idx < len(H):
            ax2.axvline(onset, color="gray", linestyle="--",
                        linewidth=0.5, alpha=0.5)

    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "fig_disease_progression.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")

    # Print disease onset summary
    print("\n=== Disease Onset Summary ===")
    for disease, onset in sorted(onset_ages.items(), key=lambda x: x[1]):
        print(f"  {disease:20s}  onset ≈ age {onset:.0f}")

    print(f"\nH(age 30) = {H[int(30/dt)]:.3f}")
    print(f"H(age 50) = {H[int(50/dt)]:.3f}")
    print(f"H(age 70) = {H[int(70/dt)]:.3f}")
    print(f"H(age 90) = {H[int(90/dt)]:.3f}")


if __name__ == "__main__":
    main()

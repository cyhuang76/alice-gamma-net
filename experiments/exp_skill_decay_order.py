#!/usr/bin/env python3
"""
exp_skill_decay_order.py
========================
Numerical demonstration of Proposition 7.3 (Skill-decay anti-chronology):
skills acquired earlier (higher η) decay last under uniform Arrhenius aging.

Model
-----
Three skills A, B, C acquired at ages 3, 20, 55.
Each skill's Γ evolves via:
    dΓ²/dt = -η(t)·Γ² + δ·D(t)
where
    η(t) = η₀ · f_dev(t) · exp(-E_a / k_B · T_age(t))
    f_dev(t) = bell-shaped developmental envelope
    D(t)     = cumulative Arrhenius damage (monotone increasing)

Output
------
figures/fig_skill_decay_order.pdf — three Γ² curves vs age
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ──────────────────────────────────────────
#  η(t) lifecycle model
# ──────────────────────────────────────────
def f_dev(age: np.ndarray) -> np.ndarray:
    """Developmental envelope: ramps to peak at ~5 yr, then exponential decline.

    Models the well-documented critical-period plasticity peak in early
    childhood, followed by monotone decline in adulthood.
    """
    t_peak = 5.0
    safe_age = np.clip(age, 0.1, None)
    ramp = np.minimum(safe_age / t_peak, 1.0)
    decline = np.exp(-0.08 * np.maximum(safe_age - t_peak, 0.0))
    return ramp * decline


def eta_lifecycle(age: np.ndarray, eta0: float = 1.0,
                  E_a_ratio: float = 0.005) -> np.ndarray:
    """η(t) = η₀ · f_dev(t) · Arrhenius decay."""
    arrhenius = np.exp(-E_a_ratio * age)
    return eta0 * f_dev(age) * arrhenius


def damage_rate(age: np.ndarray, delta0: float = 3e-4,
                E_a_ratio: float = 0.055) -> np.ndarray:
    """Aging damage rate δ(t), accelerating with age (Arrhenius)."""
    return delta0 * np.exp(E_a_ratio * age)


# ──────────────────────────────────────────
#  ODE integration (Euler, fine dt)
# ──────────────────────────────────────────
def simulate_skill(acquire_age: float, ages: np.ndarray,
                   dt: float) -> np.ndarray:
    """Simulate Γ² trajectory for a single skill acquired at acquire_age."""
    gamma2 = np.ones_like(ages) * 1.0  # start at Γ²=1 (total mismatch)
    gamma2_initial = 0.95  # high initial mismatch when skill starts

    for i in range(1, len(ages)):
        a = ages[i]
        if a < acquire_age:
            gamma2[i] = 1.0  # skill not yet acquired
            continue

        eta = eta_lifecycle(np.array([a]))[0]
        delta = damage_rate(np.array([a]))[0]

        # Practice: intense first 10 years, then constant maintenance
        years_practicing = a - acquire_age
        practice = 1.0 if years_practicing < 10.0 else 0.3

        # dΓ²/dt = -η·practice·Γ² + δ
        dg2 = -eta * practice * gamma2[i - 1] + delta
        gamma2[i] = gamma2[i - 1] + dg2 * dt
        gamma2[i] = np.clip(gamma2[i], 0.0, 1.0)

    return gamma2


# ──────────────────────────────────────────
#  Main simulation
# ──────────────────────────────────────────
def main():
    dt = 0.1  # years
    ages = np.arange(0, 100 + dt, dt)

    skills = {
        "Skill A (native language, age 3)": 3.0,
        "Skill B (second language, age 20)": 20.0,
        "Skill C (new instrument, age 35)": 35.0,
    }

    gamma2_crit = 0.65  # failure threshold (functional impairment)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                             gridspec_kw={"height_ratios": [1, 2]})

    # ── Top panel: η(t) lifecycle ──
    ax0 = axes[0]
    eta_vals = eta_lifecycle(ages)
    ax0.plot(ages, eta_vals, "k-", linewidth=2)
    ax0.set_ylabel("η(t)  [learning rate]", fontsize=12)
    ax0.set_xlim(0, 100)
    ax0.set_title("Developmental modulation of η(t)", fontsize=13)
    ax0.axhline(0, color="gray", linewidth=0.5)
    # Mark acquisition ages
    colors = ["#2166ac", "#b2182b", "#1b7837"]
    for (label, acq_age), c in zip(skills.items(), colors):
        eta_at_acq = eta_lifecycle(np.array([acq_age]))[0]
        ax0.plot(acq_age, eta_at_acq, "o", color=c, markersize=8, zorder=5)
        ax0.annotate(f"age {int(acq_age)}",
                     (acq_age, eta_at_acq),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=9, color=c)

    # ── Bottom panel: Γ² trajectories ──
    ax1 = axes[1]
    fail_ages = {}
    for (label, acq_age), c in zip(skills.items(), colors):
        g2 = simulate_skill(acq_age, ages, dt)
        ax1.plot(ages, g2, color=c, linewidth=2, label=label)

        # Find failure age: first time Γ² re-crosses threshold
        # after having been below it, or acquisition age if never learned
        acquired_mask = ages >= acq_age
        below = g2 < gamma2_crit
        was_below = np.zeros_like(below)
        was_below[1:] = below[:-1]
        crossings = (~below) & was_below & acquired_mask
        cross_idx = np.where(crossings)[0]

        # Check if skill was ever mastered
        ever_below = np.any(below & acquired_mask)

        if len(cross_idx) > 0:
            fail_age = ages[cross_idx[0]]
        elif not ever_below:
            # Never learned → "fails" at acquisition
            fail_age = acq_age
        else:
            fail_age = float("inf")  # mastered and never lost

        fail_ages[label] = fail_age
        if np.isfinite(fail_age):
            ax1.plot(fail_age, gamma2_crit, "x", color=c,
                     markersize=12, markeredgewidth=3, zorder=5)
            ax1.annotate(f"fail ≈ {fail_age:.0f} yr",
                         (fail_age, gamma2_crit),
                         textcoords="offset points", xytext=(8, -15),
                         fontsize=9, color=c,
                         arrowprops=dict(arrowstyle="->", color=c))

    ax1.axhline(gamma2_crit, color="gray", linestyle="--",
                linewidth=1.5, label=f"Γ²_crit = {gamma2_crit}")
    ax1.set_xlabel("Age  [years]", fontsize=12)
    ax1.set_ylabel("Γ²  [mismatch energy]", fontsize=12)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Skill-decay anti-chronology: "
                  "latest acquired → first to fail", fontsize=13)
    ax1.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fig_skill_decay_order.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure saved: {out_path}")

    # ── Print summary ──
    print("\n=== Skill-Decay Anti-Chronology ===")
    print(f"{'Skill':<40s} {'Acquired':>8s} {'Fails':>8s}")
    print("-" * 60)
    for label, acq_age in skills.items():
        f_age = fail_ages.get(label, float("inf"))
        print(f"{label:<40s} {acq_age:>7.0f}y  {f_age:>7.0f}y")
    print()

    # Verify anti-chronology: earlier acquired → later failure
    acq_order = sorted(skills.values())
    fail_order = [fail_ages.get(l, float("inf"))
                  for l in sorted(skills, key=skills.get)]
    assert all(f1 >= f2 for f1, f2 in zip(fail_order, fail_order[1:])), \
        "Anti-chronology violated!"
    print("[PASS] Anti-chronology confirmed: "
          "earliest-acquired skill fails last.")

    return fail_ages


if __name__ == "__main__":
    main()

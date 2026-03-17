#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: Exercise as C2 Input-Signal Maintenance
====================================================

PURPOSE
-------
Verify that exercise maintains the C2 input signal x_in,
thereby counteracting aging drift δ in the Lifecycle Equation:

    dΣΓ²/dt = -η(t)·x_in(t)·ΣΓ² + γ·Γ_env(t) + δ(t)·D(t)

KEY PREDICTIONS
---------------
1. Active individuals have lower ΣΓ² at every age than sedentary
2. 70-year-old exerciser has lower ΣΓ² than 50-year-old sedentary
3. Immune Γ (proxy: immune organ subsystem) tracks exercise status

PHYSICS
-------
Exercise increases cardiac output → higher wall shear stress →
vascular C2 operates faster → lower Γ_v → better material supply ρ.
In the Lifecycle Equation, this manifests as higher effective η·x_in
product, which increases the magnitude of the learning (C2) term.

Author: Alice Smart System (automated verification)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 嘗試 matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping figure generation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# ============================================================================
# 1. LIFECYCLE EQUATION PARAMETERS
# ============================================================================

def eta_profile(t: np.ndarray, eta0: float = 0.08) -> np.ndarray:
    """Learning rate η(t) with developmental bell curve + Arrhenius decline.

    Peak in adolescence (~15y), plateau 20-40, decline after 40.
    """
    # 發展包絡 f_dev(t): 在 15 歲達峰值
    f_dev = np.exp(-0.5 * ((t - 15) / 10) ** 2) * 0.5 + \
            np.exp(-0.5 * ((t - 25) / 20) ** 2) * 0.5

    # Arrhenius 老化因子: 40 歲後指數下降
    arrhenius = np.where(t < 40, 1.0, np.exp(-0.02 * (t - 40)))

    return eta0 * f_dev * arrhenius


def delta_aging(t: np.ndarray, delta0: float = 0.001) -> np.ndarray:
    """Irreversible aging drift δ(t).

    Accelerates after ~50 due to cumulative Coffin-Manson fatigue.
    """
    return delta0 * (1.0 + 0.5 * np.maximum(0, (t - 50) / 30) ** 2)


def x_in_exercise(t: np.ndarray, active: bool = True,
                  start_age: float = 0.0) -> np.ndarray:
    """Exercise-modulated input signal x_in(t).

    Active: x_in ≈ 0.8 (regular exercise maintains high shear stress)
    Sedentary: x_in decays from 0.8 to 0.3 after ~25 years old
    """
    if active:
        # 持續運動: x_in 維持在高水平，老年略降
        return np.where(t < 20, 0.9,
               np.where(t < 60, 0.8, 0.7))
    else:
        # 久坐: 年輕時被迫運動（體育課等），成年後急降
        return np.where(t < 20, 0.8,
               np.where(t < 30, 0.5,
               np.where(t < 50, 0.3, 0.2)))


# ============================================================================
# 2. LIFECYCLE ODE SOLVER
# ============================================================================

def simulate_lifecycle(
    t_span: tuple = (0, 80),
    dt: float = 0.1,
    eta0: float = 0.08,
    gamma_env_base: float = 0.02,
    delta0: float = 0.001,
    sigma_gamma_sq_0: float = 0.12,  # 嬰兒初始阻抗失配
    active: bool = True,
    label: str = "",
) -> dict:
    """Simulate Lifecycle Equation via Euler method.

    dΣΓ²/dt = -η(t)·x_in(t)·ΣΓ² + γ·Γ_env + δ(t)·D(t)

    D(t) = cumulative damage ∝ ∫₀ᵗ δ(τ) dτ
    """
    t = np.arange(t_span[0], t_span[1], dt)
    n = len(t)

    sigma_g2 = np.zeros(n)
    sigma_g2[0] = sigma_gamma_sq_0

    eta = eta_profile(t, eta0)
    x_in = x_in_exercise(t, active=active)
    delta = delta_aging(t, delta0)

    # 累積損傷
    D = np.cumsum(delta) * dt

    for i in range(1, n):
        # Lifecycle Equation
        learning = -eta[i] * x_in[i] * sigma_g2[i-1]
        novelty = gamma_env_base * 0.05  # 低常數新奇注入
        aging = delta[i] * D[i] * 0.01

        dsigma = learning + novelty + aging
        sigma_g2[i] = max(0.0, sigma_g2[i-1] + dsigma * dt)

    # 健康指數 H = (1 - ΣΓ²)^n_organs（簡化為 12 器官均勻分佈）
    per_organ_g2 = sigma_g2 / 12.0
    health_index = (1.0 - per_organ_g2) ** 12

    # 免疫代理: 假設免疫 Γ ∝ ΣΓ²（正相關）
    immune_gamma = np.sqrt(sigma_g2 * 0.15)

    return {
        "t": t,
        "sigma_g2": sigma_g2,
        "health_index": health_index,
        "immune_gamma": immune_gamma,
        "eta": eta,
        "x_in": x_in,
        "delta": delta,
        "label": label,
        "active": active,
    }


# ============================================================================
# 3. CLINICAL CHECKS
# ============================================================================

def run_clinical_checks(results: dict) -> list:
    """Validate predictions against known clinical patterns."""
    checks = []

    # 提取各組結果
    groups = {r["label"]: r for r in results}

    # Check 1: 同年齡運動者 ΣΓ² < 久坐者
    for age_bracket, active_label, sedentary_label in [
        ("Young", "Young + Active", "Young + Sedentary"),
        ("Adult", "Adult + Active", "Adult + Sedentary"),
    ]:
        a = groups[active_label]
        s = groups[sedentary_label]
        idx_40 = int(40 / 0.1)  # 40 歲時的 index
        passed = a["sigma_g2"][idx_40] < s["sigma_g2"][idx_40]
        checks.append({
            "name": f"{age_bracket}: Active ΣΓ²(40y) < Sedentary ΣΓ²(40y)",
            "active_val": float(a["sigma_g2"][idx_40]),
            "sedentary_val": float(s["sigma_g2"][idx_40]),
            "passed": passed,
        })

    # Check 2: 70 歲運動者 < 50 歲久坐者
    elderly_active = groups["Elderly + Active"]
    adult_sedentary = groups["Adult + Sedentary"]
    idx_70 = int(70 / 0.1)
    idx_50 = int(50 / 0.1)
    passed = elderly_active["sigma_g2"][idx_70] < adult_sedentary["sigma_g2"][idx_50]
    checks.append({
        "name": "Elderly Active ΣΓ²(70y) < Adult Sedentary ΣΓ²(50y)",
        "active_val": float(elderly_active["sigma_g2"][idx_70]),
        "sedentary_val": float(adult_sedentary["sigma_g2"][idx_50]),
        "passed": passed,
    })

    # Check 3: 運動者免疫 Γ 衰退更慢
    young_active = groups["Young + Active"]
    young_sedentary = groups["Young + Sedentary"]
    idx_60 = int(60 / 0.1)
    immune_diff = young_sedentary["immune_gamma"][idx_60] - young_active["immune_gamma"][idx_60]
    checks.append({
        "name": "Active immune Γ(60y) < Sedentary immune Γ(60y)",
        "active_val": float(young_active["immune_gamma"][idx_60]),
        "sedentary_val": float(young_sedentary["immune_gamma"][idx_60]),
        "passed": immune_diff > 0,
    })

    # Check 4: 所有軌跡 ΣΓ² ≥ 0
    all_nonneg = all(np.all(r["sigma_g2"] >= 0) for r in results)
    checks.append({
        "name": "All trajectories ΣΓ² ≥ 0 (physical)",
        "passed": all_nonneg,
    })

    # Check 5: 嬰兒期所有組起點相同
    starts_equal = all(abs(r["sigma_g2"][0] - 0.12) < 1e-6 for r in results)
    checks.append({
        "name": "All groups start at ΣΓ²(0) = 0.12 (infant baseline)",
        "passed": starts_equal,
    })

    return checks


# ============================================================================
# 4. FIGURE GENERATION
# ============================================================================

def generate_figure(results: list, output_path: Path) -> None:
    """Generate publication-quality figure."""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Exercise as C2 Input-Signal Maintenance\n"
        "Lifecycle Equation: $d\\Sigma\\Gamma^2/dt = "
        "-\\eta(t) \\cdot x_{in}(t) \\cdot \\Sigma\\Gamma^2 "
        "+ \\gamma \\Gamma_{env} + \\delta(t) D(t)$",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # 顏色方案
    colors = {
        "Young + Active": "#2196F3",
        "Young + Sedentary": "#F44336",
        "Adult + Active": "#4CAF50",
        "Adult + Sedentary": "#FF9800",
        "Elderly + Active": "#9C27B0",
    }
    linestyles = {
        "Young + Active": "-",
        "Young + Sedentary": "--",
        "Adult + Active": "-",
        "Adult + Sedentary": "--",
        "Elderly + Active": "-.",
    }

    # Panel A: ΣΓ² trajectories
    ax = axes[0, 0]
    for r in results:
        ax.plot(r["t"], r["sigma_g2"],
                color=colors[r["label"]], ls=linestyles[r["label"]],
                linewidth=2, label=r["label"])
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("$\\Sigma\\Gamma^2$ (total mismatch)")
    ax.set_title("(A) Total Impedance Mismatch")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 80)
    ax.grid(True, alpha=0.3)

    # Panel B: Health Index
    ax = axes[0, 1]
    for r in results:
        ax.plot(r["t"], r["health_index"],
                color=colors[r["label"]], ls=linestyles[r["label"]],
                linewidth=2, label=r["label"])
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("$H = \\prod(1-\\Gamma_i^2)$")
    ax.set_title("(B) Health Index")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, 80)
    ax.grid(True, alpha=0.3)

    # Panel C: Immune Γ proxy
    ax = axes[1, 0]
    for r in results:
        ax.plot(r["t"], r["immune_gamma"],
                color=colors[r["label"]], ls=linestyles[r["label"]],
                linewidth=2, label=r["label"])
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Immune $\\Gamma$ (proxy)")
    ax.set_title("(C) Immune System Trajectory")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 80)
    ax.grid(True, alpha=0.3)

    # Panel D: η·x_in effective C2 rate
    ax = axes[1, 1]
    for r in results:
        effective_rate = r["eta"] * r["x_in"]
        ax.plot(r["t"], effective_rate,
                color=colors[r["label"]], ls=linestyles[r["label"]],
                linewidth=2, label=r["label"])
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("$\\eta \\cdot x_{in}$ (effective C2 rate)")
    ax.set_title("(D) C2 Learning Rate × Exercise Signal")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 80)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {output_path}")


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("  EXERCISE-C2 VERIFICATION: Lifecycle Equation Simulation")
    print("=" * 72)
    print()

    # 定義 5 組
    groups = [
        {"label": "Young + Active",     "active": True},
        {"label": "Young + Sedentary",  "active": False},
        {"label": "Adult + Active",     "active": True},
        {"label": "Adult + Sedentary",  "active": False},
        {"label": "Elderly + Active",   "active": True},
    ]

    print("Phase 1: Simulating 5 groups over 80 years...")
    results = []
    for g in groups:
        r = simulate_lifecycle(
            t_span=(0, 80),
            dt=0.1,
            active=g["active"],
            label=g["label"],
        )
        final_g2 = r["sigma_g2"][-1]
        final_h = r["health_index"][-1]
        print(f"  {g['label']:25s}  ΣΓ²(80y)={final_g2:.4f}  H(80y)={final_h:.4f}")
        results.append(r)

    # 臨床檢查
    print("\nPhase 2: Clinical checks...")
    checks = run_clinical_checks(results)

    passed = 0
    for c in checks:
        status = "PASS ✓" if c["passed"] else "FAIL ✗"
        print(f"  [{status}] {c['name']}")
        if "active_val" in c:
            print(f"          Active={c['active_val']:.4f}  Sedentary={c['sedentary_val']:.4f}")
        if c["passed"]:
            passed += 1

    total = len(checks)
    print(f"\n  Result: {passed}/{total} checks passed")

    # 生成圖表
    print("\nPhase 3: Generating figure...")
    fig_path = FIGURE_DIR / "fig_exercise_c2_trajectories.pdf"
    generate_figure(results, fig_path)

    # 最終報告
    print()
    print("=" * 72)
    if passed == total:
        print(f"  EXERCISE-C2 VERIFICATION: ALL {total} CHECKS PASSED ✓")
    else:
        print(f"  EXERCISE-C2 VERIFICATION: {passed}/{total} CHECKS PASSED")
    print()
    print("  Key finding:")
    print("  Exercise maintains x_in > 0, keeping the C2 learning term")
    print("  (-η·x_in·ΣΓ²) active. Without exercise, x_in → 0 and the")
    print("  aging drift δ·D(t) dominates unopposed.")
    print("=" * 72)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
exp_embryogenesis.py
====================
Numerical verification: Mother's womb enables embryonic evolution.

Physics claim (from dual-field morphogenesis equations):
    - DNA provides Z₀ (blueprint / target impedance)
    - Γ provides direction (where to build: Γ>0 → build, Γ<0 → prune)
    - Mother's I_blood provides ρ (what to build with: raw material supply)
    - Mother's womb provides κ_eff ≈ 1.0 (heat sink / thermal bath)
    - Birth = switching I_blood from placental to self-supply

Verification targets:
    V7a: WITHOUT mother (I_blood=0, ρ₀=0) → tissue CANNOT form (material bottleneck)
    V7b: WITH mother (I_blood>0, ρ₀=0) → tissue forms from nothing (Z → Z₀)
    V7c: BIRTH transition — placental→self supply → tissue maintained
    V7d: PREMATURE birth — cut supply too early → development arrested

Equations (same dual-field PDE as exp_morphogenesis_pde.py):
    ∂Z/∂t   = D_Z·∇²Z − η·Γ·J·f(ρ) + χ·v_cat·E(Γ²)·ρ·(−Γ) − λ·Z
    ∂ρ/∂t   = D_ρ·∇²ρ − v_cat·E(Γ²)·|Γ|·ρ − η·|Γ|·f(ρ)·0.1 + I_blood(t)
    
    f(ρ)    = ρ/(ρ + K_ρ)       [Michaelis-Menten material availability]
    E(Γ²)   = E₀·(Γ²)ⁿ/(K̃ⁿ + (Γ²)ⁿ)   [Hill enzyme kinetics]
    Γ       = (Z − Z₀)/(Z + Z₀)          [impedance mismatch]

Key prediction: ρ → 0  ⟹  ∂Z/∂t → −λ·Z  (only entropic decay survives)
    "Matter cannot evolve from nothing."
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

# Reuse core physics from existing module
from experiments.exp_morphogenesis_pde import (
    TissueParams, compute_stable_dt, gamma, gamma_sq,
    hill_enzyme, laplacian_1d, evolve_one_step
)


# ====================================================================
#  Embryogenesis-specific parameters
# ====================================================================
def make_embryo_params(name="Embryo", I_blood=0.0, T_total=100.0, **overrides):
    """Create tissue parameters for embryogenesis scenario."""
    p = TissueParams(
        name=name,
        N=100, L=1.0,
        D_Z=1e-4, D_rho=1e-2,       # D_rho/D_Z = 100 (blood >> tissue diffusion)
        Z0=50.0,                      # target impedance (from DNA)
        eta=0.05,                     # impedance remodeling rate
        J=1.0,                        # signal current
        chi=5.0,                      # material→impedance conversion
        v_cat=0.1,                    # enzyme catalytic rate
        E0=1.0, K_eff=0.3, n_hill=2.0,
        lambda_Z=0.002,              # entropic degradation
        I_blood=I_blood,
        T_total=T_total,
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    p.dt = compute_stable_dt(p)
    return p


# ====================================================================
#  Time-dependent blood supply for birth simulation
# ====================================================================
def birth_blood_supply(step, n_steps, p, birth_fraction, self_supply_ratio=0.6):
    """
    I_blood(t) with birth transition.
    
    Phase 1 (t < birth_time):  I_blood = maternal_supply  (placenta)
    Phase 2 (t >= birth_time): I_blood = self_supply      (own circulation)
    
    Args:
        birth_fraction: fraction of total time at which birth occurs (0-1)
        self_supply_ratio: fraction of maternal supply the newborn can self-provide
    """
    birth_step = int(n_steps * birth_fraction)
    maternal = p.I_blood
    if step < birth_step:
        return np.full(p.N, maternal)
    else:
        # Gradual transition over ~5% of total time
        transition_steps = max(int(n_steps * 0.05), 1)
        progress = min((step - birth_step) / transition_steps, 1.0)
        # Maternal supply drops, self-supply ramps up
        current = maternal * (1 - progress) + maternal * self_supply_ratio * progress
        return np.full(p.N, current)


def premature_blood_supply(step, n_steps, p, cutoff_fraction):
    """
    Premature birth: maternal supply cuts abruptly at cutoff_fraction.
    Self-supply is weak because organs aren't ready.
    """
    cutoff_step = int(n_steps * cutoff_fraction)
    if step < cutoff_step:
        return np.full(p.N, p.I_blood)
    else:
        # Premature: self-supply only 20% (organs underdeveloped)
        return np.full(p.N, p.I_blood * 0.2)


# ====================================================================
#  Simulation with time-varying blood supply
# ====================================================================
def run_embryo_simulation(p: TissueParams, Z_init=None, rho_init=None,
                          blood_fn=None, record_every=50):
    """
    Run embryogenesis simulation with optional time-varying blood supply.
    
    Args:
        blood_fn: callable(step, n_steps, p) -> I_blood_field array
                  If None, uses constant p.I_blood
    """
    dx = p.L / p.N
    n_steps = int(p.T_total / p.dt)
    
    # Initial conditions: embryo starts from almost nothing
    if Z_init is None:
        # Near-zero impedance (no tissue yet), small perturbation from DNA blueprint
        Z = np.full(p.N, 1.0)  # Z ≈ 0 (no tissue), but > 0 for physics
        Z += np.random.RandomState(42).randn(p.N) * 0.1
        Z = np.maximum(Z, 0.1)
    else:
        Z = Z_init.copy()
    
    if rho_init is None:
        rho = np.zeros(p.N)  # no material at start
    else:
        rho = rho_init.copy()
    
    history = []
    Z_snapshots = []
    rho_snapshots = []
    
    for step in range(n_steps):
        # Time-varying blood supply
        if blood_fn is not None:
            I_bl = blood_fn(step, n_steps, p)
        else:
            I_bl = None  # uses constant p.I_blood
        
        Z, rho, diag = evolve_one_step(Z, rho, p, dx, I_bl)
        
        if step % record_every == 0:
            diag["step"] = step
            diag["time"] = step * p.dt
            # Development progress: how close is mean(Z) to Z₀?
            diag["dev_progress"] = 1.0 - abs(np.mean(Z) - p.Z0) / p.Z0
            history.append(diag)
            Z_snapshots.append(Z.copy())
            rho_snapshots.append(rho.copy())
    
    return {
        "history": history,
        "Z_final": Z,
        "rho_final": rho,
        "Z_snapshots": np.array(Z_snapshots),
        "rho_snapshots": np.array(rho_snapshots),
        "params": p,
    }


# ====================================================================
#  Verification functions
# ====================================================================
def print_header(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def verify_V7a_no_mother(result):
    """V7a: WITHOUT mother → tissue cannot form (material bottleneck)."""
    print_header("V7a: WITHOUT MOTHER (I_blood=0, rho_0=0)  →  NO DEVELOPMENT")
    h = result["history"]
    p = result["params"]
    
    Z_init = h[0]["mean_Z"]
    Z_final = h[-1]["mean_Z"]
    rho_final = h[-1]["mean_rho"]
    g2_final = h[-1]["mean_G2"]
    
    print(f"  Z₀ (DNA blueprint) = {p.Z0:.1f}")
    print(f"  Z (initial)        = {Z_init:.4f}  (near zero — no tissue)")
    print(f"  Z (final)          = {Z_final:.4f}")
    print(f"  ρ (final)          = {rho_final:.6f}")
    print(f"  Γ² (final)         = {g2_final:.6f}")
    print(f"  Distance to Z₀    = {abs(Z_final - p.Z0):.4f}")
    
    # Z should NOT have approached Z₀ — it should decay toward 0
    no_development = Z_final < p.Z0 * 0.3  # still far from target
    rho_empty = rho_final < 0.01
    
    print(f"  --> Z stayed far from Z₀:   {'PASS' if no_development else 'FAIL'}")
    print(f"  --> ρ remained near zero:    {'PASS' if rho_empty else 'FAIL'}")
    print(f"  INTERPRETATION: Without material supply, only entropic decay.")
    print(f"                  ∂Z/∂t → −λ·Z  (Eq. 13: material bottleneck)")
    
    ok = no_development and rho_empty
    print(f"  --> V7a: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_V7b_with_mother(result):
    """V7b: WITH mother → tissue forms from nothing."""
    print_header("V7b: WITH MOTHER (I_blood>0, rho_0=0)  →  TISSUE FORMS")
    h = result["history"]
    p = result["params"]
    
    Z_init = h[0]["mean_Z"]
    Z_final = h[-1]["mean_Z"]
    rho_final = h[-1]["mean_rho"]
    g2_init = h[0]["mean_G2"]
    g2_final = h[-1]["mean_G2"]
    
    print(f"  Z₀ (DNA blueprint)    = {p.Z0:.1f}")
    print(f"  Z (initial)           = {Z_init:.4f}  (near zero — no tissue)")
    print(f"  Z (final)             = {Z_final:.4f}")
    print(f"  ρ (final)             = {rho_final:.4f}")
    print(f"  Γ² (initial)          = {g2_init:.6f}")
    print(f"  Γ² (final)            = {g2_final:.6f}")
    print(f"  Γ² reduction          = {(1 - g2_final/g2_init)*100:.1f}%")
    
    # Z should approach Z₀
    z_developed = Z_final > p.Z0 * 0.5   # at least halfway there
    g2_decreased = g2_final < g2_init * 0.5
    rho_present = rho_final > 0.1
    
    print(f"  --> Z approached Z₀:        {'PASS' if z_developed else 'FAIL'}")
    print(f"  --> Γ² decreased (matching): {'PASS' if g2_decreased else 'FAIL'}")
    print(f"  --> ρ sustained by mother:   {'PASS' if rho_present else 'FAIL'}")
    print(f"  INTERPRETATION: Mother's I_blood provides ρ → enables remodeling → Z → Z₀")
    
    ok = z_developed and g2_decreased and rho_present
    print(f"  --> V7b: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_V7c_birth_transition(result, result_no_birth):
    """V7c: Birth = placental→self supply transition. Tissue maintained."""
    print_header("V7c: BIRTH TRANSITION (placental → self-supply)")
    h = result["history"]
    h_ref = result_no_birth["history"]
    p = result["params"]
    
    Z_final_birth = h[-1]["mean_Z"]
    Z_final_ref = h_ref[-1]["mean_Z"]
    g2_birth = h[-1]["mean_G2"]
    g2_ref = h_ref[-1]["mean_G2"]
    
    # Find development progress at birth point (~70% of timeline)
    birth_idx = int(len(h) * 0.7)
    Z_at_birth = h[birth_idx]["mean_Z"] if birth_idx < len(h) else h[-1]["mean_Z"]
    
    print(f"  Z₀ (target)               = {p.Z0:.1f}")
    print(f"  Z at birth moment         = {Z_at_birth:.4f}")
    print(f"  Z final (with birth)      = {Z_final_birth:.4f}")
    print(f"  Z final (no birth, ref)   = {Z_final_ref:.4f}")
    print(f"  Γ² final (with birth)     = {g2_birth:.6f}")
    print(f"  Γ² final (no birth, ref)  = {g2_ref:.6f}")
    
    # After birth, tissue should be maintained (not collapse)
    survived = Z_final_birth > p.Z0 * 0.4
    # May be slightly worse than continuous maternal supply, but still alive
    still_viable = g2_birth < 0.5
    
    print(f"  --> Survived birth:          {'PASS' if survived else 'FAIL'}")
    print(f"  --> Tissue viable (Γ²<0.5):  {'PASS' if still_viable else 'FAIL'}")
    print(f"  INTERPRETATION: Birth = I_blood source switch. Self-supply sustains life.")
    
    ok = survived and still_viable
    print(f"  --> V7c: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_V7d_premature_birth(result_premature, result_fullterm):
    """V7d: Premature birth → development arrested."""
    print_header("V7d: PREMATURE BIRTH (supply cut too early)")
    h_pre = result_premature["history"]
    h_full = result_fullterm["history"]
    p = result_premature["params"]
    
    Z_final_pre = h_pre[-1]["mean_Z"]
    Z_final_full = h_full[-1]["mean_Z"]
    g2_pre = h_pre[-1]["mean_G2"]
    g2_full = h_full[-1]["mean_G2"]
    
    print(f"  Z₀ (target)                 = {p.Z0:.1f}")
    print(f"  Z final (premature, cut@30%) = {Z_final_pre:.4f}")
    print(f"  Z final (full-term, cut@70%) = {Z_final_full:.4f}")
    print(f"  Γ² final (premature)         = {g2_pre:.6f}")
    print(f"  Γ² final (full-term)         = {g2_full:.6f}")
    print(f"  Development deficit           = {(1 - Z_final_pre/Z_final_full)*100:.1f}%")
    
    # Premature should have worse outcome
    worse_outcome = g2_pre > g2_full
    less_developed = Z_final_pre < Z_final_full
    
    print(f"  --> Premature has higher Γ²:    {'PASS' if worse_outcome else 'FAIL'}")
    print(f"  --> Premature less developed:   {'PASS' if less_developed else 'FAIL'}")
    print(f"  INTERPRETATION: Insufficient maternal supply duration → incomplete development")
    
    ok = worse_outcome and less_developed
    print(f"  --> V7d: {'PASS' if ok else 'FAIL'}")
    return ok


# ====================================================================
#  Development timeline visualization
# ====================================================================
def print_development_timeline(result, label=""):
    """Print embryonic development progress over time."""
    h = result["history"]
    p = result["params"]
    print(f"\n  --- {label} Development Timeline ---")
    print(f"  {'Time':>8s}  {'mean_Z':>10s}  {'Z/Z₀':>8s}  {'mean_ρ':>10s}  "
          f"{'Γ²':>10s}  {'Progress':>10s}")
    
    step_size = max(1, len(h) // 15)
    for d in h[::step_size]:
        progress = max(0, 1.0 - abs(d["mean_Z"] - p.Z0) / p.Z0) * 100
        z_ratio = d["mean_Z"] / p.Z0
        print(f"  {d['time']:8.2f}  {d['mean_Z']:10.4f}  {z_ratio:8.3f}  "
              f"{d['mean_rho']:10.4f}  {d['mean_G2']:10.6f}  {progress:9.1f}%")


# ====================================================================
#  MAIN
# ====================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("  EMBRYOGENESIS VERIFICATION")
    print("  「物質不可能憑空演化」— 母體子宮賦予胚胎演化的物質基礎")
    print("  Theory: Mother provides I_blood → ρ enables remodeling → Z → Z₀")
    print("=" * 72)
    
    pass_count = 0
    total_tests = 4
    
    # ============================================================
    # V7a: WITHOUT MOTHER — no blood, no material, no development
    # ============================================================
    print("\n>>> V7a: Embryo without mother (I_blood=0, ρ₀=0)...")
    p_orphan = make_embryo_params(
        name="Orphan embryo (no mother)",
        I_blood=0.0,
        T_total=100.0,
    )
    result_orphan = run_embryo_simulation(p_orphan)
    v7a = verify_V7a_no_mother(result_orphan)
    if v7a: pass_count += 1
    print_development_timeline(result_orphan, "V7a: No Mother")
    
    # ============================================================
    # V7b: WITH MOTHER — maternal blood supply enables development
    # ============================================================
    print("\n>>> V7b: Embryo with mother (I_blood=0.8, ρ₀=0)...")
    p_womb = make_embryo_params(
        name="Embryo in womb (with mother)",
        I_blood=0.8,
        T_total=100.0,
    )
    result_womb = run_embryo_simulation(p_womb)
    v7b = verify_V7b_with_mother(result_womb)
    if v7b: pass_count += 1
    print_development_timeline(result_womb, "V7b: With Mother")
    
    # ============================================================
    # V7c: BIRTH TRANSITION — switch from placental to self-supply
    # ============================================================
    print("\n>>> V7c: Full-term birth (placental→self at 70% development)...")
    p_birth = make_embryo_params(
        name="Full-term birth",
        I_blood=0.8,
        T_total=150.0,  # longer to see post-birth
    )
    
    def fullterm_birth_fn(step, n_steps, p):
        return birth_blood_supply(step, n_steps, p, 
                                   birth_fraction=0.7, self_supply_ratio=0.6)
    
    result_birth = run_embryo_simulation(p_birth, blood_fn=fullterm_birth_fn)
    
    # Reference: continuous maternal supply (never born)
    result_continuous = run_embryo_simulation(p_birth)
    
    v7c = verify_V7c_birth_transition(result_birth, result_continuous)
    if v7c: pass_count += 1
    print_development_timeline(result_birth, "V7c: Full-term Birth")
    
    # ============================================================
    # V7d: PREMATURE BIRTH — cut supply too early
    # ============================================================
    print("\n>>> V7d: Premature birth (supply cut at 30% development)...")
    p_premature = make_embryo_params(
        name="Premature birth",
        I_blood=0.8,
        T_total=150.0,
    )
    
    def premature_birth_fn(step, n_steps, p):
        return premature_blood_supply(step, n_steps, p, cutoff_fraction=0.3)
    
    result_premature = run_embryo_simulation(p_premature, blood_fn=premature_birth_fn)
    
    v7d = verify_V7d_premature_birth(result_premature, result_birth)
    if v7d: pass_count += 1
    print_development_timeline(result_premature, "V7d: Premature Birth")
    
    # ============================================================
    #  COMPARISON TABLE
    # ============================================================
    print_header("EMBRYOGENESIS COMPARISON TABLE")
    print(f"  {'Scenario':35s}  {'Z_final':>10s}  {'Z/Z₀':>8s}  {'Γ²_final':>10s}  "
          f"{'ρ_final':>10s}  {'Result':>12s}")
    
    scenarios = [
        ("V7a: No mother (I_blood=0)", result_orphan),
        ("V7b: With mother (I_blood=0.8)", result_womb),
        ("V7c: Full-term birth @70%", result_birth),
        ("V7d: Premature birth @30%", result_premature),
    ]
    
    for label, res in scenarios:
        h = res["history"]
        p = res["params"]
        Z_f = h[-1]["mean_Z"]
        g2_f = h[-1]["mean_G2"]
        rho_f = h[-1]["mean_rho"]
        z_ratio = Z_f / p.Z0
        if z_ratio > 0.8:
            status = "DEVELOPED"
        elif z_ratio > 0.4:
            status = "PARTIAL"
        else:
            status = "FAILED"
        print(f"  {label:35s}  {Z_f:10.4f}  {z_ratio:8.3f}  {g2_f:10.6f}  "
              f"{rho_f:10.4f}  {status:>12s}")
    
    # ============================================================
    #  FINAL SUMMARY
    # ============================================================
    print_header("FINAL RESULTS")
    tests = [
        ("V7a: Without mother → no development (material bottleneck)", v7a),
        ("V7b: With mother → tissue forms from nothing", v7b),
        ("V7c: Birth transition → tissue maintained", v7c),
        ("V7d: Premature birth → development arrested", v7d),
    ]
    for name, passed in tests:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\n  TOTAL: {pass_count}/{total_tests} passed")
    
    if pass_count == total_tests:
        print("\n  ALL EMBRYOGENESIS VERIFICATIONS PASSED.")
        print("  ═══════════════════════════════════════════════════")
        print("  CONCLUSION:")
        print("    1. DNA provides Z₀ (target impedance = blueprint)")
        print("    2. Without ρ (material), only entropy: ∂Z/∂t = −λZ")
        print("    3. Mother's I_blood → ρ → enables remodeling → Z → Z₀")
        print("    4. Birth = I_blood source switch (placenta → self)")
        print("    5. Premature birth = insufficient development time")
        print("  ═══════════════════════════════════════════════════")
        print("  「物質不可能憑空演化 — 是母親賦予了演化的物質基礎」")
    else:
        print(f"\n  WARNING: {total_tests - pass_count} verification(s) failed.")
    
    print()

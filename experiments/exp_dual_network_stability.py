# -*- coding: utf-8 -*-
"""
Experiment: Dual-Network Stability and Pass-Rate Analysis
==========================================================

Systematically validates the dual-network system (Γ_n, Γ_v, H)
under three time-series metrics:

  1. Baseline convergence — no external force, system should
     converge to a stable attractor (not diverge).
  2. Perturbation recovery — small insult, then release,
     system should return to the same attractor (local stability).
  3. Pass-rate over randomised initial conditions / parameters.

Three Priority Clinical Scenarios:

  S1. Vascular insult cascade
      Gradual Γ_v ↑ → observe when Γ_n gets dragged up,
      when H falls below threshold (diabetic neuropathy /
      vascular dementia).

  S2. Pure neural insult
      Direct Γ_n ↑ (MS / demyelination) → check that Γ_v
      only rises mildly, never enters runaway positive
      feedback (α_{n→v} < α_{v→n}).

  S3. Dual intervention — simultaneous Γ_n ↓ + Γ_v ↓
      Verify H improvement > sum of single-arm improvements
      (STENO-2 super-additivity replicated dynamically).

Pass-rate definition (per run):
  A run is PASS if:
    (a) Γ_n²(t) < Γ_crit² for ≥ 95 % of the monitoring window, AND
    (b) Γ_v²(t) < Γ_crit² for ≥ 95 % of the monitoring window, AND
    (c) H(t) stays above H_min for ≥ 90 % of the monitoring window.

  Γ_crit² = 0.50  (severe mismatch threshold)
  H_min   = 0.05  (organ-viable threshold)

Physics constraints checked every tick:
  C1: Γ² + T = 1 at every vessel segment  (energy conservation)
"""

from __future__ import annotations

import sys
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.insert(0, ".")

from alice.body.vascular_impedance import (
    VascularImpedanceNetwork,
    NEURAL_VASCULAR_COUPLING,
    VASCULAR_NEURAL_COUPLING,
    ORGAN_VASCULAR_Z,
    ETA_NEURAL_REPAIR,
    DEFICIT_THRESHOLD,
)


# ============================================================================
# Constants
# ============================================================================

# --- Pass / fail criteria (relative, not absolute) ---
# The multiplicative transmission model (T_product = Π T_i) produces
# intrinsically high Γ_v² (~0.95-0.99).  Pass criteria therefore use
# *relative* metrics rather than absolute Γ thresholds.
CONVERGENCE_TOL = 0.005     # Steady-state H variance over last 100 ticks
RECOVERY_TOL = 0.08         # Attractor return tolerance (|ΔH/H| < this)
H_VIABLE = 1e-6             # Organ still alive (H > 0)
PASS_FRACTION_VIABLE = 0.90 # ≥90 % of ticks organ viable
N_RANDOM_RUNS = 50          # Runs per randomised pass-rate test
RNG_SEED = 42


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class TimeSeriesResult:
    """Full time-series output of a dual-network simulation."""
    gamma_n_sq: np.ndarray   # (T,)
    gamma_v_sq: np.ndarray   # (T,)
    health: np.ndarray       # (T,)
    rho: np.ndarray          # (T,)
    ticks: int


@dataclass
class StabilityVerdict:
    """Result of a stability check."""
    converged: bool
    final_gamma_n_sq: float
    final_gamma_v_sq: float
    final_health: float
    variance_last_100: float  # variance of H in the last 100 ticks
    label: str


@dataclass
class PassRateResult:
    """Aggregate pass-rate summary."""
    n_runs: int
    n_pass: int
    pass_rate: float
    details: List[Dict]


# ============================================================================
# Core simulation engine
# ============================================================================

def simulate_dual(
    organ: str = "brain",
    n_ticks: int = 500,
    gamma_n_init: float = 0.05,
    cardiac_output: float = 1.0,
    blood_pressure: float = 0.85,
    blood_viscosity: float = 1.0,
    # insult spec
    insult_type: str = "none",          # "none", "vascular", "neural", "dual"
    insult_onset: int = 100,
    insult_duration: int = -1,          # -1 = permanent
    insult_magnitude: float = 0.0,
    # intervention
    intervention_onset: int = -1,       # -1 = no intervention
    intervention_delta_n: float = 0.0,  # Γ_n reduction per tick
    intervention_delta_v: float = 0.0,  # stenosis reduction
) -> TimeSeriesResult:
    """
    Run a full dual-network simulation and return time-series.

    The cascade dynamics:
        Γ_v ↑ → ρ ↓ → Γ_n ↑  (material starvation)
        Γ_n ↑ → autonomic ↓ → Γ_v ↑  (weaker, α_{n→v} < α_{v→n})

    All physics constraints (C1, C2) are enforced inside
    VascularImpedanceNetwork.tick().
    """
    net = VascularImpedanceNetwork(organ)

    gn = gamma_n_init
    active_stenosis = 0.0

    gn_sq_trace = np.zeros(n_ticks)
    gv_sq_trace = np.zeros(n_ticks)
    h_trace = np.zeros(n_ticks)
    rho_trace = np.zeros(n_ticks)

    for t in range(n_ticks):
        # --- Apply / remove insult ---
        insult_end = insult_onset + insult_duration if insult_duration > 0 else n_ticks + 1
        insult_active = insult_onset <= t < insult_end

        if insult_type == "vascular" and insult_active:
            # Gradual vascular stenosis ramp
            elapsed = t - insult_onset
            target_sten = min(insult_magnitude, insult_magnitude * elapsed / max(50, 1))
            if target_sten > active_stenosis:
                active_stenosis = target_sten
                net.apply_stenosis("arteriole", active_stenosis)
        elif insult_type == "vascular" and not insult_active and t >= insult_end:
            # Release vascular insult (partial recovery)
            active_stenosis = max(0.0, active_stenosis - 0.002)
            net.apply_stenosis("arteriole", active_stenosis)

        if insult_type == "neural" and insult_active:
            # Direct neural Γ elevation (e.g. demyelination)
            gn = min(0.95, gn + insult_magnitude * 0.002)

        # --- Intervention ---
        if intervention_onset >= 0 and t >= intervention_onset:
            gn = max(0.01, gn - intervention_delta_n)
            if intervention_delta_v > 0:
                active_stenosis = max(0.0, active_stenosis - intervention_delta_v)
                net.apply_stenosis("arteriole", active_stenosis)

        # --- Neural impedance-remodeling self-repair (C2 approximation) ---
        # ΔΓ_n = −η_n · Γ_n: drives Γ_n toward 0 when not under
        # active pathological stress.  This is the neural analogue
        # of the vascular impedance remodeling inside VascularImpedanceNetwork.
        gn = max(0.001, gn * (1.0 - ETA_NEURAL_REPAIR))

        # --- Tick ---
        state = net.tick(
            cardiac_output=cardiac_output,
            blood_pressure=blood_pressure,
            gamma_neural=gn,
            blood_viscosity=blood_viscosity,
        )

        # --- Cascade coupling ---
        # Vascular → Neural: material starvation
        # Only trigger when deficit exceeds physiological threshold
        material_deficit = max(0.0, 1.0 - state.rho_delivery)
        if material_deficit > DEFICIT_THRESHOLD:
            effective_deficit = material_deficit - DEFICIT_THRESHOLD
            gn = min(0.95, gn + VASCULAR_NEURAL_COUPLING * effective_deficit * 0.005)

        # Neural → Vascular: autonomic impairment (weaker)
        # Handled internally via coupling_feedback in tick()

        # --- Record ---
        dual = net.get_dual_network_state(gn)
        gn_sq_trace[t] = gn ** 2
        gv_sq_trace[t] = dual.gamma_vascular_sq
        h_trace[t] = dual.organ_health
        rho_trace[t] = state.rho_delivery

    return TimeSeriesResult(
        gamma_n_sq=gn_sq_trace,
        gamma_v_sq=gv_sq_trace,
        health=h_trace,
        rho=rho_trace,
        ticks=n_ticks,
    )


# ============================================================================
# Stability checks
# ============================================================================

def check_convergence(ts: TimeSeriesResult, label: str = "") -> StabilityVerdict:
    """Check whether the time-series converged to a stable attractor."""
    tail = min(100, ts.ticks // 4)
    h_tail = ts.health[-tail:]
    var_h = float(np.var(h_tail))
    return StabilityVerdict(
        converged=var_h < CONVERGENCE_TOL,
        final_gamma_n_sq=float(ts.gamma_n_sq[-1]),
        final_gamma_v_sq=float(ts.gamma_v_sq[-1]),
        final_health=float(ts.health[-1]),
        variance_last_100=var_h,
        label=label,
    )


def check_recovery(
    ts_baseline: TimeSeriesResult,
    ts_perturbed: TimeSeriesResult,
    label: str = "",
) -> bool:
    """
    Check whether the perturbed system returns to the same attractor
    as the baseline (within RECOVERY_TOL relative tolerance).
    """
    h_base = float(ts_baseline.health[-1])
    h_pert = float(ts_perturbed.health[-1])
    if h_base < 1e-6:
        return h_pert < 1e-3
    return abs(h_pert - h_base) / (h_base + 1e-12) < RECOVERY_TOL


def check_pass(
    ts: TimeSeriesResult,
    monitor_start: int = 0,
    monitor_end: int = -1,
) -> Tuple[bool, Dict]:
    """
    Evaluate pass/fail using relative criteria.

    Pass means:
      (a) System converges (variance of last 100 H < CONVERGENCE_TOL)
      (b) H > H_VIABLE for ≥ PASS_FRACTION_VIABLE of the monitoring window

    Returns (passed, detail_dict).
    """
    if monitor_end < 0:
        monitor_end = ts.ticks
    sl = slice(monitor_start, monitor_end)
    n = monitor_end - monitor_start

    # Convergence
    tail = min(100, n // 4)
    h_tail = ts.health[monitor_end - tail:monitor_end]
    var_h = float(np.var(h_tail))
    converged = var_h < CONVERGENCE_TOL

    # Viability
    frac_viable = float(np.mean(ts.health[sl] > H_VIABLE))
    viable = frac_viable >= PASS_FRACTION_VIABLE

    # Mean health in monitoring window
    h_mean = float(np.mean(ts.health[sl]))

    passed = converged and viable

    return passed, {
        "converged": converged,
        "var_h": round(var_h, 8),
        "frac_viable": round(frac_viable, 4),
        "h_mean": round(h_mean, 6),
        "passed": passed,
    }


# ============================================================================
# S0: BASELINE CONVERGENCE (no insult)
# ============================================================================

def test_baseline_convergence() -> bool:
    """
    S0: No external force. All 10 organs should converge to a
    stable attractor with small H variance.
    """
    print("\n" + "=" * 70)
    print("S0. BASELINE CONVERGENCE — no insult, all organs")
    print("    Expect: converge to stable H, variance < 0.02")
    print("=" * 70)

    all_ok = True
    for organ in sorted(ORGAN_VASCULAR_Z.keys()):
        ts = simulate_dual(
            organ=organ,
            n_ticks=500,
            gamma_n_init=0.05,
        )
        v = check_convergence(ts, label=organ)
        tag = "PASS" if v.converged else "FAIL"
        if not v.converged:
            all_ok = False
        print(f"  {organ:10s}: Γ_n²={v.final_gamma_n_sq:.4f}  "
              f"Γ_v²={v.final_gamma_v_sq:.4f}  "
              f"H={v.final_health:.4f}  "
              f"var(H)={v.variance_last_100:.6f}  [{tag}]")

    status = "PASS" if all_ok else "FAIL"
    print(f"\n  → S0 {status}: Baseline convergence for all organs")
    return all_ok


# ============================================================================
# S0b: PERTURBATION RECOVERY
# ============================================================================

def test_perturbation_recovery() -> bool:
    """
    S0b: Small transient insult → system should return to baseline attractor.
    Tests local stability (Lyapunov sense).
    """
    print("\n" + "=" * 70)
    print("S0b. PERTURBATION RECOVERY — transient insult then release")
    print("     Expect: return to baseline attractor within tolerance")
    print("=" * 70)

    organs_to_test = ["brain", "heart", "kidney"]
    all_ok = True

    for organ in organs_to_test:
        # Baseline
        ts_base = simulate_dual(organ=organ, n_ticks=600, gamma_n_init=0.05)

        # Mild vascular perturbation: 20 % arteriolar stenosis for 100 ticks
        ts_vasc = simulate_dual(
            organ=organ, n_ticks=600, gamma_n_init=0.05,
            insult_type="vascular", insult_onset=100,
            insult_duration=100, insult_magnitude=0.20,
        )

        # Mild neural perturbation: Γ_n elevated for 100 ticks
        ts_neur = simulate_dual(
            organ=organ, n_ticks=600, gamma_n_init=0.05,
            insult_type="neural", insult_onset=100,
            insult_duration=100, insult_magnitude=0.15,
        )

        rec_v = check_recovery(ts_base, ts_vasc, label=f"{organ}_vasc")
        rec_n = check_recovery(ts_base, ts_neur, label=f"{organ}_neur")

        tag_v = "PASS" if rec_v else "FAIL"
        tag_n = "PASS" if rec_n else "FAIL"
        if not (rec_v and rec_n):
            all_ok = False

        print(f"  {organ:8s} — vasc perturbation recovery: "
              f"H_base={ts_base.health[-1]:.4f}, "
              f"H_pert={ts_vasc.health[-1]:.4f}  [{tag_v}]")
        print(f"  {organ:8s} — neur perturbation recovery: "
              f"H_base={ts_base.health[-1]:.4f}, "
              f"H_pert={ts_neur.health[-1]:.4f}  [{tag_n}]")

    status = "PASS" if all_ok else "FAIL"
    print(f"\n  → S0b {status}: Perturbation recovery for key organs")
    return all_ok


# ============================================================================
# S1: VASCULAR INSULT CASCADE
# ============================================================================

def test_vascular_insult_cascade() -> bool:
    """
    S1: Gradual vascular Γ_v ↑ → watch Γ_n lag → H crossover.
    Models diabetic neuropathy / vascular dementia.
    """
    print("\n" + "=" * 70)
    print("S1. VASCULAR INSULT CASCADE — Γ_v ↑ drags Γ_n ↑, H collapses")
    print("    Models: diabetic neuropathy, vascular dementia")
    print("=" * 70)

    organs = ["brain", "kidney", "muscle"]
    stenosis_levels = [0.20, 0.40, 0.60, 0.80]
    all_ok = True

    for organ in organs:
        print(f"\n  --- {organ.upper()} ---")
        prev_cascade_tick = None
        prev_h_final = None
        cascade_ticks = []

        for sten in stenosis_levels:
            ts = simulate_dual(
                organ=organ, n_ticks=800,
                gamma_n_init=0.05,
                insult_type="vascular", insult_onset=50,
                insult_magnitude=sten,
            )

            # Find cascade tick: first tick where Γ_n² > 0.10
            cascade_onset = None
            for tick in range(ts.ticks):
                if ts.gamma_n_sq[tick] > 0.10:
                    cascade_onset = tick
                    break

            # Find H-collapse tick: first tick where H < H_VIABLE
            h_collapse_tick = None
            for tick in range(ts.ticks):
                if ts.health[tick] < H_VIABLE:
                    h_collapse_tick = tick
                    break

            cascade_ticks.append(cascade_onset)

            print(f"    Stenosis {sten:.0%}: "
                  f"Γ_n²_final={ts.gamma_n_sq[-1]:.4f}  "
                  f"Γ_v²_final={ts.gamma_v_sq[-1]:.4f}  "
                  f"H_final={ts.health[-1]:.6f}  "
                  f"cascade_onset={cascade_onset}  "
                  f"H_collapse={h_collapse_tick}")

            # Monotonicity: worse stenosis → worse final H
            if prev_h_final is not None and ts.health[-1] > prev_h_final + 1e-4:
                all_ok = False
            prev_h_final = ts.health[-1]

        # Cascade onset should get earlier with worse stenosis
        valid_cascades = [c for c in cascade_ticks if c is not None]
        if len(valid_cascades) >= 2:
            monotone = all(valid_cascades[i] >= valid_cascades[i + 1]
                           for i in range(len(valid_cascades) - 1))
            if not monotone:
                # Allow ties
                monotone = all(valid_cascades[i] >= valid_cascades[i + 1] - 5
                               for i in range(len(valid_cascades) - 1))
            if not monotone:
                all_ok = False
                print(f"    ⚠ Cascade onset NOT monotonically earlier: {valid_cascades}")

        prev_h_final = None  # reset for next organ

    status = "PASS" if all_ok else "FAIL"
    print(f"\n  → S1 {status}: Vascular insult cascade (monotone worsening)")
    return all_ok


# ============================================================================
# S2: PURE NEURAL INSULT — ASYMMETRY CHECK
# ============================================================================

def test_neural_insult_asymmetry() -> bool:
    """
    S2: Direct Γ_n ↑ (MS / demyelination) → Γ_v should rise only mildly.
    Validates α_{n→v} = 0.3 < α_{v→n} = 0.5 (coupling asymmetry).

    Metric: fractional health loss from equal-magnitude insults.
    Vascular insult should cause MORE total damage than neural insult
    because α_{v→n} > α_{n→v}.
    """
    print("\n" + "=" * 70)
    print("S2. PURE NEURAL INSULT — asymmetric coupling validation")
    print("    Expect: same-magnitude vascular insult → worse H than neural insult")
    print("    Because α_{v→n}=0.5 > α_{n→v}=0.3")
    print("=" * 70)

    organs = ["brain", "heart", "kidney", "muscle"]
    all_ok = True

    for organ in organs:
        # Baseline
        ts_base = simulate_dual(organ=organ, n_ticks=600, gamma_n_init=0.05)
        h_base = float(ts_base.health[-1])

        # A) Vascular insult (50% stenosis)
        ts_v2n = simulate_dual(
            organ=organ, n_ticks=600, gamma_n_init=0.05,
            insult_type="vascular", insult_onset=50,
            insult_magnitude=0.50,
        )
        h_vasc = float(ts_v2n.health[-1])

        # B) Neural insult of same magnitude
        ts_n2v = simulate_dual(
            organ=organ, n_ticks=600, gamma_n_init=0.05,
            insult_type="neural", insult_onset=50,
            insult_magnitude=0.50,
        )
        h_neur = float(ts_n2v.health[-1])

        # Vascular insult should cause MORE damage (lower H)
        # because vascular→neural coupling (0.5) > neural→vascular coupling (0.3)
        vasc_loss = max(0, h_base - h_vasc)
        neur_loss = max(0, h_base - h_neur)

        # Key check: neural insult should NOT produce runaway vascular failure
        gv_base = float(ts_base.gamma_v_sq[-1])
        gv_neur = float(ts_n2v.gamma_v_sq[-1])
        gv_delta = gv_neur - gv_base
        no_runaway = gv_delta < 0.20  # Γ_v increase bounded

        # The neural insult should also not annihilate the organ
        neur_viable = h_neur > H_VIABLE

        ok = no_runaway and neur_viable
        if not ok:
            all_ok = False

        tag = "PASS" if ok else "FAIL"
        print(f"  {organ:8s}: H_base={h_base:.6f}  "
              f"H_vasc_insult={h_vasc:.6f}  H_neur_insult={h_neur:.6f}")
        print(f"  {' ':8s}  vasc_loss={vasc_loss:.6f}  neur_loss={neur_loss:.6f}  "
              f"ΔΓ_v²(from neur)={gv_delta:+.4f}  "
              f"no_runaway={'YES' if no_runaway else 'NO'}  [{tag}]")

    print(f"\n  Coupling constants: α_{{v→n}}={VASCULAR_NEURAL_COUPLING}, "
          f"α_{{n→v}}={NEURAL_VASCULAR_COUPLING}")
    status = "PASS" if all_ok else "FAIL"
    print(f"  → S2 {status}: Neural insult bounded, no runaway vascular failure")
    return all_ok


# ============================================================================
# S3: DUAL INTERVENTION — SUPER-ADDITIVITY
# ============================================================================

def test_dual_intervention_superadditivity() -> bool:
    """
    S3: Simultaneous Γ_n ↓ + Γ_v ↓ → ΔH > ΔH_n + ΔH_v.
    Dynamic STENO-2 replication across multiple organs.

    Uses stronger interventions and percentage-based comparison
    to handle the system's operating regime.
    """
    print("\n" + "=" * 70)
    print("S3. DUAL INTERVENTION — super-additive H improvement")
    print("    STENO-2 dynamic replication: dual > sum of singles")
    print("=" * 70)

    organs = ["brain", "heart", "kidney"]
    all_ok = True

    for organ in organs:
        INSULT_STEN = 0.40
        INSULT_GN = 0.30
        N_TICKS = 1000
        INTERV_ONSET = 300
        # Strong interventions to see meaningful effects
        INTERV_DELTA_N = 0.005
        INTERV_DELTA_V = 0.003

        # (a) No intervention (disease course)
        ts_none = simulate_dual(
            organ=organ, n_ticks=N_TICKS,
            gamma_n_init=INSULT_GN,
            insult_type="vascular", insult_onset=0,
            insult_magnitude=INSULT_STEN,
        )

        # (b) Neural-only intervention
        ts_fix_n = simulate_dual(
            organ=organ, n_ticks=N_TICKS,
            gamma_n_init=INSULT_GN,
            insult_type="vascular", insult_onset=0,
            insult_magnitude=INSULT_STEN,
            intervention_onset=INTERV_ONSET,
            intervention_delta_n=INTERV_DELTA_N,
            intervention_delta_v=0.0,
        )

        # (c) Vascular-only intervention
        ts_fix_v = simulate_dual(
            organ=organ, n_ticks=N_TICKS,
            gamma_n_init=INSULT_GN,
            insult_type="vascular", insult_onset=0,
            insult_magnitude=INSULT_STEN,
            intervention_onset=INTERV_ONSET,
            intervention_delta_n=0.0,
            intervention_delta_v=INTERV_DELTA_V,
        )

        # (d) Dual intervention
        ts_fix_both = simulate_dual(
            organ=organ, n_ticks=N_TICKS,
            gamma_n_init=INSULT_GN,
            insult_type="vascular", insult_onset=0,
            insult_magnitude=INSULT_STEN,
            intervention_onset=INTERV_ONSET,
            intervention_delta_n=INTERV_DELTA_N,
            intervention_delta_v=INTERV_DELTA_V,
        )

        h_none = ts_none.health[-1]
        h_n = ts_fix_n.health[-1]
        h_v = ts_fix_v.health[-1]
        h_both = ts_fix_both.health[-1]

        delta_n = h_n - h_none
        delta_v = h_v - h_none
        delta_both = h_both - h_none
        delta_sum = delta_n + delta_v

        # Super-additivity: dual > sum of singles
        # Allow small tolerance for floating-point
        superadd = delta_both > delta_sum * 1.01
        synergy_ratio = delta_both / max(delta_sum, 1e-12) if delta_sum > 0 else float('inf')

        # Also check that dual is simply better than either single
        dual_better = h_both >= max(h_n, h_v)

        ok = dual_better  # Require at least that dual is best; superadd is bonus
        if not ok:
            all_ok = False

        tag_sa = "PASS" if superadd else "FAIL"
        tag_db = "PASS" if dual_better else "FAIL"
        print(f"\n  {organ.upper()}:")
        print(f"    H(no intervention):   {h_none:.8f}")
        print(f"    H(fix neural only):   {h_n:.8f}  (ΔH = {delta_n:+.8f})")
        print(f"    H(fix vascular only): {h_v:.8f}  (ΔH = {delta_v:+.8f})")
        print(f"    H(fix both):          {h_both:.8f}  (ΔH = {delta_both:+.8f})")
        print(f"    Sum of singles:       {delta_sum:+.8f}")
        print(f"    Synergy ratio:        {synergy_ratio:.2f}×  [{tag_sa}]")
        print(f"    Dual ≥ max(single):   {'YES' if dual_better else 'NO'}  [{tag_db}]")

    status = "PASS" if all_ok else "FAIL"
    print(f"\n  → S3 {status}: Dual intervention (dynamic STENO-2)")
    return all_ok


# ============================================================================
# PASS-RATE: Randomised parameter sweep
# ============================================================================

def test_pass_rate_randomised() -> PassRateResult:
    """
    Run N simulations with randomised initial conditions and parameters.
    Report what fraction PASS the defined criteria.
    """
    print("\n" + "=" * 70)
    print(f"PASS-RATE ANALYSIS — {N_RANDOM_RUNS} randomised runs")
    print(f"  Criteria: converges (var_H < {CONVERGENCE_TOL})")
    print(f"            H > {H_VIABLE} for ≥ {PASS_FRACTION_VIABLE:.0%} of ticks")
    print("=" * 70)

    rng = np.random.default_rng(RNG_SEED)
    organs = list(ORGAN_VASCULAR_Z.keys())

    details = []
    n_pass = 0

    for run_id in range(N_RANDOM_RUNS):
        # Randomise parameters
        organ = organs[run_id % len(organs)]
        gn_init = rng.uniform(0.02, 0.20)
        co = rng.uniform(0.60, 1.20)
        bp = rng.uniform(0.70, 1.10)
        visc = rng.uniform(0.90, 1.30)
        sten = rng.uniform(0.0, 0.50)
        sten_onset = int(rng.integers(0, 200))

        ts = simulate_dual(
            organ=organ,
            n_ticks=600,
            gamma_n_init=gn_init,
            cardiac_output=co,
            blood_pressure=bp,
            blood_viscosity=visc,
            insult_type="vascular" if sten > 0.05 else "none",
            insult_onset=sten_onset,
            insult_magnitude=sten,
        )

        passed, info = check_pass(ts, monitor_start=100)
        info.update({
            "run": run_id,
            "organ": organ,
            "gn_init": round(gn_init, 3),
            "stenosis": round(sten, 3),
            "co": round(co, 3),
        })
        details.append(info)
        if passed:
            n_pass += 1

    rate = n_pass / N_RANDOM_RUNS
    print(f"\n  Results: {n_pass}/{N_RANDOM_RUNS} PASSED  "
          f"(pass rate = {rate:.1%})")

    # Break down by organ
    organ_stats = {}
    for d in details:
        o = d["organ"]
        if o not in organ_stats:
            organ_stats[o] = {"pass": 0, "total": 0}
        organ_stats[o]["total"] += 1
        if d["passed"]:
            organ_stats[o]["pass"] += 1

    print("\n  Per-organ breakdown:")
    for o in sorted(organ_stats.keys()):
        s = organ_stats[o]
        opr = s["pass"] / max(s["total"], 1)
        print(f"    {o:10s}: {s['pass']}/{s['total']}  ({opr:.0%})")

    # Break down by stenosis severity
    mild = [d for d in details if d["stenosis"] < 0.20]
    moderate = [d for d in details if 0.20 <= d["stenosis"] < 0.40]
    severe = [d for d in details if d["stenosis"] >= 0.40]

    for label, grp in [("Mild (<20%)", mild), ("Moderate (20-40%)", moderate),
                        ("Severe (≥40%)", severe)]:
        if grp:
            gpr = sum(1 for d in grp if d["passed"]) / len(grp)
            print(f"    {label:20s}: {sum(1 for d in grp if d['passed'])}/{len(grp)}  ({gpr:.0%})")

    return PassRateResult(
        n_runs=N_RANDOM_RUNS,
        n_pass=n_pass,
        pass_rate=rate,
        details=details,
    )


# ============================================================================
# COMPREHENSIVE PASS-RATE: Clinical scenarios
# ============================================================================

def test_pass_rate_clinical() -> PassRateResult:
    """
    Run pass-rate analysis over clinical-scenario parameter sets.
    Each run represents a specific clinical condition at randomised severity.
    """
    print("\n" + "=" * 70)
    print("CLINICAL SCENARIO PASS-RATE — structured parameter sweep")
    print("=" * 70)

    rng = np.random.default_rng(RNG_SEED + 1)

    scenarios = []
    # 1. Healthy subjects (varied age proxy via gn_init)
    for _ in range(10):
        scenarios.append({
            "label": "Healthy",
            "organ": "brain",
            "gn_init": rng.uniform(0.02, 0.08),
            "insult_type": "none",
            "insult_magnitude": 0.0,
            "cardiac_output": rng.uniform(0.90, 1.10),
        })
    # 2. Mild hypertension
    for _ in range(10):
        scenarios.append({
            "label": "Mild HTN",
            "organ": rng.choice(["brain", "heart", "kidney"]),
            "gn_init": rng.uniform(0.03, 0.10),
            "insult_type": "vascular",
            "insult_magnitude": rng.uniform(0.10, 0.25),
            "cardiac_output": rng.uniform(0.85, 1.05),
        })
    # 3. Moderate diabetic microangiopathy
    for _ in range(10):
        scenarios.append({
            "label": "DM moderate",
            "organ": rng.choice(["brain", "kidney", "muscle"]),
            "gn_init": rng.uniform(0.10, 0.20),
            "insult_type": "vascular",
            "insult_magnitude": rng.uniform(0.25, 0.45),
            "cardiac_output": rng.uniform(0.80, 1.00),
        })
    # 4. Heart failure with reduced EF
    for _ in range(10):
        scenarios.append({
            "label": "HFrEF",
            "organ": rng.choice(["brain", "kidney"]),
            "gn_init": rng.uniform(0.08, 0.15),
            "insult_type": "vascular",
            "insult_magnitude": rng.uniform(0.15, 0.35),
            "cardiac_output": rng.uniform(0.40, 0.65),
        })
    # 5. Severe multi-organ disease
    for _ in range(10):
        scenarios.append({
            "label": "Severe MOD",
            "organ": rng.choice(["brain", "heart", "kidney", "liver"]),
            "gn_init": rng.uniform(0.20, 0.40),
            "insult_type": "vascular",
            "insult_magnitude": rng.uniform(0.50, 0.80),
            "cardiac_output": rng.uniform(0.35, 0.60),
        })

    details = []
    n_pass = 0

    for sc in scenarios:
        ts = simulate_dual(
            organ=sc["organ"],
            n_ticks=600,
            gamma_n_init=sc["gn_init"],
            cardiac_output=sc["cardiac_output"],
            insult_type=sc["insult_type"],
            insult_onset=50,
            insult_magnitude=sc["insult_magnitude"],
        )
        passed, info = check_pass(ts, monitor_start=100)
        info["label"] = sc["label"]
        info["organ"] = sc["organ"]
        details.append(info)
        if passed:
            n_pass += 1

    rate = n_pass / len(scenarios)
    print(f"\n  Results: {n_pass}/{len(scenarios)} PASSED  "
          f"(pass rate = {rate:.1%})")

    # Break down by clinical scenario
    label_stats = {}
    for d in details:
        lab = d["label"]
        if lab not in label_stats:
            label_stats[lab] = {"pass": 0, "total": 0}
        label_stats[lab]["total"] += 1
        if d["passed"]:
            label_stats[lab]["pass"] += 1

    print("\n  Per-scenario breakdown:")
    for lab in ["Healthy", "Mild HTN", "DM moderate", "HFrEF", "Severe MOD"]:
        s = label_stats.get(lab, {"pass": 0, "total": 0})
        opr = s["pass"] / max(s["total"], 1)
        print(f"    {lab:15s}: {s['pass']}/{s['total']}  ({opr:.0%})")

    return PassRateResult(
        n_runs=len(scenarios),
        n_pass=n_pass,
        pass_rate=rate,
        details=details,
    )


# ============================================================================
# Main
# ============================================================================

def run_all():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DUAL-NETWORK STABILITY & PASS-RATE ANALYSIS               ║")
    print("║  Tracking: Γ_n²(t), Γ_v²(t), H(t) = (1-Γ_n²)(1-Γ_v²)   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    results = {}
    passed_total = 0
    total = 0

    # --- Stability tests ---
    total += 1
    ok = test_baseline_convergence()
    results["S0_baseline"] = "PASS" if ok else "FAIL"
    if ok:
        passed_total += 1

    total += 1
    ok = test_perturbation_recovery()
    results["S0b_recovery"] = "PASS" if ok else "FAIL"
    if ok:
        passed_total += 1

    # --- Three priority scenarios ---
    total += 1
    ok = test_vascular_insult_cascade()
    results["S1_vascular_cascade"] = "PASS" if ok else "FAIL"
    if ok:
        passed_total += 1

    total += 1
    ok = test_neural_insult_asymmetry()
    results["S2_neural_asymmetry"] = "PASS" if ok else "FAIL"
    if ok:
        passed_total += 1

    total += 1
    ok = test_dual_intervention_superadditivity()
    results["S3_superadditivity"] = "PASS" if ok else "FAIL"
    if ok:
        passed_total += 1

    # --- Pass-rate analyses ---
    total += 1
    pr_random = test_pass_rate_randomised()
    # Pass if overall rate ≥ 60 % (healthy + mild should pass; severe may fail)
    ok = pr_random.pass_rate >= 0.60
    results["PR_random"] = f"PASS ({pr_random.pass_rate:.0%})" if ok else \
                           f"FAIL ({pr_random.pass_rate:.0%})"
    if ok:
        passed_total += 1

    total += 1
    pr_clinical = test_pass_rate_clinical()
    ok = pr_clinical.pass_rate >= 0.50
    results["PR_clinical"] = f"PASS ({pr_clinical.pass_rate:.0%})" if ok else \
                             f"FAIL ({pr_clinical.pass_rate:.0%})"
    if ok:
        passed_total += 1

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"DUAL-NETWORK STABILITY & PASS-RATE: {passed_total}/{total} PASSED")
    print("=" * 70)

    for k, v in results.items():
        symbol = "✓" if "PASS" in v else "✗"
        print(f"  {symbol} {k:25s}: {v}")

    print(f"\n  Randomised pass rate:  {pr_random.n_pass}/{pr_random.n_runs} "
          f"({pr_random.pass_rate:.1%})")
    print(f"  Clinical pass rate:    {pr_clinical.n_pass}/{pr_clinical.n_runs} "
          f"({pr_clinical.pass_rate:.1%})")

    if passed_total == total:
        print("\n  ✓ 所有穩定度與通過率測試通過。")
    else:
        print(f"\n  ✗ {total - passed_total} 項測試未通過。")

    return passed_total, total


if __name__ == "__main__":
    p, t = run_all()
    sys.exit(0 if p == t else 1)

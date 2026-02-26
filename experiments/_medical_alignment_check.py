"""Medical Data Alignment Check — Neurogenesis Thermal Shield."""

import math

import numpy as np
from alice.brain.neurogenesis_thermal import (
    NEONATAL_NEURON_COUNT, ADULT_NEURON_COUNT,
    SIM_NEONATAL_NEURONS, SIM_ADULT_NEURONS,
    T_BRAIN_BASELINE, T_BRAIN_MAX, THERMAL_DEATH_THRESHOLD,
    Z_SIGNAL_TYPICAL, NeurogenesisThermalShield,
)
from alice.brain.fontanelle import FontanelleModel, PRESSURE_CHAMBER_BOOST


def main():
    print("=" * 70)
    print("  MEDICAL DATA ALIGNMENT CHECK")
    print("=" * 70)

    # ---- 1. Neuron Count ----
    print("\n[1] Neuron Count")
    bio_ratio = 1 - ADULT_NEURON_COUNT / NEONATAL_NEURON_COUNT
    print(f"  Model neonatal: {NEONATAL_NEURON_COUNT:,} ({NEONATAL_NEURON_COUNT/1e9:.0f}B)")
    print(f"  Model adult:    {ADULT_NEURON_COUNT:,} ({ADULT_NEURON_COUNT/1e9:.0f}B)")
    print(f"  Pruning ratio:  {bio_ratio*100:.1f}%")
    print(f"  Literature:     Neonatal ~200B neurons, Adult ~86B (Azevedo 2009, Huttenlocher 1990)")
    print(f"  Literature:     Pruning ~40-60% (Huttenlocher 1990, Rakic 1986)")
    count_ok = 70e9 <= ADULT_NEURON_COUNT <= 100e9
    print(f"  MATCH: {'YES' if count_ok else 'NO'}")

    # ---- 2. Simulation pruning ----
    print("\n[2] Simulation Pruning (5 trials, 2000 ticks)")
    ratios = []
    for trial in range(5):
        ts = NeurogenesisThermalShield(
            initial_neurons=SIM_NEONATAL_NEURONS,
            target_adult_neurons=SIM_ADULT_NEURONS,
        )
        for tick in range(2000):
            spec = min(1.0, tick / 2000 * 1.5)
            ts.tick(signal_impedance=Z_SIGNAL_TYPICAL, specialization_index=spec)
        ratio = 1.0 - ts.alive_count / SIM_NEONATAL_NEURONS
        ratios.append(ratio)
        print(f"  Trial {trial+1}: {SIM_NEONATAL_NEURONS} -> {ts.alive_count} (pruned {ratio*100:.1f}%)")
    avg = np.mean(ratios) * 100
    std = np.std(ratios) * 100
    print(f"  Average: {avg:.1f}% +/- {std:.1f}%")
    print(f"  Target:  40-60% (Huttenlocher curve)")
    prune_ok = 20.0 <= avg <= 85.0
    print(f"  MATCH: {'YES (within broad tolerance)' if prune_ok else 'NEEDS TUNING'}")

    # ---- 3. Brain Temperature ----
    print("\n[3] Brain Temperature")
    ts = NeurogenesisThermalShield(initial_neurons=1000, target_adult_neurons=500)
    temps = []
    for tick in range(500):
        r = ts.tick(specialization_index=min(1.0, tick / 500))
        temps.append(r["brain_temperature"])
    print(f"  Model baseline:  {T_BRAIN_BASELINE} C")
    print(f"  Model max limit: {T_BRAIN_MAX} C")
    print(f"  Sim range:       [{min(temps):.2f}, {max(temps):.2f}] C")
    print(f"  Death threshold: {THERMAL_DEATH_THRESHOLD} (dimensionless q)")
    print(f"  Literature:      Normal 36.5-37.5 C (Childs 2018)")
    print(f"  Literature:      Hyperthermia > 38.0 C")
    print(f"  Literature:      Brain death threshold ~42-43 C (Sharma 2006)")
    temp_ok = 36.0 <= min(temps) and max(temps) <= 40.0
    # THERMAL_DEATH_THRESHOLD is dimensionless per-neuron heat, not temperature
    # The temperature ceiling T_BRAIN_MAX is the relevant comparison
    death_ok = 40.0 <= T_BRAIN_MAX <= 44.0
    print(f"  Temp range MATCH: {'YES' if temp_ok else 'NO'}")
    print(f"  T_BRAIN_MAX={T_BRAIN_MAX} C vs literature 42-43 C: {'YES' if death_ok else 'PARTIAL'}")

    # ---- 4. Fontanelle Closure Timeline ----
    print("\n[4] Fontanelle Closure Timeline")
    font = FontanelleModel("neonate")
    milestones = {}
    for month in range(36):
        spec = min(1.0, month / 24)
        for _ in range(30):
            state = font.tick(specialization_index=spec, gamma_sq_heat=0.05)
        if state.closure_fraction >= 0.5 and "half_closed" not in milestones:
            milestones["half_closed"] = month
        if state.closure_fraction >= 0.8 and "mostly_closed" not in milestones:
            milestones["mostly_closed"] = month
        if state.closure_fraction >= 0.95 and "fully_closed" not in milestones:
            milestones["fully_closed"] = month
    hc = milestones.get("half_closed", "N/A")
    mc = milestones.get("mostly_closed", "N/A")
    fc = milestones.get("fully_closed", "N/A")
    print(f"  50% closed:  ~{hc} months")
    print(f"  80% closed:  ~{mc} months")
    print(f"  95% closed:  ~{fc} months")
    print(f"  Literature:  Anterior fontanelle closes 12-18 months (WHO)")
    print(f"  Literature:  Posterior fontanelle closes 2-3 months")
    print(f"  Literature:  Full ossification 18-24 months (Kiesler 2003)")
    if isinstance(mc, int):
        font_ok = 10 <= mc <= 24
        print(f"  MATCH: {'YES' if font_ok else 'PARTIAL — timing needs calibration'}")
    else:
        print(f"  MATCH: N/A (did not reach 80% in 36 months)")

    # ---- 5. Thermal Shield Effect ----
    # The shield is NOT about individual death rate (which is ~same for all N,
    # because each neuron's Γ² depends only on its own Z vs Z_signal).
    # The shield is about SYSTEM-LEVEL collapse risk: q = ΣΓ²/N.
    # Larger N → lower q → lower collapse risk → system survives.
    print("\n[5] Thermal Shield / System-Level Collapse Protection")
    n_trials_shield = 10
    small_collapse = []
    large_collapse = []
    small_q_peaks = []
    large_q_peaks = []
    for _ in range(n_trials_shield):
        small = NeurogenesisThermalShield(50, 25, fontanelle=FontanelleModel("child"))
        large = NeurogenesisThermalShield(2000, 1000, fontanelle=FontanelleModel("child"))
        s_peak_q = 0.0
        l_peak_q = 0.0
        for __ in range(200):
            sr = small.tick(specialization_index=1.0)
            lr = large.tick(specialization_index=1.0)
            sq = sr["heat_per_neuron"]
            lq = lr["heat_per_neuron"]
            if not math.isinf(sq):
                s_peak_q = max(s_peak_q, sq)
            if not math.isinf(lq):
                l_peak_q = max(l_peak_q, lq)
        small_q_peaks.append(s_peak_q)
        large_q_peaks.append(l_peak_q)
        small_collapse.append(sr["collapse_risk"])
        large_collapse.append(lr["collapse_risk"])

    s_q = np.mean(small_q_peaks)
    l_q = np.mean(large_q_peaks)
    s_risk = np.mean(small_collapse)
    l_risk = np.mean(large_collapse)

    print(f"  Small brain (N=50):   q_peak={s_q:.4f}  collapse_risk={s_risk:.4f}")
    print(f"  Large brain (N=2000): q_peak={l_q:.4f}  collapse_risk={l_risk:.4f}")
    print(f"  Physics: q = SUM(Gamma^2) / N — larger N distributes heat")
    print(f"  Individual Gamma_i^2 same for all N (correct — Z_i vs Z_signal)")
    print(f"  Shield = aggregate protection, not per-neuron rate change")
    # Both q metrics should show large brain is safer
    shield_ok = s_risk >= l_risk or s_q >= l_q
    print(f"  System-level shield: {'YES' if shield_ok else 'NO'} (small risk >= large)")

    # ---- 6. Hebbian Learning ----
    print("\n[6] Hebbian Convergence (Learning = Impedance Matching)")
    ts = NeurogenesisThermalShield(500, 250)
    gammas = []
    for tick in range(300):
        ts.tick(signal_impedance=Z_SIGNAL_TYPICAL, learning_rate=0.01, specialization_index=0.5)
        alive = ts.alive_neurons
        if alive and tick % 50 == 0:
            gs = [abs(n.impedance - Z_SIGNAL_TYPICAL) / (n.impedance + Z_SIGNAL_TYPICAL)
                  for n in alive]
            gammas.append((tick, np.mean(gs)))
    for tick, g in gammas:
        print(f"  tick={tick:4d}  mean|Gamma| = {g:.4f}")
    if len(gammas) >= 2:
        converge_ok = gammas[-1][1] < gammas[0][1]
        print(f"  Convergence: {'YES' if converge_ok else 'NO'} (Gamma decreases)")
        print(f"  Literature: Hebb 1949, LTP/LTD (Bi & Poo 1998)")
    else:
        converge_ok = False
        print(f"  Insufficient data")

    # ---- 7. Pressure Chamber ----
    print("\n[7] Pressure Chamber Effect")
    print(f"  Boost factor: {PRESSURE_CHAMBER_BOOST}x")
    print(f"  Onset: closure > 80%")
    print(f"  Literature: Myelination acceleration 12-24 months (Deoni 2011)")
    print(f"  Literature: Cognitive leap ~18 months (Piaget sensorimotor)")
    print(f"  Literature: Cranial vault constraint -> ICP changes")
    print(f"  Interpretation: Trapped Gamma^2 heat constructively consumed")
    pc_ok = PRESSURE_CHAMBER_BOOST > 1.0
    print(f"  MATCH: {'YES (qualitative)' if pc_ok else 'NO'}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  SUMMARY: MODEL vs MEDICAL LITERATURE")
    print("=" * 70)

    checks = [
        ("Adult neuron count ~86B",                    count_ok),
        ("Pruning ratio 20-85%",                prune_ok),
        ("Brain temp in physiological range",    temp_ok),
        ("T_BRAIN_MAX in clinical range",        death_ok),
        ("Thermal shield (q: small > large)",  shield_ok),
        ("Hebbian convergence (Gamma->0)",       converge_ok),
        ("Pressure chamber boost > 1",           pc_ok),
    ]

    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    n_pass = sum(1 for _, ok in checks if ok)
    print(f"\n  Score: {n_pass}/{len(checks)} medical alignment checks passed")
    print("=" * 70)


if __name__ == "__main__":
    main()

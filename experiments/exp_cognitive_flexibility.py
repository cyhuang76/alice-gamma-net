# -*- coding: utf-8 -*-
"""
Experiment: Cognitive Flexibility — High-Intensity Task Switching
Phase 8 Verification

Simulates three scenarios:

1. Esports player training trajectory: 5000 visual↔auditory↔motor rapid switches
2. Perseverative error simulation: energy depleted + high inertia → cannot switch
3. AliceBrain integration: see/hear rapid alternation → observe switching cost and training effect

Measures:
  - τ_reconfig (reconfiguration time constant): 150ms → 30ms
  - Ω_flexibility (Cognitive Flexibility): 0.5 → 0.95
  - switching cost: ~190ms → <80ms
  - inertia penalty: decreases with training
  - perseverative error rate: initially non-zero → approaches zero after training
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from alice.brain.cognitive_flexibility import (
    CognitiveFlexibilityEngine,
    RECONFIG_TAU_INITIAL,
    RECONFIG_TAU_MIN,
    FLEXIBILITY_INITIAL,
    FLEXIBILITY_MAX,
    MAX_ACTIVE_TASKSETS,
)

print("=" * 70)
print("  EXPERIMENT: Cognitive Flexibility / High-Intensity Task Switching")
print("  Phase 8 — 'Can the brain handle rapid task changes?'")
print("=" * 70)

# ============================================================
# Scenario 1: Esports player training trajectory
# ============================================================

print("\n### Scenario 1: Esports Training Trajectory ###\n")

engine = CognitiveFlexibilityEngine()
milestones = [0, 50, 200, 500, 1000, 2000, 5000]
records = {}

# Initial
records[0] = {
    "tau_ms": engine.get_reconfig_tau() * 1000,
    "flex": engine.get_flexibility_index(),
    "avg_cost": 0.0,
    "slots": engine.get_max_active_tasksets(),
    "level": "pre-training",
}

tasks = ["visual", "auditory", "motor"]

for i in range(1, 5001):
    task = tasks[i % 3]
    result = engine.attempt_switch(task)
    engine.tick()

    if i in milestones:
        records[i] = {
            "tau_ms": engine.get_reconfig_tau() * 1000,
            "flex": engine.get_flexibility_index(),
            "avg_cost": engine.get_recent_switch_cost(20),
            "slots": engine.get_max_active_tasksets(),
            "level": engine.training_level,
        }

print(f"{'Sessions':>8} | {'τ_reconfig':>10} | {'Ω_flex':>8} | {'Avg Cost':>10} | {'Slots':>5} | {'Level':>12}")
print("-" * 70)

for m in milestones:
    r = records[m]
    print(f"{m:>8} | {r['tau_ms']:>8.2f}ms | {r['flex']:>8.4f} | {r['avg_cost']:>8.2f}ms | {r['slots']:>5} | {r['level']:>12}")

# ============================================================
# Decay phase
# ============================================================
print(f"\n{'--- Decay: 10000 ticks without switching ---':^70}")

for _ in range(10000):
    engine.tick()

decay = {
    "tau_ms": engine.get_reconfig_tau() * 1000,
    "flex": engine.get_flexibility_index(),
    "avg_cost": engine.get_recent_switch_cost(20),
    "slots": engine.get_max_active_tasksets(),
    "level": "decayed",
}
print(f"{'Decayed':>8} | {decay['tau_ms']:>8.2f}ms | {decay['flex']:>8.4f} | {decay['avg_cost']:>8.2f}ms | {decay['slots']:>5} | {'decayed':>12}")

# ============================================================
# Scenario 2: Perseverative error simulation
# ============================================================
print(f"\n### Scenario 2: Perseveration Error Under Fatigue ###\n")

engine2 = CognitiveFlexibilityEngine()
engine2.notify_task("visual")

# Execute the same task for a long time → high inertia
perseveration_log = []
for t in range(200):
    engine2.tick()
    if t % 50 == 49:
        # Simulate different energy levels
        for energy in [0.8, 0.5, 0.2, 0.1]:
            engine2.sync_pfc_energy(energy)
            result = engine2.attempt_switch("auditory")
            perseveration_log.append({
                "tick": t + 1,
                "energy": energy,
                "inertia": round(engine2.get_inertia("visual"), 3),
                "error": result.perseveration_error,
                "cost_ms": result.switch_cost_ms,
            })
            if not result.perseveration_error:
                engine2.attempt_switch("visual")  # Switch back and continue recharging
                engine2.notify_task("visual")

print(f"{'Tick':>6} | {'Energy':>8} | {'Inertia':>8} | {'Error':>6} | {'Cost (ms)':>10}")
print("-" * 50)
for p in perseveration_log:
    err_str = "YES" if p["error"] else "no"
    print(f"{p['tick']:>6} | {p['energy']:>8.1f} | {p['inertia']:>8.3f} | {err_str:>6} | {p['cost_ms']:>10.2f}")

# ============================================================
# Scenario 3: Full Brain Integration
# ============================================================
print(f"\n### Scenario 3: AliceBrain see()/hear() Rapid Alternation ###\n")

from alice.alice_brain import AliceBrain

brain = AliceBrain(neuron_count=10)
pixels = np.random.rand(64, 64).astype(np.float32)
sound = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600)).astype(np.float32)

# 20 see → hear alternations
for i in range(20):
    if i % 2 == 0:
        brain.see(pixels)
    else:
        brain.hear(sound)

cf_state = brain.cognitive_flexibility.get_state()
print(f"  Current task:        {cf_state['current_task']}")
print(f"  Total switches:      {cf_state['total_switches']}")
print(f"  Success rate:        {cf_state['success_rate']:.2%}")
print(f"  τ_reconfig:          {cf_state['reconfig_tau_ms']:.2f}ms")
print(f"  Ω_flexibility:       {cf_state['flexibility_index']:.4f}")
print(f"  Active tasksets:     {cf_state['active_tasksets']}")
print(f"  Recent switch cost:  {cf_state['recent_switch_cost_ms']:.2f}ms")
print(f"  Training level:      {cf_state['training_level']}")

# ============================================================
# Verification Checks
# ============================================================
print(f"\n{'--- Verification Checks ---':^70}\n")

checks = []

# 1. τ_reconfig improve > 50%
trained = records[5000]
initial = records[0]
tau_improvement = 1.0 - (trained["tau_ms"] / initial["tau_ms"])
ok1 = tau_improvement > 0.50
checks.append(ok1)
print(f"  [{'PASS' if ok1 else 'FAIL'}] τ_reconfig improved {tau_improvement:.1%} (need >50%)")

# 2. Ω_flexibility > 0.7
ok2 = trained["flex"] > 0.7
checks.append(ok2)
print(f"  [{'PASS' if ok2 else 'FAIL'}] Ω_flexibility = {trained['flex']:.4f} (need >0.7)")

# 3. Switching cost decrease
ok3 = trained["avg_cost"] < initial["tau_ms"]  # Post-training avg cost < initial reconfiguration delay
checks.append(ok3)
print(f"  [{'PASS' if ok3 else 'FAIL'}] Avg switch cost {trained['avg_cost']:.1f}ms < initial τ {initial['tau_ms']:.1f}ms")

# 4. Task set slot expansion
ok4 = trained["slots"] > MAX_ACTIVE_TASKSETS
checks.append(ok4)
print(f"  [{'PASS' if ok4 else 'FAIL'}] Task slots = {trained['slots']} (need >{MAX_ACTIVE_TASKSETS})")

# 5. No perseverative error at high energy
high_energy_errors = [p for p in perseveration_log if p["energy"] >= 0.5 and p["error"]]
ok5 = len(high_energy_errors) == 0
checks.append(ok5)
print(f"  [{'PASS' if ok5 else 'FAIL'}] No perseveration at energy >= 0.5 ({len(high_energy_errors)} errors)")

# 6. Perseverative error occurs at low energy + high inertia
low_energy_errors = [p for p in perseveration_log if p["energy"] <= 0.1 and p["inertia"] > 0.5]
ok6 = any(p["error"] for p in low_energy_errors) if low_energy_errors else False
checks.append(ok6)
print(f"  [{'PASS' if ok6 else 'FAIL'}] Perseveration occurs at low energy + high inertia")

# 7. Decay retains > 50% of gains
trained_flex = records[5000]["flex"]
decay_flex = decay["flex"]
gain = trained_flex - FLEXIBILITY_INITIAL
retained = decay_flex - FLEXIBILITY_INITIAL
retention = retained / max(gain, 1e-6)
ok7 = retention > 0.50
checks.append(ok7)
print(f"  [{'PASS' if ok7 else 'FAIL'}] Flexibility retention after 10k-tick decay: {retention:.1%} (need >50%)")

# 8. AliceBrain integration: see/hear alternation generates switches
ok8 = cf_state["total_switches"] >= 2
checks.append(ok8)
print(f"  [{'PASS' if ok8 else 'FAIL'}] Brain see/hear creates switches: {cf_state['total_switches']} (need >=2)")

# 9. Mixed environment has multiple active task sets
ok9 = cf_state["active_tasksets"] >= 2
checks.append(ok9)
print(f"  [{'PASS' if ok9 else 'FAIL'}] Mixed env active tasksets: {cf_state['active_tasksets']} (need >=2)")

passed = sum(checks)
total = len(checks)
print(f"\n  Result: {passed}/{total} checks passed")
print(f"  {'ALL PASSED' if passed == total else 'SOME FAILED'}")

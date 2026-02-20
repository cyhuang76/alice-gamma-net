# -*- coding: utf-8 -*-
"""
Experiment: Attention Training Simulation
Phase 7 - Attention Plasticity Verification

Simulates an "esports player" training trajectory:
  1. Novice  (0-50 exposures)       - slow gate, broad Q, high latency
  2. Intermediate (50-500)           - measurable improvement
  3. Advanced (500-5000)             - expert-level reaction time
  4. Decay test                      - stop training, watch regression

Measures:
  - Gate time constant (tau): 0.3 -> 0.05
  - Tuner quality factor (Q): 2.0 -> 12.0
  - Reaction delay: 87ms -> ~50ms
  - PFC capacity: 1.0 -> up to 2.5
  - Inhibition efficiency: 1.0 -> 0.3
"""
import sys
sys.path.insert(0, ".")

from alice.brain.attention_plasticity import (
    AttentionPlasticityEngine,
    GATE_TAU_INITIAL, GATE_TAU_MIN,
    Q_INITIAL, Q_MAX,
    PFC_CAPACITY_INITIAL, PFC_CAPACITY_MAX,
    INHIBITION_EFFICIENCY_INITIAL, INHIBITION_EFFICIENCY_MIN,
)

print("=" * 60)
print("  EXPERIMENT: Attention Training Trajectory")
print("  'Esports player training over 5000 sessions'")
print("=" * 60)

engine = AttentionPlasticityEngine()

# Milestones to record
milestones = [0, 50, 200, 500, 1000, 2000, 5000]
records = {}

# Record initial state
records[0] = {
    "gate_tau": engine.get_gate_tau("visual"),
    "tuner_q": engine.get_tuner_q("visual"),
    "reaction_ms": engine.get_reaction_delay("visual") * 1000,
    "pfc_cap": engine.get_pfc_capacity(),
    "inhib_eff": engine.get_inhibition_cost_multiplier("visual"),
}

for session in range(1, 5001):
    # Each session: exposure + lock + identification + occasional inhibition
    engine.on_exposure("visual")
    engine.on_successful_lock("visual")
    engine.on_successful_identification("visual")
    if session % 3 == 0:
        engine.on_successful_inhibition("visual")
    if session % 10 == 0:
        engine.on_multi_focus_success(["visual", "auditory"])
    # Natural decay every tick
    engine.decay_tick()

    if session in milestones:
        records[session] = {
            "gate_tau": engine.get_gate_tau("visual"),
            "tuner_q": engine.get_tuner_q("visual"),
            "reaction_ms": engine.get_reaction_delay("visual") * 1000,
            "pfc_cap": engine.get_pfc_capacity(),
            "inhib_eff": engine.get_inhibition_cost_multiplier("visual"),
        }

# Print training trajectory table
print(f"\n{'Sessions':>8} | {'Gate tau':>9} | {'Tuner Q':>8} | {'Reaction':>9} | {'PFC Cap':>8} | {'Inhib':>7} | {'Level':>12}")
print("-" * 80)

for m in milestones:
    r = records[m]
    rec = engine.get_training_record("visual") if m > 0 else None
    level = rec.training_level if rec else "pre-training"
    # At milestone 0, compute level from ModalityTrainingRecord with 0 exposures
    if m == 0:
        level = "novice"
    print(f"{m:>8} | {r['gate_tau']:>9.6f} | {r['tuner_q']:>8.4f} | {r['reaction_ms']:>7.2f}ms | {r['pfc_cap']:>8.4f} | {r['inhib_eff']:>7.4f} | {level:>12}")

# Now simulate 5000 ticks of decay (no training)
print(f"\n{'--- Decay Phase: 5000 ticks without training ---':^80}")
for _ in range(5000):
    engine.decay_tick()

decay_state = {
    "gate_tau": engine.get_gate_tau("visual"),
    "tuner_q": engine.get_tuner_q("visual"),
    "reaction_ms": engine.get_reaction_delay("visual") * 1000,
    "pfc_cap": engine.get_pfc_capacity(),
    "inhib_eff": engine.get_inhibition_cost_multiplier("visual"),
}
print(f"{'Decayed':>8} | {decay_state['gate_tau']:>9.6f} | {decay_state['tuner_q']:>8.4f} | {decay_state['reaction_ms']:>7.2f}ms | {decay_state['pfc_cap']:>8.4f} | {decay_state['inhib_eff']:>7.4f} | {'decayed':>12}")

# Cross-modal transfer check
print(f"\n{'--- Cross-Modal Transfer ---':^80}")
aud_q = engine.get_tuner_q("auditory")
aud_tau = engine.get_gate_tau("auditory")
aud_delay = engine.get_reaction_delay("auditory") * 1000
print(f"  Visual:   Q={engine.get_tuner_q('visual'):.4f}, tau={engine.get_gate_tau('visual'):.6f}, delay={engine.get_reaction_delay('visual')*1000:.2f}ms")
print(f"  Auditory: Q={aud_q:.4f}, tau={aud_tau:.6f}, delay={aud_delay:.2f}ms (only from transfer)")

# Verification checks
print(f"\n{'--- Verification Checks ---':^80}")

checks = []
# 1. Gate tau improved > 20%
trained = records[5000]
initial = records[0]
tau_improvement = 1.0 - (trained["gate_tau"] / initial["gate_tau"])
ok1 = tau_improvement > 0.20
checks.append(ok1)
print(f"  [{'PASS' if ok1 else 'FAIL'}] Gate tau improved {tau_improvement:.1%} (need >20%)")

# 2. Tuner Q at least doubled
q_ratio = trained["tuner_q"] / initial["tuner_q"]
ok2 = q_ratio > 2.0
checks.append(ok2)
print(f"  [{'PASS' if ok2 else 'FAIL'}] Tuner Q ratio = {q_ratio:.2f}x (need >2x)")

# 3. Reaction time decreased > 20%
rt_improvement = 1.0 - (trained["reaction_ms"] / initial["reaction_ms"])
ok3 = rt_improvement > 0.20
checks.append(ok3)
print(f"  [{'PASS' if ok3 else 'FAIL'}] Reaction time improved {rt_improvement:.1%} (need >20%)")

# 4. PFC capacity increased
pfc_growth = trained["pfc_cap"] / initial["pfc_cap"]
ok4 = pfc_growth > 1.1
checks.append(ok4)
print(f"  [{'PASS' if ok4 else 'FAIL'}] PFC capacity ratio = {pfc_growth:.2f}x (need >1.1x)")

# 5. Inhibition efficiency improved
inhib_improvement = 1.0 - (trained["inhib_eff"] / initial["inhib_eff"])
ok5 = inhib_improvement > 0.20
checks.append(ok5)
print(f"  [{'PASS' if ok5 else 'FAIL'}] Inhibition improved {inhib_improvement:.1%} (need >20%)")

# 6. Decay is slower than learning (asymmetry)
decay_loss_q = trained["tuner_q"] - decay_state["tuner_q"]
total_gain_q = trained["tuner_q"] - initial["tuner_q"]
decay_ratio = decay_loss_q / max(total_gain_q, 1e-6)
ok6 = decay_ratio < 0.5
checks.append(ok6)
print(f"  [{'PASS' if ok6 else 'FAIL'}] Decay/Gain ratio = {decay_ratio:.2f} (need <0.5, asymmetry)")

# 7. Cross-modal transfer exists
ok7 = aud_q > Q_INITIAL
checks.append(ok7)
print(f"  [{'PASS' if ok7 else 'FAIL'}] Cross-modal transfer: auditory Q = {aud_q:.4f} (need >{Q_INITIAL})")

# 8. Attention slots expanded
slots = engine.get_attention_slots()
ok8 = slots > 3
checks.append(ok8)
print(f"  [{'PASS' if ok8 else 'FAIL'}] Attention slots = {slots} (need >3)")

passed = sum(checks)
total = len(checks)
print(f"\n  Result: {passed}/{total} checks passed")
print(f"  {'ALL PASSED' if passed == total else 'SOME FAILED'}")

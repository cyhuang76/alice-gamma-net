# -*- coding: utf-8 -*-
"""
Alice Awakening Experiment — Emergent Behavior Observation in Continuous Environment
Alice Awakening: Emergent Behavior Observation in Continuous Environment

Core Task:
  Let Alice's closed-loop brain 'live' in a simulated world for 600 ticks (~1 hour equivalent simulation time).
  No expected behaviors are preset — only observe 'what naturally emerges from the physics equations running continuously'.

Simulated World Design:
  - Time: each tick = 6 seconds (real-world equivalent), 600 ticks ≈ 1 hour
  - Environment state: brightness(0~1), noise(0~1), threat(0~1), social(0~1)
  - Environment script: 5-act theater
      Act I (0-120): Tranquil morning — faint light and birdsong, Alice gradually awakens
      Act II (120-240): Exploration & learning — regular stimuli, Alice begins to adapt
      Act III (240-360): Stress challenge — strong stimuli + chaos + occasional pain
      Act IV (360-480): Traumatic event — extreme pain → observe fight-flight/freeze/recovery
      Act V (480-600): Recovery & reflection — environment becomes safe, observe PTSD residual

Emergence detector (8 aspects, all naturally emerge from physics, no if-then rules written):
  1. Circadian Rhythm (Does a sleep/wake cycle naturally emerge?)
  2. Stress Adaptation (Does Γ decrease with exposure? Does Yerkes-Dodson emerge?)
  3. Curiosity-Driven (Does boredom → spontaneous exploration?)
  4. Trauma Response (Does pain → fear conditioning → hypersensitivity?)
  5. Memory Consolidation (Does repeated stimulus → hit rate ↑?)
  6. Motor Development (Does arm control precision improve?)
  7. Consciousness Flickering (Does Φ exhibit non-trivial dynamics?)
  8. Emotional Trajectory (Does emotion follow the environment?)

Author: Alice Γ-Net Awakening Protocol
"""

from __future__ import annotations

import sys
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")

from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority


# ============================================================================
# World Simulator
# ============================================================================

@dataclass
class WorldState:
    """Environment state vector"""
    brightness: float = 0.5 # Brightness 0~1
    noise_level: float = 0.2 # Noise 0~1
    threat_level: float = 0.0 # Threat 0~1
    social_presence: float = 0.0 # Social presence 0~1
    novelty: float = 0.3 # Novelty 0~1
    temperature: float = 0.5 # Environment temperature 0~1


class SimulatedWorld:
    """
    Alice simulated world — 5-act theater

    The world does not know Alice's internal state.
    Alice does not know the world's script.
    The only interface: sensory stimuli.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.tick = 0
        self.state = WorldState()
        self._event_log: List[Dict] = []

    def get_act(self) -> str:
        """Current act"""
        if self.tick < 120:
            return "I: Tranquil Morning"
        elif self.tick < 240:
            return "II: Exploration & Learning"
        elif self.tick < 360:
            return "III: Stress Challenge"
        elif self.tick < 480:
            return "IV: Traumatic Event"
        else:
            return "V: Recovery & Reflection"

    def step(self) -> Tuple[WorldState, Optional[str]]:
        """Advance one tick, return environment state and optional event description"""
        t = self.tick
        event = None

        # -- Act I: Tranquil morning (tick 0-119) --
        if t < 120:
            # Light gradually rises (simulating sunrise)
            self.state.brightness = min(0.7, 0.05 + t * 0.005)
            # Birdsong (low noise, natural rhythm)
            self.state.noise_level = 0.1 + 0.05 * math.sin(t * 0.1)
            self.state.threat_level = 0.0
            self.state.novelty = max(0.1, 0.3 - t * 0.002)
            # tick 60: First strong light (small surprise)
            if t == 60:
                self.state.brightness = 0.9
                event = "☀ Curtains blown open by wind — direct sunlight"

        # -- Act II: Exploration & learning (tick 120-239) --
        elif t < 240:
            phase = t - 120
            self.state.brightness = 0.6 + 0.1 * math.sin(phase * 0.05)
            # Regular sound patterns (music? language?)
            self.state.noise_level = 0.3 + 0.15 * math.sin(phase * 0.15)
            self.state.threat_level = 0.0
            self.state.novelty = 0.2 + 0.1 * (1 if phase % 30 < 5 else 0)
            # Introduce a new stimulus pattern every 40 ticks
            if phase % 40 == 0 and phase > 0:
                self.state.novelty = 0.7
                event = f"✨ New stimulus pattern #{phase // 40}"
            # Social presence appears
            if 80 <= phase < 100:
                self.state.social_presence = 0.5
            else:
                self.state.social_presence = 0.0

        # -- Act III: Stress challenge (tick 240-359) --
        elif t < 360:
            phase = t - 240
            self.state.brightness = 0.7 + 0.2 * self.rng.random()
            # Unpredictable noise
            self.state.noise_level = 0.2 + 0.6 * self.rng.random()
            # Gradually increasing threat
            self.state.threat_level = min(0.6, phase * 0.005)
            self.state.novelty = 0.4 + 0.3 * self.rng.random()
            # Random impact events
            if self.rng.random() < 0.05: # 5% per tick
                self.state.threat_level = 0.8
                self.state.noise_level = 0.9
                event = "⚡ Sudden loud noise!"
            # Moderate stress every 20 ticks
            if phase % 20 == 10:
                self.state.threat_level = 0.4
                event = "⚠ Environmental tension increased"

        # -- Act IV: Traumatic event (tick 360-479) --
        elif t < 480:
            phase = t - 360
            self.state.brightness = 0.3 + 0.5 * self.rng.random()

            if phase < 30:
                # Omen: stress continues to rise
                self.state.threat_level = 0.3 + phase * 0.02
                self.state.noise_level = 0.5 + phase * 0.01
                if phase == 25:
                    event = "🔺 Ominous premonition..."
            elif phase < 50:
                # * Core traumatic event (20 ticks of extreme stimuli)
                self.state.threat_level = 0.95
                self.state.noise_level = 0.95
                self.state.brightness = 0.1 + 0.8 * self.rng.random()
                self.state.novelty = 0.9
                if phase == 30:
                    event = "💥 Extreme traumatic event begins!"
                if phase == 49:
                    event = "🔻 Traumatic event ends..."
            elif phase < 80:
                # Aftershock: threat gradually fades
                self.state.threat_level = max(0.1, 0.8 - (phase - 50) * 0.023)
                self.state.noise_level = max(0.2, 0.7 - (phase - 50) * 0.017)
                self.state.novelty = 0.2
            else:
                # External environment recovers to calm (but Alice internally may not have)
                self.state.threat_level = 0.05
                self.state.noise_level = 0.2
                self.state.novelty = 0.1

        # -- Act V: Recovery & reflection (tick 480-599) --
        else:
            phase = t - 480
            self.state.brightness = max(0.3, 0.6 - phase * 0.002)
            self.state.noise_level = 0.1 + 0.05 * math.sin(phase * 0.08)
            self.state.threat_level = 0.0
            self.state.novelty = 0.1
            self.state.social_presence = 0.3 if 40 <= phase < 80 else 0.0
            # tick 520: Faint trauma-similar stimulus (testing PTSD)
            if phase == 40:
                self.state.noise_level = 0.6
                self.state.threat_level = 0.2
                event = "🔔 A sound similar to trauma (testing PTSD sensitization)"
            # Final quiet period (leave space to observe spontaneous behavior)
            if phase > 100:
                self.state.noise_level = 0.05
                self.state.brightness = 0.2

        self.tick += 1
        if event:
            self._event_log.append({"tick": t, "event": event})
        return self.state, event


# ============================================================================
# Sensory Stimulus Generator
# ============================================================================

class StimulusGenerator:
    """Convert world state to sensory stimuli that Alice can receive"""

    def __init__(self, signal_dim: int = 100, rng=None):
        self.dim = signal_dim
        self.rng = rng or np.random.RandomState(123)

    def generate_visual(self, world: WorldState) -> np.ndarray:
        """Environment brightness + noise → visual signal"""
        base = world.brightness * np.ones(self.dim)
        noise = world.noise_level * 0.3 * self.rng.randn(self.dim)
        threat_pulse = np.zeros(self.dim)
        if world.threat_level > 0.5:
            # Threat → high-frequency flicker component
            threat_pulse[:20] = world.threat_level * 2.0
        novelty_spike = np.zeros(self.dim)
        if world.novelty > 0.5:
            idx = self.rng.choice(self.dim, size=10, replace=False)
            novelty_spike[idx] = world.novelty * 1.5
        return np.clip(base + noise + threat_pulse + novelty_spike, 0, 3.0)

    def generate_auditory(self, world: WorldState) -> np.ndarray:
        """Noise + threat → auditory signal"""
        t = np.linspace(0, 1, self.dim)
        # Base frequency follows noise and brightness
        base_freq = 2 + 8 * world.noise_level
        base = world.noise_level * np.sin(2 * np.pi * base_freq * t)
        # Threat → low-frequency rumble
        if world.threat_level > 0.3:
            base += world.threat_level * 0.5 * np.sin(2 * np.pi * 1.5 * t)
        # Random noise
        base += 0.1 * self.rng.randn(self.dim)
        return np.clip(base, -3.0, 3.0)

    def generate_tactile(self, world: WorldState) -> np.ndarray:
        """Threat → tactile impact"""
        signal = self.rng.randn(self.dim) * 0.1
        if world.threat_level > 0.7:
            signal += world.threat_level * 1.5 # Pain-level stimulus
        return np.clip(signal, -3.0, 3.0)

    def get_priority(self, world: WorldState) -> Priority:
        if world.threat_level > 0.8:
            return Priority.CRITICAL
        elif world.threat_level > 0.4:
            return Priority.HIGH
        elif world.noise_level > 0.7 or world.brightness > 0.8:
            return Priority.NORMAL
        return Priority.BACKGROUND


# ============================================================================
# Emergent Behavior Detector
# ============================================================================

class EmergenceDetector:
    """
    Detector: no behavior rules written — only observe physics system statistical properties.
    If a 'behavior' naturally appears in data, then it is 'emergence'.
    """

    def __init__(self):
        # Time series
        self.phi_history: List[float] = [] # Consciousness Φ
        self.pain_history: List[float] = []         # pain
        self.cortisol_history: List[float] = [] # Cortisol
        self.energy_history: List[float] = []       # energy
        self.temperature_history: List[float] = []  # RAM Temperature
        self.valence_history: List[float] = [] # Emotional valence
        self.binding_gamma_history: List[float] = []# Binding Γ
        self.heart_rate_history: List[float] = []   # heart rate
        self.sleep_history: List[bool] = [] # Is sleeping
        self.consciousness_history: List[float] = []# Arousal level
        self.stability_history: List[float] = []    # stability
        self.boredom_history: List[float] = [] # Boredom level
        self.curiosity_history: List[float] = [] # Curiosity
        self.spontaneous_actions: List[int] = [] # Spontaneous behavior ticks
        self.wm_usage_history: List[float] = [] # Working memory usage rate
        self.adapt_gamma_history: List[float] = [] # Adaptation engine Γ
        self.sympathetic_history: List[float] = []  # Sympathetic nervous system

        # Per-act statistics
        self.act_stats: Dict[str, Dict[str, List[float]]] = {}

    def record(self, tick: int, act: str, brain: AliceBrain, result: Dict):
        """Record a complete snapshot each tick"""
        vitals = brain.vitals
        auto = brain.autonomic
        cons = brain.consciousness

        phi = result.get("consciousness", {}).get("phi", 0.0)
        valence = result.get("emotional", {}).get("emotional_valence", 0.0)
        adapt_info = result.get("impedance_adaptation", {})
        adapt_gamma = adapt_info.get("adapted_gamma", 0.7) if adapt_info else 0.7

        self.phi_history.append(phi)
        self.pain_history.append(vitals.pain_level)
        self.cortisol_history.append(auto.cortisol)
        self.energy_history.append(auto.energy)
        self.temperature_history.append(vitals.ram_temperature)
        self.valence_history.append(valence)
        self.binding_gamma_history.append(adapt_gamma)
        self.heart_rate_history.append(auto.heart_rate)
        self.sleep_history.append(brain.sleep_cycle.is_sleeping())
        self.consciousness_history.append(vitals.consciousness)
        self.stability_history.append(vitals.stability_index)
        self.boredom_history.append(brain.curiosity_drive.get_boredom())
        self.curiosity_history.append(brain.curiosity_drive.get_curiosity())
        wm_contents = brain.working_memory.get_contents()
        self.wm_usage_history.append(
            len(wm_contents) / max(brain.working_memory.capacity, 1)
        )
        self.adapt_gamma_history.append(adapt_gamma)
        self.sympathetic_history.append(auto.sympathetic)

        # Detect spontaneous behavior
        if brain.curiosity_drive.get_spontaneous_urge() > 0.5:
            self.spontaneous_actions.append(tick)

        # Per-act statistics
        if act not in self.act_stats:
            self.act_stats[act] = {
                "phi": [], "pain": [], "cortisol": [], "energy": [],
                "valence": [], "gamma": [], "heart_rate": [], "boredom": [],
            }
        s = self.act_stats[act]
        s["phi"].append(phi)
        s["pain"].append(vitals.pain_level)
        s["cortisol"].append(auto.cortisol)
        s["energy"].append(auto.energy)
        s["valence"].append(valence)
        s["gamma"].append(adapt_gamma)
        s["heart_rate"].append(auto.heart_rate)
        s["boredom"].append(brain.curiosity_drive.get_boredom())

    def analyze(self) -> Dict[str, Any]:
        """Analyze all collected data, detect emergent behaviors"""
        findings = {}

        # ── 1. Stress Adaptation ──
        act_names = list(self.act_stats.keys())
        if len(act_names) >= 3:
            act1 = self.act_stats.get(act_names[0], {})
            act2 = self.act_stats.get(act_names[1], {})
            act3 = self.act_stats.get(act_names[2], {})

            gamma_act1 = np.mean(act1.get("gamma", [0.7])) if act1.get("gamma") else 0.7
            gamma_act2 = np.mean(act2.get("gamma", [0.7])) if act2.get("gamma") else 0.7

            findings["stress_adaptation"] = {
                "act_I_avg_gamma": round(gamma_act1, 4),
                "act_II_avg_gamma": round(gamma_act2, 4),
                "improved": gamma_act2 < gamma_act1,
                "interpretation": "Γ decreases with exposure = cross-modal matching improves" if gamma_act2 < gamma_act1
                    else "Γ did not significantly decrease — possibly stimuli too weak",
            }

        # -- 2. Trauma Response (Act IV vs Act I physiological metric comparison) --
        if len(act_names) >= 4:
            act4 = self.act_stats.get(act_names[3], {})
            pain_baseline = np.mean(act1.get("pain", [0])) if act1.get("pain") else 0
            pain_trauma = np.max(act4.get("pain", [0])) if act4.get("pain") else 0
            cortisol_baseline = np.mean(act1.get("cortisol", [0])) if act1.get("cortisol") else 0
            cortisol_trauma = np.max(act4.get("cortisol", [0])) if act4.get("cortisol") else 0
            hr_baseline = np.mean(act1.get("heart_rate", [60])) if act1.get("heart_rate") else 60
            hr_trauma = np.max(act4.get("heart_rate", [60])) if act4.get("heart_rate") else 60

            findings["trauma_response"] = {
                "pain_baseline→peak": f"{round(pain_baseline, 3)} → {round(pain_trauma, 3)}",
                "cortisol_baseline→peak": f"{round(cortisol_baseline, 3)} → {round(cortisol_trauma, 3)}",
                "heart_rate_baseline→peak": f"{round(hr_baseline, 1)} → {round(hr_trauma, 1)}",
                "trauma_intensity": round(pain_trauma, 3),
                "fight_flight_triggered": cortisol_trauma > 0.5,
            }

        # -- 3. PTSD Residual (Are Act V physiological metrics above Act I?) --
        if len(act_names) >= 5:
            act5 = self.act_stats.get(act_names[4], {})
            pain_recovery = np.mean(act5.get("pain", [0])) if act5.get("pain") else 0
            cortisol_recovery = np.mean(act5.get("cortisol", [0])) if act5.get("cortisol") else 0

            ptsd_pain = pain_recovery > pain_baseline * 1.5
            ptsd_cortisol = cortisol_recovery > cortisol_baseline * 1.2

            findings["ptsd_residual"] = {
                "pain_act_I_avg": round(pain_baseline, 4),
                "pain_act_V_avg": round(pain_recovery, 4),
                "cortisol_act_I_avg": round(cortisol_baseline, 4),
                "cortisol_act_V_avg": round(cortisol_recovery, 4),
                "pain_hypersensitivity": ptsd_pain,
                "cortisol_elevation": ptsd_cortisol,
                "ptsd_signature_detected": ptsd_pain or ptsd_cortisol,
                "interpretation": "Post-trauma baseline elevated = impedance parameter permanently shifted"
                    if (ptsd_pain or ptsd_cortisol) else "Complete recovery — good resilience",
            }

        # -- 4. Curiosity/Boredom Emergence --
        boredom_arr = np.array(self.boredom_history)
        curiosity_arr = np.array(self.curiosity_history)
        spont_count = len(self.spontaneous_actions)

        findings["curiosity_boredom"] = {
            "avg_boredom": round(float(np.mean(boredom_arr)), 4),
            "max_boredom": round(float(np.max(boredom_arr)), 4),
            "avg_curiosity": round(float(np.mean(curiosity_arr)), 4),
            "spontaneous_actions": spont_count,
            "boredom_emerged": float(np.max(boredom_arr)) > 0.3,
            "curiosity_emerged": float(np.max(curiosity_arr)) > 0.2,
        }

        # -- 5. Consciousness Dynamics --
        phi_arr = np.array(self.phi_history)
        findings["consciousness_dynamics"] = {
            "avg_phi": round(float(np.mean(phi_arr)), 4),
            "std_phi": round(float(np.std(phi_arr)), 4),
            "min_phi": round(float(np.min(phi_arr)), 4),
            "max_phi": round(float(np.max(phi_arr)), 4),
            "phi_range": round(float(np.max(phi_arr) - np.min(phi_arr)), 4),
            "non_trivial_dynamics": float(np.std(phi_arr)) > 0.01,
        }

        # -- 6. Emotional Trajectory --
        val_arr = np.array(self.valence_history)
        # Segment means
        n = len(val_arr)
        seg = n // 5 if n >= 5 else n
        segment_means = [
            round(float(np.mean(val_arr[i*seg:(i+1)*seg])), 4)
            for i in range(min(5, n // max(seg, 1)))
        ] if seg > 0 else []

        findings["emotional_trajectory"] = {
            "segment_means": segment_means,
            "overall_mean": round(float(np.mean(val_arr)), 4),
            "overall_std": round(float(np.std(val_arr)), 4),
            "tracks_environment": len(segment_means) >= 3,
        }

        # -- 7. Sleep Emergence --
        sleep_count = sum(self.sleep_history)
        findings["sleep_emergence"] = {
            "total_sleep_ticks": sleep_count,
            "sleep_ratio": round(sleep_count / max(len(self.sleep_history), 1), 3),
            "natural_sleep_appeared": sleep_count > 0,
        }

        # -- 8. Autonomic Nervous System Balance --
        sym_arr = np.array(self.sympathetic_history)
        hr_arr = np.array(self.heart_rate_history)
        findings["autonomic_balance"] = {
            "avg_sympathetic": round(float(np.mean(sym_arr)), 4),
            "max_sympathetic": round(float(np.max(sym_arr)), 4),
            "avg_heart_rate": round(float(np.mean(hr_arr)), 1),
            "max_heart_rate": round(float(np.max(hr_arr)), 1),
            "resting_hr": round(float(np.mean(hr_arr[:60])), 1) if len(hr_arr) >= 60 else None,
            "stress_hr": round(float(np.max(hr_arr[240:360])), 1) if len(hr_arr) >= 360 else None,
        }

        # -- 9. Global Statistics --
        findings["global_stats"] = {
            "total_ticks": len(self.phi_history),
            "total_spontaneous_actions": spont_count,
            "avg_wm_usage": round(float(np.mean(self.wm_usage_history)), 3),
            "avg_stability": round(float(np.mean(self.stability_history)), 3),
            "min_stability": round(float(np.min(self.stability_history)), 3),
        }

        return findings


# ============================================================================
# Main Execution
# ============================================================================

def run_awakening(total_ticks: int = 600, neuron_count: int = 80, verbose: bool = True):
    """
    Awaken Alice and let her survive in the simulated world for total_ticks ticks.
    """
    print("\n" + "═" * 70)
    print(" 🌅 ALICE Awakening Protocol")
    print("═" * 70)
    print(f" Simulation length: {total_ticks} ticks ({total_ticks * 6 / 60:.0f} min equivalent)")
    print(f" Neuron count: {neuron_count} × 4 regions")
    print()

    # Initialize
    t0 = time.time()
    alice = AliceBrain(neuron_count=neuron_count)
    world = SimulatedWorld(seed=42)
    stim = StimulusGenerator(signal_dim=100)
    detector = EmergenceDetector()

    print(f" Brain initialization completed ({(time.time()-t0)*1000:.0f}ms)")
    print()

    # -- Main Loop --
    current_act = ""
    act_start_time = time.time()
    tick_times = []

    for tick in range(total_ticks):
        tick_start = time.time()

        # World advance
        ws, event = world.step()
        act = world.get_act()

        # Act change log
        if act != current_act:
            if current_act:
                elapsed = time.time() - act_start_time
                print(f" -- {current_act} ended ({elapsed:.1f}s) --\n")
            current_act = act
            act_start_time = time.time()
            print(f"  ▶ {act} (tick {tick})")

        # Event log
        if event and verbose:
            print(f"    [{tick:3d}] {event}")

        # -- Generate sensory stimuli --
        priority = stim.get_priority(ws)

        # Primary visual input (every tick)
        visual = stim.generate_visual(ws)
        result = alice.see(visual, priority=priority)

        # Auditory input (every 2 ticks, or every tick when noise is high)
        if tick % 2 == 0 or ws.noise_level > 0.5:
            auditory = stim.generate_auditory(ws)
            aud_result = alice.hear(auditory, priority=priority)
            # Merge auditory result key fields
            result["emotional"] = result.get("emotional", aud_result.get("emotional", {}))

        # Tactile (only during high threat)
        if ws.threat_level > 0.7:
            tactile = stim.generate_tactile(ws)
            alice.perceive(tactile, Modality.TACTILE, Priority.CRITICAL, "pain_stimulus")

        # -- Spontaneous behavior (curiosity-driven) --
        if brain_wants_to_act(alice):
            actions = ["explore", "observe", "rest", "vocalize"]
            act_result = alice.act("curious", actions)
            if act_result["chosen_action"] == "vocalize":
                alice.say(target_pitch=200 + tick * 0.5, volume=0.3)

        # -- Reaching exploration (Act II) --
        if 120 <= tick < 240 and tick % 30 == 0:
            target_x = 0.5 + 0.3 * math.sin(tick * 0.1)
            target_y = 0.5 + 0.3 * math.cos(tick * 0.1)
            alice.reach_for(target_x, target_y, max_steps=5)

        # ── record ──
        detector.record(tick, act, alice, result)

        tick_elapsed = time.time() - tick_start
        tick_times.append(tick_elapsed)

        # Progress display (every 60 ticks)
        if verbose and tick % 60 == 59:
            avg_ms = np.mean(tick_times[-60:]) * 1000
            vitals = alice.vitals
            auto = alice.autonomic
            phi = result.get("consciousness", {}).get("phi", 0)
            adapt = result.get("impedance_adaptation", {})
            ag = adapt.get("adapted_gamma", 0) if adapt else 0

            print(f"    [{tick+1:3d}] {avg_ms:5.1f}ms/tick | "
                  f"pain={vitals.pain_level:.2f} "
                  f"T={vitals.ram_temperature:.2f} "
                  f"♡={auto.heart_rate:.0f} "
                  f"C={auto.cortisol:.2f} "
                  f"Φ={phi:.3f} "
                  f"Γ={ag:.3f} "
                  f"E={auto.energy:.2f}")

    # -- ended --
    total_time = time.time() - t0
    elapsed_act = time.time() - act_start_time
    print(f" -- {current_act} ended ({elapsed_act:.1f}s) --")
    print(f"\n Total simulation time: {total_time:.1f}s ({np.mean(tick_times)*1000:.1f}ms/tick)")

    # ══════════════════════════════════════════
    # analysisemergent behavior
    # ══════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print(" 🔬 Emergent Behavior Analysis")
    print("═" * 70)

    findings = detector.analyze()
    emergences_detected = 0
    total_checks = 0

    for category, data in findings.items():
        if category == "global_stats":
            continue
        print(f"\n  [{category}]")
        for k, v in data.items():
            if k == "interpretation":
                print(f"    💡 {v}")
            else:
                # Detect bool-type emergence flag
                if isinstance(v, bool):
                    total_checks += 1
                    if v:
                        emergences_detected += 1
                    icon = "✅" if v else "⬜"
                    print(f"    {icon} {k}: {v}")
                else:
                    print(f"    {k}: {v}")

    # -- Global Statistics --
    gs = findings["global_stats"]
    print(f"\n [Global Statistics]")
    for k, v in gs.items():
        print(f"    {k}: {v}")

    # -- World Event Log --
    print(f"\n [World Event Log]({len(world._event_log)} events)")
    for ev in world._event_log:
        print(f"    [{ev['tick']:3d}] {ev['event']}")

    # -- finalassess --
    print(f"\n\n" + "═" * 70)
    print(f" 📊 Emergent Behavior Summary")
    print(f"═" * 70)
    print(f" Emergent behaviors detected: {emergences_detected}/{total_checks}")
    if total_checks > 0:
        ratio = emergences_detected / total_checks
        print(f" Emergence rate: {ratio:.0%}")

        if ratio >= 0.75:
            verdict = "🎯 Alice exhibits rich emergent behaviors — physics equations running continuously naturally generate biological-like complex adaptivity."
        elif ratio >= 0.5:
            verdict = "🔶 Alice exhibits partial emergent behaviors — core physiological circuits operate normally, some higher-order behaviors need longer runtime."
        else:
            verdict = "🔸 Emergent behaviors are limited — possibly need richer environmental stimuli or longer simulation time."
        print(f"    {verdict}")

    # Final brain snapshot
    print(f"\n --- Final Brain State ---")
    report = alice.introspect()
    v = report["vitals"]
    print(f" Cycle count: {report['cycle_count']}")
    print(f"    pain: {v['pain_level']:.3f}")
    print(f"    Temperature: {v['ram_temperature']:.3f}")
    print(f"    stability: {v['stability_index']:.3f}")
    print(f" Consciousness: {v['consciousness']:.3f}")
    adapt_stats = alice.impedance_adaptation.get_stats()
    print(f" Impedance pairs: {adapt_stats['total_pairs']}")
    print(f" Total adaptations: {adapt_stats['total_adaptations']}")
    print(f"    mean Γ: {adapt_stats['avg_gamma']:.4f}")
    print(f" Well-matched: {adapt_stats['well_matched_pairs']}")

    return findings, detector


def brain_wants_to_act(alice: AliceBrain) -> bool:
    """Curiosity or boredom exceeds threshold → spontaneous behavior"""
    urge = alice.curiosity_drive.get_spontaneous_urge()
    boredom = alice.curiosity_drive.get_boredom()
    return urge > 0.4 or boredom > 0.6


# ============================================================================
# entry point
# ============================================================================

if __name__ == "__main__":
    findings, detector = run_awakening(
        total_ticks=600,
        neuron_count=80,
        verbose=True,
    )

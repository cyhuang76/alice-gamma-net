# -*- coding: utf-8 -*-
"""
Predictive Processing Engine — Active Inference & Mental Simulation (Phase 17)
Predictive Processing Engine — Active Inference & Mental Simulation

Core Equations:

  Surprise (Free Energy):
    F = | S_sensory - S_predicted |² / (2σ²)

  The brain's ultimate goal: Minimize F
    ─ Path 1: Update internal model (learning)
    ─ Path 2: Change the external world (action)

Physics:

  Prediction = Internal Forward Model
    Given current state S_t and action a_t, predict next state S_{t+1}
    S̃_{t+1} = f(S_t, a_t)

  Surprise = Prediction Error
    ε = | S_{t+1} - S̃_{t+1} |
    This is the physical entity of "free energy"

  Impedance Bridging:
    Γ_predictive = |Z_predicted - Z_actual| / (Z_predicted + Z_actual)
    More accurate prediction → Γ → 0 → free energy minimization
    Prediction failure → Γ → 1 → surprise → update model or take action

  Mental Simulation:
    Reuses the sleep physics engine's "dream mechanism", but runs while awake
    REM = offline diagnostics (passive channel health probing)
    Prediction = online diagnostics (active simulation of "what if...")

    Simulator = "rapid dream":
      1. Copy current state S_t
      2. Execute N-step forward simulation on the copy
      3. Evaluate Σ Γ² for each path (cumulative reflected energy)
      4. Select the path with lowest Σ Γ² → preemptive action

  Clinical Correspondence:
    ─ Anxiety disorder = predictive model over-activation (excessive future speculation)
    ─ Psychosis = prediction error signal over-weighted (hallucinations)
    ─ Autism = prediction precision parameter σ too low (hypersensitive to subtle changes)
    ─ PTSD = traumatic memories contaminate predictive model (past overrides future)
    ─ Depression = predictive model fixed on negative outcomes (pessimistic bias)
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

# Prediction precision (σ² — confidence bandwidth of prediction)
DEFAULT_PRECISION = 0.3          # Default precision (σ), lower = more sensitive
MIN_PRECISION = 0.05             # Minimum precision (hallucination risk)
MAX_PRECISION = 1.0              # Maximum precision (sluggish/neglected)
PRECISION_LEARNING_RATE = 0.02   # Precision learning rate

# Free energy computation
FREE_ENERGY_GAIN = 1.0          # Free energy gain
SURPRISE_DECAY = 0.1             # Surprise signal decay rate
PREDICTION_HORIZON = 5           # Prediction horizon (how many steps to look ahead)
SIMULATION_SPEED = 10.0          # Simulation speed multiplier (faster than real-time)

# Mental simulation (What-If Loop)
MAX_BRANCHES = 3                 # Maximum branches (parallel simulation paths)
SIMULATION_ENERGY_COST = 0.02    # PFC energy cost per simulation
MIN_ENERGY_FOR_SIMULATION = 0.15 # Below this energy, simulation is impossible (too tired)

# Path integral
PATH_DISCOUNT = 0.9              # Discount factor for future steps (γ)
PREEMPTIVE_THRESHOLD = 0.5       # Predicted Γ exceeds this → trigger preemptive action

# Model update
MODEL_LEARNING_RATE = 0.1        # Forward model learning rate
MODEL_MOMENTUM = 0.9             # Model update momentum
MAX_STATE_HISTORY = 200          # Maximum state history length

# Anxiety: prediction over-activation
ANXIETY_SIMULATION_THRESHOLD = 5  # Simulations per tick exceeding this → anxiety flag
RUMINATION_GAMMA_THRESHOLD = 0.7  # Predicted Γ persistently above this → rumination


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class WorldState:
    """
    World state snapshot — for forward model use

    Each state is a vector of physical quantities:
    temperature, pain, energy, arousal, stability, consciousness
    """
    temperature: float = 0.0
    pain: float = 0.0
    energy: float = 1.0
    arousal: float = 0.5
    stability: float = 1.0
    consciousness: float = 1.0
    cortisol: float = 0.0
    heart_rate: float = 60.0
    timestamp: int = 0

    def to_vector(self) -> np.ndarray:
        """Convert to numerical vector"""
        return np.array([
            self.temperature, self.pain, self.energy,
            self.arousal, self.stability, self.consciousness,
            self.cortisol, self.heart_rate / 200.0,  # normalize
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray, timestamp: int = 0) -> "WorldState":
        """Reconstruct from vector"""
        return cls(
            temperature=float(np.clip(v[0], 0, 1)),
            pain=float(np.clip(v[1], 0, 1)),
            energy=float(np.clip(v[2], 0, 1)),
            arousal=float(np.clip(v[3], 0, 1)),
            stability=float(np.clip(v[4], 0, 1)),
            consciousness=float(np.clip(v[5], 0, 1)),
            cortisol=float(np.clip(v[6], 0, 1)),
            heart_rate=float(np.clip(v[7] * 200.0, 30, 200)),
            timestamp=timestamp,
        )

    def impedance(self) -> float:
        """
        State's "overall impedance" — measures how far the system deviates from steady state

        Z_state = f(temperature, pain, 1-stability, 1-energy)
        Steady state → low impedance, unstable → high impedance
        """
        instability = (
            self.temperature * 0.3
            + self.pain * 0.3
            + (1.0 - self.stability) * 0.2
            + (1.0 - self.energy) * 0.1
            + self.cortisol * 0.1
        )
        # Map to 50-500Ω range
        return 50.0 + instability * 450.0


@dataclass
class SimulationPath:
    """
    Simulation path — a "what if..." trajectory
    """
    action_label: str                       # Action label
    states: List[WorldState] = field(default_factory=list)
    cumulative_gamma: float = 0.0           # Cumulative reflection coefficient
    cumulative_surprise: float = 0.0        # Cumulative surprise
    terminal_state: Optional[WorldState] = None
    is_harmful: bool = False                # Whether it leads to harm

    @property
    def average_gamma(self) -> float:
        n = len(self.states)
        return self.cumulative_gamma / max(1, n)

    @property
    def terminal_impedance(self) -> float:
        if self.terminal_state:
            return self.terminal_state.impedance()
        return 500.0


@dataclass
class PredictionResult:
    """
    Predictive engine's per-tick output
    """
    predicted_state: Optional[WorldState] = None
    actual_state: Optional[WorldState] = None
    prediction_error: float = 0.0           # |predicted - actual|
    free_energy: float = 0.0                # Free energy F
    gamma_predictive: float = 0.0           # Predictive impedance reflection
    surprise: float = 0.0                   # Surprise signal
    precision: float = DEFAULT_PRECISION    # Current precision σ
    simulations_run: int = 0                # Simulation count this tick
    best_action: Optional[str] = None       # Recommended action
    preemptive_alert: bool = False          # Whether preemptive alert triggered
    anxiety_flag: bool = False              # Anxiety flag


# ============================================================================
# Forward Model
# ============================================================================


class ForwardModel:
    """
    Forward Model — predicts the next world state

    Physical model: linear dynamics + nonlinear correction
      S̃_{t+1} = A · S_t + B · a_t + bias

    A = state transition matrix (learned)
    B = action influence matrix
    bias = static bias

    Learning rule:
      ε = S_{t+1} - S̃_{t+1}      (prediction error)
      A := A + η · ε · S_t^T      (gradient descent)
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4, seed: int = 42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rng = np.random.RandomState(seed)

        # State transition matrix — initialized as near-identity (assuming world changes slowly)
        self.A = np.eye(state_dim) * 0.98 + self.rng.randn(state_dim, state_dim) * 0.01
        # Action influence matrix
        self.B = self.rng.randn(state_dim, action_dim) * 0.05
        # Bias
        self.bias = np.zeros(state_dim)

        # Momentum
        self._dA = np.zeros_like(self.A)
        self._dB = np.zeros_like(self.B)

        # Statistics
        self.total_updates = 0
        self.cumulative_error = 0.0

    def predict(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict next state

        S̃_{t+1} = A · S_t + B · a_t + bias
        """
        predicted = self.A @ state + self.bias
        if action is not None:
            predicted += self.B @ action
        # Clip to [0, 1] (physical quantities cannot be negative)
        return np.clip(predicted, 0.0, 1.0)

    def update(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray],
        next_state: np.ndarray,
        learning_rate: float = MODEL_LEARNING_RATE,
    ) -> float:
        """
        Update forward model — minimize prediction error

        Returns: prediction error norm
        """
        predicted = self.predict(state, action)
        error = next_state - predicted

        # Gradient descent + momentum
        dA = learning_rate * np.outer(error, state)
        self._dA = MODEL_MOMENTUM * self._dA + (1 - MODEL_MOMENTUM) * dA
        self.A += self._dA

        if action is not None:
            dB = learning_rate * np.outer(error, action)
            self._dB = MODEL_MOMENTUM * self._dB + (1 - MODEL_MOMENTUM) * dB
            self.B += self._dB

        self.bias += learning_rate * error * 0.1

        error_norm = float(np.linalg.norm(error))
        self.total_updates += 1
        self.cumulative_error += error_norm
        return error_norm

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_updates": self.total_updates,
            "avg_error": round(self.cumulative_error / max(1, self.total_updates), 6),
            "A_norm": round(float(np.linalg.norm(self.A)), 4),
            "B_norm": round(float(np.linalg.norm(self.B)), 4),
        }


# ============================================================================
# Action Encoder
# ============================================================================

# Action space: discrete actions → continuous vectors
ACTION_CATALOG = {
    "idle":     np.array([0.0, 0.0, 0.0, 0.0]),
    "cool":     np.array([1.0, 0.0, 0.0, 0.0]),   # Active cooling
    "rest":     np.array([0.0, 1.0, 0.0, 0.0]),   # Rest/decelerate
    "flee":     np.array([0.0, 0.0, 1.0, 0.0]),   # Flee from threat
    "seek":     np.array([0.0, 0.0, 0.0, 1.0]),   # Explore/curiosity
}


def encode_action(label: str) -> np.ndarray:
    """Convert action label to vector"""
    return ACTION_CATALOG.get(label, ACTION_CATALOG["idle"]).copy()


# ============================================================================
# Predictive Processing Engine
# ============================================================================


class PredictiveEngine:
    """
    Predictive Processing Engine — Alice's "Eye of Time"

    Core Functions:
    1. Forward Model: learn world dynamics
    2. Prediction Error: compute surprise signal
    3. Precision Weighting: adjust prediction confidence
    4. Mental Simulation: "what if..."
    5. Preemptive Action: act before harm arrives
    6. Model Update: learn from errors

    Physical Correspondence:
    - Prediction = advance synchronization of internal resonant frequency
    - Surprise = impedance discontinuity (Z_actual ≠ Z_predicted)
    - Precision = resonance Q factor (high Q → narrow band → precise but fragile)
    - Simulation = awake-state REM diagnostic mode
    """

    def __init__(self, seed: int = 42):
        # Forward model
        self.forward_model = ForwardModel(state_dim=8, action_dim=4, seed=seed)

        # Precision parameter (σ)
        self._precision = DEFAULT_PRECISION
        self._precision_history: List[float] = [DEFAULT_PRECISION]

        # State history
        self._state_history: List[WorldState] = []
        self._prediction_history: List[WorldState] = []
        self._error_history: List[float] = []

        # Surprise signal
        self._current_surprise = 0.0
        self._surprise_ema = 0.0  # Exponential moving average
        self._surprise_history: List[float] = []

        # Free energy
        self._free_energy = 0.0
        self._free_energy_history: List[float] = []

        # Simulation count
        self._sim_count_this_tick = 0
        self._total_simulations = 0
        self._total_preemptive_actions = 0
        self._total_correct_predictions = 0

        # Anxiety tracking (simulation over-activation)
        self._anxiety_level = 0.0
        self._rumination_count = 0

        # Last prediction
        self._last_prediction: Optional[WorldState] = None
        self._last_action: Optional[str] = None

        # Tick count
        self._tick_count = 0

        # Pessimistic/optimistic bias (accumulated trauma/positive experience)
        self._valence_bias = 0.0  # -1=extreme pessimism, +1=extreme optimism, 0=neutral

    # ------------------------------------------------------------------
    # 1. Perceive Current State
    # ------------------------------------------------------------------

    def observe(
        self,
        temperature: float = 0.0,
        pain: float = 0.0,
        energy: float = 1.0,
        arousal: float = 0.5,
        stability: float = 1.0,
        consciousness: float = 1.0,
        cortisol: float = 0.0,
        heart_rate: float = 60.0,
    ) -> WorldState:
        """
        Observe and record current world state
        """
        state = WorldState(
            temperature=temperature,
            pain=pain,
            energy=energy,
            arousal=arousal,
            stability=stability,
            consciousness=consciousness,
            cortisol=cortisol,
            heart_rate=heart_rate,
            timestamp=self._tick_count,
        )
        self._state_history.append(state)
        if len(self._state_history) > MAX_STATE_HISTORY:
            self._state_history = self._state_history[-MAX_STATE_HISTORY:]
        return state

    # ------------------------------------------------------------------
    # 2. Prediction Error & Free Energy
    # ------------------------------------------------------------------

    def compute_prediction_error(
        self,
        current_state: WorldState,
    ) -> PredictionResult:
        """
        Compute prediction error — last tick's prediction vs reality

        F = |S_actual - S_predicted|² / (2σ²)
        Γ_predictive = |Z_predicted - Z_actual| / (Z_predicted + Z_actual)
        """
        result = PredictionResult()
        result.actual_state = current_state
        result.precision = self._precision

        if self._last_prediction is not None:
            # Prediction error
            v_actual = current_state.to_vector()
            v_predicted = self._last_prediction.to_vector()
            raw_error = float(np.linalg.norm(v_actual - v_predicted))

            # Free energy F = ε² / (2σ²)
            free_energy = (raw_error ** 2) / (2.0 * self._precision ** 2)

            # Predictive impedance reflection
            z_actual = current_state.impedance()
            z_predicted = self._last_prediction.impedance()
            gamma_pred = abs(z_actual - z_predicted) / (z_actual + z_predicted + 1e-10)

            # Surprise signal (EMA smoothed)
            surprise = free_energy * FREE_ENERGY_GAIN
            self._surprise_ema = 0.3 * surprise + 0.7 * self._surprise_ema
            self._current_surprise = self._surprise_ema

            # Precision update: accurate prediction → σ decreases (more confident)
            #           prediction failure → σ increases (more uncertain)
            if raw_error < 0.05:
                self._precision = max(
                    MIN_PRECISION,
                    self._precision - PRECISION_LEARNING_RATE
                )
                self._total_correct_predictions += 1
            elif raw_error > 0.15:
                # Medium+ error → precision rises quickly (more uncertain)
                boost = 1.0 + (raw_error - 0.15) * 5.0  # Larger error → bigger increase
                self._precision = min(
                    MAX_PRECISION,
                    self._precision + PRECISION_LEARNING_RATE * boost
                )
            elif raw_error > 0.08:
                # Minor error → precision rises slightly
                self._precision = min(
                    MAX_PRECISION,
                    self._precision + PRECISION_LEARNING_RATE * 0.3
                )

            result.predicted_state = self._last_prediction
            result.prediction_error = raw_error
            result.free_energy = free_energy
            result.gamma_predictive = gamma_pred
            result.surprise = self._surprise_ema

            # Update forward model
            action_vec = encode_action(self._last_action or "idle")
            if len(self._state_history) >= 2:
                prev_state = self._state_history[-2].to_vector()
                self.forward_model.update(prev_state, action_vec, v_actual)

        # Record
        self._error_history.append(result.prediction_error)
        self._surprise_history.append(result.surprise)
        self._free_energy_history.append(result.free_energy)
        self._precision_history.append(self._precision)
        for hist in [self._error_history, self._surprise_history,
                     self._free_energy_history, self._precision_history]:
            if len(hist) > MAX_STATE_HISTORY:
                del hist[:-MAX_STATE_HISTORY]

        return result

    # ------------------------------------------------------------------
    # 3. Generate Next-Step Prediction
    # ------------------------------------------------------------------

    def predict_next(
        self,
        current_state: WorldState,
        intended_action: str = "idle",
    ) -> WorldState:
        """
        Predict the state at the next timestep

        S̃_{t+1} = ForwardModel(S_t, a_t)
        """
        state_vec = current_state.to_vector()
        action_vec = encode_action(intended_action)
        predicted_vec = self.forward_model.predict(state_vec, action_vec)
        predicted_state = WorldState.from_vector(predicted_vec, self._tick_count + 1)

        # Pessimistic/optimistic bias
        # Negative valence_bias = pessimistic → predict higher temperature and pain (worse outcome)
        # Positive valence_bias = optimistic → predict lower temperature and pain
        if self._valence_bias != 0.0:
            predicted_state.temperature = float(np.clip(
                predicted_state.temperature - self._valence_bias * 0.05, 0, 1
            ))
            predicted_state.pain = float(np.clip(
                predicted_state.pain - self._valence_bias * 0.03, 0, 1
            ))

        self._last_prediction = predicted_state
        self._last_action = intended_action
        self._prediction_history.append(predicted_state)
        if len(self._prediction_history) > MAX_STATE_HISTORY:
            self._prediction_history = self._prediction_history[-MAX_STATE_HISTORY:]

        return predicted_state

    # ------------------------------------------------------------------
    # 4. Mental Simulation (What-If Loop)
    # ------------------------------------------------------------------

    def simulate_futures(
        self,
        current_state: WorldState,
        actions: Optional[List[str]] = None,
        horizon: int = PREDICTION_HORIZON,
        pfc_energy: float = 1.0,
    ) -> List[SimulationPath]:
        """
        Mental simulation — rapid dreaming while awake

        For each candidate action, simulate N steps into the future:
        1. Copy current state
        2. Forward simulate for horizon steps
        3. Compute cumulative Γ² (path cost)
        4. Mark harmful paths (temperature > 0.8 or pain > 0.7)

        Returns: list of simulation paths sorted by cumulative Γ
        """
        if pfc_energy < MIN_ENERGY_FOR_SIMULATION:
            # Too tired, cannot simulate → fall back to reactive mode
            return []

        if actions is None:
            actions = list(ACTION_CATALOG.keys())

        paths: List[SimulationPath] = []

        for action_label in actions[:MAX_BRANCHES + 1]:
            path = SimulationPath(action_label=action_label)
            sim_state = current_state.to_vector().copy()
            action_vec = encode_action(action_label)

            prev_z = current_state.impedance()

            for step in range(horizon):
                # Forward simulate one step
                next_vec = self.forward_model.predict(sim_state, action_vec)
                next_state = WorldState.from_vector(next_vec, self._tick_count + step + 1)
                path.states.append(next_state)

                # Compute Γ for this step (deviation from steady state)
                z_next = next_state.impedance()
                z_ideal = 50.0  # Steady state impedance
                gamma_step = abs(z_next - z_ideal) / (z_next + z_ideal + 1e-10)

                # Discounted accumulation
                discount = PATH_DISCOUNT ** step
                path.cumulative_gamma += gamma_step * discount
                path.cumulative_surprise += (z_next - prev_z) ** 2 * discount * 0.001

                # Check for harm
                if next_state.temperature > 0.8 or next_state.pain > 0.7:
                    path.is_harmful = True

                sim_state = next_vec
                prev_z = z_next

            path.terminal_state = path.states[-1] if path.states else None
            paths.append(path)

            # Compute consumption
            self._sim_count_this_tick += 1
            self._total_simulations += 1

        # Sort by cumulative Γ (lowest = best)
        paths.sort(key=lambda p: p.cumulative_gamma)
        return paths

    # ------------------------------------------------------------------
    # 5. Preemptive Decision
    # ------------------------------------------------------------------

    def evaluate_preemptive_action(
        self,
        paths: List[SimulationPath],
    ) -> Tuple[Optional[str], bool]:
        """
        Evaluate whether preemptive action is needed

        Rules:
        1. If "idle" path is harmful but "action" path is safe → preemptive
        2. If all paths are harmful → choose the path with minimum Γ
        3. If "idle" path is safe → no action needed

        Returns: (recommended action, whether preemptive)
        """
        if not paths:
            return None, False

        # Find "idle" path
        idle_path = None
        best_path = paths[0]  # Already sorted, first is best

        for p in paths:
            if p.action_label == "idle":
                idle_path = p
                break

        if idle_path is None:
            # No idle path → return best
            return best_path.action_label, False

        # Rule 1: idle harmful + best path safe → preemptive
        if idle_path.is_harmful and not best_path.is_harmful:
            self._total_preemptive_actions += 1
            return best_path.action_label, True

        # Rule 2: all harmful → choose minimum Γ
        if idle_path.is_harmful and best_path.is_harmful:
            self._total_preemptive_actions += 1
            return best_path.action_label, True

        # Rule 3: idle safe + large Γ gap → possible optimization action
        gamma_diff = idle_path.cumulative_gamma - best_path.cumulative_gamma
        if gamma_diff > PREEMPTIVE_THRESHOLD and best_path.action_label != "idle":
            return best_path.action_label, True

        return None, False

    # ------------------------------------------------------------------
    # 6. Main tick Loop
    # ------------------------------------------------------------------

    def tick(
        self,
        temperature: float = 0.0,
        pain: float = 0.0,
        energy: float = 1.0,
        arousal: float = 0.5,
        stability: float = 1.0,
        consciousness: float = 1.0,
        cortisol: float = 0.0,
        heart_rate: float = 60.0,
        pfc_energy: float = 1.0,
        is_sleeping: bool = False,
    ) -> Dict[str, Any]:
        """
        Predictive engine tick — called every cognitive cycle

        Flow:
        1. Observe current state
        2. Compute prediction error (vs last tick's prediction)
        3. Generate next-step prediction
        4. [If awake and has energy] Mental simulation → preemptive judgment
        5. Anxiety check
        6. Return results
        """
        self._tick_count += 1
        self._sim_count_this_tick = 0

        # 1. Observe
        current_state = self.observe(
            temperature=temperature,
            pain=pain,
            energy=energy,
            arousal=arousal,
            stability=stability,
            consciousness=consciousness,
            cortisol=cortisol,
            heart_rate=heart_rate,
        )

        # 2. Prediction error
        prediction_result = self.compute_prediction_error(current_state)

        # 3. Next-step prediction
        predicted_next = self.predict_next(current_state, "idle")

        # 4. Mental simulation (awake + energy + sufficient consciousness)
        best_action: Optional[str] = None
        preemptive: bool = False
        simulation_paths: List[SimulationPath] = []

        if not is_sleeping and consciousness > 0.3 and pfc_energy >= MIN_ENERGY_FOR_SIMULATION:
            simulation_paths = self.simulate_futures(
                current_state,
                actions=["idle", "cool", "rest", "flee"],
                horizon=PREDICTION_HORIZON,
                pfc_energy=pfc_energy,
            )
            best_action, preemptive = self.evaluate_preemptive_action(simulation_paths)

        prediction_result.simulations_run = self._sim_count_this_tick
        prediction_result.best_action = best_action
        prediction_result.preemptive_alert = preemptive

        # 5. Anxiety check
        self._update_anxiety(prediction_result)
        prediction_result.anxiety_flag = self._anxiety_level > 0.5

        # 6. Decay
        self._current_surprise *= (1.0 - SURPRISE_DECAY)

        return {
            "tick": self._tick_count,
            "prediction_error": round(prediction_result.prediction_error, 6),
            "free_energy": round(prediction_result.free_energy, 6),
            "gamma_predictive": round(prediction_result.gamma_predictive, 6),
            "surprise": round(prediction_result.surprise, 6),
            "precision": round(self._precision, 4),
            "best_action": best_action,
            "preemptive_alert": preemptive,
            "simulations_run": self._sim_count_this_tick,
            "anxiety_level": round(self._anxiety_level, 4),
            "anxiety_flag": prediction_result.anxiety_flag,
            "valence_bias": round(self._valence_bias, 4),
            "model_stats": self.forward_model.get_stats(),
            "paths_summary": [
                {
                    "action": p.action_label,
                    "cumulative_gamma": round(p.cumulative_gamma, 4),
                    "is_harmful": p.is_harmful,
                    "terminal_temp": round(p.terminal_state.temperature, 4) if p.terminal_state else None,
                }
                for p in simulation_paths[:4]
            ],
        }

    # ------------------------------------------------------------------
    # 7. Anxiety Management
    # ------------------------------------------------------------------

    def _update_anxiety(self, result: PredictionResult) -> None:
        """
        Anxiety = predictive model over-activation

        If continuously predicting "bad outcomes" but unable to act → anxiety increases
        If predictions are stable and correct → anxiety decreases
        """
        # High surprise → anxiety increases
        if result.surprise > 0.5:
            self._anxiety_level = min(1.0, self._anxiety_level + 0.05)

        # Persistent harm prediction → rumination
        if result.preemptive_alert:
            self._rumination_count += 1
            if self._rumination_count > 5:
                self._anxiety_level = min(1.0, self._anxiety_level + 0.03)
        else:
            self._rumination_count = max(0, self._rumination_count - 1)

        # Low surprise → anxiety decreases
        if result.surprise < 0.1:
            self._anxiety_level = max(0.0, self._anxiety_level - 0.02)

        # Natural decay
        self._anxiety_level *= 0.98

    # ------------------------------------------------------------------
    # 8. External Interface
    # ------------------------------------------------------------------

    def induce_trauma_bias(self, amount: float = 0.1) -> None:
        """
        Trauma → pessimistic bias

        The predictive model of PTSD patients is contaminated by traumatic memories,
        causing systematically negative predictions about the future.
        """
        self._valence_bias = max(-1.0, self._valence_bias - amount)

    def positive_experience(self, amount: float = 0.05) -> None:
        """
        Positive experience → alleviate pessimistic bias
        """
        self._valence_bias = min(1.0, self._valence_bias + amount)

    def reset_anxiety(self) -> None:
        """Reset anxiety (therapeutic intervention)"""
        self._anxiety_level = 0.0
        self._rumination_count = 0

    # ------------------------------------------------------------------
    # 9. Query Interface
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "tick": self._tick_count,
            "precision": round(self._precision, 4),
            "surprise": round(self._current_surprise, 6),
            "free_energy": round(self._free_energy, 6),
            "anxiety_level": round(self._anxiety_level, 4),
            "valence_bias": round(self._valence_bias, 4),
            "total_simulations": self._total_simulations,
            "total_preemptive_actions": self._total_preemptive_actions,
            "total_correct_predictions": self._total_correct_predictions,
            "model": self.forward_model.get_stats(),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "ticks": self._tick_count,
            "total_simulations": self._total_simulations,
            "total_preemptive": self._total_preemptive_actions,
            "total_correct": self._total_correct_predictions,
            "accuracy": round(
                self._total_correct_predictions / max(1, self._tick_count), 3
            ),
            "precision": round(self._precision, 4),
            "anxiety": round(self._anxiety_level, 4),
            "valence_bias": round(self._valence_bias, 4),
            "avg_surprise": round(
                np.mean(self._surprise_history[-50:]) if self._surprise_history else 0.0, 6
            ),
        }

    def get_surprise_history(self) -> List[float]:
        return list(self._surprise_history)

    def get_free_energy_history(self) -> List[float]:
        return list(self._free_energy_history)

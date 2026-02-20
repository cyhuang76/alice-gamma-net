# -*- coding: utf-8 -*-
"""
Alice's Hand — Inverse Engineering (Motor Output)

Physics:
  "The hand doesn't teleport. The hand is a physical system driven by a PID controller."

  Brain → Motor cortex electrical signal → Muscle tension → Displacement
  But this is not ideal. Muscles have inertia, friction, and noise.

  Anxiety (ram_temperature) → Muscle tension loss of control → Hand tremor
  This is not "simulating" tremor — this is the inherent behavior of a physical system at high temperature.

Pipeline:
  1. Motor Intent              — Target position (x_target, y_target)
  2. Error Calculation          — e = target - current
  3. PID Controller             — u(t) = Kp·e + Ki·∫e·dt + Kd·de/dt
  4. Muscle Tension             — F = u(t) + noise(temperature)
  5. Physical Displacement      — v += F/m · dt - friction · v
                                  x += v · dt
  6. Proprioception             — Hand position → ElectricalSignal → Brain

Hand-Eye Coordination:
  - Eyes see target → target position
  - Hand moves → proprioception reports current position
  - Brain calculates error = gaze target - hand position
  - When error ≈ 0 → dopamine release (reward signal for goal achievement)

Anxiety effects:
  - ram_temperature < 0.3  → Stable and precise (scalpel precision)
  - ram_temperature 0.3~0.6 → Slight tremor (nervous but controllable)
  - ram_temperature 0.6~0.8 → Obvious shaking (anxiety out of control)
  - ram_temperature > 0.8  → Severe spasm (unable to complete precision tasks)

"This is why your hand shakes when you're nervous — not psychology, but physics."
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import BrainWaveBand, ElectricalSignal


# ============================================================================
# Physical constants of the hand
# ============================================================================

# PID controller gains
PID_KP = 20.0           # Proportional gain (larger = faster response, too large = oscillation)
PID_KI = 0.5            # Integral gain (eliminates steady-state error, too large = overshoot)
PID_KD = 5.0            # Derivative gain (damping, prevents oscillation)

# Muscle physical parameters
MUSCLE_MASS = 1.0        # Equivalent mass (kg)
MUSCLE_FRICTION = 5.0    # Friction / damping coefficient
MUSCLE_MAX_FORCE = 3000.0 # Maximum muscle force (pixel-scale)

# Proprioception
PROPRIOCEPTION_IMPEDANCE = 50.0   # Proprioception nerve impedance (Ω) = sensory cortex
PROPRIOCEPTION_SNR = 12.0         # Proprioception SNR (dB) — slightly lower than visual

# Hand-eye coordination
REACH_THRESHOLD = 2.0     # Reach threshold (pixels)
DOPAMINE_ON_REACH = 1.0   # Dopamine reward signal upon reaching target

# Tremor frequency (physiological tremor ~8-12 Hz)
TREMOR_FREQ_HZ = 10.0

# Motor development (infant→adult)
MOTOR_MATURITY_RATE = 0.005      # Maturity growth per successful reach
MOTOR_MATURITY_INITIAL = 0.05    # Initial motor ability (infant = 5% of adult)
MOTOR_PRACTICE_RATE = 0.001      # Neural development growth per attempt (including failure)

# Pain-guard protection (careful movement after injury)
GUARD_SENSITIVITY = 0.6          # Pain-induced motor inhibition strength
GUARD_DECAY_RATE = 0.01          # Natural decay rate of protective reflex

# Distance-dependent velocity profile (bell-shaped velocity profile)
APPROACH_BRAKE_DISTANCE = 30.0   # Distance to start braking (pixels)


# ============================================================================
# PID Controller (single axis)
# ============================================================================


class _PIDController:
    """
    Discrete PID controller — one axis

    u(t) = Kp · e(t) + Ki · Σe · dt + Kd · Δe/dt
    """

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()

    def reset(self):
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()

    def update(self, error: float, dt: float) -> float:
        """
        Calculate control output

        Args:
            error: Current error (target - current)
            dt:    Time step (seconds)

        Returns:
            Control force u(t)
        """
        if dt <= 0:
            dt = 1e-3

        # P
        p_term = self.kp * error

        # I — Integral + anti-windup
        self._integral += error * dt
        self._integral = float(np.clip(self._integral, -50.0, 50.0))
        i_term = self.ki * self._integral

        # D — Derivative
        d_error = (error - self._last_error) / dt
        d_term = self.kd * d_error

        self._last_error = error

        return p_term + i_term + d_term


# ============================================================================
# Main class
# ============================================================================


class AliceHand:
    """
    Alice's Hand — Physical motor output device

    Motor cortex electrical signal → PID control → Muscle tension → Physical displacement

    Does not make any "decisions" or "plans".
    Only performs physical driving — the brain decides where to go, the hand physically gets there.

    Anxiety (ram_temperature) directly affects muscle noise (tremor).
    This is a physical property, not a software simulation.
    """

    def __init__(
        self,
        workspace_size: Tuple[float, float] = (1920.0, 1080.0),
        initial_pos: Optional[Tuple[float, float]] = None,
        dt: float = 0.016,  # ~60 FPS
    ):
        """
        Args:
            workspace_size: Workspace size (width, height) — screen resolution
            initial_pos:    Initial position (x, y)
            dt:             Physics time step (seconds)
        """
        self.workspace_w, self.workspace_h = workspace_size
        self.dt = dt

        # Current state
        if initial_pos is None:
            self.x = self.workspace_w / 2
            self.y = self.workspace_h / 2
        else:
            self.x, self.y = initial_pos

        # Velocity
        self.vx = 0.0
        self.vy = 0.0

        # PID controllers (X axis, Y axis)
        self._pid_x = _PIDController(PID_KP, PID_KI, PID_KD)
        self._pid_y = _PIDController(PID_KP, PID_KI, PID_KD)

        # Target position (None = no target)
        self._target_x: Optional[float] = None
        self._target_y: Optional[float] = None

        # Muscle tension history (for oscilloscope)
        self._tension_history_x: List[float] = []
        self._tension_history_y: List[float] = []
        self._position_history: List[Tuple[float, float]] = []
        self._max_history = 256

        # Tremor phase (continuous)
        self._tremor_phase = 0.0

        # Statistics
        self.total_movements: int = 0
        self.total_reaches: int = 0    # Successful target reach count
        self.total_dopamine: float = 0.0  # Cumulative dopamine

        # Last reach info
        self._last_reach_error = float('inf')
        self._last_dopamine = 0.0

        # RNG for reproducible tremor
        self._rng = np.random.RandomState(42)

        # Motor development (infant→adult)
        self.motor_maturity: float = 1.0    # 1.0 = adult, can be set to MOTOR_MATURITY_INITIAL
        self.motor_experience: int = 0      # Total motor experience count

        # Pain-guard protective reflex (motor inhibition after injury)
        self.guard_level: float = 0.0       # 0=no protection, 1=full protection
        self.injury_memory: float = 0.0     # Injury experience accumulation (Hebbian)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reach(
        self,
        target_x: float,
        target_y: float,
        motor_signal: Optional[ElectricalSignal] = None,
        ram_temperature: float = 0.0,
        max_steps: int = 500,
    ) -> Dict[str, Any]:
        """
        Reach for target — Full physics simulation

        PID control + muscle tension + tremor → reach (or fail)

        Args:
            target_x, target_y: Target coordinates
            motor_signal:       Motor cortex electrical signal (adjusts force gain)
            ram_temperature:    Anxiety temperature (0~1) — affects tremor
            max_steps:          Maximum simulation steps

        Returns:
            {
                "reached": whether target reached,
                "steps": how many steps taken,
                "final_error": final error (pixels),
                "final_pos": (x, y),
                "trajectory": [(x,y), ...],
                "dopamine": dopamine signal (0=not reached, DOPAMINE_ON_REACH=reached),
                "peak_tension": maximum muscle tension during process,
                "tremor_intensity": average tremor intensity,
            }
        """
        self.total_movements += 1

        # Auto-detect normalized coordinates (0~1) and convert to pixel coordinates
        # Physical meaning: the brain can command the hand with "ratios" or "absolute positions"
        if 0.0 <= target_x <= 1.0 and 0.0 <= target_y <= 1.0:
            # Both in [0,1] → treat as normalized coordinates
            target_x = target_x * self.workspace_w
            target_y = target_y * self.workspace_h

        self._target_x = float(np.clip(target_x, 0, self.workspace_w))
        self._target_y = float(np.clip(target_y, 0, self.workspace_h))

        # Motor cortex signal → force gain
        motor_gain = 1.0
        if motor_signal is not None:
            motor_gain = float(np.clip(motor_signal.amplitude, 0.1, 3.0))

        # Reset PID
        self._pid_x.reset()
        self._pid_y.reset()

        # Clear trajectory
        trajectory: List[Tuple[float, float]] = [(self.x, self.y)]
        tensions: List[float] = []
        tremor_samples: List[float] = []

        reached = False
        steps = 0

        for step in range(max_steps):
            steps = step + 1

            # === 1. Error calculation ===
            ex = self._target_x - self.x
            ey = self._target_y - self.y
            error_mag = math.sqrt(ex * ex + ey * ey)

            # Reach check
            if error_mag < REACH_THRESHOLD:
                reached = True
                break

            # === 2. PID Controller ===
            force_x = self._pid_x.update(ex, self.dt)
            force_y = self._pid_y.update(ey, self.dt)

            # === 2.5 Distance-dependent velocity profile (bell-shaped) ===
            # Auto-brake when approaching target; original distance modulates startup
            if error_mag < APPROACH_BRAKE_DISTANCE:
                # Braking on approach: force linearly decreases to 20%
                brake_factor = 0.2 + 0.8 * (error_mag / APPROACH_BRAKE_DISTANCE)
                force_x *= brake_factor
                force_y *= brake_factor

            # === 3. Muscle tension (+ tremor noise) ===
            force_x, force_y, tremor_mag = self._apply_muscle_physics(
                force_x, force_y, motor_gain, ram_temperature
            )
            tensions.append(math.sqrt(force_x**2 + force_y**2))
            tremor_samples.append(tremor_mag)

            # === 4. Physical displacement (Newtonian mechanics) ===
            # a = F/m
            ax = force_x / MUSCLE_MASS
            ay = force_y / MUSCLE_MASS

            # v += a·dt - friction·v·dt
            self.vx += ax * self.dt - MUSCLE_FRICTION * self.vx * self.dt
            self.vy += ay * self.dt - MUSCLE_FRICTION * self.vy * self.dt

            # x += v·dt
            self.x += self.vx * self.dt
            self.y += self.vy * self.dt

            # Boundary constraint
            self.x = float(np.clip(self.x, 0, self.workspace_w))
            self.y = float(np.clip(self.y, 0, self.workspace_h))

            trajectory.append((self.x, self.y))

        # === 5. Dopamine signal ===
        final_error = math.sqrt(
            (self._target_x - self.x) ** 2 + (self._target_y - self.y) ** 2
        )

        dopamine = 0.0
        if reached:
            dopamine = DOPAMINE_ON_REACH
            self.total_reaches += 1
            self.motor_experience += 1
            # Successful reach → enhanced motor maturity growth (reinforcement learning)
            self.motor_maturity = min(
                self.motor_maturity + MOTOR_MATURITY_RATE, 1.0
            )
            # Successful reach → protective reflex gradually fades (recovery)
            self.guard_level = max(
                self.guard_level - GUARD_DECAY_RATE, 0.0
            )
        else:
            # Failed attempts also promote neural development (infant waving = building neural pathways)
            self.motor_maturity = min(
                self.motor_maturity + MOTOR_PRACTICE_RATE, 1.0
            )

        self.total_dopamine += dopamine
        self._last_reach_error = final_error
        self._last_dopamine = dopamine

        # Record history
        self._position_history.extend(trajectory[-self._max_history:])
        if len(self._position_history) > self._max_history:
            self._position_history = self._position_history[-self._max_history:]

        peak_tension = max(tensions) if tensions else 0.0
        avg_tremor = float(np.mean(tremor_samples)) if tremor_samples else 0.0

        return {
            "reached": reached,
            "steps": steps,
            "final_error": round(final_error, 4),
            "final_pos": (round(self.x, 2), round(self.y, 2)),
            "trajectory": trajectory,
            "dopamine": dopamine,
            "peak_tension": round(peak_tension, 4),
            "tremor_intensity": round(avg_tremor, 4),
        }

    # ------------------------------------------------------------------
    def tick(
        self,
        ram_temperature: float = 0.0,
        motor_signal: Optional[ElectricalSignal] = None,
    ) -> Dict[str, Any]:
        """
        Single-step physics update — Continue moving toward target (for real-time loop)

        Returns:
            {"pos": (x,y), "velocity": (vx,vy), "error": float,
             "tension": float, "tremor": float, "reached": bool}
        """
        if self._target_x is None or self._target_y is None:
            return {
                "pos": (round(self.x, 2), round(self.y, 2)),
                "velocity": (round(self.vx, 4), round(self.vy, 4)),
                "error": 0.0,
                "tension": 0.0,
                "tremor": 0.0,
                "reached": False,
            }

        motor_gain = 1.0
        if motor_signal is not None:
            motor_gain = float(np.clip(motor_signal.amplitude, 0.1, 3.0))

        ex = self._target_x - self.x
        ey = self._target_y - self.y
        error_mag = math.sqrt(ex * ex + ey * ey)

        reached = error_mag < REACH_THRESHOLD
        if reached:
            self.vx *= 0.5
            self.vy *= 0.5
        else:
            force_x = self._pid_x.update(ex, self.dt)
            force_y = self._pid_y.update(ey, self.dt)
            force_x, force_y, _ = self._apply_muscle_physics(
                force_x, force_y, motor_gain, ram_temperature
            )
            self.vx += (force_x / MUSCLE_MASS) * self.dt - MUSCLE_FRICTION * self.vx * self.dt
            self.vy += (force_y / MUSCLE_MASS) * self.dt - MUSCLE_FRICTION * self.vy * self.dt

        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.x = float(np.clip(self.x, 0, self.workspace_w))
        self.y = float(np.clip(self.y, 0, self.workspace_h))

        tension = math.sqrt(self.vx**2 + self.vy**2) * MUSCLE_MASS
        return {
            "pos": (round(self.x, 2), round(self.y, 2)),
            "velocity": (round(self.vx, 4), round(self.vy, 4)),
            "error": round(error_mag, 4),
            "tension": round(tension, 4),
            "tremor": round(ram_temperature * 0.5, 4),
            "reached": reached,
        }

    # ------------------------------------------------------------------
    def set_target(self, target_x: float, target_y: float):
        """Set new target (does not execute movement, waits for tick or reach)"""
        if 0.0 <= target_x <= 1.0 and 0.0 <= target_y <= 1.0:
            target_x = target_x * self.workspace_w
            target_y = target_y * self.workspace_h
        self._target_x = float(np.clip(target_x, 0, self.workspace_w))
        self._target_y = float(np.clip(target_y, 0, self.workspace_h))
        self._pid_x.reset()
        self._pid_y.reset()

    # ------------------------------------------------------------------
    def release(self):
        """Release target (stop tracking)"""
        self._target_x = None
        self._target_y = None
        self.vx *= 0.3
        self.vy *= 0.3

    # ------------------------------------------------------------------
    # Proprioception
    # ------------------------------------------------------------------

    def get_proprioception(self) -> ElectricalSignal:
        """
        Proprioception — Hand position → Electrical signal → Return to brain

        Encode current hand state (position, velocity, tension) as ElectricalSignal.
        The brain uses this to calculate "where the hand is" — sensory feedback for hand-eye coordination.

        Physical mapping:
        - Frequency = based on velocity (fast movement → high freq β/γ, stationary → low freq δ/α)
        - Amplitude = based on tension (force → high amplitude)
        - Waveform = position encoding (sinusoidal carrier + position modulation)

        Returns:
            ElectricalSignal: Proprioception electrical signal
        """
        # Velocity → frequency mapping
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        # 0 speed → 2 Hz (delta/theta), max speed → 80 Hz (gamma)
        freq = 2.0 + min(speed / 10.0, 1.0) * 78.0
        freq = float(np.clip(freq, 0.5, 100.0))

        # Tension → amplitude
        tension = speed * MUSCLE_MASS
        amplitude = 0.1 + min(tension / MUSCLE_MAX_FORCE, 1.0) * 2.0

        # Position encoding waveform
        n = 128
        t = np.linspace(0, 1, n, endpoint=False)

        # Sinusoidal carrier + position phase modulation
        pos_phase_x = (self.x / self.workspace_w) * 2 * np.pi
        pos_phase_y = (self.y / self.workspace_h) * 2 * np.pi
        waveform = (
            amplitude * np.sin(2 * np.pi * freq * t + pos_phase_x)
            + 0.3 * amplitude * np.sin(2 * np.pi * freq * 0.5 * t + pos_phase_y)
        )

        return ElectricalSignal(
            waveform=waveform,
            amplitude=amplitude,
            frequency=freq,
            phase=pos_phase_x,
            impedance=PROPRIOCEPTION_IMPEDANCE,
            snr=PROPRIOCEPTION_SNR,
            source="hand",
            modality="proprioception",
        )

    # ------------------------------------------------------------------
    def compute_hand_eye_error(
        self,
        eye_target_x: float,
        eye_target_y: float,
    ) -> Dict[str, Any]:
        """
        Hand-eye coordination error — Brain calculates (gaze target - hand position) error

        This is the driving force of the dopamine circuit:
        - Large error → continue correcting
        - Error ≈ 0 → release dopamine (reward for task completion)

        Args:
            eye_target_x, eye_target_y: Position the eyes are fixating on

        Returns:
            {
                "error_x": x-axis error,
                "error_y": y-axis error,
                "error_magnitude": total error distance,
                "reached": whether within threshold,
                "dopamine": dopamine signal (0 or DOPAMINE_ON_REACH),
                "error_signal": ElectricalSignal (error → signal → return to motor cortex),
            }
        """
        ex = eye_target_x - self.x
        ey = eye_target_y - self.y
        error_mag = math.sqrt(ex * ex + ey * ey)
        reached = error_mag < REACH_THRESHOLD

        dopamine = DOPAMINE_ON_REACH if reached else 0.0

        # Error → electrical signal (return to motor cortex for correction)
        # Larger error → higher frequency (urgency), larger amplitude
        error_norm = min(error_mag / 200.0, 1.0)
        err_freq = 5.0 + error_norm * 80.0  # 5 Hz (small error) → 85 Hz (large error)
        err_amp = 0.05 + error_norm * 2.0

        n = 64
        t = np.linspace(0, 1, n, endpoint=False)
        err_waveform = err_amp * np.sin(2 * np.pi * err_freq * t)

        error_signal = ElectricalSignal(
            waveform=err_waveform,
            amplitude=err_amp,
            frequency=err_freq,
            phase=math.atan2(ey, ex),  # Error direction
            impedance=PROPRIOCEPTION_IMPEDANCE,
            snr=PROPRIOCEPTION_SNR,
            source="hand_eye_error",
            modality="proprioception",
        )

        return {
            "error_x": round(ex, 4),
            "error_y": round(ey, 4),
            "error_magnitude": round(error_mag, 4),
            "reached": reached,
            "dopamine": dopamine,
            "error_signal": error_signal,
        }

    # ------------------------------------------------------------------
    # Internal physics
    # ------------------------------------------------------------------

    def _apply_muscle_physics(
        self,
        force_x: float,
        force_y: float,
        motor_gain: float,
        ram_temperature: float,
    ) -> Tuple[float, float, float]:
        """
        Muscle physics — Applied force + tremor noise

        Physical model:
        1. Motor cortex signal → muscle force (motor_gain amplification)
        2. Anxiety (ram_temperature) → tremor noise
           - Tremor frequency ~10 Hz (physiological tremor)
           - Tremor amplitude ∝ ram_temperature² (nonlinear — more anxious = worse)
        3. Force reduction: high anxiety → coordination drops → effective force decreases
        4. Force saturation: muscles cannot exert unlimited force

        Returns:
            (force_x, force_y, tremor_magnitude)
        """
        # 1. Motor cortex gain
        force_x *= motor_gain
        force_y *= motor_gain

        # 1.5 Motor maturity × pain guard
        # Infant: maturity=0.05 → force only 5%
        # After injury: guard=0.8 → force only 52% (1-0.6×0.8)
        maturity_factor = self.motor_maturity
        guard_factor = 1.0 - GUARD_SENSITIVITY * self.guard_level
        motor_scale = maturity_factor * guard_factor
        force_x *= motor_scale
        force_y *= motor_scale

        # 2. Anxiety tremor
        # Tremor amplitude = temperature² × base amplitude (nonlinear)
        temp = float(np.clip(ram_temperature, 0.0, 1.0))
        tremor_amplitude = temp * temp * MUSCLE_MAX_FORCE * 0.05

        # Tremor = sinusoidal oscillation + random noise
        self._tremor_phase += 2 * np.pi * TREMOR_FREQ_HZ * self.dt
        if self._tremor_phase > 2 * np.pi * 1000:
            self._tremor_phase -= 2 * np.pi * 1000

        tremor_sin = math.sin(self._tremor_phase)
        tremor_noise_x = self._rng.normal(0, 0.3)
        tremor_noise_y = self._rng.normal(0, 0.3)

        tremor_x = tremor_amplitude * (0.7 * tremor_sin + 0.3 * tremor_noise_x)
        tremor_y = tremor_amplitude * (0.7 * math.cos(self._tremor_phase * 1.1) + 0.3 * tremor_noise_y)

        force_x += tremor_x
        force_y += tremor_y

        tremor_mag = math.sqrt(tremor_x ** 2 + tremor_y ** 2)

        # 3. Anxiety → coordination drops (effective force reduction)
        coordination = 1.0 - 0.4 * temp  # High anxiety = only 60% effective force
        force_x *= coordination
        force_y *= coordination

        # 4. Force saturation
        force_mag = math.sqrt(force_x ** 2 + force_y ** 2)
        if force_mag > MUSCLE_MAX_FORCE:
            scale = MUSCLE_MAX_FORCE / force_mag
            force_x *= scale
            force_y *= scale

        return force_x, force_y, tremor_mag

    # ------------------------------------------------------------------
    # Statistics and diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_movements": self.total_movements,
            "total_reaches": self.total_reaches,
            "reach_rate": round(
                self.total_reaches / max(self.total_movements, 1), 4
            ),
            "total_dopamine": round(self.total_dopamine, 4),
            "position": (round(self.x, 2), round(self.y, 2)),
            "velocity": (round(self.vx, 4), round(self.vy, 4)),
            "last_reach_error": round(self._last_reach_error, 4),
            "last_dopamine": self._last_dopamine,
            "workspace_size": (self.workspace_w, self.workspace_h),
            "motor_maturity": round(self.motor_maturity, 4),
            "motor_experience": self.motor_experience,
            "guard_level": round(self.guard_level, 4),
            "injury_memory": round(self.injury_memory, 4),
        }

    def get_trajectory(self, last_n: int = 128) -> List[Tuple[float, float]]:
        """Get recent trajectory (for frontend drawing)"""
        return self._position_history[-last_n:]

    def get_muscle_state(self, ram_temperature: float = 0.0) -> Dict[str, Any]:
        """
        Diagnostic: get current muscle state

        Returns:
            {
                "position": hand position,
                "velocity": velocity,
                "target": target position,
                "error": target error,
                "tremor_level": tremor level (based on anxiety),
                "coordination": coordination level,
                "proprioception": proprioception electrical signal,
            }
        """
        temp = float(np.clip(ram_temperature, 0.0, 1.0))
        tremor_level = temp * temp
        coordination = 1.0 - 0.4 * temp

        error = 0.0
        if self._target_x is not None and self._target_y is not None:
            error = math.sqrt(
                (self._target_x - self.x) ** 2 + (self._target_y - self.y) ** 2
            )

        return {
            "position": (round(self.x, 2), round(self.y, 2)),
            "velocity": (round(self.vx, 4), round(self.vy, 4)),
            "target": (
                round(self._target_x, 2) if self._target_x is not None else None,
                round(self._target_y, 2) if self._target_y is not None else None,
            ),
            "error": round(error, 4),
            "tremor_level": round(tremor_level, 4),
            "coordination": round(coordination, 4),
            "proprioception": self.get_proprioception(),
        }

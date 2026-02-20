# -*- coding: utf-8 -*-
"""
Alice's Temporal Calibrator — Cross-Modal Signal Binding (Action Model)

Physics:
  "In a coaxial cable, audio and video belong to the same frame
   because they arrive at the same moment."

  This is the oldest binding law — Temporal Correlation:
    If two signals are temporally close enough, they belong to the same event.

  But time is not perfect. Signals get delayed and drift.
  So calibration is needed:
    1. Primary binding = temporal window
       — Signals arriving within ±Δt → same event
    2. Secondary binding = feature matching
       — Outside the temporal window, use frequency, amplitude, phase, etc. to find correspondence
    3. Error correction = drift correction
       — Continuously track temporal offsets of each modality, dynamically compensate

Action Model:
  The complete loop of perception → action → feedback requires temporal binding:

  t₀: Eyes see target       (visual signal,    timestamp=t₀)
  t₁: Brain decision done   (motor command,    timestamp=t₁)
  t₂: Hand starts moving    (proprioception,   timestamp=t₂)
  t₃: Hand reaches target   (proprioception,   timestamp=t₃)
  t₄: Eyes confirm arrival  (visual signal,    timestamp=t₄)

  Calibration problem:
  - Are the visual signal at t₄ and the proprioception at t₃ the same event?
  - If |t₄ - t₃| < Δt → yes, bind! → dopamine
  - If |t₄ - t₃| > Δt → temporal drift → use other features to calibrate

  "This is not program logic. This is physical causality.
   Because in the real world, your brain does exactly this."

Coaxial cable analogy:
  - Each modality = one coaxial channel (visual, auditory, proprioception, motor)
  - All signals have timestamps (arrival time)
  - SignalFrame = all signals within one temporal window (= one TV frame)
  - CalibrationLoop = continuously calibrates temporal offsets of each channel
  - When calibration fails → "audio-video out of sync" → cognitive dissonance → pain

Another layer of meaning for the reflection coefficient:
  - Impedance mismatch → signal reflection → delay
  - Delay → timestamp offset → binding failure
  - Binding failure → calibration error → action model collapse
  → So impedance mismatch not only produces heat, but also "cognitive dissonance"
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import ElectricalSignal


# ============================================================================
# Physical Constants
# ============================================================================

# Temporal window — signals arriving within this many seconds are considered "same event"
#   Human audiovisual binding window ≈ 80ms
#   Alice runs on CPU, set shorter
TEMPORAL_WINDOW_MS = 50.0       # Primary binding window (ms)
EXTENDED_WINDOW_MS = 200.0      # Extended window (uses feature matching to search)

# ★ Dynamic time slice constants — calibration delay physics
#
# Core physics:
#   Consciousness's time slice is not fixed — it is the physical result of calibration delay.
#   The brain takes time to impedance-match (calibrate) cross-modal electrical signals;
#   the larger the signal volume and worse the impedance mismatch, the more time calibration needs.
#
#   T_slice = T_hardware + Σ (Γᵢ² / (1-Γᵢ²)) × τ_match
#
#   This is why:
#   - The brain goes "blank" during information overload (slice too wide, frame rate drops)
#   - Time "slows down" during meditation (fewer channels, frame rate increases)
#   - Time "slows down" during danger (amygdala fast-path bypasses calibration, frame rate surges)
#   - Watching a dubbed movie with bad lip sync is uncomfortable (calibration keeps failing, slice jitters)
#
MIN_TEMPORAL_WINDOW_MS = 20.0    # Hardware limit: minimum calibration time (ms)
MAX_TEMPORAL_WINDOW_MS = 400.0   # Upper limit: beyond this = dissociation/spacing out
MATCH_COST_MS = 15.0             # Impedance matching time cost per channel (ms)
WINDOW_EMA_RATE = 0.2            # Dynamic window exponential moving average learning rate

# Feature matching weights
WEIGHT_FREQUENCY = 0.4          # Frequency similarity
WEIGHT_AMPLITUDE = 0.3          # Amplitude similarity
WEIGHT_PHASE = 0.15             # Phase similarity
WEIGHT_WAVEFORM = 0.15          # Waveform correlation

# Calibration thresholds
BINDING_THRESHOLD = 0.5         # Feature match score ≥ this → bind
DRIFT_LEARNING_RATE = 0.1       # Drift estimation learning rate
MAX_DRIFT_MS = 100.0            # Maximum compensable drift (ms)

# Frame buffer
MAX_BUFFER_SIZE = 64            # Signal buffer queue length
MAX_FRAMES = 128                # Maximum frame history


# ============================================================================
# Signal Frame — all signals within one temporal window
# ============================================================================


@dataclass
class SignalFrame:
    """
    Signal frame = all modality signals received within one temporal window

    Like a TV frame: contains video + audio + other data;
    because they arrived at the same moment, they belong to the same "picture".

    Attributes:
        frame_id:     Frame number
        timestamp:    Frame reference time (time of earliest arriving signal)
        signals:      Modality → signal list
        bindings:     Successfully bound modality pairs
        binding_scores: Score of each binding pair
        is_complete:  Whether at least 2 modalities are present
    """
    frame_id: int = 0
    timestamp: float = 0.0
    signals: Dict[str, List[ElectricalSignal]] = field(default_factory=dict)
    bindings: List[Tuple[str, str]] = field(default_factory=list)
    binding_scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    drift_corrections: Dict[str, float] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return len(self.signals) >= 2

    @property
    def modalities(self) -> List[str]:
        return list(self.signals.keys())

    @property
    def signal_count(self) -> int:
        return sum(len(sigs) for sigs in self.signals.values())

    def get_primary_signal(self, modality: str) -> Optional[ElectricalSignal]:
        """Get the primary signal of a specified modality (first to arrive)"""
        sigs = self.signals.get(modality, [])
        return sigs[0] if sigs else None


# ============================================================================
# Feature Matcher
# ============================================================================


class _FeatureMatcher:
    """
    Signal feature matching — fallback when the temporal window is insufficient

    Compares two ElectricalSignals on:
    - Frequency similarity (closer = more similar)
    - Amplitude similarity
    - Phase similarity
    - Waveform cross-correlation

    Returns a 0~1 match score
    """

    @staticmethod
    def match(sig_a: ElectricalSignal, sig_b: ElectricalSignal) -> float:
        """
        Compute feature match score (0~1) between two signals

        Not comparing "same content", but "could they originate from the same event".
        """
        # Frequency similarity — using relative difference
        freq_max = max(sig_a.frequency, sig_b.frequency, 0.1)
        freq_diff = abs(sig_a.frequency - sig_b.frequency) / freq_max
        freq_score = max(0.0, 1.0 - freq_diff)

        # Amplitude similarity
        amp_max = max(sig_a.amplitude, sig_b.amplitude, 0.01)
        amp_diff = abs(sig_a.amplitude - sig_b.amplitude) / amp_max
        amp_score = max(0.0, 1.0 - amp_diff)

        # Phase similarity — angular difference (0~π → 0~1)
        phase_diff = abs(sig_a.phase - sig_b.phase) % (2 * math.pi)
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        phase_score = 1.0 - phase_diff / math.pi

        # Waveform cross-correlation — take maximum correlation
        waveform_score = 0.0
        if len(sig_a.waveform) > 0 and len(sig_b.waveform) > 0:
            # Take same length
            min_len = min(len(sig_a.waveform), len(sig_b.waveform))
            wa = sig_a.waveform[:min_len]
            wb = sig_b.waveform[:min_len]

            # Normalize
            norm_a = np.linalg.norm(wa)
            norm_b = np.linalg.norm(wb)
            if norm_a > 1e-10 and norm_b > 1e-10:
                correlation = float(np.dot(wa, wb) / (norm_a * norm_b))
                waveform_score = (correlation + 1.0) / 2.0  # [-1,1] → [0,1]

        # Weighted total score
        total = (
            WEIGHT_FREQUENCY * freq_score
            + WEIGHT_AMPLITUDE * amp_score
            + WEIGHT_PHASE * phase_score
            + WEIGHT_WAVEFORM * waveform_score
        )

        return float(np.clip(total, 0.0, 1.0))


# ============================================================================
# Drift Estimator — tracks temporal offset of each modality
# ============================================================================


class _DriftEstimator:
    """
    Drift Estimator — tracks temporal offset of each modality relative to reference clock

    Physical analogy:
    - TV colorburst signal — lets the receiver lock onto the transmitter's clock
    - Computes each modality's temporal offset upon every successful binding
    - Tracks drift trend via exponential moving average (EMA)
    - Automatically compensates when the next signal of that modality arrives

    Drift sources (in Alice's physical model):
    - Impedance mismatch → reflection → multipath delay
    - Different channels have different propagation speeds
    - Computational delay in the processing pipeline
    """

    def __init__(self):
        # Modality → estimated drift (ms)
        self._drift: Dict[str, float] = {}
        # Modality → drift history
        self._drift_history: Dict[str, List[float]] = {}
        self._max_history = 64

    def update(self, modality: str, observed_drift_ms: float):
        """
        Update drift estimate — EMA

        observed_drift_ms: observed drift this time (modality signal time - reference time)
        """
        clamped = float(np.clip(observed_drift_ms, -MAX_DRIFT_MS, MAX_DRIFT_MS))

        if modality not in self._drift:
            self._drift[modality] = clamped
            self._drift_history[modality] = [clamped]
        else:
            # EMA update
            self._drift[modality] = (
                (1 - DRIFT_LEARNING_RATE) * self._drift[modality]
                + DRIFT_LEARNING_RATE * clamped
            )
            self._drift_history[modality].append(self._drift[modality])
            if len(self._drift_history[modality]) > self._max_history:
                self._drift_history[modality] = self._drift_history[modality][-self._max_history:]

    def get_correction(self, modality: str) -> float:
        """Get drift correction (ms) — used to correct the modality's timestamp"""
        return -self._drift.get(modality, 0.0)

    def get_all_drifts(self) -> Dict[str, float]:
        """Get drift estimates for all modalities"""
        return {k: round(v, 4) for k, v in self._drift.items()}

    def get_drift_history(self, modality: str) -> List[float]:
        return self._drift_history.get(modality, [])


# ============================================================================
# Temporal Calibrator — Main Class
# ============================================================================


class TemporalCalibrator:
    """
    Temporal Calibrator — action model for cross-modal signal binding

    Coaxial cable physics:
    - Each channel (modality) signal has a timestamp
    - Signals arriving within the same temporal window → bound to the same frame
    - Signals outside the window → use feature matching to find correspondences
    - Continuously calibrate temporal drift for each channel

    Action loop:
    1. receive()  — receive an ElectricalSignal from any modality
    2. bind()     — attempt to bind buffered signals into a frame
    3. calibrate() — update drift estimates based on binding results
    4. get_frame() — get the latest bound frame (= input to the action model)

    "Her brain doesn't work step by step with if-else.
      Her brain simultaneously receives signals from multiple coaxial cables,
      using time to determine whether they belong to the same event.
      That is the action model."
    """

    def __init__(
        self,
        temporal_window_ms: float = TEMPORAL_WINDOW_MS,
        extended_window_ms: float = EXTENDED_WINDOW_MS,
        binding_threshold: float = BINDING_THRESHOLD,
    ):
        self.temporal_window_ms = temporal_window_ms
        self.extended_window_ms = extended_window_ms
        self.binding_threshold = binding_threshold

        # Signal buffer — signals awaiting binding
        self._buffer: Deque[ElectricalSignal] = deque(maxlen=MAX_BUFFER_SIZE)

        # Drift estimator
        self._drift = _DriftEstimator()

        # Feature matcher
        self._matcher = _FeatureMatcher()

        # Frame history
        self._frames: Deque[SignalFrame] = deque(maxlen=MAX_FRAMES)
        self._frame_counter = 0

        # Statistics
        self._total_signals = 0
        self._total_frames = 0
        self._total_bindings = 0
        self._temporal_bindings = 0    # Pure temporal bindings
        self._feature_bindings = 0      # Bindings requiring feature matching
        self._failed_bindings = 0       # Failed bindings

        # Calibration error history (larger = more cognitive dissonance)
        self._calibration_error_history: List[float] = []
        self._max_error_history = 128

        # ★ Dynamic time slicing — calibration delay physics
        #   Consciousness "frame rate" is not fixed; it is a physical consequence of
        #   calibration overhead. The more electrical signals the brain receives and
        #   the worse the impedance mismatch, the more time calibration needs
        #   → wider time slice → lower frame rate.
        self._active_window_ms = float(temporal_window_ms)  # Current actual window width
        self._calibration_load = 0.0       # Current calibration load (0~1)
        self._n_active_channels = 0        # Number of active channels
        self._avg_channel_gamma = 0.0      # Average Γ of active channels
        self._frame_rate_hz = 1000.0 / max(temporal_window_ms, 1.0)  # Consciousness frame rate
        self._temporal_resolution = 1.0    # Temporal resolution (0~1, 1=highest)
        self._window_history: List[float] = []  # Window width history
        self._max_window_history = 64

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def receive(self, signal: ElectricalSignal) -> None:
        """
        Receive a signal — place into buffer awaiting binding

        Drift compensation: adjusts timestamp based on the modality's historical drift.
        (Does not modify the original signal; compensation is only applied internally)
        """
        self._total_signals += 1
        self._buffer.append(signal)
        # ★ Update dynamic time slice
        self._update_adaptive_window()

    # ------------------------------------------------------------------
    # ★ Dynamic Time Slicing Engine
    # ------------------------------------------------------------------

    def _update_adaptive_window(self) -> None:
        """
        Dynamically compute calibration time window — time slice physics

        Core formula:
          T_slice = T_hardware + Σ (Γᵢ² / (1 - Γᵢ²)) × τ_match

        Physical meaning:
          Every active sensory channel needs calibration (impedance matching) to bind.
          The worse the impedance mismatch (Γ is larger), the longer matching takes.
          The sum of calibration time for all channels = minimum width of one time slice.

          This is why the brain "reacts slower" when processing lots of information:
          it's not that computation slowed down — the physical wait time for
          calibration got longer.

          T_hardware = minimum processing time (hardware limit of neural conduction)
          Γᵢ = impedance mismatch coefficient for channel i
          τ_match = base time cost per matching attempt
        """
        if not self._buffer:
            return

        # Collect all active modalities and their impedance characteristics
        active_modalities: Dict[str, List[float]] = {}
        for sig in self._buffer:
            if sig.modality not in active_modalities:
                active_modalities[sig.modality] = []
            active_modalities[sig.modality].append(sig.impedance)

        n_channels = len(active_modalities)
        self._n_active_channels = n_channels

        if n_channels <= 1:
            # Single channel → no cross-modal calibration needed → shortest slice (highest frame rate)
            target_window = MIN_TEMPORAL_WINDOW_MS
            self._avg_channel_gamma = 0.0
        else:
            # Compute cross-channel impedance mismatch
            # Get average impedance per modality, then compute pairwise Γ
            avg_impedances = {
                mod: float(np.mean(imps))
                for mod, imps in active_modalities.items()
            }
            impedance_values = list(avg_impedances.values())

            # Γ for all channel pairs
            gammas = []
            for i in range(len(impedance_values)):
                for j in range(i + 1, len(impedance_values)):
                    z_a, z_b = impedance_values[i], impedance_values[j]
                    gamma = abs(z_a - z_b) / max(z_a + z_b, 1e-6)
                    gammas.append(gamma)

            if gammas:
                self._avg_channel_gamma = float(np.mean(gammas))
            else:
                self._avg_channel_gamma = 0.0

            # ★ Core formula: T_slice = T_hw + Σ Γᵢ²/(1-Γᵢ²) × τ_match
            calibration_cost = 0.0
            for g in gammas:
                g_sq = min(g ** 2, 0.99)
                calibration_cost += (g_sq / (1.0 - g_sq)) * MATCH_COST_MS

            target_window = MIN_TEMPORAL_WINDOW_MS + calibration_cost

        # Clamp to physical limits
        target_window = float(np.clip(
            target_window, MIN_TEMPORAL_WINDOW_MS, MAX_TEMPORAL_WINDOW_MS
        ))

        # EMA smoothing (prevent window from jumping erratically)
        self._active_window_ms = (
            (1.0 - WINDOW_EMA_RATE) * self._active_window_ms
            + WINDOW_EMA_RATE * target_window
        )

        # Update derived metrics
        self._frame_rate_hz = 1000.0 / max(self._active_window_ms, 1.0)
        self._temporal_resolution = MIN_TEMPORAL_WINDOW_MS / max(
            self._active_window_ms, MIN_TEMPORAL_WINDOW_MS
        )
        self._calibration_load = float(np.clip(
            (self._active_window_ms - MIN_TEMPORAL_WINDOW_MS)
            / (MAX_TEMPORAL_WINDOW_MS - MIN_TEMPORAL_WINDOW_MS),
            0.0, 1.0,
        ))

        # Record history
        self._window_history.append(round(self._active_window_ms, 2))
        if len(self._window_history) > self._max_window_history:
            self._window_history = self._window_history[-self._max_window_history:]

    def bind(self) -> Optional[SignalFrame]:
        """
        Attempt to bind buffered signals into a frame

        Algorithm:
        1. Find the oldest signal in the buffer → use as reference time
        2. Find all signals within the temporal window → temporal binding
        3. For signals outside the window but within the extended window → feature matching
        4. Assemble SignalFrame
        5. Remove bound signals from buffer
        6. Update drift estimates

        Returns:
            SignalFrame if binding succeeded, None if buffer is empty or
            no valid frame could be formed
        """
        if not self._buffer:
            return None

        # 1. Find reference signal (oldest)
        # Sort by corrected time first
        corrected_times = []
        for sig in self._buffer:
            correction_ms = self._drift.get_correction(sig.modality)
            corrected_t = sig.timestamp + correction_ms / 1000.0
            corrected_times.append(corrected_t)

        # Find earliest signal
        earliest_idx = int(np.argmin(corrected_times))
        ref_signal = self._buffer[earliest_idx]
        ref_time = corrected_times[earliest_idx]

        # 2. Temporal window binding
        #    ★ Use dynamic time slice window (not a fixed constant)
        #    _active_window_ms adapts based on cross-modal impedance mismatch
        frame_signals: Dict[str, List[ElectricalSignal]] = {}
        bound_indices: List[int] = []
        drift_observations: Dict[str, float] = {}

        window_sec = self._active_window_ms / 1000.0
        # Extended window = base ratio × dynamic window (keep same extension ratio)
        base_ratio = self.extended_window_ms / max(self.temporal_window_ms, 1.0)
        extended_sec = (self._active_window_ms * base_ratio) / 1000.0

        for i, sig in enumerate(self._buffer):
            corrected_t = corrected_times[i]
            dt = abs(corrected_t - ref_time)

            if dt <= window_sec:
                # Within main window → bind directly
                modality = sig.modality
                if modality not in frame_signals:
                    frame_signals[modality] = []
                frame_signals[modality].append(sig)
                bound_indices.append(i)

                # Record drift observation (uncorrected time difference)
                raw_dt_ms = (sig.timestamp - ref_signal.timestamp) * 1000.0
                drift_observations[modality] = raw_dt_ms

        # 3. Extended window + feature matching
        if len(frame_signals) >= 1:
            for i, sig in enumerate(self._buffer):
                if i in bound_indices:
                    continue

                corrected_t = corrected_times[i]
                dt = abs(corrected_t - ref_time)

                if dt <= extended_sec and sig.modality not in frame_signals:
                    # Within extended window, and this modality not yet present → try feature matching
                    best_score = 0.0
                    for bound_sigs in frame_signals.values():
                        for bound_sig in bound_sigs:
                            score = self._matcher.match(sig, bound_sig)
                            best_score = max(best_score, score)

                    if best_score >= self.binding_threshold:
                        if sig.modality not in frame_signals:
                            frame_signals[sig.modality] = []
                        frame_signals[sig.modality].append(sig)
                        bound_indices.append(i)
                        self._feature_bindings += 1

                        raw_dt_ms = (sig.timestamp - ref_signal.timestamp) * 1000.0
                        drift_observations[sig.modality] = raw_dt_ms

        # If only one modality, still generate a single-modality frame (doesn't count as binding)
        if not frame_signals:
            return None

        # 4. Assemble SignalFrame
        self._frame_counter += 1
        frame = SignalFrame(
            frame_id=self._frame_counter,
            timestamp=ref_time,
            signals=frame_signals,
        )

        # Compute binding pairs & scores
        modalities = list(frame_signals.keys())
        for i_m in range(len(modalities)):
            for j_m in range(i_m + 1, len(modalities)):
                m_a, m_b = modalities[i_m], modalities[j_m]
                sig_a = frame_signals[m_a][0]
                sig_b = frame_signals[m_b][0]
                score = self._matcher.match(sig_a, sig_b)
                pair = (m_a, m_b)
                frame.bindings.append(pair)
                frame.binding_scores[pair] = round(score, 4)
                self._total_bindings += 1

                # Determine temporal binding vs. feature binding
                raw_dt = abs(sig_a.timestamp - sig_b.timestamp) * 1000.0
                if raw_dt <= self.temporal_window_ms:
                    self._temporal_bindings += 1

        frame.drift_corrections = {
            k: round(self._drift.get_correction(k), 4)
            for k in modalities
        }

        # 5. Remove bound signals from buffer
        remaining = deque(maxlen=MAX_BUFFER_SIZE)
        for i, sig in enumerate(self._buffer):
            if i not in bound_indices:
                remaining.append(sig)
        self._buffer = remaining

        # 6. Calibrate — update drift estimates
        for modality, drift_ms in drift_observations.items():
            self._drift.update(modality, drift_ms)

        # Record calibration error
        if drift_observations:
            max_drift = max(abs(v) for v in drift_observations.values())
            self._calibration_error_history.append(max_drift)
            if len(self._calibration_error_history) > self._max_error_history:
                self._calibration_error_history = self._calibration_error_history[-self._max_error_history:]

        # Store into frame history
        self._frames.append(frame)
        self._total_frames += 1

        return frame

    # ------------------------------------------------------------------

    def receive_and_bind(self, signal: ElectricalSignal) -> Optional[SignalFrame]:
        """
        Receive + attempt to bind — convenience method

        Tries to bind every time a new signal arrives.
        If the buffer has enough signals to form a frame → returns the frame.
        """
        self.receive(signal)
        return self.bind()

    # ------------------------------------------------------------------

    def flush(self) -> List[SignalFrame]:
        """
        Force flush buffer — bind all remaining signals into frames

        Used at end of cycle to ensure no signals are abandoned.
        """
        frames = []
        while self._buffer:
            frame = self.bind()
            if frame is None:
                # Remaining signals cannot be bound further → create single-signal frame
                if self._buffer:
                    sig = self._buffer.popleft()
                    self._frame_counter += 1
                    single_frame = SignalFrame(
                        frame_id=self._frame_counter,
                        timestamp=sig.timestamp,
                        signals={sig.modality: [sig]},
                    )
                    self._frames.append(single_frame)
                    self._total_frames += 1
                    frames.append(single_frame)
                break
            frames.append(frame)
        return frames

    # ------------------------------------------------------------------

    def get_latest_frame(self) -> Optional[SignalFrame]:
        """Get the latest bound frame"""
        return self._frames[-1] if self._frames else None

    def get_frames(self, last_n: int = 10) -> List[SignalFrame]:
        """Get recent frame history"""
        frames_list = list(self._frames)
        return frames_list[-last_n:]

    # ------------------------------------------------------------------
    # Calibration State
    # ------------------------------------------------------------------

    def get_calibration_state(self) -> Dict[str, Any]:
        """
        Get calibration state — for oscilloscope/dashboard display

        Larger calibration error = worse cross-modal temporal offset = cognitive dissonance
        """
        drifts = self._drift.get_all_drifts()

        # Calibration quality (0=fully desynchronized, 1=perfect calibration)
        if drifts:
            max_drift = max(abs(v) for v in drifts.values())
            quality = max(0.0, 1.0 - max_drift / MAX_DRIFT_MS)
        else:
            quality = 1.0

        return {
            "quality": round(quality, 4),
            "drifts_ms": drifts,
            "buffer_size": len(self._buffer),
            "total_signals": self._total_signals,
            "total_frames": self._total_frames,
            "total_bindings": self._total_bindings,
            "temporal_bindings": self._temporal_bindings,
            "feature_bindings": self._feature_bindings,
            "failed_bindings": self._failed_bindings,
            "calibration_error_history": self._calibration_error_history[-32:],
            # ★ Dynamic time slice state
            "active_window_ms": round(self._active_window_ms, 2),
            "frame_rate_hz": round(self._frame_rate_hz, 2),
            "temporal_resolution": round(self._temporal_resolution, 4),
            "calibration_load": round(self._calibration_load, 4),
            "n_active_channels": self._n_active_channels,
            "avg_channel_gamma": round(self._avg_channel_gamma, 4),
            "window_history": self._window_history[-16:],
        }

    def get_calibration_quality(self) -> float:
        """Calibration quality (0~1)"""
        state = self.get_calibration_state()
        return state["quality"]

    def get_drift_for(self, modality: str) -> float:
        """Get drift for a specific modality (ms)"""
        return self._drift.get_correction(modality)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_signals": self._total_signals,
            "total_frames": self._total_frames,
            "total_bindings": self._total_bindings,
            "temporal_bindings": self._temporal_bindings,
            "feature_bindings": self._feature_bindings,
            "failed_bindings": self._failed_bindings,
            "temporal_rate": round(
                self._temporal_bindings / max(self._total_bindings, 1), 4
            ),
            "buffer_size": len(self._buffer),
            "calibration_quality": self.get_calibration_quality(),
            "drifts": self._drift.get_all_drifts(),
            # ★ Dynamic time slice
            "active_window_ms": round(self._active_window_ms, 2),
            "frame_rate_hz": round(self._frame_rate_hz, 2),
            "temporal_resolution": round(self._temporal_resolution, 4),
            "calibration_load": round(self._calibration_load, 4),
        }

    # ------------------------------------------------------------------
    # ★ Dynamic Time Slice Public Interface
    # ------------------------------------------------------------------

    def get_active_window_ms(self) -> float:
        """Current actual time slice width (ms)"""
        return self._active_window_ms

    def get_frame_rate(self) -> float:
        """Consciousness frame rate (Hz) = 1000 / time slice width"""
        return self._frame_rate_hz

    def get_temporal_resolution(self) -> float:
        """
        Temporal resolution (0~1)

        1.0 = highest resolution (single channel / perfect match → narrowest slice)
        0.0 = lowest resolution (multiple channels severely mismatched → slice at limit)

        Physical correspondence:
        - During meditation → approaches 1.0 (time feels "slow")
        - During information overload → approaches 0.0 (dazed, can't keep up)
        - Fight-or-flight response → amygdala bypasses calibration → forced 1.0 (time "freezes")
        """
        return self._temporal_resolution

    def get_calibration_load(self) -> float:
        """Calibration load (0~1) — how much of the brain's time is spent aligning signals"""
        return self._calibration_load

    def override_window(self, window_ms: float) -> None:
        """
        Force override temporal window — simulate special states

        Use cases:
        - Amygdala fast-path → override_window(20) → time "freezes"
        - Sleep N3 → override_window(300) → consciousness gate closes
        """
        self._active_window_ms = float(np.clip(
            window_ms, MIN_TEMPORAL_WINDOW_MS, MAX_TEMPORAL_WINDOW_MS
        ))
        self._frame_rate_hz = 1000.0 / max(self._active_window_ms, 1.0)
        self._temporal_resolution = MIN_TEMPORAL_WINDOW_MS / max(
            self._active_window_ms, MIN_TEMPORAL_WINDOW_MS
        )

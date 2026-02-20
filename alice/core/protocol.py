# -*- coding: utf-8 -*-
"""
Γ-Net v4 Message Protocol Architecture — Alice Smart System Core Engine

Core Principles (from Γ-Net v4 Genesis):
  "The essence of the brain is not computation, but communication"
  "Pain is not a feeling — it is protocol collapse"

Version Notes:
  - Protocol Version: v4 (low-level communication protocol iteration, analogous to HTTP/1.1)
  - System  Version: v29.x (overall system release version, analogous to Chrome v100)
  The two evolve independently; protocol version increments only when packet format / routing logic undergoes major changes.

Full Processing Pipeline:
  Signal → MessagePacket → PriorityRouter(O(1))
        → YearRingCache(hit = zero computation)
        → BrainHemisphere(activate on demand)
        → ErrorCorrector(minimal correction)
        → Result written to cache
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xxhash

from alice.core.signal import ElectricalSignal


# ============================================================================
# 1. Message Packet Structure
# ============================================================================


class Priority(Enum):
    """Message Priority — 4-level classification"""

    BACKGROUND = 0  # Background, can be deferred
    NORMAL = 1  # Normal
    HIGH = 2  # High priority
    CRITICAL = 3  # Critical (pain / protocol collapse)


class Modality(Enum):
    """Sensory Modality"""

    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    INTERNAL = "internal"


@dataclass
class MessagePacket:
    """
    Message Packet (not a numerical array!)

    Core fields:
    - frequency_tag      : frequency tag (for year-ring matching)
    - priority           : priority (determines processing order)
    - content_hash       : content hash digest (for O(1) comparison)
    - electrical_signal   : unified electrical signal (coaxial cable model)
    """

    frequency_tag: float
    priority: Priority
    modality: Modality
    content_hash: str
    raw_data: Optional[np.ndarray] = None
    electrical_signal: Optional[ElectricalSignal] = None
    timestamp: float = field(default_factory=time.time)
    source: str = "external"

    # ------------------------------------------------------------------
    @classmethod
    def from_signal(
        cls,
        signal: Union[np.ndarray, ElectricalSignal],
        modality: Modality = Modality.VISUAL,
        priority: Priority = Priority.NORMAL,
        source: str = "external",
    ) -> MessagePacket:
        """
        Create a packet from a raw signal or electrical signal

        Supports two input types:
        1. np.ndarray     → automatically converted to ElectricalSignal
        2. ElectricalSignal → used directly
        """
        # Unified conversion to ElectricalSignal
        if isinstance(signal, ElectricalSignal):
            esig = signal
            raw = signal.waveform
        else:
            esig = ElectricalSignal.from_raw(signal, source=source, modality=modality.value)
            raw = signal

        # Frequency tag: derived from the electrical signal's actual frequency
        # freq_tag = base(1000) + carrier frequency × 10 + amplitude × 5
        freq_tag = 1000.0 + esig.frequency * 10.0 + esig.amplitude * 5.0

        content_hash = xxhash.xxh64(raw.tobytes()).hexdigest()[:8]

        return cls(
            frequency_tag=freq_tag,
            priority=priority,
            modality=modality,
            content_hash=content_hash,
            raw_data=raw,
            electrical_signal=esig,
            source=source,
        )


# ============================================================================
# 2. Priority Router — O(1) deque queue + aging to prevent starvation
# ============================================================================


class PriorityRouter:
    """
    Priority Router

    - Zero-computation classification (only checks tags, not content)
    - 4-level deque queues (O(1) push / pop)
    - Priority aging to prevent low-priority starvation
      BACKGROUND → 2 sec → NORMAL → 3 sec → HIGH
    """

    def __init__(self, aging_thresholds: Optional[Dict[Priority, float]] = None):
        self.queues: Dict[Priority, deque] = {
            Priority.CRITICAL: deque(),
            Priority.HIGH: deque(),
            Priority.NORMAL: deque(),
            Priority.BACKGROUND: deque(),
        }

        self.aging_thresholds = aging_thresholds or {
            Priority.BACKGROUND: 2.0,
            Priority.NORMAL: 3.0,
        }

        # Statistics
        self.total_routed = 0
        self.skipped_low_priority = 0
        self.aged_packets = 0
        self.aging_checks = 0
        self.last_aging_time = time.time()

    # ------------------------------------------------------------------
    def route(self, packet: MessagePacket) -> Tuple[bool, str]:
        """Route a message → (should_process_now, reason)"""
        self.total_routed += 1
        self._apply_aging()

        if packet.priority == Priority.CRITICAL:
            self.queues[Priority.CRITICAL].append((packet, time.time()))
            return True, "CRITICAL: process immediately"

        if packet.priority == Priority.HIGH:
            self.queues[Priority.HIGH].append((packet, time.time()))
            return True, "HIGH: priority processing"

        if packet.priority == Priority.NORMAL:
            self.queues[Priority.NORMAL].append((packet, time.time()))
            if not self.queues[Priority.CRITICAL] and not self.queues[Priority.HIGH]:
                return True, "NORMAL: processing"
            return False, "NORMAL: waiting for high priority to finish"

        # BACKGROUND
        self.queues[Priority.BACKGROUND].append((packet, time.time()))
        self.skipped_low_priority += 1
        return False, "BACKGROUND: deferred processing"

    # ------------------------------------------------------------------
    def _apply_aging(self):
        """
        Aging upgrade: oldest in FIFO queue is at front, stop at first non-expired → O(k)
        Check at most once every 100 ms.
        """
        now = time.time()
        if now - self.last_aging_time < 0.1:
            return
        self.last_aging_time = now
        self.aging_checks += 1

        # BACKGROUND → NORMAL
        if Priority.BACKGROUND in self.aging_thresholds:
            th = self.aging_thresholds[Priority.BACKGROUND]
            while self.queues[Priority.BACKGROUND]:
                pkt, enq = self.queues[Priority.BACKGROUND][0]
                if now - enq > th:
                    self.queues[Priority.BACKGROUND].popleft()
                    self.queues[Priority.NORMAL].append((pkt, now))
                    self.aged_packets += 1
                else:
                    break

        # NORMAL → HIGH
        if Priority.NORMAL in self.aging_thresholds:
            th = self.aging_thresholds[Priority.NORMAL]
            while self.queues[Priority.NORMAL]:
                pkt, enq = self.queues[Priority.NORMAL][0]
                if now - enq > th:
                    self.queues[Priority.NORMAL].popleft()
                    self.queues[Priority.HIGH].append((pkt, now))
                    self.aged_packets += 1
                else:
                    break

    # ------------------------------------------------------------------
    def get_next(self) -> Optional[MessagePacket]:
        """Get the next packet to process (by priority), O(1)"""
        self._apply_aging()
        for pri in (Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.BACKGROUND):
            if self.queues[pri]:
                pkt, _ = self.queues[pri].popleft()
                return pkt
        return None

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict:
        return {
            "total_routed": self.total_routed,
            "skipped_low_priority": self.skipped_low_priority,
            "aged_packets": self.aged_packets,
            "aging_checks": self.aging_checks,
            "queue_lengths": {p.name: len(q) for p, q in self.queues.items()},
            "queue_implementation": "deque (O(1) operations)",
        }


# ============================================================================
# 3. Year-Ring Cache — 8 concentric rings + dynamic capacity + Fibonacci consolidation
# ============================================================================


class YearRingCache:
    """
    Year-Ring Cache System

    Structure:
    - ring[0] = outermost layer (new memories)
    - ring[N-1] = innermost layer (stable memories)
    - Fibonacci thresholds [3,5,8,13,21,34,55,89] determine migration timing

    Key: cache hit = zero computation!
    """
    # -- Memory confidence constants --------------------------------------------------
    BASE_CONFIDENCE = 0.75   # Base confidence of the outermost ring (75% energy transmission)
    CONFIDENCE_GAIN = 0.25   # Additional confidence gained linearly with ring depth
    FIBONACCI_THRESHOLDS = [3, 5, 8, 13, 21, 34, 55, 89]

    def __init__(
        self,
        num_rings: int = 8,
        ring_capacity: int = 100,
        max_capacity: int = 500,
        min_capacity: int = 50,
    ):
        self.num_rings = num_rings
        self.ring_capacity = ring_capacity
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity

        self.rings: List[Dict[str, Any]] = [{} for _ in range(num_rings)]
        self.usage_counts: Dict[str, int] = {}

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.consolidations = 0
        self.capacity_adjustments = 0

    # ------------------------------------------------------------------
    def lookup(self, packet: MessagePacket) -> Tuple[bool, Optional[str], float]:
        """O(1) hash lookup → (hit, label, confidence)"""
        h = packet.content_hash

        for ring_idx in reversed(range(self.num_rings)):
            if h in self.rings[ring_idx]:
                self.cache_hits += 1
                entry = self.rings[ring_idx][h]
                self.usage_counts[h] = self.usage_counts.get(h, 0) + 1
                confidence = self.BASE_CONFIDENCE + self.CONFIDENCE_GAIN * (
                    ring_idx / max(1, self.num_rings - 1)
                )
                return True, entry["label"], confidence

        self.cache_misses += 1
        return False, None, 0.0

    # ------------------------------------------------------------------
    def store(self, packet: MessagePacket, label: str, computed_result: Any = None):
        """Store into the outermost ring"""
        h = packet.content_hash
        self.rings[0][h] = {
            "label": label,
            "frequency": packet.frequency_tag,
            "result": computed_result,
            "timestamp": time.time(),
        }
        self._check_consolidation(h)
        self._adjust_capacity()

        if len(self.rings[0]) > self.ring_capacity:
            self._evict_oldest(0)

    # ------------------------------------------------------------------
    def _check_consolidation(self, content_hash: str):
        """Fibonacci threshold migration: when usage count exceeds threshold → consolidate to inner ring"""
        usage = self.usage_counts.get(content_hash, 0)
        thresholds = self.FIBONACCI_THRESHOLDS

        for ring_idx in range(self.num_rings - 1):
            if content_hash in self.rings[ring_idx]:
                if ring_idx < len(thresholds) and usage >= thresholds[ring_idx]:
                    entry = self.rings[ring_idx].pop(content_hash)
                    self.rings[ring_idx + 1][content_hash] = entry
                    self.consolidations += 1
                break

    # ------------------------------------------------------------------
    def _adjust_capacity(self):
        """Dynamic capacity: hit rate >80% → expand, <40% → shrink"""
        total = self.cache_hits + self.cache_misses
        if total < 10:
            return

        hit_rate = self.cache_hits / max(1, total)

        if hit_rate > 0.8 and self.ring_capacity < self.max_capacity:
            self.ring_capacity = min(self.ring_capacity + 20, self.max_capacity)
            self.capacity_adjustments += 1
        elif hit_rate < 0.4 and self.ring_capacity > self.min_capacity:
            self.ring_capacity = max(self.ring_capacity - 15, self.min_capacity)
            self.capacity_adjustments += 1

    # ------------------------------------------------------------------
    def _evict_oldest(self, ring_idx: int):
        ring = self.rings[ring_idx]
        if not ring:
            return
        oldest = min(ring, key=lambda h: ring[h].get("timestamp", 0))
        del ring[oldest]

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict:
        total_stored = sum(len(r) for r in self.rings)
        max_possible = self.ring_capacity * self.num_rings
        hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(hit_rate, 3),
            "consolidations": self.consolidations,
            "ring_sizes": [len(r) for r in self.rings],
            "current_capacity": self.ring_capacity,
            "max_capacity": self.max_capacity,
            "total_stored": total_stored,
            "capacity_utilization": round(total_stored / max(1, max_possible), 3),
            "capacity_adjustments": self.capacity_adjustments,
        }


# ============================================================================
# 4. Left/Right Brain Hemispheres — activate on demand
# ============================================================================


class BrainHemisphere:
    """
    Brain Hemisphere (activate on demand, not computed every time!)

    - Left brain (detail) : sequential / detail processing (auditory, language)
    - Right brain (global) : parallel / holistic processing (visual, tactile)
    """

    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty  # "detail" | "global"
        self.activation_threshold = 0.3
        self.precision_levels = {"coarse": 0.1, "medium": 0.5, "fine": 1.0}
        self.current_precision = "coarse"

        self.activations = 0
        self.skipped = 0
        self.total_compute = 0.0

    # ------------------------------------------------------------------
    def should_activate(self, error_magnitude: float, modality: Modality) -> Tuple[bool, str]:
        if error_magnitude < self.activation_threshold:
            self.skipped += 1
            return False, f"Error {error_magnitude:.2f} < threshold, skipped"

        if self.specialty == "detail":
            if modality in (Modality.AUDITORY, Modality.INTERNAL):
                return True, "Left brain specialty: auditory/internal processing"
            if error_magnitude > 0.7:
                return True, "Requires left brain fine processing"
        else:
            if modality in (Modality.VISUAL, Modality.TACTILE):
                return True, "Right brain specialty: visual/tactile"
            if error_magnitude > 0.5:
                return True, "Requires right brain holistic processing"

        self.skipped += 1
        return False, "This hemisphere does not need to process"

    # ------------------------------------------------------------------
    def process(self, data: np.ndarray, error_magnitude: float) -> Tuple[np.ndarray, float]:
        self.activations += 1

        if error_magnitude > 0.7:
            self.current_precision = "fine"
            ratio = 1.0
        elif error_magnitude > 0.4:
            self.current_precision = "medium"
            ratio = 0.5
        else:
            self.current_precision = "coarse"
            ratio = 0.1

        self.total_compute += ratio

        if self.specialty == "detail":
            result = self._sequential_process(data, ratio)
        else:
            result = self._parallel_process(data, ratio)
        return result, ratio

    # ------------------------------------------------------------------
    def _sequential_process(self, data: np.ndarray, ratio: float) -> np.ndarray:
        n = max(1, int(data.size * ratio))
        flat = data.flatten().copy()
        flat[:n] = np.sort(flat[:n])
        return flat.reshape(data.shape)

    def _parallel_process(self, data: np.ndarray, ratio: float) -> np.ndarray:
        if ratio < 1.0:
            step = max(1, int(1 / ratio))
            flat = data.flatten()
            down = flat[::step]
            result = np.interp(np.arange(flat.size), np.arange(0, flat.size, step), down)
            return result.reshape(data.shape)
        return data.copy()

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict:
        total = self.activations + self.skipped
        return {
            "hemisphere": self.name,
            "activations": self.activations,
            "skipped": self.skipped,
            "avg_compute": round(self.total_compute / max(1, self.activations), 3),
            "activation_rate": round(self.activations / max(1, total), 3),
        }


# ============================================================================
# 5. Error Corrector — minimum energy (only fix the differences)
# ============================================================================


class ErrorCorrector:
    """Error Corrector: only fix the differences, like correcting a single missing stroke in a character"""

    def __init__(self, error_threshold: float = 0.1):
        self.error_threshold = error_threshold
        self.total_corrections = 0
        self.elements_corrected = 0
        self.elements_skipped = 0

    def compute_error(self, expected: np.ndarray, actual: np.ndarray) -> Tuple[float, np.ndarray]:
        diff = np.abs(expected.flatten() - actual.flatten())
        return float(np.mean(diff)), diff > self.error_threshold

    def correct(
        self, current: np.ndarray, target: np.ndarray, error_mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        self.total_corrections += 1
        fc, ft, fm = current.flatten(), target.flatten(), error_mask.flatten()
        n_correct = int(np.sum(fm))
        n_total = fm.size
        self.elements_corrected += n_correct
        self.elements_skipped += n_total - n_correct

        result = fc.copy()
        result[fm] = ft[fm]
        compute_ratio = n_correct / max(1, n_total)
        return result.reshape(current.shape), compute_ratio

    def get_stats(self) -> Dict:
        total = self.elements_corrected + self.elements_skipped
        return {
            "corrections": self.total_corrections,
            "elements_corrected": self.elements_corrected,
            "elements_skipped": self.elements_skipped,
            "average_skip_rate": round(self.elements_skipped / max(1, total), 3),
        }


# ============================================================================
# 6. Γ-Net v4 Full System Integration
# ============================================================================


class GammaNetV4Protocol:
    """
    Γ-Net v4 Message Protocol — Full System

    Processing pipeline:
    1. Message routing (zero-computation classification)
    2. Year-ring cache lookup (hit = zero computation)
    3. Left/right brain on-demand processing
    4. Minimal error correction
    5. Result written to year-ring cache
    """

    def __init__(self):
        self.router = PriorityRouter()
        self.cache = YearRingCache()
        self.left_brain = BrainHemisphere("Left", "detail")
        self.right_brain = BrainHemisphere("Right", "global")
        self.corrector = ErrorCorrector()

        self.templates: Dict[str, np.ndarray] = {}
        self.total_processed = 0
        self.cache_served = 0
        self.compute_ratios: List[float] = []

    # ------------------------------------------------------------------
    def learn(self, label: str, data: np.ndarray):
        """Learn a template and store in cache"""
        self.templates[label] = data.copy()
        packet = MessagePacket.from_signal(data)
        self.cache.store(packet, label, data)

    # ------------------------------------------------------------------
    def recognize(
        self,
        data: np.ndarray,
        priority: Priority = Priority.NORMAL,
        modality: Modality = Modality.VISUAL,
    ) -> Dict:
        """Full message protocol recognition pipeline"""
        self.total_processed += 1
        total_compute = 0.0

        packet = MessagePacket.from_signal(data, modality, priority)
        should_process, route_reason = self.router.route(packet)

        result: Dict[str, Any] = {
            "route": route_reason,
            "cache_hit": False,
            "left_activated": False,
            "right_activated": False,
            "compute_ratio": 0.0,
        }

        if not should_process:
            result["status"] = "deferred"
            return result

        # Year-ring cache O(1)
        hit, cached_label, confidence = self.cache.lookup(packet)
        if hit and confidence > 0.7:
            self.cache_served += 1
            result.update(
                status="cache_hit",
                cache_hit=True,
                label=cached_label,
                confidence=confidence,
                compute_ratio=0.0,
            )
            self.compute_ratios.append(0.0)
            return result

        # Compute: find best matching template
        best_match, best_label, min_error, error_mask = None, None, float("inf"), None
        for lbl, tmpl in self.templates.items():
            if tmpl.shape == data.shape:
                err_mag, mask = self.corrector.compute_error(tmpl, data)
                if err_mag < min_error:
                    min_error, best_match, best_label, error_mask = err_mag, tmpl, lbl, mask

        if best_match is None:
            min_error = 1.0
        result["error_magnitude"] = round(min_error, 3)

        # Left/right brain activate on demand
        left_ok, _ = self.left_brain.should_activate(min_error, modality)
        right_ok, _ = self.right_brain.should_activate(min_error, modality)

        if left_ok:
            result["left_activated"] = True
            _, lc = self.left_brain.process(data, min_error)
            total_compute += lc * 0.5

        if right_ok:
            result["right_activated"] = True
            _, rc = self.right_brain.process(data, min_error)
            total_compute += rc * 0.5

        # Error correction
        if best_match is not None and error_mask is not None:
            _, cc = self.corrector.correct(data, best_match, error_mask)
            total_compute += cc * 0.2

        result["compute_ratio"] = round(total_compute, 3)
        result["label"] = best_label or "unknown"
        result["confidence"] = round(1.0 - min_error, 3)
        result["status"] = "computed"

        if result["label"] != "unknown":
            self.cache.store(packet, result["label"])

        self.compute_ratios.append(total_compute)
        return result

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict:
        avg = float(np.mean(self.compute_ratios)) if self.compute_ratios else 0.0
        return {
            "total_processed": self.total_processed,
            "cache_served": self.cache_served,
            "cache_serve_rate": round(self.cache_served / max(1, self.total_processed), 3),
            "avg_compute_ratio": round(avg, 3),
            "router": self.router.get_stats(),
            "cache": self.cache.get_stats(),
            "left_brain": self.left_brain.get_stats(),
            "right_brain": self.right_brain.get_stats(),
            "corrector": self.corrector.get_stats(),
        }

# -*- coding: utf-8 -*-
"""
Year-Ring Cache Performance Analytics Dashboard

Provides L1/L2/L3 equivalent hit rate tracking, real-time metric monitoring, and JSON export.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CacheRingStats:
    """Single ring statistics"""

    ring_id: int
    ring_name: str
    size_history: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    hits_from_ring: int = 0
    total_hits: int = 0

    def hit_rate_from_ring(self) -> float:
        return self.hits_from_ring / max(1, self.total_hits)

    def get_avg_size(self) -> float:
        return float(np.mean(self.size_history)) if self.size_history else 0.0

    def get_max_size(self) -> int:
        return max(self.size_history) if self.size_history else 0


class CachePerformanceDashboard:
    """
    Cache Performance Dashboard

    Tracked metrics:
    - Per-ring hit rate (L1 / L2 / L3 equivalent)
    - Queries per second (QPS)
    - Capacity utilization
    - Consolidation migration count
    """

    def __init__(self, num_rings: int = 8):
        self.num_rings = num_rings

        ring_names = [f"Ring-{i}" for i in range(num_rings)]
        self.ring_stats: List[CacheRingStats] = [
            CacheRingStats(ring_id=i, ring_name=ring_names[i]) for i in range(num_rings)
        ]

        # Global statistics
        self.total_lookups = 0
        self.total_hits = 0
        self.total_misses = 0
        self.total_consolidations = 0

        # Time series
        self.hit_rate_history: List[float] = []
        self.qps_history: List[float] = []
        self.timestamps: List[float] = []

        self._start_time = time.time()
        self._last_sample_time = time.time()

    # ------------------------------------------------------------------
    def record_lookup(self, ring_id: Optional[int], hit: bool):
        """Record a single lookup"""
        self.total_lookups += 1
        if hit:
            self.total_hits += 1
            if ring_id is not None and 0 <= ring_id < self.num_rings:
                self.ring_stats[ring_id].hits_from_ring += 1
                self.ring_stats[ring_id].total_hits = self.total_hits
        else:
            self.total_misses += 1

    # ------------------------------------------------------------------
    def record_ring_size(self, ring_id: int, size: int, timestamp: Optional[float] = None):
        if 0 <= ring_id < self.num_rings:
            ts = timestamp or time.time()
            self.ring_stats[ring_id].size_history.append(size)
            self.ring_stats[ring_id].timestamps.append(ts)

    # ------------------------------------------------------------------
    def record_consolidation(self):
        self.total_consolidations += 1

    # ------------------------------------------------------------------
    def update_global_stats(self, total_capacity: int, timestamp: Optional[float] = None):
        """Sample once every 0.5 seconds"""
        ts = timestamp or time.time()
        if ts - self._last_sample_time < 0.5:
            return
        self._last_sample_time = ts

        hit_rate = self.total_hits / max(1, self.total_lookups)
        elapsed = max(0.001, ts - self._start_time)
        qps = self.total_lookups / elapsed

        self.hit_rate_history.append(hit_rate)
        self.qps_history.append(qps)
        self.timestamps.append(ts)

    # ------------------------------------------------------------------
    def get_overview(self) -> Dict[str, Any]:
        hit_rate = self.total_hits / max(1, self.total_lookups)
        elapsed = max(0.001, time.time() - self._start_time)
        return {
            "total_lookups": self.total_lookups,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(hit_rate, 4),
            "qps": round(self.total_lookups / elapsed, 1),
            "consolidations": self.total_consolidations,
            "uptime_seconds": round(elapsed, 1),
        }

    # ------------------------------------------------------------------
    def get_ring_stats(self) -> List[Dict[str, Any]]:
        """Calculate L1/L2/L3 equivalent hit rates"""
        result = []
        for rs in self.ring_stats:
            level = "L1" if rs.ring_id < 3 else ("L2" if rs.ring_id < 6 else "L3")
            result.append(
                {
                    "ring_id": rs.ring_id,
                    "name": rs.ring_name,
                    "level": level,
                    "hits": rs.hits_from_ring,
                    "hit_rate": round(rs.hit_rate_from_ring(), 4),
                    "avg_size": round(rs.get_avg_size(), 1),
                    "max_size": rs.get_max_size(),
                }
            )
        return result

    # ------------------------------------------------------------------
    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "overview": self.get_overview(),
            "rings": self.get_ring_stats(),
            "time_series": {
                "hit_rates": self.hit_rate_history[-100:],
                "qps": self.qps_history[-100:],
            },
        }

    # ------------------------------------------------------------------
    def print_dashboard(self):
        """Print dashboard to terminal"""
        ov = self.get_overview()
        print("\n" + "=" * 60)
        print("  Year-Ring Cache Performance Dashboard")
        print("=" * 60)
        print(f"  Total Lookups: {ov['total_lookups']:,}  |  Hit Rate: {ov['hit_rate']:.1%}")
        print(f"  QPS: {ov['qps']:.1f}  |  Consolidations: {ov['consolidations']}")
        print("-" * 60)

        for rs in self.get_ring_stats():
            bar = "█" * max(1, int(rs["hit_rate"] * 20))
            print(f"  [{rs['level']}] {rs['name']}: {bar} {rs['hit_rate']:.1%} ({rs['hits']} hits)")
        print("=" * 60)

    # ------------------------------------------------------------------
    def export_json(self, filepath: str):
        data = self.get_performance_metrics()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def with_analytics(cache_instance) -> CachePerformanceDashboard:
    """Attach an analytics dashboard to a YearRingCache instance"""
    dashboard = CachePerformanceDashboard(num_rings=cache_instance.num_rings)

    original_lookup = cache_instance.lookup
    original_store = cache_instance.store

    def tracked_lookup(packet):
        hit, label, conf = original_lookup(packet)
        ring_id = None
        if hit:
            for idx in reversed(range(cache_instance.num_rings)):
                if packet.content_hash in cache_instance.rings[idx]:
                    ring_id = idx
                    break
        dashboard.record_lookup(ring_id, hit)
        total = sum(len(r) for r in cache_instance.rings)
        dashboard.update_global_stats(total)
        return hit, label, conf

    def tracked_store(packet, label, computed_result=None):
        original_store(packet, label, computed_result)
        for idx in range(cache_instance.num_rings):
            dashboard.record_ring_size(idx, len(cache_instance.rings[idx]))

    cache_instance.lookup = tracked_lookup
    cache_instance.store = tracked_store
    cache_instance.dashboard = dashboard
    return dashboard

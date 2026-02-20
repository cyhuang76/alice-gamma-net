# -*- coding: utf-8 -*-
"""
Alice Smart System — Core Engine
Γ-Net v4 Communication Protocol Core Engine + Unified Electrical Signal Framework

Core Principles:
1. Message packet routing (not matrix operations) — O(1) classification
2. Year-ring cache first (hit = zero computation)
3. Left/right brain activate on demand — saves 60-80% computation
4. Error-driven minimal correction — only fix the differences
5. Unified electrical signal framework — coaxial cable model
"""

from alice.core.signal import (
    BrainWaveBand,
    ElectricalSignal,
    CoaxialChannel,
    TransmissionReport,
    SignalBus,
    REGION_IMPEDANCE,
)
from alice.core.protocol import (
    Priority,
    Modality,
    MessagePacket,
    PriorityRouter,
    YearRingCache,
    BrainHemisphere,
    ErrorCorrector,
    GammaNetV4Protocol,
)
from alice.core.cache_analytics import CachePerformanceDashboard, with_analytics
from alice.core.cache_persistence import CachePersistence, CacheCheckpoint

__all__ = [
    # Unified Electrical Signal
    "BrainWaveBand",
    "ElectricalSignal",
    "CoaxialChannel",
    "TransmissionReport",
    "SignalBus",
    "REGION_IMPEDANCE",
    # Basic Enums and Data
    "Priority",
    "Modality",
    "MessagePacket",
    # Core Engine Components
    "PriorityRouter",
    "YearRingCache",
    "BrainHemisphere",
    "ErrorCorrector",
    # Integrated System
    "GammaNetV4Protocol",
    # Analytics and Persistence
    "CachePerformanceDashboard",
    "with_analytics",
    "CachePersistence",
    "CacheCheckpoint",
]

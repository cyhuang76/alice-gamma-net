# -*- coding: utf-8 -*-
"""Alice Smart System — Performance Benchmark"""

import time
import numpy as np
from alice.alice_brain import AliceBrain
from alice.core.protocol import (
    MessagePacket, Priority, Modality,
    PriorityRouter, YearRingCache, BrainHemisphere, ErrorCorrector,
    GammaNetV4Protocol,
)
from alice.brain.fusion_brain import FusionBrain
from alice.modules.working_memory import WorkingMemory
from alice.modules.reinforcement import ReinforcementLearner
from alice.modules.causal_reasoning import CausalReasoner
from alice.modules.meta_learning import MetaLearner


def fmt(val, unit=""):
    if val >= 1_000_000:
        return f"{val/1_000_000:.2f}M{unit}"
    if val >= 1_000:
        return f"{val/1_000:.1f}K{unit}"
    return f"{val:.1f}{unit}"


def bench_message_packet(n=100_000):
    """Packet creation speed"""
    signals = [np.random.rand(8) for _ in range(n)]
    t0 = time.perf_counter()
    for sig in signals:
        MessagePacket.from_signal(sig)
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed


def bench_priority_router(n=100_000):
    """Router throughput"""
    router = PriorityRouter()
    packets = [
        MessagePacket.from_signal(
            np.random.rand(8),
            priority=[Priority.BACKGROUND, Priority.NORMAL, Priority.HIGH, Priority.CRITICAL][i % 4],
        )
        for i in range(n)
    ]
    t0 = time.perf_counter()
    for pkt in packets:
        router.route(pkt)
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed


def bench_cache_write(n=50_000):
    """Cache write speed"""
    cache = YearRingCache()
    packets = [MessagePacket.from_signal(np.random.rand(8)) for _ in range(n)]
    t0 = time.perf_counter()
    for i, pkt in enumerate(packets):
        cache.store(pkt, f"label_{i}")
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed


def bench_cache_hit(n=100_000):
    """Cache hit query speed"""
    cache = YearRingCache()
    patterns = [np.random.rand(8) for _ in range(100)]
    for i, p in enumerate(patterns):
        pkt = MessagePacket.from_signal(p)
        cache.store(pkt, f"pattern_{i}")

    # Repeatedly query known patterns
    lookup_packets = [MessagePacket.from_signal(patterns[i % 100]) for i in range(n)]
    t0 = time.perf_counter()
    hits = 0
    for pkt in lookup_packets:
        hit, _, _ = cache.lookup(pkt)
        if hit:
            hits += 1
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed, hits / n


def bench_cache_miss(n=100_000):
    """Cache miss query speed"""
    cache = YearRingCache()
    packets = [MessagePacket.from_signal(np.random.rand(8)) for _ in range(n)]
    t0 = time.perf_counter()
    for pkt in packets:
        cache.lookup(pkt)
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed


def bench_protocol_learn_recognize(n_learn=100, n_recog=10_000):
    """Full protocol: learn + recognize"""
    proto = GammaNetV4Protocol()
    patterns = {f"p_{i}": np.random.rand(16) for i in range(n_learn)}
    for label, data in patterns.items():
        proto.learn(label, data)

    # Recognize known patterns (cache hit path)
    keys = list(patterns.keys())
    t0 = time.perf_counter()
    for i in range(n_recog):
        data = patterns[keys[i % n_learn]]
        proto.recognize(data)
    elapsed = time.perf_counter() - t0
    stats = proto.get_stats()
    return n_recog / elapsed, elapsed, stats["cache_serve_rate"], stats["avg_compute_ratio"]


def bench_protocol_noisy(n_learn=50, n_recog=5_000):
    """Full protocol: noisy recognition (compute path)"""
    proto = GammaNetV4Protocol()
    patterns = {f"p_{i}": np.random.rand(16) for i in range(n_learn)}
    for label, data in patterns.items():
        proto.learn(label, data)

    keys = list(patterns.keys())
    t0 = time.perf_counter()
    for i in range(n_recog):
        data = patterns[keys[i % n_learn]] + np.random.randn(16) * 0.3
        proto.recognize(data)
    elapsed = time.perf_counter() - t0
    stats = proto.get_stats()
    return n_recog / elapsed, elapsed, stats["cache_serve_rate"], stats["avg_compute_ratio"]


def bench_fusion_brain(n=1_000):
    """FusionBrain full 5-step cycle"""
    fb = FusionBrain(neuron_count=100)
    signals = [np.random.rand(100) for _ in range(n)]
    t0 = time.perf_counter()
    for sig in signals:
        fb.process_stimulus(sig)
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed


def bench_working_memory(n=100_000):
    """Working memory store + retrieve"""
    wm = WorkingMemory(capacity=7)
    t0 = time.perf_counter()
    for i in range(n):
        wm.store(f"key_{i % 20}", f"value_{i}")
        wm.retrieve(f"key_{i % 20}")
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed


def bench_rl(n=10_000):
    """Reinforcement learning update + choose"""
    rl = ReinforcementLearner()
    actions = ["a", "b", "c", "d"]
    t0 = time.perf_counter()
    for i in range(n):
        state = f"s_{i % 50}"
        action, _ = rl.choose_action(state, actions)
        rl.update(state, action, np.random.rand(), f"s_{(i+1) % 50}", actions)
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed


def bench_causal(n=5_000):
    """Causal reasoning observe + infer"""
    cr = CausalReasoner()
    t0 = time.perf_counter()
    for i in range(n):
        cr.observe({"A": np.random.rand(), "B": np.random.rand(), "C": np.random.rand()})
    for i in range(n):
        cr.infer("A", "C")
    elapsed = time.perf_counter() - t0
    return (n * 2) / elapsed, elapsed


def bench_alice_full(n=500):
    """AliceBrain full cognitive cycle"""
    alice = AliceBrain(neuron_count=50)
    signals = [np.random.rand(50) for _ in range(n)]
    t0 = time.perf_counter()
    for sig in signals:
        alice.perceive(sig)
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed


def bench_alice_full_cycle(n=200):
    """AliceBrain perceive+think+act+learn full cycle"""
    alice = AliceBrain(neuron_count=30)
    actions = ["approach", "avoid", "observe"]
    t0 = time.perf_counter()
    for i in range(n):
        alice.perceive(np.random.rand(30))
        alice.think("why?", {"x": float(i), "y": float(i) * 0.5})
        r = alice.act(f"s{i%10}", actions)
        alice.learn_from_feedback(f"s{i%10}", r["chosen_action"], np.random.rand(), f"s{(i+1)%10}", actions)
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed


def main():
    print()
    print("=" * 72)
    print("  🧠 Alice Smart System — Performance Benchmark")
    print("=" * 72)
    print()

    results = []

    # --- Core Engine ---
    print("  ┌─ Core Engine (Layer 3) ─────────────────────────────────────┐")

    rate, t = bench_message_packet()
    results.append(("  Packet creation (100K)", rate, t))
    print(f"  │  Packet creation      {fmt(rate, '/s'):>12s}   ({t*1000:.0f}ms / 100K)      │")

    rate, t = bench_priority_router()
    results.append(("  Priority routing (100K)", rate, t))
    print(f"  │  Priority routing      {fmt(rate, '/s'):>12s}   ({t*1000:.0f}ms / 100K)      │")

    rate, t = bench_cache_write()
    results.append(("  Cache write (50K)", rate, t))
    print(f"  │  Cache write           {fmt(rate, '/s'):>12s}   ({t*1000:.0f}ms / 50K)       │")

    rate, t, hit_rate = bench_cache_hit()
    results.append(("  Cache hit query (100K)", rate, t))
    print(f"  │  Cache hit query       {fmt(rate, '/s'):>12s}   (hit rate {hit_rate:.1%})       │")

    rate, t = bench_cache_miss()
    results.append(("  Cache miss query (100K)", rate, t))
    print(f"  │  Cache miss query      {fmt(rate, '/s'):>12s}   ({t*1000:.0f}ms / 100K)      │")

    print("  └──────────────────────────────────────────────────────────┘")
    print()

    # --- Protocol System ---
    print("  ┌─ Protocol System (full flow) ─────────────────────────────┐")

    rate, t, cache_rate, compute = bench_protocol_learn_recognize()
    print(f"  │  Exact match recog    {fmt(rate, '/s'):>12s}   (cache:{cache_rate:.0%} compute:{compute:.0%})│")

    rate, t, cache_rate, compute = bench_protocol_noisy()
    print(f"  │  Noisy recognition     {fmt(rate, '/s'):>12s}   (cache:{cache_rate:.0%} compute:{compute:.0%})│")

    print("  └──────────────────────────────────────────────────────────┘")
    print()

    # --- FusionBrain ---
    print("  ┌─ FusionBrain (Layer 4) ──────────────────────────────────┐")

    rate, t = bench_fusion_brain()
    latency = t / 1000 * 1000  # ms per op
    print(f"  │  5-step cycle (100n)  {fmt(rate, '/s'):>12s}   ({latency:.2f}ms/cycle)       │")

    print("  └──────────────────────────────────────────────────────────┘")
    print()

    # --- v5 Cognitive Modules ---
    print("  ┌─ v5 Cognitive Modules (Layer 5) ──────────────────────────┐")

    rate, t = bench_working_memory()
    print(f"  │  Working memory ops   {fmt(rate, '/s'):>12s}   ({t*1000:.0f}ms / 100K)      │")

    rate, t = bench_rl()
    print(f"  │  RL update+choose  {fmt(rate, '/s'):>12s}   ({t*1000:.0f}ms / 10K)       │")

    rate, t = bench_causal()
    print(f"  │  Causal obs+infer     {fmt(rate, '/s'):>12s}   ({t*1000:.0f}ms / 10K)       │")

    print("  └──────────────────────────────────────────────────────────┘")
    print()

    # --- Full System ---
    print("  ┌─ AliceBrain Full System ───────────────────────────────┐")

    rate, t = bench_alice_full()
    lat = (t / 500) * 1000
    print(f"  │  perceive only        {fmt(rate, '/s'):>12s}   ({lat:.2f}ms/cycle)           │")

    rate, t = bench_alice_full_cycle()
    lat = (t / 200) * 1000
    print(f"  │  Full cognitive cycle  {fmt(rate, '/s'):>12s}   ({lat:.2f}ms/cycle)           │")
    print(f"  │  (perceive+think+act+learn)                            │")

    print("  └──────────────────────────────────────────────────────────┘")
    print()

    # --- Summary ---
    print("  ┌─ Performance Summary ─────────────────────────────────────┐")
    print("  │                                                          │")

    # Compute key metrics
    alice = AliceBrain(neuron_count=100)
    for i in range(100):
        alice.perceive(np.random.rand(100))
    report = alice.introspect()
    fb = report["subsystems"]["fusion_brain"]
    proto = fb["protocol"]
    lb = proto["left_brain"]
    rb = proto["right_brain"]
    cache = proto["cache"]
    corrector = proto["corrector"]

    cache_hit_rate = proto["cache_serve_rate"]
    avg_compute = proto["avg_compute_ratio"]
    left_skip = 1 - lb["activation_rate"]
    right_skip = 1 - rb["activation_rate"]
    correction_skip = corrector["average_skip_rate"]

    print(f"  │  Cache hit rate:        {cache_hit_rate:.1%}                                │")
    print(f"  │  Avg compute ratio:     {avg_compute:.1%}                                │")
    print(f"  │  Left brain skip rate:  {left_skip:.1%}                                │")
    print(f"  │  Right brain skip rate: {right_skip:.1%}                                │")
    print(f"  │  Correction skip rate:  {correction_skip:.1%}                                │")
    print(f"  │  Year-ring consolidations: {cache['consolidations']}                                   │")
    print(f"  │  Dynamic capacity adj:  {cache['capacity_adjustments']}                                   │")
    print(f"  │                                                          │")

    total_compute_saved = (1.0 - avg_compute) * 100
    print(f"  │  ★ Total compute saved: {total_compute_saved:.1f}%                                │")
    print(f"  │                                                          │")
    print("  └──────────────────────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    main()

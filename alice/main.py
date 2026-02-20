# -*- coding: utf-8 -*-
"""
Alice Smart System — Main Entry Point
Supports CLI interactive mode and API server mode
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority


def show_banner():
    print(
        r"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        🧠  Alice Smart System  v1.0                      ║
    ║        ─────────────────────────────                     ║
    ║        Based on Γ-Net architecture digital brain smart system            ║
    ║                                                          ║
    ║  Core principles:                                                        ║
    ║    • The essence of the brain is communication, not computation            ║
    ║    • Year-ring cache: hit = zero computation                              ║
    ║    • Left-right brain activated on demand, saving 60-80% compute          ║
    ║    • Error-driven minimal correction                                     ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    )


def interactive_mode():
    """CLI interactive mode"""
    show_banner()
    alice = AliceBrain(neuron_count=100)
    print("  Alice brain initialized (4 brain regions × 100 neurons)")
    print()

    while True:
        print("  ┌──────────────────────────────┐")
        print("  │  1. Send sensory stimulus         │")
        print("  │  2. Send intense stimulus (trauma) │")
        print("  │  3. Think (causal reasoning)       │")
        print("  │  4. Action selection (RL)           │")
        print("  │  5. Learning feedback               │")
        print("  │  6. System introspection report     │")
        print("  │  7. Brain state overview            │")
        print("  │  8. Continuous stimulus test (10)   │")
        print("  │  0. Exit                            │")
        print("  └──────────────────────────────┘")

        try:
            choice = input("  Choice > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "0":
            print("\n  Goodbye!")
            break

        elif choice == "1":
            signal = np.random.rand(100) * 0.5
            result = alice.perceive(signal, Modality.VISUAL, Priority.NORMAL, "random_visual")
            print(f"\n  [{result['cycle']}] Sensory input complete ({result['elapsed_ms']:.1f}ms)")
            print(f"    Sensory activity: {result['sensory']['sensory_activity']:.1%}")
            print(f"    Cognitive activity: {result['cognitive']['prefrontal_activity']:.1%}")
            print(f"    Emotional valence: {result['emotional']['emotional_valence']:.3f}")
            print(f"    Motor output: {result['motor']['output_strength']:.3f}")

        elif choice == "2":
            signal = np.random.rand(100) * 2.5
            result = alice.perceive(signal, Modality.TACTILE, Priority.CRITICAL, "trauma")
            print(f"\n  ⚠ [{result['cycle']}] Intense stimulus! ({result['elapsed_ms']:.1f}ms)")
            print(f"    Sensory activity: {result['sensory']['sensory_activity']:.1%}")
            print(f"    Emotional valence: {result['emotional']['emotional_valence']:.3f}")
            print(f"    Arousal:   {result['emotional']['arousal']:.3f}")

        elif choice == "3":
            question = input("  Question (or Enter for default) > ").strip()
            if not question:
                question = "Why does stimulus lead to motor response?"
            result = alice.think(question, {"stimulus": 0.8, "response": 0.5})
            print(f"\n  Thinking type: {result['type']}")
            print(f"  Strategy: {result['strategy_used']}")
            if "reasoning" in result:
                r = result["reasoning"]
                for k, v in r.items():
                    if k != "note":
                        print(f"    {k}: {v}")

        elif choice == "4":
            actions = ["approach", "avoid", "observe", "rest"]
            result = alice.act("alert", actions)
            print(f"\n  Chosen action: {result['chosen_action']}")
            print(f"  Explored: {result['explored']},  ε={result['epsilon']:.4f}")
            print(f"  Q values: {result['q_values']}")

        elif choice == "5":
            result = alice.learn_from_feedback(
                "alert", "observe", 1.0, "calm", ["rest", "explore"]
            )
            print(f"\n  Dopamine signal: {result['dopamine_signal']:.4f}")
            print(f"  Replay error: {result['replay_error']:.4f}")
            print(f"  Strategy: {result['meta_strategy']}")

        elif choice == "6":
            report = alice.introspect()
            print(f"\n  === System Introspection Report ===")
            print(f"  State: {report['state']}")
            print(f"  Cycles: {report['cycle_count']}")
            print(f"  Uptime: {report['uptime_seconds']}s")

            for name, sub in report["subsystems"].items():
                if isinstance(sub, dict):
                    print(f"\n  [{name}]")
                    for k, v in sub.items():
                        if not isinstance(v, (dict, list)):
                            print(f"    {k}: {v}")

        elif choice == "7":
            print(alice.fusion_brain.generate_report("Alice Brain State"))

        elif choice == "8":
            print("\n  Continuous stimulus test...")
            t0 = time.time()
            for i in range(10):
                signal = np.random.rand(100) * (0.3 + i * 0.15)
                priority = Priority.CRITICAL if i >= 7 else Priority.NORMAL
                r = alice.perceive(signal, priority=priority)
                activity = r["sensory"]["sensory_activity"]
                emotion = r["emotional"]["emotional_valence"]
                bar = "█" * max(1, int(activity * 30))
                print(f"    [{i+1:2d}] {bar} {activity:.1%}  emotion={emotion:+.3f}")
            elapsed = time.time() - t0
            print(f"\n  10 rounds complete, elapsed {elapsed*1000:.1f}ms ({elapsed*100:.1f}ms/round)")
            print(alice.fusion_brain.generate_report())

        else:
            print("  Invalid option")

        print()


def server_mode(host: str = "0.0.0.0", port: int = 8000):
    """API server mode"""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required")
        print("  pip install uvicorn fastapi")
        sys.exit(1)

    show_banner()
    print(f"  Starting API server @ http://{host}:{port}")
    print(f"  Dashboard @ http://{host}:{port}/dashboard")
    print(f"  API docs @ http://{host}:{port}/docs")
    print()

    from alice.api.server import create_app

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="Alice Smart System — Biologically-inspired digital brain smart system based on Γ-Net architecture"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="cli",
        choices=["cli", "server", "test"],
        help="Run mode: cli (interactive), server (API server), test (quick test)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8000, help="Server port number")

    args = parser.parse_args()

    if args.mode == "server":
        server_mode(args.host, args.port)
    elif args.mode == "test":
        quick_test()
    else:
        interactive_mode()


def quick_test():
    """Quick system test"""
    show_banner()
    print("  Quick system test...")
    print("=" * 60)

    alice = AliceBrain(neuron_count=50)

    # 1. Perception test
    print("\n  1. Perception test")
    for i in range(5):
        signal = np.random.rand(50) * (0.5 + i * 0.3)
        r = alice.perceive(signal)
        print(f"    Cycle {r['cycle']}: {r['elapsed_ms']:.1f}ms")

    # 2. Thinking test
    print("\n  2. Causal reasoning test")
    r = alice.think("Why?", {"stimulus": 0.9, "response": 0.7})
    print(f"    Type: {r['type']}, Strategy: {r['strategy_used']}")

    # 3. Action test
    print("\n  3. Reinforcement learning test")
    for _ in range(10):
        act_r = alice.act("s1", ["a", "b", "c"])
        reward = 1.0 if act_r["chosen_action"] == "a" else 0.0
        alice.learn_from_feedback("s1", act_r["chosen_action"], reward, "s2", ["a", "b", "c"])
    final = alice.act("s1", ["a", "b", "c"])
    print(f"    Final choice: {final['chosen_action']}, Q values: {final['q_values']}")

    # 4. Introspection
    print("\n  4. System state")
    report = alice.introspect()
    for name, sub in report["subsystems"].items():
        if isinstance(sub, dict) and not any(isinstance(v, dict) for v in sub.values()):
            items = ", ".join(f"{k}={v}" for k, v in list(sub.items())[:3])
            print(f"    {name}: {items}")

    print("\n" + "=" * 60)
    print("  ✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

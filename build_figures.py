#!/usr/bin/env python3
"""
build_figures.py — Gamma-Net figure generation API
===================================================
Unified entry point for all paper figure generation scripts.
Automatically selects a compatible Python runtime (avoids
Python 3.14 numpy segfault) and runs all figure generators.

Usage:
    python build_figures.py              # Generate all figures
    python build_figures.py --paper 0    # Generate P0 figures only
    python build_figures.py --paper 1 4  # Generate P1 and P4
    python build_figures.py --list       # List available generators
"""
import argparse
import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Candidate Python paths (order = preference) ──
PYTHON_CANDIDATES = [
    Path(r"C:\Users\pinhu\AppData\Local\Programs\Python\Python311\python.exe"),
    Path(r"C:\Python311\python.exe"),
    Path(r"C:\Python312\python.exe"),
    Path(r"C:\Python310\python.exe"),
]

# ── Paper → script mapping ──
GENERATORS = {
    0: ("Paper 0 — Framework",       "experiments/figgen/generate_paper0_figures.py"),
    1: ("Paper 1 — Topology & Mind", "experiments/figgen/generate_paper1_figures.py"),
    2: ("Paper 2 — Dual Networks",   "experiments/figgen/generate_paper_ii_figures.py"),
    3: ("Paper 3 — Temporal",        "experiments/figgen/generate_paper_iii_figures.py"),
    4: ("Paper 4 — Pathology",       "experiments/figgen/generate_paper4_figures.py"),
    5: ("Paper 5 — Verification",    "experiments/figgen/generate_paper5_figures.py"),
}


def find_compatible_python() -> str:
    """Find a Python 3.10-3.13 that can run numpy/matplotlib."""
    # Check explicit candidate paths
    for p in PYTHON_CANDIDATES:
        if p.exists():
            return str(p)

    # Fallback: search PATH for python3.11, python3.12, etc.
    for ver in ["3.11", "3.12", "3.10", "3.13"]:
        name = f"python{ver}"
        found = shutil.which(name)
        if found:
            return found

    # Last resort: use current interpreter with a warning
    major, minor = sys.version_info[:2]
    if minor >= 14:
        print(f"[WARN] Current Python {major}.{minor} may crash with numpy.")
        print("       Install Python 3.11-3.13 for reliable figure generation.")
    return sys.executable


def run_generator(python: str, paper_id: int) -> bool:
    """Run a single paper's figure generator. Returns True on success."""
    title, script = GENERATORS[paper_id]
    script_path = ROOT / script

    if not script_path.exists():
        print(f"  [SKIP] {title}: {script} not found")
        return False

    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"  Script: {script}")
    print(f"{'─' * 50}")

    result = subprocess.run(
        [python, str(script_path)],
        cwd=str(ROOT),
        capture_output=False,
    )

    if result.returncode == 0:
        print(f"  [OK] {title}")
        return True
    else:
        print(f"  [FAIL] {title} (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Gamma-Net figure generation API"
    )
    parser.add_argument(
        "--paper", "-p", type=int, nargs="*",
        help="Paper numbers to generate (0-5). Default: all."
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List available generators and exit."
    )
    parser.add_argument(
        "--python", type=str, default=None,
        help="Path to Python interpreter to use."
    )
    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available figure generators:")
        for pid, (title, script) in sorted(GENERATORS.items()):
            exists = "OK" if (ROOT / script).exists() else "MISSING"
            print(f"  P{pid}: {title:30s} [{exists}] {script}")
        return

    # Find Python
    python = args.python or find_compatible_python()
    ver = subprocess.run(
        [python, "--version"], capture_output=True, text=True
    ).stdout.strip()
    print(f"Using: {python} ({ver})")

    # Determine which papers to generate
    papers = args.paper if args.paper is not None else sorted(GENERATORS.keys())

    # Validate
    invalid = [p for p in papers if p not in GENERATORS]
    if invalid:
        print(f"[ERROR] Unknown paper(s): {invalid}")
        print(f"        Valid: {sorted(GENERATORS.keys())}")
        sys.exit(1)

    # Run
    print(f"\nGenerating figures for: {', '.join(f'P{p}' for p in papers)}")
    results = {}
    for pid in papers:
        results[pid] = run_generator(python, pid)

    # Summary
    print(f"\n{'═' * 50}")
    print("Summary:")
    ok = sum(1 for v in results.values() if v)
    total = len(results)
    for pid, success in sorted(results.items()):
        status = "OK" if success else "FAIL"
        print(f"  P{pid}: [{status}] {GENERATORS[pid][0]}")
    print(f"\n  {ok}/{total} completed successfully")
    print(f"{'═' * 50}")

    if ok < total:
        sys.exit(1)


if __name__ == "__main__":
    main()

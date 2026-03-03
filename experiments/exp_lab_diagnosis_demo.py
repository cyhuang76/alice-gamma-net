# -*- coding: utf-8 -*-
"""exp_lab_diagnosis_demo.py — Lab-Γ Diagnostic Engine CLI Demo
================================================================

Usage:
    # Run demo with built-in clinical cases
    python experiments/exp_lab_diagnosis_demo.py

    # Diagnose from JSON file
    python experiments/exp_lab_diagnosis_demo.py --input patient.json

    # Interactive mode
    python experiments/exp_lab_diagnosis_demo.py --interactive

Demo cases:
    1. Acute hepatitis (AST/ALT elevated)
    2. Acute MI (Troponin/BNP elevated)
    3. DKA (Glucose/K deranged)
    4. Sepsis (WBC/CRP/PCT/Lactate elevated)
    5. CKD with anemia (Cr/BUN/Hb deranged)
    6. Hyperthyroidism (TSH suppressed, FT4 elevated)
    7. Iron deficiency anemia (Hb low, Ferritin low)
    8. Acute pancreatitis (Amylase/Lipase elevated)
"""

from __future__ import annotations

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alice.diagnostics import GammaEngine, load_disease_templates


# ============================================================================
# Built-in Clinical Cases
# ============================================================================

DEMO_CASES = {
    "acute_hepatitis": {
        "description": "45M — 黃疸、倦怠一週 (Jaundice & fatigue × 1 week)",
        "labs": {
            "AST": 480, "ALT": 520, "Bil_total": 3.2, "Bil_direct": 2.1,
            "Albumin": 2.8, "INR": 1.8, "GGT": 180,
            "WBC": 12.5, "CRP": 45, "Hb": 13.0, "Plt": 160,
        },
    },
    "acute_mi": {
        "description": "62M — 胸痛、冒冷汗 (Chest pain, diaphoresis)",
        "labs": {
            "Troponin": 25.0, "CK_MB": 220, "BNP": 1200,
            "Glucose": 160, "K": 5.2, "Cr": 1.4,
            "WBC": 15.0, "CRP": 18, "LDL": 185, "TC": 260,
            "Hb": 14.5, "Lactate": 3.5,
        },
    },
    "dka": {
        "description": "28F — 噁心、嘔吐、呼吸急促 (Nausea, vomiting, Kussmaul breathing)",
        "labs": {
            "Glucose": 450, "HbA1c": 12.5, "K": 5.8,
            "Na": 128, "CO2": 10, "BUN": 35, "Cr": 2.0,
            "WBC": 18.0, "Hb": 15.0, "Lactate": 4.0,
        },
    },
    "sepsis": {
        "description": "70M — 發燒、低血壓、意識改變 (Fever, hypotension, altered consciousness)",
        "labs": {
            "WBC": 22.0, "Neutrophils": 88, "CRP": 180, "PCT": 15.0,
            "Lactate": 5.5, "Cr": 2.5, "BUN": 45,
            "Plt": 80, "INR": 1.6, "Albumin": 2.2,
            "AST": 85, "ALT": 72, "Bil_total": 2.0,
            "Na": 132, "K": 5.5, "Glucose": 180,
        },
    },
    "ckd_anemia": {
        "description": "55F — 慢性疲倦、水腫 (Chronic fatigue, edema)",
        "labs": {
            "Cr": 4.5, "BUN": 65, "K": 5.5, "Na": 138, "Ca": 7.8,
            "Hb": 8.5, "Hct": 26, "Ferritin": 350,
            "Albumin": 3.0, "HbA1c": 7.2, "LDL": 150,
            "ALP": 180, "Vit_D": 12, "Uric_Acid": 9.5,
        },
    },
    "hyperthyroid": {
        "description": "35F — 心悸、體重減輕、手抖 (Palpitations, weight loss, tremor)",
        "labs": {
            "TSH": 0.02, "FT4": 3.8, "FT3": 8.5,
            "Glucose": 115, "Ca": 10.8, "ALP": 160,
            "WBC": 5.0, "Hb": 12.5, "AST": 50,
        },
    },
    "iron_deficiency": {
        "description": "30F — 頭暈、蒼白、月經過多 (Dizziness, pallor, menorrhagia)",
        "labs": {
            "Hb": 7.5, "Hct": 24, "MCV": 68, "RBC": 3.2,
            "Plt": 420, "Ferritin": 8, "WBC": 6.0,
        },
    },
    "acute_pancreatitis": {
        "description": "50M — 上腹劇痛放射至背部 (Severe epigastric pain radiating to back)",
        "labs": {
            "Amylase": 850, "Lipase": 1200, "WBC": 16.0, "CRP": 95,
            "Ca": 7.5, "Glucose": 200, "AST": 65, "ALT": 55,
            "BUN": 25, "Cr": 1.2, "Hb": 14.0,
        },
    },
}


def run_demo_cases(engine: GammaEngine) -> None:
    """Run all built-in demo cases."""
    for case_id, case in DEMO_CASES.items():
        print(f"\n{'#' * 72}")
        print(f"# Case: {case_id}")
        print(f"# {case['description']}")
        print(f"{'#' * 72}")
        print(engine.format_report(case["labs"], top_n=5))
        print()


def run_json_input(engine: GammaEngine, path: str) -> None:
    """Read lab values from JSON and diagnose."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lab_values = data.get("lab_values", data)
    if "patient_info" in data:
        info = data["patient_info"]
        print(f"Patient: {info.get('age', '?')} {info.get('sex', '?')}")

    print(engine.format_report(lab_values, top_n=5))


def run_interactive(engine: GammaEngine) -> None:
    """Interactive mode: enter lab values one by one."""
    print("=" * 60)
    print("Lab-Γ Diagnostic Engine — Interactive Mode")
    print(f"Available labs: {len(engine.mapper.available_labs)}")
    print("Enter lab values as: NAME VALUE (e.g. 'AST 480')")
    print("Type 'done' to run diagnosis, 'list' to see available labs")
    print("=" * 60)

    lab_values: dict[str, float] = {}
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line.lower() == "done":
            break
        if line.lower() == "list":
            labs = engine.mapper.available_labs
            for i in range(0, len(labs), 6):
                print("  " + ", ".join(labs[i : i + 6]))
            continue
        if line.lower() in ("quit", "exit"):
            return

        parts = line.split()
        if len(parts) != 2:
            print("  Format: NAME VALUE (e.g. 'AST 480')")
            continue

        name, val_str = parts
        try:
            val = float(val_str)
        except ValueError:
            print(f"  Invalid number: {val_str}")
            continue

        if name not in engine.mapper.catalogue:
            print(f"  Unknown lab: {name}. Type 'list' for available labs.")
            continue

        lab_values[name] = val
        print(f"  ✓ {name} = {val}")

    if lab_values:
        print(engine.format_report(lab_values, top_n=5))
    else:
        print("  No lab values entered.")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    # Load engine
    templates = load_disease_templates()
    engine = GammaEngine(templates=templates)
    print(f"Lab-Γ Engine loaded: {len(templates)} disease templates, "
          f"{len(engine.mapper.available_labs)} lab items")

    # Parse args
    args = sys.argv[1:]

    if "--input" in args:
        idx = args.index("--input")
        if idx + 1 < len(args):
            run_json_input(engine, args[idx + 1])
        else:
            print("Error: --input requires a file path")
    elif "--interactive" in args:
        run_interactive(engine)
    else:
        run_demo_cases(engine)


if __name__ == "__main__":
    main()

# Documentation Index

## Core Documents

| Document | Description |
|----------|-------------|
| [SYSTEM_MANUAL.md](SYSTEM_MANUAL.md) | Complete system architecture and operation guide |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Paper series audit: data consistency and cross-reference checks |
| [EMERGENCE_STANDARD.md](EMERGENCE_STANDARD.md) | E0 emergence standard: disease arises from C1/C2/C3 alone |
| [IRREDUCIBILITY_THEOREM.md](IRREDUCIBILITY_THEOREM.md) | Proof that Γ-Net cannot be reduced to simpler models |
| [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | Documented limitations, assumptions, and scope boundaries |
| [LAB_GAMMA_ENGINE.md](LAB_GAMMA_ENGINE.md) | Lab-to-Γ mapping engine: how blood panels become impedance vectors |
| [TASK_SOP.md](TASK_SOP.md) | Standard operating procedures for development tasks |

## Related Project Files

| File | Description |
|------|-------------|
| [alice_core.md](../alice_core.md) | Core theory axioms and constraints (C1/C2/C3) |
| [alice_digest.md](../alice_digest.md) | Condensed system overview |
| [README.md](../README.md) | Main project README |
| [CHANGELOG.md](../CHANGELOG.md) | Version history |

## Build & Run

```bash
# Generate all paper figures
python build_figures.py

# Run tests
pytest tests/

# Run specific experiment
python experiments/exp_unified_verification.py
```

---
name: "Physics Bug Report"
about: "Report a violation of Alice's 3 physics constraints (C1/C2/C3)"
title: "[PHYSICS BUG] "
labels: ["physics-violation", "bug"]
assignees: []
---

## Physics Constraint Violated

- [ ] C1: Energy Conservation (G^2 + T != 1)
- [ ] C2: Hebbian Update (incorrect dZ formula)
- [ ] C3: Signal Protocol (raw float instead of ElectricalSignal)

## Module / File Affected

```
# e.g. alice/modules/synapse.py, line 42
```

## Observed Behaviour

<!-- What wrong value or error did you see? -->

## Expected Behaviour (Physics)

<!-- What the physics equations require -->

## Minimal Reproducer

```python
# Paste the smallest code snippet that triggers the bug
```

## pytest Output

```
# Paste relevant pytest -v output
```

## Physics Derivation (if known)

<!-- Reference the equation from pyproject.toml or docs/ -->

## Checklist Before Submitting

- [ ] pytest still runs (even if some tests fail)
- [ ] I have checked docs/KNOWN_LIMITATIONS.md
- [ ] I am NOT using raw floats between modules

## Pull Request — Alice Gamma-Net Physics Gate

### Summary
<!-- One-line description of what this PR does -->

### Type of Change
- [ ] Physics model upgrade (C1/C2/C3 related)
- [ ] New module / feature
- [ ] Bug fix
- [ ] Refactor / cleanup
- [ ] Docs / papers
- [ ] Test addition

---

## Physics Compliance Checklist

> All three constraints MUST pass before merge.

### C1: Energy Conservation
- [ ] `G^2 + T = 1` verified in affected channels
- [ ] No unaccounted power loss or gain introduced

### C2: Hebbian Update
- [ ] Weight update uses exactly: `dZ = -eta * G * x_pre * x_post`
- [ ] No alternative gradient rule introduced without physics derivation

### C3: Signal Protocol
- [ ] All inter-module values use `ElectricalSignal` (not raw floats)
- [ ] `Z_source` and `Z_load` metadata present in every signal

---

## Test Results

```
# Paste full pytest output here
# Required: 100% pass rate
pytest tests/ -v
```

- [ ] All existing tests pass (0 failures)
- [ ] New tests added for every new function / module
- [ ] `docs/KNOWN_LIMITATIONS.md` updated if a limitation was added

---

## Nonlinear Physics Models Affected
<!-- Check if any of these models are touched by this PR -->
- [ ] Butterworth bandwidth roll-off
- [ ] Johnson-Nyquist thermal noise
- [ ] Arrhenius aging
- [ ] Quemada viscosity
- [ ] Autocorrelation frequency estimation
- [ ] None of the above

---

## Ethics Review (if applicable)
- [ ] No consciousness-loop features added
- [ ] No deceptive identity behaviour introduced
- [ ] Consistent with Position A (Alice = physics tool)

---

## Notes for Reviewer
<!-- Any context the reviewer needs to evaluate the physics correctness -->

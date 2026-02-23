# Task SOP — Γ-Net ALICE

> Sections §1–§3 must be completed BEFORE execution begins.
> Sections §4–§5 must be completed AFTER execution ends.
> Any task without a filled SOP is considered not started.
> Success criteria in §2 must be defined before execution and cannot be modified afterward.

---

## Workflow

Human defines task → fill §1–§3 → hand off to Agent
→ Agent fills §4–§5
→ All ✅ → commit + push
→ Any ❌ → write KNOWN_LIMITATIONS.md → do NOT commit → report to human


---

## SOP-[ID] [Task Name]

**Date:**      YYYY-MM-DD
**Executor:**  Human / AI Agent
**Commit:**    (fill after execution)

---

### §1 Task Definition (fill before execution)

- **Goal:**        One sentence describing what must be achieved
- **In-scope:**    Files that will be modified
- **Out-of-scope:** Files that must NOT be touched
- **Trigger:**     Why this task is being done now

---

### §2 Success Criteria (define before execution — cannot be changed afterward)

| Criterion | Verification Method | Pass Condition |
|---|---|---|
| All tests pass | `pytest tests/ -q` | 0 failures, 0 errors |
| Physics constraint (1) | Code review | All new signals use ElectricalSignal |
| Physics constraint (2) | Code review | Learning via ΔZ = −η·Γ·x_pre·x_post only |
| Physics constraint (3) | Unit test | Γ² + T = 1 at every tick |
| Paper numbers in sync | grep test count | All files report identical counts |
| File classification correct | Directory check | No misplaced files |

---

### §3 Risk Assessment (fill before execution)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Breaking existing tests | Medium | High | Run pytest baseline before starting |
| Physics constraint violation | Low | High | Verify signal type on every new function |
| Paper number mismatch | Medium | Medium | grep all counts before final commit |
| Agent context overflow | High | Medium | Use @file precise references only |

---

### §4 Execution Log (fill during execution)

- [ ] Baseline pytest: `______ passed, ______ failed`
- [ ] Step 1:
- [ ] Step 2:
- [ ] Step 3:
- [ ] Post-execution pytest: `______ passed, ______ failed`

---

### §5 Result Verification (fill after execution)

| Criterion | Result | Notes |
|---|---|---|
| All tests pass | ✅ / ❌ | |
| Physics constraint (1) | ✅ / ❌ | |
| Physics constraint (2) | ✅ / ❌ | |
| Physics constraint (3) | ✅ / ❌ | |
| Paper numbers in sync | ✅ / ❌ | |
| File classification correct | ✅ / ❌ | |

**Task status:** ✅ Complete / ❌ Failed (see KNOWN_LIMITATIONS.md SOP-[ID])
**Commit SHA:**
**Test count change:** `______ → ______`

---

<!-- Append new SOP entries below this line — copy the full block above -->

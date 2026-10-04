# gridtrust

Decide which answers of a power-flow surrogate (a neural model such as GENCO, or DC power flow) can be
accepted without running the AC solver, and put a number on what gets missed.

Three small modules, numpy / scipy / pandas only (no torch, no Julia):

| Module | What it does |
|---|---|
| `gridtrust.newton` | AC power flow from plain arrays (polar full Newton, sparse LU). `refine(vm, va, steps=1)` takes Newton step(s) from predicted voltages and returns corrected voltages, line loadings and the remaining bus mismatch: a self-check that costs nothing extra. |
| `gridtrust.conformal` | Label-conditional split conformal band on predicted line loadings: `calibrate` → width, `flag`, `evaluate` (missed / wasted / AC-call share, Clopper-Pearson upper bound), plus `anchored` and `shift_sigma` for a band anchored on the base case. |
| `gridtrust.staged` | The decision for one state: surrogate → band → Newton step + mismatch check → needs AC, with the stage that decided and why. |

The code is extracted from an entry to the GENCO Challenge; `tests/test_equivalence_repo.py` checks it
against that entry's analysis scripts and stored results when they are present.

## Install

    pip install -e gridtrust            # from the repository root
    pip install -e "gridtrust[test]" && pytest gridtrust

## Quick start

```python
import numpy as np
from gridtrust import PowerFlowProblem, calibrate, decide

# 1. the grid state: bus types (1 PQ, 2 PV, 3 REF), setpoints, injections (gen - load, p.u.), branches
prob = PowerFlowProblem.from_branch_parameters(
    bus_type=[3, 2, 1], vm_set=[1.02, 1.01, 1.0], va_ref=0.0,
    p_inj=[0.0, 0.4, -0.9], q_inj=[0.0, 0.0, -0.3],
    f=[0, 0, 1], t=[1, 2, 2], r=[0.01, 0.02, 0.015], x=[0.1, 0.2, 0.15], rate=[1.0, 0.6, 0.6])

# 2. a surrogate's voltages (here: made up) and the loadings they imply
vm, va = np.array([1.02, 1.01, 0.96]), np.array([0.0, -0.01, -0.09])
predicted = prob.loading(prob.start(vm_pq=vm, va=va))

# 3. band width from calibration readings (predicted vs AC loading) with a 5% allowed miss rate
cal_pred, cal_ac = np.random.default_rng(0).uniform(0.9, 1.1, (2, 500))
width = calibrate(cal_pred, cal_ac, limit=1.0, alpha=0.05)

# 4. staged decision; the Newton step only runs if a line is inside the band
d = decide(predicted, width, refine=lambda: prob.refine(vm, va, steps=1), limit=1.0)
print(d.stage, d.reason, d.needs_ac, d.mismatch_mw)
```

`examples/synthetic_screening.py` runs the whole workflow on a synthetic 30-bus grid with a noisy stand-in surrogate.

## Notes

- Units: per unit on `base_mva`, angles in radians, loadings as a share of the rating. `from_matpower` takes MATPOWER tables (MW, degrees).
- The Newton solver has no step control and no generator Q limits. That is deliberate: it measures how good a start is. `solve()` reports `converged=False` instead of raising.
- `refine` uses the true network model, so its output is "surrogate plus physics". Trust it only when `result.selfcheck(threshold_mw)` holds; the threshold is yours to set from data (the entry uses 1 MW on IEEE 118).
- The conformal guarantee is marginal: on average over calibration and test draws at most a share alpha of real overloads is not flagged, provided calibration and test states are exchangeable. It does not hold per state, and it breaks under distribution shift (e.g. calibrated on N-1/N-2, used on N-4). `calibrate` returns `inf` (everything goes to AC) when fewer than `1/alpha - 1` real overloads are available.
- Split calibration and test by whole groups (e.g. outage sets), not by line reading.

MIT licence.

# Newton refinement and conformal screening for gridfm-graphkit

> **DRAFT. Not submitted anywhere.** No issue, PR or discussion has been opened. This is a local design note written against gridfm-graphkit 0.9.0.3 as installed from PyPI (the wheel ships no `tests/` directory, so the test layout below is a proposal, not a description). The prototype is `upstream/newton_refine.py`.

## Summary

Two small, optional additions to the power-flow evaluation path:

1. **Newton refinement of predicted voltages.** After the model has predicted `Vm` / `Va`, run one (or `k`) full Newton-Raphson step(s) on the AC power-flow equations of the same graph. Return corrected voltages, branch flows and loadings, and the bus power mismatch that remains. The mismatch is a physics self-check that costs nothing extra.
2. **Conformal screening.** An evaluation utility that calibrates a label-conditional split-conformal band on predicted line loadings. A state with no line inside the band is accepted without an AC solve; the share of real overloads missed this way is at most `alpha` on average. It reports the missed share, a Clopper-Pearson upper bound, the share of states sent to AC and the share of AC calls wasted on secure states.

Both use only what graphkit already has in memory at test time: the denormalised batch and the model output.

## Why (evidence from one GENCO Challenge entry)

All numbers are from released checkpoints, no training. Source file in brackets.

**One Newton step removes the error tail of the prediction.**

- GOC 500, N-1 to N-5, 14,992 scenarios, GENCO Tiny: the 99th percentile of the loading error falls from 6.5 to 0.06 points of rating [`submission/POST.md`]. Median / p99 / max: GENCO 0.325 / 6.50 / 134, after one step 0.001 / 0.06 / 22.8, after two steps 0.000 / 0.00 / 8.8 [`work/goc500/logs/analysis_newton_seed0.log`].
- IEEE 118 (366 states): p99 1.69 → 0.01, max 34.5 → 3.3 points. Texas 2000 (305 states): p99 1.93 → 0.01, max 16.4 → 0.3 points [`work/onestep_conformal.log`].
- As a screen on GOC 500, "GENCO + one Newton step, threshold = the limit" sends 12.5% of scenarios to AC (12.5% really have an overload), 0.1% of the secure ones, and misses 9 of 2,621 overloaded lines, the worst at 100.6%. Two steps miss none [`submission/POST.md`].
- The step only pays off from the model's prediction. On GOC 500, DC + one step sends 88.1% to AC and a flat start + one step 97.3% [`submission/POST.md`]. On IEEE 118 the p99 error after one step is 10.38 points from DC and 20.37 from a flat start [`work/onestep_conformal.log`].
- Scale: on GOC 2000 one step gives 6.6% sent / 0.6% secure sent / 1 of 369 lines missed. On GOC 10000 one step is not enough (55.0% / 37.0% / 7 of 507); two steps give 29.8% / 0.1% / 0 [`submission/POST.md`]. Hence `k` is a knob, not a constant.
- The mismatch catches divergence: with checkpoint seed 1 on GOC 500, one scenario's loading error after the step was 14,684 points and its mismatch 9e4 MW [`work/goc500/logs/analysis_newton_seed1.log`].
- Cost: on 10,000 buses one Newton step took 19 ms against 62 ms for GENCO Tiny inference; the step is "about a fifth of a full AC solve in the same code" [`submission/POST.md`]. Indicative only.

**A conformal band puts a number on the misses when no step is taken.**

- IEEE 118, 2,375 outage states, GENCO Base, anchored surrogate with adaptive width, `alpha` = 1%: 31.8% of states go to AC (31.2% really have an overload), 0.9% of the secure states are escalated, 0 of 792 held-out overloads are missed, 95% upper bound 0.38% [`work/conformal_case118_ieee_base.log`, row `genco_d shift 1%`; the write-up quotes 31.9% and 1.0%, which is the `genco_d global` row].
- Plain GENCO, one global width, same split: `alpha` = 5% misses 15 of 792 (1.89%, upper bound 2.90%) with 30.0% of states sent to AC; `alpha` = 1% misses 0 of 792 with 31.6% sent [`work/conformal_case118_ieee_base.log`]. This is the variant proposed below; anchoring is left out (see Open questions).
- Coverage over 100 random splits, plain GENCO, global width: mean per-line missed share 9.90 / 5.07 / 2.02 / 0.99% for `alpha` = 10 / 5 / 2 / 1% [`work/conformal_case118_ieee_base.log`].
- The band is honest about its price. On GOC 500 with raw GENCO, `alpha` = 1% sends 94.9% of scenarios to AC [`submission/POST.md`]. On the same data the example entry's F-beta threshold misses 40 of 1,327 overloaded lines, the worst at 139.6% [`submission/POST.md`].
- It breaks under shift: 6.5% and 7.1% missed at `alpha` = 5% for N-3 and N-4 outages when calibrated on N-1/N-2 [`submission/POST.md`].

## Graphkit conventions this builds on

Read from the 0.9.0.3 source. Every name below exists there.

**Graph layout** (`datasets/graph_builder.py`, `build_hetero_data`)

- Node stores `bus` and `gen`; edge stores `("bus","connects","bus")`, `("gen","connected_to","bus")`, `("bus","connected_to","gen")`; `data["scenario_id"]`.
- `data["bus"].x` has the 15 columns of `BUS_FEATURES`; `data["bus"].y` is a clone of the first five (`Pd, Qd, Qg, Vm, Va`); `data["gen"].y` is `p_mw`.
- `data["bus"].static` keeps raw `[MIN_VM_H, MAX_VM_H, MIN_QG_H, MAX_QG_H, VN_KV]`; `data["bus","connects","bus"].static` keeps raw `[ANG_MIN, ANG_MAX, RATE_A]`. The normalizer does not touch `.static`.
- **Directed edges.** Every branch appears twice: all forward copies (`from_bus → to_bus`), then all reverse copies. The forward copy carries `pf, qf, Yff, Yft`; the reverse copy carries `pt, qt, Ytt, Ytf`. So each directed edge holds the admittances of its own sending end, which is why the index constants are named `YFF_TT_*` and `YFT_TF_*`. After `RemoveInactiveBranches` (`datasets/transforms.py`) only in-service branches remain and the `B_ON` column is dropped, leaving 10 edge columns.

**Index constants** (`datasets/globals.py`)

| What | Constant = column |
|---|---|
| bus input | `PD_H`=0, `QD_H`=1, `QG_H`=2, `VM_H`=3, `VA_H`=4, `PQ_H`=5, `PV_H`=6, `REF_H`=7, `MIN_VM_H`=8, `MAX_VM_H`=9, `MIN_QG_H`=10, `MAX_QG_H`=11, `GS`=12, `BS`=13, `VN_KV`=14 |
| bus output | `VM_OUT`=0, `VA_OUT`=1, `PG_OUT`=2, `QG_OUT`=3 (`PD_OUT`=4, `QD_OUT`=5 for random masking) |
| generator | `PG_H`=0, `MIN_PG`=1, `MAX_PG`=2, `C0_H`..`C2_H`=3..5, `G_ON`=6 |
| edge | `P_E`=0, `Q_E`=1, `YFF_TT_R`=2, `YFF_TT_I`=3, `YFT_TF_R`=4, `YFT_TF_I`=5, `TAP`=6, `ANG_MIN`=7, `ANG_MAX`=8, `RATE_A`=9, `B_ON`=10 |

**Masking** (`datasets/masking.py`, `datasets/task_transforms.py`)

- `PowerFlowTransforms` = `RemoveInactiveBranches`, `RemoveInactiveGenerators`, `AddPFHeteroMask`, `ApplyMasking`.
- `AddPFHeteroMask` writes `data.mask_dict` with keys `bus`, `gen`, `branch`, `PQ`, `PV`, `REF`. Unknowns: `Vm, Va` at PQ; `Va, Qg` at PV; **`Vm` and `Qg` at REF**; `Pg` of generators at REF; branch `P_E, Q_E`. It also masks the limit columns and `RATE_A`.
- `ApplyMasking` overwrites masked entries with `args.data.mask_value`. `RemovePFMask` restores the limit columns and `RATE_A` from `.static`.
- Consequence for Newton: the slack voltage magnitude, a power-flow setpoint, is not in `x` after masking (the smoke test reads 0.0 there). The prototype takes it from `data["bus"].y` by default, or from the prediction.

**Units** (`datasets/normalizers.py`, `HeteroDataMVANormalizer`; `HeteroDataPerSampleMVANormalizer` uses the same formulas per scenario)

- `transform`: powers and `RATE_A` divided by the fitted `baseMVA` (95th percentile of non-zero powers, not the case base); `Va` and angle limits degrees → radians; admittances, `GS`, `BS` multiplied by `baseMVA_orig / baseMVA`; `Vm` untouched.
- `inverse_transform(data)`: powers back to MW / MVAr. **`Va` stays in radians** (the source says so explicitly; `ANG_MIN` / `ANG_MAX` do go back to degrees). **Admittances, `GS`, `BS` are multiplied by `baseMVA`**, which yields the case-file per-unit value times `baseMVA_orig`, not the original value. The source marks this as intentional: `V · conj(Y V)` with `V` in p.u. is then directly in MW.
- `inverse_output(output, batch)`: scales `PG_OUT`, `QG_OUT` (and `PD_OUT`, `QD_OUT` if present) of `output["bus"]` and `PG_OUT_GEN` of `output["gen"]` to MW. `Vm`, `Va` are left alone.

**Physics layers** (`models/utils.py`)

- `ComputeBranchFlow.forward(bus_data, edge_index, edge_attr)` → sending-end `(Pft, Qft)` per directed edge.
- `ComputeNodeInjection.forward(Pft, Qft, edge_index, num_bus)` → `scatter_add` over the source bus.
- `compute_shunt_power`: `p_shunt = -GS·Vm²`, `q_shunt = +BS·Vm²`.
- `ComputeNodeResiduals.forward(P_in, Q_in, bus_data_pred, bus_data_orig)` → `Pg − Pd + p_shunt − P_in` and the same for Q. That is **specified minus calculated**, the opposite sign of MATPOWER's `F(x)`. The refinement follows graphkit's sign and solves `J·dx = +residual`.
- Together these imply `Ybus[i,i] += Y_self`, `Ybus[i,j] += Y_mutual` over directed edges `i → j`, plus `GS + j·BS` on the diagonal.
- `PhysicsDecoderPF` (registered in `PHYSICS_DECODER_REGISTRY` as `"PowerFlow"`) derives `Pg` at REF and `Qg` at PV + REF from the bus balance. The refinement does the same for its output.

**Where evaluation happens** (`tasks/pf_task.py`, `tasks/utils.py`, `cli.py`)

- `PowerFlowTask.test_step` and `predict_step` both start with `inverse_transform(batch)`, `inverse_output(output, batch)`, `RemovePFMask()(batch)`, then `_build_bus_target`, `_clamp_known_to_ground_truth`, the three physics layers, and log `Active Power Loss`, `Reactive Power Loss`, `PBE Mean`, `PBE Max` and per-bus-type MSE.
- `compute_branch_predictions` (`tasks/utils.py`) builds the per-directed-edge table written by `predict`: `Pft`, `Qft`, their targets, `thermal_excess` (`relu(|S| − RATE_A)`), `rate_a`.
- Loading in graphkit is `sqrt(Pft² + Qft²) / RATE_A` per directed edge, both in MW/MVA. `datasets/postprocessing.py` has a current-based variant (`compute_loading`) on pandas/NumPy tables.
- Batching: PyG batches concatenate graphs; `batch.batch_dict["bus"]` maps buses to graphs and `local_index_per_graph` recovers per-graph bus indices.
- CLI (`__main__.py`): subcommands `train`, `finetune`, `evaluate`, `predict`, `benchmark`. `evaluate` already has an opt-in post-processing flag, `--compute_dc_ac_metrics`, which calls `compute_pf_ac_dc_metrics` (`tasks/pf_ac_dc_baseline.py`) on rank 0 after the test loop.

## Where the code would live

| New | Location | Sits next to |
|---|---|---|
| `build_ybus`, `newton_refine`, `NewtonRefineResult` | new `gridfm_graphkit/tasks/newton_refine.py` | `tasks/utils.py` (`compute_branch_predictions`, `residual_stats_by_type`), `tasks/pf_ac_dc_baseline.py` |
| Hook in the PF task | `tasks/pf_task.py`, inside `PowerFlowTask.test_step` and `predict_step`, after `_clamp_known_to_ground_truth` | the existing `PBE Mean` / `PBE Max` logging |
| `conformal_width`, `screen`, `ConformalScreenReport` | new `gridfm_graphkit/tasks/conformal_screening.py` | `tasks/pf_ac_dc_baseline.py` (same pattern: a post-hoc function over saved tables) |
| CLI flag | `__main__.py` / `cli.py`, `evaluate` and `predict` | `--compute_dc_ac_metrics` |

`models/utils.py` was considered for the Newton code, since the physics layers are there. It was rejected: those are differentiable `nn.Module`s used inside the model and under `torch.compile`; the refinement is a NumPy/SciPy post-processing step and should not be importable into the forward pass by accident.

## API sketch

```python
# gridfm_graphkit/tasks/newton_refine.py

def build_ybus(data: HeteroData) -> scipy.sparse.csr_matrix:
    """Bus admittance matrix of one denormalised graph, from the directed-edge
    admittances and the bus shunts. Units: case-file p.u. times baseMVA_orig."""

@dataclass
class NewtonRefineResult:
    bus_output: Tensor      # [num_bus, 4] VM_OUT, VA_OUT, PG_OUT, QG_OUT (Pg at REF, Qg at PV/REF from the balance)
    p_flow: Tensor          # [num_edges] sending-end MW per directed edge
    q_flow: Tensor          # [num_edges] MVAr
    loading: Tensor         # [num_edges] |S| / RATE_A, NaN where RATE_A <= 0
    residual_p: Tensor      # [num_bus] specified - calculated, MW
    residual_q: Tensor      # [num_bus] MVAr
    max_mismatch: float     # max |residual| over the Newton equations after the last step
    mismatch_history: list[float]   # [before, after step 1, ...]
    steps_taken: int
    ok: bool                # False on a singular Jacobian or a non-finite iterate

def newton_refine(
    data: HeteroData,           # one graph, after inverse_transform + RemovePFMask
    bus_output: Tensor,         # output["bus"] after inverse_output
    steps: int = 1,
    ref_vm: str = "label",      # slack Vm from data["bus"].y ("label") or from the prediction ("pred")
) -> NewtonRefineResult:
    """Run `steps` full Newton steps (polar, MATPOWER-style dS/dVa and dS/dVm) on the
    AC power-flow equations, started from the predicted voltages. Unknowns: Va at
    PV + PQ, Vm at PQ. Equations: P at PV + PQ, Q at PQ."""
```

```python
# gridfm_graphkit/tasks/conformal_screening.py

def conformal_width(pred_loading, true_loading, alpha, limit=1.0, sigma=None) -> tuple[float, int]:
    """Label-conditional split conformal. Over the calibration readings that are real
    overloads (true_loading > limit), score = (limit - pred_loading) / sigma; return the
    ceil((n+1)(1-alpha))-th smallest score and n. Returns inf if n is too small for alpha."""

def screen(pred_loading, true_loading, state_id, width, limit=1.0, sigma=None) -> ConformalScreenReport:
    """A reading is flagged when pred_loading >= limit - width * sigma. A state with a
    flagged reading goes to AC. Reports:
      ac_call_share      states sent to AC / all states
      wasted_share       states sent to AC without a real overload / states without one
      missed, real       real overloads in states accepted without AC, and all real overloads
      missed_share       missed / real
      missed_upper       Clopper-Pearson upper bound: Beta.ppf(conf, missed + 1, real - missed)
      line_missed_share  real overloads not flagged themselves (the event the guarantee covers)
      worst_missed       largest true loading among the missed"""
```

`sigma=None` means a global width in loading units. A per-reading `sigma` (per-line error scale, or the surrogate's own change against a base case) is accepted as an array so that the adaptive variants can be built outside graphkit.

## How it plugs in

- **`test_step`:** when enabled, loop over the graphs of the batch (slice buses with `batch_dict["bus"]` and edges by the graph of their source bus; whether `mask_dict` survives `to_data_list()` was not checked), call `newton_refine` on each, and log per dataset: `Newton mismatch before (MW)`, `Newton mismatch after (MW)` (max-reduced like `PBE Max`), the share of graphs with `ok=False`, and the per-bus-type MSE of the refined `Vm` / `Va` next to the existing ones.
- **`predict_step`:** add columns to the existing tables: `Vm_newton`, `Va_newton` in the bus table; `Pft_newton`, `Qft_newton`, `loading_newton` in the branch table from `compute_branch_predictions`; `newton_mismatch`, `newton_ok` per scenario.
- **Conformal screening:** runs after `predict`, on the branch table, the way `compute_pf_ac_dc_metrics` runs after `evaluate`. Needs a calibration / test split of the scenarios; calibration and test must not share an outage set (group key supplied by the caller, default `scenario`).
- **Config knobs** (all optional, read with `getattr` as the tasks already do for `verbose` and `get_embeddings`):

```yaml
evaluation:
  newton_refine:
    steps: 1            # 0 disables
    ref_vm: label       # label | pred
    mismatch_tol: 1.0   # MW; above this a state counts as failing the self-check
  conformal_screening:
    alpha: [0.05, 0.02, 0.01]
    limit: 1.0
    calibration_ratio: 0.5
    confidence: 0.95
```

- **CLI:** `--newton_steps N` on `evaluate` and `predict` as a shortcut for the config key; `--conformal_screening` on `evaluate`, mirroring `--compute_dc_ac_metrics`.

## Implementation notes

- The prototype copies the tensors to float64 NumPy and solves with `scipy.sparse.linalg.splu`. This is the simplest correct choice and is fine for post-processing on CPU. graphkit already imports `scipy.sparse` in `datasets/postprocessing.py`, but SciPy is not listed in the package's `Requires-Dist`; it would have to be declared.
- A dense `torch.linalg.solve` on the reduced Jacobian is a drop-in alternative for small graphs and would keep everything on the device and batchable (block-diagonal). It does not scale to 10,000 buses.
- The tensors are float32. On IEEE 14 the stored AC solution has a mismatch of 1.3e-4 MW when evaluated through these tensors, and graphkit's float32 physics layers report about 2.6e-4 MW on a converged float64 iterate. Tolerances below about 1e-3 MW are meaningless on this path.

## Tests that would accompany the PR

Fixtures: a 14-bus scenario built with `build_hetero_data` from small in-repo tables, passed through `HeteroDataMVANormalizer.transform`, `PowerFlowTransforms`, `inverse_transform`, `RemovePFMask`.

Newton refinement

1. `build_ybus` agrees with the layers: for random `V`, `Re/Im(V · conj(Ybus V))` equals `P_in − p_shunt`, `Q_in − q_shunt` from `ComputeBranchFlow` + `ComputeNodeInjection` + `compute_shunt_power`.
2. Fixed point: with the label voltages as the "prediction", `steps=0` gives a mismatch below 1e-3 MW and `steps=1` does not move the voltages.
3. Convergence: from a perturbed start (labels + noise), the mismatch history decreases and the voltages return to the labels within a few steps.
4. Sign: `residual_p`, `residual_q` at `steps=0` equal `ComputeNodeResiduals` on the same bus tensor.
5. Setpoints are untouched: `Vm` at PV and REF, `Va` at REF.
6. Edge order: `loading` has one entry per directed edge in `edge_index` order (all from-ends, then all to-ends) and matches `sqrt(pf² + qf²) / rate_a`, `sqrt(pt² + qt²) / rate_a` of the stored AC solution.
7. Units: the result is identical for two normalizers fitted with different `baseMVA`.
8. An out-of-service branch (`br_status = 0`) changes the result; a singular case (islanded bus) should return `ok=False` instead of raising (behaviour of the prototype here is untested).
9. A batched input raises a clear error (until batching is supported).

Conformal screening

1. Quantile index: with `n` calibration overloads, the width is the `ceil((n+1)(1−alpha))`-th smallest score; `inf` when that exceeds `n`.
2. Coverage on synthetic exchangeable data: over many random splits the mean `line_missed_share` is at most `alpha` within sampling error.
3. `missed_upper` equals `scipy.stats.beta.ppf(conf, k+1, n−k)` and is 1.0 when `k = n`.
4. `ac_call_share`, `wasted_share` on a hand-built table of five states.
5. A perfect surrogate gives width ≤ 0, no wasted calls, no misses.

## What was checked for this draft

A single run on one stored IEEE 14 scenario (`work/pf_case14_ieee_tiny`, its tiny one-epoch checkpoint, in-process, no solver):

- stored AC solution, `steps=0`: mismatch 1.3e-4 MW (Ybus, shunts and units are right);
- model prediction: mismatch 0.357 MW before, 7e-5 MW after one step; voltages then agree with the labels to 1.2e-7 p.u. and 9e-8 rad;
- `steps=0` residuals agree with `ComputeNodeResiduals` to 1e-4 MW (sign convention);
- flows agree with `ComputeBranchFlow` to 2e-4 MW; loadings after five steps agree with the stored AC flows to 3e-7.

Not checked: any grid above 14 buses with this prototype (the entry's numbers come from the NumPy reference in `analysis_warmstart.py` and `work/goc500/02_newton.py`, which this ports), `HeteroDataPerSampleMVANormalizer`, the GRIT model's 6-column output, GPU tensors, and the conformal utility, which exists only as the sketch above and as `analysis_conformal.py` in the entry.

## Open questions and limitations

- **No reactive limits.** PV buses stay PV; `MIN_QG_H` / `MAX_QG_H` are ignored. The entry's results share this limit. PV → PQ switching would need an outer loop.
- **No step control.** Plain Newton. From the model's start it converged on every scenario of the 2,000- and 10,000-bus sets; from a flat start it failed on 128 of 2,494 and on all 993 [`submission/POST.md`]. Far out of distribution (loads raised to 1.1–1.8 without re-dispatch) one step no longer rescues the prediction [`submission/POST.md`]. `ok` and the mismatch are the only safeguards.
- **Slack `Vm`.** `AddPFHeteroMask` treats `Vm` at REF as an unknown. Is that intended? For the refinement it has to be fixed; taking it from the label is correct for power flow but is, strictly, reading a target.
- **Dense or sparse.** SciPy sparse LU on CPU copies, or a torch solve that stays on the device? The first adds a declared dependency, the second does not scale.
- **Batched graphs.** The prototype takes one graph. A batch is block-diagonal, so one sparse factorisation over the whole batch works, but a single bad graph should not spoil the others; per-graph solves are safer and trivially parallel.
- **Cost.** One step is one Jacobian build and one sparse LU per graph, on CPU, inside the test loop. It should stay opt-in.
- **Not a model metric.** The step uses the true network model, so refined errors measure the model plus physics. They should be logged under their own names and never replace the raw ones.
- **Loading definition.** `|S| / RATE_A` per directed edge, as in `compute_branch_predictions`, or the current-based `compute_loading` of `datasets/postprocessing.py`? The utility should use one and say which.
- **Overload threshold.** OPF-dispatched data parks lines at exactly 100%, so `limit` moves the counts; the entry used 100.1% [`submission/POST.md`].
- **Conformal scope.** The guarantee is marginal over exchangeable outage sets, not per state, and needs enough labelled overloads: `alpha` = 1% needs about 100 [`submission/POST.md`]. Anchoring on a base-case AC solution gave the tightest bands in the entry but needs a notion of "base case" that graphkit's datasets do not have; it is left to the caller through `sigma` and a pre-anchored `pred_loading`.

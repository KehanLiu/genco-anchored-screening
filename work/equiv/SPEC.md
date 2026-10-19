# Equivalent-circuit pilot: does anchoring cancel GENCO's representation dependence?

Pre-registered experiment definition, written 10 October 2026 BEFORE any fresh operating point was scored.
Section 10 ("Development probe") holds the only numbers seen so far; they come from the inspected development pool
`work/goc500/data` and were used to choose the thresholds X, Y, X0 in section 7. Nothing in the fresh pools has been run.
Brief: an internal audit note on creative directions ("Equivalent circuit representations", "Follow-up and possible
repair", "First pilot") and its follow-up. Code: `work/equiv/transforms.py`, `work/equiv/run_equiv.py`,
tests `work/equiv/test_transforms.py`. Nothing outside `work/equiv/` was modified.

## 1. Question

The submission's start for a contingency is `v_start = v_AC(pre-outage) + v_GENCO(outage) - v_GENCO(intact)` followed by one
Newton step. GENCO's raw prediction changes substantially when an electrically identical circuit is encoded differently
(audit probes: parallel split of every branch moves GOC 500 angles by up to 29 degrees). Question: **does the anchored
predicted change, and the one-Newton-step result built on it, depend on the representation much less than the raw prediction,
and do the screening decisions survive the rewrite?** Secondary: how much of any cancellation is due to the anchoring (the
difference of two predictions on the same representation) and how much to the Newton step itself.

## 2. Mechanism and what is compared

For every stored contingency state s of a pool, with the pre-outage grid rebuilt and solved exactly as in
`analysis_hybrid_start.worker` (removed branches back in service, admittances recomputed, SciPy Newton from the stored
contingency solution), GENCO is run on the original tables (`o`) and on a rewritten, physically equivalent representation (`t`)
of BOTH the intact and the outaged graphs. Per representation r:

| column | definition |
|---|---|
| `genco_r` | raw GENCO loading: flows through the original admittances at GENCO's predicted voltages (mapped back to physical buses), max of the two ends over the rating |
| `genco_eng_r` | the engine's own per-copy flows summed back to the physical branch over the original rating (cross-check of the aggregation, must equal `genco_r` to float32) |
| `dgenco_r` | anchored loading `ac_prev + genco_r - genco_prev_r` (the predicted CHANGE of loading on the AC base) |
| `g1_r` | one Newton step from GENCO's raw voltages (unanchored; separates the step's own cancellation) |
| `hyb0_r` / `hyb1_r` | the anchored start `V_prev + (V_GENCO_outage - V_GENCO_intact)_r` and one Newton step from it (the submission's method) |
| `prev1` | one Newton step from the pre-outage AC solution: classical physics control, representation-free |

Newton always runs on the ORIGINAL problem (`work/goc2000/02_newton.py make_problem`) and the stored AC solution is the truth
for both representations. This reuse is allowed only because `verify_equivalence` proves, per state and for both the intact
and the outaged graph, that the rewrite has the same bus admittance matrix (mapped back to physical labels), specified P/Q
injections, bus types, PV/REF voltage setpoints, reference bus and angle, bus shunts, aggregated generator P/Q per bus,
summed ratings per physical branch and the same set of outaged physical branches; the maxima are stored per state and any
value above 1e-9 aborts the state (an abort anywhere aborts the run).

## 3. Transformations (frozen)

All are applied identically to the intact and to every outaged graph of a base state; random choices are drawn from
`numpy.random.default_rng([spec_seed=2026, grid, load_scenario_idx])` so that a base state and its contingencies share one rewrite.

| name | rewrite |
|---|---|
| `none` | identity (pipeline control: every `t` column must equal its `o` column to float32) |
| `permute` | random relabelling of buses plus a random reordering of branch and generator rows (ordinary equivariance control) |
| `split_random_1` | one rated, not-always-out physical branch chosen at random per load scenario becomes 2 parallel copies: stored Y columns / 2 (exact), r, x x2, b / 2, rate / 2, flows / 2; tap, shift, angle limits, status kept |
| `split_all` | every physical branch becomes 2 parallel copies as above (the extreme encoding of the audit probes) |
| `split_gen` | every in-service, non-slack generator becomes two units at the same bus with half p_mw, q_mvar, P and Q limits, cp0 (cp2 doubled); bus table, types and setpoints untouched |
| `+refvm` modifier | known-reference-voltage control: GENCO's predicted REF-bus Vm and Va are overwritten by the stored setpoints before the raw loading, the anchored start and the Newton step, for BOTH representations (the stock PF mask predicts the REF Vm; the REF Va is an unmasked input and comes back unchanged). NOTE (found in the probe): every Newton start of this pipeline (`Problem.start`) already pins the REF Vm/Va and the PV Vm to the setpoints, so `+refvm` changes only the raw `genco`/`dgenco` columns; `g1`, `hyb0`, `hyb1` are identical with and without it. A physically consistent known-REF-Vm adapter would need re-inference with a changed mask (not available for the released checkpoints) and is out of scope |

Outage mapping: a copy inherits the status of its physical branch, so an outaged split branch loses every copy; the driver
checks per state that rewriting the outaged tables equals taking every copy out of the rewritten intact grid (`outage_map_ok`).
When the randomly chosen branch of `split_random_1` happens to be an outaged branch the state is kept and counted (`split_outaged`).

## 4. Controls

`none` (identity), `permute` (equivariance), `+refvm` (known-input correction, separable from the graph effect), `prev1`
(classical start, no GENCO), `g1` (GENCO + one step without anchoring). Trust in the transformed physics rests on
`verify_equivalence`, `outage_map_ok` and the `genco_eng` cross-check, all reported per run.

## 5. Checkpoints and grids

| grid | checkpoint | role |
|---|---|---|
| GOC 500 | Tiny seed 0 (`work/goc500/pf_case500_goc_tiny`) | primary |
| GOC 500 | Tiny seed 1, Tiny seed 42 (`pf_case500_goc_tiny_seed1`, `_seed42`) | replication, `split_random_1` and `split_all` only |
| GOC 2000 | Tiny seed 0 (`work/goc2000/pf_case2000_goc_tiny`) | primary at the second grid |
| GOC 2000 | Small seed 0 (`pf_case2000_goc_small`) | replication, `split_random_1` and `split_all` only |

Released normalisers and the stock PF transforms are retained (`engine.Genco`, the same inference path as every other study).

## 6. Fresh groups and sample sizes

| pool | load scenario ids | status |
|---|---|---|
| GOC 500 fresh, `work/fresh/data500/case500_goc/raw` | 6000-6499 (datakit seed 2026; `work/fresh/datagen500.yaml`, POLICY_LOG A4) | not yet generated at the time of writing (`work/fresh/chain10000.sh` runs it after the GOC 10000 pool); never inspected; ~2,500 contingency states expected |
| GOC 2000 fresh, `work/fresh/data2000/case2000_goc/raw` | 1500-1999 (500 load scenarios, 2,496 states) | already generated; used by the fresh confirmation run `work/fresh/` (fresh N-1 and hybrid confirmation of the frozen policy), a DIFFERENT question. Its hybrid loadings have therefore been inspected once; it is fresh for the representation question, not untouched, and is reported as such |

Sample: all states of both pools for the primary checkpoints and every transform of section 3 plus `split_all+refvm` and
`split_random_1+refvm`; replication checkpoints on all states for `split_random_1` and `split_all`. No subsampling unless the
GOC 2000 runs exceed the budget, in which case the first 250 load ids in ascending order are used and this is recorded here
before inspection. Nothing is selected after seeing fresh results.

## 7. Endpoints and the pre-registered criterion

Per transform, per representation, computed by `run_equiv.py report`:

Primary endpoints
- P1 loading error of the one-step result `hyb1` (median, p99, max in points of rating over rated lines) and of `prev1`, `genco`
- P2 missed overloads: overloaded line readings (AC loading > 1.001) in states that `hyb1` does not flag, and critical states not flagged
- P3 false flags: secure states flagged by `hyb1`
- P4 Newton behaviour: convergence count and mean iterations to convergence from the anchored start, mismatch after one step

Comparison statistics (transformed vs original)
- D_s(stage) = max over rated lines of |stage_t - stage_o| per state (points); reported as median / p90 / max over states
- cancellation ratio rho(stage) = median_s D_s(stage) / median_s D_s(genco); mean-based version reported alongside
- decision agreement A(stage) = share of states where "any line > 1.001" agrees between t and o

Secondary diagnostics: raw voltage drift (max |Vm_t - Vm_o|, max |Va_t - Va_o| on the intact and the outaged graph), drift of the
predicted change, rho(dgenco), rho(hyb0), rho(g1), REF-bus raw Vm error, start mismatch, `genco_eng` vs `genco` agreement.

**"Anchoring cancels representation dependence" for a transform T** if, on a fresh pool with the primary checkpoint:
1. rho(hyb1) <= X for BOTH the median-based and the mean-based ratio (the mean catches blown-up Newton steps that a median hides), and
2. A(hyb1) >= Y, and
3. missed overloaded readings of `hyb1_t` <= those of `hyb1_o` + ceil(1% of the overloaded readings of the pool),
with **X = 0.05, Y = 0.98** (justified in section 10; `run_equiv.py report` defaults `--x 0.05 --y 0.98 --miss-margin 0.01 --x0 0.3`).
Attribution: the cancellation is credited to the anchoring (not only to the Newton step) if additionally rho(hyb0) <= X0 = 0.3,
i.e. the anchored START already removes at least 70% of the raw representation difference; if rho(hyb0) > X0 while rho(hyb1)
<= X, the cancellation is credited to the Newton step and the headline must say so.
The overall verdict "cancels" requires criterion 1-3 for `split_random_1`, `split_all` and `split_gen` on BOTH fresh pools with
the primary checkpoints. `permute` must pass with rho ~ 0 and raw drift below 1e-3 p.u. / 0.05 degrees, else the pipeline is
broken and nothing is reported. `+refvm` results are reported next to the stock-mask results; they do not enter the verdict.

## 8. Continue-or-stop rule

- CANCELS (all three transforms, both pools): stop the repair direction. Report the result as a robustness property of the
  anchored-change method (raw GENCO is representation-dependent, the anchored one-step result is not), with the attribution of
  section 7, in the existing submission. No pivot.
- DOES NOT CANCEL for `split_random_1` or `split_gen` (the encodings that occur in practice: parallel circuits and multi-unit
  plants) with a decision disagreement that changes missed overloads: continue within the 48-hour window with ONE repair,
  canonical merging of parallel copies and aggregation of co-located fixed-dispatch generators before inference, evaluated on
  the same fresh pools AND on the untouched original encodings (a repair that only restores the original input must not degrade
  them). Promote to the main contribution only if the repair improves fresh missed overloads or false flags without degrading
  ordinary cases, after a second-pass audit.
- DOES NOT CANCEL only for `split_all`, or the effect disappears under `+refvm`: document as a limited finding (extreme
  engineered encoding / known-input issue), keep the current main story, no repair.
- ABORT: any equivalence maximum above 1e-9, `outage_map_ok` false anywhere, or `none` not reproducing `o` = `t`; fix before scoring.

## 9. Commands (fresh runs; to be executed only after the GOC 500 pool exists)

```sh
cd <repo>
# GOC 500 fresh pool (ids 6000-6499), primary checkpoint, every transform
for t in none permute split_random_1 split_all split_gen split_all+refvm split_random_1+refvm; do
  work/texas/tmo 7200 .venv/bin/python -u work/equiv/run_equiv.py run --grid 500 --data work/fresh/data500 --variant tiny --seed 0 \
      --transform $t --procs 2 --tag _fresh > work/equiv/logs/fresh500_$t.out 2>&1
done
for s in 1 42; do for t in split_random_1 split_all; do
  work/texas/tmo 7200 .venv/bin/python -u work/equiv/run_equiv.py run --grid 500 --data work/fresh/data500 --variant tiny --seed $s \
      --transform $t --procs 2 --tag _fresh > work/equiv/logs/fresh500_seed${s}_$t.out 2>&1
done; done
# GOC 2000 fresh pool (ids 1500-1999, already generated, used by the fresh confirmation run for the policy question)
for t in none permute split_random_1 split_all split_gen split_all+refvm split_random_1+refvm; do
  work/texas/tmo 14400 .venv/bin/python -u work/equiv/run_equiv.py run --grid 2000 --data work/fresh/data2000 --variant tiny --seed 0 \
      --load-ids work/fresh/fresh_ids.json --transform $t --procs 2 --tag _fresh > work/equiv/logs/fresh2000_$t.out 2>&1
done
for t in split_random_1 split_all; do
  work/texas/tmo 14400 .venv/bin/python -u work/equiv/run_equiv.py run --grid 2000 --data work/fresh/data2000 --variant small --seed 0 \
      --load-ids work/fresh/fresh_ids.json --transform $t --procs 2 --tag _fresh > work/equiv/logs/fresh2000_small_$t.out 2>&1
done
# reports (criterion thresholds of section 7)
.venv/bin/python work/equiv/run_equiv.py report --grid 500  --variant tiny --seed 0 --tag _fresh   # defaults: --x 0.05 --y 0.98 --miss-margin 0.01 --x0 0.3
.venv/bin/python work/equiv/run_equiv.py report --grid 2000 --variant tiny --seed 0 --tag _fresh
.venv/bin/python work/equiv/run_equiv.py report --grid 500  --variant tiny --seed 1 --tag _fresh    # replication checkpoints likewise (seed 42; grid 2000 --variant small)
```
Outputs: `work/equiv/results/goc<N>_<variant>_seed<S>_<transform>_fresh_{lines,states}.parquet`, `summary_*.csv`, `criterion_*.csv`.
`--load-ids work/fresh/fresh_ids.json` restricts GOC 2000 to the registered ids (the pool contains only those anyway).

## 10. Development probe (DEVELOPMENT evidence only; sets X, Y, X0)

Run on 10 October 2026: `work/goc500/data` (the inspected GOC 500 development pool, load scenarios 0-19 = 100 contingency
states, 20 base states, 12 states with a real overload, 24 overloaded line readings), Tiny seed 0, `--limit 20 --tag _dev`,
~20 s per transform. Full output in `work/equiv/logs/dev_report.txt`, tables in `results/summary_goc500_tiny_dev.csv` and
`results/criterion_goc500_tiny_dev.csv`. Physics checks: equivalence maxima ybus <= 6.8e-13, injections / setpoints / types /
ratings / outaged sets 0, gen P per bus <= 1.1e-13; outage mapping ok 100/100 everywhere; engine-flow aggregation vs physics
loading <= 5.6e-5 (float32); `none` reproduces `o` = `t` exactly (max |t-o| 0). Values in points of rating (1 point = 1% of rating).

| transform | raw drift Vm / Va (median over states of the max) | state max\|t-o\| median: genco / hyb0 / hyb1 | rho hyb0 / hyb1 median [mean] / g1 | state decisions agree genco / hyb1 | hyb1 err p99 o / t | missed readings hyb1 o / t (prev1: 1) | Newton from hybrid start t: conv, it | verdict |
|---|---|---|---|---|---|---|---|---|
| none | 0 / 0 | 0 / 0 / 0 | - | 100% / 100% | 0.044 / 0.044 | 0 / 0 | 100/100, 2.47 | identity OK |
| permute | 2e-6 p.u. / 0.002 deg | 0.023 / 0.030 / 0.0000 | 1.34 / 0.001 [0.002] / 0.002 | 100% / 100% | 0.044 / 0.044 | 0 / 0 | 100/100, 2.47 | cancels (raw drift itself negligible) |
| split_random_1 | 1e-4 p.u. / 0.08 deg (p90 0.78 deg) | 0.57 / 0.075 / 0.0001 | 0.13 / 0.000 [0.001] / 0.004 | 99% / 100% | 0.044 / 0.045 | 0 / 0 | 100/100, 2.47 | cancels; start removes 87% |
| split_gen | 0.005 p.u. / 18.3 deg | 42.4 / 8.3 / 0.034 | 0.20 / 0.001 [0.004] / 0.038 | 65% / 99% | 0.044 / 0.124 | 0 / 1 | 100/100, 2.73 | cancels (1 extra miss = the allowed margin; borderline) |
| split_all | 0.097 p.u. / 21.4 deg | 502 / 132 / 7.1 | 0.26 / 0.014 [1.25] / 0.36 | 19% / 82% | 0.044 / 10.9 (max 78,990: a blown-up step) | 0 / 1 | 97/100, 4.11 | does not cancel |
| split_all+refvm | as split_all | as split_all | as split_all | 16% / 82% | as split_all | 0 / 1 | as split_all | does not cancel; refvm changes only the raw columns |

Reading: the raw prediction is strongly representation dependent even for a physically ordinary encoding (`split_gen`: 18
degrees of angle drift, 42 points of loading difference). The anchored start removes 74-87% of the raw difference (rho hyb0
0.13-0.26) and one Newton step removes essentially all of what remains (rho hyb1 <= 0.004 by mean) as long as the start is
within the step's basin; for `split_all` the start is 21 degrees off, 3 of 100 hybrid starts do not converge at all, and the
one-step result and the screening decisions change. The known-reference control does not touch this (see section 3), and
overwriting the REF Vm alone makes the raw mismatch worse (median 453 vs 17 MW: GENCO's whole magnitude profile carries its
REF Vm error of 0.023 p.u., so a local fix is inconsistent).

Choice of thresholds (made on these numbers, before any fresh scoring): X = 0.05 is one order of magnitude above the passing
mean-based ratios (0.001-0.004) and 25x below the failing one (1.25); Y = 0.98 sits far above the raw agreement of the
passing transforms (65-99%) and above their `hyb1` agreement (99-100%) minus one tie-level state per 100 (a line within 0.1
point of the limit can flip); the miss margin of 1% of overloaded readings (at least 1) absorbs the single-reading noise of
the small development count (1 reading of 24 is 4%) while allowing at most ~6 of the ~600 overloaded readings expected on a
fresh pool; X0 = 0.3 is just above the observed start-level ratios (0.13-0.26). On this probe the expected fresh outcome is
the third branch of section 8 (cancels for the natural encodings, fails for the extreme one), which would keep the current main
story and add a documented limitation; the fresh pools decide.

## 11. Open issues and provenance

- `verify_equivalence` reports `gen_n_on` (number of in-service units per bus) but does not assert it: generator splitting changes
  it by construction. Everything else is asserted at 1e-9.
- Newton runs on the original problem; the transformed problem itself is not iterated (its equations are proven identical). The
  unit tests check that the copies' flows at identical voltages sum to the original loading (1e-10) and that an outage removes
  every copy.
- `+refvm` is a raw-only control in this pipeline (section 3). The REF Va is unmasked and GENCO returns it unchanged (raw REF Va error 0).
- `split_random_1` draws one branch per load scenario; a state whose drawn branch is itself outaged is kept and counted (`split_outaged`); none occurred in the probe.
- The GOC 2000 fresh pool was inspected once for the frozen-policy question of the fresh confirmation run; the GOC 500 fresh pool (ids 6000-6499) does not exist yet.
- Code: `work/equiv/transforms.py`, `work/equiv/run_equiv.py` (imports `engine.Genco`, `analysis_prev_control.{paths,admittances,newton_module}`;
  the hybrid-start logic of `analysis_hybrid_start.worker` is copied), tests `work/equiv/test_transforms.py` (8 tests). Checkpoints as listed in
  section 5, released normalisers; datakit tables of the pools named in section 6; `work/fresh/datagen500.yaml` sha256 in `work/fresh/POLICY_LOG.md`.

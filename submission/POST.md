# Let GENCO predict the change, not the state: contingency screening with the anchored change start and one Newton step

**In one paragraph:** GENCO's predicted voltage changes improve one-step Newton screening on several benchmark grids, but the measured implementation has no demonstrated end-to-end cost advantage. On a 4,800-outage sample from the fresh GOC 2000 pool, the anchored workflow took 22.3 ms per outage with GPU batch-16 inference, versus 8.4 ms for classical screening and 13.7 ms for complete AC. All three missed zero of the 69 critical cases in that timing sample; numerical nonconvergences were treated as flagged cases. These are laptop throughput measurements, not a demonstrated deployment advantage.

| Workflow (same 4,800 outages, 24 fresh GOC 2000 bases) | Measured ms per outage, first pass | Critical cases missed in this sample |
|---|---:|---:|
| Complete AC, Newton to convergence from the solved base case | 13.7 | 0 of 69 |
| Classical screen, one Newton step from the solved base case | 8.4 | 0 of 69 |
| Anchored change start, one Newton step, GENCO on the laptop GPU in batches of 16 (amortized throughput) | 22.3 | 0 of 69 |

### Mechanism: better initialization for one Newton step

In a separate loading-only diagnostic on scenarios with up to five branch outages, the anchored start (the solved pre-outage AC base case plus GENCO's predicted voltage *change*, then one Newton step) missed 2, 5 and 0 overloaded line readings on IEEE 118, GOC 500 and GOC 2000 (of 6,986, 2,621 and 369), versus 74, 41 and 2 from the classical start after one Newton step, at 0.0% of secure scenarios flagged. Both use a converged, matching AC base state. This demonstrates better one-step initialization on these datasets; it is not an equal-cost comparison or a safety guarantee, and it stops helping at 10,000 buses.

![Overload screening after one Newton step](figures/starts.png)

*Loading-only diagnostic with cutoff 1.001 and a solved AC base state. The comparison fixes the number of Newton steps; GENCO inference cost and complete-AC performance are not shown. The measured end-to-end workflow is slower than complete AC on the separate fresh GOC 2000 timing sample.*

## In one table

Scored against the full AC solution, released checkpoints. "Classic" is one Newton step from the pre-outage solution; "anchored" is one step from the anchored change start.

| Setting | Classic | Anchored | Verdict |
|---|---|---|---|
| GOC 500: overloaded lines missed after one step | 41 of 2,621 | **5 of 2,621** | Anchored finds more |
| GOC 2000 N-1: sent to AC / critical outages missed | 29.6% / 5 of 1,451 | **2.7%** / 14 of 1,451 | Far fewer AC calls, not at equal safety |
| GOC 2000 N-1: estimated seconds per 1,000 outages | **6.5** | 19.4 | No wall-clock gain |
| GOC 10000 (Small): lines missed / secure scenarios flagged | **2 / 0.3%** | 2 / 2.1% | Ties on misses, 7 times the false flags: stops helping |
| Fresh pool, 76,512 outages, frozen policies: sent to AC / critical missed; measured ms per outage on a 4,800-outage sample (complete AC: 13.7) | 27.94% / 2 of 1,510; 8.4 | **2.17%** / 1 of 1,510; 22.3 | Confirms the referral cut; slower than complete AC |

## The idea

This entry starts from the organisers' example ([discussion #1](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/1)): GENCO Tiny screens GOC 500 under up to five branch outages, against a DC baseline. We reproduce it (93.9% recall at 32.7% of scenarios flagged; the example reports 94.3% and 32.7%) and add the pre-outage baseline, one Newton step as the check, four grid sizes, a full N-1 study with costs, reactive limits, fine-tuning and every released checkpoint at 10,000 buses. The organisers described this workflow before we built it: a "safeguarded hybrid workflow" with "re-evaluation with a classical AC solver" (GENCO paper, [section 6](https://arxiv.org/abs/2608.09921)), conformal abstention (example post), and "classical solvers provide authoritative solutions for those few cases" ([IBM Research blog](https://research.ibm.com/blog/gridfm-neural-solver-power-grid)); full quotes in the appendix. We built and measured it: we keep the authoritative AC step, seed it, and count what it costs. We do not test GENCO as a fast surrogate; we use it to predict the *change* an outage causes.

Contingency screening asks one question for thousands of grid states: *is any line overloaded?*

- **GENCO alone is not safe enough.** On the example's setup it lets through overloads up to 139.6%.
- **One Newton step from GENCO's voltages fixes that.** 9 of 2,621 overloaded lines missed on GOC 500, 12.5% of scenarios sent to the full solver, exactly the share that really has an overload. From a flat start or DC angles the same step is useless (97% and 88% flagged).
- **But the right baseline is the pre-outage solution.** For contingency analysis from a solved AC base case, the appropriate baseline is one Newton step from that same base case. That is classic single-iteration screening; it ties GENCO's start at 2,000 buses and beats it at 10,000.
- **GENCO's errors are systematic.** On Texas 2000, 97% of the errors above 3 points occur at one load level, on the same lines in every state.
- **So subtract them.** Start = pre-outage AC voltages + [GENCO(after outage) − GENCO(before outage)]. The bias cancels; what survives is GENCO's estimate of the change, plus any error in the base state itself. Two inferences (the pre-outage one is shared by all outages of a base case), one Newton step, no training.

![GENCO's error before and after the outage](figures/bias_cancels.png)

## Main result

Scenarios remove up to five branches at random (gridfm-datakit). A scenario goes to the full solver if any line loading after the step exceeds its limit.

| Grid (overloaded lines) | Pre-outage solution: missed / secure scenarios flagged | GENCO: missed / flagged | Anchored change start: missed / flagged |
|---|---|---|---|
| IEEE 118 (6,986) | 74 / 1.9% | 4 / 0.0% | **2 / 0.0%** |
| GOC 500 (2,621) | 41 / 0.2% | 9 / 0.1% | **5 / 0.0%** |
| GOC 2000 (369) | 2 / 0.0% | 1 / 0.6% | **0 / 0.0%** |

![Where you start decides what one step finds](figures/start_step.png)

- **The anchored change start begins much closer to the solution.** Median power mismatch is 6 MW on GOC 500 and GOC 2000, against 72–118 MW for the pre-outage solution and 246–435 MW for GENCO alone. Newton then converges in 2.4–2.5 iterations instead of 3.0–3.2.
- **The mechanism is measurable.** GENCO's predicted change removes 85% of the median error of assuming no change on GOC 500 and 74% on GOC 2000 (appendix).
- **The Newton step is essential.** Without it the anchored change alone misses 14–44% of overloaded lines. The step finds the overloads; GENCO improves where it starts.

## What it costs and what it is worth

GOC 2000, 24 base cases, every non-islanding single-branch outage: 76,512 outages. Mismatch thresholds are set on 6 base cases; the other 18 are scored (57,384 outages, 1,451 critical). Seconds are estimates from unit costs measured on this laptop: Newton iterations at 3.9 ms, GENCO inference batched on the GPU at 12.6 ms per state plus 2.5 ms to build its input graph. The shared base-case solve and the pre-outage GENCO inference are computed once per base case and excluded from the per-outage figures for every workflow alike.

![A complete N-1 analysis](figures/fulln1.png)

| Workflow | Sent to full AC | Critical outages missed | Newton iterations vs full AC | Seconds per 1,000 outages |
|---|---|---|---|---|
| Full AC for every outage | 100% | 0 | 100% | 11.5 |
| One step from the pre-outage solution (classic) | 29.6% | 5 of 1,451 | 57% | **6.5** |
| GENCO alone | 6.6% | 780 (54%) | 7% | 15.9 |
| One step from GENCO | 45.4% | 1 | 76% | 23.8 |
| One step from the anchored change start | **2.7%** | 14 (13 of them ≤ 100.14%) | **37%** | 19.4 |
| DC sensitivities on the AC base case | 2.1% | 303 (21%) | 3% | 2.3 |

- **Fewest Newton iterations** among workflows that miss under 1.5% of critical outages (it misses 14, the classic screen 5). Solver-call arithmetic is in the appendix.
- **No wall-clock gain.** One GENCO inference (12.6 ms batched on the GPU) costs as much as three Newton iterations (3.9 ms each), and building its input graph costs 2.5 ms more. The classic screen wins on time. Against the classic one-step screen the anchored start saves about 2.2 ms of Newton work per outage, less than the 2.5 ms the current graph builder costs, so with this builder even free inference would leave it slightly slower than the classic screen. Against full AC, the forward pass would need to cost about 4.8 ms or less, with the current 2.5 ms graph build added separately (7.2 ms together). These are unit-cost estimates, not measured end-to-end times, and the two screens do not miss the same outages (14 against 5 of 1,451), so the iteration saving is not a comparison at equal safety. A graph builder that reuses the static tensors across outages was not measured.

## Where it could win, and what that would take

Nothing in this section is measured on the target hardware. It is a unit-cost estimate built from the fresh GOC 2000 timing sample (4,800 outages, 24 bases) and the quiet-machine unit costs of the same run, written down so that the conditions are explicit. The two conditions a useful screen must meet depend on different things.

- **Accuracy, holding the hardware fixed, depends on the model.** On IEEE 118, GOC 500 and GOC 2000 the anchored start already misses fewer overloads than the classic start at the same one-step budget (2, 5, 0 against 74, 41, 2 loading-only; 1 against 2 critical outages on the fresh N-1 pool). At 10,000 buses it does not: on lines whose loading moves by more than one point, the released checkpoints remove only 28 to 39% of the median absolute error of the "no change" assumption (74 to 85% on the smaller grids), and neither a larger checkpoint nor laptop-scale fine-tuning beats the classic start. Whether a model that predicted the change as well at 10,000 buses as at 2,000 would close this gap is a hypothesis we could not test; fine-tuning on states rather than on changes did not.
- **Cost, holding the model fixed, depends on the hardware and the graph builder.** Per outage the anchored workflow needs 1.08 Newton iterations on average against 2.92 for complete AC and 1.61 for the classic screen, so it saves 1.84 iterations (7.0 ms at 3.81 ms per iteration) against complete AC and 0.53 (2.0 ms) against the classic screen. What it adds per outage is one forward pass (12.74 ms on the laptop GPU at batch 16), one graph build (2.40 ms of CPU work) and its share of the pre-outage inference (0.18 ms on this sample, one inference per base amortized over 200 timed outages). Against complete AC the forward pass would therefore have to cost about 4.4 ms or less with the current builder for the estimate to break even; against the classic screen the builder alone already exceeds the saving, so a builder that reuses the base-case graph across outages would be a precondition. A larger model raises the forward cost (Small costs about 2.2 times Tiny), so this budget is for Tiny. Excluded from all per-outage figures, for every workflow alike: model loading, the base-case AC solve, and batch assembly outside the timed loop.
- **Scale might favour the method, if accuracy held.** A Newton iteration is a sparse factorisation whose cost typically grows faster than linearly with the number of buses, while a graph network's forward cost grows roughly linearly. If that held for these solvers and models, the cost ratio would improve with grid size; we did not measure it, and grid size is where the accuracy currently fails.
- **The fair baseline is a parallel one.** Complete AC and the classic screen parallelise across CPU cores; any throughput comparison would have to give them the same cores it gives the model its GPU. On this laptop every workflow was timed single-process.

Put together: on GOC 2000 the accuracy condition is met today; whether the cost condition can be met is a question about server hardware, batched inference and a reusable graph builder that we could not measure. At 10,000 buses both conditions are open. This is a projection with its unmeasured steps named, not a result.

## Fresh operating points, frozen policies

A fresh operating-point pool confirms the reduction in full-AC referrals, without a speed gain. Across the six registered bases plus an 18-base extension (76,512 outages; 1,510 critical), the anchored screen missed one critical outage versus two for the classical screen, while referring 2.17% versus 27.94% to AC. Both missed zero on the registered six. On the same 4,800-outage timing sample, both missed zero of 69 critical cases; measured throughput was about 22.3 ms per outage with GENCO on the laptop GPU in batches of 16, versus 8.4 ms for the classical screen and 13.7 ms for full AC. This is fresh operating-point confirmation on the same network and generation recipe; the lower maximum load and different stress mix limit comparison with the old miss rates. The policies (thresholds, cushions, a shrunk start and a mismatch gate) were frozen before the pool was generated; every variant is reported in the repository, none was selected on these results.

The same pool, scored state by state for the warm starts themselves (one Newton step, cutoff 1.001; missed critical states / secure states sent to AC). Two readings of the same files: the loading-only diagnostic (any line above the cutoff) and the complete frozen policy, which also refers a state when the post-step mismatch exceeds that start's threshold (0.5 MW for the pre-outage start, 5 MW for the anchored start, both calibrated on the old GOC 2000 N-1 study and transferred unchanged). Under the full policy both starts miss zero on GOC 2000 Tiny and GOC 10000 Tiny seed 0; the other anchored checkpoints on GOC 10000 miss 1, 1 and 4. The referral counts are not an equally tuned comparison: each start keeps its own residual threshold from the old N-1 calibration, transferred without matching safety on these multi-outage states, and a residual tolerance is not proportional to a loading tolerance, so the referral gap is not to be credited to GENCO alone.

| Fresh pool (states, critical) | Start | Loading-only: missed / secure referred | Full frozen policy: missed / secure referred |
|---|---|---|---|
| GOC 2000 Tiny (2,496, 140) | pre-outage | 3 / 0.0% | 0 / 54.1% |
| GOC 2000 Tiny | anchored | **0 / 0.0%** | 0 / 0.8% |
| GOC 10000 Tiny s0 (999, 231) | pre-outage | 0 / 1.0% | 0 / 60.6% |
| GOC 10000 Tiny s0 | anchored | 0 / 2.5% | 0 / 8.1% |
| GOC 10000 Tiny s1 / Small s0 / Small s1 | anchored | 2 / 2.3%, 1 / 2.3%, 4 / 1.3% | 1 / 8.6%, 1 / 7.9%, 4 / 4.7% |

Two further frozen variants, a shrunk predicted change and a mismatch gate that picks the start with the smaller initial mismatch, lower the anchored start's false flags on GOC 10000 (to 0.4 to 0.9% loading-only) with zero to three misses depending on the checkpoint. The gains are mixed across checkpoints and have no demonstrated cost benefit, so none of them enters the headline; all are reported in the repository (`work/fresh/results_hybrid.md`).

## Where it stops working

- **10,000 buses, every released checkpoint.** No released checkpoint beats the classic screen: the classic start misses 2 lines, the anchored checkpoints miss 5, 3, 2 and 3 (Tiny seed 0, Tiny seed 1, Small seed 0, Small seed 1); the best, Small seed 0, ties on misses while referring about 7 times as many secure scenarios (2.1% against 0.3%). The bigger model improves the diagnostics (no-change error removed 35 to 39% against 28 to 31%) at about 2.2 times the Tiny inference, not the screening outcome. Table in the appendix.
- **Fine-tuning does not change it.** On GOC 10000 the anchored change start goes from 5 to 4 missed at 3.2% flagged. On GOC 500 the raw error falls from 0.33 to 0.23 points, yet after one Newton step nothing changes (5 → 5).
- **Reactive limits keep the ranking, not "almost nothing missed".** With PV-to-PQ switching in the reference solution, lines missed on GOC 500 are 23 (anchored change), 28 (GENCO) and 54 (pre-outage) of 2,570. A finding we withdraw: GENCO's 98.8% detection of voltage-limit violations on IEEE 118 becomes 52–70% against the limit-enforced truth.

## Limits of this work

- **We assume a converged AC base state** consistent with the injections, controls and network model used for the contingency. Sensitivity to state-estimation error, stale operating points and topology uncertainty was not tested.
- **No speed claim.** The savings we report are Newton iterations and AC solves.
- **Fine-tuning was small**: at most two hours per grid on a laptop, Tiny only.
- **The Newton step uses the true network model**: it measures GENCO plus physics.
- **The mismatch after a step is a heuristic**, not a bound on loading error: a two-bus case in the `gridtrust` tests passes a 1 MW check with a true overload, and the 14 outages missed in the N-1 study left a median mismatch of 0.5 MW, below the 5 MW threshold. Misses are reported, never bounded.
- **GOC 500 is a near-replication of the example**: its checkpoint and data are not published.
- **Overload means AC loading above 100.1%**; the OPF dispatch parks lines at exactly 100%, so most remaining misses are between 100.1% and 100.6%.
- **Timings were taken on a laptop** with a system process using one core; unit costs are indicative.

A bounded robustness check of the anchored start against equivalent circuit encodings (parallel branch copies, split generators) is in the [appendix](#equivalent-circuit-representations-bounded-robustness-diagnostic): it greatly reduces GENCO's sensitivity for ordinary rewrites and fails for the extreme all-branch doubling on GOC 500.

## An operator workbench

The staged workflow runs interactively on IEEE 118: pick a load level and an allowed miss rate, click lines out of service, and see which stage decided the state and why, checked against the AC solution on screen. English and Chinese.

![Workbench](figures/workbench.gif)

## Tools, code and data

- **GENCO** checkpoints from `gridfm/genco-pf-datakit` (IEEE 118, GOC 500, GOC 2000, GOC 10000 Tiny and Small), `gridfm/genco-pf-contingency` (Texas 2000) and `gridfm/genco-pf-transfer` (pretrained model).
- **gridfm-graphkit** for the model and inference; **gridfm-datakit** for scenario generation and the AC, DC and OPF solves.
- `gridtrust`: a small standalone package (numpy, scipy, pandas only) with the Newton refinement, the conformal band and the staged decision, 41 tests (with the rated-line mask passed, a non-finite prediction or loading on a rated line is never accepted as secure). A draft contribution to gridfm-graphkit is in `upstream/`.
- Code, result tables and `reproduce.sh` (`setup`, `fetch`, `checksums`, `tests`, `analysis`, `figures`; long stages behind `RUN_LONG=1`): [github.com/KehanLiu/genco-anchored-screening](https://github.com/KehanLiu/genco-anchored-screening) (MIT).

## What others have done

One-step screening from the pre-outage solution is classical (Albuyeh, Bose and Heath, [IEEE Trans. PAS, 1982](https://doi.org/10.1109/TPAS.1982.317327); Chen and Bose's "1P-1Q", [1989](https://doi.org/10.1109/59.41683), [1990](https://doi.org/10.1109/59.65923); Stott, Alsac and Monticelli, [Proc. IEEE, 1987](https://doi.org/10.1109/PROC.1987.13931)), and gridfm-datakit seeds its post-outage power flow with the pre-outage point ([PR #67](https://github.com/gridfm/gridfm-datakit/pull/67)). Learned warm starts for Newton are an active line (Newton's Lantern, [Bose, Hilmarsson and Suri, 2026](https://arxiv.org/abs/2605.11102); Li, Pandey and Pileggi, [PSCC 2024](https://arxiv.org/abs/2304.06727); Kim et al., [October 2026](https://arxiv.org/abs/2610.05959); Rahman and Kekatos, [2026](https://arxiv.org/abs/2609.36510)), as is conformal N-k screening on GridFM (Alcántara and Chatzivasileiadis, [arXiv 2602.07995](https://arxiv.org/abs/2602.07995)); the appendix says how each differs. Ours starts from the pre-outage solution plus GENCO's predicted change, is scored by overloads missed and secure scenarios flagged, and is compared with the classical one-step screen.

**Author:** Kehan Liu, independent. **Contact:** kehan_liu@outlook.com. Entered privately; public data only.

## Appendix: full results

### What the organisers envisioned

GENCO paper, [section 6](https://arxiv.org/abs/2608.09921): "we envision a safeguarded hybrid workflow in which GENCO rapidly evaluates all scenarios, while power-balance residuals and violation checks identify critical cases for re-evaluation with a classical AC solver." Example post, [discussion #1](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/1): "We plan to add conformal prediction so the classifier can quantify uncertainty and abstain when it cannot make a sufficiently reliable safety decision." [IBM Research blog](https://research.ibm.com/blog/gridfm-neural-solver-power-grid): "classical solvers provide authoritative solutions for those few cases". GENCO is presented as a fast surrogate for AC power flow; we do not test it in that role. The score is overloads missed, secure scenarios flagged and Newton iterations, never seconds: on our laptop one GENCO inference costs about three Newton iterations, and no GENCO workflow was faster than the classic screen.

### Main result: the mechanism in detail

The two 10,000-bus rows of the main-result table, scored as above:

| Grid (overloaded lines) | Pre-outage solution: missed / secure scenarios flagged | GENCO: missed / flagged | Anchored change start: missed / flagged |
|---|---|---|---|
| GOC 10000 (507), Tiny | **2 / 0.3%** | 7 / 37.0% | 5 / 3.2% |
| GOC 10000 (507), Small | **2 / 0.3%** | 6 / 28.2% | 2 / 2.1% |

- **The mechanism is measurable.** On lines whose loading moves by more than one point, GENCO's predicted change removes 85% of the median error of assuming no change on GOC 500 (2.44 → 0.36 points) and 74% on GOC 2000 (2.17 → 0.56); the sign of the change is right for 93–97% of moving lines.
- **At 10,000 buses it ties, with more false flags.** The predicted change removes only 28% (Tiny) to 35–39% (Small) of the no-change error, and no checkpoint beats the pre-outage start. Two Newton steps bring every start to 0–1 missed lines.

### N-1 study: more detail

- **Fewest Newton iterations** among workflows that miss under 1.5% of critical outages (it misses 14, the classic screen 5), and the smallest loading error after one step (99th percentile 0.006 points, against 0.032 from the pre-outage solution).
- **In operator terms (derived):** 2.7% sent to AC is about one full AC solve per 37 outages; the classic screen's 29.6% is about one per 3. The [IBM Research blog](https://research.ibm.com/blog/gridfm-neural-solver-power-grid) notes that MISO "evaluates on the order of 10,000 contingencies every few minutes"; per such round that is about 270 full AC solves against about 2,960 (derived). These are solver calls, not seconds; on wall-clock time the classic screen still wins.
- **Inside PowerModels' AC power flow** (the solver gridfm-datakit uses), GENCO's start is as reliable as the pre-outage solution (GOC 2000: 200 of 200 scenarios converge against 199; GOC 10000: 100 of 100 against 92) where flat and DC starts mostly fail (20 and 46 of 200; 0 of 100). It saves no solver time; with inference the chain is 2 to 4 times slower. Its place is where no solved neighbouring state exists.

### 10,000 buses, every released checkpoint

993 scenarios, 507 overloaded lines, GENCO on CPU.

| Start | 1 step: missed / secure flagged | 2 steps: missed / flagged | No-change error removed | Sign agreement | GENCO ms per state |
|---|---|---|---|---|---|
| Pre-outage solution | **2 / 0.3%** | 0 / 0.0% | – | – | – |
| Anchored change, Tiny seed 0 | 5 / 3.2% | 0 / 0.1% | 28% | 75.4% | 178 |
| Anchored change, Tiny seed 1 | 3 / 2.9% | 1 / 0.0% | 31% | 77.3% | 197 |
| Anchored change, Small seed 0 | 2 / 2.1% | 0 / 0.4% | 35% | 79.2% | 395 |
| Anchored change, Small seed 1 | 3 / 1.1% | 0 / 0.3% | 39% | 81.4% | 402 |

Only Small seed 0 ties the pre-outage start on missed lines (2 against 2) and does so while referring about seven times as many secure scenarios (2.1% against 0.3%); Small seed 1 misses 3 at 1.1%, Tiny seed 0 and seed 1 miss 5 and 3. The bigger model improves the diagnostics (no-change error removed 35 to 39% against 28 to 31%, sign agreement 79 to 81% against 75 to 77%) at about 2.2 times the Tiny inference, without beating the classic screen. Seeds agree on the verdict; one or two lines between checkpoints is seed noise.

### Fine-tuning

We fine-tuned the released Tiny checkpoints on new scenarios of the same kind (8,969 for GOC 500, 2,248 for GOC 2000, 1,347 for GOC 10000; half an hour to two hours each on a laptop GPU). On GOC 10000 the anchored change start goes from 5 to 4 missed at 3.2% flagged and the raw error does not improve (median 0.54 → 0.56 points). On GOC 500 the raw error falls from 0.33 to 0.23 points, yet after one Newton step nothing changes (5 → 5).

![Fine-tuning, before and after](figures/finetune.png)

### Reactive limits

With PV-to-PQ switching in the reference solution, lines missed after one step on GOC 500 are 23 (anchored change), 28 (GENCO) and 54 (pre-outage) of 2,570; the exact no-limit solution, read with the same 100.1% cutoff, misses 19: the overloads that appear only once reactive limits bind, in states the no-limit solution calls secure. That is the gap between the two formulations, not a floor; a cutoff of 100% or 95% lowers it to 9 or 0 missed, at the price of sending 5.0% or 23.5% of secure states to AC instead of 0.2%. The worst missed overload is 111%; two Newton steps with the switching logic bring it back to 5 missed, worst 100.5%. A finding we withdraw: GENCO's 98.8% detection of voltage-limit violations on IEEE 118 becomes 52–70% against the limit-enforced truth, as does the exact no-limit solution.

### Zero-shot transfer

Zero-shot transfer gives no universal start. The organisers' pretrained model, without fine-tuning, is 2 to 9 times less accurate than DC on IEEE 118, GOC 500, PEGASE 1354 and GOC 2000. Its predicted change still helps a little on the three smaller grids (lines missed after one step: 74 → 19, 12 → 3, 8 → 5) and hurts on GOC 2000.

### Embedding space

The embedding space does not warn of unfamiliar states. Distance to the calibration set separates N-3 and N-4 states from the control with AUROC 0.68 and 0.79; counting the lines out gives 1.00.

### A conformal band without a Newton step

Without a Newton step, a conformal band buys a stated miss rate at a price. A line is "possibly overloaded" when GENCO's prediction exceeds `limit − b·σ`, with `b` calibrated so that the share of real overloads missed targets α. IEEE 118, α = 1%: 31.9% of states go to AC, 31.2% really have an overload, and none of the 792 held-out overloads sits in a state accepted without AC. The band targets α per overloaded line reading, with `b` set on pooled calibration readings: over 100 random splits by outage set, 9.9 / 5.0 / 1.9 / 1.0% of real overloads are not flagged themselves for α = 10 / 5 / 2 / 1% (empirical coverage at the target, not a guarantee); because any flagged line sends the whole state to AC, the share of overloads inside accepted states is lower, 4.4 / 2.0 / 0.23 / 0.05%. DC tuned the same way escalates 81% of secure states; on Texas 2000 it ties GENCO (7.9% against 7.7%). Under N-3 and N-4 outages the promise breaks (6.5% and 7.1% missed at α = 5%). Alcántara and Chatzivasileiadis ([arXiv 2602.07995](https://arxiv.org/abs/2602.07995)) screen N-k contingencies with conformal prediction on GridFM; ours adds the Newton step and the AC-call accounting.

### Equivalent circuit representations (bounded robustness diagnostic)

**Equivalent circuit representations.** On the two primary fresh-pool experiments, anchoring plus one Newton step greatly reduced GENCO's sensitivity to one-branch and non-slack-generator splitting. The median across states of the maximum line-loading difference was below 0.05 points of rating, and loading-only decisions agreed in 99.7–100% of states. Both rewrites passed our registered tolerances, although generator splitting on GOC 500 increased missed overloaded readings from four to six and produced rare large first-step errors. Splitting all eligible branches failed the criterion on all three GOC 500 checkpoints; the primary loading-only missed count rose from four to 65. Both GOC 2000 checkpoints passed, with missed counts zero→one and zero→zero. These are bounded robustness diagnostics before mismatch abstention, not representation invariance or a safety guarantee.

Each rewrite preserves the admittance matrix (to order 1e-12, within 2e-12), injections, bus types and setpoints, and every copy of an outaged split branch is disabled together. For GOC 500 Tiny seed 0, the median across states of the largest bus-voltage drift of GENCO's raw prediction was about 26 degrees and 0.14 p.u. under all-branch doubling (maxima 52 degrees and 0.47 p.u.), which is where the one-step result also breaks (poorer initialization, large first-step errors, some runs exhausting the iteration budget or converging to a different solution). Pre-registered criterion, transformations and results: `work/equiv/SPEC.md`, `work/equiv/REPORT.md`.

![Representation effects by stage](figures/equiv.png)

### What others have done, in full

Screening with one step from the pre-outage solution is classical: Albuyeh, Bose and Heath used "the first iteration of an AC power flow" ([IEEE Trans. PAS, 1982](https://doi.org/10.1109/TPAS.1982.317327)); Chen and Bose named the fast-decoupled form "1P-1Q" ([1989](https://doi.org/10.1109/59.41683), [1990](https://doi.org/10.1109/59.65923)); Stott, Alsac and Monticelli surveyed it ([Proc. IEEE, 1987](https://doi.org/10.1109/PROC.1987.13931)). Our pre-outage baseline is this method with a full Newton step. gridfm-datakit itself seeds the post-outage power flow with the pre-outage point when it generates data ([PR #67](https://github.com/gridfm/gridfm-datakit/pull/67)).

Newton's Lantern ([Bose, Hilmarsson and Suri, 2026](https://arxiv.org/abs/2605.11102)) tunes GNN warm starts by reinforcement learning and runs Newton to convergence; no contingencies, no pre-outage baseline. Li, Pandey and Pileggi ([PSCC 2024](https://arxiv.org/abs/2304.06727)) compare a learned warm start with the pre-outage start under load-attack contingencies; absolute states, Newton to convergence. Kim et al. ([October 2026](https://arxiv.org/abs/2610.05959)) take one Newton update from calibrated or subspace-projected predictions of four models, gridfm-graphkit among them, on the 2,224-bus GB network, scored by power-balance residual. Rahman and Kekatos ([2026](https://arxiv.org/abs/2609.36510)) reuse a base-case surrogate for outages through a fixed-point iteration with compensation injections; no pre-outage anchor, no differencing, no Newton step.

Ours starts from the pre-outage solution plus GENCO's predicted change, is scored by overloads missed and secure scenarios flagged, and is compared with the classical one-step screen.

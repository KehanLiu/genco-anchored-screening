# Let GENCO predict the change, not the state: contingency screening with the anchored change start and one Newton step

**Author:** Kehan Liu, independent entry (public data only). **Contact:** kehan_liu@outlook.com. **Code, results and reproduction:** [github.com/KehanLiu/genco-anchored-screening](https://github.com/KehanLiu/genco-anchored-screening) (MIT).

> **In brief.** In a complete N-1 analysis of the GOC 2000 grid, GENCO on its own misses 54% of the outages that overload a line. Used differently, it helps. Start Newton's method from the pre-outage AC solution plus the *change* GENCO predicts for the outage, take one step, and send to the full solver only what that step flags. On 76,512 fresh outages, with every rule frozen in advance, this screen sends 2.2% of outages to the full AC solver and misses one critical outage. The classical one-step screen sends 27.9% and misses two. It saves no wall-clock time on our hardware, because one GENCO inference costs about as much as three Newton iterations. We count solver calls and misses, not seconds, and we report where the method stops working.

| Question | Answer, on the released checkpoints and grids |
|---|---|
| Is GENCO's state safe to screen with? | No. On the organisers' GOC 500 setup it lets through overloads up to 139.6% of rating. In a full GOC 2000 N-1 it misses 780 of 1,451 critical outages. |
| Does one Newton step from GENCO's state fix that? | Most of it. But the classical start, one step from the pre-outage solution, does as well at 2,000 buses and better at 10,000. "One step from GENCO" is not a GENCO result. |
| What does GENCO add, then? | Its *predicted change*. The start "pre-outage solution + [GENCO(after) − GENCO(before)]" cancels most of GENCO's systematic bias. After one step it misses 2, 5 and 0 overloaded lines on IEEE 118, GOC 500 and GOC 2000 (of 6,986, 2,621 and 369), against 74, 41 and 2 from the classical start, with 0.0% of secure scenarios flagged. |
| What does it save? | AC solver calls. In the GOC 2000 full N-1, 2.7% of outages go to AC against 29.6% for the classical screen, one full solve per 37 outages instead of one per 3, at 37% of the Newton iterations of complete AC. On the fresh pool with frozen rules: 2.17% against 27.94%, and 1 against 2 critical outages missed. |
| Is it faster? | No. Measured on the fresh pool, 22.3 ms per outage against 8.4 for the classical screen and 13.7 for complete AC, with GENCO Tiny batched on a laptop GPU. The appendix lists the conditions under which it could be. |
| Where does it stop working? | At 10,000 buses no released checkpoint beats the classical start. Laptop-scale fine-tuning does not change that. With reactive limits the ranking holds but misses rise. |

## The idea

This entry starts from the organisers' example ([discussion #1](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/1)), in which GENCO Tiny screens GOC 500 under up to five branch outages against a DC baseline. We reproduce it, with 93.9% recall at 32.7% of scenarios flagged where the example reports 94.3% and 32.7%. Then we add what an operator would ask for next: the pre-outage baseline, one Newton step as the check, four grid sizes, a full N-1 study with costs, reactive limits, fine-tuning, and every released checkpoint at 10,000 buses.

The organisers described this workflow before we built it. The GENCO paper envisions a "safeguarded hybrid workflow" with "re-evaluation with a classical AC solver" ([section 6](https://arxiv.org/abs/2608.09921)); the example post plans conformal abstention; the [IBM Research blog](https://research.ibm.com/blog/gridfm-neural-solver-power-grid) says that "classical solvers provide authoritative solutions for those few cases". Full quotes are in the appendix. We built that workflow and measured it. The useful quantity turned out to be not GENCO's state but its predicted change.

Contingency screening asks one question for thousands of grid states: *is any line overloaded?*

- GENCO alone is not safe enough. On the example's setup it lets through overloads up to 139.6%.
- One Newton step from GENCO's voltages fixes most of that: 9 of 2,621 overloaded lines missed on GOC 500, with 12.5% of scenarios sent to the full solver, which is exactly the share that really has an overload. From a flat start or DC angles the same step is useless (97% and 88% flagged).
- The right baseline, though, is the pre-outage solution. Contingency analysis begins from a solved AC base case, so the fair comparison is one Newton step from that same base case. That is classic single-iteration screening (Albuyeh, Bose and Heath, 1982). It ties GENCO's start at 2,000 buses and beats it at 10,000.
- GENCO's errors are systematic. On Texas 2000, 97% of the errors above 3 points occur at one load level, on the same lines in every state.
- So we subtract them. The start is the pre-outage AC voltages plus [GENCO(after outage) − GENCO(before outage)]. Most of the bias cancels: on lines that move by more than one point, the predicted change removes 85% of the no-change error on GOC 500 and 74% on GOC 2000. What survives is GENCO's estimate of the change, plus whatever error the base state carries. This needs two inferences, one of which is shared by all outages of a base case, one Newton step, and no training.

![GENCO's error before and after the outage](figures/bias_cancels.png)

## Main result: where you start decides what one step finds

Scenarios remove up to five branches at random (gridfm-datakit). A scenario goes to the full solver if any line loading after the step exceeds its limit (cutoff 100.1%). Every start is scored against the full AC solution, with released checkpoints and a converged AC base state.

| Grid (overloaded lines) | Pre-outage solution: missed / secure scenarios flagged | GENCO: missed / flagged | Anchored change start: missed / flagged |
|---|---|---|---|
| IEEE 118 (6,986) | 74 / 1.9% | 4 / 0.0% | **2 / 0.0%** |
| GOC 500 (2,621) | 41 / 0.2% | 9 / 0.1% | **5 / 0.0%** |
| GOC 2000 (369) | 2 / 0.0% | 1 / 0.6% | **0 / 0.0%** |
| GOC 10000 (507), Tiny | **2 / 0.3%** | 7 / 37.0% | 5 / 3.2% |
| GOC 10000 (507), Small | **2 / 0.3%** | 6 / 28.2% | 2 / 2.1% |

![Where you start decides what one step finds](figures/start_step.png)

The anchored change start begins much closer to the solution. Its median power mismatch is 6 MW on GOC 500 and GOC 2000, against 72–118 MW for the pre-outage solution and 246–435 MW for GENCO alone. Newton then converges in 2.4–2.5 iterations instead of 3.0–3.2.

The mechanism can be measured directly. On lines whose loading moves by more than one point, GENCO's predicted change removes 85% of the median error of assuming no change on GOC 500 and 74% on GOC 2000. On GOC 10000 it removes only 28–39%, and that is where the screen stops beating the classical start.

The Newton step is essential. Without it, the anchored change alone misses 14–44% of overloaded lines. The step finds the overloads; GENCO improves where the step begins.

## A complete N-1 analysis: what it costs and what it is worth

GOC 2000, 24 base cases, every non-islanding single-branch outage: 76,512 outages. Mismatch thresholds are set on 6 base cases and the other 18 are scored (57,384 outages, 1,451 critical). Seconds are estimates from unit costs measured on this laptop: a Newton iteration takes 3.9 ms, a GENCO inference batched on the GPU 12.6 ms per state, and building its input graph 2.5 ms. The base-case solve and the pre-outage inference are computed once per base case and excluded for every workflow alike.

![A complete N-1 analysis](figures/fulln1.png)

| Workflow | Sent to full AC | Critical outages missed | Newton iterations vs full AC | Seconds per 1,000 outages |
|---|---|---|---|---|
| Full AC for every outage | 100% | 0 | 100% | 11.5 |
| One step from the pre-outage solution (classic) | 29.6% | 5 of 1,451 | 57% | **6.5** |
| GENCO alone | 6.6% | 780 (54%) | 7% | 15.9 |
| One step from GENCO | 45.4% | 1 | 76% | 23.8 |
| One step from the anchored change start | **2.7%** | 14 (13 of them ≤ 100.14%) | **37%** | 19.4 |
| DC sensitivities on the AC base case | 2.1% | 303 (21%) | 3% | 2.3 |

In operator terms, 2.7% sent to AC is about one full AC solve per 37 outages, and the classical screen's 29.6% is about one per 3. MISO "evaluates on the order of 10,000 contingencies every few minutes" ([IBM Research blog](https://research.ibm.com/blog/gridfm-neural-solver-power-grid)). Per such round that would be about 270 full AC solves against about 2,960 (derived). Among the workflows that miss under 1.5% of critical outages, the anchored screen needs the fewest Newton iterations. It also leaves the smallest loading error after one step: a 99th percentile of 0.006 points, against 0.032 from the pre-outage solution.

Two caveats belong next to that table. The two screens do not miss the same outages (14 against 5 of 1,451, with 13 of the 14 within 0.14 points of the limit), so the saving is not at equal safety. And there is no wall-clock gain. One GENCO inference costs as much as three Newton iterations, and the graph build alone costs more than the 2.2 ms of Newton work the anchored start saves against the classical screen. The classical screen wins on time. What GENCO buys is fewer calls to the full solver.

## Fresh operating points, frozen policies

A screen tuned on the data it is scored on proves little. So we froze every threshold, cushion and gate, then generated a fresh GOC 2000 pool: six registered bases, extended afterwards by 18 more under the same frozen policies, 24 bases and 76,512 outages in all, 1,510 critical. Every policy variant is reported in the repository and none was selected on these results. On the six registered bases both screens missed zero critical outages. Over all 24 the anchored screen missed one critical outage and referred 2.17% of outages to AC; the classical screen missed two and referred 27.94%.

On a 4,800-outage timing sample from the same pool, both screens missed zero of 69 critical cases. Measured throughput was 22.3 ms per outage for the anchored workflow, with GENCO Tiny on the laptop GPU in batches of 16, against 8.4 ms for the classical screen and 13.7 ms for complete AC. The appendix scores the same pool state by state, including GOC 10000 with four released checkpoints and the two frozen variants we did not promote.

## Where it stops working

**10,000 buses, every released checkpoint.** No released checkpoint beats the classical screen. The classical start misses 2 lines; the anchored checkpoints miss 5, 3, 2 and 3 (Tiny seed 0, Tiny seed 1, Small seed 0, Small seed 1). The best of them, Small seed 0, ties on misses while referring about 7 times as many secure scenarios (2.1% against 0.3%). The bigger model improves the diagnostics, removing 35–39% of the no-change error against 28–31% for Tiny, at about 2.2 times the inference cost. It does not improve the screening outcome.

**Fine-tuning does not change it.** We fine-tuned the released Tiny checkpoints on new multi-outage scenarios, up to two hours per grid on a laptop GPU. On GOC 10000 the anchored start goes from 5 to 4 missed at 3.2% flagged. On GOC 500 the raw error falls from 0.33 to 0.23 points, yet after one Newton step nothing changes (5 → 5).

**Reactive limits keep the ranking, not "almost nothing missed".** With PV-to-PQ switching in the reference solution, lines missed on GOC 500 are 23 (anchored change), 28 (GENCO) and 54 (pre-outage) of 2,570. One finding we withdraw: GENCO's 98.8% detection of voltage-limit violations on IEEE 118 becomes 52–70% against the limit-enforced truth.

**No universal start.** The organisers' pretrained model, used zero-shot, is 2 to 9 times less accurate than DC on four grids. Its embedding distance does not warn of unfamiliar states either: AUROC 0.68–0.79 for N-3/N-4 against a control, where simply counting the lines out gives 1.00.

## Extending GENCO: the anchored start as a toolkit option

Two pieces of code make the extension concrete.

**A draft contribution to gridfm-graphkit** (`upstream/`, MIT). `newton_refine.py` adds two functions. `build_ybus(data)` assembles the sparse bus admittance matrix from graphkit's directed-edge admittances and bus shunts. `newton_refine(data, bus_output, steps=1, ref_vm="label")` takes one denormalised `HeteroData` graph plus `output["bus"]` and runs full Newton-Raphson steps from the predicted voltages. It returns a `NewtonRefineResult` with corrected `Vm`/`Va`, per-edge flows and loadings, bus residuals in graphkit's sign convention, the mismatch history and an `ok` flag. The hook is written for `PowerFlowTask.test_step`/`predict_step`, after `_clamp_known_to_ground_truth`, behind an opt-in config key like `--compute_dc_ac_metrics`. `PROPOSAL.md` describes the hook, a companion conformal-screening utility and the test plan. Status: a draft, checked on one IEEE 14 scenario, not yet submitted upstream.

**`gridtrust`** is a standalone numpy/scipy/pandas package with no torch dependency. Its three modules are `newton` (`PowerFlowProblem.refine` and a mismatch self-check), `conformal` (`calibrate`, `flag`, `evaluate`, `anchored`, `shift_sigma`) and `staged` (`decide`: the band first, then a Newton step with a mismatch check, then AC). 41 tests pass, including equivalence checks against the entry's stored results. `examples/synthetic_screening.py` runs the full workflow.

## An operator workbench

The staged workflow runs interactively on IEEE 118. Pick a load level and an allowed miss rate, click lines out of service, and see which stage decided the state and why, checked against the AC solution on screen. English and Chinese.

![Workbench](figures/workbench.gif)

## Limits of this work

- We assume a converged AC base state consistent with the injections, controls and network model used for the contingency. Sensitivity to state-estimation error, stale operating points and topology uncertainty was not tested.
- No speed claim is made. The savings we report are Newton iterations and AC solves. Timings were taken on a laptop, single-process, with a system process using one core. Complete AC and the classical screen parallelise across cores and were not given that advantage.
- The Newton step uses the true network model, so it measures GENCO plus physics.
- The mismatch after a step is a heuristic, not a bound on loading error. The 14 outages missed in the N-1 study left a median mismatch of 0.5 MW, below the 5 MW threshold. Misses are reported, never bounded.
- Overload means AC loading above 100.1%. The OPF dispatch parks lines at exactly 100%, so most remaining misses lie between 100.1% and 100.6%.
- GOC 500 is a near-replication of the example; its checkpoint and data are not published. Fine-tuning was small and Tiny only.

## Open science and reproducibility

- **Models and toolkits.** GENCO checkpoints from `gridfm/genco-pf-datakit` (IEEE 118, GOC 500, GOC 2000, GOC 10000 Tiny and Small), `gridfm/genco-pf-contingency` (Texas 2000) and `gridfm/genco-pf-transfer` (pretrained model). gridfm-graphkit runs the model and inference; gridfm-datakit generates scenarios and runs the AC, DC and OPF solves. Inputs are fetched by commit sha and verified against a sha256 manifest.
- **Repository.** [github.com/KehanLiu/genco-anchored-screening](https://github.com/KehanLiu/genco-anchored-screening) (MIT) holds every analysis script, result table and figure. `reproduce.sh` has stages `setup`, `fetch`, `checksums`, `tests`, `analysis` and `figures`, with the recomputation of every result table behind `RUN_LONG=1`: `sweeps`, `controls` and `finetune` (about two days), then `fresh` (the fresh pools, the six registered N-1 bases and the 18-base extension, hybrid starts and the timing sample, about 3 h) and `equiv` (the circuit-representation runs, 33 min). The `setup`, `fetch`, `checksums` and `tests` stages run from a clean clone. The `analysis` and `figures` stages need the saved result tables (not in git; produced only by the long stages) and the GOC 2000 test and fresh pools; with those copied in from the working copy that produced this post, a clean-clone replay reproduced all 17 figures byte-identical and the fresh-pool scorers regenerated their tables identically. No results archive exists, so that replay cannot yet be repeated by a third party without rerunning the long stages.
- **Registered number check.** `results/` holds the summary tables behind the result sections of this post. `python scripts/headline_numbers.py --check` recomputes 274 registered claims from those tables and compares each with the value this post stated when the claim was registered; CI fails when a value differs, a claim is missing, or the committed snapshot changes, and then runs the tests of the check itself. It does not parse this post. Outside its coverage are the equivalent-circuit section, the `gridtrust` test counts (checked by the test step instead), the quoted passages and the counts about the repository; `results/README.md` lists them, along with the ratios in the prose that are derived from registered numbers.
- **Pre-registered where it matters.** The fresh-pool policies (`work/fresh/policy_frozen.json`, `POLICY_LOG.md`) and the circuit-representation check (`work/equiv/SPEC.md`) were frozen before the data they were scored on existed.

## What others have done

**Other entries in this challenge.** The organisers' example (#1) is our starting point and the DC baseline we keep. [#2 GENCO-PTDF](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/2) audits GENCO's sensitivities on a fixed topology, which is orthogonal to outage screening. [#7 Trust, but verify](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/7) (Shooshtari) wraps GENCO in a verification step that runs Newton to convergence on every case and raises an alarm when no start converges; it also studies clustered N-10/N-20 outages near voltage collapse. The two entries are complementary and can be chained. That one decides what to do *after* a case reaches the solver. This one decides *which* cases reach it, and at what cost in solver calls. On cost it reaches the same conclusion we do: GENCO's start saves about one Newton iteration, and inference costs more than the classical solve. Two baselines it does not include are the ones that decide our result, the classical one-step screen from the pre-outage solution, which beats every GENCO start at 10,000 buses, and GENCO's *change* rather than its state as the start. Its embedding warning head predicts a different label from our embedding-distance check (solver non-convergence, supervised, against unfamiliar states, unsupervised), so the two findings do not conflict.

**Literature.** One-step screening from the pre-outage solution is classical: Albuyeh, Bose and Heath ([IEEE Trans. PAS, 1982](https://doi.org/10.1109/TPAS.1982.317327)), Chen and Bose's "1P-1Q" ([1989](https://doi.org/10.1109/59.41683), [1990](https://doi.org/10.1109/59.65923)), and the survey by Stott, Alsac and Monticelli ([Proc. IEEE, 1987](https://doi.org/10.1109/PROC.1987.13931)). gridfm-datakit itself seeds its post-outage power flow with the pre-outage point ([PR #67](https://github.com/gridfm/gridfm-datakit/pull/67)). Learned warm starts for Newton are an active line: Newton's Lantern ([Bose, Hilmarsson and Suri, 2026](https://arxiv.org/abs/2605.11102)), Li, Pandey and Pileggi ([PSCC 2024](https://arxiv.org/abs/2304.06727)), Kim et al. ([October 2026](https://arxiv.org/abs/2610.05959)) and Rahman and Kekatos ([2026](https://arxiv.org/abs/2609.36510)). Alcántara and Chatzivasileiadis ([arXiv 2602.07995](https://arxiv.org/abs/2602.07995)) screen N-k contingencies with conformal prediction on GridFM. The appendix says how each differs. Ours starts from the pre-outage solution plus GENCO's predicted change, is scored by overloads missed and secure scenarios flagged, and is compared with the classical one-step screen.

---

## Appendix: full results

### What the organisers envisioned

GENCO paper, [section 6](https://arxiv.org/abs/2608.09921): "we envision a safeguarded hybrid workflow in which GENCO rapidly evaluates all scenarios, while power-balance residuals and violation checks identify critical cases for re-evaluation with a classical AC solver." Example post, [discussion #1](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/1): "We plan to add conformal prediction so the classifier can quantify uncertainty and abstain when it cannot make a sufficiently reliable safety decision." [IBM Research blog](https://research.ibm.com/blog/gridfm-neural-solver-power-grid): "classical solvers provide authoritative solutions for those few cases". GENCO is presented as a fast surrogate for AC power flow; we do not test it in that role. The score is overloads missed, secure scenarios flagged and Newton iterations, never seconds: on our laptop one GENCO inference costs about three Newton iterations, and no GENCO workflow was faster than the classic screen.

### Measured end-to-end timing

| Workflow (same 4,800 outages, 24 fresh GOC 2000 bases) | Measured ms per outage, first pass | Critical cases missed in this sample |
|---|---:|---:|
| Complete AC, Newton to convergence from the solved base case | 13.7 | 0 of 69 |
| Classical screen, one Newton step from the solved base case | 8.4 | 0 of 69 |
| Anchored change start, one Newton step, GENCO on the laptop GPU in batches of 16 (amortized throughput) | 22.3 | 0 of 69 |

Numerical nonconvergences were treated as flagged cases. These are laptop throughput measurements, not a measurement of deployment cost either way.

![Overload screening after one Newton step](figures/starts.png)

### Where it could win, and what that would take

Nothing in this section is measured on the target hardware. It is a unit-cost estimate built from the fresh GOC 2000 timing sample (4,800 outages, 24 bases) and the quiet-machine unit costs of the same run. We write it down so that the conditions are explicit. A useful screen must be both more accurate than the classical screen and cheaper than complete AC, and the two conditions depend on different things.

**Accuracy, holding the hardware fixed, depends on the model.** On IEEE 118, GOC 500 and GOC 2000 the anchored start already misses fewer overloads than the classic start at the same one-step budget: 2, 5 and 0 against 74, 41 and 2 loading-only, and 1 against 2 critical outages on the fresh N-1 pool. At 10,000 buses it does not. On lines whose loading moves by more than one point, the released checkpoints remove only 28 to 39% of the median absolute error of the "no change" assumption, against 74 to 85% on the smaller grids. Neither a larger checkpoint nor laptop-scale fine-tuning beats the classic start there. Whether a model that predicted the change as well at 10,000 buses as at 2,000 would close the gap is a hypothesis we could not test. Fine-tuning on states rather than on changes did not close it.

**Cost, holding the model fixed, depends on the hardware and the graph builder.** Per outage the anchored workflow needs 1.08 Newton iterations on average, against 2.92 for complete AC and 1.61 for the classic screen. It therefore saves 1.84 iterations (7.0 ms at 3.81 ms per iteration) against complete AC and 0.53 (2.0 ms) against the classic screen. What it adds per outage is one forward pass (12.74 ms on the laptop GPU at batch 16), one graph build (2.40 ms of CPU work) and its share of the pre-outage inference (0.18 ms on this sample, one inference per base amortized over 200 timed outages). Against complete AC, the forward pass would have to cost about 4.4 ms or less with the current builder for the estimate to break even. Against the classic screen, the builder alone already exceeds the saving, so a builder that reuses the base-case graph across outages would be a precondition. A larger model raises the forward cost: on GOC 10000, on the CPU, Small took about 2.2 times Tiny per state. This budget is therefore for Tiny. Two scopes are mixed here on purpose and should be read apart. The measured end-to-end timings include batch assembly and device transfer and exclude model loading, the base-case AC solve and warm-up, for every workflow alike. The forward-only unit cost (12.74 ms) excludes batch assembly and transfer as well, so the budget above is a simplified estimate.

**Scale might favour the method, if accuracy held.** A Newton iteration is a sparse factorisation whose cost typically grows faster than linearly with the number of buses, while a graph network's forward cost grows roughly linearly. If that held for these solvers and models, the cost ratio would improve with grid size. We did not measure it, and grid size is where the accuracy currently fails.

**The fair baseline is a parallel one.** Complete AC and the classic screen parallelise across CPU cores. Any throughput comparison would have to give them the same cores it gives the model its GPU. On this laptop every workflow was timed single-process.

Put together: on GOC 2000 the accuracy condition is met today. Whether the cost condition can be met is a question about server hardware, batched inference and a reusable graph builder that we could not measure. At 10,000 buses both conditions are open. This is a projection with its unmeasured steps named, not a result.

### Fresh pool, state by state

Here the fresh pool is scored state by state for the warm starts themselves: one Newton step, cutoff 1.001, missed critical states and secure states sent to AC. The same files are read in two ways. The loading-only diagnostic flags a state when any line is above the cutoff. The complete frozen policy also refers a state when the post-step mismatch exceeds that start's threshold (0.5 MW for the pre-outage start, 5 MW for the anchored start, both calibrated on the old GOC 2000 N-1 study and transferred unchanged). Under the full policy both starts miss zero on GOC 2000 Tiny and on GOC 10000 Tiny seed 0; the other anchored checkpoints on GOC 10000 miss 1, 1 and 4. The referral counts are not an equally tuned comparison. Each start keeps its own residual threshold from the old N-1 calibration, transferred without matching safety on these multi-outage states, and a residual tolerance is not proportional to a loading tolerance. The referral gap is therefore not to be credited to GENCO alone.

| Fresh pool (states, critical) | Start | Loading-only: missed / secure referred | Full frozen policy: missed / secure referred |
|---|---|---|---|
| GOC 2000 Tiny (2,496, 140) | pre-outage | 3 / 0.0% | 0 / 54.1% |
| GOC 2000 Tiny | anchored | **0 / 0.0%** | 0 / 0.8% |
| GOC 10000 Tiny s0 (999, 231) | pre-outage | 0 / 1.0% | 0 / 60.5% |
| GOC 10000 Tiny s0 | anchored | 0 / 2.5% | 0 / 8.1% |
| GOC 10000 Tiny s1 / Small s0 / Small s1 | anchored | 2 / 2.3%, 1 / 2.3%, 4 / 1.3% | 1 / 8.6%, 1 / 7.9%, 4 / 4.7% |

Two further frozen variants, a shrunk predicted change and a mismatch gate that picks the start with the smaller initial mismatch, lower the anchored start's false flags on GOC 10000 (to 0.4 to 0.9% loading-only) with zero to three misses depending on the checkpoint. The gains are mixed across checkpoints and come with no measured cost benefit, so none of them enters the headline. All are reported in the repository (`work/fresh/results_hybrid.md`). This pool is a fresh set of operating points on the same network and generation recipe. Its lower maximum load and different stress mix limit any comparison with the old miss rates.

### Main result: the mechanism in detail

- **The mechanism in numbers.** On lines whose loading moves by more than one point, GENCO's predicted change removes 85% of the median error of assuming no change on GOC 500 (2.44 → 0.36 points) and 74% on GOC 2000 (2.17 → 0.56). The sign of the change is right for 93–97% of moving lines.
- **At 10,000 buses it ties, with more false flags.** The predicted change removes only 28% (Tiny) to 35–39% (Small) of the no-change error, and no checkpoint beats the pre-outage start. Two Newton steps bring every start to 0–1 missed lines.
- **Inside PowerModels' AC power flow** (the solver gridfm-datakit uses), GENCO's start is as reliable as the pre-outage solution. On GOC 2000, 200 of 200 scenarios converge against 199; on GOC 10000, 100 of 100 against 92. Flat and DC starts mostly fail (20 and 46 of 200; 0 of 100). It saves no solver time, and with inference the chain is 2 to 4 times slower. Its place is where no solved neighbouring state exists.

### 10,000 buses, every released checkpoint

993 scenarios, 507 overloaded lines, GENCO on CPU.

| Start | 1 step: missed / secure flagged | 2 steps: missed / flagged | No-change error removed | Sign agreement | GENCO ms per state |
|---|---|---|---|---|---|
| Pre-outage solution | **2 / 0.3%** | 0 / 0.0% | – | – | – |
| Anchored change, Tiny seed 0 | 5 / 3.2% | 0 / 0.1% | 28% | 75.4% | 178 |
| Anchored change, Tiny seed 1 | 3 / 2.9% | 1 / 0.0% | 31% | 77.3% | 197 |
| Anchored change, Small seed 0 | 2 / 2.1% | 0 / 0.4% | 35% | 79.2% | 395 |
| Anchored change, Small seed 1 | 3 / 1.1% | 0 / 0.3% | 39% | 81.4% | 402 |

Only Small seed 0 ties the pre-outage start on missed lines (2 against 2) and does so while referring about seven times as many secure scenarios (2.1% against 0.3%); Small seed 1 misses 3 at 1.1%, Tiny seed 0 and seed 1 miss 5 and 3. Seeds agree on the verdict; one or two lines between checkpoints is seed noise.

### Fine-tuning

We fine-tuned the released Tiny checkpoints on new scenarios of the same kind (8,969 for GOC 500, 2,248 for GOC 2000, 1,347 for GOC 10000; half an hour to two hours each on a laptop GPU). On GOC 10000 the anchored change start goes from 5 to 4 missed at 3.2% flagged and the raw error does not improve (median 0.54 → 0.56 points). On GOC 500 the raw error falls from 0.33 to 0.23 points, yet after one Newton step nothing changes (5 → 5).

![Fine-tuning, before and after](figures/finetune.png)

### Reactive limits

With PV-to-PQ switching in the reference solution, lines missed after one step on GOC 500 are 23 (anchored change), 28 (GENCO) and 54 (pre-outage) of 2,570. The exact no-limit solution, read with the same 100.1% cutoff, misses 19. Those are the overloads that appear only once reactive limits bind, in states the no-limit solution calls secure. That is the gap between the two formulations, not a floor. A cutoff of 100% or 95% lowers it to 9 or 0 missed, at the price of sending 4.9% or 23.5% of secure states to AC instead of 0.2%. The worst missed overload is 111%; two Newton steps with the switching logic bring it back to 5 missed, worst 100.5%. A finding we withdraw: GENCO's 98.8% detection of voltage-limit violations on IEEE 118 becomes 52–70% against the limit-enforced truth, as does the exact no-limit solution.

### Zero-shot transfer

Zero-shot transfer gives no universal start. The organisers' pretrained model, without fine-tuning, is 2 to 9 times less accurate than DC on IEEE 118, GOC 500, PEGASE 1354 and GOC 2000. Its predicted change still helps a little on the three smaller grids (lines missed after one step: 74 → 19, 12 → 3, 8 → 5) and hurts on GOC 2000.

### Embedding space

The embedding space does not warn of unfamiliar states. Distance to the calibration set separates N-3 and N-4 states from the control with AUROC 0.68 and 0.79; counting the lines out gives 1.00. Entry #7 trains a supervised head on the embeddings for a different label, whether the solver will fail to converge, and reports AUROC 0.96; an unsupervised distance and a supervised head on different labels are not in conflict.

### A conformal band without a Newton step

Without a Newton step, a conformal band buys a stated miss rate at a price. A line is "possibly overloaded" when GENCO's prediction exceeds `limit − b·σ`, with `b` calibrated so that the share of real overloads missed targets α. IEEE 118, α = 1%: 31.9% of states go to AC, 31.2% really have an overload, and none of the 792 held-out overloads sits in a state accepted without AC. The band targets α per overloaded line reading, with `b` set on pooled calibration readings. Over 100 random splits by outage set, 9.9 / 5.0 / 1.9 / 1.0% of real overloads are not flagged themselves for α = 10 / 5 / 2 / 1%; this is empirical coverage at the target, not a guarantee. Because any flagged line sends the whole state to AC, the share of overloads inside accepted states is lower: 4.4 / 2.0 / 0.23 / 0.05%. DC tuned the same way escalates 81% of secure states. On Texas 2000 it ties GENCO (7.9% against 7.7%), so a GENCO-versus-DC gap at equal guarantee depends on how well the DC baseline is tuned. Under N-3 and N-4 outages the promise breaks (6.5% and 7.1% missed at α = 5%).  Alcántara and Chatzivasileiadis ([arXiv 2602.07995](https://arxiv.org/abs/2602.07995)) screen N-k contingencies with conformal prediction on GridFM; ours adds the Newton step and the AC-call accounting.

### Equivalent circuit representations (bounded robustness diagnostic)

On the two primary fresh-pool experiments, anchoring plus one Newton step greatly reduced GENCO's sensitivity to one-branch and non-slack-generator splitting. The median across states of the maximum line-loading difference was below 0.05 points of rating, and loading-only decisions agreed in 99.7–100% of states. Both rewrites passed our registered tolerances, although generator splitting on GOC 500 increased missed overloaded readings from four to six and produced rare large first-step errors. Splitting all eligible branches failed the criterion on all three GOC 500 checkpoints; the primary loading-only missed count rose from four to 65. Both GOC 2000 checkpoints passed, with missed counts zero→one and zero→zero. These are bounded robustness diagnostics before mismatch abstention, not representation invariance or a safety guarantee.

Each rewrite preserves the admittance matrix (to order 1e-12, within 2e-12), injections, bus types and setpoints, and every copy of an outaged split branch is disabled together. For GOC 500 Tiny seed 0, the median across states of the largest bus-voltage drift of GENCO's raw prediction was about 26 degrees and 0.14 p.u. under all-branch doubling (maxima 52 degrees and 0.47 p.u.), which is where the one-step result also breaks. Pre-registered criterion, transformations and results: `work/equiv/SPEC.md`, `work/equiv/REPORT.md`.

![Representation effects by stage](figures/equiv.png)

### Mismatch heuristic

A two-bus case in the `gridtrust` tests passes a 1 MW mismatch check with a true overload, which is why the mismatch gate is reported as a heuristic and misses are counted rather than bounded.

### What others have done, in full

Screening with one step from the pre-outage solution is classical: Albuyeh, Bose and Heath used "the first iteration of an AC power flow" ([IEEE Trans. PAS, 1982](https://doi.org/10.1109/TPAS.1982.317327)); Chen and Bose named the fast-decoupled form "1P-1Q" ([1989](https://doi.org/10.1109/59.41683), [1990](https://doi.org/10.1109/59.65923)); Stott, Alsac and Monticelli surveyed it ([Proc. IEEE, 1987](https://doi.org/10.1109/PROC.1987.13931)). Our pre-outage baseline is this method with a full Newton step. gridfm-datakit itself seeds the post-outage power flow with the pre-outage point when it generates data ([PR #67](https://github.com/gridfm/gridfm-datakit/pull/67)).

Newton's Lantern ([Bose, Hilmarsson and Suri, 2026](https://arxiv.org/abs/2605.11102)) tunes GNN warm starts by reinforcement learning and runs Newton to convergence; no contingencies, no pre-outage baseline. Li, Pandey and Pileggi ([PSCC 2024](https://arxiv.org/abs/2304.06727)) compare a learned warm start with the pre-outage start under load-attack contingencies; absolute states, Newton to convergence. Kim et al. ([October 2026](https://arxiv.org/abs/2610.05959)) take one Newton update from calibrated or subspace-projected predictions of four models, gridfm-graphkit among them, on the 2,224-bus GB network, scored by power-balance residual. Rahman and Kekatos ([2026](https://arxiv.org/abs/2609.36510)) reuse a base-case surrogate for outages through a fixed-point iteration with compensation injections; no pre-outage anchor, no differencing, no Newton step. Shooshtari (this challenge, #7) uses GENCO's state as the first Newton start with classical fallbacks, runs to convergence on every case, and adds an unresolved-case alarm; no pre-outage start, no differencing, no screening budget.

Ours starts from the pre-outage solution plus GENCO's predicted change, is scored by overloads missed and secure scenarios flagged, and is compared with the classical one-step screen.

# A Newton step or two is enough: GENCO as the first stage of a contingency screen with a stated miss rate

**In one sentence:** on the example entry's own setup (GOC 500, up to five branches removed), a single Newton step started from GENCO's prediction sends 12.5% of scenarios to the AC solver, exactly the share that really has an overload, and finds 99.7% of the overloaded lines; the same step started from DC power flow or from a flat start sends 88% and 97%. On 2,000 buses one step still does it; on 10,000 buses it takes two.

This entry builds on the organisers' example ([discussion #1](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/1)) and does the two things it leaves open: it makes GENCO's verdict safe enough to act on without checking most states, and it puts a number on the misses that remain (the example names conformal prediction as its next step).

![One Newton step from GENCO](figures/onestep.png)

## The idea

Contingency screening asks one question for thousands of grid states: *is any line overloaded?* The AC solver answers it exactly. A surrogate answers it approximately, and the open question is which of its answers can be accepted.

The workflow has three stages. Each costs more than the one before and runs on fewer states:

![Three stages](figures/workflow.png)

1. **GENCO** predicts voltages and line loadings.
2. **One Newton step** on the real network equations, started from GENCO's voltages. It costs one Jacobian solve, about a fifth of a full AC solve in the same code. Its remaining power mismatch is a free self-check.
3. **Full AC** only for states where a line is at or above its limit, or the self-check fails.

Where the Newton step is not used, a **conformal band** decides which states to escalate. It is calibrated on real overloads: a line counts as "possibly overloaded" when the prediction is above `limit − b·σ`, and `b` is chosen so that at most a share α of real overloads falls below it.

## Results

### 1. The example setup: GOC 500, N-1 to N-5

14,992 scenarios generated with gridfm-datakit (random removal of up to five branches), the released GENCO Tiny checkpoint (`gridfm/genco-pf-datakit`, trained on N-1 only, so N-2 to N-5 are zero-shot). 12.5% of scenarios have a real overload.

We first reproduce the example: with its rule (flag a scenario when any predicted loading exceeds a threshold tuned by F-beta, β = 3) GENCO reaches 93.9% recall while flagging 32.7% of scenarios; the example reports 94.3% and 32.7%.

| Surrogate, rule | Scenarios sent to AC | Secure scenarios sent to AC | Overloaded lines missed | Worst missed loading |
|---|---|---|---|---|
| GENCO, example rule (threshold 0.970 from a calibration half) | 37.6% | 29.3% | 40 of 1,327 (3.0%) | 139.6% |
| GENCO, conformal band, α = 1% | 94.9% | 94.2% | 0 of 1,327 | none |
| **GENCO + one Newton step, threshold = the limit** | **12.5%** | **0.1%** | **9 of 2,621 (0.34%)** | **100.6%** |
| GENCO + two Newton steps | 12.5% | 0.0% | 0 of 2,621 | none |
| DC + one Newton step | 88.1% | 86.7% | 50 of 2,621 | 126.2% |
| Flat start + one Newton step | 97.3% | 96.9% | 3 of 2,621 | 102.5% |

(Rows with 1,327 lines are scored on the held-out half; rows with 2,621 need no calibration and use all scenarios.)

- The step removes GENCO's error tail: the 99th percentile of the loading error falls from 6.5 to 0.06 points of rating. The lines still missed sit between 100.1% and 100.6%.
- It only works from GENCO. From DC or a flat start one step is worse than no step.
- Checkpoint seeds 1 and 42 give the same picture: 12.5% and 12.6% sent to AC, 4 and 3 lines missed.
- Counted in Newton iterations per 100 scenarios: full AC from a flat start needs 500; GENCO + one step, continuing Newton on the escalated scenarios, needs 125.

### 2. Does it hold on larger grids?

Same setup on `case2000_goc` (2,494 scenarios) and `case10000_goc` (993 scenarios), released Tiny checkpoints, no training.

![From 500 to 10,000 buses](figures/scale.png)

| Grid | Scenarios with an overload | GENCO + one step: sent to AC / secure sent / lines missed | GENCO + two steps: sent to AC / secure sent / lines missed |
|---|---|---|---|
| GOC 500 | 12.5% | 12.5% / 0.1% / 9 of 2,621 | 12.5% / 0.0% / 0 |
| GOC 2000 | 6.1% | 6.6% / 0.6% / 1 of 369 | 6.1% / 0.0% / 0 |
| GOC 10000 | 29.7% | 55.0% / 37.0% / 7 of 507 | 29.8% / 0.1% / 0 |

- **2,000 buses:** one step is enough, and it matters more than on 500 buses: raw GENCO misses 30% of the overloaded lines there, one step leaves one line at 100.2%.
- **10,000 buses:** one step is not enough. It still leaves 37% of the secure scenarios flagged; a second step brings that to 0.1% with no line missed. The released Small checkpoints give the same picture.
- **A start Newton can use.** Our plain Newton (no step control) converges from GENCO's voltages on every scenario of both grids in 3.2 iterations on average. From a flat start it fails on 128 of the 2,494 scenarios on 2,000 buses and on all 993 on 10,000 buses; from DC angles on 101 and on all 993. Production solvers add step control and do converge, so this says how good the start is, not that the cases are unsolvable.
- **Timing, indicative only.** On 10,000 buses GENCO Tiny takes 62 ms per state on the laptop GPU and 85–99 ms on CPU; batching does not help. One Newton step takes 19 ms, Newton from GENCO to convergence 57 ms. The PowerModels AC solve that generated the data took 279 ms (median, on a loaded machine, different implementation). We do not turn this into a speed claim; it is the first grid size where GENCO is not slower than the solver it feeds.

### 3. A band with a stated miss rate, when no Newton step is taken

Outage sweeps on IEEE 118 (2,375 states, GENCO Base) and Texas 2000 (3,000 states, the paper's contingency checkpoint): N-1 and random N-2 outages at five load levels, an AC-OPF dispatch per load level. Calibration and test never share an outage set.

Two things make the band cheap. **Anchoring:** take the AC loading of the base case, which an operator has anyway, and let the surrogate predict only the change. **An adaptive width:** scale the band by how far the surrogate moved the line. Every overload that GENCO alone misses is on a line the outage moved by more than one point; such readings are 2% (Texas) and 18% (IEEE 118) of all.

![What the guarantee costs](figures/cost.png)

- **IEEE 118, α = 1%:** 31.9% of states go to AC, where 31.2% really have an overload; 1.0% of the secure states are escalated needlessly; 0 of 792 held-out overloads are missed (95% upper bound 0.38%). Over 100 random splits the measured miss rate is 9.9 / 5.0 / 1.9 / 1.0% for α = 10 / 5 / 2 / 1%.
- **DC, tuned the same way** (anchored, adaptive band), escalates 81% of the secure states at α = 1% on IEEE 118. Plain DC escalates all of them.
- **Texas 2000, α = 2%:** 8.1% of states go to AC, 0 of 95 held-out overloads missed (upper bound 3.1%).
- **An honest tie:** on Texas the tuned DC baseline does as well as GENCO (7.9% against 7.7% of states sent to AC, averaged over 100 splits). The load range there is 0.8 to 1.0 and the grid stays close to linear. GENCO's lead over a carefully tuned DC appears on IEEE 118, at loads up to 1.2.

![The promised miss rate holds on average](figures/coverage.png)

### 4. Where the guarantee breaks

![Stress test](figures/stress.png)

The band is calibrated once and then used on shifted states. It holds for unseen load levels, higher reactive demand and generation moved off the OPF dispatch. It breaks for N-3 and N-4 outages (6.5% and 7.1% missed at α = 5%) and, for GENCO without anchoring, for reactive demand ×0.7 (10.3%). The misses stay marginal: at α = 5% the worst overload in a state accepted without AC is 103.3% across all shifts. Running AC on 100 states of the new distribution and recalibrating brings α = 5% back to 4.6–5.4%. α = 1% cannot be repaired that way: it needs about 100 labelled overloads.

The guarantee is an average over outage sets. A small batch can be dominated by one hard outage: in a first draw of 80 outage sets, six contained the one branch GENCO handles worst, and the control looked broken.

**Can GENCO's embedding space warn that a state is unlike calibration? Mostly no.** We took the bus embeddings at the first, middle and last correction step, pooled them, and measured the distance to the calibration states. The last-step distance separates N-3 and N-4 from the control with AUROC 0.68 and 0.79; counting the lines out gives 1.00. The embedding does react strongly to load and reactive-demand shifts, including unseen load levels where the guarantee holds, so as a gate it raises false alarms. It does not rank the states where the band fails any better than GENCO's power-balance residual (AUROC 0.53–0.65 against 0.61–0.63).

### 5. What else we learned about GENCO

- **Voltage.** DC power flow has no voltage magnitudes. On IEEE 118, 331 states violate a voltage limit while DC reports no overload at all; GENCO finds 98.8% of the violations, and every one deeper than 0.01 p.u. On Texas violations are too rare to judge, and with tightened limits GENCO finds only about half: its largest voltage errors are at the extremes.
- **Warm start.** On IEEE 118 and Texas, Newton converges in 2.0 and 2.4 iterations from GENCO (flat start 4.0 and 4.4, pre-outage solution 3.0 and 3.1).
- **Ranking remedial actions.** For 12 overloaded states of IEEE 118 we scored 1,922 generator-pair redispatch actions by the worst loading after the action. Raw GENCO ranks them worse than the standard practice of AC base case plus DC sensitivities (Kendall τ 0.81 against 0.95), as expected off its training distribution. GENCO + one Newton step reproduces the AC ranking.
- **Residual.** GENCO's power-balance residual follows the dispatch, not the outage. On Texas its rank correlation with the worst line error drops from 0.85 to 0.15 once the base case is subtracted.
- **Errors are systematic.** On Texas, 97% of the errors above 3 points occur at one load level, on the same lines in every state. Anchoring removes them (99th percentile 1.92 → 0.31 points).
- **Far out of distribution** (loads raised to 1.1–1.8 without re-dispatch) raw GENCO is worse than DC, and one Newton step no longer rescues it.

![Newton iterations](figures/newton.png)
![Voltage](figures/voltage.png)

## An operator workbench

The same workflow runs interactively on IEEE 118: pick a load level and an allowed miss rate, click lines out of service, and see which stage decided the state and why, which lines sit inside the band, the loadings after the Newton step, and a running count of AC solves and Newton iterations avoided. Every verdict is checked against the AC solution on screen. English and Chinese.

![Workbench](figures/workbench_outage.png)

On 200 random N-1 and N-2 states the staged verdict agreed with AC in all 200 at α = 1% and 2%, with 68% of states accepted on GENCO alone.

## Limits of this work

- **No speed claim.** At 118 to 2,000 buses GENCO inference on a laptop takes longer than a full AC solve (24 ms against 5.6 ms on GOC 500). The savings reported are AC solves and Newton iterations. The 10,000-bus timings are indicative and compare different implementations.
- The Newton step uses the true network model, so stage 2 measures GENCO plus physics, not GENCO alone. On 10,000 buses two steps are close to a converged solve (3.2 iterations from GENCO); what GENCO contributes there is the start.
- GOC 500 is a near-replication: the example's checkpoint and data are not published. We use the released N-1 checkpoints and our own datakit settings. Our plain DC baseline does not reproduce the example's DC numbers.
- Generator reactive limits are not enforced in any power flow here.
- Texas has few overloads (224), so its numbers are wide: single splits miss α in about 40% of cases even though the average holds.
- Overload means AC loading above 100.1%. The OPF dispatch parks lines at exactly 100%, so the definition moves the counts; the lines missed after a Newton step are all of this kind.
- The larger grids use one checkpoint seed, and 10,000 buses only 993 scenarios.

## Tools, code and data

- **GENCO** checkpoints from `gridfm/genco-pf-datakit` (IEEE 118 Base, seeds 0/1/42; GOC 500 Tiny, seeds 0/1/42; GOC 2000 and GOC 10000 Tiny and Small) and `gridfm/genco-pf-contingency` (Texas 2000). No training was done for this entry.
- **gridfm-graphkit** for the model and inference; **gridfm-datakit** for scenario generation and the AC, DC and OPF solves (PowerModels).
- Code, result tables and a `reproduce.sh`: `<REPO_URL>` (MIT).

**Author:** `<NAME>`, independent. **Contact:** `<CONTACT_EMAIL>`. Entered privately; public data only.

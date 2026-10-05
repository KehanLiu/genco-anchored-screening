# Let GENCO predict the change, not the state: one-step contingency screening, tested until it breaks

**In one sentence:** start Newton from the operator's pre-outage AC solution plus GENCO's predicted voltage *change*, take one step, and the screen misses 2 of 6,986 overloaded lines on IEEE 118, 5 of 2,621 on GOC 500 and 0 of 369 on GOC 2000, where the classic one-step screen from the pre-outage solution alone misses 74, 41 and 2; it stops helping at 10,000 buses, fine-tuning does not change that, and on a laptop it does not pay in wall-clock time.

This entry starts from the organisers' example ([discussion #1](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/1)), reproduces it, and then asks what it would take to act on GENCO's verdict without checking most states. The answer changed twice as we added the baselines a practitioner would ask for. We report all of it, including the parts where GENCO loses.

![Which start makes one Newton step enough?](figures/starts.png)

## The idea, and how we got there

Contingency screening asks one question for thousands of grid states: *is any line overloaded?*

1. **GENCO alone is not safe enough.** On the example's setup (GOC 500, up to five branches removed, released GENCO Tiny) we reproduce its result: 93.9% recall while flagging 32.7% of scenarios (the example reports 94.3% and 32.7%). The scenarios it lets through contain overloads up to 139.6%.
2. **One Newton step from GENCO's voltages fixes that**: 12.5% of scenarios go to the full solver, exactly the share that really has an overload, and 9 of 2,621 overloaded lines are missed, the worst at 100.6%. From a flat start or DC angles the same step is useless (97% and 88% of scenarios flagged).
3. **But flat and DC starts are the wrong baseline.** An operator always has the solved pre-outage state. One Newton step from *that* is classic single-iteration screening, and it does most of the job: it beats GENCO's start at 10,000 buses and ties it at 2,000.
4. **Combine them.** The pre-outage solution is exact for the base case; GENCO knows how an outage moves voltages. Start = pre-outage AC voltages + [GENCO(after outage) − GENCO(before outage)]. GENCO's systematic bias cancels, and what remains is its estimate of the change. This start is better than either on three of four grids.

![Three stages](figures/workflow.png)

## Results

### 1. One Newton step from three starts

Scenarios remove up to five branches at random (gridfm-datakit). A scenario goes to the full solver if any line loading after the step exceeds its limit. Released checkpoints; fine-tuning is tested in section 6.

| Grid (overloaded lines) | Pre-outage solution: missed / secure scenarios flagged | GENCO: missed / flagged | Pre-outage + GENCO's change: missed / flagged |
|---|---|---|---|
| IEEE 118 (6,986) | 74 / 1.9% | 4 / 0.0% | **2 / 0.0%** |
| GOC 500 (2,621) | 41 / 0.2% | 9 / 0.1% | **5 / 0.0%** |
| GOC 2000 (369) | 2 / 0.0% | 1 / 0.6% | **0 / 0.0%** |
| GOC 10000 (507) | **2 / 0.3%** | 7 / 37.0% | 5 / 3.2% |

- The combined start begins much closer to the solution: median power mismatch 6 MW on GOC 500 and GOC 2000, against 72–118 MW for the pre-outage solution and 246–435 MW for GENCO alone. Newton then converges in 2.4–2.5 iterations instead of 3.1–3.2.
- **At 10,000 buses it fails.** The released Tiny's raw error there is larger than the change an outage causes, so its predicted change carries little information (the sign of the loading change is right for 75% of moving lines, against 93–97% on the smaller grids). Two Newton steps from any of the three starts miss nothing.
- Without a Newton step the combined prediction is not usable: it misses 14–44% of overloaded lines. The step is what finds the overloads; GENCO improves where it starts.

### 2. A complete N-1 analysis, with costs

GOC 2000, 24 base cases, every non-islanding single-branch outage: 76,512 outages. Mismatch thresholds are set on 6 base cases and everything below is scored on the other 18 (57,384 outages, 1,451 of them critical).

![A complete N-1 analysis](figures/fulln1.png)

| Workflow | Sent to full AC | Critical outages missed | Newton iterations vs full AC | Seconds per 1,000 outages |
|---|---|---|---|---|
| Full AC for every outage | 100% | 0 | 100% | 11.5 |
| One step from the pre-outage solution (classic) | 29.6% | 5 of 1,451 | 57% | **6.5** |
| GENCO alone | 6.6% | 780 (54%) | 7% | 13.5 |
| One step from GENCO | 45.4% | 1 | 76% | 21.4 |
| One step from pre-outage + GENCO's change | **2.7%** | 14 (13 of them ≤ 100.14%) | **37%** | 16.9 |
| DC sensitivities on the AC base case | 2.1% | 303 (21%) | 3% | 2.3 |

- **The combined start needs the fewest Newton iterations** among workflows that miss under 1.5% of critical outages, and after one step its loading error is the smallest (99th percentile 0.006 points, against 0.032 from the pre-outage solution).
- **It is not faster.** On this laptop one GENCO inference (12.6 ms batched on the GPU) costs as much as three Newton iterations (3.9 ms each). The classic screen wins on time; the combined workflow is slower than solving every outage outright. By these unit costs inference would have to fall to about 2 ms per state to break even.
- Neither GENCO alone nor DC sensitivities alone is a usable screen: they miss 54% and 21% of critical outages.

### 3. Inside a production solver

We fed each start to PowerModels' AC power flow (the solver gridfm-datakit uses) on 200 scenarios of GOC 2000 and 100 of GOC 10000.

| Start | GOC 2000: reaches the solution / iterations / ms | GOC 10000: reaches the solution / iterations / ms |
|---|---|---|
| Pre-outage solution (datakit's default) | 199 of 200 / 3.2 / 14.0 | 92 of 100 / 3.8 / 89 |
| Flat | 20 of 200 / 7.5 / 26.1 | 0 of 100 |
| DC angles | 46 of 200 / 6.8 / 26.0 | 0 of 100 |
| GENCO Tiny | 200 of 200 / 4.3 / 17.9 | 100 of 100 / 4.7 / 111 |

GENCO's start is as reliable as the pre-outage solution and far better than flat or DC starts, which mostly end at a different solution or none. It saves no solver time against the pre-outage start, so with inference included the chain is 2 to 4 times slower. Where no solved neighbouring state exists, GENCO gives the solver a start it can use; where one exists, it is not needed for speed.

### 4. Generator reactive limits

None of the above enforces generator reactive limits, and neither does GENCO's training data. With PV-to-PQ switching added to the reference solution, nearly every scenario has switched buses, yet the set of overloaded lines barely changes (GOC 500: 2,577 → 2,570).

- The ranking of the three starts survives. Lines missed after one step on GOC 500: combined start 23, GENCO 28, pre-outage solution 54, of 2,570. The exact solution *without* limits misses 19, so that is the floor for any limit-blind method.
- "Almost nothing missed" does not survive on GOC 500: the worst missed overload is 111%. Two Newton steps that include the switching logic bring it back to 5 missed, worst 100.5%.
- **A finding we withdraw.** We had reported that GENCO finds 98.8% of voltage-limit violations on IEEE 118. Against the limit-enforced truth it finds 52–70%, and so does the exact no-limit solution. The no-limit formulation is the problem, not GENCO's fit to it. What stands, and gets stronger: 560 states violate a voltage limit while DC screening reports no overload at all.
- No independent solver with limits was available to us, so the switching logic is validated by invariants only.

### 5. A band with a stated miss rate, when no Newton step is taken

For the case where only GENCO's own output is used, a conformal band decides which states to escalate: a line counts as "possibly overloaded" when the prediction is above `limit − b·σ`, with `b` calibrated on real overloads so that at most a share α is missed. Outage sweeps on IEEE 118 (2,375 states) and Texas 2000 (3,000 states); calibration and test never share an outage set.

![What the guarantee costs](figures/cost.png)

- **IEEE 118, α = 1%** (GENCO anchored on the base case, one band width): 31.9% of states go to AC, where 31.2% really have an overload; 1.0% of secure states are escalated needlessly; 0 of 792 held-out overloads missed (95% upper bound 0.38%). Over 100 random splits the measured miss rate is 9.9 / 5.0 / 1.9 / 1.0% for α = 10 / 5 / 2 / 1%.
- **DC tuned the same way** escalates 81% of the secure states at α = 1% on IEEE 118.
- **An honest tie:** on Texas 2000, at loads of 0.8 to 1.0, tuned DC does as well as GENCO (7.9% against 7.7% of states sent to AC).
- **Where it breaks:** calibrated once and applied to shifted states, the promise holds for unseen load levels and higher reactive demand, and breaks for N-3 and N-4 outages (6.5% and 7.1% missed at α = 5%) and for reactive demand ×0.7 without anchoring (10.3%). Recalibrating on 100 states of the new distribution repairs α = 5%, not α = 1%.

![Stress test](figures/stress.png)

### 6. Does fine-tuning help?

We fine-tuned the released Tiny checkpoints on new scenarios of the same kind (up to five branches removed), generated separately and checked to share no scenario with the test set: 8,969 scenarios for GOC 500, 2,248 for GOC 2000 and 1,347 for GOC 10000; 20,000, 6,000 and 20,000 steps at learning rate 1e-4, half an hour to two hours each on a laptop GPU. Same test scenarios as above.

![Fine-tuning, before and after](figures/finetune.png)

Lines missed after one Newton step (secure scenarios flagged in brackets):

| Grid (overloaded lines) | Pre-outage solution | GENCO: released → fine-tuned | Pre-outage + GENCO's change: released → fine-tuned |
|---|---|---|---|
| GOC 500 (2,621) | 41 (0.2%) | 9 → 10 (0.1% → 0.1%) | 5 → 5 (0.0% → 0.0%) |
| GOC 2000 (369) | 2 (0.0%) | 1 → 1 (0.6% → 0.6%) | 0 → 0 (0.0% → 0.0%) |
| GOC 10000 (507) | **2 (0.3%)** | 7 → 8 (37.0% → 42.1%) | 5 → 4 (3.2% → 3.2%) |

- **No. Fine-tuning does not rescue 10,000 buses.** The combined start still loses to the pre-outage solution alone (4 missed and 3.2% of secure scenarios flagged, against 2 and 0.3%). The validation loss moved from 0.0060 to 0.0057 and the raw error did not improve (median 0.54 → 0.56 points).
- **Why the predicted change fails there.** On lines whose loading moves by more than one point, GENCO's predicted change removes 85% of the median error of assuming no change on GOC 500 (2.44 → 0.36 points), 74% on GOC 2000 (2.17 → 0.56) and 28% on GOC 10000 (2.22 → 1.60). Fine-tuning leaves all three where they were (0.29, 0.56 and 1.63).
- **Where it has room, it improves the raw model and nothing after the step.** On GOC 500 the median raw error falls from 0.33 to 0.23 points, but the raw screen stays unusable (431 overloaded lines missed, 294 before) and after one Newton step nothing changes. A second seed gives 0.26 → 0.21 points and 3 lines missed from the combined start.
- **Choosing the training scenarios does not matter either.** Adding 1,500 scenarios chosen at random, by largest mismatch after one step, or by closeness to a limit gives 9, 8 and 9 lines missed after one step from GENCO's start; training on with no new data gives 11.
- One seed on GOC 10000, the Tiny model only, and a small training set. A larger model or far more data may behave differently; we did not test that.

### 7. What else we learned about GENCO

- **Zero-shot transfer does not give a universal start.** The organisers' pretrained model, used without fine-tuning on IEEE 118, GOC 500, PEGASE 1354 and GOC 2000, is 2 to 9 times less accurate than DC and a worse Newton start than the pre-outage solution. Its predicted *change* still helps a little on the three smaller grids (lines missed after one step: 74 → 19, 12 → 3, 8 → 5) and hurts on GOC 2000.
- **The embedding space does not warn of unfamiliar states.** Distance to the calibration set separates N-3 and N-4 states from the control with AUROC 0.68 and 0.79; counting the lines out gives 1.00.
- **Ranking remedial actions.** For 1,922 redispatch actions on IEEE 118, raw GENCO ranks worse than AC base case plus DC sensitivities (Kendall τ 0.81 against 0.95); GENCO + one Newton step reproduces the AC ranking.
- **The residual follows the dispatch, not the outage.** On Texas its rank correlation with the worst line error drops from 0.85 to 0.15 once the base case is subtracted.
- **Errors are systematic.** On Texas, 97% of the errors above 3 points occur at one load level, on the same lines in every state. This is why predicting the change works better than predicting the state.
- **Far out of distribution** (loads raised to 1.1–1.8 without re-dispatch) raw GENCO is worse than DC.

## An operator workbench

The staged workflow runs interactively on IEEE 118: pick a load level and an allowed miss rate, click lines out of service, and see which stage decided the state and why. Every verdict is checked against the AC solution on screen. English and Chinese.

![Workbench](figures/workbench.gif)

## Limits of this work

- **No speed claim.** On a laptop, with inference counted, every GENCO workflow is slower than the classic one-step screen. The savings we report are Newton iterations.
- **Fine-tuning was small.** Sections 1 to 5 and 7 use released checkpoints. Section 6 fine-tunes the Tiny model for at most two hours per grid on a laptop, with one seed on the two larger grids; it shows no gain, which does not rule out a gain from a larger model or much more data.
- The Newton step uses the true network model. Stage 2 measures GENCO plus physics.
- GOC 500 is a near-replication of the example: its checkpoint and data are not published.
- The larger grids use one checkpoint seed; GOC 10000 has 993 scenarios.
- Overload means AC loading above 100.1%. The OPF dispatch parks lines at exactly 100%, so most of the remaining misses are lines between 100.1% and 100.6%.
- Timings were taken on a laptop with a system process using one core; unit costs are indicative.

## Tools, code and data

- **GENCO** checkpoints from `gridfm/genco-pf-datakit` (IEEE 118, GOC 500, GOC 2000, GOC 10000), `gridfm/genco-pf-contingency` (Texas 2000) and `gridfm/genco-pf-transfer` (pretrained model).
- **gridfm-graphkit** for the model and inference; **gridfm-datakit** for scenario generation and the AC, DC and OPF solves.
- `gridtrust`: a small standalone package (numpy, scipy, pandas only) with the Newton refinement, the conformal band and the staged decision, 32 tests. A draft contribution to gridfm-graphkit is in `upstream/`.
- Code, result tables and `reproduce.sh`: `<REPO_URL>` (MIT).

**Author:** `<NAME>`, independent. **Contact:** `<CONTACT_EMAIL>`. Entered privately; public data only.

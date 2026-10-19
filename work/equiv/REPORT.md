# Equivalent-circuit pilot: fresh-pool results and verdict

Written 10 October 2026 after the fresh runs of `work/equiv/SPEC.md` section 9 (chain `work/equiv/fresh_chain.sh`, 17:21-17:54 CEST);
corrected 11 October 2026 after an internal audit: wording of
the verdict, the failure analysis of section 4, the loading-only labelling, the eligibility statement and the summary paragraph.
No result file, threshold or SPEC text was changed. Everything below is read from `work/equiv/results/*_fresh.*` and
`work/equiv/logs/fresh*.out`; the three helper scripts in `work/equiv/analysis/` recompute the stage table, the failure analysis and
the recovery / 5 MW tables from the parquet files. Values in points of rating (1 point = 1% of rating) unless stated.

Scope labels used throughout: "loading-only" means a screening decision based on any predicted line loading > 1.001 alone, BEFORE the
mismatch abstention of the full screening policy; the pilot registered this loading-only decision, so every miss count, agreement
share and figure value in this report (and in `figures/equiv.png`) is loading-only, before mismatch abstention. "Non-slack generators"
are the units `split_gen` splits (170 on GOC 500, 237 on GOC 2000); "eligible branches" are the rated, not-always-out branches
`split_all` splits (728 of 733 on GOC 500, five always out; 3,633 on GOC 2000).

## 1. Answer and verdict

Anchoring plus one Newton step greatly reduces GENCO's sensitivity to the tested rewrites that occur in practice, on fresh pools of
both grids. For a parallel split of one branch (`split_random_1`) and a split of every non-slack generator into two units (`split_gen`)
the raw GENCO loading moves by 0.5-40 points per state (median across states of the maximum line-loading difference) while the
anchored-plus-one-step loading `hyb1` moves by at most 0.047 points by the same statistic; loading-only screening decisions agree in
99.7-100% of states, and the change in missed overloaded readings stays within the registered allowance. Both rewrites pass the
registered criterion on the checkpoints where they were run: one-branch splitting was replicated on all five grid/checkpoint
combinations (GOC 500 Tiny seeds 0 / 1 / 42, GOC 2000 Tiny and Small seed 0); generator splitting was run only on the two primary
checkpoints (GOC 500 Tiny s0, GOC 2000 Tiny s0), as registered. "Greatly reduces sensitivity" is the strongest supported statement:
the criterion is a tolerance, not invariance. In particular GOC 500 `split_gen` has a median across states of the maximum line-loading
difference of 0.047 points, a p90 of 0.693 points and a maximum of 4,505 points (one state, see section 4); loading-only decisions
disagree in 8 of 2,494 states; missed overloaded readings rise from 4 to 6, within the registered allowance of +10, which is a tolerance
and not unchanged safety.

The extreme encoding that doubles every eligible branch (`split_all`) passes on GOC 2000 (Tiny and Small: decisions agree 99.7% / 99.4%,
loading-only misses 0 -> 1 and 0 -> 0 of 322) but fails the criterion on GOC 500 for all three checkpoints (agreement 81% / 64% / 41%,
loading-only misses 4 -> 65, 3 -> 59, 3 -> 15 of 943). The pre-registered overall verdict "cancels" (all three transforms, both pools)
is therefore NOT reached, and the second branch (repair) is not triggered because the two practical encodings pass. The applicable
branch of SPEC section 8 is, verbatim:

> DOES NOT CANCEL only for `split_all`, or the effect disappears under `+refvm`: document as a limited finding (extreme
> engineered encoding / known-input issue), keep the current main story, no repair.

How the ratio is defined (correcting the earlier draft wording): for each state the maximum over rated lines of |loading_t -
loading_o| is taken at a stage; rho is the RATIO OF MEDIANS across states of that per-state maximum (stage over raw GENCO) and,
separately, the RATIO OF MEANS. It is not the median of state-wise ratios. Both ratios must be <= X = 0.05. The mean clause is what
fails `split_all` on GOC 500 seed 0: the median ratio 0.018 passes, the mean ratio 11.9 does not (and decisions agree in only 81%).

Attribution (SPEC section 7): where the criterion passes the reduction is credited to the anchored start, rho_hyb0 (ratio of medians)
0.02-0.24 <= X0 = 0.3 on every passing split transform (the start removes 76-98% of the raw difference; the Newton step removes most of
the rest). On GOC 500 `split_all` rho_hyb0 is 0.30 (seed 0) / 0.25 / 0.35: the start removes two thirds of a raw difference that is
ten times larger than anywhere else, and one Newton step from the remainder is a poor approximation in a large share of states
(section 4). `permute` passes with rho_hyb1 <= 0.0009 and raw drift 2e-6 p.u. / 0.002 deg (its rho_hyb0 of 1.1-1.3 is a ratio of two
negligible numbers, 0.01-0.03 points). The `+refvm` controls are identical to their base transform at every stage that enters the
criterion (SPEC section 3 and open issue 11.3).

## 2. Criterion table (loading-only decisions, before mismatch abstention)

Thresholds X = 0.05 (rho_hyb1 as ratio of medians AND as ratio of means of the per-state maximum line drift), Y = 0.98 (state
decision agreement), miss margin ceil(1% of overloaded readings) = +10 on GOC 500 (943 overloaded readings, 566 critical of 2,494
states) and +4 on GOC 2000 (322 of 9.06 M readings, 140 critical of 2,496 states), X0 = 0.3. "missed" counts overloaded line readings
(AC loading > 1.001) in states that the loading-only predictor does not flag (no mismatch abstention); "agree" is the share of states
whose loading-only decision is the same under o and t. Source: `results/criterion_goc{500,2000}_{tiny,small}[_seed{1,42}]_fresh.csv`;
all 20 rows were reproduced by a second-pass re-analysis.

| grid / checkpoint | transform | rho_hyb1 median | rho_hyb1 mean | rho_hyb0 | rho_g1 | agree hyb1 (loading-only) | missed hyb1 o -> t (loading-only) | prev1 missed | verdict |
|---|---|---|---|---|---|---|---|---|---|
| GOC 500 Tiny s0 | none | - | - | - | - | 100% | 4 -> 4 | 12 | identity OK (max abs t-o = 0 at every stage) |
| GOC 500 Tiny s0 | permute | 0.0007 | 0.0020 | 1.34 | 0.0020 | 100% | 4 -> 4 | 12 | control passes |
| GOC 500 Tiny s0 | split_random_1 | 0.0002 | 0.0021 | 0.158 | 0.0045 | 100% | 4 -> 4 | 12 | CANCELS, start |
| GOC 500 Tiny s0 | split_random_1+refvm | 0.0002 | 0.0021 | 0.159 | 0.0045 | 100% | 4 -> 4 | 12 | = base (raw-only control) |
| GOC 500 Tiny s0 | split_gen | 0.0012 | 0.045 | 0.239 | 0.038 | 99.7% (8 of 2,494 disagree) | 4 -> 6 (+10 allowed) | 12 | CANCELS, start (mean 0.045 close to X; state-max t-o median 0.047 / p90 0.693 / max 4,505 pts, see section 4) |
| GOC 500 Tiny s0 | split_all | 0.018 | 11.9 | 0.304 | 2.38 | 81.1% | 4 -> 65 | 12 | DOES NOT CANCEL |
| GOC 500 Tiny s0 | split_all+refvm | 0.018 | 11.9 | 0.304 | 2.38 | 81.1% | 4 -> 65 | 12 | = base, does not cancel |
| GOC 500 Tiny s1 | split_random_1 | 0.0002 | 0.0024 | 0.172 | 0.0029 | 100% | 3 -> 3 | 12 | CANCELS, start |
| GOC 500 Tiny s1 | split_all | 0.035 | 81.3 | 0.247 | 4.74 | 63.6% | 3 -> 59 | 12 | DOES NOT CANCEL |
| GOC 500 Tiny s42 | split_random_1 | 0.0002 | 0.0015 | 0.187 | 0.0039 | 100% | 3 -> 3 | 12 | CANCELS, start |
| GOC 500 Tiny s42 | split_all | 0.231 | 25,562 | 0.350 | 183 | 40.9% | 3 -> 15 | 12 | DOES NOT CANCEL |
| GOC 2000 Tiny s0 | none | - | - | - | - | 100% | 0 -> 0 | 3 | identity OK |
| GOC 2000 Tiny s0 | permute | 0.0009 | 0.0024 | 1.09 | 0.0060 | 100% | 0 -> 0 | 3 | control passes |
| GOC 2000 Tiny s0 | split_random_1 | 1.3e-5 | 0.0003 | 0.019 | 0.0057 | 100% | 0 -> 0 | 3 | CANCELS, start |
| GOC 2000 Tiny s0 | split_random_1+refvm | 1.3e-5 | 0.0003 | 0.019 | 0.0057 | 100% | 0 -> 0 | 3 | = base |
| GOC 2000 Tiny s0 | split_gen | 0.0003 | 0.0047 | 0.102 | 0.056 | 100% | 0 -> 0 | 3 | CANCELS, start |
| GOC 2000 Tiny s0 | split_all | 0.0005 | 0.0032 | 0.090 | 0.039 | 99.7% | 0 -> 1 | 3 | CANCELS, start |
| GOC 2000 Tiny s0 | split_all+refvm | 0.0005 | 0.0032 | 0.090 | 0.039 | 99.7% | 0 -> 1 | 3 | = base |
| GOC 2000 Small s0 | split_random_1 | 1.1e-5 | 0.0003 | 0.028 | 0.0050 | 100% | 0 -> 0 | 3 | CANCELS, start |
| GOC 2000 Small s0 | split_all | 0.0005 | 0.017 | 0.098 | 0.058 | 99.4% | 0 -> 0 | 3 | CANCELS, start |

Accuracy context (summary CSVs, original encoding): `hyb1_o` loading error p99 0.029 points on GOC 500 and 0.019 on GOC 2000;
`prev1` 0.37 / 0.083; raw `genco_o` 7.5 / 11.9 with 83 / 136 missed readings (loading-only). Under `split_random_1` and `split_gen` the
`hyb1_t` p99 stays at 0.029-0.155 (GOC 500) and 0.019-0.026 (GOC 2000); under `split_all` it is 23.5 on GOC 500 and 0.042 on GOC 2000.

## 3. Stage table, primary checkpoints

Raw drift = max over buses of abs(t - o) of GENCO's predicted voltages on the outaged graph; stage columns = max over rated
lines of abs(loading_t - loading_o) per state; all as median / p90 / max over states. `split_random_1` draws one branch per
load scenario (the drawn branch was itself outaged in 12 GOC 500 and 3 GOC 2000 states; kept). Source: `analysis/stage_table.py`.

GOC 500 Tiny seed 0 (2,494 states, 499 load scenarios 6000-6499, 1.81 M rated readings)

| transform | Vm drift p.u. | Va drift deg | genco | dgenco | hyb0 | hyb1 | g1 |
|---|---|---|---|---|---|---|---|
| none | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| permute | 2.2e-6 / 4.1e-6 / 5.4e-5 | 0.0023 / 0.0049 / 0.015 | 0.025 / 0.048 / 0.15 | 0.033 / 0.064 / 0.20 | 0.033 / 0.063 / 0.20 | 1.8e-5 / 1.2e-4 / 0.0029 | 5.1e-5 / 1.8e-4 / 0.0036 |
| split_random_1 | 2.3e-4 / 9.2e-4 / 0.021 | 0.24 / 1.23 / 9.08 | 1.23 / 5.97 / 52.8 | 0.25 / 1.92 / 46.7 | 0.19 / 1.56 / 53.7 | 2.6e-4 / 0.0067 / 2.29 | 0.0055 / 0.041 / 2.32 |
| split_gen | 0.0058 / 0.013 / 0.42 | 18.5 / 22.1 / 27.3 | 40.1 / 56.8 / 825 | 12.2 / 35.9 / 691 | 9.58 / 31.6 / 1,140 | 0.047 / 0.69 / 4,510 | 1.54 / 2.54 / 16,600 |
| split_all | 0.143 / 0.418 / 0.474 | 25.8 / 34.8 / 52.0 | 474 / 1,520 / 6,390 | 127 / 376 / 5,390 | 144 / 459 / 7,480 | 8.6 / 260 / 9.6e6 | 1,130 / 1.4e5 / 6.6e9 |
| split_all+refvm | as split_all | as split_all | 474 / 1,520 / 6,390 | 127 / 376 / 5,390 | as split_all | as split_all | as split_all |
| split_random_1+refvm | 2.1e-4 / 7.9e-4 / 0.0079 | as base | 1.22 / 5.98 / 52.8 | as base | as base | as base | as base |

GOC 2000 Tiny seed 0 (2,496 states, 500 load scenarios 1500-1999, 9.06 M rated readings)

| transform | Vm drift p.u. | Va drift deg | genco | dgenco | hyb0 | hyb1 | g1 |
|---|---|---|---|---|---|---|---|
| none | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| permute | 2.6e-6 / 3.8e-6 / 9.8e-6 | 3.7e-4 / 6.5e-4 / 0.0018 | 0.010 / 0.015 / 0.029 | 0.011 / 0.017 / 0.030 | 0.011 / 0.017 / 0.030 | 8.9e-6 / 5.1e-5 / 0.0029 | 6.3e-5 / 1.3e-4 / 0.0031 |
| split_random_1 | 2.6e-4 / 9.9e-4 / 0.0043 | 0.036 / 0.18 / 3.29 | 0.47 / 1.89 / 21.5 | 0.0094 / 0.12 / 5.49 | 0.0089 / 0.080 / 6.12 | 6.0e-6 / 1.3e-4 / 0.12 | 0.0027 / 0.015 / 0.29 |
| split_gen | 0.0057 / 0.0088 / 0.040 | 14.5 / 22.3 / 26.9 | 25.9 / 35.4 / 175 | 3.82 / 13.5 / 160 | 2.63 / 10.6 / 160 | 0.0082 / 0.19 / 19.2 | 1.45 / 3.13 / 14.4 |
| split_all | 0.034 / 0.112 / 0.148 | 10.1 / 12.9 / 15.3 | 91.1 / 333 / 1,020 | 10.2 / 27.2 / 1,010 | 8.18 / 24.0 / 1,170 | 0.045 / 0.39 / 186 | 3.59 / 70 / 211 |
| +refvm variants | as base | as base | as base | as base | as base | as base | as base |

Replication checkpoints (same columns, `split_all` only): GOC 500 Tiny s1 drift 0.129 p.u. / 45.2 deg, hyb1 36.4 / 1,850 / 1.3e8;
Tiny s42 drift 0.264 p.u. / 48.5 deg, hyb1 421 / 51,100 / 8.1e10; GOC 2000 Small s0 drift 0.022 p.u. / 15.6 deg, hyb1 0.045 / 0.65 / 783.
Their `split_random_1` rows match the primary ones (hyb1 median <= 2.3e-4 points).

Reading: the raw prediction is strongly representation dependent for every non-trivial rewrite, including the physically ordinary
non-slack-generator split (15-19 degrees of angle drift, 26-40 points of loading). The anchored start removes most of it (hyb0 is 4-50
times smaller than genco) and the one Newton step removes most of the rest when the anchored start is close to the solution; where the
start is far off (GOC 500 `split_all`, and a few `split_gen` states) one step is a poor approximation, see section 4.

## 4. The GOC 500 `split_all` failure: measured first-step and recovery behaviour

`split_all` is an extreme engineered encoding: every eligible branch (728 of 733 on GOC 500, five always out; 3,633 on GOC 2000) is
replaced by two parallel copies with doubled impedance and halved rating, so the graph GENCO sees has twice the branch count of anything
it was trained on. The same transform passes on GOC 2000 with both checkpoints (section 2); the failure is specific to GOC 500, where the
raw drift is 2.5-5 times larger (25.8 / 45.2 / 48.5 degrees and 0.14 / 0.13 / 0.26 p.u. for seeds 0 / 1 / 42, against 10.1 degrees
and 0.034 p.u. on GOC 2000 Tiny) and, decisively, the drift of the PREDICTED CHANGE (outage minus intact) is 7.3 / 11.4 / 15.6
degrees median against 0.49 degrees on GOC 2000. Sources: `analysis/split_all_failure.py` and `analysis/recovery_and_gate.py`
(states parquet columns `conv_hyb_t`, `it_hyb_t`, `dv_hyb_t` = max over buses of the complex-voltage distance between the converged
Newton solution and the stored AC solution, `mis0_*`, `mis1_*`, `dv_start_hyb_*`; loadings from the lines parquet).

Because the circuit equations are proven identical for both representations (section 5) and Newton iterates the same original
problem, every difference between the `t` and `o` columns is a difference of INITIALIZATION: the anchored start under `t` carries a
different predicted change. Four distinct outcomes of that poorer initialization are measured below and must be kept apart: (a) a
poorer anchored start, (b) a large error of the one-step result, (c) exhaustion of the 30-iteration Newton budget, and (d)
convergence to a power-flow solution other than the stored one. An earlier draft of this section called (b) "leaving the Newton
basin"; the stored convergence results contradict that reading and it is withdrawn.

(a) Poorer anchored start. Mismatch of the start (max-norm, MW, median / p90 / max over states), seed 0: raw GENCO voltages o 19.6 /
45.9 / 238, t 6,970 / 11,700 / 34,800; anchored start `hyb0` o 7.1 / 33.6 / 458, t 1,185 / 7,150 / 41,600; for comparison the
pre-outage AC start has 129 / 279 / 869. The anchored start is 0.138 p.u. (median, max over buses) from the AC solution under t against
0.014 under o. Anchoring still removes 83% of the raw start mismatch (6,970 -> 1,185 MW), but what remains is 170 times the original
and nine times the classical start.

(b) Large first-step errors. After one step `hyb1` has mismatch o 0.054 / 0.67 / 57 MW, t 44.8 / 2,130 / 9.7e7 MW (`prev1`: 2.2 / 11 /
73). 359 [896 / 1,745] states (seed 0 [1 / 42]) have a `hyb1_t` maximum loading error above 100 points (o: 0) and 1,142 [1,812 /
2,329] above 10 points (o: 2). These states produce the FALSE FLAGS: 21.5% [44.7% / 75.8%] of secure states are flagged by `hyb1_t`
against 0.05% by `hyb1_o`, and of the 472 [909 / 1,475] states whose loading-only decision changes, 416 [862 / 1,463] are t-only flags.
They also drive the mean ratio (11.9 [81 / 25,562]) that the median (0.018 [0.035 / 0.23]) hides, which is what the mean clause of
the criterion was written for.

(c) and (d) Recovery when Newton is allowed to continue from the same anchored start (30-iteration budget), measured on the stored
states; "stored solution" means complex-voltage distance `dv_hyb_t` <= 1e-5 p.u. from the stored AC solution, "different solution" means
the residual-based convergence flag passes but `dv_hyb_t` > 1e-5 p.u.:

| GOC 500 `split_all`, anchored start t | first-step error > 100 points | of these: converge to the stored solution | converge to a different solution | exhaust the 30-step budget |
|---|---|---|---|---|
| Tiny seed 0 | 359 | 236 | 2 | 121 |
| Tiny seed 1 | 896 | 656 | 4 | 236 |
| Tiny seed 42 | 1,745 | 780 | 26 | 939 |

Over all states: 2,373 / 2,258 / 1,554 of 2,494 converge (o: 2,494 in every run); 121 / 236 / 940 exhaust the budget (seed 42 has
one budget failure outside the >100-point subset); 2 / 4 / 26 converge to a different solution with complex-voltage distance up to
1.08 / 2.00 / 1.33 p.u. from the stored one although the residual flag passes. Convergence and agreement with the intended operating
solution are therefore different checks. The 4,505-point `split_gen` state (GOC 500 s0, state 1756, load 6351) is the clearest
case: its one-step result is off by about 4,505 points, yet Newton from the same start converges to the stored solution in 8
iterations with complex-voltage distance 2.98e-14 p.u. (start mismatch 26,845 MW, after one step 284,631 MW). A large first-step
error is a statement about the one-step approximation, not about the attraction basin of the stored solution.

Full Newton behaviour under the transformed encoding, GOC 500 Tiny seed 0 (o = original, t = transformed):

| start | converged o | converged t | iterations to convergence o mean / p90 / max | t mean / p90 / max | t states > 5 it | t hit the 30-it cap |
|---|---|---|---|---|---|---|
| anchored `hyb` | 2494/2494 | 2373/2494 | 2.47 / 3 / 6 | 4.25 / 5 / 15 | 266 | 121 |
| raw GENCO `g` | 2494/2494 | 1284/2494 | 3.01 / 3 / 6 | 5.25 / 7 / 10 | 1595 | 1210 |
| pre-outage AC `prev` | 2494/2494 | - | 3.28 / 4 / 7 | - | - | - |

Seeds 1 / 42: anchored start converges 2258 / 1554 of 2494 (mean 4.7 / 5.3 iterations), raw start 1006 / 0 of 2494.

Where the loading-only misses sit (seed 0; seeds 1 / 42 in brackets):
- The 65 [59 / 15] MISSED readings (in 58 [48 / 12] states) do NOT come from the states with large first-step errors. In every missed
  state Newton from the same anchored start converges to the stored solution (58/58, 48/48, 12/12; complex-voltage distance <= 1.3e-8
  p.u.) in 4-5 iterations; the one-step maximum loading error there is 11.6 / 29.8 / 47.5 points (median / p90 / max; none above 100,
  37 of 58 above 10) and the mismatch after one step 62 / 277 / 953 MW (o: 0.05 MW). The missed readings have AC loadings of 100-110%,
  are predicted at 99.3% (median) by `hyb1_t`, and are caught by `hyb1_o` (100 / 102 / 109%): a start ten times further away leaves a
  residual of 10-50 points after one step, which is enough to pull a line just over its limit below it. The 4 original misses sit within
  0.1 point of the limit (AC 100-101%, predicted 99.9-100%).
- The raw-plus-one-step `g1` fails likewise and worse: 1,780 [2,493 / 2,494] states above 100 points, 78% [99.9% / 100%] of
  secure states flagged (seeds 1 and 42 flag everything, hence "0 misses"), 15 misses on seed 0 against 6 for `g1_o`.
- Contrast GOC 2000 Tiny `split_all`: the anchored start converges in all 2,496 states (mean 2.78 iterations vs 2.46 for o), the
  start mismatch is 31.8 MW median (o 6.2), 5 states exceed 100 points (o: 2), 0.3% of secure states are flagged, 7 decisions change
  (6 t-only, 1 o-only) and the one extra miss is a 100.0% reading predicted at 99.6%. GOC 2000 Small: 2496/2496 converge, 12 states
  above 100 points, 16 t-only flags (0.6%), misses 0 -> 0.

In short: on GOC 500 the all-branch doubling moves GENCO's predicted change so far (7-16 degrees median) that the anchored start is a
much poorer initialization than under the original encoding; one Newton step from it is then a poor approximation in 14-70% of the
states (false flags, blown-up mean ratio), and in a further 2% of states the one-step residual of 10-50 points hides a line within
10-50 points of its limit (the misses). Most of the badly started states still recover to the stored solution when Newton continues,
some exhaust the budget and a few converge to a different power-flow solution. Since the equations are unchanged, this is attributed to
initialization. It is a limitation of the anchored workflow, namely incomplete cancellation of a representation-sensitive predicted
change, not a defect in the anchoring code. Large first-step errors alone do not establish departure from the stored solution's Newton
basin: the measured categories above (recovery to the stored solution in 236 / 656 / 780 states, convergence elsewhere in 2 / 4 / 26,
budget exhausted in 121 / 236 / 939 for seeds 0 / 1 / 42) are what the data support, and the budget-exhausted cases do not establish
their eventual destination. The classical `prev1` and the original encoding are untouched by construction (12 and 4 misses).

Exploratory sensitivity, NOT the registered criterion: the full screening policy of the submission abstains ("refers") a state whose
post-step mismatch exceeds a threshold. Rescoring the stored `split_all` states with the loading flag OR a post-step mismatch
`mis1_hyb_r` above 5 MW (or non-finite), reproduced from the stored `mis1_hyb_o` / `mis1_hyb_t` columns by
`analysis/recovery_and_gate.py` and matching the internal audit's table exactly:

| `split_all`, 5 MW gate | original: missed readings / secure states referred | transformed: missed readings / secure states referred |
|---|---|---|
| GOC 500 Tiny s0 | 2 / 10 | 1 / 1,839 |
| GOC 500 Tiny s1 | 1 / 6 | 1 / 1,844 |
| GOC 500 Tiny s42 | 2 / 8 | 0 / 1,927 |
| GOC 2000 Tiny s0 | 0 / 18 | 0 / 73 |
| GOC 2000 Small s0 | 0 / 14 | 0 / 104 |

The 5 MW threshold is the old GOC 2000 N-1 threshold transferred unchanged, not a threshold validated for GOC 500. The table explains
the safety/referral trade-off of the full policy: the mismatch gate catches most of the loading-only misses at the cost of referring
nearly every GOC 500 state (1,839-1,927 of 1,928 secure states) under the transformed encoding. It does not rescue representation
robustness, does not make the transformation harmless, and does not replace the registered loading-only criterion; the registered
verdict stands unchanged.

## 5. Physics checks

All 20 fresh runs (logs `logs/fresh*.out`, per-state columns `eq_*`, `outage_map_ok`, `agg_check_*`):
- Equivalence maxima, contingency and intact graphs: Ybus (mapped to physical labels) 0 for `none`, `split_gen`; 4.5e-13 / 4.7e-13
  for `split_random_1`; 6.8e-13 / 1.4e-12 for `permute`; 1.1e-12 (GOC 500) / 1.6e-12 (GOC 2000) for `split_all`. Specified
  injections (S_spec, Pd, Qd), bus types, PV/REF setpoints, reference angle, Q limits, shunts, summed ratings per physical branch
  and the outaged set: 0 everywhere. Aggregated generator P / Q per bus <= 1.7e-13 / 5.7e-14. The non-zero maxima are of order
  1e-12 (all within 2e-12), far below the 1e-9 abort level.
- `gen_n_on` differs by up to 6 units per bus under `split_gen`, by construction (reported, not asserted; SPEC 11.1); 0 elsewhere.
- Outage mapping (rewriting the outaged tables equals removing every copy from the rewritten intact grid): ok in 2494/2494 (GOC 500)
  and 2496/2496 (GOC 2000) states of every run.
- Engine-flow aggregation vs physics loading (the engine's per-copy flows summed to the physical branch against flows through the
  original admittances at the same voltages): max 6.2e-5 to 1.3e-4 (float32 level), n/a for `+refvm` by design.
- Identity control `none`: max abs(t - o) = 0 at every stage on both grids. `permute`: raw drift 2e-6 p.u. / 0.002 deg (GOC 500),
  3e-6 p.u. / 0.0004 deg (GOC 2000), below the 1e-3 / 0.05 pipeline limits.
- REF bus: GENCO's raw REF Vm error is 0.019 p.u. median on both grids (setpoints 1.0295 / 1.0740); its raw REF Va error is 0 (unmasked input).
- SPEC ABORT rule (section 7 of the SPEC): since 11 October `run_equiv.py` raises `PipelineAbort` on a false `outage_map_ok`, on any
  equivalence value above 1e-9 or non-finite, and on an identity control `none` that does not reproduce o = t. Where each check runs:
  outage mapping and per-state equivalence in the worker, before the state's row exists; every stored equivalence cell again in `run()`
  after all workers return and before the parquet files are written, together with the identity control for transform `none`
  (finite / non-finite patterns of every t column against its o column first, then max |t-o| < 1e-3 points); and the same cell-level and
  identity checks in `report()` on the stored parquet files, before the summary / criterion CSVs are written. Every check validates the
  cells before any maximum is taken (pandas maxima skip NaN; an audit probe showed that one NaN equivalence cell or one NaN
  `none`-control reading with a finite counterpart previously passed through the aggregated maxima). A non-finite (o, t) pair in the
  identity control also aborts: the control must consist of finite, equal readings. `__main__` maps `PipelineAbort`, and the
  `AssertionError` that `transforms.verify_equivalence` raises in a worker on a physics mismatch, to exit code 2 with a
  "PIPELINE BROKEN - ABORT" line; no summary / criterion CSV is written on either path. Previously `outage_map_ok` was recorded without
  an assertion and the identity failure only printed "PIPELINE BROKEN". The stored fresh runs pass every control (an internal audit
  found finite equivalence values in all 20 state files and zero finite-mask mismatches or non-finite pairs in both `none` controls), so
  the stored results are unaffected. `test_abort.py` (15 tests) covers the abort paths on synthetic data, including stored-frame
  regressions for the two NaN cases and the copied-CLI exit code; a 2-load-group smoke run of `none` on the fresh GOC 500 pool confirmed
  the normal path (identity 0.00e+00 with matching finite patterns, CSVs written; outputs deleted).

Eligibility of the GOC 500 fresh pool: load scenario ids 6000-6499 were requested; the pool holds 499 load groups and 2,494 states
because load id 6353 failed OPF generation (`OPF did not converge: LOCALLY_INFEASIBLE`, recorded in
`work/fresh/data500/case500_goc/raw/error.log`, the error log named by `work/fresh/logs/datagen500.log`), and the same error log
records a PF failure at scenario 6201 (`PF did not converge`). Eligibility is therefore "successfully generated cases"; no state was
removed after inspection, and both representations are scored on exactly the same retained states, so every paired o-vs-t comparison
in this report uses identical state sets.

## 6. Limitations

- Two grids (GOC 500, GOC 2000), released checkpoints only (Tiny seeds 0 / 1 / 42, Small seed 0); no 10,000-bus run. Generator
  splitting was run on the two primary checkpoints only; one-branch splitting on all five; all-branch splitting on all five.
- One transform family per type: a 2-way parallel split (one eligible branch or all), a 2-way non-slack-generator split, a
  relabelling. No series splits, no bus splitting, no tap or shunt re-expression, no mixed rewrites.
- All miss counts, agreement shares and the plotted values of `figures/equiv.png` are loading-only, before mismatch abstention; the
  registered criterion is a descriptive tolerance (ratio of medians and of means <= 0.05, agreement >= 0.98, misses within ceil(1%)),
  not representation invariance and not a statistical safety guarantee. States within a load group are strongly dependent.
- `+refvm` is a raw-only control: every Newton start of this pipeline already pins the REF Vm / Va and PV Vm to the setpoints, so
  `g1`, `hyb0`, `hyb1` are identical with and without it; a known-REF-Vm adapter would need re-inference with a changed mask, which
  the released checkpoints do not support. The `+refvm` rows therefore say nothing about a known-input repair.
- The GOC 2000 fresh pool (ids 1500-1999) was used once by the fresh confirmation run (`work/fresh/`) for the frozen-policy question; it is fresh for the representation
  question but not untouched. The GOC 500 pool (ids 6000-6499 minus the failed id 6353) was never inspected before this run.
- No repair was tested (canonical merging of parallel copies, aggregation of co-located generators); the continue-or-stop rule did
  not call for one.
- `split_gen` on GOC 500 passes with a mean ratio of 0.045 against X = 0.05, carried by one state whose one-step result is off by
  4,505 points (state 1756; Newton from the same start converges to the stored solution in 8 iterations); three states exceed 100
  points of first-step error; misses 4 -> 6 within the +10 allowance; decisions disagree in 8 of 2,494 states. A reader may regard the
  mean ratio as borderline; the tail is disclosed in sections 1 and 4.
- Newton runs on the original problem for both representations (the equations are proven identical per state); the transformed
  problem is never iterated itself. Reactive limits are not enforced in this Newton experiment.

## 7. Summary paragraph

> **Equivalent circuit representations.** On the two primary fresh-pool experiments, anchoring plus one Newton step greatly reduced GENCO's sensitivity to one-branch and non-slack-generator splitting. The median across states of the maximum line-loading difference was below 0.05 points of rating, and loading-only decisions agreed in 99.7–100% of states. Both rewrites passed our registered tolerances, although generator splitting on GOC 500 increased missed overloaded readings from four to six and produced rare large first-step errors. Splitting all eligible branches failed the criterion on all three GOC 500 checkpoints; the primary loading-only missed count rose from four to 65. Both GOC 2000 checkpoints passed, with missed counts zero→one and zero→zero. These are bounded robustness diagnostics before mismatch abstention, not representation invariance or a safety guarantee.

## 8. Reproduction

Reports (rewrite `results/summary_*` and `results/criterion_*` with identical content and print the predictor, t-vs-o, rho and
CRITERION lines; the console output of the original report runs was not captured to a file; the report now exits with code 2 and
writes nothing if a stored run violates the SPEC ABORT rule):

```sh
cd <repo>
.venv/bin/python work/equiv/run_equiv.py report --grid 500  --variant tiny  --seed 0  --tag _fresh   # defaults --x 0.05 --y 0.98 --miss-margin 0.01 --x0 0.3
.venv/bin/python work/equiv/run_equiv.py report --grid 500  --variant tiny  --seed 1  --tag _fresh
.venv/bin/python work/equiv/run_equiv.py report --grid 500  --variant tiny  --seed 42 --tag _fresh
.venv/bin/python work/equiv/run_equiv.py report --grid 2000 --variant tiny  --seed 0  --tag _fresh
.venv/bin/python work/equiv/run_equiv.py report --grid 2000 --variant small --seed 0  --tag _fresh
# section 3 and section 4 tables (read-only)
.venv/bin/python work/equiv/analysis/stage_table.py
.venv/bin/python work/equiv/analysis/split_all_failure.py
.venv/bin/python work/equiv/analysis/recovery_and_gate.py     # recovery table, split_gen state 1756, 5 MW rescore
.venv/bin/python -m pytest -q work/equiv/test_transforms.py work/equiv/test_abort.py
```

The runs themselves: `work/equiv/fresh_chain.sh` (SPEC section 9 with `--procs 4`; 47-226 s wall per run).

Files:
- `work/equiv/SPEC.md` (pre-registration, frozen), `run_equiv.py`, `transforms.py`, `test_transforms.py`, `test_abort.py`, `fresh_chain.sh`
- `work/equiv/results/criterion_goc500_tiny_fresh.csv`, `criterion_goc500_tiny_seed1_fresh.csv`, `criterion_goc500_tiny_seed42_fresh.csv`,
  `criterion_goc2000_tiny_fresh.csv`, `criterion_goc2000_small_fresh.csv` and the matching `summary_*_fresh.csv`
- `work/equiv/results/goc<N>_<variant>_seed<S>_<transform>_fresh_{states,lines}.parquet` for the 20 runs listed in section 2
- `work/equiv/logs/fresh500_*.out`, `fresh2000_*.out` (run logs with the equivalence maxima), `fresh_chain.log`
- `work/equiv/analysis/stage_table.py`, `split_all_failure.py`, `recovery_and_gate.py` (this report's helpers)
- Development probe, not evidence: `results/*_dev.*`, `logs/dev_report.txt`

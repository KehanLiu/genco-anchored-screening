# results/: the summary tables behind the registered numbers of the write-up

`scripts/headline_numbers.py` is a registered numerical regression check. Each of its 274 claims is registered
with the value that `submission/POST.md` stated at the time of registration and is recomputed, or read, from the
files in this directory; the script writes `headline_numbers.csv` (one row per claim: the registered value, the
value from the table, the file and the formula). It does not parse the post: a number edited in the post without
re-registering it in the script is not caught. `scripts/headline_numbers.py --check` exits non-zero when a table
value differs from its registered write-up value, when a claim is MISSING or a section FAILED, or when a value
differs from the committed `headline_numbers_expected.csv`, and prints which of the three failed;
`.github/workflows/ci.yml` runs it on every push, followed by `tests/test_headline_numbers.py`, which checks
that each of those failures is caught.

## Coverage

The 274 claims read these sections of the post (counts by claim prefix in `headline_numbers.csv`): the summary
table and "The idea" (the organisers' example rule, 2 claims), "Main result" with its appendix "the mechanism in
detail" (`main_`, `nostep_`, `mech_`, `mismatch_`, `iterations_`, `pm_`: 65), "A complete N-1 analysis" and the
mismatch bullet of "Limits of this work" (`n1_`: 42), "Fresh operating points", "Measured end-to-end timing",
"Where it could win" and "Fresh pool, state by state" (`fresh_`, `freshws_`: 74), "Where it stops working" and
"10,000 buses, every released checkpoint" (`ckpt_`: 33), "Fine-tuning" (`ft_`: 13), "Reactive limits" (`qlim_`:
16), "Zero-shot transfer" (`zs_`: 8), "Embedding space" (`emb_`: 4) and "A conformal band without a Newton step"
(`conf_`: 17).

Sections of the post that no claim covers:

- **Equivalent circuit representations (bounded robustness diagnostic)**: all of its numbers come from
  `work/equiv/REPORT.md` and `work/equiv/results/*.csv`, which are not copied here.
- **Extending GENCO: the anchored start as a toolkit option**: the "41 tests pass" count is checked by the
  gridtrust test step of CI, not by the number check.
- **Mismatch heuristic**: the 1 MW two-bus case is a `gridtrust` test, not a result table.
- **Open science and reproducibility**: the figure count (17) and the claim count (274) are facts about the
  repository, not about `results/`.
- **What the organisers envisioned**: quotes; its "about three Newton iterations" is the ratio of two registered
  unit costs (12.6 / 3.9) and is not registered itself.
- **An operator workbench**, **What others have done**, **What others have done, in full**: no numbers of ours
  (entry #7's AUROC 0.96 and the organisers' 94.3% are theirs).

Numbers inside covered sections that are not registered: the 139.6% worst overload GENCO lets through on GOC 500
(summary table, "The idea"; the value is in `hybrid_start/hybrid_control_summary.csv` but no claim reads it); the
12.5%, 97% and 88% of scenarios flagged after one step from GENCO, a flat start and DC angles, and the 97% of
Texas 2000 errors at one load level ("The idea"; per-case tables under `work/`, not here); the prose ratios "one
full AC solve per 37 outages against one per 3", "about 270 against about 2,960", "about 7 times", "about three
Newton iterations" and "2 to 4 times slower" (derived in the text from registered numbers, or from
`timing/chain_*.csv`, which no claim reads); "between 100.1% and 100.6%" ("Limits of this work"); and "200 timed
outages" per base (4,800 / 24).

## Files

The files are copies of the summary tables that the analysis scripts write under `work/` (which `reproduce.sh
analysis` regenerates from the saved per-case tables; those per-case tables are parquet and not in git). Each
file is small; nothing here is a per-line or per-state table. After re-running an analysis stage, copy the
new table over its copy here, run `scripts/headline_numbers.py --write-expected`, re-register in the script any
write-up value that changed, and review the diff of `headline_numbers.csv` before committing.

| File | Copied from | Written by |
|---|---|---|
| `hybrid_start/hybrid_control_summary.csv` | `work/hybrid_control_summary.csv` | `analysis/analysis_hybrid_start.py report` (Tiny seed 0 on GOC 500, 2000, 10000: the main-result table, the no-step rows) |
| `hybrid_start/hybrid_control_summary_tiny_seed1.csv` | `work/hybrid_control_summary_tiny_seed1.csv` | `analysis/analysis_hybrid_start.py report --variant tiny --seed 1` (GOC 10000) |
| `hybrid_start/hybrid_control_summary_small_seed0.csv` | `work/hybrid_control_summary_small_seed0.csv` | `analysis/analysis_hybrid_start.py report --variant small --seed 0` (GOC 10000) |
| `hybrid_start/hybrid_control_summary_small_seed1.csv` | `work/hybrid_control_summary_small_seed1.csv` | `analysis/analysis_hybrid_start.py report --variant small --seed 1` (GOC 10000) |
| `zeroshot/summary.csv` | `work/zeroshot/summary.csv` | `analysis/analysis_zeroshot.py report` (zero-shot study; its `case118_ieee` rows `prev1`, `pg1`, `hybpg1` are the IEEE 118 row of the main-result table) |
| `finetune/summary_500.csv`, `summary_2000.csv`, `summary_10000.csv` | `work/finetune/results/` | `analysis/analysis_finetune.py --grid {500,2000,10000}` (raw error before and after fine-tuning) |
| `finetune/summary_hybrid_500.csv`, `summary_hybrid_2000.csv`, `summary_hybrid_10000.csv` | `work/finetune/results/` | `analysis/analysis_finetune_hybrid.py` (one Newton step from each start, released and fine-tuned; also the start mismatch, iteration and sign-agreement columns of the mechanism section) |
| `fulln1/summary.csv` | `work/fulln1/summary.csv` | `analysis/analysis_full_n1.py report --grid 2000` (the GOC 2000 N-1 study: workflow rows, tau sensitivity, one-step accuracy) |
| `fulln1/timing_2000.json` | `work/fulln1/timing_2000.json` | `analysis/analysis_full_n1.py timing --grid 2000` (unit costs of the N-1 study) |
| `fulln1/timing_2000_fresh.json` | `work/fulln1/timing_2000_fresh.json` | `analysis/analysis_full_n1.py timing --grid 2000 --tag _fresh` (quiet-machine unit costs of the fresh pool, `work/fresh/quiet_chain.sh`) |
| `fulln1/e2e_2000_fresh.json` | `work/fulln1/e2e_2000_fresh.json` | `analysis/analysis_full_n1.py e2e --grid 2000 --tag _fresh --policy work/fresh/policy_frozen.json` (measured end-to-end timing sample, 4,800 outages) |
| `fresh/results_n1.csv`, `results_n1_per_base.csv` | `work/fresh/` | `work/fresh/score_fresh_n1.py` (fresh N-1 pool, pooled and per base) |
| `fresh/results_hybrid.csv` | `work/fresh/results_hybrid.csv` | `work/fresh/score_fresh_hybrid.py --policy work/fresh/policy_frozen.json --tag _fresh` (warm starts on the fresh pool, loading-only and full policy) |
| `fresh/policy_frozen.json` | `work/fresh/policy_frozen.json` | hand-written before the pool was generated (thresholds, cushions, shrinkage, gate) |
| `timing/summary.csv` | `work/timing/summary.csv` | `analysis/analysis_solver_timing.py summary` (starts inside PowerModels' AC power flow) |
| `timing/chain_case2000_goc.csv`, `chain_case10000_goc.csv` | `work/timing/` | `analysis/analysis_solver_timing.py chain` (graph build, forward, solve per state) |
| `timing/goc10000_timing.csv` | `work/goc10000/timing.csv` | `work/goc10000/04_timing.py` (Tiny inference cost on GOC 10000 by device and batch) |
| `timing/goc10000_timing_small.csv` | `work/goc10000/timing_small.csv` | `work/goc10000/04_timing.py --variant small` |
| `qlim/summary.csv` | `work/qlim/summary.csv` | `analysis/analysis_qlimits.py report` (reactive limits on GOC 500, voltage limits on IEEE 118) |
| `qlim/batch2_results.json` | written by the batch-2 review script (a local review directory, not in git) | `scripts/qlim_cutoff_sweep.py` holds its `qlimits()` and reproduces the `qlimits` block from `work/qlim/lines_goc500.parquet` (the loading-cutoff sweep 1.001 / 1.0 / 0.99 / 0.95; `--compare` checks it against this file). The other blocks of the file (`conformal`, `lambdas`, `timing`, `finite_behavior`) are from the same review and nothing reads them |
| `conformal/conformal_case118_ieee_base_splits.csv` | `work/conformal_case118_ieee_base_splits.csv` | `analysis/analysis_conformal.py work/hybrid_case118_ieee_base.parquet` (100 random splits by outage set) |
| `conformal/conformal_texas_n2x450_splits.csv` | `work/texas/out/conformal_texas_n2x450_splits.csv` | `analysis/analysis_conformal.py work/texas/out/hybrid_texas_n2x450.parquet` |
| `conformal/shift_case118_fixed.csv` | `work/shift_case118_fixed.csv` | `analysis/analysis_shift.py --reuse` (one fixed band under N-3 and N-4 and the other shifts) |
| `embedding/embedding_auroc.csv` | `work/embedding_auroc.csv` | `analysis/analysis_embedding.py report` (unseeded PCA: a rerun can move the third decimal) |
| `derived/change_mechanism.csv` | computed from `work/goc{500,2000,10000}/lines_hybrid_*.parquet`, `states_hybrid_*.parquet`, `states_newton_tiny_seed0.parquet` | `scripts/derive_results.py` (what `analysis_hybrid_start.py report` prints as "lines moving > 1 point": no-change error removed, sign agreement; GENCO ms per state) |
| `derived/fulln1_w4_missed.csv` | computed from `work/fulln1/contingencies.parquet` | `scripts/derive_results.py` (the 14 critical outages the anchored workflow missed on the scored bases, with their worst loading and post-step mismatch) |
| `derived/goc500_example_rule.csv` | computed from `work/goc500/lines_tiny_seed0.parquet` | `scripts/derive_results.py` (the organisers' F3-threshold rule, as `analysis_goc500.py` prints it: recall and share flagged) |
| `derived/conformal_case118_seed0.csv` | computed from `work/hybrid_case118_ieee_base.parquet` | `scripts/derive_results.py` (the canonical seed-0 split that `analysis_conformal.py` prints but does not save) |
| `derived/finetune_scenario_counts.csv` | `work/finetune/models/*/stats/*_scenario_splits.json` | `scripts/derive_results.py` (train / val / test scenario counts of the fine-tuning runs) |
| `headline_numbers.csv` | | `scripts/headline_numbers.py` (one row per registered claim: registered write-up value, table value, method, source, formula) |
| `headline_numbers_expected.csv` | | `scripts/headline_numbers.py --write-expected` (the snapshot CI compares against; regenerate it only after a deliberate change of a table, never to make the check pass) |

`method` in `headline_numbers.csv`: `recomputed` from per-base, per-split or per-row columns by the formula in
`note`; `read` from one cell of a summary table; `derived` from `derived/*.csv`, which only
`scripts/derive_results.py` on a machine with the saved parquet tables can refresh; `MISSING` when the table
holds no value for the number; `FAILED` (one row per section, in place of its claims) when a table is absent or
a row filter matched no row.

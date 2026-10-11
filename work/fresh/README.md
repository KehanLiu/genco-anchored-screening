# Fresh operating points for the new screening policies

Written 2026-10-10 (investigation only; nothing here has been run). Companion of `work/splits/README.md`, which
freezes the calibration/test split of the *existing* files and states that those files are for development only.
This file gives the recipe for the confirmation run on operating points that no analysis has ever scored, the
inventory of what has been scored (`used_ids.json`), the measured runtimes behind the cost estimate, and the list of
parameters that must be frozen before the run starts. `run_fresh.sh` is the non-executed draft of the run.

## 1. What has been consumed (summary of `used_ids.json`)

| grid | test pool `work/goc<N>/data` (datakit seed 0) | consumed by existing analyses | untouched in pool | fine-tune pool `work/finetune/data<N>_all` (seed 1234) |
|---|---|---|---|---|
| GOC 500 | 3,000 load scenarios (`load_scenario_idx` 0..2999), 14,992 states | 3,000 / 3,000 | **0** | hours 3000..5999: train 8,969 + val 995 states (load idx 3000..4999), AL pool 4,996 states (5000..5999, scored) |
| GOC 2000 | 500 load scenarios (0..499), 2,494 states (datagen asked 1,000, killed by its 7,200 s timeout) | 500 / 500 | **0** | hours 1000..1499: train 2,248 + val 250 states |
| GOC 10000 | 199 load scenarios (0..199, idx 129 missing), 993 states | 199 / 199 | **0** | hours 200..499: train 1,347 + val 150 states |

Why everything is consumed: `analysis/analysis_hybrid_start.py run` (line 193), `analysis/analysis_prev_control.py`,
`work/goc*/02_newton.py`, `analysis/analysis_qlimits.py`, `analysis/analysis_zeroshot.py` and the Round-4 evaluations
(`work/finetune/02_eval.py`, `04_hybrid_eval.py`) all take `np.sort(status.load_scenario_idx.unique())`, i.e. the
whole pool, no subsampling. The full N-1 (`work/fulln1/contingencies.parquet`, column `base` = `load_scenario_idx`
of `work/goc2000/data`) used 24 of the 500 GOC 2000 load scenarios (ranks 0..23 evenly spaced over the ranking by
total load: 20, 25, 29, 75, 86, 97, 100, 102, 109, 115, 140, 159, 177, 231, 270, 290, 296, 309, 331, 361, 383, 431,
435, 491; calibration = ranks 2, 6, 10, 14, 18, 22 = `[435, 75, 431, 231, 86, 115]`), but the other 476 have all been
scored by the hybrid-start files, so none is fresh. The GOC 2000 load-profile parquet holds 1,000 hours but hours
500..999 were never solved (datakit is not resumable; it overwrites its output folder), so they are not usable as is.
The fine-tune pools are a separate id space (seed 1234, `01_split.py` proves zero overlap of load vectors and
(load, outage-set) pairs with the test pools); their numeric ids 0..N-1 coincide with test-pool ids by accident only.
The "new scenarios of the same kind" 8,969 / 2,248 / 1,347 of Round 4 are the fine-tuning *training* sets
(`work/finetune/data<N>_ft/splits_base/train.pt`), generated, not drawn from the test pools.

**Conclusion: fresh points must be generated.** A third datakit pool per grid, with a new seed *and* hours of the
aggregate load profile that neither the test nor the fine-tune pool used (datakit cuts the profile to its first
`load.scenarios` hours; `work/finetune/00_datagen_skip.py` generates a longer cut and drops the chunks below SKIP, so
the kept hours are exactly what a full run with that seed would have produced). Note that a new seed alone would
re-use the same aggregate hours 0..N-1 with different per-bus noise (`gridfm_datakit/perturbations/load_perturbation.py`
lines 417-470: the reference curve is deterministic, only the uniform noise, topology and admittance perturbations
depend on the seed); the SKIP mechanism avoids that.

## 2. Recipe

All commands from the repo root, `.venv/bin/python`, on AC power (`caffeinate -i`), nothing else of ours running.
`<SEED>` is the fresh datakit seed (proposed 2026; must differ from 0 and 1234). Placeholders in angle brackets.

### 2.1 Generate the fresh pools (`work/fresh/data<N>`)

Copy `work/finetune/datagen<N>.yaml` to `work/fresh/datagen<N>.yaml` and change only `load.scenarios`,
`settings.data_dir`, `settings.seed` (chunk sizes must divide SKIP and `load.scenarios`):

| grid | hours used so far | fresh hours | yaml: `scenarios` / `large_chunk_size` / `num_processes` | command |
|---|---|---|---|---|
| GOC 2000 | 0..499 (test), 1000..1499 (ft) | **1500..1999** (500 load scenarios, ~2,500 states) | 2000 / 500 / 4 | `work/texas/tmo 20000 .venv/bin/python work/finetune/00_datagen_skip.py work/fresh/datagen2000.yaml 1500` |
| GOC 10000 | 0..199 (test), 200..499 (ft) | **500..699** (200 load scenarios, ~1,000 states) | 700 / 25 / 3 | `work/texas/tmo 43200 .venv/bin/python work/finetune/00_datagen_skip.py work/fresh/datagen10000.yaml 500` |
| GOC 500 (optional) | 0..2999 (test), 3000..5999 (ft) | **6000..6499** (500 load scenarios, ~2,500 states) | 6500 / 500 / 4 | `work/texas/tmo 14400 .venv/bin/python work/finetune/00_datagen_skip.py work/fresh/datagen500.yaml 6000` |

Output lands in `work/fresh/data<N>/case<N>_goc/raw/{bus,gen,branch}_data.parquet` with `load_scenario_idx` in the
fresh hour range. Disjointness proof against both earlier pools (same hashes as `01_split.py`): run
`.venv/bin/python work/finetune/01_split.py --net case<N>_goc --all work/fresh/data<N> --test work/goc<N>/data --out work/fresh/data<N>_check --first <SKIP> --pool_from <SKIP+n>`
and, in addition, against `work/finetune/data<N>_all` (second `--test` run); keep the two `disjointness.txt`.
Record the fresh `load_scenario_idx` list in `work/fresh/fresh_ids.json` *before* any scoring.

### 2.2 GOC 2000 full N-1 on fresh base cases (loading cushion, classical-first, frozen tau)

`analysis/analysis_full_n1.py` has no load-id argument: `--bases` takes the design **ranks** 0..23 of `design()` (24 load
scenarios evenly spaced over the pool's ranking by total load, `rank % 4 == 2` = calibration). Applied to the fresh
pool, `design()` yields 24 fresh bases deterministically, so the rank argument is enough once the data path can be
redirected. Pre-register the ranks (proposal: the same six ranks `2 6 10 14 18 22` as the old calibration set, so the
load levels are comparable; 12 or 24 are affordable, see section 3):

```
GOC_DATA=work/fresh/data2000 work/texas/tmo 9000 .venv/bin/python -u analysis/analysis_full_n1.py run --grid 2000 --procs 4 \
    --bases <RANKS, e.g. 2 6 10 14 18 22> --tag _fresh          # chunks -> work/fulln1/chunks_2000_fresh/ (resumable)
GOC_DATA=work/fresh/data2000 work/texas/tmo 600 .venv/bin/python -u analysis/analysis_full_n1.py timing --grid 2000 --tag _fresh
.venv/bin/python work/fresh/score_fresh_n1.py --policy work/fresh/policy_frozen.json   # to be written: scores the chunks
```

Minimal code changes needed (not made):
1. `analysis_prev_control.paths()` (line 46-49) hard-codes `work/goc<N>/data`; add an override, e.g.
   `os.environ.get("GOC_DATA")` or a `--data` flag as in `work/goc2000/02_newton.py --data`. `engine.Genco` loads
   the checkpoint and normaliser from `work/goc<N>/pf_case<N>_goc_<variant>` (`engine.py` lines 87-95), independent
   of the data path, so the model is unchanged.
2. `analysis/analysis_full_n1.py`: `report()` ignores `--tag` (`chunk_dir(grid)`, line 446) and re-fits `tau` with
   `pick_tau` on the calibration split, then overwrites `work/fulln1/contingencies.parquet` / `summary.csv`
   (line 587). The fresh scoring must instead read `chunks_2000_fresh/`, take tau / cushion / tau1 from the frozen
   policy file and write under `work/fresh/`; simplest as a separate scorer (`score_fresh_n1.py`) that reuses
   `workflows()` with the frozen values. Columns already stored per contingency suffice for the cushion
   (`max1_prev`, `max1_hyb` vs c), classical-first (`mis1_prev` > tau1 then `max1_hyb`/`mis1_hyb`) and the mismatch
   gate (`mis0_prev`, `mis0_hyb`); only the **shrinkage start** needs a new Newton run in `task()` (line 209-211:
   add a start `shyb` = `vm_p + lam*(vm_g - vm_gi)`, `va_p + lam*(va_g - va_gi)` with the frozen lambda).
3. A measured end-to-end phase (section 4).

### 2.3 Hybrid-start evaluation on fresh GOC 10000 (and GOC 2000 / 500) load scenarios (shrinkage, mismatch gate)

```
GOC_DATA=work/fresh/data10000 work/texas/tmo 3000 .venv/bin/python -u analysis/analysis_prev_control.py run --grid 10000 --tag _fresh        # pre-outage control, model-independent
for ck in "tiny 0" "tiny 1" "small 0" "small 1"; do set -- $ck
  GOC_DATA=work/fresh/data10000 work/texas/tmo 3000 .venv/bin/python -u analysis/analysis_hybrid_start.py run --grid 10000 --variant $1 --seed $2 --tag _fresh --lam <LAMBDA_FROZEN>
done
.venv/bin/python work/fresh/score_fresh_hybrid.py --policy work/fresh/policy_frozen.json
```

`analysis/analysis_hybrid_start.py` already has `--tag` for its output names (`lines_hybrid_<variant>_seed<S><tag>.parquet`,
line 203-204) and runs every load scenario of whatever pool `paths()` returns; it needs the data override (change 1)
and a `--lam` option in `worker()` (the start is formed once, multiply the predicted change by lambda; the fitted
lambda is *not* re-estimated on the fresh data). The mismatch gate is scored from the stored `mis0_prev` /
`mis0_hyb` columns. The same three commands with `--grid 2000` / `--grid 500` cover the smaller grids.

## 3. Runtime estimates (measured sources)

| step | measured | source | fresh estimate |
|---|---|---|---|
| datakit GOC 2000 | 500 load scenarios / 2,498 states in 9:25 wall, 4 procs (1.13 s per load scenario) | `work/finetune/logs/datagen2000.log` (the `time` line) | 500 fresh load scenarios: **~10 min** |
| datakit GOC 10000 | 300 load scenarios / 1,497 states in 54:49 wall, 3 procs, ~15 GB RAM (11 s per load scenario) | `work/finetune/logs/datagen10000.log` | 200 fresh load scenarios: **~37 min** (100: ~18 min) |
| datakit GOC 500 | 6,000 load scenarios / 29,954 states in 16:57 wall (0.17 s per load scenario) | `work/finetune/logs/datagen500.log` | 500 fresh: **~2 min** (+ OPF upper-limit search) |
| full N-1 GOC 2000 | 24 bases x 3,188 outages = 76,512 contingencies in 1,660 s wall, 4 procs, 21.7 ms wall per contingency (each contingency = 1 GENCO inference on CPU + Newton to convergence from 3 starts + DC); 69 s per base | `work/fulln1/report.log` lines 4-37 (`run finished: 76,512 contingencies computed in 1660 s wall`) | 6 fresh bases (19,128 contingencies): **~7.5 min**; 12: ~14 min; 24: ~28 min |
| `timing` phase | 41 outages, single process, ~1 min | `work/fulln1/report.log` lines 39-57 | ~1 min |
| hybrid eval GOC 10000, per checkpoint | 993 states: Tiny s0 105 s, Tiny s1 93 s, Small s0 167 s, Small s1 185 s (4 procs; GENCO inference 180-400 ms per state dominates) | `work/hybrid_control.log`, `work/goc10000/logs/hybrid_tiny_seed1.log:5`, `hybrid_small_seed0.log:5`, `work/hybrid_control_small_seed1.log` | ~1,000 fresh states: **1.5-3 min per checkpoint, ~9.5 min for the four** |
| pre-outage control GOC 10000 | `states_prev_tiny_seed0` run, same order as the hybrid run without inference | `work/prev_control.log` | ~1-2 min |
| hybrid eval GOC 2000 / 500 | 2,494 states 64 s; 14,992 states 143 s | `work/hybrid_control.log` | ~1 min / ~2.5 min per checkpoint |

Whole fresh batch (GOC 2000 + 10000, four checkpoints, 6 N-1 bases): roughly **1 h 15 min** of wall time on AC power,
of which ~47 min is datakit generation. The planning estimate said "hours" for the 6 fresh bases; the
measured rate gives minutes once the pool exists. The GOC 10000 generation needs ~15 GB RAM; keep the lid rule of
`memory/training-gate.md`.

## 4. Measured end-to-end cost

`work/fulln1/timing_2000.json` is written by `analysis/analysis_full_n1.py timing` (lines 294-400; command in the docstring,
line 24): one test base, 41 random outages, each operation timed separately (min of 3 repetitions, median over
outages). Its `ms_e2e` (38.8 ms) is `engine.Genco.solve()` on CPU at batch 1 - GENCO graph build + forward +
inverse transform + flows - i.e. **GENCO-only**, not a workflow time. The "s per 1,000 outages" column of the post is
arithmetic from these unit costs (`report()` lines 515-548: Newton iterations x `ms_iter_full` + inferences x
(`ms_forward_mps_b16` + `ms_prep`) + DC solves x `ms_dc_full`). No existing script measures a per-outage wall time
for a complete workflow, and no flag produces one: `work/goc*/04_timing.py`, `analysis/analysis_solver_timing.py chain` and
`analysis/analysis_hybrid_start.py` (`ms_genco`, inside parallel workers) time single operations only; `gridtrust` has no
timing fields.

Minimal change: a new phase `e2e` in `analysis/analysis_full_n1.py` (~50 lines next to `timing`, reusing `design`,
`base_context`, `contingency_tables`), single process, quiet machine, for a pre-registered sample of fresh outages
(e.g. 200 per base), timing with `perf_counter` the complete sequence per workflow and recording the gate outcome:
- classical one step: tables -> `make_problem` -> start(prev) -> `newton(max_it=1)` -> loading + mismatch -> gate
  (max loading > c or mismatch > tau) -> if flagged, continue to convergence;
- anchored change + one step: the same plus `genco.solve()` (batch 1 on CPU, and separately the batch-16 MPS
  variant, which needs batching across outages and includes graph build and transfer) and the hybrid start; with the
  frozen lambda; with the mismatch-gate variant where both starts' `mis0` are evaluated first;
- full AC: tables -> `make_problem` -> `newton(prev)` to convergence.
Report median and p90 per workflow and per gate outcome, state what is excluded (model load, the pre-outage AC
solution and pre-outage inference, which are per base case). Misses, false flags and these times are then reported
together, per policy, in one table.

## 5. Policies to freeze before the fresh run starts (checklist)

Write `work/fresh/policy_frozen.json` *before* `run_fresh.sh` is allowed to proceed and record its hash in `POLICY_LOG.md` (nothing in this repository is committed until the user asks; the hash and timestamps stand in for a commit). Every value
below is fixed from the existing files (calibration split of `work/splits/policy_splits.json`) and is not touched
again; the fresh data only produce the reported numbers (misses, false flags, share sent to AC, Newton iterations,
measured end-to-end time).

- [x] **Shrinkage lambda** (per grid and checkpoint, one value for Va and one for Vm; `policy_frozen.json` is the
      authoritative specification): least-squares scale fitted per coordinate on the *calibration* load scenarios of
      `policy_splits.json` from the stored sufficient statistics (`analysis/analysis_hybrid_start.py` lines 165-172; formula
      in `work/splits/README.md` 3a), clipped to [0.25, 1.5]. Lambda, the calibration ids and file hashes are in
      `policy_frozen.json` and `POLICY_LOG.md`.
- [ ] **Mismatch gate**: parameter-free form fixed as "take the start with the smaller `mis0`, then one step"
      (`work/splits/README.md` 3b). State the tie rule and that it costs one inference plus one mismatch evaluation.
- [ ] **Mismatch threshold tau** for W1 / W3 / W4 (and the shrunk W4): the values of `work/fulln1/report.log`
      lines 93-95, chosen on the six calibration bases (W1 prev 0.5 MW, W3 genco 10 MW, W4 hyb 5 MW), not re-fitted
      on fresh data.
- [ ] **Loading cushion c in {1.000, 1.0005, 1.001}** for both the classical one-step screen and the anchored
      screen; truth cutoff stays 1.001; all three reported (calibration cannot separate them).
- [x] **Classical-first tau1 in {0.1, 0.2, 0.5, 1.0} MW**: the classic one-step screen (W1) runs first; GENCO (anchored
      start, one step, tau 5 MW, cutoff c) runs only where W1's one-step mismatch exceeds tau1 **or W1 flags a
      loading above c** (`policy_frozen.json` `classical_first_note`, the same rule as
      the `classical_first` rule of the independent screening re-analysis); all four tau1 reported.
- [ ] **Base-case design**: the ranks passed to `--bases` and the fresh datakit seed / SKIP / `scenarios`, with the
      resulting `load_scenario_idx` list written to `fresh_ids.json` before scoring.
- [ ] **Checkpoints**: the released Tiny/Small seed 0/1 (GOC 10000), Tiny seed 0 (GOC 2000/500); no fine-tuned
      model unless frozen here.
- [ ] **End-to-end timing protocol**: sample size per base, single process, AC power, nothing else running, batch
      sizes (CPU batch 1, MPS batch 16), reported together with the misses.
- [ ] **Reporting**: misses (critical contingencies cleared as secure), false flags (secure contingencies sent to AC)
      and measured end-to-end time in one table per policy; no new policy form or threshold is added after the run.

Set `FRESH_IDS_FROZEN=1` only after the file exists and these boxes are ticked; `run_fresh.sh` refuses otherwise.

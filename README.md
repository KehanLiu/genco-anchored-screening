# Let GENCO predict the change, not the state

An entry for the [GENCO Challenge](https://gridfm.org/porto/challenge/): contingency screening with the anchored change start and one Newton step. The write-up is [`submission/POST.md`](submission/POST.md).

In a complete N-1 analysis of the GOC 2000 grid, GENCO on its own misses 54% of the outages that overload a line. Used differently, it becomes useful: start Newton's method from the pre-outage AC solution plus the *change* GENCO predicts for the outage, take one step, and send to the full solver only what that step flags. On 76,512 fresh outages under frozen rules this screen sends 2.17% of outages to the full AC solver and misses one critical outage; the classical one-step screen sends 27.94% and misses two (the two screens keep their own residual thresholds from the old N-1 calibration, so the referral counts are not an equally tuned comparison). It does not save wall-clock time on our hardware: on a 4,800-outage timing sample the anchored workflow took 22.3 ms per outage with GENCO Tiny batched on a laptop GPU, against 8.4 ms for the classical screen and 13.7 ms for complete AC, all three missing zero of 69 critical cases. The savings we report are AC solver calls and Newton iterations, never seconds.

| Workflow (same 4,800 outages, 24 fresh GOC 2000 bases) | Measured ms per outage, first pass | Critical cases missed in this sample |
|---|---:|---:|
| Complete AC, Newton to convergence from the solved base case | 13.7 | 0 of 69 |
| Classical screen, one Newton step from the solved base case | 8.4 | 0 of 69 |
| Anchored change start, one Newton step, GENCO on the laptop GPU in batches of 16 (amortized throughput) | 22.3 | 0 of 69 |

### Mechanism: better initialization for one Newton step

In a loading-only diagnostic on scenarios with up to five branch outages, the anchored start missed 2, 5 and 0 overloaded line readings on IEEE 118, GOC 500 and GOC 2000, versus 74, 41 and 2 from the classical start after one Newton step, at 0.0% of secure scenarios flagged. Both use a converged, matching AC base state. At 10,000 buses no released checkpoint beats the classical screen: the classic start misses 2 lines, the anchored checkpoints miss 5, 3, 2 and 3; the best (Small seed 0) ties on misses while referring about seven times as many secure scenarios. The conformal band of the earlier rounds is an appendix-level component, reported without a Newton step.

![Overload screening after one Newton step](figures/starts.png)

*Loading-only diagnostic with cutoff 1.001 and a solved AC base state. The comparison fixes the number of Newton steps; GENCO inference cost and complete-AC performance are not shown.*

## What is here

| Path | What it does |
|---|---|
| `engine.py` | One grid state (load level, line outages, optional redispatch) solved by AC power flow, DC power flow and GENCO. Used by everything else. |
| `analysis/analysis_prev_control.py`, `analysis/analysis_hybrid_start.py` | The headline: one Newton step from the solved pre-outage AC base case (classic screen), and from that base case plus GENCO's predicted change (anchored change start). |
| `analysis/analysis_full_n1.py` | A complete N-1 analysis on GOC 2000 with the cost of six workflows, unit-cost timing and the measured end-to-end phase. |
| `work/fresh/` | Fresh operating points, frozen policies: `policy_frozen.json` (the policy, frozen before generation), `POLICY_LOG.md` (freeze and amendments), `score_fresh_n1.py` and `score_fresh_hybrid.py` (scorers), `run_fresh.sh` with the chain scripts (`datagen_chain.sh`, `n1_2000_chain.sh`, `n1_extend_chain.sh`, `hybrid2000_chain.sh`, `chain10000.sh`), the results `results_n1.md` and `results_hybrid.md`, and a `README.md` with the recipe. |
| `work/equiv/` | Equivalent-circuit robustness diagnostic: `SPEC.md` (pre-registered), `REPORT.md`, `transforms.py` (admittance-preserving rewrites), `run_equiv.py`, tests (`test_transforms.py`, `test_abort.py`), summaries in `results/*.csv`. |
| `work/splits/` | Frozen calibration/test splits (`policy_splits.json`) and the rules for selecting versus reporting a policy. |
| `work/goc500/`, `analysis/analysis_goc500.py`, `analysis/analysis_goc500_newton.py` | The example entry's setup (GOC 500, N-1 to N-5): its F-beta rule, the conformal band, GENCO + one Newton step. |
| `work/goc2000/` | The same on 2,000 buses: `00_datagen_first.py` (reproduces the test pool bit for bit, see Data), `01_fetch.py`, `02_newton.py`, `03_analysis.py`, `04_timing.py`; `run_all.sh` chains generation, Newton steps, analysis and timing. |
| `work/goc10000/` | The same on 10,000 buses: `00_diag.py`, `01_fetch.py`, `02_newton.py`, `03_analysis.py`, `04_timing.py` (no `run_all.sh`; `reproduce.sh sweeps` calls them). |
| `analysis/analysis_solver_timing.py` | Start points inside PowerModels' AC power flow: convergence, iterations, time. |
| `analysis/analysis_qlimits.py` | The main experiments with generator reactive limits (PV-to-PQ switching). |
| `analysis/analysis_zeroshot.py` | The pretrained model, without fine-tuning, on grids it has not seen (IEEE 118, GOC 500, PEGASE 1354, GOC 2000). |
| `analysis/analysis_finetune.py`, `analysis/analysis_finetune_hybrid.py`, `work/finetune/` | Fine-tuning the released Tiny checkpoints on new multi-outage scenarios (GOC 500, 2000, 10000) and scoring them from GENCO's start and from the anchored change start. `20_pipeline500.sh`, `20_pipeline2000.sh` and `20_pipeline10000.sh` chain training and evaluation. |
| `analysis/analysis_hybrid.py`, `work/texas/04_hybrid.py` | Outage sweeps on IEEE 118 and Texas 2000: per-line loadings from AC, DC and GENCO. |
| `analysis/analysis_conformal.py` | The conformal band (appendix): calibration, held-out scoring, coverage over 100 splits, fixed-budget curves. |
| `analysis/analysis_warmstart.py`, `analysis/analysis_onestep.py` | GENCO as a Newton start point; one Newton step as a physics correction. |
| `analysis/analysis_shift.py` | The band under distribution shift (N-3, N-4, load, power factor, redispatch) and recalibration. |
| `analysis/analysis_voltage.py` | Voltage-limit violations, which DC power flow cannot see. |
| `analysis/analysis_embedding.py` | GENCO's embedding space as a detector of states unlike calibration (a negative result). |
| `analysis/analysis_remedial.py` | Ranking redispatch actions with GENCO, GENCO + one Newton step and DC sensitivities. |
| `analysis/analysis_atlas.py` | Where GENCO's errors are: by line, outage, load level, loading. |
| `analysis/analysis_residual.py`, `analysis/analysis_redispatch.py` | Earlier studies of GENCO's power-balance residual and of off-OPF dispatch. |
| `analysis/make_figures.py` | The 17 figures in `figures/` drawn from saved results (`--help` lists their names). The workbench images and `workbench.gif` come from `workbench/tools/record_workbench_gif.py`. |
| `work/verify_round1.py` | Independent re-derivation of the numbers quoted in the round-1 write-up. |
| `gridtrust/` | Standalone package (numpy, scipy, pandas): Newton refinement, conformal band, staged decision; 41 tests. With the 23 tests of `work/equiv`, `pytest gridtrust/tests work/equiv -q` runs 64. |
| `upstream/` | Draft contribution to gridfm-graphkit (not submitted), with its own copy of the MIT licence. |
| `workbench/server.py`, `workbench/web/` | Operator workbench showing the staged workflow on IEEE 118: `python workbench/server.py`, then open http://127.0.0.1:8000 (for example `?load=1.1&out=31,103&alpha=0.01&view=newton`). |
| `workbench/whatif.py` | Every single-line outage on IEEE 118, GENCO against the AC solver, with the workbench's engine. |
| `workbench/tools/record_workbench_gif.py` | Records `figures/workbench.gif` and the workbench screenshots from a running server (needs Chrome and Pillow). |
| `work/workbench_check.py` | Checks of the workbench's staged verdict against the AC solution (`work/workbench_check.log`). |
| `Dockerfile`, `workbench/docker/` | Container for the workbench, laid out for a Hugging Face Docker Space (port 7860, non-root). Built and smoke-tested on 2026-10-11 (Apple Silicon, linux/arm64): `docker build --platform linux/arm64 -t genco-workbench:dev .` (50 min with a cold pip cache, nearly all of it wheel downloads; 4.81 GB on disk, 1.07 GB compressed), then `docker run --rm -p 7860:7860 genco-workbench:dev` and open http://localhost:7860. The build downloads the IEEE 118 Base checkpoint (Hugging Face, no token needed) and the bus layout, checks their sha256, precompiles the Julia environment and solves one state before the image is accepted. |
| `run_pretrained.py` | Fetch a released checkpoint and run it on one data shard. |
| `work/**/*.log`, `work/*.txt`, `work/**/*.csv`, `work/**/*.md` | The printed reports and small result tables behind the numbers in the write-up. |

## Reproduce

`reproduce.sh` runs one stage per call; with no argument it prints its usage and exits 1.

```sh
./reproduce.sh setup      # Python 3.12 environment from requirements.lock.txt (needs uv)
./reproduce.sh fetch      # networks (PGLib-OPF v23.07) and checkpoints (Hugging Face, pinned commits), then checksums
./reproduce.sh checksums  # sha256 of every fetched input that is present
./reproduce.sh tests      # pytest gridtrust/tests work/equiv -q (64 tests; some skip when result tables are absent)
./reproduce.sh analysis   # every table of the write-up, from the saved result tables: minutes (includes the work/fresh scorers and the work/equiv reports)
./reproduce.sh figures    # figures/*.png from the saved result tables
./reproduce.sh all        # setup, fetch, tests, analysis, figures
RUN_LONG=1 ./reproduce.sh all   # also sweeps, controls, finetune, fresh, equiv (in that order) before analysis
```

What each part needs, in three tiers:

| Tier | Commands | Needs | Wall time (Apple-silicon laptop, October 2026) |
|---|---|---|---|
| (a) clean clone, no saved data | `./reproduce.sh setup`, `fetch`, `checksums`, `tests`; the registered number check `python scripts/headline_numbers.py --check` and the package tests `python -m pytest gridtrust/tests -q` (41 tests; CI runs these two on every push with `pip install -e gridtrust pytest`) | the clone, the network (PGLib-OPF v23.07, the pinned Hugging Face checkpoints and the IEEE 118 shard), `uv`; the Texas case file only for its manifest row | under an hour, mostly downloads; `tests` collects 64 and skips 15 without saved tables |
| (b) saved result tables plus two datakit pools | `./reproduce.sh analysis`, `./reproduce.sh figures` | `work/**/*.parquet` and `work/fulln1/chunks_*/` (not in git; the small `*.csv`, `*.log`, `*.md` next to them are), the GOC 2000 test pool `work/goc2000/data/case2000_goc/raw/` (read by `analysis_full_n1.py report`) and the fresh GOC 2000 pool `work/fresh/data2000/case2000_goc/raw/` (read by `work/fresh/score_fresh_n1.py`). None of these is in git and **no results archive exists**; only tier (c) produces them | `analysis` 985 s in the replay, plus minutes for the fresh scorers and the equiv reports; `figures` 12 s; the first `import engine` adds a one-time Julia download (200 MB, about 5 min) |
| (c) full recomputation | `RUN_LONG=1 ./reproduce.sh all`, or `sweeps`, `controls`, `finetune`, `fresh`, `equiv` one at a time in that order, then `analysis` and `figures` | tier (a) plus AC power and `caffeinate -i` (`finetune`, `fresh`), about 15 GB RAM for the GOC 10000 generations, a quiet machine for the timing passes in `controls` and `fresh`; `fresh` needs the test pools from `sweeps` and the fine-tune pools from `finetune` for its disjointness proofs, `equiv` needs the fresh pools from `fresh` | about two days for `sweeps`, `controls` and `finetune`; `fresh` about 2 h 50 min sequential (the original chains overlapped, 15:08 to 17:21 on 2026-10-10); `equiv` 33 min. Every long command in `reproduce.sh` carries its measured wall time and the log it was read from |

The `fresh` stage runs the six chain scripts of `work/fresh/` in order (`datagen_chain.sh`, `hybrid2000_chain.sh`, `n1_2000_chain.sh`, `n1_extend_chain.sh`, `chain10000.sh`, `quiet_chain.sh`; `run_fresh.sh` is the earlier, never executed draft and is not called). The policies in `work/fresh/policy_frozen.json` were frozen on 2026-10-10 before any fresh pool existed (`POLICY_LOG.md`), and nothing is re-fitted on the fresh data. `n1_2000_chain.sh` runs the full N-1 on the **six registered base ranks** 2, 6, 10, 14, 18, 22 (`n1_base_ranks` of the policy file); `n1_extend_chain.sh` then extends it to all 24 ranks, the **18-base extension** registered in `POLICY_LOG.md` A2 before any fresh N-1 outcome had been inspected; the scorer reports the six, the 18 and all 24 separately. The `equiv` stage runs `work/equiv/fresh_chain.sh` (20 `run_equiv.py` runs on the fresh GOC 500 and GOC 2000 pools, pre-registered in `work/equiv/SPEC.md`); the summary and criterion CSVs it is scored with come from `run_equiv.py report` in the `analysis` stage.

Replay status: the default path was replayed from a fresh clone on 2026-10-11 (setup to tests re-checked from a fresh clone on 2026-10-19, same result) (macOS 27.0.1, Apple silicon, uv 0.12.7, Python 3.12.14). `setup`, `fetch` (53 manifest entries ok, the Texas case absent as documented), `checksums` and `tests` (64 collected; 49 pass / 15 skip with the fetched inputs only, 54 / 10 with the saved tables) ran unchanged from the clone. `analysis` (985 s) and `figures` (12 s) ran to completion **only after** the saved result tables and the GOC 2000 test pool `work/goc2000/data/case2000_goc/raw/` (read by `analysis_full_n1.py report`) had been copied into the clone from the working copy that produced the write-up (there is no archive to download them from); with those present, all 17 figures came back byte-identical to the committed ones. The `analysis` stage of that replay was its 2026-10-11 version: the fresh scorers and the `run_equiv.py report` calls were appended to it on 2026-10-22. In the same replay `work/fresh/score_fresh_n1.py` and `score_fresh_hybrid.py` were run by hand with the commands now in the stage (they also need the fresh pool `work/fresh/data2000/case2000_goc/raw/` copied in) and regenerated `work/fresh/results_n1.md` and `results_hybrid.md` identically; `run_equiv.py report` has not been replayed from a clone. The `fresh` and `equiv` stages were added on 2026-10-22 from the chain scripts that produced the write-up's tables on 2026-10-10; they were checked with `sh -n` and the scripts' `--help`, not rerun. The first `import engine` inside `analysis` installs Julia 1.13.1 through juliacall (200 MB download, about 5 min); if that download fails, re-run the stage. The `analysis_embedding.py` tables are not bit-reproducible (unseeded PCA solver).

`results/` holds copies of the summary tables behind the numbers that `scripts/headline_numbers.py` registers; `python scripts/headline_numbers.py --check` is a regression check of those registered numbers against `results/headline_numbers_expected.csv` (CI runs it on every push). It does not parse the write-up (each claim's stated value is typed into the script), and the equivalent-circuit section (`work/equiv/`) is outside its coverage. `results/README.md` maps each file to the analysis command that wrote it. After a `RUN_LONG=1` recomputation, copy the refreshed tables into `results/`, run `scripts/derive_results.py` and `scripts/headline_numbers.py --write-expected`.

## Data

- **Networks.** IEEE 118, PEGASE 1354 and the GOC 500, 2000 and 10000 cases come from PGLib-OPF at tag `v23.07`; `reproduce.sh fetch` downloads them from that tag and verifies a sha256 manifest. Texas 2000 (ACTIVSg2000, the Texas A&M synthetic grid): place `Texas2k_case1_2016summerpeak.m` in `work/texas/` yourself; it is not redistributed here (its sha256 is in the manifest).
- **Checkpoints**, pinned by Hugging Face commit: `gridfm/genco-pf-datakit` at `7014f145635f26fa29e9e003f0dc33f689d615da` (IEEE 118 Base, seeds 0, 1 and 42; GOC 500 Tiny, seeds 0, 1 and 42; GOC 2000 Tiny and Small; GOC 10000 Tiny and Small, seeds 0 and 1), `gridfm/genco-pf-contingency` at `5c625b00f5b79c8a7d4d27ecc133d4ecdf573af7` (Texas 2000), `gridfm/genco-pf-transfer` at `e7a98e99221ba8241385e491787aedb0940d5130` (the pretrained model of the zero-shot study), the dataset `gridfm/pf_small_case118_ieee` at `3f5a6e266f759be15e3777bf4d70133ff40e6fa7` (one shard), and the Texas evaluation config from gridfm-graphkit commit `95b90eecc7f5c2c86a9dd776f0c3f40434def22f`. Every file has a sha256 in `reproduce.sh`. All results except the fine-tuning section use these released checkpoints.
- **GOC test pools** (gridfm-datakit, seed 0): 3,000 / 500 / 199 load scenarios on GOC 500 / 2000 / 10000, giving 14,992 / 2,494 / 993 states. GOC 2000 asked for 1,000 scenarios and was cut to the first 500 by a timeout; `work/goc2000/00_datagen_first.py` reproduces exactly that pool bit for bit.
- **Fine-tune pools** (seed 1234, `work/finetune/`) and **fresh pools** (seed 2026, `work/fresh/`, recipe and disjointness proof in `work/fresh/README.md`) are separate id spaces, never scored by the main analyses before their use.
- The large result tables (`*.parquet`, the chunk folders) and the datakit pools are not in the repository and no archive of them exists; the long stages of `reproduce.sh` rebuild them (tier (c) above).

## Outlook

The screen pays off only if it is both more accurate than the classic one-step screen and cheaper than complete AC, and the two conditions depend on different things. Accuracy, holding the hardware fixed, depends on the model: it is met on IEEE 118, GOC 500 and GOC 2000 and not at 10,000 buses, where the released checkpoints remove only 28 to 39% of the median absolute error of the "no change" assumption on lines that move by more than one point. Cost, holding the model fixed, depends on hardware and the graph builder: in a unit-cost estimate from the fresh GOC 2000 timing sample (4,800 outages), the anchored workflow saves 1.84 Newton iterations per outage against complete AC (7.0 ms at 3.81 ms per iteration) and 0.53 against the classic screen (2.0 ms), and adds one forward pass (12.74 ms on the laptop GPU at batch 16), a 2.40 ms graph build and 0.18 ms of amortized pre-outage inference. A server GPU with large batches and a graph builder that reuses the base-case graph across outages might change the cost side; whether it would was not measured, and any such claim would have to give complete AC and the classic screen the same CPU cores it gives the model its GPU. Details and exclusions: "Where it could win, and what that would take" in the write-up.

## Limits

No speed claim is made: on a laptop, with inference counted, every GENCO workflow is slower than the classic one-step screen, and the anchored workflow is slower than complete AC on the measured fresh GOC 2000 sample (22.3 against 13.7 ms per outage). Complete AC evaluates every case; that evaluation cost is not an overload false-alarm rate. We assume a converged AC base state consistent with the injections, controls and network model used for the contingency; sensitivity to state-estimation error, stale operating points and topology uncertainty was not tested. Fine-tuning was done only at laptop scale (at most two hours per grid) and showed no gain. The savings reported are AC solves and Newton iterations. See "Limits of this work" in the write-up.

## Licences

- This repository and `gridtrust/`: MIT, see `LICENSE` (copied to `upstream/LICENSE` for the draft contribution).
- PGLib-OPF case files: Creative Commons Attribution 4.0, as stated in the [PGLib-OPF repository](https://github.com/power-grid-lib/pglib-opf) README; credit the original authors of each case.
- GENCO checkpoints (`gridfm/genco-pf-datakit`, `gridfm/genco-pf-contingency`, `gridfm/genco-pf-transfer`), gridfm-graphkit and gridfm-datakit: Apache-2.0, projects of GridFM.
- ACTIVSg2000 (Texas 2000): under the terms of Texas A&M University, see https://electricgrids.engr.tamu.edu/; not redistributed here.

**Author:** Kehan Liu. **Contact:** kehan_liu@outlook.com. Entered privately; public data only.

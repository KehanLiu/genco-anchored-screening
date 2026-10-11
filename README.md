# Let GENCO predict the change, not the state

An entry for the [GENCO Challenge](https://gridfm.org/porto/challenge/): contingency screening with the anchored change start and one Newton step. The write-up is [`submission/POST.md`](submission/POST.md).

GENCO's predicted voltage changes improve one-step Newton screening on several benchmark grids, but the measured implementation has no demonstrated end-to-end cost advantage. On a 4,800-outage sample from the fresh GOC 2000 pool, the anchored workflow took 22.3 ms per outage with GPU batch-16 inference, versus 8.4 ms for classical screening and 13.7 ms for complete AC. All three missed zero of the 69 critical cases in that timing sample; numerical nonconvergences were treated as flagged cases. These are laptop throughput measurements, not a demonstrated deployment advantage.

| Workflow (same 4,800 outages, 24 fresh GOC 2000 bases) | Measured ms per outage, first pass | Critical cases missed in this sample |
|---|---:|---:|
| Complete AC, Newton to convergence from the solved base case | 13.7 | 0 of 69 |
| Classical screen, one Newton step from the solved base case | 8.4 | 0 of 69 |
| Anchored change start, one Newton step, GENCO on the laptop GPU in batches of 16 (amortized throughput) | 22.3 | 0 of 69 |

On the full fresh pool (76,512 single-branch outages, frozen policies) the anchored screen referred 2.17% of outages to full AC against 27.94% for the classical screen and missed 1 against 2 critical outages; the two screens keep their own residual thresholds from the old N-1 calibration, so the referral counts are not an equally tuned comparison. The pooled counts and the 4,800-sample timings are different scopes. At 10,000 buses no released checkpoint beats the classical screen: the classic start misses 2 lines, the anchored checkpoints miss 5, 3, 2 and 3; the best (Small seed 0) ties on misses while referring about seven times as many secure scenarios. The conformal band of the earlier rounds is an appendix-level component, reported without a Newton step.

### Mechanism: better initialization for one Newton step

In a separate loading-only diagnostic on scenarios with up to five branch outages, the anchored start missed 2, 5 and 0 overloaded line readings on IEEE 118, GOC 500 and GOC 2000, versus 74, 41 and 2 from the classical start after one Newton step. Both use a converged, matching AC base state. This demonstrates better one-step initialization on these datasets; it is not an equal-cost comparison or a safety guarantee.

![Overload screening after one Newton step](figures/starts.png)

*Loading-only diagnostic with cutoff 1.001 and a solved AC base state. The comparison fixes the number of Newton steps; GENCO inference cost and complete-AC performance are not shown. The measured end-to-end workflow is slower than complete AC on the separate fresh GOC 2000 timing sample.*

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
./reproduce.sh analysis   # every table of the write-up, from the saved result tables: minutes
./reproduce.sh figures    # figures/*.png from the saved result tables
./reproduce.sh all        # setup, fetch, tests, analysis, figures
RUN_LONG=1 ./reproduce.sh all   # also sweeps, controls, finetune before analysis
```

`analysis` and `figures` need the saved result tables (`work/**/*.parquet`, `*.csv`, `*.log`) that are **not in the repository**; without them these two stages fail. The long stages `sweeps`, `controls` and `finetune` rebuild those tables from the fetched inputs in about two days on a laptop (`finetune` needs AC power and `caffeinate`). No results archive is available yet.

Replay status: the default path was replayed from a fresh clone of commit 4c2987b on 2026-10-11 (macOS 27.0.1, Apple silicon, uv 0.12.7, Python 3.12.14). `setup`, `fetch` (53 manifest entries ok, the Texas case absent as documented), `checksums` and `tests` (64 collected; 49 pass / 15 skip with the fetched inputs only, 54 / 10 with the saved tables) ran unchanged from the clone. `analysis` (985 s) and `figures` (12 s) ran to completion once the saved result tables and the GOC 2000 test pool `work/goc2000/data/case2000_goc/raw/` (read by `analysis_full_n1.py report`) were copied in, and all 17 figures came back byte-identical to the committed ones; `work/fresh/results_n1.md` and `results_hybrid.md` were regenerated identically by their scorers. The first `import engine` inside `analysis` installs Julia 1.13.1 through juliacall (200 MB download, about 5 min); if that download fails, re-run the stage. The `analysis_embedding.py` tables are not bit-reproducible (unseeded PCA solver).

## Data

- **Networks.** IEEE 118, PEGASE 1354 and the GOC 500, 2000 and 10000 cases come from PGLib-OPF at tag `v23.07`; `reproduce.sh fetch` downloads them from that tag and verifies a sha256 manifest. Texas 2000 (ACTIVSg2000, the Texas A&M synthetic grid): place `Texas2k_case1_2016summerpeak.m` in `work/texas/` yourself; it is not redistributed here (its sha256 is in the manifest).
- **Checkpoints**, pinned by Hugging Face commit: `gridfm/genco-pf-datakit` at `7014f145635f26fa29e9e003f0dc33f689d615da` (IEEE 118 Base, seeds 0, 1 and 42; GOC 500 Tiny, seeds 0, 1 and 42; GOC 2000 Tiny and Small; GOC 10000 Tiny and Small, seeds 0 and 1), `gridfm/genco-pf-contingency` at `5c625b00f5b79c8a7d4d27ecc133d4ecdf573af7` (Texas 2000), `gridfm/genco-pf-transfer` at `e7a98e99221ba8241385e491787aedb0940d5130` (the pretrained model of the zero-shot study), the dataset `gridfm/pf_small_case118_ieee` at `3f5a6e266f759be15e3777bf4d70133ff40e6fa7` (one shard), and the Texas evaluation config from gridfm-graphkit commit `95b90eecc7f5c2c86a9dd776f0c3f40434def22f`. Every file has a sha256 in `reproduce.sh`. All results except the fine-tuning section use these released checkpoints.
- **GOC test pools** (gridfm-datakit, seed 0): 3,000 / 500 / 199 load scenarios on GOC 500 / 2000 / 10000, giving 14,992 / 2,494 / 993 states. GOC 2000 asked for 1,000 scenarios and was cut to the first 500 by a timeout; `work/goc2000/00_datagen_first.py` reproduces exactly that pool bit for bit.
- **Fine-tune pools** (seed 1234, `work/finetune/`) and **fresh pools** (seed 2026, `work/fresh/`, recipe and disjointness proof in `work/fresh/README.md`) are separate id spaces, never scored by the main analyses before their use.
- The large result tables (`*.parquet`) are not in the repository; the long stages of `reproduce.sh` rebuild them.

## Limits

No speed claim is made: on a laptop, with inference counted, every GENCO workflow is slower than the classic one-step screen, and the anchored workflow is slower than complete AC on the measured fresh GOC 2000 sample (22.3 against 13.7 ms per outage). Complete AC evaluates every case; that evaluation cost is not an overload false-alarm rate. We assume a converged AC base state consistent with the injections, controls and network model used for the contingency; sensitivity to state-estimation error, stale operating points and topology uncertainty was not tested. Fine-tuning was done only at laptop scale (at most two hours per grid) and showed no gain. The savings reported are AC solves and Newton iterations. See "Limits of this work" in the write-up.

## Licences

- This repository and `gridtrust/`: MIT, see `LICENSE` (copied to `upstream/LICENSE` for the draft contribution).
- PGLib-OPF case files: Creative Commons Attribution 4.0, as stated in the [PGLib-OPF repository](https://github.com/power-grid-lib/pglib-opf) README; credit the original authors of each case.
- GENCO checkpoints (`gridfm/genco-pf-datakit`, `gridfm/genco-pf-contingency`, `gridfm/genco-pf-transfer`), gridfm-graphkit and gridfm-datakit: Apache-2.0, projects of GridFM.
- ACTIVSg2000 (Texas 2000): under the terms of Texas A&M University, see https://electricgrids.engr.tamu.edu/; not redistributed here.

**Author:** Kehan Liu. **Contact:** kehan_liu@outlook.com. Entered privately; public data only.

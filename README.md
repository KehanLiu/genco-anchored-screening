# Let GENCO predict the change, not the state

An entry for the [GENCO Challenge](https://gridfm.org/porto/challenge/): GENCO as the first stage of a contingency screen with a stated miss rate. The write-up is in [`submission/POST.md`](submission/POST.md); the poster outline in [`submission/POSTER.md`](submission/POSTER.md).

Start Newton from the operator's pre-outage AC solution plus GENCO's predicted voltage change, take one step, and the contingency screen misses fewer overloads than the classic one-step screen on IEEE 118, GOC 500 and GOC 2000. It stops helping at 10,000 buses and it is not faster on a laptop. Where no Newton step is taken, a conformal band decides which states go to AC and keeps the share of missed overloads at a chosen level.

![Which start makes one Newton step enough?](figures/starts.png)

## What is here

| Path | What it does |
|---|---|
| `engine.py` | One grid state (load level, line outages, optional redispatch) solved by AC power flow, DC power flow and GENCO. Used by everything else. |
| `analysis_hybrid.py`, `work/texas/04_hybrid.py` | Outage sweeps on IEEE 118 and Texas 2000: per-line loadings from AC, DC and GENCO. |
| `analysis_conformal.py` | The conformal trust layer: band calibration, held-out scoring, coverage over 100 splits, fixed-budget curves. |
| `analysis_goc500.py`, `analysis_goc500_newton.py`, `work/goc500/` | The example entry's setup (GOC 500, N-1 to N-5): its F-beta rule, the conformal band, GENCO + one Newton step. |
| `work/goc2000/`, `work/goc10000/` | The same experiment on 2,000 and 10,000 buses (`run_all.sh` chains generation, Newton steps, analysis and timing). |
| `analysis_prev_control.py`, `analysis_hybrid_start.py` | One Newton step from the pre-outage AC solution, and from that solution plus GENCO's predicted change. |
| `analysis_full_n1.py` | A complete N-1 analysis on GOC 2000 with the cost of six workflows. |
| `analysis_solver_timing.py` | Start points inside PowerModels' AC power flow: convergence, iterations, time. |
| `analysis_qlimits.py` | The main experiments with generator reactive limits (PV-to-PQ switching). |
| `analysis_zeroshot.py` | The pretrained model, without fine-tuning, on grids it has not seen. |
| `analysis_finetune.py`, `analysis_finetune_hybrid.py`, `work/finetune/` | Fine-tuning the released Tiny checkpoints on new multi-outage scenarios (GOC 500, 2000, 10000) and scoring them from GENCO's start and from the combined start. `20_pipeline500.sh`, `20_pipeline2000.sh` and `20_pipeline10000.sh` chain training and evaluation. |
| `gridtrust/` | Standalone package: Newton refinement, conformal band, staged decision; 32 tests (`pytest gridtrust/tests`). |
| `upstream/` | Draft contribution to gridfm-graphkit (not submitted). |
| `Dockerfile`, `docker/` | Container for the workbench (written, not yet built). |
| `analysis_warmstart.py`, `analysis_onestep.py` | GENCO as a Newton start point; one Newton step as a physics correction. |
| `analysis_shift.py` | The guarantee under distribution shift (N-3, N-4, load, power factor, redispatch) and recalibration. |
| `analysis_voltage.py` | Voltage-limit violations, which DC power flow cannot see. |
| `analysis_embedding.py` | GENCO's embedding space as a detector of states unlike calibration (a negative result). |
| `analysis_remedial.py` | Ranking redispatch actions with GENCO, GENCO + one Newton step and DC sensitivities. |
| `work/verify_round1.py` | Independent re-derivation of the numbers quoted in the write-up. |
| `analysis_atlas.py` | Where GENCO's errors are: by line, outage, load level, loading. |
| `analysis_residual.py`, `analysis_redispatch.py` | Earlier studies of GENCO's power-balance residual and of off-OPF dispatch. |
| `make_figures.py` | Every figure in `figures/`, drawn from saved results. |
| `server.py`, `web/` | Operator workbench showing the three-stage workflow: `python server.py`, then open http://127.0.0.1:8000 (for example `?load=1.1&out=31,103&alpha=0.01&view=newton`). |
| `run_pretrained.py` | Fetch a released checkpoint and run it on one data shard. |
| `work/**/*.log`, `work/*.txt`, `work/**/*.csv` | The printed reports and small result tables behind the numbers in the write-up. |

## Reproduce

```sh
./reproduce.sh setup      # Python 3.12 environment from requirements.lock.txt (needs uv)
./reproduce.sh fetch      # checkpoints from Hugging Face
./reproduce.sh sweeps     # AC, DC and GENCO on all grid states: several hours on a laptop
./reproduce.sh analysis   # all tables, from the saved sweeps: minutes
./reproduce.sh figures    # figures/*.png
```

The `setup`, `analysis` and `figures` stages were verified in a fresh copy of the repository (with the saved result tables in place); `fetch` and `sweeps` were run piece by piece while the work was done, not through this script.

All results except the fine-tuning section use the released checkpoints: `gridfm/genco-pf-datakit` (IEEE 118 Base and GOC 500 Tiny, seeds 0, 1 and 42; GOC 2000 and GOC 10000 Tiny and Small) and `gridfm/genco-pf-contingency` (Texas 2000).

## Data

- IEEE 118 and the GOC 500, 2000 and 10000 cases come from PGLib-OPF through gridfm-datakit.
- Texas 2000: place `Texas2k_case1_2016summerpeak.m` in `work/texas/` yourself; it is not redistributed here.
- The large result tables (`*.parquet`) are not in the repository. `./reproduce.sh sweeps` rebuilds them.

## Limits

No speed claim is made: on a laptop, with inference counted, every GENCO workflow is slower than the classic one-step screen. Fine-tuning was done only at laptop scale (at most two hours per grid) and showed no gain. The savings reported are AC solves and Newton iterations. See "Limits of this work" in the write-up.

## Licence

MIT, see `LICENSE`. GENCO, gridfm-graphkit and gridfm-datakit are Apache-2.0 projects of GridFM.

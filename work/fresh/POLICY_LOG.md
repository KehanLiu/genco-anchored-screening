> 2026-10-11: the analysis scripts were moved from the repository root to `analysis/` (`analysis_full_n1.py` -> `analysis/analysis_full_n1.py` etc.). The paths and file hashes recorded below refer to the pre-move layout.

# Policy log for the fresh confirmation run

`policy_frozen.json` is the authoritative specification of every policy form and parameter; the READMEs describe it and defer to it where they differ. Nothing in this repository is committed (the user commits on request), so file hashes and timestamps are recorded here instead. Timestamps are local file times and log lines on this machine; they are evidence of the order of events on this machine, not independent proof.

## Freeze

- `work/fresh/policy_frozen.json` written 2026-10-10 2026-10-10 15:07:29 (mtime), SHA256 `c4bac593e1f6331344222b1653191d32b0c297593f38c678e5becefd78ba2652` (unchanged since; an internal audit recorded the same hash).
- Fresh GOC 2000 generation started 2026-10-10 15:08:52 (`work/fresh/data2000/case2000_goc/raw/error.log` first line; `datagen2000.yaml` mtime 15:08:51), i.e. after the policy was written. GOC 10000 generation followed in the same chain (`datagen_chain.sh`).
- Fresh GOC 2000 hybrid-start outcomes (`results_hybrid_goc2000.md`) were first inspected at 15:20:00 (mtime). Fresh N-1 outcomes had NOT been inspected when amendment A2 below was recorded (the six-rank run started 15:22:18 and was still running; only its log header was read).

## Amendments (chronological)

- **A1** (15:2x, during an internal audit): `run_fresh.sh` N-1 `run` line gained `--policy` so the shrunk start is computed; no policy value changed.
- **A2** (15:30): N-1 design expanded from the six registered ranks {2, 6, 10, 14, 18, 22} to all 24 ranks of the fresh design, on an internal audit's recommendation (operating-point dependence in the old study). The original six remain the registered primary set and are reported separately; the other 18 are reported as the extension. Recorded before any fresh N-1 outcome was inspected; after the fresh GOC 2000 hybrid outcomes had been seen. Script `n1_extend_chain.sh`.
- **A3** (same time): README wording aligned to the JSON: separate Va/Vm scales (fresh README section 5) and the classical-first trigger 'W1 mismatch > tau1 OR W1 flags a loading above c' (fresh README 5, splits README 3d). The JSON and the scorer had this trigger from the start (it is the rule of the earlier screening re-analysis); the READMEs were incomplete.
- **A4** (same time): a fresh GOC 500 pool (seed 2026, hours 6000-6499, `datagen500.yaml`) is registered for the equivalent-circuit pilot (`work/equiv/`), not for the fresh confirmation run; generated at the end of `chain10000.sh`.

## Not changed
- No tau, c, tau1, lambda, gate rule, checkpoint or e2e protocol value has been changed since the freeze. Any later change gets a new entry here with its reason and whether outcomes had been seen.

## Reporting rules (from an internal audit)
- Every frozen variant is reported with safety (missed critical contingencies, secure sent to AC) and cost (iterations, estimated and measured ms) together, per base case (paired), for the six registered ranks and the 18-rank extension separately and pooled.
- No member of the family is selected on the fresh results and then described as independently validated. Thousands of outages from one base case are one operating point, not thousands.

## Code and data versions recorded at 15:3x on 2026-10-10 (SHA256; historical record, kept verbatim)

The policy file itself froze at 15:07:29. The code hashes below were recorded when this log was first written
(about 15:30), i.e. AFTER the freeze and after the GOC_DATA / --policy code changes of 15:11-15:21; they are the versions
that ran the six-rank N-1 (15:22) and the fresh hybrid runs. Later executed versions are appended in the next section.

| file | mtime | sha256 |
|---|---|---|
| `analysis_full_n1.py` | 2026-10-10 15:19:49 | `c7443b58b21dad0d24c9a5d33b2dc49855571d8f136ab930a44e02971c4e4c19` |
| `analysis_hybrid_start.py` | 2026-10-10 15:11:27 | `299027fa563cf9ffa09651da339c3320774c799b021b80877db54243439b9bed` |
| `analysis_prev_control.py` | 2026-10-10 15:10:39 | `ef5a18a9bfbb7eff0f5b4196d62a6d78ef8f8b3940f6c5f4e8b39437a9bf074c` |
| `engine.py` | 2026-10-03 20:01:47 | `b81fc093476c7cb70215c9081191922d185ea43b0f670366080e9ff6873def59` |
| `work/fresh/score_fresh_n1.py` | 2026-10-10 15:20:56 | `f614fba1f4c5c754f34d8bfef93e30b690b9e28c5cd75a1d303a4970a68e2615` |
| `work/fresh/score_fresh_hybrid.py` | 2026-10-10 15:19:59 | `887f705ca36308c8b95bb21e5472d610664e5074c612a639e0cfa495d16b5546` |
| `work/finetune/00_datagen_skip.py` | 2026-10-04 06:28:25 | `14a5ef6773913ff58d5edec045751c17ee7b0a3425036c266c202f9f01b05d0a` |
| `work/finetune/01_split.py` | 2026-10-04 06:26:24 | `320cb3a39ec11308c464f8a2a41d9256b259c0c5b554673f94fab66fa6329c37` |
| `work/fresh/policy_frozen.json` | 2026-10-10 15:07:29 | `c4bac593e1f6331344222b1653191d32b0c297593f38c678e5becefd78ba2652` |
| `work/fresh/datagen2000.yaml` | 2026-10-10 15:08:51 | `a4a17551837cbb71d773e3bc6f01ebe1fb3351e0024c8c0ab686e25236ad3597` |
| `work/fresh/datagen10000.yaml` | 2026-10-10 15:08:51 | `5b07b6082d3a8264e0409026043e3fc16abb5c10bcf6c36ba9d431113d166b41` |
| `work/fresh/datagen500.yaml` | 2026-10-10 15:30:28 | `86ead895e56b0d6933a5a1eeec6e57b199dd1e62e1f4fd076a88fcedd4311221` |
| `work/splits/policy_splits.json` | 2026-10-10 14:53:49 | `6c7f1c5f59e33224115bad8bf5ae8348db3cfb02ca2501f26448260a2c53545d` |

Checkpoint files (released GENCO models, downloaded from HF `gridfm/genco-pf-datakit` main; HF revision not pinned, file hashes recorded):

| file | size | sha256 |
|---|---:|---|
| `work/goc10000/pf_case10000_goc_small/config.yaml` | 795 | `c72ada9a651e95d5603375b5f90e484c5847bf9bae264523f6fe4aa26b561ab2` |
| `work/goc10000/pf_case10000_goc_small/model/best_model_state_dict.pt` | 20336211 | `30ea5c941fcc2ea63f52912aba7acd82e33750575467b8231bd0eacfdfac9b9b` |
| `work/goc10000/pf_case10000_goc_small/model/normalizer_stats.pt` | 2162 | `f502f4474627245d7b364e33c6534dcdb1bc5ce4815eb00b6838bbbb92142292` |
| `work/goc10000/pf_case10000_goc_small/released_training.yaml` | 847 | `0a8e8027c7ac9c4c93fc5fd2df7e2102e497f6d38e652010d2c62605fb46817b` |
| `work/goc10000/pf_case10000_goc_small_seed1/config.yaml` | 795 | `c72ada9a651e95d5603375b5f90e484c5847bf9bae264523f6fe4aa26b561ab2` |
| `work/goc10000/pf_case10000_goc_small_seed1/model/best_model_state_dict.pt` | 20336211 | `aa758d28e41da51b36be0bffbc35e9de918eb1795babd5368df53463aefe8c44` |
| `work/goc10000/pf_case10000_goc_small_seed1/model/normalizer_stats.pt` | 2162 | `2270a8e33ae452446efbe349e30a011b5a4fe2adef5dcc23b3aca6c25e027d11` |
| `work/goc10000/pf_case10000_goc_small_seed1/released_training.yaml` | 847 | `3d0db584ea7a1f5c027acc04a65ff810c8ebfd89448b30fdc6415b6ec8079072` |
| `work/goc10000/pf_case10000_goc_tiny/config.yaml` | 795 | `ead2e528f24d67e6238fdf432185d010038208c7f6197429efaf47e3a7f2f6d9` |
| `work/goc10000/pf_case10000_goc_tiny/model/best_model_state_dict.pt` | 5254227 | `c2a0cd5563cd156e917b6809509193b4cc2399629aa55767d96f87075f69109f` |
| `work/goc10000/pf_case10000_goc_tiny/model/normalizer_stats.pt` | 2162 | `f502f4474627245d7b364e33c6534dcdb1bc5ce4815eb00b6838bbbb92142292` |
| `work/goc10000/pf_case10000_goc_tiny/released_training.yaml` | 847 | `0679365a4920bb68c0e991083349206c14aba86d73fa9589a3920360782afed9` |
| `work/goc10000/pf_case10000_goc_tiny_seed1/config.yaml` | 795 | `ead2e528f24d67e6238fdf432185d010038208c7f6197429efaf47e3a7f2f6d9` |
| `work/goc10000/pf_case10000_goc_tiny_seed1/model/best_model_state_dict.pt` | 5254227 | `f6ad9ba7bd948f5fd116c007315cec3708812bfa38bffef979da0aff368e88e4` |
| `work/goc10000/pf_case10000_goc_tiny_seed1/model/normalizer_stats.pt` | 2162 | `2270a8e33ae452446efbe349e30a011b5a4fe2adef5dcc23b3aca6c25e027d11` |
| `work/goc10000/pf_case10000_goc_tiny_seed1/released_training.yaml` | 847 | `fef670ab4e0d91fb1500d6e07975086076afdd8a13fc05b810fbeecf2c5a1c69` |
| `work/goc2000/pf_case2000_goc_small/config.yaml` | 794 | `b2ecd33e9e0ed5657fedf2957c05079e0774b067120638fe53b5cd17671eda2a` |
| `work/goc2000/pf_case2000_goc_small/model/best_model_state_dict.pt` | 20336211 | `a435032ff715367b87c2a2375ec10be54f90a017a2d7cbe03e3ce630f3d3b3ca` |
| `work/goc2000/pf_case2000_goc_small/model/normalizer_stats.pt` | 2162 | `c4137efd78ec74216da9e40a9a56b3fa32c070937a6e7d5e4d1f98b06fb114e0` |
| `work/goc2000/pf_case2000_goc_small/released_training.yaml` | 849 | `7a757fd3def005fed18f9350044912c3c926540d997756a3e2ffd55d59a54688` |
| `work/goc2000/pf_case2000_goc_tiny/config.yaml` | 794 | `9e69bae643f2b2da743a882b640b304442304bec882e34d4b7ac24a546a5a9f7` |
| `work/goc2000/pf_case2000_goc_tiny/model/best_model_state_dict.pt` | 5254227 | `28df537e9a7a28aa82770ddff09523b2aac3ff37318b022da912902ebd428dc0` |
| `work/goc2000/pf_case2000_goc_tiny/model/normalizer_stats.pt` | 2162 | `c4137efd78ec74216da9e40a9a56b3fa32c070937a6e7d5e4d1f98b06fb114e0` |
| `work/goc2000/pf_case2000_goc_tiny/released_training.yaml` | 849 | `c98eaeb0f2ac8a714e9393882d9184ea3f5427babf03fe6defd40ab20c5ae4ea` |
| `work/goc500/pf_case500_goc_tiny/config.yaml` | 793 | `2e9c35b797372ea3ebe37bc63033417c52419492db24e1157d760f6f32c45430` |
| `work/goc500/pf_case500_goc_tiny/model/best_model_state_dict.pt` | 5278611 | `77e6b5dcc4b600d45019e41156495753a5ec678aa9af28b4b8809becc4d85d38` |
| `work/goc500/pf_case500_goc_tiny/model/normalizer_stats.pt` | 2162 | `60796cc6ceed06d141924ec6e627fba6146581ff595d58ae6b7de538bacb29cc` |
| `work/goc500/pf_case500_goc_tiny/released_training.yaml` | 801 | `48764fbedb09aefbe63b9a8c9bc77f34cfdc6022e1d456f93ba4ea348c8e5d7d` |
| `work/goc500/pf_case500_goc_tiny_seed1/config.yaml` | 793 | `2e9c35b797372ea3ebe37bc63033417c52419492db24e1157d760f6f32c45430` |
| `work/goc500/pf_case500_goc_tiny_seed1/model/best_model_state_dict.pt` | 5278611 | `c2ba0a7f60cc3727eb7096806db89099e80b21ccd8b3f920270d724913729fdf` |
| `work/goc500/pf_case500_goc_tiny_seed1/model/normalizer_stats.pt` | 2162 | `95e56b2a066504024a19ad904047d679921611a36230fc5a51f6eb76963e6ef4` |
| `work/goc500/pf_case500_goc_tiny_seed1/released_training.yaml` | 801 | `cde8e5e60aef1a050ee172789750aa9bdfd0b62da22d6bdd211e045498dbcccb` |
| `work/goc500/pf_case500_goc_tiny_seed42/config.yaml` | 793 | `2e9c35b797372ea3ebe37bc63033417c52419492db24e1157d760f6f32c45430` |
| `work/goc500/pf_case500_goc_tiny_seed42/model/best_model_state_dict.pt` | 5278611 | `3518f68d0d84a4afe494d9d0a3d970536b9ae619789b6c99c3e4c2990107eb34` |
| `work/goc500/pf_case500_goc_tiny_seed42/model/normalizer_stats.pt` | 2162 | `b2688b6bc33fba7243c1fed9e464da39caed03ae6b673b6e12841d4df4b227c8` |
| `work/goc500/pf_case500_goc_tiny_seed42/released_training.yaml` | 802 | `3d05b66572f04b9f5f0407d6118a05e0bc6166615b59d4b1e713ca0eb8943d78` |

## Executed code versions appended later (SHA256)

| file | mtime | sha256 | what changed since the record above | used by |
|---|---|---|---|---|
| `analysis_full_n1.py` | 2026-10-10 15:38:58 | `0b57b38f23113d93ddf28cd6573bae53387af2614eb8cffd7953ab74615f2de3` | `e2e` phase rewritten to the full protocol (fixed sample ids, order, warm-up, threads, MPS sync, totals, fallback costs, per-base pre-outage inference, classical-first); `task()`, `run`, chunk format, `workflows()`, `report()` unchanged, so the 24-rank chunks (15:32-16:01) and the six-rank chunks are produced by identical task code | `timing` and `e2e` (16:15-17:21). The 24-rank extension was launched at 15:32:50, before this edit was saved; the file hash recorded here is the later timing/e2e source and is not independent proof of the source the extension process loaded (the `run` phase, `task()` and chunk format were not part of the edit, so the chunks are expected to be identical to what the 15:19 version would produce). |
| `work/fresh/score_fresh_n1.py` | 2026-10-10 15:39:12 | `2a1379f1b9dec7f275eac94ec703675eb2a88d286a9db2cc853ab741b451e76a` | per-base paired tables, six / 18 / 24 sets, e2e attachment; decision logic unchanged (dev-data check still reproduces the earlier development numbers) | `results_n1.md` |
| `work/fresh/score_fresh_hybrid.py` | 2026-10-10 17:53:55 | `b7570f71390cd05977b768a8a2c49f9bb040842ebe022e054da9bc4c62e56c94` | full frozen-policy scoring (loading OR post-step mismatch > tau) added next to the loading-only diagnostic | `results_hybrid.md` |

## Protocol details specified after the freeze (A5, 2026-10-10 ~15:45, before the e2e run at 16:15)

`policy_frozen.json` froze the broad timing parameters (200 outages per base, single process, CPU batch 1, MPS batch 16,
3 repetitions, median and p90 per workflow and gate outcome, exclusions). The following implementation details were
fixed later in code (`analysis_full_n1.py e2e`, recorded in `work/fulln1/e2e_2000_fresh.json` `protocol`) and are
NOT registered parameters: sample ids `default_rng([2026, rank])`, fixed workflow order W0, W1, W4, W4s, gate, CF0.1,
CF0.2, CF0.5, CF1, then W4_mps16; 5 untimed warm-up outages per workflow and base and one untimed MPS block; 6 torch
threads (from `timing_2000.json`); `torch.mps.synchronize()` inside the timer; min-of-3 latency plus first-pass totals
and contiguous pass wall; partial MPS blocks timed as is. The timed continuation allows one screening step plus 30
further iterations, whereas the saved sweeps allow 30 in total; this adds one iteration per failed case (4 of 4,800)
and changes no verdict.

## Corrections after an internal audit (2026-10-10 evening)

- The hybrid scorer's decision was loading-only (`pred > c`), not the complete frozen policy (loading OR post-step
  mismatch > tau). Both are now reported, labelled; the frozen JSON is not narrowed; tau values are the old GOC 2000
  N-1 calibration transferred unchanged to random multi-outage states and to GOC 10000.
- The "highest-load base" explanation of the lower fresh miss count was wrong (old misses sat at 35.6 GW, below the
  fresh maximum 37.8 GW) and is withdrawn; the caveat is reduced upper-load coverage and a different stress mix.
- No policy value changed.
- Pilot (`work/equiv`): wording corrected after a further internal audit (tolerance, not invariance; loading-only decisions; measured first-step / recovery behaviour instead of a 'basin' claim); SPEC, thresholds and results unchanged. GOC 500 fresh pool eligibility = successfully generated cases (load id 6353 failed OPF, LOCALLY_INFEASIBLE).

Fresh pools: `fresh_ids.json` (written by the chains before scoring), disjointness logs `logs/disjoint*.log`, generation logs `logs/datagen*.log`.


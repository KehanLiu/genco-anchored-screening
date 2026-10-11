#!/bin/sh
# Rebuild the results and figures of the entry.
#
#   ./reproduce.sh                  no argument: print this usage and exit 1
#   ./reproduce.sh all              default path: setup, fetch, tests, analysis, figures - under an hour, mostly downloads
#   RUN_LONG=1 ./reproduce.sh all   recompute every result table first (sweeps, controls, finetune): about two days of
#                                   laptop time; a wall-time estimate and the log it was read from stand next to every
#                                   long command
#
# What runs from a clean checkout and what does not:
#   setup, fetch, tests         need only this repository, the network, and the Texas case file (see below).
#   sweeps, controls, finetune  [long] need the fetched inputs only; they recompute every result table from scratch.
#   analysis, figures           need the SAVED RESULT TABLES (work/**/*.parquet, work/**/*.csv, work/**/*.log; not in
#                               git). Only the long stages above produce them; no results archive is available yet.
#                               On a clean checkout without RUN_LONG=1 these two stages fail.
#
# Replay status: the default path was last replayed in a fresh copy of the repository in round 2 (2026-10-03). Later
# additions were syntax- and import-checked in a simulated clean checkout on 2026-10-11 (74 files compile, 49 tests
# pass, 15 skip for missing data); no full clean replay of the current script has been done.
#
# Single stages:
#   ./reproduce.sh setup      Python 3.12 environment via uv (Julia is installed by juliacall on first use)
#   ./reproduce.sh fetch      GOC networks and the two pglib files of the zero-shot study (PGLib-OPF v23.07), released
#                             GENCO checkpoints and the IEEE 118 data shard (Hugging Face, by commit sha), the Texas model
#                             config; writes work/provenance.txt and verifies the sha256 manifest, stopping on the first
#                             mismatch or missing network file
#   ./reproduce.sh tests      pytest on the gridtrust package and on work/equiv (64 tests; some skip without data)
#   ./reproduce.sh sweeps     [long] grid states with AC, DC and GENCO: datakit generation, GENCO + Newton sweeps
#   ./reproduce.sh controls   [long] pre-outage start, hybrid start, full N-1, solver timing, reactive limits, zero-shot
#   ./reproduce.sh finetune   [long] fine-tuning on GOC 500, 2000 and 10000 (AC power + caffeinate required, see below)
#   ./reproduce.sh analysis   every table of the write-up, from the saved result tables (minutes)
#   ./reproduce.sh figures    figures/*.png from the saved results
#   ./reproduce.sh checksums  verify the sha256 of every fetched input that is present (fetch runs it too)
#   ./reproduce.sh all        the default path (setup, fetch, tests, analysis, figures), plus the three long stages when RUN_LONG=1
#
# Not downloaded by this script: the Texas 2000 case file. Place Texas2k_case1_2016summerpeak.m in work/texas/ first
# (README, "Data"); its sha256 is in the manifest below. Wall times below were read from the logs in work/
# (Apple-silicon laptop, October 2026).
set -eu
cd "$(dirname "$0")"
PY=.venv/bin/python
TMO=work/texas/tmo           # tmo SECONDS cmd...: kills the command after SECONDS
HF=https://huggingface.co
# ---------------------------------------------------------------------------------------------------------------------
# Upstream revisions (commit shas recorded 2026-10-10). What this pins and what it verifies:
#   by commit sha   every Hugging Face download: the curl calls below resolve the sha directly; the Python fetchers
#                   (run_pretrained.py, which work/goc*/01_fetch.py and work/goc500/01_eval.py call) receive the same
#                   shas through HF_REV_DATAKIT and HF_REV_PF_SMALL_CASE118_IEEE and pass them as revision= to every
#                   hf_hub_download / list_repo_files call. PGLib-OPF by tag, the Texas config by commit.
#   by sha256       every file in the manifest of checksums() below: the three GOC case files, the two pglib files of
#                   the zero-shot study (case118_ieee, case1354_pegase: fetch downloads them from the v23.07 tag into
#                   gridfm_datakit's package folder, the path its loader tests first, so datakit's own downloader from
#                   the master branch never runs), the Texas case file, every checkpoint, normalizer, training yaml and
#                   config, and the three IEEE 118 data shards (scenario_partition=0 of bus/gen/branch). The model and
#                   shard hashes equal the LFS object ids the Hugging Face API reports for these shas.
#   guarded         checksums() runs at the end of fetch and again, as guard(), at the start of sweeps and finetune and
#                   right before the zero-shot generation in controls; the two pglib files are REQUIRED there (absent =
#                   failure), every other manifest entry still reports absent without failing.
#   not verified    the remaining files of the dataset repository (y_bus, runtime, logs) are not downloaded; the
#                   Python packages are pinned by requirements.lock.txt, not hashed.
#   A clean-checkout replay of the fetch and checksums stages has not been performed; checksums was run on the working copy.
# ---------------------------------------------------------------------------------------------------------------------
PGLIB_TAG=v23.07                                                   # = commit dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3
PGLIB=https://raw.githubusercontent.com/power-grid-lib/pglib-opf/$PGLIB_TAG
REV_DATAKIT=7014f145635f26fa29e9e003f0dc33f689d615da               # gridfm/genco-pf-datakit      (2026-10-01)
REV_CONTINGENCY=5c625b00f5b79c8a7d4d27ecc133d4ecdf573af7           # gridfm/genco-pf-contingency  (2026-10-01)
REV_TRANSFER=e7a98e99221ba8241385e491787aedb0940d5130              # gridfm/genco-pf-transfer     (2026-10-01)
REV_DATASET_118=3f5a6e266f759be15e3777bf4d70133ff40e6fa7           # gridfm/pf_small_case118_ieee (dataset, 2026-10-01; shards unchanged since 2026-01-26)
export HF_REV_DATAKIT=$REV_DATAKIT HF_REV_PF_SMALL_CASE118_IEEE=$REV_DATASET_118   # read by run_pretrained.py
REV_GRAPHKIT=95b90eecc7f5c2c86a9dd776f0c3f40434def22f              # gridfm/gridfm-graphkit, branch genco-paper-repro (2026-10-01)
GRAPHKIT=https://raw.githubusercontent.com/gridfm/gridfm-graphkit/$REV_GRAPHKIT
TEXAS=work/texas/pf_Texas2k_case1_2016summerpeak_base

setup() {
    uv venv --python 3.12 .venv
    uv pip install --python $PY -r requirements.lock.txt
    uv pip install --python $PY pytest==9.1.1      # not in the lock file; the tests stage needs it
}

hf_head() {  # hf_head models|datasets REPO: current commit of a Hugging Face repository (empty when offline)
    curl -sf $HF/api/$1/$2 | python3 -I -c 'import json, sys; print(json.load(sys.stdin).get("sha", ""))' 2>/dev/null || true
}

grids_dir() {  # gridfm_datakit's package folder gridfm_datakit/grids (empty when the package is not installed)
    # This is where gridfm_datakit.network.get_pglib_file_path() looks for pglib_opf_<name>.m (via
    # resources.files("gridfm_datakit.grids")) and only downloads from PGLib master when the file is missing;
    # engine.py (load_net_from_pglib) and the zero-shot datagen yamls (network.source: pglib) both go through it.
    $PY -c 'import gridfm_datakit.grids as g, os; print(os.path.dirname(g.__file__))' 2>/dev/null || true
}

fetch() {
    # ---------------------------------------------------------------------------------------------------------
    # GOC networks: PGLib-OPF, tag v23.07 (the files used for the write-up are byte-identical to this tag).
    # Saved WITHOUT the pglib_opf_ prefix, because the datagen yamls use network.source: file,
    # network_dir: work/goc<N>/net, name: case<N>_goc. gridfm_datakit writes case<N>_goc_corrected.m next to it
    # on first use.
    # ---------------------------------------------------------------------------------------------------------
    for n in 500 2000 10000; do
        mkdir -p work/goc$n/net
        [ -f work/goc$n/net/case${n}_goc.m ] || curl -L -o work/goc$n/net/case${n}_goc.m $PGLIB/pglib_opf_case${n}_goc.m
    done
    # ---------------------------------------------------------------------------------------------------------
    # IEEE 118 and case1354_pegase (zero-shot study, work/zeroshot/datagen_*.yaml with network.source: pglib; IEEE 118
    # also through engine.py for every GENCO run on that grid). gridfm_datakit would download these itself from the
    # MASTER branch of the PGLib repository the first time they are used, with no pin and after the fetch-time check.
    # Prefetching them from the pinned tag into the exact path its loader tests first (see grids_dir) bypasses that
    # downloader; checksums() then requires them. Datakit writes pglib_opf_<name>_corrected.m next to each on first
    # use (derived, not in the manifest). The copies used for the write-up are byte-identical to the v23.07 tag.
    # ---------------------------------------------------------------------------------------------------------
    GRIDS=$(grids_dir)
    [ -n "$GRIDS" ] || { echo "gridfm_datakit is not installed; run ./reproduce.sh setup first" >&2; exit 1; }
    mkdir -p "$GRIDS"
    for c in case118_ieee case1354_pegase; do
        [ -f "$GRIDS/pglib_opf_$c.m" ] || curl -L -o "$GRIDS/pglib_opf_$c.m" $PGLIB/pglib_opf_$c.m
    done
    # ---------------------------------------------------------------------------------------------------------
    # Released GENCO checkpoints (model weights, normalizer statistics, training yaml).
    # ---------------------------------------------------------------------------------------------------------
    # IEEE 118, GENCO Base, seeds 0, 1 and 42 (gridfm/genco-pf-datakit, case118_ieee/base/seed<S>/) + the first data
    # shard of gridfm/pf_small_case118_ieee at $REV_DATASET_118 (bus/gen/branch, scenario_partition=0; in the manifest)
    $PY run_pretrained.py --network case118_ieee --variant base
    $PY run_pretrained.py --network case118_ieee --variant base --seed 1
    $PY run_pretrained.py --network case118_ieee --variant base --seed 42
    # GOC 500, GENCO Tiny, seeds 0, 1 and 42 (gridfm/genco-pf-datakit, case500_goc/tiny/seed<S>/) -> work/goc500/pf_case500_goc_tiny[_seed<S>]/
    for seed in 0 1 42; do
        $PY -c "import sys, importlib; sys.path[:0] = ['work/goc500', '.']; importlib.import_module('01_eval').fetch($seed)"
    done
    # GOC 2000, Tiny and Small, seed 0; GOC 10000, Tiny and Small, seeds 0 and 1 (case<N>_goc/<variant>/seed<S>/) -> work/goc<N>/pf_case<N>_goc_<variant>[_seed1]/
    for v in tiny small; do
        $PY work/goc2000/01_fetch.py --variant $v --seed 0
        for seed in 0 1; do $PY work/goc10000/01_fetch.py --variant $v --seed $seed; done
    done
    # Texas 2000 (gridfm/genco-pf-contingency, model/training_bs_16/) -> work/texas/pf_Texas2k_case1_2016summerpeak_base/model/
    # The model config that engine.py reads (pf_<network>_<variant>/config.yaml) is the evaluation config of the GENCO
    # paper's contingency scripts, scripts/contingency/configs/caseTexas_eval.yaml in gridfm-graphkit (branch
    # genco-paper-repro, linked from the repository card of gridfm/genco-pf-contingency); the copy used for the write-up
    # is byte-identical to it (sha256 in the manifest). It is only read for the model block and the normalizer.
    mkdir -p $TEXAS/model
    for f in best_model_state_dict.pt normalizer_stats.pt; do
        [ -f $TEXAS/model/$f ] || curl -L -o $TEXAS/model/$f $HF/gridfm/genco-pf-contingency/resolve/$REV_CONTINGENCY/model/training_bs_16/$f
    done
    [ -f $TEXAS/config.yaml ] || curl -L -o $TEXAS/config.yaml $GRAPHKIT/scripts/contingency/configs/caseTexas_eval.yaml
    # Pretrained transfer model for the zero-shot study (gridfm/genco-pf-transfer, folder pretraining/) -> work/pretrained/
    mkdir -p work/pretrained
    for f in best_model_state_dict.pt HGNS_PreTrain_subnets_100.yaml train.sh; do
        [ -f work/pretrained/$f ] || curl -L -o work/pretrained/$f $HF/gridfm/genco-pf-transfer/resolve/$REV_TRANSFER/pretraining/$f
    done
    # ---------------------------------------------------------------------------------------------------------
    # Provenance record and integrity check.
    # ---------------------------------------------------------------------------------------------------------
    {
        echo "written by reproduce.sh fetch on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "pglib-opf: tag $PGLIB_TAG (commit dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3) for the GOC files and for case118_ieee and case1354_pegase (prefetched into gridfm_datakit/grids; datakit's master-branch downloader is bypassed)"
        echo "gridfm/genco-pf-datakit: pinned $REV_DATAKIT, main now $(hf_head models gridfm/genco-pf-datakit)"
        echo "gridfm/genco-pf-contingency: pinned $REV_CONTINGENCY, main now $(hf_head models gridfm/genco-pf-contingency)"
        echo "gridfm/genco-pf-transfer: pinned $REV_TRANSFER, main now $(hf_head models gridfm/genco-pf-transfer)"
        echo "gridfm/pf_small_case118_ieee (dataset): pinned $REV_DATASET_118, main now $(hf_head datasets gridfm/pf_small_case118_ieee)"
        echo "gridfm/gridfm-graphkit (Texas config): commit $REV_GRAPHKIT, branch genco-paper-repro"
        echo "gridfm_datakit $($PY -c 'import importlib.metadata as m; print(m.version("gridfm_datakit"))' 2>/dev/null || echo '(not installed)')"
        echo
        echo "sha256 of the fetched inputs present on disk:"
    } > work/provenance.txt
    # Run checksums on its own and keep its exit status: `checksums | tee` would let a mismatch pass under /bin/sh,
    # because set -e only sees the status of tee.
    tmp=$(mktemp)
    if ! checksums > "$tmp"; then
        cat "$tmp" >> work/provenance.txt; cat "$tmp"; rm -f "$tmp"
        echo "checksum mismatch, stopping" >&2
        exit 1
    fi
    cat "$tmp" >> work/provenance.txt; cat "$tmp"; rm -f "$tmp"
}

checksums() {
    # Every input the entry depends on, with the sha256 of the copy used for the write-up. Files not yet fetched or
    # that the user supplies (Texas case file) are reported as absent, not failed - except the two pglib files in
    # REQUIRED, which fetch prefetches and which gridfm_datakit would otherwise download unpinned at generation time:
    # for them absent counts as failure, so a replay cannot generate from an unverified copy.
    # Model and data-shard hashes equal the LFS object ids reported by the Hugging Face API for the revisions above.
    # Returns 1 on any mismatch or missing required file (the caller decides how to stop; as a stage, set -e ends the
    # script with status 1).
    SHA="shasum -a 256"; command -v shasum >/dev/null 2>&1 || SHA=sha256sum
    GRIDS=$(grids_dir); [ -n "$GRIDS" ] || GRIDS=work/.no_datakit
    REQUIRED="GRIDS/pglib_opf_case118_ieee.m GRIDS/pglib_opf_case1354_pegase.m"
    status=0
    while read -r sum path; do
        case "$path" in '') continue ;; esac
        label=$path; path=$(printf '%s' "$path" | sed "s#^GRIDS/#$GRIDS/#")   # GRIDS/ = gridfm_datakit's package folder
        if [ ! -f "$path" ]; then
            case " $REQUIRED " in
                *" $label "*) echo "MISSING  $label (required; run ./reproduce.sh fetch)"; status=1 ;;
                *) echo "absent   $label" ;;
            esac
            continue
        fi
        if echo "$sum  $path" | $SHA -c - >/dev/null 2>&1; then echo "ok       $label"; else echo "MISMATCH $label"; status=1; fi
    done <<EOF
36c298d571605019ef16c17dd74680adca1386d91ed47d69a0d909aebc90a1b6  work/goc500/net/case500_goc.m
af6cec27709da1f952c330e92b4eb07e0bc1673d3dc0c2e70c7d6c96a38cca6b  work/goc2000/net/case2000_goc.m
8387f73e8c135938c60e41538dfbb6b4cb58d37738553fb8a36c1e1647a66e7b  work/goc10000/net/case10000_goc.m
b1af0833849040c04babc3700631cff0d9afa66b79c5d3e13ae79bdf516cec78  GRIDS/pglib_opf_case118_ieee.m
cd6d27dff4a56684f1e4f82cfa346b36d84c4e90733228aa88331cd550e17652  GRIDS/pglib_opf_case1354_pegase.m
3744f75be7ca85456eb4ff2f365ebe0fdca78c8738ba8cce5506647207f0eb9d  work/texas/Texas2k_case1_2016summerpeak.m
337866ec2fa776519d51749e5778f8079aa64ace374f8243bf89655b850da5b7  work/pf_case118_ieee_base/data/case118_ieee/raw/bus_data.parquet/scenario_partition=0/c625d1dc0af24c64b7a79973b0197cdb-0.parquet
97e4625cb8068d4eb9dedc5ae090c0b71fac1050d91bc0f98b0f5e852075c756  work/pf_case118_ieee_base/data/case118_ieee/raw/gen_data.parquet/scenario_partition=0/b03d577113bf4767b5c12653a048f665-0.parquet
5aed0bbff009233e17e49c2925541ce0fc8cb5536a6b3bd0650d5565462b10d6  work/pf_case118_ieee_base/data/case118_ieee/raw/branch_data.parquet/scenario_partition=0/30eaa20a42244f908f4adfd2b2b539b8-0.parquet
337866ec2fa776519d51749e5778f8079aa64ace374f8243bf89655b850da5b7  work/pf_case118_ieee_base_seed1/data/case118_ieee/raw/bus_data.parquet/scenario_partition=0/c625d1dc0af24c64b7a79973b0197cdb-0.parquet
97e4625cb8068d4eb9dedc5ae090c0b71fac1050d91bc0f98b0f5e852075c756  work/pf_case118_ieee_base_seed1/data/case118_ieee/raw/gen_data.parquet/scenario_partition=0/b03d577113bf4767b5c12653a048f665-0.parquet
5aed0bbff009233e17e49c2925541ce0fc8cb5536a6b3bd0650d5565462b10d6  work/pf_case118_ieee_base_seed1/data/case118_ieee/raw/branch_data.parquet/scenario_partition=0/30eaa20a42244f908f4adfd2b2b539b8-0.parquet
337866ec2fa776519d51749e5778f8079aa64ace374f8243bf89655b850da5b7  work/pf_case118_ieee_base_seed42/data/case118_ieee/raw/bus_data.parquet/scenario_partition=0/c625d1dc0af24c64b7a79973b0197cdb-0.parquet
97e4625cb8068d4eb9dedc5ae090c0b71fac1050d91bc0f98b0f5e852075c756  work/pf_case118_ieee_base_seed42/data/case118_ieee/raw/gen_data.parquet/scenario_partition=0/b03d577113bf4767b5c12653a048f665-0.parquet
5aed0bbff009233e17e49c2925541ce0fc8cb5536a6b3bd0650d5565462b10d6  work/pf_case118_ieee_base_seed42/data/case118_ieee/raw/branch_data.parquet/scenario_partition=0/30eaa20a42244f908f4adfd2b2b539b8-0.parquet
3ea2f552b190541fe08e0e27604daa5f122232af411f8bfd9bf355a557cab065  work/pf_case118_ieee_base/model/best_model_state_dict.pt
01c93104fc47fbe4f7492d2926ad60fe7395f10f77f7788b2d71504ff6a68411  work/pf_case118_ieee_base/model/normalizer_stats.pt
2907838715518ed28278d18f31db50cf742c4e11987c52b973e0736c73c50daf  work/pf_case118_ieee_base_seed1/model/best_model_state_dict.pt
651ebfaababef72213ef66a8981062205a24722acaf6955a9ebb66cfab3b268e  work/pf_case118_ieee_base_seed1/model/normalizer_stats.pt
2706a140330159b357fbf6d355baf5ac5d5a16eeda07920a6452b57945e65a59  work/pf_case118_ieee_base_seed42/model/best_model_state_dict.pt
ae2b9b7732c24b7e92dcf2e637d627fd2a55cde56b892afa2f2b24f2d7d9e9ec  work/pf_case118_ieee_base_seed42/model/normalizer_stats.pt
77e6b5dcc4b600d45019e41156495753a5ec678aa9af28b4b8809becc4d85d38  work/goc500/pf_case500_goc_tiny/model/best_model_state_dict.pt
60796cc6ceed06d141924ec6e627fba6146581ff595d58ae6b7de538bacb29cc  work/goc500/pf_case500_goc_tiny/model/normalizer_stats.pt
48764fbedb09aefbe63b9a8c9bc77f34cfdc6022e1d456f93ba4ea348c8e5d7d  work/goc500/pf_case500_goc_tiny/released_training.yaml
c2ba0a7f60cc3727eb7096806db89099e80b21ccd8b3f920270d724913729fdf  work/goc500/pf_case500_goc_tiny_seed1/model/best_model_state_dict.pt
95e56b2a066504024a19ad904047d679921611a36230fc5a51f6eb76963e6ef4  work/goc500/pf_case500_goc_tiny_seed1/model/normalizer_stats.pt
cde8e5e60aef1a050ee172789750aa9bdfd0b62da22d6bdd211e045498dbcccb  work/goc500/pf_case500_goc_tiny_seed1/released_training.yaml
3518f68d0d84a4afe494d9d0a3d970536b9ae619789b6c99c3e4c2990107eb34  work/goc500/pf_case500_goc_tiny_seed42/model/best_model_state_dict.pt
b2688b6bc33fba7243c1fed9e464da39caed03ae6b673b6e12841d4df4b227c8  work/goc500/pf_case500_goc_tiny_seed42/model/normalizer_stats.pt
3d05b66572f04b9f5f0407d6118a05e0bc6166615b59d4b1e713ca0eb8943d78  work/goc500/pf_case500_goc_tiny_seed42/released_training.yaml
28df537e9a7a28aa82770ddff09523b2aac3ff37318b022da912902ebd428dc0  work/goc2000/pf_case2000_goc_tiny/model/best_model_state_dict.pt
c4137efd78ec74216da9e40a9a56b3fa32c070937a6e7d5e4d1f98b06fb114e0  work/goc2000/pf_case2000_goc_tiny/model/normalizer_stats.pt
c98eaeb0f2ac8a714e9393882d9184ea3f5427babf03fe6defd40ab20c5ae4ea  work/goc2000/pf_case2000_goc_tiny/released_training.yaml
a435032ff715367b87c2a2375ec10be54f90a017a2d7cbe03e3ce630f3d3b3ca  work/goc2000/pf_case2000_goc_small/model/best_model_state_dict.pt
c4137efd78ec74216da9e40a9a56b3fa32c070937a6e7d5e4d1f98b06fb114e0  work/goc2000/pf_case2000_goc_small/model/normalizer_stats.pt
7a757fd3def005fed18f9350044912c3c926540d997756a3e2ffd55d59a54688  work/goc2000/pf_case2000_goc_small/released_training.yaml
c2a0cd5563cd156e917b6809509193b4cc2399629aa55767d96f87075f69109f  work/goc10000/pf_case10000_goc_tiny/model/best_model_state_dict.pt
f502f4474627245d7b364e33c6534dcdb1bc5ce4815eb00b6838bbbb92142292  work/goc10000/pf_case10000_goc_tiny/model/normalizer_stats.pt
0679365a4920bb68c0e991083349206c14aba86d73fa9589a3920360782afed9  work/goc10000/pf_case10000_goc_tiny/released_training.yaml
f6ad9ba7bd948f5fd116c007315cec3708812bfa38bffef979da0aff368e88e4  work/goc10000/pf_case10000_goc_tiny_seed1/model/best_model_state_dict.pt
2270a8e33ae452446efbe349e30a011b5a4fe2adef5dcc23b3aca6c25e027d11  work/goc10000/pf_case10000_goc_tiny_seed1/model/normalizer_stats.pt
fef670ab4e0d91fb1500d6e07975086076afdd8a13fc05b810fbeecf2c5a1c69  work/goc10000/pf_case10000_goc_tiny_seed1/released_training.yaml
30ea5c941fcc2ea63f52912aba7acd82e33750575467b8231bd0eacfdfac9b9b  work/goc10000/pf_case10000_goc_small/model/best_model_state_dict.pt
f502f4474627245d7b364e33c6534dcdb1bc5ce4815eb00b6838bbbb92142292  work/goc10000/pf_case10000_goc_small/model/normalizer_stats.pt
0a8e8027c7ac9c4c93fc5fd2df7e2102e497f6d38e652010d2c62605fb46817b  work/goc10000/pf_case10000_goc_small/released_training.yaml
aa758d28e41da51b36be0bffbc35e9de918eb1795babd5368df53463aefe8c44  work/goc10000/pf_case10000_goc_small_seed1/model/best_model_state_dict.pt
2270a8e33ae452446efbe349e30a011b5a4fe2adef5dcc23b3aca6c25e027d11  work/goc10000/pf_case10000_goc_small_seed1/model/normalizer_stats.pt
3d0db584ea7a1f5c027acc04a65ff810c8ebfd89448b30fdc6415b6ec8079072  work/goc10000/pf_case10000_goc_small_seed1/released_training.yaml
86124547e21dd380d654495a073c493a11806651d4556036c1f62e8f5b4ceccf  work/texas/pf_Texas2k_case1_2016summerpeak_base/model/best_model_state_dict.pt
285ca02ce53cdb09d8ced26da719c077f0615517498d3d231367d23baba637e6  work/texas/pf_Texas2k_case1_2016summerpeak_base/model/normalizer_stats.pt
1aca58a53fd4a95d6e8eb620987be8a43935a0d62d8af3ef7268874266444698  work/texas/pf_Texas2k_case1_2016summerpeak_base/config.yaml
b2c9cf9b1007e24b924c83f440e57d252d500df126c4a4f11fba10c1762f5d41  work/pretrained/best_model_state_dict.pt
31db2e8b86995c66c25f09c890626915301b1d8a62cbc44d929854d8914fe356  work/pretrained/HGNS_PreTrain_subnets_100.yaml
bee30328d8b2b26e24baadaa26c05c343f00ea3e05c8c7323140fb76737eee87  work/pretrained/train.sh
EOF
    [ $status = 0 ] || { echo "checksum failure: a fetched input differs from the one used for the write-up, or a required network file is missing"; return 1; }
}

guard() {
    # Called at the start of every generation stage: refuses to run unless the manifest passes, in particular unless the
    # two prefetched pglib files are present and match (without them gridfm_datakit would download from master).
    tmp=$(mktemp)
    if ! checksums > "$tmp"; then
        grep -v '^ok ' "$tmp"; rm -f "$tmp"
        echo "inputs not verified; run ./reproduce.sh fetch first, stopping" >&2
        exit 1
    fi
    rm -f "$tmp"
}

tests() {
    $PY -m pytest gridtrust/tests work/equiv -q
}

sweeps() {
    guard
    mkdir -p work/texas/logs work/goc500/logs work/goc2000/logs work/goc10000/logs
    # outage sweeps: N-0, N-1, random N-2 at five load levels (hours each)
    $TMO 7200 $PY -u analysis/analysis_hybrid.py --network case118_ieee --variant base
    $TMO 7200 $PY -u analysis/analysis_hybrid.py --network case118_ieee --variant base_seed1
    $TMO 7200 $PY -u analysis/analysis_hybrid.py --network case118_ieee --variant base_seed42
    $TMO 7200 $PY -u work/texas/04_hybrid.py --single 150 --double 450 --tag _n2x450
    # voltages, Newton start points, shifted test sets
    $TMO 7200 $PY -u analysis/analysis_voltage.py --grid both
    $TMO 7200 $PY -u analysis/analysis_warmstart.py --grid 118
    $TMO 9000 $PY -u analysis/analysis_warmstart.py --grid texas
    $TMO 9000 $PY -u analysis/analysis_shift.py
    # the example entry's setup: GOC 500, up to five branches removed.
    # generation 11 min (work/goc500/logs/datagen.log); per seed: GENCO 2 min (logs/eval_seed0.log), Newton 5 min (logs/newton_seed0.log)
    $TMO 7200 .venv/bin/gridfm_datakit generate work/goc500/datagen.yaml
    for seed in 0 1 42; do
        $TMO 7200 $PY -u work/goc500/01_eval.py --seed $seed
        $TMO 7200 $PY -u work/goc500/02_newton.py --seed $seed
    done
    # 2,000 and 10,000 buses. Generation runs for both grids in parallel.
    # GOC 2000: the evaluation pool is load scenarios 0..499 (2,494 states), i.e. the FIRST LARGE CHUNK of the 1,000
    # that work/goc2000/datagen.yaml asks for. The original run saved that chunk after 2 h 52 min (tqdm line in
    # work/goc2000/logs/datagen.log) and was then killed by its 7,200 s timeout; hours 500..999 were never solved.
    # 00_datagen_first.py reproduces exactly that chunk (same load profile, chunk boundaries and per-chunk seeds, see
    # its docstring) and stops. It is NOT equivalent to load.scenarios: 500, which would redraw the per-bus noise of
    # every hour, and a plain `gridfm_datakit generate` of the yaml would compute all 1,000 and change the population
    # that every GOC 2000 table (Newton sweeps, pre-outage and hybrid starts, reactive limits, zero-shot, the Round-4
    # evaluations) and the 24 base cases of the full N-1 (work/fresh/README.md) are drawn from.
    # GOC 10000: 3 h 08 min for its 200 requested load scenarios, 199 solved (work/goc10000/logs/datagen.log).
    # run_all.sh waits for both, then runs the Newton sweeps (Tiny / Small: 161 / 180 s on GOC 2000, 392 / 401 s on
    # GOC 10000; work/goc<N>/logs/newton_<variant>_seed0.log), the per-grid analysis and the timing passes.
    ( $TMO 80000 $PY work/goc2000/00_datagen_first.py work/goc2000/datagen.yaml 500 > work/goc2000/logs/datagen.log 2>&1 ; echo total >> work/goc2000/logs/datagen.log ) &
    ( $TMO 80000 .venv/bin/gridfm_datakit generate work/goc10000/datagen.yaml > work/goc10000/logs/datagen.log 2>&1 ; echo total >> work/goc10000/logs/datagen.log ) &
    zsh work/goc2000/run_all.sh
    # embeddings and remedial actions
    $TMO 9000 $PY -u analysis/analysis_embedding.py extract
    $TMO 3000 $PY -u analysis/analysis_remedial.py run --budget 2400
}

controls() {
    mkdir -p work/fulln1 work/timing work/qlim work/zeroshot/logs
    # pre-outage start and hybrid start, GENCO Tiny seed 0, 4 processes: 69 / 32 / 51 s and 143 / 64 / 105 s wall for
    # GOC 500 / 2000 / 10000 (work/prev_control.log, work/hybrid_control.log)
    for grid in 500 2000 10000; do
        $TMO 3000 $PY -u analysis/analysis_prev_control.py run --grid $grid --procs 4
        $TMO 3000 $PY -u analysis/analysis_hybrid_start.py run --grid $grid --procs 4
    done
    # hybrid start with the other GOC 10000 checkpoints: 167 / 185 / 93 s wall (work/goc10000/logs/hybrid_small_seed0.log,
    # hybrid_small_seed1.log, hybrid_tiny_seed1.log)
    $TMO 3000 $PY -u analysis/analysis_hybrid_start.py run --grid 10000 --variant small --seed 0 --procs 4
    $TMO 3000 $PY -u analysis/analysis_hybrid_start.py run --grid 10000 --variant small --seed 1 --procs 4
    $TMO 3000 $PY -u analysis/analysis_hybrid_start.py run --grid 10000 --variant tiny --seed 1 --procs 4
    # full N-1 on GOC 2000: 24 base cases x 3,188 single-branch outages = 76,512 contingencies, 1,660 s wall with 4 processes
    # (work/fulln1/run_2000.out); resumable from its chunk files. The timing pass is single-process, minutes, quiet machine.
    $TMO 9000 $PY -u analysis/analysis_full_n1.py run --grid 2000 --procs 4
    $TMO 600 $PY -u analysis/analysis_full_n1.py timing --grid 2000
    # start points inside PowerModels' AC power flow: about 25 min for both grids (work/timing/run_10000.out and the file
    # times of work/timing/*.csv); run with nothing else on the machine, the numbers are timings
    $TMO 900  $PY -u analysis/analysis_solver_timing.py prep --grid 2000 --n 200
    $TMO 1800 $PY -u analysis/analysis_solver_timing.py solve --grid 2000
    $TMO 900  $PY -u analysis/analysis_solver_timing.py chain --grid 2000
    $TMO 900  $PY -u analysis/analysis_solver_timing.py prep --grid 10000 --n 100
    $TMO 1800 $PY -u analysis/analysis_solver_timing.py solve --grid 10000
    $TMO 900  $PY -u analysis/analysis_solver_timing.py chain --grid 10000
    # generator reactive limits, 3 processes: 308 s (GOC 500), 182 s (GOC 2000), 146 s (IEEE 118 voltages)
    # (work/qlim/run_goc500.out, run_goc2000.out, run_volt.out)
    $TMO 3000 $PY -u analysis/analysis_qlimits.py run --grid 500 --procs 3
    $TMO 3000 $PY -u analysis/analysis_qlimits.py run --grid 2000 --procs 3
    $TMO 3000 $PY -u analysis/analysis_qlimits.py volt
    # zero-shot transfer: datakit scenarios for IEEE 118 and case1354_pegase (work/zeroshot/logs/datagen_*.log) from the
    # two pglib files that fetch prefetched and guard re-verifies here, then 123 / 220 / 177 / 572 s wall for
    # 118 / 500 / 1354 / 2000 with 3 processes (work/zeroshot/report.log)
    guard
    $TMO 7200 .venv/bin/gridfm_datakit generate work/zeroshot/datagen_case118_ieee.yaml
    $TMO 7200 .venv/bin/gridfm_datakit generate work/zeroshot/datagen_case1354_pegase.yaml
    for grid in 118 500 1354 2000; do
        $TMO 5000 $PY -u analysis/analysis_zeroshot.py run --grid $grid --procs 3
    done
}

finetune() {
    guard
    # Every training goes through work/finetune/run_train.sh, which refuses to start unless the laptop is on AC power with
    # a sleep-prevention assertion active (run `caffeinate -i` in another terminal first; the lid may be closed). Device
    # mps = Apple GPU. The pipelines are resumable and skip finished stages; 2000 and 10000 queue behind a running training.
    # Data (`time` lines in work/finetune/logs/datagen500.log, datagen2000.log, datagen10000.log): GOC 500 generation
    # 17 min, GOC 2000 generation 9.5 min, GOC 10000 generation 55 min; splits and graphkit preprocessing add minutes.
    zsh work/finetune/00_datagen500.sh
    zsh work/finetune/00_chain_data.sh
    zsh work/finetune/00_datagen10000.sh
    # Training and evaluation with the settings of the write-up (START/END lines in work/finetune/logs/LOG.txt):
    #   GOC 500   20,000 steps, lr 1e-4: 2 fine-tuned seeds at about 60 min each, 4 active-learning rounds at about 53 min
    #             each, plus evaluations; the whole pipeline took 5 h 45 min
    #   GOC 2000  6,000 steps, lr 1e-4, training seeds 0 and 1: 34 min each; the whole pipeline took 1 h 15 min
    #   GOC 10000 20,000 steps, lr 1e-4, seed 0: training 1 h 57 min; the whole pipeline took 2 h 10 min
    # The 2000 and 10000 pipelines evaluate every checkpoint from both the GENCO start (02_eval.py ->
    # results/lines_newton_<N>_*.parquet) and the hybrid start (04_hybrid_eval.py -> results/lines_hybrid_<N>_*.parquet,
    # released baseline included). The 500 pipeline only runs the GENCO-start evaluation; its hybrid-start tables, which
    # the analysis stage reads, are produced by the three 04_hybrid_eval.py calls after it.
    zsh work/finetune/20_pipeline500.sh 20000 mps 0.0001
    # hybrid start on GOC 500: released Tiny seed 0 (reproduces work/goc500/lines_hybrid_tiny_seed0.parquet column for
    # column) and the two fine-tuned seeds, 3 processes: 178 / 165 s wall for ft_seed0 / ft_seed1
    # (work/finetune/logs/hybrid_500_ft_seed0.log, hybrid_500_ft_seed1.log). Resumable from results/parts_hybrid_500_*/.
    F=work/finetune
    [ -f $F/results/lines_hybrid_500_released.parquet ] || $TMO 3600 $PY -u $F/04_hybrid_eval.py --grid 500 --procs 3 >> $F/logs/hybrid_500_released.log 2>&1
    for s in 0 1; do
        [ -f $F/results/lines_hybrid_500_ft_seed$s.parquet ] || \
            $TMO 3600 $PY -u $F/04_hybrid_eval.py --grid 500 --model $F/models/pf_case500_goc_ft_seed$s --tag ft_seed$s --procs 3 >> $F/logs/hybrid_500_ft_seed$s.log 2>&1
    done
    zsh work/finetune/20_pipeline2000.sh 6000 mps 0.0001 "0 1"
    zsh work/finetune/20_pipeline10000.sh 20000 mps 0.0001
}

analysis() {
    for sweep in work/hybrid_case118_ieee_base work/hybrid_case118_ieee_base_seed1 work/hybrid_case118_ieee_base_seed42 work/texas/out/hybrid_texas_n2x450; do
        $PY analysis/analysis_conformal.py $sweep.parquet
        $PY analysis/analysis_atlas.py $sweep.parquet
    done
    $PY analysis/analysis_onestep.py
    $PY analysis/analysis_voltage.py --grid both --reuse
    $PY analysis/analysis_warmstart.py --grid both --reuse
    $PY analysis/analysis_shift.py --reuse
    for seed in 0 1 42; do
        $PY analysis/analysis_goc500.py --reuse work/goc500/lines_tiny_seed$seed.parquet
        $PY analysis/analysis_goc500_newton.py --seed $seed
    done
    for grid in goc2000 goc10000; do
        for variant in tiny small; do
            $PY work/$grid/03_analysis.py --variant $variant
        done
    done
    # pre-outage start and hybrid start (work/prev_control.log, work/hybrid_control*.log, work/*_summary*.csv)
    for grid in 500 2000 10000; do
        $PY analysis/analysis_prev_control.py report --grid $grid
        $PY analysis/analysis_hybrid_start.py report --grid $grid
    done
    $PY analysis/analysis_hybrid_start.py report --grid 10000 --variant small --seed 0
    $PY analysis/analysis_hybrid_start.py report --grid 10000 --variant small --seed 1
    $PY analysis/analysis_hybrid_start.py report --grid 10000 --variant tiny --seed 1
    # full N-1 on GOC 2000: six workflows and their cost (work/fulln1/report.log, summary.csv)
    $PY analysis/analysis_full_n1.py report --grid 2000
    # solver timing, reactive limits, zero-shot (work/timing/summary.csv, work/qlim/report.log, work/zeroshot/report.log)
    $PY analysis/analysis_solver_timing.py summary
    $PY analysis/analysis_qlimits.py report
    $PY analysis/analysis_zeroshot.py report
    # fine-tuning tables (only if the fine-tuned results exist)
    R=work/finetune/results
    if [ -f $R/lines_hybrid_10000_ft10000_seed0.parquet ]; then
        $PY analysis/analysis_finetune_hybrid.py released=$R/lines_hybrid_500_released.parquet ft_seed0=$R/lines_hybrid_500_ft_seed0.parquet ft_seed1=$R/lines_hybrid_500_ft_seed1.parquet
        $PY analysis/analysis_finetune_hybrid.py released=$R/lines_hybrid_2000_released.parquet ft2000_seed0=$R/lines_hybrid_2000_ft2000_seed0.parquet ft2000_seed1=$R/lines_hybrid_2000_ft2000_seed1.parquet
        $PY analysis/analysis_finetune_hybrid.py released=$R/lines_hybrid_10000_released.parquet ft10000_seed0=$R/lines_hybrid_10000_ft10000_seed0.parquet
    fi
    $PY analysis/analysis_embedding.py report
    $PY analysis/analysis_remedial.py report
    $PY work/verify_round1.py
}

figures() {
    $PY analysis/make_figures.py
}

case "${1:-}" in
    setup|fetch|tests|sweeps|controls|finetune|analysis|figures|checksums) "$1" ;;
    all)
        setup; fetch; tests
        if [ "${RUN_LONG:-0}" = 1 ]; then sweeps; controls; finetune; fi
        analysis; figures ;;
    *) awk 'NR > 1 && !/^#/ { exit } NR > 1 { print }' "$0"; exit 1 ;;
esac

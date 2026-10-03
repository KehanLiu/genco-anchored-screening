#!/bin/sh
# Rebuild every result and figure of the entry. Stages can be run one at a time:
#
#   ./reproduce.sh setup      environment (Python 3.12 via uv; Julia is installed by juliacall on first use)
#   ./reproduce.sh fetch      GENCO checkpoints and the small data slices from Hugging Face
#   ./reproduce.sh sweeps     solve the grid states with AC, DC and GENCO   (hours on a laptop)
#   ./reproduce.sh analysis   every table of the write-up, from the saved sweeps (minutes)
#   ./reproduce.sh figures    figures/*.png from the saved results
#   ./reproduce.sh all
#
# The Texas 2000 case file is not downloaded by this script: place
# Texas2k_case1_2016summerpeak.m in work/texas/ first (see README, "Data").
set -eu
cd "$(dirname "$0")"
PY=.venv/bin/python
TMO=work/texas/tmo

setup() {
    uv venv --python 3.12 .venv
    uv pip install --python $PY -r requirements.lock.txt
}

fetch() {
    # IEEE 118, GENCO Base, three checkpoint seeds (gridfm/genco-pf-datakit)
    $PY run_pretrained.py --network case118_ieee --variant base
    $PY run_pretrained.py --network case118_ieee --variant base --seed 1
    $PY run_pretrained.py --network case118_ieee --variant base --seed 42
    # GOC 500, GENCO Tiny: fetched by work/goc500/01_eval.py on first use
    # Texas 2000 (gridfm/genco-pf-contingency): checkpoint into work/texas/pf_Texas2k_case1_2016summerpeak_base/model/
    mkdir -p work/texas/pf_Texas2k_case1_2016summerpeak_base/model
    for f in best_model_state_dict.pt normalizer_stats.pt; do
        [ -f work/texas/pf_Texas2k_case1_2016summerpeak_base/model/$f ] || curl -L -o work/texas/pf_Texas2k_case1_2016summerpeak_base/model/$f \
            https://huggingface.co/gridfm/genco-pf-contingency/resolve/main/model/training_bs_16/$f
    done
}

sweeps() {
    mkdir -p work/texas/logs work/goc500/logs
    # outage sweeps: N-0, N-1, random N-2 at five load levels
    $TMO 7200 $PY -u analysis_hybrid.py --network case118_ieee --variant base
    $TMO 7200 $PY -u analysis_hybrid.py --network case118_ieee --variant base_seed1
    $TMO 7200 $PY -u analysis_hybrid.py --network case118_ieee --variant base_seed42
    $TMO 7200 $PY -u work/texas/04_hybrid.py --single 150 --double 450 --tag _n2x450
    # voltages, Newton start points, shifted test sets
    $TMO 7200 $PY -u analysis_voltage.py --grid both
    $TMO 7200 $PY -u analysis_warmstart.py --grid 118
    $TMO 9000 $PY -u analysis_warmstart.py --grid texas
    $TMO 9000 $PY -u analysis_shift.py
    # the example entry's setup: GOC 500, up to five branches removed
    $TMO 7200 .venv/bin/gridfm_datakit generate work/goc500/datagen.yaml
    for seed in 0 1 42; do
        $TMO 7200 $PY -u work/goc500/01_eval.py --seed $seed
        $TMO 7200 $PY -u work/goc500/02_newton.py --seed $seed
    done
}

analysis() {
    for sweep in work/hybrid_case118_ieee_base work/hybrid_case118_ieee_base_seed1 work/hybrid_case118_ieee_base_seed42 work/texas/out/hybrid_texas_n2x450; do
        $PY analysis_conformal.py $sweep.parquet
        $PY analysis_atlas.py $sweep.parquet
    done
    $PY analysis_onestep.py
    $PY analysis_voltage.py --grid both --reuse
    $PY analysis_warmstart.py --grid both --reuse
    $PY analysis_shift.py --reuse
    for seed in 0 1 42; do
        $PY analysis_goc500.py --reuse work/goc500/lines_tiny_seed$seed.parquet
        $PY analysis_goc500_newton.py --seed $seed
    done
}

figures() {
    $PY make_figures.py
}

case "${1:-all}" in
    setup|fetch|sweeps|analysis|figures) "$1" ;;
    all) setup; fetch; sweeps; analysis; figures ;;
    *) sed -n 2,13p "$0"; exit 1 ;;
esac

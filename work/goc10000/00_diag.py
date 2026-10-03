"""Diagnostic: where does datakit's time go on case10000_goc? One process: AC-OPF (Ipopt via PowerModels) on the
first pilot load scenarios (no cost / admittance perturbation), then the random topology generator, DC PF and AC PF.

    work/texas/tmo 1500 .venv/bin/python -u work/goc10000/00_diag.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from gridfm_datakit.network import load_net_from_file
from gridfm_datakit.process.process_network import init_julia, pf_preprocessing
from gridfm_datakit.process.solvers import run_dcpf, run_opf, run_pf
from gridfm_datakit.perturbations.topology_perturbation import RandomComponentDropGenerator

NET = f"case{HERE.name.removeprefix('goc')}_goc"
net = load_net_from_file(str(HERE / "net" / f"{NET}.m"))
print("buses", len(net.buses), "branches", len(net.branches), "gens", len(net.gens), flush=True)
jl = init_julia(200, None, print_level=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
sc = pd.read_parquet(HERE / "data_pilot" / NET / "raw" / "scenarios_agg_load_profile.parquet")
for k in (0, 1, 2):
    s = sc[sc.load_scenario == k].sort_values("load")
    n = net.copy_for_perturbation()
    n.Pd, n.Qd = s.p_mw.to_numpy(), s.q_mvar.to_numpy()
    print(f"load scenario {k}: total Pd {n.Pd.sum():.0f} MW (case: {net.Pd.sum():.0f})", flush=True)
    t0 = time.perf_counter()
    try:
        res = run_opf(n, jl)
        print(f"  OPF ok {time.perf_counter() - t0:.1f}s, Ipopt runtime {res.get('runtime', float('nan')):.1f}s, {res.get('termination_status')}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  OPF failed after {time.perf_counter() - t0:.1f}s: {e}", flush=True)
        continue
    npf = pf_preprocessing(n, res)
    t0 = time.perf_counter()
    gen = RandomComponentDropGenerator(5, 5, npf, ["branch"])
    perts = list(gen.generate(npf))
    print(f"  topology generator: {len(perts)} variants in {time.perf_counter() - t0:.1f}s", flush=True)
    for p in perts[:3]:
        t0 = time.perf_counter(); run_dcpf(p, jl, fast=True); t1 = time.perf_counter()
        try:
            run_pf(p, jl, fast=True); msg = "ok"
        except Exception as e:  # noqa: BLE001
            msg = str(e)[:80]
        print(f"    DC PF {t1 - t0:.2f}s, AC PF {time.perf_counter() - t1:.2f}s {msg}", flush=True)

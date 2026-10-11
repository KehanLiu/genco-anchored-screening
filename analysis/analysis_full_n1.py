"""Complete N-1 contingency analysis on a GOC grid and its cost under six workflows.

Unit of work: one base case (a load scenario of the stored datakit table, dispatched as datakit did; the intact grid
is rebuilt and solved as in analysis/analysis_prev_control.py) x EVERY single branch outage that does not island the grid.

Per (base, contingency), with the SciPy polar Newton of analysis_warmstart.Problem (tables constructor of
work/goc2000/02_newton.py), GENCO Tiny seed 0:
    truth    Newton from the pre-outage AC solution to convergence (fallback: GENCO / hybrid start)
    prev     pre-outage start: iterations to convergence, mismatch and loadings after ONE step
    genco    GENCO start (Vm at PQ, Va), same
    hyb      pre-outage AC voltages + [GENCO(contingency) - GENCO(pre-outage)], same
    genco raw loadings, and DC loadings anchored on the base AC case (AC base loading + DC change)

Workflows (verdict per contingency + cost), evaluated in `report`:
    W0 full AC from the pre-outage solution for every contingency
    W1 one Newton step from the pre-outage solution; continue to convergence if any loading > 1.001 or mismatch > tau
    W2 GENCO loadings only; any > 1.001 -> full AC from the pre-outage start
    W3 GENCO start + one Newton step, continuation rule as W1
    W4 hybrid start + one Newton step, continuation rule as W1 (one extra GENCO inference per base case)
    W5 anchored DC loadings; any > 1.001 -> full AC from the pre-outage start
tau is chosen per workflow on a calibration split of base cases (6 of 24), results are reported on the other 18.

    work/texas/tmo 9000 .venv/bin/python -u analysis/analysis_full_n1.py run --grid 2000 --procs 4     (resumable: chunk files)
    work/texas/tmo 600  .venv/bin/python -u analysis/analysis_full_n1.py timing --grid 2000            (single process, quiet)
    .venv/bin/python analysis/analysis_full_n1.py report --grid 2000
Writes work/fulln1/chunks_<grid>/*, contingencies.parquet, summary.csv, timing_<grid>.json, report.log.

Fresh confirmation run (work/fresh/README.md 2.2 and 4), all phases honour --tag (chunks_<grid><tag>/, timing_<grid><tag>.json,
contingencies<tag>.parquet / summary<tag>.csv; the data folder comes from analysis_prev_control.paths(), GOC_DATA override):
    run  --policy work/fresh/policy_frozen.json   adds a fourth start "shyb" = pre-outage AC voltages + lambda_vm/lambda_va x
                                                  [GENCO(contingency) - GENCO(pre-outage)], lambdas frozen in the policy file
    e2e  --policy work/fresh/policy_frozen.json   measured end-to-end ms per outage of W0 / W1 / W4 / W4s / mismatch gate / classical-first
                                                  -> e2e_<grid><tag>.parquet + .json (single process, pre-registered sample)
The frozen-policy scoring of the fresh chunks is work/fresh/score_fresh_n1.py (reuses load_chunks / workflows of this file).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here
import analysis_prev_control as pc  # noqa: E402

OUT = ROOT / "work" / "fulln1"
LOG = OUT / "report.log"
BASE, LIM, BIG = 100.0, 1.001, 9.99
N_BASES, CHUNK = 24, 120
STARTS = ("prev", "genco", "hyb")
HIST = ("prev0", "prev1", "genco1", "hyb1", "genco", "hyb0", "dc_anch")
HBIN, HMAX = 0.001, 50.0  # histogram of |loading error| in points of rating
NB = int(round(HMAX / HBIN))
TAUS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, float("inf")]  # MW, max-norm of the bus mismatch after the step


def log(*a):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_evidence(tag):
    up = subprocess.run("uptime", shell=True, capture_output=True, text=True).stdout.strip()
    ps = subprocess.run("ps -Ao pcpu,pmem,pid,comm -r | head -9", shell=True, capture_output=True, text=True).stdout.rstrip()
    log(f"[load {tag}] uptime: {up}")
    for row in ps.splitlines():
        log(f"[load {tag}]   {row[:150]}")


# ----------------------------------------------------------------------------- design: base cases and contingencies
def design(grid):
    """24 load scenarios spread evenly over the ranking by total load; candidate branches (in service, not a bridge)."""
    import networkx as nx
    import numpy as np
    import pandas as pd
    here, net, raw = pc.paths(grid)
    status = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "load_scenario_idx", "idx", "br_status", "from_bus", "to_bus"])
    always_out = (status.groupby("idx").br_status.max() == 0).sort_index().to_numpy()
    first = status.groupby("load_scenario_idx").scenario.min()
    pd_tot = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "Pd"], filters=[("scenario", "in", [int(s) for s in first.values])]).groupby("scenario").Pd.sum()
    total = pd.Series(pd_tot.reindex(first.values).to_numpy(), index=first.index).sort_values()
    pick = total.iloc[np.round(np.linspace(0, len(total) - 1, N_BASES)).astype(int)]
    bases = pd.DataFrame({"base": pick.index.astype(int), "scenario": first.reindex(pick.index).to_numpy(int), "load_mw": pick.to_numpy(),
                          "rank": np.arange(N_BASES)})
    bases["split"] = np.where(bases["rank"] % 4 == 2, "calibration", "test")
    one = status[status.scenario == status.scenario.min()].sort_values("idx")
    assert (one.idx.to_numpy() == np.arange(len(one))).all()
    f, t = one.from_bus.to_numpy(int), one.to_bus.to_numpy(int)
    live = ~always_out
    g = nx.MultiGraph()
    g.add_nodes_from(range(int(max(f.max(), t.max())) + 1))
    g.add_edges_from((int(a), int(b), int(k)) for k, (a, b) in enumerate(zip(f, t)) if live[k])
    assert nx.is_connected(g), "intact grid is not connected"
    simple = nx.Graph(g)
    bridge = np.zeros(len(one), bool)
    for a, b in nx.bridges(simple):
        if g.number_of_edges(a, b) == 1:
            bridge[list(g[a][b].keys())[0]] = True
    cand = np.flatnonzero(live & ~bridge)
    return bases, always_out, bridge, cand


# ----------------------------------------------------------------------------- worker
_W = {}


def _init(grid, lam=None):
    import engine  # Julia before torch
    import torch
    torch.set_num_threads(1)
    here, net, raw = pc.paths(grid)
    engine.WORK = here
    _W.update(genco=engine.Genco(net, "tiny"), nw=pc.newton_module(), raw=raw, ctx={}, lam=lam)


def policy_lambda(policy, grid):
    """Frozen shrinkage scales {"va": .., "vm": ..} of the Tiny seed 0 checkpoint of this grid (policy_frozen.json)."""
    lam = policy["shrinkage"]["lambda"][f"goc{grid}_tiny_seed0"]
    return {"va": float(lam["va"]), "vm": float(lam["vm"])}


def shrunk_start(prob, ctx, vm_g, va_g, lam):
    """Anchored start with the predicted change scaled: Vm_prev + lam_vm (GENCO_after - GENCO_before), Va likewise (lam_va)."""
    import numpy as np
    vm_p, va_p = np.abs(ctx["V"]), np.angle(ctx["V"])
    return prob.start(vm_pq=vm_p + lam["vm"] * (vm_g - ctx["vm_gi"]), va=va_p + lam["va"] * (va_g - ctx["va_gi"]))


def base_context(load_idx, scenario, always_out):
    """Intact-grid state of one base case: tables with the intact AC solution written in, AC / DC / GENCO loadings."""
    import numpy as np
    import pandas as pd
    if load_idx in _W["ctx"]:
        return _W["ctx"][load_idx]
    genco, nw, raw = _W["genco"], _W["nw"], _W["raw"]
    flt = [("scenario", "==", int(scenario))]
    bus0 = pd.read_parquet(raw / "bus_data.parquet", filters=flt).sort_values("bus").reset_index(drop=True)
    gen0 = pd.read_parquet(raw / "gen_data.parquet", filters=flt).sort_values("idx").reset_index(drop=True)
    branch0 = pd.read_parquet(raw / "branch_data.parquet", filters=flt).sort_values("idx").reset_index(drop=True)
    assert int(bus0.load_scenario_idx.iloc[0]) == load_idx and (branch0.idx.to_numpy() == np.arange(len(branch0))).all()
    q = gen0.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
    bus0 = bus0.merge(q, on="bus", how="left").fillna(0)
    bi = branch0.copy()
    bi["br_status"] = np.where(always_out, 0, 1).astype(branch0.br_status.dtype)
    for k, v in pc.admittances(bi).items():
        bi[f"{k}_r"], bi[f"{k}_i"] = np.where(always_out, 0.0, v.real), np.where(always_out, 0.0, v.imag)
    live0 = (branch0.br_status == 1).to_numpy()
    yc = pc.admittances(branch0[live0])
    ydiff = max(float(np.abs(yc[k] - (branch0.loc[live0, f"{k}_r"].to_numpy() + 1j * branch0.loc[live0, f"{k}_i"].to_numpy())).max()) for k in yc)
    prob_i = nw.make_problem(bus0, bi)
    v_stored = bus0.Vm.to_numpy(float) * np.exp(1j * np.radians(bus0.Va.to_numpy(float)))
    r = prob_i.newton(v_stored)
    how = "stored contingency solution"
    if not r["converged"]:
        r, how = prob_i.newton(prob_i.start()), "flat"
    assert r["converged"], f"intact grid of base {load_idx} does not solve"
    V = r["V"]
    live_i = (bi.br_status == 1).to_numpy()
    ni = int(live_i.sum())
    Sb = V * np.conj(prob_i.Ybus @ V) * BASE
    busi = bus0.copy()
    isgen = bus0.PQ.to_numpy() != 1
    busi["Vm"], busi["Va"] = np.abs(V), np.degrees(np.angle(V))
    busi["Pg"] = np.where(isgen, Sb.real + bus0.Pd.to_numpy(), bus0.Pg.to_numpy())
    busi["Qg"] = np.where(isgen, Sb.imag + bus0.Qd.to_numpy(), bus0.Qg.to_numpy())
    Sf = V[prob_i.f] * np.conj(prob_i.Yf @ V) * BASE
    St = V[prob_i.t] * np.conj(prob_i.Yt @ V) * BASE
    for c, v in (("pf", Sf.real), ("qf", Sf.imag), ("pt", St.real), ("qt", St.imag)):
        col = np.zeros(len(bi))
        col[live_i] = v
        bi[c] = col
    ac_prev = np.full(len(bi), np.nan)
    ac_prev[live_i] = prob_i.loading(V)
    dc_prev = pc.dc_loading(busi, bi)
    bp, br = genco.solve(busi, gen0, bi)
    g_prev = np.full(len(bi), np.nan)
    g_prev[live_i] = np.maximum(br.loading[:ni].to_numpy(), br.loading[ni:].to_numpy())
    ctx = dict(busi=busi, gen=gen0, bi=bi, V=V, ac_prev=ac_prev, dc_prev=dc_prev, g_prev=g_prev,
               vm_gi=bp.Vm.to_numpy(float), va_gi=np.radians(bp.Va.to_numpy(float)),
               info={"base": load_idx, "scenario": int(scenario), "intact_it": r["iterations"], "intact_start": how, "ydiff": ydiff,
                     "intact_overloads": int(np.nansum(ac_prev > LIM)), "intact_worst": float(np.nanmax(ac_prev)),
                     "pq_pg_max": float(np.abs(bus0.Pg.to_numpy()[~isgen]).max()), "n_removed_in_stored": int((~live0).sum() - always_out.sum()),
                     "load_mw": float(bus0.Pd.sum())})
    _W["ctx"].clear()  # one base in memory at a time
    _W["ctx"][load_idx] = ctx
    return ctx


ZERO_COLS = ["pf", "qf", "pt", "qt", "Yff_r", "Yff_i", "Yft_r", "Yft_i", "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i", "pf_dc", "pt_dc"]


def contingency_tables(ctx, k):
    bc = ctx["bi"].copy()
    bc.loc[k, ZERO_COLS] = 0.0
    bc.loc[k, "br_status"] = 0
    return bc


def task(job):
    import numpy as np
    import pandas as pd
    grid, load_idx, scenario, always_out, cid, branches, path = job
    if Path(path + ".parquet").exists() and Path(path + ".npz").exists():
        return path, 0, 0.0
    t_start = time.perf_counter()
    genco, nw, lam = _W["genco"], _W["nw"], _W.get("lam")
    ctx = base_context(load_idx, scenario, always_out)
    busi, V_prev = ctx["busi"], ctx["V"]
    vm_p, va_p = np.abs(V_prev), np.angle(V_prev)
    # with a frozen policy (--policy) the shrunk anchored start "shyb" is evaluated like "hyb"; truth fallback order stays STARTS
    starts_all = STARTS + (("shyb",) if lam else ())
    hist_all = HIST + (("shyb1", "shyb0") if lam else ())
    hist = {h: np.zeros(NB + 1, np.int64) for h in hist_all}
    nan_count = {h: 0 for h in hist_all}
    rows = []
    for k in branches:
        bc = contingency_tables(ctx, int(k))
        prob = nw.make_problem(busi, bc)
        live = (bc.br_status == 1).to_numpy()
        n = int(live.sum())
        idx = np.flatnonzero(live)
        bp, br = genco.solve(busi, ctx["gen"], bc)
        vm_g, va_g = bp.Vm.to_numpy(float), np.radians(bp.Va.to_numpy(float))
        g_raw = np.where(prob.rated, np.maximum(br.loading[:n].to_numpy(), br.loading[n:].to_numpy()), np.nan)
        with np.errstate(all="ignore"):
            dc_anch = np.where(prob.rated, ctx["ac_prev"][idx] + pc.dc_loading(busi, bc)[live] - ctx["dc_prev"][idx], np.nan)
        starts = {"prev": prob.start(vm_pq=vm_p, va=va_p), "genco": prob.start(vm_pq=vm_g, va=va_g),
                  "hyb": prob.start(vm_pq=vm_p + (vm_g - ctx["vm_gi"]), va=va_p + (va_g - ctx["va_gi"]))}
        if lam:
            starts["shyb"] = shrunk_start(prob, ctx, vm_g, va_g, lam)
        runs = {s: prob.newton(v0, keep=1) for s, v0 in starts.items()}
        row = {"base": load_idx, "branch": int(k), "pre_loading": float(ctx["ac_prev"][k])}
        ref = next((s for s in STARTS if runs[s]["converged"]), None)
        row["truth_conv"], row["truth_from"] = ref is not None, ref or "none"
        truth = prob.loading(runs[ref]["V"]) if ref else None
        if ref:
            row["n_real"], row["worst"] = int(np.nansum(truth > LIM)), float(np.nanmax(truth))
            row["vmin"] = float(np.abs(runs[ref]["V"]).min())
        else:
            row["n_real"], row["worst"], row["vmin"] = -1, np.nan, np.nan
        one = {}
        for s in starts_all:
            r = runs[s]
            if r["steps"]:
                v1 = r["steps"][0]
            else:  # converged without a step (start already solves the contingency) or singular Jacobian at the start
                v1 = r["V"] if r["converged"] else None
            with np.errstate(all="ignore"):
                l1 = prob.loading(v1) if v1 is not None else np.full(n, np.nan)
                mis1 = float(np.abs(prob.mismatch(v1)).max() * BASE) if v1 is not None else np.inf
            one[s] = l1
            bad = bool(np.isnan(l1[prob.rated]).any()) or not np.isfinite(mis1)
            row.update({f"conv_{s}": r["converged"], f"it_{s}": r["iterations"], f"mis0_{s}": r["mis0"], f"mis1_{s}": mis1 if np.isfinite(mis1) else np.inf,
                        f"max1_{s}": BIG if bad else float(np.nanmax(l1)),
                        f"dv_{s}": float(np.abs(r["V"] - runs[ref]["V"]).max()) if (ref and r["converged"]) else np.nan})
        row["max_genco"] = float(np.nanmax(g_raw)) if np.isfinite(g_raw[prob.rated]).all() else BIG
        row["max_dc"] = float(np.nanmax(dc_anch)) if np.isfinite(dc_anch[prob.rated]).all() else BIG
        if ref:
            preds = {"prev0": prob.loading(starts["prev"]), "prev1": one["prev"], "genco1": one["genco"], "hyb1": one["hyb"], "genco": g_raw,
                     "hyb0": prob.loading(starts["hyb"]), "dc_anch": dc_anch}
            if lam:
                preds.update(shyb1=one["shyb"], shyb0=prob.loading(starts["shyb"]))
            for h, p in preds.items():
                e = np.abs(p - truth)[prob.rated] * 100
                ok = np.isfinite(e)
                nan_count[h] += int((~ok).sum())
                hist[h] += np.bincount(np.minimum(e[ok] / HBIN, NB).astype(np.int64), minlength=NB + 1)
                row[f"emax_{h}"] = float(e[ok].max()) if ok.any() else np.nan
            # overloaded lines each predictor calls (for line-level false alarms) and the real ones it does not call
            for h in ("prev1", "genco1", "hyb1", "genco", "dc_anch") + (("shyb1",) if lam else ()):
                called = np.nan_to_num(preds[h], nan=BIG) > LIM
                row[f"lmiss_{h}"] = int(((truth > LIM) & ~called).sum())
                row[f"lfalse_{h}"] = int((~(truth > LIM) & called & prob.rated).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(path + ".tmp.parquet", index=False)
    np.savez(path + ".tmp.npz", **hist, **{f"nan_{h}": np.array(v) for h, v in nan_count.items()}, info=json.dumps(ctx["info"]))
    os.replace(path + ".tmp.npz", path + ".npz")
    os.replace(path + ".tmp.parquet", path + ".parquet")
    return path, len(rows), time.perf_counter() - t_start


def chunk_dir(grid, tag=""):
    return OUT / f"chunks_{grid}{tag}"


def load_policy(a):
    return json.loads(Path(a.policy).read_text()) if getattr(a, "policy", None) else None


def run(a):
    import numpy as np
    bases, always_out, bridge, cand = design(a.grid)
    here, net, raw = pc.paths(a.grid)
    if a.bases:
        bases = bases[bases["rank"].isin(a.bases)]
    if a.limit:
        cand = cand[np.round(np.linspace(0, len(cand) - 1, a.limit)).astype(int)]
    d = chunk_dir(a.grid, a.tag)
    d.mkdir(parents=True, exist_ok=True)
    policy = load_policy(a)
    lam = policy_lambda(policy, a.grid) if policy else None
    log(f"=== run {net}{a.tag}: {len(bases)} base cases x {len(cand):,} single-branch outages (of {len(always_out):,} branches: {int(always_out.sum())} always out, "
        f"{int(bridge.sum())} bridges skipped) = {len(bases) * len(cand):,} contingencies, {a.procs} processes; data {raw}"
        + (f"; policy {a.policy}: fourth start shyb with lambda va {lam['va']} vm {lam['vm']}" if lam else ""))
    jobs = []
    for b in bases.itertuples():
        for cid, br in enumerate(np.array_split(cand, max(int(np.ceil(len(cand) / CHUNK)), 1))):
            jobs.append((a.grid, int(b.base), int(b.scenario), always_out, cid, br, str(d / f"b{int(b.base):04d}_c{cid:03d}")))
    t0 = time.perf_counter()
    done = n = 0
    with get_context("spawn").Pool(a.procs, initializer=_init, initargs=(a.grid, lam)) as pool:
        for path, k, sec in pool.imap_unordered(task, jobs):
            done += 1
            n += k
            if done % 20 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                log(f"  {done}/{len(jobs)} chunks, {n:,} new contingencies, wall {el:.0f} s ({1e3 * el / max(n, 1):.1f} ms wall per contingency, last chunk {sec:.0f} s)")
    log(f"run finished: {n:,} contingencies computed in {time.perf_counter() - t0:.0f} s wall")


# ----------------------------------------------------------------------------- unit costs (single process)
def timing(a):
    import engine  # Julia before torch
    import numpy as np
    import torch
    from torch_geometric.data import Batch
    from gridfm_graphkit.datasets.graph_builder import build_hetero_data
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import splu
    bases, always_out, bridge, cand = design(a.grid)
    here, net, raw = pc.paths(a.grid)
    engine.WORK = here
    _W.update(genco=engine.Genco(net, "tiny"), nw=pc.newton_module(), raw=raw, ctx={})
    genco, nw = _W["genco"], _W["nw"]
    log(f"=== timing {net}{a.tag} (INDICATIVE; single process, pid {os.getpid()}): torch {torch.__version__}, {torch.get_num_threads()} CPU threads, mps {torch.backends.mps.is_available()}; "
        f"{a.states} contingencies; data {raw}")
    load_evidence("start")
    b = bases[bases.split == "test"].iloc[len(bases[bases.split == "test"]) // 2]
    ctx = base_context(int(b.base), int(b.scenario), always_out)
    rng = np.random.default_rng(0)
    ks = rng.choice(cand, a.states, replace=False)
    busi, V_prev = ctx["busi"], ctx["V"]
    res = {k: [] for k in ("tables", "build", "iter_full", "step1", "loading", "mismatch", "dc_full", "dc_factor_solve", "prep", "e2e", "its")}
    graphs = []

    def best(fn, n=3):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            out = fn()
            ts.append(1e3 * (time.perf_counter() - t0))
        return min(ts), out

    for k in ks:
        t, bc = best(lambda: contingency_tables(ctx, int(k)))
        res["tables"].append(t)
        t, prob = best(lambda: nw.make_problem(busi, bc))
        res["build"].append(t)
        v0 = prob.start(vm_pq=np.abs(V_prev), va=np.angle(V_prev))
        t, r = best(lambda: prob.newton(v0))
        if r["converged"] and r["iterations"]:
            res["iter_full"].append(t / r["iterations"])
            res["its"].append(r["iterations"])
        t, _ = best(lambda: prob.newton(v0, max_it=1))
        res["step1"].append(t)
        t, _ = best(lambda: prob.loading(r["V"]))
        res["loading"].append(t)
        t, _ = best(lambda: prob.mismatch(r["V"]))
        res["mismatch"].append(t)
        t, _ = best(lambda: pc.dc_loading(busi, bc))
        res["dc_full"].append(t)
        # DC factorisation + solve alone (matrix already assembled)
        live = (bc.br_status == 1).to_numpy()
        brl = bc[live]
        nb, m = len(busi), len(brl)
        f, tt = brl.from_bus.to_numpy(int), brl.to_bus.to_numpy(int)
        bb = brl.x.to_numpy() / (brl.r.to_numpy() ** 2 + brl.x.to_numpy() ** 2)
        A = csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[f, tt])), shape=(m, nb))
        keep = np.setdiff1d(np.arange(nb), np.flatnonzero(busi.REF.to_numpy() == 1)[:1])
        B = (A.T @ diags(bb) @ A).tocsc()[keep][:, keep]
        P = ((busi.Pg - busi.Pd).to_numpy() / BASE - busi.GS.to_numpy())[keep]
        t, _ = best(lambda: splu(B).solve(P))
        res["dc_factor_solve"].append(t)

        def prep():
            data = build_hetero_data(busi, ctx["gen"], bc)
            genco.normalizer.transform(data)
            return genco.transforms(data)
        t, g = best(prep, 1)
        res["prep"].append(t)
        graphs.append(g)
        t, _ = best(lambda: genco.solve(busi, ctx["gen"], bc), 1)
        res["e2e"].append(t)
    out = {"network": net, "torch_threads": torch.get_num_threads(), "n": int(a.states), "mean_iterations_prev": float(np.mean(res["its"]))}
    names = {"tables": "contingency branch table (pandas copy)", "build": "Newton problem build (Ybus from tables)", "iter_full": "Newton iteration (full solve / iterations)",
             "step1": "single Newton step (newton(max_it=1))", "loading": "line loadings from V", "mismatch": "bus mismatch from V", "dc_full": "DC solve incl. matrix assembly",
             "dc_factor_solve": "DC factorisation + solve only", "prep": "GENCO graph build + normalise + transforms", "e2e": "GENCO end-to-end genco.solve (cpu, batch 1)"}
    for k, label in names.items():
        v = np.asarray(res[k][1:], float)
        out[f"ms_{k}"] = float(np.median(v))
        log(f"  {label:<48} {np.median(v):8.3f} ms median (min {v.min():.3f}, p90 {np.quantile(v, .9):.3f}, n {len(v)})")
    for device in ("cpu", "mps"):
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        model = genco.model.to(device)
        for bs in (1, 16):
            if bs > len(graphs):
                continue
            per = []
            for rep in range(3):  # three different batches
                batch = Batch.from_data_list(graphs[rep * bs % max(len(graphs) - bs, 1):][:bs]).to(device)
                with torch.no_grad():
                    for _ in range(3):
                        model(batch)
                    if device == "mps":
                        torch.mps.synchronize()
                    for _ in range(7):
                        t0 = time.perf_counter()
                        model(batch)
                        if device == "mps":
                            torch.mps.synchronize()
                        per.append(1e3 * (time.perf_counter() - t0) / bs)
                del batch
            out[f"ms_forward_{device}_b{bs}"] = float(np.median(per))
            log(f"  GENCO Tiny forward {device} batch {bs:>2}                     {np.median(per):8.3f} ms/state median (min {min(per):.3f}, n {len(per)})")
        genco.model.to("cpu")
    load_evidence("end")
    tfile = OUT / f"timing_{a.grid}{a.tag}.json"
    tfile.write_text(json.dumps(out, indent=1))
    log(f"saved {tfile}")


# ----------------------------------------------------------------------------- measured end-to-end cost (single process)
def e2e(a):
    """Measured wall time of the COMPLETE per-outage sequence of each screening workflow (work/fresh/README.md section 4,
    reporting as asked by the independent batch-2 review, section 2).

    Workload (frozen, written to the json): per base case (ranks of --bases, default the policy's n1_base_ranks) a pre-registered
    sample of outages (numpy default_rng([2026, rank]).choice(cand, n, replace=False), sorted by branch id; the sample of a base does
    not depend on which other bases run); per base the order is
        1. base_context (pre-outage AC solve + intact-grid inference, untimed), then the pre-outage GENCO inference timed on its own
           (CPU batch 1 = genco.solve; MPS batch 1 = graph build + batch + transfer + forward + synchronize), min of `repetitions`;
        2. for each workflow in the fixed order W0, W1, W4, W4s, gate, CF<tau1> (every tau1 of the policy): WARMUP untimed outages
           (the first ones of the sample), then `repetitions` complete passes over the sample in id order; perf_counter around the
           complete per-outage sequence, and the contiguous wall of every pass;
        3. W4_mps<bs>: one untimed warm-up block, then `repetitions` passes over the blocks of `mps_batch` consecutive sample ids
           (the last block of a base may be shorter: timed as is, its size recorded); torch.mps.synchronize() before the shared
           timer stops; the block's shared time is split equally over its outages and added to each outage's own Newton part.
    Per outage and workflow the rows hold ms = min over repetitions, ms_first = first repetition, ms_mean = arithmetic mean;
    totals are reported without min-taking (sum of the first pass, and the pass walls) and as the sum of minima.
    Workflows:
        W0        tables -> make_problem -> Newton from the pre-outage solution to convergence (max_it 30)
        W1        tables -> make_problem -> pre-outage start -> one Newton step -> loadings + mismatch -> gate -> continue if flagged
        W4        W1 with one GENCO inference (CPU, batch 1: engine.Genco.solve) and the anchored change start, tau_hyb
        W4s       W4 with the frozen shrinkage lambdas, tau_shyb
        gate      mismatch gate: inference, BOTH starts' initial mismatch (timed), the smaller takes the step (tie / NaN -> pre-outage)
        CF<tau1>  classical-first: W1's step; where its max loading > c or its mismatch > tau1 (or not finite) the anchored start
                  (inference + hybrid start + step + gate, tau_hyb) follows; everything executed is timed
        W4_mps<bs> W4 with the inference of a block of outages as one forward on MPS; only if the batched Vm/Va reproduce genco.solve
    Gate inside the timing: max one-step loading > c = 1.001 or mismatch > the policy's tau_mw (other cushions are scored offline
    from the chunks). Continuation = the same Newton run carried on (max_it 30 more iterations); if it breaks down (singular
    Jacobian, non-finite mismatch or 30 iterations without convergence) the outage is re-solved from the pre-outage start
    (`restarted`); if that fails too the outage is `failed` (flagged, cost recorded; W0 calls it critical). The time of the
    continuation / restart part is recorded separately (ms_cont).
    Threads: torch.set_num_threads(--threads, default the torch_threads of work/fulln1/timing_<grid>.json); SciPy's splu is
    single-threaded, so this affects the GENCO forward only.
    Not timed (per base, shared by all outages): model load, pre-outage AC solve, the intact-grid inference inside base_context
    (its cost is measured separately, see 1.), warm-up.
    Writes e2e_<grid><tag>.parquet (one row per outage x workflow, with critical / missed / false_flag) and e2e_<grid><tag>.json
    (protocol, samples, per-base inference cost, pass walls, per-workflow summary).
    """
    import engine  # Julia before torch
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.data import Batch
    from gridfm_graphkit.datasets.graph_builder import build_hetero_data
    policy = load_policy(a)
    assert policy, "e2e needs --policy"
    lam, tau, cfg = policy_lambda(policy, a.grid), policy["tau_mw"], policy["e2e_timing"]
    n_policy, reps, c, bs = int(cfg["sample_outages_per_base"]), int(cfg["repetitions"]), 1.001, int(cfg.get("mps_batch", 16))
    tau1s = [float(t) for t in policy["classical_first_tau1_mw"]]
    n_sample, sample_override = n_policy, None
    if getattr(a, "sample", 0):
        n_sample = sample_override = int(a.sample)
    if a.limit:
        n_sample = min(n_sample, a.limit)
    warm = int(cfg.get("warmup_outages", 5))
    tfile = OUT / f"timing_{a.grid}.json"
    threads = int(a.threads) if getattr(a, "threads", 0) else (int(json.loads(tfile.read_text())["torch_threads"]) if tfile.exists() else torch.get_num_threads())
    torch.set_num_threads(threads)
    bases, always_out, bridge, cand = design(a.grid)
    ranks = a.bases if a.bases else policy["n1_base_ranks"]
    bases = bases[bases["rank"].isin(ranks)]
    here, net, raw = pc.paths(a.grid)
    engine.WORK = here
    _W.update(genco=engine.Genco(net, "tiny"), nw=pc.newton_module(), raw=raw, ctx={}, lam=lam)
    genco, nw = _W["genco"], _W["nw"]
    mps_ok = torch.backends.mps.is_available()
    mps_note = "not available" if not mps_ok else "pending validation"
    wf_order = ["W0", "W1", "W4", "W4s", "gate"] + [f"CF{t:g}" for t in tau1s]
    log(f"=== e2e {net}{a.tag} (single process, pid {os.getpid()}): torch {torch.__version__}, {torch.get_num_threads()} CPU threads (set explicitly), "
        f"mps {mps_ok}; {len(bases)} bases (ranks {sorted(int(r) for r in bases['rank'])}) x {n_sample} outages"
        + (f" (OVERRIDE of the policy's {n_policy})" if sample_override else "") + f", {reps} repetitions, {warm} warm-up outages per workflow and base; "
        f"workflow order {wf_order} + W4_mps{bs}; gate c = {c}, tau {tau}, tau1 {tau1s}, lambda {lam}; data {raw}")
    log("  excluded (per base, shared by all outages): model load, pre-outage AC solve, warm-up; the pre-outage GENCO inference is timed separately per base; "
        "outages sampled with rng seed [2026, rank]")
    load_evidence("start")

    def fmax(l):
        v = l[np.isfinite(l)]
        return float(v.max()) if v.size else np.nan

    def finish(prob, r, v_prev, its):
        """Carry a flagged run on to convergence; restart from the pre-outage start if it breaks down. Returns (converged, its, restarted, failed, ms)."""
        t0 = time.perf_counter()
        conv, restarted = r["converged"], False
        if not conv:
            rc = prob.newton(r["V"])
            its, conv = its + rc["iterations"], rc["converged"]
            if not conv and v_prev is not None:
                rp = prob.newton(v_prev)
                its, conv, restarted = its + rp["iterations"], rp["converged"], True
        return conv, its, restarted, not conv, 1e3 * (time.perf_counter() - t0)

    def one_step(prob, v0):
        """One Newton step from v0: (run, v1 or None, max loading, mismatch MW)."""
        r = prob.newton(v0, max_it=1, keep=1)
        v1 = r["steps"][0] if r["steps"] else (r["V"] if r["converged"] else None)
        if v1 is None:  # singular Jacobian / non-finite mismatch at the start: no step possible
            return r, None, np.nan, np.inf
        with np.errstate(all="ignore"):
            return r, v1, fmax(prob.loading(v1)), float(np.abs(prob.mismatch(v1)).max() * BASE)

    def step_gate(prob, v0, tau_s, v_prev):
        """One Newton step, loadings + mismatch, gate; if flagged carry the run on (finish)."""
        r, v1, mx, mis = one_step(prob, v0)
        its = r["iterations"]
        flag = v1 is None or not (mx <= c) or not (mis <= tau_s)
        out = {"flagged": flag, "converged": r["converged"], "iterations": its, "restarted": False, "failed": False, "ms_cont": 0.0}
        if flag and not r["converged"]:
            if v1 is None and v_prev is None:  # W1: the pre-outage start itself has no step; nothing else to try (W0 fails the same way)
                out["failed"] = True
            elif v1 is None:
                t0 = time.perf_counter()
                rp = prob.newton(v_prev)
                out.update(converged=rp["converged"], iterations=its + rp["iterations"], restarted=True, failed=not rp["converged"], ms_cont=1e3 * (time.perf_counter() - t0))
            else:
                conv, its, restarted, failed, ms = finish(prob, r, v_prev, its)
                out.update(converged=conv, iterations=its, restarted=restarted, failed=failed, ms_cont=ms)
        return out

    def problem(ctx, k):
        bc = contingency_tables(ctx, k)
        return bc, nw.make_problem(ctx["busi"], bc)

    def infer(ctx, bc):
        bp, _ = genco.solve(ctx["busi"], ctx["gen"], bc)
        return bp.Vm.to_numpy(float), np.radians(bp.Va.to_numpy(float))

    def prev_start(ctx, prob):
        return prob.start(vm_pq=np.abs(ctx["V"]), va=np.angle(ctx["V"]))

    def hyb_start(ctx, prob, vm_g, va_g):
        return prob.start(vm_pq=np.abs(ctx["V"]) + (vm_g - ctx["vm_gi"]), va=np.angle(ctx["V"]) + (va_g - ctx["va_gi"]))

    def w0(ctx, k):
        bc, prob = problem(ctx, k)
        t0 = time.perf_counter()
        r = prob.newton(prev_start(ctx, prob))
        ms = 1e3 * (time.perf_counter() - t0)
        with np.errstate(all="ignore"):
            crit = (not r["converged"]) or not (fmax(prob.loading(r["V"])) <= LIM)
        return {"flagged": True, "converged": r["converged"], "iterations": r["iterations"], "restarted": False, "failed": not r["converged"], "ms_cont": ms,
                "n_inf": 0, "critical": crit}

    def w1(ctx, k):
        bc, prob = problem(ctx, k)
        return {**step_gate(prob, prev_start(ctx, prob), tau["prev"], None), "n_inf": 0}

    def w4(ctx, k, shrink=False):
        bc, prob = problem(ctx, k)
        vm_g, va_g = infer(ctx, bc)
        v0 = shrunk_start(prob, ctx, vm_g, va_g, lam) if shrink else hyb_start(ctx, prob, vm_g, va_g)
        return {**step_gate(prob, v0, tau["shyb" if shrink else "hyb"], prev_start(ctx, prob)), "n_inf": 1}

    def wgate(ctx, k):
        bc, prob = problem(ctx, k)
        vm_g, va_g = infer(ctx, bc)
        v_prev, v_hyb = prev_start(ctx, prob), hyb_start(ctx, prob, vm_g, va_g)
        with np.errstate(all="ignore"):  # both initial mismatch evaluations are inside the timed sequence
            m_prev, m_hyb = np.abs(prob.mismatch(v_prev)).max(), np.abs(prob.mismatch(v_hyb)).max()
        use_hyb = bool(np.isfinite(m_hyb) and m_hyb < m_prev)
        out = step_gate(prob, v_hyb if use_hyb else v_prev, tau["hyb" if use_hyb else "prev"], v_prev if use_hyb else None)
        return {**out, "n_inf": 1, "start": "hyb" if use_hyb else "prev"}

    def wcf(ctx, k, tau1):
        """Classical-first: W1's step; the anchored start only where that step's loading > c or mismatch > tau1 (A3 wording)."""
        bc, prob = problem(ctx, k)
        v_prev = prev_start(ctx, prob)
        r, v1, mx, mis = one_step(prob, v_prev)
        trigger = v1 is None or not (mx <= c) or not (mis <= tau1)
        if not trigger:
            return {"flagged": False, "converged": r["converged"], "iterations": r["iterations"], "restarted": False, "failed": False, "ms_cont": 0.0, "n_inf": 0, "second": False}
        vm_g, va_g = infer(ctx, bc)
        out = step_gate(prob, hyb_start(ctx, prob, vm_g, va_g), tau["hyb"], v_prev)
        out["iterations"] += r["iterations"]
        return {**out, "n_inf": 1, "second": True}

    def prep(ctx, bc):
        data = build_hetero_data(ctx["busi"], ctx["gen"], bc)
        genco.normalizer.transform(data)
        return genco.transforms(data)

    def mps_forward(ctx, bcs):
        """Shared part of a block: graphs, batch, transfer, forward on MPS, synchronize, split."""
        batch = Batch.from_data_list([prep(ctx, bc) for bc in bcs]).to("mps")
        with torch.no_grad():
            out = genco.model(batch)["bus"][:, :2]  # Vm p.u., Va rad (inverse_output rescales Pg/Qg only)
        torch.mps.synchronize()
        vb, bidx = out.cpu().numpy(), batch["bus"].batch.cpu().numpy()
        return vb, bidx

    def mps_block(ctx, ks):
        """One block of outages: shared part (tables, problems, graphs, batch, forward on MPS, sync, split) then the Newton part per outage."""
        t0 = time.perf_counter()
        bcs = [contingency_tables(ctx, int(k)) for k in ks]
        probs = [nw.make_problem(ctx["busi"], bc) for bc in bcs]
        vb, bidx = mps_forward(ctx, bcs)
        shared = 1e3 * (time.perf_counter() - t0)
        out = []
        for j, prob in enumerate(probs):
            t1 = time.perf_counter()
            vm_g, va_g = vb[bidx == j, 0].astype(float), vb[bidx == j, 1].astype(float)
            res = step_gate(prob, hyb_start(ctx, prob, vm_g, va_g), tau["hyb"], prev_start(ctx, prob))
            out.append((1e3 * (time.perf_counter() - t1), {**res, "n_inf": 1}, vm_g, va_g))
        return shared, out

    def best_of(fn, n):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(1e3 * (time.perf_counter() - t0))
        return float(min(ts)), [float(t) for t in ts]

    wfs = {"W0": w0, "W1": w1, "W4": w4, "W4s": lambda ctx, k: w4(ctx, k, True), "gate": wgate}
    for t1 in tau1s:
        wfs[f"CF{t1:g}"] = (lambda t: (lambda ctx, k: wcf(ctx, k, t)))(t1)
    rows, samples, base_inf, pass_wall, blocks_log = [], {}, [], {w: {} for w in wf_order + [f"W4_mps{bs}"]}, []
    t_all = time.perf_counter()
    for b in bases.itertuples():
        rank, base = int(b.rank), int(b.base)
        ctx = base_context(base, int(b.scenario), always_out)  # excluded from the timing
        rng = np.random.default_rng([2026, rank])
        ks = [int(k) for k in np.sort(rng.choice(cand, min(n_sample, len(cand)), replace=False))]
        samples[str(rank)] = {"base": base, "scenario": int(b.scenario), "n_candidates": int(len(cand)), "branches": ks}
        # pre-outage (intact-grid) GENCO inference, timed on its own: CPU batch 1 and MPS batch 1
        genco.solve(ctx["busi"], ctx["gen"], ctx["bi"])  # warm-up
        ms_cpu, cpu_all = best_of(lambda: genco.solve(ctx["busi"], ctx["gen"], ctx["bi"]), reps)
        binf = {"rank": rank, "base": base, "n_sample": len(ks), "ms_cpu_b1": ms_cpu, "ms_cpu_b1_reps": cpu_all}
        if mps_ok:
            try:
                genco.model.to("mps")
                mps_forward(ctx, [ctx["bi"]])  # warm-up
                binf["ms_mps_b1"], binf["ms_mps_b1_reps"] = best_of(lambda: mps_forward(ctx, [ctx["bi"]]), reps)
            except Exception as e:  # noqa: BLE001
                binf["ms_mps_b1"], binf["mps_error"] = float("nan"), str(e)
            finally:
                genco.model.to("cpu")
        base_inf.append(binf)
        rec, crit = {}, {}
        for w in wf_order:
            fn = wfs[w]
            for k in ks[:warm]:  # untimed warm-up
                fn(ctx, k)
            walls = []
            for rep in range(reps):
                tp = time.perf_counter()
                for k in ks:
                    t0 = time.perf_counter()
                    res = fn(ctx, k)
                    ms = 1e3 * (time.perf_counter() - t0)
                    rec.setdefault((w, k), []).append((ms, res))
                walls.append(1e3 * (time.perf_counter() - tp))
            pass_wall[w][str(rank)] = walls
        for k in ks:
            crit[k] = bool(rec[("W0", k)][0][1]["critical"])
        if mps_ok:
            blocks = [ks[i:i + bs] for i in range(0, len(ks), bs)]
            wm = f"W4_mps{bs}"
            try:
                if mps_note == "pending validation":  # CPU reference (genco.solve) before the model moves to MPS
                    vm_c, va_c = infer(ctx, contingency_tables(ctx, blocks[0][0]))
                genco.model.to("mps")
                _shared, chk = mps_block(ctx, blocks[0])  # untimed warm-up block (and the validation on the first base)
                if mps_note == "pending validation":
                    dvm, dva = float(np.abs(chk[0][2] - vm_c).max()), float(np.abs(chk[0][3] - va_c).max())
                    if max(dvm, dva) > 5e-3:
                        raise ValueError(f"batched MPS forward differs from genco.solve: max |dVm| {dvm:.2e} p.u., |dVa| {dva:.2e} rad")
                    mps_note = f"validated on block 0 of base rank {rank}: max |dVm| {dvm:.1e} p.u., |dVa| {dva:.1e} rad vs genco.solve (cpu)"
                    log(f"  {wm} {mps_note}")
                walls = []
                for rep in range(reps):
                    tp = time.perf_counter()
                    for bi_, blk in enumerate(blocks):
                        shared, out = mps_block(ctx, blk)
                        own = sum(o[0] for o in out)
                        blocks_log.append({"rank": rank, "block": bi_, "size": len(blk), "rep": rep, "shared_ms": shared, "newton_ms": own, "wall_ms": shared + own})
                        for k, (ms, res, _vm, _va) in zip(blk, out):
                            rec.setdefault((wm, k), []).append((shared / len(blk) + ms, {**res, "block": bi_, "block_size": len(blk)}))
                    walls.append(1e3 * (time.perf_counter() - tp))
                pass_wall[wm][str(rank)] = walls
            except Exception as e:  # noqa: BLE001 - the variant is optional; the CPU batch-1 numbers stand on their own
                mps_ok, mps_note = False, f"skipped: {e}"
                log(f"  {wm} {mps_note}")
            genco.model.to("cpu")
        for (w, k), lst in rec.items():
            tms = [x[0] for x in lst]
            j = int(np.argmin(tms))
            res = lst[j][1]
            rows.append({"base": base, "rank": rank, "branch": k, "order": ks.index(k), "workflow": w, "ms": tms[j], "ms_first": tms[0], "ms_mean": float(np.mean(tms)),
                         "critical": crit[k], "missed": crit[k] and not res["flagged"], "false_flag": (not crit[k]) and res["flagged"],
                         **{kk: v for kk, v in res.items() if kk != "critical"}})
        el = time.perf_counter() - t_all
        log(f"  base rank {rank} (load_scenario_idx {base}): {len(ks)} outages timed, pre-outage inference cpu b1 {ms_cpu:.1f} ms"
            + (f", mps b1 {binf.get('ms_mps_b1', float('nan')):.1f} ms" if "ms_mps_b1" in binf else "") + f"; {len(rows):,} rows so far, wall {el:.0f} s")
    df = pd.DataFrame(rows)
    for col, dt in (("start", object), ("second", object), ("block", float), ("block_size", float)):
        if col not in df:
            df[col] = pd.Series(index=df.index, dtype=dt)
    df["start"] = df["start"].fillna("")
    df["second"] = df["second"].astype(object).where(df["second"].notna(), None)
    pfile, jfile = OUT / f"e2e_{a.grid}{a.tag}.parquet", OUT / f"e2e_{a.grid}{a.tag}.json"
    df.to_parquet(pfile, index=False)
    summary = []
    order = wf_order[:2] + ["W4", f"W4_mps{bs}"] + wf_order[3:]
    log(f"  MEASURED END-TO-END ms per outage (min of {reps} repetitions per outage; total = sum of the first pass, no min-taking; gate = sent to full AC)")
    log(f"    {'workflow':<10} {'gate':<8} {'n':>5} {'total s':>8} {'mean':>8} {'median':>8} {'p90':>8} {'mean it':>7} {'flagged':>7} {'restart':>7} {'failed':>6} {'crit':>5} {'missed':>6} {'false':>5}")
    for w in [x for x in order if x in set(df.workflow)]:
        part = df[df.workflow == w]
        for name, sub in (("all", part), ("flagged", part[part.flagged]), ("cleared", part[~part.flagged])):
            if not len(sub):
                continue
            r = {"workflow": w, "gate": name, "n": int(len(sub)), "ms_total_first": float(sub.ms_first.sum()), "ms_total_min": float(sub.ms.sum()),
                 "ms_total_mean_rep": float(sub.ms_mean.sum()), "ms_mean": float(sub.ms.mean()), "ms_median": float(sub.ms.median()), "ms_p90": float(sub.ms.quantile(.9)),
                 "iterations_mean": float(sub.iterations.mean()), "flagged": int(sub.flagged.sum()), "flagged_ms": float(sub.ms[sub.flagged].sum()),
                 "fallback_ms_cont": float(sub.ms_cont.sum()), "restarted": int(sub.restarted.sum()), "failed": int(sub.failed.sum()), "failed_ms": float(sub.ms[sub.failed].sum()),
                 "genco_inferences": int(sub.n_inf.sum()), "critical": int(sub.critical.sum()), "missed": int(sub.missed.sum()), "false_flag": int(sub.false_flag.sum())}
            if name == "all":
                r["pass_wall_ms_first"] = float(sum(v[0] for v in pass_wall[w].values()))
                r["pass_wall_ms_per_rep"] = [float(sum(v[i] for v in pass_wall[w].values())) for i in range(reps)] if pass_wall[w] else []
            if w == "gate":
                r["hyb_start_share"] = float((sub.start == "hyb").mean())
            if w.startswith("CF"):
                r["genco_share"] = float(sub.second.astype(bool).mean())
            summary.append(r)
            log(f"    {w:<10} {name:<8} {r['n']:>5} {r['ms_total_first'] / 1e3:>8.2f} {r['ms_mean']:>8.2f} {r['ms_median']:>8.2f} {r['ms_p90']:>8.2f} {r['iterations_mean']:>7.2f} "
                f"{r['flagged']:>7} {r['restarted']:>7} {r['failed']:>6} {r['critical']:>5} {r['missed']:>6} {r['false_flag']:>5}")
    mps_blocks = None
    if blocks_log:
        bl = pd.DataFrame(blocks_log)
        first = bl[bl.rep == 0]
        mps_blocks = {"n_blocks": int(len(first)), "n_partial": int((first["size"] < bs).sum()), "block_sizes": sorted(set(int(s) for s in first["size"])),
                      "outages": int(first["size"].sum()), "shared_ms_total_first": float(first.shared_ms.sum()), "wall_ms_total_first": float(first.wall_ms.sum()),
                      "amortized_shared_ms_per_outage": float(first.shared_ms.sum() / first["size"].sum()), "amortized_ms_per_outage": float(first.wall_ms.sum() / first["size"].sum()),
                      "shared_ms_per_full_block_median": float(first[first["size"] == bs].shared_ms.median()) if (first["size"] == bs).any() else float("nan")}
        log(f"  W4_mps{bs} blocks: {mps_blocks}")
    log(f"  pre-outage GENCO inference per base (cpu b1 / mps b1 ms): " + ", ".join(f"r{x['rank']} {x['ms_cpu_b1']:.1f}/{x.get('ms_mps_b1', float('nan')):.1f}" for x in base_inf))
    load_evidence("end")
    protocol = {
        "order": f"per base: pre-outage inference timed (CPU b1, MPS b1; min of {reps}); then for each workflow in {wf_order}: {warm} untimed warm-up outages "
                 f"(the first of the sample), then {reps} complete passes over the sample in branch-id order (per-outage perf_counter around the complete sequence, "
                 f"contiguous wall per pass recorded); then W4_mps{bs}: one untimed warm-up block, then {reps} passes over the blocks",
        "sample": f"numpy default_rng([2026, rank]).choice(candidates, {n_sample}, replace=False), sorted; candidates = in-service non-bridge branches of design(); ids in `samples`",
        "warmup": f"{warm} untimed outages per workflow and base, one untimed MPS block per base; the first base's first MPS block also validates the batched forward against genco.solve",
        "torch_threads": threads, "torch_threads_source": "--threads" if getattr(a, "threads", 0) else (f"{tfile.name} torch_threads" if tfile.exists() else "torch default"),
        "mps_sync": "torch.mps.synchronize() before the shared timer stops (then .cpu() of the outputs, also inside the timer)",
        "repetitions": f"{reps}; latency columns ms = min per outage, ms_first = first pass, ms_mean = arithmetic mean; totals: ms_total_first = sum of the first pass "
                       "(no min-taking), pass_wall_ms_first = contiguous wall of the first pass summed over bases, ms_total_min = sum of the per-outage minima; "
                       "summary mean/median/p90 are over the per-outage minima",
        "partial_batches": f"the last block of a base may hold fewer than {bs} outages; it is timed as is and its size stored per row (block_size); amortized "
                           "throughput in `mps_blocks` = total block wall / outages",
        "gate_rule": f"flag if max one-step loading > c = {c} or mismatch > tau_mw of the start (or the step is impossible)",
        "solver_failure": "newton(max_it=30): a flagged run is carried on for up to 30 more iterations; if it does not converge (or the start has no step) the outage "
                          "is re-solved from the pre-outage start (restarted=True, +30 iterations max); still not converged -> failed=True, flagged, W0 counts it critical; "
                          "ms includes everything executed, ms_cont the continuation/restart part",
        "classical_first": f"W1's step first; GENCO stage (inference + anchored start + step + gate tau_hyb) where max loading > c or mismatch > tau1 "
                           f"(tau1 in {tau1s}); all executed work timed; `second` marks outages that reached the GENCO stage",
        "mismatch_gate": "inference, both starts' initial mismatch (timed), smaller takes the step with its tau (tie / NaN -> pre-outage)",
        "pre_outage_inference": "per base, timed separately (base_inference); not in the per-outage columns; the scorer amortizes it over the base's outages",
        "excluded": ["model load", "pre-outage AC solve (base_context)", "warm-up"],
    }
    jfile.write_text(json.dumps({
        "network": net, "tag": a.tag, "data": str(raw), "torch_version": torch.__version__, "torch_threads": threads, "bases": [int(r) for r in bases["rank"]],
        "sample_outages_per_base": n_sample, "sample_policy": n_policy, "sample_override": sample_override, "limit": int(a.limit or 0), "repetitions": reps, "warmup_outages": warm,
        "gate_c": c, "tau_mw": tau, "tau1_mw": tau1s, "lambda": lam, "rng": "numpy default_rng([2026, rank])", "workflow_order": wf_order + [f"W4_mps{bs}"],
        "excluded": protocol["excluded"], "critical": "W0's own verdict (not converged or a loading > 1.001 at its solution)",
        f"W4_mps{bs}": mps_note, "protocol": protocol, "samples": samples, "base_inference": base_inf, "pass_wall_ms": pass_wall, "mps_blocks": mps_blocks,
        "summary": summary}, indent=1))
    log(f"saved {pfile} ({len(df):,} rows) and {jfile}")


# ----------------------------------------------------------------------------- report
def cp_upper(k, n, conf=0.95):
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


WF_START = {"W1": "prev", "W3": "genco", "W4": "hyb", "W4s": "shyb"}


def workflows(df, tau, lim=LIM):
    """Per-contingency outcome of each workflow: sent to full AC (bool array) and Newton iterations spent.

    lim is the decision cutoff on the one-step loadings (the loading cushion c of the frozen policy; truth stays LIM).
    W4s (shrunk anchored start) is included only when tau has a "W4s" entry and the chunks hold the shyb columns.
    """
    import numpy as np
    it_prev = df.it_prev.to_numpy()
    out = {"W0": (np.ones(len(df), bool), it_prev)}
    for w, s in WF_START.items():
        if w == "W4s" and (w not in tau or f"max1_{s}" not in df):
            continue
        flag = (df[f"max1_{s}"].to_numpy() > lim) | ~(df[f"mis1_{s}"].to_numpy() <= tau[w])
        it_s, conv = df[f"it_{s}"].to_numpy(), df[f"conv_{s}"].to_numpy()
        # continuing = the same Newton run carried on; if it breaks down, restart from the pre-outage solution
        cont = np.where(conv | (s == "prev"), it_s, it_s + it_prev)
        out[w] = (flag, np.where(flag, cont, np.minimum(it_s, 1)))
    out["W2"] = (df.max_genco.to_numpy() > lim, None)
    out["W5"] = (df.max_dc.to_numpy() > lim, None)
    for w in ("W2", "W5"):
        out[w] = (out[w][0], np.where(out[w][0], it_prev, 0))
    return {w: out[w] for w in ("W0", "W1", "W2", "W3", "W4", "W4s", "W5") if w in out}


def load_chunks(d, bases):
    """All finished chunks of one chunk folder: per-contingency frame (with split / rank / load_mw and `critical`), histograms, base info."""
    import numpy as np
    import pandas as pd
    files = [f for f in sorted(d.glob("b*_c*.parquet")) if ".tmp" not in f.name]
    if not files:
        raise FileNotFoundError(f"no finished chunks in {d}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    hist = {h: np.zeros(NB + 1, np.int64) for h in HIST}
    nanc = {h: 0 for h in HIST}
    info = {}
    for f in files:
        z = np.load(f.with_suffix(".npz"))
        for h in HIST:
            hist[h] += z[h]
            nanc[h] += int(z[f"nan_{h}"])
        i = json.loads(str(z["info"]))
        info[i["base"]] = i
    df = df.merge(bases[["base", "split", "rank", "load_mw"]], on="base")
    df["critical"] = (df.n_real > 0) | ~df.truth_conv
    return df, hist, nanc, info, files


def pick_tau(cal, w, s):
    """Largest tau on the grid with no missed critical contingency on the calibration bases (fewest iterations at zero misses)."""
    import numpy as np
    crit = ((cal.n_real > 0) | ~cal.truth_conv).to_numpy()
    table, best = [], None
    for tau in TAUS:
        flag = (cal[f"max1_{s}"].to_numpy() > LIM) | ~(cal[f"mis1_{s}"].to_numpy() <= tau)
        miss = int((crit & ~flag).sum())
        table.append((tau, float(flag.mean()), miss))
        if miss == 0:
            best = tau
    return (best if best is not None else TAUS[0]), table


def report(a):
    import numpy as np
    import pandas as pd
    bases, always_out, bridge, cand = design(a.grid)
    here, net, raw = pc.paths(a.grid)
    d = chunk_dir(a.grid, a.tag)
    df, hist, nanc, info, files = load_chunks(d, bases)
    expected = len(bases) * len(cand)
    log(f"=== report {net}{a.tag}: {len(df):,} contingencies in {len(files)} chunks of {d} (expected {expected:,}: {len(bases)} base cases x {len(cand):,} non-islanding single-branch outages; "
        f"{len(always_out):,} branches, {int(always_out.sum())} never in service, {int(bridge.sum())} bridges)")
    if a.tag:
        log(f"  NOTE: tagged report re-fits tau on the calibration split (development only); the frozen-policy scoring is work/fresh/score_fresh_n1.py. "
            f"Outputs go to contingencies{a.tag}.parquet / summary{a.tag}.csv; the untagged files are not touched.")
    inf = pd.DataFrame(info.values()).merge(bases, on="base").sort_values("rank")
    log("  base cases = 24 load scenarios of the stored datakit table, evenly spaced over the ranking of the 500 load scenarios by total load; dispatch, setpoints and "
        "perturbed admittances of the first stored scenario of each; removed branches restored; intact grid solved by Newton")
    log(f"  total load {inf.load_mw_x.min():,.0f} .. {inf.load_mw_x.max():,.0f} MW; intact-grid Newton iterations max {inf.intact_it.max()}; admittance formula check max {inf.ydiff.max():.1e}; "
        f"base cases with an overload already in the intact grid: {int((inf.intact_overloads > 0).sum())} of {len(inf)} (lines: {int(inf.intact_overloads.sum())}, worst {inf.intact_worst.max():.1%})")
    log(f"  calibration bases (rank by load) {inf[inf.split == 'calibration']['rank'].tolist()}, load_scenario_idx {inf[inf.split == 'calibration'].base.tolist()}; "
        f"test bases: the other {int((inf.split == 'test').sum())}")

    # ---- ground truth (df.critical set by load_chunks)
    crit = df.critical
    log(f"  ground truth: converged {int(df.truth_conv.sum()):,}/{len(df):,} (from the pre-outage start {int(df.conv_prev.sum()):,}; rescued by another start {int((df.truth_conv & ~df.conv_prev).sum())}; "
        f"no solution from any start {int((~df.truth_conv).sum())}); solutions from different starts differ by > 1e-5 p.u. in "
        f"{int((df[[f'dv_{s}' for s in STARTS]].max(axis=1) > 1e-5).sum())} contingencies")
    ok = df[df.truth_conv]
    log(f"  contingencies with a real overload: {int((ok.n_real > 0).sum()):,} ({(ok.n_real > 0).mean():.1%}); overloaded line readings {int(ok.n_real.sum()):,}; worst loading {ok.worst.max():.1%}; "
        f"lowest bus voltage {ok.vmin.min():.3f} p.u.")
    log("  Newton to convergence, iterations mean / median / max, converged:  " + ";  ".join(
        f"{s} {df[df[f'conv_{s}']][f'it_{s}'].mean():.3f} / {df[df[f'conv_{s}']][f'it_{s}'].median():.0f} / {df[df[f'conv_{s}']][f'it_{s}'].max()}, {int(df[f'conv_{s}'].sum()):,}" for s in STARTS))
    q = lambda v: f"{v.replace(np.inf, np.nan).median():.3g} / {v.replace(np.inf, np.nan).quantile(.99):.3g} / {v.replace(np.inf, np.nan).max():.3g}"  # noqa: E731
    log("  bus mismatch MW (max-norm) median / p99 / max:  " + ";  ".join(f"{s} start {q(df[f'mis0_{s}'])}, after one step {q(df[f'mis1_{s}'])}" for s in STARTS))

    # ---- N-1 accuracy (all 24 bases, every rated in-service line of every converged contingency)
    def hq(h, p):
        c = np.cumsum(hist[h])
        i = int(np.searchsorted(c, p * c[-1]))
        return (i + 1) * HBIN if i < NB else np.inf
    log(f"  N-1 LOADING ACCURACY, points of rating, {int(hist['prev1'].sum()):,} line readings (histogram resolution {HBIN} point; quantiles are upper bin edges)")
    log(f"    {'predictor':<34} {'median':>8} {'p99':>8} {'p99.9':>8} {'max':>9} {'NaN':>5} | per contingency worst line: median {'p99':>8} | lines: missed {'false':>8}")
    names = {"prev0": "pre-outage loading as is (0 steps)", "prev1": "pre-outage start + 1 Newton step", "genco1": "GENCO start + 1 Newton step", "hyb1": "hybrid start + 1 Newton step",
             "genco": "GENCO raw", "hyb0": "hybrid start (0 steps)", "dc_anch": "DC anchored on base AC"}
    acc = []
    for h in HIST:
        e = ok[f"emax_{h}"]
        lm = int(ok[f"lmiss_{h}"].sum()) if f"lmiss_{h}" in ok else -1
        lf = int(ok[f"lfalse_{h}"].sum()) if f"lfalse_{h}" in ok else -1
        acc.append({"network": net, "kind": "n1_accuracy", "item": h, "err_median_pp": hq(h, .5), "err_p99_pp": hq(h, .99), "err_p999_pp": hq(h, .999), "err_max_pp": float(e.max()),
                    "worst_line_err_median_pp": float(e.median()), "worst_line_err_p99_pp": float(e.quantile(.99)), "lines_missed": lm, "lines_false": lf, "line_readings": int(hist[h].sum())})
        log(f"    {names[h]:<34} {hq(h, .5):>8.3f} {hq(h, .99):>8.3f} {hq(h, .999):>8.3f} {e.max():>9.2f} {nanc[h]:>5} | {e.median():>33.3f} {e.quantile(.99):>8.2f} | "
            f"{lm if lm >= 0 else '':>13} {lf if lf >= 0 else '':>8}")

    # ---- thresholds on the calibration bases
    cal, test = df[df.split == "calibration"], df[df.split == "test"]
    tau = {}
    log(f"  MISMATCH THRESHOLD (MW, max-norm after the step), chosen on the {cal.base.nunique()} calibration bases ({len(cal):,} contingencies, {int(cal.critical.sum()):,} critical): "
        f"largest value of the grid with no critical contingency accepted")
    for w, s in (("W1", "prev"), ("W3", "genco"), ("W4", "hyb")):
        tau[w], table = pick_tau(cal, w, s)
        log(f"    {w} ({s}): tau = {tau[w]} MW;  tau: share sent to AC / missed on calibration  " + "  ".join(f"{t:g}: {f:.1%}/{m}" for t, f, m in table))

    # ---- unit costs
    tfile = OUT / f"timing_{a.grid}{a.tag}.json"
    tm = json.loads(tfile.read_text()) if tfile.exists() else None
    if tm:
        log(f"  UNIT COSTS (indicative, single process, see [load] lines of the timing run): Newton iteration {tm['ms_iter_full']:.2f} ms (single step call {tm['ms_step1']:.2f}); "
            f"GENCO Tiny forward cpu batch 1 {tm['ms_forward_cpu_b1']:.2f} ms ({tm['torch_threads']} threads), mps batch 16 {tm.get('ms_forward_mps_b16', float('nan')):.2f} ms/state; "
            f"graph build + normalise {tm['ms_prep']:.2f} ms per state (needed before any forward); DC solve {tm['ms_dc_full']:.2f} ms incl. assembly ({tm['ms_dc_factor_solve']:.2f} factor+solve only); "
            f"common to all workflows and not counted: problem build {tm['ms_build']:.2f} ms, loadings {tm['ms_loading']:.2f} ms")

    # ---- workflows
    rows = []
    label = {"W0": "full AC, every contingency", "W1": "1 Newton step from pre-outage", "W2": "GENCO only", "W3": "GENCO start + 1 step", "W4": "hybrid start + 1 step", "W5": "DC anchored on base AC"}
    for name, part in (("test", test), ("calibration", cal), ("all", df)):
        wf = workflows(part, tau)
        c = part.critical.to_numpy()
        n, n_crit, n_sec = len(part), int(c.sum()), int((~c).sum())
        n_lines = int(part.n_real.clip(lower=0).sum())
        nb = part.base.nunique()
        it0 = int(wf["W0"][1].sum())
        log(f"  WORKFLOWS on the {name} bases: {nb} base cases, {n:,} contingencies, {n_crit:,} critical ({n_crit / n:.1%}; {int((~part.truth_conv).sum())} without a solution), "
            f"{n_lines:,} overloaded line readings; W0 = {it0:,} Newton iterations ({it0 / n:.3f} per contingency)")
        log(f"    {'workflow':<34} {'to AC':>7} {'secure->AC':>10} {'missed cont.':>13} {'share':>7} {'CP95 up':>8} {'missed lines':>13} {'share':>7} {'worst':>7} {'Newton it':>10} {'vs W0':>7} "
            f"{'GENCO inf':>9} {'DC':>7} | s per 1,000 contingencies: {'cpu b1':>7} {'mps b16':>8} {'+prep cpu':>9} {'+prep mps':>9}")
        for w, (flag, its) in wf.items():
            acc_ = ~flag
            missed = acc_ & c
            k = int(missed.sum())
            ml = int(part.n_real.to_numpy()[missed].clip(min=0).sum())
            worst = float(np.nanmax(part.worst.to_numpy()[missed])) if k and np.isfinite(part.worst.to_numpy()[missed]).any() else np.nan
            n_inf = {"W0": 0, "W1": 0, "W2": n, "W3": n, "W4": n + nb, "W5": 0}[w]
            n_dc = n + nb if w == "W5" else 0
            r = {"network": net, "kind": "workflow", "bases": name, "item": w, "label": label[w], "tau_mw": tau.get(w, np.nan), "contingencies": n, "critical": n_crit,
                 "sent_to_ac": int(flag.sum()), "sent_to_ac_share": float(flag.mean()), "secure_sent_to_ac": int((flag & ~c).sum()), "secure_sent_to_ac_share": float((flag & ~c).sum() / max(n_sec, 1)),
                 "missed_contingencies": k, "missed_share_of_critical": k / max(n_crit, 1), "missed_cp95_upper_of_critical": cp_upper(k, n_crit), "missed_cp95_upper_of_all": cp_upper(k, n),
                 "missed_no_solution": int((missed & ~part.truth_conv.to_numpy()).sum()),
                 "overloaded_lines": n_lines, "missed_lines": ml, "missed_lines_share": ml / max(n_lines, 1), "missed_lines_cp95_upper": cp_upper(ml, n_lines), "worst_missed_loading": worst,
                 "newton_iterations": int(its.sum()), "iterations_vs_w0": float(its.sum() / it0), "genco_inferences": n_inf, "dc_solves": n_dc}
            wall = ["", "", "", ""]
            if tm:
                base_ms = r["newton_iterations"] * tm["ms_iter_full"] + n_dc * tm["ms_dc_full"]
                for j, (dev, prep) in enumerate((("cpu_b1", 0), ("mps_b16", 0), ("cpu_b1", 1), ("mps_b16", 1))):
                    fw = tm.get(f"ms_forward_{dev}", np.nan)
                    key = f"s_per_1000_{dev}" + ("_with_prep" if prep else "")
                    r[key] = (base_ms + n_inf * (fw + prep * tm["ms_prep"])) / n  # ms per contingency = s per 1,000
                    wall[j] = f"{r[key]:.2f}"
            rows.append(r)
            log(f"    {w + ' ' + label[w]:<34} {r['sent_to_ac_share']:>7.1%} {r['secure_sent_to_ac_share']:>10.2%} {k:>6}/{n_crit:<6} {r['missed_share_of_critical']:>7.2%} "
                f"{r['missed_cp95_upper_of_critical']:>8.2%} {ml:>6}/{n_lines:<6} {r['missed_lines_share']:>7.2%} {'' if np.isnan(worst) else f'{worst:.1%}':>7} {r['newton_iterations']:>10,} "
                f"{r['iterations_vs_w0']:>7.1%} {n_inf:>9,} {n_dc:>7,} | {'':<27} {wall[0]:>7} {wall[1]:>8} {wall[2]:>9} {wall[3]:>9}")
    # sensitivity of W1 / W3 / W4 to tau on the test bases (not used for the choice)
    c = test.critical.to_numpy()
    it0 = int(test.it_prev.sum())
    log("  sensitivity on the test bases (for information; tau was fixed on calibration): tau -> Newton iterations vs W0 / missed contingencies")
    for w, s in (("W1", "prev"), ("W3", "genco"), ("W4", "hyb")):
        cells = []
        for t in TAUS:
            flag, its = workflows(test, {**tau, w: t})[w]
            cells.append(f"{t:g}: {its.sum() / it0:.1%}/{int((~flag & c).sum())}")
            rows.append({"network": net, "kind": "tau_sensitivity", "bases": "test", "item": w, "tau_mw": t, "sent_to_ac_share": float(flag.mean()),
                         "missed_contingencies": int((~flag & c).sum()), "iterations_vs_w0": float(its.sum() / it0)})
        log(f"    {w}: " + "  ".join(cells))

    # ---- per-contingency file
    wf = workflows(df, tau)
    keep = df[["base", "branch", "split", "load_mw", "pre_loading", "truth_conv", "truth_from", "critical", "n_real", "worst", "vmin"]
              + [f"{k}_{s}" for s in STARTS for k in ("conv", "it", "mis0", "mis1", "max1")] + ["max_genco", "max_dc"]
              + [f"emax_{h}" for h in HIST] + [f"lmiss_{h}" for h in ("prev1", "genco1", "hyb1", "genco", "dc_anch")]].copy()
    keep = keep.rename(columns={"n_real": "real_overloads", "worst": "worst_loading"})
    for w, (flag, its) in wf.items():
        keep[f"{w}_to_ac"] = flag
        keep[f"{w}_verdict"] = np.where(flag, np.where(df.critical, "needs attention", "secure"), "secure")
        keep[f"{w}_missed"] = ~flag & df.critical.to_numpy()
        keep[f"{w}_iterations"] = its
    pfile, sfile = OUT / f"contingencies{a.tag}.parquet", OUT / f"summary{a.tag}.csv"
    keep.to_parquet(pfile, index=False)
    pd.concat([pd.DataFrame(rows), pd.DataFrame(acc)], ignore_index=True).to_csv(sfile, index=False)
    log(f"saved {pfile} ({len(keep):,} rows) and {sfile}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["run", "timing", "report", "e2e"])
    p.add_argument("--grid", default="2000")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--bases", type=int, nargs="*", help="ranks (0..23) of the base cases to run; default all (e2e: the policy's n1_base_ranks)")
    p.add_argument("--limit", type=int, default=0, help="pilot: this many outages per base (e2e: cap on the sample per base)")
    p.add_argument("--tag", default="", help="suffix of chunks_<grid>, timing_<grid>.json, e2e_<grid>.*, contingencies / summary (pilot and fresh runs)")
    p.add_argument("--states", type=int, default=41)
    p.add_argument("--policy", default="", help="frozen policy json (work/fresh/policy_frozen.json): run adds the shrunk start shyb; e2e needs it")
    p.add_argument("--threads", type=int, default=0, help="e2e: torch.set_num_threads (default: torch_threads of work/fulln1/timing_<grid>.json)")
    p.add_argument("--sample", type=int, default=0, help="e2e: override of the policy's sample_outages_per_base (recorded in the json as a deviation)")
    a = p.parse_args()
    {"run": run, "timing": timing, "report": report, "e2e": e2e}[a.phase](a)

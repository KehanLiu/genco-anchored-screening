"""Complete N-1 contingency analysis on a GOC grid and its cost under six workflows.

Unit of work: one base case (a load scenario of the stored datakit table, dispatched as datakit did; the intact grid
is rebuilt and solved as in analysis_prev_control.py) x EVERY single branch outage that does not island the grid.

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

    work/texas/tmo 9000 .venv/bin/python -u analysis_full_n1.py run --grid 2000 --procs 4     (resumable: chunk files)
    work/texas/tmo 600  .venv/bin/python -u analysis_full_n1.py timing --grid 2000            (single process, quiet)
    .venv/bin/python analysis_full_n1.py report --grid 2000
Writes work/fulln1/chunks_<grid>/*, contingencies.parquet, summary.csv, timing_<grid>.json, report.log.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
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


def _init(grid):
    import engine  # Julia before torch
    import torch
    torch.set_num_threads(1)
    here, net, raw = pc.paths(grid)
    engine.WORK = here
    _W.update(genco=engine.Genco(net, "tiny"), nw=pc.newton_module(), raw=raw, ctx={})


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
    genco, nw = _W["genco"], _W["nw"]
    ctx = base_context(load_idx, scenario, always_out)
    busi, V_prev = ctx["busi"], ctx["V"]
    vm_p, va_p = np.abs(V_prev), np.angle(V_prev)
    hist = {h: np.zeros(NB + 1, np.int64) for h in HIST}
    nan_count = {h: 0 for h in HIST}
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
        for s in STARTS:
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
            for h, p in preds.items():
                e = np.abs(p - truth)[prob.rated] * 100
                ok = np.isfinite(e)
                nan_count[h] += int((~ok).sum())
                hist[h] += np.bincount(np.minimum(e[ok] / HBIN, NB).astype(np.int64), minlength=NB + 1)
                row[f"emax_{h}"] = float(e[ok].max()) if ok.any() else np.nan
            # overloaded lines each predictor calls (for line-level false alarms) and the real ones it does not call
            for h in ("prev1", "genco1", "hyb1", "genco", "dc_anch"):
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


def chunk_dir(grid):
    return OUT / f"chunks_{grid}"


def run(a):
    import numpy as np
    bases, always_out, bridge, cand = design(a.grid)
    here, net, raw = pc.paths(a.grid)
    if a.bases:
        bases = bases[bases["rank"].isin(a.bases)]
    if a.limit:
        cand = cand[np.round(np.linspace(0, len(cand) - 1, a.limit)).astype(int)]
    d = OUT / (f"chunks_{a.grid}" + a.tag)
    d.mkdir(parents=True, exist_ok=True)
    log(f"=== run {net}{a.tag}: {len(bases)} base cases x {len(cand):,} single-branch outages (of {len(always_out):,} branches: {int(always_out.sum())} always out, "
        f"{int(bridge.sum())} bridges skipped) = {len(bases) * len(cand):,} contingencies, {a.procs} processes")
    jobs = []
    for b in bases.itertuples():
        for cid, br in enumerate(np.array_split(cand, max(int(np.ceil(len(cand) / CHUNK)), 1))):
            jobs.append((a.grid, int(b.base), int(b.scenario), always_out, cid, br, str(d / f"b{int(b.base):04d}_c{cid:03d}")))
    t0 = time.perf_counter()
    done = n = 0
    with get_context("spawn").Pool(a.procs, initializer=_init, initargs=(a.grid,)) as pool:
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
    log(f"=== timing {net} (INDICATIVE; single process, pid {os.getpid()}): torch {torch.__version__}, {torch.get_num_threads()} CPU threads, mps {torch.backends.mps.is_available()}; {a.states} contingencies")
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
    (OUT / f"timing_{a.grid}.json").write_text(json.dumps(out, indent=1))
    log(f"saved {OUT / f'timing_{a.grid}.json'}")


# ----------------------------------------------------------------------------- report
def cp_upper(k, n, conf=0.95):
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def workflows(df, tau):
    """Per-contingency outcome of each workflow: sent to full AC (bool array) and Newton iterations spent."""
    import numpy as np
    it_prev = df.it_prev.to_numpy()
    out = {"W0": (np.ones(len(df), bool), it_prev)}
    for w, s in (("W1", "prev"), ("W3", "genco"), ("W4", "hyb")):
        flag = (df[f"max1_{s}"].to_numpy() > LIM) | ~(df[f"mis1_{s}"].to_numpy() <= tau[w])
        it_s, conv = df[f"it_{s}"].to_numpy(), df[f"conv_{s}"].to_numpy()
        # continuing = the same Newton run carried on; if it breaks down, restart from the pre-outage solution
        cont = np.where(conv | (s == "prev"), it_s, it_s + it_prev)
        out[w] = (flag, np.where(flag, cont, np.minimum(it_s, 1)))
    out["W2"] = (df.max_genco.to_numpy() > LIM, None)
    out["W5"] = (df.max_dc.to_numpy() > LIM, None)
    for w in ("W2", "W5"):
        out[w] = (out[w][0], np.where(out[w][0], it_prev, 0))
    return {w: out[w] for w in ("W0", "W1", "W2", "W3", "W4", "W5")}


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
    d = chunk_dir(a.grid)
    files = sorted(d.glob("b*_c*.parquet"))
    files = [f for f in files if ".tmp" not in f.name]
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
    expected = len(bases) * len(cand)
    log(f"=== report {net}: {len(df):,} contingencies in {len(files)} chunks (expected {expected:,}: {len(bases)} base cases x {len(cand):,} non-islanding single-branch outages; "
        f"{len(always_out):,} branches, {int(always_out.sum())} never in service, {int(bridge.sum())} bridges)")
    inf = pd.DataFrame(info.values()).merge(bases, on="base").sort_values("rank")
    log("  base cases = 24 load scenarios of the stored datakit table, evenly spaced over the ranking of the 500 load scenarios by total load; dispatch, setpoints and "
        "perturbed admittances of the first stored scenario of each; removed branches restored; intact grid solved by Newton")
    log(f"  total load {inf.load_mw_x.min():,.0f} .. {inf.load_mw_x.max():,.0f} MW; intact-grid Newton iterations max {inf.intact_it.max()}; admittance formula check max {inf.ydiff.max():.1e}; "
        f"base cases with an overload already in the intact grid: {int((inf.intact_overloads > 0).sum())} of {len(inf)} (lines: {int(inf.intact_overloads.sum())}, worst {inf.intact_worst.max():.1%})")
    log(f"  calibration bases (rank by load) {inf[inf.split == 'calibration']['rank'].tolist()}, load_scenario_idx {inf[inf.split == 'calibration'].base.tolist()}; "
        f"test bases: the other {int((inf.split == 'test').sum())}")

    # ---- ground truth
    crit = (df.n_real > 0) | ~df.truth_conv
    df["critical"] = crit
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
    tfile = OUT / f"timing_{a.grid}.json"
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
    keep.to_parquet(OUT / "contingencies.parquet", index=False)
    pd.concat([pd.DataFrame(rows), pd.DataFrame(acc)], ignore_index=True).to_csv(OUT / "summary.csv", index=False)
    log(f"saved {OUT / 'contingencies.parquet'} ({len(keep):,} rows) and {OUT / 'summary.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["run", "timing", "report"])
    p.add_argument("--grid", default="2000")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--bases", type=int, nargs="*", help="ranks (0..23) of the base cases to run; default all")
    p.add_argument("--limit", type=int, default=0, help="pilot: this many outages per base")
    p.add_argument("--tag", default="", help="suffix of the chunk folder (pilot runs)")
    p.add_argument("--states", type=int, default=41)
    a = p.parse_args()
    {"run": run, "timing": timing, "report": report}[a.phase](a)

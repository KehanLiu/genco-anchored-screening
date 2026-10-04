"""Control experiment: "pre-outage AC solution + one / two Newton steps" against "GENCO + one / two Newton steps".

For every stored GOC scenario (branches removed after the dispatch was fixed) the pre-outage state is rebuilt:
same loads, dispatch, setpoints and perturbed admittances, removed branches back in service (their admittances
recomputed from the stored r/x/b/tap; the formula is checked against the stored admittances of in-service
branches), solved with the SciPy Newton of analysis_warmstart.Problem (constructor from work/goc2000/02_newton.py).
Scenarios whose inputs are bit-identical apart from the removed branches (hash of r, x, b, tap, Pd, Qd, bus types,
setpoints, non-slack Pg, generator status) share one intact-grid solve.

From the pre-outage voltages, on the contingency network:
    prev0 / prev1 / prev2   line loading after 0 / 1 / 2 Newton steps
    anch_dc                 AC loading of the pre-outage state + (DC loading after - DC loading before the outage)
    anch_genco              AC loading of the pre-outage state + (GENCO loading after - GENCO loading before)
(DC power flow is this script's own, checked against datakit's stored DC flows of the contingency state; GENCO
"before" is one extra inference on the intact-grid tables.)

    work/texas/tmo 3000 .venv/bin/python -u analysis_prev_control.py run --grid 500 [--procs 4] [--limit N]
    .venv/bin/python analysis_prev_control.py report
Writes work/goc<N>/lines_prev_tiny_seed0.parquet, states_prev_tiny_seed0.parquet, work/prev_control_summary.csv,
log work/prev_control.log.
"""
import argparse
import hashlib
import importlib.util
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LOG = ROOT / "work" / "prev_control.log"
BASE = 100.0
LIM = 1.001
BIG = 9.99
YCOLS = ["Yff_r", "Yff_i", "Yft_r", "Yft_i", "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i"]


def log(*a):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def paths(grid):
    here = ROOT / "work" / f"goc{grid}"
    net = f"case{grid}_goc"
    return here, net, here / "data" / net / "raw"


def newton_module():
    spec = importlib.util.spec_from_file_location("goc_newton", ROOT / "work" / "goc2000" / "02_newton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def admittances(br):
    """MATPOWER branch admittances from r, x, b, tap (no phase shifters in these cases)."""
    import numpy as np
    ys = 1.0 / (br.r.to_numpy() + 1j * br.x.to_numpy())
    tap = br.tap.to_numpy()
    ytt = ys + 0.5j * br.b.to_numpy()
    return {"Yff": ytt / tap**2, "Yft": -ys / tap, "Ytf": -ys / tap, "Ytt": ytt}


def dc_loading(bus, branch):
    """DC power flow as PowerModels' compute_dc_pf does it (b = x / (r^2 + x^2), no taps). |flow| / rate per branch idx."""
    import numpy as np
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import splu
    live = (branch.br_status == 1).to_numpy()
    br = branch[live]
    n, m = len(bus), len(br)
    f, t = br.from_bus.to_numpy(int), br.to_bus.to_numpy(int)
    r, x = br.r.to_numpy(), br.x.to_numpy()
    b = x / (r * r + x * x)
    A = csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[f, t])), shape=(m, n))
    P = (bus.Pg - bus.Pd).to_numpy() / BASE - bus.GS.to_numpy()
    ref = np.flatnonzero(bus.REF.to_numpy() == 1)[0]
    keep = np.setdiff1d(np.arange(n), [ref])
    th = np.zeros(n)
    th[keep] = splu((A.T @ diags(b) @ A).tocsc()[keep][:, keep]).solve(P[keep])
    out = np.full(len(branch), np.nan)
    with np.errstate(all="ignore"):
        out[live] = np.abs(b * (A @ th)) * BASE / br.rate_a.to_numpy()
    return out


def worker(job):
    grid, load_ids, always_out, wid = job
    import engine  # Julia before torch
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    here, net, raw = paths(grid)
    engine.WORK = here
    genco = engine.Genco(net, "tiny")
    nw = newton_module()
    lines, rows = [], []
    n_solves = n_attempts = 0
    for sub in np.array_split(load_ids, max(len(load_ids) // 40, 1)):
        flt = [("load_scenario_idx", "in", [int(i) for i in sub])]
        tabs = {t: {s: d for s, d in pd.read_parquet(raw / f"{t}_data.parquet", filters=flt).groupby("scenario")} for t in ("bus", "gen", "branch")}
        groups = {}
        for s in sorted(tabs["bus"]):
            bus = tabs["bus"][s].sort_values("bus").reset_index(drop=True)
            gen = tabs["gen"][s].sort_values("idx").reset_index(drop=True)
            branch = tabs["branch"][s].sort_values("idx").reset_index(drop=True)
            q = gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
            bus = bus.merge(q, on="bus", how="left").fillna(0)
            ref = bus.REF.to_numpy() == 1
            h = hashlib.sha1()
            for arr in (branch.r, branch.x, branch.b, branch.tap, bus.Pd, bus.Qd, bus.PQ + 2 * bus.PV + 3 * bus.REF, bus.GS, bus.BS,
                        np.where(bus.PQ.to_numpy() == 1, 0.0, bus.Vm.to_numpy()), np.round(np.where(ref, 0.0, bus.Pg.to_numpy()), 6), gen.in_service):
                h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
            groups.setdefault((int(bus.load_scenario_idx.iloc[0]), h.hexdigest()), []).append((s, bus, gen, branch))
        for (load_idx, key), members in groups.items():
            # ---- intact grid: removed branches back in service
            s0, bus0, gen0, branch0 = members[0]
            restore = (branch0.br_status.to_numpy() == 0) & ~always_out
            # union over the group: a branch out in one member but not always out is restored in all of them
            bi = branch0.copy()
            bi["br_status"] = np.where(always_out, 0, 1).astype(branch0.br_status.dtype)
            y = admittances(bi)
            for k, v in y.items():
                bi[f"{k}_r"], bi[f"{k}_i"] = np.where(always_out, 0.0, v.real), np.where(always_out, 0.0, v.imag)
            prob_i = nw.make_problem(bus0, bi)
            V_prev, it_i, tries = None, -1, 0
            for (s, bus, gen, branch) in members:  # start from a member's stored contingency solution until one converges
                tries += 1
                v_ac = bus.Vm.to_numpy(float) * np.exp(1j * np.radians(bus.Va.to_numpy(float)))
                r = prob_i.newton(v_ac)
                if r["converged"]:
                    V_prev, it_i = r["V"], r["iterations"]
                    break
            if V_prev is None:  # last resort: flat start
                tries += 1
                r = prob_i.newton(prob_i.start())
                if r["converged"]:
                    V_prev, it_i = r["V"], r["iterations"]
            n_solves += 1
            n_attempts += tries
            ok = V_prev is not None
            live_i = (bi.br_status == 1).to_numpy()
            ni = int(live_i.sum())
            ac_prev = np.full(len(bi), np.nan)
            dc_prev = np.full(len(bi), np.nan)
            g_prev = np.full(len(bi), np.nan)
            if ok:
                ac_prev[live_i] = prob_i.loading(V_prev)
                # intact-grid tables for GENCO (solution written in so that every column is consistent) and for DC
                Sb = V_prev * np.conj(prob_i.Ybus @ V_prev) * BASE
                busi = bus0.copy()
                isgen = bus0.PQ.to_numpy() != 1
                busi["Vm"], busi["Va"] = np.abs(V_prev), np.degrees(np.angle(V_prev))
                busi["Pg"] = np.where(isgen, Sb.real + bus0.Pd.to_numpy(), 0.0)
                busi["Qg"] = np.where(isgen, Sb.imag + bus0.Qd.to_numpy(), 0.0)
                Sf = V_prev[prob_i.f] * np.conj(prob_i.Yf @ V_prev) * BASE
                St = V_prev[prob_i.t] * np.conj(prob_i.Yt @ V_prev) * BASE
                for c, v in (("pf", Sf.real), ("qf", Sf.imag), ("pt", St.real), ("qt", St.imag)):
                    col = np.zeros(len(bi))
                    col[live_i] = v
                    bi[c] = col
                dc_prev = dc_loading(busi, bi)
                _, br_pred = genco.solve(busi, gen0, bi)
                g_prev[live_i] = np.maximum(br_pred.loading[:ni].to_numpy(), br_pred.loading[ni:].to_numpy())
            # ---- every contingency of this base state
            for (s, bus, gen, branch) in members:
                prob = nw.make_problem(bus, branch)
                live = (branch.br_status == 1).to_numpy()
                n = int(live.sum())
                br = branch[live]
                rate = br.rate_a.to_numpy()
                idx = br.idx.to_numpy()
                yc = admittances(br)
                ydiff = max(float(np.abs(yc[k] - (br[f"{k}_r"].to_numpy() + 1j * br[f"{k}_i"].to_numpy())).max()) for k in yc)
                v_ac = bus.Vm.to_numpy(float) * np.exp(1j * np.radians(bus.Va.to_numpy(float)))
                with np.errstate(all="ignore"):
                    s_ac = np.maximum(np.hypot(br.pf, br.qf), np.hypot(br.pt, br.qt)).to_numpy()
                    dc_c = dc_loading(bus, branch)[live]
                    dc_stored = np.abs(br.pf_dc.to_numpy()) / rate
                row = {"state": int(s), "load_idx": load_idx, "base_key": key[:12], "n_out": int((~live).sum()) - int(always_out.sum()),
                       "group_size": len(members), "intact_ok": ok, "intact_it": it_i, "intact_tries": tries, "ydiff": ydiff,
                       "dc_check_pp": float(100 * np.nanmax(np.abs(dc_c - dc_stored)[rate > 0]))}
                nanv = np.full(n, np.nan)
                p0 = p1 = p2 = nanv
                if ok:
                    v0 = prob.start(vm_pq=np.abs(V_prev), va=np.angle(V_prev))
                    r = prob.newton(v0)
                    row.update({"conv_prev": r["converged"], "it_prev": r["iterations"], "mis0_prev": r["mis0"],
                                "dv_prev": float(np.abs(r["V"] - v_ac).max()) if r["converged"] else np.nan,
                                "dv_start": float(np.abs(v0 - v_ac).max())})
                    for k in (1, 2):
                        with np.errstate(all="ignore"):
                            row[f"mis{k}_prev"] = float(np.abs(prob.mismatch(r["steps"][k - 1])).max() * BASE) if len(r["steps"]) >= k else (
                                r["mis_end"] if r["converged"] else np.inf)

                    def after(k):
                        if len(r["steps"]) >= k:
                            return prob.loading(r["steps"][k - 1])
                        return prob.loading(r["V"]) if r["converged"] else nanv

                    p0, p1, p2 = prob.loading(v0), after(1), after(2)
                else:
                    row.update({"conv_prev": False, "it_prev": -1, "mis0_prev": np.nan, "dv_prev": np.nan, "dv_start": np.nan, "mis1_prev": np.inf, "mis2_prev": np.inf})
                d = pd.DataFrame({"state": int(s), "line": idx, "ac": s_ac / rate, "prev0": p0, "prev1": p1, "prev2": p2,
                                  "ac_prev": ac_prev[idx], "dc_own": dc_c, "dc_prev": dc_prev[idx], "genco_prev": g_prev[idx]})[rate > 0]
                lines.append(d.astype({c: "float32" for c in d.columns if c not in ("state", "line")}))
                rows.append(row)
        if wid == 0:
            log(f"  worker 0: {len(rows)} scenarios done, {n_solves} intact-grid solves")
    return pd.concat(lines, ignore_index=True), pd.DataFrame(rows), n_solves, n_attempts


def run(a):
    import numpy as np
    import pandas as pd
    here, net, raw = paths(a.grid)
    status = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "load_scenario_idx", "idx", "br_status"])
    always_out = (status.groupby("idx").br_status.max() == 0).sort_index().to_numpy()
    load_ids = np.sort(status.load_scenario_idx.unique())
    if a.limit:
        load_ids = load_ids[: a.limit]
    log(f"=== run {net}: {status.scenario.nunique():,} scenarios in {len(load_ids):,} load scenarios, {int(always_out.sum())} branches always out, {a.procs} processes")
    del status
    t0 = time.perf_counter()
    jobs = [(a.grid, chunk, always_out, i) for i, chunk in enumerate(np.array_split(load_ids, a.procs))]
    with get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(worker, jobs)
    lines = pd.concat([r[0] for r in res], ignore_index=True)
    st = pd.concat([r[1] for r in res], ignore_index=True)
    lines.to_parquet(here / f"lines_prev_tiny_seed0{a.tag}.parquet", index=False)
    st.to_parquet(here / f"states_prev_tiny_seed0{a.tag}.parquet", index=False)
    solves, attempts = sum(r[2] for r in res), sum(r[3] for r in res)
    keys = st.groupby("load_idx").base_key.nunique()
    failed = st[~st.intact_ok]
    log(f"{net}: {len(st):,} scenarios, {len(lines):,} line readings, wall {time.perf_counter() - t0:.0f} s")
    log(f"  intact-grid solves: {solves:,} base states ({attempts:,} Newton runs incl. retries) for {len(st):,} contingencies = {len(st) / solves:.2f} contingencies per base state; "
        f"load scenarios with more than one distinct base state: {int((keys > 1).sum())} of {len(keys)}; "
        f"failed base states: {failed.base_key.nunique()} ({len(failed)} scenarios); intact Newton iterations mean {st[st.intact_ok].intact_it.mean():.2f} max {st.intact_it.max()}")
    log(f"  checks: admittance formula vs stored (in-service branches) max {st.ydiff.max():.1e}; own DC vs stored DC loading max {st.dc_check_pp.max():.1e} points; "
        f"prev start converged {int(st.conv_prev.sum()):,}/{len(st):,}, max |V - V_ac stored| {st.dv_prev.max():.1e} ({int((st.dv_prev > 1e-5).sum())} > 1e-5)")


def report(a):
    import numpy as np
    import pandas as pd
    out = []
    for grid in ("500", "2000", "10000"):
        here, net, raw = paths(grid)
        f = here / f"lines_prev_tiny_seed0{a.tag}.parquet"
        if not f.exists():
            continue
        prev = pd.read_parquet(f)
        stp = pd.read_parquet(here / f"states_prev_tiny_seed0{a.tag}.parquet").set_index("state")
        old = pd.read_parquet(here / "lines_newton_tiny_seed0.parquet", columns=["state", "line", "ac", "genco", "genco1", "genco2", "dc", "dc1"])
        sto = pd.read_parquet(here / "states_newton_tiny_seed0.parquet").set_index("state")
        full = prev.merge(old.rename(columns={"ac": "ac_old"}), on=["state", "line"], how="inner")
        log(f"=== report {net}: {full.state.nunique():,} scenarios, {len(full):,} line readings (prev file {len(prev):,}, GENCO file {len(old):,}); "
            f"max |ac - ac of GENCO file| {float((full.ac - full.ac_old).abs().max()):.1e}")
        full["anch_dc"] = full.ac_prev + full.dc_own - full.dc_prev
        full["anch_genco"] = full.ac_prev + full.genco - full.genco_prev
        full["prev_intact"] = full.ac_prev  # pre-outage loading used as is (no knowledge of the outage)
        sur = ["genco", "genco1", "genco2", "prev0", "prev1", "prev2", "anch_genco", "anch_dc", "dc", "dc1"]
        real = (full.ac > LIM).to_numpy()
        crit = (full.ac > LIM).groupby(full.state).any()
        n_sc, n_real = len(crit), int(real.sum())
        solves = stp.base_key.nunique()
        log(f"  {int(crit.sum()):,} of {n_sc:,} scenarios have a real overload ({crit.mean():.1%}); {n_real:,} overloaded line readings; "
            f"base states {solves:,} -> {n_sc / solves:.2f} contingencies per intact-grid AC solve; failed base states {stp[~stp.intact_ok].base_key.nunique()}")
        log(f"  Newton to convergence from pre-outage start: converged {int(stp.conv_prev.sum()):,}/{len(stp):,}, iterations mean {stp[stp.conv_prev].it_prev.mean():.2f} "
            f"median {stp[stp.conv_prev].it_prev.median():.0f} max {stp[stp.conv_prev].it_prev.max()}; from GENCO: converged {int(sto.conv_genco.sum()):,}/{len(sto):,}, "
            f"mean {sto[sto.conv_genco].it_genco.mean():.2f} median {sto[sto.conv_genco].it_genco.median():.0f} max {sto[sto.conv_genco].it_genco.max()}")
        q = lambda v: f"{v.replace(np.inf, np.nan).median():.3g} / {v.replace(np.inf, np.nan).quantile(.99):.3g} / {v.replace(np.inf, np.nan).max():.3g}"  # noqa: E731
        log(f"  bus mismatch MW max-norm, median / p99 / max: prev start {q(stp.mis0_prev)}, prev+1 {q(stp.mis1_prev)}, prev+2 {q(stp.mis2_prev)}; "
            f"GENCO start {q(sto.mis0_genco)}, GENCO+1 {q(sto.mis1_genco)}, GENCO+2 {q(sto.mis2_genco)}")
        log(f"  {'predictor':<11} {'err median':>10} {'p99':>8} {'max':>8} {'NaN':>7} {'flagged':>8} {'secure flagged':>14} {'missed scen.':>13} {'missed lines':>15} {'share':>7} {'worst':>7}")
        for s in sur:
            e = (full[s] - full.ac).abs().to_numpy() * 100
            okm = np.isfinite(e)
            flag = (full[s].fillna(BIG) > LIM).groupby(full.state).any()
            accepted = ~full.state.map(flag).to_numpy()
            missed = real & accepted
            k = int(missed.sum())
            r = {"network": net, "predictor": s, "scenarios": n_sc, "err_median_pp": float(np.median(e[okm])), "err_p99_pp": float(np.quantile(e[okm], .99)),
                 "err_max_pp": float(e[okm].max()), "nan_readings": int((~okm).sum()), "flagged": float(flag.mean()),
                 "secure_flagged": float((flag & ~crit).sum() / max((~crit).sum(), 1)), "critical_scenarios": int(crit.sum()),
                 "critical_missed": int((~flag & crit).sum()), "overloads": n_real, "overloads_missed": k, "missed_share": k / max(n_real, 1),
                 "worst_missed_loading": float(full.ac.to_numpy()[missed].max()) if k else np.nan}
            out.append(r)
            worst = f"{r['worst_missed_loading']:.1%}" if k else ""
            log(f"  {s:<11} {r['err_median_pp']:>10.4f} {r['err_p99_pp']:>8.3f} {r['err_max_pp']:>8.2f} {r['nan_readings']:>7} {r['flagged']:>8.1%} {r['secure_flagged']:>14.1%} "
                f"{r['critical_missed']:>6}/{r['critical_scenarios']:<6} {k:>7}/{n_real:<7} {r['missed_share']:>7.2%} {worst:>7}")
    pd.DataFrame(out).to_csv(ROOT / "work" / f"prev_control_summary{a.tag}.csv", index=False)
    log(f"saved {ROOT / 'work' / f'prev_control_summary{a.tag}.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["run", "report"])
    p.add_argument("--grid", default="500")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    {"run": run, "report": report}[a.phase](a)

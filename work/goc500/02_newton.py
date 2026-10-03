"""GENCO + one / two Newton steps on the GOC-500 scenarios (controls: DC + 1 step, flat + 1 step).

The Newton code is analysis_warmstart.Problem (polar full Newton, sparse LU, no step control, no Q limits);
only its constructor is replaced so that the problem is built from the stored datakit tables of each scenario:
the scenario's own perturbed branch admittances (Yff/Yft/Ytf/Ytt, in-service branches only), bus shunts,
PQ/PV/REF sets, P injections, PV/REF voltage setpoints and the reference angle. Start points:
    genco  GENCO's Vm at PQ buses and Va; setpoints at PV/REF
    dc     flat magnitudes (setpoints at PV/REF, 1.0 at PQ), stored DC angles
    flat   same magnitudes, all angles at the reference
Every start is also iterated to convergence (validation against the stored AC solution, and iteration counts).

    work/texas/tmo 7200 .venv/bin/python -u work/goc500/02_newton.py [--seed 0] [--procs 5]

Writes lines_newton_tiny_seed<S>.parquet (state, n_out, line, ac, genco, genco1, genco2, dc, dc1, flat1; NaN where a
Newton step broke down) and states_newton_tiny_seed<S>.parquet (per scenario: mismatch norms in MW/MVAr after
0/1/2 steps, iterations and ms of the full solves, ms of a single step, distance of the converged solution
from the stored AC solution).
"""
import argparse
import importlib.util
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
NET = "case500_goc"
BASE = 100.0


def _eval_module():
    spec = importlib.util.spec_from_file_location("goc_eval", HERE / "01_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_problem(bus, branch):
    import numpy as np
    from scipy.sparse import csr_matrix, diags
    from analysis_warmstart import Problem

    class TableProblem(Problem):
        def __init__(self, bus, branch):
            n = len(bus)
            assert (bus.bus.to_numpy() == np.arange(n)).all()
            self.base = BASE
            live = (branch.br_status == 1).to_numpy()
            br = branch[live]
            m = len(br)
            f, t = br.from_bus.to_numpy(int), br.to_bus.to_numpy(int)
            y = {k: br[f"{k}_r"].to_numpy() + 1j * br[f"{k}_i"].to_numpy() for k in ("Yff", "Yft", "Ytf", "Ytt")}
            rows = np.r_[np.arange(m), np.arange(m)]
            self.Yf = csr_matrix((np.r_[y["Yff"], y["Yft"]], (rows, np.r_[f, t])), shape=(m, n))
            self.Yt = csr_matrix((np.r_[y["Ytf"], y["Ytt"]], (rows, np.r_[f, t])), shape=(m, n))
            cf = csr_matrix((np.ones(m), (np.arange(m), f)), shape=(m, n))
            ct = csr_matrix((np.ones(m), (np.arange(m), t)), shape=(m, n))
            ysh = bus.GS.to_numpy() + 1j * bus.BS.to_numpy()  # stored in p.u. (checked: the stored AC solution has zero mismatch)
            self.Ybus = (cf.T @ self.Yf + ct.T @ self.Yt + diags(ysh)).tocsr()
            # P at PV/PQ and Q at PQ are the only injections Newton uses (specified quantities).
            self.S = ((bus.Pg - bus.Pd).to_numpy() + 1j * (bus.Qg - bus.Qd).to_numpy()) / BASE
            self.pq = np.flatnonzero(bus.PQ.to_numpy() == 1)
            self.pv = np.flatnonzero(bus.PV.to_numpy() == 1)
            self.ref = np.flatnonzero(bus.REF.to_numpy() == 1)
            self.pvpq = np.r_[self.pv, self.pq]
            self.vm_set = bus.Vm.to_numpy(float).copy()  # only PV/REF entries are used (setpoints)
            self.va_ref = np.radians(float(bus.Va.to_numpy()[self.ref[0]]))
            self.f, self.t = f, t
            rate = br.rate_a.to_numpy() / BASE
            self.live = np.ones(m, bool)
            self.rated = rate > 0
            self.rate = np.where(rate > 0, rate, np.nan)

    return TableProblem(bus, branch)


def worker(job):
    seed, raw, scenarios, always_out, wid = job
    import engine  # Julia before torch
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    ev = _eval_module()
    engine.WORK = HERE
    genco = engine.Genco(NET, ev.variant_of(seed))
    flt = [("scenario", ">=", int(scenarios[0])), ("scenario", "<=", int(scenarios[-1]))]
    tabs = {t: {s: d for s, d in pd.read_parquet(raw / f"{t}_data.parquet", filters=flt).groupby("scenario")} for t in ("bus", "gen", "branch")}
    lines, rows = [], []
    for i, s in enumerate(scenarios):
        bus = tabs["bus"][s].sort_values("bus").reset_index(drop=True)
        gen = tabs["gen"][s].sort_values("idx").reset_index(drop=True)
        branch = tabs["branch"][s].sort_values("idx").reset_index(drop=True)
        q = gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
        bus = bus.merge(q, on="bus", how="left").fillna(0)
        bus_pred, br_pred = genco.solve(bus, gen, branch)
        prob = make_problem(bus, branch)
        live = (branch.br_status == 1).to_numpy()
        n = int(live.sum())
        v_ac = bus.Vm.to_numpy(float) * np.exp(1j * np.radians(bus.Va.to_numpy(float)))
        vm_g, va_g = bus_pred.Vm.to_numpy(float), np.radians(bus_pred.Va.to_numpy(float))
        starts = {"genco": prob.start(vm_pq=vm_g, va=va_g), "dc": prob.start(va=np.radians(bus.Va_dc.to_numpy(float))), "flat": prob.start()}
        row = {"state": s, "n_out": int((~live).sum()) - always_out, "load_idx": int(branch.load_scenario_idx.iloc[0]),
               "mis_ac_stored": float(np.abs(prob.mismatch(v_ac)).max() * BASE)}
        runs = {}
        for name, v0 in starts.items():
            t0 = time.perf_counter()
            runs[name] = r = prob.newton(v0)
            row[f"ms_full_{name}"] = 1e3 * (time.perf_counter() - t0)
            row[f"it_{name}"], row[f"conv_{name}"], row[f"mis0_{name}"] = r["iterations"], r["converged"], r["mis0"]
            row[f"dv_{name}"] = float(np.abs(r["V"] - v_ac).max()) if r["converged"] else np.nan
            for k in (1, 2):
                with np.errstate(all="ignore"):
                    row[f"mis{k}_{name}"] = float(np.abs(prob.mismatch(r["steps"][k - 1])).max() * BASE) if len(r["steps"]) >= k else (
                        r["mis_end"] if r["converged"] else np.inf)
        t0 = time.perf_counter()
        prob.newton(starts["genco"], max_it=1)
        row["ms_step1_genco"] = 1e3 * (time.perf_counter() - t0)

        def after(name, k):
            r = runs[name]
            if len(r["steps"]) >= k:
                return prob.loading(r["steps"][k - 1])
            return prob.loading(r["V"]) if r["converged"] else np.full(n, np.nan)

        rate = branch.rate_a.to_numpy()[live]
        s_ac = np.maximum(np.hypot(branch.pf, branch.qf), np.hypot(branch.pt, branch.qt)).to_numpy()[live]
        d = pd.DataFrame({
            "state": s, "n_out": row["n_out"], "line": branch.idx.to_numpy()[live],
            "ac": s_ac / rate,
            "genco": np.maximum(br_pred.loading[:n].to_numpy(), br_pred.loading[n:].to_numpy()),
            "genco1": after("genco", 1), "genco2": after("genco", 2),
            "dc": np.abs(branch.pf_dc.to_numpy()[live]) / rate,
            "dc1": after("dc", 1), "flat1": after("flat", 1),
            "load_idx": row["load_idx"],
        })[rate > 0]
        # validation: loading of the stored AC voltages through this script's admittances vs stored AC flows,
        # and of the converged Newton solution vs stored AC flows
        row["dload_formula_pp"] = float(100 * np.nanmax(np.abs(prob.loading(v_ac) - s_ac / np.where(rate > 0, rate, np.nan))))
        best = next((runs[k] for k in ("flat", "genco", "dc") if runs[k]["converged"]), None)
        row["dload_newton_pp"] = float(100 * np.nanmax(np.abs(prob.loading(best["V"]) - s_ac / np.where(rate > 0, rate, np.nan)))) if best else np.nan
        lines.append(d.astype({c: "float32" for c in ("ac", "genco", "genco1", "genco2", "dc", "dc1", "flat1")}))
        rows.append(row)
        if wid == 0 and i % 250 == 0:
            print(time.strftime("%H:%M:%S"), f"worker 0: {i}/{len(scenarios)}  it flat/dc/genco {row['it_flat']}/{row['it_dc']}/{row['it_genco']}  "
                  f"dv {row['dv_flat']:.1e}  mis genco 0/1/2 {row['mis0_genco']:.1f}/{row['mis1_genco']:.2f}/{row['mis2_genco']:.4f} MW", flush=True)
    return pd.concat(lines, ignore_index=True), pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    raw = HERE / a.data / NET / "raw"
    status = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "idx", "br_status"])
    scenarios = np.sort(status.scenario.unique())
    always_out = int((status.groupby("idx").br_status.max() == 0).sum())
    if a.limit:
        scenarios = scenarios[: a.limit]
    jobs = [(a.seed, raw, chunk, always_out, i) for i, chunk in enumerate(np.array_split(scenarios, a.procs))]
    start = time.perf_counter()
    with get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(worker, jobs)
    lines = pd.concat([r[0] for r in res], ignore_index=True)
    st = pd.concat([r[1] for r in res], ignore_index=True)
    lines.to_parquet(HERE / f"lines_newton_tiny_seed{a.seed}{a.tag}.parquet", index=False)
    st.to_parquet(HERE / f"states_newton_tiny_seed{a.seed}{a.tag}.parquet", index=False)
    print(f"{len(st)} scenarios, {len(lines):,} line readings, wall {time.perf_counter() - start:.0f}s")
    print(f"validation: stored AC solution in this script's equations: mismatch max {st.mis_ac_stored.max():.2e} MW (median {st.mis_ac_stored.median():.2e}); "
          f"loading formula vs stored flows max {st.dload_formula_pp.max():.2e} points")
    for k in ("flat", "dc", "genco"):
        c = st[st[f"conv_{k}"]]
        print(f"  Newton from {k:<5}: converged {len(c)}/{len(st)}, iterations mean {c[f'it_{k}'].mean():.2f} (median {c[f'it_{k}'].median():.0f}, max {c[f'it_{k}'].max()}), "
              f"{c[f'ms_full_{k}'].median():.1f} ms median; max |V - V_ac stored| {c[f'dv_{k}'].max():.1e} p.u. ({(c[f'dv_{k}'] > 1e-5).sum()} scenarios > 1e-5); "
              f"start mismatch median {st[f'mis0_{k}'].median():.1f} MW")
    print(f"  converged Newton loading vs stored AC loading: max {st.dload_newton_pp.max():.2e} points")
    print(f"  one Newton step from GENCO: {st.ms_step1_genco.median():.2f} ms median")
    nan = lines[["genco1", "genco2", "dc1", "flat1"]].isna().groupby(lines.state).any().sum()
    print("  scenarios with a broken step (NaN loading):", nan.to_dict())


if __name__ == "__main__":
    main()

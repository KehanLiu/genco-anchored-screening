"""Hybrid start: pre-outage AC solution + GENCO's predicted voltage change, then 0 / 1 / 2 Newton steps.

    start = V_prev_AC + (V_GENCO(contingency) - V_GENCO(pre-outage state))      in Vm (PQ buses) and Va (all but REF)
PV/REF magnitudes stay at setpoints, the reference angle is fixed (analysis_warmstart.Problem.start). The pre-outage
state is rebuilt as in analysis/analysis_prev_control.py (helpers imported from there). GENCO Tiny seed0 runs once per
contingency and once per base state; the intact-grid GENCO input is the contingency's own stored bus/gen tables with
only the branch table changed (removed branches back in service, admittances recomputed) - no intact AC solution is
fed in.

Also measured: does GENCO's predicted change point the right way? Sign agreement and correlation between predicted
and true change of Va (buses whose true angle moves by more than 0.1 degree) and of line loading (lines whose true
loading moves by more than 1 point of rating).

    work/texas/tmo 3000 .venv/bin/python -u analysis/analysis_hybrid_start.py run --grid 500 [--procs 4] [--limit N] [--variant tiny] [--seed 0] [--tag _x] [--policy P.json]
    .venv/bin/python analysis/analysis_hybrid_start.py report [--variant tiny] [--seed 0]
With --policy (work/fresh/policy_frozen.json) a second, shrunk start "shyb" is evaluated next to the hybrid start:
    start = V_prev_AC + lambda_vm * dVm_GENCO, V_prev_AC angles + lambda_va * dVa_GENCO,   lambda from policy["shrinkage"]["lambda"]["goc<N>_<variant>_seed<S>"]
giving the extra columns shyb0/1/2 (lines) and conv_shyb, it_shyb, mis0/1/2_shyb, dv_shyb, dv_start_shyb plus the constant columns
policy_path, lam_va, lam_vm (states); the hyb columns are unchanged. GOC_DATA=<dir> in the environment reads the datakit tables
from <dir>/case<N>_goc/raw instead of work/goc<N>/data (analysis_prev_control.paths); combine with --tag so the original files stay.
Writes work/goc<N>/lines_hybrid_<variant>_seed<S>.parquet, states_hybrid_<variant>_seed<S>.parquet and
work/hybrid_control_summary.csv, log work/hybrid_control.log (for a non-default variant/seed the csv and log carry the
suffix _<variant>_seed<S>). The report compares against lines_newton_<variant>_seed<S>.parquet from 02_newton.py when it
exists, else against tiny_seed0 with a warning; the pre-outage files (*_prev_tiny_seed0) are model-independent.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here
import analysis_prev_control as pc  # noqa: E402

log = pc.log
BASE, LIM, BIG = pc.BASE, pc.LIM, pc.BIG


def model_tag(variant, seed):
    return f"{variant}_seed{seed}"


def suffix(variant, seed):
    """File-name suffix for the log and summary csv: empty for the original tiny seed0 study."""
    return "" if model_tag(variant, seed) == "tiny_seed0" else f"_{model_tag(variant, seed)}"


def set_log(variant, seed):
    pc.LOG = ROOT / "work" / f"hybrid_control{suffix(variant, seed)}.log"


def fetch_checkpoint(grid, variant, seed):
    """Checkpoint folder work/goc<grid>/pf_case<grid>_goc_<variant>[_seed<S>], fetched from HF if missing (as 02_newton.py)."""
    here, net, raw = pc.paths(grid)
    name = variant + (f"_seed{seed}" if seed else "")
    if not (here / "01_fetch.py").exists():  # work/goc500 has no fetch helper: use the folder already on disk
        folder = here / f"pf_{net}_{name}"
        if not folder.exists():
            raise FileNotFoundError(f"{folder} missing and {here / '01_fetch.py'} does not exist to fetch it")
        return name, folder
    spec = importlib.util.spec_from_file_location("goc_fetch", here / "01_fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    folder = mod.fetch(variant, seed)
    assert mod.variant_of(variant, seed) == name
    return name, folder


set_log("tiny", 0)


def shrinkage(policy, grid, variant, seed):
    """{"va": lambda_va, "vm": lambda_vm} for this grid / checkpoint from the frozen policy file, or None without a policy."""
    if not policy:
        return None
    pol = json.load(open(policy))
    key = f"goc{grid}_{model_tag(variant, seed)}"
    lam = pol["shrinkage"]["lambda"][key]
    return {"va": float(lam["va"]), "vm": float(lam["vm"])}


def worker(job):
    grid, load_ids, always_out, wid, variant, seed, lam = job
    import engine  # Julia before torch
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    set_log(variant, seed)
    here, net, raw = pc.paths(grid)
    engine.WORK = here
    genco = engine.Genco(net, fetch_checkpoint(grid, variant, seed)[0])
    nw = pc.newton_module()
    lines, rows = [], []
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
            s0, bus0, gen0, branch0 = members[0]
            bi = branch0.copy()
            bi["br_status"] = np.where(always_out, 0, 1).astype(branch0.br_status.dtype)
            for k, v in pc.admittances(bi).items():
                bi[f"{k}_r"], bi[f"{k}_i"] = np.where(always_out, 0.0, v.real), np.where(always_out, 0.0, v.imag)
            prob_i = nw.make_problem(bus0, bi)
            V_prev = None
            for (s, bus, gen, branch) in members:
                r = prob_i.newton(bus.Vm.to_numpy(float) * np.exp(1j * np.radians(bus.Va.to_numpy(float))))
                if r["converged"]:
                    V_prev = r["V"]
                    break
            if V_prev is None:
                r = prob_i.newton(prob_i.start())
                V_prev = r["V"] if r["converged"] else None
            ok = V_prev is not None
            live_i = (bi.br_status == 1).to_numpy()
            ni = int(live_i.sum())
            ac_prev = np.full(len(bi), np.nan)
            g_prev = np.full(len(bi), np.nan)
            # GENCO on the intact grid: bus0 / gen0 are the stored contingency tables (no intact AC solution in them)
            bp_i, br_i = genco.solve(bus0, gen0, bi)
            vm_gi, va_gi = bp_i.Vm.to_numpy(float), np.radians(bp_i.Va.to_numpy(float))
            g_prev[live_i] = np.maximum(br_i.loading[:ni].to_numpy(), br_i.loading[ni:].to_numpy())
            if ok:
                ac_prev[live_i] = prob_i.loading(V_prev)
                vm_p, va_p = np.abs(V_prev), np.angle(V_prev)
            for (s, bus, gen, branch) in members:
                prob = nw.make_problem(bus, branch)
                live = (branch.br_status == 1).to_numpy()
                n = int(live.sum())
                br = branch[live]
                rate = br.rate_a.to_numpy()
                idx = br.idx.to_numpy()
                vm_ac, va_ac = bus.Vm.to_numpy(float), np.radians(bus.Va.to_numpy(float))
                v_ac = vm_ac * np.exp(1j * va_ac)
                with np.errstate(all="ignore"):
                    s_ac = np.maximum(np.hypot(br.pf, br.qf), np.hypot(br.pt, br.qt)).to_numpy()
                t_g = time.perf_counter()
                bp_c, br_c = genco.solve(bus, gen, branch)
                ms_genco = 1e3 * (time.perf_counter() - t_g)
                vm_gc, va_gc = bp_c.Vm.to_numpy(float), np.radians(bp_c.Va.to_numpy(float))
                g_c = np.maximum(br_c.loading[:n].to_numpy(), br_c.loading[n:].to_numpy())
                row = {"state": int(s), "load_idx": load_idx, "base_key": key[:12], "intact_ok": ok, "ms_genco": ms_genco}
                nanv = np.full(n, np.nan)
                h0 = h1 = h2 = sh0 = sh1 = sh2 = nanv
                if ok:
                    v0 = prob.start(vm_pq=vm_p + (vm_gc - vm_gi), va=va_p + (va_gc - va_gi))
                    r = prob.newton(v0)
                    row.update({"conv_hyb": r["converged"], "it_hyb": r["iterations"], "mis0_hyb": r["mis0"],
                                "dv_hyb": float(np.abs(r["V"] - v_ac).max()) if r["converged"] else np.nan,
                                "dv_start_hyb": float(np.abs(v0 - v_ac).max()),
                                "dv_start_prev": float(np.abs(prob.start(vm_pq=vm_p, va=va_p) - v_ac).max()),
                                "dv_start_genco": float(np.abs(prob.start(vm_pq=vm_gc, va=va_gc) - v_ac).max())})
                    for k in (1, 2):
                        with np.errstate(all="ignore"):
                            row[f"mis{k}_hyb"] = float(np.abs(prob.mismatch(r["steps"][k - 1])).max() * BASE) if len(r["steps"]) >= k else (
                                r["mis_end"] if r["converged"] else np.inf)

                    def after(r_, k):
                        if len(r_["steps"]) >= k:
                            return prob.loading(r_["steps"][k - 1])
                        return prob.loading(r_["V"]) if r_["converged"] else nanv

                    h0, h1, h2 = prob.loading(v0), after(r, 1), after(r, 2)
                    if lam is not None:  # shrunk anchored start (policy): pre-outage voltages + lambda * GENCO's predicted change
                        v0s = prob.start(vm_pq=vm_p + lam["vm"] * (vm_gc - vm_gi), va=va_p + lam["va"] * (va_gc - va_gi))
                        rs = prob.newton(v0s)
                        row.update({"conv_shyb": rs["converged"], "it_shyb": rs["iterations"], "mis0_shyb": rs["mis0"],
                                    "dv_shyb": float(np.abs(rs["V"] - v_ac).max()) if rs["converged"] else np.nan,
                                    "dv_start_shyb": float(np.abs(v0s - v_ac).max())})
                        for k in (1, 2):
                            with np.errstate(all="ignore"):
                                row[f"mis{k}_shyb"] = float(np.abs(prob.mismatch(rs["steps"][k - 1])).max() * BASE) if len(rs["steps"]) >= k else (
                                    rs["mis_end"] if rs["converged"] else np.inf)
                        sh0, sh1, sh2 = prob.loading(v0s), after(rs, 1), after(rs, 2)
                    # does GENCO's predicted change point the right way? (pooled sums for sign agreement / correlation)
                    nonref = bus.REF.to_numpy() != 1
                    for name, tx, px, thr, mask in (("va", np.degrees(va_ac - va_p), np.degrees(va_gc - va_gi), 0.1, nonref),
                                                    ("vm", vm_ac - vm_p, vm_gc - vm_gi, 1e-3, bus.PQ.to_numpy() == 1)):
                        m = mask & (np.abs(tx) > thr)
                        t_, p_ = tx[m], px[m]
                        row.update({f"{name}_n": int(m.sum()), f"{name}_sign": int((np.sign(t_) == np.sign(p_)).sum()), f"{name}_st": t_.sum(), f"{name}_sp": p_.sum(),
                                    f"{name}_stt": (t_ * t_).sum(), f"{name}_spp": (p_ * p_).sum(), f"{name}_stp": (t_ * p_).sum()})
                else:
                    row.update({"conv_hyb": False, "it_hyb": -1, "mis0_hyb": np.nan, "mis1_hyb": np.inf, "mis2_hyb": np.inf})
                    if lam is not None:
                        row.update({"conv_shyb": False, "it_shyb": -1, "mis0_shyb": np.nan, "mis1_shyb": np.inf, "mis2_shyb": np.inf})
                cols = {"state": int(s), "line": idx, "ac": s_ac / rate, "hyb0": h0, "hyb1": h1, "hyb2": h2,
                        "ac_prev": ac_prev[idx], "genco_prev": g_prev[idx], "genco_now": g_c}
                if lam is not None:
                    cols.update({"shyb0": sh0, "shyb1": sh1, "shyb2": sh2})
                d = pd.DataFrame(cols)[rate > 0]
                lines.append(d.astype({c: "float32" for c in d.columns if c not in ("state", "line")}))
                rows.append(row)
        if wid == 0:
            log(f"  worker 0: {len(rows)} scenarios done")
    return pd.concat(lines, ignore_index=True), pd.DataFrame(rows)


def run(a):
    import numpy as np
    import pandas as pd
    set_log(a.variant, a.seed)
    here, net, raw = pc.paths(a.grid)
    mv = model_tag(a.variant, a.seed)
    log(f"GENCO checkpoint {mv}: {fetch_checkpoint(a.grid, a.variant, a.seed)[1]}")
    lam = shrinkage(a.policy, a.grid, a.variant, a.seed)
    if lam is not None:
        log(f"policy {a.policy}: shrunk start shyb with lambda_va {lam['va']} lambda_vm {lam['vm']} (goc{a.grid}_{mv})")
    status = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "load_scenario_idx", "idx", "br_status"])
    always_out = (status.groupby("idx").br_status.max() == 0).sort_index().to_numpy()
    load_ids = np.sort(status.load_scenario_idx.unique())
    if a.limit:
        load_ids = load_ids[: a.limit]
    log(f"=== run {net}: {len(load_ids):,} load scenarios, {a.procs} processes; data {raw}, outputs *_hybrid_{mv}{a.tag}.parquet")
    del status
    t0 = time.perf_counter()
    with get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(worker, [(a.grid, chunk, always_out, i, a.variant, a.seed, lam) for i, chunk in enumerate(np.array_split(load_ids, a.procs))])
    lines = pd.concat([r[0] for r in res], ignore_index=True)
    st = pd.concat([r[1] for r in res], ignore_index=True)
    if lam is not None:  # record the policy used (constant columns; absent without --policy so old outputs stay identical)
        st["policy_path"], st["lam_va"], st["lam_vm"] = str(a.policy), lam["va"], lam["vm"]
    lines.to_parquet(here / f"lines_hybrid_{mv}{a.tag}.parquet", index=False)
    st.to_parquet(here / f"states_hybrid_{mv}{a.tag}.parquet", index=False)
    log(f"{net} ({mv}): {len(st):,} scenarios, {len(lines):,} line readings, wall {time.perf_counter() - t0:.0f} s; hybrid start converged {int(st.conv_hyb.sum()):,}/{len(st):,}, "
        f"max |V - V_ac stored| {st.dv_hyb.max():.1e}; base states without a pre-outage solution: {st[~st.intact_ok].base_key.nunique()}; "
        f"GENCO ms per contingency state (workers in parallel) mean {st.ms_genco.mean():.0f} median {st.ms_genco.median():.0f}")
    if lam is not None:
        log(f"  shrunk start shyb converged {int(st.conv_shyb.sum()):,}/{len(st):,}, max |V - V_ac stored| {st.dv_shyb.max():.1e}; "
            f"start mismatch MW median hyb {st.mis0_hyb.median():.3g} shyb {st.mis0_shyb.median():.3g}")


def report(a):
    import numpy as np
    import pandas as pd
    set_log(a.variant, a.seed)
    mv = model_tag(a.variant, a.seed)
    out = []
    for grid in ("500", "2000", "10000"):
        here, net, raw = pc.paths(grid)
        f = here / f"lines_hybrid_{mv}{a.tag}.parquet"
        if not f.exists():
            continue
        hyb = pd.read_parquet(f)
        sth = pd.read_parquet(here / f"states_hybrid_{mv}{a.tag}.parquet").set_index("state")
        newton_tag = mv
        if not (here / f"lines_newton_{mv}.parquet").exists():
            newton_tag = "tiny_seed0"
            log(f"WARNING {net}: lines_newton_{mv}.parquet not found, GENCO + step rows (genco, genco1, genco2) and the 'GENCO contingency loading re-run vs "
                f"existing file' check use tiny_seed0 instead")
        old = pd.read_parquet(here / f"lines_newton_{newton_tag}.parquet", columns=["state", "line", "genco", "genco1", "genco2"])
        prev = pd.read_parquet(here / "lines_prev_tiny_seed0.parquet", columns=["state", "line", "prev1", "prev2", "genco_prev"]).rename(columns={"genco_prev": "genco_prev_old"})
        sto = pd.read_parquet(here / f"states_newton_{newton_tag}.parquet").set_index("state")
        stp = pd.read_parquet(here / "states_prev_tiny_seed0.parquet").set_index("state")
        full = hyb.merge(old, on=["state", "line"]).merge(prev, on=["state", "line"])
        log(f"=== report {net} (hybrid {mv}, GENCO + step {newton_tag}): {full.state.nunique():,} scenarios, {len(full):,} line readings (hybrid file {len(hyb):,}); "
            f"GENCO contingency loading re-run vs existing file max diff {float((full.genco_now - full.genco).abs().max()):.1e}; "
            f"GENCO intact loading, tables without vs with the intact AC solution written in, max diff {float((full.genco_prev - full.genco_prev_old).abs().max()):.1e}")
        real = (full.ac > LIM).to_numpy()
        crit = (full.ac > LIM).groupby(full.state).any()
        n_real = int(real.sum())
        q = lambda v: f"{v.replace(np.inf, np.nan).median():.3g} / {v.replace(np.inf, np.nan).quantile(.99):.3g}"  # noqa: E731
        log(f"  start mismatch MW max-norm, median / p99: prev {q(stp.mis0_prev)}, GENCO {q(sto.mis0_genco)}, hybrid {q(sth.mis0_hyb)};  after one step: prev {q(stp.mis1_prev)}, "
            f"GENCO {q(sto.mis1_genco)}, hybrid {q(sth.mis1_hyb)}")
        log(f"  start distance max|V - V_ac| p.u., median / p99: prev {q(sth.dv_start_prev)}, GENCO {q(sth.dv_start_genco)}, hybrid {q(sth.dv_start_hyb)}")
        c = sth[sth.conv_hyb]
        log(f"  Newton to convergence: hybrid {len(c):,}/{len(sth):,} converged, iterations mean {c.it_hyb.mean():.2f} median {c.it_hyb.median():.0f} max {c.it_hyb.max()}; "
            f"prev mean {stp[stp.conv_prev].it_prev.mean():.2f}; GENCO mean {sto[sto.conv_genco].it_genco.mean():.2f}")
        if "ms_genco" in sth:
            log(f"  GENCO ms per contingency state (workers in parallel, indicative): mean {sth.ms_genco.mean():.0f} median {sth.ms_genco.median():.0f}")
        for name, unit in (("va", "bus angles moving > 0.1 deg"), ("vm", "PQ magnitudes moving > 0.001 p.u.")):
            n, st_, sp, stt, spp, stp_ = (sth[f"{name}_{k}"].sum() for k in ("n", "st", "sp", "stt", "spp", "stp"))
            corr = (n * stp_ - st_ * sp) / np.sqrt((n * stt - st_**2) * (n * spp - sp**2))
            log(f"  GENCO predicted change vs true change, {unit}: {int(n):,} readings, sign agreement {sth[f'{name}_sign'].sum() / n:.1%}, correlation {corr:.3f}, "
                f"slope (pred on true) {(n * stp_ - st_ * sp) / (n * stt - st_**2):.3f}")
        dt, dp = (full.ac - full.ac_prev).to_numpy(), (full.genco_now - full.genco_prev).to_numpy()
        m = np.isfinite(dt) & np.isfinite(dp) & (np.abs(dt) > 0.01)
        log(f"  GENCO predicted loading change vs true change, lines moving > 1 point: {int(m.sum()):,} readings ({m.mean():.2%} of all), sign agreement "
            f"{(np.sign(dt[m]) == np.sign(dp[m])).mean():.1%}, correlation {np.corrcoef(dt[m], dp[m])[0, 1]:.3f}; raw GENCO error on these lines median "
            f"{100 * np.median(np.abs(full.genco.to_numpy()[m] - full.ac.to_numpy()[m])):.2f} points, true |change| median {100 * np.median(np.abs(dt[m])):.2f} points")
        log(f"  {'predictor':<8} {'err median':>10} {'p99':>8} {'max':>8} {'NaN':>6} {'flagged':>8} {'secure flagged':>14} {'missed scen.':>13} {'missed lines':>15} {'share':>7} {'worst':>7}")
        for s in ["genco", "genco1", "genco2", "prev1", "prev2", "hyb0", "hyb1", "hyb2"]:
            e = (full[s] - full.ac).abs().to_numpy() * 100
            okm = np.isfinite(e)
            flag = (full[s].fillna(BIG) > LIM).groupby(full.state).any()
            missed = real & ~full.state.map(flag).to_numpy()
            k = int(missed.sum())
            r = {"network": net, "model": mv, "predictor": s, "scenarios": len(crit), "err_median_pp": float(np.median(e[okm])), "err_p99_pp": float(np.quantile(e[okm], .99)),
                 "err_max_pp": float(e[okm].max()), "nan_readings": int((~okm).sum()), "flagged": float(flag.mean()),
                 "secure_flagged": float((flag & ~crit).sum() / max((~crit).sum(), 1)), "critical_scenarios": int(crit.sum()),
                 "critical_missed": int((~flag & crit).sum()), "overloads": n_real, "overloads_missed": k, "missed_share": k / max(n_real, 1),
                 "worst_missed_loading": float(full.ac.to_numpy()[missed].max()) if k else np.nan}
            out.append(r)
            worst = f"{r['worst_missed_loading']:.1%}" if k else ""
            log(f"  {s:<8} {r['err_median_pp']:>10.4f} {r['err_p99_pp']:>8.3f} {r['err_max_pp']:>8.2f} {r['nan_readings']:>6} {r['flagged']:>8.1%} {r['secure_flagged']:>14.1%} "
                f"{r['critical_missed']:>6}/{r['critical_scenarios']:<6} {k:>7}/{n_real:<7} {r['missed_share']:>7.2%} {worst:>7}")
    csv = ROOT / "work" / f"hybrid_control_summary{suffix(a.variant, a.seed)}{a.tag}.csv"
    pd.DataFrame(out).to_csv(csv, index=False)
    log(f"saved {csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["run", "report"])
    p.add_argument("--grid", default="500")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--variant", default="tiny", help="GENCO checkpoint size: tiny | small")
    p.add_argument("--seed", type=int, default=0, help="GENCO checkpoint seed")
    p.add_argument("--policy", default="", help="frozen policy json (work/fresh/policy_frozen.json): adds the shrunk start shyb with its lambda_va / lambda_vm")
    a = p.parse_args()
    {"run": run, "report": report}[a.phase](a)

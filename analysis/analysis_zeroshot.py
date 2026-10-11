"""Zero-shot transfer: the organisers' PRETRAINED GENCO Base (work/pretrained, trained on 100 subgrids cut from large
PGLib cases, gridfm/genco-pf-transfer) used WITHOUT fine-tuning on grids it has never seen. Can it serve as a universal
Newton start, or as a universal "delta" for the hybrid start?

Per scenario (datakit, random removal of up to 5 branches) the following Newton starts are built and iterated
(analysis_warmstart.Problem via work/goc2000/02_newton.make_problem; plain full Newton, no Q limits):
    flat    setpoints at PV/REF, 1.0 at PQ, flat angles
    dc      same magnitudes, stored DC angles
    prev    pre-outage AC solution (removed branches back in service, one intact-grid solve per base state)
    zs      zero-shot GENCO voltages
    hybzs   pre-outage AC voltages + [zero-shot GENCO(contingency) - zero-shot GENCO(pre-outage)]
    pg      the released per-grid GENCO checkpoint, where one exists (IEEE 118 Base, GOC 500 Tiny, GOC 2000 Tiny)
    hybpg   pre-outage AC voltages + per-grid GENCO delta
Line loadings after 0 / 1 / 2 Newton steps are stored per line; the plain screening rule (send a scenario to AC if any
predicted loading > 1.001) is evaluated in `report`.

Normalisation of the zero-shot model: the pretraining yaml asks for HeteroDataMVANormalizer, which graphkit fits PER
NETWORK (baseMVA = 95th percentile of the non-zero Pd, Qd, generator p_mw and bus Qg; vn_kv_max = largest nominal
voltage). No statistics file ships with the pretrained model, so the same recipe is applied here to the scenarios of the
unseen grid (--basemva overrides it for the sensitivity check).

    work/texas/tmo 5000 .venv/bin/python -u analysis/analysis_zeroshot.py run --grid 500 [--procs 3] [--limit N] [--tag T] [--basemva X]
    .venv/bin/python analysis/analysis_zeroshot.py report
Writes work/zeroshot/lines_<grid>.parquet, states_<grid>.parquet, summary.csv, report.log.
"""
import argparse
import hashlib
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here
OUT = ROOT / "work" / "zeroshot"
LOG = OUT / "report.log"
BASE, LIM, BIG = 100.0, 1.001, 9.99
PRETRAINED = ROOT / "work" / "pretrained" / "best_model_state_dict.pt"

# grid -> network, raw tables, per-grid released checkpoint (engine.WORK folder, variant), default cap on load scenarios
GRIDS = {
    "118": dict(net="case118_ieee", raw=OUT / "data" / "case118_ieee" / "raw", pg=(ROOT / "work", "base"), cap=0),
    "500": dict(net="case500_goc", raw=ROOT / "work" / "goc500" / "data" / "case500_goc" / "raw", pg=(ROOT / "work" / "goc500", "tiny"), cap=600),
    "1354": dict(net="case1354_pegase", raw=OUT / "data" / "case1354_pegase" / "raw", pg=None, cap=0),
    "2000": dict(net="case2000_goc", raw=ROOT / "work" / "goc2000" / "data" / "case2000_goc" / "raw", pg=(ROOT / "work" / "goc2000", "tiny"), cap=0),
}
STARTS = ["flat", "dc", "prev", "zs", "hybzs", "pg", "hybpg"]


def log(*a):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def admittances(br):
    """MATPOWER branch admittances from r, x, b, tap, shift (degrees)."""
    import numpy as np
    ys = 1.0 / (br.r.to_numpy() + 1j * br.x.to_numpy())
    tap = br.tap.to_numpy() * np.exp(1j * np.radians(br["shift"].to_numpy()))
    ytt = ys + 0.5j * br.b.to_numpy()
    return {"Yff": ytt / (tap * np.conj(tap)), "Yft": -ys / np.conj(tap), "Ytf": -ys / tap, "Ytt": ytt}


def make_zero_shot(net, stats):
    """The pretrained Base model behind engine.Genco's interface, normaliser set from `stats` (no stats file exists)."""
    import engine
    import torch
    import run_pretrained
    from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms, load_model, load_normalizer

    class ZeroShot(engine.Genco):
        def __init__(self):
            self.args = NestedNamespace(**run_pretrained.config(net, "base", "cpu"))
            self.model = load_model(self.args)
            state = torch.load(PRETRAINED, map_location="cpu")
            self.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")})
            self.model.eval()
            self.normalizer = load_normalizer(self.args)
            self.normalizer.fit_from_dict({k: torch.tensor(float(v)) for k, v in stats.items()})
            self.transforms = get_task_transforms(self.args)

    return ZeroShot()


def worker(job):
    grid, load_ids, always_out, stats, wid, tag = job
    import engine  # Julia before torch
    import numpy as np
    import pandas as pd
    import torch
    import importlib.util
    torch.set_num_threads(1)
    g = GRIDS[grid]
    net, raw = g["net"], g["raw"]
    spec = importlib.util.spec_from_file_location("goc_newton", ROOT / "work" / "goc2000" / "02_newton.py")
    nw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nw)
    models = {"zs": make_zero_shot(net, stats)}
    if g["pg"]:
        engine.WORK = g["pg"][0]
        models["pg"] = engine.Genco(net, g["pg"][1])
    part, rows, done = [], [], 0
    for sub in np.array_split(load_ids, max(len(load_ids) // 20, 1)):
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
            # ---- intact grid: removed branches back in service, admittances recomputed
            s0, bus0, gen0, branch0 = members[0]
            bi = branch0.copy()
            bi["br_status"] = np.where(always_out, 0, 1).astype(branch0.br_status.dtype)
            for k, v in admittances(bi).items():
                bi[f"{k}_r"], bi[f"{k}_i"] = np.where(always_out, 0.0, v.real), np.where(always_out, 0.0, v.imag)
            prob_i = nw.make_problem(bus0, bi)
            V_prev, it_i = None, -1
            for (s, bus, gen, branch) in members:
                r = prob_i.newton(bus.Vm.to_numpy(float) * np.exp(1j * np.radians(bus.Va.to_numpy(float))))
                if r["converged"]:
                    V_prev, it_i = r["V"], r["iterations"]
                    break
            if V_prev is None:
                r = prob_i.newton(prob_i.start())
                if r["converged"]:
                    V_prev, it_i = r["V"], r["iterations"]
            ok = V_prev is not None
            live_i = (bi.br_status == 1).to_numpy()
            ni = int(live_i.sum())
            ac_prev = np.full(len(bi), np.nan)
            if ok:
                ac_prev[live_i] = prob_i.loading(V_prev)
                vm_p, va_p = np.abs(V_prev), np.angle(V_prev)
            # GENCO on the intact grid: the stored contingency bus/gen tables with only the branch table changed
            gi = {}
            for m, model in models.items():
                bp, brp = model.solve(bus0, gen0, bi)
                load_i = np.full(len(bi), np.nan)
                load_i[live_i] = np.maximum(brp.loading[:ni].to_numpy(), brp.loading[ni:].to_numpy())
                gi[m] = (bp.Vm.to_numpy(float), np.radians(bp.Va.to_numpy(float)), load_i)
            for (s, bus, gen, branch) in members:
                prob = nw.make_problem(bus, branch)
                live = (branch.br_status == 1).to_numpy()
                n = int(live.sum())
                br = branch[live]
                rate = br.rate_a.to_numpy()
                idx = br.idx.to_numpy()
                yc = admittances(br)
                vm_ac, va_ac = bus.Vm.to_numpy(float), np.radians(bus.Va.to_numpy(float))
                v_ac = vm_ac * np.exp(1j * va_ac)
                with np.errstate(all="ignore"):
                    s_ac = np.maximum(np.hypot(br.pf, br.qf), np.hypot(br.pt, br.qt)).to_numpy()
                row = {"state": int(s), "load_idx": load_idx, "base_key": key[:12], "n_out": int((~live).sum()) - int(always_out.sum()),
                       "intact_ok": ok, "intact_it": it_i, "mis_ac_stored": float(np.abs(prob.mismatch(v_ac)).max() * BASE),
                       "ydiff": max(float(np.abs(yc[k] - (br[f"{k}_r"].to_numpy() + 1j * br[f"{k}_i"].to_numpy())).max()) for k in yc)}
                cols = {"state": int(s), "line": idx, "ac": s_ac / rate, "dc": np.abs(br.pf_dc.to_numpy()) / rate, "ac_prev": ac_prev[idx]}
                starts = {"flat": prob.start(), "dc": prob.start(va=np.radians(bus.Va_dc.to_numpy(float)))}
                if ok:
                    starts["prev"] = prob.start(vm_pq=vm_p, va=va_p)
                gc = {}
                for m, model in models.items():
                    t0 = time.perf_counter()
                    bp, brp = model.solve(bus, gen, branch)
                    row[f"ms_{m}"] = 1e3 * (time.perf_counter() - t0)
                    vm_g, va_g = bp.Vm.to_numpy(float), np.radians(bp.Va.to_numpy(float))
                    gc[m] = (vm_g, va_g)
                    cols[m] = np.maximum(brp.loading[:n].to_numpy(), brp.loading[n:].to_numpy())  # the model's own flows, no Newton
                    cols[f"{m}_prev"] = gi[m][2][idx]
                    starts[m] = prob.start(vm_pq=vm_g, va=va_g)
                    if ok:
                        starts[f"hyb{m}"] = prob.start(vm_pq=vm_p + (vm_g - gi[m][0]), va=va_p + (va_g - gi[m][1]))
                        nonref = bus.REF.to_numpy() != 1
                        for name, tx, px, thr, mask in (("va", np.degrees(va_ac - va_p), np.degrees(va_g - gi[m][1]), 0.1, nonref),
                                                        ("vm", vm_ac - vm_p, vm_g - gi[m][0], 1e-3, bus.PQ.to_numpy() == 1)):
                            mk = mask & (np.abs(tx) > thr)
                            t_, p_ = tx[mk], px[mk]
                            pre = f"{name}_{m}_"
                            row.update({pre + "n": int(mk.sum()), pre + "sign": int((np.sign(t_) == np.sign(p_)).sum()), pre + "st": t_.sum(), pre + "sp": p_.sum(),
                                        pre + "stt": (t_ * t_).sum(), pre + "spp": (p_ * p_).sum(), pre + "stp": (t_ * p_).sum()})
                nanv = np.full(n, np.nan)
                for name in STARTS:
                    if name not in starts:
                        if name in ("prev", "hybzs") or (name == "hybpg" and "pg" in models):
                            row.update({f"conv_{name}": False, f"it_{name}": -1, f"mis0_{name}": np.nan, f"mis1_{name}": np.inf, f"mis2_{name}": np.inf})
                            for k in (0, 1, 2):
                                cols[f"{name}{k}"] = nanv
                        continue
                    v0 = starts[name]
                    t0 = time.perf_counter()
                    r = prob.newton(v0)
                    row[f"ms_newton_{name}"] = 1e3 * (time.perf_counter() - t0)
                    row.update({f"conv_{name}": r["converged"], f"it_{name}": r["iterations"], f"mis0_{name}": r["mis0"],
                                f"dv_{name}": float(np.abs(r["V"] - v_ac).max()) if r["converged"] else np.nan,
                                f"dv_start_{name}": float(np.abs(v0 - v_ac).max())})
                    for k in (1, 2):
                        with np.errstate(all="ignore"):
                            row[f"mis{k}_{name}"] = float(np.abs(prob.mismatch(r["steps"][k - 1])).max() * BASE) if len(r["steps"]) >= k else (
                                r["mis_end"] if r["converged"] else np.inf)
                            cols[f"{name}{k}"] = prob.loading(r["steps"][k - 1]) if len(r["steps"]) >= k else (prob.loading(r["V"]) if r["converged"] else nanv)
                    cols[f"{name}0"] = prob.loading(v0)  # the start's voltages through the exact branch equations
                d = pd.DataFrame(cols)[rate > 0]
                part.append(d.astype({c: "float32" for c in d.columns if c not in ("state", "line")}))
                rows.append(row)
        done += len(sub)
        if wid == 0:
            print(time.strftime("%H:%M:%S"), f"worker 0: {done}/{len(load_ids)} load scenarios, {len(rows)} scenarios", flush=True)
    f = OUT / f"_part_{grid}{tag}_{wid}.parquet"
    pd.concat(part, ignore_index=True).to_parquet(f, index=False)
    return f, pd.DataFrame(rows)


def run(a):
    import numpy as np
    import pandas as pd
    import torch
    g = GRIDS[a.grid]
    net, raw = g["net"], g["raw"]
    status = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "load_scenario_idx", "idx", "br_status"])
    always_out = (status.groupby("idx").br_status.max() == 0).sort_index().to_numpy()
    load_ids = np.sort(status.load_scenario_idx.unique())
    n_all, n_load_all = status.scenario.nunique(), len(load_ids)
    cap = a.limit or g["cap"]
    if cap:
        load_ids = load_ids[:cap]
    used = status[status.load_scenario_idx.isin(load_ids)].scenario.unique()
    del status
    # graphkit's HeteroDataMVANormalizer.fit recipe on the scenarios used
    bus = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "Pd", "Qd", "Qg", "vn_kv"])
    gen = pd.read_parquet(raw / "gen_data.parquet", columns=["scenario", "p_mw"])
    bus, gen = bus[bus.scenario.isin(used)], gen[gen.scenario.isin(used)]
    vals = pd.concat([bus.Pd[bus.Pd != 0], bus.Qd[bus.Qd != 0], gen.p_mw[gen.p_mw != 0], bus.Qg[bus.Qg != 0]])
    stats = {"baseMVA_orig": 100.0, "baseMVA": float(np.percentile(vals, 95)), "vn_kv_max": float(bus.vn_kv.max())}
    recipe = stats["baseMVA"]
    if a.basemva:
        stats["baseMVA"] = a.basemva
    rel = ""
    if g["pg"]:
        folder = g["pg"][0] / f"pf_{net}_{g['pg'][1]}" / "model" / "normalizer_stats.pt"
        rs = torch.load(folder, weights_only=True)[net]
        rel = f"; released per-grid checkpoint ({g['pg'][1]}) uses baseMVA {rs['baseMVA'].item():.2f}, vn_kv_max {rs['vn_kv_max'].item():.0f}"
    log(f"=== run {net}{a.tag}: {len(used):,} of {n_all:,} scenarios ({len(load_ids):,} of {n_load_all:,} load scenarios), {int(always_out.sum())} branches always out, {a.procs} processes")
    log(f"  zero-shot normaliser: baseMVA {stats['baseMVA']:.2f} (graphkit recipe on these scenarios gives {recipe:.2f}), vn_kv_max {stats['vn_kv_max']:.0f}{rel}")
    del bus, gen
    t0 = time.perf_counter()
    with get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(worker, [(a.grid, chunk, always_out, stats, i, a.tag) for i, chunk in enumerate(np.array_split(load_ids, a.procs))])
    lines = pd.concat([pd.read_parquet(r[0]) for r in res], ignore_index=True)
    st = pd.concat([r[1] for r in res], ignore_index=True)
    st["baseMVA_zs"] = stats["baseMVA"]
    lines.to_parquet(OUT / f"lines_{a.grid}{a.tag}.parquet", index=False)
    st.to_parquet(OUT / f"states_{a.grid}{a.tag}.parquet", index=False)
    for r in res:
        r[0].unlink()
    log(f"  {net}: {len(st):,} scenarios, {len(lines):,} line readings, wall {time.perf_counter() - t0:.0f} s; base states {st.base_key.nunique():,} "
        f"(failed intact solves: {st[~st.intact_ok].base_key.nunique()}); checks: stored AC solution mismatch max {st.mis_ac_stored.max():.1e} MW, "
        f"admittance formula vs stored max {st.ydiff.max():.1e}")


def report(a):
    import numpy as np
    import pandas as pd
    out = []
    for grid, g in GRIDS.items():
        f = OUT / f"lines_{grid}{a.tag}.parquet"
        if not f.exists():
            continue
        net = g["net"]
        full = pd.read_parquet(f)
        st = pd.read_parquet(OUT / f"states_{grid}{a.tag}.parquet").set_index("state")
        has_pg = "pg" in full.columns
        real = (full.ac > LIM).to_numpy()
        crit = (full.ac > LIM).groupby(full.state).any()
        n_real = int(real.sum())
        pgname = f"per-grid {g['pg'][1].capitalize()} checkpoint" if g["pg"] else "no per-grid checkpoint"
        log(f"=== report {net}{a.tag}: {len(crit):,} scenarios, {len(full):,} line readings; zero-shot = pretrained Base, baseMVA {st.baseMVA_zs.iloc[0]:.2f}; pg = {pgname}")
        log(f"  {int(crit.sum()):,} of {len(crit):,} scenarios have a real overload ({crit.mean():.1%}); {n_real:,} overloaded line readings; "
            f"branches removed per scenario: {st.n_out.value_counts().sort_index().to_dict()}; base states {st.base_key.nunique():,} (failed {st[~st.intact_ok].base_key.nunique()})")
        q = lambda v: f"{v.replace(np.inf, np.nan).median():.3g} / {v.replace(np.inf, np.nan).quantile(.99):.3g} / {v.replace(np.inf, np.nan).max():.3g}"  # noqa: E731
        log(f"  {'Newton from':<11} {'converged':>13} {'same sol.':>9} {'it mean':>8} {'med':>4} {'max':>4}   start mismatch MW med / p99 / max      after 1 step             after 2 steps        start max|V-V_ac| med / p99")
        for name in STARTS:
            if f"conv_{name}" not in st.columns:
                continue
            c = st[st[f"conv_{name}"].astype(bool)]
            same = int((c[f"dv_{name}"] < 1e-5).sum())
            ds = st[f"dv_start_{name}"]
            log(f"  {name:<11} {len(c):>6,}/{len(st):<6,} {same:>9,} {c[f'it_{name}'].mean():>8.2f} {c[f'it_{name}'].median():>4.0f} {c[f'it_{name}'].max():>4}   "
                f"{q(st[f'mis0_{name}']):<34} {q(st[f'mis1_{name}']):<24} {q(st[f'mis2_{name}']):<24} {ds.median():.3g} / {ds.quantile(.99):.3g}")
            out.append({"network": net, "kind": "newton", "name": name, "scenarios": len(st), "converged": len(c), "same_solution": same, "it_mean": c[f"it_{name}"].mean(),
                        "it_median": c[f"it_{name}"].median(), "it_max": c[f"it_{name}"].max(), "mis0_median_mw": st[f"mis0_{name}"].median(),
                        "mis0_p99_mw": st[f"mis0_{name}"].quantile(.99), "mis1_median_mw": st[f"mis1_{name}"].replace(np.inf, np.nan).median(),
                        "mis2_median_mw": st[f"mis2_{name}"].replace(np.inf, np.nan).median(), "dv_start_median": ds.median()})
        ms = ", ".join(f"{m} {st[f'ms_{m}'].median():.0f} ms" for m in ("zs", "pg") if f"ms_{m}" in st.columns)
        log(f"  inference per scenario (1 thread, median): {ms}; full Newton from flat {st.ms_newton_flat.median():.1f} ms")
        for m in ("zs", "pg") if has_pg else ("zs",):
            for name, unit in (("va", "bus angles moving > 0.1 deg"), ("vm", "PQ magnitudes moving > 0.001 p.u.")):
                n, st_, sp, stt, spp, stp_ = (st[f"{name}_{m}_{k}"].sum() for k in ("n", "st", "sp", "stt", "spp", "stp"))
                corr = (n * stp_ - st_ * sp) / np.sqrt((n * stt - st_**2) * (n * spp - sp**2))
                log(f"  delta {m}: predicted vs true change, {unit}: {int(n):,} readings, sign agreement {st[f'{name}_{m}_sign'].sum() / n:.1%}, correlation {corr:.3f}, "
                    f"slope (pred on true) {(n * stp_ - st_ * sp) / (n * stt - st_**2):.3f}")
                out.append({"network": net, "kind": f"delta_{name}", "name": m, "readings": int(n), "sign_agreement": st[f"{name}_{m}_sign"].sum() / n, "correlation": corr})
            dt, dp = (full.ac - full.ac_prev).to_numpy(), (full[m] - full[f"{m}_prev"]).to_numpy()
            mk = np.isfinite(dt) & np.isfinite(dp) & (np.abs(dt) > 0.01)
            sign, corr = (np.sign(dt[mk]) == np.sign(dp[mk])).mean(), np.corrcoef(dt[mk], dp[mk])[0, 1]
            log(f"  delta {m}: predicted loading change vs true change, lines moving > 1 point: {int(mk.sum()):,} readings ({mk.mean():.2%} of all), sign agreement "
                f"{sign:.1%}, correlation {corr:.3f}, slope {np.polyfit(dt[mk], dp[mk], 1)[0]:.3f}; raw {m} error on these lines median "
                f"{100 * np.median(np.abs(full[m].to_numpy()[mk] - full.ac.to_numpy()[mk])):.2f} points, true |change| median {100 * np.median(np.abs(dt[mk])):.2f} points")
            out.append({"network": net, "kind": "delta_loading", "name": m, "readings": int(mk.sum()), "sign_agreement": sign, "correlation": corr})
        full["anch_zs"] = full.ac_prev + full.zs - full.zs_prev
        sur = ["zs", "zs0", "zs1", "zs2", "prev0", "prev1", "prev2", "anch_zs", "hybzs0", "hybzs1", "hybzs2"]
        if has_pg:
            sur += ["pg", "pg1", "pg2", "hybpg1", "hybpg2"]
        sur += ["dc", "dc1", "dc2", "flat1", "flat2"]
        log(f"  {'predictor':<9} {'err median':>10} {'p99':>8} {'max':>9} {'NaN':>7} {'flagged':>8} {'secure flagged':>14} {'missed scen.':>13} {'missed lines':>15} {'share':>7} {'worst':>7}")
        for s in sur:
            e = (full[s] - full.ac).abs().to_numpy() * 100
            okm = np.isfinite(e)
            flag = (full[s].fillna(BIG) > LIM).groupby(full.state).any()
            missed = real & ~full.state.map(flag).to_numpy()
            k = int(missed.sum())
            r = {"network": net, "kind": "screen", "name": s, "scenarios": len(crit), "err_median_pp": float(np.median(e[okm])), "err_p99_pp": float(np.quantile(e[okm], .99)),
                 "err_max_pp": float(e[okm].max()), "nan_readings": int((~okm).sum()), "flagged": float(flag.mean()),
                 "secure_flagged": float((flag & ~crit).sum() / max((~crit).sum(), 1)), "critical_scenarios": int(crit.sum()),
                 "critical_missed": int((~flag & crit).sum()), "overloads": n_real, "overloads_missed": k, "missed_share": k / max(n_real, 1),
                 "worst_missed_loading": float(full.ac.to_numpy()[missed].max()) if k else np.nan}
            out.append(r)
            worst = f"{r['worst_missed_loading']:.1%}" if k else ""
            log(f"  {s:<9} {r['err_median_pp']:>10.4f} {r['err_p99_pp']:>8.3f} {r['err_max_pp']:>9.2f} {r['nan_readings']:>7} {r['flagged']:>8.1%} {r['secure_flagged']:>14.1%} "
                f"{r['critical_missed']:>6}/{r['critical_scenarios']:<6} {k:>7}/{n_real:<7} {r['missed_share']:>7.2%} {worst:>7}")
    pd.DataFrame(out).to_csv(OUT / f"summary{a.tag}.csv", index=False)
    log(f"saved {OUT / f'summary{a.tag}.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["run", "report"])
    p.add_argument("--grid", default="500", choices=list(GRIDS))
    p.add_argument("--procs", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="number of load scenarios (base states)")
    p.add_argument("--tag", default="")
    p.add_argument("--basemva", type=float, default=0.0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"run": run, "report": report}[a.phase](a)

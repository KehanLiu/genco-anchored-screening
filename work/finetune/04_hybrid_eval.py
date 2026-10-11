"""Hybrid-start evaluation of ANY GENCO checkpoint folder on the EXISTING test scenarios of a GOC grid.

Per line (rated lines only), loading after 0 / 1 / 2 Newton steps from three start points:
    genco, genco1, genco2   GENCO's own voltages (as work/goc<N>/02_newton.py)
    prev1, prev2            pre-outage AC solution (model independent: read from work/goc<N>/lines_prev_tiny_seed0.parquet
                            when it covers the scenarios, otherwise computed here; --prev file|compute|auto)
    hyb0, hyb1, hyb2        HYBRID start = pre-outage AC voltages + [GENCO(contingency) - GENCO(pre-outage)] of THIS model
plus ac (stored AC loading), ac_prev (pre-outage AC loading), genco_prev (this model on the pre-outage grid), n_out.
Per scenario: start mismatch, mismatch after 1 / 2 steps and Newton iterations to convergence from the hybrid and the
GENCO start (and the pre-outage start), start distances, pooled sums for "does the predicted change carry information".
The loop is analysis_hybrid_start.worker with the model folder as a parameter; helpers are imported from
analysis/analysis_prev_control.py (pre-outage state, admittances, Newton problem) - nothing under work/goc<N> is written.

    work/texas/tmo 3600 .venv/bin/python -u work/finetune/04_hybrid_eval.py --grid 500 --procs 3 \
        [--model work/finetune/models/pf_case500_goc_ft_seed0] [--tag ft_seed0] [--limit N_LOAD_SCENARIOS]
    .venv/bin/python work/finetune/04_hybrid_eval.py --grid 500 --tag released --check     # compare with work/goc500/lines_hybrid_tiny_seed0.parquet

Defaults: --model work/goc<grid>/pf_case<grid>_goc_tiny (released Tiny seed0), --tag released.
Writes work/finetune/results/lines_hybrid_<grid>_<tag>.parquet and states_hybrid_<grid>_<tag>.parquet.
Resumable: every chunk of --chunk load scenarios is saved to results/parts_hybrid_<grid>_<tag>/ as soon as it is done;
a rerun only computes the missing chunks (parts are keyed by their load-scenario range; remove them with --clean
after the final files are written). An existing final file is not recomputed unless --force.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
RESULTS = HERE / "results"
PREDICTORS = ["genco", "genco1", "genco2", "prev1", "prev2", "hyb0", "hyb1", "hyb2"]
FLOATS = ["ac", "genco", "genco1", "genco2", "prev1", "prev2", "hyb0", "hyb1", "hyb2", "ac_prev", "genco_prev"]
_G = {}


def init(grid, model, always_out, compute_prev, parts):
    import engine  # Julia before torch
    import torch
    import analysis_prev_control as pc
    torch.set_num_threads(1)
    model = Path(model)
    here, net, raw = pc.paths(grid)
    engine.WORK = model.parent
    _G.update(genco=engine.Genco(net, model.name.removeprefix(f"pf_{net}_")), nw=pc.newton_module(), pc=pc, raw=raw,
              always_out=always_out, compute_prev=compute_prev, parts=Path(parts))


def chunk_name(ids):
    return f"{int(ids[0]):06d}_{int(ids[-1]):06d}_{len(ids)}"


def work(load_ids):
    """One chunk of load scenarios -> part files. Same computation as analysis_hybrid_start.worker (+ GENCO / pre-outage starts)."""
    import numpy as np
    import pandas as pd
    genco, nw, pc, raw, always_out = _G["genco"], _G["nw"], _G["pc"], _G["raw"], _G["always_out"]
    BASE = pc.BASE
    t0 = time.perf_counter()
    flt = [("load_scenario_idx", "in", [int(i) for i in load_ids])]
    tabs = {t: {s: d for s, d in pd.read_parquet(raw / f"{t}_data.parquet", filters=flt).groupby("scenario")} for t in ("bus", "gen", "branch")}
    groups, lines, rows = {}, [], []
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
        # ---- pre-outage state: removed branches back in service, solved with Newton (as analysis/analysis_prev_control.py)
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
        bp_i, br_i = genco.solve(bus0, gen0, bi)  # this model on the pre-outage grid (no pre-outage AC solution fed in)
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
            bp_c, br_c = genco.solve(bus, gen, branch)
            vm_gc, va_gc = bp_c.Vm.to_numpy(float), np.radians(bp_c.Va.to_numpy(float))
            g_c = np.maximum(br_c.loading[:n].to_numpy(), br_c.loading[n:].to_numpy())
            n_out = int((~live).sum()) - int(always_out.sum())
            row = {"state": int(s), "n_out": n_out, "load_idx": load_idx, "base_key": key[:12], "intact_ok": ok}
            nanv = np.full(n, np.nan)
            out = {}

            def newton_from(name, v0):
                r = prob.newton(v0)
                row.update({f"conv_{name}": r["converged"], f"it_{name}": r["iterations"], f"mis0_{name}": r["mis0"],
                            f"dv_{name}": float(np.abs(r["V"] - v_ac).max()) if r["converged"] else np.nan})
                for k in (1, 2):
                    with np.errstate(all="ignore"):
                        row[f"mis{k}_{name}"] = float(np.abs(prob.mismatch(r["steps"][k - 1])).max() * BASE) if len(r["steps"]) >= k else (
                            r["mis_end"] if r["converged"] else np.inf)
                    out[f"{name}{k}"] = prob.loading(r["steps"][k - 1]) if len(r["steps"]) >= k else (prob.loading(r["V"]) if r["converged"] else nanv)

            newton_from("genco", prob.start(vm_pq=vm_gc, va=va_gc))
            for name in (("hyb", "prev") if _G["compute_prev"] else ("hyb",)):
                if not ok:
                    row.update({f"conv_{name}": False, f"it_{name}": -1, f"mis0_{name}": np.nan, f"dv_{name}": np.nan, f"mis1_{name}": np.inf, f"mis2_{name}": np.inf})
                    out[f"{name}1"] = out[f"{name}2"] = nanv
            out["hyb0"] = nanv
            if ok:
                v0 = prob.start(vm_pq=vm_p + (vm_gc - vm_gi), va=va_p + (va_gc - va_gi))
                v0_prev = prob.start(vm_pq=vm_p, va=va_p)
                newton_from("hyb", v0)
                out["hyb0"] = prob.loading(v0)
                if _G["compute_prev"]:
                    newton_from("prev", v0_prev)
                row.update({"dv_start_hyb": float(np.abs(v0 - v_ac).max()), "dv_start_prev": float(np.abs(v0_prev - v_ac).max()),
                            "dv_start_genco": float(np.abs(prob.start(vm_pq=vm_gc, va=va_gc) - v_ac).max())})
                nonref = bus.REF.to_numpy() != 1
                for name, tx, px, thr, mask in (("va", np.degrees(va_ac - va_p), np.degrees(va_gc - va_gi), 0.1, nonref),
                                                ("vm", vm_ac - vm_p, vm_gc - vm_gi, 1e-3, bus.PQ.to_numpy() == 1)):
                    m = mask & (np.abs(tx) > thr)
                    t_, p_ = tx[m], px[m]
                    row.update({f"{name}_n": int(m.sum()), f"{name}_sign": int((np.sign(t_) == np.sign(p_)).sum()), f"{name}_st": t_.sum(), f"{name}_sp": p_.sum(),
                                f"{name}_stt": (t_ * t_).sum(), f"{name}_spp": (p_ * p_).sum(), f"{name}_stp": (t_ * p_).sum()})
            d = {"state": int(s), "n_out": n_out, "line": idx, "ac": s_ac / rate, "genco": g_c, "genco1": out["genco1"], "genco2": out["genco2"]}
            if _G["compute_prev"]:
                d.update(prev1=out["prev1"], prev2=out["prev2"])
            d.update(hyb0=out["hyb0"], hyb1=out["hyb1"], hyb2=out["hyb2"], ac_prev=ac_prev[idx], genco_prev=g_prev[idx])
            d = pd.DataFrame(d)[rate > 0]
            lines.append(d.astype({c: "float32" for c in d.columns if c in FLOATS}))
            rows.append(row)
    name = chunk_name(load_ids)
    for kind, df in (("lines", pd.concat(lines, ignore_index=True)), ("states", pd.DataFrame(rows))):
        tmp = _G["parts"] / f".{kind}_{name}.tmp"
        df.to_parquet(tmp, index=False)
        tmp.rename(_G["parts"] / f"{kind}_{name}.parquet")  # states last: its presence marks the chunk as done
    return name, len(rows), time.perf_counter() - t0


def change_stats(lines, st):
    """Does the model's predicted change (contingency minus pre-outage) carry information? Returns printable lines."""
    import numpy as np
    out = []
    for name, unit in (("va", "bus angles moving > 0.1 deg"), ("vm", "PQ magnitudes moving > 0.001 p.u.")):
        n, st_, sp, stt, spp, stp_ = (st[f"{name}_{k}"].sum() for k in ("n", "st", "sp", "stt", "spp", "stp"))
        corr = (n * stp_ - st_ * sp) / np.sqrt((n * stt - st_**2) * (n * spp - sp**2))
        out.append(f"  predicted vs true change, {unit}: {int(n):,} readings, sign agreement {st[f'{name}_sign'].sum() / n:.1%}, correlation {corr:.3f}, "
                   f"slope (pred on true) {(n * stp_ - st_ * sp) / (n * stt - st_**2):.3f}")
    dt, dp = (lines.ac - lines.ac_prev).to_numpy(), (lines.genco - lines.genco_prev).to_numpy()
    m = np.isfinite(dt) & np.isfinite(dp) & (np.abs(dt) > 0.01)
    out.append(f"  predicted vs true loading change, lines moving > 1 point: {int(m.sum()):,} readings ({m.mean():.2%} of all), sign agreement "
               f"{(np.sign(dt[m]) == np.sign(dp[m])).mean():.1%}, correlation {np.corrcoef(dt[m], dp[m])[0, 1]:.3f}; raw GENCO error on these lines median "
               f"{100 * np.median(np.abs(lines.genco.to_numpy()[m] - lines.ac.to_numpy()[m])):.2f} points, true |change| median {100 * np.median(np.abs(dt[m])):.2f} points")
    return out


def check(a, f_lines, f_states):
    """Compare with the files of analysis/analysis_hybrid_start.py / 02_newton.py / analysis/analysis_prev_control.py (released Tiny seed0)."""
    import pandas as pd
    here = ROOT / "work" / f"goc{a.grid}"
    new = pd.read_parquet(f_lines)
    print(f"check {f_lines.name}: {new.state.nunique():,} scenarios, {len(new):,} line readings")
    for ref, cols in (("lines_hybrid_tiny_seed0", {"ac": "ac", "hyb0": "hyb0", "hyb1": "hyb1", "hyb2": "hyb2", "ac_prev": "ac_prev", "genco_prev": "genco_prev", "genco_now": "genco"}),
                      ("lines_newton_tiny_seed0", {"genco": "genco", "genco1": "genco1", "genco2": "genco2"}),
                      ("lines_prev_tiny_seed0", {"prev1": "prev1", "prev2": "prev2"})):
        old = pd.read_parquet(here / f"{ref}.parquet", columns=["state", "line", *cols])
        old = old[old.state.isin(new.state.unique())].rename(columns={c: f"{c}__ref" for c in cols})
        m = new.merge(old, on=["state", "line"], how="outer", indicator=True)
        print(f"  vs work/goc{a.grid}/{ref}.parquet: rows only here {int((m._merge == 'left_only').sum())}, only there {int((m._merge == 'right_only').sum())}, common {int((m._merge == 'both').sum()):,}")
        for c, mine in cols.items():
            x, y = m[mine], m[f"{c}__ref"]
            print(f"    {mine:<10} max |diff| {float((x - y).abs().max()):.3e}   NaN here/there {int(x.isna().sum())}/{int(y.isna().sum())}   NaN pattern differs: {int((x.isna() != y.isna()).sum())}")
    st = pd.read_parquet(f_states).set_index("state")
    old = pd.read_parquet(here / "states_hybrid_tiny_seed0.parquet").set_index("state").reindex(st.index)
    for c in ("mis0_hyb", "mis1_hyb", "mis2_hyb", "it_hyb", "dv_start_hyb", "va_stp", "vm_stp"):
        x, y = st[c].astype(float).replace(float("inf"), float("nan")), old[c].astype(float).replace(float("inf"), float("nan"))
        print(f"  states {c:<13} max |diff| {float((x - y).abs().max()):.3e} (reference max |value| {float(y.abs().max()):.3e})")
    print(f"  states conv_hyb identical: {bool((st.conv_hyb == old.conv_hyb).all())}; scenarios missing in reference: {int(old.conv_hyb.isna().sum())}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, required=True, choices=[500, 2000, 10000])
    p.add_argument("--model", default="", help="folder pf_<net>_<variant> with config.yaml and model/ (default: released Tiny seed0)")
    p.add_argument("--tag", default="", help="default: 'released' for the default model, else the variant part of the folder name")
    p.add_argument("--procs", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="first N load scenarios only (5 topology variants each)")
    p.add_argument("--chunk", type=int, default=0, help="load scenarios per saved part (default 20; 4 on 10,000 buses)")
    p.add_argument("--prev", default="auto", choices=["auto", "file", "compute"])
    p.add_argument("--force", action="store_true", help="recompute even if the final files exist (existing parts are still reused)")
    p.add_argument("--clean", action="store_true", help="delete the parts folder after the final files are written")
    p.add_argument("--check", action="store_true", help="only compare existing output with the released-model reference files")
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import analysis_prev_control as pc
    here, net, raw = pc.paths(a.grid)
    model = Path(a.model).resolve() if a.model else here / f"pf_{net}_tiny"
    tag = a.tag or ("released" if not a.model else model.name.removeprefix(f"pf_{net}_"))
    RESULTS.mkdir(exist_ok=True)
    f_lines, f_states = RESULTS / f"lines_hybrid_{a.grid}_{tag}.parquet", RESULTS / f"states_hybrid_{a.grid}_{tag}.parquet"
    if a.check:
        return check(a, f_lines, f_states)
    for f in (model / "config.yaml", model / "model" / "best_model_state_dict.pt", model / "model" / "normalizer_stats.pt"):
        assert f.exists(), f"missing {f}"
    if f_lines.exists() and f_states.exists() and not a.force:
        print(f"{f_lines} exists - nothing to do (use --force to recompute)")
        return
    status = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "load_scenario_idx", "idx", "br_status"])
    always_out = (status.groupby("idx").br_status.max() == 0).sort_index().to_numpy()
    load_ids = np.sort(status.load_scenario_idx.unique())
    if a.limit:
        load_ids = load_ids[: a.limit]
    states = np.sort(status.scenario[status.load_scenario_idx.isin(load_ids)].unique())
    del status
    f_prev = here / "lines_prev_tiny_seed0.parquet"
    compute_prev = a.prev == "compute"
    if a.prev != "compute":
        have = set(pd.read_parquet(f_prev, columns=["state"]).state.unique()) if f_prev.exists() else set()
        covered = set(states) <= have
        assert covered or a.prev == "auto", f"{f_prev} does not cover the scenarios"
        compute_prev = not covered
    size = a.chunk or (4 if a.grid == 10000 else 20)
    chunks = [load_ids[i:i + size] for i in range(0, len(load_ids), size)]
    parts = RESULTS / f"parts_hybrid_{a.grid}_{tag}"
    parts.mkdir(exist_ok=True)
    meta = {"grid": a.grid, "model": str(model), "weights_bytes": (model / "model" / "best_model_state_dict.pt").stat().st_size,
            "weights_mtime": (model / "model" / "best_model_state_dict.pt").stat().st_mtime, "compute_prev": bool(compute_prev)}
    mf = parts / "meta.json"
    if mf.exists():
        assert json.loads(mf.read_text()) == meta, f"{parts} was written for another model / setting: {mf.read_text()} vs {meta} (other --tag, or move the folder away)"
    mf.write_text(json.dumps(meta))
    todo = [c for c in chunks if not (parts / f"states_{chunk_name(c)}.parquet").exists()]
    print(f"{net}: {len(states):,} scenarios in {len(load_ids):,} load scenarios, model {model}, tag {tag}; pre-outage start "
          f"{'computed here' if compute_prev else 'from ' + str(f_prev)}; {len(chunks)} chunks of {size}, {len(chunks) - len(todo)} already done, {a.procs} processes", flush=True)
    t0 = time.perf_counter()
    if todo:
        with get_context("spawn").Pool(a.procs, initializer=init, initargs=(a.grid, str(model), always_out, compute_prev, str(parts))) as pool:
            for i, (name, n, sec) in enumerate(pool.imap_unordered(work, todo), 1):
                if i == 1 or i % max(len(todo) // 20, 1) == 0 or i == len(todo):
                    print(time.strftime("%H:%M:%S"), f"chunk {name}: {n} scenarios in {sec:.0f} s; {i}/{len(todo)} chunks, wall {time.perf_counter() - t0:.0f} s", flush=True)
    lines = pd.concat([pd.read_parquet(parts / f"lines_{chunk_name(c)}.parquet") for c in chunks], ignore_index=True)
    st = pd.concat([pd.read_parquet(parts / f"states_{chunk_name(c)}.parquet") for c in chunks], ignore_index=True)
    assert set(st.state) == set(states), "scenario set of the parts differs from the request"
    if not compute_prev:
        prev = pd.read_parquet(f_prev, columns=["state", "line", "prev1", "prev2"])
        n = len(lines)
        lines = lines.merge(prev[prev.state.isin(set(states))], on=["state", "line"], how="left", validate="one_to_one")
        assert len(lines) == n
        sp = pd.read_parquet(here / "states_prev_tiny_seed0.parquet", columns=["state", "conv_prev", "it_prev", "mis0_prev", "mis1_prev", "mis2_prev"])
        st = st.merge(sp, on="state", how="left", validate="one_to_one")
    lines = lines[["state", "n_out", "line", "ac", *PREDICTORS, "ac_prev", "genco_prev"]]
    tmp = f_lines.with_suffix(".tmp")
    lines.to_parquet(tmp, index=False)
    st.to_parquet(f_states, index=False)
    tmp.rename(f_lines)
    q = lambda v: f"{v.replace(np.inf, np.nan).median():.3g} / {v.replace(np.inf, np.nan).quantile(.99):.3g}"  # noqa: E731
    print(f"{len(st):,} scenarios, {len(lines):,} line readings, wall {time.perf_counter() - t0:.0f} s -> {f_lines}")
    print(f"  base states without a pre-outage solution: {st[~st.intact_ok].base_key.nunique()}; max |V - V_ac stored| after convergence from the hybrid start {st.dv_hyb.max():.1e}")
    print(f"  start mismatch MW max-norm, median / p99: pre-outage {q(st.mis0_prev)}, GENCO {q(st.mis0_genco)}, hybrid {q(st.mis0_hyb)};  after one step: "
          f"pre-outage {q(st.mis1_prev)}, GENCO {q(st.mis1_genco)}, hybrid {q(st.mis1_hyb)}")
    print(f"  start distance max|V - V_ac| p.u., median / p99: pre-outage {q(st.dv_start_prev)}, GENCO {q(st.dv_start_genco)}, hybrid {q(st.dv_start_hyb)}")
    it = lambda k: f"{int(st[f'conv_{k}'].sum()):,}/{len(st):,} converged, iterations mean {st.loc[st[f'conv_{k}'].astype(bool), f'it_{k}'].mean():.2f}"  # noqa: E731
    print(f"  Newton to convergence: hybrid {it('hyb')}; pre-outage {it('prev')}; GENCO {it('genco')}")
    print("\n".join(change_stats(lines, st)))
    lim = pc.LIM
    real = (lines.ac > lim).to_numpy()
    crit = (lines.ac > lim).groupby(lines.state).any()
    for s in PREDICTORS:
        flag = (lines[s].fillna(pc.BIG) > lim).groupby(lines.state).any()
        missed = real & ~lines.state.map(flag).to_numpy()
        print(f"  {s:<7} flagged {flag.mean():6.1%}  secure flagged {(flag & ~crit).sum() / max((~crit).sum(), 1):6.1%}  overloaded lines missed {int(missed.sum())}/{int(real.sum())}")
    if a.clean:
        shutil.rmtree(parts)


if __name__ == "__main__":
    main()

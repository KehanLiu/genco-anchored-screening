"""Equivalent-circuit pilot: does anchoring cancel GENCO's representation dependence?  (work/equiv/SPEC.md)

For every stored contingency state the pre-outage grid is rebuilt and solved exactly as analysis_hybrid_start.worker does
(helpers imported from analysis_prev_control; the hybrid-start / one-step logic is copied here, not imported, because the
worker is one function with no seam for a second representation). GENCO (engine.Genco, the same inference path as every
other study) is run on the ORIGINAL tables and on a physically equivalent REWRITE of them - intact and outaged graphs
rewritten the same way - and for each representation r in {o, t}:
    genco_r      raw GENCO loading: physics flows (original admittances) at GENCO's predicted voltages mapped back to physical buses
    genco_eng_r  the engine's own per-copy flows summed back to the physical branch against the original rating (cross-check)
    g1_r         one Newton step from GENCO's raw voltages
    hyb0_r/hyb1_r   anchored start  V_prev_AC + (V_GENCO(outage) - V_GENCO(intact))  and one Newton step from it
    prev1        classical control: one Newton step from the pre-outage AC solution (representation-free)
Newton runs on the original problem; verify_equivalence proves the rewritten equations identical (maxima stored per state),
so the stored AC truth and the original Newton problem are reused for both representations.

    .venv/bin/python -u work/equiv/run_equiv.py run --grid 500 --data work/goc500/data --variant tiny --seed 0 --transform split_all --limit 20 --tag dev
    .venv/bin/python work/equiv/run_equiv.py report --grid 500 --variant tiny --seed 0 --tag dev [--transforms none,permute,...]
Transforms: none | permute | split_random_<n> | split_all | split_gen, combinable with '+', plus the modifier +refvm (known reference
Vm/Va written over GENCO's prediction before anchoring / Newton, for BOTH representations). Random choices are drawn from
--spec-seed (default 2026) and the load scenario id, so the intact and every contingency of a base state share one rewrite.
Outputs work/equiv/results/goc<N>_<variant>_seed<S>_<transform><tag>_{lines,states}.parquet and a log in work/equiv/logs/.
"""
import argparse
import hashlib
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(HERE))
import analysis_prev_control as pc  # noqa: E402
import transforms as tf  # noqa: E402

RESULTS = HERE / "results"
LOGS = HERE / "logs"
LIM, BIG, BASE = pc.LIM, pc.BIG, pc.BASE
STAGES = ["genco", "dgenco", "g1", "hyb0", "hyb1"]
ABORT_TOL = 1e-9        # SPEC.md section 7: any equivalence maximum above this aborts
IDENTITY_TOL = 1e-3     # points of rating; the `none` control must reproduce o = t to float32 at every stage


class PipelineAbort(RuntimeError):
    """SPEC.md section 7 ABORT rule: an equivalence value above 1e-9 (or non-finite), a false outage mapping, or the identity control
    `none` not reproducing o = t. Where the checks run:
      * per state in the worker (outage mapping, equivalence of the contingency and intact graphs), before that state's row exists;
      * in run(), after all workers have returned and BEFORE the parquet files are written: every stored equivalence cell
        (check_equivalence_cells) and, for transform `none`, the identity control (check_identity_patterns + check_identity);
      * in report(), on the stored parquet files, before the summary / criterion CSVs are written: the same cell-level and identity checks.
    Every check validates cells before any maximum is taken (pandas maxima skip NaN). __main__ maps PipelineAbort - and the AssertionError
    that transforms.verify_equivalence / the rewrites raise on a physics mismatch - to exit code 2 with a 'PIPELINE BROKEN - ABORT' line;
    no summary / criterion CSV is written on either path."""


def check_outage_map(ok, state, spec):
    if not bool(ok):
        raise PipelineAbort(f"outage mapping check failed in state {state} under {spec!r}: rewriting the outaged tables != removing "
                            f"every copy of the outaged branches from the rewritten intact grid")


def check_equivalence(eq, where, tol=ABORT_TOL):
    """`eq` maps check name (optionally prefixed eq_c_/eq_i_) -> maximum; gen_n_on is reported, not asserted (SPEC 11.1). NaN fails."""
    bad = {k: v for k, v in eq.items() if not k.endswith("gen_n_on") and not (abs(float(v)) <= tol)}
    if bad:
        raise PipelineAbort(f"equivalence maximum above {tol:g} in {where}: {bad}")


def check_equivalence_cells(S, where, tol=ABORT_TOL):
    """Every asserted equivalence cell of a states frame (columns eq_*, gen_n_on excepted) must be finite and <= tol. Runs BEFORE any
    maximum is taken: pandas maxima skip NaN, so a non-finite cell among passing rows would otherwise vanish in the aggregate."""
    import numpy as np
    states = S["state"].to_numpy() if "state" in S.columns else S.index.to_numpy()
    bad = {}
    for c in S.columns:
        if not c.startswith("eq_") or c.endswith("gen_n_on"):
            continue
        v = S[c].to_numpy(dtype=float)
        fail = ~(np.abs(v) <= tol)  # True for NaN, +-inf and any value above tol
        if fail.any():
            bad[c] = {"n_bad": int(fail.sum()), "n_nonfinite": int((~np.isfinite(v)).sum()), "states": [int(x) for x in states[fail][:5]]}
    if bad:
        raise PipelineAbort(f"equivalence cell(s) non-finite or above {tol:g} in {where}: {bad}")


def check_identity_patterns(L, where):
    """Identity control `none`, BEFORE any difference is reduced: at every stage the finite / non-finite pattern of the t column must
    equal that of its o column, and no (o, t) pair may be non-finite. A failed prediction on one side - or on both - is a failure of
    the control, not a reading to be skipped by a NaN-ignoring maximum."""
    import numpy as np
    bad = {}
    for s in STAGES:
        fo, ft = np.isfinite(L[f"{s}_o"].to_numpy(dtype=float)), np.isfinite(L[f"{s}_t"].to_numpy(dtype=float))
        mism, pair = int((fo != ft).sum()), int((~fo & ~ft).sum())
        if mism or pair:
            bad[s] = {"finite_mask_mismatch": mism, "nonfinite_pairs": pair}
    if bad:
        raise PipelineAbort(f"identity control `none` has non-finite readings in {where} (finite-mask mismatches / non-finite o,t pairs): {bad}")


def identity_max(L):
    """max |t - o| over all stages and line readings; call after check_identity_patterns (every reading is then finite)."""
    import numpy as np
    return max(float(np.abs(L[f"{s}_t"].to_numpy(dtype=float) - L[f"{s}_o"].to_numpy(dtype=float)).max()) * 100 for s in STAGES)


def dgenco(L, r):
    """Anchored loading: predicted CHANGE of loading on the AC base."""
    return L.ac_prev + L[f"genco_{r}"] - L[f"genco_prev_{r}"]


def check_identity(ident, tol=IDENTITY_TOL):
    if not (float(ident) < tol):
        raise PipelineAbort(f"identity control `none` does not reproduce o = t: max |t-o| over all stages {ident:.2e} points (limit {tol:g})")


def log(*a, file=None):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    if file:
        with open(file, "a") as f:
            f.write(line + "\n")


def parse_spec(spec):
    ops = {"permute": False, "refvm": False, "split_all": False, "split_gen": False, "split_random": 0, "k": 2}
    for p in spec.split("+"):
        if p == "none":
            continue
        elif p in ("permute", "refvm", "split_all", "split_gen"):
            ops[p] = True
        elif p.startswith("split_random_"):
            ops["split_random"] = int(p.split("_")[-1])
        else:
            raise ValueError(f"unknown transform part {p!r} in {spec!r}")
    return ops


def draw_choices(ops, rng, candidates, gen_ids):
    """Random choices of one base state (shared by its intact grid and all its contingencies)."""
    ch = {"split_ids": [], "gen_ids": [], "perm_seed": None}
    if ops["split_all"]:
        ch["split_ids"] = [int(i) for i in candidates]
    elif ops["split_random"]:
        ch["split_ids"] = sorted(int(i) for i in rng.choice(candidates, ops["split_random"], replace=False))
    if ops["split_gen"]:
        ch["gen_ids"] = [int(i) for i in gen_ids]
    if ops["permute"]:
        ch["perm_seed"] = int(rng.integers(2**31 - 1))
    return ch


def rewrite(rep, ops, ch):
    if ch["split_ids"]:
        rep = tf.split_branches(rep, ch["split_ids"], k=ops["k"])
    if ch["gen_ids"]:
        rep = tf.split_generators(rep, ch["gen_ids"])
    if ch["perm_seed"] is not None:
        rep = tf.permute(rep, ch["perm_seed"])
    return rep


def model_name(variant, seed):
    return variant + (f"_seed{seed}" if seed else "")


def out_stem(grid, variant, seed, spec, tag):
    return RESULTS / f"goc{grid}_{variant}_seed{seed}_{spec.replace('+', '_')}{tag}"


def worker(job):
    grid, raw, load_ids, always_out, wid, variant, seed, spec, spec_seed, logfile = job
    import engine  # Julia before torch
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    ops = parse_spec(spec)
    here = ROOT / "work" / f"goc{grid}"
    engine.WORK = here
    genco = engine.Genco(f"case{grid}_goc", model_name(variant, seed))
    nw = pc.newton_module()
    raw = Path(raw)

    def infer(rep, bus_orig, rate_orig):
        """GENCO on one representation -> (vm, va[rad]) in physical bus order, engine loading per physical idx, ms, raw REF prediction."""
        t0 = time.perf_counter()
        bp, br = genco.solve(*rep.tables())
        ms = 1e3 * (time.perf_counter() - t0)
        vm = rep.to_original(bp.Vm.to_numpy(float))
        va = rep.to_original(np.radians(bp.Va.to_numpy(float)))
        ref = bus_orig.REF.to_numpy() == 1
        raw_ref = (float(vm[ref][0]), float(np.degrees(va[ref][0])))
        if ops["refvm"]:
            vm, va_deg = tf.enforce_reference(vm, np.degrees(va), bus_orig)
            va = np.radians(va_deg)
        return vm, va, tf.engine_loading_to_physical(rep, br, rate_orig), ms, raw_ref

    def after(r, k, prob, nanv):
        if len(r["steps"]) >= k:
            return prob.loading(r["steps"][k - 1])
        return prob.loading(r["V"]) if r["converged"] else nanv

    def mis_after(r, k, prob):
        with np.errstate(all="ignore"):
            if len(r["steps"]) >= k:
                return float(np.abs(prob.mismatch(r["steps"][k - 1])).max() * BASE)
            return r["mis_end"] if r["converged"] else np.inf

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
            # ---- intact grid (as analysis_hybrid_start.worker): removed branches back in service, admittances recomputed
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
            rate_orig = branch0.rate_a.to_numpy(float)  # ratings are the same in every member (checked by the hash of r/x/b/tap... and below)
            live_i = (bi.br_status == 1).to_numpy()
            ac_prev = np.full(len(bi), np.nan)
            if ok:
                ac_prev[live_i] = prob_i.loading(V_prev)
                vm_p, va_p = np.abs(V_prev), np.angle(V_prev)
            # ---- the rewrite of this base state, shared by the intact grid and every contingency
            rng = np.random.default_rng([spec_seed, int(grid), load_idx])
            candidates = np.flatnonzero((rate_orig > 0) & ~always_out)
            gen_ids = gen0.idx[(gen0.in_service == 1) & (gen0.is_slack_gen == 0)].to_numpy()
            ch = draw_choices(ops, rng, candidates, gen_ids)
            rep_i = {"o": tf.Rep.original(bus0, gen0, bi)}
            rep_i["t"] = rewrite(rep_i["o"], ops, ch)
            eq_i = tf.verify_equivalence(rep_i["o"], rep_i["t"])
            gi = {r: infer(rep_i[r], bus0, rate_orig) for r in ("o", "t")}
            g_prev = {r: np.full(len(bi), np.nan) for r in ("o", "t")}
            for r in ("o", "t"):
                g_prev[r][live_i] = prob_i.loading(gi[r][0] * np.exp(1j * gi[r][1]))
            # ---- every contingency of this base state
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
                outaged = set(branch.idx[(branch.br_status == 0) & ~always_out].tolist())
                rep_c = {"o": tf.Rep.original(bus, gen, branch)}
                rep_c["t"] = rewrite(rep_c["o"], ops, ch)
                eq_c = tf.verify_equivalence(rep_c["o"], rep_c["t"])
                # outage mapping check: rewriting the outaged tables == taking every copy of the outaged branches out of the rewritten intact grid
                mapped = tf.apply_outage(rep_i["t"], outaged)
                outage_map_ok = bool(np.array_equal(mapped.branch.br_status.to_numpy(), rep_c["t"].branch.br_status.to_numpy())
                                     and np.array_equal(mapped.phys, rep_c["t"].phys))
                check_outage_map(outage_map_ok, s, spec)  # SPEC ABORT: raised here, before any result is written
                check_equivalence(eq_c, f"contingency state {s} under {spec!r}")
                check_equivalence(eq_i, f"intact grid of load scenario {load_idx} under {spec!r}")
                gc = {r: infer(rep_c[r], bus, rate_orig) for r in ("o", "t")}
                row = {"state": int(s), "load_idx": load_idx, "base_key": key[:12], "transform": spec, "intact_ok": ok,
                       "n_out": len(outaged), "split_ids": json.dumps(ch["split_ids"]) if len(ch["split_ids"]) <= 8 else f"all:{len(ch['split_ids'])}",
                       "n_split": len(ch["split_ids"]), "n_gen_split": len(ch["gen_ids"]), "perm_seed": -1 if ch["perm_seed"] is None else ch["perm_seed"],
                       "split_outaged": int(len(outaged & set(ch["split_ids"]))), "outage_map_ok": outage_map_ok,
                       "ms_genco_o": gc["o"][3], "ms_genco_t": gc["t"][3],
                       "ref_vm_raw_o": gc["o"][4][0], "ref_vm_raw_t": gc["t"][4][0], "ref_va_raw_o": gc["o"][4][1], "ref_va_raw_t": gc["t"][4][1],
                       "ref_vm_set": float(vm_ac[bus.REF.to_numpy() == 1][0]), "ref_va_set": float(np.degrees(va_ac[bus.REF.to_numpy() == 1][0]))}
                for k_, v in eq_c.items():
                    row[f"eq_c_{k_}"] = v
                for k_, v in eq_i.items():
                    row[f"eq_i_{k_}"] = v
                # raw voltage drift between representations (diagnostic), in p.u. and degrees
                nonref = bus.REF.to_numpy() != 1
                row.update({"drift_vm_c": float(np.abs(gc["t"][0] - gc["o"][0]).max()), "drift_va_c": float(np.degrees(np.abs(gc["t"][1] - gc["o"][1])[nonref]).max()),
                            "drift_vm_i": float(np.abs(gi["t"][0] - gi["o"][0]).max()), "drift_va_i": float(np.degrees(np.abs(gi["t"][1] - gi["o"][1])[nonref]).max()),
                            "drift_dvm": float(np.abs((gc["t"][0] - gi["t"][0]) - (gc["o"][0] - gi["o"][0])).max()),
                            "drift_dva": float(np.degrees(np.abs((gc["t"][1] - gi["t"][1]) - (gc["o"][1] - gi["o"][1]))[nonref]).max())})
                nanv = np.full(n, np.nan)
                cols = {"state": int(s), "line": idx, "ac": s_ac / rate, "ac_prev": ac_prev[idx]}
                if ok:
                    rp = prob.newton(prob.start(vm_pq=vm_p, va=va_p), keep=1)
                    row.update({"conv_prev": rp["converged"], "it_prev": rp["iterations"], "mis0_prev": rp["mis0"], "mis1_prev": mis_after(rp, 1, prob)})
                    cols["prev1"] = after(rp, 1, prob, nanv)
                else:
                    row.update({"conv_prev": False, "it_prev": -1, "mis0_prev": np.nan, "mis1_prev": np.inf})
                    cols["prev1"] = nanv
                for r in ("o", "t"):
                    vm_gc, va_gc, load_eng, _, _ = gc[r]
                    vm_gi, va_gi = gi[r][0], gi[r][1]
                    V_g = vm_gc * np.exp(1j * va_gc)
                    cols[f"genco_{r}"] = prob.loading(V_g)
                    cols[f"genco_eng_{r}"] = load_eng[idx]
                    cols[f"genco_prev_{r}"] = g_prev[r][idx]
                    row[f"agg_check_{r}"] = float(np.nanmax(np.abs(load_eng[idx] - cols[f"genco_{r}"]))) if not ops["refvm"] else np.nan
                    row[f"mis_raw_{r}"] = float(np.abs(prob.mismatch(V_g)).max() * BASE)
                    rg = prob.newton(prob.start(vm_pq=vm_gc, va=va_gc), keep=1)
                    row.update({f"conv_g_{r}": rg["converged"], f"it_g_{r}": rg["iterations"], f"mis0_g_{r}": rg["mis0"], f"mis1_g_{r}": mis_after(rg, 1, prob)})
                    cols[f"g1_{r}"] = after(rg, 1, prob, nanv)
                    if ok:
                        v0 = prob.start(vm_pq=vm_p + (vm_gc - vm_gi), va=va_p + (va_gc - va_gi))
                        rh = prob.newton(v0, keep=1)
                        row.update({f"conv_hyb_{r}": rh["converged"], f"it_hyb_{r}": rh["iterations"], f"mis0_hyb_{r}": rh["mis0"], f"mis1_hyb_{r}": mis_after(rh, 1, prob),
                                    f"dv_start_hyb_{r}": float(np.abs(v0 - v_ac).max()), f"dv_hyb_{r}": float(np.abs(rh["V"] - v_ac).max()) if rh["converged"] else np.nan})
                        cols[f"hyb0_{r}"], cols[f"hyb1_{r}"] = prob.loading(v0), after(rh, 1, prob, nanv)
                    else:
                        row.update({f"conv_hyb_{r}": False, f"it_hyb_{r}": -1, f"mis0_hyb_{r}": np.nan, f"mis1_hyb_{r}": np.inf, f"dv_start_hyb_{r}": np.nan, f"dv_hyb_{r}": np.nan})
                        cols[f"hyb0_{r}"], cols[f"hyb1_{r}"] = nanv, nanv
                d = pd.DataFrame(cols)[rate > 0]
                lines.append(d.astype({c: "float32" for c in d.columns if c not in ("state", "line")}))
                rows.append(row)
        if wid == 0:
            log(f"  worker 0 [{spec}]: {len(rows)} states done", file=logfile)
    return pd.concat(lines, ignore_index=True), pd.DataFrame(rows)


def run(a):
    import numpy as np
    import pandas as pd
    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    raw = (ROOT / a.data if not Path(a.data).is_absolute() else Path(a.data)) / f"case{a.grid}_goc" / "raw"
    stem = out_stem(a.grid, a.variant, a.seed, a.transform, a.tag)
    logfile = LOGS / f"{stem.name}.log"
    status = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "load_scenario_idx", "idx", "br_status"])
    always_out = (status.groupby("idx").br_status.max() == 0).sort_index().to_numpy()
    load_ids = np.sort(status.load_scenario_idx.unique())
    if a.load_ids:
        wanted = json.load(open(a.load_ids))[str(a.grid)] if a.load_ids.endswith(".json") else [int(x) for x in a.load_ids.split(",")]
        load_ids = load_ids[np.isin(load_ids, wanted)]
    if a.limit:
        load_ids = load_ids[: a.limit]
    label = "DEVELOPMENT" if "goc500/data" in str(raw) or "goc2000/data" in str(raw) or "goc10000/data" in str(raw) else "FRESH"
    log(f"=== run goc{a.grid} {model_name(a.variant, a.seed)} transform {a.transform} (spec seed {a.spec_seed}): {len(load_ids)} load scenarios, "
        f"{a.procs} process(es); data {raw} [{label} pool]; outputs {stem}_*.parquet", file=logfile)
    t0 = time.perf_counter()
    jobs = [(a.grid, str(raw), chunk, always_out, i, a.variant, a.seed, a.transform, a.spec_seed, str(logfile)) for i, chunk in enumerate(np.array_split(load_ids, a.procs))]
    if a.procs == 1:
        res = [worker(jobs[0])]
    else:
        with get_context("spawn").Pool(a.procs) as pool:
            res = pool.map(worker, jobs)
    lines = pd.concat([r[0] for r in res], ignore_index=True)
    st = pd.concat([r[1] for r in res], ignore_index=True)
    st["data"], st["pool"], st["model"] = str(raw), label, model_name(a.variant, a.seed)
    # SPEC ABORT, run level (the workers already raise per state): nothing is written unless every stored cell passed
    check_equivalence_cells(st, f"run {stem.name}")
    if not st.outage_map_ok.all():
        raise PipelineAbort(f"outage mapping check failed in {int((~st.outage_map_ok).sum())} state(s) of run {stem.name}")
    if a.transform == "none":  # identity control, before the parquet files are written: finite patterns first, then the maximum
        ident_frame = pd.DataFrame({f"{s}_{r}": (dgenco(lines, r) if s == "dgenco" else lines[f"{s}_{r}"]) for s in STAGES for r in ("o", "t")})
        check_identity_patterns(ident_frame, f"run {stem.name}")
        ident = identity_max(ident_frame)
        check_identity(ident)
        log(f"  identity control `none`: finite patterns match at every stage, max |t-o| {ident:.2e} points -> OK", file=logfile)
    lines.to_parquet(f"{stem}_lines.parquet", index=False)
    st.to_parquet(f"{stem}_states.parquet", index=False)
    eq = {c: float(st[c].max()) for c in st.columns if c.startswith("eq_")}
    log(f"done: {len(st)} states, {len(lines):,} line readings, wall {time.perf_counter() - t0:.0f} s; GENCO ms/state o {st.ms_genco_o.median():.0f} t {st.ms_genco_t.median():.0f}; "
        f"equivalence maxima {json.dumps({k: f'{v:.1e}' for k, v in eq.items()})}; outage map ok {int(st.outage_map_ok.sum())}/{len(st)}; "
        f"engine-vs-physics loading check max o {st.agg_check_o.max():.1e} t {st.agg_check_t.max():.1e}; split branch outaged in {int((st.split_outaged > 0).sum())} states", file=logfile)


def agg_max(S):
    import numpy as np
    v = S[["agg_check_o", "agg_check_t"]].to_numpy()
    return "n/a (+refvm)" if np.isnan(v).all() else f"{np.nanmax(v):.1e}"


def report(a):
    import numpy as np
    import pandas as pd
    specs = a.transforms.split(",") if a.transforms else sorted({p.name.rsplit("_lines.parquet")[0].split(f"seed{a.seed}_")[1].removesuffix(a.tag)
                                                                  for p in RESULTS.glob(f"goc{a.grid}_{a.variant}_seed{a.seed}_*{a.tag}_lines.parquet")})
    crit_rows, acc_rows = [], []
    pool_label = None
    for spec in specs:
        stem = out_stem(a.grid, a.variant, a.seed, spec, a.tag)
        if not Path(f"{stem}_lines.parquet").exists():
            print(f"missing {stem}_lines.parquet")
            continue
        L = pd.read_parquet(f"{stem}_lines.parquet")
        S = pd.read_parquet(f"{stem}_states.parquet").set_index("state")
        pool_label = S.pool.iloc[0]
        for r in ("o", "t"):
            L[f"dgenco_{r}"] = dgenco(L, r)
        real = (L.ac > LIM).to_numpy()
        crit = (L.ac > LIM).groupby(L.state).any()
        n_real, n_sc = int(real.sum()), len(crit)
        # SPEC ABORT at report time as well: no summary / criterion file is written if any stored state violates the rule.
        # Every cell is validated BEFORE the maxima below are taken (pandas maxima skip NaN).
        check_equivalence_cells(S, f"stored run {stem.name}")
        eq = {c.removeprefix("eq_"): float(S[c].max()) for c in S.columns if c.startswith("eq_")}
        check_equivalence(eq, f"stored run {stem.name}")
        if int(S.outage_map_ok.sum()) != len(S):
            raise PipelineAbort(f"outage mapping check false in {len(S) - int(S.outage_map_ok.sum())} stored state(s) of {stem.name}")
        if spec == "none":  # identity control: finite / non-finite patterns must match before any |t-o| is reduced
            check_identity_patterns(L, f"stored run {stem.name}")
        print(f"\n=== goc{a.grid} {model_name(a.variant, a.seed)}  transform {spec}  [{pool_label} pool: {S.data.iloc[0]}]")
        print(f"  {n_sc} states ({int(crit.sum())} with a real overload), {len(L):,} rated line readings, {n_real} overloaded; base states {S.base_key.nunique()}; "
              f"prev converged {int(S.conv_prev.sum())}/{n_sc}")
        print(f"  equivalence maxima: ybus {eq.get('c_ybus', np.nan):.1e} / S_spec {eq.get('c_S_spec', np.nan):.1e} / Vm_set {eq.get('c_Vm_set', np.nan):.1e} / "
              f"gen_p {eq.get('c_gen_p', np.nan):.1e} / rate_sum {eq.get('c_rate_sum', np.nan):.1e} / outaged_set {eq.get('c_outaged_set', np.nan):.0f} (contingency); "
              f"intact ybus {eq.get('i_ybus', np.nan):.1e}; outage mapping ok {int(S.outage_map_ok.sum())}/{n_sc}; split branch itself outaged in {int((S.split_outaged > 0).sum())} states; "
              f"engine-flow aggregation vs physics loading max {agg_max(S)}")
        q = lambda v: f"{np.nanmedian(v):.3g} / {np.nanquantile(v, .9):.3g} / {np.nanmax(v):.3g}"  # noqa: E731
        print(f"  raw voltage drift t-vs-o, max over buses, median / p90 / max over states: Vm outage {q(S.drift_vm_c)} p.u., Va outage {q(S.drift_va_c)} deg; "
              f"Vm intact {q(S.drift_vm_i)}, Va intact {q(S.drift_va_i)}; predicted CHANGE drift Vm {q(S.drift_dvm)}, Va {q(S.drift_dva)}")
        print(f"  REF bus: setpoint Vm {S.ref_vm_set.iloc[0]:.4f}; GENCO raw REF Vm error o {q((S.ref_vm_raw_o - S.ref_vm_set).abs())} t {q((S.ref_vm_raw_t - S.ref_vm_set).abs())}; "
              f"raw REF Va error (deg) o {q((S.ref_va_raw_o - S.ref_va_set).abs())} t {q((S.ref_va_raw_t - S.ref_va_set).abs())}")
        print(f"  Newton to convergence from the hybrid start: o conv {int(S.conv_hyb_o.sum())}/{n_sc} it mean {S.it_hyb_o[S.conv_hyb_o].mean():.2f}, "
              f"t conv {int(S.conv_hyb_t.sum())}/{n_sc} it mean {S.it_hyb_t[S.conv_hyb_t].mean():.2f}; prev it mean {S.it_prev[S.conv_prev].mean():.2f}; "
              f"from raw GENCO: o {S.it_g_o[S.conv_g_o].mean():.2f} t {S.it_g_t[S.conv_g_t].mean():.2f}")
        print(f"  mismatch MW (max-norm) median: raw GENCO o {S.mis_raw_o.median():.3g} t {S.mis_raw_t.median():.3g}; hybrid start o {S.mis0_hyb_o.median():.3g} t {S.mis0_hyb_t.median():.3g}; "
              f"after one step hyb o {S.mis1_hyb_o.replace(np.inf, np.nan).median():.3g} t {S.mis1_hyb_t.replace(np.inf, np.nan).median():.3g}, prev {S.mis1_prev.replace(np.inf, np.nan).median():.3g}")
        print(f"  {'predictor':<10} {'err median':>10} {'p99':>8} {'max':>8} {'NaN':>5} {'flagged':>8} {'secure flagged':>14} {'missed scen.':>13} {'missed lines':>13} {'share':>7}")
        preds = ["prev1"] + [f"{s}_{r}" for s in STAGES for r in ("o", "t")]
        for s in preds:
            e = (L[s] - L.ac).abs().to_numpy() * 100
            okm = np.isfinite(e)
            flag = (L[s].fillna(BIG) > LIM).groupby(L.state).any()
            missed = real & ~L.state.map(flag).to_numpy()
            k = int(missed.sum())
            rr = {"grid": a.grid, "model": model_name(a.variant, a.seed), "transform": spec, "pool": pool_label, "predictor": s, "states": n_sc,
                  "err_median_pp": float(np.median(e[okm])), "err_p99_pp": float(np.quantile(e[okm], .99)), "err_max_pp": float(e[okm].max()),
                  "nan_readings": int((~okm).sum()), "flagged": float(flag.mean()), "secure_flagged": float((flag & ~crit).sum() / max((~crit).sum(), 1)),
                  "critical": int(crit.sum()), "critical_missed": int((~flag & crit).sum()), "overloads": n_real, "overloads_missed": k, "missed_share": k / max(n_real, 1)}
            acc_rows.append(rr)
            print(f"  {s:<10} {rr['err_median_pp']:>10.4f} {rr['err_p99_pp']:>8.3f} {rr['err_max_pp']:>8.2f} {rr['nan_readings']:>5} {rr['flagged']:>8.1%} {rr['secure_flagged']:>14.1%} "
                  f"{rr['critical_missed']:>6}/{rr['critical']:<6} {k:>6}/{n_real:<6} {rr['missed_share']:>7.2%}")
        # ---- transformed vs original: per-state max |t - o| at each stage, decision agreement, cancellation ratio
        print(f"  {'t vs o':<10} {'state max|t-o| median':>22} {'p90':>8} {'max':>8} {'line median':>12} {'state flag agree':>17} {'line flag agree':>16}")
        comp = {}
        for s in STAGES:
            dd = (L[f"{s}_t"] - L[f"{s}_o"]).abs() * 100
            per_state = dd.groupby(L.state).max()
            ft, fo = (L[f"{s}_t"].fillna(BIG) > LIM).groupby(L.state).any(), (L[f"{s}_o"].fillna(BIG) > LIM).groupby(L.state).any()
            lt, lo = (L[f"{s}_t"].fillna(BIG) > LIM), (L[f"{s}_o"].fillna(BIG) > LIM)
            comp[s] = {"state_med": float(per_state.median()), "state_p90": float(per_state.quantile(.9)), "state_max": float(per_state.max()),
                       "state_mean": float(per_state.mean()), "line_med": float(dd.median()), "state_agree": float((ft == fo).mean()), "line_agree": float((lt == lo).mean())}
            c = comp[s]
            print(f"  {s:<10} {c['state_med']:>22.4f} {c['state_p90']:>8.4f} {c['state_max']:>8.3f} {c['line_med']:>12.5f} {c['state_agree']:>17.1%} {c['line_agree']:>16.1%}")
        raw_med, raw_mean = comp["genco"]["state_med"], comp["genco"]["state_mean"]
        rho = {s: (comp[s]["state_med"] / raw_med if raw_med > 0 else np.nan, comp[s]["state_mean"] / raw_mean if raw_mean > 0 else np.nan) for s in ("dgenco", "hyb0", "hyb1", "g1")}
        acc = {s: next(rr for rr in acc_rows if rr["transform"] == spec and rr["predictor"] == s) for s in ("hyb1_o", "hyb1_t", "prev1")}
        print(f"  cancellation ratio rho = median_state max|t-o| of stage / same for raw GENCO loading (mean-based in brackets): "
              + ", ".join(f"{s} {m:.3f} [{mm:.3f}]" for s, (m, mm) in rho.items()))
        verdict = {"transform": spec, "rho_hyb1_median": rho["hyb1"][0], "rho_hyb1_mean": rho["hyb1"][1], "rho_dgenco_median": rho["dgenco"][0], "rho_g1_median": rho["g1"][0],
                   "agree_hyb1": comp["hyb1"]["state_agree"], "agree_genco": comp["genco"]["state_agree"], "raw_state_med_pp": raw_med, "hyb1_state_med_pp": comp["hyb1"]["state_med"],
                   "hyb1_err_p99_o": acc["hyb1_o"]["err_p99_pp"], "hyb1_err_p99_t": acc["hyb1_t"]["err_p99_pp"],
                   "missed_hyb1_o": acc["hyb1_o"]["overloads_missed"], "missed_hyb1_t": acc["hyb1_t"]["overloads_missed"], "missed_prev1": acc["prev1"]["overloads_missed"]}
        allowed = int(np.ceil(a.miss_margin * n_real))
        rho_ok = (rho["hyb1"][0] <= a.x) and (rho["hyb1"][1] <= a.x)
        passed = rho_ok and (comp["hyb1"]["state_agree"] >= a.y) and (acc["hyb1_t"]["overloads_missed"] <= acc["hyb1_o"]["overloads_missed"] + allowed)
        attributed = rho["hyb0"][0] <= a.x0
        verdict.update({"cancels": bool(passed) if spec != "none" else None, "allowed_extra_misses": allowed, "rho_hyb0_median": rho["hyb0"][0],
                        "start_attribution": bool(attributed), "x": a.x, "y": a.y, "x0": a.x0, "miss_margin": a.miss_margin})
        crit_rows.append(verdict)
        if spec == "none":
            ident = identity_max(L)  # patterns were validated above, so every reading enters this maximum
            print(f"  IDENTITY CONTROL: finite patterns match; max |t-o| over all stages {ident:.2e} points -> {'OK' if ident < IDENTITY_TOL else 'PIPELINE BROKEN'}")
            check_identity(ident)  # SPEC ABORT: raises before the summary / criterion CSVs are written
        else:
            print(f"  CRITERION (X={a.x}, Y={a.y:.0%}, miss margin {a.miss_margin:.0%} of overloaded readings = {allowed}, X0={a.x0}): "
                  f"rho_hyb1 median {rho['hyb1'][0]:.3f} / mean {rho['hyb1'][1]:.3f} {'<=' if rho_ok else '>'} X; "
                  f"hyb1 state decisions agree {comp['hyb1']['state_agree']:.1%} {'>=' if comp['hyb1']['state_agree'] >= a.y else '<'} Y; "
                  f"missed overloaded readings hyb1 t {acc['hyb1_t']['overloads_missed']} vs o {acc['hyb1_o']['overloads_missed']} (+{allowed} allowed) "
                  f"-> {'CANCELS' if passed else 'DOES NOT CANCEL'}; attribution: rho_hyb0 {rho['hyb0'][0]:.3f} {'<=' if attributed else '>'} X0 -> "
                  f"{'anchored start removes most of the difference' if attributed else 'cancellation is mostly the Newton step'}")
    if acc_rows:
        out = RESULTS / f"summary_goc{a.grid}_{model_name(a.variant, a.seed)}{a.tag}.csv"
        pd.DataFrame(acc_rows).to_csv(out, index=False)
        pd.DataFrame(crit_rows).to_csv(RESULTS / f"criterion_goc{a.grid}_{model_name(a.variant, a.seed)}{a.tag}.csv", index=False)
        print(f"\n[{pool_label} pool] saved {out} and criterion_*.csv")
        if pool_label == "DEVELOPMENT":
            print("NOTE: these numbers come from the DEVELOPMENT pool (inspected data); they set the criterion, they do not test it.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["run", "report"])
    p.add_argument("--grid", default="500")
    p.add_argument("--data", default="work/goc500/data", help="data dir with case<N>_goc/raw (development: work/goc<N>/data; fresh: work/fresh/data<N>)")
    p.add_argument("--variant", default="tiny")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transform", default="none")
    p.add_argument("--transforms", default="", help="report: comma-separated transform list (default: every result file found)")
    p.add_argument("--spec-seed", type=int, default=2026)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--load-ids", default="", help="json file with {grid: [ids]} (e.g. work/fresh/fresh_ids.json) or a comma list; restricts the load scenarios")
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--x", type=float, default=0.05, help="criterion X: max cancellation ratio rho_hyb1, median- and mean-based (SPEC.md section 7)")
    p.add_argument("--y", type=float, default=0.98, help="criterion Y: min share of states whose hyb1 screening decision agrees (SPEC.md section 7)")
    p.add_argument("--x0", type=float, default=0.3, help="attribution X0: rho_hyb0 at or below this credits the cancellation to the anchored start")
    p.add_argument("--miss-margin", type=float, default=0.01, help="criterion: allowed extra missed overloaded readings of hyb1_t over hyb1_o, as a share of overloaded readings (ceil)")
    a = p.parse_args()
    try:
        {"run": run, "report": report}[a.phase](a)
    except (PipelineAbort, AssertionError) as e:
        # AssertionError: transforms.verify_equivalence / the rewrites assert physical identity in the worker; a failure there is the
        # same SPEC section 7 abort (it is re-raised in the parent by Pool.map) and gets the same exit code.
        kind = "" if isinstance(e, PipelineAbort) else " [physics assertion in transforms]"
        print(f"PIPELINE BROKEN - ABORT (SPEC.md section 7){kind}: {e}", file=sys.stderr, flush=True)
        sys.exit(2)

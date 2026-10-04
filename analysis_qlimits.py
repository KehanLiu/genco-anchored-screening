"""Do the screening findings survive generator reactive-power limits (PV -> PQ switching)?

Every AC power flow of this project so far lets a PV bus hold its voltage setpoint whatever reactive power that
takes. Here the AC power flow is solved WITH generator Q limits and the screening results are re-scored against
that new truth.

Q-limit power flow (class QLimNewton, wrapped around analysis_warmstart.Problem / the table-built problem of
work/goc2000/02_newton.py; the inner Newton is that class's unchanged `newton`):
    outer loop: solve; every PV bus whose required generator Q (bus injection + load Q) exceeds the summed Qmax
    or falls below the summed Qmin of its in-service generators becomes a PQ bus with Q fixed at that limit;
    re-solve; a bus fixed at Qmax whose voltage ends up ABOVE its setpoint (at Qmin: BELOW) is released back to
    PV (the generator can hold the setpoint again by moving away from the limit); repeat until no bus changes.
    All violating buses switch at once (MATPOWER enforce_q_lims=1 style). If that sequence repeats a bus-type
    pattern (cycling) or the inner Newton diverges, the solve is restarted switching one bus per outer iteration
    (largest Q excess first; releases only when no bus violates). Outer iterations are capped; failure is reported
    with its reason (newton / cycle / max_outer).
Choices:
    several generators at one bus  the bus is the unit: total required Q against the SUM of the limits of the
                                   in-service generators at the bus (all of them regulate the same bus voltage;
                                   equivalent to MATPOWER, which switches a bus when all its units are at a limit).
    reference bus                  stays the reference: it keeps its voltage magnitude and angle and its Q is not
                                   limited (pandapower's ext_grid convention). Scenarios where the reference
                                   bus's Q ends outside its generators' limits are counted, not "fixed".
    tolerances                     Q limit violation 1e-6 p.u. (1e-4 MVAr), release 1e-6 p.u. voltage,
                                   power mismatch 1e-8 p.u. (inner Newton, as before).

Phases
    run    --grid 500|2000   new truth + all predictors per GOC scenario  -> work/qlim/lines_goc<N>.parquet, states_goc<N>.parquet
    volt                     IEEE 118 voltage study with limits           -> work/qlim/voltage_case118_qlim.parquet (+ validation vs PowerModels)
    report                   tables -> work/qlim/report.log, work/qlim/summary.csv

    work/texas/tmo 5000 .venv/bin/python -u analysis_qlimits.py run --grid 500 --procs 3
    work/texas/tmo 3000 .venv/bin/python -u analysis_qlimits.py run --grid 2000 --procs 3
    work/texas/tmo 3000 .venv/bin/python -u analysis_qlimits.py volt
    .venv/bin/python analysis_qlimits.py report

Predictor columns (line loading, share of rating):
    ac        stored AC solution without Q limits (old truth)        ac_qlim   AC with Q limits (new truth)
    genco, dc raw GENCO, stored DC
    genco1/2, prev1/2, hyb1/2      plain Newton steps, no switching logic (every PV bus held at its setpoint)
    genco1s/2s, prev1s/2s, hyb1s/2s  steps with switching: bus types of the limit-enforced pre-outage state, the
                                   switching rule applied after each step
    "prev" always starts from the pre-outage state solved WITH limits; "hyb" = that state + GENCO's predicted change.
"""
import argparse
import copy
import hashlib
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "work" / "qlim"
BASE = 100.0
LIM = 1.001
BIG = 9.99
QTOL = 1e-6   # p.u.
VTOL = 1e-6   # p.u.
PLAIN = ["genco1", "genco2", "prev1", "prev2", "hyb1", "hyb2"]
SWITCH = ["genco1s", "genco2s", "prev1s", "prev2s", "hyb1s", "hyb2s"]


def stamp(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- Newton with Q limits
class QLimNewton:
    """AC power flow with generator reactive limits around a Problem (see module docstring).

    prob        analysis_warmstart.Problem-like object (Ybus, S, pv, pq, ref, vm_set, va_ref, newton, loading)
    qmin, qmax  per bus, p.u.: summed limits of the in-service generators at the bus (only PV entries are used)
    qd          per bus, p.u.: reactive load
    status      per bus: 0 regulating (or not a PV bus), +1 fixed at Qmax, -1 fixed at Qmin
    """

    def __init__(self, prob, qmin, qmax, qd):
        import numpy as np
        self.np = np
        self.p0 = prob
        self.pv0, self.pq0, self.ref = prob.pv.copy(), prob.pq.copy(), prob.ref
        self.qmin, self.qmax, self.qd = np.asarray(qmin, float), np.asarray(qmax, float), np.asarray(qd, float)
        self.vset = prob.vm_set
        self.n = len(self.vset)

    def typed(self, status):
        """Copy of the problem with the switched buses as PQ buses at their limit."""
        np = self.np
        p = copy.copy(self.p0)
        st = status[self.pv0]
        sw = self.pv0[st != 0]
        p.pv = self.pv0[st == 0]
        p.pq = np.r_[self.pq0, sw]
        p.pvpq = np.r_[p.pv, p.pq]
        S = self.p0.S.copy()
        S[sw] = S[sw].real + 1j * (np.where(status[sw] > 0, self.qmax[sw], self.qmin[sw]) - self.qd[sw])
        p.S = S
        return p

    def project(self, V, p):
        """Setpoint magnitudes at the regulating PV buses and the reference, reference angle."""
        np = self.np
        vm, va = np.abs(V), np.angle(V)
        vm[p.pv] = self.vset[p.pv]
        vm[self.ref] = self.vset[self.ref]
        va[self.ref] = self.p0.va_ref
        return vm * np.exp(1j * va)

    def qgen(self, V):
        """Generator Q each bus needs at V, p.u."""
        return (V * self.np.conj(self.p0.Ybus @ V)).imag + self.qd

    def check(self, V, status, one=False):
        """Switching rule. Returns the new status (with one=True at most one bus changes)."""
        np = self.np
        q, vm = self.qgen(V), np.abs(V)
        pv = self.pv0
        st = status[pv]
        over = np.where(st == 0, q[pv] - self.qmax[pv], -np.inf)
        under = np.where(st == 0, self.qmin[pv] - q[pv], -np.inf)
        rel = np.where(st > 0, vm[pv] - self.vset[pv], np.where(st < 0, self.vset[pv] - vm[pv], -np.inf))
        new = status.copy()
        if not one:
            new[pv[over > QTOL]] = 1
            new[pv[under > QTOL]] = -1
            new[pv[rel > VTOL]] = 0
            return new
        excess = np.maximum(over, under)
        if excess.max(initial=-np.inf) > QTOL:
            k = int(np.argmax(excess))
            new[pv[k]] = 1 if over[k] >= under[k] else -1
        elif rel.max(initial=-np.inf) > VTOL:
            new[pv[int(np.argmax(rel))]] = 0
        return new

    def _solve(self, V0, status0, one, max_outer):
        np = self.np
        status = np.zeros(self.n, np.int8) if status0 is None else status0.astype(np.int8).copy()
        seen = {status.tobytes()}
        V, inner = V0, 0
        for outer in range(1, max_outer + 1):
            p = self.typed(status)
            r = p.newton(self.project(V, p), keep=0)
            inner += r["iterations"]
            if not r["converged"]:
                return {"converged": False, "reason": "newton", "V": V, "status": status, "outer": outer, "inner": inner}
            V = r["V"]
            new = self.check(V, status, one)
            if (new == status).all():
                return {"converged": True, "reason": "", "V": V, "status": status, "outer": outer, "inner": inner, "prob": p}
            if new.tobytes() in seen:
                return {"converged": False, "reason": "cycle", "V": V, "status": status, "outer": outer, "inner": inner}
            seen.add(new.tobytes())
            status = new
        return {"converged": False, "reason": "max_outer", "V": V, "status": status, "outer": max_outer, "inner": inner}

    def solve(self, V0, status0=None):
        """All violators at once; on failure one bus at a time. Result carries `mode` and, if failed, `reason`."""
        r = self._solve(V0, status0, False, 40)
        r["mode"] = "all"
        if not r["converged"]:
            first = r["reason"]
            r = self._solve(V0, status0, True, 400)
            r["mode"] = "one"
            r["reason"] = r["reason"] and f"{first}/{r['reason']}"
        return r

    def invariants(self, r):
        """max power mismatch [MW], worst Q-limit excess at regulating PV buses [MVAr], worst wrong-side voltage at switched buses [p.u.]."""
        np = self.np
        V, status = r["V"], r["status"]
        mis = float(np.abs(r["prob"].mismatch(V)).max() * self.p0.base)
        q, vm, pv = self.qgen(V), np.abs(V), self.pv0
        st = status[pv]
        reg = pv[st == 0]
        qex = float(max(np.max(q[reg] - self.qmax[reg], initial=0.0), np.max(self.qmin[reg] - q[reg], initial=0.0)) * self.p0.base)
        side = np.where(st > 0, vm[pv] - self.vset[pv], np.where(st < 0, self.vset[pv] - vm[pv], 0.0))
        # fixed Q really is the limit
        sw = pv[st != 0]
        qfix = float(np.max(np.abs(q[sw] - np.where(status[sw] > 0, self.qmax[sw], self.qmin[sw])), initial=0.0) * self.p0.base)
        return mis, max(qex, qfix), float(np.max(side, initial=0.0))

    def steps(self, V0, status0, k=2):
        """k Newton steps with switching: types from status0, the switching rule applied after each step.
        Returns ([V after step 1..k] (None once a step broke down), status after the last check)."""
        status = status0.astype(self.np.int8).copy()
        V, out = V0, []
        for _ in range(k):
            if V is None:
                out.append(None)
                continue
            p = self.typed(status)
            r = p.newton(self.project(V, p), max_it=1, keep=1)
            if r["steps"]:
                V = r["steps"][0]
            elif r["converged"]:
                V = r["V"]
            else:
                V = None
                out.append(None)
                continue
            out.append(V)
            status = self.check(V, status)
        return out, status


def bus_limits(n, gen_bus, qmin, qmax, on):
    """Summed Q limits [same unit as given] of the in-service generators per bus."""
    import numpy as np
    b = np.asarray(gen_bus, int)[on]
    return np.bincount(b, weights=np.asarray(qmin, float)[on], minlength=n), np.bincount(b, weights=np.asarray(qmax, float)[on], minlength=n)


# ----------------------------------------------------------------------------- GOC scenarios
def worker(job):
    grid, load_ids, always_out, wid = job
    import engine  # Julia before torch
    import numpy as np
    import pandas as pd
    import torch
    import analysis_prev_control as pc
    torch.set_num_threads(1)
    here, net, raw = pc.paths(grid)
    engine.WORK = here
    genco = engine.Genco(net, "tiny")
    nw = pc.newton_module()
    lines, rows = [], []

    def qlim_of(prob, bus, gen):
        on = (gen.in_service == 1).to_numpy()
        lo, hi = bus_limits(len(bus), gen.bus, gen.min_q_mvar, gen.max_q_mvar, on)
        return QLimNewton(prob, lo / BASE, hi / BASE, bus.Qd.to_numpy(float) / BASE), lo, hi

    for sub in np.array_split(load_ids, max(len(load_ids) // 40, 1)):
        flt = [("load_scenario_idx", "in", [int(i) for i in sub])]
        tabs = {t: {s: d for s, d in pd.read_parquet(raw / f"{t}_data.parquet", filters=flt).groupby("scenario")} for t in ("bus", "gen", "branch")}
        groups = {}
        for s in sorted(tabs["bus"]):
            bus = tabs["bus"][s].sort_values("bus").reset_index(drop=True)
            gen = tabs["gen"][s].sort_values("idx").reset_index(drop=True)
            branch = tabs["branch"][s].sort_values("idx").reset_index(drop=True)
            q = gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index()  # GENCO's input columns, exactly as in the earlier runs
            bus = bus.merge(q, on="bus", how="left").fillna(0)
            ref = bus.REF.to_numpy() == 1
            h = hashlib.sha1()
            for arr in (branch.r, branch.x, branch.b, branch.tap, bus.Pd, bus.Qd, bus.PQ + 2 * bus.PV + 3 * bus.REF, bus.GS, bus.BS,
                        np.where(bus.PQ.to_numpy() == 1, 0.0, bus.Vm.to_numpy()), np.round(np.where(ref, 0.0, bus.Pg.to_numpy()), 6), gen.in_service,
                        gen.min_q_mvar, gen.max_q_mvar):
                h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
            groups.setdefault((int(bus.load_scenario_idx.iloc[0]), h.hexdigest()), []).append((s, bus, gen, branch))
        for (load_idx, key), members in groups.items():
            # ---- pre-outage state (removed branches back in service), without and with Q limits
            s0, bus0, gen0, branch0 = members[0]
            bi = branch0.copy()
            bi["br_status"] = np.where(always_out, 0, 1).astype(branch0.br_status.dtype)
            for k, v in pc.admittances(bi).items():
                bi[f"{k}_r"], bi[f"{k}_i"] = np.where(always_out, 0.0, v.real), np.where(always_out, 0.0, v.imag)
            prob_i = nw.make_problem(bus0, bi)
            ql_i, _, _ = qlim_of(prob_i, bus0, gen0)
            V_nl = None
            for (s, bus, gen, branch) in members:
                r = prob_i.newton(bus.Vm.to_numpy(float) * np.exp(1j * np.radians(bus.Va.to_numpy(float))))
                if r["converged"]:
                    V_nl = r["V"]
                    break
            if V_nl is None:
                r = prob_i.newton(prob_i.start())
                V_nl = r["V"] if r["converged"] else None
            v_first = bus0.Vm.to_numpy(float) * np.exp(1j * np.radians(bus0.Va.to_numpy(float)))
            pre = ql_i.solve(V_nl if V_nl is not None else v_first)
            ok = pre["converged"]
            bp_i, _ = genco.solve(bus0, gen0, bi)
            vm_gi, va_gi = bp_i.Vm.to_numpy(float), np.radians(bp_i.Va.to_numpy(float))
            if ok:
                V_p, st_p = pre["V"], pre["status"]
                vm_p, va_p = np.abs(V_p), np.angle(V_p)
                pre_inv = ql_i.invariants(pre)
            for (s, bus, gen, branch) in members:
                prob = nw.make_problem(bus, branch)
                ql, lo, hi = qlim_of(prob, bus, gen)
                live = (branch.br_status == 1).to_numpy()
                n = int(live.sum())
                br = branch[live]
                rate = br.rate_a.to_numpy()
                idx = br.idx.to_numpy()
                nanv = np.full(n, np.nan)
                vm_ac, va_ac = bus.Vm.to_numpy(float), np.radians(bus.Va.to_numpy(float))
                v_ac = vm_ac * np.exp(1j * va_ac)
                with np.errstate(all="ignore"):
                    s_ac = np.maximum(np.hypot(br.pf, br.qf), np.hypot(br.pt, br.qt)).to_numpy()
                pvb = prob.pv
                q_old = ql.qgen(v_ac) * BASE
                row = {"state": int(s), "load_idx": load_idx, "base_key": key[:12], "n_out": int((~live).sum()) - int(always_out.sum()),
                       "n_pv": len(pvb), "pre_ok": ok, "pre_mode": pre["mode"], "pre_reason": pre["reason"], "pre_outer": pre["outer"],
                       "pre_hi": int((pre["status"] > 0).sum()) if ok else -1, "pre_lo": int((pre["status"] < 0).sum()) if ok else -1,
                       "pre_mis_mw": pre_inv[0] if ok else np.nan,
                       "old_viol_buses": int(((q_old[pvb] > hi[pvb] + 1e-4) | (q_old[pvb] < lo[pvb] - 1e-4)).sum()),
                       "old_excess_mvar": float(max(np.max(q_old[pvb] - hi[pvb], initial=0.0), np.max(lo[pvb] - q_old[pvb], initial=0.0))),
                       "bad_limits": int((hi[pvb] < lo[pvb]).sum())}
                # ---- validation (a): limits disabled, from a DC-angle start, against the stored AC solution
                inf = np.full(len(bus), np.inf)
                r0 = QLimNewton(prob, -inf, inf, ql.qd).solve(prob.start(va=np.radians(bus.Va_dc.to_numpy(float))))
                row["nolim_conv"] = r0["converged"]
                row["nolim_dv"] = float(np.abs(r0["V"] - v_ac).max()) if r0["converged"] else np.nan
                row["nolim_switched"] = int((r0["status"] != 0).sum())
                # ---- new truth
                tr = ql.solve(v_ac)
                row["truth_src"] = "stored"
                alt = ql.solve(V_p, st_p) if ok else None   # same scenario reached from the pre-outage state and its bus types
                if not tr["converged"] and alt is not None and alt["converged"]:
                    tr, row["truth_src"] = alt, "pre-outage"
                conv = tr["converged"]
                row.update({"conv": conv, "mode": tr["mode"], "reason": tr["reason"], "outer": tr["outer"], "inner": tr["inner"]})
                la = lq = nanv
                if conv:
                    V_q, st_q = tr["V"], tr["status"]
                    mis, qex, side = ql.invariants(tr)
                    q_new = ql.qgen(V_q) * BASE
                    rb = prob.ref
                    vm_q = np.abs(V_q)
                    nonref = bus.REF.to_numpy() != 1
                    lo_v, hi_v = bus.min_vm_pu.to_numpy(), bus.max_vm_pu.to_numpy()
                    row.update({"n_hi": int((st_q > 0).sum()), "n_lo": int((st_q < 0).sum()), "mis_mw": mis, "q_excess_mvar": qex, "v_wrong_side": side,
                                "ref_q_out": bool(((q_new[rb] > hi[rb] + 1e-4) | (q_new[rb] < lo[rb] - 1e-4)).any()),
                                "ref_q_out_old": bool(((q_old[rb] > hi[rb] + 1e-4) | (q_old[rb] < lo[rb] - 1e-4)).any()),
                                "dvm_max": float(np.abs(vm_q - vm_ac).max()), "dva_max_deg": float(np.degrees(np.abs(np.angle(V_q) - va_ac)).max()),
                                "vmin_old": float(vm_ac.min()), "vmin_new": float(vm_q.min()),
                                "vviol_old": int((nonref & ((vm_ac < lo_v - 1e-4) | (vm_ac > hi_v + 1e-4))).sum()),
                                "vviol_new": int((nonref & ((vm_q < lo_v - 1e-4) | (vm_q > hi_v + 1e-4))).sum())})
                    if alt is not None and alt["converged"] and row["truth_src"] == "stored":
                        row["path_dv"] = float(np.abs(alt["V"] - V_q).max())
                        row["path_status_diff"] = int((alt["status"] != st_q).sum())
                    lq = prob.loading(V_q)
                # ---- GENCO
                bp_c, br_c = genco.solve(bus, gen, branch)
                vm_gc, va_gc = bp_c.Vm.to_numpy(float), np.radians(bp_c.Va.to_numpy(float))
                g_c = np.maximum(br_c.loading[:n].to_numpy(), br_c.loading[n:].to_numpy())
                starts = {"genco": prob.start(vm_pq=vm_gc, va=va_gc)}
                raw_start = {"genco": starts["genco"]}
                if ok:
                    starts["prev"] = prob.start(vm_pq=vm_p, va=va_p)
                    starts["hyb"] = prob.start(vm_pq=vm_p + (vm_gc - vm_gi), va=va_p + (va_gc - va_gi))
                    raw_start["prev"] = V_p
                    raw_start["hyb"] = (vm_p + (vm_gc - vm_gi)) * np.exp(1j * (va_p + (va_gc - va_gi)))
                cols = {}
                for name in ("genco", "prev", "hyb"):
                    if name not in starts:
                        for c in (f"{name}1", f"{name}2", f"{name}1s", f"{name}2s"):
                            cols[c] = nanv
                        continue
                    # (i) plain steps: every PV bus at its setpoint, no switching
                    r = prob.newton(starts[name], max_it=2)
                    for k in (1, 2):
                        cols[f"{name}{k}"] = prob.loading(r["steps"][k - 1]) if len(r["steps"]) >= k else (prob.loading(r["V"]) if r["converged"] else nanv)
                    # (ii) steps with switching, bus types of the limit-enforced pre-outage state
                    st0 = st_p if ok else np.zeros(len(bus), np.int8)
                    t0 = time.perf_counter()
                    vs, st_end = ql.steps(raw_start[name], st0, 2)
                    row[f"ms_steps_{name}"] = 1e3 * (time.perf_counter() - t0)
                    for k in (1, 2):
                        cols[f"{name}{k}s"] = prob.loading(vs[k - 1]) if vs[k - 1] is not None else nanv
                    if conv:
                        row[f"types_wrong_{name}"] = int((st_end != st_q).sum())   # after the check that follows step 2
                        row[f"types_wrong0_{name}"] = int((st0 != st_q).sum())    # pre-outage types vs true types
                d = pd.DataFrame({"state": int(s), "line": idx, "ac": s_ac / rate, "ac_qlim": lq, "genco": g_c,
                                  "dc": np.abs(br.pf_dc.to_numpy()) / rate, **cols})[rate > 0]
                lines.append(d.astype({c: "float32" for c in d.columns if c not in ("state", "line")}))
                rows.append(row)
        if wid == 0:
            stamp(f"  worker 0: {len(rows)} scenarios done")
    return pd.concat(lines, ignore_index=True), pd.DataFrame(rows)


def run(a):
    import numpy as np
    import pandas as pd
    import analysis_prev_control as pc
    OUT.mkdir(parents=True, exist_ok=True)
    here, net, raw = pc.paths(a.grid)
    status = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "load_scenario_idx", "idx", "br_status"])
    always_out = (status.groupby("idx").br_status.max() == 0).sort_index().to_numpy()
    load_ids = np.sort(status.load_scenario_idx.unique())
    if a.limit:
        load_ids = np.sort(np.random.default_rng(0).choice(load_ids, a.limit, replace=False))
    stamp(f"=== run {net}: {len(load_ids):,} load scenarios, {a.procs} processes")
    del status
    t0 = time.perf_counter()
    with get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(worker, [(a.grid, chunk, always_out, i) for i, chunk in enumerate(np.array_split(load_ids, a.procs))])
    lines = pd.concat([r[0] for r in res], ignore_index=True)
    st = pd.concat([r[1] for r in res], ignore_index=True)
    lines.to_parquet(OUT / f"lines_goc{a.grid}{a.tag}.parquet", index=False)
    st.to_parquet(OUT / f"states_goc{a.grid}{a.tag}.parquet", index=False)
    stamp(f"{net}: {len(st):,} scenarios, {len(lines):,} line readings, wall {time.perf_counter() - t0:.0f} s; limit-enforced truth converged "
          f"{int(st.conv.sum()):,}/{len(st):,}; pre-outage converged {int(st.pre_ok.sum()):,}")


# ----------------------------------------------------------------------------- IEEE 118 voltage study
def volt(a):
    import analysis_warmstart as ws  # imports engine (Julia) first
    import engine
    import numpy as np
    import pandas as pd
    from analysis_residual import cases
    from gridfm_datakit.process.process_network import _solution_arrays
    from gridfm_datakit.process.solvers import run_pf
    from gridfm_datakit.utils.idx_bus import PD, QD, VMAX, VMIN
    from gridfm_datakit.utils.idx_gen import GEN_BUS, GEN_STATUS, PG, QMAX, QMIN
    OUT.mkdir(parents=True, exist_ok=True)
    eng = ws.build("118")
    vmin, vmax = eng.base.buses[:, VMIN].copy(), eng.base.buses[:, VMAX].copy()

    def qlim_of(net):
        prob = ws.Problem(net)
        on = net.gens[:, GEN_STATUS] == 1
        lo, hi = bus_limits(len(net.buses), net.gens[:, GEN_BUS], net.gens[:, QMIN], net.gens[:, QMAX], on)
        return prob, QLimNewton(prob, lo / prob.base, hi / prob.base, net.buses[:, QD] / prob.base), lo, hi

    def pm_reference(net, status):
        """PowerModels compute_ac_pf on the final bus types: generators of switched buses become a negative load at their limit."""
        net2 = net.copy_for_perturbation()
        on = net2.idx_gens_in_service
        off = []
        for b in np.flatnonzero(status != 0):
            g = on[net2.gens[on, GEN_BUS].astype(int) == b]
            net2.buses[b, PD] -= net2.gens[g, PG].sum()
            net2.buses[b, QD] -= net2.gens[g, QMAX if status[b] > 0 else QMIN].sum()
            off += list(g)
        if off:
            net2.deactivate_gens(np.array(off))
        ac = run_pf(net2, eng.jl, fast=True)
        _, _, ac_bus = _solution_arrays(ac, net2)
        return ac_bus[:, 0] * np.exp(1j * ac_bus[:, 1])

    pre = {}   # load scale -> status of the limit-enforced N-0 state

    def pre_status(scale):
        key = round(scale, 4)
        if key not in pre:
            net0 = eng._dispatch(scale)[0].copy_for_perturbation()
            prob0, ql0, _, _ = qlim_of(net0)
            r = ql0.solve(prob0.start(vm_pq=net0.buses[:, 7], va=np.radians(net0.buses[:, 8])))
            pre[key] = (r["status"] if r["converged"] else np.zeros(len(net0.buses), np.int8), r["converged"], r["V"])
        return pre[key]

    frames, rows, checks = [], [], []
    todo = list(enumerate(cases(eng, 300)))
    if a.limit:
        todo = todo[:: max(len(todo) // a.limit, 1)]
    t0 = time.perf_counter()
    for state, (scale, outages) in todo:
        try:
            result = eng.solve(scale, outages)
        except engine.SolveError:
            continue
        net = eng._dispatch(scale)[0].copy_for_perturbation()
        if outages:
            net.deactivate_branches(sorted(set(int(o) for o in outages)))
        prob, ql, lo, hi = qlim_of(net)
        b = pd.DataFrame(result["buses"])
        br = pd.DataFrame(result["branches"])
        v_ac = b.vm_ac.to_numpy(float) * np.exp(1j * np.radians(b.va_ac.to_numpy(float)))
        vm_g, va_g = b.vm_genco.to_numpy(float), np.radians(b.va_genco.to_numpy(float))
        n = len(b)
        inf = np.full(n, np.inf)
        r0 = QLimNewton(prob, -inf, inf, ql.qd).solve(prob.start())
        tr = ql.solve(v_ac)
        st_p, pre_ok, V_p = pre_status(scale)
        if not tr["converged"] and pre_ok:
            alt = ql.solve(V_p, st_p)
            tr = alt if alt["converged"] else tr
        q_old = ql.qgen(v_ac) * prob.base
        pvb = prob.pv
        row = {"state": state, "load_scale": scale, "outages": "+".join(map(str, outages)), "conv": tr["converged"], "mode": tr["mode"], "reason": tr["reason"],
               "outer": tr["outer"], "nolim_dv": float(np.abs(r0["V"] - v_ac).max()) if r0["converged"] else np.nan, "nolim_switched": int((r0["status"] != 0).sum()),
               "mis_ac_mw": float(np.abs(prob.mismatch(v_ac)).max() * prob.base), "pre_switched": int((st_p != 0).sum()),
               "old_viol_buses": int(((q_old[pvb] > hi[pvb] + 1e-4) | (q_old[pvb] < lo[pvb] - 1e-4)).sum())}
        nan = np.full(n, np.nan)
        vm_q = vm_1s = vm_2s = vm_1 = nan
        status = np.zeros(n, np.int8)
        over_q = -1
        # GENCO + Newton steps: (ii) with switching from the N-0 types, (i) plain
        vs, _ = ql.steps(prob.start(vm_pq=vm_g, va=va_g), st_p, 2)
        vm_1s = np.abs(vs[0]) if vs[0] is not None else nan
        vm_2s = np.abs(vs[1]) if vs[1] is not None else nan
        rp = prob.newton(prob.start(vm_pq=vm_g, va=va_g), max_it=1)
        vm_1 = np.abs(rp["steps"][0]) if rp["steps"] else (np.abs(rp["V"]) if rp["converged"] else nan)
        if tr["converged"]:
            V_q, status = tr["V"], tr["status"]
            vm_q = np.abs(V_q)
            mis, qex, side = ql.invariants(tr)
            lq = prob.loading(V_q)
            over_q = int((lq > LIM).sum())
            rb = prob.ref
            q_new = ql.qgen(V_q) * prob.base
            row.update({"n_hi": int((status > 0).sum()), "n_lo": int((status < 0).sum()), "mis_mw": mis, "q_excess_mvar": qex, "v_wrong_side": side,
                        "ref_q_out": bool(((q_new[rb] > hi[rb] + 1e-4) | (q_new[rb] < lo[rb] - 1e-4)).any()),
                        "dload_max_pp": float(100 * np.nanmax(np.abs(lq - prob.loading(v_ac)))),
                        "over_ac_own": int((prob.loading(v_ac) > LIM).sum()), "over_q": over_q})
            if (status != 0).any() and len(checks) < a.pm_checks and state % 7 == 0:
                try:
                    v_pm = pm_reference(net, status)
                    checks.append({"state": state, "switched": int((status != 0).sum()), "dv": float(np.abs(v_pm - V_q).max()),
                                   "dload_pp": float(100 * np.nanmax(np.abs(prob.loading(v_pm) - lq)))})
                except RuntimeError as err:
                    checks.append({"state": state, "switched": int((status != 0).sum()), "dv": np.nan, "dload_pp": np.nan, "error": str(err)[:80]})
        lines = br[br.in_service & br.loading_ac.notna()]
        frames.append(pd.DataFrame({
            "state": state, "load_scale": scale, "outages": row["outages"], "bus": b.id.astype("int32"), "type": b.type,
            "vm_ac": b.vm_ac.astype(float), "vm_qlim": vm_q, "status": status, "vm_genco": vm_g, "vm_genco1": vm_1, "vm_genco1s": vm_1s, "vm_genco2s": vm_2s,
            "vmin": vmin, "vmax": vmax, "over_ac": int((lines.loading_ac > LIM).sum()), "over_dc": int((lines.loading_dc > LIM).sum()), "over_qlim": over_q,
            "conv": tr["converged"]}))
        rows.append(row)
        if len(rows) % 200 == 0:
            stamp(f"  {len(rows)} states, {time.perf_counter() - t0:.0f} s")
    pd.concat(frames, ignore_index=True).to_parquet(OUT / f"voltage_case118_qlim{a.tag}.parquet", index=False)
    pd.DataFrame(rows).to_parquet(OUT / f"states_case118{a.tag}.parquet", index=False)
    pd.DataFrame(checks).to_csv(OUT / f"pm_check_case118{a.tag}.csv", index=False)
    stamp(f"IEEE 118: {len(rows)} states, wall {time.perf_counter() - t0:.0f} s, {len(checks)} PowerModels checks")


# ----------------------------------------------------------------------------- report
def report(a):
    import numpy as np
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"report{a.tag}.log", "w")

    def log(*x):
        line = " ".join(str(v) for v in x)
        print(line, flush=True)
        logf.write(line + "\n")

    out = []

    def table(full, truth, preds, net, label, variant_of):
        t = full[truth]
        real = (t > LIM).to_numpy()
        crit = (t > LIM).groupby(full.state).any()
        n_real = int(real.sum())
        log(f"  [{label}] {len(crit):,} scenarios, {int(crit.sum()):,} with an overload, {n_real:,} overloaded line readings")
        log(f"  {'predictor':<9} {'err median':>10} {'p99':>8} {'max':>9} {'NaN':>6} {'flagged':>8} {'secure flagged':>14} {'missed scen.':>13} {'missed lines':>15} {'share':>7} {'worst':>7}")
        for s in preds:
            e = (full[s] - t).abs().to_numpy() * 100
            okm = np.isfinite(e)
            flag = (full[s].fillna(BIG) > LIM).groupby(full.state).any()
            missed = real & ~full.state.map(flag).to_numpy()
            k = int(missed.sum())
            r = {"network": net, "truth": label, "variant": variant_of(s), "predictor": s, "scenarios": len(crit), "err_median_pp": float(np.median(e[okm])),
                 "err_p99_pp": float(np.quantile(e[okm], .99)), "err_max_pp": float(e[okm].max()), "nan_readings": int((~okm).sum()), "flagged": float(flag.mean()),
                 "secure_flagged": float((flag & ~crit).sum() / max((~crit).sum(), 1)), "critical_scenarios": int(crit.sum()),
                 "critical_missed": int((~flag & crit).sum()), "overloads": n_real, "overloads_missed": k, "missed_share": k / max(n_real, 1),
                 "worst_missed_loading": float(t.to_numpy()[missed].max()) if k else np.nan}
            out.append(r)
            worst = f"{r['worst_missed_loading']:.1%}" if k else ""
            log(f"  {s:<9} {r['err_median_pp']:>10.4f} {r['err_p99_pp']:>8.3f} {r['err_max_pp']:>9.2f} {r['nan_readings']:>6} {r['flagged']:>8.1%} {r['secure_flagged']:>14.1%} "
                f"{r['critical_missed']:>6}/{r['critical_scenarios']:<6} {k:>7}/{n_real:<7} {r['missed_share']:>7.2%} {worst:>7}")

    variant = lambda s: "raw" if s in ("genco", "dc") else ("switching" if s.endswith("s") else "plain")  # noqa: E731
    for grid in ("500", "2000"):
        f = OUT / f"lines_goc{grid}{a.tag}.parquet"
        if not f.exists():
            continue
        net = f"case{grid}_goc"
        here = ROOT / "work" / f"goc{grid}"
        full = pd.read_parquet(f)
        st = pd.read_parquet(OUT / f"states_goc{grid}{a.tag}.parquet").set_index("state")
        log(f"\n{'=' * 150}\n{net}: {len(st):,} scenarios, {len(full):,} rated line readings")
        # ---- validation
        log(f"validation (a) limits disabled, DC-angle start vs stored AC solution: converged {int(st.nolim_conv.sum()):,}/{len(st):,}, max |V - V_stored| {st.nolim_dv.max():.1e} p.u. "
            f"({int((st.nolim_dv > 1e-5).sum())} scenarios > 1e-5), buses switched {int(st.nolim_switched.sum())}")
        c = st[st.conv]
        log(f"validation (b) invariants of the limit-enforced solutions ({len(c):,} converged): power mismatch max {c.mis_mw.max():.1e} MW; worst Q outside a limit at a regulating "
            f"PV bus / fixed Q off its limit {c.q_excess_mvar.max():.1e} MVAr; worst voltage on the wrong side of the setpoint at a switched bus {c.v_wrong_side.max():.1e} p.u.; "
            f"PV buses with Qmin > Qmax: {int(st.bad_limits.max())}")
        p = c[c.path_dv.notna()] if "path_dv" in c else c.iloc[:0]
        if len(p):
            log(f"  same scenario solved from the pre-outage state and its bus types ({len(p):,} scenarios): identical switched set in {int((p.path_status_diff == 0).sum()):,} "
                f"({(p.path_status_diff == 0).mean():.2%}); max |dV| {p.path_dv.max():.1e} p.u., scenarios with |dV| > 1e-5: {int((p.path_dv > 1e-5).sum())}")
        log(f"  solve mode: all-at-once {int((c['mode'] == 'all').sum()):,}, one-at-a-time fallback {int((c['mode'] == 'one').sum()):,}; outer iterations median {c.outer.median():.0f} "
            f"max {c.outer.max()}; inner Newton iterations median {c.inner.median():.0f}; truth taken from the pre-outage path: {int((c.truth_src != 'stored').sum())}")
        # ---- how much the truth changes
        nc = st[~st.conv]
        log(f"stored (no-limit) solutions: {int((st.old_viol_buses > 0).sum()):,} of {len(st):,} scenarios ({(st.old_viol_buses > 0).mean():.1%}) have a PV bus outside its Q limits; "
            f"buses per scenario median {st.old_viol_buses.median():.0f} of {st.n_pv.median():.0f} PV buses, max {st.old_viol_buses.max()}; worst excess {st.old_excess_mvar.max():.0f} MVAr")
        log(f"pre-outage states with limits: converged for {int(st.pre_ok.sum()):,}/{len(st):,} scenarios; switched buses median {st[st.pre_ok].pre_hi.add(st[st.pre_ok].pre_lo).median():.0f} "
            f"(at Qmax {st[st.pre_ok].pre_hi.median():.0f}, at Qmin {st[st.pre_ok].pre_lo.median():.0f})")
        log(f"limit-enforced truth: NOT converged {len(nc):,} of {len(st):,} ({len(nc) / len(st):.2%}) reasons {nc.reason.value_counts().to_dict() if len(nc) else {}}")
        nsw = c.n_hi + c.n_lo
        log(f"  scenarios with at least one PV->PQ switch: {int((nsw > 0).sum()):,} of {len(c):,} ({(nsw > 0).mean():.1%}); switched buses per scenario median {nsw.median():.0f} "
            f"mean {nsw.mean():.1f} max {nsw.max()} (at Qmax mean {c.n_hi.mean():.1f}, at Qmin mean {c.n_lo.mean():.1f}); share of PV buses switched {nsw.sum() / c.n_pv.sum():.1%}")
        log(f"  reference bus Q outside its limits: {int(c.ref_q_out.sum()):,} scenarios with limits ({int(c.ref_q_out_old.sum()):,} without) -- not enforced")
        log(f"  voltage change: max |dVm| per scenario median {c.dvm_max.median():.4f} p99 {c.dvm_max.quantile(.99):.4f} max {c.dvm_max.max():.4f} p.u.; "
            f"lowest Vm: median {c.vmin_old.median():.4f} -> {c.vmin_new.median():.4f}, min {c.vmin_old.min():.4f} -> {c.vmin_new.min():.4f}; "
            f"scenarios with a bus outside its voltage limits: {int((c.vviol_old > 0).sum()):,} -> {int((c.vviol_new > 0).sum()):,} (bus-states {int(c.vviol_old.sum()):,} -> {int(c.vviol_new.sum()):,})")
        full = full[full.state.isin(c.index)].reset_index(drop=True)
        d = (full.ac_qlim - full.ac).abs() * 100
        per = d.groupby(full.state).max()
        o_old, o_new = (full.ac > LIM), (full.ac_qlim > LIM)
        s_old, s_new = o_old.groupby(full.state).any(), o_new.groupby(full.state).any()
        log(f"  line loading change |ac_qlim - ac|, points of rating: median {d.median():.4f} / p99 {d.quantile(.99):.3f} / max {d.max():.2f}; worst line per scenario median "
            f"{per.median():.3f} / p99 {per.quantile(.99):.2f}; readings changing by more than 1 point: {(d > 1).mean():.2%}, more than 5: {(d > 5).mean():.3%}")
        log(f"  overloaded line readings: {int(o_old.sum()):,} without limits -> {int(o_new.sum()):,} with limits; appear {int((o_new & ~o_old).sum()):,}, disappear {int((o_old & ~o_new).sum()):,}; "
            f"scenarios with an overload: {int(s_old.sum()):,} -> {int(s_new.sum()):,} (appear {int((s_new & ~s_old).sum())}, disappear {int((s_old & ~s_new).sum())})")
        for nm in ("genco", "prev", "hyb"):
            if f"types_wrong_{nm}" in c:
                log(f"  bus types after 2 switching steps from {nm:<5}: scenarios with exactly the true switched set {(c[f'types_wrong_{nm}'] == 0).mean():.1%} "
                    f"(pre-outage types already right: {(c[f'types_wrong0_{nm}'] == 0).mean():.1%}); wrong buses mean {c[f'types_wrong_{nm}'].mean():.2f} max {c[f'types_wrong_{nm}'].max():.0f}; "
                    f"2 switching steps take {c[f'ms_steps_{nm}'].median():.1f} ms median")
        # ---- screening: old numbers on the same scenarios, then against the new truth
        old = pd.read_parquet(here / "lines_newton_tiny_seed0.parquet", columns=["state", "line", "genco", "genco1", "genco2", "dc"]).rename(
            columns={k: f"o_{k}" for k in ("genco", "genco1", "genco2", "dc")})
        prev = pd.read_parquet(here / "lines_prev_tiny_seed0.parquet", columns=["state", "line", "prev1", "prev2"]).rename(columns={"prev1": "o_prev1", "prev2": "o_prev2"})
        hyb = pd.read_parquet(here / "lines_hybrid_tiny_seed0.parquet", columns=["state", "line", "hyb1", "hyb2"]).rename(columns={"hyb1": "o_hyb1", "hyb2": "o_hyb2"})
        m = full.merge(old, on=["state", "line"]).merge(prev, on=["state", "line"]).merge(hyb, on=["state", "line"])
        assert len(m) == len(full)
        log(f"  re-run vs earlier files: raw GENCO loading max diff {float((m.genco - m.o_genco).abs().max()):.1e}, GENCO + 1 plain step max diff {float((m.genco1 - m.o_genco1).abs().max()):.1e}")
        log("\nA. BEFORE: earlier predictors (no limits anywhere, pre-outage state without limits) against the no-limit truth, same scenarios")
        mo = m[["state", "ac"] + [c_ for c_ in m.columns if c_.startswith("o_")]].rename(columns=lambda c_: c_[2:] if c_.startswith("o_") else c_)
        table(mo, "ac", ["genco", "dc", "genco1", "genco2", "prev1", "prev2", "hyb1", "hyb2"], net, "no limits", variant)
        log("\nB. AFTER: against the limit-enforced truth. raw predictors and (i) plain Newton steps (no switching; pre-outage start solved with limits)")
        table(full, "ac_qlim", ["genco", "dc"] + PLAIN, net, "Q limits", variant)
        log("\nC. AFTER: against the limit-enforced truth. (ii) Newton steps with switching (bus types of the limit-enforced pre-outage state, rule applied after each step)")
        table(full, "ac_qlim", SWITCH, net, "Q limits", variant)
        log("\nD. reference: the no-limit AC solution itself used as a predictor of the limit-enforced truth (the best any limit-blind method can do)")
        table(full, "ac_qlim", ["ac"], net, "Q limits", lambda s: "no-limit AC")

    f = OUT / f"voltage_case118_qlim{a.tag}.parquet"
    if f.exists():
        data = pd.read_parquet(f)
        st = pd.read_parquet(OUT / f"states_case118{a.tag}.parquet")
        TOL = 1e-4
        vio = lambda vm, lo, hi: (vm < lo - TOL) | (vm > hi + TOL)  # noqa: E731
        log(f"\n{'=' * 150}\nIEEE 118 voltage study: {data.state.nunique():,} states")
        oldf = ROOT / "work" / "voltage_case118.parquet"
        if oldf.exists() and not a.tag:
            o = pd.read_parquet(oldf, columns=["state", "bus", "vm_ac", "vm_genco"]).merge(data[["state", "bus", "vm_ac", "vm_genco"]], on=["state", "bus"], suffixes=("_o", ""))
            log(f"rebuilt states vs work/voltage_case118.parquet: {o.state.nunique():,} common states, max |vm_ac diff| {(o.vm_ac - o.vm_ac_o).abs().max():.1e}, max |vm_genco diff| {(o.vm_genco - o.vm_genco_o).abs().max():.1e}")
        log(f"validation (a) limits disabled, flat start vs engine AC (PowerModels): max |V diff| {st.nolim_dv.max():.1e} p.u., buses switched {int(st.nolim_switched.sum())}; "
            f"engine AC solution in this script's equations: mismatch max {st.mis_ac_mw.max():.1e} MW")
        c = st[st.conv]
        log(f"validation (b) invariants ({len(c):,} converged): mismatch max {c.mis_mw.max():.1e} MW, Q-limit excess max {c.q_excess_mvar.max():.1e} MVAr, wrong-side voltage max {c.v_wrong_side.max():.1e} p.u.")
        pmf = OUT / f"pm_check_case118{a.tag}.csv"
        if pmf.exists() and pmf.stat().st_size > 5:
            pm = pd.read_csv(pmf)
            good = pm[pm.dv.notna()]
            log(f"validation (b) independent solver: PowerModels compute_ac_pf on the final bus types (switched generators as fixed injections at their limit), {len(good)} states "
                f"with {good.switched.min()}-{good.switched.max()} switched buses ({len(pm) - len(good)} PowerModels failures): max |V diff| {good.dv.max():.1e} p.u., max loading diff {good.dload_pp.max():.1e} points. "
                f"This checks the solution for a given switched set; the switching decisions themselves are checked by the invariants only (no pandapower / MATPOWER installed).")
        nc = st[~st.conv]
        nsw = c.n_hi + c.n_lo
        log(f"no-limit solutions with a PV bus outside its Q limits: {int((st.old_viol_buses > 0).sum()):,} of {len(st):,} states; N-0 states with switched buses: {int((st.pre_switched > 0).sum())} states")
        log(f"limit-enforced truth: NOT converged {len(nc)} of {len(st)} {nc.reason.value_counts().to_dict() if len(nc) else ''}; states with a switch {int((nsw > 0).sum()):,} ({(nsw > 0).mean():.1%}), "
            f"switched buses median {nsw.median():.0f} mean {nsw.mean():.1f} max {nsw.max()} (Qmax {int(c.n_hi.sum()):,}, Qmin {int(c.n_lo.sum()):,} bus-states); one-at-a-time fallback {int((c['mode'] == 'one').sum())}; "
            f"reference bus Q outside limits {int(c.ref_q_out.sum())} states")
        log(f"  line loading change: worst line per state median {c.dload_max_pp.median():.3f} / p99 {c.dload_max_pp.quantile(.99):.2f} / max {c.dload_max_pp.max():.2f} points; "
            f"states with an overload {int((c.over_ac_own > 0).sum())} -> {int((c.over_q > 0).sum())}; overloaded lines {int(c.over_ac_own.sum())} -> {int(c.over_q.sum())}")
        d = data[data.conv].copy()
        n_states = d.state.nunique()
        pqm = d.type == "PQ"
        swm = (d.type == "PV") & (d.status != 0)
        d["real_old"] = vio(d.vm_ac, d.vmin, d.vmax) & pqm
        d["real_new"] = vio(d.vm_qlim, d.vmin, d.vmax) & (d.type != "REF")
        dc_silent = d.groupby("state").over_dc.first() == 0

        def states_of(mask):
            return pd.Index(d.state[mask].unique())

        so, sn_pq, sn_sw, sn = states_of(d.real_old), states_of(d.real_new & pqm), states_of(d.real_new & swm), states_of(d.real_new)
        log(f"\nvoltage-limit violations (case limits {d.vmin.min()}-{d.vmax.max()}, tolerance 1e-4), {n_states} states with a limit-enforced solution:")
        log(f"  without Q limits (PQ buses; PV buses sit on their setpoints): {int(d.real_old.sum()):,} bus-states in {len(so):,} states ({len(so) / n_states:.1%}); low {int((d.real_old & (d.vm_ac < d.vmin)).sum())}, high {int((d.real_old & (d.vm_ac > d.vmax)).sum())}; "
            f"states with no DC overload: {int(dc_silent.reindex(so).sum())}")
        log(f"  with Q limits, PQ buses:          {int((d.real_new & pqm).sum()):,} bus-states in {len(sn_pq):,} states; low {int((d.real_new & pqm & (d.vm_qlim < d.vmin)).sum())}, high {int((d.real_new & pqm & (d.vm_qlim > d.vmax)).sum())}")
        log(f"  with Q limits, switched PV buses: {int((d.real_new & swm).sum()):,} bus-states in {len(sn_sw):,} states (of {int(swm.sum()):,} switched bus-states); low {int((d.real_new & swm & (d.vm_qlim < d.vmin)).sum())}, high {int((d.real_new & swm & (d.vm_qlim > d.vmax)).sum())}")
        log(f"  with Q limits, all buses:         {int(d.real_new.sum()):,} bus-states in {len(sn):,} states ({len(sn) / n_states:.1%}); states with no DC overload: {int(dc_silent.reindex(sn).sum())}; "
            f"states violating only with limits {len(sn.difference(so))}, only without {len(so.difference(sn))}")
        dv = (d.vm_qlim - d.vm_ac).abs()
        log(f"  |Vm change| from the limits: PQ buses median {dv[pqm].median():.5f} p99 {dv[pqm].quantile(.99):.5f} max {dv[pqm].max():.5f}; switched PV buses median {dv[swm].median():.5f} max {dv[swm].max():.5f} p.u.")
        log(f"\n  {'predictor':<34} {'buses scored':<14} {'real':>6} {'called':>7} {'hit':>6} {'missed':>7} {'false+':>7} {'recall':>7} {'precis.':>8} {'states real':>11} {'st. unflagged':>13} {'|Vm err| p99':>12} {'max':>8}")

        def score(name, col, truth_col, mask, label):
            real = d[truth_col] & mask if truth_col == "real_new" else d.real_old
            truth_vm = d.vm_qlim if truth_col == "real_new" else d.vm_ac
            called = vio(d[col], d.vmin, d.vmax) & mask
            hit = real & called
            s_real, s_called = set(d.state[real]), set(d.state[called])
            err = (d[col] - truth_vm).abs()[mask]
            r = {"network": "case118_ieee", "truth": "Q limits" if truth_col == "real_new" else "no limits", "variant": "voltage", "predictor": name, "buses": label,
                 "real": int(real.sum()), "called": int(called.sum()), "missed": int((real & ~called).sum()), "false_alarms": int((called & ~real).sum()),
                 "recall": hit.sum() / max(real.sum(), 1), "precision": hit.sum() / max(called.sum(), 1), "states_real": len(s_real), "states_unflagged": len(s_real - s_called),
                 "vm_err_p99": float(err.quantile(.99)), "vm_err_max": float(err.max())}
            out.append(r)
            log(f"  {name:<34} {label:<14} {r['real']:>6,} {r['called']:>7,} {int(hit.sum()):>6,} {r['missed']:>7,} {r['false_alarms']:>7,} {r['recall']:>7.1%} {r['precision']:>8.1%} "
                f"{r['states_real']:>11} {r['states_unflagged']:>13} {r['vm_err_p99']:>12.5f} {r['vm_err_max']:>8.5f}")

        allm = d.type != "REF"
        score("GENCO vs no-limit truth (before)", "vm_genco", "real_old", pqm, "PQ")
        for name, col in (("GENCO", "vm_genco"), ("GENCO + 1 plain step", "vm_genco1"), ("GENCO + 1 step with switching", "vm_genco1s"), ("GENCO + 2 steps with switching", "vm_genco2s"),
                          ("no-limit AC solution", "vm_ac")):
            score(f"{name}", col, "real_new", pqm, "PQ")
            score(f"{name}", col, "real_new", allm, "PQ + PV")
        log("  (PV buses: GENCO and plain steps put them on the setpoint, so a violation at a switched PV bus can only be seen by a method that switches.)")

    pd.DataFrame(out).to_csv(OUT / f"summary{a.tag}.csv", index=False)
    log(f"\nsaved {OUT / f'summary{a.tag}.csv'}")
    logf.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["run", "volt", "report"])
    p.add_argument("--grid", default="500")
    p.add_argument("--procs", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="run: random subset of load scenarios (seed 0); volt: about this many states")
    p.add_argument("--pm-checks", type=int, default=60)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    {"run": run, "volt": volt, "report": report}[a.phase](a)

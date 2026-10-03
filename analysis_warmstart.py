"""GENCO as a warm start for Newton-Raphson AC power flow, and one Newton step as a physics correction.

For each grid state (load level x outage set) the AC power flow is solved by a plain full Newton
(polar, MATPOWER-style, sparse LU, no step control, no Q limits -- the same problem PowerModels'
compute_ac_pf solves for the engine) from four start points:

    flat   PV/REF buses at their voltage setpoint, PQ buses at 1.0 p.u., all angles at the reference
    dc     flat magnitudes, DC power-flow angles
    genco  GENCO's Vm (PQ buses) and Va; PV/REF magnitudes and the reference angle are the setpoints
    prev   the solution of the dispatch the state was derived from (AC-OPF point before outages /
           load change) -- this is what engine.py hands to PowerModels, i.e. the operational baseline

and line loadings are computed after 0, 1 and 2 Newton steps to see what one Jacobian solve buys.

Sets:  normal  N-0/N-1/N-2/N-3 at load levels with an AC-OPF dispatch
       stress  loads scaled beyond the dispatched level WITHOUT re-dispatch (slack picks up the
               difference), combined with N-0/N-1/N-3/N-4 outages. GENCO is out of distribution here.

GENCO's input for a state is built without any AC solution (a placeholder result is fed through
datakit's post-processing; the model masks those fields -- checked in validate()).

    work/texas/tmo 3000 .venv/bin/python -u analysis_warmstart.py --grid 118
    work/texas/tmo 6000 .venv/bin/python -u analysis_warmstart.py --grid texas
    .venv/bin/python analysis_warmstart.py --grid both --reuse
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import engine  # noqa: E402  (loads Julia before torch)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix, diags, hstack, vstack  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

from gridfm_datakit.network import load_net_from_file, makeYbus  # noqa: E402
from gridfm_datakit.process.process_network import _solution_arrays  # noqa: E402
from gridfm_datakit.process.solvers import run_dcpf, run_pf  # noqa: E402
from gridfm_datakit.utils.idx_brch import BR_STATUS, F_BUS, RATE_A, T_BUS  # noqa: E402
from gridfm_datakit.utils.idx_bus import BUS_I, BUS_TYPE, PD, PQ, PV, QD, REF, VA, VM  # noqa: E402
from gridfm_datakit.utils.idx_gen import GEN_BUS, GEN_STATUS  # noqa: E402

TEXAS = ROOT / "work" / "texas"
OUT = {"118": ROOT / "work" / "warmstart_case118.parquet", "texas": TEXAS / "out" / "warmstart_texas.parquet"}
LEVELS = {"118": (0.7, 0.8, 0.9, 1.0, 1.1, 1.2), "texas": (0.8, 0.85, 0.9, 0.95, 1.0)}
# Stress: (dispatch level, load multipliers applied on top of it without re-dispatch).
STRESS = {
    "118": ((1.0, 1.2), (1.1, 1.2, 1.3, 1.4, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0)),  # nothing solves from x2.0
    "texas": ((1.0,), (1.05, 1.1, 1.15, 1.2, 1.22, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3, 1.4)),  # nothing solves at x1.3
}
STARTS = ("flat", "dc", "genco", "prev")
TOL = 1e-8  # p.u. (1e-6 MW on a 100 MVA base), max-norm of the bus power mismatch
MAX_IT = 30
SAME = 1e-5  # p.u. / rad: two solutions closer than this are the same


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- Newton power flow
class Problem:
    """Admittance matrices, injections and bus sets of one datakit Network state."""

    def __init__(self, net):
        assert (net.buses[:, BUS_I] == np.arange(len(net.buses))).all()
        self.base = net.baseMVA
        self.Ybus, self.Yf, self.Yt = makeYbus(net.baseMVA, net.buses, net.branches)
        self.Ybus = self.Ybus.tocsr()
        on = net.gens[:, GEN_STATUS] == 1
        gbus = net.gens[on, GEN_BUS].astype(int)
        n = len(net.buses)
        pg = np.bincount(gbus, weights=net.gens[on, 1], minlength=n)
        qg = np.bincount(gbus, weights=net.gens[on, 2], minlength=n)
        self.S = ((pg - net.buses[:, PD]) + 1j * (qg - net.buses[:, QD])) / self.base
        kind = net.buses[:, BUS_TYPE]
        self.pq = np.flatnonzero(kind == PQ)
        self.pv = np.flatnonzero(kind == PV)
        self.ref = np.flatnonzero(kind == REF)
        self.pvpq = np.r_[self.pv, self.pq]
        self.vm_set = net.buses[:, VM].copy()  # setpoints at PV/REF (the dispatch's voltages)
        self.va_ref = np.radians(net.buses[self.ref[0], VA])
        self.f = net.branches[:, F_BUS].astype(int)
        self.t = net.branches[:, T_BUS].astype(int)
        self.live = net.branches[:, BR_STATUS] == 1
        rate = net.branches[:, RATE_A] / self.base
        self.rated = self.live & (rate > 0)
        self.rate = np.where(rate > 0, rate, np.nan)

    def start(self, vm_pq=None, va=None):
        """Complex start vector: setpoints at PV/REF, given (or 1.0) magnitudes at PQ, given (or flat) angles."""
        vm = self.vm_set.copy()
        vm[self.pq] = 1.0 if vm_pq is None else vm_pq[self.pq]
        ang = np.full(len(vm), self.va_ref) if va is None else np.asarray(va, dtype=float).copy()
        ang[self.ref] = self.va_ref
        return vm * np.exp(1j * ang)

    def mismatch(self, V):
        mis = V * np.conj(self.Ybus @ V) - self.S
        return np.r_[mis[self.pvpq].real, mis[self.pq].imag]

    def jacobian(self, V):
        Ibus = self.Ybus @ V
        dV, dI, dn = diags(V), diags(Ibus), diags(V / np.abs(V))
        dS_dVa = 1j * dV @ np.conj(dI - self.Ybus @ dV)
        dS_dVm = dV @ np.conj(self.Ybus @ dn) + np.conj(dI) @ dn
        dS_dVa, dS_dVm = csr_matrix(dS_dVa), csr_matrix(dS_dVm)
        pvpq, pq = self.pvpq, self.pq
        return vstack([
            hstack([dS_dVa[pvpq][:, pvpq].real, dS_dVm[pvpq][:, pq].real]),
            hstack([dS_dVa[pq][:, pvpq].imag, dS_dVm[pq][:, pq].imag]),
        ]).tocsc()

    def newton(self, V0, max_it=MAX_IT, keep=2):
        """Full Newton from V0. Returns dict(converged, iterations, mis0 [MW], V, steps=[V after 1..keep steps])."""
        V = V0.copy()
        va, vm = np.angle(V), np.abs(V)
        npv = len(self.pvpq)
        F = self.mismatch(V)
        norm = np.abs(F).max()
        out = {"mis0": norm * self.base, "steps": [], "iterations": 0, "converged": bool(norm < TOL)}
        while not out["converged"] and out["iterations"] < max_it:
            try:
                with np.errstate(all="ignore"):
                    dx = splu(self.jacobian(V)).solve(-F)
            except RuntimeError:  # singular Jacobian
                break
            va[self.pvpq] += dx[:npv]
            vm[self.pq] += dx[npv:]
            V = vm * np.exp(1j * va)
            out["iterations"] += 1
            with np.errstate(all="ignore"):
                F = self.mismatch(V)
            if not np.isfinite(F).all():
                break
            if len(out["steps"]) < keep:
                out["steps"].append(V.copy())
            norm = np.abs(F).max()
            out["converged"] = bool(norm < TOL)
        out["V"] = V
        out["mis_end"] = float(norm * self.base) if np.isfinite(norm) else np.inf
        return out

    def loading(self, V):
        """Loading of every branch (share of rating, NaN if out of service / unrated)."""
        with np.errstate(all="ignore"):
            sf = np.abs(V[self.f] * np.conj(self.Yf @ V))
            st = np.abs(V[self.t] * np.conj(self.Yt @ V))
            return np.where(self.rated, np.maximum(sf, st) / self.rate, np.nan)


# ----------------------------------------------------------------------------- GENCO without an AC solution
def placeholder_ac(net):
    """A stand-in 'AC result' so datakit builds GENCO's input tables for a state nobody has solved.
    Everything the result contributes is either a setpoint or masked by the PF task."""
    rev = net.reverse_bus_index_mapping
    vm, va = net.buses[:, VM], np.radians(net.buses[:, VA])
    gens = net.idx_gens_in_service
    return {
        "solve_time": 0.0,
        "solution": {
            "pf": True,
            "bus": {str(rev[i]): {"vm": float(vm[i]), "va": float(va[i])} for i in range(len(vm))},
            "gen": {str(i + 1): {"pg": float(net.gens[i, 1] / net.baseMVA), "qg": float(net.gens[i, 2] / net.baseMVA)} for i in gens},
            "branch": {str(i + 1): {"pf": 0.0, "qf": 0.0, "pt": 0.0, "qt": 0.0} for i in range(len(net.branches))},
        },
    }


def genco_voltage(eng, net, ac, dc):
    bus, gen, branch = engine.tables(net, ac, dc)
    start = time.perf_counter()
    bus_pred, _ = eng.genco.solve(bus, gen, branch)
    seconds = time.perf_counter() - start
    vm, va = bus_pred.Vm.to_numpy(float), np.radians(bus_pred.Va.to_numpy(float))
    residual = float(bus_pred.res_p.abs().max())
    return vm, va, residual, seconds


# ----------------------------------------------------------------------------- one state
def solve_state(eng, net, meta, want_lines=True):
    """All start points on one state. Returns (state row, line frame or None)."""
    prob = Problem(net)
    row = dict(meta)
    dc = run_dcpf(net, eng.jl, fast=True)
    _, _, dc_bus = _solution_arrays(dc, net)
    dc_flow, _, _ = _solution_arrays(dc, net)
    va_dc = dc_bus[:, 1]
    # The engine's own AC solve (PowerModels compute_ac_pf / NLsolve, started at the dispatch point).
    try:
        start = time.perf_counter()
        ac = run_pf(net, eng.jl, fast=True)
        row["engine_ms"] = 1e3 * (time.perf_counter() - start)
        _, _, ac_bus = _solution_arrays(ac, net)
        v_engine = ac_bus[:, 0] * np.exp(1j * ac_bus[:, 1])
        row["engine_ok"] = True
    except RuntimeError:
        v_engine, row["engine_ok"], row["engine_ms"] = None, False, np.nan

    vm_g, va_g, row["genco_residual_mw"], genco_s = genco_voltage(eng, net, placeholder_ac(net), dc)
    row["genco_ms"] = 1e3 * genco_s
    v_genco_raw = vm_g * np.exp(1j * va_g)
    starts = {
        "flat": prob.start(),
        "dc": prob.start(va=va_dc),
        "genco": prob.start(vm_pq=vm_g, va=va_g),
        "prev": prob.start(vm_pq=net.buses[:, VM], va=np.radians(net.buses[:, VA])),
    }
    runs = {}
    for name, v0 in starts.items():
        start = time.perf_counter()
        runs[name] = prob.newton(v0)
        row[f"ms_{name}"] = 1e3 * (time.perf_counter() - start)
        row[f"conv_{name}"] = runs[name]["converged"]
        row[f"it_{name}"] = runs[name]["iterations"]
        row[f"mis0_{name}"] = runs[name]["mis0"]

    # Reference solution: the engine's if it converged, else the solution most starts agree on.
    if v_engine is not None:
        v_ref, row["ref"] = v_engine, "engine"
    else:
        sols = [runs[n]["V"] for n in STARTS if runs[n]["converged"]]
        if sols:
            votes = [sum(np.abs(s - o).max() < SAME for o in sols) for s in sols]
            v_ref, row["ref"] = sols[int(np.argmax(votes))], "newton"
        else:
            v_ref, row["ref"] = None, "none"
    for name in STARTS:
        ok = runs[name]["converged"] and v_ref is not None
        row[f"dv_{name}"] = float(np.abs(runs[name]["V"] - v_ref).max()) if ok else np.nan
        row[f"same_{name}"] = bool(ok and row[f"dv_{name}"] < SAME)
        row[f"vmin_{name}"] = float(np.abs(runs[name]["V"]).min()) if runs[name]["converged"] else np.nan
    if v_ref is None:
        return row, None
    row["vmin_ref"] = float(np.abs(v_ref).min())
    row["mis_ref_mw"] = float(np.abs(prob.mismatch(v_ref)).max() * prob.base)
    row["genco_dvm"] = float(np.abs(vm_g - np.abs(v_ref)).max())
    row["genco_dva_deg"] = float(np.degrees(np.abs(va_g - np.angle(v_ref))).max())
    if not want_lines:
        return row, None

    def after(name, k):
        steps = runs[name]["steps"]
        if len(steps) >= k:
            return prob.loading(steps[k - 1])
        # converged in fewer than k steps -> final point; broke down -> NaN
        return prob.loading(runs[name]["V"]) if runs[name]["converged"] else np.full(len(prob.rate), np.nan)

    dc_loading = np.full(len(prob.rate), np.nan)
    dc_loading[net.idx_branches_in_service] = np.abs(dc_flow[:, 0])
    lines = pd.DataFrame({
        "state": meta["state"],
        "line": np.arange(len(prob.rate)),
        "ac": prob.loading(v_ref),
        "genco": prob.loading(v_genco_raw),
        "genco1": after("genco", 1),
        "genco2": after("genco", 2),
        "dc": np.where(prob.rated, dc_loading / prob.rate, np.nan),
        "dc1": after("dc", 1),
        "dc2": after("dc", 2),
        "flat1": after("flat", 1),
        "flat2": after("flat", 2),
        "prev1": after("prev", 1),
    })[prob.rated]
    return row, lines.astype({c: "float32" for c in lines.columns if c not in ("state", "line")})


# ----------------------------------------------------------------------------- validation
def validate(eng, grid):
    """Own Newton vs the engine's AC, GENCO from placeholder vs real tables, loading formula vs engine."""
    rng = np.random.default_rng(7)
    rated = [b["id"] for b in eng.static["branches"] if b["rate_a"] is not None and not b["bridge"]]
    worst = {"newton_dV": 0.0, "newton_dloading_pp": 0.0, "genco_placeholder_dVm": 0.0, "genco_placeholder_dVa_deg": 0.0,
             "genco_loading_vs_engine_pp": 0.0, "ac_loading_vs_engine_pp": 0.0}
    n = 0
    for scale in (LEVELS[grid][0], LEVELS[grid][-1]):
        for k in (0, 1, 2, 3):
            outages = sorted(int(i) for i in rng.choice(rated, k, replace=False))
            try:
                result = eng.solve(scale, outages)
            except engine.SolveError:
                continue
            net = eng._dispatch(scale)[0].copy_for_perturbation()
            if outages:
                net.deactivate_branches(outages)
            dc, ac = run_dcpf(net, eng.jl, fast=True), run_pf(net, eng.jl, fast=True)
            _, _, ac_bus = _solution_arrays(ac, net)
            v_engine = ac_bus[:, 0] * np.exp(1j * ac_bus[:, 1])
            prob = Problem(net)
            run = prob.newton(prob.start())
            assert run["converged"]
            worst["newton_dV"] = max(worst["newton_dV"], np.abs(run["V"] - v_engine).max())
            worst["newton_dloading_pp"] = max(worst["newton_dloading_pp"], 100 * np.nanmax(np.abs(prob.loading(run["V"]) - prob.loading(v_engine))))
            vm_a, va_a, _, _ = genco_voltage(eng, net, ac, dc)
            vm_b, va_b, _, _ = genco_voltage(eng, net, placeholder_ac(net), dc)
            worst["genco_placeholder_dVm"] = max(worst["genco_placeholder_dVm"], np.abs(vm_a - vm_b).max())
            worst["genco_placeholder_dVa_deg"] = max(worst["genco_placeholder_dVa_deg"], np.degrees(np.abs(va_a - va_b)).max())
            br = pd.DataFrame(result["branches"])
            mine_g = prob.loading(vm_a * np.exp(1j * va_a))
            both = prob.rated & br.loading_genco.notna().to_numpy()
            worst["genco_loading_vs_engine_pp"] = max(worst["genco_loading_vs_engine_pp"], 100 * np.abs(mine_g - br.loading_genco.to_numpy(float))[both].max())
            worst["ac_loading_vs_engine_pp"] = max(worst["ac_loading_vs_engine_pp"], 100 * np.abs(prob.loading(v_engine) - br.loading_ac.to_numpy(float))[both].max())
            n += 1
    print(f"\n[validate {grid}] {n} states; worst differences:")
    for k, v in worst.items():
        print(f"    {k:<32} {v:.3e}")
    return worst


# ----------------------------------------------------------------------------- sweeps
def build(grid):
    engine.layout = lambda network, n, edges: np.zeros((n, 2))  # no coordinates needed (and no download)
    if grid == "118":
        return engine.Engine("case118_ieee", "base")
    sys.path.insert(0, str(TEXAS))
    c = __import__("03_classical")
    engine.WORK = TEXAS
    engine.load_net_from_pglib = lambda name: c.in_service_gens_only(load_net_from_file(str(c.CASE)))
    return engine.Engine(c.NET, "base")


def outage_sets(eng, rng, counts):
    """counts: {k: number of sets of k simultaneous outages}; sets that island the grid are redrawn."""
    rated = [b["id"] for b in eng.static["branches"] if b["rate_a"] is not None and not b["bridge"]]
    base = eng._dispatch(1.0)[0]
    sets = []
    for k, n in counts.items():
        found = set()
        tries = 0
        while len(found) < n and tries < 50 * n:
            tries += 1
            pick = tuple(sorted(int(i) for i in rng.choice(rated, k, replace=False)))
            if pick in found:
                continue
            if k:
                net = base.copy_for_perturbation()
                net.deactivate_branches(list(pick))
                if not net.check_single_connected_component():
                    continue
            found.add(pick)
            if k == 0:
                break
        sets += sorted(found)
    return sets


def sweep(eng, grid, n_per_class, n_stress):
    rng = np.random.default_rng(1)
    rows, frames, state = [], [], 0
    normal_sets = outage_sets(eng, rng, {0: 1, 1: n_per_class, 2: n_per_class, 3: n_per_class})
    stress_sets = outage_sets(eng, rng, {0: 1, 1: n_stress, 3: n_stress, 4: n_stress})

    def run(kind, scale, mult, outages):
        nonlocal state
        net = eng._dispatch(scale)[0].copy_for_perturbation()
        if mult != 1.0:
            net.buses[:, PD] *= mult
            net.buses[:, QD] *= mult
        if outages:
            net.deactivate_branches(list(outages))
        meta = {"state": state, "grid": grid, "set": kind, "dispatch_scale": scale, "load_mult": mult,
                "n_out": len(outages), "outages": "+".join(map(str, outages))}
        row, lines = solve_state(eng, net, meta)
        rows.append(row)
        if lines is not None:
            frames.append(lines)
        state += 1

    for scale in LEVELS[grid]:
        try:
            eng._dispatch(scale)
        except engine.SolveError as err:
            log(f"load x{scale}: {err.message} -> level skipped")
            continue
        for outages in normal_sets:
            run("normal", scale, 1.0, outages)
        log(f"normal, load x{scale}: {state} states so far")
    scales, mults = STRESS[grid]
    for scale in scales:
        for mult in mults:
            for outages in stress_sets:
                run("stress", scale, mult, outages)
            log(f"stress, dispatch x{scale} loads x{mult}: {state} states so far")
    # N-4 at the dispatched load: stress by topology only
    extra = outage_sets(eng, rng, {4: 3 * n_stress})
    for scale in (LEVELS[grid][-1],):
        for outages in extra:
            run("stress_n4", scale, 1.0, outages)
    log(f"N-4 at load x{LEVELS[grid][-1]}: {state} states so far")
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- report
def convergence_table(states, title):
    print(f"\n{title}  ({len(states)} states; engine/PowerModels converged on {int(states.engine_ok.sum())}, "
          f"a reference solution exists for {int((states.ref != 'none').sum())})")
    print(f"    {'start':<6} {'converged':>10} {'same sol.':>10} {'iters median':>13} {'mean':>6} {'max':>4} {'start mismatch MW median':>25} {'p90':>9}")
    for s in STARTS:
        conv = states[f"conv_{s}"]
        it = states.loc[conv, f"it_{s}"]
        print(f"    {s:<6} {conv.mean():>9.1%} {states[f'same_{s}'].mean():>10.1%} {it.median() if len(it) else np.nan:>13.0f} "
              f"{it.mean() if len(it) else np.nan:>6.2f} {it.max() if len(it) else np.nan:>4.0f} "
              f"{states[f'mis0_{s}'].median():>25.1f} {states[f'mis0_{s}'].quantile(0.9):>9.1f}")


def report(states, lines, grid):
    print(f"\n{'=' * 30} {grid} {'=' * 30}")
    normal = states[states.set == "normal"]
    convergence_table(normal, "1. NORMAL SET, Newton from each start")
    print("    iterations (mean) by number of outages:")
    print(normal.groupby("n_out")[[f"it_{s}" for s in STARTS]].mean().round(2).to_string().replace("\n", "\n    "))
    print("    iterations (mean) by load level:")
    print(normal.groupby("dispatch_scale")[[f"it_{s}" for s in STARTS]].mean().round(2).to_string().replace("\n", "\n    "))
    print(f"    GENCO vs AC on this set: max |dVm| median {normal.genco_dvm.median():.4f} p.u., max |dVa| median {normal.genco_dva_deg.median():.3f} deg")
    print("    indicative wall-clock, median ms (noisy; Newton here is unoptimised SciPy): "
          + ", ".join(f"{s} {normal[f'ms_{s}'].median():.1f}" for s in STARTS)
          + f", GENCO inference {normal.genco_ms.median():.1f}, engine PowerModels {normal.engine_ms.median():.1f}")

    for kind, title in (("stress", "2a. STRESS SET: loads scaled without re-dispatch"), ("stress_n4", "2b. STRESS SET: N-4 at the top dispatched load")):
        st = states[states.set == kind]
        if st.empty:
            continue
        convergence_table(st, title)
        if kind == "stress":
            print("    converged share by load multiplier (all dispatch levels and outage sets):")
            tab = st.groupby("load_mult")[[f"conv_{s}" for s in STARTS] + ["engine_ok"]].mean().round(2)
            tab["n"] = st.groupby("load_mult").size()
            print(tab.to_string().replace("\n", "\n    "))
        solvable = st[st.ref != "none"]
        f, g, d, p = solvable.same_flat, solvable.same_genco, solvable.same_dc, solvable.same_prev
        print(f"    of {len(solvable)} states with a solution: GENCO start reaches it where flat fails: {int((g & ~f).sum())}, "
              f"where DC fails: {int((g & ~d).sum())}, where flat AND DC fail: {int((g & ~f & ~d).sum())}; "
              f"flat reaches it where GENCO fails: {int((f & ~g).sum())}; DC where GENCO fails: {int((d & ~g).sum())}; "
              f"prev where GENCO fails: {int((p & ~g).sum())}; GENCO where prev fails: {int((g & ~p).sum())}")
        other = {s: int((solvable[f"conv_{s}"] & ~solvable[f"same_{s}"]).sum()) for s in STARTS}
        print(f"    converged to a DIFFERENT solution than the reference: {other}")

    print("\n3. LINE LOADING after k Newton steps (error in points of rating vs the converged AC solution)")
    for kind in ("normal", "stress"):
        ids = states.state[(states.set == kind) & (states.ref != "none")]
        sub = lines[lines.state.isin(ids)]
        if sub.empty:
            continue
        real = sub.ac > engine.OVERLOAD
        print(f"  {kind}: {sub.state.nunique()} states, {len(sub):,} line-states, {int(real.sum())} real overloads")
        print(f"    {'method':<8} {'median':>8} {'p99':>8} {'max':>9} {'missed':>7} {'false':>6} {'no value':>9}")
        for m in ("genco", "genco1", "genco2", "dc", "dc1", "dc2", "flat1", "flat2", "prev1"):
            err = (sub[m] - sub.ac).abs() * 100
            called = sub[m] > engine.OVERLOAD
            print(f"    {m:<8} {err.median():>8.3f} {err.quantile(0.99):>8.2f} {err.max():>9.1f} {int((real & ~called).sum()):>7} "
                  f"{int((~real & called).sum()):>6} {int(sub[m].isna().sum()):>9}")


def lines_path(grid):
    return OUT[grid].with_name(OUT[grid].stem + "_lines.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=("118", "texas", "both"), default="118")
    parser.add_argument("--n", type=int, default=20, help="outage sets per class (N-1, N-2, N-3) per load level")
    parser.add_argument("--stress", type=int, default=3, help="outage sets per class (N-1, N-3, N-4) per stress level")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    grids = ("118", "texas") if args.grid == "both" else (args.grid,)
    if not args.reuse and len(grids) > 1:
        parser.error("solve one grid per process (the engine is patched per grid); --grid both only with --reuse")
    for grid in grids:
        if args.reuse:
            states, lines = pd.read_parquet(OUT[grid]), pd.read_parquet(lines_path(grid))
        else:
            start = time.perf_counter()
            log(f"building engine for {grid}")
            eng = build(grid)
            validate(eng, grid)
            if args.validate_only:
                return
            states, lines = sweep(eng, grid, args.n, args.stress)
            states.to_parquet(OUT[grid], index=False)
            lines.to_parquet(lines_path(grid), index=False)
            log(f"saved {OUT[grid]} and {lines_path(grid)} after {time.perf_counter() - start:.0f} s")
        report(states, lines, grid)


if __name__ == "__main__":
    main()

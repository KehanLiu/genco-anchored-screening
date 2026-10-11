"""Same-solver comparison of AC power flow start points inside PowerModels (the solver datakit used).

Every stored scenario is rebuilt as the PowerModels data dict datakit solved (datakit's own _gfm_state path:
perturbed r/x/b, removed branches, loads, setpoints), turned into PowerModels' PowerFlowData
(instantiate_pf_data) and solved with PowerModels._compute_ac_pf, i.e. the NLsolve call behind compute_ac_pf,
which returns the NLsolve result (iterations, residual norm, converged flags). Only the start point differs:

    default   what datakit used: the AC-OPF point of the intact grid (bus vm/va -> vm_start/va_start, gen pg/qg ->
              pg_start/qg_start). The OPF voltages are not stored; they are recovered as the power flow solution of
              the same scenario with the removed branches put back (same injections and setpoints => same point).
    flat      PowerModels flat start: 1.0 p.u. at PQ, setpoints at PV/REF, angles 0, unknown injections 0
    dc        stored DC angles, flat magnitudes
    tiny      GENCO Tiny: Vm at PQ buses and all angles; PV/REF magnitudes are setpoints
    small     GENCO Small
  suffix _qc: the unknown injections of PowerModels' formulation (Q at PV buses, P and Q at the slack) start at the
              values that balance those buses for the start voltages (what the default start has by construction);
  suffix _q0: they start at 0 (what PowerModels does when no qg_start is given).
  suffix _nt: NLsolve method=:newton (no step control) instead of the default :trust_region.
  suffix _p : "primed". PowerModels 0.21.6 evaluates its Jacobian (jsp!) from the voltages left behind by the last
              residual call (f!), not from the x it is given, and NLsolve asks for the first Jacobian BEFORE the first
              residual. On a fresh PowerFlowData (what compute_ac_pf / datakit does) the first Jacobian is therefore the
              flat-voltage Jacobian whatever the start point is. Without suffix the solver runs exactly like that
              (internal voltages reset to the freshly instantiated state before every solve); with _p the internal
              voltages are set to the start point first, so the first Jacobian belongs to the start point.

    work/texas/tmo 900  .venv/bin/python -u analysis/analysis_solver_timing.py prep  --grid 2000 --n 200
    work/texas/tmo 1800 .venv/bin/python -u analysis/analysis_solver_timing.py solve --grid 2000
    work/texas/tmo 900  .venv/bin/python -u analysis/analysis_solver_timing.py chain --grid 2000
    .venv/bin/python analysis/analysis_solver_timing.py summary
Outputs in work/timing/.
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "timing"
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here
LOG = OUT / "timing.log"
SEED = 0
MAXIT = 60  # NLsolve's default is 1000; only a cap for diverging / stalling starts
REPS = 3
# (name, voltage source, unknown injections consistent with the start voltages?, NLsolve method)
CONFIGS = [
    ("default", "default", None, "trust_region", 0),
    ("flat_q0", "flat", False, "trust_region", 0),
    ("dc_qc", "dc", True, "trust_region", 0),
    ("tiny_qc", "tiny", True, "trust_region", 0),
    ("tiny_q0", "tiny", False, "trust_region", 0),
    ("small_qc", "small", True, "trust_region", 0),
    ("default_p", "default", None, "trust_region", 1),
    ("flat_q0_p", "flat", False, "trust_region", 1),
    ("dc_qc_p", "dc", True, "trust_region", 1),
    ("tiny_qc_p", "tiny", True, "trust_region", 1),
    ("small_qc_p", "small", True, "trust_region", 1),
    ("default_nt", "default", None, "newton", 0),
    ("flat_q0_nt", "flat", False, "newton", 0),
    ("tiny_qc_nt", "tiny", True, "newton", 0),
    ("default_nt_p", "default", None, "newton", 1),
    ("flat_q0_nt_p", "flat", False, "newton", 1),
    ("dc_qc_nt_p", "dc", True, "newton", 1),
    ("tiny_qc_nt_p", "tiny", True, "newton", 1),
]

JULIA = r"""
using SparseArrays, LinearAlgebra

function gfmt_inst(data)
    t = time_ns()
    pf = PowerModels.instantiate_pf_data(data)
    return pf, (time_ns() - t) / 1e6
end

gfmt_order(pf) = (collect(pf.am.idx_to_bus), collect(pf.bus_type_idx))

function _gfmt_full(pf, vm, va)
    n = length(pf.bus_type_idx)
    vmf = [pf.bus_type_idx[i] == 1 ? Float64(vm[i]) : pf.vm_idx[i] for i in 1:n]
    vaf = [pf.bus_type_idx[i] == 3 ? pf.va_idx[i] : Float64(va[i]) for i in 1:n]
    return vmf, vaf
end

# x0 of PowerModels' formulation for start voltages (vm used at PQ, va at PQ/PV; both in admittance-matrix order)
function gfmt_x0(pf, vm, va, consistent::Bool)
    vmf, vaf = _gfmt_full(pf, vm, va)
    V = vmf .* cis.(vaf)
    S = V .* conj.(pf.am.matrix * V)
    n = length(vmf)
    x = zeros(2n)
    for i in 1:n
        t = pf.bus_type_idx[i]
        if t == 1
            x[2i-1] = vmf[i]; x[2i] = vaf[i]
        elseif t == 2
            x[2i-1] = consistent ? -(pf.q_delta_base_idx[i] + imag(S[i])) : 0.0
            x[2i] = vaf[i]
        else
            x[2i-1] = consistent ? -(pf.p_delta_base_idx[i] + real(S[i])) : 0.0
            x[2i] = consistent ? -(pf.q_delta_base_idx[i] + imag(S[i])) : 0.0
        end
    end
    return x
end

function _gfmt_volt(pf, x)
    n = length(pf.bus_type_idx)
    vm = [pf.bus_type_idx[i] == 1 ? x[2i-1] : pf.vm_idx[i] for i in 1:n]
    va = [pf.bus_type_idx[i] == 3 ? pf.va_idx[i] : x[2i] for i in 1:n]
    return vm, va
end

# inf-norm of PowerModels' residual at x (own formula, same equations as f!)
function gfmt_resid(pf, x)
    vm, va = _gfmt_volt(pf, x)
    V = vm .* cis.(va)
    S = V .* conj.(pf.am.matrix * V)
    r = 0.0
    for i in eachindex(vm)
        t = pf.bus_type_idx[i]
        p = pf.p_delta_base_idx[i] + real(S[i]) + (t == 3 ? x[2i-1] : 0.0)
        q = pf.q_delta_base_idx[i] + imag(S[i]) + (t == 2 ? x[2i-1] : (t == 3 ? x[2i] : 0.0))
        r = max(r, abs(p), abs(q))
    end
    return r
end

# write x0 into the data dict the way PowerModels reads its warm start (vm_start/va_start, pg_start/qg_start)
function gfmt_set!(pf, x0)
    data = pf.data
    for (_, g) in data["gen"]
        g["pg_start"] = 0.0
        g["qg_start"] = 0.0
    end
    for (i, bid) in enumerate(pf.am.idx_to_bus)
        bus = data["bus"]["$(bid)"]
        t = pf.bus_type_idx[i]
        if t == 1
            bus["vm_start"] = x0[2i-1]; bus["va_start"] = x0[2i]
        elseif t == 2
            bus["va_start"] = x0[2i]
            pf.bus_gens[bid][1]["qg_start"] = -x0[2i-1]
        else
            pf.bus_gens[bid][1]["pg_start"] = -x0[2i-1]
            pf.bus_gens[bid][1]["qg_start"] = -x0[2i]
        end
    end
    return nothing
end

function gfmt_dv(pf, x, vm_ref, va_ref)
    vm, va = _gfmt_volt(pf, x)
    return maximum(abs.(vm .* cis.(va) .- vm_ref .* cis.(va_ref)))
end

# internal voltage state of PowerFlowData: as freshly instantiated (prime=false) or at the start point (prime=true)
function gfmt_state!(pf, x0, prime::Bool)
    if prime
        vm, va = _gfmt_volt(pf, x0)
        pf.vm_idx .= vm; pf.va_idx .= va
    else
        for i in eachindex(pf.bus_type_idx)
            if pf.bus_type_idx[i] == 1
                pf.vm_idx[i] = 1.0
            end
            pf.va_idx[i] = 0.0
        end
    end
    fill!(pf.p_inject_idx, 0.0); fill!(pf.q_inject_idx, 0.0)
    return nothing
end

function gfmt_bench(pf, X0py, meths::String, primes::String, reps::Int, maxit::Int, offset::Int, vm_ref_py, va_ref_py)
    pr = [p == "1" for p in split(primes, ",")]
    X0 = Array{Float64}(X0py)
    vm_ref = Array{Float64}(vm_ref_py); va_ref = Array{Float64}(va_ref_py)
    ms = split(meths, ",")
    k = size(X0, 2)
    T = fill(NaN, k, reps)
    its = fill(-1, k); fc = fill(-1, k); gc = fill(-1, k)
    res = fill(NaN, k); res0 = fill(NaN, k); dv = fill(NaN, k); x0err = fill(NaN, k)
    conv = fill(false, k)
    Z = fill(NaN, size(X0, 1), k)
    for rep in 1:reps
        for jj in 1:k
            j = mod1(jj + offset + rep - 1, k)
            if rep > 1 && !(res[j] <= 1e-8)   # not converged in the first repetition: no timing needed
                continue
            end
            x0 = X0[:, j]
            gfmt_set!(pf, x0)
            fill!(pf.x0, 0.0)
            gfmt_state!(pf, x0, pr[j])
            m = Symbol(ms[j])
            try
                t = time_ns()
                r = PowerModels._compute_ac_pf(pf; method=m, iterations=maxit)
                T[j, rep] = (time_ns() - t) / 1e6
                if rep == 1
                    its[j] = r.iterations; fc[j] = r.f_calls; gc[j] = r.g_calls
                    res[j] = r.residual_norm
                    conv[j] = (r.x_converged || r.f_converged) && isfinite(r.residual_norm)
                    x0err[j] = maximum(abs.(pf.x0 .- x0))
                    res0[j] = gfmt_resid(pf, x0)
                    Z[:, j] = r.zero
                    dv[j] = all(isfinite, r.zero) ? gfmt_dv(pf, r.zero, vm_ref, va_ref) : NaN
                end
            catch err
                if rep == 1
                    res0[j] = gfmt_resid(pf, x0)
                end
            end
        end
    end
    return (T=T, its=its, fc=fc, gc=gc, res=res, res0=res0, dv=dv, x0err=x0err, conv=conv, Z=Z)
end

gfmt_env() = (Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads(), string(VERSION), Sys.CPU_NAME)
"""


def log(*a):
    OUT.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_evidence(tag):
    up = subprocess.run("uptime", shell=True, capture_output=True, text=True).stdout.strip()
    ps = subprocess.run("ps -Ao pcpu,pmem,pid,comm -r | head -9", shell=True, capture_output=True, text=True).stdout.rstrip()
    log(f"[load {tag}] uptime: {up}")
    for row in ps.splitlines():
        log(f"[load {tag}]   {row[:150]}")


def machine():
    out = subprocess.run("sysctl -n machdep.cpu.brand_string hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize; sw_vers -productVersion",
                         shell=True, capture_output=True, text=True).stdout.split("\n")
    log(f"machine: {out[0]}, {out[1]} cores ({out[2]} + {out[3]} by performance level), {int(out[4]) / 2**30:.0f} GB, macOS {out[5]}")


def paths(grid):
    here = ROOT / "work" / f"goc{grid}"
    net = f"case{grid}_goc"
    return here, net, here / "data" / net / "raw"


def read_tables(raw, ids):
    import pandas as pd
    flt = [("scenario", "in", [int(i) for i in ids])]
    return {t: {s: d for s, d in pd.read_parquet(raw / f"{t}_data.parquet", filters=flt).groupby("scenario")} for t in ("bus", "gen", "branch")}


def state_of(tabs, s):
    bus = tabs["bus"][s].sort_values("bus").reset_index(drop=True)
    gen = tabs["gen"][s].sort_values("idx").reset_index(drop=True)
    branch = tabs["branch"][s].sort_values("idx").reset_index(drop=True)
    q = gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
    return bus.merge(q, on="bus", how="left").fillna(0), gen, branch


def newton_module(here):
    spec = importlib.util.spec_from_file_location("goc_newton", here / "02_newton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- prep: scenarios + GENCO predictions
def prep(a):
    import engine  # Julia before torch
    import numpy as np
    import pandas as pd
    import torch
    here, net, raw = paths(a.grid)
    engine.WORK = here
    log(f"=== prep {net}: {a.n} scenarios, seed {SEED}; torch {torch.__version__}, {torch.get_num_threads()} threads")
    all_ids = np.sort(pd.read_parquet(raw / "runtime_data.parquet").scenario.unique())
    ids = np.sort(np.random.default_rng(SEED).choice(all_ids, size=min(a.n, len(all_ids)), replace=False))
    tabs = read_tables(raw, ids)
    out = {"ids": ids}
    for variant in ("tiny", "small"):
        genco = engine.Genco(net, variant)
        vm, va, ms = [], [], []
        for s in ids:
            bus, gen, branch = state_of(tabs, s)
            t0 = time.perf_counter()
            bp, _ = genco.solve(bus, gen, branch)
            ms.append(1e3 * (time.perf_counter() - t0))
            vm.append(bp.Vm.to_numpy(float))
            va.append(np.radians(bp.Va.to_numpy(float)))
        out[f"vm_{variant}"], out[f"va_{variant}"], out[f"ms_{variant}"] = np.array(vm), np.array(va), np.array(ms)
        log(f"  GENCO {variant}: {len(ids)} predictions, end-to-end {np.median(ms):.1f} ms median (not a quiet-machine timing, see chain)")
    np.savez(OUT / f"genco_pred_{net}.npz", **out)
    log(f"  saved {OUT / f'genco_pred_{net}.npz'}")


# ----------------------------------------------------------------------------- solve: PowerModels from every start
def solve(a):
    import juliacall  # noqa: F401
    import numpy as np
    import pandas as pd
    from gridfm_datakit.network import load_net_from_file
    from gridfm_datakit.process.process_network import init_julia
    from gridfm_datakit.process import solvers
    from gridfm_datakit.utils.idx_brch import BR_B, BR_R, BR_STATUS, BR_X, F_BUS, T_BUS
    from gridfm_datakit.utils.idx_bus import BUS_I, BUS_TYPE, PD, QD, VA, VM
    from gridfm_datakit.utils.idx_gen import GEN_BUS, GEN_STATUS, PG, QG

    here, net_name, raw = paths(a.grid)
    machine()
    log(f"=== solve {net_name}: PowerModels._compute_ac_pf (NLsolve), {REPS} repetitions per start, interleaved, iteration cap {MAXIT}")
    pred = np.load(OUT / f"genco_pred_{net_name}.npz")
    ids = pred["ids"][: a.limit] if a.limit else pred["ids"]
    jl = init_julia(max_iter=200)
    jl.seval(JULIA)
    env = jl.gfmt_env()
    log(f"julia {env[2]}, Julia threads {env[0]}, BLAS threads {env[1]}, cpu {env[3]}; single process")
    net = load_net_from_file(str(here / "net" / f"{net_name}.m"))
    row_of_bus = net.buses[:, BUS_I].astype(int)  # table index of every net.buses row
    base_status = net.branches[:, BR_STATUS].copy()
    tabs = read_tables(raw, ids)
    rt = pd.read_parquet(raw / "runtime_data.parquet").set_index("scenario")
    names = [c[0] for c in CONFIGS]
    meths = ",".join(c[3] for c in CONFIGS)
    primes = ",".join(str(c[4]) for c in CONFIGS)

    def data_dict(bus, gen, branch, status):
        t0 = time.perf_counter()
        net.buses[:, PD] = bus.Pd.to_numpy()[row_of_bus]
        net.buses[:, QD] = bus.Qd.to_numpy()[row_of_bus]
        net.buses[:, BUS_TYPE] = (bus.PQ + 2 * bus.PV + 3 * bus.REF).to_numpy()[row_of_bus]
        net.buses[:, VM] = bus.Vm.to_numpy()[row_of_bus]
        net.buses[:, VA] = bus.Va.to_numpy()[row_of_bus]
        net.gens[:, PG] = gen.p_mw.to_numpy()
        net.gens[:, QG] = gen.q_mvar.to_numpy()
        net.gens[:, GEN_STATUS] = gen.in_service.to_numpy()
        net.branches[:, BR_STATUS] = status
        net.branches[:, BR_R] = branch.r.to_numpy()
        net.branches[:, BR_X] = branch.x.to_numpy()
        net.branches[:, BR_B] = branch.b.to_numpy()
        d = solvers._julia_pm_data(net, jl)
        return d, 1e3 * (time.perf_counter() - t0)

    def one(k, s, reps):
        bus, gen, branch = state_of(tabs, s)
        assert (branch.from_bus.to_numpy() == net.branches[:, F_BUS]).all() and (branch.to_bus.to_numpy() == net.branches[:, T_BUS]).all()
        assert (gen.bus.to_numpy() == net.gens[:, GEN_BUS]).all()
        status = branch.br_status.to_numpy().astype(float)
        assert (status <= base_status).all()
        n_out = int((base_status - status).sum())
        vm_tab, va_tab = bus.Vm.to_numpy(float), np.radians(bus.Va.to_numpy(float))
        # --- datakit's start: the OPF point = power flow solution with the removed branches back in service
        d, _ = data_dict(bus, gen, branch, base_status)
        pf, _ = jl.gfmt_inst(d)
        order, types = (np.asarray(x) for x in jl.gfmt_order(pf))
        perm = np.array([net.bus_index_mapping[int(b)] for b in order])
        vm_s, va_s = vm_tab[perm], va_tab[perm]
        x_st = np.asarray(jl.gfmt_x0(pf, vm_s, va_s, True))
        r = jl.gfmt_bench(pf, x_st.reshape(-1, 1), "trust_region", "1", 1, MAXIT, 0, vm_s, va_s)
        x_default = np.asarray(r.Z)[:, 0].copy()
        row = {"scenario": int(s), "load_idx": int(bus.load_scenario_idx.iloc[0]), "n_out": n_out,
               "intact_conv": bool(np.asarray(r.conv)[0]), "intact_it": int(np.asarray(r.its)[0]), "intact_dv_vs_stored": float(np.asarray(r.dv)[0])}
        # --- the scenario as datakit solved it
        d, row["dict_ms"] = data_dict(bus, gen, branch, status)
        pf, row["inst_ms"] = jl.gfmt_inst(d)
        order2, types2 = (np.asarray(x) for x in jl.gfmt_order(pf))
        assert (order2 == order).all() and (types2 == types).all()
        # validation: stored AC solution in PowerModels' own equations, and a solve started from it
        x_st = np.asarray(jl.gfmt_x0(pf, vm_s, va_s, True))
        row["stored_res0"] = float(jl.gfmt_resid(pf, x_st))
        ones, zeros = np.ones(len(perm)), np.zeros(len(perm))
        va_dc = np.radians(bus.Va_dc.to_numpy(float))
        va_dc = (va_dc - va_dc[bus.REF.to_numpy() == 1][0])[perm]
        volt = {"flat": (ones, zeros), "dc": (ones, va_dc),
                "tiny": (pred["vm_tiny"][k][perm], pred["va_tiny"][k][perm]), "small": (pred["vm_small"][k][perm], pred["va_small"][k][perm])}
        cols = [x_default if src == "default" else np.asarray(jl.gfmt_x0(pf, *volt[src], bool(qc))) for _, src, qc, _, _ in CONFIGS]
        X0 = np.column_stack(cols + [x_st])
        r = jl.gfmt_bench(pf, X0, meths + ",trust_region", primes + ",1", reps, MAXIT, k, vm_s, va_s)
        T = np.asarray(r.T)
        for j, name in enumerate(names + ["stored"]):
            row[f"pmconv_{name}"] = bool(np.asarray(r.conv)[j])  # PowerModels' flag: x_converged || f_converged
            row[f"conv_{name}"] = bool(np.asarray(r.res)[j] <= 1e-8)  # residual below NLsolve's ftol (x_converged alone = stalled trust region)
            row[f"it_{name}"] = int(np.asarray(r.its)[j])
            row[f"fcalls_{name}"] = int(np.asarray(r.fc)[j])
            row[f"jcalls_{name}"] = int(np.asarray(r.gc)[j])
            row[f"res0_{name}"] = float(np.asarray(r.res0)[j])
            row[f"res_{name}"] = float(np.asarray(r.res)[j])
            row[f"dv_{name}"] = float(np.asarray(r.dv)[j])
            row[f"x0err_{name}"] = float(np.asarray(r.x0err)[j])
            row[f"ms_min_{name}"] = float(np.nanmin(T[j])) if np.isfinite(T[j]).any() else np.nan
            row[f"ms_med_{name}"] = float(np.nanmedian(T[j])) if np.isfinite(T[j]).any() else np.nan
        row["stored_runtime_ms"] = 1e3 * float(rt.ac.loc[s])
        return row

    t0 = time.perf_counter()
    for kk in (0, 1):  # warm-up: compiles every path (both NLsolve methods)
        one(kk, ids[kk], 1)
    log(f"warm-up (2 scenarios, all starts, includes Julia compilation): {time.perf_counter() - t0:.1f} s")
    load_evidence("start")
    rows = []
    t0 = time.perf_counter()
    for k, s in enumerate(ids):
        rows.append(one(k, s, REPS))
        if k % max(len(ids) // 10, 1) == 0:
            w = rows[-1]
            log(f"  {k}/{len(ids)} scenario {s} n_out {w['n_out']}: it default/flat/dc_qc/tiny_qc {w['it_default']}/{w['it_flat_q0']}/{w['it_dc_qc']}/{w['it_tiny_qc']}; "
                f"ms {w['ms_min_default']:.1f}/{w['ms_min_flat_q0']:.1f}/{w['ms_min_dc_qc']:.1f}/{w['ms_min_tiny_qc']:.1f}; dv {w['dv_default']:.1e}; stored res0 {w['stored_res0']:.1e}")
    log(f"measured {len(rows)} scenarios in {time.perf_counter() - t0:.0f} s")
    load_evidence("end")
    df = pd.DataFrame(rows)
    out = OUT / f"solver_starts_{net_name}{a.tag}.parquet"
    df.to_parquet(out, index=False)
    log(f"saved {out}")
    report(df, net_name)


def report(df, net_name):
    import numpy as np
    log(f"--- {net_name}: {len(df)} scenarios; validation: stored AC solution in PowerModels' equations, residual max {df.stored_res0.max():.2e} p.u. "
        f"(median {df.stored_res0.median():.2e}); solve started from it: {df.it_stored.mean():.2f} iterations, max |dV| {df.dv_stored.max():.1e}; "
        f"x0 written through the dict vs intended, max diff {max(df[c].max() for c in df if c.startswith('x0err_')):.1e}")
    log(f"    intact-grid solves (default start): converged {df.intact_conv.sum()}/{len(df)}, iterations mean {df.intact_it.mean():.2f}; "
        f"n_out==0 scenarios: {(df.n_out == 0).sum()}, their |V_intact - V_stored| max {df[df.n_out == 0].intact_dv_vs_stored.max() if (df.n_out == 0).any() else float('nan'):.1e}")
    log(f"    data dict (datakit _gfm_state, incl. numpy writes) {df.dict_ms.median():.2f} ms median; instantiate_pf_data {df.inst_ms.median():.2f} ms median; "
        f"datakit stored runtime for these scenarios {df.stored_runtime_ms.median():.1f} ms median")
    log(f"    {'start':<12} {'conv':>8} {'it mean':>8} {'med':>4} {'max':>4} {'Jac':>6} {'ms med(min3)':>12} {'p90':>8} {'ms med(med3)':>12} {'res0 med':>10} {'max dV':>9} {'same':>8}")
    for name in [c[0] for c in CONFIGS]:
        c = df[df[f"conv_{name}"]]
        same = int((c[f"dv_{name}"] < 1e-5).sum())
        if len(c) == 0:
            log(f"    {name:<12} {0:>4}/{len(df):<3} (no convergence; start residual median {df[f'res0_{name}'].median():.2e} p.u.)")
            continue
        log(f"    {name:<12} {len(c):>4}/{len(df):<3} (PM flag {int(df[f'pmconv_{name}'].sum()):>3}) {c[f'it_{name}'].mean():>8.2f} {c[f'it_{name}'].median():>4.0f} {c[f'it_{name}'].max():>4} {c[f'jcalls_{name}'].mean():>6.2f} "
            f"{c[f'ms_min_{name}'].median():>12.2f} {c[f'ms_min_{name}'].quantile(.9):>8.2f} {c[f'ms_med_{name}'].median():>12.2f} {df[f'res0_{name}'].median():>10.2e} "
            f"{c[f'dv_{name}'].max():>9.1e} {same:>4}/{len(c):<3}")


# ----------------------------------------------------------------------------- chain: GENCO inference + SciPy Newton
def chain(a):
    import engine  # Julia before torch
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.data import Batch
    from gridfm_graphkit.datasets.graph_builder import build_hetero_data
    here, net, raw = paths(a.grid)
    engine.WORK = here
    nw = newton_module(here)
    machine()
    log(f"=== chain {net}: torch {torch.__version__}, {torch.get_num_threads()} CPU threads, mps {torch.backends.mps.is_available()}; {a.states} states")
    load_evidence("start")
    ids = np.load(OUT / f"genco_pred_{net}.npz")["ids"][: a.states]
    tabs = read_tables(raw, ids)
    rows = []

    def add(variant, item, device, batch, vals, note=""):
        vals = np.asarray(vals, float)
        rows.append({"network": net, "variant": variant, "item": item, "device": device, "batch": batch,
                     "ms_per_state": float(np.median(vals)), "ms_min": float(vals.min()), "n": len(vals), "note": note})
        log(f"  {variant:<5} {item:<52} {device:<3} batch {batch:>2}: {np.median(vals):8.2f} ms/state median (min {vals.min():.2f}, n {len(vals)}) {note}")

    def sync(device):
        if device == "mps":
            torch.mps.synchronize()

    for variant in ("tiny", "small"):
        genco = engine.Genco(net, variant)
        graphs, t_prep = [], []
        for s in ids:
            bus, gen, branch = state_of(tabs, s)
            t0 = time.perf_counter()
            data = build_hetero_data(bus, gen, branch)
            genco.normalizer.transform(data)
            graphs.append(genco.transforms(data))
            t_prep.append(1e3 * (time.perf_counter() - t0))
        add(variant, "graph build + normalise + transforms", "cpu", 1, t_prep[1:])
        for device in ("cpu", "mps"):
            if device == "mps" and not torch.backends.mps.is_available():
                continue
            model = genco.model.to(device)
            for bs in (1, 4, 16):
                batch = Batch.from_data_list(graphs[:bs]).to(device)
                with torch.no_grad():
                    for _ in range(3):
                        model(batch)
                    sync(device)
                    ts = []
                    for _ in range(7):
                        t0 = time.perf_counter()
                        model(batch)
                        sync(device)
                        ts.append(1e3 * (time.perf_counter() - t0) / bs)
                add(variant, "GENCO forward", device, bs, ts)
                del batch
            genco.model.to("cpu")
        e2e, build, step1, step2, full, its = [], [], [], [], [], []
        for s in ids:
            bus, gen, branch = state_of(tabs, s)
            t0 = time.perf_counter()
            bp, _ = genco.solve(bus, gen, branch)
            e2e.append(1e3 * (time.perf_counter() - t0))
            t0 = time.perf_counter()
            prob = nw.make_problem(bus, branch)
            build.append(1e3 * (time.perf_counter() - t0))
            v = prob.start(vm_pq=bp.Vm.to_numpy(float), va=np.radians(bp.Va.to_numpy(float)))
            for store, kw in ((step1, {"max_it": 1}), (step2, {"max_it": 2}), (full, {})):
                best = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    r = prob.newton(v, **kw)
                    best.append(1e3 * (time.perf_counter() - t0))
                store.append(min(best))
            its.append(r["iterations"] if r["converged"] else np.nan)
        add(variant, "GENCO end-to-end single state (graph+model+inverse+flows)", "cpu", 1, e2e[1:])
        add(variant, "SciPy Newton problem build", "cpu", 1, build[1:])
        add(variant, "SciPy one Newton step from GENCO", "cpu", 1, step1[1:], "min of 3 per state")
        add(variant, "SciPy two Newton steps from GENCO", "cpu", 1, step2[1:], "min of 3 per state")
        add(variant, "SciPy full Newton from GENCO", "cpu", 1, full[1:], f"min of 3 per state; {np.nanmean(its):.2f} iterations, converged {int(np.isfinite(its).sum())}/{len(its)}")
    load_evidence("end")
    pd.DataFrame(rows).to_csv(OUT / f"chain_{net}.csv", index=False)
    log(f"saved {OUT / f'chain_{net}.csv'}")


# ----------------------------------------------------------------------------- summary
def summary(a):
    import numpy as np
    import pandas as pd
    rows = []
    for grid in ("2000", "10000"):
        net = f"case{grid}_goc"
        f = OUT / f"solver_starts_{net}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        report(df, net)
        for name in [c[0] for c in CONFIGS]:
            c = df[df[f"conv_{name}"]]
            rows.append({"network": net, "kind": "PowerModels solve", "item": name, "n": len(df), "converged": len(c), "pm_flag_converged": int(df[f"pmconv_{name}"].sum()),
                         "it_mean": c[f"it_{name}"].mean(), "it_median": c[f"it_{name}"].median(), "it_max": c[f"it_{name}"].max(),
                         "jac_evals_mean": c[f"jcalls_{name}"].mean(),
                         "ms_median": c[f"ms_min_{name}"].median(), "ms_p90": c[f"ms_min_{name}"].quantile(.9), "ms_median_of_median3": c[f"ms_med_{name}"].median(),
                         "start_residual_pu_median": df[f"res0_{name}"].median(), "max_dv_vs_stored": c[f"dv_{name}"].max(),
                         "same_solution": int((c[f"dv_{name}"] < 1e-5).sum())})
        for col, item in (("dict_ms", "data dict (datakit _gfm_state)"), ("inst_ms", "instantiate_pf_data (Ybus + Jacobian pattern)"),
                          ("stored_runtime_ms", "datakit stored AC runtime (generation, loaded machine)")):
            rows.append({"network": net, "kind": "PowerModels overhead", "item": col + ": " + item, "n": len(df), "ms_median": df[col].median(), "ms_p90": df[col].quantile(.9)})
        g = OUT / f"chain_{net}.csv"
        if g.exists():
            ch = pd.read_csv(g)
            for _, r in ch.iterrows():
                rows.append({"network": net, "kind": "chain", "item": f"{r.variant}: {r['item']} [{r.device}, batch {r.batch}]", "n": r.n, "ms_median": r.ms_per_state, "note": r.note})
            pm = {name: df[df[f"conv_{name}"]][f"ms_min_{name}"].median() for name in [c[0] for c in CONFIGS]}

            def get(variant, item, device, batch):
                m = ch[(ch.variant == variant) & (ch["item"].str.startswith(item)) & (ch.device == device) & (ch.batch == batch)]
                return float(m.ms_per_state.iloc[0]) if len(m) else np.nan

            for variant in ("tiny", "small"):
                graph = get(variant, "graph build", "cpu", 1)
                for device in ("cpu", "mps"):
                    for bs in (1, 4, 16):
                        fwd = get(variant, "GENCO forward", device, bs)
                        if not np.isfinite(fwd):
                            continue
                        for sfx in ("", "_p"):
                            g_, d_, f_, c_ = (pm[k + sfx] for k in (f"{variant}_qc", "default", "flat_q0", "dc_qc"))
                            total = graph + fwd + g_
                            rows.append({"network": net, "kind": "chain total", "item": f"graph + GENCO {variant} forward [{device}, batch {bs}] + PowerModels solve from {variant}_qc{sfx}",
                                         "ms_median": total, "note": f"graph {graph:.2f} + forward {fwd:.2f} + solve {g_:.2f}; vs default{sfx} {d_:.2f} "
                                         f"(x{total / d_:.2f}), flat_q0{sfx} {f_:.2f} (x{total / f_:.2f}), dc_qc{sfx} {c_:.2f} (x{total / c_:.2f})"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_colwidth", 200, "display.max_rows", 500, "display.float_format", "{:.3g}".format):
        print(out.to_string(index=False))
    log(f"saved {OUT / 'summary.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["prep", "solve", "chain", "summary"])
    p.add_argument("--grid", default="2000")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--states", type=int, default=17)
    p.add_argument("--tag", default="")
    p.add_argument("--maxit", type=int, default=MAXIT)
    a = p.parse_args()
    MAXIT = a.maxit
    OUT.mkdir(parents=True, exist_ok=True)
    {"prep": prep, "solve": solve, "chain": chain, "summary": summary}[a.phase](a)

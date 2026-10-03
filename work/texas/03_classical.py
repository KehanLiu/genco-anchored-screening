"""Classical AC / DC power flow on Texas-2k on this machine, and new states built from the case file.

Part A  rebuilds official test states (test_2 partition 0) in a datakit Network from the case
        file (loads, dispatch, PV setpoints, perturbed r/x, outages), solves them with datakit's
        PowerModels AC and DC power flow from a flat start, checks the result against the stored
        AC/DC solution and times each solve.
Part B  the IEEE-118 engine's own path (engine.Engine, unmodified): base case -> load scaling ->
        AC-OPF dispatch -> random line outages -> AC + DC power flow + GENCO, with the engine's
        timings and the same accuracy metrics as 02_eval_testset.py.

    work/texas/tmo 3000 .venv/bin/python -u work/texas/03_classical.py [N_TEST_STATES] [N_OUTAGE_SETS]
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import engine  # noqa: E402  (loads Julia before torch)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gridfm_datakit.network import Network, load_net_from_file  # noqa: E402
from gridfm_datakit.process.solvers import run_dcpf, run_pf  # noqa: E402
from gridfm_datakit.utils.idx_brch import BR_R, BR_STATUS, BR_X, F_BUS, T_BUS  # noqa: E402
from gridfm_datakit.utils.idx_bus import BUS_TYPE, PQ, PV, REF, VA, VM  # noqa: E402
from gridfm_datakit.utils.idx_gen import GEN_STATUS, PG, QG, VG  # noqa: E402

NET = "Texas2k_case1_2016summerpeak"
CASE = HERE / f"{NET}.m"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stats(x):
    x = np.asarray(x) * 1e3
    return f"median {np.median(x):.1f} ms, mean {x.mean():.1f}, p90 {np.quantile(x, 0.9):.1f}, min {x.min():.1f} (n={len(x)})"


def part_a(jl, n_states):
    base = load_net_from_file(str(CASE))
    tables = {t: pd.read_parquet(HERE / "data" / "test_2" / f"{t}_p0.parquet") for t in ("bus", "gen", "branch")}
    # The case file and the dataset must describe the same grid in the same order.
    br0 = tables["branch"][tables["branch"].scenario == 0].sort_values("idx")
    assert len(br0) == len(base.branches)
    same = (br0.from_bus.to_numpy() == base.branches[:, F_BUS]) & (br0.to_bus.to_numpy() == base.branches[:, T_BUS])
    swapped = (br0.from_bus.to_numpy() == base.branches[:, T_BUS]) & (br0.to_bus.to_numpy() == base.branches[:, F_BUS])
    assert (same | swapped).all()
    # Two plain lines (tap 1, shift 0) are stored with from/to reversed; physically identical,
    # so their stored "pf" is our "pt".
    print(f"[A] branch order matches the case file; {int(swapped.sum())} plain lines stored with from/to swapped")
    rows, t_ac, t_dc = [], [], []
    for s in sorted(tables["bus"].scenario.unique())[:n_states]:
        bus = tables["bus"][tables["bus"].scenario == s].sort_values("bus")
        gen = tables["gen"][tables["gen"].scenario == s].sort_values("idx")
        br = tables["branch"][tables["branch"].scenario == s].sort_values("idx")
        net = base.copy_for_perturbation()
        net.buses[:, BUS_TYPE] = np.select([bus.REF == 1, bus.PV == 1], [REF, PV], PQ)
        net.Pd, net.Qd = bus.Pd.to_numpy(), bus.Qd.to_numpy()
        net.gens[:, GEN_STATUS] = gen.in_service.to_numpy()
        net.gens[:, PG] = gen.p_mw.to_numpy()
        net.gens[:, QG] = gen.q_mvar.to_numpy()
        vm_set = bus.Vm.to_numpy()
        net.gens[:, VG] = vm_set[gen.bus.to_numpy()]
        # Flat start: PQ buses at 1 p.u., all angles at 0; PV/REF buses at their setpoint.
        net.buses[:, VM] = np.where(bus.PQ == 1, 1.0, vm_set)
        net.buses[:, VA] = 0.0
        net.branches[:, BR_R] = br.r.to_numpy()
        net.branches[:, BR_X] = br.x.to_numpy()
        net.branches[:, BR_STATUS] = br.br_status.to_numpy()
        try:
            start = time.perf_counter()
            ac = run_pf(net, jl, fast=True)
            t1 = time.perf_counter() - start
            start = time.perf_counter()
            dc = run_dcpf(net, jl, fast=True)
            t2 = time.perf_counter() - start
        except RuntimeError as err:
            rows.append({"state": s, "ok": False, "err": str(err)[:120]})
            continue
        t_ac.append(t1)
        t_dc.append(t2)
        sol = ac["solution"]
        n = len(bus)
        vm = np.array([sol["bus"][str(base.reverse_bus_index_mapping[i])]["vm"] for i in range(n)])
        va = np.degrees([sol["bus"][str(base.reverse_bus_index_mapping[i])]["va"] for i in range(n)])
        live = br.br_status.to_numpy() == 1
        pf = np.array([sol["branch"][str(k + 1)]["pt" if swapped[k] else "pf"] * 100 if live[k] else np.nan for k in range(len(br))])
        dsol = dc["solution"]
        va_dc = np.degrees([dsol["bus"][str(base.reverse_bus_index_mapping[i])]["va"] for i in range(n)])
        rows.append({
            "state": s, "ok": True, "n_out": int((~live).sum()),
            "vm_diff": np.abs(vm - bus.Vm.to_numpy()).max(),
            "va_diff_deg": np.abs((va - va[bus.REF.to_numpy() == 1][0]) - (bus.Va - bus.Va[bus.REF == 1].iloc[0]).to_numpy()).max(),
            "pf_diff_mw": np.nanmax(np.abs(pf - br.pf.to_numpy())),
            "va_dc_diff_deg": np.abs((va_dc - va_dc[bus.REF.to_numpy() == 1][0]) - (bus.Va_dc - bus.Va_dc[bus.REF == 1].iloc[0]).to_numpy()).max(),
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "rebuild_test_states.csv", index=False)
    ok = res[res.ok]
    print(f"\n[A] rebuilt {len(res)} official test states from the case file; AC converged on {len(ok)}")
    print(f"    vs stored AC: max |dVm| {ok.vm_diff.max():.2e} p.u., max |dVa| {ok.va_diff_deg.max():.2e} deg, "
          f"max |dPflow| {ok.pf_diff_mw.max():.2e} MW; vs stored DC: max |dVa_dc| {ok.va_dc_diff_deg.max():.2e} deg")
    print(f"    AC PF (PowerModels compute_ac_pf, flat start, incl. data hand-off): {stats(t_ac[1:])}")
    print(f"    DC PF (PowerModels compute_dc_pf, incl. data hand-off):            {stats(t_dc[1:])}")
    return t_ac[1:], t_dc[1:]


def in_service_gens_only(net):
    """engine.py assumes every generator is in service (true for the IEEE cases); the Texas case has
    out-of-service units, so they are dropped from the case before the engine sees it."""
    on = net.mpc["gen"][:, GEN_STATUS] == 1
    mpc = dict(net.mpc, gen=net.mpc["gen"][on], gencost=net.mpc["gencost"][on])
    log(f"dropping {int((~on).sum())} out-of-service generators of {len(on)}")
    return Network(mpc)


def part_b(n_sets):
    engine.WORK = HERE
    engine.load_net_from_pglib = lambda name: in_service_gens_only(load_net_from_file(str(CASE)))
    engine.layout = lambda network, n, edges: np.zeros((n, 2))  # no coordinates needed (and no download)
    log("building engine (initial AC-OPF dispatch at load x1.0)")
    start = time.perf_counter()
    eng = engine.Engine(NET, "base")
    log(f"engine ready in {time.perf_counter() - start:.1f} s")
    rng = np.random.default_rng(0)
    n_br = len(eng.static["branches"])
    rated = [b["id"] for b in eng.static["branches"] if b["rate_a"] is not None and not b["bridge"]]
    sets = [()] + [tuple(rng.choice(rated, 1)) for _ in range(n_sets)] + [tuple(rng.choice(rated, 2, replace=False)) for _ in range(n_sets)]
    frames, times, fails = [], [], {}
    levels, opf_s = [], {}
    for scale in (0.7, 0.8, 0.9, 1.0, 1.05):
        # A failed AC-OPF dispatch is not cached by the engine and would be retried (~15-100 s) for
        # every outage set, so each load level is checked once first.
        start = time.perf_counter()
        try:
            eng._dispatch(scale)
            opf_s[scale] = round(time.perf_counter() - start, 1)
        except engine.SolveError as err:
            log(f"load x{scale}: {err.message} -> level skipped")
            fails[f"opf_failed@x{scale}"] = len(sets)
            continue
        levels.append(scale)
        for outages in sets:
            try:
                r = eng.solve(scale, outages)
            except engine.SolveError as err:
                fails[err.reason] = fails.get(err.reason, 0) + 1
                continue
            times.append({"load_scale": scale, "n_out": len(outages), **r["timing_ms"]})
            lines = pd.DataFrame(r["branches"])
            lines = lines[lines.in_service]
            buses = pd.DataFrame(r["buses"])
            lines["state"] = f"{scale}:{outages}"
            lines["res_max"] = buses.res_p.abs().max()
            lines["res_mean"] = buses.res_p.abs().mean()
            frames.append(lines)
        log(f"load x{scale}: {len(frames)} states solved so far, failures {fails}")
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(OUT / "new_states_lines.parquet", index=False)
    tm = pd.DataFrame(times)
    tm.to_csv(OUT / "new_states_timing.csv", index=False)
    n_states = data.state.nunique()
    print(f"\n[B] new states from the case file via engine.Engine: {n_states} solved "
          f"({len(levels)} load levels {levels} x {len(sets)} outage sets: N-0, {n_sets} N-1, {n_sets} N-2), failures {fails}")
    print(f"    AC-OPF dispatch per load level (s, x1.0 was solved at engine start): {opf_s}")
    for c in ("ac", "dc", "genco"):
        x = tm[c].iloc[1:] / 1e3
        print(f"    engine timing {c:>5}: {stats(x)}")
    r = data[data.loading_ac.notna()]
    real = r.loading_ac > engine.OVERLOAD
    for m in ("genco", "dc"):
        perr = (data[f"p_{m}"] - data.p_ac).abs()
        lerr = (r[f"loading_{m}"] - r.loading_ac).abs() * 100
        called = r[f"loading_{m}"] > engine.OVERLOAD
        print(f"    {m:>5}: P flow err mean {perr.mean():.2f} / max {perr.max():.1f} MW; loading err median {lerr.median():.2f} / p99 {lerr.quantile(0.99):.1f} pp; "
              f"overloads real {int(real.sum())}, missed {int((real & ~called).sum())}, false {int((~real & called).sum())}")
    st = data.groupby("state")[["res_mean", "res_max"]].first()
    print(f"    GENCO P residual: mean over buses {st.res_mean.mean():.2f} MW, max {st.res_max.max():.1f} MW")


def main():
    n_states = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_sets = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    jl = engine.init_julia(max_iter=200)
    part_a(jl, n_states)
    part_b(n_sets)


if __name__ == "__main__":
    main()

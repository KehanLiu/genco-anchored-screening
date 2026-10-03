"""Accuracy and speed of the published GENCO Texas-2k checkpoint on official test states.

Reads the slice fetched by 01_fetch.py (default: test_2 = N-2 test set, partition 0, 200
states), runs GENCO on every state the same way engine.Genco does for IEEE 118, and
compares GENCO and DC power flow (stored in the same files) against the stored AC solution.
Then times GENCO single-state and batched on CPU and MPS.

    work/texas/tmo 1800 .venv/bin/python -u work/texas/02_eval_testset.py [LEVEL] [PARTITION ...]

Writes work/texas/out/testset_lines_<level>.parquet, testset_buses_<level>.parquet,
timing_genco.csv and prints the summary tables.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import engine  # noqa: E402  (loads Julia before torch, as everywhere else in the project)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from gridfm_graphkit.datasets.graph_builder import build_hetero_data  # noqa: E402

NET = "Texas2k_case1_2016summerpeak"
OVERLOAD = engine.OVERLOAD  # 1.001, same threshold as the IEEE 118 engine
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(level, parts):
    tables = {}
    for t in ("bus", "gen", "branch", "runtime"):
        tables[t] = pd.concat([pd.read_parquet(HERE / "data" / level / f"{t}_p{p}.parquet") for p in parts])
    return tables


def state_tables(tables, s):
    bus = tables["bus"][tables["bus"].scenario == s].sort_values("bus").reset_index(drop=True)
    gen = tables["gen"][tables["gen"].scenario == s].sort_values("idx").reset_index(drop=True)
    branch = tables["branch"][tables["branch"].scenario == s].sort_values("idx").reset_index(drop=True)
    q = gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
    bus = bus.merge(q, on="bus", how="left").fillna(0)
    return bus, gen, branch


def bus_p_injection(bus, branch, vm, va_deg):
    """Active power leaving each bus through in-service branches and shunts (MW), AC equations."""
    v = vm * np.exp(1j * np.radians(va_deg))
    br = branch[branch.br_status == 1]
    f, t = br.from_bus.to_numpy(), br.to_bus.to_numpy()
    y = {k: br[f"{k}_r"].to_numpy() + 1j * br[f"{k}_i"].to_numpy() for k in ("Yff", "Yft", "Ytf", "Ytt")}
    sf = v[f] * np.conj(y["Yff"] * v[f] + y["Yft"] * v[t]) * 100
    st = v[t] * np.conj(y["Ytt"] * v[t] + y["Ytf"] * v[f]) * 100
    p = np.zeros(len(bus))
    np.add.at(p, f, sf.real)
    np.add.at(p, t, st.real)
    return p + bus.GS.to_numpy() * vm**2


def evaluate_state(genco, bus, gen, branch):
    start = time.perf_counter()
    bus_pred, br_pred = genco.solve(bus, gen, branch)
    t_solve = time.perf_counter() - start

    live = (branch.br_status == 1).to_numpy()
    n_live = int(live.sum())
    rate = branch.rate_a.where(branch.rate_a > 0).to_numpy()
    lines = pd.DataFrame({"idx": branch.idx, "rate": rate})[live].reset_index(drop=True)
    lines["p_ac"] = branch.pf[live].to_numpy()
    lines["p_genco"] = br_pred.p[:n_live].to_numpy()
    lines["p_dc"] = branch.pf_dc[live].to_numpy()
    s_ac = np.maximum(np.hypot(branch.pf, branch.qf), np.hypot(branch.pt, branch.qt))[live].to_numpy()
    lines["load_ac"] = s_ac / lines.rate
    lines["load_genco"] = np.maximum(br_pred.loading[:n_live].to_numpy(), br_pred.loading[n_live:].to_numpy())
    lines.loc[lines.rate.isna(), "load_genco"] = np.nan
    lines["load_dc"] = lines.p_dc.abs() / lines.rate

    # Active power-balance residual per bus, MW. GENCO: graphkit's own residual (as engine.py);
    # cross-checked with the numpy version. DC: AC equations evaluated at Vm = 1, Va = Va_dc with
    # the DC dispatch (as the paper's dc_bus_residuals). AC: sanity check, should be ~0.
    pd_ = bus.Pd.to_numpy()
    buses = pd.DataFrame({"bus": bus.bus})
    buses["res_genco"] = bus_pred.res_p.to_numpy()
    buses["res_genco_np"] = bus_pred.Pg.to_numpy() - pd_ - bus_p_injection(bus, branch, bus_pred.Vm.to_numpy(), bus_pred.Va.to_numpy())
    buses["res_dc"] = bus.Pg_dc.to_numpy() - pd_ - bus_p_injection(bus, branch, np.ones(len(bus)), bus.Va_dc.to_numpy())
    buses["res_ac"] = bus.Pg.to_numpy() - pd_ - bus_p_injection(bus, branch, bus.Vm.to_numpy(), bus.Va.to_numpy())
    buses["vm_err"] = (bus_pred.Vm - bus.Vm).to_numpy()
    buses["va_err_deg"] = (bus_pred.Va - bus.Va).to_numpy()
    return lines, buses, t_solve


def summarize(lines, buses, n_states):
    rated = lines[lines.rate.notna()]
    real = rated.load_ac > OVERLOAD
    rows = []
    for m, name in (("genco", "GENCO base"), ("dc", "DC power flow")):
        perr = (lines[f"p_{m}"] - lines.p_ac).abs()
        res = buses[f"res_{m}"].abs()
        lerr = (rated[f"load_{m}"] - rated.load_ac).abs() * 100
        called = rated[f"load_{m}"] > OVERLOAD
        rows.append({
            "method": name,
            "P flow err mean MW": perr.mean(), "P flow err max MW": perr.max(),
            "P residual mean MW": res.mean(), "P residual max MW": res.max(),
            "loading err median pp": lerr.median(), "loading err p99 pp": lerr.quantile(0.99),
            "real overloads": int(real.sum()), "missed": int((real & ~called).sum()),
            "false": int((~real & called).sum()),
        })
    table = pd.DataFrame(rows).set_index("method")
    print(f"\n{n_states} states, {len(lines):,} in-service line readings ({len(rated):,} rated), "
          f"{len(buses):,} bus readings; overload = loading > {OVERLOAD}")
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(table.T)
    print(f"sanity: AC ground-truth residual mean/max {buses.res_ac.abs().mean():.4f}/{buses.res_ac.abs().max():.4f} MW; "
          f"GENCO residual graphkit vs numpy max diff {(buses.res_genco - buses.res_genco_np).abs().max():.4f} MW")
    print(f"GENCO voltage: |Vm err| mean {buses.vm_err.abs().mean():.5f} max {buses.vm_err.abs().max():.4f} p.u.; "
          f"|Va err| mean {buses.va_err_deg.abs().mean():.4f} max {buses.va_err_deg.abs().max():.3f} deg")
    per_state = lines.groupby("state").apply(lambda d: pd.Series({
        "genco": (d.p_genco - d.p_ac).abs().max(), "dc": (d.p_dc - d.p_ac).abs().max()}), include_groups=False)
    print(f"per-state worst P-flow error, median over states: GENCO {per_state.genco.median():.1f} MW, DC {per_state.dc.median():.1f} MW")
    return table


def prepare_graph(genco, bus, gen, branch):
    data = build_hetero_data(bus, gen, branch)
    genco.normalizer.transform(data)
    return genco.transforms(data)


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def time_batched(genco, graphs, device, sizes, reps=5):
    model = genco.model.to(device)
    rows = []
    for bs in sizes:
        if bs > len(graphs):
            continue
        try:
            batch = Batch.from_data_list(graphs[:bs]).to(device)
            with torch.no_grad():
                for _ in range(2):  # warm-up
                    model(batch)
                sync(device)
                times = []
                for _ in range(reps):
                    start = time.perf_counter()
                    model(batch)
                    sync(device)
                    times.append(time.perf_counter() - start)
            t = float(np.median(times))
            rows.append({"device": device, "batch": bs, "ms_per_batch": 1e3 * t, "ms_per_state": 1e3 * t / bs, "status": "ok"})
            log(f"  {device} batch {bs:>4}: {1e3 * t:9.1f} ms/batch  {1e3 * t / bs:8.2f} ms/state")
        except Exception as err:  # noqa: BLE001  (e.g. op not implemented on MPS, out of memory)
            rows.append({"device": device, "batch": bs, "status": f"{type(err).__name__}: {str(err)[:200]}"})
            log(f"  {device} batch {bs}: FAILED {type(err).__name__}: {str(err)[:200]}")
            break
    genco.model.to("cpu")
    return rows


def main():
    level = sys.argv[1] if len(sys.argv) > 1 else "test_2"
    parts = [int(p) for p in sys.argv[2:]] or [0]
    log(f"torch {torch.__version__}, threads {torch.get_num_threads()}, mps available {torch.backends.mps.is_available()}")
    engine.WORK = HERE  # Genco reads WORK / pf_<network>_<variant>
    genco = engine.Genco(NET, "base")
    log(f"model loaded: {sum(p.numel() for p in genco.model.parameters()):,} parameters")
    tables = load(level, parts)
    scenarios = sorted(tables["bus"].scenario.unique())
    log(f"{level} partitions {parts}: {len(scenarios)} states")

    all_lines, all_buses, t_single = [], [], []
    for i, s in enumerate(scenarios):
        bus, gen, branch = state_tables(tables, s)
        lines, buses, t = evaluate_state(genco, bus, gen, branch)
        lines["state"], buses["state"] = s, s
        lines["n_out"] = int((branch.br_status == 0).sum())
        all_lines.append(lines)
        all_buses.append(buses)
        t_single.append(t)
        if i % 25 == 0:
            log(f"state {i}/{len(scenarios)}: GENCO end-to-end {1e3 * t:.0f} ms, max |dP| {(lines.p_genco - lines.p_ac).abs().max():.1f} MW")
    lines = pd.concat(all_lines, ignore_index=True)
    buses = pd.concat(all_buses, ignore_index=True)
    lines.to_parquet(OUT / f"testset_lines_{level}.parquet", index=False)
    buses.to_parquet(OUT / f"testset_buses_{level}.parquet", index=False)
    summarize(lines, buses, len(scenarios))
    base_out = tables["branch"].groupby("scenario").br_status.apply(lambda s: int((s == 0).sum()))
    print(f"out-of-service branches per state: {base_out.value_counts().sort_index().to_dict()}")

    # ---- speed ----
    t_single = np.array(t_single[1:])  # first call includes lazy initialisation
    print(f"\nGENCO single state, CPU, end-to-end (graph build + normalise + model + inverse + residual): "
          f"median {1e3 * np.median(t_single):.1f} ms, p90 {1e3 * np.quantile(t_single, 0.9):.1f} ms over {len(t_single)} states")
    log("building graphs for batched timing")
    graphs = []
    t_prep = []
    for s in scenarios[:128]:
        bus, gen, branch = state_tables(tables, s)
        start = time.perf_counter()
        graphs.append(prepare_graph(genco, bus, gen, branch))
        t_prep.append(time.perf_counter() - start)
    print(f"graph build + normalise + transforms per state (CPU): median {1e3 * np.median(t_prep):.1f} ms")

    # Batched predictions must equal one-by-one predictions.
    with torch.no_grad():
        one = genco.model(Batch.from_data_list(graphs[:1]))["bus"]
        many = genco.model(Batch.from_data_list(graphs[:4]))["bus"][: one.shape[0]]
    print(f"batched vs single prediction, max abs diff (normalised units): {(one - many).abs().max().item():.2e}")

    sizes = [1, 4, 16, 32, 64, 128]
    rows = time_batched(genco, graphs, "cpu", sizes)
    if torch.backends.mps.is_available():
        rows += time_batched(genco, graphs, "mps", sizes)
        try:  # does MPS give the same answer?
            m = genco.model.to("mps")
            with torch.no_grad():
                out_mps = m(Batch.from_data_list(graphs[:1]).to("mps"))["bus"].cpu()
            print(f"MPS vs CPU prediction, max abs diff (normalised units): {(out_mps - one).abs().max().item():.2e}")
        except Exception as err:  # noqa: BLE001
            print(f"MPS check failed: {err}")
        genco.model.to("cpu")
    timing = pd.DataFrame(rows)
    timing.to_csv(OUT / "timing_genco.csv", index=False)
    print("\nGENCO model forward only (graphs pre-built, in memory):")
    print(timing.to_string(index=False))
    rt = tables["runtime"]
    print(f"\nstored datakit solver runtime for these states (generated on gridfm's machine): "
          f"AC median {1e3 * rt.ac.median():.0f} ms, DC median {1e3 * rt.dc.median():.0f} ms")


if __name__ == "__main__":
    main()

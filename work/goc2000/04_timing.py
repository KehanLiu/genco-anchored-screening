"""Per-state wall-clock on this machine for this folder's grid (folder name -> case<N>_goc).

  GENCO forward pass (graphs pre-built in memory), CPU and MPS, batch sizes 1/4/16/64  (pattern: time_batched in work/texas/02_eval_testset.py)
  GENCO end-to-end single state on CPU (graph build + normalise + model + inverse + flows), and graph build alone
  SciPy Newton (analysis_warmstart.Problem via 02_newton.make_problem): problem build, one step from GENCO,
      full solve to 1e-6 MW from flat and from GENCO
  datakit / PowerModels AC and DC power flow: the runtime columns stored at generation time (runtime_data.parquet;
      measured while datagen ran with several processes)

    work/texas/tmo 1800 .venv/bin/python -u work/goc2000/04_timing.py [--variant tiny] [--states 64] [--newton 40]
Writes timing.csv (timing_<variant>.csv for other variants).
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
NET = f"case{HERE.name.removeprefix('goc')}_goc"

import engine  # noqa: E402  (Julia before torch)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402
from gridfm_graphkit.datasets.graph_builder import build_hetero_data  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


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
                for _ in range(2):
                    model(batch)
                sync(device)
                times = []
                for _ in range(reps):
                    start = time.perf_counter()
                    model(batch)
                    sync(device)
                    times.append(time.perf_counter() - start)
            t = float(np.median(times))
            rows.append({"item": "GENCO forward", "device": device, "batch": bs, "ms_per_state": 1e3 * t / bs, "n": reps, "note": "ok"})
            log(f"  {device} batch {bs:>4}: {1e3 * t:9.1f} ms/batch  {1e3 * t / bs:8.2f} ms/state")
            del batch
        except Exception as err:  # noqa: BLE001
            rows.append({"item": "GENCO forward", "device": device, "batch": bs, "ms_per_state": np.nan, "n": 0, "note": f"FAILED {type(err).__name__}: {str(err)[:200]}"})
            log(f"  {device} batch {bs}: FAILED {type(err).__name__}: {str(err)[:300]}")
            break
    genco.model.to("cpu")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="tiny")
    p.add_argument("--data", default="data")
    p.add_argument("--states", type=int, default=64)
    p.add_argument("--newton", type=int, default=40)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("goc_newton", HERE / "02_newton.py")
    nw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nw)
    engine.WORK = HERE
    genco = engine.Genco(NET, a.variant)
    log(f"{NET} {a.variant}: {sum(p.numel() for p in genco.model.parameters()):,} parameters; torch {torch.__version__}, {torch.get_num_threads()} CPU threads, "
        f"mps available {torch.backends.mps.is_available()}")
    raw = HERE / a.data / NET / "raw"
    ids = np.sort(pd.read_parquet(raw / "runtime_data.parquet").scenario.unique())[: a.states]
    flt = [("scenario", ">=", int(ids[0])), ("scenario", "<=", int(ids[-1]))]
    tabs = {t: {s: d for s, d in pd.read_parquet(raw / f"{t}_data.parquet", filters=flt).groupby("scenario")} for t in ("bus", "gen", "branch")}

    def state(s):
        bus = tabs["bus"][s].sort_values("bus").reset_index(drop=True)
        gen = tabs["gen"][s].sort_values("idx").reset_index(drop=True)
        branch = tabs["branch"][s].sort_values("idx").reset_index(drop=True)
        q = gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
        return bus.merge(q, on="bus", how="left").fillna(0), gen, branch

    rows, graphs, t_prep = [], [], []
    for s in ids:
        bus, gen, branch = state(s)
        t0 = time.perf_counter()
        data = build_hetero_data(bus, gen, branch)
        genco.normalizer.transform(data)
        graphs.append(genco.transforms(data))
        t_prep.append(time.perf_counter() - t0)
    rows.append({"item": "graph build + normalise + transforms (from pandas tables)", "device": "cpu", "batch": 1, "ms_per_state": 1e3 * np.median(t_prep), "n": len(t_prep), "note": "median"})
    log(f"{len(graphs)} graphs built, {1e3 * np.median(t_prep):.1f} ms/state")
    with torch.no_grad():
        one = genco.model(Batch.from_data_list(graphs[:1]))["bus"]
        many = genco.model(Batch.from_data_list(graphs[:4]))["bus"][: one.shape[0]]
    log(f"batched vs single prediction, max abs diff (normalised units): {(one - many).abs().max().item():.2e}")
    sizes = [1, 4, 16, 64]
    rows += time_batched(genco, graphs, "cpu", sizes)
    if torch.backends.mps.is_available():
        rows += time_batched(genco, graphs, "mps", sizes)
        try:
            m = genco.model.to("mps")
            with torch.no_grad():
                gpu = m(Batch.from_data_list(graphs[:1]).to("mps"))["bus"].cpu()
            log(f"MPS vs CPU prediction, max abs diff (normalised units): {(gpu - one).abs().max().item():.2e}")
        except Exception as err:  # noqa: BLE001
            log(f"MPS check failed: {type(err).__name__}: {str(err)[:200]}")
        genco.model.to("cpu")
    else:
        rows.append({"item": "GENCO forward", "device": "mps", "batch": 0, "ms_per_state": np.nan, "n": 0, "note": "MPS not available"})

    t = {k: [] for k in ("e2e", "build", "step", "flat", "genco")}
    its = {"flat": [], "genco": []}
    conv = {"flat": [], "genco": []}
    for s in ids[: a.newton]:
        bus, gen, branch = state(s)
        t0 = time.perf_counter()
        bus_pred, _ = genco.solve(bus, gen, branch)
        t["e2e"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        prob = nw.make_problem(bus, branch)
        t["build"].append(time.perf_counter() - t0)
        v_g = prob.start(vm_pq=bus_pred.Vm.to_numpy(float), va=np.radians(bus_pred.Va.to_numpy(float)))
        v_f = prob.start()
        for _ in range(2):
            t0 = time.perf_counter()
            prob.newton(v_g, max_it=1)
            dt = time.perf_counter() - t0
        t["step"].append(dt)
        for name, v0 in (("flat", v_f), ("genco", v_g)):
            t0 = time.perf_counter()
            r = prob.newton(v0)
            t[name].append(time.perf_counter() - t0)
            conv[name].append(r["converged"])
            its[name].append(r["iterations"])
    med = {k: 1e3 * float(np.median(v[1:])) for k, v in t.items()}
    n = len(t["e2e"]) - 1
    rows += [
        {"item": "GENCO end-to-end single state (graph build + model + inverse + flows)", "device": "cpu", "batch": 1, "ms_per_state": med["e2e"], "n": n, "note": "median"},
        {"item": "Newton problem build (Ybus from stored admittances)", "device": "cpu", "batch": 1, "ms_per_state": med["build"], "n": n, "note": "median"},
        {"item": "one Newton step from GENCO (SciPy)", "device": "cpu", "batch": 1, "ms_per_state": med["step"], "n": n, "note": "median"},
        {"item": "full Newton from flat (SciPy)", "device": "cpu", "batch": 1, "ms_per_state": med["flat"], "n": n, "note": f"median; {np.mean(its['flat']):.2f} iterations; converged {sum(conv['flat'])}/{len(conv['flat'])} (not converged = stopped at 30 iterations)"},
        {"item": "full Newton from GENCO (SciPy)", "device": "cpu", "batch": 1, "ms_per_state": med["genco"], "n": n, "note": f"median; {np.mean(its['genco']):.2f} iterations; converged {sum(conv['genco'])}/{len(conv['genco'])}"},
    ]
    rt = pd.read_parquet(raw / "runtime_data.parquet")
    for c, name in (("ac", "datakit PowerModels AC power flow (stored runtime, pf_fast)"), ("dc", "datakit PowerModels DC power flow (stored runtime, dcpf_fast)")):
        rows.append({"item": name, "device": "cpu", "batch": 1, "ms_per_state": 1e3 * rt[c].median(), "n": len(rt), "note": f"median, measured during generation under load; mean {1e3 * rt[c].mean():.1f}, p90 {1e3 * rt[c].quantile(.9):.1f}"})
    out = pd.DataFrame(rows)
    out.insert(0, "variant", a.variant)
    out.insert(0, "network", NET)
    csv = HERE / ("timing.csv" if a.variant == "tiny" else f"timing_{a.variant}.csv")
    out.to_csv(csv, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.float_format", "{:.2f}".format):
        print(out.drop(columns=["network"]).to_string(index=False))
    print(f"saved {csv}")


if __name__ == "__main__":
    main()

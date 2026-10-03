"""GENCO Tiny (released case500_goc checkpoint, trained on N-1) on the datakit scenarios of datagen.yaml.

Per-line table: state, n_out (branches removed on top of the 5 the pglib case already has out), line,
ac / genco / dc loading as a fraction of rate_a (in-service rated lines only), plus load_idx.
Loading as in engine.py / work/texas/02_eval_testset.py: AC and GENCO = max apparent power of both
ends / rate_a, DC = |P_dc| / rate_a.

    work/texas/tmo 7200 .venv/bin/python -u work/goc500/01_eval.py [--seed 0] [--data data] [--procs 5]
"""
import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
NET = "case500_goc"


def variant_of(seed):
    return "tiny" + (f"_seed{seed}" if seed else "")


def fetch(seed):
    import yaml
    import run_pretrained
    folder = HERE / f"pf_{NET}_{variant_of(seed)}"
    prefix = f"{NET}/tiny/seed{seed}"
    for f in ("best_model_state_dict.pt", "normalizer_stats.pt"):
        run_pretrained.place("gridfm/genco-pf-datakit", f"{prefix}/{f}", folder / "model" / f)
    run_pretrained.place("gridfm/genco-pf-datakit", f"{prefix}/{NET}_tiny_seed{seed}.yaml", folder / "released_training.yaml")
    cfg = folder / "config.yaml"
    if not cfg.exists():
        cfg.write_text(yaml.safe_dump(run_pretrained.config(NET, "tiny", "cpu"), sort_keys=False))


def worker(job):
    seed, raw, scenarios, always_out, wid = job
    import engine  # Julia before torch
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    engine.WORK = HERE
    genco = engine.Genco(NET, variant_of(seed))
    lo, hi = int(scenarios[0]), int(scenarios[-1])
    flt = [("scenario", ">=", lo), ("scenario", "<=", hi)]
    tabs = {t: {s: d for s, d in pd.read_parquet(raw / f"{t}_data.parquet", filters=flt).groupby("scenario")} for t in ("bus", "gen", "branch")}
    out, times = [], []
    for i, s in enumerate(scenarios):
        bus = tabs["bus"][s].sort_values("bus").reset_index(drop=True)
        gen = tabs["gen"][s].sort_values("idx").reset_index(drop=True)
        branch = tabs["branch"][s].sort_values("idx").reset_index(drop=True)
        q = gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
        bus = bus.merge(q, on="bus", how="left").fillna(0)
        start = time.perf_counter()
        _, br = genco.solve(bus, gen, branch)
        times.append(time.perf_counter() - start)
        live = (branch.br_status == 1).to_numpy()
        n = int(live.sum())
        rate = branch.rate_a.to_numpy()[live]
        s_ac = np.maximum(np.hypot(branch.pf, branch.qf), np.hypot(branch.pt, branch.qt)).to_numpy()[live]
        d = pd.DataFrame({
            "state": s,
            "n_out": int((~live).sum()) - always_out,
            "line": branch.idx.to_numpy()[live],
            "ac": s_ac / rate,
            "genco": np.maximum(br.loading[:n].to_numpy(), br.loading[n:].to_numpy()),
            "dc": np.abs(branch.pf_dc.to_numpy()[live]) / rate,
            "load_idx": int(branch.load_scenario_idx.iloc[0]),
        })
        out.append(d[rate > 0])
        if wid == 0 and i % 250 == 0:
            print(time.strftime("%H:%M:%S"), f"worker 0: {i}/{len(scenarios)}, {1e3 * np.median(times):.0f} ms/state", flush=True)
    return pd.concat(out, ignore_index=True), times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    fetch(a.seed)
    raw = HERE / a.data / NET / "raw"
    status = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "idx", "br_status"])
    scenarios = np.sort(status.scenario.unique())
    always_out = int((status.groupby("idx").br_status.max() == 0).sum())
    if a.limit:
        scenarios = scenarios[: a.limit]
    print(f"{len(scenarios)} scenarios, {always_out} branches out of service in every scenario (pglib case)", flush=True)
    jobs = [(a.seed, raw, chunk, always_out, i) for i, chunk in enumerate(np.array_split(scenarios, a.procs))]
    start = time.perf_counter()
    with get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(worker, jobs)
    lines = pd.concat([r[0] for r in res], ignore_index=True)
    t = np.concatenate([r[1] for r in res])
    out = HERE / (a.out or f"lines_tiny_seed{a.seed}.parquet")
    lines.to_parquet(out, index=False)
    print(f"wrote {out}: {len(lines):,} line readings, {lines.state.nunique()} states, wall {time.perf_counter() - start:.0f}s; "
          f"GENCO end-to-end per state (1 thread, {a.procs} procs in parallel): median {1e3 * np.median(t):.0f} ms")
    print("n_out:", lines.groupby("state").n_out.first().value_counts().sort_index().to_dict())
    rt = pd.read_parquet(raw / "runtime_data.parquet")
    print(f"datakit solver runtime on this machine: AC median {1e3 * rt.ac.median():.1f} ms, DC median {1e3 * rt.dc.median():.1f} ms")


if __name__ == "__main__":
    main()

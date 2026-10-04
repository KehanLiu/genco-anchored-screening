"""Evaluate any GENCO checkpoint folder with the EXISTING pipeline (work/goc<N>/02_newton.py worker, unchanged):
GENCO -> per-line loadings, + one / two Newton steps, + full Newton iteration counts. Only the model folder and the
output location are redirected; nothing under work/goc<N> is written.

    work/texas/tmo 7200 .venv/bin/python -u work/finetune/02_eval.py --grid 500 --model work/finetune/models/pf_case500_goc_ft_seed0 --tag ft_seed0
    ... --data work/finetune/data500_ft --role pool        # label-free scores on the active-learning pool

Writes work/finetune/results/lines_newton_<grid>_<tag>.parquet and states_newton_<grid>_<tag>.parquet
(same columns as work/goc500/lines_newton_tiny_seed0.parquet / states_newton_tiny_seed0.parquet).
"""
import argparse
import importlib.util
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def worker(job):
    grid, model, scenarios, raw, always_out, wid = job
    model = Path(model)
    src = ROOT / "work" / f"goc{grid}" / "02_newton.py"
    spec = importlib.util.spec_from_file_location(f"newton{grid}", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    variant = model.name.removeprefix(f"pf_{mod.NET}_")
    mod.HERE = model.parent                                   # -> engine.WORK inside the original worker
    mod._eval_module = lambda: SimpleNamespace(variant_of=lambda *a: variant)
    inner = (0, raw, scenarios, always_out, wid) if grid == 500 else (0, raw, scenarios, always_out, wid, "x")
    return mod.worker(inner)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, required=True)
    p.add_argument("--model", required=True, help="folder pf_<net>_<variant> with config.yaml and model/")
    p.add_argument("--tag", required=True)
    p.add_argument("--data", default="", help="default: work/goc<grid>/data (the test set)")
    p.add_argument("--role", default="", help="with a fine-tuning dataset: only scenarios of this role in scenario_map.parquet")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    net = f"case{a.grid}_goc"
    data = Path(a.data) if a.data else ROOT / "work" / f"goc{a.grid}" / "data"
    raw = (data / net / "raw").resolve()
    status = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "idx", "br_status"])
    scenarios = np.sort(status.scenario.unique())
    always_out = int((status.groupby("idx").br_status.max() == 0).sum())
    if a.role:
        m = pd.read_parquet(data / "scenario_map.parquet")
        scenarios = np.sort(m.scenario[m.role == a.role].to_numpy())
    if a.limit:
        scenarios = scenarios[: a.limit]
    model = str(Path(a.model).resolve())
    jobs = [(a.grid, model, chunk, raw, always_out, i) for i, chunk in enumerate(np.array_split(scenarios, a.procs))]
    print(f"{net}: {len(scenarios)} scenarios from {raw}, model {model}, {a.procs} procs", flush=True)
    start = time.perf_counter()
    with get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(worker, jobs)
    lines = pd.concat([r[0] for r in res], ignore_index=True)
    st = pd.concat([r[1] for r in res], ignore_index=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    lines.to_parquet(out / f"lines_newton_{a.grid}_{a.tag}.parquet", index=False)
    st.to_parquet(out / f"states_newton_{a.grid}_{a.tag}.parquet", index=False)
    c = st[st.conv_genco]
    print(f"{len(st)} scenarios, {len(lines):,} line readings, wall {time.perf_counter() - start:.0f}s; Newton from GENCO converged {len(c)}/{len(st)}, "
          f"iterations mean {c.it_genco.mean():.3f}; start mismatch median {st.mis0_genco.median():.1f} MW; "
          f"raw loading error p99 {np.quantile((lines.genco - lines.ac).abs(), .99) * 100:.2f} points")


if __name__ == "__main__":
    main()

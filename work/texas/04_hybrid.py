"""The safeguarded hybrid workflow (analysis_hybrid.py) on Texas-2k.

Same rule search, held-out calibration on whole outage sets and DC baseline as on IEEE 118; only the
state sweep differs: Texas has ~3,200 branches, so single and double outages are sampled instead of
enumerated, and only load levels whose AC-OPF dispatch is feasible are used.

On Texas GENCO's residual is set by the load level's dispatch, not by the outage, so the rule search
is also run on the residual *excess* over the N-0 state of the same dispatch (known without AC).

    work/texas/tmo 3600 .venv/bin/python -u work/texas/04_hybrid.py [--single N] [--double N] [--margin M] [--tag T]
    .venv/bin/python work/texas/04_hybrid.py --reuse [--tag T]
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import engine  # noqa: E402  (loads Julia before torch)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis_hybrid  # noqa: E402
from gridfm_datakit.network import load_net_from_file  # noqa: E402

c = __import__("03_classical")
NET, CASE, OUT = c.NET, c.CASE, c.OUT
LOAD_SCALES = (0.8, 0.85, 0.9, 0.95, 1.0)


BANDS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.075, 0.10]
EXCESS = [np.inf, 5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]  # MW above the N-0 residual of the same dispatch


def build(margin):
    engine.WORK = HERE
    engine.load_net_from_pglib = lambda name: c.in_service_gens_only(load_net_from_file(str(CASE)))
    engine.layout = lambda network, n, edges: np.zeros((n, 2))
    return engine.Engine(NET, "base", dispatch_margin=margin)


def sweep(eng, n_single, n_double):
    rng = np.random.default_rng(1)
    rated = [b["id"] for b in eng.static["branches"] if b["rate_a"] is not None and not b["bridge"]]
    singles = rng.choice(rated, n_single, replace=False)
    doubles = set()
    while len(doubles) < n_double:
        doubles.add(tuple(sorted(rng.choice(rated, 2, replace=False).tolist())))
    sets = [()] + [(int(i),) for i in singles] + sorted(doubles)

    frames, fails, state = [], {}, 0
    for scale in LOAD_SCALES:
        try:
            eng._dispatch(scale)  # check once: a failed dispatch is not cached and would be retried per set
        except engine.SolveError as err:
            c.log(f"load x{scale}: {err.message} -> level skipped")
            continue
        for outages in sets:
            try:
                result = eng.solve(scale, outages)
            except engine.SolveError as err:
                fails[err.reason] = fails.get(err.reason, 0) + 1
                continue
            lines = pd.DataFrame(result["branches"])
            lines = lines[lines.in_service & lines.loading_ac.notna()]
            frames.append(pd.DataFrame({
                "state": state,
                "load_scale": scale,
                "outages": "+".join(map(str, outages)),
                "line": lines.id,
                "ac": lines.loading_ac,
                "genco": lines.loading_genco,
                "dc": lines.loading_dc,
                "residual": pd.DataFrame(result["buses"]).res_p.abs().max(),
            }))
            state += 1
        c.log(f"load x{scale}: {state} states solved so far, failures {fails}")
    return pd.concat(frames, ignore_index=True)


def report(data):
    print("\n=== rules from analysis_hybrid.py (absolute residual) ===")
    analysis_hybrid.report(data)

    base = data[data.outages == ""].groupby("load_scale").residual.first()
    excess = data.assign(residual=data.residual - data.load_scale.map(base))
    analysis_hybrid.BANDS, analysis_hybrid.RESIDUALS = BANDS, EXCESS
    print("\n=== finer bands, residual as excess over the N-0 state of the same dispatch ===")
    analysis_hybrid.report(excess)

    keys = np.array(sorted(data.outages.unique()))
    np.random.default_rng(0).shuffle(keys)
    test = excess[excess.outages.isin(keys[len(keys) // 2:])]
    print("\nband only, test half:")
    for band in BANDS:
        print(analysis_hybrid.row(analysis_hybrid.rule_name(band, np.inf), analysis_hybrid.score(test, "genco", band)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=int, default=150)
    parser.add_argument("--double", type=int, default=150)
    parser.add_argument("--margin", type=float, default=engine.DISPATCH_MARGIN)
    parser.add_argument("--tag", default="")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    out = OUT / f"hybrid_texas{args.tag}.parquet"
    if args.reuse:
        data = pd.read_parquet(out)
    else:
        start = time.perf_counter()
        data = sweep(build(args.margin), args.single, args.double)
        data.to_parquet(out, index=False)
        c.log(f"saved {out} after {time.perf_counter() - start:.0f} s (margin {args.margin})")
    report(data)


if __name__ == "__main__":
    main()

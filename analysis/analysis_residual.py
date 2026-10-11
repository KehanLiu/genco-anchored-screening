"""Does GENCO's own power-balance residual warn when its prediction is wrong?

The residual needs no classical solver, so if it tracks the real error it can
tell an operator when to fall back to AC power flow. This sweeps single and
double line outages at several load levels, and checks how well the residual
ranks the states where GENCO is actually wrong.

    .venv/bin/python analysis/analysis_residual.py
"""

# engine first: it loads Julia, which has to happen before torch.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here

from engine import WORK, Engine, SolveError  # noqa: E402

import argparse
import itertools
import random

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

LOAD_SCALES = [0.7, 0.85, 1.0, 1.1, 1.2]
N_DOUBLE = 150  # random double outages per load level


def cases(engine, n_double=N_DOUBLE):
    """(load scale, outages) pairs: no outage, every single outage, and random double outages."""
    ids = [b["id"] for b in engine.static["branches"] if not b["bridge"]]
    rng = random.Random(0)
    doubles = rng.sample(list(itertools.combinations(ids, 2)), n_double)
    out = [(s, ()) for s in LOAD_SCALES]
    out += [(s, (i,)) for s in LOAD_SCALES for i in ids]
    out += [(s, pair) for s in LOAD_SCALES for pair in doubles]
    return out


def collect(engine):
    rows = []
    for load_scale, outages in cases(engine):
        try:
            result = engine.solve(load_scale, outages)
        except SolveError:
            continue
        buses = pd.DataFrame(result["buses"])
        branches = pd.DataFrame(result["branches"])
        live = branches[branches.in_service]
        summary = result["summary"]
        rows.append({
            "load_scale": load_scale,
            "n_out": len(outages),
            "outages": "+".join(map(str, outages)),
            # GENCO's self-check, available without the AC solver
            "res_p_max": buses.res_p.abs().max(),
            "res_p_sum": buses.res_p.abs().sum(),
            "res_q_max": buses.res_q.abs().max(),
            # the real error, known only after the AC solve
            "flow_err_max": (live.p_genco - live.p_ac).abs().max(),
            "flow_err_mean": (live.p_genco - live.p_ac).abs().mean(),
            "vm_err_max": (buses.vm_genco - buses.vm_ac).abs().max(),
            "loading_err_max": (live.loading_genco - live.loading_ac).abs().max(),
            "slack_err": (buses.pg_genco - buses.pg_ac)[buses.type == "REF"].abs().max(),
            "missed": len(summary["missed"]),
            "false_alarms": len(summary["false_alarms"]),
        })
    return pd.DataFrame(rows)


def auc(score, positive):
    """Probability that a random bad state has a higher residual than a random good one."""
    if positive.sum() == 0 or (~positive).sum() == 0:
        return float("nan")
    return mannwhitneyu(score[positive], score[~positive]).statistic / (positive.sum() * (~positive).sum())


def report(data):
    print(f"\n{len(data)} solved states: {dict(data.n_out.value_counts().sort_index())} by number of lines out")
    print("\nreal error by number of lines out (median / 90th percentile / max):")
    for n, group in data.groupby("n_out"):
        q = group.flow_err_max.quantile([0.5, 0.9, 1.0]).round(1).tolist()
        print(f"  {n} out: worst-line flow error {q} MW, states with a missed overload {(group.missed > 0).mean():.0%}")

    print("\nrank correlation (Spearman) between the residual and the real error:")
    for res in ("res_p_max", "res_p_sum", "res_q_max"):
        cells = [f"{err} {spearmanr(data[res], data[err]).statistic:+.2f}" for err in ("flow_err_max", "vm_err_max", "slack_err")]
        print(f"  {res:<10} vs  " + "   ".join(cells))

    print("\ncan the residual pick out the bad states? (AUC: 0.5 = useless, 1.0 = perfect)")
    bad = {
        "worst-line flow error > 5 MW": data.flow_err_max > 5,
        "worst-line flow error > 10 MW": data.flow_err_max > 10,
        "worst-line loading error > 5 points": data.loading_err_max > 0.05,
        "at least one missed overload": data.missed > 0,
    }
    for name, flag in bad.items():
        cells = "   ".join(f"{res} {auc(data[res], flag):.2f}" for res in ("res_p_max", "res_p_sum"))
        print(f"  {name:<38} {flag.mean():>4.0%} of states   {cells}")

    print("\nif the operator re-checks with AC whenever res_p_max exceeds a threshold:")
    print(f"  {'threshold':>10} {'states re-checked':>18} {'>10 MW errors caught':>22} {'missed overloads caught':>25}")
    big, miss = data.flow_err_max > 10, data.missed > 0
    for threshold in (0.25, 0.5, 1.0, 2.0):
        flagged = data.res_p_max > threshold
        print(f"  {threshold:>7.2f} MW {flagged.mean():>17.0%} {(flagged & big).sum() / max(big.sum(), 1):>21.0%} {(flagged & miss).sum() / max(miss.sum(), 1):>24.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="case118_ieee")
    parser.add_argument("--variant", default="base")
    args = parser.parse_args()

    data = collect(Engine(args.network, args.variant))
    out = WORK / f"residual_{args.network}_{args.variant}.csv"
    data.to_csv(out, index=False)
    print(f"saved {out}")
    report(data)


if __name__ == "__main__":
    main()

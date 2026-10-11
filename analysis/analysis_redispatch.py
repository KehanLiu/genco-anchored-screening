"""Does GENCO stay accurate when generation is moved by hand?

GENCO was trained on states dispatched by AC-OPF. A game lets the player shift
output between generators freely, which leaves that distribution. This moves
increasing amounts of power between random generator pairs, starting from the
OPF dispatch, and compares GENCO with AC power flow at each level.

    .venv/bin/python analysis/analysis_redispatch.py
"""

# engine first: it loads Julia, which has to happen before torch.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here

from engine import WORK, Engine, SolveError  # noqa: E402

import argparse
import random

import pandas as pd

SHIFT_MW = [0, 50, 150, 300, 600, 1000]  # total power moved between generators
SAMPLES = 40
PAIRS = 4  # generator pairs sharing each shift


def random_redispatch(gens, total, rng):
    """Move `total` MW from some generators to others, within their limits. Returns ({gen: delta}, MW moved)."""
    free = gens[~gens.slack]
    room_up = (free.p_max - free.p_set).clip(lower=0)
    room_down = (free.p_set - free.p_min).clip(lower=0)
    delta, moved = {}, 0.0
    for _ in range(PAIRS):
        ups = [g for g in free.id if room_up[g] > 1 and g not in delta]
        downs = [g for g in free.id if room_down[g] > 1 and g not in delta]
        if not ups or not downs:
            break
        up = rng.choice(ups)
        down = rng.choice([g for g in downs if g != up] or downs)
        if up == down:
            break
        amount = min(total / PAIRS, room_up[up], room_down[down])
        delta[up], delta[down] = amount, -amount
        moved += amount
    return delta, moved


def collect(engine, load_scale):
    rng = random.Random(0)
    base = pd.DataFrame(engine.solve(load_scale)["gens"]).set_index("id", drop=False)
    rows = []
    for target in SHIFT_MW:
        for _ in range(SAMPLES if target else 1):
            delta, moved = random_redispatch(base, target, rng) if target else ({}, 0.0)
            row = {"target_mw": target, "moved_mw": moved, "solved": False}
            try:
                result = engine.solve(load_scale, redispatch=delta)
            except SolveError:
                rows.append(row)
                continue
            buses = pd.DataFrame(result["buses"])
            branches = pd.DataFrame(result["branches"])
            summary = result["summary"]
            rows.append({
                **row,
                "solved": True,
                "flow_err_max": (branches.p_genco - branches.p_ac).abs().max(),
                "flow_err_mean": (branches.p_genco - branches.p_ac).abs().mean(),
                "vm_err_max": summary["max_vm_err"],
                "slack_err": abs(summary["slack_p_genco_mw"] - summary["slack_p_ac_mw"]),
                "res_p_max": buses.res_p.abs().max(),
                "overloads_ac": len(summary["overloads_ac"]),
                "missed": len(summary["missed"]),
                "false_alarms": len(summary["false_alarms"]),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="case118_ieee")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--load_scale", type=float, default=1.0)
    args = parser.parse_args()

    data = collect(Engine(args.network, args.variant), args.load_scale)
    out = WORK / f"redispatch_{args.network}_{args.variant}.csv"
    data.to_csv(out, index=False)
    print(f"saved {out}")

    solved = data[data.solved]
    table = solved.groupby("target_mw").agg(
        states=("solved", "size"),
        moved_mw=("moved_mw", "median"),
        flow_err_median=("flow_err_max", "median"),
        flow_err_worst=("flow_err_max", "max"),
        vm_err_median=("vm_err_max", "median"),
        slack_err_median=("slack_err", "median"),
        res_p_median=("res_p_max", "median"),
        ac_overloads=("overloads_ac", "mean"),
        missed=("missed", "mean"),
        false_alarms=("false_alarms", "mean"),
    )
    print(f"\n{args.network} at load x{args.load_scale}: GENCO vs AC as generation is moved by hand")
    print("(flow_err = worst-line active-flow error in MW; missed / false_alarms = lines per state)")
    print(table.round(3).to_string())
    print(f"\nAC power flow failed to converge in {(~data.solved).sum()} of {len(data)} states")


if __name__ == "__main__":
    main()

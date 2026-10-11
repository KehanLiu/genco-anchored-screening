"""Sweep every single-line outage and compare GENCO with the classical AC solver.

Uses the same engine as the workbench (workbench/server.py). Needs the checkpoint that
`run_pretrained.py` downloads:

    .venv/bin/python run_pretrained.py --network case118_ieee --variant base
    .venv/bin/python workbench/whatif.py --network case118_ieee --variant base
"""

# engine first: it loads Julia, which has to happen before torch.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here

from engine import Engine, SolveError  # noqa: E402

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="case118_ieee")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--load_scale", type=float, default=1.0)
    args = parser.parse_args()

    engine = Engine(args.network, args.variant)
    rows = []
    for branch in engine.static["branches"]:
        try:
            result = engine.solve(args.load_scale, [branch["id"]])
        except SolveError:
            continue
        rows.append({
            "outage": branch["id"],
            "ac_ms": result["timing_ms"]["ac"] + result["timing_ms"]["dc"],
            "genco_ms": result["timing_ms"]["genco"],
            **{k: v for k, v in result["summary"].items() if k.startswith("max_")},
            "missed": len(result["summary"]["missed"]),
            "false_alarms": len(result["summary"]["false_alarms"]),
        })

    result = pd.DataFrame(rows)
    errors = [c for c in result.columns if c.startswith("max_")]
    print(f"\n{len(result)} single-line outages solved on {args.network} (load x{args.load_scale})")
    print(f"latency per state, median: AC+DC solve {result.ac_ms.median():.1f} ms, GENCO {result.genco_ms.median():.1f} ms")
    print(f"outages with an overload GENCO missed: {(result.missed > 0).sum()}, with a false alarm: {(result.false_alarms > 0).sum()}")
    print("\nworst-bus / worst-branch error per outage, GENCO vs AC:")
    print(result[errors].describe().loc[["mean", "50%", "max"]].to_string())
    print("\n5 outages where GENCO is furthest from AC on branch active flow:")
    print(result.nlargest(5, "max_p_flow_err_mw").to_string(index=False))


if __name__ == "__main__":
    main()

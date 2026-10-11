"""Conformal screening with "GENCO + one Newton step" as the surrogate.

Reads the per-line loadings written by analysis/analysis_warmstart.py (normal set: N-0 to N-3 at dispatched
load levels) and runs the same label-conditional split conformal screening as analysis/analysis_conformal.py
with one global band, for each start point after 0, 1 and 2 Newton steps. States are split at
random into calibration and test halves, repeated over many splits.

    .venv/bin/python analysis/analysis_onestep.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analysis_conformal import ALPHAS, conformal_width, evaluate  # noqa: E402

WORK = ROOT / "work"
GRIDS = {"IEEE 118": WORK / "warmstart_case118", "Texas 2000": WORK / "texas/out/warmstart_texas"}
SURROGATES = {"genco": "GENCO", "genco1": "GENCO + 1 Newton step", "genco2": "GENCO + 2 steps", "dc": "DC", "dc1": "DC + 1 step",
              "flat1": "flat start + 1 step", "flat2": "flat start + 2 steps", "prev1": "pre-outage solution + 1 step"}
SPLITS = 200


def main():
    rows = []
    for grid, stem in GRIDS.items():
        states = pd.read_parquet(f"{stem}.parquet")
        normal = states.state[states.set == "normal"]
        lines = pd.read_parquet(f"{stem}_lines.parquet")
        lines = lines[lines.state.isin(normal)].dropna(subset=["ac"])
        ids = np.array(sorted(lines.state.unique()))
        print(f"\n{grid}: {len(ids)} states (normal set), {(lines.ac > 1.001).sum()} real overloads, "
              f"{lines.state[lines.ac > 1.001].nunique() / len(ids):.0%} of states have one")
        print(f"  {'surrogate':<30} {'p99 err (pt)':>12} {'max':>6} | alpha: " + "".join(f"{a:>26.0%}" for a in ALPHAS))
        print(f"  {'':<30} {'':>12} {'':>6} |        " + "".join(f"{'AC on secure / missed':>26}" for _ in ALPHAS))
        for column, label in SURROGATES.items():
            error = (lines[column] - lines.ac).abs() * 100
            cells = []
            for alpha in ALPHAS:
                results = []
                for seed in range(SPLITS):
                    order = np.random.default_rng(seed).permutation(ids)
                    cal, test = lines[lines.state.isin(order[: len(ids) // 2])], lines[lines.state.isin(order[len(ids) // 2:])]
                    width, _ = conformal_width(cal, column, np.ones(len(cal)), alpha)
                    results.append(evaluate(test, test.state.nunique(), column, np.ones(len(test)), width))
                r = pd.DataFrame(results)
                cells.append(f"{r.wasted.mean():>14.1%} / {r.missed_share.mean():>8.2%}")
                rows.append({"grid": grid, "surrogate": label, "alpha": alpha, "ac_calls": r.ac_calls.mean(), "wasted": r.wasted.mean(),
                             "missed_share": r.missed_share.mean(), "line_missed_share": r.line_missed_share.mean()})
            print(f"  {label:<30} {error.quantile(0.99):>12.2f} {error.max():>6.1f} |        " + "".join(f"{c:>26}" for c in cells))
    pd.DataFrame(rows).to_csv(WORK / "onestep_conformal.csv", index=False)
    print(f"\nsaved {WORK / 'onestep_conformal.csv'}  (mean over {SPLITS} random splits of the states)")


if __name__ == "__main__":
    main()

"""Where GENCO's line loadings go wrong: a failure atlas from the saved outage sweeps.

Reads a sweep written by analysis_hybrid.py or work/texas/04_hybrid.py and reports how the loading
error depends on the number of outages, the load level, how close the line is to its limit, how
far the outage moved the line, and which lines and outage sets carry the large errors.

    .venv/bin/python analysis_atlas.py work/hybrid_case118_ieee_base.parquet
    .venv/bin/python analysis_atlas.py work/texas/out/hybrid_texas_n2x450.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OVERLOAD = 1.001  # same as engine.OVERLOAD (not imported: engine starts Julia)
LARGE = 0.03      # an error of 3 points of rating: enough to flip a verdict inside any sensible band


def summary(frame, by, label):
    g = frame.groupby(by, observed=True)
    out = pd.DataFrame({
        "readings": g.size(),
        "genco median": g.e_genco.median() * 100,
        "genco p99": g.e_genco.quantile(0.99) * 100,
        "genco max": g.e_genco.max() * 100,
        "dc p99": g.e_dc.quantile(0.99) * 100,
        "bias (genco-ac)": g.bias.mean() * 100,
        "large errors": g.large.sum(),
    })
    print(f"\nloading error in points of rating, by {label}:")
    print(out.round(2).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", type=Path)
    args = parser.parse_args()

    data = pd.read_parquet(args.sweep)
    base = data[data.outages == ""].set_index(["load_scale", "line"]).genco.rename("genco0")
    data = data.join(base, on=["load_scale", "line"])
    data["n_out"] = np.where(data.outages == "", 0, data.outages.str.count(r"\+") + 1)
    data["bias"] = data.genco - data.ac
    data["e_genco"] = data.bias.abs()
    data["e_dc"] = (data.dc - data.ac).abs()
    data["large"] = data.e_genco > LARGE
    data["shift"] = (data.genco - data.genco0).abs()
    print(f"{args.sweep.name}: {data.state.nunique()} states, {len(data):,} line readings, {data.line.nunique()} lines")
    print(f"large errors (> {LARGE:.0%} of rating): {data.large.sum():,} readings ({data.large.mean():.3%}), in {data.state[data.large].nunique()} states")

    summary(data, "n_out", "number of lines out")
    summary(data, "load_scale", "load level")
    summary(data, pd.cut(data.ac, [0, 0.5, 0.8, 0.95, 1.0, 1.05, 10], labels=["<50%", "50-80%", "80-95%", "95-100%", "100-105%", ">105%"]), "true loading of the line")
    summary(data, pd.cut(data["shift"] * 100, [-1, 0.1, 1, 5, 10, 20, 1e4], labels=["<0.1", "0.1-1", "1-5", "5-10", "10-20", ">20"]),
            "how far GENCO moved the line from its base-case loading (points)")
    state = data.groupby("state").agg(residual=("residual", "first"), worst=("e_genco", "max"), load_scale=("load_scale", "first"))
    base_res = data[data.outages == ""].groupby("load_scale").residual.first()
    state["excess"] = state.residual - state.load_scale.map(base_res)
    print("\nper state: rank correlation (Spearman) of the worst line error with GENCO's residual "
          f"{state.worst.corr(state.residual, method='spearman'):.2f}, with the residual in excess of the base case {state.worst.corr(state.excess, method='spearman'):.2f}")

    # Concentration: how few lines and outage sets carry the large errors.
    large = data[data.large]
    by_line = large.groupby("line").agg(large=("large", "size"), states=("state", "nunique"), worst=("e_genco", "max"), bias=("bias", "mean"))
    by_line = by_line.sort_values("large", ascending=False)
    top = by_line.head(10)
    print(f"\n{len(by_line)} of {data.line.nunique()} lines ever have a large error; the 10 worst carry {top.large.sum() / max(len(large), 1):.0%} of them:")
    print((top.assign(worst=top.worst * 100, bias=top.bias * 100)).round(1).to_string())
    by_set = large.groupby("outages").size().sort_values(ascending=False)
    print(f"\n{len(by_set)} of {data.outages.nunique()} outage sets ever cause a large error; the 10 worst cause {by_set.head(10).sum() / max(len(large), 1):.0%} of them:")
    print(by_set.head(10).to_string())

    real = data.ac > OVERLOAD
    missed = data[real & (data.genco <= OVERLOAD)]
    print(f"\nreal overloads GENCO alone calls safe: {len(missed)} of {real.sum()} ({len(missed) / max(real.sum(), 1):.1%}), on {missed.line.nunique()} lines")
    print("  under-call in points: " + ", ".join(f"{q:.0%} quantile {v * 100:.1f}" for q, v in (missed.ac - missed.genco).quantile([0.5, 0.9, 1.0]).items()))
    print("  share of these where the line had moved > 1 point from its base case: "
          f"{(missed['shift'] > 0.01).mean():.0%}  (all readings: {(data['shift'] > 0.01).mean():.1%})")
    print(missed.groupby("line").agg(missed=("ac", "size"), worst_ac=("ac", "max"), genco_at_worst=("genco", "min"), outage_sets=("outages", "nunique"))
          .sort_values("missed", ascending=False).head(8).round(3).to_string())

    out = args.sweep.with_name(args.sweep.stem.replace("hybrid", "atlas") + "_lines.csv")
    per_line = data.groupby("line").agg(p99=("e_genco", lambda e: e.quantile(0.99)), worst=("e_genco", "max"), bias=("bias", "mean"),
                                        large=("large", "sum"), mean_loading=("ac", "mean"))
    per_line.to_csv(out)
    print(f"\nsaved per-line table {out}")


if __name__ == "__main__":
    main()

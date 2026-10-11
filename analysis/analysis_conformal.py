"""A conformal trust layer: which grid states can skip the AC solver, with a guaranteed miss rate.

A surrogate (GENCO, or DC power flow) predicts every line loading. A line is "possibly overloaded"
when its prediction is above  limit - b * sigma ; a state with such a line goes to the AC solver,
every other state is accepted as secure without AC.

b comes from label-conditional split conformal prediction: on calibration states, take every line
reading that really is overloaded (AC), measure how far below the limit the surrogate put it in
units of sigma, and take the ceil((n+1)(1-alpha))-th smallest of these n scores. Under
exchangeability a new real overload is then flagged with probability >= 1 - alpha, so at most
alpha of the real overloads are missed.

Surrogates
    genco, dc        the raw prediction
    genco_d, dc_d    anchored on the base case: AC loading of the N-0 state of the same dispatch
                     (which an operator has anyway) plus the surrogate's predicted change. For DC
                     this is the usual way DC sensitivities are used in practice.
sigma (what makes the band adaptive)
    global   1: one band width for all lines
    line     a high quantile of that line's absolute error on a separate fit split
    shift    how far the surrogate moved the line away from its base-case loading, plus a floor:
             lines the outage does not touch get a narrow band

Outage sets are split as whole groups: fit / calibration / test never share an outage set. The N-0
states are the anchor and are left out of the scoring.

    .venv/bin/python analysis/analysis_conformal.py work/hybrid_case118_ieee_base.parquet
    .venv/bin/python analysis/analysis_conformal.py work/texas/out/hybrid_texas_n2x450.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gridtrust" / "src"))
import gridtrust.conformal  # noqa: E402  (the standalone package holds the calibration and scoring)

OVERLOAD = 1.001  # same as engine.OVERLOAD (not imported: engine starts Julia)
ALPHAS = [0.10, 0.05, 0.02, 0.01]
SURROGATES = ["genco", "genco_d", "dc", "dc_d"]
SIGMAS = ["global", "line", "shift"]
LINE_QUANTILE = 0.99    # per-line error quantile used as sigma for "line"
LINE_FLOOR = 0.002      # 0.2 points of rating
SHIFT_FLOOR = 0.005     # 0.5 points of rating; picked on the calibration half, results barely move between 0.2 and 1
KEEP = 0.5              # line readings where AC and every surrogate are below this never matter


def load(path):
    """Line readings near enough to the limit to matter, per-line errors of all readings, states per outage set."""
    data = pd.read_parquet(path)
    base = data[data.outages == ""].set_index(["load_scale", "line"])[["ac", "genco", "dc"]].add_suffix("0")
    data = data[data.outages != ""].join(base, on=["load_scale", "line"]).dropna(subset=["ac0"])
    for s in ("genco", "dc"):
        data[f"{s}_d"] = data.ac0 + data[s] - data[f"{s}0"]
        data[f"shift_{s}"] = data[f"shift_{s}_d"] = (data[s] - data[f"{s}0"]).abs() + SHIFT_FLOOR
    errors = data[["outages", "line"]].copy()
    for s in SURROGATES:
        errors[s] = (data[s] - data.ac).abs()
    n_states = data.groupby("outages").state.nunique()
    data = data[data[["ac"] + SURROGATES].max(axis=1) > KEEP].reset_index(drop=True)
    return data, errors, n_states


def split_keys(keys, seed):
    """Outage sets -> fit / calibration / test (quarter, quarter, half)."""
    keys = np.array(sorted(keys))
    np.random.default_rng(seed).shuffle(keys)
    half = len(keys) // 2
    return set(keys[: half // 2]), set(keys[half // 2: half]), set(keys[half:])


def sigma_of(data, surrogate, kind, line_sigma):
    if kind == "global":
        return np.ones(len(data))
    if kind == "shift":
        return data[f"shift_{surrogate}"].to_numpy()
    sigma = line_sigma[surrogate]
    return data.line.map(sigma).fillna(sigma.max()).to_numpy()


def conformal_width(cal, surrogate, sigma, alpha):
    """Band width b (in sigma units) such that at most alpha of real overloads fall below limit - b*sigma."""
    return gridtrust.conformal.calibrate(cal[surrogate].to_numpy(), cal.ac.to_numpy(), limit=OVERLOAD, alpha=alpha, sigma=sigma, return_n=True)


def evaluate(test, n_states, surrogate, sigma, width):
    """Screening: every state with a possibly overloaded line goes to AC; the rest is accepted as secure."""
    return gridtrust.conformal.evaluate(test[surrogate].to_numpy(), test.ac.to_numpy(), test.state.to_numpy(), width,
                                        limit=OVERLOAD, sigma=sigma, n_states=n_states)


def run(data, errors, n_states, seed, alphas=ALPHAS):
    """All surrogates, sigma kinds and alphas on one split of the outage sets."""
    fit, cal_keys, test_keys = split_keys(n_states.index, seed)
    test = data[data.outages.isin(test_keys)]
    n_test = n_states[n_states.index.isin(test_keys)].sum()
    line_sigma = errors[errors.outages.isin(fit)].groupby("line")[SURROGATES].quantile(LINE_QUANTILE).clip(lower=LINE_FLOOR)
    cals = {True: data[data.outages.isin(cal_keys)], False: data[data.outages.isin(cal_keys | fit)]}
    rows = []
    for surrogate in SURROGATES:
        for kind in SIGMAS:
            cal = cals[kind == "line"]  # only "line" needs the fit split; the others calibrate on fit + calibration
            sigma_cal, sigma_test = sigma_of(cal, surrogate, kind, line_sigma), sigma_of(test, surrogate, kind, line_sigma)
            for alpha in alphas:
                width, n_cal = conformal_width(cal, surrogate, sigma_cal, alpha)
                rows.append({"surrogate": surrogate, "sigma": kind, "alpha": alpha, "seed": seed, "width": width, "n_cal": n_cal,
                             **evaluate(test, n_test, surrogate, sigma_test, width)})
    return rows


def budget_curves(data, n_states, seed=0):
    """Rank test states by risk (how close the worst line is to the limit, in sigma) and send them to AC in that order."""
    _, _, test_keys = split_keys(n_states.index, seed)
    test = data[data.outages.isin(test_keys)]
    n_test = n_states[n_states.index.isin(test_keys)].sum()
    overloads = (test.ac > OVERLOAD).groupby(test.state).sum()
    frames = []
    for surrogate in SURROGATES:
        for kind in ("global", "shift"):
            risk = ((test[surrogate] - OVERLOAD) / sigma_of(test, surrogate, kind, None)).groupby(test.state).max()
            order = risk.sort_values(ascending=False).index
            found = overloads.reindex(order).cumsum().to_numpy() / overloads.sum()
            frames.append(pd.DataFrame({"surrogate": surrogate, "sigma": kind, "ac_calls": np.arange(1, len(order) + 1) / n_test, "found": found}))
    return pd.concat(frames, ignore_index=True)


def table(rows):
    lines = [f"  {'surrogate':<8} {'sigma':<7} {'alpha':>5} {'n cal':>6} {'width':>9} {'AC calls':>9} {'on secure':>9} {'missed':>10} {'of real':>8} "
             f"{'95% upper':>9} {'per-line':>9} {'worst':>7}"]
    for r in rows:
        width = "all->AC" if np.isinf(r["width"]) else (f"{r['width'] * 100:.2f} pt" if r["sigma"] == "global" else f"{r['width']:.2f} sig")
        worst = "" if np.isnan(r["worst_missed"]) else f"{r['worst_missed']:.1%}"
        lines.append(f"  {r['surrogate']:<8} {r['sigma']:<7} {r['alpha']:>5.0%} {r['n_cal']:>6} {width:>9} {r['ac_calls']:>9.1%} {r['wasted']:>9.1%} "
                     f"{r['missed']:>4}/{r['real']:<5} {r['missed_share']:>8.2%} {r['missed_upper95']:>9.2%} {r['line_missed_share']:>9.2%} {worst:>7}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", type=Path, help="parquet written by analysis/analysis_hybrid.py or work/texas/04_hybrid.py")
    parser.add_argument("--splits", type=int, default=100, help="random re-splits for the coverage check")
    args = parser.parse_args()
    stem = args.sweep.stem.replace("hybrid", "conformal")

    data, errors, n_states = load(args.sweep)
    real = data.ac > OVERLOAD
    print(f"{args.sweep.name}: {n_states.sum()} states with outages, {len(n_states)} outage sets, {real.sum()} real overloads on {data.line[real].nunique()} lines")
    print(f"states with a real overload: {data.state[real].nunique() / n_states.sum():.1%}  (AC runs on these are never wasted)")
    print("absolute loading error in points of rating, all readings (median / 99th percentile / max):")
    for s in SURROGATES:
        e = errors[s] * 100
        print(f"  {s:<8} {e.median():.3f} / {e.quantile(0.99):.2f} / {e.max():.1f}")

    print("\ncanonical split (seed 0; its test half was not used to design the scores):")
    print(table(run(data, errors, n_states, 0)))

    curves = budget_curves(data, n_states)
    curves.to_csv(args.sweep.with_name(f"{stem}_budget.csv"), index=False)
    print("\nfixed AC budget on the canonical test half: share of real overloads found when the riskiest x% of states go to AC")
    budgets = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    print(f"  {'surrogate':<8} {'sigma':<7} " + " ".join(f"{b:>6.0%}" for b in budgets) + "   AC share needed to find all")
    for (s, k), g in curves.groupby(["surrogate", "sigma"], sort=False):
        found = [g.found[g.ac_calls <= b].max() for b in budgets]
        print(f"  {s:<8} {k:<7} " + " ".join(f"{f:>6.1%}" for f in found) + f"   {g.ac_calls[g.found >= 1 - 1e-12].min():.1%}")

    splits = pd.DataFrame([r for seed in range(1, args.splits + 1) for r in run(data, errors, n_states, seed)])
    out = args.sweep.with_name(f"{stem}_splits.csv")
    splits.to_csv(out, index=False)
    print(f"\ncoverage check over {args.splits} random splits of the outage sets (saved {out}):")
    print(f"  {'surrogate':<8} {'sigma':<7} {'alpha':>5} {'AC calls mean':>13} {'p10-p90':>13} {'on secure':>9} {'missed mean':>11} {'p90':>7} {'per-line missed':>15} {'splits > alpha':>14}")
    for (s, k, a), g in splits.groupby(["surrogate", "sigma", "alpha"], sort=False):
        print(f"  {s:<8} {k:<7} {a:>5.0%} {g.ac_calls.mean():>13.1%} {g.ac_calls.quantile(0.1):>6.1%}-{g.ac_calls.quantile(0.9):<6.1%} {g.wasted.mean():>9.1%} "
              f"{g.missed_share.mean():>11.2%} {g.missed_share.quantile(0.9):>7.2%} {g.line_missed_share.mean():>15.2%} {(g.missed_share > a).mean():>14.0%}")


if __name__ == "__main__":
    main()

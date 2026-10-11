"""The safeguarded hybrid workflow, measured: GENCO first, AC power flow only when in doubt.

For each grid state GENCO predicts the line loadings. The state is escalated to the classical
AC solver when GENCO's own power-balance residual is high or when it puts any line within a
band around 100% loading; otherwise its verdict on overloads is accepted as is. AC power flow
solves a whole state, so the cost of the safeguard is the share of states escalated.

The band and the residual threshold are chosen on half of the outage sets and scored on the
other half. The same procedure is run for DC power flow (band only; it has no residual).

    .venv/bin/python analysis/analysis_hybrid.py            # solve the sweep, then report
    .venv/bin/python analysis/analysis_hybrid.py --reuse    # report from the saved sweep
"""

# engine first: it loads Julia, which has to happen before torch.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here

from engine import OVERLOAD, WORK, Engine, SolveError  # noqa: E402

import argparse
import itertools

import numpy as np
import pandas as pd

from analysis_residual import cases

BANDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]  # half-width around 100% loading
RESIDUALS = [np.inf, 2.0, 1.0, 0.5, 0.35, 0.25, 0.15]            # MW; inf = residual not used
TARGETS = [0.0, 0.01]  # share of real overloads the workflow may miss on the calibration half


def collect(engine, n_double):
    frames = []
    for state, (load_scale, outages) in enumerate(cases(engine, n_double)):
        try:
            result = engine.solve(load_scale, outages)
        except SolveError:
            continue
        lines = pd.DataFrame(result["branches"])
        lines = lines[lines.in_service & lines.loading_ac.notna()]
        frames.append(pd.DataFrame({
            "state": state,
            "load_scale": load_scale,
            "outages": "+".join(map(str, outages)),
            "line": lines.id,
            "ac": lines.loading_ac,
            "genco": lines.loading_genco,
            "dc": lines.loading_dc,
            "residual": pd.DataFrame(result["buses"]).res_p.abs().max(),
        }))
    return pd.concat(frames, ignore_index=True)


def score(data, solver, band, residual=np.inf):
    """Run the workflow with one rule. Returns AC-call share and what the accepted states get wrong."""
    in_band = (data[solver] - 1).abs() <= band if band else pd.Series(False, index=data.index)
    escalate = in_band.groupby(data.state).transform("any") | (data.residual > residual)
    real = data.ac > OVERLOAD
    called = data[solver] > OVERLOAD
    missed = real & ~called & ~escalate
    false_alarm = ~real & called & ~escalate
    return {
        "ac_calls": data.state[escalate].nunique() / data.state.nunique(),
        "missed": int(missed.sum()),
        "missed_share": missed.sum() / max(real.sum(), 1),
        "states_with_miss": data.state[missed].nunique(),
        "false_alarms": int(false_alarm.sum()),
    }


def calibrate(data, solver, target):
    """Cheapest rule (fewest AC calls) that misses at most `target` of the real overloads."""
    residuals = RESIDUALS if solver == "genco" else [np.inf]
    best = None
    for band, residual in itertools.product(BANDS, residuals):
        result = score(data, solver, band, residual)
        if result["missed_share"] <= target and (best is None or result["ac_calls"] < best[2]["ac_calls"]):
            best = (band, residual, result)
    return best


def rule_name(band, residual):
    parts = []
    if band:
        parts.append(f"band {1 - band:.1%}-{1 + band:.1%}")
    if np.isfinite(residual):
        parts.append(f"residual > {residual} MW")
    return " or ".join(parts) or "never escalate"


def row(label, result):
    return (f"  {label:<52} {result['ac_calls']:>8.1%} {result['missed']:>7} {result['missed_share']:>8.2%} "
            f"{result['states_with_miss']:>7} {result['false_alarms']:>7}")


def report(data):
    # Whole outage sets go to one half, so the test half has outages the calibration never saw.
    keys = np.array(sorted(data.outages.unique()))
    np.random.default_rng(0).shuffle(keys)
    calibration = data[data.outages.isin(keys[: len(keys) // 2])]
    test = data[data.outages.isin(keys[len(keys) // 2:])]

    real = data.ac > OVERLOAD
    print(f"\n{data.state.nunique()} states ({len(keys)} outage sets x {data.load_scale.nunique()} load levels), {len(data):,} line readings")
    print(f"real overloads: {real.sum():,} on {data.line[real].nunique()} distinct lines; {data.state[real].nunique() / data.state.nunique():.0%} of states have at least one")
    err = (data[["genco", "dc"]].sub(data.ac, axis=0) * 100).abs()
    print("loading error in points of rating (median / 99th percentile): "
          f"GENCO {err.genco.median():.2f} / {err.genco.quantile(0.99):.1f}, DC {err.dc.median():.2f} / {err.dc.quantile(0.99):.1f}")

    header = f"  {'workflow':<52} {'AC calls':>8} {'missed':>7} {'of real':>8} {'states':>7} {'false+':>7}"
    print(f"\nscored on the test half ({test.state.nunique()} states, {(test.ac > OVERLOAD).sum():,} real overloads):")
    print(header)
    print(row("GENCO alone", score(test, "genco", 0)))
    print(row("DC power flow alone", score(test, "dc", 0)))
    for target in TARGETS:
        for solver, name in (("genco", "GENCO"), ("dc", "DC")):
            best = calibrate(calibration, solver, target)
            if best is None:
                print(f"  {name}: no rule keeps misses within {target:.0%} on the calibration half")
                continue
            band, residual, fitted = best
            label = f"{name} + AC when {rule_name(band, residual)}"
            print(row(label, score(test, solver, band, residual)) + f"   (calibration: {fitted['ac_calls']:.1%} calls, {fitted['missed']} missed; target <= {target:.0%})")
    print(row("AC power flow always", {"ac_calls": 1.0, "missed": 0, "missed_share": 0.0, "states_with_miss": 0, "false_alarms": 0}))

    print("\ntrade-off on the test half, GENCO with the residual rule at 0.5 MW and a widening band:")
    print(header)
    for band in BANDS:
        print(row(rule_name(band, 0.5), score(test, "genco", band, 0.5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="case118_ieee")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--doubles", type=int, default=300, help="random double outages per load level")
    parser.add_argument("--reuse", action="store_true", help="report from the saved sweep instead of solving again")
    args = parser.parse_args()

    out = WORK / f"hybrid_{args.network}_{args.variant}.parquet"
    if args.reuse and out.exists():
        data = pd.read_parquet(out)
    else:
        data = collect(Engine(args.network, args.variant), args.doubles)
        data.to_parquet(out, index=False)
        print(f"saved {out}")
    report(data)


if __name__ == "__main__":
    main()

"""Voltage problems DC power flow cannot see: how well does GENCO detect them?

DC power flow assumes every voltage magnitude is 1.0 p.u., so it can never flag a voltage-limit
violation. GENCO predicts Vm at every bus. This sweeps the same states as the hybrid analyses
(IEEE 118: analysis_hybrid; Texas 2000: work/texas/04_hybrid.py), keeps the per-bus voltages of
the AC solver and of GENCO, and scores GENCO's voltage-violation calls against AC.

Voltage limits are the per-bus VMIN/VMAX of the case file (IEEE 118: 0.94/1.06, Texas: 0.9/1.1;
the same limits the AC-OPF dispatch respects). Since violations of those are rare, tightened
operational bands are scored as well. PV and REF buses have their Vm given to every solver, so
errors and violations are counted on PQ buses only.

    work/texas/tmo 3600 .venv/bin/python -u analysis/analysis_voltage.py [--grid 118|texas|both]
    .venv/bin/python analysis/analysis_voltage.py --reuse
"""

# engine first: it loads Julia, which has to happen before torch.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here

import engine  # noqa: E402
from engine import OVERLOAD, Engine, SolveError  # noqa: E402

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gridfm_datakit.utils.idx_bus import VMAX, VMIN

OUT_118 = ROOT / "work" / "voltage_case118.parquet"
OUT_TEXAS = ROOT / "work" / "texas" / "out" / "voltage_texas.parquet"

TOL = 1e-4                     # p.u.; the OPF parks voltages exactly on a limit, as it does line loadings
BANDS = [0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]  # p.u. either side of a limit
TARGETS = [0.0, 0.01]          # share of real violations the workflow may miss on the calibration half
TARGET_STATE_SHARE = 0.05      # the "rare event" band is picked so about this share of states violates


def bus_rows(result, state, vmin, vmax):
    """Per-bus voltages of one solved state, with the state's line-overload counts attached."""
    buses = pd.DataFrame(result["buses"])
    lines = pd.DataFrame(result["branches"])
    lines = lines[lines.in_service & lines.loading_ac.notna()]
    return pd.DataFrame({
        "state": state,
        "load_scale": result["load_scale"],
        "outages": "+".join(map(str, result["outages"])),
        "bus": buses.id.astype("int32"),
        "type": buses.type,
        "vm_ac": buses.vm_ac.astype(float),
        "vm_genco": buses.vm_genco.astype(float),
        "vmin": vmin,
        "vmax": vmax,
        "over_ac": int((lines.loading_ac > OVERLOAD).sum()),
        "over_genco": int((lines.loading_genco > OVERLOAD).sum()),
        "over_dc": int((lines.loading_dc > OVERLOAD).sum()),
        "residual": buses.res_p.abs().max(),
    })


def limits(eng):
    return eng.base.buses[:, VMIN].copy(), eng.base.buses[:, VMAX].copy()


def collect_118(n_double=300):
    from analysis_residual import cases
    eng = Engine("case118_ieee", "base")
    vmin, vmax = limits(eng)
    frames = []
    for state, (load_scale, outages) in enumerate(cases(eng, n_double)):
        try:
            result = eng.solve(load_scale, outages)
        except SolveError:
            continue
        frames.append(bus_rows(result, state, vmin, vmax))
    return pd.concat(frames, ignore_index=True)


def collect_texas(n_single=150, n_double=150):
    """Texas states exactly as 04_hybrid.sweep samples them; the per-bus tables are tapped off engine.solve."""
    sys.path.insert(0, str(ROOT / "work" / "texas"))
    hybrid = __import__("04_hybrid")
    eng = hybrid.build(engine.DISPATCH_MARGIN)
    vmin, vmax = limits(eng)
    frames = []
    solve = eng.solve

    def tapped(load_scale=1.0, outages=(), redispatch=None):
        result = solve(load_scale, outages, redispatch)
        frames.append(bus_rows(result, len(frames), vmin, vmax))  # sweep numbers solved states the same way
        return result

    eng.solve = tapped
    hybrid.sweep(eng, n_single, n_double)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- report

def violates(vm, lo, hi):
    return (vm < lo - TOL) | (vm > hi + TOL)


def score(data, lo, hi, band=None):
    """GENCO's violation calls against AC on PQ buses. With `band`, a state where GENCO puts any PQ bus
    within `band` p.u. of a limit goes to the AC solver, and only the accepted states can be wrong."""
    real = violates(data.vm_ac, lo, hi)
    called = violates(data.vm_genco, lo, hi)
    if not band:
        escalate = pd.Series(False, index=data.index)
    else:
        near = ((data.vm_genco - lo).abs() <= band) | ((data.vm_genco - hi).abs() <= band)
        escalate = near.groupby(data.state).transform("any")
    missed = real & ~called & ~escalate
    false_alarm = ~real & called & ~escalate
    hit = real & called
    state_real = set(data.state[real])
    state_called = set(data.state[called | escalate])
    return {
        "real": int(real.sum()),
        "states_real": len(state_real),
        "called": int(called.sum()),
        "missed": int(missed.sum()),
        "missed_share": missed.sum() / max(real.sum(), 1),
        "false_alarms": int(false_alarm.sum()),
        "recall": hit.sum() / real.sum() if real.sum() else np.nan,
        "precision": hit.sum() / called.sum() if called.sum() else np.nan,
        "states_missed": len(state_real - state_called),       # violating states GENCO raises no flag for
        "states_with_miss": data.state[missed].nunique(),
        "ac_calls": data.state[escalate].nunique() / data.state.nunique(),
    }


def calibrate(data, lo, hi, target):
    best = None
    for band in BANDS:
        result = score(data, lo, hi, band)
        if result["missed_share"] <= target and (best is None or result["ac_calls"] < best[1]["ac_calls"]):
            best = (band, result)
    return best


def rare_band(pq, vmin, vmax):
    """Band, tighter or wider than the case limits, in which about TARGET_STATE_SHARE of states violate.

    The band is a share t of the way from the case limits to 1.0: lo = vmin + t (1 - vmin), same for hi.
    """
    n_states = pq.state.nunique()
    best = None
    for t in np.round(np.arange(-1.0, 0.951, 0.025), 3):  # t < 0: wider than the case limits
        lo, hi = vmin + t * (1 - vmin), vmax - t * (vmax - 1)
        share = pq.state[violates(pq.vm_ac, lo, hi)].nunique() / n_states
        if share > 0 and (best is None or abs(share - TARGET_STATE_SHARE) < abs(best[2] - TARGET_STATE_SHARE)):
            best = (round(lo, 4), round(hi, 4), share)
    return best


def report(data, name):
    n_states = data.state.nunique()
    keys = np.array(sorted(data.outages.unique()))
    np.random.default_rng(0).shuffle(keys)
    pq = data[data.type == "PQ"]
    calibration = pq[pq.outages.isin(keys[: len(keys) // 2])]
    test = pq[pq.outages.isin(keys[len(keys) // 2:])]
    vmin, vmax = float(data.vmin.min()), float(data.vmax.max())
    uniform = data.vmin.nunique() == 1 and data.vmax.nunique() == 1

    print(f"\n{'=' * 100}\n{name}: {n_states} states ({len(keys)} outage sets x {data.load_scale.nunique()} load levels), "
          f"{data.bus.nunique()} buses ({pq.bus.nunique()} PQ), {len(pq):,} PQ bus readings")
    print(f"case voltage limits: {vmin}-{vmax} p.u.{'' if uniform else ' (not uniform across buses; widest shown)'}")
    err = (pq.vm_genco - pq.vm_ac).abs()
    worst = err.groupby(pq.state).max()
    print(f"GENCO |Vm error| on PQ buses, p.u.: median {err.median():.5f} / p99 {err.quantile(0.99):.5f} / "
          f"p99.9 {err.quantile(0.999):.5f} / max {err.max():.5f}")
    print(f"  worst bus per state: median {worst.median():.5f} / p99 {worst.quantile(0.99):.5f}; "
          f"readings off by more than 0.005: {(err > 0.005).sum():,} ({(err > 0.005).mean():.3%}), by more than 0.01: {(err > 0.01).sum():,}")
    print(f"AC Vm on PQ buses: min {pq.vm_ac.min():.4f}, 0.1% {pq.vm_ac.quantile(0.001):.4f}, median {pq.vm_ac.median():.4f}, "
          f"99.9% {pq.vm_ac.quantile(0.999):.4f}, max {pq.vm_ac.max():.4f}")
    dc_err = (pq.vm_ac - 1).abs()
    print(f"DC's implicit Vm = 1.0 is off by: median {dc_err.median():.4f} / p99 {dc_err.quantile(0.99):.4f} / max {dc_err.max():.4f}")
    gens = data[data.type != "PQ"]
    out = violates(gens.vm_ac, gens.vmin, gens.vmax)
    print(f"PV/REF buses outside the case limits (setpoints, known to every solver): {out.sum()} bus-states")

    bands = [("case limits", vmin, vmax), ("0.95-1.05", 0.95, 1.05)]
    rare = rare_band(pq, vmin, vmax)
    if rare:
        bands.append((f"{rare[0]}-{rare[1]} (rare-event band)", rare[0], rare[1]))
    bands.append((f"undervoltage only (< {vmin})", vmin, np.inf))
    seen = set()
    bands = [b for b in bands if not (b[1:] in seen or seen.add(b[1:]))]

    state_info = data.groupby("state")[["over_ac", "over_dc", "over_genco"]].first()
    print(f"states with a real line overload (AC): {(state_info.over_ac > 0).sum()}; DC flags an overload in {(state_info.over_dc > 0).sum()} states")

    print("\nGENCO alone, all states, PQ buses (low/high = real violations below / above the band):")
    print(f"  {'band':<34} {'real':>7} {'low':>6} {'high':>6} {'states':>7} {'share':>6} {'missed':>7} {'false+':>7} {'recall':>7} {'precis.':>7} "
          f"{'st.unflag':>9} {'V only':>7} {'V,DC ok':>8}")
    for label, lo, hi in bands:
        r = score(pq, lo, hi)
        real = violates(pq.vm_ac, lo, hi)
        v_states = pd.Index(pq.state[real].unique())
        no_overload = int((state_info.over_ac.reindex(v_states) == 0).sum())   # voltage violation, no line overload in AC
        dc_silent = int((state_info.over_dc.reindex(v_states) == 0).sum())     # ... and DC shows no overload either
        print(f"  {label:<34} {r['real']:>7,} {(pq.vm_ac[real] < lo).sum():>6,} {(pq.vm_ac[real] > hi).sum():>6,} "
              f"{r['states_real']:>7} {r['states_real'] / n_states:>6.1%} {r['missed']:>7,} {r['false_alarms']:>7,} "
              f"{r['recall']:>7.1%} {r['precision']:>7.1%} {r['states_missed']:>9} {no_overload:>7} {dc_silent:>8}")
    print("  st.unflag = violating states where GENCO flags no bus at all; V only = violating states with no real line overload;")
    print("  V,DC ok = violating states where DC power flow shows no line overload (no warning of any kind from DC screening).")

    depth = np.maximum(vmin - pq.vm_ac, pq.vm_ac - vmax)            # how far AC is outside the case limits
    depth_genco = np.maximum(vmin - pq.vm_genco, pq.vm_genco - vmax)
    real, called = depth > TOL, depth_genco > TOL
    print(f"\nhow big are the violations of the case limits, and GENCO's mistakes on them (PQ buses, all states):")
    for d in (TOL, 0.002, 0.005, 0.01, 0.02):
        deep = depth > d
        print(f"  AC outside by more than {d:<6}: {deep.sum():>6,} bus-states in {pq.state[deep].nunique():>5} states; "
              f"GENCO calls no violation on {(deep & ~called).sum():>4,}")
    if (real & ~called).any():
        print(f"  missed violations: AC outside by median {depth[real & ~called].median():.5f}, max {depth[real & ~called].max():.5f} p.u.")
    if (called & ~real).any():
        print(f"  false alarms: AC inside the limit by median {(-depth[called & ~real]).median():.5f}, max {(-depth[called & ~real]).max():.5f} p.u.")
    on_limit = (depth.abs() <= 0.005).groupby(pq.state).any().mean()
    print(f"  states where AC has a PQ bus within 0.005 of a case limit: {on_limit:.1%} (the OPF dispatch parks voltages on the limit)")

    print(f"\nescalation to AC when GENCO puts a PQ bus within +-d of a limit; scored on the test half ({test.state.nunique()} states):")
    header = f"  {'band / rule':<58} {'AC calls':>8} {'real':>6} {'missed':>7} {'of real':>8} {'states':>7} {'false+':>7}"
    for label, lo, hi in bands:
        print(f"{header}\n  [{label}]")
        rows = [("GENCO alone", score(test, lo, hi), "")]
        rows.append(("GENCO + AC within +-0.005 of a limit (fixed)", score(test, lo, hi, 0.005), ""))
        for target in TARGETS:
            best = calibrate(calibration, lo, hi, target)
            if best is None:
                print(f"    no band keeps misses within {target:.0%} on the calibration half")
                continue
            band, fitted = best
            rule = f"GENCO + AC within +-{band} (calibrated, target <= {target:.0%})" if band else f"never escalate (calibrated, target <= {target:.0%})"
            rows.append((rule, score(test, lo, hi, band),
                         f"   (calibration: {fitted['ac_calls']:.1%} calls, {fitted['missed']} of {fitted['real']} missed)"))
        for rule, r, note in rows:
            print(f"    {rule:<56} {r['ac_calls']:>8.1%} {r['real']:>6,} {r['missed']:>7,} {r['missed_share']:>8.2%} "
                  f"{r['states_with_miss']:>7} {r['false_alarms']:>7,}{note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=["118", "texas", "both"], default="both")
    parser.add_argument("--reuse", action="store_true", help="report from the saved sweeps instead of solving again")
    args = parser.parse_args()
    jobs = [("118", "IEEE 118", OUT_118, collect_118), ("texas", "Texas 2000", OUT_TEXAS, collect_texas)]
    for key, name, out, collect in jobs:
        if args.grid not in (key, "both"):
            continue
        if args.reuse and out.exists():
            data = pd.read_parquet(out)
        else:
            if out.exists():
                raise SystemExit(f"{out} exists; pass --reuse or move it away (this script never overwrites a sweep)")
            start = time.perf_counter()
            data = collect()
            data.to_parquet(out, index=False)
            print(f"saved {out} after {time.perf_counter() - start:.0f} s", flush=True)
        report(data, name)


if __name__ == "__main__":
    main()

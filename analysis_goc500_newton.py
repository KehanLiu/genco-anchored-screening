"""GOC 500: does ONE Newton step on top of GENCO Tiny fix the heavy error tail? Same rule comparison as
analysis_goc500.py (their F3 threshold rule vs our label-conditional split conformal screening), for

    genco, dc            raw surrogates (as before)
    genco1, genco2       GENCO voltages + 1 / 2 full Newton steps
    dc1, flat1           controls: DC angles / flat start + 1 Newton step

plus a mismatch-scaled conformal band: sigma of a scenario = bus power mismatch left after the step(s)
(max |dP|,|dQ| in MW/MVAr, free to compute, no AC solution needed) + a floor; a line is "possibly overloaded"
when prediction >= limit - b * sigma, b calibrated conformally. The floor is picked inside the calibration half.

Reads work/goc500/lines_newton_tiny_seed<S>.parquet and states_newton_tiny_seed<S>.parquet (work/goc500/02_newton.py).

    .venv/bin/python analysis_goc500_newton.py --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_conformal as ac
import analysis_goc500 as g5
from analysis_conformal import conformal_width, evaluate

SURROGATES = ["genco", "genco1", "genco2", "dc", "dc1", "flat1"]
STEPPED = {"genco1": "mis1_genco", "genco2": "mis2_genco", "dc1": "mis1_dc", "flat1": "mis1_flat"}
g5.NAMES.update({"genco1": "GENCO+1", "genco2": "GENCO+2", "dc1": "DC+1", "flat1": "flat+1"})
FLOORS = [0.01, 0.1, 1.0, 10.0]  # MW
BIG = 9.99  # loading used when a Newton step broke down (always escalated)
ROOT = Path(__file__).resolve().parent / "work" / "goc500"


def outcome(test, n_test, n_critical, surrogate, sigma, width):
    r = evaluate(test, n_test, surrogate, sigma, width)
    r["critical"], r["critical_missed"] = n_critical, r["states_with_miss"]
    r["recall"] = 1 - r["critical_missed"] / max(n_critical, 1)
    r["precision"] = (n_critical - r["critical_missed"]) / max(round(r["ac_calls"] * n_test), 1)
    return r


def table(rows):
    out = [f"  {'model':<8} {'rule':<30} {'band/t':>10} {'recall':>7} {'AC calls':>8} {'on secure':>9} {'missed scen.':>13} "
           f"{'missed lines':>13} {'of real':>8} {'95% upper':>9} {'worst':>7}"]
    for r in rows:
        worst = "" if np.isnan(r["worst_missed"]) else f"{r['worst_missed']:.1%}"
        out.append(f"  {g5.NAMES[r['surrogate']]:<8} {r['rule']:<30} {r['band']:>10} {r['recall']:>7.1%} {r['ac_calls']:>8.1%} {r['wasted']:>9.1%} "
                   f"{r['critical_missed']:>6}/{r['critical']:<6} {r['missed']:>6}/{r['real']:<6} {r['missed_share']:>8.2%} {r['missed_upper95']:>9.2%} {worst:>7}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", type=int, default=100)
    a = p.parse_args()
    full = pd.read_parquet(ROOT / f"lines_newton_tiny_seed{a.seed}.parquet")
    st = pd.read_parquet(ROOT / f"states_newton_tiny_seed{a.seed}.parquet").set_index("state")
    broken = {s: int(full[s].isna().groupby(full.state).any().sum()) for s in STEPPED}
    full[list(STEPPED)] = full[list(STEPPED)].fillna(BIG)
    states = np.sort(full.state.unique())
    print(f"GENCO Tiny seed {a.seed}: {len(states):,} scenarios, {len(full):,} line readings; real overload = AC loading > {ac.OVERLOAD}; "
          f"scenarios with a broken Newton step: {broken}")

    # accuracy
    print("\n1. absolute loading error, points of rating: median / p99 / p99.9 / max")
    k = full.n_out.to_numpy()
    near = (full.ac > 0.9).to_numpy()
    print(f"  {'':<9} {'all readings':>30} {'AC loading > 90%':>30} {'n_out = 1':>30} {'n_out = 5':>30}")
    for s in SURROGATES:
        e = (full[s] - full.ac).abs().to_numpy() * 100
        cells = [f"{np.median(e[m]):.3f} / {np.quantile(e[m], .99):.2f} / {np.quantile(e[m], .999):.2f} / {e[m].max():.1f}"
                 for m in (np.ones(len(e), bool), near, k == 1, k == 5)]
        print(f"  {g5.NAMES[s]:<9} " + " ".join(f"{c:>30}" for c in cells))
    print("  bus power mismatch after the step, MW/MVAr (median / p90 / p99 / max): " + "; ".join(
        f"{c} {st[c].median():.3g} / {st[c].quantile(.9):.3g} / {st[c].quantile(.99):.3g} / {st[c].max():.3g}" for c in ("mis0_genco", "mis1_genco", "mis2_genco", "mis1_dc", "mis1_flat")))

    per_state = full.groupby("state")[["ac"] + SURROGATES].max()
    per_state["critical"] = per_state.ac > ac.OVERLOAD
    print("  raw verdict at t = limit (no band): " + "; ".join(
        f"{g5.NAMES[s]} flags {(per_state[s] > ac.OVERLOAD).mean():.1%}, recall {((per_state[s] > ac.OVERLOAD) & per_state.critical).sum() / per_state.critical.sum():.2%}" for s in SURROGATES))

    data = full[full[["ac"] + SURROGATES].max(axis=1) > g5.KEEP].reset_index(drop=True)
    for s, col in STEPPED.items():
        data[f"mis_{s}"] = data.state.map(st[col]).to_numpy()
    cal_states, test_states = g5.split(states, 0)
    cal, test = data[data.state.isin(cal_states)], data[data.state.isin(test_states)]
    n_test, n_crit = len(test_states), int(per_state.critical.loc[test_states].sum())
    print(f"\ncanonical split (seed 0, same as analysis_goc500.py): {len(cal_states):,} calibration / {n_test:,} test scenarios, "
          f"{n_crit} critical ({n_crit / n_test:.1%}), {(test.ac > ac.OVERLOAD).sum():,} overloaded lines in the test half")

    def sigma_of(frame, s, floor):
        return np.clip(frame[f"mis_{s}"].to_numpy(), 0, 1e6) + floor

    def pick_floor(frame, frame_states, s, alpha, seed=0):
        """Floor chosen inside the calibration half: calibrate on one half of it, count AC calls on the other."""
        a_s, b_s = g5.split(frame_states, 1000 + seed)
        fa, fb = frame[frame.state.isin(a_s)], frame[frame.state.isin(b_s)]
        best = (np.inf, FLOORS[0])
        for floor in FLOORS:
            w, _ = conformal_width(fa, s, sigma_of(fa, s, floor), alpha)
            calls = evaluate(fb, len(b_s), s, sigma_of(fb, s, floor), w)["ac_calls"]
            if calls < best[0]:
                best = (calls, floor)
        return best[1]

    # their rule, t from calibration
    rows = []
    for s in SURROGATES:
        t, _ = g5.fbeta_threshold(per_state.loc[cal_states, s].to_numpy(), per_state.loc[cal_states, "critical"].to_numpy())
        rows.append({"surrogate": s, "rule": "F3, t from calibration", "band": f"t={t:.3f}", **outcome(test, n_test, n_crit, s, 1.0, ac.OVERLOAD - np.nextafter(t, 2))})
    print("\n2. THEIR rule (scenario flagged if any predicted loading > t; t maximises scenario-level F3 on the calibration half), test half")
    print(table(rows))

    # conformal
    crows = []
    for s in SURROGATES:
        for alpha in ac.ALPHAS:
            w, n_cal = conformal_width(cal, s, 1.0, alpha)
            crows.append({"surrogate": s, "rule": f"conformal global a={alpha:.0%}", "alpha": alpha, "band": f"{w * 100:.2f} pt", **outcome(test, n_test, n_crit, s, 1.0, w)})
    print(f"\n3. OUR rule, label-conditional split conformal, one global band (calibrated on {n_cal:,} real overloads), test half")
    print(table(crows))
    mrows = []
    for s in STEPPED:
        for alpha in ac.ALPHAS:
            floor = pick_floor(cal, cal_states, s, alpha)
            w, _ = conformal_width(cal, s, sigma_of(cal, s, floor), alpha)
            mrows.append({"surrogate": s, "rule": f"conformal mismatch a={alpha:.0%}", "alpha": alpha, "band": f"{w * 100:.3f}pt/MW" + f" f={floor:g}",
                          **outcome(test, n_test, n_crit, s, sigma_of(test, s, floor), w)})
    print("\n4. mismatch-scaled band: possibly overloaded if prediction >= limit - b * (post-step mismatch [MW] + floor f [MW]); b conformal, f picked inside the calibration half")
    print(table(mrows))

    # re-splits
    out = []
    for seed in range(1, a.splits + 1):
        c_s, t_s = g5.split(states, seed)
        c, t = data[data.state.isin(c_s)], data[data.state.isin(t_s)]
        for s in SURROGATES:
            for alpha in ac.ALPHAS:
                w, _ = conformal_width(c, s, 1.0, alpha)
                out.append({"surrogate": s, "sigma": "global", "alpha": alpha, **evaluate(t, len(t_s), s, 1.0, w)})
                if s in STEPPED:
                    floor = pick_floor(c, c_s, s, alpha, seed)
                    w, _ = conformal_width(c, s, sigma_of(c, s, floor), alpha)
                    out.append({"surrogate": s, "sigma": "mismatch", "alpha": alpha, **evaluate(t, len(t_s), s, sigma_of(t, s, floor), w)})
    sp = pd.DataFrame(out)
    sp.to_csv(ROOT / f"conformal_newton_tiny_seed{a.seed}_splits.csv", index=False)
    print(f"\n5. coverage check, {a.splits} random 50/50 re-splits of scenarios")
    print(f"   {'model':<8} {'sigma':<9} {'alpha':>5} {'per-line miss rate':>18} {'splits > alpha':>14} {'AC calls mean':>13} {'p10-p90':>13} {'on secure':>9} {'missed (state-level)':>20}")
    for (s, kind, al), g in sp.groupby(["surrogate", "sigma", "alpha"], sort=False):
        print(f"   {g5.NAMES[s]:<8} {kind:<9} {al:>5.0%} {g.line_missed_share.mean():>18.2%} {(g.line_missed_share > al).mean():>14.0%} {g.ac_calls.mean():>13.1%} "
              f"{g.ac_calls.quantile(0.1):>6.1%}-{g.ac_calls.quantile(0.9):<6.1%} {g.wasted.mean():>9.1%} {g.missed_share.mean():>20.2%}")

    # budget
    overloads = (test.ac > ac.OVERLOAD).groupby(test.state).sum().reindex(test_states, fill_value=0)
    frames = []
    for s in SURROGATES + ["oracle"]:
        order = per_state.loc[test_states, "ac" if s == "oracle" else s].sort_values(ascending=False).index
        frames.append(pd.DataFrame({"surrogate": s, "ac_calls": np.arange(1, n_test + 1) / n_test,
                                    "found_lines": overloads.loc[order].cumsum().to_numpy() / overloads.sum(),
                                    "found_scenarios": per_state.critical.loc[order].cumsum().to_numpy() / n_crit}))
    curves = pd.concat(frames, ignore_index=True)
    csv = ROOT / f"budget_newton_tiny_seed{a.seed}.csv"
    curves.to_csv(csv, index=False)
    print(f"\n6. fixed AC budget, test half: share of real overloaded lines (critical scenarios) found when the riskiest x% (by max predicted loading) go to AC  (saved {csv})")
    print(f"  {'model':<8} " + " ".join(f"{b:>15.0%}" for b in g5.BUDGETS) + "   AC share needed to find all lines")
    for s, g in curves.groupby("surrogate", sort=False):
        cells = [f"{g.found_lines[g.ac_calls <= b].max():.1%} ({g.found_scenarios[g.ac_calls <= b].max():.1%})" for b in g5.BUDGETS]
        print(f"  {g5.NAMES.get(s, s):<8} " + " ".join(f"{c:>15}" for c in cells) + f"   {g.ac_calls[g.found_lines >= 1 - 1e-12].min():.1%}")

    # cost
    it_flat, it_genco = st.it_flat.mean(), st.it_genco.mean()
    print("\n7. cost in Newton iterations (one iteration = one Jacobian build + sparse LU + solve; same SciPy implementation; GENCO inference NOT counted)")
    print(f"   full Newton to 1e-6 MW: from flat {it_flat:.2f} iterations ({st.ms_full_flat.median():.1f} ms median), from DC angles {st.it_dc.mean():.2f} ({st.ms_full_dc.median():.1f} ms), "
          f"from GENCO {it_genco:.2f} ({st.ms_full_genco.median():.1f} ms); one step from GENCO {st.ms_step1_genco.median():.2f} ms "
          f"= {st.ms_step1_genco.median() / st.ms_full_flat.median():.0%} of a full flat-start solve")
    print(f"   Newton iterations per 100 scenarios; baseline full AC everywhere from flat start = {100 * it_flat:.0f}")
    print(f"   {'rule':<44} {'AC calls':>8} {'missed lines':>12} {'escalated: flat-start AC':>25} {'escalated: continue Newton':>27}")
    extra = {"genco": 0, "dc": 0, "genco1": 1, "genco2": 2, "dc1": 1, "flat1": 1}
    cont = {"genco": it_genco, "dc": st.it_dc.mean(), "genco1": it_genco - 1, "genco2": max(it_genco - 2, 0), "dc1": st.it_dc.mean() - 1, "flat1": it_flat - 1}
    for r in rows + crows + mrows:
        s = r["surrogate"]
        if s in ("dc", "flat1") or (r["rule"].startswith("conformal") and r.get("alpha") not in (0.05, 0.01)):
            continue
        base = 100 * extra[s]
        print(f"   {g5.NAMES[s] + ' ' + r['rule']:<44} {r['ac_calls']:>8.1%} {r['missed']:>5}/{r['real']:<6} {base + 100 * r['ac_calls'] * it_flat:>25.0f} {base + 100 * r['ac_calls'] * cont[s]:>27.0f}")
    print("   ('continue Newton' = escalated scenarios keep iterating from the surrogate's point instead of restarting flat; mean iterations used, not per-scenario)")


if __name__ == "__main__":
    main()

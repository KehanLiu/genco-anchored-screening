"""Report for the GENCO + Newton-step experiment on this folder's grid (folder name -> case<N>_goc).

Reads lines_newton_<variant>_seed<S><tag>.parquet / states_newton_... (02_newton.py) and prints:
  1. loading error (points of rating) of GENCO, GENCO+1, GENCO+2, DC, DC+1, flat+1
  2. the plain rule "send the scenario to AC iff any predicted loading > 1.001" on ALL scenarios
  3. the example-post rule (t maximising scenario-level F-beta, beta = 3, on a calibration half; scored on the test half)
  4. Newton iterations / convergence, post-step mismatch, blow-ups, validation against the stored AC solution
  5. budget curve CSV budget_newton_<variant>_seed<S>.csv (test half, same layout as work/goc500/budget_newton_tiny_seed0.csv)
Helpers (split, fbeta_threshold, evaluate) are imported from analysis/analysis_goc500.py / analysis/analysis_conformal.py unchanged.

    .venv/bin/python work/goc2000/03_analysis.py --variant tiny --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import analysis_conformal as ac  # noqa: E402
import analysis_goc500 as g5  # noqa: E402

NET = f"case{HERE.name.removeprefix('goc')}_goc"
SUR = ["genco", "genco1", "genco2", "dc", "dc1", "flat1"]
NAMES = {"genco": "GENCO", "genco1": "GENCO+1", "genco2": "GENCO+2", "dc": "DC", "dc1": "DC+1", "flat1": "flat+1"}
STEPPED = ["genco1", "genco2", "dc1", "flat1"]
BIG = 9.99  # loading used where a Newton step broke down (NaN): the scenario is always escalated
LIM = ac.OVERLOAD


def plain(full, per_state, s, t=LIM, states=None):
    """Scenario goes to AC iff max predicted loading > t. Outcome over `states` (default all)."""
    ps = per_state if states is None else per_state.loc[states]
    flag = ps[s] > t
    crit = ps.critical
    sub = full if states is None else full[full.state.isin(states)]
    real = (sub.ac > LIM).to_numpy()
    accepted = ~sub.state.map(flag).to_numpy()
    missed = real & accepted
    n_real, k = int(real.sum()), int(missed.sum())
    return {"flagged": flag.mean(), "on_secure": (flag & ~crit).sum() / max((~crit).sum(), 1),
            "recall": (flag & crit).sum() / max(crit.sum(), 1), "crit_missed": int((~flag & crit).sum()), "crit": int(crit.sum()),
            "missed": k, "real": n_real, "share": k / max(n_real, 1),
            "upper95": float(beta.ppf(0.95, k + 1, n_real - k)) if n_real > k else 1.0,  # one-sided 95% Clopper-Pearson
            "worst": float(sub.ac.to_numpy()[missed].max()) if k else np.nan}


def fmt(name, rule, r):
    worst = "" if np.isnan(r["worst"]) else f"{r['worst']:.1%}"
    return (f"  {name:<8} {rule:<12} {r['flagged']:>8.1%} {r['on_secure']:>9.1%} {r['recall']:>7.1%} {r['crit_missed']:>6}/{r['crit']:<6} "
            f"{r['missed']:>6}/{r['real']:<6} {r['share']:>8.2%} {r['upper95']:>9.2%} {worst:>7}")


HEAD = (f"  {'model':<8} {'rule':<12} {'flagged':>8} {'on secure':>9} {'recall':>7} {'missed scen.':>13} {'missed lines':>13} {'of real':>8} {'95% upper':>9} {'worst':>7}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="tiny")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    stem = f"newton_{a.variant}_seed{a.seed}{a.tag}"
    full = pd.read_parquet(HERE / f"lines_{stem}.parquet")
    st = pd.read_parquet(HERE / f"states_{stem}.parquet").set_index("state")
    broken = {s: int(full[s].isna().groupby(full.state).any().sum()) for s in STEPPED}
    states = np.sort(full.state.unique())
    n_out = full.groupby("state").n_out.first()
    print(f"{NET}, GENCO {a.variant} seed {a.seed}: {len(states):,} scenarios, {len(full):,} in-service rated line readings, {full.line.nunique():,} lines; "
          f"real overload = AC loading > {LIM}")
    print("branches removed per scenario:", n_out.value_counts().sort_index().to_dict())
    print("scenarios with a broken Newton step (NaN loading, treated as flagged):", broken)

    print("\n0. validation of the Newton code against the stored AC solution")
    print(f"  stored AC voltages in this script's equations: mismatch max {st.mis_ac_stored.max():.2e} MW (median {st.mis_ac_stored.median():.2e}); "
          f"loading formula vs stored flows max {st.dload_formula_pp.max():.2e} points")
    for k in ("flat", "dc", "genco"):
        c = st[st[f"conv_{k}"]]
        far = int((c[f"dv_{k}"] > 1e-5).sum())
        print(f"  Newton from {k:<5}: converged {len(c):,}/{len(st):,}; iterations mean {c[f'it_{k}'].mean():.2f} median {c[f'it_{k}'].median():.0f} max {c[f'it_{k}'].max()}; "
              f"{c[f'ms_full_{k}'].median():.1f} ms median; max |V - V_ac stored| {c[f'dv_{k}'].max():.1e} p.u., {far} converged to a different solution (> 1e-5); "
              f"start mismatch median {st[f'mis0_{k}'].median():.1f} MW")
    print(f"  converged Newton loading vs stored AC loading: max {st.dload_newton_pp.max():.2e} points; one step from GENCO {st.ms_step1_genco.median():.2f} ms median")

    print("\n1. absolute loading error, points of rating: median / p99 / p99.9 / max   (NaN steps excluded here)")
    near = (full.ac > 0.9).to_numpy()
    print(f"  {'':<9} {'all readings':>32} {'AC loading > 90%':>32}")
    for s in SUR:
        e = (full[s] - full.ac).abs().to_numpy() * 100
        ok = np.isfinite(e)
        cells = [f"{np.median(e[m]):.3f} / {np.quantile(e[m], .99):.2f} / {np.quantile(e[m], .999):.2f} / {e[m].max():.1f}" for m in (ok, ok & near)]
        print(f"  {NAMES[s]:<9} " + " ".join(f"{c:>32}" for c in cells))
    print("  blow-ups: readings with |error| > 50 points: " + ", ".join(f"{NAMES[s]} {int(((full[s] - full.ac).abs() > 0.5).sum()):,}" for s in SUR)
          + "; scenarios with any such reading: " + ", ".join(f"{NAMES[s]} {full.state[(full[s] - full.ac).abs() > 0.5].nunique()}" for s in SUR))
    print("  bus power mismatch, MW/MVAr max-norm, median / p90 / p99 / max:")
    for c in ("mis0_genco", "mis1_genco", "mis2_genco", "mis0_dc", "mis1_dc", "mis0_flat", "mis1_flat"):
        v = st[c].replace(np.inf, np.nan)
        print(f"    {c:<11} {v.median():.3g} / {v.quantile(.9):.3g} / {v.quantile(.99):.3g} / {v.max():.3g}" + (f"   ({int(v.isna().sum())} non-finite)" if v.isna().any() else ""))

    full[STEPPED] = full[STEPPED].fillna(BIG)
    per_state = full.groupby("state")[["ac"] + SUR].max()
    per_state["critical"] = per_state.ac > LIM
    real_all = full.ac > LIM
    print(f"\n2. prevalence: {per_state.critical.sum():,} scenarios with a real overload = {per_state.critical.mean():.1%}; {real_all.sum():,} overloaded line readings on "
          f"{full.line[real_all].nunique()} distinct lines; AC > 1.000 -> {(per_state.ac > 1.0).mean():.1%}, > 1.01 -> {(per_state.ac > 1.01).mean():.1%}")
    print("   by n_out: " + ", ".join(f"{i}: {per_state.critical[n_out.loc[per_state.index] == i].mean():.1%}" for i in sorted(n_out.unique())))
    print(f"\n   PLAIN rule, all {len(states):,} scenarios: send to AC iff any predicted loading > {LIM}")
    print(HEAD)
    for s in SUR:
        print(fmt(NAMES[s], f"t={LIM}", plain(full, per_state, s)))

    cal_states, test_states = g5.split(states, 0)
    print(f"\n3. EXAMPLE-POST rule: t maximising scenario-level F3 on the calibration half ({len(cal_states):,}), scored on the test half ({len(test_states):,}; "
          f"{int(per_state.critical.loc[test_states].sum())} critical)")
    print(HEAD)
    for s in SUR:
        t, f = g5.fbeta_threshold(per_state.loc[cal_states, s].to_numpy(), per_state.loc[cal_states, "critical"].to_numpy())
        if t is None:
            print(f"  {NAMES[s]:<8} no threshold (no true positives on the calibration half)")
            continue
        print(fmt(NAMES[s], f"t={t:.3f}", plain(full, per_state, s, t, test_states)) + f"   F3(cal)={f:.3f}")

    test = full[full.state.isin(test_states)]
    overloads = (test.ac > LIM).groupby(test.state).sum().reindex(test_states, fill_value=0)
    n_test, n_crit = len(test_states), int(per_state.critical.loc[test_states].sum())
    frames = []
    for s in SUR + ["oracle"]:
        order = per_state.loc[test_states, "ac" if s == "oracle" else s].sort_values(ascending=False).index
        frames.append(pd.DataFrame({"surrogate": s, "ac_calls": np.arange(1, n_test + 1) / n_test,
                                    "found_lines": overloads.loc[order].cumsum().to_numpy() / max(overloads.sum(), 1),
                                    "found_scenarios": per_state.critical.loc[order].cumsum().to_numpy() / max(n_crit, 1)}))
    curves = pd.concat(frames, ignore_index=True)
    csv = HERE / f"budget_{stem}.csv"
    curves.to_csv(csv, index=False)
    print(f"\n4. fixed AC budget, test half: share of real overloaded lines (critical scenarios) found when the riskiest x% go to AC  (saved {csv})")
    print(f"  {'model':<8} " + " ".join(f"{b:>15.0%}" for b in g5.BUDGETS) + "   AC share needed to find all lines")
    for s, g in curves.groupby("surrogate", sort=False):
        cells = [f"{g.found_lines[g.ac_calls <= b].max():.1%} ({g.found_scenarios[g.ac_calls <= b].max():.1%})" for b in g5.BUDGETS]
        print(f"  {NAMES.get(s, s):<8} " + " ".join(f"{c:>15}" for c in cells) + f"   {g.ac_calls[g.found_lines >= 1 - 1e-12].min():.1%}")
    if "ms_genco_e2e" in st:
        print(f"\n5. per-state ms while {'several'} workers ran in parallel (indicative): GENCO end-to-end {st.ms_genco_e2e.median():.0f}, Newton full flat {st.ms_full_flat.median():.0f}, "
              f"from GENCO {st.ms_full_genco.median():.0f}, one step {st.ms_step1_genco.median():.0f}")


if __name__ == "__main__":
    main()

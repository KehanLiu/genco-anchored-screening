"""Head-to-head on the organisers' example setup (GOC 500, random removal of up to 5 branches):
their fixed-threshold rule (t maximising scenario-level F-beta, beta=3) against our label-conditional
split conformal screening rule, for GENCO Tiny and for DC power flow.

Reads the per-line table written by work/goc500/01_eval.py (state, n_out, line, ac, genco, dc, load_idx).
One definition of a real overload for every rule: AC loading > OVERLOAD (default 1.001 as in engine.py /
analysis/analysis_conformal.py; --overload 1.0 to change). A scenario is critical if any in-service rated line is
overloaded. Scenarios are split 50/50 at random (seed 0) into calibration and test.

    .venv/bin/python analysis/analysis_goc500.py --reuse work/goc500/lines_tiny_seed0.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))  # sibling analysis modules, also when imported from work/
import analysis_conformal as ac  # noqa: E402
from analysis_conformal import conformal_width, evaluate

SURROGATES = ["genco", "dc"]
NAMES = {"genco": "GENCO", "dc": "DC"}
BETA = 3.0
GRID = np.round(np.arange(0.500, 1.2001, 0.001), 3)
BUDGETS = [0.05, 0.10, 0.20, 0.35, 0.50]
KEEP = 0.5  # readings with AC and both surrogates below this cannot change any decision for thresholds >= KEEP


def split(states, seed, groups=None):
    """Random 50/50 split of scenarios (or of whole load scenarios when groups is given)."""
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(states)
        return perm[: len(perm) // 2], perm[len(perm) // 2:]
    keys = rng.permutation(np.unique(groups))
    cal = np.isin(groups, keys[: len(keys) // 2])
    return states[cal], states[~cal]


def fbeta_threshold(score, truth):
    """t maximising scenario-level F-beta of 'max predicted loading > t'."""
    best = (-1.0, None)
    n_true = truth.sum()
    for t in GRID:
        flag = score > t
        tp = (flag & truth).sum()
        if tp == 0:
            continue
        prec, rec = tp / flag.sum(), tp / n_true
        f = (1 + BETA**2) * prec * rec / (BETA**2 * prec + rec)
        if f > best[0]:
            best = (f, t)
    return best[1], best[0]


def rule(test, per_state, test_states, surrogate, threshold):
    """Send every scenario with a predicted loading >= threshold to AC. Scenario- and line-level outcome."""
    r = evaluate(test, len(test_states), surrogate, 1.0, ac.OVERLOAD - threshold)
    ps = per_state.loc[test_states]
    flag, truth = (ps[surrogate] >= threshold).to_numpy(), ps.critical.to_numpy()
    r["recall"] = (flag & truth).sum() / max(truth.sum(), 1)
    r["precision"] = (flag & truth).sum() / max(flag.sum(), 1)
    r["critical_missed"] = int((~flag & truth).sum())
    r["critical"] = int(truth.sum())
    return r


def rule_table(rows):
    out = [f"  {'model':<6} {'rule':<26} {'t':>6} {'recall':>7} {'flagged':>8} {'precision':>9} {'on secure':>9} {'missed scen.':>13} "
           f"{'missed lines':>13} {'of real':>8} {'95% upper':>9} {'worst':>7}"]
    for r in rows:
        worst = "" if np.isnan(r["worst_missed"]) else f"{r['worst_missed']:.1%}"
        t = "all->AC" if np.isinf(r["t"]) else f"{r['t']:.3f}"
        out.append(f"  {NAMES[r['surrogate']]:<6} {r['rule']:<26} {t:>7} {r['recall']:>7.1%} {r['ac_calls']:>8.1%} {r['precision']:>9.1%} {r['wasted']:>9.1%} "
                   f"{r['critical_missed']:>6}/{r['critical']:<6} {r['missed']:>6}/{r['real']:<6} {r['missed_share']:>8.2%} {r['missed_upper95']:>9.2%} {worst:>7}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reuse", type=Path, default=Path("work/goc500/lines_tiny_seed0.parquet"), help="per-line parquet from work/goc500/01_eval.py")
    p.add_argument("--splits", type=int, default=100)
    p.add_argument("--overload", type=float, default=ac.OVERLOAD)
    p.add_argument("--budget-csv", type=Path, default=None)
    a = p.parse_args()
    ac.OVERLOAD = a.overload  # the imported conformal functions read this module constant
    full = pd.read_parquet(a.reuse)
    states = np.sort(full.state.unique())
    n_out = full.groupby("state").n_out.first()
    print(f"{a.reuse}: {len(states):,} scenarios, {len(full):,} in-service rated line readings, {full.line.nunique()} lines; "
          f"real overload = AC loading > {ac.OVERLOAD}")
    print("branches removed per scenario:", n_out.value_counts().sort_index().to_dict())

    # a. accuracy
    print("\na. absolute loading error, points of rating: median / p99 / max  (checkpoint was trained on N-1 only; n_out >= 2 is zero-shot)")
    print(f"  {'n_out':<6} {'readings':>10}   {'GENCO':>24}   {'DC':>24}")
    err = {s: (full[s] - full.ac).abs().to_numpy() * 100 for s in SURROGATES}
    k = full.n_out.to_numpy()
    for label, m in [("all", np.ones(len(full), bool))] + [(str(i), k == i) for i in sorted(n_out.unique())]:
        cells = [f"{np.median(err[s][m]):.3f} / {np.quantile(err[s][m], 0.99):.2f} / {err[s][m].max():.1f}" for s in SURROGATES]
        print(f"  {label:<6} {m.sum():>10,}   {cells[0]:>24}   {cells[1]:>24}")
    near = (full.ac > 0.9).to_numpy()
    print("  readings with AC loading > 90%: " + ", ".join(f"{NAMES[s]} {np.median(err[s][near]):.3f} / {np.quantile(err[s][near], 0.99):.2f} / {err[s][near].max():.1f}" for s in SURROGATES)
          + f"  ({near.sum():,} readings)")

    # b. prevalence
    per_state = full.groupby("state")[["ac"] + SURROGATES].max()
    per_state["critical"] = per_state.ac > ac.OVERLOAD
    real_all = full.ac > ac.OVERLOAD
    print(f"\nb. prevalence: {real_all.sum():,} overloaded line readings ({real_all.mean():.3%} of readings) on {full.line[real_all].nunique()} distinct lines; "
          f"{per_state.critical.sum():,} critical scenarios = {per_state.critical.mean():.1%}")
    print("   critical share by n_out: " + ", ".join(f"{i}: {per_state.critical[n_out.loc[per_state.index] == i].mean():.1%}" for i in sorted(n_out.unique())))
    print(f"   sensitivity to the definition: AC > 1.000 -> {(per_state.ac > 1.0).mean():.1%} of scenarios, > 1.001 -> {(per_state.ac > 1.001).mean():.1%}, "
          f"> 1.01 -> {(per_state.ac > 1.01).mean():.1%}  (the OPF dispatch parks congested lines at exactly 100%)")
    print("   what the raw predictions say at t = limit: " + ", ".join(
        f"{NAMES[s]} flags {(per_state[s] > ac.OVERLOAD).mean():.1%} of scenarios, recall {((per_state[s] > ac.OVERLOAD) & per_state.critical).sum() / per_state.critical.sum():.1%}"
        for s in SURROGATES))

    data = full[full[["ac"] + SURROGATES].max(axis=1) > KEEP].reset_index(drop=True)
    assert set(data.state.unique()) == set(states)
    groups = full.groupby("state").load_idx.first().loc[states].to_numpy() if "load_idx" in full else None
    cal_states, test_states = split(states, 0)
    cal, test = data[data.state.isin(cal_states)], data[data.state.isin(test_states)]
    n_real_test = int((test.ac > ac.OVERLOAD).sum())
    print(f"\ncanonical split (seed 0): {len(cal_states):,} calibration / {len(test_states):,} test scenarios; test half has "
          f"{per_state.critical.loc[test_states].sum():,} critical scenarios ({per_state.critical.loc[test_states].mean():.1%}), {n_real_test:,} overloaded lines")

    # c. their rule
    rows = []
    for s in SURROGATES:
        for name, st in (("F3, t from calibration", cal_states), ("F3, t tuned on test", test_states)):
            t, f = fbeta_threshold(per_state.loc[st, s].to_numpy(), per_state.loc[st, "critical"].to_numpy())
            rows.append({"surrogate": s, "rule": name, "t": t, "fbeta": f, **rule(test, per_state, test_states, s, np.nextafter(t, 2))})
    print("\nc. THEIR rule: scenario critical if any predicted loading > t, t maximising scenario-level F-beta (beta=3); evaluated on the test half")
    print(rule_table(rows))
    print("   F3 at the chosen t: " + ", ".join(f"{NAMES[r['surrogate']]} {r['rule'].split(', ')[1]} {r['fbeta']:.3f}" for r in rows))
    print("   example post (their own Tiny, trained on this perturbation): GENCO recall 94.3%, 32.7% flagged; DC recall 99.8%, 98.5% flagged; prevalence 10.7%")

    # d. our rule
    crows = []
    for s in SURROGATES:
        for alpha in ac.ALPHAS:
            width, n_cal = conformal_width(cal, s, 1.0, alpha)
            crows.append({"surrogate": s, "rule": f"conformal alpha={alpha:.0%}", "t": ac.OVERLOAD - width, "alpha": alpha, "n_cal": n_cal, "width": width,
                          **rule(test, per_state, test_states, s, ac.OVERLOAD - width)})
    print(f"\nd. OUR rule: label-conditional split conformal (analysis_conformal.conformal_width / evaluate, sigma = global), "
          f"calibrated on {crows[0]['n_cal']:,} real overloads of the calibration half; evaluated on the test half")
    print(rule_table(crows))
    print("   band width below the limit, points of rating: " + "; ".join(
        f"{NAMES[s]} " + " / ".join(f"{r['width'] * 100:.1f}" for r in crows if r["surrogate"] == s) for s in SURROGATES) + "  (alpha 10/5/2/1%)")
    print("   'flagged' = AC-call share, 'on secure' = share of secure scenarios sent to AC anyway, 'missed lines' = real overloads in scenarios accepted without AC")

    def resplit(grp):
        out = []
        for seed in range(1, a.splits + 1):
            c_s, t_s = split(states, seed, grp)
            c, t = data[data.state.isin(c_s)], data[data.state.isin(t_s)]
            for s in SURROGATES:
                for alpha in ac.ALPHAS:
                    width, _ = conformal_width(c, s, 1.0, alpha)
                    out.append({"surrogate": s, "alpha": alpha, "seed": seed, "width": width, **evaluate(t, len(t_s), s, 1.0, width)})
        return pd.DataFrame(out)

    for title, grp in (("random 50/50 re-splits of scenarios", None), ("re-splits by whole load scenario (variants of one dispatch stay together)", groups)):
        if title.startswith("re-splits by") and grp is None:
            continue
        sp = resplit(grp)
        print(f"\n   coverage check, {a.splits} {title}:")
        print(f"   {'model':<6} {'alpha':>5} {'per-line miss rate':>18} {'p10-p90':>15} {'splits > alpha':>14} {'AC calls mean':>13} {'p10-p90':>13} {'on secure':>9} {'missed (state-level)':>20}")
        for (s, al), g in sp.groupby(["surrogate", "alpha"], sort=False):
            print(f"   {NAMES[s]:<6} {al:>5.0%} {g.line_missed_share.mean():>18.2%} {g.line_missed_share.quantile(0.1):>7.2%}-{g.line_missed_share.quantile(0.9):<7.2%} "
                  f"{(g.line_missed_share > al).mean():>14.0%} {g.ac_calls.mean():>13.1%} {g.ac_calls.quantile(0.1):>6.1%}-{g.ac_calls.quantile(0.9):<6.1%} {g.wasted.mean():>9.1%} {g.missed_share.mean():>20.2%}")

    # e. fixed budget
    overloads = (test.ac > ac.OVERLOAD).groupby(test.state).sum().reindex(test_states, fill_value=0)
    frames = []
    for s in SURROGATES + ["oracle"]:
        risk = per_state.loc[test_states, "ac" if s == "oracle" else s]
        order = risk.sort_values(ascending=False).index
        found = overloads.loc[order].cumsum().to_numpy() / overloads.sum()
        crit = per_state.critical.loc[order].cumsum().to_numpy() / per_state.critical.loc[test_states].sum()
        frames.append(pd.DataFrame({"surrogate": s, "ac_calls": np.arange(1, len(order) + 1) / len(order), "found_lines": found, "found_scenarios": crit}))
    curves = pd.concat(frames, ignore_index=True)
    out = a.budget_csv or a.reuse.with_name("budget.csv" if a.reuse.stem == "lines_tiny_seed0" else f"budget_{a.reuse.stem.removeprefix('lines_')}.csv")
    curves.to_csv(out, index=False)
    print(f"\ne. fixed AC budget on the test half: share of real overloaded lines (critical scenarios) found when the riskiest x% of scenarios, "
          f"ranked by max predicted loading, go to AC  (saved {out})")
    print(f"  {'model':<7} " + " ".join(f"{b:>15.0%}" for b in BUDGETS) + "   AC share needed to find all lines")
    for s, g in curves.groupby("surrogate", sort=False):
        cells = [f"{g.found_lines[g.ac_calls <= b].max():.1%} ({g.found_scenarios[g.ac_calls <= b].max():.1%})" for b in BUDGETS]
        print(f"  {NAMES.get(s, s):<7} " + " ".join(f"{c:>15}" for c in cells) + f"   {g.ac_calls[g.found_lines >= 1 - 1e-12].min():.1%}")


if __name__ == "__main__":
    main()

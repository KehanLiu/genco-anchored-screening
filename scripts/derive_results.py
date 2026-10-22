#!/usr/bin/env python
"""Derive the few post numbers that live only in the gitignored per-case parquet tables.

Writes results/derived/*.csv (small, tracked). scripts/headline_numbers.py reads those files in CI;
this script needs the saved result tables under work/ (see reproduce.sh, "analysis") and runs only
on a machine that has them. Every value here is also printed by an existing analysis script, but
only to its log; this writes the same quantities as tables so that the CI check does not read logs.

    .venv/bin/python scripts/derive_results.py

Outputs
    change_mechanism.csv        per grid/checkpoint: GENCO's predicted change vs the true change on
                                lines moving > 1 point (analysis_hybrid_start.py report, "lines
                                moving > 1 point" line) and GENCO ms per state
    fulln1_w4_missed.csv        the critical outages missed by the anchored workflow (W4) on the
                                18 scored bases of the GOC 2000 N-1 study (analysis_full_n1.py report)
    goc500_example_rule.csv     the organisers' F3-threshold rule on GOC 500 (analysis_goc500.py, c.)
    conformal_case118_seed0.csv the canonical seed-0 split of the IEEE 118 conformal study
                                (analysis_conformal.py, "canonical split")
    finetune_scenario_counts.csv train/val/test scenario counts of the fine-tuning runs
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "derived"
sys.path.insert(0, str(ROOT / "analysis"))

OVERLOAD = 1.001  # analysis_conformal.OVERLOAD, the loading cutoff of every screen


def change_mechanism():
    """Same mask and formulas as analysis_hybrid_start.py report (lines moving > 1 point)."""
    rows = []
    for grid, ckpt in [(500, "tiny_seed0"), (2000, "tiny_seed0"), (10000, "tiny_seed0"), (10000, "tiny_seed1"),
                       (10000, "small_seed0"), (10000, "small_seed1")]:
        d = ROOT / "work" / f"goc{grid}"
        full = pd.read_parquet(d / f"lines_hybrid_{ckpt}.parquet", columns=["ac", "ac_prev", "genco_prev", "genco_now"])
        dt = (full.ac - full.ac_prev).to_numpy()
        dp = (full.genco_now - full.genco_prev).to_numpy()
        m = np.isfinite(dt) & np.isfinite(dp) & (np.abs(dt) > 0.01)
        nochange = 100 * np.median(np.abs(dt[m]))
        err = 100 * np.median(np.abs(dp[m] - dt[m]))
        st = pd.read_parquet(d / f"states_hybrid_{ckpt}.parquet")
        if "ms_genco" in st:
            ms = float(st.ms_genco.median())  # analysis_hybrid_start.py: "GENCO ms per contingency state ... median"
        else:  # tiny seed 0 was timed in the Newton sweep (work/goc10000/03_analysis.py: "GENCO end-to-end")
            sn = pd.read_parquet(d / f"states_newton_{ckpt}.parquet")
            ms = float(sn.ms_genco_e2e.median()) if "ms_genco_e2e" in sn else float("nan")
        rows.append({"grid": grid, "checkpoint": ckpt, "moving_readings": int(m.sum()),
                     "nochange_err_median_pp": nochange, "change_err_median_pp": err,
                     "removed_share": 1 - err / nochange,
                     "sign_agreement": float((np.sign(dt[m]) == np.sign(dp[m])).mean()),
                     "genco_ms_per_state_median": ms})
    return pd.DataFrame(rows)


def fulln1_w4_missed():
    c = pd.read_parquet(ROOT / "work" / "fulln1" / "contingencies.parquet")
    c = c[c.split.eq("test") & c.critical & c.W4_missed]  # the report's own miss flag, written per outage
    out = c[["base", "branch", "worst_loading", "mis1_hyb", "max1_hyb"]]
    return out.sort_values("worst_loading").reset_index(drop=True)


def goc500_example_rule():
    import analysis_goc500 as g  # noqa: E402
    full = pd.read_parquet(ROOT / "work" / "goc500" / "lines_tiny_seed0.parquet", columns=["state", "ac", "genco", "dc"])
    states = np.sort(full.state.unique())
    per_state = full.groupby("state")[["ac", "genco", "dc"]].max()
    per_state["critical"] = per_state.ac > OVERLOAD
    cal_states, test_states = g.split(states, 0)
    rows = []
    for s in ("genco", "dc"):
        for name, st in (("F3, t from calibration", cal_states), ("F3, t tuned on test", test_states)):
            t, f = g.fbeta_threshold(per_state.loc[st, s].to_numpy(), per_state.loc[st, "critical"].to_numpy())
            ps = per_state.loc[test_states]
            flag, truth = (ps[s] >= np.nextafter(t, 2)).to_numpy(), ps.critical.to_numpy()
            rows.append({"surrogate": s, "rule": name, "t": t, "fbeta": f, "test_states": len(test_states),
                         "recall": (flag & truth).sum() / truth.sum(), "flagged_share": flag.mean(),
                         "critical_missed": int((~flag & truth).sum()), "critical": int(truth.sum())})
    return pd.DataFrame(rows)


def conformal_case118_seed0():
    import analysis_conformal as ac  # noqa: E402
    data, errors, n_states = ac.load(ROOT / "work" / "hybrid_case118_ieee_base.parquet")
    real = data.ac > OVERLOAD
    rows = pd.DataFrame(ac.run(data, errors, n_states, 0))
    rows.insert(0, "states_with_real_overload_share", data.state[real].nunique() / n_states.sum())
    return rows


def finetune_scenario_counts():
    rows = []
    for grid, model in ((500, "pf_case500_goc_ft_seed0"), (2000, "pf_case2000_goc_ft2000_seed0"), (10000, "pf_case10000_goc_ft10000_seed0")):
        p = ROOT / "work" / "finetune" / "models" / model / "stats" / f"case{grid}_goc_scenario_splits.json"
        d = json.loads(p.read_text())
        rows.append({"grid": grid, "model": model, **{k: len(d[k]) for k in ("train", "val", "test") if k in d}})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("change_mechanism", change_mechanism), ("fulln1_w4_missed", fulln1_w4_missed),
                     ("goc500_example_rule", goc500_example_rule), ("conformal_case118_seed0", conformal_case118_seed0),
                     ("finetune_scenario_counts", finetune_scenario_counts)):
        df = fn()
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"== {name}.csv ({len(df)} rows)")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

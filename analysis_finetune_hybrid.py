"""Before / after fine-tuning with the HYBRID start: tables from work/finetune/04_hybrid_eval.py outputs.

Predictors (line loading after k Newton steps): genco1 / genco2 from GENCO's voltages, prev1 / prev2 from the
pre-outage AC solution (model independent -> the baseline row of every table), hyb1 / hyb2 from the hybrid start
(pre-outage AC voltages + this model's predicted change). Plain rule as in analysis_finetune.py: a scenario is
flagged (goes to the full solver) if any predicted loading > 1.001; an overloaded line (AC loading > 1.001) is
missed if its scenario is not flagged. The upper bound on the miss share is the one-sided 95% Clopper-Pearson bound
(lines treated as independent trials - optimistic, lines of one scenario are correlated). Tables for all scenarios
and for the plain N-1 subset (n_out == 1), plus per model: start mismatch, Newton iterations, and whether the
predicted change carries information (sign agreement / correlation of predicted vs true loading change).

    .venv/bin/python analysis_finetune_hybrid.py released=work/finetune/results/lines_hybrid_500_released.parquet \
        ft_seed0=work/finetune/results/lines_hybrid_500_ft_seed0.parquet  released=work/finetune/results/lines_hybrid_2000_released.parquet ...

Each argument is label=path of a lines_hybrid_<grid>_<tag>.parquet (grid read from the file name, or label=grid:path);
the states parquet is found by replacing "lines_" with "states_". One table per grid; missing files are skipped with a
warning. Writes work/finetune/results/summary_hybrid_<grid>.csv.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

OVERLOAD = 1.001
BIG = 9.99
ROOT = Path(__file__).resolve().parent
MODEL_PREDICTORS = ["genco1", "hyb1", "genco2", "hyb2"]
BASELINE = ["prev1", "prev2"]


def upper95(k, n):
    return float(beta.ppf(0.95, k + 1, n - k)) if n > k else 1.0


def rule(lines, col):
    e = (lines[col] - lines.ac).abs().to_numpy() * 100
    e = e[np.isfinite(e)]
    pred = lines[col].fillna(BIG)
    per = pd.DataFrame({"ac": lines.ac, "p": pred}).groupby(lines.state).max()
    flagged, critical = per.p > OVERLOAD, per.ac > OVERLOAD
    over = lines.ac.to_numpy() > OVERLOAD
    missed = over & ~lines.state.map(flagged).to_numpy()
    k, n = int(missed.sum()), int(over.sum())
    return {"scenarios": len(per), "err_median": float(np.median(e)), "err_p99": float(np.quantile(e, .99)), "err_max": float(e.max()),
            "nan_readings": int(lines[col].isna().sum()), "flagged": int(flagged.sum()), "secure": int((~critical).sum()),
            "secure_flagged": int((flagged & ~critical).sum()), "critical": int(critical.sum()), "critical_missed": int((critical & ~flagged).sum()),
            "overloaded": n, "missed": k, "missed_share": k / max(n, 1), "missed_upper95": upper95(k, n) if n else np.nan,
            "worst": float(lines.ac.to_numpy()[missed].max()) if k else np.nan}


def change_info(lines):
    dt, dp = (lines.ac - lines.ac_prev).to_numpy(), (lines.genco - lines.genco_prev).to_numpy()
    m = np.isfinite(dt) & np.isfinite(dp) & (np.abs(dt) > 0.01)
    if m.sum() < 3:
        return {"chg_n": int(m.sum()), "chg_sign": np.nan, "chg_corr": np.nan, "chg_slope": np.nan}
    return {"chg_n": int(m.sum()), "chg_sign": float((np.sign(dt[m]) == np.sign(dp[m])).mean()), "chg_corr": float(np.corrcoef(dt[m], dp[m])[0, 1]),
            "chg_slope": float(np.polyfit(dt[m], dp[m], 1)[0])}


def state_info(st):
    f = lambda v: v.replace(np.inf, np.nan)  # noqa: E731
    out = {}
    for k in ("prev", "genco", "hyb"):
        if f"mis0_{k}" not in st:
            continue
        conv = st[f"conv_{k}"].fillna(False).astype(bool)
        out.update({f"mis0_{k}": f(st[f"mis0_{k}"]).median(), f"mis0_{k}_p99": f(st[f"mis0_{k}"]).quantile(.99), f"mis1_{k}": f(st[f"mis1_{k}"]).median(),
                    f"mis1_{k}_p99": f(st[f"mis1_{k}"]).quantile(.99), f"it_{k}": st.loc[conv, f"it_{k}"].mean(), f"notconv_{k}": int((~conv).sum())})
    return out


def show(df, title):
    print(f"\n{title}")
    print(f"  {'model':<22} {'predictor':<9} {'err median':>10} {'p99':>8} {'max':>8} | {'flagged':>13} {'secure flagged':>20} | {'lines missed':>14} {'share':>7} {'<= (95%)':>8} {'worst':>7}")
    last = None
    for r in df.itertuples():
        if last is not None and r.model != last:
            print("  " + "-" * 138)
        last = r.model
        worst = "" if np.isnan(r.worst) else f"{r.worst:.1%}"
        up = "" if np.isnan(r.missed_upper95) else f"{r.missed_upper95:.2%}"
        print(f"  {r.model:<22} {r.predictor:<9} {r.err_median:>10.4f} {r.err_p99:>8.3f} {r.err_max:>8.2f} | "
              f"{f'{r.flagged} ({r.flagged / r.scenarios:.1%})':>13} {f'{r.secure_flagged}/{r.secure} ({r.secure_flagged / max(r.secure, 1):.1%})':>20} | "
              f"{f'{r.missed}/{r.overloaded}':>14} {r.missed_share:>7.2%} {up:>8} {worst:>7}")


def parse(spec):
    label, path = spec.split("=", 1)
    grid = None
    m = re.match(r"^(\d+):(.*)$", path)
    if m:
        grid, path = int(m.group(1)), m.group(2)
    path = Path(path)
    if grid is None:
        m = re.search(r"lines_hybrid_(\d+)_", path.name) or re.search(r"goc(\d+)", str(path))
        assert m, f"cannot read the grid from {path}; use label=GRID:path"
        grid = int(m.group(1))
    return label, grid, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="label=lines_hybrid parquet (or label=GRID:path)")
    p.add_argument("--out", default=str(ROOT / "work" / "finetune" / "results"))
    a = p.parse_args()
    by_grid = {}
    for spec in a.models:
        label, grid, path = parse(spec)
        if not path.exists():
            print(f"WARNING: {path} missing - {label} skipped")
            continue
        by_grid.setdefault(grid, []).append((label, path))
    for grid, models in by_grid.items():
        rows, info, ref_states = [], [], None
        for i, (label, path) in enumerate(models):
            lines = pd.read_parquet(path)
            st = pd.read_parquet(path.with_name(path.name.replace("lines_", "states_")))
            states = set(lines.state.unique())
            if ref_states is None:
                ref_states = states
            elif states != ref_states:
                print(f"WARNING: {label} covers {len(states):,} scenarios, the first model {len(ref_states):,} - rows are not comparable")
            for subset, sel, ssel in (("all", slice(None), slice(None)), ("n_out=1", lines.n_out == 1, st.n_out == 1)):
                sub = lines[sel]
                if not len(sub):
                    continue
                if i == 0:  # pre-outage baseline: model independent, taken from the first file
                    rows += [{"grid": grid, "subset": subset, "model": "pre-outage (no model)", "predictor": c, **rule(sub, c)} for c in BASELINE]
                rows += [{"grid": grid, "subset": subset, "model": label, "predictor": c, **rule(sub, c)} for c in MODEL_PREDICTORS]
                info.append({"grid": grid, "subset": subset, "model": label, **change_info(sub), **state_info(st[ssel])})
        df, inf = pd.DataFrame(rows), pd.DataFrame(info)
        for subset, g in df.groupby("subset", sort=False):
            base = g[g.model == "pre-outage (no model)"]
            g = pd.concat([base, g[g.model != "pre-outage (no model)"]])
            show(g, f"GOC {grid}, {subset}: {g.scenarios.iloc[0]:,} scenarios, {g.overloaded.iloc[0]:,} overloaded lines in {g.critical.iloc[0]:,} scenarios. "
                    f"Loading error in points of rating; flagged = any predicted loading > {OVERLOAD}; missed = overloaded lines in unflagged scenarios; "
                    f"'<= (95%)' = one-sided Clopper-Pearson upper bound of the miss share; worst = highest AC loading among missed")
            print(f"  {'model':<22} {'start mismatch MW median/p99: pre-outage':>42} {'GENCO':>14} {'hybrid':>14} | {'after 1 step: pre-outage':>26} {'GENCO':>14} {'hybrid':>14} | "
                  f"{'Newton it: prev/GENCO/hyb':>26} | predicted vs true loading change (lines moving > 1 point): n, sign agreement, correlation, slope")
            for r in inf[inf.subset == subset].itertuples():
                pair = lambda k: f"{getattr(r, k):.3g} / {getattr(r, k + '_p99'):.3g}"  # noqa: E731
                print(f"  {r.model:<22} {pair('mis0_prev'):>42} {pair('mis0_genco'):>14} {pair('mis0_hyb'):>14} | {pair('mis1_prev'):>26} {pair('mis1_genco'):>14} {pair('mis1_hyb'):>14} | "
                      f"{f'{r.it_prev:.2f} / {r.it_genco:.2f} / {r.it_hyb:.2f}':>26} | {r.chg_n:,}, {r.chg_sign:.1%}, {r.chg_corr:.3f}, {r.chg_slope:.3f}")
            nc = inf[(inf.subset == subset) & ((inf.notconv_hyb > 0) | (inf.notconv_genco > 0))]
            if len(nc):
                print("  not converged (hybrid / GENCO start): " + ", ".join(f"{r.model} {r.notconv_hyb} / {r.notconv_genco}" for r in nc.itertuples()))
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        df.merge(inf, on=["grid", "subset", "model"], how="left").to_csv(out / f"summary_hybrid_{grid}.csv", index=False)
        print(f"\nsaved {out / f'summary_hybrid_{grid}.csv'}")


if __name__ == "__main__":
    main()

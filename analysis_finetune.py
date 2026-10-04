"""Before / after fine-tuning: released GENCO Tiny vs fine-tuned checkpoints on the EXISTING test scenarios.

Plain rule (no calibration): a scenario is flagged if any predicted loading > 1.001 (analysis_conformal.OVERLOAD);
an overloaded line (AC loading > 1.001) is "missed" if its scenario is not flagged. Reported for raw GENCO and
after one / two Newton steps, on all scenarios and on the plain N-1 subset (n_out == 1).

    .venv/bin/python analysis_finetune.py --grid 500 released=work/goc500/lines_newton_tiny_seed0.parquet \
        ft_seed0=work/finetune/results/lines_newton_500_ft_seed0.parquet ...

Each argument is label=path of a lines_newton parquet; the states parquet is found by replacing "lines_" with "states_".
Writes work/finetune/results/summary_<grid>.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OVERLOAD = 1.001
BIG = 9.99
ROOT = Path(__file__).resolve().parent


def rule(lines, col):
    per = lines.groupby("state")[["ac", col]].max()
    flagged = per[col] > OVERLOAD
    critical = per.ac > OVERLOAD
    over = lines[lines.ac > OVERLOAD]
    missed = over[~over.state.map(flagged).to_numpy()]
    return {
        "flagged": int(flagged.sum()), "secure": int((~critical).sum()), "secure_flagged": int((flagged & ~critical).sum()),
        "critical": int(critical.sum()), "critical_missed": int((critical & ~flagged).sum()),
        "overloaded": len(over), "missed": len(missed), "worst": float(missed.ac.max()) if len(missed) else np.nan,
    }


def summarise(label, lines, st, subset):
    row = {"model": label, "subset": subset, "scenarios": lines.state.nunique()}
    e = (lines.genco - lines.ac).abs().to_numpy() * 100
    row.update(err_median=np.median(e), err_p99=np.quantile(e, .99), err_p999=np.quantile(e, .999), err_max=e.max())
    for k, col in (("raw", "genco"), ("n1", "genco1"), ("n2", "genco2")):
        row.update({f"{k}_{a}": b for a, b in rule(lines, col).items()})
        e = (lines[col].where(lines[col] < BIG) - lines.ac).abs().to_numpy() * 100
        row[f"{k}_err_p99"], row[f"{k}_err_max"] = np.nanquantile(e, .99), np.nanmax(e)
    c = st[st.conv_genco]
    row.update(it_genco=c.it_genco.mean(), it_flat=st.it_flat[st.conv_flat].mean(), not_converged=int((~st.conv_genco).sum()),
               mis0=st.mis0_genco.median(), mis1=st.mis1_genco.median(), mis1_p99=st.mis1_genco.quantile(.99), mis2_p99=st.mis2_genco.quantile(.99))
    return row


def show(df, title):
    print(f"\n{title}")
    print(f"  {'model':<22} {'raw err med/p99/max (pt)':>26} | {'RAW: flagged':>12} {'secure fl.':>12} {'lines missed':>13} {'worst':>7} | "
          f"{'+1: flagged':>11} {'secure fl.':>12} {'missed':>9} {'worst':>7} | {'+2: secure fl.':>14} {'missed':>7} | {'Newton it':>9} {'mis0 MW':>8} {'mis1 MW':>8}")
    for r in df.itertuples():
        w = lambda x: "" if np.isnan(x) else f"{x:.1%}"
        print(f"  {r.model:<22} {f'{r.err_median:.3f} / {r.err_p99:.2f} / {r.err_max:.0f}':>26} | {r.raw_flagged:>12} {f'{r.raw_secure_flagged}/{r.raw_secure}':>12} "
              f"{f'{r.raw_missed}/{r.raw_overloaded}':>13} {w(r.raw_worst):>7} | {r.n1_flagged:>11} {f'{r.n1_secure_flagged}/{r.n1_secure}':>12} {f'{r.n1_missed}/{r.n1_overloaded}':>9} {w(r.n1_worst):>7} | "
              f"{f'{r.n2_secure_flagged}/{r.n2_secure}':>14} {r.n2_missed:>7} | {r.it_genco:>9.3f} {r.mis0:>8.1f} {r.mis1:>8.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=500)
    p.add_argument("models", nargs="+", help="label=lines_newton parquet")
    a = p.parse_args()
    rows = []
    for spec in a.models:
        label, path = spec.split("=", 1)
        path = Path(path)
        lines = pd.read_parquet(path, columns=["state", "n_out", "line", "ac", "genco", "genco1", "genco2"])
        st = pd.read_parquet(path.with_name(path.name.replace("lines_", "states_")))
        broken = int(lines[["genco1", "genco2"]].isna().any(axis=1).groupby(lines.state).any().sum())
        lines[["genco1", "genco2"]] = lines[["genco1", "genco2"]].fillna(BIG)
        rows.append({**summarise(label, lines, st, "all"), "broken_step": broken})
        one = lines.n_out == 1
        rows.append({**summarise(label, lines[one], st[st.n_out == 1], "n_out=1"), "broken_step": broken})
    df = pd.DataFrame(rows)
    out = ROOT / "work" / "finetune" / "results" / f"summary_{a.grid}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    for subset, g in df.groupby("subset", sort=False):
        show(g, f"GOC {a.grid}, {subset}: {g.scenarios.iloc[0]:,} scenarios; flagged = any predicted loading > {OVERLOAD}; 'secure fl.' = secure scenarios flagged; "
                f"missed = overloaded lines in unflagged scenarios; worst = highest AC loading among missed")
    print(f"\nscenarios with a broken Newton step: {dict(zip(df.model[df.subset == 'all'], df.broken_step[df.subset == 'all']))}; "
          f"not converged from GENCO: {dict(zip(df.model[df.subset == 'all'], df.not_converged[df.subset == 'all']))}; saved {out}")


if __name__ == "__main__":
    main()

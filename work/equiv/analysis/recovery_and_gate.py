"""REPORT.md sections 4 and 6 helpers (read-only, fresh parquet files only):
  1. Newton recovery after a large first-step loading error under GOC 500 `split_all` (anchored start, transformed encoding):
     of the states whose `hyb1_t` maximum loading error exceeds 100 points, how many later converge to the stored AC solution
     (complex-voltage distance `dv_hyb_t` <= 1e-5 p.u.), converge to a different PF solution, or exhaust the 30-iteration budget.
  2. The worst GOC 500 `split_gen` state (first-step |t-o| about 4,505 points) and its Newton outcome.
  3. Exploratory 5 MW mismatch-gate rescore (NOT the registered criterion): a state is referred when its loading-only flag fires
     OR its post-step mismatch `mis1_hyb_r` is > 5 MW or non-finite; the 5 MW threshold is the old GOC 2000 N-1 threshold transferred
     unchanged. Reproduces the independent batch-4 review's table (section 3).
Columns used: lines parquet `ac`, `hyb1_o`, `hyb1_t`; states parquet `conv_hyb_t`, `it_hyb_t`, `dv_hyb_t`, `mis0_hyb_t`, `mis1_hyb_o`, `mis1_hyb_t`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
LIM, BIG, DV_SAME, TAU_MW = 1.001, 1e9, 1e-5, 5.0


def load(key, transform):
    L = pd.read_parquet(RES / f"{key}_{transform}_fresh_lines.parquet")
    S = pd.read_parquet(RES / f"{key}_{transform}_fresh_states.parquet").set_index("state")
    return L, S


def recovery(key):
    L, S = load(key, "split_all")
    err = ((L.hyb1_t - L.ac).abs() * 100).groupby(L.state).max()
    flag = (L.hyb1_t.fillna(BIG) > LIM).groupby(L.state).any()
    crit = (L.ac > LIM).groupby(L.state).any()
    missed = crit & ~flag
    conv = S.conv_hyb_t
    elsewhere = conv & (S.dv_hyb_t > DV_SAME)
    large = err > 100
    return {"states": len(S), "converged": int(conv.sum()), "budget_exhausted_all": int((~conv).sum()), "it_cap": int(S.it_hyb_t.max()),
            "converged_elsewhere_all": int(elsewhere.sum()), "max_dv_converged_pu": float(S.dv_hyb_t[conv].max()),
            "err>100": int(large.sum()), "err>100_to_stored": int((large & conv & ~elsewhere).sum()), "err>100_elsewhere": int((large & elsewhere).sum()),
            "err>100_budget": int((large & ~conv).sum()), "missed_states": int(missed.sum()), "missed_to_stored": int((missed & conv & ~elsewhere).sum()),
            "missed_max_dv": float(S.dv_hyb_t[missed].max())}


def gate(key, tau=TAU_MW):
    """(missed overloaded readings, secure states referred) per representation, loading flag OR mismatch > tau / non-finite."""
    L, S = load(key, "split_all")
    crit = (L.ac > LIM).groupby(L.state).any()
    real = L.ac > LIM
    out = {}
    for r in ("o", "t"):
        flag = (L[f"hyb1_{r}"].fillna(BIG) > LIM).groupby(L.state).any()
        m1 = S[f"mis1_hyb_{r}"].reindex(flag.index)
        screen = flag | ~(m1 <= tau)
        out[r] = (int((real & ~L.state.map(screen)).sum()), int((~crit & screen).sum()))
    return out


if __name__ == "__main__":
    keys = sys.argv[1:] or ["goc500_tiny_seed0", "goc500_tiny_seed1", "goc500_tiny_seed42"]
    print("## GOC 500 split_all: Newton recovery after a first-step loading error > 100 points (anchored start, transformed encoding)")
    print("| checkpoint | states | err > 100 | -> stored solution | -> different solution | 30-it budget exhausted | all converged elsewhere (max dv p.u.) | not converged, all | missed states -> stored (max dv) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        r = recovery(k)
        print(f"| {k} | {r['states']} | {r['err>100']} | {r['err>100_to_stored']} | {r['err>100_elsewhere']} | {r['err>100_budget']} | "
              f"{r['converged_elsewhere_all']} ({r['max_dv_converged_pu']:.3g}) | {r['budget_exhausted_all']} | {r['missed_to_stored']}/{r['missed_states']} ({r['missed_max_dv']:.2g}) |")
    if (RES / "goc500_tiny_seed0_split_gen_fresh_states.parquet").exists():
        L, S = load("goc500_tiny_seed0", "split_gen")
        d = ((L.hyb1_t - L.hyb1_o).abs() * 100).groupby(L.state).max()
        w = int(d.idxmax())
        e_t = ((L.hyb1_t - L.ac).abs() * 100).groupby(L.state).max()
        print(f"\n## GOC 500 split_gen: state max |hyb1_t - hyb1_o| median {d.median():.4f} / p90 {d.quantile(.9):.3f} / max {d.max():,.0f} points; "
              f"states with first-step error > 100 points: t {int((e_t > 100).sum())}, o {int((((L.hyb1_o - L.ac).abs() * 100).groupby(L.state).max() > 100).sum())}")
        print(f"   worst state {w} (load {int(S.loc[w, 'load_idx'])}): converged {bool(S.loc[w, 'conv_hyb_t'])} in {int(S.loc[w, 'it_hyb_t'])} iterations, "
              f"dv_hyb_t {S.loc[w, 'dv_hyb_t']:.3g} p.u., mismatch start {S.loc[w, 'mis0_hyb_t']:,.0f} MW, after one step {S.loc[w, 'mis1_hyb_t']:,.0f} MW")
    print(f"\n## Exploratory {TAU_MW:g} MW mismatch-gate rescore of split_all (NOT the registered criterion; old GOC 2000 N-1 threshold transferred unchanged)")
    print("| checkpoint | original: missed readings / secure states referred | transformed: missed / secure referred |")
    print("|---|---|---|")
    for k in (sys.argv[1:] or ["goc500_tiny_seed0", "goc500_tiny_seed1", "goc500_tiny_seed42", "goc2000_tiny_seed0", "goc2000_small_seed0"]):
        g = gate(k)
        print(f"| {k} | {g['o'][0]} / {g['o'][1]:,} | {g['t'][0]} / {g['t'][1]:,} |")

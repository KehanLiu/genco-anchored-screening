"""GOC 500 split_all failure analysis for REPORT.md: Newton behaviour of the anchored start under the transformed encoding,
where the missed overloads sit, and whether the raw+one-step (g1) fails likewise. Reads the fresh parquet files only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
BIG, LIM = 1e9, 1.001

def analyse(key):
    L = pd.read_parquet(RES / f"{key}_split_all_fresh_lines.parquet")
    S = pd.read_parquet(RES / f"{key}_split_all_fresh_states.parquet").set_index("state")
    n = len(S)
    print(f"\n## {key} split_all: {n} states, {len(L):,} rated readings, {(L.ac > LIM).sum()} overloaded, {int((L.ac > LIM).groupby(L.state).any().sum())} critical states")
    # Newton behaviour per start, o vs t
    print("| start | converged o | converged t | iterations to convergence o mean/p90/max | t mean/p90/max | t: states >5 it | t: not converged (30 it cap) |")
    print("|---|---|---|---|---|---|---|")
    for name, k in (("hyb (anchored)", "hyb"), ("g (raw GENCO)", "g")):
        co, ct = S[f"conv_{k}_o"], S[f"conv_{k}_t"]
        io, it = S[f"it_{k}_o"], S[f"it_{k}_t"]
        fmt = lambda v, c: f"{v[c].mean():.2f} / {v[c].quantile(.9):.0f} / {v[c].max()}" if c.any() else "n/a"
        print(f"| {name} | {int(co.sum())}/{n} | {int(ct.sum())}/{n} | {fmt(io, co)} | {fmt(it, ct)} | {int((it > 5).sum())} | {int((~ct).sum())} |")
    print(f"| prev (AC pre-outage) | {int(S.conv_prev.sum())}/{n} | - | {S.it_prev[S.conv_prev].mean():.2f} / {S.it_prev.quantile(.9):.0f} / {S.it_prev.max()} | - | - | - |")
    qq = lambda v: f"{np.nanmedian(v):.3g} / {np.nanquantile(v, .9):.3g} / {np.nanmax(v):.3g}"
    print("\nmismatch MW (max-norm), median / p90 / max over states:")
    for lab, c in (("raw GENCO voltages", "mis_raw"), ("anchored start hyb0", "mis0_hyb"), ("after one step hyb1", "mis1_hyb"), ("raw start g0", "mis0_g"), ("after one step g1", "mis1_g")):
        print(f"- {lab}: o {qq(S[c+'_o'].replace(np.inf, np.nan))}; t {qq(S[c+'_t'].replace(np.inf, np.nan))}")
    print(f"- pre-outage AC start prev0: {qq(S.mis0_prev)}; after one step prev1: {qq(S.mis1_prev.replace(np.inf, np.nan))}")
    print(f"- start distance |V_start - V_AC| max over buses (p.u., complex): hyb o {qq(S.dv_start_hyb_o)}; hyb t {qq(S.dv_start_hyb_t)}")
    print(f"- raw drift t-vs-o: Vm {qq(S.drift_vm_c)} p.u., Va {qq(S.drift_va_c)} deg; drift of the predicted change: Vm {qq(S.drift_dvm)}, Va {qq(S.drift_dva)} deg")
    # per-state loading error after one step
    real = L.ac > LIM
    for s in ("hyb1", "g1"):
        for r in ("o", "t"):
            col = f"{s}_{r}"
            e = (L[col] - L.ac).abs() * 100
            emax = e.groupby(L.state).max()
            flag = (L[col].fillna(BIG) > LIM).groupby(L.state).any()
            missed = real & ~L.state.map(flag)
            ms = L.state[missed]
            nan = int(L[col].isna().sum())
            print(f"\n### {col}: NaN readings {nan}; state max loading error pts: {qq(emax)}; states with max error > 100 pts: {int((emax > 100).sum())}, > 10 pts: {int((emax > 10).sum())}")
            print(f"  missed overloaded readings {int(missed.sum())} in {ms.nunique()} states; critical states not flagged {int((~flag & (real.groupby(L.state).any())).sum())}; secure states flagged {float((flag & ~real.groupby(L.state).any()).sum())/max((~real.groupby(L.state).any()).sum(),1):.1%}")
            if missed.sum():
                st = ms.unique()
                em = emax.loc[st]
                conv = S.loc[st, f"conv_{'hyb' if s == 'hyb1' else 'g'}_{r}"]
                its = S.loc[st, f"it_{'hyb' if s == 'hyb1' else 'g'}_{r}"]
                m1 = S.loc[st, f"mis1_{'hyb' if s == 'hyb1' else 'g'}_{r}"]
                print(f"  in the missed states: max loading error pts {qq(em)}; > 100 pts in {int((em > 100).sum())}/{len(st)}; > 10 pts in {int((em > 10).sum())}/{len(st)}; "
                      f"Newton converged {int(conv.sum())}/{len(st)}, iterations {qq(its)}, mismatch after one step MW {qq(m1.replace(np.inf, np.nan))}")
                # how far below the limit do the missed readings sit under t, and the same readings under o
                mm = L[missed]
                print(f"  missed readings: AC loading {qq(mm.ac*100)} %; predicted {col} {qq(mm[col]*100)} %; same readings under {s}_o {qq(mm[s+'_o']*100)} %")
                print(f"  load scenarios of the missed states: {sorted(S.loc[st, 'load_idx'].unique().tolist())[:30]}{' ...' if S.loc[st, 'load_idx'].nunique() > 30 else ''} ({S.loc[st, 'load_idx'].nunique()} scenarios)")
    # does the hyb1 blow-up coincide with hyb1_t misses? compare states by error class
    e_t = ((L.hyb1_t - L.ac).abs() * 100).groupby(L.state).max()
    e_o = ((L.hyb1_o - L.ac).abs() * 100).groupby(L.state).max()
    flag_t = (L.hyb1_t.fillna(BIG) > LIM).groupby(L.state).any(); flag_o = (L.hyb1_o.fillna(BIG) > LIM).groupby(L.state).any()
    crit = real.groupby(L.state).any()
    dis = flag_t != flag_o
    print(f"\n### decision disagreement hyb1 t vs o: {int(dis.sum())} states ({dis.mean():.1%}); of these flagged by t only {int((dis & flag_t).sum())}, by o only {int((dis & flag_o).sum())}")
    print(f"  hyb1_t state max error in disagreeing states: {qq(e_t[dis])}; in agreeing states {qq(e_t[~dis])}")
    print(f"  states with hyb1_t max error > 100 pts: {int((e_t > 100).sum())}, of which decisions disagree {int(((e_t > 100) & dis).sum())}; share of t-only flags among secure states {float((flag_t & ~crit).mean()):.1%} (o: {float((flag_o & ~crit).mean()):.1%})")
    print(f"  Newton from the anchored start t in states with error > 100 pts: converged {int(S.conv_hyb_t[e_t[e_t > 100].index].sum())}/{int((e_t > 100).sum())}, it > 5 in {int((S.it_hyb_t[e_t[e_t > 100].index] > 5).sum())}")

if __name__ == "__main__":
    for key in sys.argv[1:] or ["goc500_tiny_seed0", "goc500_tiny_seed1", "goc500_tiny_seed42", "goc2000_tiny_seed0", "goc2000_small_seed0"]:
        analyse(key)

"""Independent re-derivation of the numbers quoted in submission/POST.md that came from subagent reports.

Reads only saved result tables; shares no code with the scripts that produced the reports.

    .venv/bin/python work/verify_round1.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).parent
OV = 1.001


def fbeta(tp, fp, fn, beta=3):
    return (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp) if tp else 0.0


print("== voltage, IEEE 118 (PQ buses, case limits, tolerance 1e-4) ==")
v = pd.read_parquet(W / "voltage_case118.parquet")
v = v[v.type == "PQ"]
tol = 1e-4
real = (v.vm_ac < v.vmin - tol) | (v.vm_ac > v.vmax + tol)
call = (v.vm_genco < v.vmin - tol) | (v.vm_genco > v.vmax + tol)
depth = np.maximum(v.vmin - v.vm_ac, v.vm_ac - v.vmax)
st = v.assign(real=real).groupby("state").agg(real=("real", "any"), over_ac=("over_ac", "first"), over_dc=("over_dc", "first"))
print(f"states {len(st)}, violating states {st.real.sum()} ({st.real.mean():.1%}); bus readings violating {real.sum()}")
print(f"GENCO: missed {(real & ~call).sum()}, false {(~real & call).sum()}, recall {(real & call).sum() / real.sum():.1%}")
print(f"violations deeper than 0.01 p.u.: {(real & (depth > 0.01)).sum()}, missed by GENCO: {(real & ~call & (depth > 0.01)).sum()}")
print(f"violating states with no DC overload: {(st.real & (st.over_dc == 0)).sum()}; with no AC overload: {(st.real & (st.over_ac == 0)).sum()}")
print(f"GENCO Vm error p99 {(v.vm_genco - v.vm_ac).abs().quantile(0.99):.5f}")

print("\n== Newton start points (normal set) ==")
for name, f in (("IEEE 118", "warmstart_case118.parquet"), ("Texas 2000", "texas/out/warmstart_texas.parquet")):
    s = pd.read_parquet(W / f)
    n = s[s.set == "normal"]
    print(f"{name}: {len(n)} states; mean iterations " + ", ".join(f"{k} {n[f'it_{k}'].mean():.2f}" for k in ("flat", "dc", "prev", "genco"))
          + "; all converged: " + str(bool(n[[f"conv_{k}" for k in ("flat", "dc", "prev", "genco")]].all().all())))
    stress = s[s.set != "normal"]
    conv = stress[[f"conv_{k}" for k in ("flat", "dc", "prev", "genco")]]
    print(f"   stress states {len(stress)}: converged per start {conv.sum().to_dict()}; GENCO converged where flat or dc failed: "
          f"{int((conv.conv_genco & ~(conv.conv_flat & conv.conv_dc)).sum())}; GENCO failed where flat converged: {int((~conv.conv_genco & conv.conv_flat).sum())}")

print("\n== one Newton step on IEEE 118 / Texas (normal set) ==")
for name, stem in (("IEEE 118", "warmstart_case118"), ("Texas 2000", "texas/out/warmstart_texas")):
    s = pd.read_parquet(W / f"{stem}.parquet")
    lines = pd.read_parquet(W / f"{stem}_lines.parquet")
    lines = lines[lines.state.isin(s.state[s.set == "normal"])].dropna(subset=["ac"])
    real = lines.ac > OV
    for c in ("genco", "genco1", "dc1", "prev1"):
        e = (lines[c] - lines.ac).abs() * 100
        print(f"{name} {c:7} p99 {e.quantile(0.99):.2f} max {e.max():.1f} missed {(real & (lines[c] <= OV)).sum()}/{real.sum()} false {(~real & (lines[c] > OV)).sum()}")

print("\n== stress test, IEEE 118, alpha 5% ==")
f = pd.read_csv(W / "shift_case118_fixed.csv")
f["shift"] = f["shift"].str.replace(r"^redispatch.*", "redispatch", regex=True)
a = f[f.alpha == 0.05]
print(a[a["shift"].isin(["control", "n3", "n4", "load", "pf Qd x0.7", "pf Qd x1.2", "pf Qd x1.5", "redispatch"])]
      .pivot_table(index="shift", columns=["surrogate", "sigma"], values="line_missed_share").mul(100).round(1).to_string())
print("worst missed loading, genco_d rows:", f[f.surrogate == "genco_d"].worst_missed.max())
# re-derive one cell from the raw sweep: N-3, genco_d + shift, width from the csv
n3 = pd.read_parquet(W / "shift_case118_n3.parquet")
print("n3 sweep columns:", list(n3.columns))

print("\n== GOC 500: example rule and conformal on raw GENCO (seed 0, same 50/50 split as the report) ==")
d = pd.read_parquet(W / "goc500/lines_tiny_seed0.parquet")
ids = np.array(sorted(d.state.unique()))
print("scenarios", len(ids), "columns", list(d.columns))
rng = np.random.default_rng(0)
perm = rng.permutation(ids)
cal_ids, test_ids = set(perm[: len(ids) // 2]), set(perm[len(ids) // 2:])
real = d.ac > OV
crit = d.state[real].unique()
mx = d.groupby("state")[["genco", "dc"]].max()
truth = pd.Series(mx.index.isin(crit), index=mx.index)
cal = mx.index.isin(list(cal_ids))
for col in ("genco", "dc"):
    best = max(np.arange(0.70, 1.05, 0.005), key=lambda t: fbeta(((mx[col] > t) & truth & cal).sum(), ((mx[col] > t) & ~truth & cal).sum(), ((mx[col] <= t) & truth & cal).sum()))
    flag = (mx[col] > best) & ~cal
    tt = truth & ~cal
    sent = set(mx.index[flag])
    test_lines = d[d.state.isin(list(test_ids))]
    r = test_lines.ac > OV
    missed = r & ~test_lines.state.isin(list(sent))
    print(f"{col}: F3 threshold from calibration half {best:.3f}; test: recall {(flag & tt).sum() / tt.sum():.1%}, flagged {flag.sum() / (~cal).sum():.1%}, "
          f"secure flagged {(flag & ~truth).sum() / (~truth & ~cal).sum():.1%}, missed lines {missed.sum()}/{r.sum()} worst {test_lines.ac[missed].max():.3f}")
# conformal alpha = 1% on raw GENCO, global band
cl = d[d.state.isin(list(cal_ids)) & real]
scores = np.sort((OV - cl.genco).to_numpy())
k = int(np.ceil((len(scores) + 1) * 0.99))
width = scores[k - 1]
tl = d[d.state.isin(list(test_ids))]
sent = set(tl.state[tl.genco >= OV - width])
tr = tl.ac > OV
print(f"conformal 1%: width {width * 100:.2f} pt, sent to AC {len(sent) / len(test_ids):.1%}, missed {(tr & ~tl.state.isin(list(sent))).sum()}/{tr.sum()}")
e = (d.genco - d.ac).abs() * 100
print(f"raw GENCO loading error p99 {e.quantile(0.99):.2f}")

print("\n== stress test: N-3 cell re-derived from the raw sweeps (genco_d + shift, alpha 5%) ==")
def prep(path):
    x = pd.read_parquet(path)
    key = [c for c in ("anchor",) if c in x.columns]
    if key:
        base = x[x.outages == ""].set_index(["anchor", "line"])[["ac", "genco"]].add_suffix("0")
        x = x[x.outages != ""].join(base, on=["anchor", "line"])
    else:
        base = x[x.outages == ""].set_index(["load_scale", "line"])[["ac", "genco"]].add_suffix("0")
        x = x[x.outages != ""].join(base, on=["load_scale", "line"])
    x = x.dropna(subset=["ac0"])
    x["pred"] = x.ac0 + x.genco - x.genco0
    x["sigma"] = (x.genco - x.genco0).abs() + 0.005
    return x
calib = prep(W / "hybrid_case118_ieee_base.parquet")
cr = calib[calib.ac > OV]
sc = np.sort(((OV - cr.pred) / cr.sigma).to_numpy())
b = sc[int(np.ceil((len(sc) + 1) * 0.95)) - 1]
for name in ("control", "n3", "n4"):
    t = prep(W / f"shift_case118_{name}.parquet")
    r = t.ac > OV
    flagged = t.pred >= OV - b * t.sigma
    sent = set(t.state[flagged])
    state_missed = r & ~t.state.isin(list(sent))
    print(f"{name}: width {b:.3f} sigma, real overloads {r.sum()}, not flagged {(r & ~flagged).sum() / r.sum():.1%}, state-level missed {state_missed.sum()} worst {t.ac[state_missed].max():.3f}")

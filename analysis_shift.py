"""Does the conformal trust layer survive distribution shift?

analysis_conformal.py guarantees "at most alpha of the real overloads are not flagged" only if the
calibration states and the states it is used on are exchangeable. Here the band widths are calibrated
ONCE on the in-distribution sweep (N-1 + random N-2 at five load levels, OPF dispatch) and then applied,
unchanged, to states from shifted distributions:

    control      fresh random N-2 outage sets, same load levels          (no shift: sanity check)
    n3, n4       triple / quadruple outages
    load         load levels the calibration never saw
    pf           reactive demand scaled alone (power-factor shift), dispatch re-optimised by AC-OPF
    redispatch   generation moved by hand away from the OPF dispatch, then N-1 / N-2

Every shifted dispatch gets its own N-0 state: it is the anchor of the genco_d / dc_d surrogates and of
the "shift" sigma, exactly as in analysis_conformal.py.

One repair is tested: the operator runs AC on a small random sample of the shifted states and
recalibrates on it (alone, or pooled with the original calibration set).

    work/texas/tmo 3000 .venv/bin/python -u analysis_shift.py                 # solve all shifts, then report
    .venv/bin/python analysis_shift.py --reuse                                # report from the saved sweeps
    work/texas/tmo 3000 .venv/bin/python -u analysis_shift.py --grid texas    # N-3 and unseen load levels on Texas 2000
"""
import argparse
import importlib.util
import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
ALPHAS = [0.10, 0.05, 0.02, 0.01]
COMBOS = [("genco_d", "global"), ("genco_d", "shift"), ("genco", "shift"), ("dc_d", "shift")]
CAL_SCALES = {"case118": [0.7, 0.85, 1.0, 1.1, 1.2], "texas": [0.8, 0.85, 0.9, 0.95, 1.0]}
NEW_SCALES = {"case118": [0.6, 0.775, 0.925, 1.05, 1.15], "texas": [0.825, 0.875, 0.925, 0.975]}
Q_SCALES = [1.2, 1.5, 0.7]
REDISPATCH_MW = (250, 700)  # target; generator limits cut what is actually moved (reported)
CAL_FILE = {"case118": WORK / "hybrid_case118_ieee_base.parquet", "texas": WORK / "texas/out/hybrid_texas_n2x450.parquet"}
RECAL_SIZES = [50, 100]
RECAL_DRAWS = 20
BOOT = 2000


TAG = ""  # set by --tag (smoke tests)


def out_path(grid, name):
    return WORK / f"shift_{grid}_{name}{TAG}.parquet"


# ---------------------------------------------------------------- solving the shifted states

def make_engine(grid):
    """An Engine whose reactive demand can be scaled on its own; engine.py is left untouched."""
    import engine  # loads Julia; must come before torch

    class ShiftEngine(engine.Engine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._qd0 = self.base_qd.copy()
            self._caches = {1.0: self.dispatched}

        def set_q(self, q_scale):
            """Scale Qd only (P unchanged). Each q_scale keeps its own cache of OPF dispatches."""
            self.base_qd = self._qd0 * q_scale
            self.dispatched = self._caches.setdefault(q_scale, {})

    if grid == "case118":
        return engine, ShiftEngine("case118_ieee", "base")
    here = WORK / "texas"
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location("texas_hybrid", here / "04_hybrid.py")
    tex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tex)
    engine.Engine = ShiftEngine          # build() instantiates engine.Engine
    return engine, tex.build(engine.DISPATCH_MARGIN)


def draw_sets(ids, k, n, rng, exclude=()):
    seen, out = set(exclude), []
    while len(out) < n:
        s = tuple(sorted(rng.sample(ids, k)))
        if "+".join(map(str, s)) not in seen:
            seen.add("+".join(map(str, s)))
            out.append(s)
    return out


def outage_pool(rated, n_sets, seed, known):
    """Fresh N-2 sets (not in the calibration file) shared by all shifts, so that shifts are compared on the
    same outages; each set also gets two further lines that turn it into an N-3 and an N-4 set."""
    rng = random.Random(f"{seed}-pool")
    doubles = draw_sets(rated, 2, n_sets, rng, known)
    extra = [rng.sample([i for i in rated if i not in d], 2) for d in doubles]
    return doubles, extra


def tasks_for(name, grid, eng, engine, rng, known, n_sets, seed):
    """List of (anchor, load_scale, q_scale, outages, redispatch, moved_mw). N-0 anchors are added by the caller."""
    rated = [b["id"] for b in eng.static["branches"] if not b["bridge"] and (grid == "case118" or b["rate_a"] is not None)]
    doubles, extra = outage_pool(rated, n_sets, seed, known)
    scales = CAL_SCALES[grid]
    plain = lambda s, o: (f"L{s}", s, 1.0, o, None, 0.0)
    if name == "control":
        return [plain(s, o) for o in doubles for s in scales]
    if name in ("n3", "n4"):
        k = int(name[1]) - 2
        return [plain(s, tuple(sorted(o + tuple(e[:k])))) for o, e in zip(doubles, extra) for s in scales]
    if name == "load":
        return [plain(s, o) for o in doubles for s in NEW_SCALES[grid]]
    if name == "pf":
        return [(f"L{s}Q{q}", s, q, o, None, 0.0) for q in Q_SCALES for o in doubles for s in scales]
    if name == "redispatch":
        from analysis_redispatch import random_redispatch
        per_dispatch, tasks = 8, []
        n_dispatch = max(n_sets * len(scales) // per_dispatch, 1)
        for i in range(n_dispatch):
            s = scales[i % len(scales)]
            eng.set_q(1.0)
            gens = pd.DataFrame(eng.solve(s)["gens"]).set_index("id", drop=False)
            delta, moved = random_redispatch(gens, rng.uniform(*REDISPATCH_MW), rng)
            tasks += [(f"L{s}R{i}", s, 1.0, o, delta, moved) for o in rng.sample(doubles, min(per_dispatch, len(doubles)))]
        return tasks
    raise ValueError(name)


def sweep(name, grid, eng, engine, known, n_sets, seed=7):
    rng = random.Random(f"{seed}-{name}")
    tasks = tasks_for(name, grid, eng, engine, rng, known, n_sets, seed)
    anchors = {}
    for t in tasks:  # one N-0 state per dispatch, solved first
        anchors.setdefault(t[0], (t[0], t[1], t[2], (), t[4], t[5]))
    frames, fails, dead, state, start = [], {}, set(), 0, time.perf_counter()
    for anchor, scale, q, outages, redispatch, moved in list(anchors.values()) + tasks:
        if anchor in dead:
            continue
        eng.set_q(q)
        try:
            result = eng.solve(scale, outages, redispatch=redispatch)
        except engine.SolveError as err:
            fails[err.reason] = fails.get(err.reason, 0) + 1
            if not outages:
                dead.add(anchor)  # no base case -> nothing to anchor on
                print(f"  {name}: anchor {anchor} failed ({err.reason}); its states are skipped", flush=True)
            continue
        lines = pd.DataFrame(result["branches"])
        lines = lines[lines.in_service & lines.loading_ac.notna()]
        frames.append(pd.DataFrame({
            "state": state, "load_scale": scale, "outages": "+".join(map(str, outages)), "line": lines.id,
            "ac": lines.loading_ac, "genco": lines.loading_genco, "dc": lines.loading_dc,
            "residual": pd.DataFrame(result["buses"]).res_p.abs().max(),
            "anchor": anchor, "q_scale": q, "moved_mw": moved,
        }))
        state += 1
        if state % 100 == 0:
            print(f"  {name}: {state} states solved, failures {fails}, {time.perf_counter() - start:.0f} s", flush=True)
    data = pd.concat(frames, ignore_index=True)
    data.attrs = {}
    print(f"  {name}: {state} states solved ({len(anchors) - len(dead)} anchors), failures {fails}, {time.perf_counter() - start:.0f} s", flush=True)
    return data, fails


# ---------------------------------------------------------------- scoring

def prep(data):
    """analysis_conformal.load, with the base case looked up per dispatch ('anchor') instead of per load level."""
    from analysis_conformal import KEEP, SHIFT_FLOOR, SURROGATES
    if "anchor" not in data:
        data = data.assign(anchor="L" + data.load_scale.astype(str))
    base = data[data.outages == ""].set_index(["anchor", "line"])[["ac", "genco", "dc"]].add_suffix("0")
    data = data[data.outages != ""].join(base, on=["anchor", "line"]).dropna(subset=["ac0"])
    for s in ("genco", "dc"):
        data[f"{s}_d"] = data.ac0 + data[s] - data[f"{s}0"]
        data[f"shift_{s}"] = data[f"shift_{s}_d"] = (data[s] - data[f"{s}0"]).abs() + SHIFT_FLOOR
    errors = {s: (data[s] - data.ac).abs() * 100 for s in SURROGATES}
    n_states = data.state.nunique()
    data = data[data[["ac"] + SURROGATES].max(axis=1) > KEEP].reset_index(drop=True)
    return data, errors, n_states


def widths(cal):
    from analysis_conformal import conformal_width, sigma_of
    return {(s, k, a): conformal_width(cal, s, sigma_of(cal, s, k, None), a)[0] for s, k in COMBOS for a in ALPHAS}


def line_miss_ci(test, surrogate, sigma, width, rng):
    """Per-line miss rate with a 90% interval from resampling whole outage sets: overloads cluster in states, and
    GENCO's hard cases cluster in a few outage sets (all load levels of a set stand or fall together)."""
    from analysis_conformal import OVERLOAD
    real = (test.ac > OVERLOAD).to_numpy()
    miss = real & (test[surrogate].to_numpy() < OVERLOAD - width * sigma)
    per = pd.DataFrame({"real": real, "miss": miss}).groupby(test.outages.to_numpy()).sum()
    per = per[per.real > 0].to_numpy()
    if not len(per):
        return np.nan, np.nan
    idx = rng.integers(0, len(per), (BOOT, len(per)))
    rate = per[idx, 1].sum(axis=1) / per[idx, 0].sum(axis=1)
    return np.quantile(rate, 0.05), np.quantile(rate, 0.95)


def fmt_width(w, kind):
    return "all->AC" if np.isinf(w) else (f"{w * 100:.2f} pt" if kind == "global" else f"{w:.2f} sig")


def score_fixed(label, test, errors, n_states, W, out):
    from analysis_conformal import OVERLOAD, evaluate, sigma_of
    rng = np.random.default_rng(0)
    real = test.ac > OVERLOAD
    out(f"\n--- {label}")
    out(f"states {n_states}, real overloads {int(real.sum())} on {test.line[real].nunique()} lines, "
        f"states with an overload {test.state[real].nunique()} ({test.state[real].nunique() / n_states:.1%})")
    out("abs loading error, points of rating (median / p99 / max): " + "; ".join(
        f"{s} {errors[s].median():.3f} / {errors[s].quantile(0.99):.2f} / {errors[s].max():.1f}" for s in ("genco", "genco_d", "dc_d")))
    out(f"  {'surrogate':<8} {'sigma':<6} {'alpha':>5} {'width':>9} | {'per-line miss':>13} {'90% interval':>14} {'vs alpha':>9} | "
        f"{'missed (state-level)':>20} {'share':>6} {'states':>6} {'worst':>7} | {'AC calls':>8} {'on secure':>9}")
    rows = []
    for s, k in COMBOS:
        sigma = sigma_of(test, s, k, None)
        for a in ALPHAS:
            w = W[(s, k, a)]
            r = evaluate(test, n_states, s, sigma, w)
            lo, hi = line_miss_ci(test, s, sigma, w, rng)
            verdict = "n/a" if not r["real"] else "BROKEN" if lo > a else "above" if r["line_missed_share"] > a else "ok"
            worst = "" if np.isnan(r["worst_missed"]) else f"{r['worst_missed']:.1%}"
            out(f"  {s:<8} {k:<6} {a:>5.0%} {fmt_width(w, k):>9} | {r['line_missed_share']:>13.2%} {f'{lo:.1%}-{hi:.1%}':>14} {verdict:>9} | "
                f"{r['missed']:>14}/{r['real']:<5} {r['missed_share']:>6.2%} {r['states_with_miss']:>6} {worst:>7} | {r['ac_calls']:>8.1%} {r['wasted']:>9.1%}")
            rows.append({"shift": label, "surrogate": s, "sigma": k, "alpha": a, "width": w, "n_states": n_states, "ci_lo": lo, "ci_hi": hi, "verdict": verdict,
                         "n_insecure": test.state[real].nunique(), **r})
    return rows


def score_recal(label, test, all_ids, n_states, cal, W, out):
    """Operator labels m random shifted states with AC, recalibrates, and screens the rest."""
    from analysis_conformal import conformal_width, evaluate, sigma_of
    out(f"\n--- {label}: recalibration on m labelled shifted states, scored on the rest (mean of {RECAL_DRAWS} draws)")
    out(f"  {'surrogate':<8} {'sigma':<6} {'alpha':>5} {'m':>4} | {'fixed widths':^21} | {'shifted sample only':^32} | {'pooled with original':^21}")
    out(f"  {'':<8} {'':<6} {'':>5} {'':>4} | {'line miss':>9} {'AC calls':>9}   | {'line miss':>9} {'AC calls':>9} {'all->AC':>8}    | {'line miss':>9} {'AC calls':>9}")
    rows = []
    for m in RECAL_SIZES:
        if m * 2 > n_states:
            continue
        draws = []
        for d in range(RECAL_DRAWS):
            rng = np.random.default_rng(1000 + d)
            # sample m of ALL shifted states; those without a relevant reading simply add no scores
            picked = rng.choice(all_ids, m, replace=False)
            in_sample = test.state.isin(picked).to_numpy()
            draws.append((test[in_sample], test[~in_sample], n_states - m))
        for s, k in COMBOS:
            for a in ALPHAS:
                acc = {"fixed": [], "only": [], "pooled": []}
                for sample, rest, n_rest in draws:
                    sig_rest = sigma_of(rest, s, k, None)
                    w_only = conformal_width(sample, s, sigma_of(sample, s, k, None), a)[0]
                    both = pd.concat([cal, sample], ignore_index=True)
                    w_pool = conformal_width(both, s, sigma_of(both, s, k, None), a)[0]
                    for key, w in (("fixed", W[(s, k, a)]), ("only", w_only), ("pooled", w_pool)):
                        r = evaluate(rest, n_rest, s, sig_rest, w)
                        acc[key].append((r["line_missed_share"], r["ac_calls"], np.isinf(w), r["missed_share"], r["wasted"]))
                mean = {key: np.mean(v, axis=0) for key, v in acc.items()}
                out(f"  {s:<8} {k:<6} {a:>5.0%} {m:>4} | {mean['fixed'][0]:>9.2%} {mean['fixed'][1]:>9.1%}   | "
                    f"{mean['only'][0]:>9.2%} {mean['only'][1]:>9.1%} {mean['only'][2]:>8.0%}    | {mean['pooled'][0]:>9.2%} {mean['pooled'][1]:>9.1%}")
                rows.append({"shift": label, "surrogate": s, "sigma": k, "alpha": a, "m": m,
                             **{f"{key}_{n}": mean[key][i] for key in acc for i, n in enumerate(["line_miss", "ac_calls", "all_to_ac", "state_miss", "wasted"])}})
    return rows


def report(grid, names, out):
    cal, cal_err, n_cal = prep(pd.read_parquet(CAL_FILE[grid]))
    from analysis_conformal import OVERLOAD
    W = widths(cal)
    out(f"=== {grid}: calibrated once on {CAL_FILE[grid].name}: {n_cal} states with outages, {int((cal.ac > OVERLOAD).sum())} real overloads")
    out("fixed widths: " + "; ".join(f"{s}+{k} " + "/".join(fmt_width(W[(s, k, a)], k) for a in ALPHAS) for s, k in COMBOS) + "   (alpha 10/5/2/1 %)")
    out("verdict: 'BROKEN' = the 90% interval of the per-line miss rate (resampling whole outage sets) lies wholly above alpha; "
        "'above' = point estimate above alpha but within sampling noise; 'ok' = at or below alpha.")
    fixed, recal = [], []
    for name in names:
        path = out_path(grid, name)
        if not path.exists():
            out(f"\n--- {name}: no sweep at {path}")
            continue
        raw = pd.read_parquet(path)
        parts = [(name, raw)]
        if name == "pf":
            parts = [(f"pf Qd x{q}", raw[raw.q_scale == q]) for q in sorted(raw.q_scale.unique())]
        if name == "load":
            parts += [(f"load x{s}", raw[raw.load_scale == s]) for s in sorted(raw.load_scale.unique())]
        for label, part in parts:
            test, errors, n_states = prep(part)
            all_ids = part.state[part.outages != ""].unique()
            if name == "redispatch":
                label += f" (median {part.moved_mw.median():.0f} MW moved, range {part.moved_mw.min():.0f}-{part.moved_mw.max():.0f})"
            fixed += score_fixed(label, test, errors, n_states, W, out)
            if not label.startswith("load x"):
                recal += score_recal(label, test, all_ids, n_states, cal, W, out)
    summary = pd.DataFrame(fixed)
    if len(summary):
        summary = summary[~summary["shift"].str.startswith("load x")]
        out("\n=== summary: per-line miss rate with the fixed widths (guarantee: <= alpha); * = BROKEN, i.e. 90% interval wholly above alpha")
        out(f"  {'shift':<28} {'states':>6} {'real':>5} | " + " | ".join(f"{s + '+' + k:^27}" for s, k in COMBOS))
        out(f"  {'':<28} {'':>6} {'':>5} | " + " | ".join(" ".join(f"{a:>6.0%}" for a in ALPHAS) for _ in COMBOS))
        for label, g in summary.groupby("shift", sort=False):
            cells = []
            for s, k in COMBOS:
                c = g[(g.surrogate == s) & (g.sigma == k)].set_index("alpha")
                cells.append(" ".join(f"{c.line_missed_share[a] * 100:>5.1f}{'*' if c.verdict[a] == 'BROKEN' else ' '}" for a in ALPHAS))
            out(f"  {label[:28]:<28} {g.n_states.iloc[0]:>6} {g.real.iloc[0]:>5} | " + " | ".join(cells))
        out("\n=== summary: AC-call share with the fixed widths (in brackets: share of secure states sent to AC)")
        out(f"  {'shift':<28} {'insecure':>8} | " + " | ".join(f"{s + '+' + k + ' @5% / @1%':^27}" for s, k in COMBOS))
        for label, g in summary.groupby("shift", sort=False):
            cells = []
            for s, k in COMBOS:
                c = g[(g.surrogate == s) & (g.sigma == k)].set_index("alpha")
                cells.append(f"{c.ac_calls[0.05]:>5.1%} ({c.wasted[0.05]:>5.1%}) {c.ac_calls[0.01]:>5.1%} ({c.wasted[0.01]:>5.1%})")
            out(f"  {label[:28]:<28} {g.n_insecure.iloc[0] / g.n_states.iloc[0]:>8.1%} | " + " | ".join(cells))
    pd.DataFrame(fixed).to_csv(WORK / f"shift_{grid}_fixed{TAG}.csv", index=False)
    pd.DataFrame(recal).to_csv(WORK / f"shift_{grid}_recal{TAG}.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="case118", choices=["case118", "texas"])
    parser.add_argument("--shifts", default=None, help="comma-separated; default all for the grid")
    parser.add_argument("--sets", type=int, default=None, help="outage sets per shift (each is solved at every load level)")
    parser.add_argument("--reuse", action="store_true", help="report from saved sweeps; solve only the missing ones")
    parser.add_argument("--report-only", action="store_true", help="never solve")
    parser.add_argument("--report", type=Path, default=None, help="also write the report to this text file")
    parser.add_argument("--tag", default="", help="suffix for the sweep files")
    args = parser.parse_args()
    global TAG
    TAG = args.tag
    default = ["control", "n3", "n4", "load", "pf", "redispatch"] if args.grid == "case118" else ["control", "n3", "load"]
    names = args.shifts.split(",") if args.shifts else default
    n_sets = args.sets or (240 if args.grid == "case118" else 60)

    todo = [] if args.report_only else [n for n in names if not (args.reuse and out_path(args.grid, n).exists())]
    if todo:
        engine, eng = make_engine(args.grid)
        known = set(pd.read_parquet(CAL_FILE[args.grid], columns=["outages"]).outages.unique())
        for name in todo:
            if out_path(args.grid, name).exists() and not args.reuse:
                # never overwrite a sweep silently
                print(f"{out_path(args.grid, name)} exists; pass --reuse to keep it or delete it by hand to re-solve")
                continue
            data, _ = sweep(name, args.grid, eng, engine, known, n_sets)
            data.to_parquet(out_path(args.grid, name), index=False)
            print(f"saved {out_path(args.grid, name)}", flush=True)

    lines = []

    def out(text):
        print(text, flush=True)
        lines.append(text)

    report(args.grid, names, out)
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
        print(f"\nreport written to {args.report}")


if __name__ == "__main__":
    main()

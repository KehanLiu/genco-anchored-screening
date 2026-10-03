"""Remedial-action ranking: can a surrogate rank redispatch actions the way AC power flow does?

For a dozen overloaded N-1 / N-2 states of IEEE 118, ~200 candidate actions "move d MW from
generator a to generator b" are scored by the worst line loading after the action, with

    ac        full AC power flow (truth)
    genco     raw GENCO
    newton    one Newton step on the network equations started from GENCO's Vm / Va
              (analysis_warmstart.Problem, the same calls as stage 2 of Engine._staged)
    dc        DC power flow
    dc_anch   AC loading of the contingency state before the action + DC-predicted change
              (sign(P_ac0) * (P_dc_after - P_dc_before) / rating): the sensitivity practice
    genco_anch  AC loading before the action + GENCO-predicted change of loading

GENCO is off its training distribution here: the dispatch is moved away from the OPF point.

The run is resumable and saves after every chunk of actions; each call works for --budget seconds:

    .venv/bin/python -u analysis_remedial.py run --budget 45     # repeat until "all done"
    .venv/bin/python -u analysis_remedial.py report
"""

# engine first: it loads Julia, which has to happen before torch.
from engine import OVERLOAD, WORK, Engine, SolveError

import argparse
import json
import random
import sys
import time

import numpy as np
import pandas as pd

NETWORK, VARIANT = "case118_ieee", "base"
SEED = 0
LOADS = (1.0, 1.05, 1.1, 1.15, 1.2)
N_STATES = 12
PER_KIND = 6  # N-1 and N-2 states each
PER_LOAD = 3  # at most this many states per load level
WORST_RANGE = (1.02, 1.30)  # worst AC loading of an eligible state
MAX_CANDIDATES = 1500
N_ACTIONS = 200
AMOUNTS = (20, 50, 100)  # MW moved
CHUNK = 20  # actions per saved file

PARTS = WORK / "remedial_case118_parts"
STATES = PARTS / "states.json"
CANDIDATES = PARTS / "candidates.json"
OUT = WORK / "remedial_case118.parquet"
LOG = WORK / "remedial_case118.log"
METHODS = ("genco", "newton", "dc", "dc_anch", "genco_anch")


def log(*a):
    text = " ".join(str(x) for x in a)
    print(text, flush=True)
    with LOG.open("a") as f:
        f.write(text + "\n")


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ----------------------------------------------------------------------------- one state, all methods
def solve_all(eng, load, outages, redispatch):
    """Per-branch loadings of one state by every method, or None if AC fails."""
    from analysis_warmstart import Problem

    try:
        res = eng.solve(load, outages, redispatch or None)
    except SolveError:
        return None
    br = pd.DataFrame(res["branches"])
    bus = pd.DataFrame(res["buses"])
    # the same network state Engine.solve built, for the Newton step
    net, _ = eng._dispatch(load)
    state = net.copy_for_perturbation()
    if redispatch:
        pg = state.Pg_gen[state.idx_gens_in_service].copy()
        for g, delta in redispatch.items():
            pg[int(g)] += delta
        state.Pg_gen = pg
    state.deactivate_branches(sorted(outages))
    prob = Problem(state)
    v0 = prob.start(vm_pq=bus.vm_genco.to_numpy(float), va=np.radians(bus.va_genco.to_numpy(float)))
    step = prob.newton(v0, max_it=1, keep=1)
    ok = (step["iterations"] == 1 or step["converged"]) and bool(np.isfinite(step["V"]).all())
    newton = prob.loading(step["V"]) if ok else np.full(len(br), np.nan)
    out = pd.DataFrame({
        "branch": br.id.astype(int),
        "loading_ac": br.loading_ac.astype(float),
        "loading_genco": br.loading_genco.astype(float),
        "loading_newton": np.where(br.in_service, newton, np.nan),
        "loading_dc": br.loading_dc.astype(float),
        "p_ac": br.p_ac.astype(float),
        "p_dc": br.p_dc.astype(float),
    })
    out["newton_mismatch_mw"] = float(step["mis_end"]) if ok else np.inf
    return out, pd.DataFrame(res["gens"])


# ----------------------------------------------------------------------------- choosing states and actions
def choose_states(eng, deadline):
    """Walk a seeded list of outage candidates until N_STATES eligible ones are found. Resumable."""
    if STATES.exists():
        return json.loads(STATES.read_text())
    rng = random.Random(SEED)
    lines = [b["id"] for b in eng.static["branches"] if not b["bridge"] and b["rate_a"]]
    cands = []
    for i in range(MAX_CANDIDATES):
        k = 1 if i % 2 == 0 else 2
        cands.append((rng.choice(LOADS), sorted(rng.sample(lines, k))))
    seen = json.loads(CANDIDATES.read_text()) if CANDIDATES.exists() else {}
    chosen, kinds, loads, keys = [], {1: 0, 2: 0}, {}, set()
    for i, (load, outs) in enumerate(cands):
        k, key = len(outs), (load, tuple(outs))
        if kinds[k] >= PER_KIND or loads.get(load, 0) >= PER_LOAD or key in keys:
            continue
        if str(i) not in seen:
            if time.time() > deadline:
                save_json(CANDIDATES, seen)
                log(f"state search paused after candidate {i}, {len(chosen)} states so far")
                return None
            got = solve_all(eng, load, outs, None)
            if got is None:
                seen[str(i)] = None
            else:
                seen[str(i)] = float(got[0].loading_ac.max())
            if len(seen) % 20 == 0:
                save_json(CANDIDATES, seen)
        worst = seen[str(i)]
        if worst is None or not WORST_RANGE[0] <= worst <= WORST_RANGE[1]:
            continue
        keys.add(key)
        kinds[k] += 1
        loads[load] = loads.get(load, 0) + 1
        chosen.append({"state": len(chosen), "candidate": i, "load": load, "outages": outs, "worst_ac": worst})
        if len(chosen) == N_STATES:
            break
    save_json(CANDIDATES, seen)
    # candidate actions, from the generator table of the un-redispatched contingency state
    for st in chosen:
        _, gens = solve_all(eng, st["load"], st["outages"], None)
        free = gens[~gens.slack]
        rng = random.Random(SEED * 1000 + st["state"] + 1)
        room_down, room_up = free.p_set - free.p_min, free.p_max - free.p_set
        valid = [(int(a), int(b), d) for d in AMOUNTS for a, ra in zip(free.id, room_down) if ra >= d
                 for b, rb in zip(free.id, room_up) if rb >= d and a != b]
        actions = rng.sample(valid, min(N_ACTIONS, len(valid)))
        st["n_valid_actions"] = len(valid)
        st["actions"] = sorted(actions)
        st["n_free_gens"] = int(len(free))
    save_json(STATES, chosen)
    log(f"chose {len(chosen)} states from {len(seen)} solved candidates (seed {SEED}):")
    for st in chosen:
        log(f"  state {st['state']:2d}: load x{st['load']:.2f}, outages {st['outages']}, worst AC loading "
            f"{100 * st['worst_ac']:.1f}%, {len(st['actions'])} actions")
    return chosen


def run(budget):
    PARTS.mkdir(exist_ok=True)
    deadline = time.time() + budget
    eng = Engine(NETWORK, VARIANT)
    states = choose_states(eng, deadline)
    if states is None:
        return False
    for st in states:
        todo = [(-1, None)] + list(enumerate(st["actions"]))  # -1 is "no action", the anchor
        for c in range(0, len(todo), CHUNK):
            part = PARTS / f"s{st['state']:02d}_c{c // CHUNK:02d}.parquet"
            if part.exists():
                continue
            if time.time() > deadline:
                log(f"paused before state {st['state']} chunk {c // CHUNK}")
                return False
            frames = []
            for aid, act in todo[c:c + CHUNK]:
                red = None if act is None else {act[0]: -float(act[2]), act[1]: float(act[2])}
                got = solve_all(eng, st["load"], st["outages"], red)
                if got is None:
                    f = pd.DataFrame({"branch": [-1]})
                else:
                    f = got[0]
                f.insert(0, "action", aid)
                f["solved"] = got is not None
                f["gen_down"], f["gen_up"], f["mw"] = (-1, -1, 0) if act is None else act
                frames.append(f)
            data = pd.concat(frames, ignore_index=True)
            data.insert(0, "state", st["state"])
            data["load"], data["n_outages"] = st["load"], len(st["outages"])
            data["outages"] = ",".join(map(str, st["outages"]))
            tmp = part.with_suffix(".tmp")
            data.to_parquet(tmp)
            tmp.replace(part)
    parts = sorted(PARTS.glob("s*_c*.parquet"))
    pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(OUT)
    log(f"all done: {len(parts)} parts -> {OUT}")
    return True


# ----------------------------------------------------------------------------- report
def scores(data):
    """One row per (state, action): worst line loading after the action, by method."""
    base = data[data.action == -1].set_index(["state", "branch"])
    act = data[(data.action >= 0) & data.solved].join(base[["loading_ac", "loading_genco", "p_ac", "p_dc"]],
                                                      on=["state", "branch"], rsuffix="_0")
    rate = (act.p_dc_0.abs() / base.loading_dc.reindex(pd.MultiIndex.from_frame(act[["state", "branch"]])).to_numpy())
    act["loading_dc_anch"] = act.loading_ac_0 + np.sign(act.p_ac_0) * (act.p_dc - act.p_dc_0) / rate
    act["loading_genco_anch"] = act.loading_ac_0 + act.loading_genco - act.loading_genco_0
    cols = {m: f"loading_{m}" for m in ("ac",) + METHODS}
    g = act.groupby(["state", "action"])
    out = g[list(cols.values())].max().rename(columns={v: k for k, v in cols.items()})
    out["mismatch"] = g.newton_mismatch_mw.first()
    out["mw"] = g.mw.first()
    return out.reset_index(), base


def report():
    from scipy.stats import kendalltau, spearmanr

    data = pd.read_parquet(OUT)
    sc, base = scores(data)
    failed = data[(data.action >= 0) & ~data.solved].groupby("state").action.nunique()
    rng = np.random.default_rng(SEED)
    rows, pooled = [], {m: np.zeros(3, int) for m in METHODS}
    log("\n=== per state ===")
    for s, d in sc.groupby("state"):
        d = d.sort_values("action").reset_index(drop=True)
        ac = d.ac.to_numpy()
        clear = ac <= OVERLOAD
        order_ac = np.argsort(ac, kind="stable")
        w0 = base.loc[s].loading_ac.max()
        n_over0 = int((base.loc[s].loading_ac > OVERLOAD).sum())
        log(f"state {s}: before {100 * w0:.1f}% ({n_over0} lines over), {len(d)} actions solved "
            f"({int(failed.get(s, 0))} AC failures), {int(clear.sum())} clear all overloads, "
            f"best achievable {100 * ac.min():.1f}%, median {100 * np.median(ac):.1f}%")
        for m in METHODS + ("random",):
            x = rng.permutation(len(d)).astype(float) if m == "random" else d[m].to_numpy()
            order = np.argsort(x, kind="stable")
            row = {"state": s, "method": m, "kendall": kendalltau(x, ac).statistic, "spearman": spearmanr(x, ac).statistic,
                   "top5": len(set(order[:5]) & set(order_ac[:5])), "top10": len(set(order[:10]) & set(order_ac[:10])),
                   "regret": 100 * (ac[order[0]] - ac.min()),
                   "score_err": np.nan if m == "random" else 100 * np.abs(x - ac).mean()}
            if m != "random":
                pred = x <= OVERLOAD
                tp, fp, fn = int((pred & clear).sum()), int((pred & ~clear).sum()), int((~pred & clear).sum())
                pooled[m] += (tp, fp, fn)
                row["precision"] = tp / (tp + fp) if tp + fp else np.nan
                row["recall"] = tp / (tp + fn) if tp + fn else np.nan
            if clear.any():
                # random: expected position of the first clearing action in a random order
                row["checks"] = (len(d) + 1) / (clear.sum() + 1) if m == "random" else int(np.flatnonzero(clear[order])[0]) + 1
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(WORK / "remedial_case118_per_state.csv", index=False)

    def ms(x):
        x = x.dropna()
        return f"{x.mean():.2f}+-{x.std():.2f}" if len(x) else "n/a"

    table = []
    for m in METHODS + ("random",):
        r = res[res.method == m]
        ch = r.checks.dropna()
        tp, fp, fn = pooled.get(m, (0, 0, 0))
        table.append({
            "method": m, "kendall": ms(r.kendall), "spearman": ms(r.spearman), "top5/5": ms(r.top5), "top10/10": ms(r.top10),
            "regret_pts": f"{r.regret.mean():.2f} (max {r.regret.max():.2f})",
            "score_MAE_pts": f"{r.score_err.mean():.2f}" if m != "random" else "-",
            "clear_prec": f"{tp / (tp + fp):.3f}" if tp + fp else "-", "clear_rec": f"{tp / (tp + fn):.3f}" if tp + fn else "-",
            "pred_clear": tp + fp,
            "AC_checks med/max": f"{ch.median():.1f}/{ch.max():.1f}" if len(ch) else "n/a",
        })
    n_none = int(res[res.method == "genco"].checks.isna().sum())
    log(f"\n=== {NETWORK}: {res.state.nunique()} states, mean+-sd over states; precision/recall pooled over all actions ===")
    log(pd.DataFrame(table).to_string(index=False))
    log(f"states where no candidate clears all overloads (AC): {n_none} of {res.state.nunique()}; "
        f"AC_checks over the other {res.state.nunique() - n_none}")
    log(f"actions: {len(sc)} solved, {int(failed.sum())} AC failures; clearing actions {int((sc.ac <= OVERLOAD).sum())}")
    log(f"Newton step: mismatch after the step median {sc.mismatch.median():.3f} MW, max {sc.mismatch.max():.2f} MW")
    for mw, d in sc.groupby("mw"):
        log(f"  d={mw:3d} MW: score MAE in points  " + "  ".join(f"{m} {100 * (d[m] - d.ac).abs().mean():.2f}" for m in METHODS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--budget", type=float, default=45, help="seconds of work per call")
    args = ap.parse_args()
    if args.cmd == "run":
        sys.exit(0 if run(args.budget) else 3)
    report()

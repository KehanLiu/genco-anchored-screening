"""Frozen-policy scoring of the fresh GOC 2000 full N-1 (work/fresh/README.md sections 2.2, 4 and 5).

    .venv/bin/python work/fresh/score_fresh_n1.py --policy work/fresh/policy_frozen.json

Reads  work/fulln1/chunks_<grid>_fresh/           per-contingency chunks of `analysis/analysis_full_n1.py run --tag _fresh --policy ...`
       work/fresh/policy_frozen.json              tau per start, loading cushions c, classical-first tau1, lambda (all frozen)
       work/fulln1/timing_<grid>_fresh.json       unit costs of the fresh timing run (estimated s per 1,000, with graph prep)
       work/fulln1/e2e_<grid>_fresh.parquet/.json measured end-to-end ms per outage (analysis/analysis_full_n1.py e2e), if present
Writes work/fresh/results_n1.md, results_n1.csv (pooled rows) and results_n1_per_base.csv (paired per-base rows).
Nothing is fitted or chosen here; the chunk loading and the continuation rule are imported from analysis_full_n1
(load_chunks, workflows), not re-implemented.

Policies scored (every row at every cushion c of the policy; truth stays > 1.001):
    W0   full AC from the pre-outage solution                       W1   classical one step, tau_prev
    W4   anchored change start + one step, tau_hyb                   W4s  shrunk anchored start (frozen lambdas), tau_shyb
    gate mismatch gate: the start with the smaller initial mismatch (tie / NaN -> pre-outage) takes the step with its tau
    classical-first: W1 first; the anchored start (W4 rule) only where W1's one-step mismatch > tau1 or W1 flags
                     (mirrors the `classical_first` rule of the independent screening re-analysis)

Base sets (POLICY_LOG.md A2): the six ranks of policy_frozen.json n1_base_ranks were registered first; the other 18 ranks of
the 24-rank design are an extension recorded before any fresh N-1 outcome was inspected. Every variant is reported per base
(paired) and pooled over (i) the six registered ranks, (ii) the 18-rank extension, (iii) all 24; bases missing from the chunks
are marked.

--existing-dev-data  DEVELOPMENT ONLY: scores the TEST split of work/fulln1/contingencies.parquet (the files every policy
                     was developed on) instead of the fresh chunks, checks the numbers against
                     work/fresh/review_checks/screening_results.json and writes results_n1_devcheck.* -- never the fresh outputs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rel_path(p):
    """Path relative to the repository root when it lies inside it (keeps the report free of machine-specific paths)."""
    q = Path(p).resolve()
    return str(q.relative_to(ROOT)) if q.is_relative_to(ROOT) else str(p)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

REPORTED = ("W0", "W1", "W4", "W4s", "gate")
LABEL = {"W0": "full AC, every contingency", "W1": "classical one step (pre-outage start)", "W4": "anchored change + one step",
         "W4s": "shrunk anchored start + one step", "gate": "mismatch gate (smaller mis0 start + one step)"}
E2E_ORDER = ("W0", "W1", "W4", "W4_mps16", "W4s", "gate", "CF0.1", "CF0.2", "CF0.5", "CF1")


def score(df, flag, its, n_inf, it0, tm):
    import numpy as np
    crit = df.critical.to_numpy()
    missed = ~flag & crit
    n = len(df)
    worst = df["worst_loading"] if "worst_loading" in df else df["worst"]
    r = {"contingencies": n, "critical": int(crit.sum()), "missed": int(missed.sum()), "missed_share_of_critical": float(missed.sum() / max(crit.sum(), 1)),
         "sent_to_ac": int(flag.sum()), "sent_to_ac_share": float(flag.mean()), "secure_sent_to_ac": int((flag & ~crit).sum()),
         "newton_iterations": int(its.sum()), "iterations_vs_w0": float(its.sum() / it0), "genco_inferences": int(n_inf),
         "worst_missed_loading": float(np.nanmax(worst.to_numpy()[missed])) if missed.any() and np.isfinite(worst.to_numpy()[missed]).any() else np.nan}
    if tm:
        r["est_s_per_1000_with_prep"] = (its.sum() * tm["ms_iter_full"] + n_inf * (tm["ms_forward_mps_b16"] + tm["ms_prep"])) / n
    return r


def policies(df, policy, tm):
    """All policy rows of the frozen policy on one frame (critical, it_*, conv_*, mis0_*, mis1_*, max1_* columns)."""
    import numpy as np
    import analysis_full_n1 as fn1
    tau = policy["tau_mw"]
    tau_wf = {"W1": tau["prev"], "W3": tau["genco"], "W4": tau["hyb"], "W4s": tau["shyb"]}
    n, nb = len(df), df.base.nunique()
    it0 = int(df.it_prev.sum())
    base = df.base.to_numpy()
    has_shyb = "max1_shyb" in df
    rows = []
    for c in policy["loading_cushion_c"]:
        wf = fn1.workflows(df, tau_wf, lim=c)
        n_inf = {"W0": 0, "W1": 0, "W4": n + nb, "W4s": n + nb, "gate": n + nb}
        for w in ("W0", "W1", "W4", "W4s"):
            if w not in wf:
                continue
            flag, its = wf[w]
            rows.append({"family": "loading_cushion", "policy": w, "label": LABEL[w], "c": c, "tau_mw": tau_wf.get(w, np.nan), "tau1_mw": np.nan,
                         **score(df, flag, its, n_inf[w], it0, tm)})
        # mismatch gate: per outage the start with the smaller initial mismatch (tie / NaN -> pre-outage), then that start's W1 / W4 rule
        m_prev, m_hyb = df.mis0_prev.to_numpy(float), df.mis0_hyb.to_numpy(float)
        use_hyb = np.isfinite(m_hyb) & (m_hyb < m_prev)
        flag = np.where(use_hyb, wf["W4"][0], wf["W1"][0])
        its = np.where(use_hyb, wf["W4"][1], wf["W1"][1])
        r = score(df, flag, its, n_inf["gate"], it0, tm)
        r["hyb_start_share"] = float(use_hyb.mean())
        rows.append({"family": "loading_cushion", "policy": "gate", "label": LABEL["gate"], "c": c, "tau_mw": f"{tau['prev']}/{tau['hyb']}", "tau1_mw": np.nan, **r})
        # classical-first (screening_reanalysis.py `classical_first` with second start hyb; its default cutoff 1.001 is the c = 1.001 row)
        first_cost = np.minimum(df.it_prev.to_numpy(), 1)
        for tau1 in policy["classical_first_tau1_mw"]:
            first = (df.max1_prev.to_numpy() > c) | ~(df.mis1_prev.to_numpy() <= tau1)
            second_flag, second_its = wf["W4"]
            flag = first & second_flag
            its = np.where(first, first_cost + second_its, first_cost)
            n_inf = int(first.sum()) + len(np.unique(base[first]))
            r = score(df, flag, its, n_inf, it0, tm)
            r["genco_share"] = float(first.mean())
            rows.append({"family": "classical_first", "policy": f"CF tau1={tau1:g}", "label": "W1 first, anchored start where W1 mismatch > tau1 or W1 flags",
                         "c": c, "tau_mw": tau["hyb"], "tau1_mw": tau1, **r})
    if not has_shyb:
        rows.append({"family": "loading_cushion", "policy": "W4s", "label": LABEL["W4s"] + " -- NOT AVAILABLE (no shyb columns in these chunks)", "c": np.nan})
    return rows


def variant_key(r):
    """(policy, c, tau1) identifies one frozen variant across base sets and bases."""
    import numpy as np
    t1 = r.get("tau1_mw", np.nan)
    return (r["policy"], r.get("c"), None if t1 is None or (isinstance(t1, float) and np.isnan(t1)) else float(t1))


def e2e_workflow(r):
    """Name of the e2e workflow timed for this policy row (the timed gate ran with c = 1.001), or None."""
    w, c, t1 = variant_key(r)
    if w == "W0":
        return "W0"
    if c != 1.001:
        return None
    return f"CF{t1:g}" if w.startswith("CF") else w


def e2e_stats(part):
    """Measured cost and accuracy of one workflow on the identical outage sample (rows of e2e_<grid><tag>.parquet)."""
    import numpy as np
    n = len(part)
    if not n:
        return None
    first = part["ms_first"] if "ms_first" in part else part["ms"]
    r = {"e2e_n": n, "e2e_bases": int(part.base.nunique()), "e2e_total_first_ms": float(first.sum()), "e2e_total_min_ms": float(part.ms.sum()),
         "e2e_ms_mean": float(part.ms.mean()), "e2e_ms_median": float(part.ms.median()), "e2e_ms_p90": float(part.ms.quantile(.9)),
         "e2e_flagged": int(part.flagged.sum()), "e2e_flagged_ms": float(part.ms[part.flagged].sum()), "e2e_iterations_mean": float(part.iterations.mean())}
    if "critical" in part:
        miss = (part.critical & ~part.flagged) if "missed" not in part else part.missed.astype(bool)
        ff = (~part.critical & part.flagged) if "false_flag" not in part else part.false_flag.astype(bool)
        r.update(e2e_critical=int(part.critical.sum()), e2e_missed=int(miss.sum()), e2e_false_flag=int(ff.sum()))
    if "failed" in part or "converged" in part:  # failed = no solution after all fallbacks (older files: not converged)
        nc = part.failed.astype(bool) if "failed" in part else ~part.converged.astype(bool)
        r.update(e2e_not_converged=int(nc.sum()), e2e_not_converged_ms=float(part.ms[nc].sum()))
    if "restarted" in part:
        r["e2e_restarted"] = int(part.restarted.astype(bool).sum())
    if "ms_cont" in part:
        r["e2e_fallback_ms"] = float(np.nansum(part.ms_cont.to_numpy(float)))
    if "block_size" in part and part.block_size.notna().any():
        r["e2e_partial_blocks"] = int((part.drop_duplicates(["base", "block"]).block_size < part.block_size.max()).sum()) if "block" in part else np.nan
    return r


def base_inference(ej, df):
    """Pre-outage GENCO inference per base (CPU b1 / MPS b1 ms, from the e2e json) and its amortisation over the base's N-1 sweep."""
    if not ej or not ej.get("base_inference"):
        return None
    per_base = df.groupby("base").size()
    rows = []
    for b in ej["base_inference"]:
        n_sweep = int(per_base.get(b["base"], 0))
        rows.append({**b, "n_outages_sweep": n_sweep,
                     "ms_cpu_b1_amortized_sweep": b["ms_cpu_b1"] / n_sweep if n_sweep else float("nan"),
                     "ms_cpu_b1_amortized_sample": b["ms_cpu_b1"] / b["n_sample"] if b.get("n_sample") else float("nan")})
    return rows


def attach_e2e(rows, e2e, ranks_of_set):
    """Measured e2e cost + per-sample misses / false flags of the matching workflow, restricted to the bases of each row's set."""
    if e2e is None:
        return
    for r in rows:
        w = e2e_workflow(r)
        if w is None or w not in set(e2e.workflow):
            continue
        part = e2e[(e2e.workflow == w) & e2e["rank"].isin(ranks_of_set[r["set"]])]
        st = e2e_stats(part)
        if st:
            r.update(st)
            if w == "W4" and "W4_mps16" in set(e2e.workflow):
                m = e2e_stats(e2e[(e2e.workflow == "W4_mps16") & e2e["rank"].isin(ranks_of_set[r["set"]])])
                r.update({k.replace("e2e_", "e2e_mps16_"): v for k, v in m.items() if k in ("e2e_ms_mean", "e2e_ms_median", "e2e_ms_p90", "e2e_total_first_ms")})


POOLED_COLS = [("set", "bases"), ("policy", "policy"), ("c", "c"), ("tau_mw", "tau MW"), ("tau1_mw", "tau1 MW"), ("contingencies", "contingencies"), ("critical", "critical"),
               ("missed", "missed"), ("missed_share_of_critical", "missed/critical"), ("sent_to_ac", "to AC"), ("sent_to_ac_share", "share to AC"),
               ("secure_sent_to_ac", "secure to AC"), ("newton_iterations", "Newton it"), ("iterations_vs_w0", "vs W0"), ("genco_inferences", "GENCO inf"),
               ("est_s_per_1000_with_prep", "est s/1000 (+prep)"), ("e2e_n", "e2e n"), ("e2e_ms_mean", "e2e ms mean"), ("e2e_ms_median", "e2e ms med"),
               ("e2e_ms_p90", "e2e ms p90"), ("e2e_total_first_ms", "e2e total ms (pass 1)"), ("e2e_missed", "e2e missed"), ("e2e_false_flag", "e2e false flags"),
               ("worst_missed_loading", "worst missed")]
BASE_COLS = [("rank", "rank"), ("registered", "set"), ("base", "load_scenario_idx"), ("load_mw", "load MW"), ("contingencies", "contingencies"), ("critical", "critical"),
             ("missed", "missed"), ("secure_sent_to_ac", "secure to AC"), ("sent_to_ac_share", "share to AC"), ("worst_missed_loading", "worst missed"),
             ("newton_iterations", "Newton it"), ("iterations_vs_w0", "vs W0")]


def fmt(k, v):
    import numpy as np
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if k in ("missed_share_of_critical", "sent_to_ac_share", "iterations_vs_w0"):
        return f"{v:.2%}"
    if k.startswith("est_") or k.startswith("e2e_ms") or k.startswith("e2e_total") or k.startswith("e2e_mps16_ms") or k.startswith("ms_"):
        return f"{v:.2f}"
    if k == "worst_missed_loading":
        return f"{v:.1%}"
    if k in ("c", "load_mw"):
        return f"{v:g}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def markdown(rows, cols, head=()):
    out = list(head) + ["", "| " + " | ".join(c[1] for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(k, r.get(k)) for k, _ in cols) + " |")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--grid", default="2000")
    p.add_argument("--tag", default="_fresh")
    p.add_argument("--data", default="", help="fresh datakit pool (default work/fresh/data<grid>); sets GOC_DATA for design()")
    p.add_argument("--existing-dev-data", action="store_true", help="DEVELOPMENT ONLY: score the test split of work/fulln1/contingencies.parquet and check "
                                                                   "against work/fresh/review_checks/screening_results.json; writes results_n1_devcheck.*")
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    policy = json.loads(Path(a.policy).read_text())
    out_dir = ROOT / "work" / "fresh"
    fulln1 = ROOT / "work" / "fulln1"
    six = [int(r) for r in policy["n1_base_ranks"]]
    if a.existing_dev_data:
        src = fulln1 / "contingencies.parquet"
        df = pd.read_parquet(src)
        if "rank" not in df:  # the development table has no rank column: design() ranks the 24 bases by total load
            order = df.drop_duplicates("base").sort_values("load_mw").base.to_numpy()
            df = df.merge(pd.DataFrame({"base": order, "rank": range(len(order))}), on="base")
        df = df[df.split == "test"].reset_index(drop=True)
        design_ranks = sorted(int(r) for r in df["rank"].unique())
        tfile, efile = fulln1 / f"timing_{a.grid}.json", None
        stem, bases_note = "results_n1_devcheck", f"test split of {src.relative_to(ROOT)} ({df.base.nunique()} bases)"
        head = ["# DEVELOPMENT CHECK of the fresh scorer on the existing (development) N-1 files -- NOT a fresh result", ""]
        sets = [("test split", design_ranks)]
        set_note = "- base sets: the development test split only (no registered / extension distinction on development data)"
    else:
        os.environ["GOC_DATA"] = a.data or str(ROOT / "work" / "fresh" / f"data{a.grid}")
        import analysis_full_n1 as fn1
        d = fn1.chunk_dir(a.grid, a.tag)
        if not [f for f in d.glob("b*_c*.parquet") if ".tmp" not in f.name]:
            sys.exit(f"score_fresh_n1: no fresh chunks in {d}; run `analysis/analysis_full_n1.py run --tag {a.tag} --policy ...` first (nothing scored, "
                     f"work/fulln1/contingencies.parquet is never used for the fresh result)")
        bases, always_out, bridge, cand = fn1.design(a.grid)
        df, hist, nanc, info, files = fn1.load_chunks(d, bases)
        ids_file = out_dir / "fresh_ids.json"
        if a.tag == "_fresh" and ids_file.exists():  # the fresh result may only come from the fresh pool's load scenarios
            fresh = set(json.loads(ids_file.read_text())[str(a.grid)])
            assert set(df.base.unique()) <= fresh, f"chunk bases {sorted(set(df.base.unique()) - fresh)} are not in {ids_file}"
        tfile, efile = fulln1 / f"timing_{a.grid}{a.tag}.json", fulln1 / f"e2e_{a.grid}{a.tag}.parquet"
        stem = "results_n1"
        design_ranks = [int(r) for r in bases["rank"]]
        present = sorted(int(r) for r in df["rank"].unique())
        # incomplete bases (fewer outages than the design) are still scored but marked
        n_per = df.groupby("rank").size()
        bases_note = (f"{len(files)} chunks of {d.relative_to(ROOT)}, design ranks {present} = load_scenario_idx "
                      f"{df.groupby('rank').base.first().tolist()} of {rel_path(os.environ['GOC_DATA'])} (expected {len(cand):,} outages per base; "
                      f"incomplete: {sorted(int(r) for r in n_per.index[n_per < len(cand)])})")
        head = [f"# Fresh GOC {a.grid} full N-1 under the frozen policies ({a.policy})", ""]
        ext = [r for r in design_ranks if r not in six]
        sets = [("six registered", [r for r in present if r in six]), ("18-rank extension", [r for r in present if r in ext]), ("all 24", present)]
        set_note = (f"- base sets (POLICY_LOG.md A2): the six ranks {six} were REGISTERED FIRST (policy_frozen.json n1_base_ranks); the 18 other ranks "
                    f"{ext} are an EXTENSION recorded before any fresh N-1 outcome was inspected. Present in the chunks: {len(sets[0][1])}/6 registered, "
                    f"{len(sets[1][1])}/18 extension, {len(present)}/24 in all; missing ranks: {[r for r in design_ranks if r not in present] or 'none'}")
    sets = [(name, ranks) for name, ranks in sets if ranks]
    ranks_of_set = dict(sets)
    tm = json.loads(tfile.read_text()) if tfile.exists() else None
    e2e = pd.read_parquet(efile) if efile and efile.exists() else None
    ej = efile.with_suffix(".json") if efile else None
    ej = json.loads(ej.read_text()) if ej and ej.exists() else None
    proto = (ej or {}).get("protocol", {})
    head += [f"- contingencies: {len(df):,} from {bases_note}; critical (truth > 1.001 or no solution): {int(df.critical.sum()):,}",
             set_note,
             f"- frozen policy: tau MW {policy['tau_mw']}, loading cushions c {policy['loading_cushion_c']}, classical-first tau1 MW {policy['classical_first_tau1_mw']}, "
             f"lambda {policy['shrinkage']['lambda'].get(f'goc{a.grid}_tiny_seed0')}; nothing re-fitted",
             f"- unit costs for the estimate: {tfile.relative_to(ROOT) if tm else 'none (no timing json) -> estimate column empty'}"
             + (f" (Newton iteration {tm['ms_iter_full']:.2f} ms, GENCO forward mps b16 {tm.get('ms_forward_mps_b16', float('nan')):.2f} ms + graph prep {tm['ms_prep']:.2f} ms)" if tm else ""),
             f"- measured end-to-end: {efile.relative_to(ROOT) if e2e is not None else 'none (no e2e parquet) -> e2e columns empty'}"
             + (f" ({ej['sample_outages_per_base']} outages x {len(ej['bases'])} bases (ranks {ej['bases']}), {ej['repetitions']} repetitions, gate c = {ej['gate_c']}, "
                f"torch threads {ej.get('torch_threads')}; excluded: {', '.join(ej['excluded'])}; W4_mps16: {ej.get('W4_mps16')})" if ej else ""),
             "- W0 = Newton to convergence from the pre-outage solution; `vs W0` = Newton iterations relative to it; secure to AC = secure contingencies sent to full AC",
             "- e2e columns (c = 1.001 rows and W0 only, the cushion inside the timed gate; restricted to the bases of the row's set): n timed outages, per-outage ms "
             "(min over repetitions) mean / median / p90, total ms of the first (untrimmed) pass over the sample, and misses / false flags of the timed workflow on "
             "that identical sample. Pre-outage GENCO inference per base is reported separately below, not inside these columns."]
    pooled = []
    for name, ranks in sets:
        sub = df[df["rank"].isin(ranks)].reset_index(drop=True)
        for r in policies(sub, policy, tm):
            r["set"] = f"{name} ({len(ranks)})"
            pooled.append(r)
    ranks_of_set = {f"{name} ({len(ranks)})": ranks for name, ranks in sets}
    attach_e2e(pooled, e2e, ranks_of_set)
    # paired per-base rows (same frozen variants, one base at a time; vs W0 is that base's own full-AC iteration count)
    per_base = []
    meta = df.drop_duplicates("base").set_index("rank")[["base", "load_mw"]]
    for rank in sorted(int(r) for r in df["rank"].unique()):
        sub = df[df["rank"] == rank].reset_index(drop=True)
        for r in policies(sub, policy, None):
            if "missed" not in r:
                continue
            r.update(rank=rank, base=int(meta.loc[rank, "base"]), load_mw=float(meta.loc[rank, "load_mw"]), registered="R" if rank in six else "ext")
            per_base.append(r)
    lines = list(head)
    for fam, title in (("loading_cushion", "## Loading cushion: W0, W1, W4, W4s and the mismatch gate at every c (pooled per base set)"),
                       ("classical_first", "## Classical-first: W1, then the anchored start where W1's mismatch > tau1 or W1 flags (pooled per base set)")):
        lines += ["", title] + markdown([r for r in pooled if r["family"] == fam], POOLED_COLS)
    if e2e is not None:
        lines += ["", "## Measured end-to-end cost per workflow (analysis_full_n1.py e2e; identical outage sample for every workflow)", "",
                  f"- protocol: {json.dumps(proto, default=str)[:1500] if proto else 'see the e2e json'}",
                  "- ms per outage = min over repetitions; total (pass 1) = sum of the first repetition's times, no min-taking; total (min) = sum of the per-outage minima;",
                  "  flagged = sent to full AC (continuation executed) with the total ms of those outages; restarted = continuation broke down and was re-solved from the "
                  "pre-outage start; not converged = no solution within max_it after all fallbacks (counted critical by W0, flagged by every screen) with the ms spent on them",
                  "", "| workflow | gate | n | bases | total ms pass 1 | total ms (min) | mean ms | median ms | p90 ms | mean it | flagged | flagged ms | restarted | not conv | not conv ms | critical | missed | false flags |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for w in [x for x in E2E_ORDER if x in set(e2e.workflow)] + sorted(set(e2e.workflow) - set(E2E_ORDER)):
            part = e2e[e2e.workflow == w]
            for name, sub in (("all", part), ("flagged", part[part.flagged]), ("cleared", part[~part.flagged])):
                s = e2e_stats(sub)
                if s:
                    g = lambda k, f="{:.2f}": (f.format(s[k]) if k in s and s[k] == s[k] else "")  # noqa: E731
                    lines.append(f"| {w} | {name} | {s['e2e_n']:,} | {s['e2e_bases']} | {g('e2e_total_first_ms')} | {g('e2e_total_min_ms')} | {g('e2e_ms_mean')} | {g('e2e_ms_median')} | "
                                 f"{g('e2e_ms_p90')} | {g('e2e_iterations_mean')} | {s['e2e_flagged']} | {g('e2e_flagged_ms')} | {g('e2e_restarted', '{}')} | {g('e2e_not_converged', '{}')} | "
                                 f"{g('e2e_not_converged_ms')} | {g('e2e_critical', '{}')} | {g('e2e_missed', '{}')} | {g('e2e_false_flag', '{}')} |")
        if ej and ej.get("mps_blocks"):
            mb = ej["mps_blocks"]
            lines += ["", f"- W4_mps16 blocks: {mb}"]
        bi = base_inference(ej, df)
        if bi:
            lines += ["", "### Pre-outage GENCO inference per base (needed once per base by the anchored starts; NOT inside the per-outage columns)", "",
                      "| rank | load_scenario_idx | ms CPU b1 | ms MPS b1 | outages in sweep | amortized ms/outage (sweep) | e2e sample | amortized ms/outage (sample) |", "|---|---|---|---|---|---|---|---|"]
            for b in bi:
                lines.append(f"| {b['rank']} | {b['base']} | {b['ms_cpu_b1']:.2f} | {b.get('ms_mps_b1', float('nan')):.2f} | {b['n_outages_sweep']:,} | {b['ms_cpu_b1_amortized_sweep']:.4f} | "
                             f"{b.get('n_sample', '')} | {b['ms_cpu_b1_amortized_sample']:.4f} |")
            tot = sum(b["ms_cpu_b1"] for b in bi)
            sw = sum(b["n_outages_sweep"] for b in bi)
            lines.append(f"\n- total pre-outage inference (CPU b1) over the {len(bi)} timed bases: {tot:.1f} ms = {tot / max(sw, 1):.4f} ms per sweep outage "
                         f"({tot / max(sum(b.get('n_sample', 0) for b in bi), 1):.4f} ms per e2e-sample outage)")
    lines += ["", "## Per-base paired counts (one table per frozen variant; R = registered six, ext = 18-rank extension; pooled rows at the end of each table)",
              "", f"- design ranks {design_ranks}; bases NOT in the chunks are listed as missing; rows are the same frozen variants as above, vs W0 is per base",
              f"- missing from the chunks: {[r for r in design_ranks if r not in set(int(x) for x in df['rank'].unique())] or 'none'}"]
    keys = []
    for r in pooled:
        k = variant_key(r)
        if "missed" in r and k not in keys:
            keys.append(k)
    for k in keys:
        w, c, t1 = k
        lines += ["", f"### {w}  c = {c:g}" + (f"  tau1 = {t1:g} MW" if t1 is not None else "")]
        rows_k = [r for r in per_base if variant_key(r) == k]
        rows_k.sort(key=lambda r: r["rank"])
        have = {r["rank"] for r in rows_k}
        for rank in design_ranks:
            if rank not in have:
                rows_k.append({"rank": rank, "registered": "R" if rank in six else "ext", "base": "MISSING from chunks"})
        rows_k.sort(key=lambda r: r["rank"])
        pooled_k = [{**r, "rank": "pooled", "registered": r["set"], "base": "", "load_mw": np.nan} for r in pooled if variant_key(r) == k]
        lines += markdown(rows_k + pooled_k, BASE_COLS)
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(pooled).to_csv(out_dir / f"{stem}.csv", index=False)
    pd.DataFrame(per_base).to_csv(out_dir / f"{stem}_per_base.csv", index=False)
    print("\n".join(lines))
    print(f"\nsaved {out_dir / f'{stem}.md'}, {out_dir / f'{stem}.csv'} and {out_dir / f'{stem}_per_base.csv'}")

    if a.existing_dev_data:  # reproduce the development numbers of the independent re-analysis
        ref = json.loads((ROOT / "work" / "fresh" / "review_checks" / "screening_results.json").read_text())
        got = {(r["policy"], r["c"]): r for r in pooled if "missed" in r}
        checks = []
        for x in ref["fixed_tau_loading_cushion"]:
            w = {"prev": "W1", "hyb": "W4", "genco": None}[x["start"]]
            if w:
                g = got[(w, x["decision_cutoff"])]
                checks.append((f"{w} c={x['decision_cutoff']}", (x["missed_contingencies"], x["sent_to_full_ac"], x["newton_iterations"]), (g["missed"], g["sent_to_ac"], g["newton_iterations"])))
        for x in ref["classical_first"]:
            if x["second_start"] == "hyb" and x["first_mismatch_threshold_mw"] in policy["classical_first_tau1_mw"]:
                g = got[(f"CF tau1={x['first_mismatch_threshold_mw']:g}", 1.001)]
                checks.append((f"classical-first tau1={x['first_mismatch_threshold_mw']}", (x["missed_contingencies"], x["sent_to_full_ac"], x["newton_iterations"], x["genco_inferences"]),
                               (g["missed"], g["sent_to_ac"], g["newton_iterations"], g["genco_inferences"])))
        # paired rows must add up to the pooled row
        for k in {variant_key(r) for r in pooled if "missed" in r}:
            pb = [r for r in per_base if variant_key(r) == k]
            po = [r for r in pooled if variant_key(r) == k][0]
            checks.append((f"per-base sum {k[0]} c={k[1]:g}", (po["missed"], po["sent_to_ac"], po["newton_iterations"]),
                           (sum(r["missed"] for r in pb), sum(r["sent_to_ac"] for r in pb), sum(r["newton_iterations"] for r in pb))))
        ok = True
        for name, exp, g in checks:
            ok &= exp == g
            print(f"  check {name:<34} expected (missed, sent, iterations[, inferences]) {exp} got {g} {'OK' if exp == g else 'MISMATCH'}")
        print("DEV CHECK PASSED: scorer reproduces work/fresh/review_checks/screening_results.json" if ok else "DEV CHECK FAILED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

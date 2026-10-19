"""Score the frozen screening policy (hybrid / shrunk starts and the mismatch gate) on the FRESH load scenarios.

    .venv/bin/python work/fresh/score_fresh_hybrid.py --policy work/fresh/policy_frozen.json [--tag _fresh]
    .venv/bin/python work/fresh/score_fresh_hybrid.py --policy work/fresh/policy_frozen.json --existing-dev-data   # development only

Reads, per grid / checkpoint of policy["checkpoints"], work/goc<N>/{lines,states}_prev_tiny_seed0<tag>.parquet (pre-outage
start, analysis/analysis_prev_control.py) and {lines,states}_hybrid_<ckpt><tag>.parquet (analysis/analysis_hybrid_start.py --policy, which adds
the shrunk start shyb). Starts scored after ONE Newton step: prev1, hyb1, shyb1, and the mismatch gate of the policy (per
outage the start with the smaller initial bus mismatch mis0; tie or NaN -> pre-outage): gate = prev1 | hyb1, sgate = prev1 | shyb1.
Definitions follow analysis_hybrid_start.report and the independent re-analysis: per state the maximum loading
over its lines; a missing (NaN / diverged) prediction counts as flagged (filled with 9.99); truth is a line > 1.001 of
rating; states without any finite true loading are excluded; Newton iterations are averaged over converged full Newton runs
of the start used.

Two decision rules are scored for every grid / checkpoint / start / cushion c (policy["loading_cushion_c"]):
  policy = loading_only : loading-only warm-start diagnostic (decision: any line > c; mismatch ignored). The original tables.
  policy = full_policy  : complete frozen policy (decision: any line > c OR post-step mismatch > tau of that start; tau =
                          policy["tau_mw"], transferred unchanged from the old GOC 2000 N-1 calibration, work/fulln1/report.log
                          lines 93-95; never re-fitted). The post-step mismatch is the stored mis1_prev (states_prev) and
                          mis1_hyb / mis1_shyb (states_hybrid); for gate / sgate the tau of the start chosen per state applies
                          (prev 0.5, hyb / shyb 5 MW); a non-finite or missing mis1 counts as above tau (referred).
                          added_referrals = secure referrals under the full policy minus loading-only.
An internal audit found that the original scorer ignored tau; the full-policy
rows are verified against work/fresh/review_checks/batch3_hybrid_tau_reanalysis.csv when it exists (the run fails on any difference).
Writes work/fresh/results_hybrid.md and .csv (or results_hybrid_existing_dev.* with --existing-dev-data).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
BIG = 9.99
TRUTH = 1.001
RECHECK_CSV = ROOT / "work" / "fresh" / "review_checks" / "batch3_hybrid_tau_reanalysis.csv"
LABEL_LOADING = "loading-only warm-start diagnostic (decision: any line > c; mismatch ignored)"
LABEL_FULL = ("complete frozen policy (decision: any line > c OR post-step mismatch > tau of that start; tau transferred unchanged "
              "from the old GOC 2000 N-1 calibration, work/fulln1/report.log lines 93-95; never re-fitted)")


def per_state(path, preds, truth=False):
    """Per-state maxima of the prediction columns (NaN -> BIG) and, with truth, max ac, number of lines and of lines > TRUTH."""
    cols = ["state", *preds] + (["ac"] if truth else [])
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=250000, columns=cols):
        x = batch.to_pandas()
        x[preds] = x[preds].fillna(BIG)
        g = x.groupby("state")
        m = g[preds].max()
        if truth:
            m["ac"] = g.ac.max()
            m["n_lines"] = g.size()
            m["n_real"] = (x.ac > TRUTH).groupby(x.state).sum()
        else:
            m["n_lines"] = g.size()
        parts.append(m)
    allp = pd.concat(parts)
    g = allp.groupby(level=0)
    out = g[[c for c in allp.columns if c not in ("n_lines", "n_real")]].max()
    out["n_lines"] = g.n_lines.sum()
    if truth:
        out["n_real"] = g.n_real.sum()
    return out


def score(grid, ckpt, tag, pol, notes):
    here = ROOT / "work" / f"goc{grid}"
    files = {k: here / f"{k}{tag}.parquet" for k in ("lines_prev_tiny_seed0", "states_prev_tiny_seed0", f"lines_hybrid_{ckpt}", f"states_hybrid_{ckpt}")}
    missing = [str(f.relative_to(ROOT)) for f in files.values() if not f.exists()]
    if missing:
        return None, missing
    lh, lp = files[f"lines_hybrid_{ckpt}"], files["lines_prev_tiny_seed0"]
    hyb_cols = pq.ParquetFile(lh).schema.names
    has_shyb = "shyb1" in hyb_cols
    sth = pd.read_parquet(files[f"states_hybrid_{ckpt}"]).set_index("state")
    stp = pd.read_parquet(files["states_prev_tiny_seed0"]).set_index("state")
    ah = per_state(lh, ["hyb1"] + (["shyb1"] if has_shyb else []), truth=True)
    ap = per_state(lp, ["prev1"])
    if not ah.index.equals(ap.index) or not (ah.n_lines.to_numpy() == ap.n_lines.loc[ah.index].to_numpy()).all():
        notes.append(f"goc{grid} {ckpt}: hybrid and prev line files differ in states or lines per state (hybrid {len(ah)}, prev {len(ap)} states); inner join used")
    a = ah.join(ap.drop(columns="n_lines"), how="inner")
    scols = ["mis0_hyb", "mis1_hyb", "it_hyb", "conv_hyb"] + (["mis0_shyb", "mis1_shyb", "it_shyb", "conv_shyb"] if has_shyb else [])
    a = a.join(sth[scols], how="inner").join(stp[["mis0_prev", "mis1_prev", "it_prev", "conv_prev"]], how="inner")
    missing_truth = int((~np.isfinite(a.ac)).sum())
    a = a[np.isfinite(a.ac)]
    real = (a.ac > TRUTH).to_numpy()
    n, n_crit, n_sec = len(a), int(real.sum()), int((~real).sum())
    n_real_lines = int(a.n_real.sum())
    tau = pol["tau_mw"]
    full = lambda v: np.full(n, float(v))  # noqa: E731
    # per start: one-step loading, Newton iterations, converged, gate share, post-step mismatch mis1, tau of the start (per state), tau label
    starts = {"prev1": (a.prev1.to_numpy(), a.it_prev.to_numpy(), a.conv_prev.to_numpy(bool), None, a.mis1_prev.to_numpy(float), full(tau["prev"]), f"prev {tau['prev']}"),
              "hyb1": (a.hyb1.to_numpy(), a.it_hyb.to_numpy(), a.conv_hyb.to_numpy(bool), None, a.mis1_hyb.to_numpy(float), full(tau["hyb"]), f"hyb {tau['hyb']}")}
    gate = (a.mis0_hyb < a.mis0_prev).to_numpy()  # NaN on either side -> False -> pre-outage start; tie -> pre-outage
    starts["gate"] = (np.where(gate, a.hyb1, a.prev1), np.where(gate, a.it_hyb, a.it_prev), np.where(gate, a.conv_hyb, a.conv_prev).astype(bool), float(gate.mean()),
                      np.where(gate, a.mis1_hyb, a.mis1_prev).astype(float), np.where(gate, tau["hyb"], tau["prev"]).astype(float), f"chosen start: prev {tau['prev']} or hyb {tau['hyb']}")
    if has_shyb:
        starts["shyb1"] = (a.shyb1.to_numpy(), a.it_shyb.to_numpy(), a.conv_shyb.to_numpy(bool), None, a.mis1_shyb.to_numpy(float), full(tau["shyb"]), f"shyb {tau['shyb']}")
        sgate = (a.mis0_shyb < a.mis0_prev).to_numpy()
        starts["sgate"] = (np.where(sgate, a.shyb1, a.prev1), np.where(sgate, a.it_shyb, a.it_prev), np.where(sgate, a.conv_shyb, a.conv_prev).astype(bool), float(sgate.mean()),
                           np.where(sgate, a.mis1_shyb, a.mis1_prev).astype(float), np.where(sgate, tau["shyb"], tau["prev"]).astype(float), f"chosen start: prev {tau['prev']} or shyb {tau['shyb']}")
    lam = pol["shrinkage"]["lambda"].get(f"goc{grid}_{ckpt}", {})
    rows = []
    for name, (pred, it, conv, share, mis, tau_s, tau_label) in starts.items():
        for c in pol["loading_cushion_c"]:
            flag_loading = pred > c
            flag_full = flag_loading | ~np.isfinite(mis) | (mis > tau_s)  # non-finite / missing mis1 -> referred
            sec_loading = int((~real & flag_loading).sum())
            for policy, flag in (("loading_only", flag_loading), ("full_policy", flag_full)):
                missed = real & ~flag
                sec = int((~real & flag).sum())
                rows.append({"grid": int(grid), "checkpoint": ckpt, "start": name, "c": c, "policy": policy, "states": n, "critical_states": n_crit, "secure_states": n_sec,
                             "missed_critical_states": int(missed.sum()), "overloaded_lines": n_real_lines, "missed_overloaded_lines": int(a.n_real.to_numpy()[missed].sum()),
                             "secure_flagged": sec, "secure_flagged_pct": 100 * float(sec / max(n_sec, 1)),
                             "flagged_pct": 100 * float(flag.mean()), "newton_converged": int(conv.sum()), "mean_newton_iters": float(it[conv].mean()) if conv.any() else np.nan,
                             "genco_start_share_pct": 100 * share if share is not None else np.nan, "excluded_missing_truth": missing_truth,
                             "tau_mw": tau_label if policy == "full_policy" else "", "added_referrals": sec - sec_loading if policy == "full_policy" else np.nan,
                             "lam_va": lam.get("va", np.nan) if name in ("shyb1", "sgate") else (1.0 if name in ("hyb1", "gate") else np.nan),
                             "lam_vm": lam.get("vm", np.nan) if name in ("shyb1", "sgate") else (1.0 if name in ("hyb1", "gate") else np.nan)})
    if not has_shyb:
        notes.append(f"goc{grid} {ckpt}: no shyb columns in {lh.relative_to(ROOT)} (run without --policy); shyb1 / sgate not scored")
    return pd.DataFrame(rows), []


def markdown(df, full_policy=False):
    cols = ["checkpoint", "start", "c", "states", "critical_states", "missed_critical_states", "overloaded_lines", "missed_overloaded_lines",
            "secure_states", "secure_flagged", "secure_flagged_pct", "flagged_pct", "genco_start_share_pct", "mean_newton_iters", "newton_converged"]
    if full_policy:
        cols = cols[:2] + ["tau_mw"] + cols[2:] + ["added_referrals"]
    text_cols = ("checkpoint", "start", "tau_mw")
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---:" if c not in text_cols else ":---" for c in cols) + "|\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "c":
                cells.append(f"{v:.4f}")
            elif c == "added_referrals":
                cells.append("" if pd.isna(v) else f"{int(v):+d}")
            elif isinstance(v, float) and not float(v).is_integer() or c in ("secure_flagged_pct", "flagged_pct", "genco_start_share_pct", "mean_newton_iters"):
                cells.append("" if pd.isna(v) else f"{v:.2f}")
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def verify_against_recheck(res, path):
    """Compare both decision rules with the independent re-analysis CSV; return the comparison text; raise on any difference."""
    rev = pd.read_csv(path)
    rev["grid"] = rev.grid.astype(int)
    keys = ["grid", "checkpoint", "start", "c"]
    for d in (rev, res):
        d["c"] = d.c.astype(float).round(6)
    lo = res[res.policy == "loading_only"].set_index(keys)[["missed_critical_states", "secure_flagged"]].rename(columns={"missed_critical_states": "load_only_missed", "secure_flagged": "load_only_secure_flags"})
    fu = res[res.policy == "full_policy"].set_index(keys)[["missed_critical_states", "secure_flagged", "added_referrals", "secure_states"]].rename(columns={"missed_critical_states": "tau_missed", "secure_flagged": "tau_secure_flags"})
    ours = lo.join(fu, how="inner")
    cmp = rev.set_index(keys).join(ours, how="inner", lsuffix="_rev", rsuffix="_ours")
    missing = rev.set_index(keys).index.difference(ours.index)
    checked = ["load_only_missed", "load_only_secure_flags", "tau_missed", "tau_secure_flags", "secure_states"]
    bad = pd.Series(False, index=cmp.index)
    for col in checked:
        bad |= cmp[f"{col}_rev"].to_numpy() != cmp[f"{col}_ours"].to_numpy()
    # added_flags in the re-analysis counts all states (secure and critical) newly referred; ours counts secure states only; compare only when they must agree
    show = cmp[[f"{c}_{s}" for c in ("tau_missed", "tau_secure_flags") for s in ("rev", "ours")] + ["load_only_missed_rev", "load_only_missed_ours", "load_only_secure_flags_rev", "load_only_secure_flags_ours"]].copy()
    show["ok"] = np.where(bad, "DIFF", "ok")
    text = f"Verification against {path.relative_to(ROOT)}: {len(cmp)} rows compared ({len(rev)} in the re-analysis CSV, {len(missing)} not scored in this run), " \
           f"columns {checked}: {int((~bad).sum())} match, {int(bad.sum())} differ\n" + show.reset_index().to_string(index=False) + "\n"
    if bad.any() or (len(missing) and len(cmp) == 0):
        print(text)
        sys.exit("score_fresh_hybrid.py: FULL-POLICY RESULTS DIFFER FROM THE INDEPENDENT RE-ANALYSIS; see the comparison above")
    return text


def summary_table(res, c=1.001):
    s = res[np.isclose(res.c, c)]
    lo = s[s.policy == "loading_only"].set_index(["grid", "checkpoint", "start"])
    fu = s[s.policy == "full_policy"].set_index(["grid", "checkpoint", "start"])
    lines = [f"Summary at c = {c} (missed critical states / secure states sent to AC in % of secure states)",
             "| grid | checkpoint | start | loading-only missed | loading-only secure % | full-policy missed | full-policy secure % | added secure referrals |",
             "|---:|:---|:---|---:|---:|---:|---:|---:|"]
    for k, r in lo.iterrows():
        f = fu.loc[k]
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {r.missed_critical_states} | {r.secure_flagged_pct:.2f} ({r.secure_flagged}) | "
                     f"{f.missed_critical_states} | {f.secure_flagged_pct:.2f} ({f.secure_flagged}) | {int(f.added_referrals):+d} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--tag", default="_fresh", help="file tag of the fresh run outputs (default _fresh)")
    p.add_argument("--grid", nargs="*", default=None, help="score only these grids (default: every grid in the policy); the output file gets a _goc<N> suffix")
    p.add_argument("--existing-dev-data", action="store_true",
                   help="DEVELOPMENT ONLY: score the existing (non-fresh, tag-less) evaluation files to check the scorer against physics_pilots")
    p.add_argument("--recheck-csv", default=str(RECHECK_CSV), help="independent re-analysis to verify the fresh full-policy rows against (skipped if absent or with --existing-dev-data)")
    a = p.parse_args()
    pol = json.load(open(a.policy))
    tag = "" if a.existing_dev_data else a.tag
    if tag == "" and not a.existing_dev_data:
        sys.exit("score_fresh_hybrid.py: an empty --tag would score the original files; use --existing-dev-data explicitly for that")
    out = ROOT / "work" / "fresh" / (("results_hybrid_existing_dev" if a.existing_dev_data else "results_hybrid") + ("" if not a.grid else "_goc" + "_".join(a.grid)))
    header = (f"# Hybrid / shrunk start and mismatch gate, policy {a.policy}\n\n")
    if a.existing_dev_data:
        header += ("**DEVELOPMENT ONLY - scored on the EXISTING evaluation files (work/goc<N>/*_tiny_seed0.parquet etc., the data the policy was "
                   "calibrated on), NOT on fresh load scenarios. These numbers are a scorer check against the independent re-analysis and "
                   "must not be reported as fresh results.**\n\n")
    else:
        header += f"Fresh load scenarios: files work/goc<N>/*{tag}.parquet only.\n\n"
    header += ("Truth: a line above 1.001 of rating after full AC. Columns: missed_critical_states = critical states cleared as secure; "
               "missed_overloaded_lines = overloaded lines in those states; secure_flagged = secure states sent to AC (and % of secure states); "
               "flagged_pct = share of all states sent to AC; genco_start_share_pct = outages where the gate picked the GENCO-anchored start; "
               "mean_newton_iters = full Newton iterations from that start (converged runs). gate = prev1 | hyb1 and sgate = prev1 | shyb1 by "
               f"smaller mis0 (tie or NaN -> prev). Cushions c = {pol['loading_cushion_c']}.\n\n"
               f"Two tables per grid. Table A: {LABEL_LOADING}. These are the original tables, unchanged. "
               f"Table B: {LABEL_FULL}. tau (MW) = {pol['tau_mw']}; the post-step mismatch is the stored mis1_prev / mis1_hyb / mis1_shyb; "
               "for gate / sgate the tau of the start chosen per state applies; a non-finite or missing mis1 counts as above tau (referred); "
               "added_referrals = secure referrals under the full policy minus loading-only (same c, same start). "
               "Audit finding: the original scorer decided on loading only, although "
               "policy_frozen.json specifies one step with the chosen start's tau.\n\n")
    tables, notes, missing_all = [], [], []
    for grid, ckpts in pol["checkpoints"].items():
        if a.grid and str(grid) not in a.grid:
            continue
        for ckpt in ckpts:
            df, missing = score(grid, ckpt, tag, pol, notes)
            if missing:
                missing_all += missing
                continue
            tables.append(df)
    if missing_all:
        msg = "score_fresh_hybrid.py: refusing to score, missing files:\n  " + "\n  ".join(missing_all)
        if not a.existing_dev_data:
            sys.exit(msg + "\nRun work/fresh/run_fresh.sh (analysis/analysis_prev_control.py / analysis/analysis_hybrid_start.py with --tag _fresh and GOC_DATA) first.")
        notes.append(msg)
    if not tables:
        sys.exit("score_fresh_hybrid.py: nothing scored")
    res = pd.concat(tables, ignore_index=True)
    text = header
    for grid, df in res.groupby("grid", sort=False):
        lam = {k: v for k, v in pol["shrinkage"]["lambda"].items() if k.startswith(f"goc{grid}_")}
        text += f"## GOC {grid}\n\nshrinkage lambda (va, vm): " + ", ".join(f"{k.split('_', 1)[1]} ({v['va']}, {v['vm']})" for k, v in lam.items()) + "\n\n"
        text += f"### GOC {grid} A: {LABEL_LOADING}\n\n" + markdown(df[df.policy == "loading_only"]) + "\n"
        text += f"### GOC {grid} B: {LABEL_FULL}\n\n" + markdown(df[df.policy == "full_policy"], full_policy=True) + "\n"
    text += "## " + summary_table(res) + "\n"
    if notes:
        text += "Notes:\n" + "".join(f"- {n}\n" for n in notes)
    verification = ""
    if not a.existing_dev_data and tag == "_fresh" and Path(a.recheck_csv).exists():
        verification = verify_against_recheck(res.copy(), Path(a.recheck_csv))
        text += "\n```\n" + verification + "```\n"
    print(text)
    out.with_suffix(".md").write_text(text)
    res.to_csv(out.with_suffix(".csv"), index=False)
    print(f"saved {out.with_suffix('.md')} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

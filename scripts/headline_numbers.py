#!/usr/bin/env python
"""Registered numerical regression check of the write-up (stdlib + pandas only).

Each `claim(...)` call below registers one number of submission/POST.md together with the value the write-up
stated at the time of registration (the `post` argument), and recomputes or reads that number from the summary
tables in results/. The script does not parse the post: it compares the registered value with the table value,
so a number edited in the post without re-registering it here is not caught. The claims cover the result
sections of the post; results/README.md gives the count and lists the sections outside the coverage.

    python scripts/headline_numbers.py                  print one row per claim, write results/headline_numbers.csv
    python scripts/headline_numbers.py --check          exit 1 when (a) a table value differs from its registered
                                                        write-up value, (b) a claim is MISSING or a section FAILED,
                                                        or (c) a value differs from results/headline_numbers_expected.csv;
                                                        prints which of the three failed (what CI runs)
    python scripts/headline_numbers.py --write-expected refresh results/headline_numbers_expected.csv after a
                                                        deliberate change of a result table (then re-register the
                                                        changed write-up values here as well)

Columns of the output
    claim        short id, grouped by the section of the post
    post         the registered write-up value (a range "a–b" when the write-up gives one)
    value        the number from results/, full precision
    shown        value formatted the way the write-up rounds it
    matches_post shown == post (or inside the range)
    method       recomputed  from per-case, per-base or per-split rows, or from several cells by the formula in `note`
                 read        one cell of a summary table that an analysis script wrote
                 derived     one cell of results/derived/*.csv, which scripts/derive_results.py computes from the
                             gitignored parquet tables (see results/README.md); read here, recomputable only locally
                 MISSING     the table holds no value for the number (an empty cell)
                 FAILED      the section's function raised (a table is absent, or a row filter matched no row);
                             one row per failed section, named <section>_FAILED, in place of its claims
    source       the file under results/ (several, separated by " + ")
    note         the row filter and the formula
"""
import argparse
import csv
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "headline_numbers.csv"
EXPECTED = RES / "headline_numbers_expected.csv"

ROWS = []


@lru_cache(maxsize=None)
def table(rel):
    return pd.read_csv(RES / rel)


@lru_cache(maxsize=None)
def jsonfile(rel):
    return json.loads((RES / rel).read_text())


def one(df, **sel):
    """The single row of df matching every column==value in sel."""
    m = pd.Series(True, index=df.index)
    for k, v in sel.items():
        m &= df[k] == v
    if m.sum() != 1:
        raise KeyError(f"{m.sum()} rows match {sel}")
    return df[m].iloc[0]


# ----------------------------------------------------------------------------------------------- formats
def pct(d):
    return lambda v: f"{100 * v:.{d}f}%"


def fixed(d):
    return lambda v: f"{v:.{d}f}"


def integer(v):
    return f"{int(round(v)):,}"


def claim(cid, post, value, fmt, method, source, note=""):
    missing = value is None or (isinstance(value, float) and math.isnan(value))
    shown = "MISSING" if missing else fmt(value)
    if missing:
        ok = False
    elif "–" in post:  # a range the post states, e.g. "3.0–3.2"
        lo, hi = (float(x.rstrip("%")) for x in post.split("–"))
        ok = lo <= float(shown.rstrip("%").replace(",", "")) <= hi
    else:
        ok = shown == post
    ROWS.append({"claim": cid, "post": post, "value": "MISSING" if missing else f"{value:.8g}", "shown": shown,
                 "matches_post": ok, "method": "MISSING" if missing else method, "source": source, "note": note})


# ----------------------------------------------------------------------------------------------- 1. main result (up to five outages)
def hybrid_start():
    zs = "zeroshot/summary.csv"
    z = table(zs)
    z = z[(z.network == "case118_ieee") & (z.kind == "screen")]
    claim("main_118_overloads", "6,986", one(z, name="prev1").overloads, integer, "read", zs, "network==case118_ieee kind==screen overloads")
    for start, name in (("prev", "prev1"), ("genco", "pg1"), ("anch", "hybpg1")):
        r = one(z, name=name)
        claim(f"main_118_{start}_missed", {"prev": "74", "genco": "4", "anch": "2"}[start], r.overloads_missed, integer, "read", zs, f"name=={name} overloads_missed")
        claim(f"main_118_{start}_flagged", {"prev": "1.9%", "genco": "0.0%", "anch": "0.0%"}[start], r.secure_flagged, pct(1), "read", zs, f"name=={name} secure_flagged")
    post = {500: {"overloads": "2,621", "prev": ("41", "0.2%"), "genco": ("9", "0.1%"), "anch": ("5", "0.0%")},
            2000: {"overloads": "369", "prev": ("2", "0.0%"), "genco": ("1", "0.6%"), "anch": ("0", "0.0%")},
            10000: {"overloads": "507", "prev": ("2", "0.3%"), "genco": ("7", "37.0%"), "anch": ("5", "3.2%")}}
    hs = "hybrid_start/hybrid_control_summary.csv"
    h = table(hs)
    for grid, p in post.items():
        net = f"case{grid}_goc"
        claim(f"main_{grid}_overloads", p["overloads"], one(h, network=net, predictor="hyb1").overloads, integer, "read", hs, f"network=={net} overloads")
        for start, pred in (("prev", "prev1"), ("genco", "genco1"), ("anch", "hyb1")):
            r = one(h, network=net, predictor=pred)
            claim(f"main_{grid}_{start}_missed", p[start][0], r.overloads_missed, integer, "read", hs, f"network=={net} predictor=={pred} overloads_missed")
            claim(f"main_{grid}_{start}_flagged", p[start][1], r.secure_flagged, pct(1), "read", hs, f"network=={net} predictor=={pred} secure_flagged")
    sm = "hybrid_start/hybrid_control_summary_small_seed0.csv"
    s = table(sm)
    for start, pred, p in (("prev", "prev1", ("2", "0.3%")), ("genco", "genco1", ("6", "28.2%")), ("anch", "hyb1", ("2", "2.1%"))):
        r = one(s, predictor=pred)
        claim(f"main_10000small_{start}_missed", p[0], r.overloads_missed, integer, "read", sm, f"predictor=={pred} overloads_missed")
        claim(f"main_10000small_{start}_flagged", p[1], r.secure_flagged, pct(1), "read", sm, f"predictor=={pred} secure_flagged")
    # no Newton step: the anchored change alone misses 14–44% of overloaded lines
    for grid in (500, 2000, 10000):
        r = one(h, network=f"case{grid}_goc", predictor="hyb0")
        claim(f"nostep_{grid}_missed_share", "14%–44%", r.missed_share, pct(0), "read", hs, f"network==case{grid}_goc predictor==hyb0 missed_share")


# ----------------------------------------------------------------------------------------------- 2. 10,000 buses, every checkpoint
def checkpoints():
    files = {"tiny_seed0": "hybrid_start/hybrid_control_summary.csv", "tiny_seed1": "hybrid_start/hybrid_control_summary_tiny_seed1.csv",
             "small_seed0": "hybrid_start/hybrid_control_summary_small_seed0.csv", "small_seed1": "hybrid_start/hybrid_control_summary_small_seed1.csv"}
    post = {"tiny_seed0": ("5", "3.2%", "0", "0.1%", "28%", "75.4%", "178"), "tiny_seed1": ("3", "2.9%", "1", "0.0%", "31%", "77.3%", "197"),
            "small_seed0": ("2", "2.1%", "0", "0.4%", "35%", "79.2%", "395"), "small_seed1": ("3", "1.1%", "0", "0.3%", "39%", "81.4%", "402")}
    t = table(files["tiny_seed0"])
    t = t[t.network == "case10000_goc"]
    claim("ckpt_scenarios", "993", one(t, predictor="hyb1").scenarios, integer, "read", files["tiny_seed0"], "network==case10000_goc scenarios")
    for step, p in (("prev1", ("2", "0.3%")), ("prev2", ("0", "0.0%"))):
        r = one(t, predictor=step)
        claim(f"ckpt_{step}_missed", p[0], r.overloads_missed, integer, "read", files["tiny_seed0"], f"network==case10000_goc predictor=={step} overloads_missed")
        claim(f"ckpt_{step}_flagged", p[1], r.secure_flagged, pct(1), "read", files["tiny_seed0"], f"network==case10000_goc predictor=={step} secure_flagged")
    mech = "derived/change_mechanism.csv"
    for ck, p in post.items():
        d = table(files[ck])
        d = d[d.network == "case10000_goc"]
        for i, step in enumerate(("hyb1", "hyb2")):
            r = one(d, predictor=step)
            claim(f"ckpt_{ck}_{step}_missed", p[2 * i], r.overloads_missed, integer, "read", files[ck], f"predictor=={step} overloads_missed")
            claim(f"ckpt_{ck}_{step}_flagged", p[2 * i + 1], r.secure_flagged, pct(1), "read", files[ck], f"predictor=={step} secure_flagged")
        m = one(table(mech), grid=10000, checkpoint=ck)
        claim(f"ckpt_{ck}_nochange_removed", p[4], m.removed_share, pct(0), "derived", mech, f"grid==10000 checkpoint=={ck} removed_share = 1 - change_err_median_pp / nochange_err_median_pp")
        claim(f"ckpt_{ck}_sign_agreement", p[5], m.sign_agreement, pct(1), "derived", mech, f"grid==10000 checkpoint=={ck} sign_agreement")
        claim(f"ckpt_{ck}_genco_ms", p[6], m.genco_ms_per_state_median, fixed(0), "derived", mech, f"grid==10000 checkpoint=={ck} genco_ms_per_state_median")


# ----------------------------------------------------------------------------------------------- 3. mechanism, mismatch, iterations
def mechanism():
    mech = "derived/change_mechanism.csv"
    for grid, p in ((500, ("2.44", "0.36", "85%")), (2000, ("2.17", "0.56", "74%"))):
        m = one(table(mech), grid=grid, checkpoint="tiny_seed0")
        claim(f"mech_{grid}_nochange_err", p[0], m.nochange_err_median_pp, fixed(2), "derived", mech, f"grid=={grid} nochange_err_median_pp")
        claim(f"mech_{grid}_change_err", p[1], m.change_err_median_pp, fixed(2), "derived", mech, f"grid=={grid} change_err_median_pp")
        claim(f"mech_{grid}_removed", p[2], m.removed_share, pct(0), "derived", mech, f"grid=={grid} removed_share")
    for grid in (500, 2000):
        f = f"finetune/summary_hybrid_{grid}.csv"
        r = one(table(f), grid=grid, subset="all", model="released", predictor="hyb1")
        claim(f"mech_{grid}_sign_agreement", "93%–97%", r.chg_sign, pct(0), "read", f, "subset==all model==released chg_sign")
        claim(f"mismatch_{grid}_anch_mw", "6", r.mis0_hyb, fixed(0), "read", f, "subset==all model==released mis0_hyb (median start mismatch, MW)")
        claim(f"mismatch_{grid}_prev_mw", "72–118", r.mis0_prev, fixed(0), "read", f, "mis0_prev")
        claim(f"mismatch_{grid}_genco_mw", "246–435", r.mis0_genco, fixed(0), "read", f, "mis0_genco")
        claim(f"iterations_{grid}_anch", "2.4–2.5", r.it_hyb, fixed(1), "read", f, "it_hyb (mean Newton iterations to convergence)")
        claim(f"iterations_{grid}_prev", "3.0–3.2", r.it_prev, fixed(1), "read", f, "it_prev")
        claim(f"iterations_{grid}_genco", "3.0–3.2", r.it_genco, fixed(1), "read", f, "it_genco")
    ex = "derived/goc500_example_rule.csv"
    r = one(table(ex), surrogate="genco", rule="F3, t tuned on test")
    claim("example_recall", "93.9%", r.recall, pct(1), "derived", ex, "surrogate==genco rule=='F3, t tuned on test' recall")
    claim("example_flagged", "32.7%", r.flagged_share, pct(1), "derived", ex, "flagged_share")


# ----------------------------------------------------------------------------------------------- 4. full N-1 on GOC 2000
def full_n1():
    f, tj = "fulln1/summary.csv", "fulln1/timing_2000.json"
    s = table(f)
    w = s[(s.kind == "workflow")]
    test, allb = w[w.bases == "test"], w[w.bases == "all"]
    claim("n1_outages_total", "76,512", one(allb, item="W0").contingencies, integer, "read", f, "kind==workflow bases==all item==W0 contingencies")
    claim("n1_outages_scored", "57,384", one(test, item="W0").contingencies, integer, "read", f, "bases==test item==W0 contingencies")
    claim("n1_critical_scored", "1,451", one(test, item="W0").critical, integer, "read", f, "bases==test item==W0 critical")
    t = jsonfile(tj)
    claim("n1_ms_per_iteration", "3.9", t["ms_iter_full"], fixed(1), "read", tj, "ms_iter_full")
    claim("n1_ms_forward_b16", "12.6", t["ms_forward_mps_b16"], fixed(1), "read", tj, "ms_forward_mps_b16")
    claim("n1_ms_prep", "2.5", t["ms_prep"], fixed(1), "read", tj, "ms_prep")
    post = {"W0": ("100%", "0", "100%", "11.5"), "W1": ("29.6%", "5", "57%", "6.5"), "W2": ("6.6%", "780", "7%", "15.9"),
            "W3": ("45.4%", "1", "76%", "23.8"), "W4": ("2.7%", "14", "37%", "19.4"), "W5": ("2.1%", "303", "3%", "2.3")}
    it0 = one(test, item="W0").newton_iterations
    for wf, p in post.items():
        r = one(test, item=wf)
        claim(f"n1_{wf}_sent_to_ac", p[0], r.sent_to_ac / r.contingencies, pct(0 if wf == "W0" else 1), "recomputed", f, f"bases==test item=={wf} sent_to_ac / contingencies")
        claim(f"n1_{wf}_missed", p[1], r.missed_contingencies, integer, "read", f, f"item=={wf} missed_contingencies")
        claim(f"n1_{wf}_iterations_vs_w0", p[2], r.newton_iterations / it0, pct(0), "recomputed", f, f"item=={wf} newton_iterations / W0 newton_iterations")
        secs = (r.newton_iterations * t["ms_iter_full"] + r.dc_solves * t["ms_dc_full"] + r.genco_inferences * (t["ms_forward_mps_b16"] + t["ms_prep"])) / r.contingencies
        claim(f"n1_{wf}_s_per_1000", p[3], secs, fixed(1), "recomputed", f + " + " + tj,
              "(newton_iterations*ms_iter_full + dc_solves*ms_dc_full + genco_inferences*(ms_forward_mps_b16+ms_prep)) / contingencies")
    claim("n1_W2_missed_share", "54%", one(test, item="W2").missed_contingencies / one(test, item="W2").critical, pct(0), "recomputed", f, "item==W2 missed_contingencies / critical")
    claim("n1_W5_missed_share", "21%", one(test, item="W5").missed_contingencies / one(test, item="W5").critical, pct(0), "recomputed", f, "item==W5 missed_contingencies / critical")
    claim("n1_W1_tau_mw", "0.5", one(test, item="W1").tau_mw, fixed(1), "read", f, "item==W1 tau_mw")
    claim("n1_W4_tau_mw", "5", one(test, item="W4").tau_mw, fixed(0), "read", f, "item==W4 tau_mw")
    acc = s[s.kind == "n1_accuracy"]
    claim("n1_p99_err_anch", "0.006", one(acc, item="hyb1").err_p99_pp, fixed(3), "read", f, "kind==n1_accuracy item==hyb1 err_p99_pp")
    claim("n1_p99_err_prev", "0.032", one(acc, item="prev1").err_p99_pp, fixed(3), "read", f, "kind==n1_accuracy item==prev1 err_p99_pp")
    w1, w4, w0 = one(test, item="W1"), one(test, item="W4"), one(test, item="W0")
    claim("n1_saving_vs_classic_ms", "2.2", (w1.newton_iterations - w4.newton_iterations) / w4.contingencies * t["ms_iter_full"], fixed(1), "recomputed", f + " + " + tj, "(W1 - W4 newton_iterations) / contingencies * ms_iter_full")
    be = (w0.newton_iterations - w4.newton_iterations) / w4.contingencies * t["ms_iter_full"] - t["ms_prep"]
    claim("n1_breakeven_forward_ms", "4.8", be, fixed(1), "recomputed", f + " + " + tj, "(W0 - W4 newton_iterations) / contingencies * ms_iter_full - ms_prep")
    claim("n1_breakeven_forward_plus_prep_ms", "7.2", be + t["ms_prep"], fixed(1), "recomputed", f + " + " + tj, "previous + ms_prep")
    d = "derived/fulln1_w4_missed.csv"
    m = table(d)
    claim("n1_W4_missed_rows", "14", len(m), integer, "derived", d, "rows (test bases, W4_missed)")
    claim("n1_W4_missed_below_10014", "13", (m.worst_loading <= 1.0014).sum(), integer, "derived", d, "worst_loading <= 1.0014")
    claim("n1_W4_missed_median_mismatch_mw", "0.5", m.mis1_hyb.median(), fixed(1), "derived", d, "median mis1_hyb")
    # inside PowerModels' AC power flow
    tf = "timing/summary.csv"
    pm = table(tf)
    pm = pm[pm.kind == "PowerModels solve"]
    for net, p in (("case2000_goc", {"tiny_qc": "200", "default": "199", "flat_q0": "20", "dc_qc": "46"}), ("case10000_goc", {"tiny_qc": "100", "default": "92", "flat_q0": "0", "dc_qc": "0"})):
        for item, v in p.items():
            claim(f"pm_{net.split('_')[0]}_{item}_same_solution", v, one(pm, network=net, item=item).same_solution, integer, "read", tf, f"kind=='PowerModels solve' network=={net} item=={item} same_solution")


# ----------------------------------------------------------------------------------------------- 5. fresh pool
def fresh():
    pb = "fresh/results_n1_per_base.csv"
    p = table(pb)
    p = p[p.c == 1.001]
    for wf, name, p_missed, p_share in (("W1", "classic", "2", "27.94%"), ("W4", "anch", "1", "2.17%")):
        g = p[p.policy == wf]
        claim(f"fresh_{name}_outages", "76,512", g.contingencies.sum(), integer, "recomputed", pb, f"policy=={wf} c==1.001 sum contingencies over 24 bases")
        claim(f"fresh_{name}_critical", "1,510", g.critical.sum(), integer, "recomputed", pb, "sum critical")
        claim(f"fresh_{name}_missed", p_missed, g.missed.sum(), integer, "recomputed", pb, "sum missed")
        claim(f"fresh_{name}_sent_to_ac", p_share, g.sent_to_ac.sum() / g.contingencies.sum(), pct(2), "recomputed", pb, "sum sent_to_ac / sum contingencies")
        claim(f"fresh_{name}_missed_registered6", "0", g[g.registered == "R"].missed.sum(), integer, "recomputed", pb, "registered==R sum missed")
    ej = "fulln1/e2e_2000_fresh.json"
    e = jsonfile(ej)
    summ = {(r["workflow"], r["gate"]): r for r in e["summary"]}
    for wf, name, ms in (("W0", "fullac", "13.7"), ("W1", "classic", "8.4"), ("W4_mps16", "anch", "22.3")):
        r = summ[(wf, "all")]
        claim(f"fresh_timing_{name}_n", "4,800", r["n"], integer, "read", ej, f"summary workflow=={wf} gate==all n")
        claim(f"fresh_timing_{name}_critical", "69", r["critical"], integer, "read", ej, "critical")
        claim(f"fresh_timing_{name}_missed", "0", r["missed"], integer, "read", ej, "missed")
        claim(f"fresh_timing_{name}_ms_per_outage", ms, r["ms_total_first"] / r["n"], fixed(1), "recomputed", ej, "ms_total_first / n")
    claim("fresh_timing_bases", "24", len(e["bases"]), integer, "read", ej, "len(bases)")
    it = {wf: summ[(wf, "all")]["iterations_mean"] for wf in ("W0", "W1", "W4")}
    claim("fresh_iterations_anch", "1.08", it["W4"], fixed(2), "read", ej, "summary W4 gate==all iterations_mean")
    claim("fresh_iterations_fullac", "2.92", it["W0"], fixed(2), "read", ej, "W0 iterations_mean")
    claim("fresh_iterations_classic", "1.61", it["W1"], fixed(2), "read", ej, "W1 iterations_mean")
    tj = "fulln1/timing_2000_fresh.json"
    t = jsonfile(tj)
    claim("fresh_ms_per_iteration", "3.81", t["ms_iter_full"], fixed(2), "read", tj, "ms_iter_full")
    claim("fresh_ms_forward_b16", "12.74", t["ms_forward_mps_b16"], fixed(2), "read", tj, "ms_forward_mps_b16")
    claim("fresh_ms_prep", "2.40", t["ms_prep"], fixed(2), "read", tj, "ms_prep")
    pre = sum(b["ms_cpu_b1"] for b in e["base_inference"]) / sum(b["n_sample"] for b in e["base_inference"])
    claim("fresh_ms_preoutage_share", "0.18", pre, fixed(2), "recomputed", ej, "sum base_inference ms_cpu_b1 / sum n_sample")
    claim("fresh_saved_iterations_vs_fullac", "1.84", it["W0"] - it["W4"], fixed(2), "recomputed", ej, "W0 - W4 iterations_mean")
    claim("fresh_saved_ms_vs_fullac", "7.0", (it["W0"] - it["W4"]) * t["ms_iter_full"], fixed(1), "recomputed", ej + " + " + tj, "(W0 - W4 iterations_mean) * ms_iter_full")
    claim("fresh_saved_iterations_vs_classic", "0.53", it["W1"] - it["W4"], fixed(2), "recomputed", ej, "W1 - W4 iterations_mean")
    claim("fresh_saved_ms_vs_classic", "2.0", (it["W1"] - it["W4"]) * t["ms_iter_full"], fixed(1), "recomputed", ej + " + " + tj, "(W1 - W4 iterations_mean) * ms_iter_full")
    claim("fresh_breakeven_forward_ms", "4.4", (it["W0"] - it["W4"]) * t["ms_iter_full"] - t["ms_prep"] - pre, fixed(1), "recomputed", ej + " + " + tj, "saved ms vs full AC - ms_prep - pre-outage share")
    ti, ts = "timing/goc10000_timing.csv", "timing/goc10000_timing_small.csv"
    a = one(table(ti), item="GENCO forward", device="cpu", batch=16).ms_per_state
    b = one(table(ts), item="GENCO forward", device="cpu", batch=16).ms_per_state
    claim("fresh_small_vs_tiny_ms_ratio", "2.2", b / a, fixed(1), "recomputed", ts + " + " + ti, "item=='GENCO forward' device==cpu batch==16: small ms_per_state / tiny ms_per_state")
    # warm-start table on the fresh pool
    hf = "fresh/results_hybrid.csv"
    h = table(hf)
    h = h[h.c == 1.001]
    claim("freshws_2000_states", "2,496", one(h, grid=2000, checkpoint="tiny_seed0", start="hyb1", policy="loading_only").states, integer, "read", hf, "grid==2000 states")
    claim("freshws_2000_critical", "140", one(h, grid=2000, checkpoint="tiny_seed0", start="hyb1", policy="loading_only").critical_states, integer, "read", hf, "grid==2000 critical_states")
    claim("freshws_10000_states", "999", one(h, grid=10000, checkpoint="tiny_seed0", start="hyb1", policy="loading_only").states, integer, "read", hf, "grid==10000 states")
    claim("freshws_10000_critical", "231", one(h, grid=10000, checkpoint="tiny_seed0", start="hyb1", policy="loading_only").critical_states, integer, "read", hf, "grid==10000 critical_states")
    rows = [(2000, "tiny_seed0", "prev1", ("3", "0.0%", "0", "54.1%")), (2000, "tiny_seed0", "hyb1", ("0", "0.0%", "0", "0.8%")),
            (10000, "tiny_seed0", "prev1", ("0", "1.0%", "0", "60.5%")), (10000, "tiny_seed0", "hyb1", ("0", "2.5%", "0", "8.1%")),
            (10000, "tiny_seed1", "hyb1", ("2", "2.3%", "1", "8.6%")), (10000, "small_seed0", "hyb1", ("1", "2.3%", "1", "7.9%")),
            (10000, "small_seed1", "hyb1", ("4", "1.3%", "4", "4.7%"))]
    for grid, ck, start, p in rows:
        for i, pol in enumerate(("loading_only", "full_policy")):
            r = one(h, grid=grid, checkpoint=ck, start=start, policy=pol)
            claim(f"freshws_{grid}_{ck}_{start}_{pol}_missed", p[2 * i], r.missed_critical_states, integer, "read", hf, f"grid=={grid} checkpoint=={ck} start=={start} policy=={pol} c==1.001 missed_critical_states")
            claim(f"freshws_{grid}_{ck}_{start}_{pol}_flagged", p[2 * i + 1], r.secure_flagged / r.secure_states, pct(1), "recomputed", hf, "secure_flagged / secure_states")
    v = h[(h.grid == 10000) & (h.policy == "loading_only") & h.start.isin(["shyb1", "sgate"])]
    claim("freshws_variants_flagged_min", "0.4%", (v.secure_flagged / v.secure_states).min(), pct(1), "recomputed", hf, "grid==10000 policy==loading_only start in {shyb1,sgate}: min secure_flagged/secure_states")
    claim("freshws_variants_flagged_max", "0.9%", (v.secure_flagged / v.secure_states).max(), pct(1), "recomputed", hf, "same rows: max")
    claim("freshws_variants_missed_min", "0", v.missed_critical_states.min(), integer, "recomputed", hf, "same rows: min missed_critical_states")
    claim("freshws_variants_missed_max", "3", v.missed_critical_states.max(), integer, "recomputed", hf, "same rows: max missed_critical_states")
    pf = "fresh/policy_frozen.json"
    claim("fresh_tau_prev_mw", "0.5", jsonfile(pf)["tau_mw"]["prev"], fixed(1), "read", pf, "tau_mw.prev")
    claim("fresh_tau_hyb_mw", "5", jsonfile(pf)["tau_mw"]["hyb"], fixed(0), "read", pf, "tau_mw.hyb")


# ----------------------------------------------------------------------------------------------- 6. fine-tuning
def finetune():
    f = "finetune/summary_hybrid_10000.csv"
    for model, p in (("released", "5"), ("ft10000_seed0", "4")):
        r = one(table(f), grid=10000, subset="all", model=model, predictor="hyb1")
        claim(f"ft_10000_{model}_missed", p, r.missed, integer, "read", f, f"subset==all model=={model} predictor==hyb1 missed")
        claim(f"ft_10000_{model}_flagged", "3.2%", r.secure_flagged / r.secure, pct(1), "recomputed", f, "secure_flagged / secure")
    f = "finetune/summary_10000.csv"
    claim("ft_10000_released_raw_err", "0.54", one(table(f), model="released", subset="all").err_median, fixed(2), "read", f, "model==released subset==all err_median")
    claim("ft_10000_ft_raw_err", "0.56", one(table(f), model="ft10000_seed0", subset="all").err_median, fixed(2), "read", f, "model==ft10000_seed0 err_median")
    f = "finetune/summary_500.csv"
    claim("ft_500_released_raw_err", "0.33", one(table(f), model="released_seed0", subset="all").err_median, fixed(2), "read", f, "model==released_seed0 subset==all err_median")
    claim("ft_500_ft_raw_err", "0.23", one(table(f), model="ft_seed0", subset="all").err_median, fixed(2), "read", f, "model==ft_seed0 err_median")
    f = "finetune/summary_hybrid_500.csv"
    for model in ("released", "ft_seed0"):
        claim(f"ft_500_{model}_missed", "5", one(table(f), grid=500, subset="all", model=model, predictor="hyb1").missed, integer, "read", f, f"model=={model} predictor==hyb1 missed")
    d = "derived/finetune_scenario_counts.csv"
    for grid, p in ((500, "8,969"), (2000, "2,248"), (10000, "1,347")):
        claim(f"ft_{grid}_train_scenarios", p, one(table(d), grid=grid).train, integer, "derived", d, f"grid=={grid} train")


# ----------------------------------------------------------------------------------------------- 7. reactive limits
def qlimits():
    f = "qlim/summary.csv"
    q = table(f)
    sw = q[(q.network == "case500_goc") & (q.truth == "Q limits") & (q.variant == "switching")]
    for pred, p in (("hyb1s", "23"), ("genco1s", "28"), ("prev1s", "54")):
        claim(f"qlim_{pred}_missed", p, one(sw, predictor=pred).overloads_missed, integer, "read", f, f"network==case500_goc truth=='Q limits' variant==switching predictor=={pred} overloads_missed")
    claim("qlim_overloads", "2,570", one(sw, predictor="hyb1s").overloads, integer, "read", f, "overloads")
    claim("qlim_hyb1s_worst_missed", "111%", one(sw, predictor="hyb1s").worst_missed_loading, pct(0), "read", f, "predictor==hyb1s worst_missed_loading")
    claim("qlim_hyb2s_missed", "5", one(sw, predictor="hyb2s").overloads_missed, integer, "read", f, "predictor==hyb2s overloads_missed")
    claim("qlim_hyb2s_worst_missed", "100.5%", one(sw, predictor="hyb2s").worst_missed_loading, pct(1), "read", f, "predictor==hyb2s worst_missed_loading")
    nl = one(q, network="case500_goc", truth="Q limits", variant="no-limit AC", predictor="ac")
    claim("qlim_nolimit_ac_missed", "19", nl.overloads_missed, integer, "read", f, "variant=='no-limit AC' predictor==ac overloads_missed")
    claim("qlim_nolimit_ac_secure_flagged", "0.2%", nl.secure_flagged, pct(1), "read", f, "secure_flagged")
    j = "qlim/batch2_results.json"
    sweep = {s["cutoff"]: s for s in jsonfile(j)["qlimits"]["sweep"]}
    claim("qlim_cutoff100_missed", "9", sweep[1.0]["missed_readings_in_accepted_states"], integer, "read", j, "qlimits.sweep cutoff==1.0 missed_readings_in_accepted_states")
    claim("qlim_cutoff100_secure_sent", "4.9%", sweep[1.0]["secure_sent_percent"] / 100, pct(1), "read", j, "cutoff==1.0 secure_sent_percent")
    claim("qlim_cutoff95_missed", "0", sweep[0.95]["missed_readings_in_accepted_states"], integer, "read", j, "cutoff==0.95 missed_readings_in_accepted_states")
    claim("qlim_cutoff95_secure_sent", "23.5%", sweep[0.95]["secure_sent_percent"] / 100, pct(1), "read", j, "cutoff==0.95 secure_sent_percent")
    vo = q[(q.network == "case118_ieee") & (q.variant == "voltage")]
    claim("qlim_v118_detection_before", "98.8%", one(vo, truth="no limits", predictor="GENCO vs no-limit truth (before)").recall, pct(1), "read", f, "network==case118_ieee variant==voltage truth=='no limits' recall")
    claim("qlim_v118_detection_pq", "70%", one(vo, truth="Q limits", predictor="GENCO", buses="PQ").recall, pct(0), "read", f, "truth=='Q limits' predictor==GENCO buses==PQ recall")
    claim("qlim_v118_detection_pqpv", "52%", one(vo, truth="Q limits", predictor="GENCO", buses="PQ + PV").recall, pct(0), "read", f, "buses=='PQ + PV' recall")


# ----------------------------------------------------------------------------------------------- 8. conformal band, embedding, zero-shot
def conformal():
    d = "derived/conformal_case118_seed0.csv"
    s0 = table(d)
    r = one(s0, surrogate="genco_d", sigma="global", alpha=0.01)
    claim("conf_118_a1_ac_calls", "31.9%", r.ac_calls, pct(1), "derived", d, "seed 0, surrogate==genco_d sigma==global alpha==0.01 ac_calls")
    claim("conf_118_states_with_overload", "31.2%", r.states_with_real_overload_share, pct(1), "derived", d, "states_with_real_overload_share")
    claim("conf_118_a1_missed", "0", r.missed, integer, "derived", d, "missed")
    claim("conf_118_a1_real", "792", r.real, integer, "derived", d, "real (held-out overloads)")
    claim("conf_118_dc_secure_escalated", "81%", one(s0, surrogate="dc_d", sigma="shift", alpha=0.01).wasted, pct(0), "derived", d, "surrogate==dc_d sigma==shift alpha==0.01 wasted")
    f = "conformal/conformal_case118_ieee_base_splits.csv"
    sp = table(f)
    sp = sp[(sp.surrogate == "genco_d") & (sp.sigma == "global")]
    for a, p_line, p_state in ((0.10, "9.9%", "4.4%"), (0.05, "5.0%", "2.0%"), (0.02, "1.9%", "0.23%"), (0.01, "1.0%", "0.05%")):
        g = sp[sp.alpha == a]
        claim(f"conf_118_splits_a{int(100 * a)}_line_missed", p_line, g.line_missed_share.mean(), pct(1), "recomputed", f, f"surrogate==genco_d sigma==global alpha=={a}: mean line_missed_share over {len(g)} splits")
        claim(f"conf_118_splits_a{int(100 * a)}_state_missed", p_state, g.missed_share.mean(), pct(2 if a <= 0.02 else 1), "recomputed", f, "mean missed_share")
    f = "conformal/conformal_texas_n2x450_splits.csv"
    tx = table(f)
    claim("conf_texas_dc_ac_calls", "7.9%", tx[(tx.surrogate == "dc_d") & (tx.sigma == "shift") & (tx.alpha == 0.02)].ac_calls.mean(), pct(1), "recomputed", f, "surrogate==dc_d sigma==shift alpha==0.02 mean ac_calls")
    claim("conf_texas_genco_ac_calls", "7.7%", tx[(tx.surrogate == "genco") & (tx.sigma == "shift") & (tx.alpha == 0.02)].ac_calls.mean(), pct(1), "recomputed", f, "surrogate==genco sigma==shift alpha==0.02 mean ac_calls")
    f = "conformal/shift_case118_fixed.csv"
    for sh, p in (("n3", "6.5%"), ("n4", "7.1%")):
        claim(f"conf_118_{sh}_line_missed", p, one(table(f), shift=sh, surrogate="genco_d", sigma="shift", alpha=0.05).line_missed_share, pct(1), "read", f, f"shift=={sh} surrogate==genco_d sigma==shift alpha==0.05 line_missed_share")


def embedding():
    f = "embedding/embedding_auroc.csv"
    r = one(table(f), score="L11.mean|knn5")
    claim("emb_auroc_n3", "0.68", r.n3, fixed(2), "read", f, "score=='L11.mean|knn5' n3")
    claim("emb_auroc_n4", "0.79", r.n4, fixed(2), "read", f, "n4")
    claim("emb_auroc_nout_n3", "1.00", one(table(f), score="n_out").n3, fixed(2), "read", f, "score==n_out n3")
    claim("emb_auroc_nout_n4", "1.00", one(table(f), score="n_out").n4, fixed(2), "read", f, "score==n_out n4")


def zeroshot():
    f = "zeroshot/summary.csv"
    z = table(f)
    z = z[z.kind == "screen"]
    ratios = {net: one(z, network=net, name="zs").err_median_pp / one(z, network=net, name="dc").err_median_pp for net in ("case118_ieee", "case500_goc", "case1354_pegase", "case2000_goc")}
    claim("zs_err_ratio_min", "2–9", min(ratios.values()), fixed(1), "recomputed", f, "min over grids of err_median_pp[zs] / err_median_pp[dc] (kind==screen)")
    claim("zs_err_ratio_max", "2–9", max(ratios.values()), fixed(1), "recomputed", f, "max of the same")
    for net, p in (("case118_ieee", ("74", "19")), ("case500_goc", ("12", "3")), ("case1354_pegase", ("8", "5"))):
        claim(f"zs_{net.split('_')[0]}_prev1_missed", p[0], one(z, network=net, name="prev1").overloads_missed, integer, "read", f, f"network=={net} name==prev1 overloads_missed")
        claim(f"zs_{net.split('_')[0]}_hybzs1_missed", p[1], one(z, network=net, name="hybzs1").overloads_missed, integer, "read", f, f"network=={net} name==hybzs1 overloads_missed")


# ----------------------------------------------------------------------------------------------- main
SECTIONS = (hybrid_start, checkpoints, mechanism, full_n1, fresh, finetune, qlimits, conformal, embedding, zeroshot)
ABSENT = ("MISSING", "FAILED")  # `method` values of a claim without a table value


def compute():
    """Register every claim of every section; returns the rows (re-entrant: clears the caches and the row list)."""
    ROWS.clear()
    table.cache_clear()
    jsonfile.cache_clear()
    for fn in SECTIONS:
        try:
            fn()
        except Exception as exc:  # a missing table must not stop the other sections
            ROWS.append({"claim": f"{fn.__name__}_FAILED", "post": "", "value": "MISSING", "shown": "FAILED", "matches_post": False,
                         "method": "FAILED", "source": "", "note": f"{type(exc).__name__}: {exc}"})
    return list(ROWS)


def check(rows, expected=None):
    """The three gates of --check. Returns [(gate, [detail lines])] for every gate that failed; [] when all pass."""
    expected = EXPECTED if expected is None else expected
    failures = []
    wrong = [r for r in rows if r["method"] not in ABSENT and not r["matches_post"]]
    if wrong:
        failures.append(("registered value", [f"{r['claim']}: registered {r['post']}, table gives {r['shown']} ({r['source']})" for r in wrong]))
    absent = [r for r in rows if r["method"] in ABSENT]
    if absent:
        failures.append(("missing claim", [f"{r['claim']}: {r['method']} {r['note']}".rstrip() for r in absent]))
    if not expected.exists():
        failures.append(("expected snapshot", [f"no {expected}; run with --write-expected first"]))
    else:
        with expected.open(newline="") as fh:
            exp = {r["claim"]: r["value"] for r in csv.DictReader(fh)}
        got = {r["claim"]: r["value"] for r in rows}
        diffs = [f"{c}: expected {exp.get(c)}, got {got.get(c)}" for c in sorted(set(exp) | set(got)) if exp.get(c) != got.get(c)]
        if diffs:
            failures.append(("expected snapshot", diffs))
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when a table value differs from its registered write-up value, a claim is MISSING or a section "
                         "FAILED, or a value differs from results/headline_numbers_expected.csv; prints which gate(s) failed")
    ap.add_argument("--write-expected", action="store_true", help="overwrite results/headline_numbers_expected.csv")
    ap.add_argument("--quiet", action="store_true", help="print only mismatches and the summary")
    a = ap.parse_args(argv)
    rows = compute()
    w = max(len(r["claim"]) for r in rows)
    for r in rows:
        if a.quiet and r["matches_post"]:
            continue
        flag = "ok " if r["matches_post"] else "!! "
        print(f"{flag}{r['claim']:<{w}}  registered {r['post']:>10}  got {r['shown']:>10}  {r['method']:<10} {r['source']}")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    n_absent = sum(r["method"] in ABSENT for r in rows)
    n_wrong = sum(r["method"] not in ABSENT and not r["matches_post"] for r in rows)
    n_claims = len(rows) - sum(r["method"] == "FAILED" for r in rows)
    print(f"\n{n_claims} registered claims: {len(rows) - n_wrong - n_absent} equal their registered write-up value, {n_wrong} differ, "
          f"{n_absent} MISSING or FAILED (the post itself is not parsed); written {OUT.relative_to(ROOT)}")
    if a.write_expected:
        with EXPECTED.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["claim", "value"])
            wr.writerows((r["claim"], r["value"]) for r in rows)
        print(f"wrote {EXPECTED.relative_to(ROOT)}")
    if a.check:
        failures = check(rows)
        if failures:
            print(f"\nCHECK FAILED ({', '.join(g for g, _ in failures)}):")
            for gate, lines in failures:
                print(f"  [{gate}] {len(lines)} claim(s):")
                for line in lines:
                    print(f"    {line}")
            return 1
        print(f"check passed: {n_claims} claims equal their registered write-up value, none MISSING or FAILED, "
              f"all values equal {EXPECTED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

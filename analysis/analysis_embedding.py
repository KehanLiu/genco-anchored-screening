"""Can GENCO's own embedding space warn that a state is "unlike calibration"?

analysis/analysis_shift.py showed that the conformal trust layer (calibrated on N-1 / random N-2 states) loses its
promise under N-3 / N-4 outages and under reactive demand x0.7. Here the bus embeddings of the model are
turned into out-of-distribution scores and tested as that warning.

    extract   solve the calibration sweep (reference) and every shifted sweep of analysis/analysis_shift.py again, with a
              forward hook on the model (the input of the bus head `mlp_bus` at each of the 12 correction steps,
              i.e. the bus embedding after step i). The states are generated through analysis_shift's own
              functions with the same seeds, so state ids match work/shift_case118_*.parquet (checked).
    report    distance scores (Mahalanobis with Ledoit-Wolf shrinkage, kNN) on standardised, PCA-reduced pooled
              embeddings; (a) separation shifted vs control, (b) ranking of trust-layer failures, (c) a gate
              "escalate to AC above the reference quantile", (d) the same for cheap non-embedding scores.

    work/texas/tmo 1500 .venv/bin/python -u analysis/analysis_embedding.py extract
    work/texas/tmo 900  .venv/bin/python -u analysis/analysis_embedding.py report
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here
SETS = ["ref", "control", "n3", "n4", "load", "pf", "redispatch"]
LAYERS = [0, 5, 11]          # first, middle, last correction step
N_SETS, SEED = 240, 7        # analysis_shift defaults for case118
REF_DOUBLES = 300            # random N-2 sets per load level in work/hybrid_case118_ieee_base.parquet (2 of them island)
LINE_FILE = {"ref": WORK / "hybrid_case118_ieee_base.parquet",
             **{n: WORK / f"shift_case118_{n}.parquet" for n in SETS[1:]}}
SURR, SIGMA = "genco_d", "shift"     # the trust layer that is gated
GATE_Q = [0.95, 0.90]
BROKEN = ["n3", "n4", "pf Qd x0.7"]


def emb_path(name):
    return WORK / f"embedding_states_{name}.parquet"


# ---------------------------------------------------------------- extraction

def extract(names):
    import analysis_shift as S
    engine, eng = S.make_engine("case118")
    from analysis_residual import cases

    grabbed = []
    eng.genco.model.mlp_bus.register_forward_hook(lambda m, inp, out: grabbed.append(inp[0].detach().numpy().copy()))
    known = set(pd.read_parquet(S.CAL_FILE["case118"], columns=["outages"]).outages.unique())

    for name in names:
        if emb_path(name).exists():
            print(f"{emb_path(name).name} exists, skipped", flush=True)
            continue
        if name == "ref":   # exactly analysis_hybrid.collect: state id = position in cases(), failures leave gaps
            items = [(i, f"L{s}", s, 1.0, tuple(o), None, 0.0) for i, (s, o) in enumerate(cases(eng, REF_DOUBLES))]
        else:               # exactly analysis_shift.sweep: anchors first, state id counts the solved states
            rng = random.Random(f"{SEED}-{name}")
            tasks = S.tasks_for(name, "case118", eng, engine, rng, known, N_SETS, SEED)
            anchors = {}
            for t in tasks:
                anchors.setdefault(t[0], (t[0], t[1], t[2], (), t[4], t[5]))
            items = [(None,) + t for t in list(anchors.values()) + tasks]
        rows, anchor_h, dead, fails, counter, start = [], {}, set(), {}, 0, time.perf_counter()
        for sid, anchor, scale, q, outages, redispatch, moved in items:
            if anchor in dead:
                continue
            eng.set_q(q)
            grabbed.clear()
            try:
                result = eng.solve(scale, outages, redispatch=redispatch)
            except engine.SolveError as err:
                fails[err.reason] = fails.get(err.reason, 0) + 1
                if not outages:
                    dead.add(anchor)
                continue
            assert len(grabbed) == 12, len(grabbed)
            H = {L: grabbed[L] for L in LAYERS}
            if not outages:
                anchor_h[anchor] = H
            buses = pd.DataFrame(result["buses"])
            row = {"set": name, "state": counter if sid is None else sid, "load_scale": scale, "q_scale": q, "anchor": anchor,
                   "outages": "+".join(map(str, outages)), "n_out": len(outages), "moved_mw": moved,
                   "res_p": buses.res_p.abs().max(), "res_q": buses.res_q.abs().max()}
            for L in LAYERS:
                d = H[L] - anchor_h[anchor][L]
                row[f"e{L}_mean"], row[f"e{L}_max"] = H[L].mean(0), H[L].max(0)
                row[f"d{L}_absmax"] = np.abs(d).max(0)          # per feature: largest change at any bus vs the N-0 state
                row[f"bdn{L}"] = np.linalg.norm(d, axis=1)      # per bus: size of the change vs the N-0 state
            rows.append(row)
            counter += 1
            if counter % 500 == 0:
                print(f"  {name}: {counter} states, failures {fails}, {time.perf_counter() - start:.0f} s", flush=True)
        pd.DataFrame(rows).to_parquet(emb_path(name), index=False)
        print(f"{name}: {counter} states, failures {fails}, {time.perf_counter() - start:.0f} s -> {emb_path(name).name}", flush=True)


# ---------------------------------------------------------------- report

def stack(col):
    return np.stack(col.to_numpy()).astype(np.float64)


def auroc(score, positive):
    from scipy.stats import rankdata
    positive = np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def top_share(score, positive, frac):
    """Share of the positives among the top `frac` states by score (ties broken at random, fixed seed)."""
    positive = np.asarray(positive, bool)
    if not positive.any():
        return np.nan
    perm = np.random.default_rng(0).permutation(len(score))
    order = perm[np.argsort(-np.asarray(score)[perm], kind="stable")]
    return positive[order[: int(round(frac * len(score)))]].mean() * len(order[: int(round(frac * len(score)))]) / positive.sum()


class Detector:
    """standardise -> PCA (99% variance, at most 64 components) -> Mahalanobis (Ledoit-Wolf) and kNN distances."""

    def __init__(self, X):
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd = np.where(self.sd < 1e-6 * max(self.sd.max(), 1e-12), np.inf, self.sd)   # constant features carry nothing
        Z = (X - self.mu) / self.sd
        full = PCA().fit(Z)
        k = int(min(np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.99) + 1, 64, Z.shape[1]))
        self.pca = PCA(n_components=k).fit(Z)
        P = self.pca.transform(Z)
        self.lw = LedoitWolf().fit(P)
        self.nn = NearestNeighbors(n_neighbors=11).fit(P)
        self.k = k

    def score(self, X, train=False):
        P = self.pca.transform((X - self.mu) / self.sd)
        dist = self.nn.kneighbors(P)[0]
        off = 1 if train else 0   # a training point is its own nearest neighbour
        return {"maha": np.sqrt(self.lw.mahalanobis(P)), "knn5": dist[:, 4 + off], "knn10": dist[:, 9 + off]}


def report(out):
    import analysis_shift as S
    from analysis_conformal import OVERLOAD, sigma_of
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold

    emb = {n: pd.read_parquet(emb_path(n)) for n in SETS}
    lines = {n: pd.read_parquet(LINE_FILE[n]) for n in SETS}
    if "anchor" not in lines["ref"]:
        lines["ref"] = lines["ref"].assign(anchor="L" + lines["ref"].load_scale.astype(str))

    out("=== state matching with the saved sweeps (state id, outages, load, GENCO residual)")
    for n in SETS:
        a = emb[n].set_index("state")
        b = lines[n].drop_duplicates("state").set_index("state")
        same = len(a) == len(b) and a.index.equals(b.index) and (a.outages == b.outages).all() and (a.load_scale == b.load_scale).all()
        dres = float((a.res_p - b.residual.reindex(a.index)).abs().max())
        out(f"  {n:<11} {len(a):>5} states here, {len(b):>5} in {LINE_FILE[n].name}: ids/outages/load {'identical' if same else 'DIFFERENT'}, max |residual diff| {dres:.5f} MW")
        if not same:
            raise SystemExit("state mismatch")

    # ---- per-state tables: meta, cheap scores, feature matrices
    cal, _, _ = S.prep(lines["ref"])
    W = S.widths(cal)
    parts = {"ref": ("ref", None), "control": ("control", None), "n3": ("n3", None), "n4": ("n4", None), "load": ("load", None),
             "pf Qd x0.7": ("pf", 0.7), "pf Qd x1.2": ("pf", 1.2), "pf Qd x1.5": ("pf", 1.5), "redispatch": ("redispatch", None)}
    meta, feat = {}, {}
    for label, (n, q) in parts.items():
        e, ln = emb[n], lines[n]
        if q is not None:
            e, ln = e[e.q_scale == q], ln[ln.q_scale == q]
        anchors = e[e.outages == ""].set_index("anchor")
        s = e[e.outages != ""].reset_index(drop=True)
        m = s[["state", "load_scale", "q_scale", "anchor", "outages", "n_out", "moved_mw", "res_p", "res_q"]].copy()
        m["res_excess"] = m.res_p - m.anchor.map(anchors.res_p).to_numpy()
        base = ln[ln.outages == ""].set_index(["anchor", "line"]).genco.rename("genco0")
        j = ln[ln.outages != ""].join(base, on=["anchor", "line"])
        m["dload_max"] = m.state.map((j.genco - j.genco0).abs().groupby(j.state).max()).to_numpy()
        # trust layer on this part, exactly as analysis_shift.score_fixed
        test, _, n_states = S.prep(ln)
        assert n_states == len(m), (label, n_states, len(m))
        real = (test.ac > OVERLOAD).to_numpy()
        sig = sigma_of(test, SURR, SIGMA, None)
        m["n_real"] = m.state.map(pd.Series(real).groupby(test.state.to_numpy()).sum()).fillna(0).astype(int).to_numpy()
        for a in (0.05, 0.01):
            flagged = test[SURR].to_numpy() >= OVERLOAD - W[(SURR, SIGMA, a)] * sig
            g = pd.DataFrame({"unfl": real & ~flagged, "fl": flagged}).groupby(test.state.to_numpy()).sum()
            m[f"unflagged{a}"] = m.state.map(g.unfl).fillna(0).astype(int).to_numpy()   # real overloads not flagged themselves
            m[f"sent{a}"] = m.state.map(g.fl).fillna(0).to_numpy() > 0                   # state goes to AC anyway
        meta[label] = m
        F = {}
        for L in LAYERS:
            for pool in ("mean", "max"):
                X = stack(s[f"e{L}_{pool}"])
                F[f"L{L}.{pool}"] = X
                F[f"L{L}.{pool}-N0"] = X - stack(anchors[f"e{L}_{pool}"].loc[s.anchor])
            F[f"L{L}.absmaxdiff"] = stack(s[f"d{L}_absmax"])
            m[f"L{L}.busnorm_max"] = stack(s[f"bdn{L}"]).max(1)     # local score: the most changed bus
            F[f"L{L}.busnorm"] = stack(s[f"bdn{L}"])                # per-bus profile of the change (118 numbers)
        feat[label] = F

    # ---- distance scores. Reference scores are out-of-fold (whole outage sets held out), so the gate thresholds are honest.
    ref = meta["ref"]
    folds = list(GroupKFold(5).split(ref, groups=ref.outages))
    dims = {}
    for f in feat["ref"]:
        Xr = feat["ref"][f]
        det = Detector(Xr)
        dims[f] = det.k
        oof = {k: np.full(len(Xr), np.nan) for k in ("maha", "knn5", "knn10")}
        for tr, te in folds:
            for k, v in Detector(Xr[tr]).score(Xr[te]).items():
                oof[k][te] = v
        for k in oof:
            ref[f"{f}|{k}"] = oof[k]
        for label in meta:
            if label != "ref":
                for k, v in det.score(feat[label][f]).items():
                    meta[label][f"{f}|{k}"] = v
    cheap = ["n_out", "res_p", "res_excess", "res_q", "dload_max"]
    local = [f"L{L}.busnorm_max" for L in LAYERS]
    embs = [c for c in ref.columns if "|" in c]
    scores = cheap + local + embs
    allm = pd.concat([m.assign(part=label) for label, m in meta.items()], ignore_index=True)
    allm.to_parquet(WORK / "embedding_scores.parquet", index=False)
    out(f"\nreference: {len(ref)} states ({(ref.n_out == 1).sum()} N-1, {(ref.n_out == 2).sum()} N-2; the whole calibration sweep). "
        f"Embedding width {feat['ref']['L11.mean'].shape[1]}, 118 buses. PCA components kept: "
        + ", ".join(f"{f} {k}" for f, k in dims.items()))
    out(f"scores: {len(cheap)} cheap, {len(local)} local embedding scalars, {len(embs)} embedding distances "
        f"(3 layers x 6 feature sets x maha/knn5/knn10)")

    shifts = [l for l in meta if l not in ("ref", "control")]
    ctrl = meta["control"]

    # ---- (a) separation
    rows = []
    for sc in scores:
        r = {"score": sc}
        for l in shifts:
            r[l] = auroc(np.r_[meta[l][sc], ctrl[sc]], np.r_[np.ones(len(meta[l]), bool), np.zeros(len(ctrl), bool)])
        n2 = ref[ref.n_out == 2]
        r["control_vs_refN2"] = auroc(np.r_[ctrl[sc], n2[sc]], np.r_[np.ones(len(ctrl), bool), np.zeros(len(n2), bool)])
        r["rho_load_ref"] = spearmanr(ref[sc], ref.load_scale)[0]
        r["rho_nout_ref"] = spearmanr(ref[sc], ref.n_out)[0]
        r["rho_dload_ref"] = spearmanr(ref[sc], ref.dload_max)[0]
        rows.append(r)
    A = pd.DataFrame(rows).set_index("score")
    A["mean_broken"] = A[BROKEN].mean(axis=1)
    A.to_csv(WORK / "embedding_auroc.csv")

    # ---- (b) ranking of trust-layer failures within each test set
    rows = []
    for l in ["control"] + shifts:
        m = meta[l]
        for a in (0.05, 0.01):
            fail = (m[f"unflagged{a}"] > 0).to_numpy()
            accepted_fail = fail & ~m[f"sent{a}"].to_numpy()       # state accepted without AC although it has an overload
            insecure = (m.n_real > 0).to_numpy()
            for sc in scores:
                x = m[sc].to_numpy()
                rows.append({"shift": l, "alpha": a, "score": sc, "n_states": len(m), "n_fail": int(fail.sum()), "n_insecure": int(insecure.sum()),
                             "auroc": auroc(x, fail), "auroc_insecure_only": auroc(x[insecure], fail[insecure]),
                             "top10": top_share(x, fail, 0.10), "top20": top_share(x, fail, 0.20),
                             "n_accepted_fail": int(accepted_fail.sum()), "auroc_accepted_fail": auroc(x, accepted_fail)})
    B = pd.DataFrame(rows)
    B.to_csv(WORK / "embedding_failrank.csv", index=False)

    # ---- (c) gate
    rows = []
    for sc in scores:
        for q in GATE_Q:
            thr = ref[sc].quantile(q)
            for l in ["control"] + shifts:
                m = meta[l]
                gated = (m[sc] > thr).to_numpy()
                secure = (m.n_real == 0).to_numpy()
                r = {"score": sc, "ref_q": q, "threshold": thr, "shift": l, "n_states": len(m),
                     "gated": gated.mean(), "gated_secure": gated[secure].mean()}
                for a in (0.05, 0.01):
                    n_real = m.n_real.sum()
                    sent = m[f"sent{a}"].to_numpy()
                    acc_real = np.where(sent, 0, m.n_real)   # real overloads in states accepted without AC
                    r.update({f"extra_ac@{a}": (gated & ~sent).mean(), f"ac_before@{a}": sent.mean(),
                              f"line_miss_before@{a}": m[f"unflagged{a}"].sum() / n_real,
                              f"line_miss_after@{a}": m[f"unflagged{a}"][~gated].sum() / n_real,
                              f"state_miss_before@{a}": acc_real.sum() / n_real, f"state_miss_after@{a}": acc_real[~gated].sum() / n_real})
                rows.append(r)
    C = pd.DataFrame(rows)
    C.to_csv(WORK / "embedding_gate.csv", index=False)

    # ---- printing: all cheap + local scores, and the embedding distances that are best on each question (picked post hoc)
    best_sep = A.loc[embs].mean_broken.sort_values(ascending=False).index[:4].tolist()
    b5 = B[(B.alpha == 0.05) & B["shift"].isin(BROKEN)].groupby("score").auroc_insecure_only.mean()
    best_fail = b5.loc[embs].sort_values(ascending=False).index[:3].tolist()
    fixed = ["L11.mean|maha", "L11.mean|knn5", "L11.mean-N0|maha", "L11.mean-N0|knn5"]
    show = cheap + local + list(dict.fromkeys(fixed + best_sep + best_fail))
    W_ = max(len(s) for s in show)

    out("\n=== (a) separation: AUROC of each score, shifted states vs control states (0.5 = no separation)")
    out("    last columns: control vs the N-2 reference states (should be 0.5), Spearman rho with load level / lines out / max GENCO loading change in the reference")
    out(f"  {'score':<{W_}} | " + " ".join(f"{l.replace('pf Qd ', 'Qd'):>7}" for l in shifts) + " | ctl/ref  rho_load rho_nout rho_dload")
    for sc in show:
        r = A.loc[sc]
        out(f"  {sc:<{W_}} | " + " ".join(f"{r[l]:>7.3f}" for l in shifts) + f" | {r.control_vs_refN2:>7.3f} {r.rho_load_ref:>9.2f} {r.rho_nout_ref:>8.2f} {r.rho_dload_ref:>9.2f}")
    out("  all embedding distances, range over the 54: " + "; ".join(f"{l} {A.loc[embs, l].min():.2f}-{A.loc[embs, l].max():.2f}" for l in shifts))
    out("  by layer/feature set/distance, mean AUROC over n3, n4, Qd x0.7:")
    t = A.loc[embs].assign(feature=[s.split("|")[0] for s in embs], dist=[s.split("|")[1] for s in embs])
    for f, g in t.groupby("feature", sort=False):
        out(f"    {f:<16} " + "  ".join(f"{d} n3 {g[g.dist == d].n3.iloc[0]:.3f} n4 {g[g.dist == d].n4.iloc[0]:.3f} Qd0.7 {g[g.dist == d]['pf Qd x0.7'].iloc[0]:.3f} load {g[g.dist == d]['load'].iloc[0]:.3f}" for d in ("maha", "knn5")))

    for a in (0.05, 0.01):
        out(f"\n=== (b) does the score rank the trust-layer failures first? ({SURR}+{SIGMA}, fixed width, alpha {a:.0%})")
        out("    failure = state with a real overload reading that is not flagged. Per shift: AUROC over all states / AUROC among states that")
        out("    have a real overload (a state without overloads cannot fail, so the first number rewards any 'is it stressed' score) / share of failures in top 10% / top 20%")
        ls = ["control"] + shifts
        nf = B[(B.alpha == a) & (B.score == "n_out")].set_index("shift")
        out(f"  {'failing states / insecure / all':<{W_}} | " + " | ".join(f"{l.replace('pf Qd ', 'Qd'):^9} {nf.n_fail[l]:>3}/{nf.n_insecure[l]:>3}/{nf.n_states[l]:>4}" for l in ls))
        for sc in show:
            g = B[(B.alpha == a) & (B.score == sc)].set_index("shift")
            out(f"  {sc:<{W_}} | " + " | ".join(f"{g.auroc[l]:.2f} {g.auroc_insecure_only[l]:.2f} {g.top10[l]:>4.0%} {g.top20[l]:>4.0%}" for l in ls))

    for q in GATE_Q:
        out(f"\n=== (c) gate: escalate to AC when the score is above the {q:.0%} quantile of the reference (out-of-fold reference scores)")
        out("    per shift: states gated / secure states gated / extra AC calls at alpha 5% (gated and not already sent) | per-line miss before -> after, alpha 5% | alpha 1%")
        out("    a blind gate of the same size would cut the miss rate by the gated share; only a larger cut is information")
        for l in ["control"] + shifts:
            g0 = C[(C.ref_q == q) & (C["shift"] == l)].set_index("score")
            r0 = g0.iloc[0]
            out(f"  -- {l}: trust layer alone: AC calls {r0['ac_before@0.05']:.1%} / {r0['ac_before@0.01']:.1%}, per-line miss {r0['line_miss_before@0.05']:.2%} (alpha 5%) / {r0['line_miss_before@0.01']:.2%} (alpha 1%)")
            for sc in show:
                r = g0.loc[sc]
                cut5 = 1 - r["line_miss_after@0.05"] / r["line_miss_before@0.05"] if r["line_miss_before@0.05"] else np.nan
                out(f"     {sc:<{W_}} gated {r.gated:>6.1%}  secure {r.gated_secure:>6.1%}  extra AC {r['extra_ac@0.05']:>6.1%} | "
                    f"{r['line_miss_before@0.05']:>6.2%} -> {r['line_miss_after@0.05']:>6.2%} {'OK ' if r['line_miss_after@0.05'] <= 0.05 else '>5%'} (cut {cut5:>4.0%}) | "
                    f"{r['line_miss_before@0.01']:>6.2%} -> {r['line_miss_after@0.01']:>6.2%} {'OK ' if r['line_miss_after@0.01'] <= 0.01 else '>1%'}")

    out("\n=== (d) embedding vs cheap scores, one line per question")
    for name, group in (("cheap", cheap), ("local embedding scalar", local), ("embedding distance", embs)):
        best = A.loc[group].mean_broken.idxmax()
        out(f"  separation (mean AUROC n3/n4/Qd x0.7): best {name}: {best} {A.loc[best, 'mean_broken']:.3f}  (n3 {A.loc[best, 'n3']:.3f}, n4 {A.loc[best, 'n4']:.3f}, Qd0.7 {A.loc[best, 'pf Qd x0.7']:.3f})")
    for name, group in (("cheap", cheap), ("local embedding scalar", local), ("embedding distance", embs)):
        best = b5.loc[group].idxmax()
        g = B[(B.alpha == 0.05) & (B.score == best)].set_index("shift")
        out(f"  failure ranking among insecure states (mean AUROC n3/n4/Qd x0.7, alpha 5%): best {name}: {best} {b5[best]:.3f}  ("
            + ", ".join(f"{l} {g.auroc_insecure_only[l]:.2f}" for l in ["control"] + BROKEN) + ")")
    c = C[(C.ref_q == 0.95)]
    restored = c[c["shift"].isin(BROKEN)].groupby("score").apply(
        lambda g: pd.Series({"ok5": (g["line_miss_after@0.05"] <= 0.05).all(), "ok1": (g["line_miss_after@0.01"] <= 0.01).all(),
                             "extra": g["extra_ac@0.05"].mean()}), include_groups=False)
    cg = c[c["shift"] == "control"].set_index("score").gated
    for tag, col in (("alpha 5%", "ok5"), ("alpha 5% and 1%", "ok1")):
        ok = restored[restored.ok5 & restored[col]].sort_values("extra")
        out(f"  95% gate brings n3, n4 and Qd x0.7 back under {tag}: {len(ok)} of {len(scores)} scores. Cheapest: "
            + ("; ".join(f"{s} (extra AC {r.extra:.1%} of shifted states, control gated {cg[s]:.1%})" for s, r in ok.head(6).iterrows()) or "none"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["extract", "report"])
    p.add_argument("--sets", default=",".join(SETS))
    args = p.parse_args()
    if args.phase == "extract":
        extract(args.sets.split(","))
        return
    text = []

    def out(s):
        print(s, flush=True)
        text.append(s)

    report(out)
    (WORK / "embedding_report.log").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    sys.exit(main())

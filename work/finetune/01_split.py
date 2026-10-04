"""Build the fine-tuning dataset from a fresh datakit run and prove it is disjoint from the test set.

Input : --all  (datakit output generated with a different seed and MORE load scenarios than the test set)
        --test (the existing test set; read only)
Kept  : scenarios whose load_scenario_idx >= --first (the test set only uses indices below that, i.e. other hours
        of the aggregate load profile; on top of that the per-bus load noise comes from a different seed).
Roles by load_scenario_idx (never by scenario, so the 5 topology variants of one load case stay together):
        train  [first, pool_from) minus a random --val_share of those load indices
        val    that held-out share
        pool   [pool_from, end)   -> unlabelled pool for the active-learning round
Output: <out>/<net>/raw/{bus,gen,branch}_data.parquet (flat files, scenario renumbered 0..N-1),
        <out>/scenario_map.parquet (scenario, scenario_orig, load_idx, role, n_out),
        <out>/splits_base/{train,val,test}.pt (test.pt = the val ids again; graphkit wants the file, we never use it),
        <out>/disjointness.txt.

    .venv/bin/python work/finetune/01_split.py --net case500_goc --all work/finetune/data500_all \
        --test work/goc500/data --out work/finetune/data500_ft --first 3000 --pool_from 5000
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def fingerprints(raw, scenarios=None):
    """Per scenario: hash of the (Pd, Qd) vector rounded to 1e-6 MW, and the tuple of out-of-service branches."""
    flt = None if scenarios is None else [("scenario", "in", list(map(int, scenarios)))]
    bus = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "bus", "Pd", "Qd", "load_scenario_idx"], filters=flt).sort_values(["scenario", "bus"])
    br = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "idx", "br_status"], filters=flt)
    out = br[br.br_status == 0].groupby("scenario").idx.apply(lambda s: tuple(sorted(s)))
    rows = {}
    for s, g in bus.groupby("scenario", sort=True):
        v = np.round(np.c_[g.Pd.to_numpy(), g.Qd.to_numpy()], 6)
        rows[s] = (hashlib.sha1(v.tobytes()).hexdigest(), float(g.Pd.sum()), int(g.load_scenario_idx.iloc[0]))
    f = pd.DataFrame.from_dict(rows, orient="index", columns=["load_hash", "pd_sum", "load_idx"])
    f["topo"] = out.reindex(f.index).apply(lambda t: t if isinstance(t, tuple) else ())
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", required=True)
    p.add_argument("--all", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--pool_from", type=int, required=True)
    p.add_argument("--val_share", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    src, test, out = Path(a.all) / a.net / "raw", Path(a.test) / a.net / "raw", Path(a.out)
    raw = out / a.net / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    br = pd.read_parquet(src / "branch_data.parquet")
    status = br.groupby("scenario").agg(load_idx=("load_scenario_idx", "first"), dead=("br_status", lambda s: int((s == 0).sum())))
    always_out = int((br.groupby("idx").br_status.max() == 0).sum())
    keep = status[status.load_idx >= a.first].sort_index()
    rng = np.random.default_rng(a.seed)
    train_loads = np.arange(a.first, a.pool_from)
    present = np.intersect1d(train_loads, keep.load_idx.unique())
    val_loads = rng.choice(present, int(round(a.val_share * len(present))), replace=False)
    role = np.where(keep.load_idx >= a.pool_from, "pool", np.where(keep.load_idx.isin(val_loads), "val", "train"))
    m = pd.DataFrame({"scenario": np.arange(len(keep)), "scenario_orig": keep.index.to_numpy(), "load_idx": keep.load_idx.to_numpy(),
                      "role": role, "n_out": keep.dead.to_numpy() - always_out})
    new_id = dict(zip(m.scenario_orig, m.scenario))
    for t in ("bus", "gen", "branch"):
        d = br if t == "branch" else pd.read_parquet(src / f"{t}_data.parquet")
        d = d[d.scenario.isin(new_id)].copy()
        d["scenario"] = d.scenario.map(new_id).astype("int64")
        d = d.sort_values(["scenario", {"bus": "bus", "gen": "idx", "branch": "idx"}[t]]).drop(columns=["scenario_partition"], errors="ignore")
        d.to_parquet(raw / f"{t}_data.parquet", index=False)
        del d
    m.to_parquet(out / "scenario_map.parquet", index=False)
    sp = out / "splits_base"
    sp.mkdir(exist_ok=True)
    ids = {r: torch.tensor(m.scenario[m.role == r].to_numpy()) for r in ("train", "val")}
    torch.save(ids["train"], sp / "train.pt")
    torch.save(ids["val"], sp / "val.pt")
    torch.save(ids["val"], sp / "test.pt")

    # ---- disjointness from the test set
    new = fingerprints(raw)
    old = fingerprints(test)
    new_pairs = set(zip(new.load_hash, new.topo))
    old_pairs = set(zip(old.load_hash, old.topo))
    args_new = (src / "args.log").read_text() if (src / "args.log").exists() else ""
    args_old = (test / "args.log").read_text() if (test / "args.log").exists() else ""
    seed_of = lambda s: next((l.strip() for l in s.splitlines() if "seed" in l), "?")
    lines = [
        f"network {a.net}: new dataset {raw} vs test set {test} (read only)",
        f"datakit seed: new [{seed_of(args_new)}]  test [{seed_of(args_old)}]",
        f"scenarios kept: {len(m)} (train {int((m.role == 'train').sum())}, val {int((m.role == 'val').sum())}, pool {int((m.role == 'pool').sum())}); "
        f"load cases train {m.load_idx[m.role == 'train'].nunique()}, val {m.load_idx[m.role == 'val'].nunique()}, pool {m.load_idx[m.role == 'pool'].nunique()}",
        f"load_scenario_idx: new {new.load_idx.min()}..{new.load_idx.max()}   test {old.load_idx.min()}..{old.load_idx.max()}   "
        f"-> common indices: {len(set(new.load_idx) & set(old.load_idx))}",
        f"load vectors (Pd,Qd of all buses, rounded to 1e-6 MW): new {new.load_hash.nunique()} distinct, test {old.load_hash.nunique()} distinct, common: {len(set(new.load_hash) & set(old.load_hash))}",
        f"(load vector, set of removed branches) pairs: new {len(new_pairs)}, test {len(old_pairs)}, common: {len(new_pairs & old_pairs)}",
        f"for information: removed-branch sets that also occur in the test set: {len(set(new.topo) & set(old.topo))} of {new.topo.nunique()} distinct new sets "
        f"({new.topo.isin(set(old.topo)).mean():.1%} of new scenarios; mostly single outages)",
        f"n_out distribution (new): {m.n_out.value_counts().sort_index().to_dict()}",
        f"total load range MW: new {new.pd_sum.min():.0f}..{new.pd_sum.max():.0f}, test {old.pd_sum.min():.0f}..{old.pd_sum.max():.0f}",
    ]
    assert not (set(new.load_idx) & set(old.load_idx)) and not (set(new.load_hash) & set(old.load_hash)) and not (new_pairs & old_pairs), "NOT DISJOINT"
    lines.append("DISJOINT: yes (assert passed)")
    (out / "disjointness.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

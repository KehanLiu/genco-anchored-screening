"""Active-learning selection on the unlabelled pool: K scenarios where the fine-tuned model is least trustworthy.

Scores use ONLY label-free quantities of the pool evaluation (02_eval.py --role pool):
    mis1       bus power mismatch (max |dP|,|dQ|, MW/MVAr) left after ONE Newton step from GENCO's voltages (largest first)
    mis0       power-balance residual of GENCO's own voltages (largest first)
    nearlimit  distance of the closest GENCO-predicted line loading to its limit, min |loading - 1| (smallest first;
               = scenarios with a line predicted inside +-3% when K is smaller than their number)
    random     K random pool scenarios (seed)
The AC columns of the evaluation files (ac, dv_*, it_*) are never read for the selection; they are read afterwards
only to DESCRIBE what each criterion picked. Writes <data>/splits_al_<criterion>/{train,val,test}.pt
(train = base train + selected) and <data>/al_selection.csv.

    .venv/bin/python work/finetune/03_select.py --data work/finetune/data500_ft --pool_tag pool_ft_seed0 --k 1500
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--grid", type=int, default=500)
    p.add_argument("--pool_tag", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    data = Path(a.data)
    st = pd.read_parquet(HERE / "results" / f"states_newton_{a.grid}_{a.pool_tag}.parquet", columns=["state", "mis0_genco", "mis1_genco"]).set_index("state")
    pred = pd.read_parquet(HERE / "results" / f"lines_newton_{a.grid}_{a.pool_tag}.parquet", columns=["state", "genco"])
    m = pd.read_parquet(data / "scenario_map.parquet")
    pool = np.sort(m.scenario[m.role == "pool"].to_numpy())
    assert set(pool) == set(st.index), "pool evaluation incomplete"
    dist = (pred.genco - 1).abs().groupby(pred.state).min()
    in_band = int((dist <= 0.03).sum())
    rng = np.random.default_rng(a.seed)
    picks = {
        "mis1": st.mis1_genco.replace(np.inf, 1e9).fillna(1e9).sort_values(ascending=False).index[: a.k].to_numpy(),
        "mis0": st.mis0_genco.sort_values(ascending=False).index[: a.k].to_numpy(),
        "nearlimit": dist.sort_values().index[: a.k].to_numpy(),
        "random": rng.choice(pool, a.k, replace=False),
    }
    base = {r: torch.load(data / "splits_base" / f"{r}.pt", weights_only=True) for r in ("train", "val", "test")}
    for name, ids in picks.items():
        d = data / f"splits_al_{name}"
        d.mkdir(exist_ok=True)
        assert not set(ids.tolist()) & set(base["train"].tolist()) and not set(ids.tolist()) & set(base["val"].tolist())
        torch.save(torch.cat([base["train"], torch.tensor(np.sort(ids))]), d / "train.pt")
        torch.save(base["val"], d / "val.pt")
        torch.save(base["test"], d / "test.pt")
    pd.DataFrame({k: np.sort(v) for k, v in picks.items()}).to_csv(data / "al_selection.csv", index=False)

    # ---- description only (uses labels AFTER the selection is fixed)
    lab = pd.read_parquet(HERE / "results" / f"lines_newton_{a.grid}_{a.pool_tag}.parquet", columns=["state", "n_out", "ac", "genco"])
    lab["err"] = (lab.genco - lab.ac).abs() * 100
    per = lab.groupby("state").agg(n_out=("n_out", "first"), err_max=("err", "max"), ac_max=("ac", "max"), g_max=("genco", "max"))
    per["missed"] = (per.ac_max > 1.001) & (per.g_max <= 1.001)
    print(f"pool {len(pool)} scenarios, K = {a.k} ({a.k / len(base['train']):.0%} of the {len(base['train'])} base training scenarios); "
          f"{in_band} pool scenarios have a line predicted within +-3% of its limit")
    print(f"  {'criterion':<10} {'mean n_out':>10} {'median max line err (pt)':>25} {'critical':>9} {'missed by plain rule':>21} {'overlap with mis1':>18}")
    for name, ids in {"(whole pool)": pool, **picks}.items():
        s = per.loc[ids]
        print(f"  {name:<12} {s.n_out.mean():>8.2f} {s.err_max.median():>25.2f} {(s.ac_max > 1.001).mean():>9.1%} {int(s.missed.sum()):>10}/{int(per.missed.sum()):<10} "
              f"{len(set(ids.tolist()) & set(picks['mis1'].tolist())) / len(ids):>17.0%}")


if __name__ == "__main__":
    main()

"""Fetch the released GENCO checkpoint (model, normalizer stats, training yaml) for this folder's grid.
The grid is derived from the folder name (work/goc2000 -> case2000_goc), so the same file serves work/goc10000.

    .venv/bin/python work/goc2000/01_fetch.py --variant tiny --seed 0
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
NET = f"case{HERE.name.removeprefix('goc')}_goc"


def variant_of(variant, seed):
    return variant + (f"_seed{seed}" if seed else "")


def fetch(variant, seed):
    import yaml
    import run_pretrained
    folder = HERE / f"pf_{NET}_{variant_of(variant, seed)}"
    prefix = f"{NET}/{variant}/seed{seed}"
    for f in ("best_model_state_dict.pt", "normalizer_stats.pt"):
        run_pretrained.place("gridfm/genco-pf-datakit", f"{prefix}/{f}", folder / "model" / f)
    run_pretrained.place("gridfm/genco-pf-datakit", f"{prefix}/{NET}_{variant}_seed{seed}.yaml", folder / "released_training.yaml")
    cfg = folder / "config.yaml"
    if not cfg.exists():
        cfg.write_text(yaml.safe_dump(run_pretrained.config(NET, variant, "cpu"), sort_keys=False))
    return folder


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="tiny")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(fetch(a.variant, a.seed))

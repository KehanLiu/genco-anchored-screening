"""Stage table for REPORT.md: raw GENCO drift and per-state max |t-o| loading per stage, read from the fresh parquet files.
Usage: .venv/bin/python work/equiv/analysis/stage_table.py [goc500_tiny_seed0 goc2000_tiny_seed0 ...]"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
STAGES = ("genco", "dgenco", "hyb0", "hyb1", "g1")
BIG, LIM = 1e9, 1.001

def q(v):
    return f"{np.nanmedian(v):.3g} / {np.nanquantile(v, .9):.3g} / {np.nanmax(v):.3g}"

def main(keys):
    for key in keys:
        print(f"\n## {key}")
        print("| transform | Vm drift p.u. (med/p90/max) | Va drift deg (med/p90/max) | " + " | ".join(f"{s} max\\|t-o\\| pts (med/p90/max)" for s in STAGES) + " |")
        print("|---|---|---|" + "---|" * len(STAGES))
        for lp in sorted(RES.glob(f"{key}_*_fresh_lines.parquet")):
            tr = lp.name.split(f"{key}_")[1].removesuffix("_fresh_lines.parquet")
            L = pd.read_parquet(lp)
            S = pd.read_parquet(str(lp).replace("_lines.parquet", "_states.parquet"))
            for r in ("o", "t"):
                L[f"dgenco_{r}"] = L.ac_prev + L[f"genco_{r}"] - L[f"genco_prev_{r}"]
            cells = []
            for s in STAGES:
                dd = ((L[f"{s}_t"] - L[f"{s}_o"]).abs() * 100).groupby(L.state).max()
                cells.append(q(dd))
            print(f"| {tr} | {q(S.drift_vm_c)} | {q(S.drift_va_c)} | " + " | ".join(cells) + " |")

if __name__ == "__main__":
    main(sys.argv[1:] or ["goc500_tiny_seed0", "goc2000_tiny_seed0", "goc500_tiny_seed1", "goc500_tiny_seed42", "goc2000_small_seed0"])

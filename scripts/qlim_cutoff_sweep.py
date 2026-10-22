#!/usr/bin/env python
"""Loading-cutoff sweep under reactive limits on GOC 500: the `qlimits` block of results/qlim/batch2_results.json.

Reads the per-line table work/qlim/lines_goc500.parquet (written by `analysis/analysis_qlimits.py run --grid 500`,
gitignored, about 800 MB, needs pyarrow) and, for each loading cutoff 1.001 / 1.0 / 0.99 / 0.95 applied to the
no-limit AC loadings, counts the overloaded readings (truth: AC with PV-to-PQ switching, cutoff 1.001) that sit in
states not sent to AC, and the secure states sent. scripts/headline_numbers.py reads `qlimits.sweep` of the json
(claims qlim_cutoff100_* and qlim_cutoff95_*; the write-up's "A cutoff of 100% or 95% lowers it to 9 or 0 missed,
at the price of sending 4.9% or 23.5% of secure states to AC").

    .venv/bin/python scripts/qlim_cutoff_sweep.py                      print the block
    .venv/bin/python scripts/qlim_cutoff_sweep.py --compare            also compare it with the committed json

The function is the `qlimits()` of the batch-2 review script (reports/codex_review/batch2_checks.py, a local
review directory that is not in git), copied here unchanged except for reading only the three columns it uses.
The committed json was written by that review script and also holds `conformal`, `lambdas`, `timing` and
`finite_behavior` blocks from the same review; nothing in results/ or scripts/ reads those. This script runs
only on a machine with the saved parquet table; it does not run in CI.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "work" / "qlim" / "lines_goc500.parquet"
JSON = ROOT / "results" / "qlim" / "batch2_results.json"
OVERLOAD = 1.001  # the loading cutoff of every screen; also the truth cutoff here
CUTOFFS = (1.001, 1.0, .99, .95)


def qlimits(table=TABLE):
    df = pd.read_parquet(table, columns=["state", "ac", "ac_qlim"])
    maxima = df.groupby("state")["ac_qlim"].max()
    eligible = maxima.index[np.isfinite(maxima)]
    excluded = len(maxima) - len(eligible)
    df = df[df.state.isin(eligible)]
    real = df.ac_qlim > OVERLOAD
    insecure = np.unique(df.loc[real, "state"])
    secure = np.setdiff1d(eligible, insecure)
    sweep = []
    for cutoff in CUTOFFS:
        flagged = df.ac > cutoff
        sent = np.unique(df.loc[flagged, "state"])
        missed = real & ~df.state.isin(sent)
        wasted = len(np.intersect1d(sent, secure))
        sweep.append({"cutoff": cutoff, "missed_states": int(df.loc[missed, "state"].nunique()),
                      "missed_readings_in_accepted_states": int(missed.sum()),
                      "direct_reading_misses": int((real & ~flagged).sum()),
                      "secure_sent_to_ac": wasted, "secure_sent_percent": 100 * wasted / len(secure)})
    return {"eligible_states": len(eligible), "excluded_states_without_finite_truth": excluded,
            "insecure_states": len(insecure), "secure_states": len(secure),
            "overloaded_readings": int(real.sum()), "sweep": sweep}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, default=TABLE, help=f"per-line parquet table (default {TABLE.relative_to(ROOT)})")
    ap.add_argument("--compare", action="store_true", help=f"compare with the qlimits block of {JSON.relative_to(ROOT)}")
    a = ap.parse_args(argv)
    block = qlimits(a.table)
    print(json.dumps({"qlimits": block}, indent=2))
    if a.compare:
        committed = json.loads(JSON.read_text())["qlimits"]
        same = json.loads(json.dumps(block)) == committed  # through json so that floats compare as written
        print(f"{'equal to' if same else 'DIFFERS from'} {JSON.relative_to(ROOT)} qlimits")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

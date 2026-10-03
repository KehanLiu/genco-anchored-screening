"""Checks of the workbench's staged verdict (GENCO band -> one Newton step -> full AC) on IEEE 118.

    .venv/bin/python work/workbench_check.py --mismatch      # in-process: mismatch after one Newton step on 600 states
    .venv/bin/python work/workbench_check.py                 # against a running server.py: ~200 random N-1/N-2 states

Both print to stdout; redirect to work/workbench_check.log.
"""
import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LEVELS = (0.8, 0.9, 1.0, 1.1, 1.2)


def post(url, body):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def mismatch():
    """What one Newton step from GENCO leaves as bus power mismatch: the basis of engine.NEWTON_MISMATCH_MW."""
    sys.path.insert(0, str(ROOT))
    import engine

    eng = engine.Engine("case118_ieee", "base")
    rated = [b["id"] for b in eng.static["branches"] if b["rate_a"] is not None and not b["bridge"]]
    rng = np.random.default_rng(11)
    rows = []
    for scale in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
        done = 0
        while done < 100:
            k = (1, 2, 3)[done % 3]
            try:
                result = eng.solve(scale, [int(i) for i in rng.choice(rated, k, replace=False)], alpha=0.01)
            except engine.SolveError:
                continue
            newton = result["staged"]["newton"]
            rows.append((newton["mismatch_start_mw"], newton["mismatch_mw"], newton["max_loading_err"]))
            done += 1
    start, after, err = (np.array(c, dtype=float) for c in zip(*rows))
    print(f"mismatch after one Newton step from GENCO, {len(rows)} states (N-1/N-2/N-3, load x0.7-1.2), MW:")
    print(f"  start (GENCO's voltages): median {np.median(start):.1f}, max {start.max():.1f}")
    print(f"  after the step: median {np.median(after):.3f}, p99 {np.quantile(after, 0.99):.2f}, p99.9 {np.quantile(after, 0.999):.2f}, max {after.max():.2f}")
    print(f"  loading error after the step, points of rating: p99 {100 * np.quantile(err, 0.99):.3f}, max {100 * err.max():.3f}")
    print(f"  loading error where the mismatch is above 1 MW: max {100 * err[after > 1].max() if (after > 1).any() else 0:.3f} ({int((after > 1).sum())} states)")
    print(f"  threshold in engine.py: {engine.NEWTON_MISMATCH_MW} MW -> {int((after > engine.NEWTON_MISMATCH_MW).sum())} of these states fail the self-check")


def sweep(url, n, alpha):
    grid = json.load(urllib.request.urlopen(f"{url}/api/grid"))
    rated = [b["id"] for b in grid["branches"] if b["rate_a"] is not None and not b["bridge"]]
    rng = np.random.default_rng(5)
    stage, check, missed_lines, rows = Counter(), Counter(), [], []
    flat = used = 0
    while len(rows) < n:
        scale = LEVELS[len(rows) % len(LEVELS)]
        k = 1 if len(rows) % 2 == 0 else 2
        outages = sorted(int(i) for i in rng.choice(rated, k, replace=False))
        result = post(f"{url}/api/solve", {"load_scale": scale, "outages": outages, "alpha": alpha})
        if not result["ok"]:
            continue
        s = result["staged"]
        rows.append(s)
        stage[s["stage"]] += 1
        check[s["check"]] += 1
        if s["check"] == "missed":
            worst = max(result["branches"][i]["loading_ac"] for i in s["ac"]["overloads"])
            missed_lines.append((scale, outages, s["stage"], round(100 * worst, 2)))
        if s["iterations"]["saved"] is not None:
            flat += s["iterations"]["flat"]
            used += s["iterations"]["staged"]
    insecure = sum(bool(s["ac"]["overloads"]) for s in rows)
    print(f"\nalpha {alpha:.0%}: {n} random states (N-1 and N-2 alternating, load x{LEVELS[0]}-{LEVELS[-1]}), {insecure} with a real overload ({insecure / n:.1%})")
    print(f"  accepted at stage 1 (GENCO band):   {stage[1]:>4} ({stage[1] / n:.1%})")
    print(f"  accepted after the Newton step:     {stage[2]:>4} ({stage[2] / n:.1%})")
    print(f"  sent to full AC:                    {stage[3]:>4} ({stage[3] / n:.1%})")
    print(f"  staged verdict vs AC: correct {check['correct']}, missed overloads {check['missed']} (states), false alarms {check['false_alarm']} (secure states sent to AC)")
    for m in missed_lines:
        print(f"    missed: load x{m[0]}, outages {m[1]}, accepted at stage {m[2]}, worst real loading {m[3]}%")
    print(f"  self-check failures: {sum(not s['newton']['selfcheck_ok'] for s in rows)}; "
          f"worst mismatch after the step {max(s['newton']['mismatch_mw'] for s in rows):.2f} MW")
    print(f"  Newton iterations: flat start {flat}, staged {used} (saved {flat - used}, {1 - used / flat:.0%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mismatch", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    if args.mismatch:
        mismatch()
    else:
        for alpha in (0.01, 0.02, 0.05):
            sweep(args.url, args.n, alpha)


if __name__ == "__main__":
    main()

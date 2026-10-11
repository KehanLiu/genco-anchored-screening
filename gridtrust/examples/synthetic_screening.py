"""Staged screening on a synthetic grid: a noisy "surrogate", a conformal band, one Newton step, full AC.

A 30-bus ring with chords is solved exactly for 400 states (random loads, one line out each).
The surrogate is the exact solution plus an error on the voltages. Half of the states calibrate the
band; the other half is decided by the staged check and compared with the exact answer.

    python examples/synthetic_screening.py
"""
import numpy as np

from gridtrust import PowerFlowProblem, anchored, calibrate, decide, evaluate, shift_sigma

LIMIT, ALPHA, N_BUS, N_STATES = 1.0, 0.05, 30, 400
rng = np.random.default_rng(0)

# ---- a grid: ring + chords, bus 0 is the reference, every fifth bus holds its voltage (PV)
f = np.r_[np.arange(N_BUS), np.arange(0, N_BUS, 3)]
t = np.r_[(np.arange(N_BUS) + 1) % N_BUS, (np.arange(0, N_BUS, 3) + 7) % N_BUS]
r, x = np.full(len(f), 0.01), rng.uniform(0.05, 0.12, len(f))
bus_type = np.ones(N_BUS, int)
bus_type[::5] = 2
bus_type[0] = 3
load0 = rng.uniform(0.2, 0.6, N_BUS)
gen0 = np.where(bus_type > 1, load0.sum() / (bus_type > 1).sum(), 0.0)


def problem(load, out=None, rate=None):
    status = np.ones(len(f), bool)
    if out is not None:
        status[out] = False
    return PowerFlowProblem.from_branch_parameters(bus_type, np.full(N_BUS, 1.02), 0.0, gen0 - load, -0.3 * load,
                                                   f, t, r, x, rate=rate, status=status)


# ratings: 170% of the flow in the undisturbed grid (at least 40 MVA), so some outages overload a line
base = problem(load0, rate=np.ones(len(f)))
rate = np.maximum(1.7 * base.loading(base.solve()["V"]), 0.4)


BIAS_VM, BIAS_VA = rng.normal(0, 0.002, N_BUS), rng.normal(0, 0.003, N_BUS)


def surrogate(V):
    """Stand-in for a learned model: the true voltages with a systematic error per bus (which anchoring
    on the base case removes) and a smaller random one."""
    return np.abs(V) + BIAS_VM + rng.normal(0, 3e-4, N_BUS), np.angle(V) + BIAS_VA + rng.normal(0, 5e-4, N_BUS)


# ---- states: a load level, its base case (the anchor), then one line out
rows = []
for s in range(N_STATES):
    load = load0 * rng.uniform(0.85, 1.1)
    anchor = problem(load, rate=rate)
    v_anchor = anchor.solve()["V"]
    prob = problem(load, out=int(rng.integers(len(f))), rate=rate)
    run = prob.solve()
    if not run["converged"]:
        continue
    vm, va = surrogate(run["V"])
    vm0, va0 = surrogate(v_anchor)
    rows.append({"prob": prob, "vm": vm, "va": va, "actual": prob.loading(run["V"]),
                 "pred": prob.loading(prob.start(vm_pq=vm, va=va)),
                 "pred0": anchor.loading(anchor.start(vm_pq=vm0, va=va0)), "actual0": anchor.loading(v_anchor)})
for row in rows:  # anchored prediction and adaptive band scale, per line
    row["anchored"] = anchored(row["pred"], row["pred0"], row["actual0"])
    row["sigma"] = shift_sigma(row["pred"], row["pred0"])

cal, test = rows[: len(rows) // 2], rows[len(rows) // 2:]


def stack(part, key):
    return np.concatenate([row[key] for row in part])


def live(part):
    return ~np.isnan(stack(part, "actual"))


# ---- calibrate the band on real overloads of the calibration half
keep = live(cal)
width, n = calibrate(stack(cal, "anchored")[keep], stack(cal, "actual")[keep], LIMIT, ALPHA, stack(cal, "sigma")[keep], return_n=True)
print(f"{len(rows)} states, band width {width:.2f} sigma from {n} calibration overloads (alpha = {ALPHA:.0%})")

keep = live(test)
state = np.repeat(np.arange(len(test)), len(f))[keep]
band = evaluate(stack(test, "anchored")[keep], stack(test, "actual")[keep], state, width, LIMIT, stack(test, "sigma")[keep], n_states=len(test))
print(f"band alone: {band['ac_calls']:.1%} of states to AC ({band['wasted']:.1%} of the secure ones), "
      f"{band['missed']} of {band['real']} overloads missed (95% upper bound {band['missed_upper95']:.1%})")

# ---- the staged decision per state
stages, wrong = {1: 0, 2: 0, 3: 0}, 0
for row in test:
    d = decide(row["anchored"], width, refine=lambda: row["prob"].refine(row["vm"], row["va"]), limit=LIMIT, sigma=row["sigma"],
               rated=row["prob"].rated)
    stages[d.stage] += 1
    wrong += (not d.needs_ac) and bool(np.nanmax(row["actual"]) > LIMIT)
insecure = sum(bool(np.nanmax(row["actual"]) > LIMIT) for row in test)
print(f"staged: {stages[1]} states decided by the surrogate, {stages[2]} after one Newton step, {stages[3]} need AC "
      f"({insecure} really have an overload); {wrong} overloaded states accepted")

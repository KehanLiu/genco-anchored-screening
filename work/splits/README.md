# Frozen splits for new screening policies

`policy_splits.json` fixes which load scenarios (GOC 500 / 2000 / 10000 hybrid-start files) and which base cases
(GOC 2000 full N-1) may be used to *select* a policy and which to *report* it. Written 2026-10-10 after the
an internal audit proposed four new policies (shrink the predicted change, mismatch gate, loading cushion,
classical-first) and evaluated them on existing files.

Rules
1. Select parameters on `calibration`, report on `test`. Do not re-split, do not tune on `test`.
2. A policy whose *form* was chosen after looking at results is not validated by those results, whether or not it
   has a fitted parameter. This applies to all four forms below (the parameter-free mismatch gate included: its form
   was assessed on the whole GOC hybrid files) and to the loading cushion and classical-first thresholds on the N-1
   study. The existing files are for *development* only. Confirmation needs fresh operating points that were never
   scored: new GOC 2000 base cases for the N-1 policies and new load scenarios for the hybrid-start policies
   (recipe in `work/fresh/README.md`). The complete policy (form, parameters, mismatch thresholds, loading cutoff)
   is frozen in a file before the fresh run starts, and misses, false flags and measured end-to-end cost are
   reported together.
3. Pre-registered forms for the next batch, with their single free parameter:
   (a) **shrinkage** (scale of the predicted change): start = pre-outage AC voltages + lambda_va * (GENCO(after) -
       GENCO(before)) on Va and lambda_vm * (...) on Vm. Development objective, per coordinate:
       lambda = sum(dy_true * dy_pred) / sum(dy_pred^2) over the mover readings of the `calibration` load scenarios
       (Va: non-reference buses with true |dVa| > 0.1 degrees; Vm: PQ buses with true |dVm| > 0.001 p.u.), i.e. the
       least-squares scale of the predicted polar-voltage change, clipped to [0.25, 1.5]. Amended 2026-10-10 before
       any fresh data existed: an earlier draft said one lambda from the Va movers for both coordinates, but the
       calibration fit gives lambda_vm > 1 on every checkpoint. Values frozen in `work/fresh/policy_frozen.json`.
       This objective is a proxy: the reported outcome is misses and secure-flagged share after one Newton step,
       never the SSE.
   (b) **mismatch gate** (parameter-free): per outage take the start (pre-outage AC or anchored change) with the
       smaller initial bus mismatch, then one Newton step; costs one GENCO inference and one extra mismatch
       evaluation per outage.
   (c) **loading cushion**: decision cutoff c in {1.000, 1.0005, 1.001} for both the classical one-step screen and the
       anchored screen, truth cutoff stays 1.001. On the existing N-1 split both c = 1.001 and c = 1.0005 give zero
       calibration misses, so calibration cannot choose between them; the fresh run reports all three.
   (d) **classical-first**: the classic one-step screen runs first; GENCO (anchored start + one step) only where
       that screen leaves mismatch > tau1 or flags a loading above the cutoff c; tau1 in {0.1, 0.2, 0.5, 1.0} MW,
       reported for all four. `work/fresh/policy_frozen.json` is the authoritative statement of every form.

import numpy as np

from gridtrust.newton import RefineResult
from gridtrust.staged import decide


def refined(loading, mismatch, ok=True):
    n = len(loading)
    return RefineResult(np.ones(n), np.zeros(n), np.array(loading, float), mismatch, 50.0, 1, False, ok)


def test_stage1_band_clear_does_not_run_newton():
    calls = []
    d = decide(np.array([0.5, 0.9, np.nan]), width=0.02, limit=1.0, refine=lambda: calls.append(1))
    assert (d.stage, d.decided_by, d.reason, d.needs_ac, d.verdict) == (1, "surrogate", "band_clear", False, "secure")
    assert not calls and len(d.flagged) == 0 and d.mismatch_mw is None


def test_stage2_newton_clear():
    d = decide(np.array([0.5, 0.99]), width=0.02, limit=1.0, refine=lambda: refined([0.5, 0.97, np.nan], 0.1))
    assert (d.stage, d.decided_by, d.reason, d.needs_ac) == (2, "newton", "newton_clear", False)
    assert d.flagged.tolist() == [1] and d.selfcheck_ok and d.mismatch_mw == 0.1


def test_stage3_newton_overload():
    d = decide(np.array([0.5, 0.99]), width=0.02, limit=1.0, refine=refined([0.5, 1.03], 0.1))
    assert (d.stage, d.decided_by, d.reason, d.needs_ac) == (3, "ac", "newton_overload", True)
    assert d.newton_overloads.tolist() == [1]


def test_stage3_selfcheck_fails():
    d = decide(np.array([0.5, 0.99]), width=0.02, limit=1.0, refine=refined([0.5, 0.9], 5.0))
    assert (d.stage, d.reason, d.selfcheck_ok) == (3, "selfcheck", False)
    assert decide(np.array([0.99]), 0.02, limit=1.0, refine=refined([0.9], 5.0), mismatch_threshold_mw=10).stage == 2
    broken = decide(np.array([0.99]), 0.02, limit=1.0, refine=refined([np.nan], np.inf, ok=False))
    assert (broken.stage, broken.reason) == (3, "selfcheck")


def test_adaptive_band_anchor_and_no_refine():
    predicted, sigma = np.array([0.9, 0.9]), np.array([0.01, 0.06])
    d = decide(predicted, width=2.0, limit=1.0, sigma=sigma)           # band limit 0.98 and 0.88
    assert d.flagged.tolist() == [1] and (d.stage, d.reason) == (3, "no_refine")
    a = decide(np.array([0.1]), 0.02, limit=1.0, anchor=True)
    assert (a.stage, a.decided_by, a.needs_ac) == (3, "anchor", True)
    assert decide(np.array([0.1]), np.inf, limit=1.0).stage == 3       # uncalibrated band: everything to AC

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
    d = decide(np.array([0.5, 0.99, np.nan]), width=0.02, limit=1.0, refine=lambda: refined([0.5, 0.97, np.nan], 0.1))
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


def test_selfcheck_is_not_a_loading_certificate():
    """Two-bus counterexample (independent review, 2026-10-10): one Newton step from a poor start leaves 0.95 MW of
    mismatch, below the 1 MW threshold, while the one-step loading is 0.98 against a true 1.02. Stage 2 accepts the
    state; a second step finds the overload. The mismatch check is an acceptance heuristic, not a bound."""
    from gridtrust.newton import PowerFlowProblem

    problem = PowerFlowProblem.from_branch_parameters(bus_type=[3, 1], vm_set=[1.0, 1.0], va_ref=0.0, p_inj=[0.0, -0.3],
                                                       q_inj=[0.0, -0.1], f=[0], t=[1], r=[0.05], x=[0.3],
                                                       rate=[0.326710132607248], base_mva=100.0)
    truth = problem.solve()
    assert problem.loading(truth["V"])[0] > 1.02
    vm0, va0 = np.array([1.0, 0.9305263157894736]), np.array([0.0, -0.16842105263157894])
    one = problem.refine(vm0, va0, steps=1)
    assert one.mismatch_mw < 1.0 and one.loading[0] < 0.98
    assert decide(np.array([1.05]), width=0.02, refine=one).stage == 2  # accepted as secure: the known failure mode
    two = problem.refine(vm0, va0, steps=2)
    assert two.loading[0] > 1.02 and decide(np.array([1.05]), width=0.02, refine=two).needs_ac


def test_all_nan_prediction_is_not_secure():
    """Review finding (2026-10-10): `flag` leaves NaN unflagged, so an all-NaN prediction used to pass stage 1."""
    d = decide(np.array([np.nan, np.nan]), width=0.02, limit=1.0, refine=refined([0.5, 0.5], 0.1))
    assert (d.stage, d.decided_by, d.reason, d.needs_ac, d.verdict) == (3, "ac", "invalid", True, "needs_ac")
    assert d.invalid.tolist() == [0, 1]


def test_nan_on_rated_line_is_not_secure():
    rated = np.array([True, True, False])
    d = decide(np.array([0.5, np.nan, np.nan]), width=0.02, limit=1.0, rated=rated, refine=refined([0.5, 0.5, np.nan], 0.1))
    assert (d.stage, d.reason, d.needs_ac) == (3, "invalid", True)
    assert d.invalid.tolist() == [1]
    assert decide(np.array([0.5, np.inf, np.nan]), 0.02, limit=1.0, rated=rated).reason == "invalid"


def test_nan_loading_on_rated_line_after_newton_is_not_secure():
    rated = np.array([True, True, False])
    calls = []
    d = decide(np.array([0.5, 0.99, np.nan]), width=0.02, limit=1.0, rated=rated,
               refine=lambda: calls.append(1) or refined([0.5, np.nan, np.nan], 0.1))
    assert calls and d.selfcheck_ok and (d.stage, d.reason, d.needs_ac) == (3, "invalid", True)
    assert d.invalid.tolist() == [1] and len(d.newton_overloads) == 0
    # the legacy convention (no mask, NaN marks an unrated line) must also refuse NaN loadings on predicted lines
    d = decide(np.array([0.5, 0.99, np.nan]), width=0.02, limit=1.0, refine=refined([np.nan, np.nan, np.nan], 0.1))
    assert (d.stage, d.reason, d.needs_ac) == (3, "invalid", True) and d.invalid.tolist() == [0, 1]


def test_unrated_branch_with_undefined_loading_does_not_escalate():
    rated = np.array([True, False, False])   # rate 0 and out of service: no reading, no escalation
    d = decide(np.array([0.5, np.nan, np.nan]), width=0.02, limit=1.0, rated=rated)
    assert (d.stage, d.reason, d.needs_ac) == (1, "band_clear", False) and len(d.invalid) == 0
    d = decide(np.array([0.99, np.nan, np.inf]), width=0.02, limit=1.0, rated=rated, refine=refined([0.9, np.nan, np.nan], 0.1))
    assert (d.stage, d.reason, d.needs_ac) == (2, "newton_clear", False) and len(d.invalid) == 0
    # without a mask the old convention holds: NaN is an unrated line, as long as some line has a reading
    assert decide(np.array([0.5, np.nan]), width=0.02, limit=1.0).reason == "band_clear"


def test_rated_mask_length_must_match():
    import pytest

    with pytest.raises(ValueError):
        decide(np.array([0.5, 0.5]), width=0.02, limit=1.0, rated=np.array([True]))
    with pytest.raises(ValueError):
        decide(np.array([0.99]), width=0.02, limit=1.0, refine=refined([0.9, 0.9], 0.1))


def test_nan_start_fails_mismatch_and_finite_checks():
    """A NaN start into the real Newton step: the step breaks (ok False, mismatch inf) and the loadings are NaN,
    so the self-check and the finiteness check both refuse it, whichever is evaluated first."""
    from gridtrust.newton import PowerFlowProblem

    problem = PowerFlowProblem.from_branch_parameters(bus_type=[3, 1], vm_set=[1.0, 1.0], va_ref=0.0, p_inj=[0.0, -0.3],
                                                       q_inj=[0.0, -0.1], f=[0], t=[1], r=[0.05], x=[0.3], rate=[0.5])
    r = problem.refine(np.array([1.0, np.nan]), np.array([0.0, np.nan]))
    assert not r.ok and not np.isfinite(r.mismatch_mw) and np.isnan(r.loading).all()
    d = decide(np.array([1.05]), width=0.02, limit=1.0, rated=problem.rated, refine=r)
    assert d.needs_ac and d.stage == 3 and d.selfcheck_ok is False and d.invalid.tolist() == [0]


def test_stage1_flags_only_rated_lines_with_explicit_mask():
    """Review finding (batch 2, section 5): an unrated line with a finite prediction above the limit must not
    escalate the state when the explicit mask excludes it. Without a mask the legacy behaviour is unchanged."""
    d = decide(np.array([0.5, 1.2]), width=0.02, limit=1.0, rated=np.array([True, False]))
    assert (d.stage, d.reason, d.needs_ac) == (1, "band_clear", False) and d.flagged.tolist() == []
    d = decide(np.array([0.5, 1.2]), width=0.02, limit=1.0, rated=np.array([True, False]), refine=refined([0.5, 1.5], 0.1))
    assert (d.stage, d.reason) == (1, "band_clear")  # the Newton step is not even run
    legacy = decide(np.array([0.5, 1.2]), width=0.02, limit=1.0)
    assert (legacy.stage, legacy.reason) == (3, "no_refine") and legacy.flagged.tolist() == [1]
    both = decide(np.array([0.5, 1.2]), width=0.02, limit=1.0, rated=np.array([True, True]))
    assert (both.stage, both.reason) == (3, "no_refine") and both.flagged.tolist() == [1]


def test_fallback_without_mask_warns_once_and_keeps_legacy_verdict():
    """Review finding (batch 2, section 5): with rated=None a non-finite prediction next to finite ones passes as an
    unrated line (band_clear; a partial inf plus an infinite post-step loading even gives newton_clear). The verdict
    is kept for backward compatibility, but the fallback warns once per process; the explicit mask refuses both."""
    import warnings

    import pytest

    from gridtrust import staged

    staged._warned_fallback = False
    with pytest.warns(RuntimeWarning, match="pass rated="):
        d = decide(np.array([0.5, np.nan]), width=0.02, limit=1.0)
    assert d.reason == "band_clear" and staged._warned_fallback
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second mixed prediction in the same process does not warn again
        assert decide(np.array([0.5, np.nan]), width=0.02, limit=1.0).reason == "band_clear"
        d = decide(np.array([0.99, np.inf]), width=0.02, limit=1.0, refine=refined([0.9, np.inf], 0.1))
        assert (d.stage, d.reason, d.needs_ac) == (2, "newton_clear", False)  # the documented fallback limitation
        # no warning when the mask is explicit, the prediction is all finite, or has no finite reading at all
        assert decide(np.array([0.5, np.nan]), width=0.02, limit=1.0, rated=np.array([True, True])).reason == "invalid"
        assert decide(np.array([0.5, 0.5]), width=0.02, limit=1.0).reason == "band_clear"
        assert decide(np.array([np.nan, np.nan]), width=0.02, limit=1.0).reason == "invalid"
    staged._warned_fallback = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = decide(np.array([0.99, np.inf]), width=0.02, limit=1.0, rated=np.array([True, True]), refine=refined([0.9, np.inf], 0.1))
    assert (d.stage, d.reason, d.needs_ac) == (3, "invalid", True) and d.invalid.tolist() == [1]
    d = decide(np.array([0.99, 0.5]), width=0.02, limit=1.0, rated=np.array([True, True]), refine=refined([0.9, np.inf], 0.1))
    assert (d.stage, d.needs_ac) == (3, True) and d.invalid.tolist() == [1]  # inf loading: overload and invalid, both to AC
    assert not staged._warned_fallback

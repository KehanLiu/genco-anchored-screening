import numpy as np
import pytest

from gridtrust import conformal as cf


def synthetic(rng, n_states=400, n_lines=30):
    """Loadings around the limit, a surrogate with heteroscedastic error, and a base case to anchor on."""
    base = rng.uniform(0.3, 1.0, n_lines)
    moved = rng.random((n_states, n_lines)) < 0.2
    change = np.where(moved, rng.normal(0.05, 0.15, (n_states, n_lines)), rng.normal(0, 0.002, (n_states, n_lines)))
    actual = base + change
    bias = rng.normal(0, 0.02, n_lines)                      # systematic per-line error, removed by anchoring
    predicted = actual + bias + np.abs(change) * rng.normal(0, 0.15, change.shape) + rng.normal(0, 0.001, change.shape)
    state = np.repeat(np.arange(n_states), n_lines)
    return {"actual": actual.ravel(), "predicted": predicted.ravel(), "state": state,
            "actual_base": np.tile(base, n_states), "predicted_base": np.tile(base + bias, n_states)}


def split(data, rng):
    states = np.unique(data["state"])
    cal = np.isin(data["state"], rng.permutation(states)[: len(states) // 2])
    return {k: v[cal] for k, v in data.items()}, {k: v[~cal] for k, v in data.items()}


@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.02])
@pytest.mark.parametrize("adaptive", [False, True])
def test_coverage_over_many_splits(alpha, adaptive):
    rng = np.random.default_rng(3)
    data = synthetic(rng)
    missed = []
    for _ in range(200):
        cal, test = split(data, rng)
        sig = (lambda d: cf.shift_sigma(d["predicted"], d["predicted_base"])) if adaptive else (lambda d: None)
        pred = (lambda d: cf.anchored(d["predicted"], d["predicted_base"], d["actual_base"])) if adaptive else (lambda d: d["predicted"])
        width = cf.calibrate(pred(cal), cal["actual"], alpha=alpha, sigma=sig(cal))
        assert np.isfinite(width)
        out = cf.evaluate(pred(test), test["actual"], test["state"], width, sigma=sig(test))
        assert out["missed_share"] <= out["line_missed_share"] <= 1
        missed.append(out["line_missed_share"])
    assert np.mean(missed) <= alpha + 0.01          # the guarantee, on average over splits
    assert np.mean(missed) >= alpha / 3             # and the band is not trivially wide


def test_width_monotone_in_alpha_and_infinite_when_too_few():
    data = synthetic(np.random.default_rng(1))
    widths = [cf.calibrate(data["predicted"], data["actual"], alpha=a) for a in (0.2, 0.1, 0.05, 0.02, 0.01)]
    assert all(a <= b for a, b in zip(widths, widths[1:]))
    over = np.flatnonzero(data["actual"] > cf.LIMIT)[:50]                     # 50 overloads: alpha = 1% needs 99
    few_p, few_a = data["predicted"][over], data["actual"][over]
    assert cf.calibrate(few_p, few_a, alpha=0.01) == np.inf
    width, n = cf.calibrate(few_p, few_a, alpha=0.05, return_n=True)
    assert np.isfinite(width) and n == 50
    assert cf.calibrate(few_p[:18], few_a[:18], alpha=0.05) == np.inf         # 18 < 1/0.05 - 1
    assert np.isfinite(cf.calibrate(few_p[:19], few_a[:19], alpha=0.05))
    assert cf.calibrate([0.5, 0.6], [0.5, 0.6], alpha=0.1) == np.inf          # no overload at all
    assert cf.flag(np.array([0.1, np.nan]), np.inf).all()                     # infinite width: everything to AC


def test_calibrate_picks_the_right_order_statistic():
    # 9 overloads predicted 1..9 points below the limit: alpha = 0.2 -> ceil(10 * 0.8) = 8th smallest score
    predicted = 1.0 - 0.01 * np.arange(1, 10)
    actual = np.full(9, 1.1)
    assert cf.calibrate(predicted, actual, limit=1.0, alpha=0.2) == pytest.approx(0.08)
    assert cf.calibrate(predicted, actual, limit=1.0, alpha=0.2, sigma=np.full(9, 0.02)) == pytest.approx(4.0)
    flagged = cf.flag(predicted, 0.08, limit=1.0)
    assert flagged.sum() == 8 and not flagged[-1]
    assert not cf.flag(np.array([np.nan]), 0.08)[0]


def test_helpers():
    np.testing.assert_allclose(cf.anchored([0.9], [0.7], [0.75]), [0.95])
    np.testing.assert_allclose(cf.shift_sigma([0.9, 0.7], [0.7, 0.7], floor=0.01), [0.21, 0.01])
    assert cf.clopper_pearson_upper(0, 792) == pytest.approx(1 - 0.05 ** (1 / 792))   # exact for zero events
    assert cf.clopper_pearson_upper(5, 5) == 1.0
    assert 0.05 < cf.clopper_pearson_upper(3, 100) < 0.08


def test_evaluate_counts_states():
    # 4 states x 2 lines. State 0: overload, flagged. State 1: overload on line 1 not flagged, but line 0 is
    # (found at state level, missed at line level). State 2: overload, nothing flagged (missed). State 3: secure, flagged.
    actual = np.array([1.2, 0.5, 0.6, 1.1, 1.05, 0.4, 0.3, 0.9])
    predicted = np.array([1.1, 0.5, 1.0, 0.8, 0.90, 0.4, 0.3, 1.0])
    state = np.repeat(np.arange(4), 2)
    out = cf.evaluate(predicted, actual, state, width=0.02, limit=1.0)
    assert out["ac_calls"] == 0.75 and out["wasted"] == 1.0
    assert (out["real"], out["missed"], out["states_with_miss"]) == (3, 1, 1)
    assert out["missed_share"] == pytest.approx(1 / 3) and out["line_missed_share"] == pytest.approx(2 / 3)
    assert out["worst_missed"] == 1.05
    assert cf.evaluate(predicted, actual, state, width=0.02, limit=1.0, n_states=6)["ac_calls"] == 0.5

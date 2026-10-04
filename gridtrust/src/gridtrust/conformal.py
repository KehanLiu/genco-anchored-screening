"""Label-conditional split conformal screening of line overloads.

A surrogate predicts every line loading. A line is "possibly overloaded" when its prediction is
at or above  limit - width * sigma ; a state with such a line goes to the AC solver, every other
state is accepted as secure without AC.

`width` comes from the calibration readings that really are overloaded: measure how far below the
limit the surrogate put each of them, in units of sigma, and take the ceil((n+1)(1-alpha))-th
smallest of these n scores. Under exchangeability a new real overload is then flagged with
probability >= 1 - alpha, so on average at most a share alpha of the real overloads is missed.
The guarantee is marginal (an average over calibration and test draws), not per state.

sigma makes the band adaptive. Two choices used here:
    None           one band width for all readings (sigma = 1)
    shift_sigma()  how far the surrogate moved the line away from its base-case loading, plus a
                   floor: lines an outage does not touch get a narrow band
and `anchored()` builds the matching prediction: the AC loading of the base case (which an
operator has anyway) plus the surrogate's predicted change.

All functions take plain arrays with one entry per line reading.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

LIMIT = 1.001        # loading above which a line counts as overloaded (just over 100% for solver tolerance)
SHIFT_FLOOR = 0.005  # 0.5 points of rating


def _sigma(sigma, n):
    return np.ones(n) if sigma is None else np.broadcast_to(np.asarray(sigma, dtype=float), (n,))


def anchored(predicted, predicted_base, actual_base):
    """Base-case AC loading plus the surrogate's predicted change from its own base-case prediction."""
    return np.asarray(actual_base, dtype=float) + np.asarray(predicted, dtype=float) - np.asarray(predicted_base, dtype=float)


def shift_sigma(predicted, predicted_base, floor=SHIFT_FLOOR):
    """Band scale per reading: how far the surrogate moved the line from its base-case loading, plus a floor."""
    return np.abs(np.asarray(predicted, dtype=float) - np.asarray(predicted_base, dtype=float)) + floor


def calibrate(predicted, actual, limit=LIMIT, alpha=0.05, sigma=None, return_n=False):
    """Band width (in sigma units) such that at most a share alpha of real overloads falls below limit - width * sigma.

    predicted, actual: surrogate and true (AC) loading of the calibration readings.
    Returns inf when there are too few real overloads in the calibration data for this alpha
    (fewer than 1/alpha - 1): everything is then flagged. With `return_n` also returns the
    number of real overloads the width rests on.
    """
    predicted, actual = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    real = actual > limit
    scores = np.sort(((limit - predicted) / _sigma(sigma, len(predicted)))[real])
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    width = np.inf if k > n else float(scores[k - 1])
    return (width, n) if return_n else width


def flag(predicted, width, limit=LIMIT, sigma=None):
    """Boolean per reading: possibly overloaded (prediction inside or above the band). NaN predictions are not flagged,
    except with an infinite width, which flags everything."""
    predicted = np.asarray(predicted, dtype=float)
    if np.isinf(width) and width > 0:
        return np.ones(len(predicted), bool)
    with np.errstate(invalid="ignore"):
        return predicted >= limit - width * _sigma(sigma, len(predicted))


def clopper_pearson_upper(k, n, confidence=0.95):
    """One-sided upper confidence bound on a proportion after k events in n trials."""
    return float(beta.ppf(confidence, k + 1, n - k)) if n > k else 1.0


def evaluate(predicted, actual, state, width, limit=LIMIT, sigma=None, n_states=None, confidence=0.95):
    """Screening outcome on test readings: every state with a flagged reading goes to AC, the rest is accepted.

    state: id of the grid state each reading belongs to. n_states: number of states screened, if
    some have no reading in the arrays (e.g. after dropping readings far from the limit).
    Returns a dict:
        ac_calls            share of states sent to AC
        wasted              AC runs on states without an overload, as a share of those states
        real, missed        real overloads; those sitting in a state that was accepted without AC
        missed_share        missed / real
        missed_upper95      Clopper-Pearson upper bound on the miss rate (at `confidence`)
        line_missed_share   real overloads not flagged themselves: the event the guarantee is about
        worst_missed        highest true loading among the missed readings (NaN if none)
        states_with_miss    accepted states that contain a real overload
    """
    predicted, actual, state = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float), np.asarray(state)
    real = actual > limit
    flagged = flag(predicted, width, limit, sigma)
    sent = np.unique(state[flagged])
    escalated = np.isin(state, sent)
    missed = real & ~escalated
    insecure = np.unique(state[real])
    if n_states is None:
        n_states = len(np.unique(state))
    n_real, k = int(real.sum()), int(missed.sum())
    return {
        "ac_calls": len(sent) / n_states,
        "wasted": len(np.setdiff1d(sent, insecure)) / max(n_states - len(insecure), 1),
        "real": n_real,
        "missed": k,
        "missed_share": k / max(n_real, 1),
        "missed_upper95": clopper_pearson_upper(k, n_real, confidence),
        "line_missed_share": (real & ~flagged).sum() / max(n_real, 1),
        "worst_missed": float(actual[missed].max()) if k else np.nan,
        "states_with_miss": len(np.unique(state[missed])),
    }

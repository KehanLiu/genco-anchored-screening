"""The staged decision for one grid state: surrogate -> conformal band -> Newton step + self-check -> full AC.

    stage 1  no line reading inside the band        -> accepted as secure on the surrogate alone
    stage 2  a Newton step from the surrogate's voltages leaves a mismatch below the threshold
             and shows no line above the limit      -> accepted as secure after the step
    stage 3  otherwise                              -> the state needs a full AC solve

The base case (no outage) is the anchor of the band and is always solved by AC.

A reading that is not a finite number shows nothing. Lines without a rating (or out of service) have
no reading by design and are excluded by the `rated` mask; a non-finite prediction or post-step loading
on a rated line is invalid input and sends the state to AC rather than passing as "not above the limit".
Strict protection needs the explicit mask (`rated=problem.rated`): without it, a non-finite prediction next to
finite ones is taken to mark an unrated line and is not treated as invalid input.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .conformal import LIMIT, flag

MISMATCH_MW = 1.0  # largest bus mismatch after the Newton step up to which the corrected loadings are accepted (a heuristic, not a bound on loading error)


@dataclass
class Decision:
    stage: int                 # 1, 2 or 3: the stage that decided
    decided_by: str            # "surrogate", "newton", "ac" or "anchor"
    reason: str                # "band_clear", "newton_clear", "newton_overload", "selfcheck", "no_refine", "invalid" or "anchor"
    needs_ac: bool             # True: run the full AC solve; False: accepted as secure
    flagged: np.ndarray = field(repr=False)            # indices of readings inside or above the band (stage 1)
    newton_overloads: np.ndarray = field(repr=False)   # indices of lines above the limit after the step (empty if not run)
    mismatch_mw: float | None = None                   # self-check value, None if the step was not run
    selfcheck_ok: bool | None = None
    refined: object = field(repr=False, default=None)  # the RefineResult, if the step was run
    invalid: np.ndarray = field(repr=False, default_factory=lambda: np.array([], dtype=int))  # rated lines whose reading was not finite

    @property
    def verdict(self):
        return "needs_ac" if self.needs_ac else "secure"


_FALLBACK_WARNING = ("decide: a non-finite prediction next to finite ones is taken as an unrated line because no "
                     "rated= mask was given; pass rated= (e.g. problem.rated) for strict non-finite protection")
_warned_fallback = False  # the fallback warning is emitted once per process


def _rated_mask(rated, n, predicted):
    """Boolean per line: has a rating and is in service, i.e. its reading must be a finite number.

    None falls back to the convention that a NaN prediction marks an unrated line; a prediction without any
    finite reading cannot be all unrated lines, so every line then counts as rated (and invalid). The fallback
    cannot tell a failed prediction on a rated line from an unrated line, so it warns once per process when it
    is actually relied on (some readings finite, some not)."""
    global _warned_fallback
    if rated is None:
        finite = np.isfinite(predicted)
        if not finite.any():
            return np.ones(n, bool)
        if not finite.all() and not _warned_fallback:
            _warned_fallback = True
            warnings.warn(_FALLBACK_WARNING, RuntimeWarning, stacklevel=3)
        return finite
    rated = np.asarray(rated, dtype=bool)
    if rated.shape != (n,):
        raise ValueError(f"rated mask has shape {rated.shape}, expected ({n},)")
    return rated


def _invalid(values, rated, n):
    """Indices of rated lines whose reading is not finite."""
    values = np.asarray(values, dtype=float)
    if values.shape != (n,):
        raise ValueError(f"expected one reading per line ({n}), got shape {values.shape}")
    return np.flatnonzero(rated & ~np.isfinite(values))


def decide(predicted, width, refine=None, limit=LIMIT, sigma=None, mismatch_threshold_mw=MISMATCH_MW, anchor=False,
           rated=None):
    """Decide one state.

    predicted: the surrogate's loading per line (anchored or raw; NaN for lines without a rating).
    width, sigma: the calibrated band (see gridtrust.conformal).
    refine: a RefineResult, or a zero-argument callable returning one (e.g.
        `lambda: problem.refine(vm, va)`); a callable is only invoked when stage 1 flags a line.
        None means there is no stage 2: a flagged state goes straight to AC.
    anchor: True for the base case, which is solved by AC whatever the surrogate says.
    rated: boolean per line, True where the line has a rating and is in service (e.g. `problem.rated`).
        A non-finite prediction or post-step loading on such a line is invalid input: the state goes to
        AC (reason "invalid"), and stage 1 only flags rated lines. None keeps the convention that NaN in
        `predicted` marks an unrated line; a prediction without any finite reading is still never accepted.
        Strict non-finite protection requires the explicit mask: with rated=None a non-finite prediction
        (or post-step loading) on a line that does have a rating is silently treated as unrated, and such a
        state can be accepted as secure. The fallback warns once per process (RuntimeWarning) when a
        prediction mixes finite and non-finite readings.
    """
    none = np.array([], dtype=int)
    predicted = np.asarray(predicted, dtype=float)
    n = len(predicted)
    explicit = rated is not None
    rated = _rated_mask(rated, n, predicted)
    flags = flag(predicted, width, limit, sigma)
    if explicit:
        flags = flags & rated  # unrated lines have no reading and cannot flag the state
    flagged = np.flatnonzero(flags)
    if anchor:
        return Decision(3, "anchor", "anchor", True, flagged, none)
    invalid = _invalid(predicted, rated, n)
    if len(invalid) or not rated.any():
        return Decision(3, "ac", "invalid", True, flagged, none, invalid=invalid)
    if len(flagged) == 0:
        return Decision(1, "surrogate", "band_clear", False, flagged, none)
    if refine is None:
        return Decision(3, "ac", "no_refine", True, flagged, none)
    result = refine() if callable(refine) else refine
    ok = result.selfcheck(mismatch_threshold_mw)
    loading = np.asarray(result.loading, dtype=float)
    invalid = _invalid(loading, rated, n)
    with np.errstate(invalid="ignore"):
        over = np.flatnonzero(rated & (loading > limit))
    mismatch = float(result.mismatch_mw)
    if ok and len(over) == 0 and len(invalid) == 0:
        return Decision(2, "newton", "newton_clear", False, flagged, over, mismatch, True, result)
    reason = "selfcheck" if not ok else "newton_overload" if len(over) else "invalid"
    return Decision(3, "ac", reason, True, flagged, over, mismatch, ok, result, invalid)

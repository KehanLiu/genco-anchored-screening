"""The staged decision for one grid state: surrogate -> conformal band -> Newton step + self-check -> full AC.

    stage 1  no line reading inside the band        -> accepted as secure on the surrogate alone
    stage 2  a Newton step from the surrogate's voltages leaves a mismatch below the threshold
             and shows no line above the limit      -> accepted as secure after the step
    stage 3  otherwise                              -> the state needs a full AC solve

The base case (no outage) is the anchor of the band and is always solved by AC.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conformal import LIMIT, flag

MISMATCH_MW = 1.0  # largest bus mismatch after the Newton step up to which the corrected loadings are trusted


@dataclass
class Decision:
    stage: int                 # 1, 2 or 3: the stage that decided
    decided_by: str            # "surrogate", "newton", "ac" or "anchor"
    reason: str                # "band_clear", "newton_clear", "newton_overload", "selfcheck", "no_refine" or "anchor"
    needs_ac: bool             # True: run the full AC solve; False: accepted as secure
    flagged: np.ndarray = field(repr=False)            # indices of readings inside or above the band (stage 1)
    newton_overloads: np.ndarray = field(repr=False)   # indices of lines above the limit after the step (empty if not run)
    mismatch_mw: float | None = None                   # self-check value, None if the step was not run
    selfcheck_ok: bool | None = None
    refined: object = field(repr=False, default=None)  # the RefineResult, if the step was run

    @property
    def verdict(self):
        return "needs_ac" if self.needs_ac else "secure"


def decide(predicted, width, refine=None, limit=LIMIT, sigma=None, mismatch_threshold_mw=MISMATCH_MW, anchor=False):
    """Decide one state.

    predicted: the surrogate's loading per line (anchored or raw; NaN for lines without a rating).
    width, sigma: the calibrated band (see gridtrust.conformal).
    refine: a RefineResult, or a zero-argument callable returning one (e.g.
        `lambda: problem.refine(vm, va)`); a callable is only invoked when stage 1 flags a line.
        None means there is no stage 2: a flagged state goes straight to AC.
    anchor: True for the base case, which is solved by AC whatever the surrogate says.
    """
    none = np.array([], dtype=int)
    flagged = np.flatnonzero(flag(predicted, width, limit, sigma))
    if anchor:
        return Decision(3, "anchor", "anchor", True, flagged, none)
    if len(flagged) == 0:
        return Decision(1, "surrogate", "band_clear", False, flagged, none)
    if refine is None:
        return Decision(3, "ac", "no_refine", True, flagged, none)
    result = refine() if callable(refine) else refine
    ok = result.selfcheck(mismatch_threshold_mw)
    with np.errstate(invalid="ignore"):
        over = np.flatnonzero(np.asarray(result.loading, dtype=float) > limit)
    mismatch = float(result.mismatch_mw)
    if ok and len(over) == 0:
        return Decision(2, "newton", "newton_clear", False, flagged, over, mismatch, True, result)
    return Decision(3, "ac", "newton_overload" if ok else "selfcheck", True, flagged, over, mismatch, ok, result)

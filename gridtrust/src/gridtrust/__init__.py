"""gridtrust: decide which answers of a power-flow surrogate can be accepted without an AC solve.

    newton     AC power flow from plain arrays; `refine` = Newton step(s) from predicted voltages + self-check
    conformal  label-conditional split conformal band on predicted line loadings
    staged     surrogate -> band -> Newton step + mismatch check -> needs AC

Depends on numpy, scipy and pandas only.
"""
from . import conformal, newton, staged
from .conformal import anchored, calibrate, clopper_pearson_upper, evaluate, flag, shift_sigma
from .newton import PowerFlowProblem, RefineResult
from .staged import Decision, decide

__version__ = "0.1.0"
__all__ = [
    "conformal", "newton", "staged",
    "PowerFlowProblem", "RefineResult",
    "calibrate", "flag", "anchored", "shift_sigma", "evaluate", "clopper_pearson_upper",
    "decide", "Decision",
]

"""Serve the what-if workbench: the engine under /api and the front-end from web/.

    .venv/bin/python workbench/server.py            # then open http://127.0.0.1:8000
"""

# engine first: it loads Julia, which has to happen before torch.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]  # engine / run_pretrained at the root, sibling analysis modules here

from engine import STAGE_ALPHA_DEFAULT, Engine, SolveError  # noqa: E402

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

WEB = Path(__file__).parent / "web"
app = FastAPI()
engine = None


class Scenario(BaseModel):
    load_scale: float = Field(1.0, ge=0.1, le=3.0)
    outages: list[int] = []
    alpha: float = Field(STAGE_ALPHA_DEFAULT, gt=0, lt=1)  # allowed miss rate of the staged check


# Handlers are `async def` on purpose: they run on the event loop in the main
# thread, which is the only thread Julia may be called from.
@app.get("/api/grid")
async def grid():
    # "staged" is null for a grid without a calibrated band; the front-end then shows the plain advisor.
    return {**engine.static, "staged": engine.staged_info()}


@app.post("/api/solve")
async def solve(scenario: Scenario):
    n_branches = len(engine.static["branches"])
    if any(not 0 <= o < n_branches for o in scenario.outages):
        return {"ok": False, "reason": "ac_failed", "message": "Unknown branch id."}
    try:
        return engine.solve(scenario.load_scale, scenario.outages, alpha=scenario.alpha)
    except SolveError as err:
        return {"ok": False, "reason": err.reason, "message": err.message}


def main():
    global engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="case118_ieee")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    engine = Engine(args.network, args.variant)
    WEB.mkdir(exist_ok=True)
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

# Poster outline (one page, A0 portrait or 16:9 landscape)

**Title:** A Newton step or two is enough: GENCO as the first stage of a contingency screen with a stated miss rate

**Headline number (largest type on the page):** 12.5% of scenarios sent to AC, 99.7% of overloaded lines found. GOC 500, up to five branches removed, released GENCO Tiny.

## Layout: three columns

### Left: the question and the workflow
- One sentence: thousands of grid states, one question (is any line overloaded?), and a surrogate whose answers must be accepted or checked.
- `figures/workflow.png`: the three stages with what happens to 1,000 scenarios.
- Box "what is new against the example entry": the Newton step, a miss rate that is promised and measured, a DC baseline tuned as a practitioner would, a stress test.

### Centre: the main result
- `figures/onestep.png`, large.
- Small table under it: scenarios sent to AC / secure scenarios sent / lines missed, for GENCO with the example rule, GENCO + one step, DC + one step, flat + one step.
- One line: "the step only works from GENCO".
- `figures/scale.png`: 500, 2,000 and 10,000 buses. One step holds to 2,000 buses; 10,000 needs two.

### Right: the trust layer and its limits
- `figures/cost.png`: what a promised miss rate costs, GENCO against tuned and plain DC.
- `figures/coverage.png`: the promise holds on average.
- `figures/stress.png`: where it breaks (N-3, N-4, low reactive demand) and what recalibration restores.

## Bottom strip
- `figures/workbench_outage.png`, small: the operator workbench showing which stage decided a state.
- Findings about GENCO, one line each: its embedding space does not warn of unfamiliar states better than counting the lines out; voltage violations DC cannot see (`figures/voltage.png`, small); Newton iterations halved (`figures/newton.png`, small); errors are systematic per dispatch and anchoring removes them.
- Limits, stated plainly: no speed claim on a laptop; tuned DC ties with GENCO on Texas 2000; near-replication of the example setup.
- Tools: GENCO, gridfm-graphkit, gridfm-datakit. Code link and QR code. Author and contact.

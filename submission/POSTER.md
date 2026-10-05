# Poster outline (one page, A0 portrait or 16:9 landscape)

**Title:** Let GENCO predict the change, not the state: one-step contingency screening, tested until it breaks

**Headline (largest type):** Pre-outage AC solution + GENCO's predicted change + one Newton step: 2 of 6,986 overloaded lines missed on IEEE 118, 5 of 2,621 on GOC 500, 0 of 369 on GOC 2000. The classic one-step screen misses 74, 41 and 2.

## Layout: three columns

### Left: the question and how the answer changed
- One sentence: thousands of grid states, one question (is any line overloaded?), and a surrogate whose answers must be accepted or checked.
- Four numbered steps (as in the write-up): GENCO alone is not safe; one Newton step fixes it; the pre-outage solution is the baseline that matters; combine them.
- `figures/workflow.png`.

### Centre: the main result
- `figures/starts.png`, large.
- `figures/fulln1.png`: a complete N-1 analysis with costs.
- One line each: fewest Newton iterations; not faster on a laptop; break-even needs inference near 2 ms per state.

### Right: tested until it breaks
- 10,000 buses: the predicted change stops helping, and fine-tuning does not bring it back (`figures/finetune.png`): 5 → 4 lines missed, against 2 from the pre-outage solution alone.
- Reactive limits: ranking survives, "almost nothing missed" does not; a withdrawn voltage finding.
- `figures/stress.png` and `figures/cost.png`: a conformal band with a stated miss rate, where it holds and where it breaks.
- Zero-shot transfer and embedding-space detection: negative results, one line each.

## Bottom strip
- `figures/workbench.gif` first frame or `figures/workbench_outage.png`: the operator workbench.
- Limits, stated plainly: no speed claim; fine-tuning only at laptop scale; near-replication of the example setup.
- Tools: GENCO, gridfm-graphkit, gridfm-datakit; the `gridtrust` package. Code link and QR code. Author and contact.

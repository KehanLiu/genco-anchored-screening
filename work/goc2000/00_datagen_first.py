"""datakit generation that stops after the first KEEP load scenarios; the later ones are never computed.

The GOC 2000 evaluation pool (work/goc2000/data) holds load scenarios 0..499 of a run that asked for 1,000
(work/goc2000/datagen.yaml): datakit saved its first large chunk (settings.large_chunk_size = 500) and the run was then
killed by its 7,200 s timeout (work/goc2000/logs/datagen.log). That chunk is NOT reproduced by setting load.scenarios
to 500: datakit draws the per-bus load noise as a single array of shape (buses, load.scenarios) under the global seed
(gridfm_datakit/perturbations/load_perturbation.py, uniform noise after the profile cut), so with a different count
every bus but the first receives different noise for hours 0..499. With load.scenarios left at 1,000 the load profile,
the large-chunk boundaries and the per-chunk seeds (seed * 20000 + start + 1, process_scenario_chunk in
gridfm_datakit/process/process_network.py) are those of the original run, and dropping the large chunks whose end is
above KEEP gives exactly the scenarios the original run saved. Mirror image of work/finetune/00_datagen_skip.py.

    .venv/bin/python work/goc2000/00_datagen_first.py work/goc2000/datagen.yaml 500
"""
import sys

import yaml

import gridfm_datakit.generate as g

if __name__ == "__main__":
    cfg, keep = sys.argv[1], int(sys.argv[2])
    c = yaml.safe_load(open(cfg))
    total, chunk = c["load"]["scenarios"], c["settings"]["large_chunk_size"]
    assert keep % chunk == 0 and total % chunk == 0, "KEEP and load.scenarios must be multiples of large_chunk_size"
    orig = g._split_range

    def patched(start, stop, n_parts):
        bounds = orig(start, stop, n_parts)
        if start == 0 and stop == total:  # the top-level split into large chunks
            bounds = [b for b in bounds if b[1] <= keep]
            print(f"keeping load scenarios below {keep}; large chunks kept: {bounds}", flush=True)
        return bounds

    g._split_range = patched
    print(g.generate_power_flow_data_distributed(cfg))

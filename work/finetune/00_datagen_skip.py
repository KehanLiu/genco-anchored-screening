"""datakit generation that skips the first SKIP load scenarios (they would only be thrown away).

datakit cuts the aggregate load profile to its first `load.scenarios` hours, so load scenarios that are disjoint
from a test set built from hours 0..SKIP-1 need load.scenarios > SKIP. This wrapper leaves datakit untouched and
only drops the large chunks that start below SKIP (SKIP must be a multiple of settings.large_chunk_size), so the
kept scenarios are exactly what a full run would have produced for those indices (same per-chunk seeds).

    .venv/bin/python work/finetune/00_datagen_skip.py work/finetune/datagen2000.yaml 1000
"""
import sys

import yaml

import gridfm_datakit.generate as g

if __name__ == "__main__":
    cfg, skip = sys.argv[1], int(sys.argv[2])
    c = yaml.safe_load(open(cfg))
    total, chunk = c["load"]["scenarios"], c["settings"]["large_chunk_size"]
    assert skip % chunk == 0 and total % chunk == 0, "SKIP and load.scenarios must be multiples of large_chunk_size"
    orig = g._split_range

    def patched(start, stop, n_parts):
        bounds = orig(start, stop, n_parts)
        if start == 0 and stop == total:  # the top-level split into large chunks
            bounds = [b for b in bounds if b[0] >= skip]
            print(f"skipping load scenarios below {skip}; large chunks kept: {bounds}", flush=True)
        return bounds

    g._split_range = patched
    print(g.generate_power_flow_data_distributed(cfg))

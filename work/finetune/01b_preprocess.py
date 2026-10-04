"""Build graphkit's per-scenario cache (<data>/<net>/processed) ahead of training. No training, no model.

    .venv/bin/python work/finetune/01b_preprocess.py case500_goc work/finetune/data500_ft
"""
import sys
import time
from pathlib import Path

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk

net, data = sys.argv[1], Path(sys.argv[2])
t = time.perf_counter()
ds = HeteroGridDatasetDisk(root=str(data / net), data_normalizer=None, stream_partitions="off")
print(f"{len(ds)} scenarios cached in {ds.processed_dir}, {time.perf_counter() - t:.0f}s")

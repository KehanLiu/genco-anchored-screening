"""Read only the parquet footers of the first partition of each table over HTTP range requests.

Prints row groups, rows, and compressed bytes per column so the download can be sized.
"""
import json
import sys
import urllib.request

import fsspec
import pyarrow.parquet as pq

REPO = "https://huggingface.co/gridfm/genco-pf-contingency"
level = sys.argv[1] if len(sys.argv) > 1 else "test_1"
part = sys.argv[2] if len(sys.argv) > 2 else "0"
fs = fsspec.filesystem("http")
for table in ("bus", "gen", "branch", "runtime"):
    path = f"data/{level}/Texas2k_case1_2016summerpeak/raw/{table}_data.parquet/scenario_partition={part}"
    listing = json.load(urllib.request.urlopen(f"{REPO.replace('.co/', '.co/api/models/')}/tree/main/{path}", timeout=30))
    for entry in listing:
        url = f"{REPO}/resolve/main/{entry['path']}"
        with fs.open(url, block_size=256 * 1024, cache_type="readahead") as handle:
            meta = pq.ParquetFile(handle).metadata
            print(f"\n{table} {entry['path'].split('/')[-1]} size={entry['size'] / 1e6:.1f}MB rows={meta.num_rows} row_groups={meta.num_row_groups}")
            sizes = {}
            for g in range(meta.num_row_groups):
                group = meta.row_group(g)
                for c in range(group.num_columns):
                    col = group.column(c)
                    sizes[col.path_in_schema] = sizes.get(col.path_in_schema, 0) + col.total_compressed_size
            print("  rows per group:", [meta.row_group(g).num_rows for g in range(min(meta.num_row_groups, 8))])
            print("  " + ", ".join(f"{k}={v / 1e6:.2f}" for k, v in sizes.items()))
            stats = meta.row_group(0).column(0).statistics
            last = meta.row_group(meta.num_row_groups - 1).column(0).statistics
            print(f"  first column {meta.row_group(0).column(0).path_in_schema}: min={stats.min} max={last.max}")

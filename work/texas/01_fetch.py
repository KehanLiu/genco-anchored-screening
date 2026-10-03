"""Fetch a slice of the Texas contingency test sets with HTTP range requests.

Note: an earlier version skipped r, x and the eight admittance columns assuming they
were constant. They are not: the test sets perturb every branch's r and x per scenario
(checked on test_2 partition 0), so all columns are read.

usage: 01_fetch.py LEVEL PARTITION [PARTITION ...]     e.g. 01_fetch.py test_1 0 1
Writes work/texas/data/LEVEL/{bus,gen,branch,runtime}_pPARTITION.parquet and
appends the exact bytes transferred to work/texas/data/downloads.csv.
"""
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

HERE = Path(__file__).parent
REPO = "gridfm/genco-pf-contingency"
RAW = "data/{level}/Texas2k_case1_2016summerpeak/raw/{table}_data.parquet/scenario_partition={part}"
STATIC = ["Yff_r", "Yff_i", "Yft_r", "Yft_i", "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i"]
SKIP = STATIC + ["r", "x"]  # r and x are not model inputs
session = requests.Session()


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a URL; every read is one range request, counted."""

    def __init__(self, url, size):
        self.url, self.size, self.pos, self.fetched = url, size, 0, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        for attempt in range(4):
            try:
                reply = session.get(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"}, timeout=120)
                reply.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2)
        data = reply.content
        assert len(data) == end - self.pos, (len(data), end - self.pos)
        self.pos = end
        self.fetched += len(data)
        return data


def listing(path):
    url = f"https://huggingface.co/api/models/{REPO}/tree/main/{path}"
    return json.load(urllib.request.urlopen(url, timeout=30))


def fetch(level, part):
    out = HERE / "data" / level
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for table in ("runtime", "gen", "bus", "branch"):
        if (out / f"{table}_p{part}.parquet").exists():
            print(f"{level} p{part} {table}: already on disk", flush=True)
            continue
        frames = []
        for entry in listing(RAW.format(level=level, table=table, part=part)):
            url = f"https://huggingface.co/{REPO}/resolve/main/{entry['path']}"
            handle = RangeFile(url, entry["size"])
            parquet = pq.ParquetFile(handle, pre_buffer=True)
            frame = parquet.read().to_pandas()
            frames.append(frame)
            log.append({"level": level, "partition": part, "table": table, "file": entry["path"],
                        "file_bytes": entry["size"], "fetched_bytes": handle.fetched, "rows": len(frame)})
            print(f"{level} p{part} {table}: {len(frame)} rows, {handle.fetched / 1e6:.1f} of {entry['size'] / 1e6:.1f} MB", flush=True)
        pd.concat(frames).to_parquet(out / f"{table}_p{part}.parquet", index=False)
    log_path = HERE / "data" / "downloads.csv"
    pd.DataFrame(log).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


if __name__ == "__main__":
    level = sys.argv[1]
    for part in sys.argv[2:]:
        fetch(level, int(part))

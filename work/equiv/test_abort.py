"""pytest work/equiv/test_abort.py  -- the SPEC.md section 7 ABORT rule of run_equiv.py (no engine / Julia import, synthetic data only).

Checks that a false outage mapping, an equivalence maximum above 1e-9 and an identity control that does not reproduce o = t
raise PipelineAbort, and that `report` on such data writes no summary / criterion file, while clean data still produces both.
Stored-frame regressions (independent batch-5 review probe): one NaN equivalence cell among passing
rows, and one non-finite `none`-control reading whose counterpart is finite, must abort too (pandas maxima skip NaN); the copied CLI
must exit 2 on both without writing a CSV."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_equiv as rq  # noqa: E402


def test_outage_map_check_raises():
    rq.check_outage_map(True, 7, "split_all")
    with pytest.raises(rq.PipelineAbort, match="outage mapping"):
        rq.check_outage_map(False, 7, "split_all")
    with pytest.raises(rq.PipelineAbort):
        rq.check_outage_map(np.bool_(False), 7, "split_all")


def test_equivalence_check_raises_above_1e_9_and_ignores_gen_n_on():
    rq.check_equivalence({"ybus": 1e-12, "S_spec": 0.0, "gen_n_on": 6.0, "eq_c_gen_n_on": 6.0}, "unit")
    with pytest.raises(rq.PipelineAbort, match="ybus"):
        rq.check_equivalence({"ybus": 2e-9, "S_spec": 0.0}, "unit")
    with pytest.raises(rq.PipelineAbort):
        rq.check_equivalence({"eq_c_ybus": float("nan")}, "unit")


def test_identity_check_raises():
    rq.check_identity(0.0)
    with pytest.raises(rq.PipelineAbort, match="identity control"):
        rq.check_identity(0.5)


def _synthetic(n_states=3, n_lines=4, broken_identity=False, broken_map=False, bad_eq=False):
    """Minimal lines / states frames with every column `report` reads, transform `none`."""
    rng = np.random.default_rng(0)
    rows, lines = [], []
    for s in range(n_states):
        ac = rng.uniform(0.5, 0.95, n_lines)
        if s == 0:
            ac[0] = 1.05  # one real overload so the miss count is defined
        d = {"state": s, "line": np.arange(n_lines), "ac": ac, "ac_prev": ac * 0.98, "prev1": ac + 1e-3}
        for st in rq.STAGES:
            for r in ("o", "t"):
                d[f"{st}_{r}"] = ac + 1e-4
            d[f"genco_prev_{r}"] = ac * 0.98
        d["genco_prev_o"] = ac * 0.98
        if broken_identity and s == 1:
            d["hyb1_t"] = ac + 0.2  # 20 points off its o column
        lines.append(pd.DataFrame(d))
        row = {"state": s, "load_idx": 6000 + s, "base_key": "abc", "pool": "FRESH", "data": "synthetic", "conv_prev": True, "it_prev": 3,
               "outage_map_ok": not (broken_map and s == 2), "split_outaged": 0, "agg_check_o": 1e-5, "agg_check_t": 1e-5,
               "eq_c_ybus": 1e-8 if bad_eq else 0.0, "eq_c_S_spec": 0.0, "eq_c_gen_n_on": 0.0, "eq_i_ybus": 0.0,
               "drift_vm_c": 0.0, "drift_va_c": 0.0, "drift_vm_i": 0.0, "drift_va_i": 0.0, "drift_dvm": 0.0, "drift_dva": 0.0,
               "ref_vm_set": 1.03, "ref_vm_raw_o": 1.01, "ref_vm_raw_t": 1.01, "ref_va_set": 0.0, "ref_va_raw_o": 0.0, "ref_va_raw_t": 0.0,
               "mis_raw_o": 10.0, "mis_raw_t": 10.0, "mis1_prev": 1.0}
        for k in ("hyb", "g"):
            for r in ("o", "t"):
                row.update({f"conv_{k}_{r}": True, f"it_{k}_{r}": 3, f"mis0_{k}_{r}": 5.0, f"mis1_{k}_{r}": 0.1})
        rows.append(row)
    return pd.concat(lines, ignore_index=True), pd.DataFrame(rows)


def _report_args(tag):
    return argparse.Namespace(grid="500", variant="tiny", seed=0, transforms="none", tag=tag, x=0.05, y=0.98, x0=0.3, miss_margin=0.01)


@pytest.mark.parametrize("kind", ["identity", "map", "eq"])
def test_report_aborts_and_writes_no_csv(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(rq, "RESULTS", tmp_path)
    L, S = _synthetic(broken_identity=kind == "identity", broken_map=kind == "map", bad_eq=kind == "eq")
    stem = rq.out_stem("500", "tiny", 0, "none", "_abort")
    L.to_parquet(f"{stem}_lines.parquet", index=False)
    S.to_parquet(f"{stem}_states.parquet", index=False)
    with pytest.raises(rq.PipelineAbort):
        rq.report(_report_args("_abort"))
    assert not list(tmp_path.glob("summary_*.csv")) and not list(tmp_path.glob("criterion_*.csv"))


def test_report_clean_identity_writes_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(rq, "RESULTS", tmp_path)
    L, S = _synthetic()
    stem = rq.out_stem("500", "tiny", 0, "none", "_ok")
    L.to_parquet(f"{stem}_lines.parquet", index=False)
    S.to_parquet(f"{stem}_states.parquet", index=False)
    rq.report(_report_args("_ok"))
    assert (tmp_path / "summary_goc500_tiny_ok.csv").exists() and (tmp_path / "criterion_goc500_tiny_ok.csv").exists()


# ---- stored-frame regressions for the two report-time holes found by the batch-5 review -------------------------------------------

def _write(tmp_path, L, S, tag):
    stem = rq.out_stem("500", "tiny", 0, "none", tag)
    L.to_parquet(f"{stem}_lines.parquet", index=False)
    S.to_parquet(f"{stem}_states.parquet", index=False)


def _no_csv(tmp_path):
    return not list(tmp_path.glob("summary_*.csv")) and not list(tmp_path.glob("criterion_*.csv"))


def test_equivalence_cells_check_fails_on_nonfinite_or_large_cell():
    _, S = _synthetic()
    rq.check_equivalence_cells(S, "unit")
    for bad in (np.nan, np.inf, -np.inf, 2e-9):
        S2 = S.copy()
        S2.loc[1, "eq_c_ybus"] = bad
        with pytest.raises(rq.PipelineAbort, match="eq_c_ybus"):
            rq.check_equivalence_cells(S2, "unit")
    S3 = S.copy()
    S3["eq_c_gen_n_on"] = np.nan  # reported, not asserted (SPEC 11.1)
    rq.check_equivalence_cells(S3, "unit")


def test_report_aborts_partial_nan_equivalence_cell(tmp_path, monkeypatch):
    """Reviewer case partial_nan_eq: one eq_c_ybus = NaN among otherwise zero rows was reduced away by S[c].max()."""
    monkeypatch.setattr(rq, "RESULTS", tmp_path)
    L, S = _synthetic()
    S.loc[0, "eq_c_ybus"] = np.nan
    _write(tmp_path, L, S, "_pnan")
    with pytest.raises(rq.PipelineAbort, match="non-finite"):
        rq.report(_report_args("_pnan"))
    assert _no_csv(tmp_path)


def test_identity_patterns_check():
    L, _ = _synthetic()
    rq.check_identity_patterns(L, "unit")
    L2 = L.copy()
    L2.loc[(L2.state == 1) & (L2.line == 0), "hyb1_t"] = np.nan
    with pytest.raises(rq.PipelineAbort, match="finite-mask"):
        rq.check_identity_patterns(L2, "unit")
    L3 = L.copy()
    L3.loc[(L3.state == 1) & (L3.line == 0), ["g1_o", "g1_t"]] = np.inf  # matching failure on both sides is a failure too
    with pytest.raises(rq.PipelineAbort, match="nonfinite_pairs.: 1"):
        rq.check_identity_patterns(L3, "unit")


@pytest.mark.parametrize("kind", ["mismatch", "pair"])
def test_report_aborts_identity_nonfinite(tmp_path, monkeypatch, kind):
    """Reviewer case identity_nan_mismatch: one none-control hyb1_t = NaN with a finite hyb1_o was reported as identity 0 -> OK."""
    monkeypatch.setattr(rq, "RESULTS", tmp_path)
    L, S = _synthetic()
    sel = (L.state == 1) & (L.line == 0)
    L.loc[sel, ["hyb1_t"] if kind == "mismatch" else ["hyb1_o", "hyb1_t"]] = np.nan
    _write(tmp_path, L, S, "_inan")
    with pytest.raises(rq.PipelineAbort, match="identity control"):
        rq.report(_report_args("_inan"))
    assert _no_csv(tmp_path)


@pytest.mark.parametrize("case", ["partial_nan_eq", "identity_nan_mismatch", "clean"])
def test_cli_report_exit_code_on_stored_frames(tmp_path, case):
    """The reviewer's probe: a copy of the runner CLI on synthetic parquets must exit 2 with no CSV (or 0 with both CSVs when clean)."""
    root = Path(rq.ROOT)
    (tmp_path / "results").mkdir()
    shutil.copyfile(Path(rq.__file__), tmp_path / "run_equiv.py")
    L, S = _synthetic()
    if case == "partial_nan_eq":
        S.loc[0, "eq_c_ybus"] = np.nan
    if case == "identity_nan_mismatch":
        L.loc[(L.state == 1) & (L.line == 0), "hyb1_t"] = np.nan
    stem = tmp_path / "results/goc500_tiny_seed0_none_probe"
    L.to_parquet(f"{stem}_lines.parquet", index=False)
    S.to_parquet(f"{stem}_states.parquet", index=False)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(root / "analysis"), str(root / "work/equiv")]))
    p = subprocess.run([sys.executable, str(tmp_path / "run_equiv.py"), "report", "--grid", "500", "--variant", "tiny", "--seed", "0",
                        "--transforms", "none", "--tag", "_probe"], cwd=tmp_path, env=env, capture_output=True, text=True)
    csv = sorted(x.name for x in (tmp_path / "results").glob("*.csv"))
    if case == "clean":
        assert p.returncode == 0 and csv == ["criterion_goc500_tiny_probe.csv", "summary_goc500_tiny_probe.csv"], p.stderr
    else:
        assert p.returncode == 2 and not csv and "PIPELINE BROKEN - ABORT" in p.stderr, (p.returncode, csv, p.stderr)

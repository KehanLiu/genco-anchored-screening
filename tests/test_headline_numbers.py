"""Regression tests for the three gates of `scripts/headline_numbers.py --check`.

Each test points the script at a scratch copy of results/ (csv and json only; no parquet, no pyarrow) and breaks
one thing: a table value, a table file, a table cell, the claim list, or the expected snapshot. CI runs them with
`python -m pytest tests -q` after installing gridtrust (numpy, scipy, pandas) and pytest.
"""
import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "headline_numbers.py"
N_CLAIMS = 274  # the registered claims; bump it together with the claim list and headline_numbers_expected.csv


def load_script():
    spec = importlib.util.spec_from_file_location("headline_numbers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hn(tmp_path, monkeypatch):
    """The script with ROOT, RES, OUT and EXPECTED pointed at a scratch copy of results/."""
    res = tmp_path / "results"
    shutil.copytree(ROOT / "results", res, ignore=shutil.ignore_patterns("headline_numbers.csv"))
    mod = load_script()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "RES", res)
    monkeypatch.setattr(mod, "OUT", res / "headline_numbers.csv")
    monkeypatch.setattr(mod, "EXPECTED", res / "headline_numbers_expected.csv")
    return mod


def run_check(hn, capsys):
    code = hn.main(["--check", "--quiet"])
    out = capsys.readouterr().out
    gates = [g for g in ("registered value", "missing claim", "expected snapshot") if f"[{g}]" in out]
    return code, out, gates


def written_rows(hn):
    return pd.read_csv(hn.OUT).set_index("claim")


def test_passes_on_the_committed_results(hn, capsys):
    code, out, gates = run_check(hn, capsys)
    assert code == 0 and gates == []
    assert "check passed" in out
    assert f"{N_CLAIMS} registered claims: {N_CLAIMS} equal their registered write-up value, 0 differ, 0 MISSING or FAILED" in out
    rows = written_rows(hn)
    assert len(rows) == N_CLAIMS and rows.matches_post.all()
    assert set(rows.method) <= {"read", "recomputed", "derived"}


def test_changed_table_value_fails_registered_value_and_snapshot(hn, capsys):
    f = hn.RES / "zeroshot" / "summary.csv"
    z = pd.read_csv(f)
    m = (z.network == "case118_ieee") & (z.kind == "screen") & (z.name == "prev1")
    assert m.sum() == 1 and z.loc[m, "overloads_missed"].item() == 74
    z.loc[m, "overloads_missed"] = 75
    z.to_csv(f, index=False)
    code, out, gates = run_check(hn, capsys)
    assert code == 1 and gates == ["registered value", "expected snapshot"]
    assert "CHECK FAILED (registered value, expected snapshot)" in out
    assert "main_118_prev_missed: registered 74, table gives 75" in out
    assert "main_118_prev_missed: expected 74, got 75" in out
    assert not written_rows(hn).loc["main_118_prev_missed", "matches_post"]


def test_absent_table_fails_missing_claim_and_snapshot(hn, capsys):
    (hn.RES / "embedding" / "embedding_auroc.csv").unlink()
    code, out, gates = run_check(hn, capsys)
    assert code == 1 and gates == ["missing claim", "expected snapshot"]
    assert "embedding_FAILED: FAILED FileNotFoundError" in out
    assert "emb_auroc_nout_n3: expected 1, got None" in out
    rows = written_rows(hn)
    assert rows.loc["embedding_FAILED", "method"] == "FAILED"
    assert "emb_auroc_n3" not in rows.index
    assert f"{N_CLAIMS - 4} registered claims" in out  # the four embedding claims are gone, the FAILED row is not a claim


def test_empty_cell_is_a_missing_claim(hn, capsys):
    f = hn.RES / "embedding" / "embedding_auroc.csv"
    e = pd.read_csv(f)
    e.loc[e.score == "n_out", "n3"] = float("nan")
    e.to_csv(f, index=False)
    code, out, gates = run_check(hn, capsys)
    assert code == 1 and gates == ["missing claim", "expected snapshot"]
    assert "emb_auroc_nout_n3: MISSING" in out
    assert "emb_auroc_nout_n3: expected 1, got MISSING" in out
    assert written_rows(hn).loc["emb_auroc_nout_n3", "method"] == "MISSING"


def test_removed_claims_fail_the_snapshot_only(hn, capsys, monkeypatch):
    monkeypatch.setattr(hn, "SECTIONS", tuple(fn for fn in hn.SECTIONS if fn is not hn.embedding))
    code, out, gates = run_check(hn, capsys)
    assert code == 1 and gates == ["expected snapshot"]
    assert "[expected snapshot] 4 claim(s)" in out
    assert "emb_auroc_nout_n3: expected 1, got None" in out
    assert f"{N_CLAIMS - 4} registered claims: {N_CLAIMS - 4} equal" in out


def test_changed_expected_row_fails_the_snapshot_only(hn, capsys):
    exp = pd.read_csv(hn.EXPECTED, dtype=str)
    assert exp.loc[exp.claim == "main_118_overloads", "value"].item() == "6986"
    exp.loc[exp.claim == "main_118_overloads", "value"] = "6987"
    exp.to_csv(hn.EXPECTED, index=False)
    code, out, gates = run_check(hn, capsys)
    assert code == 1 and gates == ["expected snapshot"]
    assert "[expected snapshot] 1 claim(s)" in out
    assert "main_118_overloads: expected 6987, got 6986" in out
    assert f"{N_CLAIMS} registered claims: {N_CLAIMS} equal" in out


def test_missing_expected_file_fails(hn, capsys):
    hn.EXPECTED.unlink()
    code, out, gates = run_check(hn, capsys)
    assert code == 1 and gates == ["expected snapshot"]
    assert "run with --write-expected first" in out


def test_without_check_the_exit_code_stays_zero(hn, capsys):
    """Plain runs are reports; only --check is a gate."""
    (hn.RES / "embedding" / "embedding_auroc.csv").unlink()
    assert hn.main(["--quiet"]) == 0
    assert "CHECK FAILED" not in capsys.readouterr().out

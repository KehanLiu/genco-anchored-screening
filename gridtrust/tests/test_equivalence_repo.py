"""Equivalence with the analysis scripts of the repository this package was extracted from.

Skipped unless the repository's scripts and stored result files are next to the package.
Nothing here starts Julia or loads torch: analysis_warmstart imports `engine` (which does), so a
stub module stands in for it while the Newton class is imported.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from gridtrust import conformal as cf
from gridtrust.newton import PowerFlowProblem

pd = pytest.importorskip("pandas")
ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "work" / "hybrid_case118_ieee_base.parquet"
GOC = ROOT / "work" / "goc500"
RAW = GOC / "data" / "case500_goc" / "raw"
SCENARIO = 7


# ----------------------------------------------------------------- conformal
@pytest.fixture(scope="module")
def sweep():
    if not (SWEEP.exists() and (ROOT / "analysis_conformal.py").exists()):
        pytest.skip("repository sweep / analysis_conformal.py not present")
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(ROOT))
    try:
        import analysis_conformal as ac
    finally:
        sys.path.remove(str(ROOT))
    data, _, n_states = ac.load(SWEEP)
    fit, cal_keys, test_keys = ac.split_keys(n_states.index, 0)
    cal, test = data[data.outages.isin(fit | cal_keys)], data[data.outages.isin(test_keys)]
    return ac, cal, test, int(n_states[n_states.index.isin(test_keys)].sum())


@pytest.mark.parametrize("surrogate,kind,alpha", [
    ("genco_d", "shift", 0.01), ("genco_d", "shift", 0.05), ("genco", "global", 0.05), ("dc_d", "shift", 0.02), ("dc", "global", 0.10),
])
def test_conformal_matches_analysis_script(sweep, surrogate, kind, alpha):
    ac, cal, test, n_test = sweep
    raw = surrogate.removesuffix("_d")

    def mine(frame):
        if kind == "global":
            sigma = None
        else:
            sigma = cf.shift_sigma(frame[raw], frame[f"{raw}0"], floor=ac.SHIFT_FLOOR)
        pred = cf.anchored(frame[raw], frame[f"{raw}0"], frame.ac0) if surrogate.endswith("_d") else frame[raw].to_numpy()
        return pred, sigma

    pred_cal, sigma_cal = mine(cal)
    pred_test, sigma_test = mine(test)
    np.testing.assert_array_equal(pred_cal, cal[surrogate].to_numpy())
    width_ref, n_ref = ac.conformal_width(cal, surrogate, ac.sigma_of(cal, surrogate, kind, None), alpha)
    width, n = cf.calibrate(pred_cal, cal.ac, limit=ac.OVERLOAD, alpha=alpha, sigma=sigma_cal, return_n=True)
    assert (width, n) == (width_ref, n_ref) and np.isfinite(width)
    ref = ac.evaluate(test, n_test, surrogate, ac.sigma_of(test, surrogate, kind, None), width_ref)
    out = cf.evaluate(pred_test, test.ac, test.state, width, limit=ac.OVERLOAD, sigma=sigma_test, n_states=n_test)
    assert set(out) == set(ref)
    for key, value in ref.items():
        np.testing.assert_equal(out[key], value, err_msg=key)


# ----------------------------------------------------------------- Newton
@pytest.fixture(scope="module")
def scenario():
    if not (RAW / "bus_data.parquet").exists():
        pytest.skip("stored GOC-500 scenarios not present")
    pytest.importorskip("pyarrow")
    flt = [("scenario", "==", SCENARIO)]
    bus = pd.read_parquet(RAW / "bus_data.parquet", filters=flt).sort_values("bus").reset_index(drop=True)
    branch = pd.read_parquet(RAW / "branch_data.parquet", filters=flt).sort_values("idx").reset_index(drop=True)
    y = {k: branch[f"{k}_r"].to_numpy() + 1j * branch[f"{k}_i"].to_numpy() for k in ("Yff", "Yft", "Ytf", "Ytt")}
    kind = 1 * bus.PQ.to_numpy(int) + 2 * bus.PV.to_numpy(int) + 3 * bus.REF.to_numpy(int)
    ref = int(np.flatnonzero(kind == 3)[0])
    prob = PowerFlowProblem.from_admittance(
        kind, bus.Vm.to_numpy(float), np.radians(float(bus.Va[ref])),
        (bus.Pg - bus.Pd).to_numpy() / 100.0, (bus.Qg - bus.Qd).to_numpy() / 100.0,
        branch.from_bus.to_numpy(int), branch.to_bus.to_numpy(int), y["Yff"], y["Yft"], y["Ytf"], y["Ytt"],
        shunt=bus.GS.to_numpy() + 1j * bus.BS.to_numpy(), rate=branch.rate_a.to_numpy() / 100.0,
        status=branch.br_status.to_numpy() == 1, base_mva=100.0)
    return prob, bus, branch


def test_newton_matches_saved_results(scenario):
    """Against what work/goc500/02_newton.py stored for this scenario (flat and DC start need no GENCO)."""
    prob, bus, branch = scenario
    states, lines = GOC / "states_newton_tiny_seed0.parquet", GOC / "lines_newton_tiny_seed0.parquet"
    if not (states.exists() and lines.exists()):
        pytest.skip("stored Newton results not present")
    row = pd.read_parquet(states, filters=[("state", "==", SCENARIO)]).iloc[0]
    saved = pd.read_parquet(lines, filters=[("state", "==", SCENARIO)]).set_index("line")
    v_ac = bus.Vm.to_numpy(float) * np.exp(1j * np.radians(bus.Va.to_numpy(float)))
    assert prob.mismatch_mw(v_ac) == pytest.approx(row.mis_ac_stored, abs=1e-6)
    for name, v0 in (("flat", prob.start()), ("dc", prob.start(va=np.radians(bus.Va_dc.to_numpy(float))))):
        run = prob.solve(v0)
        assert (run["iterations"], run["converged"]) == (row[f"it_{name}"], row[f"conv_{name}"])
        assert run["mis0"] == pytest.approx(row[f"mis0_{name}"], rel=1e-9)
        assert prob.mismatch_mw(run["steps"][0]) == pytest.approx(row[f"mis1_{name}"], rel=1e-8)
        assert prob.mismatch_mw(run["steps"][1]) == pytest.approx(row[f"mis2_{name}"], rel=1e-7)
        assert float(np.abs(run["V"] - v_ac).max()) == pytest.approx(row[f"dv_{name}"], abs=1e-9)
        one = prob.loading(run["steps"][0])
        np.testing.assert_allclose(one[saved.index.to_numpy()], saved[f"{name}1"].to_numpy(), rtol=2e-6, atol=1e-7)  # stored as float32
    np.testing.assert_allclose(prob.loading(v_ac)[saved.index.to_numpy()], saved.ac.to_numpy(), rtol=2e-6, atol=1e-7)


def test_newton_matches_analysis_class(scenario):
    """Against analysis_warmstart.Problem itself (built by 02_newton.make_problem), one and two steps from a perturbed start."""
    prob, bus, branch = scenario
    if "juliacall" in sys.modules or not (ROOT / "analysis_warmstart.py").exists():
        pytest.skip("analysis_warmstart.py not present")
    pytest.importorskip("gridfm_datakit")
    stub = "engine" not in sys.modules
    sys.path.insert(0, str(ROOT))
    try:
        if stub:
            sys.modules["engine"] = types.ModuleType("engine")  # the real one starts Julia; Problem does not use it
        spec = importlib.util.spec_from_file_location("goc_newton", GOC / "02_newton.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ref = mod.make_problem(bus, branch)
    finally:
        sys.path.remove(str(ROOT))
        if stub:
            sys.modules.pop("engine", None)
            sys.modules.pop("analysis_warmstart", None)
    assert "juliacall" not in sys.modules
    live = (branch.br_status == 1).to_numpy()
    rng = np.random.default_rng(0)
    vm = bus.Vm.to_numpy(float) + rng.normal(0, 0.01, len(bus))     # a surrogate-sized error
    va = np.radians(bus.Va.to_numpy(float)) + rng.normal(0, 0.01, len(bus))
    v0, v0_ref = prob.start(vm_pq=vm, va=va), ref.start(vm_pq=vm, va=va)
    np.testing.assert_array_equal(v0, v0_ref)
    np.testing.assert_allclose(prob.mismatch(v0), ref.mismatch(v0_ref), rtol=0, atol=1e-12)
    assert abs(prob.jacobian(v0) - ref.jacobian(v0_ref)).max() < 1e-10
    run, run_ref = prob.solve(v0), ref.newton(v0_ref)
    assert (run["iterations"], run["converged"]) == (run_ref["iterations"], run_ref["converged"])
    for k in (0, 1):
        np.testing.assert_allclose(run["steps"][k], run_ref["steps"][k], rtol=0, atol=1e-11)
        np.testing.assert_allclose(prob.loading(run["steps"][k])[live], ref.loading(run_ref["steps"][k]), rtol=0, atol=1e-10)
    res = prob.refine(vm, va, steps=1)
    np.testing.assert_allclose(res.V, run_ref["steps"][0], rtol=0, atol=1e-11)
    assert res.mismatch_mw == pytest.approx(float(np.abs(ref.mismatch(run_ref["steps"][0])).max() * 100), rel=1e-6)

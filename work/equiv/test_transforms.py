"""pytest work/equiv/test_transforms.py  (uses development scenario 0 of work/goc500/data; no engine / Julia import)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transforms as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "work" / "goc500" / "data" / "case500_goc" / "raw"
TOL = 1e-10


@pytest.fixture(scope="module")
def orig():
    if not RAW.exists():
        pytest.skip(f"{RAW} missing")
    flt = [("scenario", "==", 0)]
    bus, gen, branch = (pd.read_parquet(RAW / f"{t}_data.parquet", filters=flt) for t in ("bus", "gen", "branch"))
    q = gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
    bus = bus.merge(q, on="bus", how="left").fillna(0)
    return tf.Rep.original(bus, gen, branch)


def v_ac(rep):
    """Stored AC voltages of the representation's buses, in its own bus order."""
    b = rep.bus
    return b.Vm.to_numpy(float) * np.exp(1j * np.radians(b.Va.to_numpy(float)))


def loading_of(rep, V_rep, rate_orig):
    Sf, St = tf.branch_flows(rep, V_rep)
    live = (rep.branch.br_status == 1).to_numpy()
    return tf.loading_to_physical(rep.phys[live], Sf.real, Sf.imag, St.real, St.imag, rate_orig)


def test_identity_is_equivalent(orig):
    d = tf.verify_equivalence(orig, orig)
    assert max(v for k, v in d.items()) == 0.0


def test_split_one_branch(orig):
    live = orig.branch[orig.branch.br_status == 1]
    pid = int(live.loc[live.rate_a > 0, "idx"].iloc[3])
    rep = tf.split_branches(orig, [pid], k=2)
    assert len(rep.branch) == len(orig.branch) + 1
    assert (rep.phys == pid).sum() == 2
    d = tf.verify_equivalence(orig, rep)
    assert d["ybus"] <= TOL and d["S_spec"] <= TOL and d["rate_sum"] <= TOL
    rows = rep.branch[rep.phys == pid]
    o = orig.branch[orig.phys == pid].iloc[0]
    assert np.allclose(rows.r, 2 * o.r) and np.allclose(rows.x, 2 * o.x) and np.allclose(rows.b, o.b / 2)
    assert np.allclose(rows.rate_a, o.rate_a / 2) and (rows.tap == o.tap).all()
    # at identical voltages every copy carries half the flow and the summed loading is the original one
    rate = orig.branch.rate_a.to_numpy()
    lo, lr = loading_of(orig, v_ac(orig), rate), loading_of(rep, v_ac(rep), rate)
    assert np.nanmax(np.abs(lo - lr)) <= TOL
    Sf, _ = tf.branch_flows(rep, v_ac(rep))
    live_r = (rep.branch.br_status == 1).to_numpy()
    copies = np.flatnonzero(rep.phys[live_r] == pid)
    Sf_o, _ = tf.branch_flows(orig, v_ac(orig))
    j = np.flatnonzero(orig.phys[(orig.branch.br_status == 1).to_numpy()] == pid)[0]
    assert np.allclose(Sf[copies], Sf_o[j] / 2)


def test_split_all_branches_k3(orig):
    rep = tf.split_branches(orig, orig.phys, k=3)
    assert len(rep.branch) == 3 * len(orig.branch)
    d = tf.verify_equivalence(orig, rep)
    assert d["ybus"] <= TOL and d["rate_sum"] <= TOL
    rate = orig.branch.rate_a.to_numpy()
    assert np.nanmax(np.abs(loading_of(orig, v_ac(orig), rate) - loading_of(rep, v_ac(rep), rate))) <= TOL
    # loading of the stored AC voltages reproduces the stored flows (sanity of the flow formula itself)
    br = orig.branch[(orig.branch.br_status == 1) & (orig.branch.rate_a > 0)]
    stored = np.maximum(np.hypot(br.pf, br.qf), np.hypot(br.pt, br.qt)) / br.rate_a
    assert np.nanmax(np.abs(loading_of(orig, v_ac(orig), rate)[br.idx.to_numpy()] - stored.to_numpy())) < 1e-6


def test_split_generators(orig):
    on = orig.gen[(orig.gen.in_service == 1) & (orig.gen.is_slack_gen == 0)]
    rep = tf.split_generators(orig, on.idx)
    assert len(rep.gen) == len(orig.gen) + len(on)
    d = tf.verify_equivalence(orig, rep)
    assert d["gen_p"] <= TOL and d["gen_q"] <= TOL and d["ybus"] == 0.0
    assert d["gen_n_on"] == len(on[on.bus == on.bus.mode()[0]]) or d["gen_n_on"] >= 1
    halves = rep.gen[rep.gen_phys == int(on.idx.iloc[0])]
    assert len(halves) == 2 and np.allclose(halves.p_mw, on.p_mw.iloc[0] / 2) and np.allclose(halves.max_p_mw, on.max_p_mw.iloc[0] / 2)


def test_permute_maps_back(orig):
    rep = tf.permute(orig, seed=7)
    assert (rep.bus.bus.to_numpy() == np.arange(len(rep.bus))).all()
    assert not (rep.bus_map == np.arange(len(rep.bus))).all()
    d = tf.verify_equivalence(orig, rep)
    assert d["ybus"] <= TOL and d["type"] == 0.0 and d["Va_ref"] == 0.0
    # a per-bus array of the permuted representation comes back in physical order
    assert np.array_equal(rep.to_original(rep.bus.Pd.to_numpy()), orig.bus.Pd.to_numpy())
    rate = orig.branch.rate_a.to_numpy()
    assert np.nanmax(np.abs(loading_of(orig, v_ac(orig), rate) - loading_of(rep, v_ac(rep), rate))) <= TOL


def test_composed_split_then_permute(orig):
    rep = tf.permute(tf.split_generators(tf.split_branches(orig, orig.phys[::5], k=2), orig.gen.idx[::3]), seed=3)
    d = tf.verify_equivalence(orig, rep)
    assert d["ybus"] <= TOL and d["gen_p"] <= TOL and d["S_spec"] <= TOL
    rate = orig.branch.rate_a.to_numpy()
    assert np.nanmax(np.abs(loading_of(orig, v_ac(orig), rate) - loading_of(rep, v_ac(rep), rate))) <= TOL


def test_outage_disables_all_copies(orig):
    pid = int(orig.branch[orig.branch.br_status == 1].idx.iloc[10])
    rep = tf.split_branches(orig, [pid], k=2)
    out_o = tf.apply_outage(orig, [pid])
    out_r = tf.apply_outage(rep, [pid])
    assert (out_r.branch.br_status.to_numpy()[rep.phys == pid] == 0).all()
    assert tf.outaged_physical(out_r) == tf.outaged_physical(out_o) == tf.outaged_physical(orig) | {pid}
    d = tf.verify_equivalence(out_o, out_r)
    assert d["ybus"] <= TOL and d["outaged_set"] == 0.0
    # removing a single copy is a different circuit and must be caught
    half = rep.branch.copy()
    half.loc[np.flatnonzero(rep.phys == pid)[0], "br_status"] = 0
    with pytest.raises(AssertionError):
        tf.verify_equivalence(out_o, tf.Rep(rep.bus, rep.gen, half, rep.bus_map, rep.phys, rep.gen_phys))


def test_enforce_reference(orig):
    vm = np.full(len(orig.bus), 0.5)
    va = np.zeros(len(orig.bus))
    vm2, va2 = tf.enforce_reference(vm, va, orig.bus)
    ref = orig.bus.REF.to_numpy() == 1
    assert ref.sum() == 1 and np.allclose(vm2[ref], orig.bus.Vm[ref]) and np.allclose(va2[ref], orig.bus.Va[ref])
    assert (vm2[~ref] == 0.5).all() and (vm == 0.5).all()

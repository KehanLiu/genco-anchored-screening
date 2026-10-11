"""Equivalent-circuit rewrites of the per-scenario datakit tables GENCO consumes (bus / gen / branch
frames with the columns of work/goc<N>/data/case<N>_goc/raw/*.parquet).

A representation (`Rep`) is a set of tables plus two maps back to the physical circuit:
    bus_map[i]   row/id of physical bus i in this representation (identity unless permuted)
    phys[j]      physical branch idx of branch row j (a split branch has k rows with the same phys)
    gen_phys[j]  physical generator idx of generator row j

Rewrites (all return a new Rep, inputs untouched):
    split_branches(rep, branch_ids, k)   k parallel copies: Y columns / k (exact), r,x * k, b / k, rate_a / k, flows / k;
                                         tap, shift, angle limits and br_status kept
    split_generators(rep, gen_ids)       two units at the same bus with half p_mw / q_mvar / P and Q limits / cp0 (cp2 doubled)
    permute(rep, seed)                   relabel buses, shuffle branch and generator rows
    apply_outage(rep, phys_ids)          every copy of an outaged physical branch is taken out of service
    verify_equivalence(orig, rep)        Ybus (mapped back), specified injections, bus types, setpoints, reference: max differences
    aggregate_flows / loading_to_physical  copy flows summed back to the physical branch, loading against the ORIGINAL rating
    enforce_reference(vm, va, bus)       known-reference-voltage control: predicted REF Vm / Va replaced by the setpoints
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, diags

YCOLS = ["Yff_r", "Yff_i", "Yft_r", "Yft_i", "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i"]
FLOW_COLS = ["pf", "qf", "pt", "qt", "pf_dc", "pt_dc"]
GEN_HALVED = ["p_mw", "q_mvar", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar", "cp0_eur", "p_mw_dc"]
BASE = 100.0


@dataclass
class Rep:
    bus: pd.DataFrame
    gen: pd.DataFrame
    branch: pd.DataFrame
    bus_map: np.ndarray
    phys: np.ndarray
    gen_phys: np.ndarray

    @classmethod
    def original(cls, bus, gen, branch):
        """Identity representation of stored tables (bus sorted by id, branch by idx, gen by idx)."""
        bus = bus.sort_values("bus").reset_index(drop=True)
        gen = gen.sort_values("idx").reset_index(drop=True)
        branch = branch.sort_values("idx").reset_index(drop=True)
        assert (bus.bus.to_numpy() == np.arange(len(bus))).all()
        return cls(bus, gen, branch, np.arange(len(bus)), branch.idx.to_numpy().copy(), gen.idx.to_numpy().copy())

    def tables(self):
        return self.bus, self.gen, self.branch

    def to_original(self, per_bus):
        """Per-bus array (or 2-d with buses on axis 0) of this representation in physical bus order."""
        return np.asarray(per_bus)[self.bus_map]

    def copies_of(self, phys_ids):
        return np.isin(self.phys, np.asarray(list(phys_ids)))


# ----------------------------------------------------------------------------------------------- rewrites
def split_branches(rep, branch_ids, k=2):
    """Each selected physical branch becomes k parallel copies placed where the original row was."""
    branch_ids = np.asarray(sorted(set(int(i) for i in branch_ids)))
    k = int(k)
    assert k >= 1
    sel = np.isin(rep.phys, branch_ids)
    if not sel.any() or k == 1:
        return replace(rep, bus=rep.bus.copy(), gen=rep.gen.copy(), branch=rep.branch.copy())
    take = np.repeat(np.arange(len(rep.branch)), np.where(sel, k, 1))
    branch = rep.branch.iloc[take].reset_index(drop=True)
    split = sel[take]
    branch.loc[split, YCOLS] = branch.loc[split, YCOLS] / k          # exact: the sum over copies is the stored admittance
    branch.loc[split, ["r", "x"]] = branch.loc[split, ["r", "x"]] * k
    branch.loc[split, "b"] = branch.loc[split, "b"] / k
    branch.loc[split, "rate_a"] = branch.loc[split, "rate_a"] / k
    for c in FLOW_COLS:
        if c in branch:
            branch.loc[split, c] = branch.loc[split, c] / k
    branch["idx"] = np.arange(len(branch))
    return Rep(rep.bus.copy(), rep.gen.copy(), branch, rep.bus_map.copy(), rep.phys[take], rep.gen_phys.copy())


def split_generators(rep, gen_ids):
    """Each selected physical generator becomes two units at the same bus with half the dispatch and limits."""
    gen_ids = np.asarray(sorted(set(int(i) for i in gen_ids)))
    sel = np.isin(rep.gen_phys, gen_ids)
    if not sel.any():
        return replace(rep, bus=rep.bus.copy(), gen=rep.gen.copy(), branch=rep.branch.copy())
    take = np.repeat(np.arange(len(rep.gen)), np.where(sel, 2, 1))
    gen = rep.gen.iloc[take].reset_index(drop=True)
    split = sel[take]
    for c in GEN_HALVED:
        if c in gen:
            gen.loc[split, c] = gen.loc[split, c] / 2
    if "cp2_eur_per_mw2" in gen:
        gen.loc[split, "cp2_eur_per_mw2"] = gen.loc[split, "cp2_eur_per_mw2"] * 2  # same total cost at the same total output
    gen["idx"] = np.arange(len(gen))
    return Rep(rep.bus.copy(), gen, rep.branch.copy(), rep.bus_map.copy(), rep.phys.copy(), rep.gen_phys[take])


def permute(rep, seed):
    """Relabel buses by a random permutation and shuffle branch and generator rows consistently."""
    rng = np.random.default_rng(seed)
    n = len(rep.bus)
    perm = rng.permutation(n)  # current id -> new id
    bus = rep.bus.copy()
    bus["bus"] = perm[bus.bus.to_numpy()]
    bus = bus.sort_values("bus").reset_index(drop=True)
    gen = rep.gen.copy()
    gen["bus"] = perm[gen.bus.to_numpy()]
    gorder = rng.permutation(len(gen))
    gen = gen.iloc[gorder].reset_index(drop=True)
    branch = rep.branch.copy()
    branch["from_bus"] = perm[branch.from_bus.to_numpy()]
    branch["to_bus"] = perm[branch.to_bus.to_numpy()]
    border = rng.permutation(len(branch))
    branch = branch.iloc[border].reset_index(drop=True)
    return Rep(bus, gen, branch, perm[rep.bus_map], rep.phys[border], rep.gen_phys[gorder])


def apply_outage(rep, phys_ids):
    """Take every copy of the given physical branches out of service (an outage of a split branch removes all copies)."""
    branch = rep.branch.copy()
    out = rep.copies_of(phys_ids)
    branch["br_status"] = np.where(out, 0, branch.br_status.to_numpy()).astype(branch.br_status.dtype)
    return replace(rep, branch=branch)


def outaged_physical(rep):
    """Physical branches out of service in this representation; asserts the copies of a branch agree on status."""
    status = pd.Series(rep.branch.br_status.to_numpy(), index=rep.phys)
    agg = status.groupby(level=0).agg(["min", "max"])
    assert (agg["min"] == agg["max"]).all(), "copies of one physical branch disagree on br_status"
    return set(agg.index[agg["min"] == 0].tolist())


def enforce_reference(vm, va, bus):
    """Known-reference-voltage control: the predicted REF Vm (p.u.) and Va (same unit as bus.Va) replaced by the stored setpoints.
    `bus` is the table of the same representation as vm/va. Returns copies."""
    vm, va = np.array(vm, dtype=float), np.array(va, dtype=float)
    ref = bus.REF.to_numpy() == 1
    vm[ref] = bus.Vm.to_numpy(float)[ref]
    va[ref] = bus.Va.to_numpy(float)[ref]
    return vm, va


# ----------------------------------------------------------------------------------------------- physics checks
def ybus(rep, in_original_labels=True):
    """Sparse bus admittance matrix of the in-service branches plus bus shunts; rows/cols in physical bus labels if asked."""
    bus, br = rep.bus, rep.branch[rep.branch.br_status == 1]
    n = len(bus)
    f, t = br.from_bus.to_numpy(int), br.to_bus.to_numpy(int)
    y = {k: br[f"{k}_r"].to_numpy() + 1j * br[f"{k}_i"].to_numpy() for k in ("Yff", "Yft", "Ytf", "Ytt")}
    Y = coo_matrix((np.r_[y["Yff"], y["Yft"], y["Ytf"], y["Ytt"]], (np.r_[f, f, t, t], np.r_[f, t, f, t])), shape=(n, n)).tocsr()
    Y = Y + diags(bus.GS.to_numpy() + 1j * bus.BS.to_numpy())
    if in_original_labels:
        Y = Y[rep.bus_map][:, rep.bus_map]
    return Y.tocsr()


def _per_bus(rep):
    """Specified quantities per physical bus (arrays in physical bus order)."""
    bus = rep.bus.iloc[rep.bus_map]
    n = len(bus)
    gen = rep.gen[rep.gen.in_service == 1]
    inv = np.empty(n, int)
    inv[rep.bus_map] = np.arange(n)  # representation id -> physical id
    gbus = inv[gen.bus.to_numpy(int)]
    out = {
        "S_spec": ((bus.Pg - bus.Pd).to_numpy() + 1j * (bus.Qg - bus.Qd).to_numpy()) / BASE,
        "Pd": bus.Pd.to_numpy(), "Qd": bus.Qd.to_numpy(),
        "type": (bus.PQ + 2 * bus.PV + 3 * bus.REF).to_numpy(),
        "Vm_set": np.where(bus.PQ.to_numpy() == 1, 0.0, bus.Vm.to_numpy(float)),
        "Va_ref": np.where(bus.REF.to_numpy() == 1, bus.Va.to_numpy(float), 0.0),
        "gen_p": np.bincount(gbus, weights=gen.p_mw.to_numpy(), minlength=n),
        "gen_q": np.bincount(gbus, weights=gen.q_mvar.to_numpy(), minlength=n),
        "gen_n_on": np.bincount(gbus, minlength=n).astype(float),
        "q_limits": (bus.min_q_mvar.to_numpy() + 1j * bus.max_q_mvar.to_numpy()) if "min_q_mvar" in bus else np.zeros(n),
        "shunt": bus.GS.to_numpy() + 1j * bus.BS.to_numpy(),
    }
    return out


def verify_equivalence(orig, rep, tol=1e-9):
    """Max absolute differences between the circuit equations of two representations of the same physical state.
    Raises AssertionError when any of them exceeds `tol` (gen_n_on is reported but not asserted: splitting changes it)."""
    assert len(orig.bus) == len(rep.bus), "different number of buses"
    d = {}
    dy = (ybus(orig) - ybus(rep))
    d["ybus"] = float(np.abs(dy.data).max()) if dy.nnz else 0.0
    a, b = _per_bus(orig), _per_bus(rep)
    for k in a:
        d[k] = float(np.abs(a[k] - b[k]).max())
    rate_o = pd.Series(orig.branch.rate_a.to_numpy(), index=orig.phys).groupby(level=0).sum()
    rate_r = pd.Series(rep.branch.rate_a.to_numpy(), index=rep.phys).groupby(level=0).sum()
    d["rate_sum"] = float((rate_o - rate_r.reindex(rate_o.index)).abs().max())
    d["outaged_set"] = 0.0 if outaged_physical(orig) == outaged_physical(rep) else 1.0
    bad = {k: v for k, v in d.items() if k != "gen_n_on" and not (v <= tol)}
    assert not bad, f"representations differ: {bad}"
    return d


# ----------------------------------------------------------------------------------------------- flows back to the physical branch
def aggregate_flows(phys_live, p_f, q_f, p_t, q_t, n_phys):
    """Sum per-copy terminal flows (arrays over the in-service rows, table order) back to the physical branch.
    Returns arrays indexed by physical idx (NaN for branches without an in-service copy)."""
    phys_live = np.asarray(phys_live, int)
    present = np.bincount(phys_live, minlength=n_phys) > 0
    out = {}
    for name, v in (("p_f", p_f), ("q_f", q_f), ("p_t", p_t), ("q_t", q_t)):
        s = np.bincount(phys_live, weights=np.asarray(v, float), minlength=n_phys)
        out[name] = np.where(present, s, np.nan)
    return out


def loading_to_physical(phys_live, p_f, q_f, p_t, q_t, rate_orig):
    """Fractional loading per physical branch against the ORIGINAL rating (MVA units for flows and rate; NaN if unrated)."""
    agg = aggregate_flows(phys_live, p_f, q_f, p_t, q_t, len(rate_orig))
    rate = np.where(np.asarray(rate_orig, float) > 0, rate_orig, np.nan)
    with np.errstate(all="ignore"):
        return np.maximum(np.hypot(agg["p_f"], agg["q_f"]), np.hypot(agg["p_t"], agg["q_t"])) / rate


def engine_loading_to_physical(rep, branch_out, rate_orig):
    """Genco.solve's branch table (in-service from-ends in table order, then to-ends; flows in MW/MVAr) -> loading per physical idx."""
    live = (rep.branch.br_status == 1).to_numpy()
    n = int(live.sum())
    assert len(branch_out) == 2 * n
    p, q = branch_out.p.to_numpy(float), branch_out.q.to_numpy(float)
    return loading_to_physical(rep.phys[live], p[:n], q[:n], p[n:], q[n:], rate_orig)


def branch_flows(rep, V_rep):
    """Terminal complex flows (MVA) of the in-service rows at the given voltages (in this representation's bus order)."""
    br = rep.branch[rep.branch.br_status == 1]
    n, m = len(rep.bus), len(br)
    f, t = br.from_bus.to_numpy(int), br.to_bus.to_numpy(int)
    y = {k: br[f"{k}_r"].to_numpy() + 1j * br[f"{k}_i"].to_numpy() for k in ("Yff", "Yft", "Ytf", "Ytt")}
    rows = np.r_[np.arange(m), np.arange(m)]
    Yf = csr_matrix((np.r_[y["Yff"], y["Yft"]], (rows, np.r_[f, t])), shape=(m, n))
    Yt = csr_matrix((np.r_[y["Ytf"], y["Ytt"]], (rows, np.r_[f, t])), shape=(m, n))
    Sf = V_rep[f] * np.conj(Yf @ V_rep) * BASE
    St = V_rep[t] * np.conj(Yt @ V_rep) * BASE
    return Sf, St

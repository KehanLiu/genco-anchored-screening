"""Newton refinement of predicted voltages -- DRAFT PROTOTYPE, NOT SUBMITTED ANYWHERE.

Patch-style prototype written against gridfm-graphkit 0.9.0.3. If it were proposed
upstream it would live next to ``ComputeBranchFlow`` / ``ComputeNodeInjection`` /
``ComputeNodeResiduals`` (``gridfm_graphkit/models/utils.py``) or in a new
``gridfm_graphkit/tasks/newton_refine.py`` beside ``tasks/utils.py``. See
``upstream/PROPOSAL.md``.

What it does
------------
Takes one *unbatched* ``HeteroData`` power-flow graph and the model output for it,
both already brought back to physical units, i.e. after

    normalizer.inverse_transform(data)
    normalizer.inverse_output(output, data)
    RemovePFMask()(data)

(the three calls at the top of ``PowerFlowTask.test_step`` / ``predict_step``), and
runs ``steps`` full Newton-Raphson steps (polar form, MATPOWER-style dS/dVa, dS/dVm)
on the AC power-flow equations of that graph, started from the predicted voltages.
It returns corrected voltages, the flow and loading of every directed edge and the
bus power mismatch before and after each step.

Conventions taken from graphkit (all read from its source, see PROPOSAL.md)
----------------------------------------------------------------------------
* ``data["bus"].x`` columns: ``PD_H, QD_H, QG_H, VM_H, VA_H, PQ_H, PV_H, REF_H, ...,
  GS, BS`` (``datasets/globals.py``); ``data["gen"].x[:, PG_H]``.
* Model output ``output["bus"]`` columns: ``VM_OUT, VA_OUT, PG_OUT, QG_OUT``.
* After ``inverse_transform``: powers are in MW / MVAr, ``Va`` stays in RADIANS, and
  the admittances (edge columns ``YFF_TT_R .. YFT_TF_I``, bus columns ``GS``, ``BS``)
  are the case-file per-unit values times ``baseMVA_orig``. ``V * conj(Y V)`` with
  ``V`` in p.u. is therefore directly in MW / MVAr and no base is needed here.
* Directed edges: ``("bus", "connects", "bus")`` holds every in-service branch twice,
  all from-ends first and then all to-ends. Each directed edge (i -> j) carries the
  admittances of ITS OWN sending end: (Yff, Yft) on the forward copy, (Ytt, Ytf) on
  the reverse copy. So ``Ybus[i, i] += Y_self`` and ``Ybus[i, j] += Y_mutual`` over
  all directed edges, plus ``GS + j BS`` on the diagonal. This is exactly what
  ``ComputeBranchFlow`` + ``ComputeNodeInjection`` + ``compute_shunt_power`` imply.
* Sign of the mismatch: graphkit's ``ComputeNodeResiduals`` returns
  ``Pg - Pd + p_shunt - P_in`` = specified minus calculated. MATPOWER's ``F`` is the
  opposite. This module follows graphkit, so the Newton update solves
  ``J dx = +residual``.
* Masking: ``AddPFHeteroMask`` + ``ApplyMasking`` overwrite the unknowns in ``x`` with
  ``mask_value``; only the known entries are read here (Vm at PV, Va at REF, Pd, Qd,
  Qg at PQ, Pg of generators not at REF). ``AddPFHeteroMask`` also masks Vm at the
  REF bus, so the slack voltage setpoint is not in ``x``; it is taken from the
  label ``data["bus"].y`` by default (``ref_vm="label"``) or from the prediction.

Not handled: generator reactive limits (no PV -> PQ switching), step control,
islands, batched graphs. Numerics run in float64 on NumPy copies with a SciPy sparse
LU; the tensors themselves are float32, which bounds how small the mismatch can get.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import splu

from gridfm_graphkit.datasets.globals import (
    BS,
    GS,
    PD_H,
    PG_H,
    PG_OUT,
    PQ_H,
    PV_H,
    QD_H,
    QG_H,
    QG_OUT,
    RATE_A,
    REF_H,
    VA_H,
    VA_OUT,
    VM_H,
    VM_OUT,
    YFF_TT_I,
    YFF_TT_R,
    YFT_TF_I,
    YFT_TF_R,
)

BUS_EDGE = ("bus", "connects", "bus")
GEN_EDGE = ("gen", "connected_to", "bus")
# data[BUS_EDGE].static columns are [ANG_MIN, ANG_MAX, RATE_A] (datasets/graph_builder.py)
STATIC_RATE_A = 2


@dataclass
class NewtonRefineResult:
    """Outcome of ``newton_refine``. Powers in MW / MVAr, angles in radians."""

    bus_output: torch.Tensor  # [num_bus, 4] = VM_OUT, VA_OUT, PG_OUT, QG_OUT, same layout as output["bus"]
    p_flow: torch.Tensor  # [num_edges] sending-end active flow per directed edge
    q_flow: torch.Tensor  # [num_edges] sending-end reactive flow per directed edge
    loading: torch.Tensor  # [num_edges] |S| / RATE_A per directed edge, NaN where RATE_A <= 0
    residual_p: torch.Tensor  # [num_bus] specified - calculated P after the last step (0 at REF)
    residual_q: torch.Tensor  # [num_bus] specified - calculated Q after the last step (0 at PV, REF)
    max_mismatch: float  # max |residual| over the Newton equations after the last step
    mismatch_history: list[float] = field(default_factory=list)  # [before, after step 1, ...]
    steps_taken: int = 0
    ok: bool = True  # False if the Jacobian was singular or the iterate became non-finite

    @property
    def vm(self) -> torch.Tensor:
        return self.bus_output[:, VM_OUT]

    @property
    def va(self) -> torch.Tensor:
        return self.bus_output[:, VA_OUT]


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def build_ybus(data) -> csr_matrix:
    """Bus admittance matrix of one denormalised graph (case-file p.u. times baseMVA_orig)."""
    edges = data[BUS_EDGE]
    n = data["bus"].x.size(0)
    src, dst = edges.edge_index.cpu().numpy()
    attr = _np(edges.edge_attr)
    y_self = attr[:, YFF_TT_R] + 1j * attr[:, YFF_TT_I]
    y_mut = attr[:, YFT_TF_R] + 1j * attr[:, YFT_TF_I]
    bus_x = _np(data["bus"].x)
    ybus = csr_matrix((np.r_[y_self, y_mut], (np.r_[src, src], np.r_[src, dst])), shape=(n, n))
    return (ybus + diags(bus_x[:, GS] + 1j * bus_x[:, BS])).tocsr()


def _jacobian(ybus, v, pvpq, pq):
    """d(S_calc)/d(Va, Vm), reduced to the Newton unknowns (MATPOWER dSbus_dV, polar)."""
    ibus = ybus @ v
    d_v, d_i, d_n = diags(v), diags(ibus), diags(v / np.abs(v))
    ds_dva = csr_matrix(1j * d_v @ np.conj(d_i - ybus @ d_v))
    ds_dvm = csr_matrix(d_v @ np.conj(ybus @ d_n) + np.conj(d_i) @ d_n)
    return vstack(
        [
            hstack([ds_dva[pvpq][:, pvpq].real, ds_dvm[pvpq][:, pq].real]),
            hstack([ds_dva[pq][:, pvpq].imag, ds_dvm[pq][:, pq].imag]),
        ],
    ).tocsc()


def newton_refine(data, bus_output: torch.Tensor, steps: int = 1, ref_vm: str = "label") -> NewtonRefineResult:
    """Run ``steps`` full Newton steps on the AC power-flow equations, started from the prediction.

    Args:
        data: one unbatched PF ``HeteroData`` graph after ``inverse_transform`` and
            ``RemovePFMask`` (see module docstring). Not modified.
        bus_output: ``output["bus"]`` after ``inverse_output``, at least the columns
            ``VM_OUT`` and ``VA_OUT`` (Va in radians). Not modified.
        steps: number of Newton steps (0 only evaluates the mismatch of the prediction).
        ref_vm: where the slack voltage magnitude comes from, ``"label"``
            (``data["bus"].y[:, VM_H]``, the power-flow setpoint) or ``"pred"``.

    Returns:
        NewtonRefineResult.
    """
    if getattr(data["bus"], "batch", None) is not None and int(data["bus"].batch.max()) > 0:
        raise ValueError("newton_refine expects a single graph, not a batch")
    if ref_vm not in ("label", "pred"):
        raise ValueError("ref_vm must be 'label' or 'pred'")

    bus_x = _np(data["bus"].x)
    n = bus_x.shape[0]
    pq = np.flatnonzero(bus_x[:, PQ_H] == 1)
    pv = np.flatnonzero(bus_x[:, PV_H] == 1)
    ref = np.flatnonzero(bus_x[:, REF_H] == 1)
    if len(ref) == 0:
        raise ValueError("graph has no REF bus")
    pvpq = np.r_[pv, pq]
    n_ang = len(pvpq)

    # Specified injections (MW / MVAr). Only P at PV+PQ and Q at PQ enter the Newton equations.
    gen_idx, gen_bus = data[GEN_EDGE].edge_index.cpu().numpy()
    pg = np.bincount(gen_bus, weights=_np(data["gen"].x)[gen_idx, PG_H], minlength=n)
    p_spec = pg - bus_x[:, PD_H]
    q_spec = bus_x[:, QG_H] - bus_x[:, QD_H]

    # Start point: prediction at the unknowns, setpoints at the knowns.
    out = _np(bus_output)
    vm, va = out[:, VM_OUT].copy(), out[:, VA_OUT].copy()
    vm[pv] = bus_x[pv, VM_H]
    va[ref] = bus_x[ref, VA_H]
    if ref_vm == "label":
        vm[ref] = _np(data["bus"].y)[ref, VM_H]

    ybus = build_ybus(data)

    def residual(v):
        """Specified minus calculated, graphkit's sign (ComputeNodeResiduals)."""
        s_calc = v * np.conj(ybus @ v)
        return p_spec - s_calc.real, q_spec - s_calc.imag

    def stacked(res_p, res_q):
        return np.r_[res_p[pvpq], res_q[pq]]

    v = vm * np.exp(1j * va)
    f = stacked(*residual(v))
    history = [float(np.abs(f).max()) if f.size else 0.0]
    ok, taken = True, 0
    for _ in range(steps):
        try:
            with np.errstate(all="ignore"):
                dx = splu(_jacobian(ybus, v, pvpq, pq)).solve(f)  # J dx = +(spec - calc)
        except RuntimeError:  # singular Jacobian
            ok = False
            break
        va_new, vm_new = va.copy(), vm.copy()
        va_new[pvpq] += dx[:n_ang]
        vm_new[pq] += dx[n_ang:]
        v_new = vm_new * np.exp(1j * va_new)
        with np.errstate(all="ignore"):
            f_new = stacked(*residual(v_new))
        if not (np.isfinite(f_new).all() and (vm_new > 0).all()):
            ok = False
            break
        va, vm, v, f = va_new, vm_new, v_new, f_new
        taken += 1
        history.append(float(np.abs(f).max()))

    # Flows per directed edge, same formula as ComputeBranchFlow (sending end of each directed edge).
    edges = data[BUS_EDGE]
    src, dst = edges.edge_index.cpu().numpy()
    attr = _np(edges.edge_attr)
    y_self = attr[:, YFF_TT_R] + 1j * attr[:, YFF_TT_I]
    y_mut = attr[:, YFT_TF_R] + 1j * attr[:, YFT_TF_I]
    s_edge = v[src] * np.conj(y_self * v[src] + y_mut * v[dst])
    # RATE_A in edge_attr is zeroed by the PF mask until RemovePFMask restores it; .static keeps the raw MVA value.
    rate = _np(edges.static)[:, STATIC_RATE_A] if hasattr(edges, "static") else attr[:, RATE_A]
    with np.errstate(all="ignore"):
        loading = np.where(rate > 0, np.abs(s_edge) / rate, np.nan)

    # Pg at REF and Qg at PV + REF follow from the bus balance, as in PhysicsDecoderPF.
    s_calc = v * np.conj(ybus @ v)
    pg_out = pg.copy()
    pg_out[ref] = s_calc.real[ref] + bus_x[ref, PD_H]
    qg_out = bus_x[:, QG_H].copy()
    pvref = np.r_[pv, ref]
    qg_out[pvref] = s_calc.imag[pvref] + bus_x[pvref, QD_H]
    res_p = pg_out - bus_x[:, PD_H] - s_calc.real
    res_q = qg_out - bus_x[:, QD_H] - s_calc.imag

    like = dict(dtype=bus_output.dtype, device=bus_output.device)
    refined = torch.zeros((n, 4), **like)
    for col, values in ((VM_OUT, vm), (VA_OUT, va), (PG_OUT, pg_out), (QG_OUT, qg_out)):
        refined[:, col] = torch.as_tensor(values, **like)
    return NewtonRefineResult(
        bus_output=refined,
        p_flow=torch.as_tensor(s_edge.real, **like),
        q_flow=torch.as_tensor(s_edge.imag, **like),
        loading=torch.as_tensor(loading, **like),
        residual_p=torch.as_tensor(res_p, **like),
        residual_q=torch.as_tensor(res_q, **like),
        max_mismatch=history[-1],
        mismatch_history=history,
        steps_taken=taken,
        ok=ok,
    )

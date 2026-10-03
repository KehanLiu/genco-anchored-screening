"""In-process what-if engine: one grid, a classical AC solver and GENCO side by side.

A grid state is the base case with its loads scaled and some branches tripped.
Generation is re-dispatched by AC-OPF when the load changes (as in the data
GENCO was trained on); tripping branches keeps the dispatch and only re-runs
power flow. Each state is solved by datakit's AC and DC power flow (PowerModels,
via Julia) and by a pretrained GENCO checkpoint.
"""

# Julia has to be loaded before torch (juliacall warns of a possible segfault otherwise).
import juliacall  # noqa: F401

from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.process.process_network import init_julia, pf_post_processing, pf_preprocessing
from gridfm_datakit.process.solvers import run_dcpf, run_opf, run_pf
from gridfm_datakit.utils.idx_brch import RATE_A as RATE_A_COLUMN
from gridfm_datakit.utils.column_names import (
    BRANCH_COLUMNS,
    BUS_COLUMNS,
    DC_BRANCH_COLUMNS,
    DC_BUS_COLUMNS,
    DC_GEN_COLUMNS,
    GEN_COLUMNS,
)

import json
import time
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml

import gridfm_graphkit  # noqa: F401  registers models, normalizers and transforms
from gridfm_graphkit.datasets.globals import RATE_A
from gridfm_graphkit.datasets.graph_builder import build_hetero_data
from gridfm_graphkit.datasets.masking import RemovePFMask
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms, load_model, load_normalizer
from gridfm_graphkit.models.utils import ComputeBranchFlow, ComputeNodeInjection, ComputeNodeResiduals

WORK = Path(__file__).parent / "work"
LABELS = {"case14_ieee": "IEEE 14", "case30_ieee": "IEEE 30", "case57_ieee": "IEEE 57", "case118_ieee": "IEEE 118"}
# AC-OPF on the IEEE cases stays feasible in this range (case118 fails at 1.3).
LOAD_SCALE = {"min": 0.6, "max": 1.2, "step": 0.01, "default": 1.0}
# Generation is dispatched as if lines could only carry this share of their rating, so the
# undisturbed grid keeps some headroom. At 1.0 the OPF parks congested lines at exactly 100%
# and every state has lines no approximate solver can call.
DISPATCH_MARGIN = 0.95
# Loading above which a line counts as overloaded; just over 100% to absorb solver tolerance.
OVERLOAD = 1.001
# How far GENCO's verdict can be trusted, from analysis_hybrid.py and analysis_residual.py
# (IEEE 118, base model; the thresholds were picked on the same 1,635 states they were scored on).
# A line GENCO puts inside the band is too close to call; outside it GENCO's call is taken as is.
UNCERTAIN_BAND = (0.98, 1.02)
# GENCO's worst bus power-balance residual, in MW, above which its answer is less reliable.
RESIDUAL_MEDIUM, RESIDUAL_LOW = 0.25, 0.5


class SolveError(Exception):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason
        self.message = message


class Genco:
    """A pretrained GENCO power-flow model held in memory."""

    def __init__(self, network, variant):
        folder = WORK / f"pf_{network}_{variant}"
        self.args = NestedNamespace(**yaml.safe_load((folder / "config.yaml").read_text()))
        self.model = load_model(self.args)
        state = torch.load(folder / "model" / "best_model_state_dict.pt", map_location="cpu")
        self.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")})
        self.model.eval()
        self.normalizer = load_normalizer(self.args)
        self.normalizer.fit_from_dict(torch.load(folder / "model" / "normalizer_stats.pt", weights_only=True)[network])
        self.transforms = get_task_transforms(self.args)

    def solve(self, bus, gen, branch):
        """Predict one grid state. Returns per-bus and per-directed-branch tables.

        Branch rows cover in-service branches only: all from-ends in table
        order, then all to-ends.
        """
        data = build_hetero_data(bus, gen, branch)
        self.normalizer.transform(data)
        data = self.transforms(data)
        with torch.no_grad():
            out = self.model(data)
        self.normalizer.inverse_transform(data)
        self.normalizer.inverse_output(out, data)
        RemovePFMask()(data)

        edges = data["bus", "connects", "bus"]
        p_flow, q_flow = ComputeBranchFlow()(out["bus"], edges.edge_index, edges.edge_attr)
        p_in, q_in = ComputeNodeInjection()(p_flow, q_flow, edges.edge_index, data["bus"].x.size(0))
        res_p, res_q = ComputeNodeResiduals()(p_in, q_in, out["bus"], data["bus"].x)
        bus_out = pd.DataFrame(out["bus"].numpy(), columns=["Vm", "Va", "Pg", "Qg"])
        bus_out["Va"] = np.degrees(bus_out["Va"])
        bus_out["res_p"], bus_out["res_q"] = res_p.numpy(), res_q.numpy()
        branch_out = pd.DataFrame({
            "from_bus": edges.edge_index[0].numpy(),
            "to_bus": edges.edge_index[1].numpy(),
            "p": p_flow.numpy(),
            "q": q_flow.numpy(),
            "loading": (torch.sqrt(p_flow**2 + q_flow**2) / edges.edge_attr[:, RATE_A]).numpy(),
        })
        return bus_out, branch_out


def tables(state, ac, dc):
    """Turn a solved datakit network into the bus/gen/branch tables graphkit reads."""
    out = pf_post_processing(0, state, ac, dc, include_dc_res=True)
    bus = pd.DataFrame(out["bus"], columns=["scenario"] + BUS_COLUMNS[1:] + DC_BUS_COLUMNS)
    gen = pd.DataFrame(out["gen"], columns=["scenario"] + GEN_COLUMNS[1:] + DC_GEN_COLUMNS)
    branch = pd.DataFrame(out["branch"], columns=["scenario"] + BRANCH_COLUMNS[1:] + DC_BRANCH_COLUMNS)
    for frame, cols in ((bus, ["bus"]), (gen, ["bus"]), (branch, ["from_bus", "to_bus"])):
        frame[cols] = frame[cols].astype(int)
    q_limits = gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
    bus = bus.merge(q_limits, on="bus", how="left").fillna(0)
    return bus, gen, branch


def layout(network, n_buses, edges):
    """Bus coordinates in [0, 1], y growing downward."""
    case = network.removesuffix("_ieee")
    cache = WORK / "geo" / f"{case}.json"
    try:
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            url = (
                "https://raw.githubusercontent.com/e2nIEE/pandapower/v2.13.1/"
                f"pandapower/networks/power_system_test_case_jsons/{case}.json"
            )
            urllib.request.urlretrieve(url, cache)
        geo = json.loads(json.loads(cache.read_text())["_object"]["bus_geodata"]["_object"])
        coords = {int(i): (row[0], row[1]) for i, row in zip(geo["index"], geo["data"])}
        xy = np.array([coords[b] for b in range(n_buses)], dtype=float)
    except (OSError, KeyError, ValueError):
        graph = nx.Graph(edges)
        graph.add_nodes_from(range(n_buses))
        pos = nx.kamada_kawai_layout(graph)
        xy = np.array([pos[b] for b in range(n_buses)], dtype=float)
    xy = (xy - xy.min(axis=0)) / np.ptp(xy, axis=0)
    xy[:, 1] = 1 - xy[:, 1]
    return xy


def records(frame):
    """JSON-safe list of row dicts: plain floats, NaN and inf as None."""
    clean = frame.astype(object).where(frame.notna() & ~frame.isin([np.inf, -np.inf]), None)
    return [
        {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}
        for row in clean.to_dict(orient="records")
    ]


class Engine:
    def __init__(self, network="case118_ieee", variant="base", dispatch_margin=DISPATCH_MARGIN):
        self.network = network
        self.dispatch_margin = dispatch_margin
        self.jl = init_julia(max_iter=200)
        self.genco = Genco(network, variant)
        self.base = load_net_from_pglib(network)
        self.base_pd = self.base.Pd.copy()
        self.base_qd = self.base.Qd.copy()
        self.dispatched = {}
        self.static = self._static()

    def _static(self):
        """Grid description that does not change with the scenario."""
        net, _ = self._dispatch(LOAD_SCALE["default"])
        bus, gen, branch = tables(net, run_pf(net, self.jl, fast=True), run_dcpf(net, self.jl, fast=True))
        edges = list(zip(branch.from_bus, branch.to_bus))
        xy = layout(self.network, len(bus), edges)
        # A bridge is a branch whose loss splits the grid; a parallel circuit is never one.
        multigraph = nx.MultiGraph(edges)
        bridges = {frozenset(e) for e in nx.bridges(nx.Graph(edges)) if multigraph.number_of_edges(*e) == 1}
        gens_at = gen.groupby("bus").size()
        kind = bus[["PQ", "PV", "REF"]].idxmax(axis=1)
        buses = pd.DataFrame({
            "id": bus.bus,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "type": kind,
            "vn_kv": bus.vn_kv,
            "n_gens": bus.bus.map(gens_at).fillna(0).astype(int),
            "pd_base": self.base_pd,
            "qd_base": self.base_qd,
        })
        vn = bus.vn_kv.to_numpy()
        branches = pd.DataFrame({
            "id": range(len(branch)),
            "from": branch.from_bus,
            "to": branch.to_bus,
            "rate_a": branch.rate_a.where(branch.rate_a > 0),
            "transformer": (branch.tap != 1) | (vn[branch.from_bus] != vn[branch.to_bus]),
            "bridge": [frozenset(e) in bridges for e in edges],
        })
        return {
            "network": self.network,
            "label": LABELS.get(self.network, self.network),
            "load_scale": LOAD_SCALE,
            "advisor": {"band": list(UNCERTAIN_BAND), "residual_medium": RESIDUAL_MEDIUM, "residual_low": RESIDUAL_LOW},
            "buses": records(buses),
            "branches": records(branches),
        }

    def _dispatch(self, load_scale):
        """Base case at this load with an AC-OPF dispatch. Returns (net, opf seconds or None if cached)."""
        key = round(load_scale, 4)
        if key in self.dispatched:
            return self.dispatched[key], None
        net = self.base.copy_for_perturbation()
        net.Pd = self.base_pd * load_scale
        net.Qd = self.base_qd * load_scale
        planned = net.copy_for_perturbation()
        planned.branches[:, RATE_A_COLUMN] *= self.dispatch_margin
        start = time.perf_counter()
        try:
            opf = run_opf(planned, self.jl)
        except RuntimeError as err:
            raise SolveError("opf_failed", f"No feasible generation dispatch at load x{load_scale:.2f}.") from err
        seconds = time.perf_counter() - start
        self.dispatched[key] = pf_preprocessing(net, opf)
        return self.dispatched[key], seconds

    def solve(self, load_scale=1.0, outages=(), redispatch=None):
        """Solve one state. `redispatch` maps generator id to a MW change from the OPF dispatch;
        whatever imbalance that leaves is picked up by the reference bus."""
        outages = sorted(set(int(o) for o in outages))
        net, t_opf = self._dispatch(load_scale)
        state = net.copy_for_perturbation()
        if redispatch:
            pg = state.Pg_gen[state.idx_gens_in_service].copy()
            for gen_id, delta in redispatch.items():
                pg[int(gen_id)] += delta
            state.Pg_gen = pg
        if outages:
            state.deactivate_branches(outages)
            if not state.check_single_connected_component():
                raise SolveError("islanded", "Tripping these lines splits the grid into separate islands.")
        try:
            start = time.perf_counter()
            dc = run_dcpf(state, self.jl, fast=True)
            t_dc = time.perf_counter() - start
            start = time.perf_counter()
            ac = run_pf(state, self.jl, fast=True)
            t_ac = time.perf_counter() - start
        except RuntimeError as err:
            raise SolveError("ac_failed", "The classical AC power flow did not converge for this state.") from err
        bus, gen, branch = tables(state, ac, dc)

        start = time.perf_counter()
        bus_pred, branch_pred = self.genco.solve(bus, gen, branch)
        t_genco = time.perf_counter() - start

        buses = pd.DataFrame({
            "id": bus.bus,
            "type": bus[["PQ", "PV", "REF"]].idxmax(axis=1),
            "pd": bus.Pd,
            "qd": bus.Qd,
            "vm_ac": bus.Vm,
            "vm_genco": bus_pred.Vm,
            "va_ac": bus.Va,
            "va_genco": bus_pred.Va,
            "va_dc": bus.Va_dc,
            "pg_ac": bus.Pg,
            "pg_genco": bus_pred.Pg,
            "qg_ac": bus.Qg,
            "qg_genco": bus_pred.Qg,
            "res_p": bus_pred.res_p,
            "res_q": bus_pred.res_q,
        })

        live = (branch.br_status == 1).to_numpy()
        n_live = int(live.sum())
        rate = branch.rate_a.where(branch.rate_a > 0).to_numpy()
        nan = np.full(len(branch), np.nan)
        p_genco, q_genco, loading_genco = nan.copy(), nan.copy(), nan.copy()
        p_genco[live] = branch_pred.p[:n_live]
        q_genco[live] = branch_pred.q[:n_live]
        loading_genco[live] = np.maximum(branch_pred.loading[:n_live].to_numpy(), branch_pred.loading[n_live:].to_numpy())
        s_ac = np.maximum(np.hypot(branch.pf, branch.qf), np.hypot(branch.pt, branch.qt))
        branches = pd.DataFrame({
            "id": range(len(branch)),
            "in_service": live,
            "p_ac": branch.pf.where(live),
            "p_genco": p_genco,
            "p_dc": branch.pf_dc.where(live),
            "q_ac": branch.qf.where(live),
            "q_genco": q_genco,
            "loading_ac": (s_ac / rate).where(live),
            "loading_genco": np.where(np.isnan(rate), np.nan, loading_genco),
            "loading_dc": (branch.pf_dc.abs() / rate).where(live),
        })

        # GENCO's three-way call on each line, made without the AC solution.
        low, high = UNCERTAIN_BAND
        verdict = pd.Series("safe", index=branches.index, dtype=object)
        verdict[branches.loading_genco >= low] = "uncertain"
        verdict[branches.loading_genco > high] = "overload"
        verdict[branches.loading_genco.isna()] = None
        branches["verdict"] = verdict

        # p_set is the dispatch given to both solvers; the reference bus's real output is in buses.
        gens = pd.DataFrame({
            "id": range(len(gen)),
            "bus": gen.bus,
            "p_set": state.Pg_gen[state.idx_gens_in_service],
            "p_min": gen.min_p_mw,
            "p_max": gen.max_p_mw,
            "slack": gen.is_slack_gen == 1,
        })

        over_ac = set(branches.id[branches.loading_ac > OVERLOAD])
        over_genco = set(branches.id[branches.loading_genco > OVERLOAD])
        ref = buses.type == "REF"
        summary = {
            "total_load_mw": float(bus.Pd.sum()),
            "slack_p_ac_mw": float(buses.pg_ac[ref].sum()),
            "slack_p_genco_mw": float(buses.pg_genco[ref].sum()),
            "max_vm_err": float((buses.vm_genco - buses.vm_ac).abs().max()),
            "max_va_err_deg": float((buses.va_genco - buses.va_ac).abs().max()),
            "max_p_flow_err_mw": float((branches.p_genco - branches.p_ac).abs().max()),
            "max_res_p_mw": float(buses.res_p.abs().max()),
            "max_res_q_mvar": float(buses.res_q.abs().max()),
            "overloads_ac": sorted(over_ac),
            "overloads_genco": sorted(over_genco),
            "missed": sorted(over_ac - over_genco),
            "false_alarms": sorted(over_genco - over_ac),
        }
        # The advisor's view: confidence from the residual, lines in three tiers, and how
        # the confident calls hold up against AC.
        residual = summary["max_res_p_mw"]
        called_over = set(branches.id[verdict == "overload"])
        uncertain = set(branches.id[verdict == "uncertain"])
        called_safe = set(branches.id[verdict == "safe"])
        summary["advisor"] = {
            "confidence": "low" if residual > RESIDUAL_LOW else "medium" if residual > RESIDUAL_MEDIUM else "high",
            "overload": sorted(called_over),
            "uncertain": sorted(uncertain),
            "uncertain_overloaded": sorted(uncertain & over_ac),
            "confirmed": sorted(called_over & over_ac),
            "missed": sorted(called_safe & over_ac),
            "false_alarms": sorted(called_over - over_ac),
        }
        return {
            "ok": True,
            "load_scale": load_scale,
            "outages": outages,
            "timing_ms": {
                "opf": None if t_opf is None else round(1e3 * t_opf, 2),
                "ac": round(1e3 * t_ac, 2),
                "dc": round(1e3 * t_dc, 2),
                "genco": round(1e3 * t_genco, 2),
            },
            "buses": records(buses),
            "branches": records(branches),
            "gens": records(gens),
            "summary": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in summary.items()},
        }

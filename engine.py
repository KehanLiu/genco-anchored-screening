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
import sys
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
# How far GENCO's verdict can be trusted, from analysis/analysis_hybrid.py and analysis/analysis_residual.py
# (IEEE 118, base model; the thresholds were picked on the same 1,635 states they were scored on).
# A line GENCO puts inside the band is too close to call; outside it GENCO's call is taken as is.
UNCERTAIN_BAND = (0.98, 1.02)
# GENCO's worst bus power-balance residual, in MW, above which its answer is less reliable.
RESIDUAL_MEDIUM, RESIDUAL_LOW = 0.25, 0.5

# ---- staged check (GENCO -> one Newton step -> full AC), only computed when solve() gets an alpha ----
# Stage 1 is the conformal band of analysis/analysis_conformal.py: surrogate "genco_d" (AC loading of the N-0
# state of the same dispatch + GENCO's predicted change), sigma "shift". The allowed miss rates:
STAGE_ALPHAS = (0.05, 0.02, 0.01)
STAGE_ALPHA_DEFAULT = 0.01
# Stage 2 self-check: the largest bus power mismatch (MW / MVAr) left after one Newton step from GENCO's
# voltages, above which the corrected loadings are not trusted and the state goes to full AC.
# work/warmstart_case118.parquet has no mismatch-after-one-step column, so it was measured through
# Engine.solve on 600 random IEEE 118 states (N-1/N-2/N-3 at load x0.7-1.2; work/workbench_check.py
# --mismatch, numbers in work/workbench_check.log): median 0.01, 99th percentile 0.42 MW, and a single
# state above 1 MW (19.9 MW), which is also the only one whose loadings were still off after the step
# (5.1 points of rating; 99th percentile 0.13). With one outlier in 600 the 99.9th percentile (8.5 MW)
# is that outlier, so the threshold is put in the gap below it instead: 1 MW.
NEWTON_MISMATCH_MW = 1.0


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


def conformal_band(network, variant):
    """Band widths of stage 1, {"widths": {alpha: b}, "sigma_floor": ...}, or None without a sweep.

    Calibrated as in analysis/analysis_conformal.py (label-conditional split conformal, surrogate genco_d,
    sigma "shift") on the fit + calibration outage sets of the canonical split (seed 0) of the
    sweep written by analysis/analysis_hybrid.py, so the widths are the ones in work/conformal_*.log and
    the test half of that split stays untouched. Cached in a small JSON next to the sweep.
    """
    cache = WORK / f"workbench_conformal_{network}_{variant}.json"
    sweep = WORK / f"hybrid_{network}_{variant}.parquet"
    if cache.exists():
        band = json.loads(cache.read_text())
    elif sweep.exists():
        if str(Path(__file__).resolve().parent / "analysis") not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
        import analysis_conformal as conformal

        data, _, n_states = conformal.load(sweep)
        fit, cal_keys, _ = conformal.split_keys(n_states.index, 0)
        cal = data[data.outages.isin(fit | cal_keys)]
        sigma = conformal.sigma_of(cal, "genco_d", "shift", None)
        widths, n_cal = {}, 0
        for alpha in STAGE_ALPHAS:
            width, n_cal = conformal.conformal_width(cal, "genco_d", sigma, alpha)
            widths[str(alpha)] = float(width)
        band = {"widths": widths, "sigma_floor": conformal.SHIFT_FLOOR, "surrogate": "genco_d", "sigma": "shift",
                "n_calibration_overloads": int(n_cal), "sweep": sweep.name, "split_seed": 0}
        cache.write_text(json.dumps(band, indent=1))
    else:
        return None
    if not all(str(a) in band["widths"] and np.isfinite(band["widths"][str(a)]) for a in STAGE_ALPHAS):
        return None
    return {"widths": {a: band["widths"][str(a)] for a in STAGE_ALPHAS}, "sigma_floor": band["sigma_floor"],
            "n_calibration_overloads": band.get("n_calibration_overloads")}


class Engine:
    def __init__(self, network="case118_ieee", variant="base", dispatch_margin=DISPATCH_MARGIN):
        self.network = network
        self.genco_variant = variant
        self.dispatch_margin = dispatch_margin
        self.jl = init_julia(max_iter=200)
        self.genco = Genco(network, variant)
        self.base = load_net_from_pglib(network)
        self.base_pd = self.base.Pd.copy()
        self.base_qd = self.base.Qd.copy()
        self.dispatched = {}
        self.anchors = {}
        self._staged_info = False  # loaded on first use
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

    def staged_info(self):
        """Settings of the staged check, or None if this grid has no outage sweep to calibrate the band on."""
        if self._staged_info is False:
            band = conformal_band(self.network, self.genco_variant)
            self._staged_info = band and {
                "alphas": list(STAGE_ALPHAS),
                "default_alpha": STAGE_ALPHA_DEFAULT,
                "mismatch_threshold_mw": NEWTON_MISMATCH_MW,
                "limit": OVERLOAD,
                **band,
            }
        return self._staged_info

    def _anchor(self, load_scale, redispatch):
        """AC and GENCO line loadings of the N-0 state of this dispatch: what the band is anchored on."""
        key = (round(load_scale, 4), tuple(sorted((redispatch or {}).items())))
        if key not in self.anchors:
            self.solve(load_scale, (), redispatch)  # stores the anchor
        return self.anchors[key]

    def _staged(self, prob, branches, buses, bus_pred, load_scale, outages, redispatch, alpha):
        """The three-stage verdict on one state. Adds per-line and per-bus columns in place."""
        info = self.staged_info()
        if info is None:
            return None
        alpha = min(info["widths"], key=lambda a: abs(float(a) - alpha))
        width, floor = info["widths"][alpha], info["sigma_floor"]
        live = branches.in_service.to_numpy()
        ids = branches.id.to_numpy()
        loading_ac = branches.loading_ac.to_numpy(float)
        over_ac = loading_ac > OVERLOAD

        # Stage 1: GENCO's predicted change on top of the base-case AC loading, with a band that
        # widens with how far GENCO moved the line.
        genco = branches.loading_genco.to_numpy(float)
        if outages:
            ac0, genco0 = self._anchor(load_scale, redispatch)
        else:
            ac0, genco0 = loading_ac, genco
        with np.errstate(invalid="ignore"):
            anchored = ac0 + genco - genco0
            band_limit = OVERLOAD - width * (np.abs(genco - genco0) + floor)
            flagged = anchored >= band_limit
        branches["loading_anchored"] = anchored
        branches["band_limit"] = np.where(np.isnan(anchored), np.nan, band_limit)
        stage1 = pd.Series(np.where(flagged, "possible", "safe"), index=branches.index, dtype=object)
        stage1[np.isnan(anchored)] = None
        branches["stage1"] = stage1

        # Stage 2: one Newton step on the network equations from GENCO's voltages.
        v0 = prob.start(vm_pq=bus_pred.Vm.to_numpy(float), va=np.radians(bus_pred.Va.to_numpy(float)))
        step = prob.newton(v0, max_it=1, keep=1)
        stepped = (step["iterations"] == 1 or step["converged"]) and bool(np.isfinite(step["V"]).all())
        loading_newton = prob.loading(step["V"]) if stepped else np.full(len(branches), np.nan)
        loading_newton = np.where(live, loading_newton, np.nan)
        mismatch = float(step["mis_end"]) if stepped else np.inf
        selfcheck = bool(mismatch <= NEWTON_MISMATCH_MW)
        with np.errstate(invalid="ignore"):
            over_newton = loading_newton > OVERLOAD
        branches["loading_newton"] = loading_newton
        stage2 = pd.Series(np.where(over_newton, "over", "ok"), index=branches.index, dtype=object)
        stage2[np.isnan(loading_newton)] = None
        branches["stage2"] = stage2
        buses["vm_newton"] = np.abs(step["V"]) if stepped else np.nan
        buses["va_newton"] = np.degrees(np.angle(step["V"])) if stepped else np.nan

        # Newton iterations of the same plain solver: from a flat start, and from GENCO's voltages to convergence.
        flat = prob.newton(prob.start())
        full = prob.newton(v0)
        it_flat = flat["iterations"] if flat["converged"] else None
        it_genco = full["iterations"] if full["converged"] else None

        if not outages:
            # The base case is solved by full AC whatever the surrogate says: it is the anchor.
            stage, decided_by, reason, used = 3, "anchor", "anchor", it_flat
        elif not flagged.any():
            stage, decided_by, reason, used = 1, "genco", "band_clear", 0
        elif selfcheck and not over_newton.any():
            stage, decided_by, reason, used = 2, "newton", "newton_clear", 1
        else:
            stage, decided_by, reason = 3, "ac", "newton_overload" if selfcheck else "selfcheck"
            # Full AC continues Newton from the stage-2 point; if that fails, a flat start after the wasted step.
            used = it_genco if it_genco is not None else None if it_flat is None else it_flat + 1
        if stage < 3:
            verdict, check = "secure", "missed" if over_ac.any() else "correct"
        else:
            verdict = "overloaded" if over_ac.any() else "secure"
            check = "false_alarm" if decided_by == "ac" and not over_ac.any() else "correct"

        worst = int(np.nanargmax(loading_newton)) if stepped and not np.isnan(loading_newton).all() else None
        return {
            "alpha": float(alpha),
            "width": width,
            "sigma_floor": floor,
            "stage": stage,
            "decided_by": decided_by,
            "reason": reason,
            "verdict": verdict,
            "check": check,
            "flagged": [int(i) for i in ids[flagged]],
            "newton": {
                "mismatch_start_mw": round(float(step["mis0"]), 4),
                "mismatch_mw": round(mismatch, 4) if np.isfinite(mismatch) else None,
                "threshold_mw": NEWTON_MISMATCH_MW,
                "selfcheck_ok": selfcheck,
                "overloads": [int(i) for i in ids[over_newton]],
                "worst_line": worst,
                "worst_loading": None if worst is None else round(float(loading_newton[worst]), 5),
                "max_loading_err": round(float(np.nanmax(np.abs(loading_newton - loading_ac))), 5) if worst is not None else None,
            },
            "ac": {"needed": stage == 3, "overloads": [int(i) for i in ids[over_ac]]},
            "iterations": {
                "flat": it_flat,
                "genco_start": it_genco,
                "staged": used,
                "saved": None if it_flat is None or used is None else it_flat - used,
            },
        }

    def solve(self, load_scale=1.0, outages=(), redispatch=None, alpha=None):
        """Solve one state. `redispatch` maps generator id to a MW change from the OPF dispatch;
        whatever imbalance that leaves is picked up by the reference bus.

        With `alpha` (an allowed miss rate, one of STAGE_ALPHAS) the result also carries the staged
        check: extra columns in "branches" and "buses" and a "staged" entry. Without it the result
        is what it always was."""
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
        prob = None
        if alpha is not None and self.staged_info() is not None:
            from analysis_warmstart import Problem  # imports this module, so not at the top
            prob = Problem(state)
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
        if not outages:
            key = (round(load_scale, 4), tuple(sorted((redispatch or {}).items())))
            self.anchors.setdefault(key, (branches.loading_ac.to_numpy(float), branches.loading_genco.to_numpy(float)))
        staged = None
        if prob is not None:
            staged = self._staged(prob, branches, buses, bus_pred, load_scale, outages, redispatch, alpha)
        result = {
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
        if alpha is not None:
            result["staged"] = staged
        return result

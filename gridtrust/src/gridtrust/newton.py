"""AC power flow by a plain full Newton, built from plain arrays.

Polar formulation, MATPOWER-style Jacobian, sparse LU, no step control, no generator Q limits.
The point of the module is `PowerFlowProblem.refine`: start Newton at the voltages a surrogate
predicted, take one (or a few) steps on the real network equations, and report the corrected
voltages, the line loadings and the bus power mismatch that is left. That mismatch is a
self-check that costs nothing extra: a small value means the corrected state satisfies the
network equations, a large one means the surrogate's start was too far off for the step.

Conventions
    buses are numbered 0..n-1; everything is per unit on `base_mva` unless a name ends in `_mw`
    angles are radians (MATPOWER tables in degrees are converted by `from_matpower`)
    bus types follow MATPOWER: 1 = PQ, 2 = PV, 3 = REF
    injections are generation minus load; P is used at PV and PQ buses, Q at PQ buses only
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import splu

PQ, PV, REF = 1, 2, 3
TOL = 1e-8   # p.u. (1e-6 MW on a 100 MVA base), max-norm of the bus power mismatch
MAX_IT = 30


@dataclass
class RefineResult:
    """Outcome of `PowerFlowProblem.refine`."""
    vm: np.ndarray             # corrected voltage magnitudes, p.u. (NaN if the step broke down)
    va: np.ndarray             # corrected voltage angles, rad
    loading: np.ndarray        # per branch, share of rating (NaN: out of service, unrated, or broken step)
    mismatch_mw: float         # largest bus power mismatch left after the steps (MW / MVAr); inf if broken
    mismatch_start_mw: float   # the same before the first step, i.e. of the surrogate's voltages
    steps: int                 # Newton steps actually taken
    converged: bool            # mismatch below the solver tolerance
    ok: bool                   # the requested steps went through and gave finite voltages
    V: np.ndarray = field(repr=False, default=None)  # complex voltages

    def selfcheck(self, threshold_mw: float) -> bool:
        """True if the steps went through and the remaining bus mismatch is at most the threshold.

        This is a heuristic acceptance rule, not a bound on the loading error: a small residual mismatch
        does not certify that the corrected loadings are close to the converged solution (see
        tests/test_staged.py::test_selfcheck_is_not_a_loading_certificate)."""
        return bool(self.ok and self.mismatch_mw <= threshold_mw)


class PowerFlowProblem:
    """Admittance matrices, specified injections and bus sets of one grid state."""

    def __init__(self, Ybus, Yf, Yt, f, t, S, bus_type, vm_set, va_ref, rate=None, live=None, base_mva=100.0):
        """Low-level constructor; most users want `from_admittance`, `from_branch_parameters` or `from_matpower`.

        Ybus (n x n), Yf, Yt (m x n): sparse admittance matrices in p.u.
        f, t: from / to bus of each of the m branches.
        S: complex specified injection per bus, p.u. (generation minus load).
        bus_type: 1 PQ, 2 PV, 3 REF per bus.
        vm_set: voltage magnitude per bus, p.u.; only the PV / REF entries are used (setpoints).
        va_ref: angle of the reference bus, rad.
        rate: branch rating in p.u. (MVA / base_mva); 0, negative or NaN = unrated.
        live: boolean per branch, False = out of service (its rows of Yf / Yt must then be zero).
        """
        self.base = float(base_mva)
        self.Ybus = csr_matrix(Ybus)
        self.Yf, self.Yt = csr_matrix(Yf), csr_matrix(Yt)
        self.S = np.asarray(S, dtype=complex)
        kind = np.asarray(bus_type)
        self.pq = np.flatnonzero(kind == PQ)
        self.pv = np.flatnonzero(kind == PV)
        self.ref = np.flatnonzero(kind == REF)
        if len(self.ref) == 0:
            raise ValueError("no reference bus (type 3)")
        if len(self.pq) + len(self.pv) + len(self.ref) != len(kind):
            raise ValueError("bus_type must be 1 (PQ), 2 (PV) or 3 (REF) for every bus")
        self.pvpq = np.r_[self.pv, self.pq]
        self.vm_set = np.asarray(vm_set, dtype=float).copy()
        self.va_ref = float(va_ref)
        self.f = np.asarray(f, dtype=int)
        self.t = np.asarray(t, dtype=int)
        m = len(self.f)
        self.live = np.ones(m, bool) if live is None else np.asarray(live, dtype=bool)
        rate = np.full(m, np.nan) if rate is None else np.asarray(rate, dtype=float)
        with np.errstate(invalid="ignore"):
            positive = rate > 0
        self.rated = self.live & positive
        self.rate = np.where(positive, rate, np.nan)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_admittance(cls, bus_type, vm_set, va_ref, p_inj, q_inj, f, t, yff, yft, ytf, ytt,
                        shunt=None, rate=None, status=None, base_mva=100.0):
        """From per-branch admittance blocks (p.u., complex), as gridfm-datakit stores them.

        p_inj, q_inj: specified injections per bus in p.u. (generation minus load).
        shunt: complex bus shunt admittance in p.u. (Gs + jBs) or None.
        status: 1 / True for branches in service; branches out of service are dropped from the
            matrices (their loading is NaN).
        """
        n = len(np.asarray(bus_type))
        f, t = np.asarray(f, dtype=int), np.asarray(t, dtype=int)
        m = len(f)
        live = np.ones(m, bool) if status is None else np.asarray(status).astype(bool)
        blocks = [np.where(live, np.asarray(y, dtype=complex), 0) for y in (yff, yft, ytf, ytt)]
        rows = np.r_[np.arange(m), np.arange(m)]
        cols = np.r_[f, t]
        Yf = csr_matrix((np.r_[blocks[0], blocks[1]], (rows, cols)), shape=(m, n))
        Yt = csr_matrix((np.r_[blocks[2], blocks[3]], (rows, cols)), shape=(m, n))
        cf = csr_matrix((np.ones(m), (np.arange(m), f)), shape=(m, n))
        ct = csr_matrix((np.ones(m), (np.arange(m), t)), shape=(m, n))
        ysh = np.zeros(n, complex) if shunt is None else np.asarray(shunt, dtype=complex)
        Ybus = cf.T @ Yf + ct.T @ Yt + diags(ysh)
        S = np.asarray(p_inj, dtype=float) + 1j * np.asarray(q_inj, dtype=float)
        return cls(Ybus, Yf, Yt, f, t, S, bus_type, vm_set, va_ref, rate=rate, live=live, base_mva=base_mva)

    @classmethod
    def from_branch_parameters(cls, bus_type, vm_set, va_ref, p_inj, q_inj, f, t, r, x, b=None, tap=None, shift=None,
                               shunt=None, rate=None, status=None, base_mva=100.0):
        """From series impedance r + jx, total line charging b, tap ratio and phase shift (rad), all p.u.

        The branch model is MATPOWER's pi model with the transformer at the from end; tap 0 means 1.
        """
        f = np.asarray(f, dtype=int)
        m = len(f)
        zeros = np.zeros(m)
        b = zeros if b is None else np.asarray(b, dtype=float)
        ratio = np.ones(m) if tap is None else np.asarray(tap, dtype=float).copy()
        ratio[ratio == 0] = 1.0
        shift = zeros if shift is None else np.asarray(shift, dtype=float)
        tap_c = ratio * np.exp(1j * shift)
        ys = 1.0 / (np.asarray(r, dtype=float) + 1j * np.asarray(x, dtype=float))
        ytt = ys + 1j * b / 2
        yff = ytt / (tap_c * np.conj(tap_c))
        yft = -ys / np.conj(tap_c)
        ytf = -ys / tap_c
        return cls.from_admittance(bus_type, vm_set, va_ref, p_inj, q_inj, f, t, yff, yft, ytf, ytt,
                                   shunt=shunt, rate=rate, status=status, base_mva=base_mva)

    @classmethod
    def from_matpower(cls, base_mva, bus, gen, branch):
        """From MATPOWER-style arrays (bus >= 9 columns, gen >= 8, branch >= 11; MW, MVAr, degrees).

        Bus numbers may be arbitrary; they are mapped to 0..n-1 in table order. PV / REF setpoints
        are the generators' VG where a bus has a generator in service, else the bus table's VM.
        Generation is the tables' PG / QG (the REF bus's P and all PV / REF Q are not used by Newton).
        """
        bus, gen, branch = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (bus, gen, branch))
        index = {int(b): i for i, b in enumerate(bus[:, 0])}
        n = len(bus)
        on = gen[:, 7] > 0
        gbus = np.array([index[int(b)] for b in gen[on, 0]], dtype=int)
        pg = np.bincount(gbus, weights=gen[on, 1], minlength=n)
        qg = np.bincount(gbus, weights=gen[on, 2], minlength=n)
        vm_set = bus[:, 7].copy()
        vm_set[gbus] = gen[on, 5]
        kind = bus[:, 1].astype(int)
        ref = np.flatnonzero(kind == REF)
        if len(ref) == 0:
            raise ValueError("no reference bus (type 3)")
        f = np.array([index[int(b)] for b in branch[:, 0]], dtype=int)
        t = np.array([index[int(b)] for b in branch[:, 1]], dtype=int)
        return cls.from_branch_parameters(
            kind, vm_set, np.radians(bus[ref[0], 8]),
            (pg - bus[:, 2]) / base_mva, (qg - bus[:, 3]) / base_mva,
            f, t, branch[:, 2], branch[:, 3], branch[:, 4], branch[:, 8], np.radians(branch[:, 9]),
            shunt=(bus[:, 4] + 1j * bus[:, 5]) / base_mva, rate=branch[:, 5] / base_mva,
            status=branch[:, 10] > 0, base_mva=base_mva)

    # ------------------------------------------------------------------ the equations
    def start(self, vm_pq=None, va=None):
        """Complex start vector: setpoints at PV / REF, given (or 1.0) magnitudes at PQ, given (or flat) angles.

        `vm_pq` and `va` are full per-bus arrays; of `vm_pq` only the PQ entries are used, and the
        reference angle always overrides `va` at the reference bus.
        """
        mag = self.vm_set.copy()
        mag[self.pq] = 1.0 if vm_pq is None else np.asarray(vm_pq, dtype=float)[self.pq]
        ang = np.full(len(mag), self.va_ref) if va is None else np.asarray(va, dtype=float).copy()
        ang[self.ref] = self.va_ref
        return mag * np.exp(1j * ang)

    def mismatch(self, V):
        """Power mismatch vector F(V): P at PV and PQ buses, then Q at PQ buses (p.u.)."""
        mis = V * np.conj(self.Ybus @ V) - self.S
        return np.r_[mis[self.pvpq].real, mis[self.pq].imag]

    def mismatch_mw(self, V):
        """Largest bus power mismatch in MW / MVAr."""
        return float(np.abs(self.mismatch(V)).max() * self.base)

    def jacobian(self, V):
        """dF/d(Va at PV+PQ, Vm at PQ), sparse CSC."""
        Ibus = self.Ybus @ V
        dV, dI, dn = diags(V), diags(Ibus), diags(V / np.abs(V))
        dS_dVa = 1j * dV @ np.conj(dI - self.Ybus @ dV)
        dS_dVm = dV @ np.conj(self.Ybus @ dn) + np.conj(dI) @ dn
        dS_dVa, dS_dVm = csr_matrix(dS_dVa), csr_matrix(dS_dVm)
        pvpq, pq = self.pvpq, self.pq
        return vstack([
            hstack([dS_dVa[pvpq][:, pvpq].real, dS_dVm[pvpq][:, pq].real]),
            hstack([dS_dVa[pq][:, pvpq].imag, dS_dVm[pq][:, pq].imag]),
        ]).tocsc()

    def step(self, V):
        """One full Newton step from V. Returns the new complex voltages, or None if the Jacobian is singular."""
        va, vm = np.angle(V), np.abs(V)
        npv = len(self.pvpq)
        try:
            with np.errstate(all="ignore"):
                dx = splu(self.jacobian(V)).solve(-self.mismatch(V))
        except RuntimeError:
            return None
        va[self.pvpq] += dx[:npv]
        vm[self.pq] += dx[npv:]
        return vm * np.exp(1j * va)

    def solve(self, V0=None, max_it=MAX_IT, keep=2, tol=TOL):
        """Full Newton from V0 (default: flat start).

        Returns dict(converged, iterations, mis0 [MW], mis_end [MW], V, steps=[V after 1..keep steps]).
        Stops without converging when the Jacobian is singular or the mismatch is no longer finite.
        """
        V = self.start() if V0 is None else np.asarray(V0, dtype=complex).copy()
        va, vm = np.angle(V), np.abs(V)
        npv = len(self.pvpq)
        F = self.mismatch(V)
        norm = np.abs(F).max()
        out = {"mis0": norm * self.base, "steps": [], "iterations": 0, "converged": bool(norm < tol)}
        while not out["converged"] and out["iterations"] < max_it:
            try:
                with np.errstate(all="ignore"):
                    dx = splu(self.jacobian(V)).solve(-F)
            except RuntimeError:  # singular Jacobian
                break
            va[self.pvpq] += dx[:npv]
            vm[self.pq] += dx[npv:]
            V = vm * np.exp(1j * va)
            out["iterations"] += 1
            with np.errstate(all="ignore"):
                F = self.mismatch(V)
            if not np.isfinite(F).all():
                break
            if len(out["steps"]) < keep:
                out["steps"].append(V.copy())
            norm = np.abs(F).max()
            out["converged"] = bool(norm < tol)
        out["V"] = V
        out["mis_end"] = float(norm * self.base) if np.isfinite(norm) else np.inf
        return out

    newton = solve  # the name used in the analysis scripts this was ported from

    # ------------------------------------------------------------------ flows
    def flows(self, V):
        """Complex power entering each branch at its from and to end, p.u. (zero for branches out of service)."""
        return V[self.f] * np.conj(self.Yf @ V), V[self.t] * np.conj(self.Yt @ V)

    def loading(self, V):
        """Loading of every branch (share of rating, NaN if out of service / unrated)."""
        with np.errstate(all="ignore"):
            sf = np.abs(V[self.f] * np.conj(self.Yf @ V))
            st = np.abs(V[self.t] * np.conj(self.Yt @ V))
            return np.where(self.rated, np.maximum(sf, st) / self.rate, np.nan)

    # ------------------------------------------------------------------ refinement of a surrogate's answer
    def refine(self, vm, va, steps=1):
        """Newton step(s) from predicted voltages, with the self-check.

        vm, va: predicted magnitude (p.u.) and angle (rad) per bus. As in a power flow, PV / REF
        magnitudes and the reference angle are taken from the setpoints, not from the prediction.
        Returns a RefineResult; `result.selfcheck(threshold_mw)` says whether to trust it.
        """
        v0 = self.start(vm_pq=vm, va=va)
        run = self.solve(v0, max_it=steps, keep=0)
        ok = (run["iterations"] == steps or run["converged"]) and bool(np.isfinite(run["V"]).all())
        m, n = len(self.f), len(v0)
        if ok:
            V = run["V"]
            return RefineResult(np.abs(V), np.angle(V), self.loading(V), float(run["mis_end"]), float(run["mis0"]),
                                run["iterations"], run["converged"], True, V)
        return RefineResult(np.full(n, np.nan), np.full(n, np.nan), np.full(m, np.nan), np.inf, float(run["mis0"]),
                            run["iterations"], False, False, run["V"])

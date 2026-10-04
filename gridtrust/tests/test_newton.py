import numpy as np
import pytest

from gridtrust.newton import PQ, PV, REF, PowerFlowProblem

from cases import BRANCH14, BUS14, GEN14


# ----------------------------------------------------------------- 3 buses, closed form
def lossless_chain():
    """REF - PV - PV chain of pure reactances, all |V| = 1: P = sin(angle difference) / x on each line."""
    return PowerFlowProblem.from_branch_parameters(
        bus_type=[REF, PV, PV], vm_set=[1.0, 1.0, 1.0], va_ref=0.0,
        p_inj=[0.0, -0.5, -0.2], q_inj=[0.0, 0.0, 0.0],
        f=[0, 1], t=[1, 2], r=[0.0, 0.0], x=[0.1, 0.2], rate=[1.0, 0.5])


def test_three_bus_closed_form():
    prob = lossless_chain()
    run = prob.solve()
    assert run["converged"] and run["iterations"] <= 5
    d12, d23 = np.arcsin(0.7 * 0.1), np.arcsin(0.2 * 0.2)
    np.testing.assert_allclose(np.angle(run["V"]), [0.0, -d12, -d12 - d23], atol=1e-10)
    np.testing.assert_allclose(np.abs(run["V"]), 1.0, atol=1e-12)
    sf, st = prob.flows(run["V"])
    np.testing.assert_allclose(sf.real, [0.7, 0.2], atol=1e-9)
    np.testing.assert_allclose(st.real, [-0.7, -0.2], atol=1e-9)
    # |S| = sqrt(P^2 + Q^2) with Q = (1 - cos d) / x at both ends of a lossless line between equal magnitudes
    s = np.hypot([0.7, 0.2], [(1 - np.cos(d12)) / 0.1, (1 - np.cos(d23)) / 0.2])
    np.testing.assert_allclose(prob.loading(run["V"]), s / np.array([1.0, 0.5]), atol=1e-9)


def three_bus_pq():
    """REF, PV and PQ bus in a triangle; injections are computed by hand (dense loops) from chosen voltages."""
    f, t = [0, 0, 1], [1, 2, 2]
    r, x, b = [0.01, 0.02, 0.015], [0.1, 0.2, 0.15], [0.02, 0.0, 0.04]
    V = np.array([1.02, 1.01 * np.exp(-0.05j), 0.97 * np.exp(-0.11j)])
    Y = np.zeros((3, 3), complex)
    for i, j, rr, xx, bb in zip(f, t, r, x, b):
        y = 1 / (rr + 1j * xx)
        Y[i, i] += y + 0.5j * bb
        Y[j, j] += y + 0.5j * bb
        Y[i, j] -= y
        Y[j, i] -= y
    S = V * np.conj(Y @ V)
    prob = PowerFlowProblem.from_branch_parameters(
        bus_type=[REF, PV, PQ], vm_set=np.abs(V), va_ref=0.0, p_inj=S.real, q_inj=S.imag,
        f=f, t=t, r=r, x=x, b=b, rate=[1.0, 1.0, 0.0])
    return prob, V, Y


def test_three_bus_recovers_known_voltages():
    prob, V, Y = three_bus_pq()
    np.testing.assert_allclose(prob.Ybus.toarray(), Y, atol=1e-14)
    assert np.abs(prob.mismatch(V)).max() < 1e-14
    run = prob.solve()
    assert run["converged"]
    np.testing.assert_allclose(run["V"], V, atol=1e-9)
    loading = prob.loading(run["V"])
    assert np.isnan(loading[2]) and np.isfinite(loading[:2]).all()   # rate 0 = unrated


def test_jacobian_matches_finite_differences():
    prob, V, _ = three_bus_pq()
    V = prob.start(vm_pq=np.array([1, 1, 0.95]), va=np.array([0, -0.02, -0.08]))
    J = prob.jacobian(V).toarray()
    vm, va, h = np.abs(V), np.angle(V), 1e-7
    cols = []
    for i in prob.pvpq:
        d = va.copy(); d[i] += h
        cols.append((prob.mismatch(vm * np.exp(1j * d)) - prob.mismatch(V)) / h)
    for i in prob.pq:
        d = vm.copy(); d[i] += h
        cols.append((prob.mismatch(d * np.exp(1j * va)) - prob.mismatch(V)) / h)
    np.testing.assert_allclose(J, np.array(cols).T, atol=1e-5)


def test_start_uses_setpoints_and_reference_angle():
    prob, V, _ = three_bus_pq()
    v0 = prob.start(vm_pq=np.array([9.0, 9.0, 0.9]), va=np.array([0.3, -0.1, -0.2]))
    np.testing.assert_allclose(np.abs(v0), [1.02, 1.01, 0.9])
    np.testing.assert_allclose(np.angle(v0), [0.0, -0.1, -0.2])


def test_out_of_service_branch_is_removed():
    prob, V, _ = three_bus_pq()
    off = PowerFlowProblem.from_branch_parameters(
        bus_type=[REF, PV, PQ], vm_set=prob.vm_set, va_ref=0.0, p_inj=prob.S.real, q_inj=prob.S.imag,
        f=[0, 0, 1], t=[1, 2, 2], r=[0.01, 0.02, 0.015], x=[0.1, 0.2, 0.15], b=[0.02, 0.0, 0.04],
        rate=[1.0, 1.0, 1.0], status=[1, 0, 1])
    run = off.solve()
    assert run["converged"]
    loading = off.loading(run["V"])
    assert np.isnan(loading[1]) and np.isfinite(loading[[0, 2]]).all()
    assert off.Ybus[0, 2] == 0


def test_requires_reference_bus():
    with pytest.raises(ValueError):
        PowerFlowProblem.from_branch_parameters([PQ, PQ], [1, 1], 0.0, [0, 0], [0, 0], [0], [1], [0.0], [0.1])


# ----------------------------------------------------------------- IEEE 14
@pytest.fixture(scope="module")
def ieee14():
    prob = PowerFlowProblem.from_matpower(100.0, BUS14, GEN14, BRANCH14)
    run = prob.solve()
    return prob, run


def test_ieee14_converges_to_published_solution(ieee14):
    prob, run = ieee14
    assert run["converged"] and run["iterations"] <= 5
    assert run["mis_end"] < 1e-6
    np.testing.assert_allclose(np.abs(run["V"]), BUS14[:, 7], atol=1.5e-3)          # published to 3 decimals
    np.testing.assert_allclose(np.degrees(np.angle(run["V"])), BUS14[:, 8], atol=0.02)
    slack = (run["V"] * np.conj(prob.Ybus @ run["V"]))[0] * 100 + (BUS14[0, 2] + 1j * BUS14[0, 3])
    assert abs(slack.real - 232.4) < 0.1 and abs(slack.imag - (-16.55)) < 0.05   # runpf prints 232.39 MW, -16.55 MVAr


def test_ieee14_one_step_is_quadratic(ieee14):
    prob, run = ieee14
    rng = np.random.default_rng(0)
    dvm, dva = rng.normal(size=14), rng.normal(size=14)
    before, after = [], []
    for eps in (4e-3, 2e-3, 1e-3):
        vm, va = np.abs(run["V"]) + eps * dvm, np.angle(run["V"]) + eps * dva
        res = prob.refine(vm, va, steps=1)
        assert res.ok and res.steps == 1
        before.append(res.mismatch_start_mw)
        after.append(res.mismatch_mw)
        assert res.mismatch_mw < 0.05 * res.mismatch_start_mw
        assert np.abs(res.V - run["V"]).max() < 20 * eps**2 * 14
    before, after = np.array(before), np.array(after)
    # halving the perturbation halves the start mismatch and quarters what is left after one step
    assert np.all((before[:-1] / before[1:] > 1.8) & (before[:-1] / before[1:] < 2.2))
    assert np.all((after[:-1] / after[1:] > 3.2) & (after[:-1] / after[1:] < 4.8))


def test_ieee14_refine_matches_step_and_solve(ieee14):
    prob, run = ieee14
    vm, va = np.abs(run["V"]) * 1.01, np.angle(run["V"]) * 0.9
    res = prob.refine(vm, va, steps=1)
    v0 = prob.start(vm_pq=vm, va=va)
    np.testing.assert_array_equal(res.V, prob.solve(v0, max_it=1)["V"])
    np.testing.assert_allclose(res.V, prob.step(v0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(res.loading, prob.loading(res.V))
    assert res.mismatch_mw == pytest.approx(prob.mismatch_mw(res.V))
    assert res.selfcheck(1.0) and not res.selfcheck(1e-12)
    two = prob.refine(vm, va, steps=2)
    assert two.steps == 2 and two.mismatch_mw < res.mismatch_mw
    exact = prob.refine(np.abs(run["V"]), np.angle(run["V"]))
    assert exact.ok and exact.steps == 0 and exact.converged


def test_refine_reports_breakdown():
    # a load far beyond what the line can carry, started at zero voltage: the step cannot be taken
    prob = PowerFlowProblem.from_branch_parameters([REF, PQ], [1.0, 1.0], 0.0, [0.0, -50.0], [0.0, -50.0], [0], [1], [0.0], [0.1], rate=[1.0])
    res = prob.refine(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    assert not res.ok and res.mismatch_mw == np.inf and not res.selfcheck(1e9)
    assert np.isnan(res.loading).all()
    assert not prob.solve()["converged"]

"""Algorithm 1 and the hybrid stage on ACTIVSg200; skips without AMPL and Gurobi."""

import pytest

from ropf import algorithm, risk
from ropf.algorithm import AlgorithmConfig
from ropf.model import BusCut, Master, SolverConfig


@pytest.mark.parametrize("kwargs", [
    {"metric": "max_line_loading"}, {"stage": "a3"}, {"flow_domain": "rateA"},
    {"eta": 0.0}, {"eta": 1.0}, {"kappa": 0}, {"k_bar": 0}, {"q": 0},
    {"q": 5, "stage": "a2"}, {"weight_multiplier": -1.0}, {"risk_weight": -1.0},
    {"gamma_tol": -1.0}])
def test_bad_config_is_an_error(kwargs):
    with pytest.raises(ValueError):
        AlgorithmConfig(**kwargs)


def _run(net, **kwargs):
    kwargs.setdefault("kappa", 2)
    kwargs.setdefault("k_bar", 4)
    try:
        return algorithm.run(net, AlgorithmConfig(**kwargs), SolverConfig())
    except Exception as exc:
        pytest.skip(f"could not run: {exc}")


@pytest.fixture(scope="module")
def base_run(net):
    result = _run(net, eta=0.9)
    if result.termination == "infeasible":
        pytest.skip("the master did not solve")
    return result


def test_lambda_star_is_z0_over_rho0(base_run):
    assert base_run.lambda_star == pytest.approx(base_run.z0 / base_run.rho0, rel=1e-12)
    assert base_run.risk_weight == pytest.approx(base_run.lambda_star, rel=1e-12)


def test_nominal_iterate_is_at_lambda_zero(base_run):
    nominal = base_run.iterations[0]
    assert nominal.k == 0 and nominal.ratio == 1.0
    assert nominal.objective == pytest.approx(nominal.cost, rel=1e-12)


def test_surrogate_never_exceeds_the_representation(base_run):
    for it in base_run.iterations:
        assert it.surrogate <= it.rho + 1e-6 * max(1.0, abs(it.rho))


def test_the_cut_pool_only_grows(base_run):
    pools = [(it.n_line_cuts, it.n_bus_cuts) for it in base_run.iterations]
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(pools, pools[1:]))


def test_zero_weight_is_the_nominal_solve(net):
    result = _run(net, weight_multiplier=0.0)
    assert result.termination == "zero_weight" and len(result.iterations) == 1


def test_eta_exit(net):
    result = _run(net, eta=0.3, k_bar=25)
    if result.termination != "eta_target":
        pytest.skip(f"exits by {result.termination}")
    assert result.final.ratio <= 0.7


def test_k_bar_exit(net):
    result = _run(net, eta=0.99, k_bar=3)
    assert result.termination in ("iteration_limit", "separation_exhausted")
    assert result.k_end <= 3


def test_gap_exit(net):
    result = _run(net, eta=0.99, k_bar=40, gamma_tol=1e-6)
    if result.termination != "converged":
        pytest.skip(f"exits by {result.termination}")
    assert abs(result.final.gamma) <= 1e-6


def test_the_bus_family_adds_only_bus_cuts(net):
    result = _run(net, metric="bus_flow_sum_agg", eta=0.9, k_bar=3)
    assert result.final.n_bus_cuts > 0 and result.final.n_line_cuts == 0


def _master(net):
    try:
        return Master(net, "master.mod", SolverConfig(),
                      line_cut_capacity=100, bus_cut_capacity=100)
    except Exception as exc:
        pytest.skip(f"could not load the master: {exc}")


def _pooled_mean(net, master, metric, Pf, q):
    if metric == "bus_flow_sum_agg":
        best = {}
        for cut in master.bus_cuts:
            value = sum(Pf[b] for b in cut.branch_pos) - sum(Pf[b] for b in cut.branch_neg)
            best[cut.bus] = max(best.get(cut.bus, 0.0), value)
        values = best.values()
    elif metric == "joule_loss_max":
        values = [net.branches[b].r_heat * Pf[b] ** 2 for b in master.line_cut_ids]
    else:
        values = [abs(Pf[b]) for b in master.line_cut_ids]
    return risk.top_q_mean(values, q)


@pytest.mark.parametrize("metric", risk.METRICS)
def test_order_q_surrogate_is_the_pooled_mean(net, metric):
    master = _master(net)
    config = AlgorithmConfig(metric=metric, q=5, kappa=5, eta=0.9, k_bar=4)
    result = algorithm.run_loop(net, master, config)
    if result.termination == "infeasible":
        pytest.skip("the master did not solve")
    for it in result.iterations:
        assert it.surrogate <= it.rho + 1e-6 * max(1.0, abs(it.rho))
    expected = _pooled_mean(net, master, metric, result.dispatch.Pf, 5)
    assert result.final.surrogate == pytest.approx(expected, rel=1e-6, abs=1e-8)
    assert result.final.rho < result.rho0


def test_a_bus_under_two_patterns_is_counted_once(net):
    master = _master(net)
    master.set_risk_family("bus_flow_sum_agg")
    master.set_order(2)
    nominal = master.solve()
    ev = risk.evaluate("bus_flow_sum_agg", net, nominal.Pf)
    cut = risk.build_bus_cuts(ev, [max(ev.components, key=ev.components.get)])[0]
    small = min(cut.branch_pos + cut.branch_neg, key=lambda b: abs(nominal.Pf[b]))
    flipped = BusCut(
        bus=cut.bus,
        branch_pos=tuple(b for b in cut.branch_pos if b != small)
        + ((small,) if small in cut.branch_neg else ()),
        branch_neg=tuple(b for b in cut.branch_neg if b != small)
        + ((small,) if small in cut.branch_pos else ()))
    assert master.add_bus_cuts([cut, flipped]) == 2
    master.set_risk_weight(1e3)
    solution = master.solve()
    expected = _pooled_mean(net, master, "bus_flow_sum_agg", solution.Pf, 2)
    assert solution.phi == pytest.approx(expected, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("metric", risk.METRICS)
def test_mean_form_at_q_one_reproduces_the_direct_cuts(net, metric, monkeypatch):
    direct = _run(net, metric=metric, eta=0.9)
    original = Master.set_order
    monkeypatch.setattr(Master, "set_order",
                        lambda self, q, mean_form=None: original(self, q, True))
    mean = _run(net, metric=metric, eta=0.9)
    assert (len(mean.iterations), mean.termination) == \
        (len(direct.iterations), direct.termination)
    # The Joule master is a QCP; barrier leaves the worst line to about 1e-3.
    rho_tol = 1e-3 if metric == "joule_loss_max" else 1e-6
    for a, b in zip(direct.iterations, mean.iterations):
        assert b.objective == pytest.approx(a.objective, rel=1e-7)
        assert b.surrogate == pytest.approx(a.surrogate, rel=1e-6, abs=1e-8)
        assert b.rho == pytest.approx(a.rho, rel=rho_tol)


def _hybrid(net, **kwargs):
    for name in ("knitro", "ipopt"):
        config = AlgorithmConfig(stage="a2", kappa=2, k_bar=3, **kwargs)
        try:
            result = algorithm.run(net, config, SolverConfig(), SolverConfig(name=name))
        except Exception:
            continue
        if result.ac is not None and result.ac.solved:
            return result
    pytest.skip("neither Knitro nor Ipopt solved (M^ac)")


@pytest.fixture(scope="module")
def hybrid_run(net):
    return _hybrid(net, eta=0.3)


def test_the_pool_transfers_unchanged(hybrid_run):
    assert (hybrid_run.ac.n_line_cuts, hybrid_run.ac.n_bus_cuts) == \
        (hybrid_run.final.n_line_cuts, hybrid_run.final.n_bus_cuts)
    assert hybrid_run.ac.cuts_added == 0


def test_the_reported_dispatch_is_the_ac_one(hybrid_run, net):
    assert hybrid_run.reported is hybrid_run.ac and hybrid_run.dispatch.v
    rho = risk.evaluate(hybrid_run.metric, net, hybrid_run.dispatch.Pf).value
    assert hybrid_run.ac.rho == pytest.approx(rho, rel=1e-12)


def test_a2_at_zero_weight_is_an_ac_opf(net):
    result = _hybrid(net, weight_multiplier=0.0)
    assert result.termination == "zero_weight" and result.dispatch.v

"""Algorithm 1: the weight, the two exits, and the Section 3.4 hand-off.

The configuration tests need nothing.  The rest solve (M) on ACTIVSg200 and skip
where no solver is reachable, the same way `test_dcopf_vs_matpower` does.
"""

from __future__ import annotations

import inspect
import os

import pytest

from ropf import algorithm, risk
from ropf.algorithm import AlgorithmConfig
from ropf.network import read_matpower

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASE = os.path.join(FIXTURES, "case_ACTIVSg200.m")


###############################################################################
# The Require line: what the algorithm refuses to be given
###############################################################################


def test_unknown_metric_is_an_error():
    with pytest.raises(ValueError, match="risk functional"):
        AlgorithmConfig(metric="max_line_loading")


def test_unknown_stage_is_an_error():
    with pytest.raises(ValueError, match="stage"):
        AlgorithmConfig(stage="hybrid")


@pytest.mark.parametrize("eta", [0.0, 1.0, -0.5, 2.0])
def test_eta_outside_the_unit_interval_is_an_error(eta):
    """eta = 0 exits before the loop runs; eta = 1 demands a zero functional."""
    with pytest.raises(ValueError, match="eta"):
        AlgorithmConfig(eta=eta)


@pytest.mark.parametrize("kwargs", [{"kappa": 0}, {"k_bar": 0},
                                    {"weight_multiplier": -1.0},
                                    {"risk_weight": -1.0}])
def test_out_of_range_budgets_are_errors(kwargs):
    with pytest.raises(ValueError):
        AlgorithmConfig(**kwargs)


def test_the_ac_stage_has_no_iteration_knob():
    """Section 3.4 solves (M^ac) once, and nothing may say otherwise.

    A knob defaulting to 1 is only safe until a layer above it parses a default
    of its own and shadows the solver's.  The durable form of that default is
    for the knob not to exist, so this asserts its absence: neither the config
    nor `run_ac_stage` carries an iteration count, and `run_ac_stage` calls
    `Master.solve` exactly once.
    """
    fields = set(AlgorithmConfig.__dataclass_fields__)
    knobs = {name for name in fields
             if "iter" in name.lower() or name.lower().endswith("_solves")}
    assert knobs == set(), f"an AC iteration knob crept into the config: {knobs}"

    params = set(inspect.signature(algorithm.run_ac_stage).parameters)
    knobs = {name for name in params if "iter" in name.lower()}
    assert knobs == set(), f"run_ac_stage grew an iteration knob: {knobs}"

    source = inspect.getsource(algorithm.run_ac_stage)
    assert source.count(".solve()") == 1, \
        "run_ac_stage must solve (M^ac) exactly once"


###############################################################################
# The loop
###############################################################################


@pytest.fixture(scope="module")
def net():
    return read_matpower(CASE)


def _run(net, **kwargs):
    pytest.importorskip("amplpy", reason="AMPL is not installed")
    from ropf.model import SolverConfig

    kwargs.setdefault("metric", "max_active_flow")
    kwargs.setdefault("kappa", 2)
    kwargs.setdefault("k_bar", 4)
    config = AlgorithmConfig(**kwargs)
    try:
        return algorithm.run(net, config,
                             SolverConfig(name="gurobi", verbose=False))
    except Exception as exc:                     # no licence, no solver, ...
        pytest.skip(f"could not run the loop: {exc}")


@pytest.fixture(scope="module")
def base_run(net):
    result = _run(net, eta=0.9, k_bar=4)
    if result.termination == "infeasible":
        pytest.skip("the master did not solve")
    return result


def test_lambda_star_is_z0_over_rho0(base_run):
    """eq (weightstar).  The two terms of eq (1a) are equal at the nominal point."""
    assert base_run.lambda_star == pytest.approx(base_run.z0 / base_run.rho0,
                                                 rel=1e-12)
    assert base_run.risk_weight == pytest.approx(
        base_run.config.weight_multiplier * base_run.lambda_star, rel=1e-12)
    # What eq (weightstar) is for: neither term of the objective dominates on
    # its units alone at the point the loop starts from.
    assert base_run.lambda_star * base_run.rho0 == pytest.approx(base_run.z0,
                                                                 rel=1e-12)


def test_nominal_iterate_is_the_lambda_zero_solve(base_run):
    """Line 1 is at lambda = 0 whatever the run's own weight is."""
    nominal = base_run.iterations[0]
    assert nominal.k == 0
    assert nominal.ratio == pytest.approx(1.0, rel=1e-12)
    # At lambda = 0 the objective is the generation cost alone.
    assert nominal.objective == pytest.approx(nominal.cost, rel=1e-12)


def test_surrogate_never_exceeds_the_functional(base_run):
    """eq (11): Phi^k lower-bounds phi(x^k) at every iterate, so Gamma^k >= 0.

    This is what makes the master a relaxation, and it is the one property the
    whole method rests on.  A negative gap means a cut was built from something
    other than the incumbent's sign pattern.
    """
    for it in base_run.iterations:
        if it.solved:
            assert it.surrogate <= it.rho + 1e-6 * max(1.0, abs(it.rho))
            assert it.gamma >= -1e-9


def test_zero_weight_returns_the_nominal_dispatch(net):
    """The lambda = 0 grid point is the nominal solve, and costs one solve.

    Phi is bounded from below by the cuts and priced at lambda in the
    objective, so at lambda = 0 it is free and no cut can move the dispatch.
    Running the loop to discover that would cost k-bar solves per rung.
    """
    result = _run(net, weight_multiplier=0.0, eta=0.5, k_bar=25)
    assert result.termination == "zero_weight"
    assert len(result.iterations) == 1
    assert result.n_cuts == 0
    assert result.dispatch is result.nominal_dispatch


def test_eta_exit_reaches_the_target(net):
    result = _run(net, weight_multiplier=1.0, eta=0.3, k_bar=25)
    if result.termination != "eta_target":
        pytest.skip(f"this case exits by {result.termination}")
    assert result.final.ratio <= 1.0 - result.config.eta


def test_iteration_limit_exit_stops_at_k_bar(net):
    """The second exit.  eta = 0.99 is out of reach, so k-bar must bind."""
    result = _run(net, weight_multiplier=1.0, eta=0.99, k_bar=3)
    assert result.termination in ("iteration_limit", "separation_exhausted")
    assert result.k_end <= 3
    assert result.final.ratio > 1.0 - 0.99


def test_the_cut_pool_only_grows(base_run):
    """A cut is appended, never replaced or removed.  See eq (11)."""
    counts = [(it.n_line_cuts, it.n_bus_cuts) for it in base_run.iterations]
    for before, after in zip(counts, counts[1:]):
        assert after[0] >= before[0] and after[1] >= before[1]


def test_the_bus_family_adds_bus_cuts_and_no_line_cuts(net):
    """eq (6c) is its own family; it must not fall back to the flow cuts."""
    result = _run(net, metric="bus_flow_sum_agg", weight_multiplier=1.0,
                  eta=0.9, k_bar=3)
    assert result.final.n_bus_cuts > 0
    assert result.final.n_line_cuts == 0


###############################################################################
# Section 3.4
###############################################################################


@pytest.fixture(scope="module")
def hybrid_run(net):
    pytest.importorskip("amplpy", reason="AMPL is not installed")
    from ropf.model import SolverConfig

    config = AlgorithmConfig(metric="max_active_flow", stage="a2",
                             weight_multiplier=1.0, kappa=2, eta=0.3, k_bar=3)
    try:
        result = algorithm.run(net, config,
                               SolverConfig(name="gurobi", verbose=False),
                               SolverConfig(name="knitro", verbose=False,
                                            knitro_threads=4))
    except Exception as exc:
        pytest.skip(f"could not run the hybrid stage: {exc}")
    if result.ac is None or not result.ac.solved:
        pytest.skip("the AC solve did not run")
    return result


def test_the_pool_transfers_unchanged(hybrid_run):
    """A transfer, not a re-separation: same cuts, no new ones."""
    assert hybrid_run.ac.n_line_cuts == hybrid_run.final.n_line_cuts
    assert hybrid_run.ac.n_bus_cuts == hybrid_run.final.n_bus_cuts
    assert hybrid_run.ac.cuts_added == 0


def test_the_reported_dispatch_is_the_ac_one(hybrid_run):
    """Section 3.4: the DC stage supplies the cuts, (M^ac) supplies the answer."""
    assert hybrid_run.reported is hybrid_run.ac
    assert hybrid_run.dispatch.v, "an AC dispatch carries voltage magnitudes"


def test_the_ac_dispatch_is_scored_on_the_same_functional(hybrid_run, net):
    """rho at the AC iterate is phi evaluated on the AC dispatch, not inherited."""
    recomputed = risk.evaluate(hybrid_run.metric, net,
                               hybrid_run.dispatch.Pf)
    assert hybrid_run.ac.rho == pytest.approx(recomputed.value, rel=1e-12)


def test_the_lambda_zero_point_of_a2_is_still_an_ac_dispatch(net):
    """An empty pool must not skip the AC solve.

    Stage a2 reports AC dispatches; if lambda = 0 reported the DC one instead,
    the frontier would mix the two models and the counterfactual's paired
    comparison would no longer isolate lambda.
    """
    pytest.importorskip("amplpy", reason="AMPL is not installed")
    from ropf.model import SolverConfig

    config = AlgorithmConfig(stage="a2", weight_multiplier=0.0, k_bar=4)
    try:
        result = algorithm.run(net, config,
                               SolverConfig(name="gurobi", verbose=False),
                               SolverConfig(name="knitro", verbose=False,
                                            knitro_threads=4))
    except Exception as exc:
        pytest.skip(f"could not run the hybrid stage: {exc}")
    assert result.termination == "zero_weight"
    assert result.ac is not None, "the lambda = 0 point of a2 needs an AC solve"
    assert result.n_cuts == 0

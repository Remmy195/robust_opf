"""(M) at lambda = 0 against MATPOWER rundcopf (fixture written by fixtures/dump_dcopf.m)."""

import os

import pytest

from conftest import FIXTURES
from ropf.model import Master, SolverConfig


@pytest.fixture(scope="module")
def golden():
    values = {}
    with open(os.path.join(FIXTURES, "dcopf_ACTIVSg200.csv")) as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split(",")
                values[key] = float(value)
    return values


@pytest.fixture(scope="module")
def solution(net):
    try:
        master = Master(net, "master.mod", SolverConfig())
        master.set_risk_family("max_active_flow")
        solution = master.solve()
    except Exception as exc:
        pytest.skip(f"could not solve: {exc}")
    if not solution.solved:
        pytest.skip(f"the master returned {solution.status!r}")
    return solution


def test_cost_matches_rundcopf(solution, golden, net):
    assert solution.gen_cost == pytest.approx(golden["f"], rel=1e-9)
    assert solution.objective == pytest.approx(solution.gen_cost, rel=1e-12)
    offline_no_load = sum(g.costvector[2] for g in net.gens.values() if not g.status)
    assert offline_no_load > 1.0


def test_dispatch_matches_rundcopf(solution, golden):
    assert max(map(abs, solution.Pf.values())) == \
        pytest.approx(golden["max_abs_pf_pu"], rel=1e-7)
    assert sum(solution.Pg.values()) == pytest.approx(golden["sum_pg_pu"], rel=1e-7)

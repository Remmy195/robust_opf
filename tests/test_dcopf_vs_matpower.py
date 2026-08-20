"""End to end: (M) at lambda = 0 is MATPOWER's DC-OPF.

`test_matpower_parity` checks the network model element by element.  This checks
the consequence: solving `modfiles/master.mod` with the risk weight set to zero
and no cuts must reproduce what MATPOWER's own ``rundcopf`` returns on the same
case, in both objective and dispatch.

That is the claim Algorithm 1 line 1 rests on -- the nominal dispatch it starts
from, and the z0 and rho0 that set lambda* -- so it is checked directly.

Needs AMPL and an LP solver, so it skips where the parity test does not.  To
regenerate the golden values::

    OCTAVE_HOME=$HOME/miniconda3 $OCTAVE_HOME/bin/octave-cli --no-gui \
        tests/fixtures/dump_dcopf.m tests/fixtures/case_ACTIVSg200.m \
        tests/fixtures/dcopf_ACTIVSg200.csv
"""

from __future__ import annotations

import os

import pytest

from ropf.network import read_matpower

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

#: Both sides solve the same LP with different codes, so agreement is limited by
#: the solvers' own tolerances rather than by the formulation.  The observed
#: relative difference on ACTIVSg200 is 2.2e-13.
RTOL = 1e-9


def _read_golden(name):
    values = {}
    with open(os.path.join(FIXTURES, name)) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            key, val = line.strip().split(",")
            values[key] = float(val)
    return values


@pytest.fixture(scope="module")
def dc_solution():
    pytest.importorskip("amplpy", reason="AMPL is not installed")
    from ropf.model import Master, SolverConfig

    case = os.path.join(FIXTURES, "case_ACTIVSg200.m")
    net = read_matpower(case)
    try:
        master = Master(net, "master.mod",
                        SolverConfig(name="gurobi", verbose=False))
        master.set_risk_weight(0.0)
        master.set_risk_family("max_active_flow")
        solution = master.solve()
    except Exception as exc:                      # no licence, no solver, ...
        pytest.skip(f"could not solve the DC master: {exc}")
    if not solution.solved:
        pytest.skip(f"DC master returned {solution.status!r}")
    return solution, _read_golden("dcopf_ACTIVSg200.csv"), net


def test_objective_matches_rundcopf(dc_solution):
    """The generation cost at lambda = 0 is MATPOWER's DC-OPF optimum."""
    solution, golden, _ = dc_solution
    assert solution.gen_cost == pytest.approx(golden["f"], rel=RTOL)


def test_dispatch_matches_rundcopf(dc_solution):
    """Same optimum, same flows: the largest line flow agrees."""
    solution, golden, _ = dc_solution
    observed = max(abs(v) for v in solution.Pf.values())
    assert observed == pytest.approx(golden["max_abs_pf_pu"], rel=1e-7)


def test_total_generation_matches_rundcopf(dc_solution):
    solution, golden, _ = dc_solution
    assert sum(solution.Pg.values()) == pytest.approx(
        golden["sum_pg_pu"], rel=1e-7)


def test_offline_units_carry_no_cost(dc_solution):
    """An out-of-service unit contributes nothing, no-load cost included.

    Zeroing only the output bounds leaves the constant term of the cost
    polynomial in the objective, because it is the one term that survives
    Pg == 0.  On ACTIVSg200 that is 11 units and $7,173.15 on every reported
    cost -- large enough to move a headline number, small enough to look like
    a solver difference rather than a modelling error.
    """
    solution, golden, net = dc_solution
    offline_no_load = sum(g.costvector[2] for g in net.gens.values()
                          if not g.status)
    assert offline_no_load > 0, "case has no offline units; test proves nothing"
    # If the no-load costs of offline units leaked in, the cost would sit
    # exactly this far above MATPOWER's.
    assert solution.gen_cost == pytest.approx(golden["f"], rel=RTOL)
    assert abs(solution.gen_cost - (golden["f"] + offline_no_load)) > 1.0


def test_zero_weight_means_no_risk_term(dc_solution):
    """At lambda = 0 the objective is the generation cost alone."""
    solution, _, _ = dc_solution
    assert solution.objective == pytest.approx(solution.gen_cost, rel=1e-12)

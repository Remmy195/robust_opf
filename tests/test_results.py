"""The three artifacts: their schema, their normalization, and nothing else.

Built from synthetic `Result` objects, so this runs without a solver.  What is
under test is the reporting, not the optimization.
"""

from __future__ import annotations

import csv
import json
import os

import pytest

from ropf import results
from ropf.algorithm import AlgorithmConfig, Iterate, Result
from ropf.config import RunConfig
from ropf.model import Solution
from ropf.network import read_matpower

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASE = os.path.join(FIXTURES, "case_ACTIVSg200.m")


@pytest.fixture(scope="module")
def net():
    return read_matpower(CASE)


def _iterate(k, cost, rho, surrogate=None, rho0=10.0):
    surrogate = rho if surrogate is None else surrogate
    return Iterate(k=k, status="solved", cost=cost, surrogate=surrogate,
                   rho=rho, gamma=(rho - surrogate) / rho if rho else 0.0,
                   ratio=rho / rho0, objective=cost, n_line_cuts=2 * k,
                   n_bus_cuts=0, cuts_added=2 if k else 0, solve_time_s=0.5)


def _result(multiplier, cost, rho, stage="baseline", n_gens=3):
    dispatch = Solution(status="solved", objective=cost, gen_cost=cost, phi=rho,
                        Pg={g: 0.5 * g for g in range(1, n_gens + 1)},
                        Pf={1: rho, 2: -0.5})
    return Result(case=CASE, metric="max_active_flow", stage=stage,
                  lambda_star=100.0, risk_weight=100.0 * multiplier,
                  weight_multiplier=multiplier,
                  z0=1000.0, rho0=10.0, phi0=0.0,
                  iterations=[_iterate(0, 1000.0, 10.0), _iterate(1, cost, rho)],
                  termination="eta_target", dispatch=dispatch,
                  nominal_dispatch=dispatch, total_time_s=1.0,
                  config=AlgorithmConfig(weight_multiplier=multiplier))


@pytest.fixture
def sweep():
    return [_result(0.0, 1000.0, 10.0), _result(1.0, 1100.0, 6.0),
            _result(4.0, 1400.0, 4.0)]


###############################################################################
# Exactly three files
###############################################################################


def test_write_leaves_three_artifacts_and_no_others(tmp_path, net, sweep):
    outdir = str(tmp_path / "run")
    results.write(outdir, net, RunConfig(case=CASE), sweep)
    assert sorted(os.listdir(outdir)) == sorted([
        results.SOLUTION_SUMMARY, results.FRONTIER_CSV, results.FRONTIER_JSON])


def test_the_csv_columns_are_the_declared_schema(tmp_path, net, sweep):
    outdir = str(tmp_path / "run")
    paths = results.write(outdir, net, RunConfig(case=CASE), sweep)
    with open(paths["efficient_frontier_csv"]) as handle:
        reader = csv.reader(handle)
        assert tuple(next(reader)) == results.FRONTIER_COLUMNS
        assert len(list(reader)) == len(sweep)


def test_a_column_outside_the_schema_is_an_error(tmp_path):
    """A CSV whose columns depend on the run cannot be read back."""
    with pytest.raises(ValueError, match="outside the declared schema"):
        results._write_csv(str(tmp_path / "f.csv"),
                           [{"case": "x", "surprise": 1}])


def test_the_json_frontier_carries_the_same_rows(tmp_path, net, sweep):
    outdir = str(tmp_path / "run")
    paths = results.write(outdir, net, RunConfig(case=CASE), sweep)
    with open(paths["efficient_frontier_json"]) as handle:
        payload = json.load(handle)
    with open(paths["efficient_frontier_csv"]) as handle:
        rows = list(csv.DictReader(handle))
    assert len(payload["rows"]) == len(rows)
    assert payload["rows"][0]["cost"] == float(rows[0]["cost"])
    assert "provenance" in payload and "config" in payload


###############################################################################
# Normalization
###############################################################################


def test_the_frontier_normalizes_against_the_stage_own_zero_weight_row(net, sweep):
    rows = results.frontier_rows(net, RunConfig(case=CASE), sweep)
    base = rows[0]
    assert base["weight_multiplier"] == 0.0
    assert base["cost_ratio"] == pytest.approx(1.0)
    assert base["risk_reduction"] == pytest.approx(0.0)
    assert rows[1]["risk_ratio"] == pytest.approx(6.0 / 10.0)
    assert rows[1]["cost_ratio"] == pytest.approx(1100.0 / 1000.0)


def test_a_sweep_without_a_zero_weight_row_reports_the_base_as_missing(net):
    """A missing base is reported as missing, never substituted."""
    rows = results.frontier_rows(net, RunConfig(case=CASE),
                                 [_result(1.0, 1100.0, 6.0)])
    assert rows[0]["risk_ratio"] != rows[0]["risk_ratio"]      # NaN
    assert rows[0]["cost_ratio"] != rows[0]["cost_ratio"]


###############################################################################
# Dominance
###############################################################################


def test_a_genuinely_dominated_point_is_flagged_and_kept(net):
    sweep = [_result(0.0, 1000.0, 10.0), _result(1.0, 1200.0, 8.0),
             _result(2.0, 1100.0, 6.0)]
    rows = results.frontier_rows(net, RunConfig(case=CASE), sweep)
    assert [r["dominated"] for r in rows] == [False, True, False]
    assert len(rows) == len(sweep), "a dominated point is information, not noise"


def test_solver_noise_is_not_reported_as_dominance(net):
    """Two weights reaching the same dispatch differ in the tenth figure."""
    sweep = [_result(1.0, 29963.176996248476, 1.707500000002562),
             _result(2.0, 29963.176965744464, 1.707500000000007)]
    rows = results.frontier_rows(net, RunConfig(case=CASE), sweep)
    assert not any(r["dominated"] for r in rows)


###############################################################################
# What the counterfactual reads back
###############################################################################


def test_the_dispatch_round_trips(tmp_path, net, sweep):
    """The counterfactual takes P* as given; a summary without it is unusable."""
    outdir = str(tmp_path / "run")
    paths = results.write(outdir, net, RunConfig(case=CASE), sweep)
    recovered = results.read_dispatch(paths["solution_summary"], 1.0)
    assert recovered == sweep[1].dispatch.Pg


def test_reading_a_weight_the_sweep_does_not_hold_says_what_it_does(
        tmp_path, net, sweep):
    outdir = str(tmp_path / "run")
    paths = results.write(outdir, net, RunConfig(case=CASE), sweep)
    with pytest.raises(KeyError) as exc:
        results.read_dispatch(paths["solution_summary"], 0.5)
    assert "0.0" in str(exc.value) and "4.0" in str(exc.value)


def test_line_flows_are_summarized_not_dumped(tmp_path, net, sweep):
    """At 88,207 branches over six weights the vectors would be most of the file."""
    outdir = str(tmp_path / "run")
    paths = results.write(outdir, net, RunConfig(case=CASE), sweep)
    with open(paths["solution_summary"]) as handle:
        summary = json.load(handle)
    dispatch = summary["runs"][0]["dispatch"]
    assert "Pf_pu" not in dispatch
    assert dispatch["max_abs_Pf_pu"] == pytest.approx(10.0)


def test_provenance_pins_the_case_file(net):
    """Two copies of ACTIVSg2000.m can differ; the digest is what pins them."""
    record = results.provenance(net)
    assert record["case_sha256"] and len(record["case_sha256"]) == 64
    assert record["ropf_version"]

"""The frontier tracer: the box, the bisection, and the grid it recommends.

Every test here runs against a SYNTHETIC frontier, not a solver.  What the
tracer is is a geometry -- an endpoint walk, a fixed normalization and a
longest-chord bisection -- and a geometry tested through Gurobi is tested
through something that can disagree with it for reasons of its own.  The one
thing the stub keeps faithful is the shape the real thing has: tau saturates on
a flat tail, because Algorithm 1 exits the moment rho^k/rho^0 <= 1 - eta, and
that tail is what the tracer exists to find the edge of.
"""

from __future__ import annotations

import json
import os

import pytest

from ropf import algorithm
from ropf.config import ConfigError, RunConfig
from ropf.study import frontier

Z0 = 1000.0
RHO0 = 10.0
#: Where tau stops falling.  With eta = 0.5 the real loop stops at 0.5; this is
#: the same shape at a different floor, so a test cannot pass by knowing 0.5.
TAIL = 0.4


class FakeNetwork:
    """Everything the tracer touches on a network, which is one attribute."""

    casefile = "case_FAKE.m"


def curve(multiplier: float) -> tuple:
    """A monotone frontier with a flat tail: tau falls, then stops."""
    tau = max(TAIL, 1.0 / (1.0 + 3.0 * multiplier))
    return Z0 * (1.0 + 0.5 * (1.0 - tau)), tau


def result_at(multiplier: float, cost: float, tau: float,
              status: str = "solved",
              termination: str = "eta_target") -> algorithm.Result:
    iterate = algorithm.Iterate(
        k=1, status=status, cost=cost, surrogate=tau * RHO0 * 0.9,
        rho=tau * RHO0, gamma=0.1, ratio=tau, objective=cost,
        n_line_cuts=1, n_bus_cuts=0, cuts_added=1, solve_time_s=0.01)
    return algorithm.Result(
        case="case_FAKE.m", metric="max_active_flow", stage="baseline",
        lambda_star=Z0 / RHO0, risk_weight=multiplier * Z0 / RHO0,
        weight_multiplier=multiplier, z0=Z0, rho0=RHO0, phi0=RHO0 * 0.9,
        iterations=[iterate], termination=termination)


@pytest.fixture
def solved(monkeypatch):
    """Replace `algorithm.run` with the synthetic curve.  Records every call."""
    calls = []

    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        multiplier = config.weight_multiplier
        calls.append(multiplier)
        cost, tau = curve(multiplier)
        return result_at(multiplier, cost, tau,
                         termination=("zero_weight" if multiplier == 0.0
                                      else "eta_target"))

    monkeypatch.setattr(algorithm, "run", fake_run)
    return calls


def run_trace(**kwargs) -> frontier.FrontierTrace:
    config = RunConfig(case="case_FAKE.m", **kwargs)
    return frontier.trace(config, None, FakeNetwork())


###############################################################################
# What it refuses
###############################################################################


def test_an_absolute_lambda_has_no_frontier_to_trace(solved):
    """`risk_weight` pins lambda, so every point would be the same run."""
    config = RunConfig(case="case_FAKE.m", risk_weight=500.0)
    with pytest.raises(ConfigError, match="risk_weight"):
        frontier.trace(config, None, FakeNetwork())
    assert solved == [], "it must refuse before solving anything"


def test_a_config_with_no_case_is_refused(solved):
    with pytest.raises(ConfigError, match="nothing to trace"):
        frontier.trace(RunConfig(), None, FakeNetwork())


@pytest.mark.parametrize("kwargs", [
    {"frontier_seed": 0.0}, {"frontier_seed": -1.0},
    {"frontier_hi_cap": 0.1}, {"frontier_max_points": 1},
    {"frontier_tol": 0.0}, {"frontier_saturation_tol": 0.0},
    {"frontier_saturation_tol": 1.0}, {"frontier_grid_points": 1},
])
def test_the_tracer_keys_are_validated_when_the_config_is_read(kwargs):
    """A tracer parameter is refused when the file is read, not an hour in."""
    with pytest.raises(ConfigError):
        RunConfig(**kwargs)


###############################################################################
# The endpoints
###############################################################################


def test_the_left_endpoint_is_the_nominal_solve(solved):
    out = run_trace()
    left = out.left
    assert left is not None
    assert left.multiplier == 0.0
    assert left.tau == pytest.approx(1.0)
    assert left.cost == pytest.approx(Z0)
    assert left.lambda_star == pytest.approx(Z0 / RHO0)
    assert solved[0] == 0.0, "the nominal solve comes first; it anchors the box"


def test_the_walk_doubles_from_the_seed_and_stops_on_the_tail(solved):
    out = run_trace(frontier_seed=0.25, frontier_saturation_tol=0.01)
    assert out.walk_outcome == "saturated"

    # The doubling sequence, which `_confirm_saturation` may follow with one
    # probe at the cap; that probe is not part of the doubling.
    walk = sorted(p.multiplier for p in out.points
                  if p.role in ("walk", "right"))
    doubling = [m for m in walk if m < out.grid_resolution + 64.0]
    doubling = [m for m in walk if m != 64.0]
    assert doubling == pytest.approx(
        [0.25 * 2 ** i for i in range(len(doubling))]), \
        "the walk must double, not step"
    assert solved[:1 + len(doubling)] == pytest.approx([0.0] + doubling), \
        "the endpoints are found before anything is refined"

    right = out.right
    assert right is not None
    assert right.tau == pytest.approx(TAIL), \
        "a saturated walk ends on the flat tail"
    assert right.multiplier == min(
        p.multiplier for p in out.usable_points
        if p.tau == pytest.approx(TAIL)), \
        "the endpoint is the cheapest weight reaching the floor"


def test_the_seed_always_gets_one_doubling(solved):
    """A tiny seed means the frontier has not started, not that it has ended.

    Comparing the seed against the nominal point would stop the walk there.
    """
    out = run_trace(frontier_seed=1e-6, frontier_saturation_tol=0.01)
    assert len([m for m in solved if m > 0]) >= 2
    assert out.right.multiplier > 1e-6


def test_a_false_plateau_does_not_end_the_walk(monkeypatch):
    """A staircase frontier: flat, then a second drop further out.

    MEASURED on ACTIVSg500 under max_active_flow -- tau is 0.7810 at
    lambda = 0.5, 0.7807 at 1 and 0.7807 at 2, then 0.7181 at 4.  A walk that
    believes the first plateau stops at 1 and never sees the rest, so the
    recommended grid spans LESS of the frontier than the geometric grid it was
    meant to improve on.
    """
    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        m = config.weight_multiplier
        tau = 1.0 if m < 0.25 else (0.781 if m < 4.0 else 0.718)
        return result_at(m, Z0 * (1.0 + 0.5 * (1.0 - tau)), tau)

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace(frontier_seed=0.25, frontier_hi_cap=64.0,
                    frontier_saturation_tol=0.005, frontier_max_points=16)

    assert out.walk_outcome == "capped", \
        "the plateau at lambda = 1 was mistaken for the tail"
    assert out.right.tau == pytest.approx(0.718), \
        "the right endpoint must reach past the plateau"
    assert max(p.multiplier for p in out.usable_points) >= 4.0


def test_a_real_plateau_is_still_saturation(monkeypatch):
    """The cap probe must not turn every saturated walk into a capped one."""
    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        m = config.weight_multiplier
        tau = max(TAIL, 1.0 / (1.0 + 3.0 * m))
        return result_at(m, Z0 * (1.0 + 0.5 * (1.0 - tau)), tau)

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace(frontier_seed=0.25, frontier_saturation_tol=0.01)
    assert out.walk_outcome == "saturated"


def test_the_cap_is_recorded_as_a_cap_not_as_saturation(solved):
    """Reaching the cap means the tail was never found, and says so."""
    out = run_trace(frontier_seed=0.25, frontier_hi_cap=0.5,
                    frontier_saturation_tol=1e-9)
    assert out.walk_outcome == "capped"
    assert out.right.multiplier == pytest.approx(0.5)
    assert max(solved) <= 0.5, "the cap is a cap"


def test_a_zero_risk_nominal_stops_with_one_point(monkeypatch):
    """rho0 = 0 makes lambda* and tau undefined: there is no frontier."""
    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        result = result_at(0.0, Z0, float("nan"),
                           termination="nominal_risk_zero")
        result.rho0 = 0.0
        result.lambda_star = float("nan")
        return result

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace()
    assert out.termination == "nominal_risk_zero"
    assert len(out.points) == 1 and out.n_solves == 1
    assert out.normalization is None


def test_an_infeasible_nominal_stops_the_trace(monkeypatch):
    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        return result_at(0.0, float("nan"), float("nan"), status="infeasible",
                         termination="infeasible")

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace()
    assert out.termination == "infeasible"
    assert out.points[0].usable is False


def test_a_frontier_that_does_not_move_is_named(monkeypatch):
    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        return result_at(config.weight_multiplier, Z0, 1.0)

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace()
    assert out.termination == "flat_frontier"
    assert out.normalization.cost_is_flat and out.normalization.tau_is_flat


###############################################################################
# The box
###############################################################################


def test_the_box_puts_the_endpoints_at_the_corners(solved):
    out = run_trace()
    norm = out.normalization
    assert (norm.cost_of(out.left.cost), norm.tau_of(out.left.tau)) == \
        pytest.approx((0.0, 0.0))
    assert (norm.cost_of(out.right.cost), norm.tau_of(out.right.tau)) == \
        pytest.approx((1.0, 1.0))


def test_every_point_lands_inside_the_box(solved):
    """The frontier is monotone, so no solved point can sit outside [0,1]^2."""
    out = run_trace()
    norm = out.normalization
    for point in out.usable_points:
        assert -1e-9 <= norm.cost_of(point.cost) <= 1.0 + 1e-9
        assert -1e-9 <= norm.tau_of(point.tau) <= 1.0 + 1e-9


def test_the_box_is_anchored_on_the_reported_values_not_on_z0(monkeypatch):
    """Stage a2 reports an AC cost against a DC z0; the box uses the AC one.

    Anchoring on `Result.z0` there would put the left endpoint off the origin
    and carry the DC-to-AC model change into the chord lengths as if it were a
    response to lambda.  Same rule as `ropf.results._normalize_within_stage`.
    """
    ac_offset = 137.0

    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        multiplier = config.weight_multiplier
        cost, tau = curve(multiplier)
        result = result_at(multiplier, cost, tau)
        # What stage a2 does: the reported iterate is the AC solve, on a cost
        # base of its own, while z0 stays the DC nominal.
        result.ac = result.iterations[-1]
        result.z0 = cost - ac_offset if multiplier == 0.0 else Z0
        return result

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace(stage="a2")
    assert out.normalization.cost_lo == pytest.approx(out.left.cost)
    assert out.normalization.cost_lo != pytest.approx(out.left.z0)
    assert out.normalization.cost_of(out.left.cost) == pytest.approx(0.0)


###############################################################################
# Refinement
###############################################################################


def test_refinement_splits_at_the_arithmetic_midpoint(solved):
    """Every refined point is the midpoint in m of two points already solved.

    `solved` is the call order, so the check is against what was known at the
    moment each split was chosen rather than against the finished trace.
    """
    out = run_trace(frontier_max_points=8, frontier_tol=1e-6)
    n_walk = sum(1 for p in out.points if p.role in ("left", "walk", "right"))
    assert len(solved) > n_walk, "the trace refined nothing"

    for index in range(n_walk, len(solved)):
        earlier = sorted(solved[:index])
        midpoints = [0.5 * (a + b) for a, b in zip(earlier, earlier[1:])]
        assert any(solved[index] == pytest.approx(mid) for mid in midpoints), \
            f"{solved[index]} is not the midpoint of an adjacent pair"


def test_the_point_budget_is_a_solve_budget(solved):
    out = run_trace(frontier_max_points=7, frontier_tol=1e-9)
    assert len(out.points) == 7
    assert out.n_solves == 7, "the memo makes points and solves the same number"
    assert len(solved) == 7, "no multiplier is solved twice"
    assert out.termination == "point_budget"


def test_the_chord_tolerance_stops_before_the_budget(solved):
    out = run_trace(frontier_max_points=40, frontier_tol=0.35)
    assert out.termination == "chord_tolerance"
    assert out.longest_chord < 0.35
    assert len(out.points) < 40


def test_the_longest_chord_is_the_one_that_gets_split(solved):
    """After the trace stops, no adjacent pair may exceed the tolerance."""
    out = run_trace(frontier_max_points=40, frontier_tol=0.2)
    points = sorted(out.usable_points, key=lambda p: p.multiplier)
    norm = out.normalization
    for a, b in zip(points, points[1:]):
        assert frontier._chord(norm, a, b) < 0.2


def test_refinement_concentrates_points_where_the_frontier_bends(solved):
    """The point of the whole module: not on the flat tail.

    The geometric grid puts five of its six points above lambda*; the tracer
    must not, because above the bend every dispatch is the same dispatch.
    """
    out = run_trace(frontier_max_points=12, frontier_tol=1e-9)
    right = out.right.multiplier
    # Inside the traced range; the cap probe of `_confirm_saturation` sits
    # outside it and is diagnostic, not a point on the frontier.
    inside = [p for p in out.usable_points if p.multiplier <= right]
    below = [p for p in inside if p.multiplier < right / 2.0]
    assert len(below) > len(inside) / 2, \
        "most points must sit below half the saturating weight"


def test_a_point_is_never_solved_twice(solved):
    run_trace(frontier_max_points=16, frontier_tol=1e-9)
    assert len(solved) == len(set(solved))


def test_an_infeasible_interior_point_is_kept_and_not_retried(monkeypatch):
    """A failed split is a record that the multiplier was tried, not a gap."""
    failed = []

    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        multiplier = config.weight_multiplier
        cost, tau = curve(multiplier)
        if multiplier not in (0.0,) and multiplier % 0.25 != 0.0:
            failed.append(multiplier)
            return result_at(multiplier, float("nan"), float("nan"),
                             status="infeasible", termination="infeasible")
        return result_at(multiplier, cost, tau)

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace(frontier_max_points=10, frontier_tol=1e-9)
    assert failed, "the test did not exercise a failed split"
    assert len(failed) == len(set(failed)), "a failed split was retried"
    assert any(p.usable is False for p in out.points)
    assert out.termination in ("point_budget", "no_split_left")


def test_a_step_in_the_frontier_is_not_bisected_forever(monkeypatch):
    """The defect this floor exists for, on the shape that produced it.

    On ACTIVSg200 under max_active_flow the dispatch jumps once, near
    lambda = 0.167 lambda*: tau goes 1 -> 0.46 between two adjacent
    multipliers. The chord across that pair is the full diagonal of the box and
    does NOT shrink when the pair is split, so a longest-chord rule with no
    floor bisects into the jump until the budget is gone, reports
    "point_budget" as though more points would have helped, and recommends six
    weights within a thousandth of each other.
    """
    step = 0.167

    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        multiplier = config.weight_multiplier
        tau = 1.0 if multiplier < step else TAIL
        return result_at(multiplier, Z0 * (1.0 + 0.5 * (1.0 - tau)), tau)

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace(frontier_max_points=40, frontier_tol=0.05)

    assert out.termination == "no_split_left"
    assert "STEP in the frontier" in out.termination_detail
    assert len(out.points) < 40, "the step ate the whole budget"

    points = sorted(out.usable_points, key=lambda p: p.multiplier)
    narrowest = min(b.multiplier - a.multiplier
                    for a, b in zip(points, points[1:]))
    assert narrowest > out.grid_resolution / 2.0, \
        "refinement went finer than the recommendation can express"


def test_a_step_recommends_the_risk_levels_it_actually_has(monkeypatch):
    """Two dispatches means two grid points, not six weights on two dispatches.

    Padding back up to K would put four more weights in the campaign that
    evaluate a dispatch already in it -- the near-duplicate evaluation this
    module exists to stop, moved from the tail to the step.
    """
    step = 0.167

    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        multiplier = config.weight_multiplier
        tau = 1.0 if multiplier < step else TAIL
        return result_at(multiplier, Z0 * (1.0 + 0.5 * (1.0 - tau)), tau)

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace(frontier_max_points=40, frontier_tol=0.05,
                    frontier_grid_points=6)

    by_multiplier = {round(p.multiplier, out.grid_decimals): p
                     for p in out.usable_points}
    taus = sorted({round(by_multiplier[m].tau, 9)
                   for m in out.recommended_weight_grid})
    assert len(out.recommended_weight_grid) == len(taus) == 2, \
        f"a two-level frontier got {out.recommended_weight_grid}"
    assert taus == [pytest.approx(TAIL), pytest.approx(1.0)]


def test_identical_tau_picks_the_cheapest_weight_not_the_furthest(monkeypatch):
    """MEASURED on ACTIVSg200 under bus_flow_sum_agg: the walk returns
    tau = 0.7204100 at lambda = 0.5, 1 and 64 -- the same dispatch three times
    -- but the three floats differ in the twelfth digit.  An exact `min` picked
    lambda = 64, and the recommended grid ran out to a weight sixteen times
    past anything the paper states, to buy nothing."""
    import itertools
    wobble = itertools.count()

    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        m = config.weight_multiplier
        if m < 0.25:
            tau = 1.0
        elif m < 0.5:
            tau = 0.76591
        else:
            # The same dispatch, spelled with last-digit float noise.
            tau = 0.72041 + (next(wobble) % 3 - 1) * 1e-12
        return result_at(m, Z0 * (1.0 + 0.5 * (1.0 - tau)), tau)

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace(frontier_seed=0.25, frontier_hi_cap=64.0,
                    frontier_saturation_tol=0.005, frontier_max_points=10)

    assert out.right.multiplier <= 1.0, \
        f"the endpoint ran out to {out.right.multiplier}, past where tau stopped moving"
    assert max(out.recommended_weight_grid) <= 1.0, \
        f"the grid names a useless far-out weight: {out.recommended_weight_grid}"


def test_the_last_grid_point_is_re_chosen_after_refinement(monkeypatch):
    """MEASURED on ACTIVSg500 under max_active_flow: the walk sees tau 0.7807
    at lambda = 1 and 0.7178 at 64, so its endpoint is 64.  Refinement then
    finds lambda = 3.46 reaching 0.7181 -- the same risk within the saturation
    threshold, for $88.7k against $89.2k.  The grid must name 3.46, not 64."""
    # Gradual, as the real one is -- a pure step would send refinement
    # step-chasing and never reach the far plateau at all, which is a
    # different defect with its own test above.
    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        m = config.weight_multiplier
        if m <= 1.0:
            tau = 1.0 - 0.2193 * m                       # 1.0 -> 0.7807
        elif m < 3.4:
            tau = 0.7807 - 0.06261 * (m - 1.0) / 2.4     # -> 0.71809
        elif m < 32.0:
            tau = 0.71809                                # the floor
        else:
            tau = 0.71780                                # noise past it
        return result_at(m, Z0 * (1.0 + 0.5 * (1.0 - tau)), tau)

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace(frontier_seed=0.25, frontier_hi_cap=64.0,
                    frontier_saturation_tol=0.005, frontier_max_points=20)

    top = max(out.recommended_weight_grid)
    assert 3.0 <= top < 32.0, \
        f"the grid should end where the floor is first reached, got {top}"
    assert out.right.multiplier < 32.0, \
        "the endpoint sat past the floor, where more weight buys nothing"


###############################################################################
# The recommendation
###############################################################################


def test_the_recommendation_carries_both_endpoints(solved):
    out = run_trace(frontier_max_points=12, frontier_tol=1e-9,
                    frontier_grid_points=6)
    grid = out.recommended_weight_grid
    assert grid[0] == 0.0, "the campaign needs its lambda = 0 base"
    assert grid[-1] == pytest.approx(round(out.right.multiplier,
                                           out.grid_decimals))


def test_the_recommendation_is_k_points_ascending(solved):
    out = run_trace(frontier_max_points=12, frontier_tol=1e-9,
                    frontier_grid_points=6)
    grid = out.recommended_weight_grid
    assert len(grid) == 6
    assert list(grid) == sorted(grid)
    assert len(set(grid)) == len(grid)


def test_the_recommendation_spreads_in_tau_not_in_lambda(solved):
    """Evenly in tau is the whole point: K distinguishable risk levels.

    A grid spread evenly in lambda would put most of its points on the tail,
    where every dispatch is the same dispatch and the campaign re-evaluates
    near-duplicates.
    """
    out = run_trace(frontier_max_points=14, frontier_tol=1e-9,
                    frontier_grid_points=6)
    # The grid is snapped, so points are matched to it on the snapped value.
    by_multiplier = {round(p.multiplier, out.grid_decimals): p
                     for p in out.usable_points}
    taus = [by_multiplier[m].tau for m in out.recommended_weight_grid]
    assert len(taus) == 6, "every recommended multiplier is a solved point"

    gaps = [a - b for a, b in zip(taus, taus[1:])]
    assert all(gap > 0 for gap in gaps), "no two grid points share a tau"
    even = (out.left.tau - out.right.tau) / (len(taus) - 1)
    assert max(gaps) < 3.0 * even, \
        f"the tau spacing is not even: {gaps}"


@pytest.mark.parametrize("span, decimals", [
    (64.0, 4), (1.0, 4), (0.5, 5), (0.002, 7), (1e-9, 9),
])
def test_the_precision_keeps_four_digits_across_the_traced_span(span, decimals):
    """Coarse where the frontier is wide, fine where it is narrow."""
    assert frontier._grid_decimals(span) == decimals


def test_a_narrow_bend_is_not_rounded_into_one_grid_point(monkeypatch):
    """The defect the adaptive precision exists for, on the shape that caused it.

    On ACTIVSg200 the whole bend is 0.002 wide in lambda/lambda*, tau falling
    from 1 to 0.46 across it.  At a fixed four decimals two points carrying
    tau of 0.602 and 0.579 both write down as 0.1675, so a grid that named
    them would silently be one risk level short of what the trace found.
    """
    lo, hi = 0.166, 0.168

    def fake_run(network, config, solver=None, ac_solver=None, log=None):
        multiplier = config.weight_multiplier
        fraction = min(1.0, max(0.0, (multiplier - lo) / (hi - lo)))
        tau = 1.0 - (1.0 - TAIL) * fraction
        return result_at(multiplier, Z0 * (1.0 + 0.5 * (1.0 - tau)), tau)

    monkeypatch.setattr(algorithm, "run", fake_run)
    out = run_trace(frontier_max_points=24, frontier_tol=1e-9,
                    frontier_grid_points=6)

    assert out.grid_decimals > 4, "a 0.002-wide bend needs more than 4 decimals"
    grid = out.recommended_weight_grid
    assert len(grid) == 6, f"a risk level was rounded away: {grid}"

    by_multiplier = {round(p.multiplier, out.grid_decimals): p
                     for p in out.usable_points}
    taus = [by_multiplier[m].tau for m in grid]
    assert len(set(taus)) == 6, "two grid points carry the same risk level"


def test_the_recommendation_is_snapped_so_it_can_be_written_down(solved):
    out = run_trace(frontier_max_points=16, frontier_tol=1e-9)
    for multiplier in out.recommended_weight_grid:
        assert multiplier == round(multiplier, out.grid_decimals)
    line = out.weight_grid_line()
    assert line.startswith("weight_grid = ")
    parsed = tuple(float(p) for p in line.split("=", 1)[1].split(","))
    assert parsed == pytest.approx(out.recommended_weight_grid)


def test_the_recommended_line_is_a_config_the_parser_accepts(solved, tmp_path):
    """The output is meant to be pasted; this pastes it."""
    from ropf.config import read_config

    out = run_trace(frontier_max_points=12, frontier_tol=1e-9)
    path = tmp_path / "pasted.conf"
    path.write_text(out.weight_grid_line() + "\n")
    assert read_config(str(path)).weight_grid == \
        pytest.approx(out.recommended_weight_grid)


def test_fewer_points_than_k_recommends_what_there_is(solved):
    out = run_trace(frontier_max_points=4, frontier_tol=1e-9,
                    frontier_grid_points=6)
    inside = [p for p in out.usable_points
              if p.multiplier <= out.right.multiplier]
    assert len(out.recommended_weight_grid) == len(inside) < 6


def test_the_point_budget_bounds_the_walk_as_well_as_the_refinement(solved):
    """A budget that only bounded step 4 would be no budget at all: the walk
    doubles, and on a large instance those runs are the cost that matters."""
    # A seed far below the bend, so tau is still falling fast when the
    # budget runs out: the walk stops for the budget and not for the tail.
    out = run_trace(frontier_max_points=4, frontier_seed=0.001,
                    frontier_hi_cap=1024.0, frontier_saturation_tol=1e-12)
    assert out.walk_outcome == "budget"
    assert len(out.points) == 4 and out.n_solves == 4
    assert out.right is not None, "a budgeted walk still leaves an endpoint"


###############################################################################
# What it leaves behind, and what it leaves alone
###############################################################################


def test_the_trace_writes_one_file_and_it_round_trips(solved, tmp_path):
    config = RunConfig(case="case_FAKE.m", frontier_max_points=6)
    out = frontier.trace(config, None, FakeNetwork())
    path = frontier.write(str(tmp_path), config, out, None)

    assert os.path.basename(path) == frontier.TRACE_JSON
    assert sorted(os.listdir(tmp_path)) == [frontier.TRACE_JSON], \
        "the tracer writes one file; the three artifacts are results.write's"

    with open(path) as handle:
        payload = json.load(handle)
    assert payload["trace"]["recommended_weight_grid"] == \
        list(out.recommended_weight_grid)
    assert len(payload["trace"]["points"]) == len(out.points)
    assert payload["config"]["weight_grid"] == list(config.weight_grid), \
        "the trace records the grid it did NOT change"


def test_the_production_grid_is_untouched(solved):
    """The tracer recommends; it does not edit.  Nothing it runs may move
    WEIGHT_GRID, the config's grid, or the ladder's."""
    from ropf.study.ladder import LadderConfig

    before = algorithm.WEIGHT_GRID
    config = RunConfig(case="case_FAKE.m", frontier_max_points=6)
    frontier.trace(config, None, FakeNetwork())
    assert algorithm.WEIGHT_GRID == before
    assert config.weight_grid == before
    assert LadderConfig(damping_per_load=1.0).weight_grid == before


def test_the_tracer_never_touches_the_campaign(solved):
    """It calls `algorithm.run` and nothing else that solves."""
    import inspect

    source = inspect.getsource(frontier)
    for forbidden in ("evaluate_campaign", "build_campaign", "PostEvent",
                      "disfigure", "run_combo"):
        assert forbidden not in source, \
            f"the tracer reaches into the campaign: {forbidden}"


def test_the_termination_is_always_a_declared_one(solved):
    for kwargs in ({}, {"frontier_max_points": 3, "frontier_tol": 1e-9},
                   {"frontier_tol": 5.0}, {"frontier_hi_cap": 0.25}):
        out = run_trace(**kwargs)
        assert out.termination in frontier.TRACE_TERMINATIONS
        assert out.walk_outcome in frontier.WALK_OUTCOMES + ("",)

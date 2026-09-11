"""An adaptive tracer for the cost-risk frontier of ONE case, metric and stage.

WHAT THIS IS FOR.  The ladder campaign runs the full Section 4 evaluation at
every point of `weight_grid`.  With a geometric grid and eta = 0.5 most of that
grid lands on the flat tail -- every weight reaching the eta floor returns a
dispatch at the same tau -- so the campaign re-evaluates near-duplicates.  This
module finds where the frontier BENDS and recommends K multipliers to paste
into the ladder config.

WHAT THIS IS NOT.  It never runs the campaign, writes no frontier artifact, and
nothing here is read by `ropf solve` or by the ladder.  Its output is a
recommendation a human pastes in, which is what keeps the campaign grid a
declared constant rather than something a solver chose mid-study.

THE FRONTIER IS MONOTONE, which is what makes it traceable.  Each point is one
run of Algorithm 1 at lambda = m lambda*:

    P(m) = (cost, tau)      cost = Result.reported.cost
                            tau  = Result.reported.ratio = rho^end/rho^0

tau is non-increasing and cost non-decreasing in m, so the solved points in m
order are the frontier in order and the gap between adjacent points is a
well-defined length to bisect.

    1  LEFT ENDPOINT.  m = 0.  Anchors the box and defines lambda* = z0/rho0.
    2  RIGHT ENDPOINT.  Double m from `frontier_seed` until tau stops moving --
       the eta floor -- or `frontier_hi_cap` is hit.  Which one is recorded: a
       capped walk means the box is not the whole frontier.
    3  NORMALIZE ONCE to those endpoints, into [0,1]^2, and never again.  A box
       that moved as points were added would make an early chord and a late one
       incomparable.
    4  REFINE.  Split the adjacent pair with the longest normalized chord at the
       midpoint in m, until the longest chord is below `frontier_tol` or
       `frontier_max_points` points are solved.

`frontier_max_points` bounds the WHOLE trace, not just step 4: a doubling walk
from 0.25 to a cap of 64 is nine runs before a single point is placed.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Optional,
                    Sequence, Tuple)

from .. import algorithm, results as results_module
from ..config import ConfigError, RunConfig
from ..network import Network, read_matpower

#: The one artifact a trace leaves behind -- a recommendation, not a result,
#: written to its own directory so it cannot overwrite a `ropf solve`.
TRACE_JSON = "frontier_trace.json"

#: Decimals the multiplier is rounded to before use as a memo key; the rounded
#: value is what is solved.  Bisection reaches the same point by paths differing
#: in the last bits, and without a stable key the memo misses.
MULTIPLIER_DECIMALS = 12

#: Significant digits the recommended grid keeps across the traced span, and
#: with it the finest interval refinement will split -- the same number, since
#: refining below what the recommendation can express buys nothing.
#:
#: ADAPTIVE, because a fixed precision is either too coarse or illegible.  On
#: ACTIVSg200 under max_active_flow the whole bend is 0.002 wide in m, tau
#: falling 1 to 0.46 across it, and four decimals collapses distinct risk levels
#: onto one point: 0.16748 and 0.167542 carry tau 0.602 and 0.579 and both write
#: as 0.1675.
GRID_SIGNIFICANT = 4
GRID_DECIMALS_MIN = 4
GRID_DECIMALS_MAX = 9

#: The floor exists because WITHOUT ONE A STEP IN THE FRONTIER EATS THE WHOLE
#: BUDGET: the chord across a pair spanning a sharp switch is the full diagonal
#: and does not shrink when split, so a longest-chord rule bisects into the jump
#: until it runs out of points.  Below the floor there is nothing between the
#: two sides for the recommendation to name.


def _grid_decimals(span: float) -> int:
    """Decimals that keep `GRID_SIGNIFICANT` digits across `span`, clamped."""
    if not math.isfinite(span) or span <= 0.0:
        return GRID_DECIMALS_MIN
    wanted = int(math.ceil(GRID_SIGNIFICANT - math.log10(span)))
    return max(GRID_DECIMALS_MIN, min(GRID_DECIMALS_MAX, wanted))

#: Two points whose tau differs by less than this fraction of the target
#: spacing are one risk level, and will not get two of the K points.
TAU_DISTINCT_FRACTION = 0.05

#: A span below this, relative to the endpoint it is measured from, is no span
#: at all: dividing by the tenth-significant-figure difference between two
#: weights that reach the same dispatch would turn LP noise into a full unit of
#: normalized length.  As `ropf.results.DOMINANCE_RTOL`.
SPAN_RTOL = 1e-9

#: Every way a trace can stop.  A stop for a reason not in this list is a bug.
TRACE_TERMINATIONS = (
    "chord_tolerance",     # every adjacent chord is shorter than frontier_tol
    "point_budget",        # frontier_max_points points have been solved
    "no_split_left",       # every splittable pair's midpoint is already solved
    "nominal_risk_zero",   # rho0 <= 0: lambda* and tau are undefined
    "flat_frontier",       # the walk moved neither cost nor tau
    "infeasible",          # an endpoint solve did not solve
)

#: How the step-2 walk ended.  "budget" means `frontier_max_points` was reached
#: before the tail or the cap, so the endpoint is wherever the walk got to.
WALK_OUTCOMES = ("saturated", "capped", "budget", "infeasible")


def _noop(_message: str) -> None:
    return None


###############################################################################
# Points
###############################################################################


@dataclass
class FrontierPoint:
    """One solved multiplier: the two coordinates, and enough to audit them."""

    multiplier: float
    #: z at the reported dispatch.  The AC one where the stage has an AC stage.
    cost: float
    #: tau = rho^end/rho^0, the quantity the eta exit tests.
    tau: float
    risk_weight: float
    lambda_star: float
    z0: float
    rho0: float
    #: Where this point came from: the left endpoint, the doubling walk, the
    #: right endpoint the walk stopped at, or a refinement split.
    role: str
    termination: str
    status: str
    k_end: int
    n_cuts: int
    solve_time_s: float
    total_time_s: float
    #: False when the point cannot carry geometry -- the solve failed, or a
    #: coordinate is not finite.  Kept as the record that the multiplier was
    #: tried, and excluded from the chord arithmetic.
    usable: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "cost": self.cost,
            "tau": self.tau,
            "risk_weight": self.risk_weight,
            "lambda_star": self.lambda_star,
            "z0": self.z0,
            "rho0": self.rho0,
            "role": self.role,
            "termination": self.termination,
            "status": self.status,
            "k_end": self.k_end,
            "n_cuts": self.n_cuts,
            "solve_time_s": self.solve_time_s,
            "total_time_s": self.total_time_s,
            "usable": self.usable,
        }


def _point(result: algorithm.Result, multiplier: float,
           role: str) -> FrontierPoint:
    """One run of Algorithm 1, read as a frontier point."""
    reported = result.reported
    usable = (reported.status == "solved"
              and math.isfinite(reported.cost) and math.isfinite(reported.ratio))
    return FrontierPoint(
        multiplier=float(multiplier),
        cost=reported.cost,
        tau=reported.ratio,
        risk_weight=result.risk_weight,
        lambda_star=result.lambda_star,
        z0=result.z0,
        rho0=result.rho0,
        role=role,
        termination=result.termination,
        status=reported.status,
        k_end=result.k_end,
        n_cuts=reported.n_line_cuts + reported.n_bus_cuts,
        solve_time_s=reported.solve_time_s,
        total_time_s=result.total_time_s,
        usable=usable)


###############################################################################
# The box
###############################################################################


@dataclass
class Normalization:
    """The [0,1]^2 box of step 3, anchored on the endpoints OWN reported values,
    not on `Result.z0` and tau = 1.

    On baseline and a3 those agree.  On a2 they do not -- Section 3.4 solves
    (M^ac) even at lambda = 0 -- so anchoring on z0 would move the left endpoint
    off the origin and make the chords carry the DC-to-AC model change as a
    response to lambda.  The rule `ropf.results` normalizes by, for the same
    reason: within a stage, against that stage own lambda = 0 row.
    """

    cost_lo: float
    cost_hi: float
    tau_lo: float
    tau_hi: float

    @property
    def cost_span(self) -> float:
        return self.cost_hi - self.cost_lo

    @property
    def tau_span(self) -> float:
        return self.tau_lo - self.tau_hi

    @property
    def cost_is_flat(self) -> bool:
        return self.cost_span <= SPAN_RTOL * max(1.0, abs(self.cost_lo))

    @property
    def tau_is_flat(self) -> bool:
        return self.tau_span <= SPAN_RTOL * max(1.0, abs(self.tau_lo))

    def cost_of(self, cost: float) -> float:
        """(cost - cost_lo)/(cost_hi - cost_lo); 0 on an axis that did not move."""
        return 0.0 if self.cost_is_flat else (cost - self.cost_lo) / self.cost_span

    def tau_of(self, tau: float) -> float:
        """(tau_lo - tau)/(tau_lo - tau_hi); 0 on an axis that did not move."""
        return 0.0 if self.tau_is_flat else (self.tau_lo - tau) / self.tau_span

    def as_dict(self) -> Dict[str, Any]:
        return {"cost_lo": self.cost_lo, "cost_hi": self.cost_hi,
                "tau_lo": self.tau_lo, "tau_hi": self.tau_hi,
                "cost_is_flat": self.cost_is_flat,
                "tau_is_flat": self.tau_is_flat}


def _cheapest_at_lowest_tau(points: Sequence[FrontierPoint],
                            tolerance: float) -> FrontierPoint:
    """The smallest multiplier reaching within `tolerance` of the lowest tau.

    "Lowest risk" is a plateau, not a point, and the cheapest weight on it is
    the one worth naming.  `tolerance` is the walk own saturation threshold, so
    this rule and the walk stopping rule are one statement.
    """
    lowest = min(p.tau for p in points)
    return min((p for p in points if p.tau <= lowest + tolerance),
               key=lambda p: p.multiplier)


def _chord(norm: Normalization, a: FrontierPoint, b: FrontierPoint) -> float:
    """The normalized distance between two adjacent points.

    A knee metric -- split the pair whose midpoint deviates furthest from the
    chord -- places points better but needs the midpoint SOLVED to measure the
    deviation, so it buys that placement with more solves.  Chord length needs
    nothing but the pair.
    """
    return math.hypot(norm.cost_of(b.cost) - norm.cost_of(a.cost),
                      norm.tau_of(b.tau) - norm.tau_of(a.tau))


###############################################################################
# The trace
###############################################################################


@dataclass
class FrontierTrace:
    """What one trace returns: the points, the box, and the recommendation."""

    case: str
    metric: str
    stage: str

    #: Every solved multiplier, in m order.  Unusable points are kept; see
    #: `FrontierPoint.usable`.
    points: List[FrontierPoint] = field(default_factory=list)
    #: The box of step 3.  None when the trace stopped before both endpoints
    #: were in hand.
    normalization: Optional[Normalization] = None

    #: How the step-2 walk ended: one of `WALK_OUTCOMES`, or "" if it never ran.
    walk_outcome: str = ""
    termination: str = "chord_tolerance"
    termination_detail: str = ""

    #: Decimals the recommended grid is written to, chosen once from the span
    #: between the endpoints.  See `_grid_decimals`.
    grid_decimals: int = GRID_DECIMALS_MIN
    #: The finest interval in m refinement would split: 10^-grid_decimals.
    grid_resolution: float = 10.0 ** -GRID_DECIMALS_MIN

    #: The longest adjacent chord left unrefined.  0.0 when there is no pair.
    longest_chord: float = 0.0
    #: The number of times `algorithm.run` was actually called.  Equal to the
    #: point count: the memo is what makes that true.
    n_solves: int = 0
    total_time_s: float = 0.0

    #: The K multipliers to paste into the ladder config's `weight_grid`.
    recommended_weight_grid: Tuple[float, ...] = ()

    # -- the two endpoints ----------------------------------------------------

    @property
    def usable_points(self) -> List[FrontierPoint]:
        return [p for p in self.points if p.usable]

    @property
    def left(self) -> Optional[FrontierPoint]:
        """The nominal point, m = 0.  Always the first solved."""
        return next((p for p in self.points if p.role == "left"), None)

    @property
    def right(self) -> Optional[FrontierPoint]:
        """The point the doubling walk stopped at."""
        return next((p for p in self.points if p.role == "right"), None)

    @property
    def endpoint_multipliers(self) -> Tuple[float, ...]:
        return tuple(p.multiplier for p in (self.left, self.right)
                     if p is not None)

    def weight_grid_line(self) -> str:
        """The recommendation as the config line it is meant to become."""
        return "weight_grid = " + ", ".join(
            f"{m:g}" for m in self.recommended_weight_grid)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case,
            "metric": self.metric,
            "stage": self.stage,
            "termination": self.termination,
            "termination_detail": self.termination_detail,
            "walk_outcome": self.walk_outcome,
            "grid_decimals": self.grid_decimals,
            "grid_resolution": self.grid_resolution,
            "longest_chord": self.longest_chord,
            "n_points": len(self.points),
            "n_solves": self.n_solves,
            "total_time_s": self.total_time_s,
            "normalization": (None if self.normalization is None
                              else self.normalization.as_dict()),
            "endpoint_multipliers": list(self.endpoint_multipliers),
            "recommended_weight_grid": list(self.recommended_weight_grid),
            "weight_grid_line": self.weight_grid_line(),
            "points": [p.as_dict() for p in self.points],
        }


###############################################################################
# Solving, once per multiplier
###############################################################################


class _Solves:
    """`algorithm.run`, memoized on the multiplier for one trace.  Bisection
    revisits multipliers and each call is a whole run of Algorithm 1, so the
    memo is what makes the point budget a budget."""

    def __init__(self, network: Network, config: RunConfig,
                 emit: Callable[[str], None]):
        self.network = network
        self.config = config
        self.emit = emit
        self.dc_solver = config.dc_solver()
        self.ac_solver = config.ac_solver()
        self._results: Dict[float, algorithm.Result] = {}
        self.n_solves = 0

    @staticmethod
    def key(multiplier: float) -> float:
        return round(float(multiplier), MULTIPLIER_DECIMALS)

    def solved(self, multiplier: float) -> bool:
        return self.key(multiplier) in self._results

    def __call__(self, multiplier: float) -> Tuple[float, algorithm.Result]:
        """Run at `multiplier`, or return the last run at it.  Returns the
        ROUNDED multiplier: that is the weight actually solved."""
        key = self.key(multiplier)
        if key in self._results:
            return key, self._results[key]
        self.n_solves += 1
        self.emit(f"\n tracer: solve {self.n_solves}, "
                  f"lambda = {key:g} lambda*\n")
        result = algorithm.run(self.network, self.config.algorithm_config(key),
                               self.dc_solver, self.ac_solver, self.emit)
        self._results[key] = result
        return key, result


###############################################################################
# trace
###############################################################################


def trace(config: RunConfig,
          log: Optional[Callable[[str], None]] = None,
          network: Optional[Network] = None) -> FrontierTrace:
    """Trace the frontier of one case, metric and stage.  Campaign-free.
    `network` is optional, for a caller that has already read the case."""
    emit = log or _noop
    started = time.time()

    if config.risk_weight is not None:
        raise ConfigError(
            "risk_weight pins lambda to an absolute value, so every point of "
            "the trace would be the same run. The tracer sweeps multiples of "
            "lambda*; clear risk_weight to trace a frontier.")
    if config.exchange_rate is not None:
        raise ConfigError(
            f"the (xi, tau_lo, tau_hi) triple pins the exchange rate at "
            f"{config.exchange_rate:g}, which IS the multiplier, so every "
            f"point of the trace would be the same run. The tracer sweeps the "
            f"rate to find where the frontier bends; clear the triple to trace "
            f"a frontier, and read the recommended grid as the rates to use.")
    if not config.case:
        raise ConfigError("no `case` given; there is nothing to trace")

    if network is None:
        network = read_matpower(config.case, emit)

    out = FrontierTrace(case=os.path.basename(network.casefile or config.case),
                        metric=config.metric, stage=config.stage)
    solves = _Solves(network, config, emit)

    # ---- 1. the left endpoint: m = 0 ---------------------------------------
    emit("\n tracer step 1: the nominal point, lambda = 0\n")
    _, nominal = solves(0.0)
    left = _point(nominal, 0.0, "left")
    out.points.append(left)

    if nominal.termination == "nominal_risk_zero":
        # rho0 = 0 makes lambda* and tau undefined: no axis to trade against,
        # so no frontier.  Reported as degenerate, not as a one-point trace.
        return _finish(out, solves, started, "nominal_risk_zero",
                       nominal.termination_detail, emit)
    if not left.usable:
        return _finish(out, solves, started, "infeasible",
                       f"the nominal solve returned {left.status!r}, so the "
                       f"frontier has no left endpoint", emit)

    emit(f" tracer: lambda* = {left.lambda_star:.6f} $/p.u., "
         f"z0 = {left.cost:.6f} $, tau = {left.tau:.6f}\n")

    # ---- 2. the right endpoint: double until tau stops moving --------------
    emit(f"\n tracer step 2: doubling from {config.frontier_seed:g} "
         f"lambda* to the eta floor\n")
    walked, outcome = _walk_right(solves, config, left, emit)
    out.points.extend(walked)
    out.walk_outcome = outcome

    usable_walk = [p for p in walked if p.usable]
    if outcome == "infeasible" or not usable_walk:
        return _finish(out, solves, started, "infeasible",
                       f"the walk stopped at lambda = "
                       f"{walked[-1].multiplier:g} lambda*, which returned "
                       f"{walked[-1].status!r}", emit)

    # THE CHEAPEST WEIGHT THAT REACHED THE LOWEST TAU, not the last point the
    # walk solved: the cap probe sits far past the plateau at the same tau, and
    # taking it would stretch the box over a dead span.
    #
    # TOLERANT, necessarily.  On ACTIVSg200 under bus_flow_sum_agg tau = 0.7204100
    # at lambda = 0.5, 1 and 64 -- one dispatch three times -- but the floats
    # differ in the twelfth digit, so an exact `min` picked 64.
    right = _cheapest_at_lowest_tau(usable_walk,
                                    config.frontier_saturation_tol)
    right.role = "right"
    emit(f" tracer: right endpoint lambda = {right.multiplier:g} lambda*, "
         f"tau = {right.tau:.6f} ({outcome})\n")

    # ---- 3. normalize once, and never again --------------------------------
    out.normalization = Normalization(cost_lo=left.cost, cost_hi=right.cost,
                                      tau_lo=left.tau, tau_hi=right.tau)
    out.grid_decimals = _grid_decimals(right.multiplier - left.multiplier)
    out.grid_resolution = 10.0 ** -out.grid_decimals
    emit(f" tracer step 3: box cost [{left.cost:.6f}, {right.cost:.6f}], "
         f"tau [{right.tau:.6f}, {left.tau:.6f}]; the grid is written to "
         f"{out.grid_decimals} decimals\n")

    if out.normalization.cost_is_flat and out.normalization.tau_is_flat:
        return _finish(out, solves, started, "flat_frontier",
                       f"neither cost nor tau moved between lambda = 0 and "
                       f"lambda = {right.multiplier:g} lambda*; there is no "
                       f"frontier to place points on", emit)

    # ---- 4. refine the longest normalized chord ----------------------------
    emit("\n tracer step 4: splitting the longest normalized chord\n")
    reason, detail, longest = _refine(out.points, solves, out.normalization,
                                      config, out.grid_resolution, emit)
    out.longest_chord = longest
    return _finish(out, solves, started, reason, detail, emit)


def _finish(out: FrontierTrace, solves: _Solves, started: float,
            termination: str, detail: str,
            emit: Callable[[str], None]) -> FrontierTrace:
    """Sort by m, recommend a grid, and record how the trace stopped."""
    if termination not in TRACE_TERMINATIONS:
        raise ValueError(f"{termination!r} is not one of "
                         f"{list(TRACE_TERMINATIONS)}")
    out.points.sort(key=lambda p: p.multiplier)
    out.termination = termination
    out.termination_detail = detail
    out.n_solves = solves.n_solves
    out.total_time_s = time.time() - started
    out.recommended_weight_grid = tuple(
        recommend(out, solves.config.frontier_grid_points,
                  solves.config.frontier_saturation_tol))

    emit(f"\n tracer: {len(out.points)} points in {out.n_solves} solves, "
         f"{out.total_time_s:.1f}s: {termination} ({detail})\n")
    wanted = solves.config.frontier_grid_points
    if not out.recommended_weight_grid:
        emit(f" tracer: no grid to recommend; there is no usable point to "
             f"build one from ({termination})\n")
        return out
    if len(out.recommended_weight_grid) < wanted:
        emit(f" tracer: {len(out.recommended_weight_grid)} multipliers, not "
             f"the {wanted} asked for: the frontier does not offer {wanted} "
             f"distinct risk levels, and a weight that repeats one already in "
             f"the grid costs a full campaign for a dispatch already "
             f"evaluated\n")
    emit(f" tracer: recommended grid -> {out.weight_grid_line()}\n")
    return out


def _walk_right(solves: _Solves, config: RunConfig, left: FrontierPoint,
                emit: Callable[[str], None]) -> Tuple[List[FrontierPoint], str]:
    """Step 2: double m from the seed until tau stops moving, or the cap.

    The drop is measured WITHIN the doubling sequence, between m and 2m, so the
    seed always gets one doubling.  Measured against the nominal point instead,
    a seed too small to do anything would stop the walk -- the opposite of the
    tail this looks for.
    """
    cap = float(config.frontier_hi_cap)
    walked: List[FrontierPoint] = []

    multiplier, result = solves(config.frontier_seed)
    walked.append(_point(result, multiplier, "walk"))
    if not walked[-1].usable:
        return walked, "infeasible"
    emit(f" tracer: lambda = {multiplier:g} lambda*, "
         f"tau = {walked[-1].tau:.6f} (nominal tau {left.tau:.6f})\n")
    if multiplier >= cap:
        return walked, "capped"
    if _spent(walked, config):
        return walked, "budget"

    while True:
        previous = walked[-1]
        multiplier, result = solves(min(previous.multiplier * 2.0, cap))
        point = _point(result, multiplier, "walk")
        walked.append(point)
        if not point.usable:
            return walked, "infeasible"

        drop = previous.tau - point.tau
        emit(f" tracer: lambda = {multiplier:g} lambda*, tau = "
             f"{point.tau:.6f}, drop {drop:.6f} vs saturation_tol "
             f"{config.frontier_saturation_tol:g}\n")
        if drop < config.frontier_saturation_tol:
            return _confirm_saturation(solves, config, walked, emit)
        if multiplier >= cap:
            return walked, "capped"
        if _spent(walked, config):
            return walked, "budget"


def _confirm_saturation(solves: _Solves, config: RunConfig,
                        walked: List[FrontierPoint],
                        emit: Callable[[str], None]
                        ) -> Tuple[List[FrontierPoint], str]:
    """One probe at the cap before a plateau is believed to be the end.

    A PLATEAU IS NOT NECESSARILY THE TAIL: the frontier can be a staircase.  On
    ACTIVSg500 under max_active_flow tau sits at 0.7810, 0.7807, 0.7807 for
    lambda = 0.5, 1, 2 and then drops to 0.7181 at 4, so a walk believing the
    first plateau leaves 6 points of risk reduction outside the traced range.

    One solve settles it.  An unmoved cap means the plateau was the tail; a
    moved one makes the cap the right endpoint, and refinement finds the step.
    """
    plateau = walked[-1]
    cap = float(config.frontier_hi_cap)
    if plateau.multiplier >= cap or _spent(walked, config):
        return walked, "saturated"

    multiplier, result = solves(cap)
    probe = _point(result, multiplier, "walk")
    walked.append(probe)
    if not probe.usable:
        # The plateau stands: the cap says nothing, so nothing is claimed of it.
        return walked, "saturated"

    further = plateau.tau - probe.tau
    if further < config.frontier_saturation_tol:
        emit(f" tracer: the cap at {multiplier:g} lambda* confirms the plateau "
             f"(tau {probe.tau:.6f}); saturated\n")
        return walked, "saturated"

    emit(f" tracer: the plateau at {plateau.multiplier:g} lambda* was false -- "
         f"the cap at {multiplier:g} drops tau a further {further:.6f}, to "
         f"{probe.tau:.6f}. The frontier is a staircase; the cap is the right "
         f"endpoint and refinement will find the step.\n")
    return walked, "capped"


def _spent(walked: List[FrontierPoint], config: RunConfig) -> bool:
    """True when the left endpoint and the walk have used the point budget."""
    return 1 + len(walked) >= config.frontier_max_points


def _refine(points: List[FrontierPoint], solves: _Solves,
            norm: Normalization, config: RunConfig, resolution: float,
            emit: Callable[[str], None]) -> Tuple[str, str, float]:
    """Step 4.  Appends to `points`; returns (termination, detail, longest).

    The tolerance is tested BEFORE the budget: a trace that has both converged
    and spent its budget has converged.
    """
    longest = 0.0
    while True:
        usable = sorted((p for p in points if p.usable),
                        key=lambda p: p.multiplier)
        pairs = sorted(((_chord(norm, a, b), a, b)
                        for a, b in zip(usable, usable[1:])),
                       key=lambda item: -item[0])
        if not pairs:
            return ("no_split_left",
                    "fewer than two usable points, so there is no pair to "
                    "split", longest)

        longest = pairs[0][0]
        if longest < config.frontier_tol:
            return ("chord_tolerance",
                    f"the longest normalized chord is {longest:.6f}, below "
                    f"frontier_tol = {config.frontier_tol:g}", longest)
        if len(points) >= config.frontier_max_points:
            return ("point_budget",
                    f"{len(points)} points solved, the limit is "
                    f"frontier_max_points = {config.frontier_max_points}; the "
                    f"longest chord left unsplit is {longest:.6f}", longest)

        # The longest pair that can still be split.  Unsplittable when its
        # midpoint is already solved (so it failed, and splitting rediscovers
        # the failure) or when the pair is narrower than `resolution`, which is
        # what a step in the frontier looks like from here.
        chosen = None
        stepped = False
        for length, a, b in pairs:
            if length < config.frontier_tol:
                break
            if b.multiplier - a.multiplier <= resolution:
                stepped = True
                continue
            midpoint = 0.5 * (a.multiplier + b.multiplier)
            if not solves.solved(midpoint):
                chosen = (length, a, b, midpoint)
                break
        if chosen is None:
            step = ("; a chord that stays long across a pair that narrow is a "
                    "STEP in the frontier and not a curve -- the dispatch "
                    "changes discontinuously there and there is nothing "
                    "between the two sides to find" if stepped else "")
            return ("no_split_left",
                    f"every pair longer than frontier_tol = "
                    f"{config.frontier_tol:g} is either already split at its "
                    f"midpoint or narrower than {resolution:g} in "
                    f"lambda/lambda*, which is the finest the recommendation "
                    f"can express; the longest is {longest:.6f}{step}",
                    longest)

        length, a, b, midpoint = chosen
        emit(f" tracer: chord {length:.6f} between lambda = {a.multiplier:g} "
             f"and {b.multiplier:g} lambda*; splitting at {midpoint:g}\n")
        solved_at, result = solves(midpoint)
        points.append(_point(result, solved_at, "interior"))


###############################################################################
# The recommendation
###############################################################################


def recommend(out: FrontierTrace, k: int,
              saturation_tol: float = 0.0) -> List[float]:
    """`k` multipliers spread evenly in TAU, endpoints included.

    Not in m and not in cost: tau is what the campaign compares dispatches on,
    and a grid even in m spends most of its points on the flat tail.  The
    endpoints go in first and unconditionally -- the nominal dispatch is the
    base of every paired comparison, and the walk endpoint is the most de-risked
    dispatch found.

    FEWER THAN K IS THE FINDING, not a shortfall: a stepped frontier offers two
    distinct dispatches however many weights are solved on it, and padding back
    to K would put more weights in the campaign evaluating a dispatch already
    in it.
    """
    left = out.left
    if left is None or out.right is None or not out.usable_points:
        return _snap((p.multiplier for p in out.usable_points),
                     out.grid_decimals)

    # THE LAST GRID POINT IS RE-CHOSEN HERE, after refinement.  The walk picks
    # from the doubling sequence alone, which on a staircase forces it to the
    # cap: on ACTIVSg500 the walk sees tau 0.7807 at lambda = 1 and 0.7178 at 64,
    # but refinement finds 3.46 reaching 0.7181 for $88.7k against $89.2k.
    right = _cheapest_at_lowest_tau(out.usable_points, saturation_tol)

    # ONLY THE POINTS INSIDE THE TRACED RANGE: `_confirm_saturation` can leave a
    # probe at the cap, past the right endpoint at the same tau.  That probe is
    # evidence the plateau is real, not a point on the frontier.
    usable = [p for p in out.usable_points
              if left.multiplier <= p.multiplier <= right.multiplier]
    if k < 2 or len(usable) <= k:
        return _snap((p.multiplier for p in usable), out.grid_decimals)
    chosen: Dict[float, FrontierPoint] = {left.multiplier: left,
                                          right.multiplier: right}
    pool = [p for p in usable if p is not left and p is not right]

    # Two points this close in tau are one risk level.
    spacing = abs(left.tau - right.tau) / (k - 1)
    apart = TAU_DISTINCT_FRACTION * spacing

    # The k - 2 interior targets, strictly between the endpoints' tau.
    for j in range(1, k - 1):
        if not pool:
            break
        target = left.tau + (right.tau - left.tau) * j / (k - 1)
        # Ties on tau go to the smaller multiplier: on the flat tail the
        # cheapest weight reaching a tau is the one worth running.
        best = min(pool, key=lambda p: (abs(p.tau - target), p.multiplier))
        pool.remove(best)
        if any(abs(best.tau - taken.tau) <= apart
               for taken in chosen.values()):
            continue
        chosen[best.multiplier] = best

    return _snap(sorted(chosen), out.grid_decimals)


def _snap(multipliers: Iterable[float], decimals: int) -> List[float]:
    """Round to `decimals`, in order, dropping what collides."""
    snapped: List[float] = []
    for multiplier in multipliers:
        value = round(float(multiplier), decimals)
        if value not in snapped:
            snapped.append(value)
    return snapped


###############################################################################
# Writing
###############################################################################


def write(outdir: str, config: RunConfig, out: FrontierTrace,
          network: Optional[Network] = None,
          log: Optional[Callable[[str], None]] = None) -> str:
    """Write `frontier_trace.json` into `outdir` and return its path.  One file,
    and not a result: a measurement taken to choose a parameter."""
    emit = log or _noop
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, TRACE_JSON)
    payload = {
        "provenance": results_module.provenance(network),
        "config": config.as_dict(),
        "trace": out.as_dict(),
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    emit(f" wrote frontier_trace: {path} ({os.path.getsize(path)} bytes)\n")
    return path

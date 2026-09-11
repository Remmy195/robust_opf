"""Model (D): can the surviving network serve its demand, and at what loss.

Section 4.2.  Given a pre-event dispatch P* and a disfigurement:

    excluded   the surviving network is not connected -- a different operating
               problem from the one (D) represents.  Decided before any solve.
    collapse   the frequency screen rejected it.  The screen runs BEFORE (D).
    survival   (D) answered, or the survivors cannot be curtailed to their own
               demand inside the response window and (D) was never run.  Lost
               load and the rating violation are SEPARATE numbers, both always
               reported.

THE RATING IS SOFT.  Bounding `Pf` by ``beta * U`` is not a measurement at
gamma = 0: the window pins Pg at P*, so a branch outage DETERMINES the flows and
bounding a determined quantity removes the solution instead of scoring it.  The
model carries GO3 slack ``s_jtk^+``, eq (157)-(160), so every admitted draw
returns `overload_max_pu`, `overload_sum_pu` and `worst_loading`.

TWO PATHS, ONE ANSWER.  A branch-only event at gamma = 0 leaves (D) no dispatch
decision, so `frozen` gets the same flows from one sparse linear solve.  Both
paths fill the same fields; `n_solves` is the only way to tell which ran.

ONE AMPL INSTANCE FOR THE WHOLE CAMPAIGN.  The network is loaded once and only
parameters are written thereafter: the disfigurement through the ``alive_*``
vectors, the phases through ``shed_allowed`` and the objective.  The lifecycle
rule of `ropf.model` applies unchanged.

THE RESPONSE WINDOW IS COMPUTED HERE, NOT IN AMPL, because P* can sit a hair
outside ``[Pmin, Pmax]`` after a solve and the guard for that needs to be
somewhere a test can reach it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, FrozenSet, Iterable, Optional, Set,
                    Tuple)

from amplpy import AMPL

from . import frozen
from ..model import MODFILE_DIR, SolverConfig
from ..network import Network

POSTEVENT_MODFILE = "postevent.mod"

#: RELATIVE to system demand, not absolute.  Both test a sum over the whole
#: fleet, so an absolute tolerance at Gurobi FeasibilityTol is exactly the slack
#: the solver may leave on one row of that sum.  On ACTIVSg2000 the master
#: balance residual runs 6.6e-10 to 3.9e-6 p.u. and straddled the old constant:
#: three of eighteen weights lost their whole branch-class screen to solver
#: noise.  See analysis/FINDING_n_overload_zero.md.
SHED_RTOL = 1e-6
BALANCE_RTOL = 1e-6

OUTCOMES = ("excluded", "collapse", "survival")


def tolerance(rtol: float, scale: float) -> float:
    """``rtol`` on a fleet of size ``scale``, floored at ``rtol`` itself."""
    return rtol * max(1.0, abs(scale))


def shed_tolerance(network: Network) -> float:
    """The one shed tolerance, p.u., for a network.  Section 3.6.  Derived here
    so a campaign CSV can be read with the number the evaluation used."""
    return tolerance(SHED_RTOL, network.load_pu)


def _noop(_message: str) -> None:
    return None


###############################################################################
# Inputs
###############################################################################


@dataclass(frozen=True)
class PostEventParams:
    """The declared parameters of (D).

    gamma
        Response window scale, p.u.; a window has half-width ``pi_g * gamma``.
        No case file here carries ramp-rate data, so gamma is a declared study
        axis that is swept, never assumed at one value and reported as data.
    beta
        The emergency rating factor of eq (6e).  Defaults to 1, the NORMAL
        rating: rateB and rateC are zero across every ACTIVSg distribution, but
        all six .aux files carry the same LimitSet record -- rate set "A" for
        the base case and again for the CONTINGENCY case at LSLinePercent 100 --
        so the published rule is rating A at 100 percent.  1.2 stays in the
        sweep as a short-term-overload sensitivity and is labelled an
        assumption where it is used.  Re-reading a campaign at another factor
        is free: `worst_loading` does not mention beta.
    """

    gamma: float
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.gamma < 0:
            raise ValueError(f"gamma is a window half-width scale and cannot be "
                             f"negative, got {self.gamma}")
        if self.beta < 1:
            raise ValueError(f"beta scales the normal rating upward and cannot "
                             f"be below 1, got {self.beta}")


@dataclass(frozen=True)
class Disfigurement:
    """What a disfigurement removes, as network counts.  A removed bus deletes
    every incident line and every unit at it; `survivors` applies that closure,
    so a caller may pass buses alone."""

    buses: FrozenSet[int] = frozenset()
    branches: FrozenSet[int] = frozenset()
    gens: FrozenSet[int] = frozenset()
    label: str = ""

    @property
    def size(self) -> int:
        """K.  Section 4.1.2 counts buses and lines; a unit removed with its bus
        is not a component in its own right."""
        return len(self.buses) + len(self.branches)


@dataclass
class PostEventResult:
    """One evaluation of one disfigurement against one dispatch."""

    outcome: str
    reason: str = ""
    #: L, p.u.  None when (D) was neither solved nor short-circuited; 0.0 when
    #: an uncurtailable surplus made the answer known without a solve (see
    #: `PostEvent.evaluate`).
    lost_load: Optional[float] = None
    #: z-hat, the post-event generation cost, $.
    cost: Optional[float] = None
    #: Which phase produced the answer: 'cost', 'overload', 'loadshed', or
    #: 'curtailed' for the uncurtailable-surplus short-circuit, which reaches no
    #: phase at all.  Named so a campaign can count how many rows never reached
    #: the flow model.
    phase: Optional[str] = None
    shed_by_bus: Dict[int, float] = field(default_factory=dict)
    Pg: Dict[int, float] = field(default_factory=dict)

    # --- eq (6e): what the ratings had to give -------------------------------
    #: max_e |Pf_e| / U_e over surviving RATED branches.  The parameter-free
    #: severity number: it does not mention beta, so one campaign can be re-read
    #: at any emergency rating factor without re-solving.
    worst_loading: Optional[float] = None
    #: The branch attaining `worst_loading`.  One at-rating branch can decide a
    #: whole instance: ACTIVSg500 branch 144 does.
    worst_branch: Optional[int] = None
    #: max_e s_e, p.u.  GO3's largest s_jtk^+, eq (157)-(160).
    overload_max_pu: Optional[float] = None
    #: sum_e s_e, p.u.  GO3's summed violation penalty.
    overload_sum_pu: Optional[float] = None
    #: The absolute shed tolerance applying to `lost_load`, p.u.  Section 3.6.
    shed_tol: float = SHED_RTOL

    n_buses: int = 0
    n_branches: int = 0
    n_gens: int = 0
    #: Demand the removed buses took with them, p.u.  Not lost load (eq (6f)
    #: sums over survivors), but L is not comparable across events without it.
    removed_demand_pu: float = 0.0
    solve_time_s: float = 0.0
    n_solves: int = 0

    @property
    def survived(self) -> bool:
        return self.outcome == "survival"

    @property
    def served_all_demand(self) -> bool:
        return (self.outcome == "survival"
                and (self.lost_load or 0.0) <= self.shed_tol)

    @property
    def overloaded(self) -> bool:
        """A rating violation the post-event system could not dispatch away.  At
        gamma = 0 on the branch class this IS the binding set."""
        return (self.outcome == "survival"
                and (self.overload_max_pu or 0.0) > 0.0)


###############################################################################
# The surviving network
###############################################################################


def survivors(network: Network,
              disfigurement: Disfigurement) -> Tuple[Set[int], Set[int], Set[int]]:
    """Apply the removal closure.  Returns the SURVIVING (buses, branches, gens).

    A unit already out of service is never alive, so it cannot be removed twice
    nor counted as a survivor contributing capacity.
    """
    dead_buses = {int(b) for b in disfigurement.buses}
    dead_branches = {int(e) for e in disfigurement.branches}
    dead_gens = {int(g) for g in disfigurement.gens}

    for count, branch in network.branches.items():
        if branch.id_f in dead_buses or branch.id_t in dead_buses:
            dead_branches.add(int(count))

    for count, gen in network.gens.items():
        if not gen.status:
            dead_gens.add(int(count))

    for bus_count in dead_buses:
        bus = network.buses.get(bus_count)
        if bus is not None:
            dead_gens.update(int(g) for g in bus.genidsbycount)

    return (set(network.buses) - dead_buses,
            set(network.branches) - dead_branches,
            set(network.gens) - dead_gens)


def is_connected(network: Network,
                 live_buses: Set[int],
                 live_branches: Set[int]) -> bool:
    """True when the surviving buses form one connected component.  Section 4.2
    excludes a split network, which is why (D) needs no per-island reference bus
    or power balance."""
    if not live_buses:
        return False

    adjacency: Dict[int, list] = {bus: [] for bus in live_buses}
    for count in live_branches:
        branch = network.branches[count]
        f, t = int(branch.id_f), int(branch.id_t)
        if f in adjacency and t in adjacency:
            adjacency[f].append(t)
            adjacency[t].append(f)

    start = next(iter(live_buses))
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen) == len(live_buses)


def response_window(network: Network,
                    P_star: Dict[int, float],
                    pi: Dict[int, float],
                    gamma: float) -> Tuple[Dict[int, float], Dict[int, float]]:
    """eq (6d), intersected with the operating range.  Returns (lo, hi).

    Symmetric and deliberately no wider: a wider window would let (D) use the
    same slack to resolve a genuine emergency-rating violation.

    The guard matters.  P* comes from a solved master and can sit outside
    ``[Pmin, Pmax]`` by the solver tolerance, making the naive intersection
    empty and (D) infeasible for a reason unrelated to the disfigurement; there
    the window collapses to the nearest feasible output.
    """
    lo: Dict[int, float] = {}
    hi: Dict[int, float] = {}
    for count, gen in network.gens.items():
        star = float(P_star.get(count, 0.0))
        half = abs(float(pi.get(count, 0.0))) * float(gamma)
        low = max(float(gen.Pmin), star - half)
        high = min(float(gen.Pmax), star + half)
        if low > high:
            low = high = min(max(star, float(gen.Pmin)), float(gen.Pmax))
        lo[count], hi[count] = low, high
    return lo, hi


def capacity_participation(network: Network) -> Dict[int, float]:
    """pi_g proportional to capacity, over the in-service fleet.  The surrogate
    used when no AGC factor is available; `dynamics` reads the real ones where
    the distribution ships them."""
    total = sum(gen.Pmax for gen in network.gens.values() if gen.status)
    if total <= 0:
        return {count: 0.0 for count in network.gens}
    return {count: (gen.Pmax / total if gen.status else 0.0)
            for count, gen in network.gens.items()}


###############################################################################
# The model
###############################################################################


class PostEvent:
    """(D), loaded once with one network and evaluated many times."""

    def __init__(self,
                 network: Network,
                 solver: Optional[SolverConfig] = None,
                 log: Optional[Callable[[str], None]] = None,
                 modfile: str = POSTEVENT_MODFILE):
        self.network = network
        self.log = log or _noop
        self.solver = solver or SolverConfig(name="gurobi", verbose=False)

        path = modfile if os.path.isabs(modfile) else os.path.join(MODFILE_DIR,
                                                                   modfile)
        if not os.path.exists(path):
            raise FileNotFoundError(f"model file not found: {path}")

        self.log(f" loading {os.path.basename(path)}\n")
        self.ampl = AMPL()
        self.ampl.read(path)
        self.ampl.eval("option display_precision 0;")
        self.ampl.setOption("presolve", 0)
        self.solver.apply(self.ampl, self.log)
        self._load_network()

        # What the last evaluation left alive, so only the difference is
        # written next time.  Rewriting all 88,207 entries per solve is most of
        # the cost at the top instance.
        self._alive_bus: Set[int] = set(network.buses)
        self._alive_br: Set[int] = set(network.branches)
        self._alive_gen: Set[int] = {c for c, g in network.gens.items() if g.status}
        self._write_alive_gen(self._alive_gen)

        self._shed_tol = shed_tolerance(network)

        # The fleet as the case ships it.  `_alive_gen` moves with every solve;
        # this does not, and `evaluate` needs the fixed one to ask what an event
        # actually took out -- a vendor list names already-offline units.
        self._in_service: Set[int] = {c for c, g in network.gens.items()
                                      if g.status}

    # -- construction ------------------------------------------------------

    def _load_network(self) -> None:
        net = self.network
        ampl = self.ampl
        started = time.time()

        ampl.getSet("buses").setValues(list(net.buses))
        ampl.getSet("gens").setValues(list(net.gens))
        ampl.getSet("branches").setValues(list(net.branches))

        Pd = {c: bus.Pd for c, bus in net.buses.items()}
        Gs = {c: bus.Gs for c, bus in net.buses.items()}
        ampl.get_parameter("Pd").setValues(Pd)
        ampl.get_parameter("Gs").setValues(Gs)

        bus_f, bus_t, bdc, Pfinj, U = {}, {}, {}, {}, {}
        for count, branch in net.branches.items():
            bus_f[count], bus_t[count] = branch.id_f, branch.id_t
            bdc[count], Pfinj[count] = branch.bdc, branch.Pfinj
            U[count] = branch.limit
        for name, values in (("bus_f", bus_f), ("bus_t", bus_t), ("bdc", bdc),
                             ("Pfinj", Pfinj), ("U", U)):
            ampl.get_parameter(name).setValues(values)

        c2, c1, c0 = {}, {}, {}
        for count, gen in net.gens.items():
            c2[count], c1[count], c0[count] = gen.costvector
        for name, values in (("c2", c2), ("c1", c1), ("c0", c0)):
            ampl.get_parameter(name).setValues(values)

        branches_f = ampl.getSet("branches_f")
        branches_t = ampl.getSet("branches_t")
        bus_gens = ampl.getSet("bus_gens")
        for count, bus in net.buses.items():
            branches_f[count].setValues(list(bus.frombranchids.values()))
            branches_t[count].setValues(list(bus.tobranchids.values()))
            bus_gens[count].setValues(list(bus.genidsbycount))

        # Required before the first solve; the evaluation overwrites it.
        ampl.get_parameter("Pg_lo").setValues({c: 0.0 for c in net.gens})
        ampl.get_parameter("Pg_hi").setValues({c: 0.0 for c in net.gens})
        ampl.get_parameter("ref_bus").set(int(net.refbus or min(net.buses)))

        self.log(f" (D) loaded {net.numbuses} buses, {net.numbranches} branches,"
                 f" {net.numgens} gens in {time.time() - started:.1f}s\n")

    # -- parameter writes --------------------------------------------------

    def _write_alive(self, name: str, previous: Set[int], now: Set[int]) -> None:
        """Write only what changed between two evaluations."""
        changed = {}
        for count in previous - now:
            changed[count] = 0
        for count in now - previous:
            changed[count] = 1
        if changed:
            self.ampl.get_parameter(name).setValues(changed)

    def _write_alive_gen(self, live: Set[int]) -> None:
        self.ampl.get_parameter("alive_gen").setValues(
            {c: (1 if c in live else 0) for c in self.network.gens})

    # -- evaluation --------------------------------------------------------

    def evaluate(self,
                 P_star: Dict[int, float],
                 pi: Dict[int, float],
                 params: PostEventParams,
                 disfigurement: Disfigurement,
                 screen: Optional[Callable[[Set[int]], Tuple[bool, str]]] = None
                 ) -> PostEventResult:
        """One disfigurement against one dispatch.  See the module docstring.

        `screen` takes the surviving generator counts and returns
        ``(cleared, reason)``, and runs after the connectivity exclusion and
        before (D) -- the order Section 4.2 states.
        """
        live_buses, live_branches, live_gens = survivors(self.network,
                                                         disfigurement)
        removed_demand = sum(self.network.buses[c].Pd
                             for c in set(self.network.buses) - live_buses)
        counts = dict(n_buses=len(live_buses), n_branches=len(live_branches),
                      n_gens=len(live_gens), removed_demand_pu=removed_demand)

        # ---- exclusion, before any solve -----------------------------------
        if not is_connected(self.network, live_buses, live_branches):
            return PostEventResult(outcome="excluded",
                                   reason="the surviving network is disconnected",
                                   shed_tol=self._shed_tol, **counts)

        # ---- the frequency screen, before (D) ------------------------------
        if screen is not None:
            cleared, why = screen(live_gens)
            if not cleared:
                return PostEventResult(outcome="collapse", reason=why,
                                       shed_tol=self._shed_tol, **counts)

        # ---- WHAT DID THE EVENT ACTUALLY REMOVE? ---------------------------
        # Decides both the guard below and the routing rule after it, so it is
        # asked once.  `determined` means the event takes nothing off either
        # side of the balance -- no bus, no injection -- so at gamma = 0 the
        # flows follow from P* and the surviving topology alone.
        #
        # NAMING A UNIT IS NOT REMOVING ONE: the test is on the injection taken
        # out, not the set named.  A vendor contingency naming an offline unit,
        # or one at Pmin = Pmax = 0, is the undisturbed network under another
        # name; deciding those by set membership sends 114 rows per weight to an
        # LP pinned by the balance residual.
        live_demand = sum(self.network.buses[b].Pd for b in live_buses)
        window = response_window(self.network, P_star, pi, params.gamma)
        lost_supply = sum(window[0][g] for g in self._in_service - live_gens)
        determined = (not disfigurement.buses
                      and lost_supply <= self._shed_tol)

        # ---- an UNCURTAILABLE surplus is never put to (D) -------------------
        # If every surviving unit sits at the bottom of its eq (6d) window and
        # the survivors still overproduce, the cost phase is infeasible for a
        # reason belonging to gamma, not to the disfigurement.  The operator
        # curtails the excess; widening the window instead would let (D) paper
        # over a genuine rating violation elsewhere.
        #
        # AN EVENT THAT REMOVES NOTHING HAS NO SURPLUS TO FIND, so the guard is
        # skipped there.  Otherwise the test reduces to the balance residual, a
        # scalar not mentioning the contingency, and goes all-or-nothing over
        # the whole branch class.
        if not determined:
            floor = sum(window[0][g] for g in live_gens)
            if floor > live_demand + tolerance(BALANCE_RTOL,
                                               self.network.load_pu):
                return PostEventResult(
                    outcome="survival", lost_load=0.0, phase="curtailed",
                    shed_tol=self._shed_tol,
                    reason="the survivors cannot be curtailed to their own "
                           "demand inside the response window; the excess is "
                           "curtailed by the operator and not evaluated by (D)",
                    **counts)

        # ---- the routing rule ----------------------------------------------
        # removes no injection, gamma = 0   closed form, no AMPL
        # removes no injection, gamma > 0   LP
        # removes a bus or an injection     LP
        #
        # The first row is not a screen and not a bound: no dispatch decision is
        # left for the LP to make, so the closed form is exact and the LP is a
        # slower route to the same numbers.  Both fill the same fields.
        if determined and params.gamma == 0.0:
            return self._frozen(params, live_buses, live_branches, live_gens,
                                window, counts)
        return self._solve(params, live_buses, live_branches, live_gens,
                           window, counts, shed_can_move=True)

    # -- the two paths -----------------------------------------------------

    def _frozen(self, params: PostEventParams,
                live_buses: Set[int], live_branches: Set[int],
                live_gens: Set[int],
                window: Tuple[Dict[int, float], Dict[int, float]],
                counts: Dict[str, Any]) -> PostEventResult:
        """(D) in closed form.  See `ropf.counterfactual.frozen`."""
        started = time.time()
        buses = sorted(live_buses)
        index = {bus: row for row, bus in enumerate(buses)}
        pinned = {g: window[0][g] for g in live_gens}

        injection = {}
        for count in buses:
            bus = self.network.buses[count]
            injection[count] = (sum(pinned.get(int(g), 0.0)
                                    for g in bus.genidsbycount)
                                - bus.Pd - bus.Gs)

        pf = frozen.flows(self.network, buses, index, live_branches, injection,
                          ref=self._reference(live_buses))
        load = frozen.summarize(self.network, pf, params.beta)

        # The same two outcomes the LP path reaches, and the same words for
        # them: nothing downstream may be able to tell which path ran.
        if load.overload_max_pu > 0.0:
            phase, reason = "overload", (
                "no operating point respects the emergency ratings, and at "
                "gamma = 0 nothing may move to relieve them")
        else:
            phase, reason = "cost", "all demand served"

        return PostEventResult(
            outcome="survival", phase=phase, reason=reason,
            lost_load=0.0, cost=self._generation_cost(pinned), Pg=pinned,
            worst_loading=load.worst_loading, worst_branch=load.worst_branch,
            overload_max_pu=load.overload_max_pu,
            overload_sum_pu=load.overload_sum_pu, shed_tol=self._shed_tol,
            solve_time_s=time.time() - started, n_solves=0, **counts)

    def _solve(self, params: PostEventParams,
               live_buses: Set[int], live_branches: Set[int],
               live_gens: Set[int],
               window: Tuple[Dict[int, float], Dict[int, float]],
               counts: Dict[str, Any],
               shed_can_move: bool) -> PostEventResult:
        """(D) as the LP, in the lexicographic order `postevent.mod` declares.

        THE CHEAP PHASE IS TRIED FIRST.  A draw with a zero-overload, zero-shed
        operating point has overload optimum 0 by inspection and the cost phase
        IS that optimum, so trying it first leaves the common case at one solve
        instead of two.  The answer is the same either way.

        `shed_can_move` is False exactly on an event removing no injection at
        gamma = 0, where sum(L) is pinned at the balance residual and the
        load-shed phase feasible set IS the cost phase one.  That case normally
        takes the closed-form path and reaches this method only when a caller
        forces the LP, which the path-agreement test does.
        """
        self._apply(live_buses, live_branches, live_gens, params, window)
        started = time.time()
        solves = 0

        # ---- phase `cost`: the cheapest dispatch that sheds nothing and
        #      violates no rating.  eq (6e) as a hard bound is s_cap = 0.
        self.ampl.get_parameter("s_capped").set(1)
        self.ampl.get_parameter("s_cap").set(0.0)
        self.ampl.get_parameter("shed_allowed").set(0)
        self.ampl.eval("objective gen_cost;")
        self.ampl.solve()
        solves += 1

        if str(self.ampl.get_value("solve_result")) == "solved":
            load = self._loading(live_branches, params.beta)
            return PostEventResult(
                outcome="survival", reason="all demand served",
                lost_load=0.0,
                cost=float(self.ampl.get_objective("gen_cost").value()),
                phase="cost", Pg=self._live_values("Pg", live_gens),
                worst_loading=load.worst_loading,
                worst_branch=load.worst_branch,
                overload_max_pu=load.overload_max_pu,
                overload_sum_pu=load.overload_sum_pu, shed_tol=self._shed_tol,
                solve_time_s=time.time() - started, n_solves=solves, **counts)

        # ---- phase `overload`: the smallest rating violation admitted.
        #      Shedding is allowed exactly when the event allows it at all: the
        #      cost phase can fail for a balance reason as well as a rating one,
        #      and only shedding answers the first.  Ratings before load is the
        #      ordering the hard-rating model already had.
        self.ampl.get_parameter("s_capped").set(0)
        self.ampl.get_parameter("shed_allowed").set(1 if shed_can_move else 0)
        self.ampl.eval("objective overload;")
        self.ampl.solve()
        solves += 1
        status = str(self.ampl.get_value("solve_result"))

        if status != "solved":
            # Not a rating failure: `s` absorbs every rating.  The survivors
            # cannot balance inside the response window even with every bus
            # shed, which the guard above is meant to have caught.
            return PostEventResult(
                outcome="survival", phase="overload", shed_tol=self._shed_tol,
                reason=f"the overload phase returned {status!r}; the ratings "
                       f"cannot cause that, so the survivors do not balance "
                       f"inside the response window at any lost load",
                solve_time_s=time.time() - started, n_solves=solves, **counts)

        s_star = float(self.ampl.get_objective("overload").value())

        if not shed_can_move:
            load = self._loading(live_branches, params.beta)
            Pg = self._live_values("Pg", live_gens)
            return PostEventResult(
                outcome="survival", phase="overload", Pg=Pg,
                reason="no operating point respects the emergency ratings, "
                       "and at gamma = 0 nothing may move to relieve them",
                lost_load=0.0, cost=self._generation_cost(Pg),
                worst_loading=load.worst_loading,
                worst_branch=load.worst_branch,
                overload_max_pu=load.overload_max_pu,
                overload_sum_pu=load.overload_sum_pu, shed_tol=self._shed_tol,
                solve_time_s=time.time() - started, n_solves=solves, **counts)

        # ---- phase `loadshed`: the smallest lost load at that overload.
        #      Held at s_star, not zero, so this phase never has to buy back an
        #      overload the network could not avoid.  Feasible whenever the
        #      overload phase was, so a failure here IS a modelling error.
        self.ampl.get_parameter("s_capped").set(1)
        self.ampl.get_parameter("s_cap").set(
            s_star + tolerance(BALANCE_RTOL, self.network.load_pu))
        self.ampl.get_parameter("shed_allowed").set(1)
        self.ampl.eval("objective lost_load;")
        self.ampl.solve()
        solves += 1
        status = str(self.ampl.get_value("solve_result"))
        elapsed = time.time() - started

        if status != "solved":
            return PostEventResult(
                outcome="survival", phase="loadshed", shed_tol=self._shed_tol,
                reason=f"the load-shed phase returned {status!r}, which cannot "
                       f"happen once the overload phase has solved: its "
                       f"solution sheds nothing and is still feasible here",
                solve_time_s=elapsed, n_solves=solves, **counts)

        shed = {bus: value
                for bus, value in self._live_values("L", live_buses).items()
                if value > self._shed_tol}
        Pg = self._live_values("Pg", live_gens)
        load = self._loading(live_branches, params.beta)
        return PostEventResult(
            outcome="survival", phase="loadshed",
            reason="no zero-shed operating point exists",
            lost_load=float(self.ampl.get_objective("lost_load").value()),
            cost=self._generation_cost(Pg), shed_by_bus=shed, Pg=Pg,
            worst_loading=load.worst_loading, worst_branch=load.worst_branch,
            overload_max_pu=load.overload_max_pu,
            overload_sum_pu=load.overload_sum_pu, shed_tol=self._shed_tol,
            solve_time_s=elapsed, n_solves=solves, **counts)

    def _loading(self, live_branches: Set[int], beta: float) -> frozen.Loading:
        """The severity numbers of the solve that just finished.  Read off `Pf`,
        not `s`: outside the overload phase `s` is held only by the budget row
        and may sit above its minimum, and the closed form has no `s`."""
        return frozen.summarize(self.network,
                                self._live_values("Pf", live_branches), beta)

    def _apply(self, live_buses, live_branches, live_gens,
               params: PostEventParams,
               window: Tuple[Dict[int, float], Dict[int, float]]) -> None:
        """Write the disfigurement and the window into the loaded instance.  The
        window is passed in, not recomputed: `evaluate` needs its floor to
        decide whether (D) is asked at all."""
        self._write_alive("alive_bus", self._alive_bus, live_buses)
        self._write_alive("alive_br", self._alive_br, live_branches)
        self._write_alive("alive_gen", self._alive_gen, live_gens)
        self._alive_bus, self._alive_br, self._alive_gen = \
            set(live_buses), set(live_branches), set(live_gens)

        lo, hi = window
        self.ampl.get_parameter("Pg_lo").setValues(lo)
        self.ampl.get_parameter("Pg_hi").setValues(hi)
        self.ampl.get_parameter("beta").set(float(params.beta))
        self.ampl.get_parameter("ref_bus").set(self._reference(live_buses))

    def _reference(self, live_buses: Set[int]) -> int:
        """eq (6g).  The pre-event reference bus where it survives, else any; the
        network is connected by here, so the choice only fixes the gauge."""
        if self.network.refbus in live_buses:
            return int(self.network.refbus)
        for count in sorted(live_buses):
            if self.network.buses[count].nodetype == 3:
                return count
        return min(live_buses)

    def _live_values(self, name: str, live: Iterable[int]) -> Dict[int, float]:
        raw = self.ampl.get_variable(name).get_values().to_dict()
        live = set(live)
        return {int(k): float(v) for k, v in raw.items() if int(k) in live}

    def _generation_cost(self, Pg: Dict[int, float]) -> float:
        """z-hat at a dispatch.  Read from the objective in phase 1; recomputed
        in phase 2, where the active objective is the lost load."""
        total = 0.0
        for count, value in Pg.items():
            quad, lin, const = self.network.gens[count].costvector
            total += const + lin * value + quad * value * value
        return total

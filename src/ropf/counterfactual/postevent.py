"""Model (D): can the surviving network serve its demand, and at what loss.

Section 4.2.  Given a pre-event dispatch P* and a disfigurement, the evaluation
has three outcomes and reaches the optimization only for the third:

    excluded   the surviving network is not connected.  A split system is a
               different operating problem from the one (D) represents, so the
               sample drops it.  Decided here, before any solve.
    collapse   the frequency screen rejected it.  Decided by the screen, which
               runs BEFORE (D); see `ropf.counterfactual.frequency`.
    survival   (D) solved.  Lost load is zero when a zero-shed dispatch exists
               inside the response window and the ratings, and positive when
               none does.

ONE AMPL INSTANCE FOR THE WHOLE CAMPAIGN.  A campaign solves (D) tens of
thousands of times against the same network.  `PostEvent` loads the network once
and thereafter only writes parameters, so AMPL never regenerates the model: the
disfigurement enters through the ``alive_*`` vectors of ``postevent.mod``, and
the two phases through ``shed_allowed`` and the choice of objective.  The
lifecycle rule of `ropf.model` applies unchanged -- nothing is closed or dropped
between solves.

THE RESPONSE WINDOW IS COMPUTED HERE, NOT IN AMPL.  Equation (6d) intersects
``P* +/- pi_g gamma`` with the unit's operating range, and P* can sit a hair
outside ``[Pmin, Pmax]`` after a solve, which makes the intersection empty.
Doing it in Python puts the guard somewhere a test can reach it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, Optional, Sequence, Set, Tuple

from amplpy import AMPL

from ..model import MODFILE_DIR, SolverConfig
from ..network import Network

POSTEVENT_MODFILE = "postevent.mod"

#: A shed below this is a rounding artefact of the LP, not lost load.
SHED_TOL = 1e-9

OUTCOMES = ("excluded", "collapse", "survival")


def _noop(_message: str) -> None:
    return None


###############################################################################
# Inputs
###############################################################################


@dataclass(frozen=True)
class PostEventParams:
    """The declared parameters of (D).

    gamma
        The response window scale, p.u.  A unit's window has half-width
        ``pi_g * gamma``.  No case file in this study carries ramp-rate data, so
        gamma is a declared study axis, not a measured quantity: it is swept,
        never assumed at one value and reported as if it were data.
    beta
        The emergency rating factor of eq (6e).  1 is the NORMAL rating and is
        the default.  No case here carries a usable emergency rating -- rateB
        and rateC are zero across the ACTIVSg distributions -- so a value above
        1 needs an external, cited source rather than a convention.
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
    """What a disfigurement removes, as network counts.

    A removed bus deletes every line incident to it and every unit located at
    it; a removed line deletes that line alone.  `survivors` applies that
    closure, so a caller may pass buses alone.
    """

    buses: FrozenSet[int] = frozenset()
    branches: FrozenSet[int] = frozenset()
    gens: FrozenSet[int] = frozenset()
    label: str = ""

    @property
    def size(self) -> int:
        """K, the number of components the disfigurement names.

        Units removed with their bus are not components in their own right;
        Section 4.1.2 counts buses and lines.
        """
        return len(self.buses) + len(self.branches)


@dataclass
class PostEventResult:
    """One evaluation of one disfigurement against one dispatch."""

    outcome: str
    reason: str = ""
    #: L, p.u.  None when (D) was never solved.
    lost_load: Optional[float] = None
    #: z-hat, the post-event generation cost, $.
    cost: Optional[float] = None
    #: Which phase produced the answer: 'cost' or 'loadshed'.
    phase: Optional[str] = None
    shed_by_bus: Dict[int, float] = field(default_factory=dict)
    Pg: Dict[int, float] = field(default_factory=dict)

    n_buses: int = 0
    n_branches: int = 0
    n_gens: int = 0
    #: Demand the removed buses took with them, p.u.  Not lost load -- eq (6f)
    #: sums over the surviving buses -- but two disfigurements are not
    #: comparable on L without it.
    removed_demand_pu: float = 0.0
    solve_time_s: float = 0.0
    n_solves: int = 0

    @property
    def survived(self) -> bool:
        return self.outcome == "survival"

    @property
    def served_all_demand(self) -> bool:
        return self.outcome == "survival" and (self.lost_load or 0.0) <= SHED_TOL


###############################################################################
# The surviving network
###############################################################################


def survivors(network: Network,
              disfigurement: Disfigurement) -> Tuple[Set[int], Set[int], Set[int]]:
    """Apply the removal closure.  Returns the SURVIVING (buses, branches, gens).

    A unit already out of service in the case file is never alive, so it cannot
    be "removed" twice and cannot be counted as a survivor that contributes
    capacity.  Out-of-service branches are not in `Network.branches` at all.
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
    """True when the surviving buses form one connected component.

    Section 4.2 excludes a disfigurement that splits the network.  This is the
    test that decides it, and it is why (D) never needs a per-island reference
    bus or a per-island power balance.
    """
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

    The guard matters: P* comes from a solved master and can sit outside
    ``[Pmin, Pmax]`` by the solver's own tolerance, which makes the naive
    intersection empty and (D) infeasible for a reason that has nothing to do
    with the disfigurement.  Where that happens the window collapses to the
    nearest feasible output rather than being reported as a contingency.
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
    """pi_g proportional to capacity, normalized over the in-service fleet.

    The default when no AGC participation factor is available for a case.  It
    is a surrogate and is named one: `ropf.counterfactual.dynamics` reads the
    real factors where the distribution ships them.
    """
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

        # Which components the last evaluation left alive, so that only the
        # difference is written on the next one.  A campaign changes a handful
        # of entries at a time; rewriting all 88,207 of them on
        # every solve is most of the cost at the top rung.
        self._alive_bus: Set[int] = set(network.buses)
        self._alive_br: Set[int] = set(network.branches)
        self._alive_gen: Set[int] = {c for c, g in network.gens.items() if g.status}
        self._write_alive_gen(self._alive_gen)

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

        # A window is required for every unit before the first solve; the
        # evaluation overwrites it.
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
        ``(cleared, reason)``.  It runs after the connectivity exclusion and
        before (D), which is the order Section 4.2 states.
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
                                   **counts)

        # ---- the frequency screen, before (D) ------------------------------
        if screen is not None:
            cleared, why = screen(live_gens)
            if not cleared:
                return PostEventResult(outcome="collapse", reason=why, **counts)

        self._apply(live_buses, live_branches, live_gens, P_star, pi, params)

        started = time.time()
        # ---- phase 1: the cheapest dispatch that sheds nothing -------------
        self.ampl.get_parameter("shed_allowed").set(0)
        self.ampl.eval("objective gen_cost;")
        self.ampl.solve()
        status = str(self.ampl.get_value("solve_result"))

        if status == "solved":
            return PostEventResult(
                outcome="survival", reason="all demand served",
                lost_load=0.0,
                cost=float(self.ampl.get_objective("gen_cost").value()),
                phase="cost", Pg=self._live_values("Pg", live_gens),
                solve_time_s=time.time() - started, n_solves=1, **counts)

        # ---- phase 2: the smallest lost load -------------------------------
        # Reaching here means no zero-shed operating point exists inside the
        # response window and the ratings.  Phase 2 always has a solution --
        # shedding every bus is feasible -- so a phase 2 that does not solve is
        # a modelling error and is reported as one, never as a survival.
        self.ampl.get_parameter("shed_allowed").set(1)
        self.ampl.eval("objective lost_load;")
        self.ampl.solve()
        status = str(self.ampl.get_value("solve_result"))
        elapsed = time.time() - started

        if status != "solved":
            return PostEventResult(
                outcome="survival", phase="loadshed",
                reason=f"the load-shed phase returned {status!r}, which cannot "
                       f"happen for a connected network: shedding every bus is "
                       f"feasible",
                solve_time_s=elapsed, n_solves=2, **counts)

        shed = {bus: value
                for bus, value in self._live_values("L", live_buses).items()
                if value > SHED_TOL}
        Pg = self._live_values("Pg", live_gens)
        return PostEventResult(
            outcome="survival", phase="loadshed",
            reason="no zero-shed operating point exists",
            lost_load=float(self.ampl.get_objective("lost_load").value()),
            cost=self._generation_cost(Pg), shed_by_bus=shed, Pg=Pg,
            solve_time_s=elapsed, n_solves=2, **counts)

    def _apply(self, live_buses, live_branches, live_gens,
               P_star, pi, params: PostEventParams) -> None:
        self._write_alive("alive_bus", self._alive_bus, live_buses)
        self._write_alive("alive_br", self._alive_br, live_branches)
        self._write_alive("alive_gen", self._alive_gen, live_gens)
        self._alive_bus, self._alive_br, self._alive_gen = \
            set(live_buses), set(live_branches), set(live_gens)

        lo, hi = response_window(self.network, P_star, pi, params.gamma)
        self.ampl.get_parameter("Pg_lo").setValues(lo)
        self.ampl.get_parameter("Pg_hi").setValues(hi)
        self.ampl.get_parameter("beta").set(float(params.beta))
        self.ampl.get_parameter("ref_bus").set(self._reference(live_buses))

    def _reference(self, live_buses: Set[int]) -> int:
        """eq (6g).  The pre-event reference bus where it survives, else any.

        The network is connected by the time this is reached, so the choice
        only fixes the gauge.
        """
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

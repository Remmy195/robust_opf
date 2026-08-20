"""The AMPL boundary.

Every optimization model in this study is an AMPL model file under
``modfiles/``, and this is the only module that touches AMPL.  Keeping that
boundary in one place is what makes the lifecycle rule below checkable by
inspection rather than by discipline.

    AMPL LIFECYCLE RULE.  Never call ``ampl.close()``, and never drop the last
    reference to an entity object (``ampl.get_variable("Pf")``), while a loop
    that will solve again is still running.  Doing either can hang the process
    on the entity destructor rather than raising.  A `Master` therefore holds
    its AMPL instance for its own lifetime, resolves entities once in
    ``__init__``, and exposes no teardown at all: the instance is released when
    the `Master` is garbage collected, after the loop is over.

The cut pool only ever grows.  A cut is appended, never replaced or removed, so
the master stays a relaxation of the true functional at every iteration, which
is what equation (11) of the manuscript rests on.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from amplpy import AMPL

from .network import Network

MODFILE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "modfiles")

#: Finite stand-in for a free bus angle.  The reference bus is pinned by
#: collapsing its bounds to zero rather than by a constraint row.
ANGLE_LIMIT_RAD = 6.28318530718

#: Manuscript risk functionals, mapped to the cut family that bounds them.
#: `max_line_loading` and the apparent-power Joule variants that the earlier
#: codebase carried are deliberately absent: they are not functionals in the
#: paper, and supporting them meant carrying a fourth cut family with no
#: consumer.
RISK_FAMILY = {
    "max_active_flow": 0,      # eq (4b), cut family (6a)
    "joule_loss_max": 1,       # eq (4c), cut family (6b)
    "bus_flow_sum_agg": 0,     # eq (4a), cut family (6c) -- see add_bus_cuts
}


def _noop(_message: str) -> None:
    return None


###############################################################################
# Solver configuration
###############################################################################


@dataclass
class SolverConfig:
    """Which solver to call and under what budget.

    Every limit is stated, never left to the solver's own default, so that a
    solve which does not finish is a recorded outcome against a declared budget
    rather than an unbounded run.
    """

    name: str = "gurobi"
    time_limit_s: float = 3600.0
    #: Gurobi LP method: primal, dual, barrier, concurrent, or auto.
    gurobi_method: Optional[str] = None
    #: Knitro algorithm; 1 is interior-direct.
    knitro_algorithm: int = 1
    knitro_threads: int = 40
    ipopt_max_iter: int = 3000
    ipopt_tol: float = 1e-6
    verbose: bool = True

    def apply(self, ampl: AMPL, log: Callable[[str], None]) -> None:
        name = self.name.lower()
        ampl.setOption("solver", name)
        if name == "gurobi":
            method_map = {"primal": 0, "simplex": 0, "dual": 1,
                          "barrier": 2, "interior": 2, "ipm": 2,
                          "concurrent": 3, "auto": None}
            opts = []
            code = method_map.get(str(self.gurobi_method).lower()) \
                if self.gurobi_method else None
            if code is not None:
                opts.append(f"method={code}")
            opts.append(f"timelim={self.time_limit_s:g}")
            opts.append(f"outlev={1 if self.verbose else 0}")
            ampl.setOption("gurobi_options", " ".join(opts))
        elif name == "knitro":
            opts = [f"algorithm={self.knitro_algorithm}",
                    f"numthreads={self.knitro_threads}",
                    "blasoptionlib=1", "linsolver=7",
                    f"maxtime_real={self.time_limit_s:g}"]
            if self.knitro_algorithm in (1, 6):
                opts.insert(1, "bar_murule=1")
            ampl.setOption("knitro_options", " ".join(opts))
        elif name == "ipopt":
            # The earlier codebase configured Gurobi and Knitro only, so IPOPT
            # ran with no declared limits at all -- an AC solve that stalled
            # simply never returned.  These three make the budget explicit.
            ampl.setOption("ipopt_options",
                           f"max_cpu_time={self.time_limit_s:g} "
                           f"max_iter={self.ipopt_max_iter} "
                           f"tol={self.ipopt_tol:g}")
        else:
            log(f" note: no option profile for solver '{name}'; using defaults\n")
        log(f" solver {name}, time limit {self.time_limit_s:g}s\n")


###############################################################################
# Results
###############################################################################


@dataclass
class Solution:
    """One solve of a master problem."""

    status: str
    objective: float
    gen_cost: float
    phi: float
    Pg: Dict[int, float] = field(default_factory=dict)
    Pf: Dict[int, float] = field(default_factory=dict)
    Pt: Dict[int, float] = field(default_factory=dict)
    Qf: Dict[int, float] = field(default_factory=dict)
    Qt: Dict[int, float] = field(default_factory=dict)
    Qg: Dict[int, float] = field(default_factory=dict)
    v: Dict[int, float] = field(default_factory=dict)
    theta: Dict[int, float] = field(default_factory=dict)
    solve_time_s: float = 0.0

    @property
    def solved(self) -> bool:
        return self.status == "solved"


@dataclass(frozen=True)
class BusCut:
    """One instance of eq (6c), the bus family cut at a single bus.

    ``branch_pos``/``branch_neg`` and ``gen_pos``/``gen_neg`` carry the incident
    components split by the sign they take at the incumbent; a component whose
    value is zero there contributes nothing and appears in neither.  ``demand``
    is the constant |P_di|.
    """

    bus: int
    branch_pos: Sequence[int]
    branch_neg: Sequence[int]
    gen_pos: Sequence[int]
    gen_neg: Sequence[int]
    demand: float

    @property
    def key(self) -> tuple:
        """What identifies this hyperplane, for the Section 3 exclusion.

        The bus AND its sign pattern together: a repeated bus under a new sign
        pattern is a different hyperplane and a genuinely new cut.  Keyed off
        the cut rather than off the evaluation that produced it, so that the
        pool's record of what it holds and the separation's record of what to
        skip cannot drift apart -- there is one definition, and it is this one.
        """
        return (self.bus,
                tuple(sorted(self.branch_pos)), tuple(sorted(self.branch_neg)),
                tuple(sorted(self.gen_pos)), tuple(sorted(self.gen_neg)))

    @property
    def is_empty(self) -> bool:
        """A cut with no terms reads ``Phi >= 0``, which the model already has."""
        return not (self.branch_pos or self.branch_neg
                    or self.gen_pos or self.gen_neg or self.demand)


###############################################################################
# The master problem
###############################################################################


class Master:
    """(M) or (M^ac), loaded with one network and solved repeatedly.

    Construct once per run and call `solve` as many times as the loop needs.
    `add_line_cuts` and `add_bus_cuts` append to the pool between solves.
    """

    def __init__(self,
                 network: Network,
                 modfile: str = "master.mod",
                 solver: Optional[SolverConfig] = None,
                 log: Optional[Callable[[str], None]] = None,
                 line_cut_capacity: int = 0,
                 bus_cut_capacity: int = 0):
        """`*_cut_capacity` sizes the pool; the network's own size is the floor.

        MAX_CUTS and MAX_BUS_CUTS index declared entities, so they are set once
        here and never afterwards.  The line families cannot outgrow the network
        -- the separation excludes branches already cut -- but the bus family
        can: it excludes by bus AND sign pattern, so one bus may legitimately
        carry several cuts and a long run can need more than `numbuses` of them.
        The caller therefore states the budget it intends to spend.
        """
        self.network = network
        self.log = log or _noop
        self.solver = solver or SolverConfig()
        self.modfile = modfile
        self.is_ac = os.path.basename(modfile).endswith("_ac.mod")
        self.line_cut_capacity = max(1, network.numbranches, int(line_cut_capacity))
        self.bus_cut_capacity = max(1, network.numbuses, int(bus_cut_capacity))

        path = modfile if os.path.isabs(modfile) else os.path.join(MODFILE_DIR, modfile)
        if not os.path.exists(path):
            raise FileNotFoundError(f"model file not found: {path}")

        self.log(f" loading {os.path.basename(path)}\n")
        self.ampl = AMPL()
        self.ampl.read(path)
        self.ampl.eval("option display_precision 0;")
        self.ampl.eval("option expand_precision 0;")
        self.ampl.setOption("presolve", 0)
        self.solver.apply(self.ampl, self.log)

        self._n_line_cuts = 0
        self._n_bus_cuts = 0
        self._line_cut_ids: List[int] = []
        self._bus_cuts: List[BusCut] = []

        self._load_network()

    # -- construction ------------------------------------------------------

    def _load_network(self) -> None:
        net = self.network
        t0 = time.time()
        ampl = self.ampl

        ampl.getSet("buses").setValues(list(net.buses))
        ampl.getSet("gens").setValues(list(net.gens))
        ampl.getSet("branches").setValues(list(net.branches))

        # --- buses ---------------------------------------------------------
        Pd, Gs, theta_min, theta_max = {}, {}, {}, {}
        Qd, Bs, Vmax, Vmin, Vinit = {}, {}, {}, {}, {}
        for count, bus in net.buses.items():
            Pd[count] = bus.Pd
            Gs[count] = bus.Gs
            # The reference bus angle is pinned by collapsing its bounds rather
            # than by a constraint row, so eq (1e)/(2i) costs no row.
            if count == net.refbus:
                theta_min[count] = theta_max[count] = 0.0
            else:
                theta_min[count] = -ANGLE_LIMIT_RAD
                theta_max[count] = ANGLE_LIMIT_RAD
            if self.is_ac:
                Qd[count] = bus.Qd
                Bs[count] = bus.Bs
                Vmax[count] = bus.Vmax
                Vmin[count] = bus.Vmin
                Vinit[count] = 1.0

        ampl.get_parameter("Pd").setValues(Pd)
        ampl.get_parameter("Gs").setValues(Gs)
        ampl.get_parameter("theta_min").setValues(theta_min)
        ampl.get_parameter("theta_max").setValues(theta_max)

        # --- branches ------------------------------------------------------
        bus_f, bus_t, U, r = {}, {}, {}, {}
        maxangle, minangle = {}, {}
        bdc, Pfinj, thetadiffinit = {}, {}, {}
        Gff, Bff, Gft, Bft, Gtf, Btf, Gtt, Btt = ({} for _ in range(8))
        for count, br in net.branches.items():
            bus_f[count] = br.id_f
            bus_t[count] = br.id_t
            U[count] = br.limit
            # The nonnegative heat coefficient, not the raw series
            # resistance.  See `Branch.r_heat`.
            r[count] = br.r_heat
            maxangle[count] = br.maxangle_rad
            minangle[count] = br.minangle_rad
            bdc[count] = br.bdc
            Pfinj[count] = br.Pfinj
            if self.is_ac:
                thetadiffinit[count] = 0.0
                Gff[count], Bff[count] = br.Gff, br.Bff
                Gft[count], Bft[count] = br.Gft, br.Bft
                Gtf[count], Btf[count] = br.Gtf, br.Btf
                Gtt[count], Btt[count] = br.Gtt, br.Btt

        ampl.get_parameter("bus_f").setValues(bus_f)
        ampl.get_parameter("bus_t").setValues(bus_t)
        ampl.get_parameter("U").setValues(U)
        ampl.get_parameter("r").setValues(r)
        ampl.get_parameter("maxangle").setValues(maxangle)
        ampl.get_parameter("minangle").setValues(minangle)

        if self.is_ac:
            ampl.get_parameter("Qd").setValues(Qd)
            ampl.get_parameter("Bs").setValues(Bs)
            ampl.get_parameter("Vmax").setValues(Vmax)
            ampl.get_parameter("Vmin").setValues(Vmin)
            ampl.get_parameter("Vinit").setValues(Vinit)
            ampl.get_parameter("thetadiffinit").setValues(thetadiffinit)
            for name, values in (("Gff", Gff), ("Bff", Bff), ("Gft", Gft),
                                 ("Bft", Bft), ("Gtf", Gtf), ("Btf", Btf),
                                 ("Gtt", Gtt), ("Btt", Btt)):
                ampl.get_parameter(name).setValues(values)
        else:
            ampl.get_parameter("bdc").setValues(bdc)
            ampl.get_parameter("Pfinj").setValues(Pfinj)

        # --- generators ----------------------------------------------------
        Pmax, Pmin, fixedcost, lincost, quadcost = {}, {}, {}, {}, {}
        Qmax, Qmin = {}, {}
        for count, gen in net.gens.items():
            # An out-of-service unit is held at zero rather than removed, so the
            # generator index set matches the case file row for row.
            Pmax[count] = gen.Pmax if gen.status else 0.0
            Pmin[count] = gen.Pmin if gen.status else 0.0
            quadcost[count], lincost[count], fixedcost[count] = gen.costvector
            # The no-load term must be zeroed too, not just the bounds.  It is
            # the only cost that survives Pg == 0, so leaving it in charges the
            # system for units that are not running: on ACTIVSg200 that is 11
            # units and $7,173.15 added to every reported cost.  MATPOWER drops
            # offline units outright, which is what makes this visible as a
            # constant offset against `rundcopf`.
            if not gen.status:
                fixedcost[count] = 0.0
            if self.is_ac:
                Qmax[count] = gen.Qmax if gen.status else 0.0
                Qmin[count] = gen.Qmin if gen.status else 0.0

        ampl.get_parameter("Pmax").setValues(Pmax)
        ampl.get_parameter("Pmin").setValues(Pmin)
        ampl.get_parameter("fixedcost").setValues(fixedcost)
        ampl.get_parameter("lincost").setValues(lincost)
        ampl.get_parameter("quadcost").setValues(quadcost)
        if self.is_ac:
            ampl.get_parameter("Qmax").setValues(Qmax)
            ampl.get_parameter("Qmin").setValues(Qmin)

        # --- incidence sets -------------------------------------------------
        branches_f = ampl.getSet("branches_f")
        branches_t = ampl.getSet("branches_t")
        bus_gens = ampl.getSet("bus_gens")
        for count, bus in net.buses.items():
            branches_f[count].setValues(list(bus.frombranchids.values()))
            branches_t[count].setValues(list(bus.tobranchids.values()))
            bus_gens[count].setValues(list(bus.genidsbycount))

        ampl.get_parameter("MAX_CUTS").set(self.line_cut_capacity)
        ampl.get_parameter("MAX_BUS_CUTS").set(self.bus_cut_capacity)
        ampl.get_parameter("nCUT").set(0)
        ampl.get_parameter("nBUSCUT").set(0)

        self.log(f" loaded {net.numbuses} buses, {net.numbranches} branches,"
                 f" {net.numgens} gens in {time.time() - t0:.1f}s\n")

    # -- risk parameters ---------------------------------------------------

    def set_risk_weight(self, weight: float) -> None:
        """lambda in eq (1a)."""
        self.ampl.get_parameter("risk_weight").set(float(weight))

    def set_risk_family(self, metric: str) -> None:
        if metric not in RISK_FAMILY:
            raise ValueError(
                f"unknown risk functional {metric!r}; "
                f"expected one of {sorted(RISK_FAMILY)}")
        self.ampl.get_parameter("risk_family").set(RISK_FAMILY[metric])

    # -- cuts --------------------------------------------------------------

    @property
    def n_line_cuts(self) -> int:
        return self._n_line_cuts

    @property
    def n_bus_cuts(self) -> int:
        return self._n_bus_cuts

    @property
    def line_cut_ids(self) -> List[int]:
        return list(self._line_cut_ids)

    @property
    def bus_cuts(self) -> List[BusCut]:
        """The eq (6c) cuts this master holds, in the order they were added.

        The pool is readable because Section 3.4 transfers it: the DC master's
        pool is appended to the AC master unchanged.
        """
        return list(self._bus_cuts)

    @property
    def bus_cut_keys(self) -> set:
        """Exclusion keys for what the pool already holds.  See `BusCut.key`."""
        return {cut.key for cut in self._bus_cuts}

    def add_line_cuts(self, branch_ids: Sequence[int]) -> int:
        """Append eq (6a) or (6b) cuts for `branch_ids`.  Returns how many."""
        new = [int(b) for b in branch_ids]
        if not new:
            return 0
        if self._n_line_cuts + len(new) > self.line_cut_capacity:
            raise ValueError(
                f"line cut pool would reach {self._n_line_cuts + len(new)} cuts, "
                f"past the capacity of {self.line_cut_capacity} declared at "
                f"construction; raise line_cut_capacity")
        choose = self.ampl.get_parameter("choose")
        for branch_id in new:
            self._n_line_cuts += 1
            self._line_cut_ids.append(branch_id)
            choose[self._n_line_cuts] = branch_id
        self.ampl.get_parameter("nCUT").set(self._n_line_cuts)
        return len(new)

    def add_bus_cuts(self, cuts: Sequence[BusCut]) -> int:
        """Append eq (6c) cuts.  Returns how many were actually added.

        A cut whose terms are all zero is skipped: it would read ``Phi >= 0``,
        which the variable's own bound already imposes.
        """
        added = 0
        pos_br = self.ampl.getSet("bus_cut_br_pos")
        neg_br = self.ampl.getSet("bus_cut_br_neg")
        pos_gen = self.ampl.getSet("bus_cut_gen_pos")
        neg_gen = self.ampl.getSet("bus_cut_gen_neg")
        demand = self.ampl.get_parameter("bus_cut_demand")

        wanted = sum(1 for cut in cuts if not cut.is_empty)
        if self._n_bus_cuts + wanted > self.bus_cut_capacity:
            raise ValueError(
                f"bus cut pool would reach {self._n_bus_cuts + wanted} cuts, "
                f"past the capacity of {self.bus_cut_capacity} declared at "
                f"construction; raise bus_cut_capacity")

        for cut in cuts:
            if cut.is_empty:
                continue
            self._n_bus_cuts += 1
            k = self._n_bus_cuts
            self._bus_cuts.append(cut)
            pos_br[k].setValues(list(cut.branch_pos))
            neg_br[k].setValues(list(cut.branch_neg))
            pos_gen[k].setValues(list(cut.gen_pos))
            neg_gen[k].setValues(list(cut.gen_neg))
            demand[k] = float(cut.demand)
            added += 1

        if added:
            self.ampl.get_parameter("nBUSCUT").set(self._n_bus_cuts)
        return added

    # -- solving -----------------------------------------------------------

    def solve(self) -> Solution:
        t0 = time.time()
        self.ampl.solve()
        elapsed = time.time() - t0
        status = str(self.ampl.get_value("solve_result"))

        objective = float(self.ampl.get_objective("total_cost").value())
        phi = float(self.ampl.get_variable("Phi").value())
        weight = float(self.ampl.get_parameter("risk_weight").value())
        # The reported cost is the generation term alone; the objective carries
        # lambda * Phi on top of it, and the two are not comparable across the
        # weight grid.
        gen_cost = objective - weight * phi

        Pg = self._values("Pg")
        Pf = self._values("Pf")
        solution = Solution(status=status, objective=objective,
                            gen_cost=gen_cost, phi=phi,
                            Pg=Pg, Pf=Pf, solve_time_s=elapsed)
        if self.is_ac:
            solution.Pt = self._values("Pt")
            solution.Qf = self._values("Qf")
            solution.Qt = self._values("Qt")
            solution.Qg = self._values("Qg")
            solution.v = self._values("v")
        else:
            # Lossless DC: the to-end flow is not a variable, but downstream
            # consumers expect it.  See modfiles/master.mod.
            solution.Pt = {k: -val for k, val in Pf.items()}
        solution.theta = self._values("theta")

        self.log(f" solve_result {status}, objective {objective:.6f},"
                 f" cost {gen_cost:.6f}, Phi {phi:.6f}, {elapsed:.1f}s\n")
        return solution

    def _values(self, name: str) -> Dict[int, float]:
        raw = self.ampl.get_variable(name).get_values().to_dict()
        return {int(k): float(v) for k, v in raw.items()}

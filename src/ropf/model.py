"""The AMPL boundary: (M) and (M^ac), loaded from ``modfiles/``.

AMPL LIFECYCLE RULE.  Never call ``ampl.close()`` and never drop the last
reference to an entity object while a loop that will solve again is running;
either can hang the process on the entity destructor rather than raising.  A
`Master` holds its AMPL instance for its own lifetime and exposes no teardown.

The cut pool only ever grows, so the master stays a relaxation of the true
functional at every iteration -- eq (11).
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

#: Finite stand-in for a free bus angle; the reference bus is pinned by
#: collapsing its bounds rather than by a constraint row.
ANGLE_LIMIT_RAD = 6.28318530718

#: Risk functional -> the cut family that bounds it.
RISK_FAMILY = {
    "max_active_flow": 0,      # eq (4b), cut family (6a)
    "joule_loss_max": 1,       # eq (4c), cut family (6b)
    "bus_flow_sum_agg": 0,     # eq (4a), cut family (6c) -- see add_bus_cuts
}

#: Solvers whose AMPL driver can write the presolved problem to a file.
LP_WRITERS = ("gurobi", "cplex", "xpress", "copt", "mosek")


def _noop(_message: str) -> None:
    return None


###############################################################################
# Solver configuration
###############################################################################


@dataclass
class SolverConfig:
    """Which solver to call and under what budget.  Every limit is stated, so a
    solve that does not finish is a recorded outcome against a declared budget."""

    name: str = "gurobi"
    time_limit_s: float = 3600.0
    #: Gurobi LP method: primal, dual, barrier, concurrent, or auto.
    gurobi_method: Optional[str] = None
    #: Crossover after barrier.  0 returns the interior point, which spreads
    #: the answer across the optimal face rather than concentrating it at a
    #: vertex: the total and the cost are unaffected, a per-bus pattern is not.
    gurobi_crossover: Optional[int] = None
    #: Set this to the PHYSICAL core count.  Letting Gurobi default to a thread
    #: per LOGICAL cpu cost 5.7x wall clock on ACTIVSg70k for the same answer.
    gurobi_threads: Optional[int] = None
    gurobi_numericfocus: Optional[int] = None
    gurobi_scaleflag: Optional[int] = None
    #: Knitro algorithm; 1 is interior-direct.
    knitro_algorithm: int = 1
    knitro_threads: int = 40
    ipopt_max_iter: int = 3000
    ipopt_tol: float = 1e-6
    verbose: bool = True

    @property
    def option_key(self) -> str:
        return f"{self.name.lower()}_options"

    @property
    def writes_lp(self) -> bool:
        return self.name.lower() in LP_WRITERS

    def options(self) -> str:
        """The solver option string for `name`, or "" where none is profiled."""
        name = self.name.lower()
        if name == "gurobi":
            methods = {"primal": 0, "simplex": 0, "dual": 1, "barrier": 2,
                       "interior": 2, "ipm": 2, "concurrent": 3, "auto": None}
            opts = []
            code = (methods.get(str(self.gurobi_method).lower())
                    if self.gurobi_method else None)
            if code is not None:
                opts.append(f"method={code}")
            for flag, value in (("crossover", self.gurobi_crossover),
                                ("threads", self.gurobi_threads),
                                ("numericfocus", self.gurobi_numericfocus),
                                ("scale", self.gurobi_scaleflag)):
                if value is not None:
                    opts.append(f"{flag}={value}")
            opts.append(f"timelim={self.time_limit_s:g}")
            opts.append(f"outlev={1 if self.verbose else 0}")
            return " ".join(opts)
        if name == "knitro":
            opts = [f"algorithm={self.knitro_algorithm}",
                    f"numthreads={self.knitro_threads}",
                    "blasoptionlib=1", "linsolver=7",
                    f"maxtime_real={self.time_limit_s:g}"]
            if self.knitro_algorithm in (1, 6):
                opts.insert(1, "bar_murule=1")
            return " ".join(opts)
        if name == "ipopt":
            return (f"max_cpu_time={self.time_limit_s:g} "
                    f"max_iter={self.ipopt_max_iter} tol={self.ipopt_tol:g}")
        return ""

    def apply(self, ampl: AMPL, log: Callable[[str], None]) -> None:
        name = self.name.lower()
        ampl.setOption("solver", name)
        options = self.options()
        if options:
            ampl.setOption(self.option_key, options)
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
    #: Where the LP of this solve was written, when LP output is on.
    lp_path: Optional[str] = None

    @property
    def solved(self) -> bool:
        return self.status == "solved"


@dataclass(frozen=True)
class BusCut:
    """One instance of eq (6c): the incident lines at a bus, split by the sign
    they take at the incumbent.  A line whose flow is zero appears in neither."""

    bus: int
    branch_pos: Sequence[int]
    branch_neg: Sequence[int]

    @property
    def key(self) -> tuple:
        """The bus AND its sign pattern: a repeated bus under a new pattern is a
        different hyperplane and a genuinely new cut."""
        return (self.bus,
                tuple(sorted(self.branch_pos)), tuple(sorted(self.branch_neg)))

    @property
    def is_empty(self) -> bool:
        """A cut with no terms reads Phi >= 0, which the model already has."""
        return not (self.branch_pos or self.branch_neg)


###############################################################################
# The master problem
###############################################################################


class Master:
    """(M) or (M^ac), loaded with one network and solved repeatedly.

    Construct once per run, call `solve` as often as the loop needs, and append
    to the pool with `add_line_cuts`/`add_bus_cuts` between solves.
    """

    def __init__(self,
                 network: Network,
                 modfile: str = "master.mod",
                 solver: Optional[SolverConfig] = None,
                 log: Optional[Callable[[str], None]] = None,
                 line_cut_capacity: int = 0,
                 bus_cut_capacity: int = 0):
        """`*_cut_capacity` sizes the pool; the network's own size is the floor.

        MAX_CUTS and MAX_BUS_CUTS index declared entities and are set once here.
        The bus family can outgrow the bus count -- one bus may carry several
        sign patterns -- so the caller states the budget it intends to spend.
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

        self._lp_dir: Optional[str] = None
        self._lp_stem = "master"
        #: The label the next LP file carries.  Algorithm 1 sets it to k.
        self.lp_iteration = 0

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
            r[count] = br.r_heat          # the heat coefficient, not raw series r
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
            # generator index set matches the case file row for row.  The
            # no-load cost must be zeroed too: it is the only cost surviving
            # Pg == 0, and MATPOWER drops offline units outright.
            Pmax[count] = gen.Pmax if gen.status else 0.0
            Pmin[count] = gen.Pmin if gen.status else 0.0
            quadcost[count], lincost[count], fixedcost[count] = gen.costvector
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
            raise ValueError(f"unknown risk functional {metric!r}; "
                             f"expected one of {sorted(RISK_FAMILY)}")
        self.ampl.get_parameter("risk_family").set(RISK_FAMILY[metric])

    # -- LP output ---------------------------------------------------------

    def set_lp_output(self, directory: Optional[str],
                      stem: str = "master") -> None:
        """Write the problem of every subsequent solve to `directory`.

        One file per solve, named ``<stem>_k<iteration>.lp`` from
        `lp_iteration`, which Algorithm 1 sets to k.  The driver writes it as
        part of the solve, so this costs no extra solve; it does cost the file.
        Columns and rows carry solver-generated names -- AMPL's own entity
        names are not exposed through the driver -- so the LP is for inspecting
        the model's shape and size, and `ropf.risk.exposed` is what names the
        components.
        """
        if directory and not self.solver.writes_lp:
            self.log(f" note: solver '{self.solver.name}' cannot write an LP;"
                     f" LP output is off\n")
            self._lp_dir = None
            return
        self._lp_dir = directory
        self._lp_stem = stem
        if directory:
            os.makedirs(directory, exist_ok=True)
            self.log(f" LP output on: {directory}"
                     f"{os.sep}{stem}_k<iteration>.lp\n")

    def lp_path(self) -> Optional[str]:
        """Where the next solve would write its LP, or None when output is off."""
        if not self._lp_dir:
            return None
        return os.path.join(self._lp_dir,
                            f"{self._lp_stem}_k{self.lp_iteration:03d}.lp")

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
        """The eq (6c) cuts held, in the order added.  Readable because Section
        3.4 transfers the DC pool to the AC master unchanged."""
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
        """Append eq (6c) cuts.  Returns how many were added; an empty cut is
        skipped, since it would read Phi >= 0."""
        added = 0
        pos_br = self.ampl.getSet("bus_cut_br_pos")
        neg_br = self.ampl.getSet("bus_cut_br_neg")

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
            self._bus_cuts.append(cut)
            pos_br[self._n_bus_cuts].setValues(list(cut.branch_pos))
            neg_br[self._n_bus_cuts].setValues(list(cut.branch_neg))
            added += 1

        if added:
            self.ampl.get_parameter("nBUSCUT").set(self._n_bus_cuts)
        return added

    # -- solving -----------------------------------------------------------

    def solve(self) -> Solution:
        lp_path = self.lp_path()
        options = self.solver.options()
        if lp_path:
            options = (f"{options} writeprob="
                       f"{lp_path.replace(os.sep, '/')}").strip()
        if options:
            self.ampl.setOption(self.solver.option_key, options)

        t0 = time.time()
        self.ampl.solve()
        elapsed = time.time() - t0
        status = str(self.ampl.get_value("solve_result"))

        objective = float(self.ampl.get_objective("total_cost").value())
        phi = float(self.ampl.get_variable("Phi").value())
        weight = float(self.ampl.get_parameter("risk_weight").value())
        # The reported cost is the generation term alone; the objective carries
        # lambda * Phi on top and is not comparable across the weight grid.
        gen_cost = objective - weight * phi

        Pf = self._values("Pf")
        solution = Solution(status=status, objective=objective,
                            gen_cost=gen_cost, phi=phi,
                            Pg=self._values("Pg"), Pf=Pf,
                            solve_time_s=elapsed, lp_path=lp_path)
        if self.is_ac:
            solution.Pt = self._values("Pt")
            solution.Qf = self._values("Qf")
            solution.Qt = self._values("Qt")
            solution.Qg = self._values("Qg")
            solution.v = self._values("v")
        else:
            # Lossless DC: Pt is not a variable, but consumers expect it.
            solution.Pt = {k: -val for k, val in Pf.items()}
        solution.theta = self._values("theta")

        self.log(f" solve_result {status}, objective {objective:.6f},"
                 f" cost {gen_cost:.6f}, Phi {phi:.6f}, {elapsed:.1f}s\n")
        if lp_path and os.path.exists(lp_path):
            self.log(f" wrote LP {lp_path} ({os.path.getsize(lp_path)} bytes)\n")
        return solution

    def _values(self, name: str) -> Dict[int, float]:
        raw = self.ampl.get_variable(name).get_values().to_dict()
        return {int(k): float(v) for k, v in raw.items()}

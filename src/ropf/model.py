"""The AMPL master problems (M) and (M^ac), loaded from modfiles/."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from amplpy import AMPL, OutputHandler

from .network import Network

MODFILE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "modfiles")

ANGLE_LIMIT_RAD = 6.28318530718

# Exposure representation -> risk_family in the model; bus cuts have their own block.
RISK_FAMILY = {"max_active_flow": 0, "joule_loss_max": 1, "bus_flow_sum_agg": 0}


def _noop(_message: str) -> None:
    return None


class _SolverOutput(OutputHandler):
    """Forwards AMPL and solver output to the log, indented."""

    def __init__(self, log: Callable[[str], None]):
        self.log = log
        self.at_line_start = True

    def output(self, kind, msg: str) -> None:
        text = []
        for line in msg.replace("\r\n", "\n").splitlines(keepends=True):
            text.append(("    " if self.at_line_start else "") + line)
            self.at_line_start = line.endswith("\n")
        self.log("".join(text))

    def end_line(self) -> None:
        if not self.at_line_start:
            self.log("\n")
            self.at_line_start = True


@dataclass
class SolverConfig:
    name: str = "gurobi"
    time_limit_s: float = 3600.0
    threads: Optional[int] = None
    options: str = ""
    verbose: bool = False

    def option_string(self) -> str:
        name = self.name.lower()
        opts: List[str] = []
        if name == "gurobi":
            if self.threads:
                opts.append(f"threads={self.threads}")
            opts += [f"timelim={self.time_limit_s:g}",
                     f"outlev={1 if self.verbose else 0}"]
        elif name == "knitro":
            opts = ["algorithm=1", "bar_murule=1"]
            if self.threads:
                opts.append(f"numthreads={self.threads}")
            opts += ["blasoptionlib=1", "linsolver=7",
                     f"maxtime_real={self.time_limit_s:g}"]
        elif name == "ipopt":
            opts = [f"max_cpu_time={self.time_limit_s:g}", "max_iter=3000",
                    "tol=1e-06"]
        if self.options:
            opts.append(self.options)
        return " ".join(opts)

    def apply(self, ampl: AMPL) -> None:
        ampl.setOption("solver", self.name.lower())
        options = self.option_string()
        if options:
            ampl.setOption(f"{self.name.lower()}_options", options)


@dataclass
class Solution:
    status: str
    objective: float
    gen_cost: float
    phi: float
    Pg: Dict[int, float] = field(default_factory=dict)
    Pf: Dict[int, float] = field(default_factory=dict)
    theta: Dict[int, float] = field(default_factory=dict)
    Pt: Dict[int, float] = field(default_factory=dict)
    Qf: Dict[int, float] = field(default_factory=dict)
    Qt: Dict[int, float] = field(default_factory=dict)
    Qg: Dict[int, float] = field(default_factory=dict)
    v: Dict[int, float] = field(default_factory=dict)
    solve_time_s: float = 0.0

    @property
    def solved(self) -> bool:
        return self.status == "solved"


@dataclass(frozen=True)
class BusCut:
    """Bus cut at `bus`: incident lines split by their sign at the incumbent."""

    bus: int
    branch_pos: Sequence[int]
    branch_neg: Sequence[int]

    @property
    def key(self) -> tuple:
        return (self.bus,
                tuple(sorted(self.branch_pos)), tuple(sorted(self.branch_neg)))

    @property
    def is_empty(self) -> bool:
        return not (self.branch_pos or self.branch_neg)


class Master:
    """(M) or (M^ac) for one network; solve, append cuts, solve again.

    Never call ampl.close() mid-loop: it can hang on the entity destructor.
    """

    def __init__(self,
                 network: Network,
                 modfile: str = "master.mod",
                 solver: Optional[SolverConfig] = None,
                 log: Optional[Callable[[str], None]] = None,
                 line_cut_capacity: int = 0,
                 bus_cut_capacity: int = 0):
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

        self.ampl = AMPL()
        self._output = _SolverOutput(self.log)
        self.ampl.set_output_handler(self._output)
        self.ampl.read(path)
        self.ampl.eval("option display_precision 0;")
        self.ampl.eval("option expand_precision 0;")
        self.ampl.setOption("presolve", 0)
        self.solver.apply(self.ampl)
        version = self.ampl.get_option("version").splitlines()[0]
        self.log(f"{os.path.basename(path)}: {version}, {self.solver.name}"
                 f" '{self.solver.option_string()}'\n")

        self._line_cut_ids: List[int] = []
        self._bus_cuts: List[BusCut] = []
        self._load_network()

    def _load_network(self) -> None:
        net, ampl, ac = self.network, self.ampl, self.is_ac

        ampl.getSet("buses").setValues(list(net.buses))
        ampl.getSet("gens").setValues(list(net.gens))
        ampl.getSet("branches").setValues(list(net.branches))

        bus = {name: {} for name in ("Pd", "Gs", "theta_min", "theta_max")}
        if ac:
            bus.update({name: {} for name in ("Qd", "Bs", "Vmax", "Vmin", "Vinit")})
        for count, b in net.buses.items():
            bus["Pd"][count] = b.Pd
            bus["Gs"][count] = b.Gs
            ref = count == net.refbus
            bus["theta_min"][count] = 0.0 if ref else -ANGLE_LIMIT_RAD
            bus["theta_max"][count] = 0.0 if ref else ANGLE_LIMIT_RAD
            if ac:
                bus["Qd"][count] = b.Qd
                bus["Bs"][count] = b.Bs
                bus["Vmax"][count] = b.Vmax
                bus["Vmin"][count] = b.Vmin
                bus["Vinit"][count] = 1.0

        branch_attrs = {"bus_f": "id_f", "bus_t": "id_t", "U": "limit",
                        "r": "r_heat", "maxangle": "maxangle_rad",
                        "minangle": "minangle_rad"}
        if ac:
            branch_attrs.update({n: n for n in ("Gff", "Bff", "Gft", "Bft",
                                                "Gtf", "Btf", "Gtt", "Btt")})
        else:
            branch_attrs.update({"bdc": "bdc", "Pfinj": "Pfinj"})
        branch = {name: {c: getattr(br, attr) for c, br in net.branches.items()}
                  for name, attr in branch_attrs.items()}
        if ac:
            branch["thetadiffinit"] = {c: 0.0 for c in net.branches}

        gen = {name: {} for name in ("Pmax", "Pmin", "quadcost", "lincost",
                                     "fixedcost")}
        if ac:
            gen.update({"Qmax": {}, "Qmin": {}})
        for count, g in net.gens.items():
            on = bool(g.status)
            gen["Pmax"][count] = g.Pmax if on else 0.0
            gen["Pmin"][count] = g.Pmin if on else 0.0
            quad, lin, fixed = g.costvector
            gen["quadcost"][count] = quad
            gen["lincost"][count] = lin
            # An offline unit is held at zero, so its no-load cost must go too.
            gen["fixedcost"][count] = fixed if on else 0.0
            if ac:
                gen["Qmax"][count] = g.Qmax if on else 0.0
                gen["Qmin"][count] = g.Qmin if on else 0.0

        for table in (bus, branch, gen):
            for name, values in table.items():
                ampl.get_parameter(name).setValues(values)

        branches_f = ampl.getSet("branches_f")
        branches_t = ampl.getSet("branches_t")
        bus_gens = ampl.getSet("bus_gens")
        for count, b in net.buses.items():
            branches_f[count].setValues(b.frombranchids)
            branches_t[count].setValues(b.tobranchids)
            bus_gens[count].setValues(b.genidsbycount)

        ampl.get_parameter("MAX_CUTS").set(self.line_cut_capacity)
        ampl.get_parameter("MAX_BUS_CUTS").set(self.bus_cut_capacity)
        ampl.get_parameter("nCUT").set(0)
        ampl.get_parameter("nBUSCUT").set(0)

    def set_risk_weight(self, weight: float) -> None:
        self.ampl.get_parameter("risk_weight").set(float(weight))

    def set_risk_family(self, metric: str) -> None:
        if metric not in RISK_FAMILY:
            raise ValueError(f"unknown risk functional {metric!r}")
        self.ampl.get_parameter("risk_family").set(RISK_FAMILY[metric])

    def set_order(self, q: int, mean_form: Optional[bool] = None) -> None:
        """q > 1 uses the eq:cutmean block; (M^ac) has none."""
        if q < 1:
            raise ValueError(f"the order q must be at least 1, got {q}")
        mean_form = q > 1 if mean_form is None else bool(mean_form)
        if self.is_ac:
            if mean_form:
                raise ValueError("order q > 1 runs on (M) only")
            return
        self.ampl.get_parameter("q").set(int(q))
        self.ampl.get_parameter("mean_form").set(int(mean_form))

    @property
    def n_line_cuts(self) -> int:
        return len(self._line_cut_ids)

    @property
    def n_bus_cuts(self) -> int:
        return len(self._bus_cuts)

    @property
    def line_cut_ids(self) -> List[int]:
        return list(self._line_cut_ids)

    @property
    def bus_cuts(self) -> List[BusCut]:
        return list(self._bus_cuts)

    @property
    def bus_cut_keys(self) -> set:
        return {cut.key for cut in self._bus_cuts}

    def add_line_cuts(self, branch_ids: Sequence[int]) -> int:
        new = [int(b) for b in branch_ids]
        if self.n_line_cuts + len(new) > self.line_cut_capacity:
            raise ValueError("line cut pool is past line_cut_capacity")
        choose = self.ampl.get_parameter("choose")
        for branch_id in new:
            self._line_cut_ids.append(branch_id)
            choose[self.n_line_cuts] = branch_id
        if new:
            self.ampl.get_parameter("nCUT").set(self.n_line_cuts)
        return len(new)

    def add_bus_cuts(self, cuts: Sequence[BusCut]) -> int:
        cuts = [cut for cut in cuts if not cut.is_empty]
        if self.n_bus_cuts + len(cuts) > self.bus_cut_capacity:
            raise ValueError("bus cut pool is past bus_cut_capacity")
        pos_br = self.ampl.getSet("bus_cut_br_pos")
        neg_br = self.ampl.getSet("bus_cut_br_neg")
        bus_of = None if self.is_ac else self.ampl.get_parameter("bus_cut_bus")
        for cut in cuts:
            self._bus_cuts.append(cut)
            k = self.n_bus_cuts
            pos_br[k].setValues(list(cut.branch_pos))
            neg_br[k].setValues(list(cut.branch_neg))
            if bus_of is not None:
                bus_of[k] = cut.bus
        if cuts:
            self.ampl.get_parameter("nBUSCUT").set(self.n_bus_cuts)
        return len(cuts)

    def solve(self) -> Solution:
        t0 = time.time()
        self.ampl.solve()
        elapsed = time.time() - t0
        self._output.end_line()
        status = str(self.ampl.get_value("solve_result"))
        objective = float(self.ampl.get_objective("total_cost").value())
        phi = float(self.ampl.get_variable("Phi").value())
        weight = float(self.ampl.get_parameter("risk_weight").value())

        solution = Solution(status=status, objective=objective,
                            gen_cost=objective - weight * phi, phi=phi,
                            Pg=self._values("Pg"), Pf=self._values("Pf"),
                            theta=self._values("theta"), solve_time_s=elapsed)
        if self.is_ac:
            for name in ("Pt", "Qf", "Qt", "Qg", "v"):
                setattr(solution, name, self._values(name))
        self.log(f" {status}, cost {solution.gen_cost:.6f}, Phi {phi:.6f},"
                 f" {elapsed:.1f}s\n")
        return solution

    def _values(self, name: str) -> Dict[int, float]:
        raw = self.ampl.get_variable(name).get_values().to_dict()
        return {int(k): float(v) for k, v in raw.items()}

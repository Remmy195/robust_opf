"""Algorithm 1 on (M), and the hybrid stage a2: one (M^ac) solve with the DC cut pool."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import risk
from .model import Master, Solution, SolverConfig
from .network import Network

STAGES = ("baseline", "a2")
DC_MODFILE = "master.mod"
AC_MODFILE = "master_ac.mod"


def _noop(_message: str) -> None:
    return None


@dataclass
class AlgorithmConfig:
    metric: str = "max_active_flow"
    stage: str = "baseline"
    flow_domain: str = "all"
    q: int = 1
    weight_multiplier: float = 1.0
    risk_weight: Optional[float] = None
    kappa: int = 5
    eta: float = 0.85
    k_bar: int = 40
    gamma_tol: float = 1e-6

    def __post_init__(self) -> None:
        if self.metric not in risk.METRICS:
            raise ValueError(f"unknown exposure representation {self.metric!r}; "
                             f"expected one of {list(risk.METRICS)}")
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}; "
                             f"expected one of {list(STAGES)}")
        if self.flow_domain not in risk.FLOW_DOMAINS:
            raise ValueError(f"unknown flow domain {self.flow_domain!r}; "
                             f"expected one of {list(risk.FLOW_DOMAINS)}")
        if self.q < 1:
            raise ValueError(f"the order q must be at least 1, got {self.q}")
        if self.q > 1 and self.stage != "baseline":
            raise ValueError("order q > 1 runs on (M) only")
        if self.kappa < 1:
            raise ValueError(f"kappa must be at least 1, got {self.kappa}")
        if not 0.0 < self.eta < 1.0:
            raise ValueError(f"eta must lie in (0, 1), got {self.eta}")
        if self.k_bar < 1:
            raise ValueError(f"k_bar must be at least 1, got {self.k_bar}")
        if self.gamma_tol < 0.0:
            raise ValueError(f"gamma_tol cannot be negative, got {self.gamma_tol}")
        if self.weight_multiplier < 0 or (self.risk_weight or 0.0) < 0:
            raise ValueError("the exposure weight cannot be negative")


@dataclass
class Iterate:
    k: int
    status: str
    cost: float
    surrogate: float
    rho: float
    gamma: float
    ratio: float
    objective: float
    n_line_cuts: int
    n_bus_cuts: int
    cuts_added: int
    solve_time_s: float

    @property
    def solved(self) -> bool:
        return self.status == "solved"


@dataclass
class Result:
    case: str
    metric: str
    stage: str
    lambda_ref: float
    risk_weight: float
    weight_multiplier: Optional[float]
    z0: float
    rho0: float
    phi0: float
    flow_domain: str = "all"
    q: int = 1
    iterations: List[Iterate] = field(default_factory=list)
    ac: Optional[Iterate] = None
    termination: str = "iteration_limit"
    termination_detail: str = ""
    dispatch: Optional[Solution] = None
    total_time_s: float = 0.0
    config: Optional[AlgorithmConfig] = None

    @property
    def final(self) -> Iterate:
        return self.iterations[-1]

    @property
    def reported(self) -> Iterate:
        return self.ac if self.ac is not None else self.final

    @property
    def k_end(self) -> int:
        return self.final.k


def run(network: Network,
        config: AlgorithmConfig,
        solver: Optional[SolverConfig] = None,
        ac_solver: Optional[SolverConfig] = None,
        log: Optional[Callable[[str], None]] = None,
        lp_dir: Optional[str] = None) -> Result:
    """`lp_dir` writes each master solve there as an LP file."""
    emit = log or _noop
    started = time.time()
    emit(f"\n{config.stage}: {config.metric} (q = {config.q}), "
         f"lambda = {config.weight_multiplier:g} lambda_ref, kappa = "
         f"{config.kappa}, eta = {config.eta:g}, k_bar = {config.k_bar}\n")

    budget = config.kappa * config.k_bar
    master = Master(network, DC_MODFILE, solver or SolverConfig(), emit,
                    line_cut_capacity=budget, bus_cut_capacity=budget,
                    lp_dir=lp_dir)
    result = run_loop(network, master, config, emit)

    if config.stage == "a2" and result.termination != "infeasible":
        run_ac_stage(network, master, result, config,
                     ac_solver or SolverConfig(name="knitro"), emit, lp_dir)

    result.total_time_s = time.time() - started
    emit(f"{result.termination}: {result.termination_detail}"
         f" ({result.total_time_s:.1f}s)\n")
    return result


def run_loop(network: Network,
             master: Master,
             config: AlgorithmConfig,
             log: Optional[Callable[[str], None]] = None) -> Result:
    emit = log or _noop
    metric = config.metric
    master.set_risk_family(metric)
    master.set_order(config.q)

    # ALG1-L1
    emit("ALG1-L1  nominal solve\n")
    master.set_risk_weight(0.0)
    nominal = master.solve()

    result = Result(
        case=network.casefile or "?",
        metric=metric,
        stage=config.stage,
        flow_domain=config.flow_domain,
        q=config.q,
        lambda_ref=float("nan"),
        risk_weight=float("nan"),
        weight_multiplier=(None if config.risk_weight is not None
                           else config.weight_multiplier),
        z0=nominal.gen_cost,
        rho0=float("nan"),
        phi0=nominal.phi,
        dispatch=nominal,
        config=config,
    )

    if not nominal.solved:
        result.termination = "infeasible"
        result.termination_detail = f"nominal solve returned {nominal.status!r}"
        result.iterations.append(_iterate(0, nominal, float("nan"),
                                          float("nan"), master, 0))
        return result

    # ALG1-L2
    ev = risk.evaluate(metric, network, nominal.Pf, config.flow_domain, config.q)
    rho0 = result.rho0 = ev.value
    result.iterations.append(_iterate(0, nominal, rho0, rho0, master, 0))
    emit(f"ALG1-L2  z0 = {result.z0:.6f}, rho0 = {rho0:.6f}\n")

    if rho0 <= 0.0:
        result.risk_weight = 0.0
        result.termination = "nominal_risk_zero"
        result.termination_detail = f"rho0 = {rho0:g}"
        return result

    result.lambda_ref = result.z0 / rho0
    risk_weight = (float(config.risk_weight) if config.risk_weight is not None
                   else config.weight_multiplier * result.lambda_ref)
    result.risk_weight = risk_weight
    emit(f"lambda_ref = {result.lambda_ref:.6f}, lambda = {risk_weight:.6f}\n")

    if risk_weight == 0.0:
        # Phi is free at lambda = 0, so no cut can move the dispatch.
        result.termination = "zero_weight"
        result.termination_detail = "lambda = 0"
        return result

    master.set_risk_weight(risk_weight)
    incumbent = ev

    for k in range(1, config.k_bar + 1):  # ALG1-L3, L4
        # ALG1-L5
        selected = risk.select(
            incumbent, config.kappa,
            existing=(master.bus_cut_keys if incumbent.is_bus_family
                      else set(master.line_cut_ids)))

        # ALG1-L6
        if incumbent.is_bus_family:
            added = master.add_bus_cuts(risk.build_bus_cuts(incumbent, selected))
        else:
            added = master.add_line_cuts(selected)
        if added == 0:
            result.termination = "selection_exhausted"
            result.termination_detail = f"no new cut at k = {k}"
            break

        # ALG1-L7
        emit(f"ALG1-L7  k = {k}, {master.n_line_cuts} line + "
             f"{master.n_bus_cuts} bus cuts\n")
        solution = master.solve()
        if not solution.solved:
            result.termination = "infeasible"
            result.termination_detail = f"master returned {solution.status!r} at k = {k}"
            result.iterations.append(
                _iterate(k, solution, float("nan"), rho0, master, added))
            break

        # ALG1-L8
        ev_k = risk.evaluate(metric, network, solution.Pf, config.flow_domain,
                             config.q)
        iterate = _iterate(k, solution, ev_k.value, rho0, master, added)
        result.iterations.append(iterate)
        result.dispatch = solution
        incumbent = ev_k
        emit(f"ALG1-L8  rho = {iterate.rho:.6f}, Phi = {iterate.surrogate:.6f}, "
             f"Gamma = {iterate.gamma:.3e}, rho/rho0 = {iterate.ratio:.6f}\n")

        # ALG1-L9
        if iterate.ratio <= 1.0 - config.eta:
            result.termination = "eta_target"
            result.termination_detail = f"rho/rho0 = {iterate.ratio:.6f} at k = {k}"
            break
        if iterate.gamma <= config.gamma_tol:
            result.termination = "converged"
            result.termination_detail = f"Gamma = {iterate.gamma:.3e} at k = {k}"
            break
        if k == config.k_bar:
            result.termination = "iteration_limit"
            result.termination_detail = f"rho/rho0 = {iterate.ratio:.6f} at k = {k}"

    return result


def run_ac_stage(network: Network,
                 dc_master: Master,
                 result: Result,
                 config: AlgorithmConfig,
                 solver: SolverConfig,
                 log: Optional[Callable[[str], None]] = None,
                 lp_dir: Optional[str] = None) -> Iterate:
    """Append the DC pool to (M^ac) and solve once; with an empty pool this is the AC-OPF."""
    emit = log or _noop
    emit(f"AC stage  {dc_master.n_line_cuts} line + {dc_master.n_bus_cuts}"
         f" bus cuts to (M^ac)\n")
    ac = Master(network, AC_MODFILE, solver, emit,
                line_cut_capacity=max(1, dc_master.n_line_cuts),
                bus_cut_capacity=max(1, dc_master.n_bus_cuts),
                lp_dir=lp_dir)
    ac.set_risk_family(config.metric)
    ac.set_risk_weight(result.risk_weight)
    ac.add_line_cuts(dc_master.line_cut_ids)
    ac.add_bus_cuts(dc_master.bus_cuts)

    solution = ac.solve()
    ev = risk.evaluate(config.metric, network, solution.Pf, config.flow_domain,
                       config.q)
    result.ac = _iterate(result.k_end, solution, ev.value, result.rho0, ac, 0)
    if solution.solved:
        result.dispatch = solution
    else:
        result.termination_detail += f"; AC solve returned {solution.status!r}"
    return result.ac


def _iterate(k: int, solution: Solution, rho: float, rho0: float,
             master: Master, cuts_added: int) -> Iterate:
    return Iterate(k=k,
                   status=solution.status,
                   cost=solution.gen_cost,
                   surrogate=solution.phi,
                   rho=rho,
                   gamma=(rho - solution.phi) / rho if rho else float("nan"),
                   ratio=rho / rho0 if rho0 else float("nan"),
                   objective=solution.objective,
                   n_line_cuts=master.n_line_cuts,
                   n_bus_cuts=master.n_bus_cuts,
                   cuts_added=cuts_added,
                   solve_time_s=solution.solve_time_s)

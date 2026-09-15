"""The configuration file: one declared key table, and no silent acceptance.

A run is a plain text file of ``key = value`` lines, ``#`` for a comment.  Every
key is a field of `RunConfig`, with a type and a default; the file may set any
of them and nothing else.

An unknown key is an error, because a config that ignores what it does not
recognise turns a typo into a study that measured the wrong thing.  A duplicate
key is an error, because there is no defensible answer to which value was meant.
Values are converted by the field's declared type, so ``kappa = 2.5`` is an
error and not a silent truncation.
"""

from __future__ import annotations

import difflib
import os
import typing
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .algorithm import STAGES, WEIGHT_GRID, AlgorithmConfig, exchange_rate
from .model import SolverConfig
from .risk import EXPOSURE_FRACTION, FLOW_DOMAINS, METRICS

#: Where a bare case name is looked for, after the config file's own directory.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data")

#: Subdirectory of the output tree the per-iteration LP files go to.
LP_SUBDIR = "lp"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """The configuration file is wrong, and the run does not start."""


###############################################################################
# The key table
###############################################################################


@dataclass
class RunConfig:
    """Every key a configuration file may set.  The field list IS the key table,
    so a study parameter is documented, typed, defaulted and accepted at once."""

    # --- what to solve -------------------------------------------------------
    #: MATPOWER case file, resolved against the config's directory, then
    #: `DATA_DIR`, then the working directory.
    case: str = ""
    #: One of `ropf.risk.METRICS`.
    metric: str = "max_active_flow"
    #: One of `ropf.algorithm.STAGES`.
    stage: str = "baseline"
    #: One of `ropf.risk.FLOW_DOMAINS`: the component set the active risk
    #: functional is maximized over.  It restricts the surrogate ALONE -- the
    #: case file is not edited and the branches keep their flows and their
    #: eq (1d) limit.  See `ropf.risk.FLOW_DOMAINS` for how the restriction
    #: reads on the bus functional, which carries no rating of its own.
    flow_domain: str = "all"

    # --- Algorithm 1 ---------------------------------------------------------
    #: kappa, cuts appended per iteration.
    kappa: int = 1
    #: eta, the target reduction in the functional.
    eta: float = 0.5
    #: k-bar, the iteration limit.
    k_bar: int = 25
    #: |Gamma| at which the surrogate has converged on the functional.
    #: Zero, the default, disables the test.
    gamma_tol: float = 0.0
    #: The weight grid, as multiples of lambda* = z0/rho0.
    weight_grid: Tuple[float, ...] = WEIGHT_GRID
    #: An absolute lambda in $/p.u.  Setting it bypasses the grid AND lambda*,
    #: and produces a single run rather than a frontier.
    risk_weight: Optional[float] = None

    # --- eq (weightstar): the weight as an exchange rate ----------------------
    #     xi      + band  ->  one exchange rate, so one run
    #     xi_grid + band  ->  one rate per xi, so a frontier
    # The rate IS the multiplier of `weight_grid`, so a sweep stated this way
    # and the same sweep stated as multipliers are the same study.  Neither the
    # tolerance nor the band is defaulted: they state what the operator is
    # willing to pay, and a default would put a price nobody chose on record.
    #: xi, the cost tolerance, as a fraction of z0.
    xi: Optional[float] = None
    #: A grid of xi at a fixed band.  At a band of width 0.2, xi = 0, .05, .1,
    #: .2, .4, .8 gives exactly the multipliers 0, 1/4, 1/2, 1, 2, 4.
    xi_grid: Tuple[float, ...] = ()
    #: tau_lo, the bottom of the risk band, as a fraction of rho0.
    tau_lo: Optional[float] = None
    #: tau_hi, the top of the risk band, as a fraction of rho0.
    tau_hi: Optional[float] = None

    # --- the frontier tracer -------------------------------------------------
    # `ropf trace` only; nothing below is read by `ropf solve` or the ladder.
    #: The first non-zero multiplier the right-endpoint walk tries, doubling.
    frontier_seed: float = 0.25
    #: The largest multiplier the walk will run.  Reaching it is recorded as a
    #: capped walk, not a saturated one: the eta floor was never found.
    frontier_hi_cap: float = 64.0
    #: The total points -- Algorithm 1 runs -- one trace may solve, the
    #: endpoint walk included.
    frontier_max_points: int = 12
    #: Refinement stops when the longest chord in the normalized box falls
    #: below this.
    frontier_tol: float = 0.05
    #: The walk calls the right endpoint saturated when the drop in
    #: tau = rho^end/rho^0 between doublings falls below this.
    frontier_saturation_tol: float = 0.01
    #: K, the number of multipliers the trace recommends for `weight_grid`.
    frontier_grid_points: int = 6

    # --- solvers -------------------------------------------------------------
    #: Solver for (M) and for (D).
    solver_dc: str = "gurobi"
    #: Solver for (M^ac).
    solver_ac: str = "knitro"
    time_limit_s: float = 3600.0
    #: Set this to the PHYSICAL core count, not the logical one: a thread per
    #: logical cpu made one (D) evaluation on ACTIVSg70k 5.7x slower for the
    #: same answer.
    threads: int = 40
    #: Let the solver print its own log.  Off by default: at 70,000 buses it is
    #: larger than everything else the run writes together.
    solver_verbose: bool = False
    #: Gurobi LP method for (M)/(D): primal, dual, barrier, concurrent, auto.
    gurobi_method: Optional[str] = None
    gurobi_numericfocus: Optional[int] = None
    gurobi_scaleflag: Optional[int] = None

    # --- output --------------------------------------------------------------
    #: Directory for the three artifacts.  See `ropf.results`.
    outdir: str = "results"
    #: A label for this run, used in the output directory name.
    tag: str = ""
    #: A component is reported badly exposed at this fraction of rho^k.  See
    #: `ropf.risk.exposed`.
    exposure_fraction: float = EXPOSURE_FRACTION
    #: Write every master solve to ``<outdir>/lp/master_k<k>.lp``.  Off by
    #: default: it is one file per iteration per weight, and the DC solver has
    #: to be one that can write a problem file.
    write_lp: bool = False

    # -- validation -----------------------------------------------------------

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ConfigError(f"metric = {self.metric!r} is not one of "
                              f"{list(METRICS)}")
        if self.stage not in STAGES:
            raise ConfigError(f"stage = {self.stage!r} is not one of "
                              f"{list(STAGES)}")
        if self.flow_domain not in FLOW_DOMAINS:
            raise ConfigError(f"flow_domain = {self.flow_domain!r} is not one "
                              f"of {list(FLOW_DOMAINS)}")
        if not self.weight_grid:
            raise ConfigError("weight_grid is empty; there is nothing to solve")
        if any(w < 0 for w in self.weight_grid):
            raise ConfigError(f"weight_grid = {list(self.weight_grid)} carries a "
                              f"negative weight; lambda prices risk and cannot "
                              f"be negative")
        if self.threads < 1:
            raise ConfigError(f"threads = {self.threads} must be at least 1")
        if self.time_limit_s <= 0:
            raise ConfigError(f"time_limit_s = {self.time_limit_s} must be "
                              f"positive; every solve is given a stated budget")
        if not 0.0 < self.exposure_fraction <= 1.0:
            raise ConfigError(
                f"exposure_fraction = {self.exposure_fraction} is a fraction of "
                f"rho and must lie in (0, 1]")
        self._check_tracer()
        self._check_exchange_rate()
        # kappa, eta and k-bar are Algorithm 1's own and are checked by its
        # config, so the rule lives in one place.
        self.algorithm_config()

    def _check_exchange_rate(self) -> None:
        """The keys of eq (weightstar): a numerator, a denominator, or neither.

        A partial statement is an error, not a fallback: the only way to carry
        on is to invent the missing half, which would put a price nobody chose
        on record.
        """
        if self.xi is not None and self.xi_grid:
            raise ConfigError(
                "xi and xi_grid both set the cost tolerance, one as a single "
                "value and one as a sweep. Which applies has no defensible "
                "answer, so neither is taken; delete one.")

        numerator = "xi" if self.xi is not None else (
            "xi_grid" if self.xi_grid else None)
        band = {name for name in ("tau_lo", "tau_hi")
                if getattr(self, name) is not None}

        if numerator is None and not band:
            return
        if numerator is None:
            raise ConfigError(
                f"{', '.join(sorted(band))} sets the risk band, but neither xi "
                f"nor xi_grid sets the cost tolerance it divides. The band is "
                f"the denominator of eq (weightstar) and does not price risk "
                f"on its own.")
        if band != {"tau_lo", "tau_hi"}:
            raise ConfigError(
                f"{numerator} sets the cost tolerance, but the risk band is "
                f"{'incomplete' if band else 'missing'}: eq (weightstar) "
                f"divides by tau_hi - tau_lo and needs both ends. Set "
                f"tau_lo and tau_hi.")
        if self.risk_weight is not None:
            raise ConfigError(
                f"risk_weight and {numerator} both set the weight, one as an "
                f"absolute lambda in $/p.u. and the other as a multiple of "
                f"lambda*. Which applies has no defensible answer, so neither "
                f"is taken; delete one.")

        try:
            for value in ((self.xi,) if self.xi is not None else self.xi_grid):
                exchange_rate(value, self.tau_lo, self.tau_hi)
        except ValueError as exc:
            raise ConfigError(str(exc)) from None

    def _check_tracer(self) -> None:
        """The six `frontier_*` keys, checked when the file is read even when
        nothing traces -- a tracer parameter is a study parameter."""
        if self.frontier_seed < 1e-9:
            raise ConfigError(
                f"frontier_seed = {self.frontier_seed} must be at least 1e-9. "
                f"The trace already has lambda = 0 as its left endpoint, and a "
                f"seed that rounds to zero would make the walk re-read it.")
        if self.frontier_hi_cap < self.frontier_seed:
            raise ConfigError(
                f"frontier_hi_cap = {self.frontier_hi_cap} is below "
                f"frontier_seed = {self.frontier_seed}, so the walk would be "
                f"capped before it began")
        if self.frontier_max_points < 2:
            raise ConfigError(
                f"frontier_max_points = {self.frontier_max_points} must be at "
                f"least 2; the two endpoints are two points")
        if self.frontier_tol <= 0:
            raise ConfigError(
                f"frontier_tol = {self.frontier_tol} must be positive; a "
                f"tolerance of zero refines until the point budget runs out")
        if not 0.0 < self.frontier_saturation_tol < 1.0:
            raise ConfigError(
                f"frontier_saturation_tol = {self.frontier_saturation_tol} is "
                f"a drop in a ratio and must lie in (0, 1)")
        if self.frontier_grid_points < 2:
            raise ConfigError(
                f"frontier_grid_points = {self.frontier_grid_points} must be "
                f"at least 2; the recommendation always carries both endpoints")

    # -- derived objects ------------------------------------------------------

    def algorithm_config(self, weight_multiplier: float = 1.0,
                         stage: Optional[str] = None,
                         lp_dir: Optional[str] = None) -> AlgorithmConfig:
        """The Require line of Algorithm 1 at one point of the weight grid."""
        return AlgorithmConfig(metric=self.metric,
                               stage=stage or self.stage,
                               flow_domain=self.flow_domain,
                               weight_multiplier=weight_multiplier,
                               risk_weight=self.risk_weight,
                               kappa=self.kappa,
                               eta=self.eta,
                               k_bar=self.k_bar,
                               gamma_tol=self.gamma_tol,
                               exposure_fraction=self.exposure_fraction,
                               lp_dir=lp_dir)

    def lp_dir(self, outdir: str, weight_multiplier: float) -> Optional[str]:
        """Where one weight's LP files go, or None when `write_lp` is off.

        One directory per weight: the files are labelled by iteration, so two
        weights sharing a directory would overwrite each other's k.
        """
        if not self.write_lp:
            return None
        return os.path.join(outdir, LP_SUBDIR, f"w{weight_multiplier:g}")

    def dc_solver(self) -> SolverConfig:
        return SolverConfig(name=self.solver_dc,
                            time_limit_s=self.time_limit_s,
                            knitro_threads=self.threads,
                            gurobi_threads=self.threads,
                            gurobi_method=self.gurobi_method,
                            gurobi_numericfocus=self.gurobi_numericfocus,
                            gurobi_scaleflag=self.gurobi_scaleflag,
                            verbose=self.solver_verbose)

    def ac_solver(self) -> SolverConfig:
        return SolverConfig(name=self.solver_ac,
                            time_limit_s=self.time_limit_s,
                            knitro_threads=self.threads,
                            gurobi_threads=self.threads,
                            verbose=self.solver_verbose)

    @property
    def exchange_rate(self) -> Optional[float]:
        """xi/(tau_hi - tau_lo), or None when a single xi is not set."""
        if self.xi is None:
            return None
        return exchange_rate(self.xi, self.tau_lo, self.tau_hi)

    @property
    def exchange_rates(self) -> Tuple[float, ...]:
        """One rate per xi in `xi_grid`, or () when that grid is not set."""
        if not self.xi_grid:
            return ()
        return tuple(exchange_rate(x, self.tau_lo, self.tau_hi)
                     for x in self.xi_grid)

    @property
    def weights(self) -> Tuple[float, ...]:
        """The grid this run sweeps.  A single point when the weight is pinned:
        an absolute lambda bypasses lambda* and reports as a multiplier of 1,
        while an (xi, band) triple IS a multiplier."""
        if self.risk_weight is not None:
            return (1.0,)
        rate = self.exchange_rate
        if rate is not None:
            return (rate,)
        return self.exchange_rates or tuple(self.weight_grid)

    def as_dict(self) -> Dict[str, Any]:
        """Every key and its effective value, for the run's own record."""
        record = asdict(self)
        record["weight_grid"] = list(self.weight_grid)
        return record


#: The declared keys, in declaration order.
KEYS = tuple(f.name for f in fields(RunConfig))


###############################################################################
# Parsing
###############################################################################


def parse_keyfile(path: str, cls: type) -> Dict[str, Any]:
    """Parse ``key = value`` lines into keyword arguments for `cls`.

    The two refusals live here and nowhere else, so a run config and a ladder
    config reject an unknown key and a duplicate key on the same terms.  `cls`
    is any dataclass whose field list is its key table.
    """
    keys = tuple(f.name for f in fields(cls))
    try:
        with open(path, "r") as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise ConfigError(f"cannot open config file {path}: {exc}") from exc

    values, seen = {}, {}
    for lineno, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ConfigError(
                f"{path}:{lineno}: expected 'key = value', got {line!r}")

        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()

        if key not in keys:
            raise ConfigError(f"{path}:{lineno}: unknown key {key!r}"
                              + _suggest(key, keys))
        if key in seen:
            raise ConfigError(
                f"{path}:{lineno}: {key!r} is already set on line {seen[key]}. "
                f"Which of the two values applies has no defensible answer, so "
                f"neither is taken; delete one.")

        seen[key] = lineno
        values[key] = _convert(key, value, path, lineno, cls)

    return values


def read_config(path: str) -> RunConfig:
    """Parse a configuration file into a `RunConfig`."""
    config = _build(parse_keyfile(path, RunConfig), path, RunConfig)
    if config.case:
        config.case = resolve_case(config.case,
                                   os.path.dirname(os.path.abspath(path)))
    return config


def _suggest(key: str, keys: Sequence[str] = None) -> str:
    keys = tuple(keys) if keys is not None else KEYS
    close = difflib.get_close_matches(key, keys, n=3, cutoff=0.6)
    if close:
        return f"; did you mean {', '.join(repr(c) for c in close)}?"
    return f". Known keys: {', '.join(keys)}"


def _build(values: Dict[str, Any], path: str, cls: type = RunConfig) -> Any:
    try:
        return cls(**values)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _convert(key: str, value: str, path: str, lineno: int,
             cls: type = RunConfig) -> Any:
    """Coerce one value to the declared type of its field."""
    declared = typing.get_type_hints(cls)[key]
    where = f"{path}:{lineno}: {key} = {value!r}"

    origin = typing.get_origin(declared)
    args = typing.get_args(declared)

    # Optional[X] -- an explicit "none" clears it back to the default of None.
    if origin is typing.Union and type(None) in args:
        if value.lower() in ("none", ""):
            return None
        declared = next(a for a in args if a is not type(None))
        origin, args = typing.get_origin(declared), typing.get_args(declared)

    if origin is tuple:
        item = args[0] if args else float
        parts = [p.strip() for p in value.replace(",", " ").split()]
        # An empty value is an empty list, not an error: writing no values is
        # how a study axis is switched off.  The keys for which nothing is not
        # a legal answer refuse it in their own __post_init__.
        return tuple(_scalar(item, p, where) for p in parts)

    return _scalar(declared, value, where)


def _scalar(declared: type, value: str, where: str) -> Any:
    if declared is bool:
        low = value.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ConfigError(f"{where}: expected a boolean "
                          f"({'/'.join(sorted(_TRUE | _FALSE))})")
    if declared is int:
        try:
            return int(value)
        except ValueError:
            raise ConfigError(
                f"{where}: expected an integer. A value like 2.5 is rejected "
                f"rather than truncated.") from None
    if declared is float:
        try:
            return float(value)
        except ValueError:
            raise ConfigError(f"{where}: expected a number") from None
    return value


###############################################################################
# Case files
###############################################################################


def resolve_case(case: str, near: Optional[str] = None) -> str:
    """Find a case file named absolutely, relatively, or bare.

    Searched as given, beside the config file, in `DATA_DIR`, and with a ``.m``
    appended in each.  Not found raises with the full list of what was tried.
    """
    candidates: List[str] = []
    stems = [case] if case.endswith(".m") else [case, case + ".m"]
    for stem in stems:
        if os.path.isabs(stem):
            candidates.append(stem)
            continue
        if near:
            candidates.append(os.path.join(near, stem))
        candidates.append(os.path.join(DATA_DIR, stem))
        candidates.append(os.path.abspath(stem))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    tried = "\n  ".join(candidates)
    raise ConfigError(f"case file {case!r} not found. Tried:\n  {tried}\n"
                      f"The ACTIVSg cases ship in data/.")

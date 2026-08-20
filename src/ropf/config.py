"""The configuration file: one declared key table, and no silent acceptance.

A run is described by a plain text file of ``key = value`` lines, ``#`` for a
comment.  Every key this study understands is a field of `RunConfig` below, with
a type and a default; the file may set any of them and nothing else.

TWO KINDS OF MISTAKE ARE ERRORS HERE, NOT WARNINGS.

An UNKNOWN key is an error because a config that silently ignores what it does
not recognise turns a typo into a study that measured something other than what
was asked for, and leaves no trace of it in the output.  ``metrik = joule`` must
stop the run, not fall back to a default.

A DUPLICATE key is an error because there is no defensible answer to which of
the two values was meant.  Last-one-wins is the usual choice and it is the worst
one: a file that sets ``k_bar = 25`` at the top and ``k_bar = 3`` sixty lines
down looks, at the point a reader checks it, like it does what it says.

Values are converted by the field's declared type, so ``kappa = 2.5`` is an
error and not a silent truncation to 2.
"""

from __future__ import annotations

import difflib
import os
import typing
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .algorithm import STAGES, WEIGHT_GRID, AlgorithmConfig
from .model import SolverConfig
from .risk import METRICS

#: Where a bare case name is looked for, after the config file's own directory.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """The configuration file is wrong, and the run does not start."""


###############################################################################
# The key table
###############################################################################


@dataclass
class RunConfig:
    """Every key a configuration file may set.  The field list IS the key table.

    Adding a study parameter means adding a field here, and it is then
    documented, typed, defaulted and accepted by the parser at once.  There is
    no second place to register it and therefore no way for the two to drift.
    """

    # --- what to solve -------------------------------------------------------
    #: MATPOWER case file.  Resolved against the config file's directory, then
    #: `DATA_DIR`, then the working directory.
    case: str = ""
    #: One of `ropf.risk.METRICS`.
    metric: str = "max_active_flow"
    #: One of `ropf.algorithm.STAGES`.
    stage: str = "baseline"

    # --- Algorithm 1 ---------------------------------------------------------
    #: kappa, cuts appended per iteration.
    kappa: int = 1
    #: eta, the target reduction in the functional.
    eta: float = 0.5
    #: k-bar, the iteration limit.
    k_bar: int = 25
    #: The weight grid, as multiples of lambda* = z0/rho0.
    weight_grid: Tuple[float, ...] = WEIGHT_GRID
    #: An absolute lambda in $/p.u.  Setting it bypasses the grid AND lambda*,
    #: and produces a single run rather than a frontier.
    risk_weight: Optional[float] = None

    # --- solvers -------------------------------------------------------------
    #: Solver for (M) and for (D).
    solver_dc: str = "gurobi"
    #: Solver for (M^ac).
    solver_ac: str = "knitro"
    time_limit_s: float = 3600.0
    threads: int = 40
    #: Let the solver print its own log.  Off by default: at 70,000 buses the
    #: solver's output is larger than everything else the run writes together.
    solver_verbose: bool = False

    # --- output --------------------------------------------------------------
    #: Directory for the three artifacts.  See `ropf.results`.
    outdir: str = "results"
    #: A label for this run, used in the output directory name.
    tag: str = ""

    # -- validation -----------------------------------------------------------

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ConfigError(f"metric = {self.metric!r} is not one of "
                              f"{list(METRICS)}")
        if self.stage not in STAGES:
            raise ConfigError(f"stage = {self.stage!r} is not one of "
                              f"{list(STAGES)}")
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
        # kappa, eta and k-bar are Algorithm 1's own, and are checked by its
        # config so that the rule lives in one place.
        self.algorithm_config()

    # -- derived objects ------------------------------------------------------

    def algorithm_config(self, weight_multiplier: float = 1.0,
                         stage: Optional[str] = None) -> AlgorithmConfig:
        """The Require line of Algorithm 1 at one point of the weight grid."""
        return AlgorithmConfig(metric=self.metric,
                               stage=stage or self.stage,
                               weight_multiplier=weight_multiplier,
                               risk_weight=self.risk_weight,
                               kappa=self.kappa,
                               eta=self.eta,
                               k_bar=self.k_bar)

    def dc_solver(self) -> SolverConfig:
        return SolverConfig(name=self.solver_dc,
                            time_limit_s=self.time_limit_s,
                            knitro_threads=self.threads,
                            verbose=self.solver_verbose)

    def ac_solver(self) -> SolverConfig:
        return SolverConfig(name=self.solver_ac,
                            time_limit_s=self.time_limit_s,
                            knitro_threads=self.threads,
                            verbose=self.solver_verbose)

    @property
    def weights(self) -> Tuple[float, ...]:
        """The grid this run sweeps.  An absolute lambda is a single point."""
        if self.risk_weight is not None:
            return (1.0,)
        return tuple(self.weight_grid)

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


def read_config(path: str) -> RunConfig:
    """Parse a configuration file into a `RunConfig`.

    Raises `ConfigError` on an unknown key, a duplicate key, a line that is not
    ``key = value``, or a value the field's type will not accept.
    """
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

        if key not in KEYS:
            raise ConfigError(f"{path}:{lineno}: unknown key {key!r}"
                              + _suggest(key))
        if key in seen:
            raise ConfigError(
                f"{path}:{lineno}: {key!r} is already set on line {seen[key]}. "
                f"Which of the two values applies has no defensible answer, so "
                f"neither is taken; delete one.")

        seen[key] = lineno
        values[key] = _convert(key, value, path, lineno)

    config = _build(values, path)
    if config.case:
        config.case = resolve_case(config.case, os.path.dirname(
            os.path.abspath(path)))
    return config


def _suggest(key: str) -> str:
    close = difflib.get_close_matches(key, KEYS, n=3, cutoff=0.6)
    if close:
        return f"; did you mean {', '.join(repr(c) for c in close)}?"
    return f". Known keys: {', '.join(KEYS)}"


def _build(values: Dict[str, Any], path: str) -> RunConfig:
    try:
        return RunConfig(**values)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _convert(key: str, value: str, path: str, lineno: int) -> Any:
    """Coerce one value to the declared type of its field."""
    hints = typing.get_type_hints(RunConfig)
    declared = hints[key]
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
        if not parts:
            raise ConfigError(f"{where}: expected a list of values")
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

    Searched in order: as given, beside the config file, in `DATA_DIR`, and
    with a ``.m`` suffix appended in each of those.  A case that is not found
    raises with the full list of what was tried, since "no such file" on a bare
    name like ``ACTIVSg2000`` says nothing about where it was looked for.
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
                      f"Run data/fetch.py to download the ACTIVSg cases.")

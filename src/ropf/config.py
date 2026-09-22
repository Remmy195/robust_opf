"""Run configuration: `key = value` lines, `#` comments. Unknown or repeated keys are errors."""

from __future__ import annotations

import difflib
import os
import typing
from dataclasses import dataclass
from typing import Any, Optional

from .algorithm import AlgorithmConfig
from .model import SolverConfig

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data")

_BOOL = {"1": True, "true": True, "yes": True, "on": True,
         "0": False, "false": False, "no": False, "off": False}


class ConfigError(ValueError):
    pass


@dataclass
class RunConfig:
    case: str = ""
    metric: str = "max_active_flow"
    stage: str = "baseline"
    flow_domain: str = "all"
    q: int = 1
    weight_multiplier: float = 1.0
    risk_weight: Optional[float] = None
    kappa: int = 1
    eta: float = 0.5
    k_bar: int = 25
    gamma_tol: float = 0.0
    solver_dc: str = "gurobi"
    solver_ac: str = "knitro"
    solver_dc_options: str = ""
    solver_ac_options: str = ""
    time_limit_s: float = 3600.0
    threads: Optional[int] = None
    solver_verbose: bool = False
    write_lp: bool = False
    outdir: str = ""

    def __post_init__(self) -> None:
        if self.time_limit_s <= 0:
            raise ConfigError(f"time_limit_s must be positive, got {self.time_limit_s}")
        if self.threads is not None and self.threads < 1:
            raise ConfigError(f"threads must be at least 1, got {self.threads}")
        try:
            self.algorithm_config()
        except ValueError as exc:
            raise ConfigError(str(exc)) from None

    def algorithm_config(self) -> AlgorithmConfig:
        return AlgorithmConfig(metric=self.metric, stage=self.stage,
                               flow_domain=self.flow_domain, q=self.q,
                               weight_multiplier=self.weight_multiplier,
                               risk_weight=self.risk_weight, kappa=self.kappa,
                               eta=self.eta, k_bar=self.k_bar,
                               gamma_tol=self.gamma_tol)

    def dc_solver(self) -> SolverConfig:
        return SolverConfig(self.solver_dc, self.time_limit_s, self.threads,
                            self.solver_dc_options, self.solver_verbose)

    def ac_solver(self) -> SolverConfig:
        return SolverConfig(self.solver_ac, self.time_limit_s, self.threads,
                            self.solver_ac_options, self.solver_verbose)


def read_config(path: str) -> RunConfig:
    hints = typing.get_type_hints(RunConfig)
    try:
        with open(path) as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise ConfigError(f"cannot open config file {path}: {exc}") from exc

    values, seen = {}, {}
    for lineno, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{path}:{lineno}"
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep:
            raise ConfigError(f"{where}: expected 'key = value', got {line!r}")
        if key not in hints:
            close = difflib.get_close_matches(key, list(hints), n=3, cutoff=0.6)
            hint = (f"; did you mean {', '.join(map(repr, close))}?" if close
                    else f". Known keys: {', '.join(hints)}")
            raise ConfigError(f"{where}: unknown key {key!r}{hint}")
        if key in seen:
            raise ConfigError(f"{where}: {key!r} is already set on line {seen[key]}")
        seen[key] = lineno
        values[key] = _convert(hints[key], value, f"{where}: {key}")

    config = RunConfig(**values)
    if config.case:
        config.case = resolve_case(config.case, os.path.dirname(os.path.abspath(path)))
    return config


def _convert(declared: Any, value: str, where: str) -> Any:
    if typing.get_origin(declared) is typing.Union:
        if value.lower() in ("", "none"):
            return None
        declared = next(a for a in typing.get_args(declared) if a is not type(None))
    try:
        if declared is bool:
            return _BOOL[value.lower()]
        return declared(value)
    except (KeyError, ValueError):
        raise ConfigError(f"{where}: expected {declared.__name__}, got {value!r}") from None


def resolve_case(case: str, near: Optional[str] = None) -> str:
    """As given, beside the config, in data/, then the working directory; `.m` optional."""
    candidates = []
    for stem in ([case] if case.endswith(".m") else [case, case + ".m"]):
        if os.path.isabs(stem):
            candidates.append(stem)
            continue
        if near:
            candidates.append(os.path.join(near, stem))
        candidates += [os.path.join(DATA_DIR, stem), os.path.abspath(stem)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise ConfigError(f"case file {case!r} not found. Tried:\n  " + "\n  ".join(candidates))

"""`ropf CONFIG`: one run of Algorithm 1, written to result.json and run.log."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import amplpy

from . import __version__, algorithm
from .config import ConfigError, RunConfig, read_config
from .network import CaseFormatError, Network, read_matpower

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Log:
    def __init__(self, path: str, echo: bool = True):
        self.echo = echo
        self._handle = open(path, "w", buffering=1)

    def __call__(self, message: str) -> None:
        if self.echo:
            sys.stdout.write(message)
            sys.stdout.flush()
        self._handle.write(message)

    def __enter__(self) -> "Log":
        return self

    def __exit__(self, *_exc) -> None:
        self._handle.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ropf", description="Risk-aware optimal power flow by cutting planes.")
    parser.add_argument("config", help="configuration file")
    parser.add_argument("--quiet", action="store_true", help="log to run.log only")
    parser.add_argument("--version", action="version", version=f"ropf {__version__}")
    args = parser.parse_args(argv)
    try:
        return solve(args.config, echo=not args.quiet)
    except (ConfigError, CaseFormatError, FileNotFoundError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


def output_dir(config: RunConfig) -> str:
    if config.outdir:
        return config.outdir
    case = os.path.splitext(os.path.basename(config.case))[0]
    order = f"_q{config.q}" if config.q > 1 else ""
    return os.path.join("results", f"{case}_{config.metric}{order}_{config.stage}")


def solve(path: str, echo: bool = True) -> int:
    config = read_config(path)
    if not config.case:
        raise ConfigError(f"{path}: no case given")
    outdir = output_dir(config)
    os.makedirs(outdir, exist_ok=True)
    started = time.time()
    with Log(os.path.join(outdir, "run.log"), echo) as log:
        network = read_matpower(config.case)
        meta = metadata(path, network, outdir)
        log_header(log, meta, config)
        result = algorithm.run(network, config.algorithm_config(),
                               config.dc_solver(), config.ac_solver(), log,
                               os.path.join(outdir, "lp") if config.write_lp else None)
        meta["finished"] = _now()
        meta["wall_s"] = round(time.time() - started, 3)
        meta["solver_s"] = round(sum(it.solve_time_s for it in result.iterations)
                                 + (result.ac.solve_time_s if result.ac else 0.0), 3)
        log_summary(log, meta, result)
        with open(os.path.join(outdir, "result.json"), "w") as handle:
            json.dump({"metadata": meta, "config": asdict(config),
                       "result": asdict(result)}, handle, indent=1)
        with open(os.path.join(outdir, "result.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(result.iterations[0])))
            writer.writeheader()
            writer.writerows(asdict(it) for it in result.iterations)
        if result.dispatch is not None:
            write_dispatch_csvs(outdir, network, result.dispatch)
    return 1 if result.termination == "infeasible" else 0


def write_dispatch_csvs(outdir: str, network: Network, solution) -> None:
    with open(os.path.join(outdir, "dispatch_bus.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bus", "Pd", "Qd", "theta", "v"])
        for i, bus in network.buses.items():
            writer.writerow([bus.nodeID, bus.Pd, bus.Qd,
                             solution.theta.get(i, ""), solution.v.get(i, "")])
    with open(os.path.join(outdir, "dispatch_branch.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["branch", "from", "to", "rated", "limit", "Pf", "Pt", "Qf", "Qt"])
        for i, br in network.branches.items():
            writer.writerow([i, br.id_f, br.id_t, br.constrainedflow, br.limit,
                             solution.Pf.get(i, ""), solution.Pt.get(i, ""),
                             solution.Qf.get(i, ""), solution.Qt.get(i, "")])
    with open(os.path.join(outdir, "dispatch_gen.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gen", "bus", "Pmin", "Pmax", "Qmin", "Qmax", "Pg", "Qg"])
        for i, gen in network.gens.items():
            writer.writerow([i, gen.nodeID, gen.Pmin, gen.Pmax, gen.Qmin, gen.Qmax,
                             solution.Pg.get(i, ""), solution.Qg.get(i, "")])


def metadata(path: str, network: Network, outdir: str) -> Dict[str, Any]:
    with open(network.casefile, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {
        "ropf": __version__,
        "git": _git_commit(),
        "started": _now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "amplpy": amplpy.__version__,
        "config_file": os.path.abspath(path),
        "case_file": os.path.abspath(network.casefile),
        "case_sha256": digest,
        "buses": network.numbuses,
        "branches": network.numbranches,
        "unrated_branches": len(network.unconstrained_branches()),
        "gens": network.numgens,
        "gens_online": sum(g.status for g in network.gens.values()),
        "baseMVA": network.baseMVA,
        "load_MW": network.sumPd,
        "outdir": os.path.abspath(outdir),
    }


def log_header(log, meta: Dict[str, Any], config: RunConfig) -> None:
    git = f", git {meta['git']}" if meta["git"] else ""
    log(f"ropf {meta['ropf']}{git}\n"
        f"started  {meta['started']}\n"
        f"host     {meta['host']}, {meta['platform']}\n"
        f"python   {meta['python']}, amplpy {meta['amplpy']}\n"
        f"config   {meta['config_file']}\n"
        f"case     {meta['case_file']}\n"
        f"sha256   {meta['case_sha256']}\n"
        f"network  {meta['buses']} buses, {meta['branches']} branches"
        f" ({meta['unrated_branches']} unrated), {meta['gens']} gens"
        f" ({meta['gens_online']} online), load {meta['load_MW']:.2f} MW,"
        f" base {meta['baseMVA']:g} MVA\n"
        f"output   {meta['outdir']}\n\n")
    for key, value in asdict(config).items():
        log(f"  {key} = {'none' if value is None else value}\n")


def log_summary(log, meta: Dict[str, Any], result: algorithm.Result) -> None:
    final = result.final
    log(f"\ncost     {_change(result.z0, final.cost)}\n"
        f"exposure {_change(result.rho0, final.rho)}\n"
        f"cuts     {final.n_line_cuts} line + {final.n_bus_cuts} bus at k = {final.k}\n")
    if result.ac is not None:
        log(f"AC       {result.ac.status}, cost {result.ac.cost:.6f},"
            f" exposure {result.ac.rho:.6f}\n")
    log(f"finished {meta['finished']}, wall {meta['wall_s']:.1f}s,"
        f" solver {meta['solver_s']:.1f}s\n")


def _change(before: float, after: float) -> str:
    text = f"{before:.6f} -> {after:.6f}"
    if math.isfinite(before) and math.isfinite(after) and before:
        text += f" ({(after / before - 1) * 100:+.2f}%)"
    return text


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_commit() -> Optional[str]:
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        return None
    try:
        out = subprocess.run(["git", "-C", ROOT, "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None

"""The command line: ``ropf <command>``.

    ropf solve CONFIG     sweep the weight grid for one case, metric and stage,
                          and write the three artifacts
    ropf keys             print the configuration key table
    ropf fetch [CASE...]  download the ACTIVSg cases into data/

THE CONFIG FILE IS THE ONLY PLACE A STUDY PARAMETER IS SET.  No command-line
flag overrides a value in it.  The two flags that exist -- ``--quiet`` and
``--dry-run`` -- change what is printed and whether anything is solved, and
neither can change a number.  An override would mean the same config file
producing two different studies depending on how it was invoked, and the output
directory could no longer be read as a record of what was run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import typing
from dataclasses import fields
from typing import List, Optional, Sequence

from . import __version__
from .config import ConfigError, RunConfig, read_config
from .log import Log

#: The run transcript.  Written beside the three artifacts; it is a transcript,
#: not a result, and nothing reads it back.
TRANSCRIPT = "run.log"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except ConfigError as exc:
        sys.stderr.write(f"config error: {exc}\n")
        return 2
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ropf",
        description="Risk-aware optimal power flow by cutting planes.")
    parser.add_argument("--version", action="version",
                        version=f"ropf {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    solve = sub.add_parser(
        "solve", help="sweep the weight grid for one case, metric and stage",
        description=("Run Algorithm 1 at every point of the configuration's "
                     "weight grid and write solution_summary.json, "
                     "efficient_frontier.csv and efficient_frontier.json."))
    solve.add_argument("config", metavar="CONFIG",
                       help="configuration file; see `ropf keys`")
    solve.add_argument("--quiet", action="store_true",
                       help="write the transcript to the file only")
    solve.add_argument("--dry-run", action="store_true",
                       help="report the effective configuration and stop")
    solve.set_defaults(handler=_solve)

    keys = sub.add_parser("keys", help="print the configuration key table",
                          description="Every key a configuration file may set.")
    keys.set_defaults(handler=_keys)

    fetch = sub.add_parser(
        "fetch", help="download the ACTIVSg cases into data/",
        description="Download and unpack the test systems the study uses.")
    fetch.add_argument("cases", metavar="CASE", nargs="*",
                       help="rung names, e.g. activs200; default is all of them")
    fetch.add_argument("--list", action="store_true",
                       help="list the known cases and stop")
    fetch.set_defaults(handler=_fetch)

    return parser


###############################################################################
# ropf solve
###############################################################################


def _output_dir(config: RunConfig) -> str:
    """``outdir/tag``, or ``outdir/<case>_<metric>_<stage>`` when no tag is set."""
    if config.tag:
        return os.path.join(config.outdir, config.tag)
    case = os.path.splitext(os.path.basename(config.case))[0] or "case"
    return os.path.join(config.outdir, f"{case}_{config.metric}_{config.stage}")


def _solve(args: argparse.Namespace) -> int:
    from . import algorithm, results
    from .network import read_matpower

    config = read_config(args.config)
    if not config.case:
        raise ConfigError(f"{args.config}: no `case` given; there is nothing "
                          f"to solve")

    outdir = _output_dir(config)
    if args.dry_run:
        _report_config(config, outdir, sys.stdout.write)
        return 0

    os.makedirs(outdir, exist_ok=True)
    with Log(os.path.join(outdir, TRANSCRIPT), echo=not args.quiet) as log:
        log.section(f"ropf {__version__}: {os.path.basename(args.config)}")
        _report_config(config, outdir, log)

        network = read_matpower(config.case, log)
        dc_solver, ac_solver = config.dc_solver(), config.ac_solver()

        runs = []
        for multiplier in config.weights:
            log.section(f"lambda = {multiplier:g} lambda*"
                        f"   [{config.metric}, stage {config.stage}]")
            runs.append(algorithm.run(
                network, config.algorithm_config(multiplier),
                dc_solver, ac_solver, log))

        log.section("results")
        results.write(outdir, network, config, runs, log)
        log(f"\n total wall clock {log.elapsed_s:.1f}s\n")

        failed = [r for r in runs if r.termination == "infeasible"]
        if failed:
            log(f" {len(failed)} of {len(runs)} weights did not solve\n")
            return 1
    return 0


def _report_config(config: RunConfig, outdir: str, emit) -> None:
    emit(f"\n case          {config.case}\n")
    emit(f" metric        {config.metric}\n")
    emit(f" stage         {config.stage}\n")
    emit(f" weights       {', '.join(f'{w:g}' for w in config.weights)}"
         f" x lambda*\n")
    if config.risk_weight is not None:
        emit(f" risk_weight   {config.risk_weight:g} $/p.u. (absolute; "
             f"lambda* is bypassed)\n")
    emit(f" kappa         {config.kappa}\n")
    emit(f" eta           {config.eta:g}\n")
    emit(f" k-bar         {config.k_bar}\n")
    emit(f" solvers       {config.solver_dc} (DC), {config.solver_ac} (AC),"
         f" {config.time_limit_s:g}s each\n")
    emit(f" output        {outdir}\n")


###############################################################################
# ropf keys
###############################################################################


def _keys(_args: argparse.Namespace) -> int:
    hints = typing.get_type_hints(RunConfig)
    defaults = RunConfig()
    width = max(len(f.name) for f in fields(RunConfig))
    print("Configuration keys.  A file sets `key = value`, one per line, "
          "`#` for a comment.")
    print("An unknown key and a repeated key are both errors.\n")
    for field in fields(RunConfig):
        value = getattr(defaults, field.name)
        if isinstance(value, tuple):
            value = ", ".join(f"{v:g}" for v in value)
        shown = "" if value in ("", None) else value
        print(f"  {field.name:<{width}}  {_typename(hints[field.name]):<16}"
              f"  default: {shown}")
    return 0


def _typename(declared) -> str:
    origin = typing.get_origin(declared)
    args = typing.get_args(declared)
    if origin is typing.Union and type(None) in args:
        inner = next(a for a in args if a is not type(None))
        return f"{_typename(inner)} or none"
    if origin is tuple:
        return "list of numbers"
    return {int: "integer", float: "number", bool: "true/false",
            str: "text"}.get(declared, getattr(declared, "__name__", "text"))


###############################################################################
# ropf fetch
###############################################################################


def _fetch(args: argparse.Namespace) -> int:
    from .data import fetch

    if args.list:
        for name, source in sorted(fetch.SOURCES.items()):
            print(f"  {name:<12}  {source.description}")
        return 0
    return fetch.fetch(args.cases or None, log=sys.stdout.write)


if __name__ == "__main__":
    raise SystemExit(main())

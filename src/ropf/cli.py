"""The command line: ``ropf <command>``.

    ropf solve CONFIG     sweep the weight grid for one case, metric and stage,
                          and write the three artifacts
    ropf ladder CONFIG    run the ladder study, one combo per invocation
    ropf keys             print the configuration key table
    ropf fetch [CASE...]  download the ACTIVSg cases into data/

THE CONFIG FILE IS THE ONLY PLACE A STUDY PARAMETER IS SET.  No command-line
flag overrides a value in it.  The flags that exist change what is printed
(``--quiet``, ``--status``), whether anything is solved (``--dry-run``), which
of the declared work is done now (``--all``, ``--reclaim``) -- and none of them
can change a number.  An override would mean the same config file producing two
different studies depending on how it was invoked, and the output directory
could no longer be read as a record of what was run.

``--reclaim`` is the case worth naming: it acts on claims older than
``stale_claim_hours``, and that threshold is a number, so it lives in the config
file.  The flag decides whether to act, not what the threshold is.
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

    ladder = sub.add_parser(
        "ladder", help="run the ladder study, one combo per invocation",
        description=("Claim one (rung, metric, stage) combo, run its frontier "
                     "and its Section 4 campaign, and exit. Run it again -- or "
                     "in several shells at once -- to work through the rest."))
    ladder.add_argument("config", metavar="CONFIG",
                        help="ladder configuration file")
    ladder.add_argument("--all", action="store_true",
                        help="keep claiming combos until none are left")
    ladder.add_argument("--status", action="store_true",
                        help="print the progress table and stop")
    ladder.add_argument("--dry-run", action="store_true",
                        help="report the combos and the campaign size, and "
                             "solve nothing")
    ladder.add_argument("--reclaim", action="store_true",
                        help="release claims older than stale_claim_hours "
                             "before starting")
    ladder.add_argument("--quiet", action="store_true",
                        help="write the transcript to the file only")
    ladder.set_defaults(handler=_ladder)

    fetch = sub.add_parser(
        "fetch", help="download the ACTIVSg cases into data/",
        description="Download and unpack the test systems the study uses.")
    fetch.add_argument("cases", metavar="CASE", nargs="*",
                       help="rung names, e.g. activs200; default is all of them")
    fetch.add_argument("--list", action="store_true",
                       help="list the known cases and stop")
    fetch.add_argument("--from", dest="source_dir", metavar="DIR",
                       help="take the cases AND their dynamics from local "
                            "ACTIVSg distributions already downloaded into DIR")
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
# ropf ladder
###############################################################################


def _ladder(args: argparse.Namespace) -> int:
    from .study import ladder as ladder_module

    config = ladder_module.read_ladder_config(args.config)
    todo = ladder_module.combos(config)

    if args.status:
        rows = ladder_module.status_rows(config)
        width = max(len(r["combo"]) for r in rows)
        for row in rows:
            age = ("" if row["claim_age_h"] is None
                   else f"  claimed {row['claim_age_h']:.1f}h ago")
            print(f"  {row['combo']:<{width}}  {row['state']:<8}{age}")
        tally = {}
        for row in rows:
            tally[row["state"]] = tally.get(row["state"], 0) + 1
        print("\n  " + ", ".join(f"{n} {state}"
                                 for state, n in sorted(tally.items())))
        return 0

    if args.dry_run:
        per = config.evaluations_per_combo()
        print(f"  {len(todo)} combos: "
              f"{len(config.rungs)} rungs x {len(config.metrics)} metrics "
              f"x {len(config.stages)} stages")
        for combo in todo:
            print(f"    {combo.name}")
        print(f"\n  weight grid       {len(config.weight_grid)} points")
        if config.campaign:
            print(f"  disfigurements    {config.n_disfigurements} "
                  f"({len(config.gen_k)} top-K + "
                  f"{len(config.walk_k)}x{config.walk_draws} walks)")
            print(f"  gamma             {len(config.gamma)} "
                  f"({', '.join(f'{g:g}' for g in config.gamma)})")
            print(f"  evaluations       {per:,} per combo, "
                  f"{per * len(todo):,} over the ladder")
        else:
            print("  campaign          off; the frontier only")
        print(f"  output            {config.outdir}")
        return 0

    if args.reclaim:
        ladder_module.reclaim_stale(config, sys.stdout.write)

    ran = 0
    while True:
        combo = ladder_module.next_combo(config, sys.stdout.write)
        if combo is None:
            if ran == 0:
                print("  nothing left to claim; every combo is done or "
                      "running")
            else:
                print(f"  ran {ran} combo(s); nothing left to claim")
            return 0

        directory = ladder_module.combo_dir(config, combo)
        os.makedirs(directory, exist_ok=True)
        with Log(os.path.join(directory, TRANSCRIPT),
                 echo=not args.quiet) as log:
            log.section(f"ropf {__version__}: {combo.name}")
            try:
                ladder_module.run_combo(config, combo, log)
            except Exception as exc:
                # The claim is released so the combo is retried rather than
                # looking permanently taken by a process that is gone.
                ladder_module.release(directory)
                log(f"\n {combo.name} FAILED: {type(exc).__name__}: {exc}\n")
                sys.stderr.write(f"{combo.name} failed: {exc}\n")
                return 1
            log(f"\n {combo.name} done in {log.elapsed_s:.1f}s\n")
        ran += 1
        if not args.all:
            return 0


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
    if args.source_dir:
        return fetch.adopt(args.source_dir, args.cases or None,
                           log=sys.stdout.write)
    return fetch.fetch(args.cases or None, log=sys.stdout.write)


if __name__ == "__main__":
    raise SystemExit(main())

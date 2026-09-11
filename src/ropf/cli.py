"""The command line: ``ropf <command>``.

    ropf solve CONFIG     sweep the weight grid for one case, metric and stage,
                          and write the three artifacts
    ropf trace CONFIG     trace the cost-risk frontier and recommend a grid
    ropf ladder CONFIG    run the ladder study, one combo per invocation
    ropf score CONFIG     run the Section 4 campaign against dispatches that
                          already exist, one combo per invocation
    ropf keys             print the configuration key table

THE CONFIG FILE IS THE ONLY PLACE A STUDY PARAMETER IS SET.  No flag overrides
a value in it: the flags change what is printed (``--quiet``, ``--status``),
whether anything is solved (``--dry-run``), or which of the declared work is
done now (``--all``, ``--reclaim``).  None of them can change a number, so the
output directory stays readable as a record of what was run.

`score` is `ladder` without Algorithm 1: same flags, same claiming, same two
campaign artifacts, but it reads its dispatches out of `frontier_dir` instead
of computing them.  They do not share a config because a score config has no
`weight_grid`, `kappa`, `eta` or `k_bar` to set.
"""

from __future__ import annotations

import argparse
import os
import sys
import typing
from dataclasses import fields
from typing import Optional, Sequence

from . import __version__
from .config import ConfigError, RunConfig, read_config
from .log import Log

#: The run transcript, written beside the artifacts.  Nothing reads it back.
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
    _add_run_flags(solve, "report the effective configuration and stop")
    solve.set_defaults(handler=_solve)

    trace = sub.add_parser(
        "trace", help="trace the frontier and recommend a weight grid",
        description=("Adaptively trace the cost-risk frontier of one case, "
                     "metric and stage, and recommend a fixed weight grid to "
                     "write into a config. Solves nothing but Algorithm 1: no "
                     "Section 4 campaign is run and no study artifact is "
                     "written. Its parameters are the frontier_* config keys."))
    _add_run_flags(trace, "report the effective configuration and stop")
    trace.set_defaults(handler=_trace)

    keys = sub.add_parser("keys", help="print the configuration key table",
                          description="Every key a configuration file may set.")
    keys.set_defaults(handler=_keys)

    ladder = sub.add_parser(
        "ladder", help="run the ladder study, one combo per invocation",
        description=("Claim one (instance, metric, stage) combo, run its "
                     "frontier and its Section 4 campaign, and exit. Run it "
                     "again -- or in several shells at once -- for the rest."))
    _add_claim_flags(ladder, "report the combos and the campaign size, and "
                             "solve nothing")
    ladder.set_defaults(handler=_ladder)

    score = sub.add_parser(
        "score", help="run the Section 4 campaign against existing dispatches",
        description=("Claim one (instance, metric, stage) combo, read its "
                     "frontier out of the tree named by `frontier_dir`, run "
                     "the Section 4 campaign against those dispatches, and "
                     "exit. Algorithm 1 does not run. The source tree is "
                     "read-only; the campaign is written to `outdir`."))
    _add_claim_flags(score, "report the combos, their source trees and the "
                            "overheads they carry, and solve nothing")
    score.set_defaults(handler=_score)

    return parser


def _add_run_flags(parser: argparse.ArgumentParser, dry_help: str) -> None:
    parser.add_argument("config", metavar="CONFIG",
                        help="configuration file; see `ropf keys`")
    parser.add_argument("--quiet", action="store_true",
                        help="write the transcript to the file only")
    parser.add_argument("--dry-run", action="store_true", help=dry_help)


def _add_claim_flags(parser: argparse.ArgumentParser, dry_help: str) -> None:
    _add_run_flags(parser, dry_help)
    parser.add_argument("--all", action="store_true",
                        help="keep claiming combos until none are left")
    parser.add_argument("--status", action="store_true",
                        help="print the progress table and stop")
    parser.add_argument("--reclaim", action="store_true",
                        help="release claims older than stale_claim_hours "
                             "before starting")


###############################################################################
# ropf solve
###############################################################################


def _output_dir(config: RunConfig) -> str:
    """``outdir/tag``, or ``outdir/<case>_<metric>_<stage>`` when no tag is set."""
    if config.tag:
        return os.path.join(config.outdir, config.tag)
    case = os.path.splitext(os.path.basename(config.case))[0] or "case"
    return os.path.join(config.outdir, f"{case}_{config.metric}_{config.stage}")


def _trace_dir(config: RunConfig) -> str:
    """`_output_dir` with a ``_trace`` suffix, so tracing a config cannot
    overwrite the transcript of the last solve of that same config."""
    return _output_dir(config) + "_trace"


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
                network,
                config.algorithm_config(
                    multiplier, lp_dir=config.lp_dir(outdir, multiplier)),
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
    if config.flow_domain != "all":
        emit(f" flow_domain   {config.flow_domain}   (the maximum in eq (4b) "
             f"runs over this component set; the model files, the feasible set "
             f"and eq (1d) are untouched)\n")
    emit(f" stage         {config.stage}\n")
    emit(f" weights       {', '.join(f'{w:g}' for w in config.weights)}"
         f" x lambda*\n")
    if config.risk_weight is not None:
        emit(f" risk_weight   {config.risk_weight:g} $/p.u. (absolute; "
             f"lambda* is bypassed)\n")
    if config.exchange_rate is not None or config.exchange_rates:
        # Where the weights came from, not just what they are: a sweep stated
        # as tolerances and the same sweep stated as multipliers are the same
        # study, so the record has to say which was written down.
        tolerances = ((config.xi,) if config.xi is not None else config.xi_grid)
        emit(f" xi            {', '.join(f'{x:g}' for x in tolerances)}"
             f"   (cost tolerance, as a fraction of z0)\n")
        emit(f" tau band      {config.tau_lo:g} to {config.tau_hi:g}, width "
             f"{config.tau_hi - config.tau_lo:g}   (as a fraction of rho0)\n")
        emit(f" exchange rate xi/(tau_hi - tau_lo) = "
             f"{', '.join(f'{r:g}' for r in config.weights)}"
             f"   [eq (weightstar)]\n")
    emit(f" kappa         {config.kappa}\n")
    emit(f" eta           {config.eta:g}\n")
    emit(f" k-bar         {config.k_bar}\n")
    emit(f" exposure      components at or above {config.exposure_fraction:g}"
         f" of rho are reported bad\n")
    emit(f" solvers       {config.solver_dc} (DC), {config.solver_ac} (AC),"
         f" {config.time_limit_s:g}s each\n")
    emit(f" output        {outdir}\n")
    if config.write_lp:
        emit(f" LP files      {os.path.join(outdir, 'lp')}"
             f"{os.sep}w<lambda>{os.sep}master_k<iteration>.lp\n")


###############################################################################
# ropf trace
###############################################################################


def _trace(args: argparse.Namespace) -> int:
    from .network import read_matpower
    from .study import frontier

    config = read_config(args.config)
    if not config.case:
        raise ConfigError(f"{args.config}: no `case` given; there is nothing "
                          f"to trace")

    outdir = _trace_dir(config)
    if args.dry_run:
        _report_config(config, outdir, sys.stdout.write)
        _report_tracer(config, sys.stdout.write)
        return 0

    os.makedirs(outdir, exist_ok=True)
    with Log(os.path.join(outdir, TRANSCRIPT), echo=not args.quiet) as log:
        log.section(f"ropf {__version__}: trace "
                    f"{os.path.basename(args.config)}")
        _report_config(config, outdir, log)
        _report_tracer(config, log)

        network = read_matpower(config.case, log)
        out = frontier.trace(config, log, network)

        log.section("recommended weight grid")
        _report_trace(out, log)
        frontier.write(outdir, config, out, network, log)
        log(f"\n total wall clock {log.elapsed_s:.1f}s\n")

        if out.termination in ("infeasible", "nominal_risk_zero",
                               "flat_frontier"):
            log(f" no frontier was traced: {out.termination}\n")
            return 1
    return 0


def _report_tracer(config: RunConfig, emit) -> None:
    emit(" trace note    the weights above are the config's CURRENT grid, "
         "which the\n               trace reads nothing from and does not "
         "change; it recommends a\n               replacement to paste in\n")
    emit(f" trace seed    {config.frontier_seed:g} x lambda*, doubling to a "
         f"cap of {config.frontier_hi_cap:g}\n")
    emit(f" trace budget  {config.frontier_max_points} points, chord "
         f"tolerance {config.frontier_tol:g}, saturation "
         f"{config.frontier_saturation_tol:g}\n")
    emit(f" trace grid    {config.frontier_grid_points} recommended "
         f"multipliers\n")


def _report_trace(out, emit) -> None:
    """The points, then the line the whole command exists to produce."""
    emit(f"\n {'lambda/lambda*':>15}  {'cost':>16}  {'tau':>10}  "
         f"{'role':<9}  termination\n")
    for point in out.points:
        emit(f" {point.multiplier:>15g}  {point.cost:>16.6f}  "
             f"{point.tau:>10.6f}  {point.role:<9}  {point.termination}\n")
    emit(f"\n {len(out.points)} points in {out.n_solves} solves; walk "
         f"{out.walk_outcome or 'not run'}; {out.termination} "
         f"({out.termination_detail})\n")
    if not out.recommended_weight_grid:
        emit("\n There is no grid to recommend: the trace found no usable "
             "point to build one from.\n\n")
        return
    emit("\n Paste this into the ladder config to fix the campaign grid:\n\n")
    emit(f"     {out.weight_grid_line()}\n\n")


###############################################################################
# ropf ladder and ropf score
###############################################################################


def _ladder(args: argparse.Namespace) -> int:
    from .study import ladder as ladder_module

    config = ladder_module.read_ladder_config(args.config)
    if args.status:
        _print_status(ladder_module.status_rows(config))
        return 0
    if args.dry_run:
        return _report_ladder(config, ladder_module)
    if args.reclaim:
        ladder_module.reclaim_stale(config, sys.stdout.write)
    return _claim_loop(args, config, ladder_module.run_combo)


def _score(args: argparse.Namespace) -> int:
    from .study import ladder as ladder_module

    config = ladder_module.read_score_config(args.config)
    if args.status:
        _print_status(ladder_module.status_rows(config))
        return 0
    if args.dry_run:
        return _report_score(config, ladder_module)
    if args.reclaim:
        ladder_module.reclaim_stale(config, sys.stdout.write)
    return _claim_loop(args, config, ladder_module.score_combo)


def _report_combos(config, todo) -> None:
    print(f"  {len(todo)} combos: "
          f"{len(config.instances)} instances x {len(config.metrics)} metrics "
          f"x {len(config.stages)} stages")


def _report_ladder(config, ladder_module) -> int:
    todo = ladder_module.combos(config)
    per = config.evaluations_per_combo()
    _report_combos(config, todo)
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


def _report_score(config, ladder_module) -> int:
    """What would be scored, out of which tree, and at what cost overheads.

    The overheads are a property of the source trees rather than of the config,
    so this READS them.  It still solves nothing: the artifacts it opens are
    the ones `score_combo` would open, for the same checks.
    """
    todo = ladder_module.combos(config)
    _report_combos(config, todo)
    print(f"  reading from      {config.frontier_dir}   (read-only)")
    print(f"  disfigurements    {config.n_disfigurements} sampled "
          f"({len(config.gen_k)} top-K + "
          f"{len(config.walk_k)}x{config.walk_draws} walks)"
          + (", plus the vendor list" if config.vendor_list else ""))
    print(f"  gamma             {len(config.gamma)} "
          f"({', '.join(f'{g:g}' for g in config.gamma)}), "
          f"beta = {config.beta:g}")
    print(f"  output            {config.outdir}")
    print()

    points, missing = 0, 0
    width = max(len(c.name) for c in todo)
    for combo in todo:
        source = ladder_module.frontier_combo_dir(config, combo)
        try:
            read = ladder_module.read_frontier(source, combo, combo.case_path())
        except (OSError, ValueError) as exc:
            missing += 1
            print(f"    {combo.name:<{width}}  -- {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0]}")
            continue
        points += len(read.runs) * len(config.gamma)
        overheads = ", ".join(
            f"{read.overhead[r.weight_multiplier] * 100:.2f}"
            for r in read.runs if r.weight_multiplier in read.overhead)
        print(f"    {combo.name:<{width}}  {len(read.runs)} weights, "
              f"overhead % {overheads}")

    # The vendor class is enumerated from the case's .aux, not set in the
    # config, so the only honest count here is the sampled one.  `build_campaign`
    # logs what it actually mapped, per combo, as it runs.
    print(f"\n  dispatch points   {points:,} (weight x gamma) over "
          f"{len(todo) - missing} combos"
          + (f"; {missing} combo(s) carry no frontier to score" if missing
             else ""))
    print(f"  evaluations       {points * config.n_disfigurements:,} against "
          f"the sampled classes"
          + (", plus the vendor list, whose size is a property of the case "
             "and not of this config" if config.vendor_list else ""))
    return 1 if missing else 0


def _print_status(rows) -> None:
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


def _claim_loop(args: argparse.Namespace, config, runner) -> int:
    """Claim a combo, run `runner` on it, exit -- or keep going under --all.

    One loop for both drivers: the claim is what makes the study parallel and
    resumable, and a second copy would be a second chance to get the
    release-on-failure wrong.
    """
    from .study import ladder as ladder_module

    ran = 0
    while True:
        combo = ladder_module.next_combo(config, sys.stdout.write)
        if combo is None:
            print("  nothing left to claim; every combo is done or running"
                  if ran == 0 else
                  f"  ran {ran} combo(s); nothing left to claim")
            return 0

        directory = ladder_module.combo_dir(config, combo)
        os.makedirs(directory, exist_ok=True)
        with Log(os.path.join(directory, TRANSCRIPT),
                 echo=not args.quiet) as log:
            log.section(f"ropf {__version__}: {combo.name}")
            try:
                runner(config, combo, log)
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


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Cross-evaluation: every de-risked dispatch scored under every representation.

Edit the settings below and run the script; there is no config file and no
command line.  One run of Algorithm 1 per representation at lambda =
WEIGHT_MULTIPLIER lambda_ref, then each of those dispatches, and the plain OPF,
is scored under all of them.  The solves are the whole cost: scoring a
representation on a dispatch is arithmetic on Pf, so the off-diagonal entries
are free.

Read the matrix by column.  A column is one representation, and the entry that
ought to be lowest in it is the dispatch de-risked under that representation;
the diagonal reproduces the tau of a frontier run at the same weight.  An
off-diagonal entry above 1 is relocation: de-risking under one representation
raised the quantity another names.

Writes xeval.csv (one row per dispatch x representation), xeval.json, xeval.png
and run.log.  Analysis only: it never edits a config or a solve directory.

Usage:  python analysis/xeval.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, fields

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from ropf import risk                                            # noqa: E402
from ropf.algorithm import AlgorithmConfig, Result, run          # noqa: E402
from ropf.cli import Log                                         # noqa: E402
from ropf.config import ConfigError, resolve_case                # noqa: E402
from ropf.model import SolverConfig                              # noqa: E402
from ropf.network import CaseFormatError, Network, read_matpower  # noqa: E402

###############################################################################
# Settings.  The keys are those of a ropf config; see the table in README.md.
###############################################################################

CASE = "case_ACTIVSg200"        # a bare name is looked up in data/
REPRESENTATIONS = ("max_active_flow", "joule_loss_max", "bus_flow_sum_agg")
WEIGHT_MULTIPLIER = 1.0         # the one weight every representation is solved at
STAGE = "baseline"              # baseline, or a2 to add the hybrid AC solve
FLOW_DOMAIN = "all"             # all, or rated to drop branches with no rateA
Q = 1                           # order: mean of the q largest contributions

KAPPA = 5                       # cuts per iteration
ETA = 0.85                      # stop when rho/rho0 <= 1 - eta
K_BAR = 40                      # iteration limit
GAMMA_TOL = 1e-6                # stop when the gap Gamma <= gamma_tol

SOLVER_DC = "gurobi"
SOLVER_AC = "knitro"            # used only under STAGE = "a2"
TIME_LIMIT_S = 3600.0           # per solve
THREADS = None                  # None leaves the solver's own default
SOLVER_VERBOSE = False

#: Where the artifacts go.  Empty means results/<case>_xeval.
OUTDIR = ""

#: Also draw xeval.png.  Needs matplotlib, which ropf itself does not.
PLOT = True

###############################################################################

#: The plain OPF, which every ratio is taken against.  Not a representation.
NOMINAL = "nominal"

#: How each representation is named in a table, in a plot, and in a colour.
SHORT = {"max_active_flow": "phi^flow", "joule_loss_max": "phi^joule",
         "bus_flow_sum_agg": "phi^bus", NOMINAL: "plain OPF"}
SYMBOL = {"max_active_flow": r"$\phi^{\mathrm{flow}}$",
          "joule_loss_max": r"$\phi^{\mathrm{joule}}$",
          "bus_flow_sum_agg": r"$\phi^{\mathrm{bus}}$"}
COLOR = {"max_active_flow": "#1f4e79", "joule_loss_max": "#c0392b",
         "bus_flow_sum_agg": "#2e7d32"}
NOMINAL_COLOR = "#8d99a6"
GRID_COLOR = "#dfe3e8"


def algorithm_settings(metric: str, multiplier: float) -> AlgorithmConfig:
    """The settings above under one representation at one weight."""
    return AlgorithmConfig(metric=metric, stage=STAGE, flow_domain=FLOW_DOMAIN,
                           q=Q, weight_multiplier=multiplier, kappa=KAPPA,
                           eta=ETA, k_bar=K_BAR, gamma_tol=GAMMA_TOL)


def solver(name: str) -> SolverConfig:
    return SolverConfig(name=name, time_limit_s=TIME_LIMIT_S, threads=THREADS,
                        verbose=SOLVER_VERBOSE)


@dataclass
class Cell:
    """One dispatch scored under one representation.  Field order is the CSV order."""

    solved_under: str        # NOMINAL, or the representation the dispatch was found under
    evaluated_under: str     # the representation doing the scoring
    diagonal: bool           # the two agree: the representation scoring its own dispatch
    exposure: float
    exposure_ratio: float    # against the plain OPF scored under evaluated_under
    element: int             # where the largest contribution sits
    element_id: str          # that element named as the case names it
    cost: float
    cost_overhead_pct: float
    k_end: int
    termination: str
    status: str


###############################################################################
# Solving and scoring
###############################################################################


def cross_evaluate(network: Network, log=None) -> tuple[dict, list[Cell]]:
    """Solve under each representation, then score every dispatch under each."""
    emit = log or (lambda _message: None)
    meta = {
        "case": os.path.basename(network.casefile),
        "buses": network.numbuses,
        "branches": network.numbranches,
        "gens": network.numgens,
        "weight_multiplier": WEIGHT_MULTIPLIER,
        "stage": STAGE,
        "flow_domain": FLOW_DOMAIN,
        "q": Q,
    }
    dc, ac = solver(SOLVER_DC), solver(SOLVER_AC)

    # The plain OPF is one solve, not one per representation: it does not depend
    # on the representation, which only enters through the exposure weight.
    emit("\n=== plain OPF ===\n")
    runs = {NOMINAL: run(network, algorithm_settings(REPRESENTATIONS[0], 0.0),
                         dc, ac, emit)}
    if not runs[NOMINAL].reported.solved:
        raise ConfigError("the plain OPF did not solve; every ratio is taken "
                          "against it, so there is nothing to cross-evaluate")
    for metric in REPRESENTATIONS:
        emit(f"\n=== de-risked under {SHORT[metric]} ===\n")
        runs[metric] = run(network, algorithm_settings(metric, WEIGHT_MULTIPLIER),
                           dc, ac, emit)

    z0 = runs[NOMINAL].reported.cost
    nominal_Pf = runs[NOMINAL].dispatch.Pf
    reference = {metric: risk.evaluate(metric, network, nominal_Pf, FLOW_DOMAIN, Q).value
                 for metric in REPRESENTATIONS}

    cells = []
    for solved_under, result in runs.items():
        for evaluated_under in REPRESENTATIONS:
            cells.append(_cell(network, solved_under, evaluated_under, result,
                               reference[evaluated_under], z0))
    return meta, cells


def _cell(network: Network, solved_under: str, evaluated_under: str,
          result: Result, reference: float, z0: float) -> Cell:
    """Score one dispatch under one representation.  A dispatch whose solve did
    not reach `solved` is scored anyway and flagged: the numbers are then a
    pre-feasible iterate's, and the table and the plot leave them out."""
    iterate = result.reported
    evaluation = risk.evaluate(evaluated_under, network, result.dispatch.Pf,
                               FLOW_DOMAIN, Q)
    element = evaluation.ranked()[0][0] if evaluation.components else -1
    return Cell(
        solved_under=solved_under,
        evaluated_under=evaluated_under,
        diagonal=solved_under == evaluated_under,
        exposure=evaluation.value,
        exposure_ratio=evaluation.value / reference if reference else float("nan"),
        element=element,
        element_id=_element_id(network, evaluated_under, element),
        cost=iterate.cost,
        cost_overhead_pct=(iterate.cost / z0 - 1) * 100 if z0 else float("nan"),
        k_end=result.k_end,
        termination=result.termination,
        status=iterate.status,
    )


def _element_id(network: Network, metric: str, element: int) -> str:
    """The element carrying the largest contribution, as the case names it."""
    if element < 0:
        return ""
    if metric == "bus_flow_sum_agg":
        return f"bus {network.buses[element].nodeID}"
    branch = network.branches[element]
    return f"branch {branch.f}-{branch.t}"


###############################################################################
# Output
###############################################################################


def write_csv(path: str, meta: dict, cells) -> None:
    columns = [f.name for f in fields(Cell)]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(meta) + columns)
        writer.writeheader()
        for cell in cells:
            writer.writerow({**meta, **asdict(cell)})


def write_json(path: str, meta: dict, cells) -> None:
    settings = {"case": CASE, "representations": list(REPRESENTATIONS),
                "weight_multiplier": WEIGHT_MULTIPLIER, "stage": STAGE,
                "flow_domain": FLOW_DOMAIN, "q": Q, "kappa": KAPPA, "eta": ETA,
                "k_bar": K_BAR, "gamma_tol": GAMMA_TOL, "solver_dc": SOLVER_DC,
                "solver_ac": SOLVER_AC, "time_limit_s": TIME_LIMIT_S,
                "threads": THREADS}
    with open(path, "w") as handle:
        json.dump({"metadata": meta, "settings": settings,
                   "cells": [asdict(c) for c in cells]}, handle, indent=1)


def write_table(log, cells) -> None:
    """The matrix as read: rows are dispatches, columns are representations.

    `*` marks the lowest entry of a column and `d` the diagonal.  They coincide
    when each representation is the best way to reduce what it names."""
    by_key = {(c.solved_under, c.evaluated_under): c for c in cells}
    rows = [NOMINAL] + [m for m in REPRESENTATIONS
                        if by_key[(m, m)].status == "solved"]
    best = {column: min((by_key[(r, column)].exposure_ratio for r in rows))
            for column in REPRESENTATIONS}

    log("\nexposure relative to the plain OPF"
        f" (lambda = {WEIGHT_MULTIPLIER:g} lambda_ref)\n\n")
    log(f"{'dispatch':>12}" + "".join(f"{SHORT[c]:>14}" for c in REPRESENTATIONS)
        + f"{'+cost %':>10}\n")
    for row in rows:
        log(f"{SHORT[row]:>12}")
        for column in REPRESENTATIONS:
            cell = by_key[(row, column)]
            mark = ("d" if cell.diagonal else " ") + (
                "*" if cell.exposure_ratio <= best[column] else " ")
            log(f"{cell.exposure_ratio:12.4f}{mark}")
        log(f"{by_key[(row, REPRESENTATIONS[0])].cost_overhead_pct:10.3f}\n")

    for row in rows[1:]:
        raised = [SHORT[c] for c in REPRESENTATIONS
                  if by_key[(row, c)].exposure_ratio > 1.0]
        if raised:
            log(f"\n{SHORT[row]} relocates: it raises {', '.join(raised)}\n")


def write_plot(path: str, meta: dict, cells) -> None:
    """The matrix as grouped bars, one group per representation.  Needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {(c.solved_under, c.evaluated_under): c for c in cells}
    dispatches = [m for m in REPRESENTATIONS if by_key[(m, m)].status == "solved"]
    if not dispatches:
        return

    figure, axes = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    axes.set_axisbelow(True)
    axes.grid(True, axis="y", color=GRID_COLOR, linewidth=0.5)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)

    width = 0.8 / len(dispatches)
    for index, dispatch in enumerate(dispatches):
        offset = (index - (len(dispatches) - 1) / 2) * width
        bars = axes.bar([i + offset for i in range(len(REPRESENTATIONS))],
                        [by_key[(dispatch, c)].exposure_ratio
                         for c in REPRESENTATIONS],
                        width * 0.9, color=COLOR[dispatch], zorder=3,
                        label=SYMBOL[dispatch])
        for column, bar in zip(REPRESENTATIONS, bars):
            if column == dispatch:        # the representation scoring its own work
                bar.set_edgecolor("black")
                bar.set_linewidth(1.2)
        axes.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    axes.axhline(1.0, color=NOMINAL_COLOR, linestyle=":", linewidth=1.0, zorder=2)
    axes.annotate("plain OPF", (len(REPRESENTATIONS) - 0.5, 1.0), fontsize=8,
                  color=NOMINAL_COLOR, ha="right", va="bottom")
    axes.set_xticks(range(len(REPRESENTATIONS)))
    axes.set_xticklabels([SYMBOL[c] for c in REPRESENTATIONS])
    axes.set_xlabel("evaluated under")
    axes.set_ylabel("exposure, relative to the plain OPF")
    axes.set_title(rf"{meta['case']}, $\lambda = {WEIGHT_MULTIPLIER:g}\,\lambda^{{\mathrm{{ref}}}}$,"
                   f" stage {meta['stage']}", fontsize=10, pad=8)
    axes.tick_params(labelsize=9)
    axes.margins(y=0.16)
    axes.legend(title="de-risked under", fontsize=8, title_fontsize=8,
                frameon=False, ncols=len(dispatches), loc="upper center")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def output_dir(case: str) -> str:
    stem = os.path.splitext(os.path.basename(case))[0]
    return os.path.join("results", f"{stem}_xeval")


###############################################################################
# Entry point
###############################################################################


def main() -> int:
    try:
        case = resolve_case(CASE)
        for metric in REPRESENTATIONS:       # validated before any solve
            algorithm_settings(metric, WEIGHT_MULTIPLIER)
        if WEIGHT_MULTIPLIER <= 0:
            raise ConfigError("the weight must be positive: at zero every "
                              "dispatch is the plain OPF")
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    outdir = OUTDIR or output_dir(case)
    os.makedirs(outdir, exist_ok=True)
    with Log(os.path.join(outdir, "run.log")) as log:
        try:
            meta, cells = cross_evaluate(read_matpower(case), log)
        except (CaseFormatError, ConfigError) as exc:
            log(f"{exc}\n")
            return 2
        write_csv(os.path.join(outdir, "xeval.csv"), meta, cells)
        write_json(os.path.join(outdir, "xeval.json"), meta, cells)
        write_table(log, cells)
        if PLOT:
            try:
                write_plot(os.path.join(outdir, "xeval.png"), meta, cells)
            except ImportError:   # the CSV is written either way
                log("\nno matplotlib: xeval.png not written\n")
        log(f"\nwritten to {os.path.abspath(outdir)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

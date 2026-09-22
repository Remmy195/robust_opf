#!/usr/bin/env python3
"""The cost-exposure frontier: one run of Algorithm 1 per exposure weight.

Usage:  python analysis/frontier.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, fields, replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from ropf.algorithm import Result, run                           # noqa: E402
from ropf.cli import Log                                         # noqa: E402
from ropf.config import ConfigError, RunConfig, resolve_case     # noqa: E402
from ropf.network import CaseFormatError, read_matpower          # noqa: E402

###############################################################################
# Settings.  The keys are those of a ropf config; see the table in README.md.
###############################################################################

CASE = "case_ACTIVSg10k"        # a bare name is looked up in data/
METRIC = "max_active_flow"      # max_active_flow, joule_loss_max, bus_flow_sum_agg
STAGE = "baseline"              # baseline, or a2 to add the hybrid AC solve
FLOW_DOMAIN = "all"             # all, or rated to drop branches with no rateA
Q = 1                           # order: mean of the q largest contributions

KAPPA = 5                       # cuts per iteration
ETA = 0.8                      # stop when rho/rho0 <= 1 - eta
K_BAR = 15                      # iteration limit
GAMMA_TOL = 1e-6                # stop when the gap Gamma <= gamma_tol; 0 disables

SOLVER_DC = "gurobi"
SOLVER_AC = "knitro"            # used only under STAGE = "a2"
TIME_LIMIT_S = 3600.0           # per solve
THREADS = None                  # None leaves the solver's own default
SOLVER_VERBOSE = False

#: The weights to trace, as multiples of lambda*: the plain OPF, then a
#: geometric grid over three octaves either side.  A case whose frontier bends
#: between two of these wants a grid of its own.
WEIGHTS = (0, 0.001, 0.002, 0.0039, 0.0078, 0.0156, 0.0312, 34)

#: Where the artifacts go.  Empty means results/<case>_<metric>_<stage>_frontier.
OUTDIR = ""

#: Also draw frontier.png.  Needs matplotlib, which ropf itself does not.
PLOT = True

###############################################################################

#: Two points this close on an axis are one point, not one dominating the other:
#: weights that reach the same dispatch differ in the last significant figures.
DOMINANCE_RTOL = 1e-7

#: Plot colours: the traced curve, the plain-OPF anchor, a dominated point, the grid.
CURVE_COLOR = "#1f4e79"
BASELINE_COLOR = "#c0392b"
DOMINATED_COLOR = "#8d99a6"
GRID_COLOR = "#dfe3e8"


def settings() -> RunConfig:
    """The settings above as a ropf config, validated as `ropf solve` validates it."""
    return RunConfig(case=resolve_case(CASE), metric=METRIC, stage=STAGE,
                     flow_domain=FLOW_DOMAIN, q=Q, kappa=KAPPA, eta=ETA,
                     k_bar=K_BAR, gamma_tol=GAMMA_TOL, solver_dc=SOLVER_DC,
                     solver_ac=SOLVER_AC, time_limit_s=TIME_LIMIT_S,
                     threads=THREADS, solver_verbose=SOLVER_VERBOSE,
                     outdir=OUTDIR)


@dataclass
class Point:
    """One traced weight.  Field order is the CSV column order."""

    weight_multiplier: float
    risk_weight: float
    lambda_star: float
    cost: float
    exposure: float
    surrogate: float
    gamma: float
    cost_ratio: float        # cost / z0
    exposure_ratio: float    # rho / rho0, the tau of the paper
    z0: float
    rho0: float
    k_end: int
    n_line_cuts: int
    n_bus_cuts: int
    termination: str
    status: str
    solve_time_s: float
    dominated: bool = False


###############################################################################
# Tracing
###############################################################################


def trace(config: RunConfig, weights, log=None) -> tuple[dict, list[Point]]:
    """Run Algorithm 1 at each weight.  Returns the run metadata and the points."""
    emit = log or (lambda _message: None)
    network = read_matpower(config.case)
    meta = {
        "case": os.path.basename(network.casefile),
        "buses": network.numbuses,
        "branches": network.numbranches,
        "gens": network.numgens,
        "metric": config.metric,
        "stage": config.stage,
        "flow_domain": config.flow_domain,
        "q": config.q,
    }
    emit(f"{meta['case']}: {meta['buses']} buses, {meta['branches']} branches,"
         f" {meta['gens']} gens\n"
         f"{config.metric}, stage {config.stage}, {len(weights)} weights\n")

    points = []
    for index, multiplier in enumerate(weights, start=1):
        emit(f"\n=== {index}/{len(weights)}  lambda = {multiplier:g} lambda* ===\n")
        # The swept weight wins over both weight keys of the config.
        settings = replace(config.algorithm_config(),
                           weight_multiplier=multiplier, risk_weight=None)
        result = run(network, settings, config.dc_solver(), config.ac_solver(), emit)
        points.append(_point(multiplier, result))
    mark_dominated(points)
    return meta, points


def _point(multiplier: float, result: Result) -> Point:
    """The reported iterate of a run -- the AC solve under stage a2, else the last."""
    iterate = result.reported
    return Point(
        weight_multiplier=multiplier,
        risk_weight=result.risk_weight,
        lambda_star=result.lambda_star,
        cost=iterate.cost,
        exposure=iterate.rho,
        surrogate=iterate.surrogate,
        gamma=iterate.gamma,
        cost_ratio=iterate.cost / result.z0 if result.z0 else float("nan"),
        exposure_ratio=iterate.ratio,
        z0=result.z0,
        rho0=result.rho0,
        k_end=result.k_end,
        n_line_cuts=iterate.n_line_cuts,
        n_bus_cuts=iterate.n_bus_cuts,
        termination=result.termination,
        status=iterate.status,
        solve_time_s=round(result.total_time_s, 3),
    )


def mark_dominated(points) -> None:
    """Flag a point another beats on both axes.  A dominated point is real
    information -- a loop that stopped at a different k, or an AC solve that
    found another local solution -- so it is flagged and kept, never dropped."""
    priced = [p for p in points if math.isfinite(p.cost) and math.isfinite(p.exposure)]
    for point in priced:
        point.dominated = any(
            other is not point
            and _no_worse(other.cost, point.cost)
            and _no_worse(other.exposure, point.exposure)
            and (_better(other.cost, point.cost)
                 or _better(other.exposure, point.exposure))
            for other in priced)


def _slack(value: float) -> float:
    return DOMINANCE_RTOL * max(1.0, abs(value))


def _no_worse(candidate: float, reference: float) -> bool:
    return candidate <= reference + _slack(reference)


def _better(candidate: float, reference: float) -> bool:
    return candidate < reference - _slack(reference)


###############################################################################
# Output
###############################################################################


def _kept(point: Point) -> float:
    """Exposure kept, rho/rho0.  Across, in the plot."""
    return point.exposure_ratio


def _overhead(point: Point) -> float:
    """Cost above the plain OPF, in percent.  Up, in the plot."""
    return (point.cost_ratio - 1) * 100


def write_csv(path: str, meta: dict, points) -> None:
    columns = [f.name for f in fields(Point)]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(meta) + columns)
        writer.writeheader()
        for point in points:
            writer.writerow({**meta, **asdict(point)})


def write_json(path: str, meta: dict, config: RunConfig, points) -> None:
    """The points, plus the settings they share.  The two weight keys of the
    config are left out: the weight is per point, and is in every row."""
    shared = {key: value for key, value in asdict(config).items()
              if key not in ("weight_multiplier", "risk_weight")}
    with open(path, "w") as handle:
        json.dump({"metadata": meta, "settings": shared,
                   "points": [asdict(p) for p in points]}, handle, indent=1)


def write_table(log, points) -> None:
    """The curve as read: what each weight costs and what it buys."""
    log(f"\n{'lambda/lambda*':>14} {'cost':>14} {'+cost %':>9}"
        f" {'exposure':>12} {'tau':>8} {'k':>4}  termination\n")
    for point in points:
        flag = "  *dominated" if point.dominated else ""
        log(f"{point.weight_multiplier:14g} {point.cost:14.2f}"
            f" {_overhead(point):9.3f} {point.exposure:12.6f}"
            f" {_kept(point):8.4f} {point.k_end:4d}"
            f"  {point.termination}{flag}\n")


def write_plot(path: str, meta: dict, points) -> None:
    """The frontier: exposure kept across, cost paid up.  Needs matplotlib.

    The plain OPF sits at (1, 0) by construction and is drawn as the anchor the
    rest of the curve is priced against.  Dominated points are drawn apart from
    the curve, since a line through them would not be a frontier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solved = [p for p in points if p.status == "solved"]
    if not solved:
        return
    curve = sorted((p for p in solved if not p.dominated), key=_kept)
    dominated = [p for p in solved if p.dominated]
    baseline = next((p for p in curve if p.weight_multiplier == 0), None)

    figure, axes = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    axes.set_axisbelow(True)
    axes.grid(True, color=GRID_COLOR, linewidth=0.5)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)

    if baseline is not None:   # crosshair on the anchor, drawn under everything
        axes.axhline(_overhead(baseline), color=BASELINE_COLOR,
                     linewidth=0.7, linestyle=":", alpha=0.6, zorder=1)
        axes.axvline(_kept(baseline), color=BASELINE_COLOR,
                     linewidth=0.7, linestyle=":", alpha=0.6, zorder=1)

    axes.plot([_kept(p) for p in curve], [_overhead(p) for p in curve],
              marker="o", markersize=5, markerfacecolor="white",
              markeredgewidth=1.3, linewidth=1.5, color=CURVE_COLOR,
              zorder=3, label="de-risked dispatch")
    if dominated:
        axes.scatter([_kept(p) for p in dominated], [_overhead(p) for p in dominated],
                     marker="x", s=30, linewidths=1.2, color=DOMINATED_COLOR,
                     zorder=2, label="dominated")
    if baseline is not None:
        axes.scatter([_kept(baseline)], [_overhead(baseline)], marker="o", s=55,
                     color=BASELINE_COLOR, zorder=4,
                     label=r"plain OPF ($\lambda = 0$)")

    deepest = curve[0] if curve else None
    if deepest is not None and deepest is not baseline:
        axes.annotate(rf"$\lambda = {deepest.weight_multiplier:g}\,\lambda^*$",
                      (_kept(deepest), _overhead(deepest)),
                      textcoords="offset points", xytext=(10, 7),
                      fontsize=8, color=CURVE_COLOR)

    axes.set_xlabel(r"exposure kept, $\rho / \rho^0$")
    axes.set_ylabel("cost above the plain OPF (%)")
    axes.set_title(f"{meta['case']}, {meta['metric']}, stage {meta['stage']}",
                   fontsize=10, pad=8)
    axes.margins(x=0.12, y=0.14)
    axes.tick_params(labelsize=9)
    axes.legend(fontsize=8, frameon=False, loc="upper right")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def output_dir(config: RunConfig) -> str:
    case = os.path.splitext(os.path.basename(config.case))[0]
    return os.path.join("results", f"{case}_{config.metric}_{config.stage}_frontier")


###############################################################################
# Entry point
###############################################################################


def main() -> int:
    try:
        config = settings()
        if any(weight < 0 for weight in WEIGHTS):
            raise ConfigError("an exposure weight cannot be negative")
    except (ConfigError, FileNotFoundError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    outdir = OUTDIR or output_dir(config)
    os.makedirs(outdir, exist_ok=True)
    with Log(os.path.join(outdir, "run.log")) as log:
        try:
            meta, points = trace(config, sorted(set(WEIGHTS)), log)
        except CaseFormatError as exc:
            log(f"{exc}\n")
            return 2
        write_csv(os.path.join(outdir, "frontier.csv"), meta, points)
        write_json(os.path.join(outdir, "frontier.json"), meta, config, points)
        write_table(log, points)
        if PLOT:
            try:
                write_plot(os.path.join(outdir, "frontier.png"), meta, points)
            except ImportError:   # the CSV is written either way
                log("\nno matplotlib: frontier.png not written\n")
        log(f"\nwritten to {os.path.abspath(outdir)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

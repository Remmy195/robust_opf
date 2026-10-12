#!/usr/bin/env python3
"""Nominal vs. de-risked branch flows: two runs of Algorithm 1, at
weight_multiplier = 0 and at the configured weight, compared branch by branch.

Usage:  python analysis/flows.py
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from ropf.algorithm import run                                   # noqa: E402
from ropf.cli import Log, write_dispatch_csvs                    # noqa: E402
from ropf.config import ConfigError, RunConfig, resolve_case     # noqa: E402
from ropf.network import CaseFormatError, read_matpower          # noqa: E402

###############################################################################
# Settings.  The keys are those of a ropf config; see the table in README.md.
###############################################################################

CASE = "pglib_opf_case30000_goc__api"        # a bare name is looked up in data/
METRIC = "joule_loss_max"      # max_active_flow, joule_loss_max, bus_flow_sum_agg
STAGE = "baseline"              # baseline, or a2 to add the hybrid AC solve
FLOW_DOMAIN = "all"             # all, or rated to drop branches with no rateA
Q = 10                          # order: mean of the q largest contributions

WEIGHT_MULTIPLIER = 0.1         # exposure weight as a multiple of lambda_ref
KAPPA = 5                       # cuts per iteration
ETA = 0.85                      # stop when rho/rho0 <= 1 - eta
K_BAR = 40                      # iteration limit
GAMMA_TOL = 1e-6                # stop when the gap Gamma <= gamma_tol

SOLVER_DC = "gurobi"
SOLVER_AC = "knitro"            # used only under STAGE = "a2"
TIME_LIMIT_S = 3600.0           # per solve
THREADS = None                  # None leaves the solver's own default
SOLVER_VERBOSE = False

#: Where the artifacts go.  Empty means results/<case>_<metric>_<stage>_flows.
OUTDIR = ""

#: Also draw the plots.  Needs matplotlib, which ropf itself does not.
PLOT = True

###############################################################################


def settings() -> RunConfig:
    """The settings above as a ropf config, validated as `ropf solve` validates it."""
    return RunConfig(case=resolve_case(CASE), metric=METRIC, stage=STAGE,
                     flow_domain=FLOW_DOMAIN, q=Q, weight_multiplier=WEIGHT_MULTIPLIER,
                     kappa=KAPPA, eta=ETA, k_bar=K_BAR, gamma_tol=GAMMA_TOL,
                     solver_dc=SOLVER_DC, solver_ac=SOLVER_AC, time_limit_s=TIME_LIMIT_S,
                     threads=THREADS, solver_verbose=SOLVER_VERBOSE, outdir=OUTDIR)


def branch_rows(network, solution):
    """{branch: (from, to, limit, Pf, abs_Pf, loading, joule_loss)}; MW, except
    loading (fraction of limit). joule_loss is r_heat * Pf_pu^2 (the same surrogate
    as risk.py's joule_loss_max) scaled once by baseMVA, so it comes out in MW."""
    base = network.baseMVA
    rows = {}
    for i, br in network.branches.items():
        pf = solution.Pf.get(i, 0.0)
        rows[i] = (br.id_f, br.id_t, br.limit * base, pf * base,
                   abs(pf) * base, abs(pf) / br.limit, br.r_heat * pf * pf * base)
    return rows


def write_csv(path: str, rows) -> None:
    """`rows`: ordered (branch, from, to, limit, Pf, abs_Pf, loading, joule_loss) tuples."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "branch", "from", "to", "limit", "Pf", "abs_Pf",
                         "loading", "joule_loss"])
        for rank, row in enumerate(rows, start=1):
            writer.writerow([rank, *row])


def write_plot(path: str, values, ylabel: str, hline: float = None) -> None:
    """`values`, already sorted highest first, plotted against rank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    axes.plot(range(1, len(values) + 1), values, linewidth=1.5, color="#1f4e79")
    if hline is not None:
        axes.axhline(hline, color="#c0392b", linewidth=0.8, linestyle=":", label="limit")
        axes.legend(fontsize=8, frameon=False)
    axes.set_xscale("log")
    axes.set_xlabel("branch rank")
    axes.set_ylabel(ylabel)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def write_compare_csv(path: str, nominal, derisked) -> None:
    """Every branch, ranked by nominal loading, nominal vs. de-risked side by side."""
    order = sorted(nominal, key=lambda i: nominal[i][5], reverse=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "branch", "from", "to", "limit",
                         "loading_nominal", "loading_derisked",
                         "Pf_nominal", "Pf_derisked",
                         "joule_loss_nominal", "joule_loss_derisked"])
        for rank, i in enumerate(order, start=1):
            f, t, limit, pf_n, _, load_n, loss_n = nominal[i]
            _, _, _, pf_d, _, load_d, loss_d = derisked[i]
            writer.writerow([rank, i, f, t, limit, load_n, load_d, pf_n, pf_d,
                             loss_n, loss_d])


def write_compare_plot(path: str, nominal, derisked, column: int, ylabel: str,
                       scale: float = 1.0, hline: float = None,
                       derisked_label: str = "de-risked") -> None:
    """`column` by rank, nominal and de-risked, each series sorted on its own value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = sorted((v[column] * scale for v in nominal.values()), reverse=True)
    d = sorted((v[column] * scale for v in derisked.values()), reverse=True)

    figure, axes = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    axes.plot(range(1, len(n) + 1), n, linewidth=1.5, color="#c0392b", label="nominal")
    axes.plot(range(1, len(d) + 1), d, linewidth=1.5, color="#1f4e79", label=derisked_label)
    if hline is not None:
        axes.axhline(hline, color="#8d99a6", linewidth=0.8, linestyle=":")
    axes.set_xscale("log")
    axes.set_xlabel("branch rank")
    axes.set_ylabel(ylabel)
    axes.legend(fontsize=8, frameon=False)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def output_dir(config: RunConfig) -> str:
    case = os.path.splitext(os.path.basename(config.case))[0]
    return os.path.join("results", f"{case}_{config.metric}_{config.stage}_flows")


def main() -> int:
    try:
        config = settings()
    except (ConfigError, FileNotFoundError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    outdir = OUTDIR or output_dir(config)
    os.makedirs(outdir, exist_ok=True)
    with Log(os.path.join(outdir, "run.log")) as log:
        try:
            network = read_matpower(config.case)
            nominal_config = replace(config.algorithm_config(),
                                     weight_multiplier=0.0, risk_weight=None)
            log("\n--- nominal (lambda = 0) ---\n")
            nominal = run(network, nominal_config, config.dc_solver(), config.ac_solver(), log)
            log("\n--- de-risked ---\n")
            derisked = run(network, config.algorithm_config(),
                           config.dc_solver(), config.ac_solver(), log)
        except CaseFormatError as exc:
            log(f"{exc}\n")
            return 2
        if nominal.dispatch is None or derisked.dispatch is None:
            log("no dispatch: nothing to write\n")
            return 1

        nominal_dir = os.path.join(outdir, "nominal")
        derisked_dir = os.path.join(outdir, "derisked")
        os.makedirs(nominal_dir, exist_ok=True)
        os.makedirs(derisked_dir, exist_ok=True)
        write_dispatch_csvs(nominal_dir, network, nominal.dispatch)
        write_dispatch_csvs(derisked_dir, network, derisked.dispatch)

        nominal_rows = branch_rows(network, nominal.dispatch)
        derisked_rows = branch_rows(network, derisked.dispatch)
        by_loading = [(i, *v) for i, v in
                     sorted(derisked_rows.items(), key=lambda kv: kv[1][5], reverse=True)]
        by_flow = [(i, *v) for i, v in
                  sorted(derisked_rows.items(), key=lambda kv: kv[1][4], reverse=True)]
        by_joule_loss = [(i, *v) for i, v in
                         sorted(derisked_rows.items(), key=lambda kv: kv[1][6], reverse=True)]
        write_csv(os.path.join(outdir, "flows_loading.csv"), by_loading)
        write_csv(os.path.join(outdir, "flows_flow.csv"), by_flow)
        write_csv(os.path.join(outdir, "flows_joule_loss.csv"), by_joule_loss)
        write_compare_csv(os.path.join(outdir, "flows_compare.csv"),
                          nominal_rows, derisked_rows)
        if PLOT:
            try:
                derisked_label = f"de-risked (q={config.q})"
                write_plot(os.path.join(outdir, "flows_loading.png"),
                          [row[6] * 100 for row in by_loading],
                          "loading (% of limit)", hline=100)
                write_plot(os.path.join(outdir, "flows_flow.png"),
                          [row[5] for row in by_flow], "flow, |Pf| (MW)")
                write_plot(os.path.join(outdir, "flows_joule_loss.png"),
                          [row[7] for row in by_joule_loss], "joule loss (MW)")
                write_compare_plot(os.path.join(outdir, "flows_compare_loading.png"),
                                   nominal_rows, derisked_rows, column=5,
                                   ylabel="loading (% of limit)", scale=100, hline=100,
                                   derisked_label=derisked_label)
                write_compare_plot(os.path.join(outdir, "flows_compare_flow.png"),
                                   nominal_rows, derisked_rows, column=4,
                                   ylabel="flow, |Pf| (MW)", derisked_label=derisked_label)
                write_compare_plot(os.path.join(outdir, "flows_compare_joule_loss.png"),
                                   nominal_rows, derisked_rows, column=6,
                                   ylabel="joule loss (MW)", derisked_label=derisked_label)
            except ImportError:
                log("no matplotlib: flows_*.png not written\n")
        log(f"written to {os.path.abspath(outdir)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

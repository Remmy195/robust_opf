"""The artifacts a study run leaves behind.

    solution_summary.json   the configuration, the provenance, and for each
                            weight the full trace and the dispatch P*
    efficient_frontier.csv  one row per weight, for reading and plotting
    efficient_frontier.json the same rows with the header block
    dispatch_{bus,gen,branch}.csv   every entity at every weight

Unit outputs are persisted in full, because the counterfactual takes P* as
given and a summary without them cannot be replayed.  Line flows are not, in
the summary: at 88,207 branches over six weights they would be most of the
file, so they go to the three dispatch tables instead, where the cost falls on
whoever opens them.

THE FRONTIER NORMALIZES WITHIN A STAGE.  Each stage's cost and risk columns are
given relative to that stage's own lambda = 0 row, not to the nominal solve
inside Algorithm 1.  On stage a2 those differ -- the algorithm's nominal solve
is on (M) while every reported dispatch is on (M^ac) -- so normalizing by the
algorithm's rho0 would attribute the model change to lambda.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import __version__, risk
from .algorithm import Result
from .config import RunConfig
from .network import Network

SOLUTION_SUMMARY = "solution_summary.json"
FRONTIER_CSV = "efficient_frontier.csv"
FRONTIER_JSON = "efficient_frontier.json"

DISPATCH_BUS_CSV = "dispatch_bus.csv"
DISPATCH_GEN_CSV = "dispatch_gen.csv"
DISPATCH_BRANCH_CSV = "dispatch_branch.csv"

#: The CSV columns, in order.  These tuples are the schema; the row builders
#: fill exactly these keys and the writer refuses anything else, because a CSV
#: whose columns depend on the run cannot be read back.
FRONTIER_COLUMNS = (
    "case", "n_buses", "n_branches", "n_gens",
    "metric", "stage", "flow_domain",
    "weight_multiplier", "risk_weight", "lambda_star",
    "cost", "risk", "surrogate", "gamma",
    "cost_ratio", "risk_ratio", "risk_reduction",
    "z0", "rho0", "ratio_vs_algorithm_nominal",
    "k_end", "n_line_cuts", "n_bus_cuts",
    "n_exposed", "n_overloaded",
    "termination", "status", "dominated",
    "solve_time_s", "total_time_s",
)

BUS_COLUMNS = (
    "weight_multiplier", "bus", "bus_id", "nodetype",
    "Pd_pu", "Qd_pu", "Vbase_kV", "Vmin_pu", "Vmax_pu",
    "v_pu", "theta_rad", "degree",
    "Pg_at_bus_pu", "Qg_at_bus_pu", "bus_flow_sum_pu",
    "is_risk_argmax", "high_exposure",
)

GEN_COLUMNS = (
    "weight_multiplier", "gen", "bus_id", "bus",
    "Pmin_pu", "Pmax_pu", "Qmin_pu", "Qmax_pu",
    "Pg_pu", "Qg_pu", "Pg_at_Pmax", "Pg_at_Pmin",
    "Pg_nominal_pu", "dPg_from_nominal_pu",
    "cost_quadratic", "cost_linear", "cost_constant", "gen_cost",
)

BRANCH_COLUMNS = (
    "weight_multiplier", "branch", "from_bus_id", "to_bus_id",
    "from_bus", "to_bus", "r", "x", "r_heat", "ratio", "angle_deg",
    "limit_pu", "rated", "Pf_pu", "Pt_pu", "Qf_pu", "Qt_pu",
    "abs_Pf_pu", "loading", "joule_pu",
    "Pf_nominal_pu", "dPf_from_nominal_pu",
    "is_risk_argmax", "high_exposure", "overloaded",
)


def _noop(_message: str) -> None:
    return None


###############################################################################
# Provenance
###############################################################################


def _git_commit() -> Optional[str]:
    """The commit this ran at, when the source is a git checkout."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def file_digest(path: Optional[str]) -> Optional[str]:
    """SHA-256 of the case file.  The ACTIVSg distributions are not versioned,
    so two copies of ``ACTIVSg2000.m`` can differ; this is what makes "the same
    case" checkable."""
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(network: Optional[Network] = None) -> Dict[str, Any]:
    """Enough to identify what produced a number, months later."""
    case = network.casefile if network else None
    return {
        "ropf_version": __version__,
        "git_commit": _git_commit(),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "case_file": case,
        "case_sha256": file_digest(case),
    }


###############################################################################
# Rows
###############################################################################


def frontier_rows(network: Network,
                  config: RunConfig,
                  results: Sequence[Result]) -> List[Dict[str, Any]]:
    """One row per weight, normalized within the stage.  See the module note."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        reported = result.reported
        dispatch = result.dispatch
        rows.append({
            "case": os.path.basename(network.casefile or "?"),
            "n_buses": network.numbuses,
            "n_branches": network.numbranches,
            "n_gens": network.numgens,
            "metric": result.metric,
            "stage": result.stage,
            "flow_domain": result.flow_domain,
            "weight_multiplier": result.weight_multiplier,
            "risk_weight": result.risk_weight,
            "lambda_star": result.lambda_star,
            "cost": reported.cost,
            "risk": reported.rho,
            "surrogate": reported.surrogate,
            "gamma": reported.gamma,
            "cost_ratio": float("nan"),
            "risk_ratio": float("nan"),
            "risk_reduction": float("nan"),
            "z0": result.z0,
            "rho0": result.rho0,
            "ratio_vs_algorithm_nominal": reported.ratio,
            "k_end": result.k_end,
            "n_line_cuts": reported.n_line_cuts,
            "n_bus_cuts": reported.n_bus_cuts,
            "n_exposed": reported.n_exposed,
            "n_overloaded": (len(risk.overloaded(network, dispatch.Pf))
                             if dispatch is not None and dispatch.Pf else 0),
            "termination": result.termination,
            "status": reported.status,
            "dominated": False,
            "solve_time_s": reported.solve_time_s,
            "total_time_s": result.total_time_s,
        })

    _normalize_within_stage(rows)
    _mark_dominated(rows)
    return rows


def _normalize_within_stage(rows: List[Dict[str, Any]]) -> None:
    """Fill the ratio columns from this stage's own lambda = 0 row.

    Left as NaN when the sweep has no lambda = 0 point.  A missing base is
    reported as missing rather than substituted with the algorithm's own rho0,
    which for stage a2 is on the other model.
    """
    base = next((r for r in rows if r["weight_multiplier"] == 0.0), None)
    if base is None or not base["cost"] or not base["risk"]:
        return
    for row in rows:
        row["cost_ratio"] = row["cost"] / base["cost"]
        row["risk_ratio"] = row["risk"] / base["risk"]
        row["risk_reduction"] = 1.0 - row["risk_ratio"]


#: Relative tolerance for the dominance comparison.  Looser than the LP solvers'
#: own convergence, tighter than any difference the weight grid produces: two
#: weights reaching the same dispatch differ in the tenth significant figure,
#: and an exact test would report that as one point dominating the other.
DOMINANCE_RTOL = 1e-7


def _mark_dominated(rows: List[Dict[str, Any]]) -> None:
    """Flag a point another beats on both axes.  A dominated point is real
    information -- an AC solve that found another local solution, or a loop that
    exited at a different k -- so it is flagged and kept, never dropped."""
    for row in rows:
        row["dominated"] = any(
            other is not row
            and _no_worse(other["cost"], row["cost"])
            and _no_worse(other["risk"], row["risk"])
            and (_better(other["cost"], row["cost"])
                 or _better(other["risk"], row["risk"]))
            for other in rows)


def _slack(value: float) -> float:
    return DOMINANCE_RTOL * max(1.0, abs(value))


def _no_worse(candidate: float, reference: float) -> bool:
    return candidate <= reference + _slack(reference)


def _better(candidate: float, reference: float) -> bool:
    return candidate < reference - _slack(reference)


###############################################################################
# Writing
###############################################################################


def write(outdir: str,
          network: Network,
          config: RunConfig,
          results: Sequence[Result],
          log: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
    """Write the artifacts into `outdir`.  Returns their paths by name."""
    emit = log or _noop
    os.makedirs(outdir, exist_ok=True)
    rows = frontier_rows(network, config, results)

    paths = {
        "solution_summary": os.path.join(outdir, SOLUTION_SUMMARY),
        "efficient_frontier_csv": os.path.join(outdir, FRONTIER_CSV),
        "efficient_frontier_json": os.path.join(outdir, FRONTIER_JSON),
    }
    header = {
        "provenance": provenance(network),
        "config": config.as_dict(),
        "network": {
            "buses": network.numbuses,
            "branches": network.numbranches,
            "generators": network.numgens,
            "baseMVA": network.baseMVA,
            "sumPd_MW": network.sumPd,
            "reference_bus_id": network.slackbus,
            "isolated_buses": network.numisolated,
            "branches_with_substituted_rating":
                len(network.unconstrained_branches()),
        },
    }

    _write_json(paths["solution_summary"],
                dict(header, runs=[_run_record(r) for r in results]))
    _write_json(paths["efficient_frontier_json"], dict(header, rows=rows))
    _write_csv(paths["efficient_frontier_csv"], rows)

    for name, path in paths.items():
        emit(f" wrote {name}: {path} ({os.path.getsize(path)} bytes)\n")

    paths.update(write_dispatch_detail(outdir, network, config, results, emit))
    return paths


def _run_record(result: Result) -> Dict[str, Any]:
    """One weight point, in full: trace, outcome, and the dispatch P*."""
    dispatch = result.dispatch
    record: Dict[str, Any] = {
        "metric": result.metric,
        "stage": result.stage,
        "flow_domain": result.flow_domain,
        "weight_multiplier": result.weight_multiplier,
        "risk_weight": result.risk_weight,
        "lambda_star": result.lambda_star,
        "z0": result.z0,
        "rho0": result.rho0,
        "phi0": result.phi0,
        "termination": result.termination,
        "termination_detail": result.termination_detail,
        "k_end": result.k_end,
        "total_time_s": result.total_time_s,
        "algorithm_config": asdict(result.config) if result.config else None,
        "iterations": [asdict(it) for it in result.iterations],
        "ac_stage": asdict(result.ac) if result.ac else None,
    }
    if dispatch is not None:
        # Keyed by generator count, as `ropf.network` indexes them, and written
        # as strings because JSON object keys are strings.  The counterfactual
        # reads this back as P*.
        record["dispatch"] = {
            "Pg_pu": {str(g): v for g, v in sorted(dispatch.Pg.items())},
            "max_abs_Pf_pu": max((abs(v) for v in dispatch.Pf.values()),
                                 default=0.0),
            "sum_Pg_pu": sum(dispatch.Pg.values()),
            "objective": dispatch.objective,
            "gen_cost": dispatch.gen_cost,
            "status": dispatch.status,
        }
    return record


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=_jsonable)
        handle.write("\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def _write_table(path: str, columns: Sequence[str],
                 rows: Sequence[Dict[str, Any]]) -> None:
    """The schema is the column tuple, not the rows."""
    for row in rows:
        unknown = set(row) - set(columns)
        if unknown:
            raise ValueError(
                f"{os.path.basename(path)} row carries columns outside the "
                f"declared schema: {sorted(unknown)}. Add them to the column "
                f"tuple or drop them; a CSV whose columns depend on the run "
                f"cannot be read back.")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """`_write_table` at the frontier schema."""
    _write_table(path, FRONTIER_COLUMNS, rows)


def read_dispatch(summary_path: str,
                  weight_multiplier: float) -> Dict[int, float]:
    """Read one pre-event dispatch P* back out of a summary, so a disfigurement
    campaign can be re-run or extended without repeating the dispatch."""
    with open(summary_path) as handle:
        summary = json.load(handle)
    for run in summary.get("runs", []):
        if run.get("weight_multiplier") == weight_multiplier:
            dispatch = run.get("dispatch")
            if not dispatch:
                raise KeyError(
                    f"{summary_path}: the run at lambda = {weight_multiplier} "
                    f"x lambda* carries no dispatch")
            return {int(g): float(v) for g, v in dispatch["Pg_pu"].items()}
    available = sorted(r.get("weight_multiplier") for r in summary.get("runs", []))
    raise KeyError(f"{summary_path}: no run at lambda = {weight_multiplier} x "
                   f"lambda*; the file holds {available}")


###############################################################################
# The dispatch, entity by entity
###############################################################################
#
# Nothing in this codebase reads these three files: they exist for a reader of
# the STUDY asking which units re-dispatched and which lines unloaded as lambda
# rose, which the frontier's one row per weight cannot answer.  Each file
# carries every weight, keyed by `weight_multiplier`, because the question is a
# comparison across lambda.


def write_dispatch_detail(outdir: str,
                          network: Network,
                          config: RunConfig,
                          results: Sequence[Result],
                          log: Optional[Callable[[str], None]] = None
                          ) -> Dict[str, str]:
    """Write the bus, generator and branch tables for every weight.

    The ``*_nominal_pu`` columns diff against THIS SWEEP'S lambda = 0 dispatch,
    the same base the frontier normalizes to.  A run that did not solve
    contributes no rows: an empty dispatch would be written as a system at
    rest, which is not what happened.
    """
    emit = log or _noop
    base = _base_dispatch(results)
    fraction = config.exposure_fraction

    bus_rows: List[Dict[str, Any]] = []
    gen_rows: List[Dict[str, Any]] = []
    branch_rows: List[Dict[str, Any]] = []

    for result in results:
        solution = result.dispatch
        if solution is None or not solution.Pg:
            continue
        weight = result.weight_multiplier
        # The functional's own component table, so "which bus/line was the
        # argmax" is answered by the code the separation used.  THE DOMAIN IS
        # PART OF THAT: omitting it defaulted this evaluation to all of E while
        # the separation ran over the rated set, and `is_risk_argmax` then
        # marked the unrestricted argmax beside a rated maximum.
        evaluation = risk.evaluate(result.metric, network, solution.Pf,
                                   result.flow_domain)
        argmax = evaluation.argmax
        exposed = set(risk.exposed(evaluation, fraction))
        overloaded = set(risk.overloaded(network, solution.Pf))
        bus_flow = (evaluation.components if evaluation.is_bus_family
                    else risk.evaluate("bus_flow_sum_agg", network,
                                       solution.Pf).components)
        is_bus = evaluation.is_bus_family

        bus_rows.extend(_bus_rows(network, solution, weight, bus_flow,
                                  argmax if is_bus else None,
                                  exposed if is_bus else set()))
        gen_rows.extend(_gen_rows(network, solution, weight, base))
        branch_rows.extend(_branch_rows(
            network, solution, weight, base, evaluation,
            argmax if not is_bus else None,
            set() if is_bus else exposed, overloaded))

    paths = {
        "dispatch_bus": os.path.join(outdir, DISPATCH_BUS_CSV),
        "dispatch_gen": os.path.join(outdir, DISPATCH_GEN_CSV),
        "dispatch_branch": os.path.join(outdir, DISPATCH_BRANCH_CSV),
    }
    for key, columns, rows in (
            ("dispatch_bus", BUS_COLUMNS, bus_rows),
            ("dispatch_gen", GEN_COLUMNS, gen_rows),
            ("dispatch_branch", BRANCH_COLUMNS, branch_rows)):
        _write_table(paths[key], columns, rows)
        emit(f" wrote {key}: {paths[key]} ({len(rows)} rows, "
             f"{os.path.getsize(paths[key])} bytes)\n")
    return paths


def _base_dispatch(results: Sequence[Result]):
    """The sweep's own lambda = 0 dispatch, or None when it has no such point."""
    for result in results:
        if result.weight_multiplier == 0.0 and result.dispatch is not None:
            return result.dispatch
    return None


def _bus_rows(network: Network, solution, weight, bus_flow,
              argmax: Optional[int], exposed) -> List[Dict[str, Any]]:
    rows = []
    for count, bus in network.buses.items():
        gens = bus.genidsbycount
        rows.append({
            "weight_multiplier": weight,
            "bus": count,
            "bus_id": bus.nodeID,
            "nodetype": bus.nodetype,
            "Pd_pu": bus.Pd,
            "Qd_pu": bus.Qd,
            "Vbase_kV": bus.Vbase,
            "Vmin_pu": bus.Vmin,
            "Vmax_pu": bus.Vmax,
            "v_pu": solution.v.get(count, ""),
            "theta_rad": solution.theta.get(count, ""),
            "degree": bus.degree,
            "Pg_at_bus_pu": sum(solution.Pg.get(g, 0.0) for g in gens),
            "Qg_at_bus_pu": sum(solution.Qg.get(g, 0.0) for g in gens),
            "bus_flow_sum_pu": bus_flow.get(count, 0.0),
            "is_risk_argmax": int(argmax is not None and count == argmax),
            "high_exposure": int(count in exposed),
        })
    return rows


def _gen_rows(network: Network, solution, weight, base) -> List[Dict[str, Any]]:
    rows = []
    for count, gen in network.gens.items():
        Pg = solution.Pg.get(count, 0.0)
        nominal = None if base is None else base.Pg.get(count)
        a, b, c = (list(gen.costvector) + [0.0, 0.0, 0.0])[:3]
        rows.append({
            "weight_multiplier": weight,
            "gen": count,
            "bus_id": gen.nodeID,
            "bus": network.id_to_count.get(gen.nodeID, ""),
            "Pmin_pu": gen.Pmin,
            "Pmax_pu": gen.Pmax,
            "Qmin_pu": gen.Qmin,
            "Qmax_pu": gen.Qmax,
            "Pg_pu": Pg,
            "Qg_pu": solution.Qg.get(count, ""),
            "Pg_at_Pmax": int(_at_bound(Pg, gen.Pmax)),
            "Pg_at_Pmin": int(_at_bound(Pg, gen.Pmin)),
            "Pg_nominal_pu": "" if nominal is None else nominal,
            "dPg_from_nominal_pu": "" if nominal is None else Pg - nominal,
            "cost_quadratic": a,
            "cost_linear": b,
            "cost_constant": c,
            "gen_cost": a * Pg * Pg + b * Pg + c,
        })
    return rows


def _branch_rows(network: Network, solution, weight, base, evaluation,
                 argmax: Optional[int], exposed, overloaded
                 ) -> List[Dict[str, Any]]:
    joule = (evaluation.components if evaluation.metric == "joule_loss_max"
             else None)
    rows = []
    for count, branch in network.branches.items():
        Pf = solution.Pf.get(count, 0.0)
        nominal = None if base is None else base.Pf.get(count)
        rows.append({
            "weight_multiplier": weight,
            "branch": count,
            "from_bus_id": branch.id_f,
            "to_bus_id": branch.id_t,
            "from_bus": branch.f,
            "to_bus": branch.t,
            "r": branch.r,
            "x": branch.x,
            "r_heat": branch.r_heat,
            "ratio": branch.ratio,
            "angle_deg": branch.angle,
            "limit_pu": branch.limit,
            # 0 where the rating was the big-M substitution of `ropf.network`,
            # so a loading of 0.01 there reads as "no rating given".
            "rated": branch.constrainedflow,
            "Pf_pu": Pf,
            "Pt_pu": solution.Pt.get(count, ""),
            "Qf_pu": solution.Qf.get(count, ""),
            "Qt_pu": solution.Qt.get(count, ""),
            "abs_Pf_pu": abs(Pf),
            "loading": (abs(Pf) / branch.limit) if branch.limit else "",
            "joule_pu": (joule.get(count) if joule is not None
                         else branch.r_heat * Pf * Pf),
            "Pf_nominal_pu": "" if nominal is None else nominal,
            "dPf_from_nominal_pu": "" if nominal is None else Pf - nominal,
            "is_risk_argmax": int(argmax is not None and count == argmax),
            # Bad under the run's own functional, and bad against its own
            # rateA.  Two different questions, so two columns.
            "high_exposure": int(count in exposed),
            "overloaded": int(count in overloaded),
        })
    return rows


#: Relative tolerance for calling a unit pinned at a bound.  Looser than the LP
#: solver's own feasibility tolerance, tighter than any redispatch.
BOUND_RTOL = 1e-6


def _at_bound(value: float, bound: float) -> bool:
    return abs(value - bound) <= BOUND_RTOL * max(1.0, abs(bound))

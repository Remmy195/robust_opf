"""The three artifacts a study run leaves behind, and nothing else.

    solution_summary.json   everything about one sweep: the configuration, the
                            provenance, and for each weight the full iteration
                            trace and the dispatch P* the counterfactual needs.
    efficient_frontier.csv  one row per weight, for reading and for plotting.
    efficient_frontier.json the same rows with the header block, so a reader
                            holding only this file can regenerate the CSV.

An earlier codebase wrote spreadsheets, per-iteration dumps and a plotting stack
several thousand lines long.  The study reads three files.  Anything a fourth
file would carry either belongs in one of these or is not a result -- the run
transcript the command line writes is a transcript, not a result, and nothing
reads it.

WHAT IS AND IS NOT PERSISTED.  Unit outputs are, in full: the counterfactual
takes the pre-event dispatch P* as given, so a summary without them cannot be
replayed.  Line flows are not: nothing downstream reads them, they are
recoverable from the angles and the network, and at 88,207 branches over six
weights they would be most of the file.  What is kept of them is the functional
they were scored by and where its maximum sat.

THE FRONTIER NORMALIZES WITHIN A STAGE.  Each stage's cost and risk columns are
also given relative to that stage's own lambda = 0 row, not to the nominal solve
inside Algorithm 1.  On stage a2 those differ: the algorithm's nominal solve is
on (M) while every reported dispatch is on (M^ac), so normalizing by the
algorithm's rho0 would report an AC dispatch against a DC base and attribute the
model change to lambda.
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

from . import __version__
from .algorithm import Iterate, Result
from .config import RunConfig
from .network import Network

SOLUTION_SUMMARY = "solution_summary.json"
FRONTIER_CSV = "efficient_frontier.csv"
FRONTIER_JSON = "efficient_frontier.json"

#: The CSV columns, in order.  This tuple is the schema; the row builder below
#: fills exactly these keys and the writer asserts nothing else appears.
FRONTIER_COLUMNS = (
    "case", "n_buses", "n_branches", "n_gens",
    "metric", "stage",
    "weight_multiplier", "risk_weight", "lambda_star",
    "cost", "risk", "surrogate", "gamma",
    "cost_ratio", "risk_ratio", "risk_reduction",
    "z0", "rho0", "ratio_vs_algorithm_nominal",
    "k_end", "n_line_cuts", "n_bus_cuts",
    "termination", "status", "dominated",
    "solve_time_s", "total_time_s",
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


def _file_digest(path: Optional[str]) -> Optional[str]:
    """SHA-256 of the case file.

    A reproducibility claim rests on the inputs being the same file, and the
    ACTIVSg distributions are not versioned: two copies of ``ACTIVSg2000.m`` can
    differ.  The digest is what makes "the same case" checkable.
    """
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
        "case_sha256": _file_digest(case),
    }


###############################################################################
# Rows
###############################################################################


def _iterate_row(it: Iterate) -> Dict[str, Any]:
    return asdict(it)


def frontier_rows(network: Network,
                  config: RunConfig,
                  results: Sequence[Result]) -> List[Dict[str, Any]]:
    """One row per weight, normalized within the stage.  See the module note."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        reported = result.reported
        rows.append({
            "case": os.path.basename(network.casefile or "?"),
            "n_buses": network.numbuses,
            "n_branches": network.numbranches,
            "n_gens": network.numgens,
            "metric": result.metric,
            "stage": result.stage,
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

    Left as NaN when the sweep has no lambda = 0 point -- an absolute
    ``risk_weight`` produces a single run, and there is then no base to divide
    by.  A missing base is reported as missing rather than substituted with the
    algorithm's own rho0, which for stage a2 is on the other model.
    """
    base = next((r for r in rows if r["weight_multiplier"] == 0.0), None)
    if base is None or not base["cost"] or not base["risk"]:
        return
    for row in rows:
        row["cost_ratio"] = row["cost"] / base["cost"]
        row["risk_ratio"] = row["risk"] / base["risk"]
        row["risk_reduction"] = 1.0 - row["risk_ratio"]


def _mark_dominated(rows: List[Dict[str, Any]]) -> None:
    """Flag a point another point beats on both axes.

    Each point solves its own weighted problem, so in exact arithmetic the set
    is a frontier.  A dominated point is real information -- an AC solve that
    found a different local solution, or a loop that exited at a different k --
    so it is flagged and kept, never dropped.

    The comparison is tolerant on purpose.  Two weights that reach the same
    dispatch return costs that differ in the tenth significant figure, and an
    exact test reports that difference as one point dominating the other, which
    is a claim about the study where the truth is a claim about the LP solver.
    """
    for row in rows:
        row["dominated"] = any(
            other is not row
            and _no_worse(other["cost"], row["cost"])
            and _no_worse(other["risk"], row["risk"])
            and (_better(other["cost"], row["cost"])
                 or _better(other["risk"], row["risk"]))
            for other in rows)


#: Relative tolerance for the dominance comparison.  Looser than the LP solvers'
#: own convergence, tighter than any difference the weight grid produces.
DOMINANCE_RTOL = 1e-7


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
    """Write the three artifacts into `outdir`.  Returns their paths by name."""
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
    return paths


def _run_record(result: Result) -> Dict[str, Any]:
    """One weight point, in full: trace, outcome, and the dispatch P*."""
    dispatch = result.dispatch
    record: Dict[str, Any] = {
        "metric": result.metric,
        "stage": result.stage,
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
        "iterations": [_iterate_row(it) for it in result.iterations],
        "ac_stage": _iterate_row(result.ac) if result.ac else None,
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


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    for row in rows:
        unknown = set(row) - set(FRONTIER_COLUMNS)
        if unknown:
            raise ValueError(
                f"frontier row carries columns outside the declared schema: "
                f"{sorted(unknown)}. Add them to FRONTIER_COLUMNS or drop them; "
                f"a CSV whose columns depend on the run cannot be read back.")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FRONTIER_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in FRONTIER_COLUMNS})


def read_dispatch(summary_path: str,
                  weight_multiplier: float) -> Dict[int, float]:
    """Read one pre-event dispatch P* back out of a summary.

    The counterfactual runs from the summary rather than from a live solve, so
    that a disfigurement campaign can be re-run, extended, or re-run at a
    different K without repeating the dispatch it is evaluating.
    """
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

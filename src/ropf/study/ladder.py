"""The ladder study: six rungs, three functionals, three stages.

THE UNIT OF WORK IS ONE COMBO -- one (rung, metric, stage) triple.  A combo
runs Algorithm 1 across the weight grid, writes the three artifacts, then runs
the Section 4 counterfactual campaign against the dispatches that sweep
produced.  One invocation claims one combo, runs it, and exits.

    ropf ladder configs/ladder.conf --dry-run    # what would run, and how big
    ropf ladder configs/ladder.conf              # claim one combo and run it
    ropf ladder configs/ladder.conf --all        # keep going until none left
    ropf ladder configs/ladder.conf --status     # the progress table

WHY ONE COMBO PER INVOCATION.  The combos are independent -- nothing a combo
computes is read by another -- so the whole ladder parallelizes by running this
command in as many shells as there are cores to spare, and it resumes after a
crash by being run again.  Both of those follow from the claim, and neither
needs a scheduler.

    A combo directory is claimed by creating `.claim/` inside it, which is
    atomic on POSIX: two processes racing for the same combo, one wins and the
    other moves on.  A combo is finished when `DONE` exists, which is written
    last, after every artifact.  A directory with a claim and no DONE is either
    running now or died; `--reclaim` releases the ones older than
    `stale_claim_hours`, which is a number and therefore lives in the config,
    not in a flag.

THE RANKING IS TAKEN ONCE, AT LAMBDA = 0.  Section 4.1.1's top-K class ranks
units on the NOMINAL dispatch and holds that list fixed across the whole
campaign.  Re-ranking per dispatch would flatter the de-risked ones, whose
top-K carries less power by construction -- the campaign would be scoring each
dispatch against an adversary that had been told to go easy on it.

THE DISFIGUREMENTS ARE BUILT ONCE PER COMBO, before any dispatch is evaluated,
and every dispatch faces the same list.  The random walks are drawn from a
declared seed for the same reason: two dispatches compared on different
disfigurements are not compared at all.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import algorithm, results as results_module
from ..algorithm import STAGES, WEIGHT_GRID
from ..config import ConfigError, RunConfig, _build, parse_keyfile
from ..counterfactual import disfigure, dynamics, frequency, postevent
from ..counterfactual.postevent import Disfigurement, PostEventParams
from ..data.fetch import DATA_DIR, LADDER, SOURCES
from ..log import Log
from ..model import SolverConfig
from ..network import Network, read_matpower
from ..risk import METRICS

#: Written last, after every artifact.  Its presence is the only thing that
#: means a combo is finished.
DONE = "DONE"
CLAIM = ".claim"
CAMPAIGN_JSON = "campaign.json"
CAMPAIGN_CSV = "campaign.csv"
TRANSCRIPT = "run.log"

CAMPAIGN_COLUMNS = (
    "weight_multiplier", "risk_weight", "gamma", "disfigurement", "kind", "K",
    "outcome", "reason", "lost_load_pu", "removed_demand_pu", "cost",
    "phase", "n_solves", "solve_time_s",
    "cleared", "failed_test", "rocof_hz_s", "nadir_hz", "H_sys_s",
    "delta_P_pu", "n_responsive", "n_buses", "n_branches", "n_gens",
)


def _noop(_message: str) -> None:
    return None


###############################################################################
# The key table
###############################################################################


@dataclass
class LadderConfig:
    """Every key a ladder configuration file may set.

    Same discipline as `ropf.config.RunConfig`: the field list IS the key
    table, an unknown key is an error and a duplicate key is an error.  Both
    refusals come from `ropf.config.parse_keyfile`, which is the only place
    they are implemented.
    """

    # --- the grid ------------------------------------------------------------
    #: Rungs to run, by the names in `ropf.data.fetch.SOURCES`.
    rungs: Tuple[str, ...] = LADDER
    metrics: Tuple[str, ...] = METRICS
    stages: Tuple[str, ...] = STAGES

    # --- Algorithm 1 ---------------------------------------------------------
    kappa: int = 1
    eta: float = 0.5
    k_bar: int = 25
    weight_grid: Tuple[float, ...] = WEIGHT_GRID

    # --- Section 4: the campaign ---------------------------------------------
    #: Run the counterfactual campaign at all.  Off gives the frontier only.
    campaign: bool = True
    #: Class (a), Section 4.1.1: disable the K highest-output units.
    gen_k: Tuple[int, ...] = (1, 2, 3, 5, 8, 12, 18, 25)
    #: Class (b), Section 4.1.2: random-walk component sets of size K.
    walk_k: Tuple[int, ...] = (5, 15, 40)
    walk_draws: int = 100
    #: The walk seed.  Declared, so the campaign is reproducible.
    walk_seed: int = 20260820
    #: gamma, the response-window scale of (D).  A LIST: gamma is a declared
    #: study axis and not a measured quantity, so it is swept.  Every value
    #: multiplies the campaign, which `--dry-run` reports before anything runs.
    gamma: Tuple[float, ...] = (0.5,)
    #: beta, the emergency rating factor of eq (6e).  Above 1 needs a cited
    #: source: rateB and rateC are zero across the ACTIVSg distributions.
    beta: float = 1.0

    # --- Section 4.2: the frequency screen -----------------------------------
    #: None of these four is defaulted.  `frequency.ScreenConfig` refuses to
    #: run without them rather than supply a plausible number that would then
    #: be reported as if it were data; the config file inherits that refusal.
    f0_hz: Optional[float] = None
    rocof_max_hz_s: Optional[float] = None
    f_under_hz: Optional[float] = None
    #: The load damping, on ONE of the two bases -- set exactly one.  They
    #: differ by the load-to-baseMVA ratio, which is 671 on ACTIVSg2000, and
    #: getting it wrong fails quietly: the nadir just comes out deeper.
    damping_per_load: Optional[float] = None
    damping_on_system: Optional[float] = None
    horizon_s: float = 30.0
    dt_s: float = 0.01
    #: Directories searched for each rung's .dyr and .aux, comma separated.
    #: The ACTIVSg dynamics ship alongside the cases and are not in data/.
    dynamics_search: str = ""

    # --- solvers -------------------------------------------------------------
    solver_dc: str = "gurobi"
    solver_ac: str = "knitro"
    time_limit_s: float = 3600.0
    threads: int = 40
    solver_verbose: bool = False
    #: Crossover for (D).  0 disables it; see `ropf.model.SolverConfig`, where
    #: the measurement behind this is recorded.  It is worth setting only at
    #: the top rung, and it changes the per-bus shed pattern, not the total.
    postevent_crossover: Optional[int] = None

    # --- output --------------------------------------------------------------
    outdir: str = os.path.join("results", "ladder")
    #: How old a claim must be before `--reclaim` releases it.
    stale_claim_hours: float = 24.0

    def __post_init__(self) -> None:
        for rung in self.rungs:
            if rung not in SOURCES:
                raise ConfigError(
                    f"rungs names {rung!r}, which is not a known rung; known: "
                    f"{', '.join(LADDER)}")
        for metric in self.metrics:
            if metric not in METRICS:
                raise ConfigError(f"metrics names {metric!r}, which is not one "
                                  f"of {list(METRICS)}")
        for stage in self.stages:
            if stage not in STAGES:
                raise ConfigError(f"stages names {stage!r}, which is not one "
                                  f"of {list(STAGES)}")
        if not (self.rungs and self.metrics and self.stages):
            raise ConfigError("rungs, metrics and stages must each name at "
                              "least one value; there is nothing to run")
        if self.campaign:
            if not self.gamma:
                raise ConfigError("gamma is empty; (D) has no response window")
            if any(k < 1 for k in self.gen_k):
                raise ConfigError(f"gen_k = {list(self.gen_k)} carries a K "
                                  f"below 1")
            if any(k < 1 for k in self.walk_k):
                raise ConfigError(f"walk_k = {list(self.walk_k)} carries a K "
                                  f"below 1")
            if self.walk_draws < 1:
                raise ConfigError(f"walk_draws = {self.walk_draws} must be at "
                                  f"least 1")
        if (self.damping_per_load is None) == (self.damping_on_system is None):
            raise ConfigError(
                "set exactly one of damping_per_load and damping_on_system. "
                "The textbook 1-2 %/% figure is on the LOAD base and the swing "
                "equation needs the SYSTEM base; the two differ by the "
                "load-to-baseMVA ratio, so a damping without its base is not a "
                "damping.")

    # -- derived objects ------------------------------------------------------

    def damping(self) -> frequency.LoadDamping:
        if self.damping_per_load is not None:
            return frequency.LoadDamping.per_load(self.damping_per_load)
        return frequency.LoadDamping.on_system_base(self.damping_on_system)

    def screen_config(self) -> frequency.ScreenConfig:
        """The Section 4.2 constants.  Raises if the config left one unset."""
        missing = [name for name in ("f0_hz", "rocof_max_hz_s", "f_under_hz")
                   if getattr(self, name) is None]
        if missing:
            raise ConfigError(
                f"the frequency screen needs {', '.join(missing)}, and none of "
                f"them is defaulted: they are grid-code quantities and belong "
                f"in the config with a citation. Set them, or set "
                f"campaign = false to run the frontier only.")
        return frequency.ScreenConfig(
            f0_hz=self.f0_hz, rocof_max_hz_s=self.rocof_max_hz_s,
            f_under_hz=self.f_under_hz, damping=self.damping(),
            horizon_s=self.horizon_s, dt_s=self.dt_s)

    def dynamics_dirs(self) -> List[str]:
        return [d.strip() for d in self.dynamics_search.split(",") if d.strip()]

    def run_config(self, combo: "Combo", case: str) -> RunConfig:
        """The `RunConfig` this combo is, so the frontier machinery is reused."""
        return RunConfig(
            case=case, metric=combo.metric, stage=combo.stage,
            kappa=self.kappa, eta=self.eta, k_bar=self.k_bar,
            weight_grid=self.weight_grid,
            solver_dc=self.solver_dc, solver_ac=self.solver_ac,
            time_limit_s=self.time_limit_s, threads=self.threads,
            solver_verbose=self.solver_verbose,
            outdir=self.outdir, tag=combo.name)

    def postevent_solver(self) -> SolverConfig:
        return SolverConfig(name=self.solver_dc,
                            time_limit_s=self.time_limit_s,
                            gurobi_crossover=self.postevent_crossover,
                            gurobi_threads=self.threads,
                            knitro_threads=self.threads,
                            verbose=self.solver_verbose)

    @property
    def n_disfigurements(self) -> int:
        return len(self.gen_k) + len(self.walk_k) * self.walk_draws

    def evaluations_per_combo(self) -> int:
        if not self.campaign:
            return 0
        return (len(self.weight_grid) * len(self.gamma)
                * self.n_disfigurements)

    def as_dict(self) -> Dict[str, Any]:
        record = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in list(record.items()):
            if isinstance(value, tuple):
                record[key] = list(value)
        return record


def read_ladder_config(path: str) -> LadderConfig:
    """Parse a ladder configuration file.  Same two refusals as a run config."""
    return _build(parse_keyfile(path, LadderConfig), path, LadderConfig)


###############################################################################
# Combos
###############################################################################


@dataclass(frozen=True)
class Combo:
    """One unit of work: a rung, a functional and a stage."""

    rung: str
    metric: str
    stage: str

    @property
    def name(self) -> str:
        return f"{self.rung}_{self.metric}_{self.stage}"

    @property
    def distribution(self) -> str:
        """The TAMU distribution name, e.g. ACTIVSg2000.  Not the rung name."""
        return SOURCES[self.rung].rung

    def case_path(self, data_dir: str = DATA_DIR) -> str:
        return os.path.join(data_dir, SOURCES[self.rung].case)


def combos(config: LadderConfig) -> List[Combo]:
    """Every combo the config declares, cheap rungs first.

    The rung order is the ladder's own, not the config's, so a partial run
    covers the cheap rungs first however the file happens to list them.
    """
    order = {name: i for i, name in enumerate(LADDER)}
    return [Combo(rung, metric, stage)
            for rung in sorted(config.rungs, key=lambda r: order[r])
            for metric in config.metrics
            for stage in config.stages]


###############################################################################
# Claiming
###############################################################################


def combo_dir(config: LadderConfig, combo: Combo) -> str:
    return os.path.join(config.outdir, combo.name)


def is_done(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, DONE))


def claim_age_hours(directory: str) -> Optional[float]:
    """How long the claim on `directory` has stood, or None if unclaimed."""
    claim = os.path.join(directory, CLAIM)
    if not os.path.isdir(claim):
        return None
    return (time.time() - os.path.getmtime(claim)) / 3600.0


def claim(directory: str) -> bool:
    """Take the claim on a combo directory, atomically.

    `os.mkdir` either creates the directory or raises: there is no window in
    which two processes both believe they created it, which is the whole
    reason the claim is a directory and not a file that gets written.
    """
    os.makedirs(directory, exist_ok=True)
    try:
        os.mkdir(os.path.join(directory, CLAIM))
    except FileExistsError:
        return False
    with open(os.path.join(directory, CLAIM, "who"), "w") as handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "python": platform.python_version()}, handle, indent=2)
    return True


def release(directory: str) -> None:
    """Drop a claim.  Used by --reclaim and when a combo fails."""
    claim_dir = os.path.join(directory, CLAIM)
    who = os.path.join(claim_dir, "who")
    if os.path.isfile(who):
        os.remove(who)
    if os.path.isdir(claim_dir):
        os.rmdir(claim_dir)


def next_combo(config: LadderConfig,
               log: Callable[[str], None] = _noop) -> Optional[Combo]:
    """Claim and return the next combo with work left, or None."""
    for combo in combos(config):
        directory = combo_dir(config, combo)
        if is_done(directory):
            continue
        if claim(directory):
            return combo
        log(f" {combo.name}: claimed by another process, skipping\n")
    return None


def reclaim_stale(config: LadderConfig,
                  log: Callable[[str], None] = _noop) -> List[Combo]:
    """Release claims older than `stale_claim_hours` on unfinished combos."""
    released = []
    for combo in combos(config):
        directory = combo_dir(config, combo)
        if is_done(directory):
            continue
        age = claim_age_hours(directory)
        if age is not None and age >= config.stale_claim_hours:
            release(directory)
            released.append(combo)
            log(f" released {combo.name}, claimed {age:.1f}h ago\n")
    return released


def status_rows(config: LadderConfig) -> List[Dict[str, Any]]:
    rows = []
    for combo in combos(config):
        directory = combo_dir(config, combo)
        age = claim_age_hours(directory)
        rows.append({
            "combo": combo.name,
            "state": ("done" if is_done(directory)
                      else "running" if age is not None else "pending"),
            "claim_age_h": None if age is None else round(age, 2),
        })
    return rows


###############################################################################
# The campaign
###############################################################################


@dataclass
class Campaign:
    """The disfigurements and the screen, built once and reused per dispatch."""

    disfigurements: List[Disfigurement]
    kinds: Dict[str, str]
    screen_config: Optional[frequency.ScreenConfig]
    unit_dynamics: Dict[int, Any] = field(default_factory=dict)
    participation: Dict[int, float] = field(default_factory=dict)
    participation_source: str = ""
    dynamics_source: str = ""
    #: What the .dyr reader accepted and rejected, per model.  Recorded because
    #: a rejected record is a unit with no inertia, and a campaign run against
    #: a file that mostly failed to parse would otherwise look normal.
    dynamics_report: Optional[Any] = None


def build_campaign(config: LadderConfig,
                   combo: Combo,
                   network: Network,
                   nominal_Pg: Dict[int, float],
                   log: Callable[[str], None] = _noop) -> Campaign:
    """Everything the campaign needs that does not depend on the dispatch.

    The ranking is taken here, from the nominal dispatch, and is never retaken.
    """
    ranking = disfigure.rank_units(nominal_Pg)
    disfigurements = [disfigure.top_k_units(ranking, k) for k in config.gen_k]
    kinds = {d.label: "top_k" for d in disfigurements}

    for K in config.walk_k:
        # A seed per K, so two campaigns at different K do not consume each
        # other's stream and adding a K does not move the existing ones.
        drawn = disfigure.walk_draws(network, K, config.walk_draws,
                                     seed=config.walk_seed + K)
        for index, item in enumerate(drawn):
            labelled = Disfigurement(buses=item.buses, branches=item.branches,
                                     gens=item.gens,
                                     label=f"walk{K}_{index:03d}")
            disfigurements.append(labelled)
            kinds[labelled.label] = "walk"
    log(f" {len(disfigurements)} disfigurements "
        f"({len(config.gen_k)} top-K, "
        f"{len(config.walk_k) * config.walk_draws} walks, "
        f"seed {config.walk_seed})\n")

    screen_config = config.screen_config()
    files = dynamics.locate(combo.distribution, config.dynamics_dirs())
    unit_dyn, participation, pi_source, source = {}, {}, "", "none"
    dynamics_report = None

    if files.dyr:
        records, report = dynamics.parse_dyr(files.dyr, files.archive, log)
        unit_dyn = dynamics.unit_inertia(network, records, log)
        source = report.path
        dynamics_report = report
    else:
        raise ConfigError(
            f"no .dyr found for {combo.distribution} under "
            f"{config.dynamics_dirs() or '(dynamics_search is empty)'}. The "
            f"frequency screen needs it: without inertia every surviving "
            f"system has H_sys = 0 and the screen would fail every "
            f"disfigurement for the wrong reason. Set dynamics_search, or set "
            f"campaign = false.")

    aux = None
    if files.aux:
        aux = dynamics.parse_participation(files.aux, files.archive, log)
    participation, pi_source = dynamics.participation_factors(network, aux, log)

    return Campaign(disfigurements=disfigurements, kinds=kinds,
                    screen_config=screen_config, unit_dynamics=unit_dyn,
                    participation=participation, participation_source=pi_source,
                    dynamics_source=source, dynamics_report=dynamics_report)


def _screen_for(network: Network, campaign: Campaign,
                disfigurement: Disfigurement, P_star: Dict[int, float],
                sink: List[frequency.ScreenResult]) -> Callable:
    """The Section 4.2 screen as (D) wants it: live gens in, cleared/why out.

    `evaluate` hands the callback the surviving generators only, and the screen
    needs the surviving buses too, so the closure recomputes them from the
    disfigurement it was built for.  The full `ScreenResult` goes into `sink`,
    because the campaign records the nadir and the RoCoF, not just the verdict.
    """
    def _screen(live_gens):
        live_buses, _, _ = postevent.survivors(network, disfigurement)
        result = frequency.screen(campaign.screen_config, network,
                                  campaign.unit_dynamics, live_buses,
                                  live_gens, P_star)
        sink.append(result)
        return result.cleared, result.reason
    return _screen


def evaluate_campaign(config: LadderConfig,
                      network: Network,
                      campaign: Campaign,
                      runs: Sequence[algorithm.Result],
                      log: Callable[[str], None] = _noop) -> List[Dict]:
    """Every (weight, gamma, disfigurement) against (D).  One row each."""
    evaluator = postevent.PostEvent(network, config.postevent_solver(), log)
    rows: List[Dict[str, Any]] = []
    started = time.time()
    total = len(runs) * len(config.gamma) * len(campaign.disfigurements)
    done = 0

    for run in runs:
        if run.dispatch is None or not run.dispatch.Pg:
            log(f" lambda x{run.weight_multiplier}: no dispatch to evaluate "
                f"({run.termination}); skipped\n")
            continue
        P_star = run.dispatch.Pg
        for gamma in config.gamma:
            params = PostEventParams(gamma=gamma, beta=config.beta)
            for item in campaign.disfigurements:
                sink: List[frequency.ScreenResult] = []
                result = evaluator.evaluate(
                    P_star, campaign.participation, params, item,
                    _screen_for(network, campaign, item, P_star, sink))
                rows.append(_campaign_row(run, gamma, item, campaign, result,
                                          sink[0] if sink else None))
                done += 1
                if done % 250 == 0:
                    rate = done / max(time.time() - started, 1e-9)
                    log(f"   {done}/{total} evaluations, {rate:.1f}/s, "
                        f"{(total - done) / max(rate, 1e-9) / 60:.1f} min left\n")

    log(f" campaign: {len(rows)} evaluations in "
        f"{time.time() - started:.1f}s\n")
    return rows


def _campaign_row(run: algorithm.Result, gamma: float,
                  item: Disfigurement, campaign: Campaign,
                  result: postevent.PostEventResult,
                  screen: Optional[frequency.ScreenResult]) -> Dict[str, Any]:
    return {
        "weight_multiplier": run.weight_multiplier,
        "risk_weight": run.risk_weight,
        "gamma": gamma,
        "disfigurement": item.label,
        "kind": campaign.kinds.get(item.label, ""),
        "K": item.size or len(item.gens),
        "outcome": result.outcome,
        "reason": result.reason,
        "lost_load_pu": result.lost_load,
        "removed_demand_pu": result.removed_demand_pu,
        "cost": result.cost,
        "phase": result.phase,
        "n_solves": result.n_solves,
        "solve_time_s": round(result.solve_time_s, 4),
        "cleared": None if screen is None else screen.cleared,
        "failed_test": "" if screen is None else screen.failed_test,
        "rocof_hz_s": None if screen is None else screen.rocof_hz_s,
        "nadir_hz": None if screen is None else screen.nadir_hz,
        "H_sys_s": None if screen is None else screen.H_sys_s,
        "delta_P_pu": None if screen is None else screen.delta_P_pu,
        "n_responsive": None if screen is None else screen.n_responsive,
        "n_buses": result.n_buses,
        "n_branches": result.n_branches,
        "n_gens": result.n_gens,
    }


###############################################################################
# Running one combo
###############################################################################


def run_combo(config: LadderConfig, combo: Combo,
              log: Callable[[str], None] = _noop) -> Dict[str, Any]:
    """One combo end to end: the frontier, then the campaign, then DONE."""
    directory = combo_dir(config, combo)
    case = combo.case_path()
    if not os.path.isfile(case):
        raise FileNotFoundError(
            f"{combo.rung}: {case} is not there. Run `ropf fetch "
            f"{combo.rung}` to see how to obtain it.")

    started = time.time()
    network = read_matpower(case, log)
    run_config = config.run_config(combo, case)
    dc_solver, ac_solver = run_config.dc_solver(), run_config.ac_solver()

    runs = []
    for multiplier in run_config.weights:
        log(f"\n lambda = {multiplier:g} lambda*   [{combo.name}]\n")
        runs.append(algorithm.run(network, run_config.algorithm_config(
            multiplier), dc_solver, ac_solver, log))

    results_module.write(directory, network, run_config, runs, log)

    record: Dict[str, Any] = {
        "combo": {"rung": combo.rung, "metric": combo.metric,
                  "stage": combo.stage, "distribution": combo.distribution},
        "provenance": results_module.provenance(network),
        "config": config.as_dict(),
        "frontier_runs": len(runs),
    }

    if config.campaign:
        nominal = next((r.nominal_dispatch for r in runs
                        if r.nominal_dispatch is not None), None)
        if nominal is None:
            raise RuntimeError(
                "no nominal dispatch was produced, so the top-K ranking has "
                "nothing to rank; the campaign cannot run")
        campaign = build_campaign(config, combo, network, nominal.Pg, log)
        rows = evaluate_campaign(config, network, campaign, runs, log)
        record.update({
            "dynamics_source": campaign.dynamics_source,
            "dynamics_report": (None if campaign.dynamics_report is None
                                else vars(campaign.dynamics_report)),
            "participation_source": campaign.participation_source,
            "screen": {"f0_hz": config.f0_hz,
                       "rocof_max_hz_s": config.rocof_max_hz_s,
                       "f_under_hz": config.f_under_hz,
                       "damping": campaign.screen_config.damping.describe(
                           network),
                       "horizon_s": config.horizon_s, "dt_s": config.dt_s},
            "n_disfigurements": len(campaign.disfigurements),
            "n_evaluations": len(rows),
        })
        _write_json(os.path.join(directory, CAMPAIGN_JSON),
                    dict(record, rows=rows))
        _write_csv(os.path.join(directory, CAMPAIGN_CSV), rows)
        log(f" wrote {CAMPAIGN_JSON} and {CAMPAIGN_CSV} ({len(rows)} rows)\n")

    record["total_time_s"] = round(time.time() - started, 1)
    with open(os.path.join(directory, DONE), "w") as handle:
        json.dump(record, handle, indent=2, default=str)
    return record


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    import csv
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CAMPAIGN_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

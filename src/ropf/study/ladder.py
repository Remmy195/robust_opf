"""The ladder study: six instances, three functionals, three stages.

THE UNIT OF WORK IS ONE COMBO -- one (instance, metric, stage) triple.  A combo
runs Algorithm 1 across the weight grid, writes the artifacts, then runs the
Section 4 campaign against the dispatches that sweep produced.  One invocation
claims one combo, runs it, and exits.

    ropf ladder configs/ladder.conf --dry-run    # what would run, and how big
    ropf ladder configs/ladder.conf              # claim one combo and run it
    ropf ladder configs/ladder.conf --all        # keep going until none left
    ropf ladder configs/ladder.conf --status     # the progress table

Combos are independent, so the ladder parallelizes by running the command in as
many shells as there are cores and resumes after a crash by being run again.
Both follow from the claim: a combo directory is claimed by creating `.claim/`
inside it, atomically; it is finished when `DONE` exists, written last.  A claim
with no DONE is running or died, and `--reclaim` releases those older than
`stale_claim_hours`.

THE RANKING IS TAKEN ONCE, AT LAMBDA = 0, and the disfigurements are built once
per combo from a declared seed.  Re-ranking per dispatch would score each
dispatch against an adversary told to go easy on it, and two dispatches compared
on different disfigurements are not compared at all.

`ropf score` runs the same campaign against a results tree that already holds
dispatches, without running Algorithm 1.  See `score_combo`.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import algorithm, results as results_module
from ..algorithm import STAGES, WEIGHT_GRID
from ..config import DATA_DIR, ConfigError, RunConfig, _build, parse_keyfile
from ..counterfactual import disfigure, dynamics, frequency, postevent, vendor
from ..counterfactual.postevent import Disfigurement, PostEventParams
from ..model import Solution, SolverConfig
from ..network import Network, read_matpower
from ..risk import METRICS

#: The six instances: the TAMU distribution and the MATPOWER file, smallest
#: first, which is the order a partial run follows.  Both names are needed:
#: `texas2k` is what a config says, `ACTIVSg2000` what the .dyr and .aux are
#: called.  All of them ship in data/.
INSTANCES: Dict[str, Tuple[str, str]] = {
    "activs200": ("ACTIVSg200", "case_ACTIVSg200.m"),
    "activs500": ("ACTIVSg500", "case_ACTIVSg500.m"),
    "texas2k": ("ACTIVSg2000", "case_ACTIVSg2000.m"),
    "activs10k": ("ACTIVSg10k", "case_ACTIVSg10k.m"),
    "activs25k": ("ACTIVSg25k", "case_ACTIVSg25k.m"),
    "activs70k": ("ACTIVSg70k", "case_ACTIVSg70k.m"),
}

#: The ladder, smallest first.
LADDER: Tuple[str, ...] = tuple(INSTANCES)

#: Written last, after every artifact.  Its presence is the only thing that
#: means a combo is finished.
DONE = "DONE"
CLAIM = ".claim"
CAMPAIGN_JSON = "campaign.json"
CAMPAIGN_CSV = "campaign.csv"
TRANSCRIPT = "run.log"

#: The campaign CSV schema.  Columns are APPENDED, never reordered, so a reader
#: written against an older file keeps working.
#:
#: The eq (6e) group: `overload_max_pu`/`overload_sum_pu` are GO3 s_jtk^+ at its
#: largest and summed; `worst_loading` is the parameter-free severity number,
#: max |Pf|/U over rated survivors, which does not mention beta; `worst_branch`
#: is the branch attaining it, since one at-rating branch can decide an instance.
#:
#: `cost_overhead` is the axis the security claim is made on: z(lambda)/z(0) - 1
#: from the sweep the dispatch came from.  `weight_multiplier` is NOT that axis
#: -- on ACTIVSg200 five of six multipliers buy the same 9.04 percent -- and a
#: campaign no longer sits beside its frontier.  Blank, never zero, when the
#: sweep left no lambda = 0 row.
CAMPAIGN_COLUMNS = (
    "weight_multiplier", "risk_weight", "gamma", "disfigurement", "kind", "K",
    "outcome", "reason", "lost_load_pu", "removed_demand_pu", "cost",
    "phase", "n_solves", "solve_time_s",
    "cleared", "failed_test", "rocof_hz_s", "nadir_hz", "H_sys_s",
    "delta_P_pu", "n_responsive", "n_buses", "n_branches", "n_gens",
    "overload_max_pu", "overload_sum_pu", "worst_loading", "worst_branch",
    "cost_overhead",
)


def _noop(_message: str) -> None:
    return None


###############################################################################
# The key table
###############################################################################
#
# TWO DRIVERS, ONE CAMPAIGN.  What they share is Section 4 -- the disfigurement
# classes, the screen, gamma, beta, the post-event solver -- and that is
# `CampaignConfig`.  What they do not share is Algorithm 1, so kappa, eta, k-bar
# and the weight grid are ladder keys alone: a score config naming them would
# claim something that did not happen.  `ropf score` therefore refuses
# `weight_grid` as an unknown key and `ropf ladder` refuses `frontier_dir`,
# both falling out of the field list being the key table.


@dataclass
class CampaignConfig:
    """Every key BOTH drivers may set: Section 4, the screen, and the output.
    Same discipline as `ropf.config.RunConfig`; both refusals come from
    `parse_keyfile`, the only place they are implemented."""

    # --- the grid ------------------------------------------------------------
    #: Instances to run, by the names in `INSTANCES`.
    instances: Tuple[str, ...] = LADDER
    metrics: Tuple[str, ...] = METRICS
    stages: Tuple[str, ...] = STAGES

    # --- Section 4: the campaign ---------------------------------------------
    #: Class (a), Section 4.1.1: disable the K highest-output units.
    gen_k: Tuple[int, ...] = (1, 2, 3, 5, 8, 12, 18, 25)
    #: Class (b), Section 4.1.2: random-walk component sets of size K.
    walk_k: Tuple[int, ...] = (5, 15, 40)
    walk_draws: int = 100
    #: The walk seed.  Declared, so the campaign is reproducible.
    walk_seed: int = 20260820
    #: Class (c): the vendor N-1 list in the case PowerWorld ``.aux``.  Unlike
    #: the sampled classes this one is ENUMERATED, so the worst case over it is
    #: exact rather than an order statistic, which makes it their control.  Its
    #: size is a property of the case, so it is not in `n_disfigurements`;
    #: `build_campaign` logs what it mapped.  ACTIVSg70k ships no list.
    vendor_list: bool = True
    #: gamma, the response-window scale of (D).  A LIST: a declared study axis,
    #: not a measured quantity, so it is swept.  Every value multiplies the
    #: campaign, which `--dry-run` reports before anything runs.
    gamma: Tuple[float, ...] = (0.5,)
    #: beta, the emergency rating factor of eq (6e).  1 by default: rateB and
    #: rateC are zero across the ACTIVSg distributions, but all six .aux files
    #: monitor contingencies against rate set "A" at 100 percent.  1.2 is the
    #: declared short-term-overload sensitivity beside it, and `worst_loading`
    #: does not mention beta, so one run reads at both.
    beta: float = 1.0

    # --- Section 4.2: the frequency screen -----------------------------------
    #: None of these four is defaulted: they are grid-code quantities, and a
    #: plausible substitute would be reported as if it were data.
    f0_hz: Optional[float] = None
    rocof_max_hz_s: Optional[float] = None
    f_under_hz: Optional[float] = None
    #: The load damping, on ONE of the two bases -- set exactly one.  They differ
    #: by the load-to-baseMVA ratio (671 on ACTIVSg2000) and getting it wrong
    #: fails quietly: the nadir just comes out deeper.
    damping_per_load: Optional[float] = None
    damping_on_system: Optional[float] = None
    horizon_s: float = 30.0
    dt_s: float = 0.01
    #: Directories searched for each instance's .dyr and .aux, comma separated.
    #: The ACTIVSg dynamics ship alongside the cases and are not in data/.
    dynamics_search: str = ""

    # --- solvers -------------------------------------------------------------
    solver_dc: str = "gurobi"
    time_limit_s: float = 3600.0
    threads: int = 40
    solver_verbose: bool = False
    #: Crossover for (D).  0 disables it; worth setting only at the top
    #: instance, and it changes the per-bus shed pattern, not the total.
    postevent_crossover: Optional[int] = None

    # --- output --------------------------------------------------------------
    outdir: str = os.path.join("results", "ladder")
    #: How old a claim must be before `--reclaim` releases it.
    stale_claim_hours: float = 24.0

    def __post_init__(self) -> None:
        self._check_grid()
        self._check_campaign()
        self._check_damping()

    # -- kept apart so a subclass says which of them apply --------------------

    def _check_grid(self) -> None:
        for instance in self.instances:
            if instance not in INSTANCES:
                raise ConfigError(
                    f"instances names {instance!r}, which is not a known instance; known: "
                    f"{', '.join(LADDER)}")
        for metric in self.metrics:
            if metric not in METRICS:
                raise ConfigError(f"metrics names {metric!r}, which is not one "
                                  f"of {list(METRICS)}")
        for stage in self.stages:
            if stage not in STAGES:
                raise ConfigError(f"stages names {stage!r}, which is not one "
                                  f"of {list(STAGES)}")
        if not (self.instances and self.metrics and self.stages):
            raise ConfigError("instances, metrics and stages must each name at "
                              "least one value; there is nothing to run")

    def _check_campaign(self) -> None:
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

    def _check_damping(self) -> None:
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

    def as_dict(self) -> Dict[str, Any]:
        record = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in list(record.items()):
            if isinstance(value, tuple):
                record[key] = list(value)
        return record


@dataclass
class LadderConfig(CampaignConfig):
    """`CampaignConfig` plus Algorithm 1: the keys `ropf ladder` adds."""

    # --- Algorithm 1 ---------------------------------------------------------
    kappa: int = 1
    eta: float = 0.5
    k_bar: int = 25
    weight_grid: Tuple[float, ...] = WEIGHT_GRID
    solver_ac: str = "knitro"

    #: Run the counterfactual campaign at all.  Off gives the frontier only.
    #: A ladder key alone: a score config that turned it off would have nothing
    #: left to do.
    campaign: bool = True
    #: Write every master solve to ``<combo dir>/lp/w<lambda>/master_k<k>.lp``.
    write_lp: bool = False

    def __post_init__(self) -> None:
        self._check_grid()
        if self.campaign:
            self._check_campaign()
        self._check_damping()

    # -- derived objects ------------------------------------------------------

    def run_config(self, combo: "Combo", case: str) -> RunConfig:
        """The `RunConfig` this combo is, so the frontier machinery is reused."""
        return RunConfig(
            case=case, metric=combo.metric, stage=combo.stage,
            kappa=self.kappa, eta=self.eta, k_bar=self.k_bar,
            weight_grid=self.weight_grid,
            solver_dc=self.solver_dc, solver_ac=self.solver_ac,
            time_limit_s=self.time_limit_s, threads=self.threads,
            solver_verbose=self.solver_verbose,
            outdir=self.outdir, tag=combo.name, write_lp=self.write_lp)

    def evaluations_per_combo(self) -> int:
        if not self.campaign:
            return 0
        return (len(self.weight_grid) * len(self.gamma)
                * self.n_disfigurements)


@dataclass
class ScoreConfig(CampaignConfig):
    """`CampaignConfig` plus the tree the dispatches are read out of.

    THE WEIGHT GRID IS NOT A KEY HERE: it is a property of the tree named by
    `frontier_dir`, and two statements of one fact can disagree.
    """

    #: The results tree holding the frontier artifacts to score.  READ-ONLY:
    #: nothing is written into it, and `outdir` is where the campaign goes.
    frontier_dir: str = ""
    #: The suffix the combo directories under `frontier_dir` carry.  Their names
    #: come from the `tag` of the config that wrote them:
    #: ``<distribution>_<metric>_<stage>_frontier`` -- the DISTRIBUTION name.
    frontier_suffix: str = "_frontier"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.frontier_dir:
            raise ConfigError(
                "frontier_dir is empty; `ropf score` scores dispatches that "
                "already exist, and there is no tree to read them from. Name "
                "the results tree, e.g. frontier_dir = results/frontier_v1.")

    def evaluations_per_combo(self, n_weights: int) -> int:
        """The count for a combo whose source tree carries `n_weights` points."""
        return n_weights * len(self.gamma) * self.n_disfigurements


def read_ladder_config(path: str) -> LadderConfig:
    """Parse a ladder configuration file.  Same two refusals as a run config."""
    return _build(parse_keyfile(path, LadderConfig), path, LadderConfig)


def read_score_config(path: str) -> ScoreConfig:
    """Parse a score configuration file.  Same two refusals as a run config."""
    return _build(parse_keyfile(path, ScoreConfig), path, ScoreConfig)


###############################################################################
# Combos
###############################################################################


@dataclass(frozen=True)
class Combo:
    """One unit of work: a instance, a functional and a stage."""

    instance: str
    metric: str
    stage: str

    @property
    def name(self) -> str:
        return f"{self.instance}_{self.metric}_{self.stage}"

    @property
    def distribution(self) -> str:
        """The TAMU distribution name, e.g. ACTIVSg2000.  Not the instance name."""
        return INSTANCES[self.instance][0]

    def case_path(self, data_dir: str = DATA_DIR) -> str:
        return os.path.join(data_dir, INSTANCES[self.instance][1])


def combos(config: CampaignConfig) -> List[Combo]:
    """Every combo the config declares, cheap instances first.  The order is the
    ladder own, not the file one, so a partial run covers the cheap ones."""
    order = {name: i for i, name in enumerate(LADDER)}
    return [Combo(instance, metric, stage)
            for instance in sorted(config.instances, key=lambda r: order[r])
            for metric in config.metrics
            for stage in config.stages]


###############################################################################
# Claiming
###############################################################################


def combo_dir(config: CampaignConfig, combo: Combo) -> str:
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
    """Take the claim on a combo directory, atomically.  `os.mkdir` creates or
    raises, with no window in which two processes both think they won -- which
    is why the claim is a directory and not a written file."""
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


def next_combo(config: CampaignConfig,
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


def reclaim_stale(config: CampaignConfig,
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


def status_rows(config: CampaignConfig) -> List[Dict[str, Any]]:
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
    #: What the .dyr reader accepted and rejected.  A rejected record is a unit
    #: with no inertia, and a mostly-failed parse would otherwise look normal.
    dynamics_report: Optional[Any] = None
    #: What the vendor reader mapped and skipped: a rate over a class that
    #: silently lost entries is a rate over an unnamed denominator.
    vendor_report: Optional[Any] = None


def build_campaign(config: CampaignConfig,
                   combo: Combo,
                   network: Network,
                   nominal_Pg: Dict[int, float],
                   log: Callable[[str], None] = _noop) -> Campaign:
    """Everything the campaign needs that does not depend on the dispatch.  The
    ranking is taken here, from the nominal dispatch, and never retaken."""
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
    n_sampled = len(disfigurements)

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

    # Class (c) comes last: it needs the .aux found above, and appending rather
    # than interleaving keeps the sampled classes labels fixed.
    vendor_report = None
    if config.vendor_list:
        if not files.aux:
            log(f" vendor list: no .aux found for {combo.distribution}; "
                f"class (c) is absent for this instance\n")
        else:
            drawn, vendor_report = vendor.vendor_draws(network, files.aux,
                                                       files.archive, log)
            for item in drawn:
                if item.label in kinds:
                    raise ConfigError(
                        f"vendor contingency {item.label!r} collides with an "
                        f"existing disfigurement label. Labels index the "
                        f"campaign CSV, so a collision would silently merge "
                        f"two different events into one row.")
                disfigurements.append(item)
                kinds[item.label] = "vendor"

    log(f" {len(disfigurements)} disfigurements "
        f"({len(config.gen_k)} top-K, "
        f"{len(config.walk_k) * config.walk_draws} walks, "
        f"seed {config.walk_seed}; "
        f"{len(disfigurements) - n_sampled} vendor)\n")

    return Campaign(disfigurements=disfigurements, kinds=kinds,
                    vendor_report=vendor_report,
                    screen_config=screen_config, unit_dynamics=unit_dyn,
                    participation=participation, participation_source=pi_source,
                    dynamics_source=source, dynamics_report=dynamics_report)


def _screen_for(network: Network, campaign: Campaign,
                disfigurement: Disfigurement, P_star: Dict[int, float],
                sink: List[frequency.ScreenResult]) -> Callable:
    """The Section 4.2 screen as (D) wants it: live gens in, cleared/why out.

    `evaluate` passes the surviving generators only and the screen needs the
    buses too, so the closure recomputes them.  The full `ScreenResult` goes to
    `sink`: the campaign records the nadir and RoCoF, not just the verdict.
    """
    def _screen(live_gens):
        live_buses, _, _ = postevent.survivors(network, disfigurement)
        result = frequency.screen(campaign.screen_config, network,
                                  campaign.unit_dynamics, live_buses,
                                  live_gens, P_star)
        sink.append(result)
        return result.cleared, result.reason
    return _screen


def evaluate_campaign(config: CampaignConfig,
                      network: Network,
                      campaign: Campaign,
                      runs: Sequence[algorithm.Result],
                      log: Callable[[str], None] = _noop,
                      overhead: Optional[Dict[float, float]] = None
                      ) -> List[Dict]:
    """Every (weight, gamma, disfigurement) against (D).  One row each.

    `overhead` is each weight cost overhead -- the frontier `cost_ratio` minus
    one -- passed in rather than derived, since it belongs to the sweep and has
    one definition, in `ropf.results`.  A missing weight gets a blank column,
    never a zero.
    """
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
                rows.append(_campaign_row(
                    run, gamma, item, campaign, result,
                    sink[0] if sink else None,
                    (overhead or {}).get(run.weight_multiplier)))
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
                  screen: Optional[frequency.ScreenResult],
                  overhead: Optional[float] = None) -> Dict[str, Any]:
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
        "overload_max_pu": result.overload_max_pu,
        "overload_sum_pu": result.overload_sum_pu,
        "worst_loading": result.worst_loading,
        "worst_branch": result.worst_branch,
        "cost_overhead": overhead,
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
            f"{combo.instance}: {case} is not there. The ACTIVSg cases "
            f"ship in data/.")

    started = time.time()
    network = read_matpower(case, log)
    run_config = config.run_config(combo, case)
    dc_solver, ac_solver = run_config.dc_solver(), run_config.ac_solver()

    runs = []
    for multiplier in run_config.weights:
        log(f"\n lambda = {multiplier:g} lambda*   [{combo.name}]\n")
        runs.append(algorithm.run(
            network,
            run_config.algorithm_config(
                multiplier, lp_dir=run_config.lp_dir(directory, multiplier)),
            dc_solver, ac_solver, log))

    results_module.write(directory, network, run_config, runs, log)

    record: Dict[str, Any] = {
        "combo": {"instance": combo.instance, "metric": combo.metric,
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
        # The overhead the campaign rows are indexed by comes from the same
        # normalization the frontier CSV reports, not from a second one here.
        overhead = _overhead_of(results_module.frontier_rows(
            network, run_config, runs))
        rows = evaluate_campaign(config, network, campaign, runs, log,
                                 overhead)
        record.update(_campaign_record(config, network, campaign, rows))
        _write_json(os.path.join(directory, CAMPAIGN_JSON),
                    dict(record, rows=rows))
        _write_csv(os.path.join(directory, CAMPAIGN_CSV), rows)
        log(f" wrote {CAMPAIGN_JSON} and {CAMPAIGN_CSV} ({len(rows)} rows)\n")

    record["total_time_s"] = round(time.time() - started, 1)
    with open(os.path.join(directory, DONE), "w") as handle:
        json.dump(record, handle, indent=2, default=str)
    return record


def _campaign_record(config: CampaignConfig, network: Network,
                     campaign: Campaign,
                     rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What both drivers record about the campaign they just ran.  One block, so
    a score tree and a ladder tree stay mergeable."""
    return {
        "dynamics_source": campaign.dynamics_source,
        "dynamics_report": (None if campaign.dynamics_report is None
                            else vars(campaign.dynamics_report)),
        "participation_source": campaign.participation_source,
        "screen": {"f0_hz": config.f0_hz,
                   "rocof_max_hz_s": config.rocof_max_hz_s,
                   "f_under_hz": config.f_under_hz,
                   "damping": campaign.screen_config.damping.describe(network),
                   "horizon_s": config.horizon_s, "dt_s": config.dt_s},
        "n_disfigurements": len(campaign.disfigurements),
        "n_evaluations": len(rows),
        # So a rate over the vendor class can state its denominator.
        "vendor_list": (None if campaign.vendor_report is None else {
            "total": campaign.vendor_report.total,
            "mapped": campaign.vendor_report.mapped,
            "skipped_shunt": campaign.vendor_report.skipped_shunt,
            "skipped_multi": campaign.vendor_report.skipped_multi,
            "unmapped_branch": campaign.vendor_report.unmapped_branch,
            "unmapped_gen": campaign.vendor_report.unmapped_gen,
            "unmapped_other": campaign.vendor_report.unmapped_other,
            "examples": campaign.vendor_report.examples,
        }),
    }


def _overhead_of(rows: Sequence[Dict[str, Any]]) -> Dict[float, float]:
    """``{weight_multiplier: cost_ratio - 1}`` over the rows that carry one.  A
    NaN ratio is LEFT OUT: it means the sweep had no lambda = 0 row, and a
    missing overhead has to read as missing in the CSV."""
    out: Dict[float, float] = {}
    for row in rows:
        multiplier, ratio = row.get("weight_multiplier"), row.get("cost_ratio")
        if multiplier is None or ratio is None:
            continue
        ratio = float(ratio)
        if ratio == ratio:                        # not NaN
            out[float(multiplier)] = ratio - 1.0
    return out


###############################################################################
# Scoring a frontier that already exists
###############################################################################
#
# WHY THIS PATH EXISTS.  `run_combo` can only score a dispatch it just
# computed, so it scores whatever `weight_grid` its config names -- and on cost
# overhead, the axis the study argues, a multiplier grid bunches: five of the
# six ACTIVSg200 max_active_flow multipliers buy the same 9.04 percent.  The
# dispatches at the overheads the study wants already exist in
# `results/frontier_v1/`; what was missing was a way to read them.
#
# ALGORITHM 1 DOES NOT RUN HERE.  Not an optimization -- the definition of the
# command.  A second sweep would produce a second set of dispatches.
#
# THE SOURCE TREE IS READ-ONLY.  The campaign goes to `outdir` under the ladder
# own combo names, so a score tree and a ladder tree read the same.
# `source.json` records which frontier directory each combo came from and what
# the sweep said about each weight.

#: Written beside the campaign by `score_combo`.  Where the dispatches came
#: from, and what the sweep said about each one.
SOURCE_JSON = "source.json"

#: The artifacts a combo directory must carry to be scorable.  P* is read from
#: the summary; the other two are cross-checked against it.
FRONTIER_ARTIFACTS = (results_module.SOLUTION_SUMMARY,
                      results_module.FRONTIER_CSV,
                      results_module.DISPATCH_GEN_CSV)

@dataclass
class FrontierSource:
    """One combo's frontier, read back as Algorithm 1 would have returned it."""

    directory: str
    runs: List[algorithm.Result]
    #: The nominal dispatch: the lambda = 0 row of THIS tree.  Section 4.1.1's
    #: ranking is taken from it once and never retaken.
    nominal: Solution
    #: ``{weight_multiplier: cost_ratio - 1}`` off `efficient_frontier.csv`.
    overhead: Dict[float, float]
    #: One record per weight, for `source.json`: what the sweep said.
    weights: List[Dict[str, Any]]
    case_sha256: Optional[str] = None


def frontier_combo_dir(config: ScoreConfig, combo: Combo) -> str:
    """``<frontier_dir>/<distribution>_<metric>_<stage><frontier_suffix>``.  The
    DISTRIBUTION name, because that is what the configs that wrote those trees
    tagged them with."""
    return os.path.join(
        config.frontier_dir,
        f"{combo.distribution}_{combo.metric}_{combo.stage}"
        f"{config.frontier_suffix}")


def read_frontier(directory: str, combo: Combo, case: str,
                  log: Callable[[str], None] = _noop) -> FrontierSource:
    """Read one combo frontier tree back into `algorithm.Result` objects.

    `evaluate_campaign` reads five things off a run -- the multiplier, the
    weight, `dispatch.Pg`, `termination`, and the nominal dispatch -- so a
    `Result` carrying those is enough.  The iteration traces are NOT rebuilt:
    nothing in Section 4 reads them, and a half-populated trace would invite
    someone to.

    FOUR REFUSALS, each a way to score the wrong thing silently:

    * the tree metric or stage is not this combo -- otherwise the directory
      name is the only thing saying what was scored;
    * the case digest in the tree provenance is not the digest of the case
      about to be loaded, so the dispatch and the campaign network would be two
      different systems;
    * `dispatch_gen.csv` and the summary disagree about P*;
    * there is no lambda = 0 row, so the ranking has nothing to rank and the
      campaign no base to normalize cost against.
    """
    missing = [name for name in FRONTIER_ARTIFACTS
               if not os.path.isfile(os.path.join(directory, name))]
    if missing:
        raise FileNotFoundError(
            f"{directory}: {', '.join(missing)} missing, so there is no "
            f"frontier here to score. `ropf score` reads dispatches, it does "
            f"not compute them.")

    summary_path = os.path.join(directory, results_module.SOLUTION_SUMMARY)
    with open(summary_path) as handle:
        summary = json.load(handle)

    digest = summary.get("provenance", {}).get("case_sha256")
    local = results_module.file_digest(case)
    if digest and local and digest != local:
        raise ConfigError(
            f"{directory} was computed on a different "
            f"{os.path.basename(case)}: the sweep records sha256 "
            f"{digest[:16]}..., the file in data/ is "
            f"{local[:16]}.... The dispatch and the campaign network would be "
            f"from two different systems, and nothing downstream would say so.")

    gen_csv = _read_dispatch_gen(os.path.join(
        directory, results_module.DISPATCH_GEN_CSV))
    overhead, frontier_rows = _read_frontier_csv(os.path.join(
        directory, results_module.FRONTIER_CSV))

    runs, weights = [], []
    for record in summary.get("runs", []):
        metric, stage = record.get("metric"), record.get("stage")
        if metric != combo.metric or stage != combo.stage:
            raise ConfigError(
                f"{directory} holds a {metric}/{stage} sweep, and it was read "
                f"for {combo.metric}/{combo.stage}. The directory name is the "
                f"only thing that said otherwise, and a directory name is not "
                f"a record of what was solved.")
        multiplier = record.get("weight_multiplier")
        if multiplier is None:
            # An absolute `risk_weight` run: one point, no multiplier, so
            # nothing to key the frontier normalization off.
            raise ConfigError(
                f"{directory} carries a run with no weight_multiplier, so it "
                f"cannot be indexed against the frontier's cost_ratio. `ropf "
                f"score` reads multiplier sweeps.")
        multiplier = float(multiplier)
        dispatch = _solution_of(record, multiplier, directory, gen_csv)
        runs.append(algorithm.Result(
            case=os.path.basename(case),
            metric=metric, stage=stage,
            flow_domain=record.get("flow_domain") or "all",
            lambda_star=record.get("lambda_star", float("nan")),
            risk_weight=record.get("risk_weight", float("nan")),
            weight_multiplier=multiplier,
            z0=record.get("z0", float("nan")),
            rho0=record.get("rho0", float("nan")),
            phi0=record.get("phi0", float("nan")),
            termination=record.get("termination", "?"),
            termination_detail=record.get("termination_detail", ""),
            dispatch=dispatch,
            total_time_s=record.get("total_time_s", 0.0)))
        row = frontier_rows.get(multiplier, {})
        weights.append({
            "weight_multiplier": multiplier,
            "risk_weight": record.get("risk_weight"),
            "lambda_star": record.get("lambda_star"),
            "termination": record.get("termination", "?"),
            "k_end": record.get("k_end"),
            "cost": row.get("cost"),
            "risk": row.get("risk"),
            "cost_ratio": row.get("cost_ratio"),
            "cost_overhead": overhead.get(multiplier),
            "risk_reduction": row.get("risk_reduction"),
            "has_dispatch": dispatch is not None,
        })

    nominal = next((r.dispatch for r in runs
                    if r.weight_multiplier == 0.0 and r.dispatch is not None),
                   None)
    if nominal is None:
        raise ConfigError(
            f"{directory} has no lambda = 0 dispatch. Section 4.1.1 ranks "
            f"units on the NOMINAL dispatch once and holds that list "
            f"fixed, and the "
            f"nominal dispatch is this sweep's own zero-weight row; there is "
            f"nothing else it could legitimately be.")

    unconverged = [r.weight_multiplier for r in runs
                   if r.termination in ("iteration_limit", "infeasible")]
    log(f" read {len(runs)} weights from {directory}\n")
    shown = ", ".join(f"{overhead[r.weight_multiplier] * 100:.2f}%"
                      for r in runs if r.weight_multiplier in overhead)
    log(f" cost overhead {shown}\n")
    if unconverged:
        # In the transcript of every score run: a number quoted off a cell
        # that did not converge has to say so.  Raising k-bar is not the fix.
        log(f" {len(unconverged)} of {len(runs)} weights terminated "
            f"iteration_limit or infeasible: "
            f"{', '.join(f'x{m:g}' for m in unconverged)}. A number quoted "
            f"from one of these cells must say so.\n")

    return FrontierSource(directory=directory, runs=runs, nominal=nominal,
                          overhead=overhead, weights=weights,
                          case_sha256=digest)


def _solution_of(record: Dict[str, Any], multiplier: float, directory: str,
                 gen_csv: Dict[float, Dict[int, float]]):
    """The dispatch of one summary run, checked against `dispatch_gen.csv`.

    The summary is the source; the CSV is the same numbers written for a reader
    and is CHECKED, not parsed as a second source.  Two sources of one dispatch
    can drift, one source and a check cannot.
    """
    dispatch = record.get("dispatch")
    if not dispatch:
        return None
    Pg = {int(g): float(v) for g, v in dispatch["Pg_pu"].items()}

    table = gen_csv.get(multiplier)
    if table is not None:
        if set(table) != set(Pg):
            raise ConfigError(
                f"{directory}: dispatch_gen.csv and solution_summary.json name "
                f"different generators at lambda = {multiplier:g} lambda* "
                f"({len(table)} against {len(Pg)}).")
        gap = max((abs(table[g] - Pg[g]) for g in Pg), default=0.0)
        if gap > DISPATCH_ATOL:
            raise ConfigError(
                f"{directory}: dispatch_gen.csv and solution_summary.json "
                f"disagree about P* at lambda = {multiplier:g} lambda* by "
                f"{gap:.3e} p.u., past {DISPATCH_ATOL:g}. One of the two "
                f"is not the dispatch that sweep produced.")

    return Solution(status=dispatch.get("status", "solved"),
                    objective=dispatch.get("objective", float("nan")),
                    gen_cost=dispatch.get("gen_cost", float("nan")),
                    phi=float("nan"), Pg=Pg)


#: How far `dispatch_gen.csv` may sit from the summary before the tree is
#: inconsistent.  Both come from one `Solution` in one process, so the only
#: difference is the CSV decimal round trip -- zero at repr precision, and
#: measured at 0 across all nine frontier_v1 baseline trees.
DISPATCH_ATOL = 1e-9


def _read_dispatch_gen(path: str) -> Dict[float, Dict[int, float]]:
    """``{weight_multiplier: {gen count: Pg}}`` off `dispatch_gen.csv`."""
    out: Dict[float, Dict[int, float]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(float(row["weight_multiplier"]), {})[
                int(row["gen"])] = float(row["Pg_pu"])
    return out


def _read_frontier_csv(path: str) -> Tuple[Dict[float, float],
                                           Dict[float, Dict[str, Any]]]:
    """The cost overhead per weight, and the rows it came from.  `cost_ratio` is
    READ, not recomputed: `ropf.results` normalizes within a stage, and a second
    implementation of that rule here could disagree with it."""
    overhead: Dict[float, float] = {}
    rows: Dict[float, Dict[str, Any]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            multiplier = float(row["weight_multiplier"])
            rows[multiplier] = {
                key: _float_or_none(row.get(key))
                for key in ("cost", "risk", "cost_ratio", "risk_ratio",
                            "risk_reduction")}
            ratio = rows[multiplier]["cost_ratio"]
            if ratio is not None and ratio == ratio:      # present, not NaN
                overhead[multiplier] = ratio - 1.0
    return overhead, rows


def _float_or_none(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def score_combo(config: ScoreConfig, combo: Combo,
                log: Callable[[str], None] = _noop) -> Dict[str, Any]:
    """One combo campaign against dispatches that already exist, then DONE.

    Deliberately the same shape as `run_combo`: same disfigurement build, same
    evaluator, same artifacts in the same schema.  The one difference is where
    `runs` comes from, and Algorithm 1 is not in this function.
    """
    directory = combo_dir(config, combo)
    source_dir = frontier_combo_dir(config, combo)
    case = combo.case_path()
    if not os.path.isfile(case):
        raise FileNotFoundError(
            f"{combo.instance}: {case} is not there. The ACTIVSg cases "
            f"ship in data/.")

    started = time.time()
    source = read_frontier(source_dir, combo, case, log)
    network = read_matpower(case, log)

    campaign = build_campaign(config, combo, network, source.nominal.Pg, log)
    rows = evaluate_campaign(config, network, campaign, source.runs, log,
                             source.overhead)

    record: Dict[str, Any] = {
        "combo": {"instance": combo.instance, "metric": combo.metric,
                  "stage": combo.stage, "distribution": combo.distribution},
        "provenance": results_module.provenance(network),
        "config": config.as_dict(),
        "scored": {
            "frontier_dir": source.directory,
            "case_sha256": source.case_sha256,
            "weights": source.weights,
        },
        "frontier_runs": len(source.runs),
    }
    record.update(_campaign_record(config, network, campaign, rows))

    _write_json(os.path.join(directory, SOURCE_JSON), record["scored"])
    _write_json(os.path.join(directory, CAMPAIGN_JSON), dict(record, rows=rows))
    _write_csv(os.path.join(directory, CAMPAIGN_CSV), rows)
    log(f" wrote {SOURCE_JSON}, {CAMPAIGN_JSON} and {CAMPAIGN_CSV} "
        f"({len(rows)} rows)\n")

    record["total_time_s"] = round(time.time() - started, 1)
    with open(os.path.join(directory, DONE), "w") as handle:
        json.dump(record, handle, indent=2, default=str)
    return record


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CAMPAIGN_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

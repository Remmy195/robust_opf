"""The two study drivers: the grid they enumerate, and the claim that makes it safe.

The claim is the part worth testing hardest.  It is what lets the study be run
in several shells at once and resumed after a crash, and both of those are
silently wrong if two processes can hold the same combo -- the second would
overwrite the first's artifacts with a half-finished campaign and the DONE
marker would say it was fine.

The score driver's tests are at the bottom, and they are hard for the same
reason: `ropf score` must not run Algorithm 1, and every way of quietly scoring
the WRONG dispatch produces a perfectly normal-looking campaign.csv.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import fields

import pytest

from ropf import results
from ropf.algorithm import Result
from ropf.config import ConfigError
from ropf.counterfactual import postevent
from ropf.counterfactual.postevent import Disfigurement
from ropf.risk import METRICS
from ropf.study import ladder
from ropf.study.ladder import (Combo, LadderConfig, read_ladder_config,
                               read_score_config)


def base(**kwargs):
    """A config that passes validation, so a test can vary one thing."""
    defaults = {"damping_per_load": 1.0}
    defaults.update(kwargs)
    return LadderConfig(**defaults)


def write(tmp_path, text, name="ladder.conf"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


###############################################################################
# The key table
###############################################################################


def test_the_two_refusals_are_inherited(tmp_path):
    """Unknown and duplicate keys are errors here for the same reason they are
    in a run config -- and by the same code, not a second copy of it."""
    with pytest.raises(ConfigError, match="unknown key 'instance'"):
        read_ladder_config(write(tmp_path, "instance = activs200\n"))
    with pytest.raises(ConfigError, match="already set on line 1"):
        read_ladder_config(write(tmp_path, "k_bar = 5\nk_bar = 9\n"))


def test_a_config_file_round_trips(tmp_path):
    path = write(tmp_path, "instances = activs200, texas2k\n"
                           "metrics = max_active_flow\n"
                           "stages = baseline\n"
                           "gamma = 0.25, 0.5\n"
                           "damping_per_load = 1.5\n")
    config = read_ladder_config(path)
    assert config.instances == ("activs200", "texas2k")
    assert config.gamma == (0.25, 0.5)
    assert config.damping_per_load == 1.5


def test_an_unknown_instance_is_named(tmp_path):
    with pytest.raises(ConfigError, match="activs300"):
        base(instances=("activs300",))


def test_an_unknown_metric_is_rejected():
    with pytest.raises(ConfigError, match="joule"):
        base(metrics=("joule",))


def test_an_unknown_stage_is_rejected():
    with pytest.raises(ConfigError, match="a4"):
        base(stages=("a4",))


def test_an_empty_axis_is_an_error():
    with pytest.raises(ConfigError, match="at least one"):
        base(metrics=())


###############################################################################
# The damping base
###############################################################################


def test_a_damping_without_a_base_is_refused():
    """Neither given: the screen would have no damping at all."""
    with pytest.raises(ConfigError, match="exactly one"):
        LadderConfig()


def test_two_damping_bases_are_refused():
    """Both given: which one applies has no defensible answer, and the failure
    is quiet -- the nadir just comes out at the wrong depth."""
    with pytest.raises(ConfigError, match="exactly one"):
        LadderConfig(damping_per_load=1.0, damping_on_system=0.02)


def test_the_damping_carries_its_base():
    assert base(damping_per_load=1.5).damping().base == "load"
    per_system = LadderConfig(damping_on_system=0.02)
    assert per_system.damping().base == "system"


###############################################################################
# The screen constants
###############################################################################


def test_the_screen_refuses_to_invent_its_constants():
    """f0, RoCoF-bar and f_under are grid-code quantities. A config that omits
    them must stop, not supply a plausible number that gets reported as data."""
    with pytest.raises(ConfigError) as exc:
        base().screen_config()
    message = str(exc.value)
    for name in ("f0_hz", "rocof_max_hz_s", "f_under_hz"):
        assert name in message
    assert "campaign = false" in message, "the error must name the way out"


def test_a_complete_screen_config_builds():
    config = base(f0_hz=60.0, rocof_max_hz_s=0.5, f_under_hz=59.3)
    screen = config.screen_config()
    assert screen.f0_hz == 60.0
    assert screen.f_under_hz == 59.3


###############################################################################
# Combos
###############################################################################


def test_the_full_grid_is_six_by_three_by_three():
    assert len(ladder.combos(base())) == 54


def test_combos_run_the_cheap_instances_first_whatever_the_file_says():
    """A partial run must cover the cheap instances first, so the order is the
    ladder's own and not the order the config happens to list."""
    config = base(instances=("activs70k", "activs200", "texas2k"),
                  metrics=("max_active_flow",), stages=("baseline",))
    assert [c.instance for c in ladder.combos(config)] == [
        "activs200", "texas2k", "activs70k"]


def test_a_combo_names_its_distribution_not_its_instance():
    """`locate` wants ACTIVSg2000; the instance is called texas2k."""
    assert Combo("texas2k", "max_active_flow", "baseline").distribution \
        == "ACTIVSg2000"
    assert Combo("activs200", "max_active_flow", "baseline").name \
        == "activs200_max_active_flow_baseline"


def test_the_evaluation_count_is_the_declared_grid():
    """6 weights x 1 gamma x (8 top-K + 3 x 100 walks)."""
    assert base().evaluations_per_combo() == 1848
    assert base(gamma=(0.25, 0.5, 1.0)).evaluations_per_combo() == 5544
    assert base(campaign=False).evaluations_per_combo() == 0


def test_a_combo_becomes_a_run_config():
    config = base(kappa=3, eta=0.9, k_bar=7, weight_grid=(0.0, 1.0))
    combo = Combo("activs200", "joule_loss_max", "a2")
    run = config.run_config(combo, "/tmp/case.m")
    assert (run.metric, run.stage) == ("joule_loss_max", "a2")
    assert (run.kappa, run.eta, run.k_bar) == (3, 0.9, 7)
    assert run.tag == combo.name


def test_the_thread_count_reaches_gurobi():
    """A thread per logical cpu cost 5.7x on the top instance; the declared count
    has to actually arrive at the solver, not just at Knitro."""
    solver = base(threads=16).postevent_solver()
    assert solver.gurobi_threads == 16
    assert solver.knitro_threads == 16


###############################################################################
# Claiming
###############################################################################


def test_a_combo_can_only_be_claimed_once(tmp_path):
    directory = str(tmp_path / "combo")
    assert ladder.claim(directory) is True
    assert ladder.claim(directory) is False, \
        "a second process must not get the same combo"


def test_a_claim_records_who_took_it(tmp_path):
    directory = str(tmp_path / "combo")
    ladder.claim(directory)
    assert os.path.isfile(os.path.join(directory, ladder.CLAIM, "who"))


def test_a_released_claim_can_be_retaken(tmp_path):
    directory = str(tmp_path / "combo")
    ladder.claim(directory)
    ladder.release(directory)
    assert ladder.claim(directory) is True


def test_done_is_what_finished_means(tmp_path):
    directory = str(tmp_path / "combo")
    ladder.claim(directory)
    assert not ladder.is_done(directory), "a claim is not a completion"
    open(os.path.join(directory, ladder.DONE), "w").close()
    assert ladder.is_done(directory)


def test_next_combo_skips_the_finished_and_the_claimed(tmp_path):
    config = base(instances=("activs200",), metrics=("max_active_flow",),
                  stages=("baseline", "a2", "a3"), outdir=str(tmp_path))
    first = ladder.next_combo(config)
    second = ladder.next_combo(config)
    assert first is not None and second is not None
    assert first != second, "the claim on the first must push the second on"

    # finish both, and the third is the only one left
    for combo in (first, second):
        open(os.path.join(ladder.combo_dir(config, combo), ladder.DONE),
             "w").close()
    third = ladder.next_combo(config)
    assert third not in (first, second)
    assert ladder.next_combo(config) is None, "nothing should be left"


def test_a_stale_claim_is_released_only_when_it_is_stale(tmp_path):
    config = base(instances=("activs200",), metrics=("max_active_flow",),
                  stages=("baseline",), outdir=str(tmp_path),
                  stale_claim_hours=24.0)
    combo = ladder.next_combo(config)
    directory = ladder.combo_dir(config, combo)

    assert ladder.reclaim_stale(config) == [], "a fresh claim is not stale"

    old = os.path.join(directory, ladder.CLAIM)
    os.utime(old, (0, 0))          # claimed at the epoch
    assert ladder.reclaim_stale(config) == [combo]
    assert ladder.claim(directory) is True


def test_a_finished_combo_is_never_reclaimed(tmp_path):
    """A DONE combo with an old claim left behind must stay done."""
    config = base(instances=("activs200",), metrics=("max_active_flow",),
                  stages=("baseline",), outdir=str(tmp_path))
    combo = ladder.next_combo(config)
    directory = ladder.combo_dir(config, combo)
    open(os.path.join(directory, ladder.DONE), "w").close()
    os.utime(os.path.join(directory, ladder.CLAIM), (0, 0))
    assert ladder.reclaim_stale(config) == []


def test_status_reports_all_three_states(tmp_path):
    config = base(instances=("activs200",), metrics=("max_active_flow",),
                  stages=("baseline", "a2", "a3"), outdir=str(tmp_path))
    done = ladder.next_combo(config)
    open(os.path.join(ladder.combo_dir(config, done), ladder.DONE), "w").close()
    ladder.next_combo(config)          # leaves one running

    states = {row["combo"]: row["state"]
              for row in ladder.status_rows(config)}
    assert sorted(states.values()) == ["done", "pending", "running"]


###############################################################################
# Scoring a frontier that already exists
###############################################################################
#
# WHAT THESE TEST.  `ropf score` reads dispatches out of a results tree and runs
# the Section 4 campaign against them.  Two things make it worth testing hard:
# it must not run Algorithm 1 -- that is the whole command -- and it must refuse
# every way of quietly scoring the wrong dispatch, because none of them shows up
# in the output.  A campaign against the wrong case, or against a tree whose two
# copies of P* disagree, produces a perfectly normal-looking campaign.csv.


def frontier_tree(tmp_path, name="ACTIVSg200_max_active_flow_baseline_frontier",
                  metric="max_active_flow", stage="baseline",
                  weights=((0.0, 0.0, "zero_weight", 100.0),
                           (0.5, 12.0, "iteration_limit", 102.0),
                           (2.0, 48.0, "eta_target", 110.0)),
                  Pg=((1, 0.6), (2, 0.4)),
                  digest=None):
    """A minimal frontier tree: the three artifacts `read_frontier` reads.

    Written by hand rather than by running a solve, for the same reason
    `test_results.py` builds `Result` objects by hand: what is under test is the
    reading, and a test that had to solve first would not run without a licence.
    """
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)

    runs, rows = [], []
    base = weights[0][3]
    for multiplier, weight, termination, cost in weights:
        runs.append({
            "metric": metric, "stage": stage,
            "weight_multiplier": multiplier, "risk_weight": weight,
            "lambda_star": 24.0, "z0": base, "rho0": 5.0, "phi0": 5.0,
            "termination": termination, "k_end": 3,
            "dispatch": {"Pg_pu": {str(g): v + multiplier for g, v in Pg},
                         "objective": cost, "gen_cost": cost,
                         "status": "solved"},
        })
        rows.append({"weight_multiplier": multiplier, "risk_weight": weight,
                     "cost": cost, "risk": 5.0 - multiplier,
                     "cost_ratio": cost / base,
                     "risk_ratio": (5.0 - multiplier) / 5.0,
                     "risk_reduction": multiplier / 5.0,
                     "termination": termination})

    summary = {"provenance": {"case_sha256": digest}, "runs": runs}
    (directory / "solution_summary.json").write_text(json.dumps(summary))

    with open(directory / "efficient_frontier.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gen_rows = [{"weight_multiplier": multiplier, "gen": g,
                 "Pg_pu": repr(v + multiplier)}
                for multiplier, _w, _t, _c in weights for g, v in Pg]
    with open(directory / "dispatch_gen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=["weight_multiplier", "gen", "Pg_pu"])
        writer.writeheader()
        writer.writerows(gen_rows)
    return str(directory)


def score_base(tmp_path, **kwargs):
    defaults = {"damping_per_load": 1.0,
                "frontier_dir": str(tmp_path),
                "instances": ("activs200",),
                "metrics": ("max_active_flow",),
                "stages": ("baseline",),
                "gamma": (0.0,)}
    defaults.update(kwargs)
    return ladder.ScoreConfig(**defaults)


COMBO = Combo("activs200", "max_active_flow", "baseline")


# -- the key table ------------------------------------------------------------


def test_a_score_config_has_no_algorithm_1_keys(tmp_path):
    """The command does not run Algorithm 1, so a file that set its parameters
    would be describing something that did not happen."""
    for key in ("weight_grid = 0, 1", "kappa = 5", "eta = 0.85", "k_bar = 40",
                "solver_ac = knitro"):
        with pytest.raises(ConfigError, match="unknown key"):
            read_score_config(write(tmp_path, key + "\n", "score.conf"))


def test_a_ladder_config_has_no_frontier_dir(tmp_path):
    """And the refusal runs the other way too: the ladder computes its own
    dispatches and has no tree to read them from."""
    with pytest.raises(ConfigError, match="unknown key 'frontier_dir'"):
        read_ladder_config(write(tmp_path, "frontier_dir = results/x\n"))


def test_a_score_config_without_a_source_tree_is_refused():
    with pytest.raises(ConfigError, match="frontier_dir is empty"):
        ladder.ScoreConfig(damping_per_load=1.0)


def test_the_shared_campaign_keys_are_shared(tmp_path):
    """Both drivers inherit one key table for Section 4, so the two trees are
    comparable key for key.  A second copy could drift."""
    shared = {f.name for f in fields(ladder.CampaignConfig)}
    for name in ("gamma", "beta", "gen_k", "walk_k", "walk_seed", "vendor_list",
                 "f0_hz", "damping_per_load", "dynamics_search", "outdir"):
        assert name in shared
    assert shared <= {f.name for f in fields(LadderConfig)}
    assert shared <= {f.name for f in fields(ladder.ScoreConfig)}


def test_the_source_directory_is_named_by_distribution(tmp_path):
    """A frontier tree names its directories after the DISTRIBUTION, and the
    ladder names its combos after the instance; the score path crosses between
    them and getting that backwards finds nothing."""
    config = score_base(tmp_path, frontier_dir="results/frontier_v1")
    assert ladder.frontier_combo_dir(config, Combo(
        "texas2k", "joule_loss_max", "baseline")) == os.path.join(
            "results/frontier_v1",
            "ACTIVSg2000_joule_loss_max_baseline_frontier")


def test_the_suffix_is_a_key_not_a_convention(tmp_path):
    config = score_base(tmp_path, frontier_dir="t", frontier_suffix="")
    assert ladder.frontier_combo_dir(config, COMBO) == os.path.join(
        "t", "ACTIVSg200_max_active_flow_baseline")


# -- reading a tree back ------------------------------------------------------


def test_a_tree_reads_back_as_the_runs_the_campaign_wants(tmp_path):
    directory = frontier_tree(tmp_path)
    source = ladder.read_frontier(directory, COMBO, "no_such_case.m")

    assert [r.weight_multiplier for r in source.runs] == [0.0, 0.5, 2.0]
    assert [r.risk_weight for r in source.runs] == [0.0, 12.0, 48.0]
    assert [r.termination for r in source.runs] == [
        "zero_weight", "iteration_limit", "eta_target"]
    assert source.runs[1].dispatch.Pg == {1: 1.1, 2: 0.9}
    assert source.nominal.Pg == {1: 0.6, 2: 0.4}, \
        "the ranking base is this tree's own lambda = 0 row"


def test_the_overhead_comes_off_the_frontier_csv(tmp_path):
    """cost_ratio - 1, read rather than recomputed: `ropf.results` normalizes
    within a stage for a reason written down there, and a second implementation
    of that rule could disagree with it."""
    source = ladder.read_frontier(frontier_tree(tmp_path), COMBO, "nope.m")
    assert source.overhead == pytest.approx({0.0: 0.0, 0.5: 0.02, 2.0: 0.10})


def test_the_source_record_carries_the_termination_beside_the_overhead(tmp_path):
    """Most of these cells exit `iteration_limit`, and a number quoted from one
    of them has to say so.  The two facts have to arrive together."""
    source = ladder.read_frontier(frontier_tree(tmp_path), COMBO, "nope.m")
    at_half = next(w for w in source.weights if w["weight_multiplier"] == 0.5)
    assert at_half["termination"] == "iteration_limit"
    assert at_half["cost_overhead"] == pytest.approx(0.02)


def test_a_tree_of_the_wrong_metric_is_refused(tmp_path):
    """Otherwise the directory name is the only thing that said what was
    scored, and a directory name is not a record of what was solved."""
    directory = frontier_tree(tmp_path, metric="joule_loss_max")
    with pytest.raises(ConfigError, match="joule_loss_max"):
        ladder.read_frontier(directory, COMBO, "nope.m")


def test_a_tree_of_the_wrong_stage_is_refused(tmp_path):
    directory = frontier_tree(tmp_path, stage="a2")
    with pytest.raises(ConfigError, match="a2"):
        ladder.read_frontier(directory, COMBO, "nope.m")


def test_a_tree_computed_on_another_copy_of_the_case_is_refused(tmp_path):
    """The ACTIVSg distributions are not versioned: two copies of one .m can
    differ.  The dispatch and the campaign network must be the same system, and
    nothing downstream would say if they were not."""
    case = tmp_path / "case.m"
    case.write_text("not the case that was solved")
    directory = frontier_tree(tmp_path, digest="0" * 64)
    with pytest.raises(ConfigError, match="different case.m"):
        ladder.read_frontier(directory, COMBO, str(case))


def test_the_matching_digest_passes(tmp_path):
    case = tmp_path / "case.m"
    case.write_text("the case that was solved")
    directory = frontier_tree(tmp_path, digest=results.file_digest(str(case)))
    assert len(ladder.read_frontier(directory, COMBO, str(case)).runs) == 3


def test_the_two_copies_of_the_dispatch_must_agree(tmp_path):
    """`dispatch_gen.csv` is the summary's P* written for a reader. If they
    differ, one of the two is not what the sweep produced -- and the campaign
    would silently take the summary's."""
    directory = frontier_tree(tmp_path)
    path = os.path.join(directory, "dispatch_gen.csv")
    rows = list(csv.DictReader(open(path)))
    rows[0]["Pg_pu"] = str(float(rows[0]["Pg_pu"]) + 1e-3)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ConfigError, match="disagree about"):
        ladder.read_frontier(directory, COMBO, "nope.m")


def test_a_tree_with_no_nominal_dispatch_is_refused(tmp_path):
    """Section 4.1.1 ranks units on the nominal dispatch once and holds that
    list fixed; the nominal dispatch is this sweep's own zero-weight row."""
    directory = frontier_tree(tmp_path, weights=(
        (0.5, 12.0, "iteration_limit", 102.0),
        (2.0, 48.0, "eta_target", 110.0)))
    with pytest.raises(ConfigError, match="no lambda = 0 dispatch"):
        ladder.read_frontier(directory, COMBO, "nope.m")


def test_a_directory_with_no_frontier_says_which_file_is_missing(tmp_path):
    directory = frontier_tree(tmp_path)
    os.remove(os.path.join(directory, "dispatch_gen.csv"))
    with pytest.raises(FileNotFoundError, match="dispatch_gen.csv"):
        ladder.read_frontier(directory, COMBO, "nope.m")


# -- the overhead reaches the campaign row ------------------------------------


def test_the_campaign_row_carries_the_cost_overhead():
    """The axis the security claim is made on. `weight_multiplier` is not it:
    five of vendor_gamma0's six ACTIVSg200 multipliers buy one overhead."""
    run = Result(case="c", metric="max_active_flow", stage="baseline",
                 lambda_star=1.0, risk_weight=2.0, weight_multiplier=0.5,
                 z0=1.0, rho0=1.0, phi0=1.0)
    row = ladder._campaign_row(
        run, 0.0, Disfigurement(label="x"),
        ladder.Campaign(disfigurements=[], kinds={}, screen_config=None),
        postevent.PostEventResult(outcome="survival"), None, 0.0157)
    assert row["cost_overhead"] == 0.0157
    assert "cost_overhead" in ladder.CAMPAIGN_COLUMNS


def test_a_weight_with_no_overhead_gets_a_blank_not_a_zero():
    """A sweep with no lambda = 0 row has no base to normalize against, and a
    missing overhead has to read as missing."""
    run = Result(case="c", metric="max_active_flow", stage="baseline",
                 lambda_star=1.0, risk_weight=2.0, weight_multiplier=0.5,
                 z0=1.0, rho0=1.0, phi0=1.0)
    row = ladder._campaign_row(
        run, 0.0, Disfigurement(label="x"),
        ladder.Campaign(disfigurements=[], kinds={}, screen_config=None),
        postevent.PostEventResult(outcome="survival"), None, None)
    assert row["cost_overhead"] is None


def test_a_nan_cost_ratio_is_dropped_rather_than_carried():
    assert ladder._overhead_of([
        {"weight_multiplier": 0.0, "cost_ratio": 1.0},
        {"weight_multiplier": 1.0, "cost_ratio": float("nan")},
    ]) == {0.0: 0.0}


# -- Algorithm 1 does not run -------------------------------------------------


def test_score_combo_never_runs_algorithm_1(tmp_path, monkeypatch):
    """The point of the command, asserted rather than described.

    Everything after the read is stubbed -- the campaign needs a .dyr, a
    network and a solver, and none of those is what this is testing -- but
    `algorithm.run` is replaced with a detonator, so a score path that grew a
    solve would fail here and not in a fourteen-hour campaign.
    """
    def detonate(*_args, **_kwargs):
        raise AssertionError("score_combo ran Algorithm 1")

    monkeypatch.setattr(ladder.algorithm, "run", detonate)
    monkeypatch.setattr(ladder, "read_matpower", lambda *_a, **_k: object())
    monkeypatch.setattr(ladder, "build_campaign", lambda *_a, **_k: ladder.Campaign(
        disfigurements=[], kinds={}, screen_config=None))
    monkeypatch.setattr(ladder, "evaluate_campaign",
                        lambda *_a, **_k: [{"weight_multiplier": 0.0}])
    monkeypatch.setattr(ladder, "_campaign_record", lambda *_a, **_k: {})
    monkeypatch.setattr(ladder.results_module, "provenance", lambda *_a: {})
    monkeypatch.setattr(Combo, "case_path", lambda self, **_k: __file__)

    source = tmp_path / "src"
    source.mkdir()
    frontier_tree(source)
    config = score_base(tmp_path, frontier_dir=str(source),
                        outdir=str(tmp_path / "out"))
    directory = ladder.combo_dir(config, COMBO)
    os.makedirs(directory, exist_ok=True)

    record = ladder.score_combo(config, COMBO)
    assert record["frontier_runs"] == 3
    assert os.path.isfile(os.path.join(directory, ladder.DONE))
    assert os.path.isfile(os.path.join(directory, ladder.CAMPAIGN_CSV))


def test_score_combo_writes_nothing_into_the_source_tree(tmp_path, monkeypatch):
    """A frontier tree is an input.  A campaign that wrote into it would make
    the tree a record of two different things."""
    monkeypatch.setattr(ladder, "read_matpower", lambda *_a, **_k: object())
    monkeypatch.setattr(ladder, "build_campaign", lambda *_a, **_k: ladder.Campaign(
        disfigurements=[], kinds={}, screen_config=None))
    monkeypatch.setattr(ladder, "evaluate_campaign", lambda *_a, **_k: [])
    monkeypatch.setattr(ladder, "_campaign_record", lambda *_a, **_k: {})
    monkeypatch.setattr(ladder.results_module, "provenance", lambda *_a: {})
    monkeypatch.setattr(Combo, "case_path", lambda self, **_k: __file__)

    source = tmp_path / "src"
    source.mkdir()
    directory = frontier_tree(source)
    before = sorted(os.listdir(directory))

    config = score_base(tmp_path, frontier_dir=str(source),
                        outdir=str(tmp_path / "out"))
    os.makedirs(ladder.combo_dir(config, COMBO), exist_ok=True)
    ladder.score_combo(config, COMBO)

    assert sorted(os.listdir(directory)) == before
    assert os.path.isfile(os.path.join(
        ladder.combo_dir(config, COMBO), ladder.SOURCE_JSON))


def test_source_json_says_which_tree_was_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "read_matpower", lambda *_a, **_k: object())
    monkeypatch.setattr(ladder, "build_campaign", lambda *_a, **_k: ladder.Campaign(
        disfigurements=[], kinds={}, screen_config=None))
    monkeypatch.setattr(ladder, "evaluate_campaign", lambda *_a, **_k: [])
    monkeypatch.setattr(ladder, "_campaign_record", lambda *_a, **_k: {})
    monkeypatch.setattr(ladder.results_module, "provenance", lambda *_a: {})
    monkeypatch.setattr(Combo, "case_path", lambda self, **_k: __file__)

    source = tmp_path / "src"
    source.mkdir()
    directory = frontier_tree(source)
    config = score_base(tmp_path, frontier_dir=str(source),
                        outdir=str(tmp_path / "out"))
    os.makedirs(ladder.combo_dir(config, COMBO), exist_ok=True)
    ladder.score_combo(config, COMBO)

    written = json.load(open(os.path.join(
        ladder.combo_dir(config, COMBO), ladder.SOURCE_JSON)))
    assert written["frontier_dir"] == directory
    assert [w["weight_multiplier"] for w in written["weights"]] == [0.0, 0.5, 2.0]
    assert [w["termination"] for w in written["weights"]] == [
        "zero_weight", "iteration_limit", "eta_target"]

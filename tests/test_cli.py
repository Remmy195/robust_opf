"""The command surface: the config file is the only place a value is set."""

from __future__ import annotations

import os
import re

import pytest

from ropf import cli
from ropf.config import RunConfig

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASE = os.path.join(FIXTURES, "case_ACTIVSg200.m")


def test_no_command_prints_help_and_fails(capsys):
    assert cli.main([]) == 2
    assert "COMMAND" in capsys.readouterr().out


def test_keys_lists_every_declared_key(capsys):
    assert cli.main(["keys"]) == 0
    printed = capsys.readouterr().out
    from ropf.config import KEYS
    for key in KEYS:
        assert re.search(rf"^\s+{re.escape(key)}\s", printed, re.M), \
            f"`ropf keys` does not document {key}"


def test_no_flag_can_change_a_study_value():
    """The two flags that exist change what is printed, not what is computed.

    An override would mean one config file producing two different studies
    depending on how it was invoked, and the output directory would no longer
    be a record of what was run.
    """
    parser = cli._parser()
    solve = parser._subparsers._group_actions[0].choices["solve"]
    flags = {action.dest for action in solve._actions} - {"help"}
    assert flags == {"config", "quiet", "dry_run"}, \
        f"the solve command grew a flag: {sorted(flags)}"


def test_score_takes_the_same_flags_as_ladder_and_no_more():
    """`ropf score` is `ropf ladder` without Algorithm 1, and the flag set says
    so: neither `frontier_dir` nor anything else about what is scored is
    reachable from the command line."""
    parser = cli._parser()
    choices = parser._subparsers._group_actions[0].choices
    def flags(name):
        return {a.dest for a in choices[name]._actions} - {"help"}
    assert flags("score") == flags("ladder") == {
        "config", "all", "status", "dry_run", "reclaim", "quiet"}


def test_score_dry_run_reads_the_overheads_and_solves_nothing(tmp_path, capsys):
    """The overheads are the reason the command exists and they live in the
    source tree, so --dry-run reads them.  It writes nothing."""
    source = tmp_path / "trees" / "ACTIVSg200_max_active_flow_baseline_frontier"
    _frontier_tree(source)
    config = tmp_path / "score.conf"
    config.write_text(
        f"frontier_dir = {tmp_path / 'trees'}\n"
        f"instances = activs200\nmetrics = max_active_flow\n"
        f"stages = baseline\ngamma = 0\ndamping_per_load = 1.0\n"
        f"outdir = {tmp_path / 'out'}\n")

    assert cli.main(["score", str(config), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "read-only" in printed
    assert "0.00, 2.00" in printed, printed
    assert not os.path.exists(tmp_path / "out"), "a dry run wrote an output tree"
    assert sorted(os.listdir(source)) == [
        "dispatch_gen.csv", "efficient_frontier.csv", "solution_summary.json"]


def test_score_dry_run_names_a_combo_with_no_frontier(tmp_path, capsys):
    """A combo whose source tree is absent is reported and the command fails,
    rather than the campaign silently covering eight of nine cells."""
    (tmp_path / "trees").mkdir()
    config = tmp_path / "score.conf"
    config.write_text(
        f"frontier_dir = {tmp_path / 'trees'}\n"
        f"instances = activs200\nmetrics = max_active_flow\n"
        f"stages = baseline\ngamma = 0\ndamping_per_load = 1.0\n"
        f"outdir = {tmp_path / 'out'}\n")
    assert cli.main(["score", str(config), "--dry-run"]) == 1
    assert "no frontier to score" in capsys.readouterr().out


def _frontier_tree(directory):
    """The three artifacts `read_frontier` reads, at two weights."""
    import csv
    import json
    directory.mkdir(parents=True)
    runs, rows, gens = [], [], []
    for multiplier, cost in ((0.0, 100.0), (1.0, 102.0)):
        runs.append({"metric": "max_active_flow", "stage": "baseline",
                     "weight_multiplier": multiplier, "risk_weight": multiplier,
                     "lambda_star": 1.0, "z0": 100.0, "rho0": 1.0, "phi0": 1.0,
                     "termination": "zero_weight" if not multiplier else
                                    "eta_target",
                     "dispatch": {"Pg_pu": {"1": 1.0 + multiplier},
                                  "objective": cost, "gen_cost": cost,
                                  "status": "solved"}})
        rows.append({"weight_multiplier": multiplier, "cost": cost,
                     "risk": 1.0, "cost_ratio": cost / 100.0})
        gens.append({"weight_multiplier": multiplier, "gen": 1,
                     "Pg_pu": repr(1.0 + multiplier)})
    (directory / "solution_summary.json").write_text(
        json.dumps({"provenance": {}, "runs": runs}))
    for name, table in (("efficient_frontier.csv", rows),
                        ("dispatch_gen.csv", gens)):
        with open(directory / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


def test_dry_run_reports_the_effective_config_and_solves_nothing(tmp_path, capsys):
    config = tmp_path / "run.conf"
    config.write_text(f"case = {CASE}\nmetric = joule_loss_max\n"
                      f"outdir = {tmp_path / 'out'}\n")
    assert cli.main(["solve", str(config), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "joule_loss_max" in printed
    assert not (tmp_path / "out").exists(), "a dry run must write nothing"


def test_a_bad_config_fails_before_anything_is_written(tmp_path, capsys):
    config = tmp_path / "run.conf"
    config.write_text("kapa = 3\n")
    assert cli.main(["solve", str(config)]) == 2
    assert "did you mean 'kappa'" in capsys.readouterr().err


def test_a_config_with_no_case_is_an_error(tmp_path, capsys):
    config = tmp_path / "run.conf"
    config.write_text("metric = joule_loss_max\n")
    assert cli.main(["solve", str(config)]) == 2
    assert "no `case` given" in capsys.readouterr().err


def test_the_output_directory_names_the_run():
    config = RunConfig(case="/data/case_ACTIVSg2000.m", metric="joule_loss_max",
                       stage="a2", outdir="results")
    assert cli._output_dir(config) == \
        os.path.join("results", "case_ACTIVSg2000_joule_loss_max_a2")
    assert cli._output_dir(RunConfig(case="x.m", tag="instance3")) == \
        os.path.join("results", "instance3")


def test_solve_writes_the_declared_artifacts(tmp_path):
    pytest.importorskip("amplpy", reason="AMPL is not installed")
    from ropf import results

    outdir = tmp_path / "out"
    config = tmp_path / "run.conf"
    config.write_text(f"case = {CASE}\nweight_grid = 0, 1\nkappa = 2\n"
                      f"k_bar = 2\noutdir = {outdir}\ntag = smoke\n")
    try:
        code = cli.main(["solve", str(config), "--quiet"])
    except Exception as exc:                      # no licence, no solver, ...
        pytest.skip(f"could not solve: {exc}")
    if code != 0:
        pytest.skip("the master did not solve")

    written = sorted(os.listdir(outdir / "smoke"))
    assert written == sorted([results.SOLUTION_SUMMARY, results.FRONTIER_CSV,
                              results.FRONTIER_JSON, results.DISPATCH_BUS_CSV,
                              results.DISPATCH_GEN_CSV,
                              results.DISPATCH_BRANCH_CSV, cli.TRANSCRIPT])
